"""
watchdog_monitor.py
Real-time folder monitor.
Watches one or more directories and automatically scans any new file dropped in.
Uses Python's built-in watchdog pattern via polling (no extra dependencies needed).
Optionally uses the 'watchdog' library for faster inotify-based detection.

Usage:
    python -m utils.watchdog_monitor /path/to/watch [/another/path]
    python main.py --watch /downloads --watch /tmp/uploads
"""

import os
import sys
import time
import threading
import hashlib
from datetime import datetime
from typing import Callable, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from settings import (
    MAX_FILE_SIZE_MB,
    WATCHDOG_DEBOUNCE_S,
    SUPPORTED_TYPES,
)
from logger import get_logger, ScanLogger

logger = get_logger('watchdog')


# ── File event ───────────────────────────────────────────

class FileEvent:
    def __init__(self, path: str, event_type: str = 'created'):
        self.path       = path
        self.event_type = event_type
        self.timestamp  = datetime.utcnow().isoformat()
        self.size       = os.path.getsize(path) if os.path.isfile(path) else 0


# ── Polling-based watcher ────────────────────────────────

class PollingWatcher:
    """
    Polls a directory every `interval` seconds for new or modified files.
    Works on all platforms with zero extra dependencies.
    """

    def __init__(self, paths: List[str], interval: float = 2.0,
                 recursive: bool = True):
        self.paths     = [os.path.abspath(p) for p in paths]
        self.interval  = interval
        self.recursive = recursive
        self._seen: dict = {}      # path → (mtime, size)
        self._handlers: List[Callable] = []
        self._running  = False
        self._thread: Optional[threading.Thread] = None

    def on_file(self, handler: Callable[[FileEvent], None]) -> None:
        """Register a callback called for every new/modified file."""
        self._handlers.append(handler)

    def start(self) -> None:
        self._running = True
        # Take initial snapshot (don't fire events for pre-existing files)
        for path in self.paths:
            self._snapshot(path)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info(f"Watchdog started on: {self.paths}")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Watchdog stopped.")

    def _run(self) -> None:
        while self._running:
            for path in self.paths:
                try:
                    self._check(path)
                except Exception as e:
                    logger.warning(f"Watchdog error on {path}: {e}")
            time.sleep(self.interval)

    def _snapshot(self, dirpath: str) -> None:
        """Record current state of all files."""
        for root, dirs, files in os.walk(dirpath):
            for fname in files:
                fp = os.path.join(root, fname)
                try:
                    st = os.stat(fp)
                    self._seen[fp] = (st.st_mtime, st.st_size)
                except OSError:
                    pass
            if not self.recursive:
                break

    def _check(self, dirpath: str) -> None:
        """Detect new or modified files."""
        for root, dirs, files in os.walk(dirpath):
            for fname in files:
                fp = os.path.join(root, fname)
                try:
                    st = os.stat(fp)
                    key = (st.st_mtime, st.st_size)
                    if fp not in self._seen:
                        self._seen[fp] = key
                        self._fire(FileEvent(fp, 'created'))
                    elif self._seen[fp] != key:
                        self._seen[fp] = key
                        self._fire(FileEvent(fp, 'modified'))
                except OSError:
                    pass
            if not self.recursive:
                break

    def _fire(self, event: FileEvent) -> None:
        for handler in self._handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Handler error: {e}")


# ── Auto-scanner ─────────────────────────────────────────

class AutoScanner:
    """
    Combines PollingWatcher with the CrossGuard scanner.
    Automatically scans any new file that appears in watched directories.
    """

    def __init__(self, watch_paths: List[str],
                 method: str = 'meta',
                 recursive: bool = True,
                 poll_interval: float = 2.0,
                 on_result: Optional[Callable] = None):
        """
        Args:
            watch_paths:   directories to monitor
            method:        ensemble method ('meta' | 'weighted' | 'average')
            recursive:     watch subdirectories
            poll_interval: seconds between directory polls
            on_result:     optional callback(result_dict) for each scan
        """
        self.watch_paths   = watch_paths
        self.method        = method
        self.on_result     = on_result
        self._scan_logger  = ScanLogger()
        self._watcher      = PollingWatcher(watch_paths, poll_interval, recursive)
        self._watcher.on_file(self._handle_file)
        self._scanner      = None   # lazy-loaded
        self._pending: dict = {}    # path → time of last write (debounce)
        self._debounce_thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        """Start monitoring. Blocks until stop() is called."""
        logger.info("AutoScanner starting...")
        self._running = True
        self._watcher.start()
        self._debounce_thread = threading.Thread(
            target=self._debounce_loop, daemon=True
        )
        self._debounce_thread.start()

        print("\n" + "="*55)
        print("  CrossGuard AutoScanner — ACTIVE")
        print("="*55)
        for p in self.watch_paths:
            print(f"  Watching: {p}")
        print("  Press Ctrl+C to stop.\n")

        try:
            while self._running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self) -> None:
        self._running = False
        self._watcher.stop()
        logger.info("AutoScanner stopped.")

    # ── Debounce logic ───────────────────────────────────

    def _handle_file(self, event: FileEvent) -> None:
        """Called by watcher — debounces rapid write events."""
        size_mb = event.size / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            logger.info(f"Skipping oversized file: {event.path} ({size_mb:.1f} MB)")
            return
        # Record latest write time; debounce loop will scan after quiet period
        self._pending[event.path] = time.time()

    def _debounce_loop(self) -> None:
        """Wait until file is fully written before scanning."""
        while self._running:
            now = time.time()
            ready = [
                path for path, ts in list(self._pending.items())
                if now - ts >= WATCHDOG_DEBOUNCE_S
            ]
            for path in ready:
                del self._pending[path]
                self._scan_file(path)
            time.sleep(0.2)

    # ── Scan ─────────────────────────────────────────────

    def _scan_file(self, filepath: str) -> None:
        if not os.path.isfile(filepath):
            return

        # Lazy-load scanner (heavy — don't load until first file arrives)
        if self._scanner is None:
            from scanner import Scanner
            self._scanner = Scanner(method=self.method)

        logger.info(f"Auto-scanning: {filepath}")
        try:
            result = self._scanner.scan(filepath)
            self._scan_logger.log(result)
            self._print_alert(result)
            if self.on_result:
                self.on_result(result)
        except Exception as e:
            logger.error(f"Scan failed for {filepath}: {e}")

    # ── Output ───────────────────────────────────────────

    @staticmethod
    def _print_alert(result: dict) -> None:
        v = result['verdict']
        icons = {'MALWARE': '🚨', 'SUSPICIOUS': '⚠️ ', 'CLEAN': '✅'}
        ts = datetime.utcnow().strftime('%H:%M:%S')

        print(f"\n[{ts}] {icons.get(v,'?')} {v}")
        print(f"  File:     {result['filename']}")
        print(f"  Platform: {result['platform']}")
        print(f"  Score:    {_format_score(result['final_score'])}  "
              f"(LightGBM={_format_score(result['lgbm_score'])}, "
              f"MalConv2={_format_score(result['malconv_score'])})")
        print(f"  Confidence: {result['confidence']}")
        if v == 'MALWARE':
            print(f"  ⛔ SHA256: {result['sha256']}")


def _format_score(score) -> str:
    if score is None:
        return 'unavailable'
    return f"{score:.4f}"


# ── CLI entry point ──────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='CrossGuard AutoScanner')
    parser.add_argument('paths', nargs='+', help='Directories to monitor')
    parser.add_argument('--method', default='meta',
                        choices=['meta', 'weighted', 'average'])
    parser.add_argument('--no-recursive', action='store_true')
    args = parser.parse_args()

    monitor = AutoScanner(
        watch_paths=args.paths,
        method=args.method,
        recursive=not args.no_recursive,
    )
    monitor.start()
