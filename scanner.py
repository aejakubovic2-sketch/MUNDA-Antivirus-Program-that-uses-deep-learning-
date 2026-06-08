"""
scanner.py
Main scanning engine — ties together file identification,
feature extraction, LightGBM, MalConv2, and ensemble.
"""

import os
import time
import hashlib
import json
import subprocess
import sys
import importlib.util
from typing import Optional

from file_identifier import identify_file, format_size
from feature_extractor import extract_features
from lightgbm_model import LightGBMDetector
from ensemble import EnsembleDetector
from settings import SUPPORTED_TYPES

MACOS_TYPES = {'MACHO', 'DMG', 'MACAPP'}


class _UnavailableDetector:
    def __init__(self, reason: str):
        self.reason = reason

    def predict(self, *_args, **_kwargs) -> float:
        raise RuntimeError(self.reason)


class _SubprocessMalConv2Detector:
    def __init__(self, device: str = None, timeout: int = 120):
        self.device = device
        self.timeout = timeout

    def predict(self, filepath: str) -> float:
        script = os.path.join(os.path.dirname(__file__), 'malconv2_model.py')
        command = [sys.executable, script, '--predict-json', filepath]
        env = os.environ.copy()
        if self.device:
            command.extend(['--device', self.device])
            env['MUNDA_MALCONV2_DEVICE'] = self.device

        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
                env=env,
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                f"MalConv2 timed out after {self.timeout}s while scanning "
                f"{os.path.basename(filepath)}."
            ) from e

        payload = self._parse_json_payload(completed.stdout)
        if completed.returncode != 0:
            if payload and 'error' in payload:
                raise RuntimeError(payload['error'])
            message = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(message or 'MalConv2 subprocess failed')

        if payload is None:
            raise RuntimeError(
                f"MalConv2 returned invalid output: {completed.stdout.strip()}"
            )

        if 'error' in payload:
            raise RuntimeError(payload['error'])
        return float(payload['score'])

    @staticmethod
    def _parse_json_payload(stdout: str):
        try:
            return json.loads(stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            return None


class Scanner:
    """
    MUNDA malware scanner.

    Usage:
        scanner = Scanner()
        result  = scanner.scan('/path/to/file.exe')
        print(result)
    """

    def __init__(self, method: str = 'meta', device: str = None):
        """
        Args:
            method: ensemble method — 'meta' | 'weighted' | 'average'
            device: torch device — 'cuda' | 'cpu' | None (auto-detect)
        """
        print("[Scanner] Initialising MUNDA...")
        self.lgbm    = LightGBMDetector()
        self.malconv = self._init_malconv(device)
        self.ensemble = EnsembleDetector(self.lgbm, self.malconv, method=method)
        print(f"[Scanner] Ready. Ensemble method: {method}")

    # ── Public API ───────────────────────────────────────

    def scan(self, filepath: str) -> dict:
        """
        Scan a single file.

        Returns a result dict with all detection details.
        Raises FileNotFoundError if file doesn't exist.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        start_time = time.time()

        # Step 1: Identify file type
        file_info = identify_file(filepath)
        file_type  = file_info['file_type']
        platform   = file_info['platform']
        size       = file_info['size_bytes']

        # Step 2: Compute SHA256 hash
        sha256 = self._sha256(filepath) if os.path.isfile(filepath) else 'N/A'

        if file_type not in SUPPORTED_TYPES:
            elapsed = round(time.time() - start_time, 3)
            return self._unsupported_result(
                filepath, file_type, platform, size, sha256, elapsed
            )

        # Step 3: Extract EMBER v3 features
        try:
            features = extract_features(filepath)
        except Exception as e:
            print(f"[Scanner] Feature extraction failed: {e}")
            import numpy as np
            features = __import__('numpy').zeros(2568, dtype='float32')

        # Step 4: Run available detector models.
        result = self.ensemble.predict(features, filepath, file_type)

        elapsed = round(time.time() - start_time, 3)

        return {
            # File info
            'filepath':      filepath,
            'filename':      os.path.basename(filepath),
            'file_type':     file_type,
            'platform':      platform,
            'size':          format_size(size),
            'size_bytes':    size,
            'sha256':        sha256,

            # Detection scores
            'final_score':   result['final_score'],
            'lgbm_score':    result['lgbm_score'],
            'malconv_score': result['malconv_score'],

            # Verdict
            'verdict':       result['verdict'],
            'confidence':    result['confidence'],
            'is_malware':    result['verdict'] == 'MALWARE',

            # Meta
            'method':        result['method'],
            'model_errors':  result.get('model_errors', {}),
            'scan_time_s':   elapsed,
        }

    def scan_directory(self, dirpath: str,
                       recursive: bool = True,
                       extensions: list = None) -> list:
        """
        Scan all files in a directory.

        Args:
            dirpath:    directory to scan
            recursive:  scan subdirectories
            extensions: optional list of extensions to filter, e.g. ['.exe', '.apk']
        Returns:
            list of result dicts, sorted by final_score descending
        """
        results = []
        for root, dirs, files in os.walk(dirpath):
            for fname in files:
                if extensions:
                    if not any(fname.lower().endswith(ext) for ext in extensions):
                        continue
                fpath = os.path.join(root, fname)
                try:
                    result = self.scan(fpath)
                    results.append(result)
                    self._print_result(result)
                except Exception as e:
                    print(f"[Scanner] Error scanning {fpath}: {e}")
            if not recursive:
                break

        results.sort(
            key=lambda r: r['final_score'] if r['final_score'] is not None else -1,
            reverse=True,
        )
        return results

    # ── Helpers ──────────────────────────────────────────

    @staticmethod
    def _init_malconv(device: str = None):
        enable_flag = (
            os.environ.get('MUNDA_ENABLE_MALCONV2') or
            os.environ.get('CROSSGUARD_ENABLE_MALCONV2')
        )
        if enable_flag == '0':
            return _UnavailableDetector(
                "MalConv2 is disabled by environment configuration."
            )

        if importlib.util.find_spec('torch') is None:
            return _UnavailableDetector(
                "MalConv2 requires PyTorch, but `torch` is not installed. "
                "Run `python -m pip install -r requirements.txt`."
            )

        checkpoint = os.path.join(
            os.path.dirname(__file__),
            'data',
            'malconv2',
            'malconvGCT_nocat.checkpoint',
        )
        legacy_checkpoint = os.path.join(
            os.path.dirname(__file__),
            'data',
            'malconv2',
            'malconv2_pretrained.pt',
        )
        if enable_flag == '1' or os.path.isfile(checkpoint) or os.path.isfile(legacy_checkpoint):
            timeout = int(os.environ.get('MUNDA_MALCONV2_TIMEOUT', '120'))
            return _SubprocessMalConv2Detector(device=device, timeout=timeout)

        return _UnavailableDetector(
            "MalConv2 checkpoint is not downloaded yet. Run "
            "`python3.12 main.py --download-models` to add it."
        )

    @staticmethod
    def _sha256(filepath: str) -> str:
        h = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _unsupported_result(filepath: str, file_type: str, platform: str,
                            size: int, sha256: str, elapsed: float) -> dict:
        return {
            'filepath':      filepath,
            'filename':      os.path.basename(filepath),
            'file_type':     file_type,
            'platform':      platform,
            'size':          format_size(size),
            'size_bytes':    size,
            'sha256':        sha256,
            'final_score':   None,
            'lgbm_score':    None,
            'malconv_score': None,
            'verdict':       'UNSUPPORTED',
            'confidence':    'N/A',
            'is_malware':    False,
            'method':        'none',
            'model_errors': {
                'scanner': _unsupported_message(file_type)
            },
            'scan_time_s':   elapsed,
        }

    @staticmethod
    def _print_result(result: dict) -> None:
        icon = {'MALWARE': '[!]', 'SUSPICIOUS': '[?]', 'CLEAN': '[OK]'}
        v    = result['verdict']
        print(
            f"{icon.get(v, '?')} [{v:<10}] "
            f"{result['filename']:<40} "
            f"score={_format_score(result['final_score'])}  "
            f"({result['platform']})  "
            f"{result['scan_time_s']}s"
        )


def _format_score(score) -> str:
    if score is None:
        return 'N/A'
    return f"{score:.3f}"


def _unsupported_message(file_type: str) -> str:
    if file_type in MACOS_TYPES:
        return (
            'macOS file type recognized, but MUNDA does not include a trained '
            'macOS malware model yet. Current scoring models support Windows PE, '
            'Linux ELF, Android APK, and PDF targets.'
        )
    return (
        'Unsupported file type. MUNDA currently scans PE, ELF, APK, PDF, '
        'and .NET files with the LightGBM model.'
    )
