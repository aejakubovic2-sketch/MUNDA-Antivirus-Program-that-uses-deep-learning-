"""
scanner.py
Main scanning engine — ties together file identification,
feature extraction, LightGBM, MalConv2, and ensemble.
"""

import os
import time
import hashlib
from typing import Optional

from file_identifier import identify_file, format_size
from feature_extractor import extract_features
from lightgbm_model import LightGBMDetector
from ensemble import EnsembleDetector
from settings import SUPPORTED_TYPES


class _UnavailableDetector:
    def __init__(self, reason: str):
        self.reason = reason

    def predict(self, *_args, **_kwargs) -> float:
        raise RuntimeError(self.reason)


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
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        start_time = time.time()

        # Step 1: Identify file type
        file_info = identify_file(filepath)
        file_type  = file_info['file_type']
        platform   = file_info['platform']
        size       = file_info['size_bytes']

        # Step 2: Compute SHA256 hash
        sha256 = self._sha256(filepath)

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
        if os.environ.get('CROSSGUARD_ENABLE_MALCONV2') != '1':
            return _UnavailableDetector(
                "MalConv2 is disabled because no public pretrained checkpoint is configured."
            )

        try:
            from malconv2_model import MalConv2Detector
            return MalConv2Detector(device=device)
        except Exception as e:
            print(f"[Scanner] MalConv2 unavailable: {e}")
            return _UnavailableDetector(str(e))

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
                'scanner': (
                    'Unsupported file type. MUNDA currently scans PE, ELF, APK, '
                    'PDF, and .NET files with the LightGBM model.'
                )
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
