"""
logger.py
Structured logging for CrossGuard.
  - Human-readable console output
  - Structured JSONL scan log (one JSON object per line)
  - Log rotation
"""

import json
import logging
import logging.handlers
import os
import sys
from datetime import datetime, timezone

from settings import LOG_FILE, LOG_LEVEL, SCAN_LOG_FILE


def get_logger(name: str = 'crossguard') -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    fmt = logging.Formatter(
        '[%(asctime)s] %(levelname)-8s %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    try:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            LOG_FILE,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except OSError:
        pass

    return logger


class ScanLogger:
    """
    Appends one JSON object per scan to SCAN_LOG_FILE.
    Each line is a valid JSON object - easy to parse, grep, or stream.
    """

    def __init__(self, path: str = SCAN_LOG_FILE):
        self.path = path
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        except OSError:
            pass

    def log(self, result: dict) -> None:
        entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            **result,
        }
        try:
            with open(self.path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry) + '\n')
        except OSError:
            pass

    def read_all(self) -> list:
        if not os.path.isfile(self.path):
            return []

        records = []
        with open(self.path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return records

    def read_recent(self, n: int = 100) -> list:
        return self.read_all()[-n:]

    def stats(self) -> dict:
        records = self.read_all()
        if not records:
            return {'total': 0, 'malware': 0, 'suspicious': 0, 'clean': 0}

        verdicts = [record.get('verdict', 'UNKNOWN') for record in records]
        return {
            'total': len(records),
            'malware': verdicts.count('MALWARE'),
            'suspicious': verdicts.count('SUSPICIOUS'),
            'clean': verdicts.count('CLEAN'),
            'by_platform': _count_by(records, 'platform'),
            'avg_score': round(
                sum(record.get('final_score', 0) for record in records) / len(records),
                4,
            ),
        }


def _count_by(records: list, key: str) -> dict:
    counts = {}
    for record in records:
        value = record.get(key, 'Unknown')
        counts[value] = counts.get(value, 0) + 1
    return counts
