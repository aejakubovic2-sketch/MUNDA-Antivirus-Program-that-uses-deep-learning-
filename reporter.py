"""
reporter.py
Generates scan reports in multiple formats: JSON, CSV, plain text.
For PDF output use the pdf skill (requires reportlab).
"""

import os
import csv
import json
from datetime import datetime, timezone
from typing import List, Dict

from settings import REPORT_DIR
from logger import get_logger

logger = get_logger('reporter')


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ReportGenerator:
    """
    Generates scan reports from a list of scan result dicts.
    Supports JSON, CSV, and plain text output.
    """

    def __init__(self, results: List[Dict]):
        self.results = results
        self.timestamp = _utc_now().strftime('%Y%m%d_%H%M%S')
        os.makedirs(REPORT_DIR, exist_ok=True)

    def save_json(self, path: str = None) -> str:
        path = path or os.path.join(REPORT_DIR, f'scan_{self.timestamp}.json')
        payload = {
            'generated': _utc_now().isoformat(),
            'total_scans': len(self.results),
            'summary': self._summary(),
            'results': self.results,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)
        logger.info(f"JSON report -> {path}")
        return path

    def save_csv(self, path: str = None) -> str:
        path = path or os.path.join(REPORT_DIR, f'scan_{self.timestamp}.csv')
        fields = [
            'timestamp', 'filename', 'platform', 'file_type', 'size',
            'verdict', 'confidence', 'final_score',
            'lgbm_score', 'malconv_score', 'sha256', 'scan_time_s',
        ]
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            for result in self.results:
                row = {key: result.get(key, '') for key in fields}
                row.setdefault('timestamp', self.timestamp)
                writer.writerow(row)
        logger.info(f"CSV report -> {path}")
        return path

    def save_text(self, path: str = None) -> str:
        path = path or os.path.join(REPORT_DIR, f'scan_{self.timestamp}.txt')
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self._format_text())
        logger.info(f"Text report -> {path}")
        return path

    def print_summary(self) -> None:
        print(self._format_summary())

    def _summary(self) -> dict:
        verdicts = [result.get('verdict', 'UNKNOWN') for result in self.results]
        scores = [
            result.get('final_score', 0) for result in self.results
            if result.get('final_score') is not None
        ]
        return {
            'total': len(self.results),
            'malware': verdicts.count('MALWARE'),
            'suspicious': verdicts.count('SUSPICIOUS'),
            'clean': verdicts.count('CLEAN'),
            'avg_score': round(sum(scores) / len(scores), 4) if scores else 0,
            'max_score': round(max(scores), 4) if scores else 0,
        }

    def _format_summary(self) -> str:
        summary = self._summary()
        lines = [
            '',
            '  +-----------------------------------+',
            '  |     CROSSGUARD SCAN SUMMARY      |',
            '  +-----------------------------------+',
            f"  |  Total scanned:  {summary['total']:<15} |",
            f"  |  Malware:        {summary['malware']:<15} |",
            f"  |  Suspicious:     {summary['suspicious']:<15} |",
            f"  |  Clean:          {summary['clean']:<15} |",
            f"  |  Avg score:      {summary['avg_score']:<15} |",
            f"  |  Max score:      {summary['max_score']:<15} |",
            '  +-----------------------------------+',
            '',
        ]
        return '\n'.join(lines)

    def _format_text(self) -> str:
        sep = '-' * 80
        lines = [
            '=' * 80,
            '  CROSSGUARD - Scan Report',
            f'  Generated: {_utc_now().isoformat()}',
            '=' * 80,
            self._format_summary(),
            '',
            '  DETAILED RESULTS',
            sep,
        ]

        order = {'MALWARE': 0, 'SUSPICIOUS': 1, 'CLEAN': 2}
        sorted_results = sorted(
            self.results,
            key=lambda result: (
                order.get(result.get('verdict', 'CLEAN'), 3),
                -result.get('final_score', 0),
            ),
        )

        for result in sorted_results:
            verdict = result.get('verdict', 'UNKNOWN')
            label = {
                'MALWARE': '[MALWARE]',
                'SUSPICIOUS': '[SUSPECT]',
                'CLEAN': '[  CLEAN]',
            }.get(verdict, '[?]')
            lines += [
                f"  {label}  {result.get('filename', 'unknown')}",
                f"           Score: {result.get('final_score', 0):.4f}  "
                f"| Platform: {result.get('platform', '?')}  "
                f"| Size: {result.get('size', '?')}  "
                f"| Confidence: {result.get('confidence', '?')}",
                f"           LightGBM={_format_score(result.get('lgbm_score'))}  "
                f"MalConv2={_format_score(result.get('malconv_score'))}",
                f"           SHA256: {result.get('sha256', '?')[:64]}",
                '',
            ]

        lines.append('=' * 80)
        return '\n'.join(lines)


def _format_score(score) -> str:
    if score is None:
        return 'unavailable'
    return f"{score:.4f}"
