"""
reporter.py
Generates scan reports in multiple formats: JSON, CSV, plain text.
For PDF output use the pdf skill (requires reportlab).
"""

import os
import csv
import json
from datetime import datetime
from typing import List, Dict

from settings import REPORT_DIR
from logger import get_logger

logger = get_logger('reporter')


class ReportGenerator:
    """
    Generates scan reports from a list of scan result dicts.
    Supports JSON, CSV, and plain text output.
    """

    def __init__(self, results: List[Dict]):
        self.results   = results
        self.timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        os.makedirs(REPORT_DIR, exist_ok=True)

    # ── Public API ───────────────────────────────────────

    def save_json(self, path: str = None) -> str:
        path = path or os.path.join(REPORT_DIR, f'scan_{self.timestamp}.json')
        payload = {
            'generated':   datetime.utcnow().isoformat(),
            'total_scans': len(self.results),
            'summary':     self._summary(),
            'results':     self.results,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)
        logger.info(f"JSON report → {path}")
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
            for r in self.results:
                row = {k: r.get(k, '') for k in fields}
                row.setdefault('timestamp', self.timestamp)
                writer.writerow(row)
        logger.info(f"CSV report → {path}")
        return path

    def save_text(self, path: str = None) -> str:
        path = path or os.path.join(REPORT_DIR, f'scan_{self.timestamp}.txt')
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self._format_text())
        logger.info(f"Text report → {path}")
        return path

    def print_summary(self) -> None:
        print(self._format_summary())

    # ── Internals ────────────────────────────────────────

    def _summary(self) -> dict:
        verdicts = [r.get('verdict', 'UNKNOWN') for r in self.results]
        scores = [
            r.get('final_score', 0) for r in self.results
            if r.get('final_score') is not None
        ]
        return {
            'total':      len(self.results),
            'malware':    verdicts.count('MALWARE'),
            'suspicious': verdicts.count('SUSPICIOUS'),
            'clean':      verdicts.count('CLEAN'),
            'avg_score':  round(sum(scores) / len(scores), 4) if scores else 0,
            'max_score':  round(max(scores), 4) if scores else 0,
        }

    def _format_summary(self) -> str:
        s = self._summary()
        lines = [
            '',
            '  ╔══════════════════════════════════╗',
            '  ║     CROSSGUARD SCAN SUMMARY      ║',
            '  ╠══════════════════════════════════╣',
            f"  ║  Total scanned:  {s['total']:<15} ║",
            f"  ║  🚨 Malware:     {s['malware']:<15} ║",
            f"  ║  ⚠️  Suspicious:  {s['suspicious']:<15} ║",
            f"  ║  ✅ Clean:       {s['clean']:<15} ║",
            f"  ║  Avg score:     {s['avg_score']:<15} ║",
            f"  ║  Max score:     {s['max_score']:<15} ║",
            '  ╚══════════════════════════════════╝',
            '',
        ]
        return '\n'.join(lines)

    def _format_text(self) -> str:
        sep = '-' * 80
        lines = [
            '=' * 80,
            '  CROSSGUARD — Scan Report',
            f'  Generated: {datetime.utcnow().isoformat()}',
            '=' * 80,
            self._format_summary(),
            '',
            '  DETAILED RESULTS',
            sep,
        ]

        # Sort: malware first, then suspicious, then clean
        order = {'MALWARE': 0, 'SUSPICIOUS': 1, 'CLEAN': 2}
        sorted_results = sorted(
            self.results,
            key=lambda r: (order.get(r.get('verdict', 'CLEAN'), 3),
                           -r.get('final_score', 0))
        )

        for r in sorted_results:
            v = r.get('verdict', 'UNKNOWN')
            icon = {'MALWARE': '[MALWARE]', 'SUSPICIOUS': '[SUSPECT]',
                    'CLEAN': '[  CLEAN]'}.get(v, '[?]')
            lines += [
                f"  {icon}  {r.get('filename', 'unknown')}",
                f"           Score: {r.get('final_score', 0):.4f}  "
                f"| Platform: {r.get('platform', '?')}  "
                f"| Size: {r.get('size', '?')}  "
                f"| Confidence: {r.get('confidence', '?')}",
                f"           LightGBM={_format_score(r.get('lgbm_score'))}  "
                f"MalConv2={_format_score(r.get('malconv_score'))}",
                f"           SHA256: {r.get('sha256', '?')[:64]}",
                '',
            ]

        lines.append('=' * 80)
        return '\n'.join(lines)


def _format_score(score) -> str:
    if score is None:
        return 'unavailable'
    return f"{score:.4f}"
