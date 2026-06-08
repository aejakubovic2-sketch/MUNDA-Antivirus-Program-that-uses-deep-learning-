"""
evaluator.py
Comprehensive evaluation of the CrossGuard ensemble.
Computes ROC AUC, PR AUC, TPR at fixed FPR, confusion matrix,
per-platform breakdown, and generates a full PDF/text report.
"""

import os
import json
import time
import numpy as np
from datetime import datetime

from settings import (
    THRESHOLD_MALWARE, THRESHOLD_SUSPICIOUS,
    REPORT_DIR,
)
from logger import get_logger
from ember_compat import get_thrember

logger = get_logger('evaluator')


# ── Core metrics ─────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_scores: np.ndarray) -> dict:
    """
    Compute a full suite of detection metrics.

    Args:
        y_true:   ground truth labels (0=benign, 1=malware)
        y_scores: predicted probabilities (0.0–1.0)

    Returns:
        dict of metrics
    """
    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores).astype(float)

    y_pred = (y_scores >= THRESHOLD_MALWARE).astype(int)

    try:
        from sklearn.metrics import (
            roc_auc_score, average_precision_score, roc_curve,
        )
        roc_auc = roc_auc_score(y_true, y_scores)
        pr_auc = average_precision_score(y_true, y_scores)
        fpr_arr, tpr_arr, _thresholds = roc_curve(y_true, y_scores)
    except ModuleNotFoundError:
        roc_auc = _roc_auc_numpy(y_true, y_scores)
        pr_auc = _average_precision_numpy(y_true, y_scores)
        fpr_arr, tpr_arr = _roc_curve_numpy(y_true, y_scores)

    # TPR at fixed FPR thresholds (industry standard)
    tpr_at_1pct_fpr  = _tpr_at_fpr(fpr_arr, tpr_arr, 0.01)
    tpr_at_01pct_fpr = _tpr_at_fpr(fpr_arr, tpr_arr, 0.001)

    # Confusion matrix
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr       = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    accuracy  = (tp + tn) / len(y_true)

    return {
        'roc_auc':         round(roc_auc, 6),
        'pr_auc':          round(pr_auc, 6),
        'tpr_at_1pct_fpr': round(tpr_at_1pct_fpr, 6),
        'tpr_at_01pct_fpr':round(tpr_at_01pct_fpr, 6),
        'accuracy':        round(accuracy, 6),
        'precision':       round(precision, 6),
        'recall':          round(recall, 6),
        'f1':              round(f1, 6),
        'fpr':             round(fpr, 6),
        'fnr':             round(fnr, 6),
        'tp': int(tp), 'tn': int(tn),
        'fp': int(fp), 'fn': int(fn),
        'total':           int(len(y_true)),
        'threshold_used':  THRESHOLD_MALWARE,
    }


def _tpr_at_fpr(fpr_arr, tpr_arr, target_fpr: float) -> float:
    """Return the TPR at the closest FPR value to target_fpr."""
    idx = np.searchsorted(fpr_arr, target_fpr)
    if idx >= len(tpr_arr):
        return float(tpr_arr[-1])
    return float(tpr_arr[idx])


def _roc_auc_numpy(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    pos = y_scores[y_true == 1]
    neg = y_scores[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0

    comparisons = pos[:, None] - neg[None, :]
    wins = np.sum(comparisons > 0)
    ties = np.sum(comparisons == 0)
    return float((wins + 0.5 * ties) / comparisons.size)


def _average_precision_numpy(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    positives = np.sum(y_true == 1)
    if positives == 0:
        return 0.0

    order = np.argsort(-y_scores, kind='mergesort')
    sorted_true = y_true[order]
    tp = np.cumsum(sorted_true == 1)
    precision = tp / (np.arange(len(sorted_true)) + 1)
    return float(np.sum(precision[sorted_true == 1]) / positives)


def _roc_curve_numpy(y_true: np.ndarray, y_scores: np.ndarray):
    positives = np.sum(y_true == 1)
    negatives = np.sum(y_true == 0)
    if positives == 0 or negatives == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 0.0])

    order = np.argsort(-y_scores, kind='mergesort')
    sorted_true = y_true[order]
    tp = np.r_[0, np.cumsum(sorted_true == 1)]
    fp = np.r_[0, np.cumsum(sorted_true == 0)]
    return fp / negatives, tp / positives


# ── Per-platform evaluation ───────────────────────────────

def evaluate_by_platform(results: list) -> dict:
    """
    Given a list of scan result dicts (with 'verdict', 'final_score',
    'platform', and 'label' keys), compute metrics per platform.
    """
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in results:
        grouped[r.get('platform', 'Unknown')].append(r)

    platform_metrics = {}
    for platform, items in grouped.items():
        y_true   = np.array([r['label'] for r in items])
        y_scores = np.array([r['final_score'] for r in items])
        if len(np.unique(y_true)) < 2:
            continue   # need both classes to compute AUC
        platform_metrics[platform] = compute_metrics(y_true, y_scores)

    return platform_metrics


# ── Full benchmark runner ────────────────────────────────

class Evaluator:
    """
    Runs the full CrossGuard evaluation pipeline:
      1. Load EMBER2024 test set
      2. Run ensemble on all test samples
      3. Compute metrics (overall + per-platform)
      4. Run challenge set (evasive malware)
      5. Save report
    """

    def __init__(self, scanner=None):
        self.scanner = scanner
        self._results = []

    def run(self, max_samples: int = None) -> dict:
        """
        Full evaluation. Returns dict of all metrics.
        """
        logger.info("Starting MUNDA evaluation...")
        start = time.time()

        # Load EMBER2024 test features
        try:
            from settings import DATASET_DIR
            thrember = get_thrember()
            X_test, y_test = thrember.read_vectorized_features(
                DATASET_DIR, subset='test'
            )
        except ImportError:
            logger.error("thrember not installed. Run: pip install -r requirements.txt")
            return {}
        except Exception as e:
            logger.error(f"Could not load test set: {e}")
            return {}

        if max_samples:
            X_test = X_test[:max_samples]
            y_test = y_test[:max_samples]

        logger.info(f"Evaluating on {len(X_test):,} test samples...")

        # Get scores from scanner's LightGBM (feature-based)
        lgbm_scores = []
        for fv in X_test:
            score = self.scanner.lgbm.predict(fv, 'UNKNOWN')
            lgbm_scores.append(score)

        lgbm_scores = np.array(lgbm_scores)
        metrics = compute_metrics(y_test, lgbm_scores)
        metrics['eval_time_s'] = round(time.time() - start, 2)
        metrics['num_samples'] = len(X_test)

        # Challenge set (evasive malware)
        challenge_metrics = self._eval_challenge()

        report = {
            'timestamp':        datetime.utcnow().isoformat(),
            'overall':          metrics,
            'challenge_set':    challenge_metrics,
        }

        self._save_report(report)
        self._print_report(report)
        return report

    def _eval_challenge(self) -> dict:
        logger.info("Evaluating on EMBER2024 challenge set (evasive malware)...")
        try:
            from settings import DATASET_DIR
            thrember = get_thrember()
            X_ch, y_ch = thrember.read_vectorized_features(
                DATASET_DIR, subset='challenge'
            )
            scores = np.array([
                self.scanner.lgbm.predict(fv, 'UNKNOWN') for fv in X_ch
            ])
            m = compute_metrics(y_ch, scores)
            logger.info(f"Challenge set ROC AUC: {m['roc_auc']}")
            logger.info(f"Challenge set PR  AUC: {m['pr_auc']} "
                        f"(expected ~0.57 per EMBER2024 paper)")
            return m
        except Exception as e:
            logger.warning(f"Challenge set evaluation skipped: {e}")
            return {}

    def _save_report(self, report: dict) -> None:
        ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        json_path = os.path.join(REPORT_DIR, f'eval_{ts}.json')
        txt_path  = os.path.join(REPORT_DIR, f'eval_{ts}.txt')

        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)

        with open(txt_path, 'w') as f:
            f.write(self._format_report(report))

        logger.info(f"Report saved → {json_path}")
        logger.info(f"Report saved → {txt_path}")

    @staticmethod
    def _format_report(report: dict) -> str:
        m  = report.get('overall', {})
        ch = report.get('challenge_set', {})
        ts = report.get('timestamp', '')

        lines = [
            '=' * 60,
            '  CROSSGUARD — Evaluation Report',
            f'  Generated: {ts}',
            '=' * 60,
            '',
            '  TEST SET RESULTS',
            '  ─────────────────────────────────',
            f"  Samples:         {m.get('total', 0):,}",
            f"  ROC AUC:         {m.get('roc_auc', 0):.4f}",
            f"  PR  AUC:         {m.get('pr_auc', 0):.4f}",
            f"  TPR @ 1%  FPR:   {m.get('tpr_at_1pct_fpr', 0):.4f}",
            f"  TPR @ 0.1% FPR:  {m.get('tpr_at_01pct_fpr', 0):.4f}",
            f"  Accuracy:        {m.get('accuracy', 0):.4f}",
            f"  Precision:       {m.get('precision', 0):.4f}",
            f"  Recall:          {m.get('recall', 0):.4f}",
            f"  F1:              {m.get('f1', 0):.4f}",
            f"  FPR:             {m.get('fpr', 0):.4f}",
            f"  FNR:             {m.get('fnr', 0):.4f}",
            '',
            '  CONFUSION MATRIX',
            '  ─────────────────────────────────',
            f"  TP (correct malware):  {m.get('tp', 0):,}",
            f"  TN (correct benign):   {m.get('tn', 0):,}",
            f"  FP (false alarms):     {m.get('fp', 0):,}",
            f"  FN (missed malware):   {m.get('fn', 0):,}",
        ]

        if ch:
            lines += [
                '',
                '  CHALLENGE SET (evasive malware)',
                '  ─────────────────────────────────',
                f"  Samples:  {ch.get('total', 0):,}",
                f"  ROC AUC:  {ch.get('roc_auc', 0):.4f}",
                f"  PR  AUC:  {ch.get('pr_auc', 0):.4f}  "
                "(~0.57 expected per EMBER2024 paper)",
            ]

        lines += ['', '=' * 60]
        return '\n'.join(lines)

    @staticmethod
    def _print_report(report: dict) -> None:
        print('\n' + Evaluator._format_report(report) + '\n')
