"""
ensemble.py
Meta-learner ensemble that combines LightGBM + MalConv2 scores.
Three combination methods: simple average, weighted average, meta-learner (best).
"""

import os
import json
import numpy as np

ENSEMBLE_CACHE = os.path.join(os.path.dirname(__file__), 'data', 'ensemble')
MALCONV_SUPPORTED_TYPES = {'WIN32', 'WIN64', 'DOTNET'}


class EnsembleDetector:
    """
    Combines LightGBM + MalConv2 predictions using a meta-learner.

    Methods:
      - 'average'   : (lgbm + malconv) / 2
      - 'weighted'  : 0.4*lgbm + 0.6*malconv
      - 'meta'      : small LightGBM trained on both scores (best accuracy)
    """

    def __init__(self, lgbm_detector, malconv_detector, method: str = 'meta'):
        self.lgbm     = lgbm_detector
        self.malconv  = malconv_detector
        self.method   = method
        self._meta    = None   # meta-learner model (loaded lazily)

        # Weights for weighted average method
        self.lgbm_weight    = 0.40
        self.malconv_weight = 0.60

    # ── Main predict ─────────────────────────────────────

    def predict(self,
                feature_vector: np.ndarray,
                filepath: str,
                file_type: str = 'UNKNOWN') -> dict:
        """
        Run both models and combine their scores.

        Returns dict:
          {
            'final_score':    float,   # 0=clean, 1=malware
            'lgbm_score':     float | None,
            'malconv_score':  float | None,
            'verdict':        str,     # 'MALWARE' | 'SUSPICIOUS' | 'CLEAN'
            'confidence':     str,     # 'HIGH' | 'MEDIUM' | 'LOW'
            'method':         str,
          }
        """
        scores = {}
        errors = {}

        try:
            scores['lgbm'] = self.lgbm.predict(feature_vector, file_type)
        except Exception as e:
            errors['lgbm'] = str(e)

        if file_type in MALCONV_SUPPORTED_TYPES:
            try:
                scores['malconv'] = self.malconv.predict(filepath)
            except Exception as e:
                errors['malconv'] = str(e)
        else:
            errors['malconv'] = (
                'MalConv2 is only used for Windows PE files '
                '(WIN32, WIN64, DOTNET).'
            )

        if not scores:
            details = '; '.join(
                f"{model}: {error}" for model, error in errors.items()
            )
            raise RuntimeError(
                "No malware detection model is available. "
                f"Model errors: {details}"
            )

        lgbm_score = scores.get('lgbm')
        malconv_score = scores.get('malconv')
        final_score = self._combine_available(scores)

        return {
            'final_score':   round(final_score, 4),
            'lgbm_score':    self._round_score(lgbm_score),
            'malconv_score': self._round_score(malconv_score),
            'verdict':       self._verdict(final_score),
            'confidence':    self._confidence(lgbm_score, malconv_score, final_score),
            'method':        self._method_used(scores),
            'model_errors':  errors,
        }

    # ── Score combination ────────────────────────────────

    def _combine(self, lgbm_score: float, malconv_score: float) -> float:
        if self.method == 'average':
            return (lgbm_score + malconv_score) / 2.0

        elif self.method == 'weighted':
            return (self.lgbm_weight * lgbm_score +
                    self.malconv_weight * malconv_score)

        elif self.method == 'meta':
            meta = self._load_meta()
            if meta is None:
                # Fallback to weighted if meta not trained yet
                return (self.lgbm_weight * lgbm_score +
                        self.malconv_weight * malconv_score)
            features = np.array([[lgbm_score, malconv_score]])
            return float(meta.predict(features)[0])

        raise ValueError(f"Unknown method: {self.method}")

    def _combine_available(self, scores: dict) -> float:
        if 'lgbm' in scores and 'malconv' in scores:
            return self._combine(scores['lgbm'], scores['malconv'])

        return next(iter(scores.values()))

    def _method_used(self, scores: dict) -> str:
        if 'lgbm' in scores and 'malconv' in scores:
            return self.method
        return next(iter(scores.keys()))

    # ── Individual model wrappers ────────────────────────

    def _get_lgbm_score(self, feature_vector: np.ndarray, file_type: str) -> float:
        return self.lgbm.predict(feature_vector, file_type)

    def _get_malconv_score(self, filepath: str) -> float:
        return self.malconv.predict(filepath)

    # ── Meta-learner training ────────────────────────────

    def train_meta(self,
                   filepaths:       list,
                   feature_vectors: np.ndarray,
                   labels:          list,
                   file_types:      list = None) -> None:
        """
        Train the meta-learner on (lgbm_score, malconv_score) pairs.

        Args:
            filepaths:        list of file paths
            feature_vectors:  np.ndarray of shape (N, 2568)
            labels:           list of int (0=benign, 1=malware)
            file_types:       optional list of file type strings
        """
        import lightgbm as lgb
        from tqdm import tqdm

        if file_types is None:
            file_types = ['UNKNOWN'] * len(filepaths)

        print("[Ensemble] Collecting base model scores for meta-learner training...")
        scores = []
        valid_labels = []

        for i, (fp, fv, ft, lbl) in enumerate(
                tqdm(zip(filepaths, feature_vectors, file_types, labels),
                     total=len(filepaths))):
            try:
                ls = self._get_lgbm_score(fv, ft)
                ms = self._get_malconv_score(fp)
                scores.append([ls, ms])
                valid_labels.append(lbl)
            except Exception as e:
                print(f"[Ensemble] Skipping {fp}: {e}")

        X = np.array(scores)
        y = np.array(valid_labels)

        print(f"[Ensemble] Training meta-learner on {len(X):,} samples...")
        meta = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=16,
            objective='binary',
            metric='auc',
            verbose=-1,
        )
        meta.fit(X, y)

        os.makedirs(ENSEMBLE_CACHE, exist_ok=True)
        save_path = os.path.join(ENSEMBLE_CACHE, 'meta_lgbm.txt')
        meta.booster_.save_model(save_path)
        self._meta = meta.booster_
        print(f"[Ensemble] Meta-learner saved → {save_path}")

    def _load_meta(self):
        if self._meta is None:
            path = os.path.join(ENSEMBLE_CACHE, 'meta_lgbm.txt')
            if os.path.isfile(path):
                import lightgbm as lgb
                self._meta = lgb.Booster(model_file=path)
        return self._meta

    def is_meta_trained(self) -> bool:
        return os.path.isfile(os.path.join(ENSEMBLE_CACHE, 'meta_lgbm.txt'))

    # ── Verdict helpers ──────────────────────────────────

    @staticmethod
    def _verdict(score: float) -> str:
        if score >= 0.75:
            return 'MALWARE'
        elif score >= 0.40:
            return 'SUSPICIOUS'
        return 'CLEAN'

    @staticmethod
    def _round_score(score) -> float:
        if score is None:
            return None
        return round(float(score), 4)

    @staticmethod
    def _confidence(lgbm: float, malconv: float, final: float) -> str:
        """
        HIGH   — both models agree strongly
        MEDIUM — both models agree but score is in mid-range
        LOW    — models disagree with each other
        """
        if lgbm is None or malconv is None:
            return EnsembleDetector._single_model_confidence(final)
        agreement = abs(lgbm - malconv)
        if agreement < 0.15 and (final > 0.80 or final < 0.20):
            return 'HIGH'
        elif agreement < 0.30:
            return 'MEDIUM'
        return 'LOW'

    @staticmethod
    def _single_model_confidence(score: float) -> str:
        """
        Confidence when only one detector is available.

        With no second model to compare against, confidence is based on how far
        the score is from the uncertain middle range rather than forced to LOW.
        """
        if score >= 0.85 or score <= 0.20:
            return 'HIGH'
        if score >= 0.60 or score <= 0.40:
            return 'MEDIUM'
        return 'LOW'
