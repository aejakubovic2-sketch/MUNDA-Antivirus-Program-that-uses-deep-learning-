"""
lightgbm_model.py
Wraps the EMBER2024 pretrained LightGBM classifier.
Downloads from HuggingFace if not already cached.
"""

import os
import numpy as np

MODEL_CACHE = os.path.join(os.path.dirname(__file__), 'data', 'lgbm')
HUGGINGFACE_REPO = 'joyce8/EMBER2024-benchmark-models'


class LightGBMDetector:
    """
    Loads the EMBER2024 pretrained LightGBM model.
    Predicts a malware probability score (0.0 = clean, 1.0 = malware).
    """

    # Map file type → which of the 14 EMBER2024 models to use
    MODEL_MAP = {
        'WIN32':   'lgbm_win32.txt',
        'WIN64':   'lgbm_win64.txt',
        'DOTNET':  'lgbm_dotnet.txt',
        'ELF':     'lgbm_elf.txt',
        'APK':     'lgbm_apk.txt',
        'PDF':     'lgbm_pdf.txt',
        'UNKNOWN': 'lgbm_all.txt',    # fallback: all-format model
    }

    def __init__(self):
        self._models = {}   # lazy-loaded per file type

    def predict(self, feature_vector: np.ndarray, file_type: str = 'UNKNOWN') -> float:
        """
        Args:
            feature_vector: numpy array of shape (2568,)
            file_type: one of WIN32, WIN64, DOTNET, ELF, APK, PDF, UNKNOWN
        Returns:
            float in [0, 1] — probability of being malware
        """
        model = self._load_model(file_type)
        score = model.predict(feature_vector.reshape(1, -1))[0]
        # LightGBM returns raw probability for binary classification
        return float(np.clip(score, 0.0, 1.0))

    def _load_model(self, file_type: str):
        if file_type not in self._models:
            filename = self.MODEL_MAP.get(file_type, self.MODEL_MAP['UNKNOWN'])
            model_path = os.path.join(MODEL_CACHE, filename)

            if not os.path.isfile(model_path):
                self._download_model(filename, model_path)

            import lightgbm as lgb
            self._models[file_type] = lgb.Booster(model_file=model_path)

        return self._models[file_type]

    def _download_model(self, filename: str, dest_path: str):
        """Download pretrained model from HuggingFace hub."""
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        try:
            from huggingface_hub import hf_hub_download
            print(f"[LightGBM] Downloading {filename} from HuggingFace...")
            hf_hub_download(
                repo_id=HUGGINGFACE_REPO,
                filename=filename,
                local_dir=MODEL_CACHE,
            )
            print(f"[LightGBM] Downloaded → {dest_path}")
        except Exception as e:
            raise RuntimeError(
                f"Could not download {filename}.\n"
                f"Please manually download from: "
                f"https://huggingface.co/{HUGGINGFACE_REPO}\n"
                f"and place it at: {dest_path}\n"
                f"Error: {e}"
            )

    def is_available(self) -> bool:
        """Check if at least the all-format fallback model exists."""
        return os.path.isfile(
            os.path.join(MODEL_CACHE, self.MODEL_MAP['UNKNOWN'])
        )

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              file_type: str = 'all') -> None:
        """
        Train a new LightGBM model on provided data.
        Used when fine-tuning on custom data.

        Args:
            X_train: feature matrix (N, 2568)
            y_train: labels (N,) — 0=benign, 1=malware
            X_val:   optional validation features
            y_val:   optional validation labels
            file_type: tag for saving the model
        """
        import lightgbm as lgb

        params = {
            'objective':        'binary',
            'metric':           'auc',
            'boosting_type':    'gbdt',
            'num_leaves':       64,
            'learning_rate':    0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq':     5,
            'verbose':          -1,
            'n_jobs':           -1,
        }

        dtrain = lgb.Dataset(X_train, label=y_train)
        callbacks = [lgb.log_evaluation(period=50)]

        eval_sets = []
        if X_val is not None and y_val is not None:
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
            eval_sets = [dval]
            callbacks.append(lgb.early_stopping(stopping_rounds=20))

        print(f"[LightGBM] Training on {len(X_train):,} samples...")
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=500,
            valid_sets=eval_sets,
            callbacks=callbacks,
        )

        os.makedirs(MODEL_CACHE, exist_ok=True)
        save_path = os.path.join(MODEL_CACHE, f'lgbm_{file_type}.txt')
        model.save_model(save_path)
        self._models[file_type] = model
        print(f"[LightGBM] Model saved → {save_path}")
