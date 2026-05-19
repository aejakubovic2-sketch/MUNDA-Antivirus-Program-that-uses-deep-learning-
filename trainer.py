"""
trainer.py
Full training pipeline:
  1. Download EMBER2024 dataset
  2. Train LightGBM on EMBER features
  3. Fine-tune MalConv2 on raw binaries
  4. Train meta-learner ensemble
  5. Evaluate on test set + challenge set
"""

import os
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def download_dataset():
    """
    Download EMBER2024 dataset from HuggingFace.
    Requires: pip install -r requirements.txt
    """
    print("[Trainer] Downloading EMBER2024 dataset...")
    print("[Trainer] This may take a while (3.2M files).")
    try:
        import thrember
        thrember.download_dataset(DATA_DIR)
        print("[Trainer] Dataset downloaded.")
    except ImportError:
        print("[Trainer] thrember not installed. Run: pip install -r requirements.txt")
        print("[Trainer] Or clone: https://github.com/FutureComputing4AI/EMBER2024")
    except Exception as e:
        print(f"[Trainer] Download failed: {e}")


def download_pretrained_models():
    """
    Download the pretrained LightGBM models and the MalConv2 checkpoint.
    """
    print("[Trainer] Downloading EMBER2024 pretrained LightGBM models...")
    try:
        from lightgbm_model import LightGBMDetector
        LightGBMDetector().download_all()
        print("[Trainer] LightGBM models downloaded.")
    except ModuleNotFoundError as e:
        print(f"[Trainer] {e.name} not installed. Run: pip install -r requirements.txt")
    except Exception as e:
        print(f"[Trainer] LightGBM download failed: {e}")

    print("[Trainer] MalConv2 checkpoint download skipped.")
    print("[Trainer] No public MalConv2 checkpoint is configured; scans will use LightGBM unless you train or add one.")


def train_lightgbm(file_type: str = 'all'):
    """
    Train LightGBM on vectorized EMBER2024 features.
    Requires dataset to be downloaded first.
    """
    import thrember
    from lightgbm_model import LightGBMDetector

    print(f"[Trainer] Loading EMBER2024 features for: {file_type}")
    X_train, y_train = thrember.read_vectorized_features(
        DATA_DIR, subset='train', file_type=file_type
    )
    X_test, y_test = thrember.read_vectorized_features(
        DATA_DIR, subset='test', file_type=file_type
    )

    detector = LightGBMDetector()
    detector.train(X_train, y_train, X_test, y_test, file_type=file_type)

    # Evaluate
    from sklearn.metrics import roc_auc_score
    preds = np.array([detector.predict(x, file_type) for x in X_test])
    auc   = roc_auc_score(y_test, preds)
    print(f"[Trainer] LightGBM ROC AUC on test set: {auc:.4f}")
    return auc


def train_all():
    """
    Train LightGBM for every supported file type.
    """
    file_types = ['all', 'win32', 'win64', 'dotnet', 'elf', 'apk', 'pdf']
    results = {}
    for ft in file_types:
        print(f"\n{'='*50}")
        print(f"Training LightGBM for: {ft.upper()}")
        print('='*50)
        try:
            auc = train_lightgbm(ft)
            results[ft] = auc
        except Exception as e:
            print(f"[Trainer] Failed for {ft}: {e}")
            results[ft] = None
    print("\n[Trainer] Training complete. Results:")
    for ft, auc in results.items():
        status = f"{auc:.4f}" if auc else "FAILED"
        print(f"  {ft:<10}: AUC = {status}")
    return results


def evaluate_challenge_set():
    """
    Evaluate the ensemble on the EMBER2024 challenge set.
    These are evasive malware samples that initially fooled ~70 AV products.
    This is the hardest test.
    """
    import thrember
    from scanner import Scanner
    from sklearn.metrics import roc_auc_score, average_precision_score

    print("[Trainer] Loading EMBER2024 challenge set...")
    X_challenge, y_challenge = thrember.read_vectorized_features(
        DATA_DIR, subset='challenge'
    )

    scanner = Scanner(method='meta')
    scores = []
    import os, tempfile
    for fv in X_challenge:
        # We only have feature vectors for challenge set (no raw files)
        # So use LightGBM only for this evaluation
        score = scanner.lgbm.predict(fv, 'UNKNOWN')
        scores.append(score)

    scores = np.array(scores)
    auc_roc = roc_auc_score(y_challenge, scores)
    auc_pr  = average_precision_score(y_challenge, scores)

    print(f"\n[Trainer] Challenge Set Results (evasive malware):")
    print(f"  ROC AUC : {auc_roc:.4f}")
    print(f"  PR  AUC : {auc_pr:.4f}  (main metric for evasive malware)")
    print(f"  Note: PR AUC ~0.57 is expected per the EMBER2024 paper.")
    return {'roc_auc': auc_roc, 'pr_auc': auc_pr}
