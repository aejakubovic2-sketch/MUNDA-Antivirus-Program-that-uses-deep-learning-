"""
settings.py
Central configuration for CrossGuard.
All paths, thresholds, model params, and runtime settings live here.
Override any setting via environment variables.
"""

import os

# ── Base paths ───────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, 'data')
LOG_DIR     = os.path.join(BASE_DIR, 'logs')
REPORT_DIR  = os.path.join(BASE_DIR, 'reports')

# Model subdirectories
LGBM_DIR      = os.path.join(DATA_DIR, 'lgbm')
MALCONV_DIR   = os.path.join(DATA_DIR, 'malconv2')
ENSEMBLE_DIR  = os.path.join(DATA_DIR, 'ensemble')
DATASET_DIR   = os.path.join(DATA_DIR, 'ember2024')

def _safe_makedirs(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except PermissionError:
        pass


# Ensure directories exist when possible without making imports fail.
for _d in [DATA_DIR, LOG_DIR, REPORT_DIR, LGBM_DIR, MALCONV_DIR, ENSEMBLE_DIR, DATASET_DIR]:
    _safe_makedirs(_d)

# ── HuggingFace repos ────────────────────────────────────
HF_EMBER2024_MODELS = 'joyce8/EMBER2024-benchmark-models'
HF_MALCONV2         = 'FutureComputing4AI/MalConvGCT'

# ── Detection thresholds ─────────────────────────────────
THRESHOLD_MALWARE    = 0.75   # score >= this → MALWARE
THRESHOLD_SUSPICIOUS = 0.40   # score >= this → SUSPICIOUS (else CLEAN)

# ── Scanning limits ──────────────────────────────────────
MAX_FILE_SIZE_MB   = 50       # skip files larger than this
MAX_FILE_BYTES     = 2_000_000  # MalConv2 input truncation (2 MB)
SCAN_TIMEOUT_S     = 30       # seconds before a scan is aborted

# ── MalConv2 architecture ────────────────────────────────
MALCONV_NUM_FILTERS  = 128
MALCONV_FILTER_SIZE  = 500
MALCONV_EMB_SIZE     = 8

# ── LightGBM training defaults ───────────────────────────
LGBM_PARAMS = {
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
LGBM_NUM_ROUNDS      = 500
LGBM_EARLY_STOPPING  = 20

# ── Ensemble weights (weighted method) ───────────────────
ENSEMBLE_LGBM_WEIGHT    = 0.40
ENSEMBLE_MALCONV_WEIGHT = 0.60

# ── Dashboard ────────────────────────────────────────────
DASHBOARD_HOST = os.environ.get('CROSSGUARD_HOST', '0.0.0.0')
DASHBOARD_PORT = int(os.environ.get('CROSSGUARD_PORT', 5000))
DASHBOARD_MAX_UPLOAD_MB = 50

# ── Logging ──────────────────────────────────────────────
LOG_LEVEL       = os.environ.get('CROSSGUARD_LOG_LEVEL', 'INFO')
LOG_FILE        = os.path.join(LOG_DIR, 'crossguard.log')
SCAN_LOG_FILE   = os.path.join(LOG_DIR, 'scans.jsonl')  # one JSON per line

# ── Supported file types ─────────────────────────────────
SUPPORTED_TYPES = ['WIN32', 'WIN64', 'DOTNET', 'ELF', 'APK', 'PDF']

PLATFORM_LABELS = {
    'WIN32':   'Windows 32-bit',
    'WIN64':   'Windows 64-bit',
    'DOTNET':  'Windows .NET',
    'ELF':     'Linux',
    'APK':     'Android',
    'PDF':     'Document',
    'MACHO':   'macOS Mach-O',
    'DMG':     'macOS disk image',
    'MACAPP':  'macOS app bundle',
    'UNKNOWN': 'Unknown',
}

# ── Feature vector ───────────────────────────────────────
EMBER_VECTOR_SIZE   = 2568   # EMBER v3 full feature vector dims
EMBER_NONPE_DIMS    = 696    # dims populated for non-PE files

# ── Monitoring / alerts ──────────────────────────────────
ALERT_ON_MALWARE    = True
ALERT_EMAIL         = os.environ.get('CROSSGUARD_ALERT_EMAIL', '')
SMTP_HOST           = os.environ.get('CROSSGUARD_SMTP_HOST', '')
SMTP_PORT           = int(os.environ.get('CROSSGUARD_SMTP_PORT', 587))
SMTP_USER           = os.environ.get('CROSSGUARD_SMTP_USER', '')
SMTP_PASS           = os.environ.get('CROSSGUARD_SMTP_PASS', '')

# ── Watchdog (real-time folder monitor) ──────────────────
WATCHDOG_DEBOUNCE_S = 1.0    # seconds to wait after file write before scanning
