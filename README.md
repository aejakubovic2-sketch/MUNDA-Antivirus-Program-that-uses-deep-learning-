# MUNDA — Cross-Platform Malware Detector
## Malware detection dashboard using EMBER2024 features, LightGBM, and MalConv2

---

## Overview
MUNDA is a cross-platform malware detection system that uses:
- **EMBER2024 dataset** — 3.2M files across Windows, Linux, Android, PDF
- **LightGBM** — Fast feature-based detection with pretrained EMBER2024 models
- **MalConv2** — Raw-byte deep learning model for Windows PE files
- **Flask dashboard** — Local browser interface for uploading and scanning files

Unsupported file types are reported as `UNSUPPORTED` with an `N/A` score instead of receiving a misleading threat percentage.

## Supported Platforms
This table describes the files MUNDA can scan, not the operating system running MUNDA. The dashboard can run on macOS, but macOS app formats are not currently supported scan targets.

| Target platform | Scan support |
|---|---|
| Windows 32/64-bit | Supported: `.exe`, `.dll`, `.sys` |
| Windows .NET apps | Supported: .NET assemblies |
| Android | Supported: `.apk` |
| Linux | Supported: ELF binaries, `.elf` |
| Documents | Supported: `.pdf` |
| macOS | Not supported yet: `.app`, `.dmg`, Mach-O binaries |

## Project Structure
```
MUNDA/
├── app.py                     # Flask backend
├── index.html                 # Dashboard frontend
├── main.py                    # CLI entry point
├── scanner.py                 # Main scanning engine
├── feature_extractor.py       # EMBER feature extraction
├── file_identifier.py         # Detect file type (PE/ELF/APK/PDF)
├── ember_compat.py            # EMBER2024 import compatibility helpers
├── lightgbm_model.py          # LightGBM model loader/predictor
├── malconv2_model.py          # MalConv2 checkpoint loader/predictor
├── ensemble.py                # Combines available model scores
├── data/                      # Downloaded models/dataset (not committed)
├── requirements.txt
└── README.md
```

## Quick Start
```bash
# 1. Install dependencies (Python 3.12 recommended)
python3.12 -m pip install -r requirements.txt

# 2. Download pretrained LightGBM models and MalConv2 checkpoint
python3.12 main.py --download-models

# 3. Scan a file
python3.12 main.py --scan /path/to/suspicious.exe

# 4. Launch web dashboard
python3.12 main.py --dashboard --port 5050
```

Then open:

```text
http://localhost:5050
```

On macOS, if LightGBM cannot find `libomp.dylib`, install OpenMP with:

```bash
brew install libomp
```

## Scoring
MUNDA scores supported files from `0%` to `100%`:

| Score | Verdict |
|---|---|
| 0–39% | Clean |
| 40–74% | Suspicious |
| 75–100% | Malware |

Confidence is based on how decisive the active model score is. If only LightGBM is available, MUNDA no longer forces confidence to `LOW`.

MalConv2 is only used for Windows PE files (`.exe`, `.dll`, `.sys`, and .NET assemblies). PDF, APK, and ELF files are scanned with the EMBER2024 LightGBM models because the public MalConv2 checkpoint was trained for PE-style raw binaries.

## Safe Testing
Do not download random malware onto a personal laptop. For safe testing, use harmless test files or controlled research datasets. The standard EICAR string is safe, but the plain `eicar.com` file is not a supported MUNDA file type and may show as `UNSUPPORTED`.

## Training Your Own Model
```bash
# Download EMBER2024 dataset
python3.12 main.py --download-dataset

# Train the full ensemble
python3.12 main.py --train

# Evaluate on test + challenge set
python3.12 main.py --evaluate
```
