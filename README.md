# MUNDA — Cross-Platform Malware Detector
## Malware detection dashboard using EMBER2024 features and LightGBM

---

## Overview
MUNDA is a cross-platform malware detection system that uses:
- **EMBER2024 dataset** — 3.2M files across Windows, Linux, Android, PDF
- **LightGBM** — Fast feature-based detection with pretrained EMBER2024 models
- **Flask dashboard** — Local browser interface for uploading and scanning files
- **MalConv2 support** — Present in the code, but disabled by default unless a compatible checkpoint is added

Unsupported file types are reported as `UNSUPPORTED` with an `N/A` score instead of receiving a misleading threat percentage.

## Supported Platforms
| Platform | Format |
|---|---|
| Windows 32/64-bit | .exe, .dll, .sys |
| Android | .apk |
| Linux | .elf, ELF binaries |
| Documents | .pdf |
| .NET apps | .NET assemblies |

## Project Structure
```
MUNDA/
├── app.py                     # Flask backend
├── index.html                 # Dashboard frontend
├── main.py                    # CLI entry point
├── scanner.py                 # Main scanning engine
├── feature_extractor.py       # EMBER feature extraction
├── file_identifier.py         # Detect file type (PE/ELF/APK/PDF)
├── lightgbm_model.py          # LightGBM model loader/predictor
├── malconv2_model.py          # Optional MalConv2 model support
├── ensemble.py                # Combines available model scores
├── data/                      # Downloaded models/dataset (not committed)
├── requirements.txt
└── README.md
```

## Quick Start
```bash
# 1. Install dependencies (Python 3.12 recommended)
python3.12 -m pip install -r requirements.txt

# 2. Download pretrained LightGBM models
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
