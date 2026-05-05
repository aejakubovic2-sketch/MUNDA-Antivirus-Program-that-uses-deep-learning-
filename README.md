# CrossGuard — Cross-Platform Malware Detector
## Deep Learning Antivirus using EMBER2024 + LightGBM + MalConv2

---

## Overview
!!!!!THE CURRENT BUILD DOESNT WORK ONLY THE DASHBOARD FUNCTION BUT THE MODELS THEMSELFS DONT WORK!!!!
CrossGuard is a cross-platform malware detection system that combines:
- **EMBER2024 dataset** — 3.2M files across Windows, Linux, Android, PDF
- **LightGBM** — Fast feature-based detection (99.69% AUC)
- **MalConv2** — Deep raw-byte neural network (99.82% AUC)
- **Meta-Learner Ensemble** — Combines both for ~99.9%+ AUC

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
malware_detector/
├── core/
│   ├── feature_extractor.py   # EMBER v3 feature extraction
│   └── file_identifier.py     # Detect file type (PE/ELF/APK/PDF)
├── models/
│   ├── lightgbm_model.py      # LightGBM wrapper + pretrained loader
│   ├── malconv2_model.py      # MalConv2 deep learning model
│   └── ensemble.py            # Meta-learner ensemble (best accuracy)
├── scanner/
│   └── scanner.py             # Main scanning engine
├── dashboard/
│   └── app.py                 # Web dashboard (Flask)
│   └── templates/index.html   # Frontend UI
├── utils/
│   ├── dataset.py             # EMBER2024 dataset downloader/loader
│   └── trainer.py             # Training pipeline
├── data/                      # Models + dataset stored here
├── requirements.txt
└── main.py                    # Entry point
```

## Quick Start
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download pretrained models
python main.py --download-models

# 3. Scan a file
python main.py --scan /path/to/suspicious.exe

# 4. Launch web dashboard
python main.py --dashboard
```

## Training Your Own Model
```bash
# Download EMBER2024 dataset
python main.py --download-dataset

# Train the full ensemble
python main.py --train

# Evaluate on test + challenge set
python main.py --evaluate
```
