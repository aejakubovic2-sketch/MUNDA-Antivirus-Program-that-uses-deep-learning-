# MUNDA 🛡️

### Cross-Platform Malware Detection System

MUNDA is a malware detection platform that combines traditional machine learning and deep learning models to analyze files from multiple operating systems and document formats.

Built using **EMBER2024**, **LightGBM**, **MalConv2**, **Python**, and **Flask**, MUNDA provides both a command-line interface and a web-based dashboard for malware analysis.

> **Note:** The `main` branch contains the stable version of the project. Other branches were used for experimentation and development purposes.

---

## Features

* 🔍 Malware detection across multiple platforms
* 🤖 Ensemble-based detection using LightGBM and MalConv2
* 📊 Browser-based dashboard built with Flask
* ⚡ Fast static analysis using EMBER2024 features
* 📁 Support for Windows, Linux, Android, and PDF files
* 🚫 Unsupported formats are clearly identified instead of receiving misleading threat scores
* 🖥️ Cross-platform dashboard (Windows, Linux, macOS)

---

## Detection Architecture

MUNDA uses a hybrid detection pipeline:

### LightGBM (EMBER2024)

Feature-based malware detection using pretrained models trained on the EMBER2024 dataset.

### MalConv2

Deep learning model that analyzes raw bytes directly from executable files.

### Ensemble Scoring

When multiple models are available, MUNDA combines predictions into a single threat score and confidence rating.

---

## Supported File Types

| Target Platform   | Status             | File Types                      |
| ----------------- | ------------------ | ------------------------------- |
| Windows           | ✅ Supported        | `.exe`, `.dll`, `.sys`          |
| .NET Applications | ✅ Supported        | .NET Assemblies                 |
| Android           | ✅ Supported        | `.apk`                          |
| Linux             | ✅ Supported        | ELF binaries, `.elf`            |
| Documents         | ✅ Supported        | `.pdf`                          |
| macOS             | ⚠️ Recognized Only | `.app`, `.dmg`, Mach-O binaries |

> macOS files can be identified by MUNDA but are currently not scored because no macOS malware model is included.

---

## Project Structure

```text
MUNDA/
├── app.py                     # Flask backend
├── index.html                 # Dashboard frontend
├── main.py                    # CLI entry point
├── scanner.py                 # Main scanning engine
├── feature_extractor.py       # EMBER2024 feature extraction
├── file_identifier.py         # File type detection
├── ember_compat.py            # EMBER2024 compatibility helpers
├── lightgbm_model.py          # LightGBM model loader
├── malconv2_model.py          # MalConv2 model loader
├── ensemble.py                # Score aggregation
├── data/                      # Downloaded models and datasets
├── requirements.txt
└── README.md
```

---

## Dataset

MUNDA utilizes the **EMBER2024** malware dataset containing approximately:

* 3.2 million samples
* Windows PE files
* Linux ELF binaries
* Android APKs
* PDF documents

---

## Installation

### Requirements

* Python 3.12 (recommended)
* pip
* Git

### Clone Repository

```bash
git clone https://github.com/your-username/MUNDA.git
cd MUNDA
```

### Install Dependencies

```bash
python3.12 -m pip install -r requirements.txt
```

### Download Pretrained Models

```bash
python3.12 main.py --download-models
```

---

## Quick Start

### Scan a File

```bash
python3.12 main.py --scan /path/to/file.exe
```

### Launch the Dashboard

```bash
python3.12 main.py --dashboard --port 5050
```

Open your browser and navigate to:

```text
http://localhost:5050
```

---

## Dashboard

The Flask dashboard allows users to:

* Upload files
* View threat scores
* Check confidence levels
* Identify file types
* Review scan verdicts

---

## Threat Scoring

| Score   | Verdict       |
| ------- | ------------- |
| 0–39%   | ✅ Clean       |
| 40–74%  | ⚠️ Suspicious |
| 75–100% | 🚨 Malware    |

Confidence is calculated from the decisiveness of the active detection models.

---

## Model Usage

| File Type  | LightGBM | MalConv2 |
| ---------- | -------- | -------- |
| Windows PE | ✅        | ✅        |
| .NET       | ✅        | ✅        |
| APK        | ✅        | ❌        |
| ELF        | ✅        | ❌        |
| PDF        | ✅        | ❌        |

MalConv2 is only used for Windows PE-style binaries because the publicly available checkpoint was trained specifically on PE files.

---

## Training

### Download Dataset

```bash
python3.12 main.py --download-dataset
```

### Train Models

```bash
python3.12 main.py --train
```

### Evaluate Models

```bash
python3.12 main.py --evaluate
```

---

## Troubleshooting

### PyTorch Installation Issues

PyTorch support may lag behind the newest Python releases.

```bash
python3.12 -m pip install -r requirements.txt
```

### macOS: Missing OpenMP

```bash
brew install libomp
```

### Homebrew Not Installed

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### Port Already In Use

```bash
python3.12 main.py --dashboard --port 5050
```

Choose a different port if necessary.

### Models Missing

```bash
python3.12 main.py --download-models
```

### File Marked as UNSUPPORTED

Only the following formats are currently analyzed:

* PE (`.exe`, `.dll`, `.sys`)
* .NET assemblies
* APK (`.apk`)
* ELF (`.elf`)
* PDF (`.pdf`)

---

## Limitations

* Detection results are probabilistic and should not be treated as guarantees.
* macOS malware detection is not yet implemented.
* MalConv2 is only available for Windows PE files.
* Accuracy depends on the quality of pretrained models and the characteristics of analyzed samples.

---

## Safe Testing

Avoid downloading real malware onto personal systems.

For safe testing, use:

* EICAR test files
* Sandbox environments
* Academic malware datasets
* Virtual machines

---

## Disclaimer

MUNDA is intended for **educational, research, and defensive security purposes only**.

The authors are not responsible for decisions made based on scan results. Always perform additional verification before classifying a file as safe or malicious.

---

## License

Specify your project's license here.

```text
MIT License
```

or

```text
GNU GPL v3
```


## Credits
This project uses pretrained EMBER2024 LightGBM models and the public MalConv2 checkpoint from FutureComputing4AI.

- EMBER2024: https://github.com/FutureComputing4AI/EMBER2024
- MalConv2: https://github.com/FutureComputing4AI/MalConv2


## Poster
<img width="1491" height="1055" alt="poster1" src="https://github.com/user-attachments/assets/8a90f8b7-7f8c-49c3-83aa-74e16492e8e2" />


