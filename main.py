"""
main.py
MUNDA - Malware Detector
Entry point for CLI usage.

Usage:
  python main.py --scan /path/to/file.exe
  python main.py --scan-dir /path/to/folder
  python main.py --dashboard
  python main.py --download-models
  python main.py --download-dataset
  python main.py --train
  python main.py --evaluate
"""

import argparse
import json
import os
import sys

# Make sure all modules are importable
sys.path.insert(0, os.path.dirname(__file__))


def main():
    parser = argparse.ArgumentParser(
        prog='munda',
        description='MUNDA - Malware Detection\n'
        'Dataset: EMBER2024 | Model: LightGBM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--scan', metavar='FILE', help='Scan a single file')
    parser.add_argument('--scan-dir', metavar='DIR', help='Scan all files in a directory')
    parser.add_argument(
        '--recursive',
        action='store_true',
        default=True,
        help='Scan subdirectories (default: True)',
    )
    parser.add_argument(
        '--method',
        default='meta',
        choices=['meta', 'weighted', 'average'],
        help='Ensemble method (default: meta)',
    )
    parser.add_argument('--dashboard', action='store_true', help='Launch web dashboard')
    parser.add_argument('--port', type=int, default=5000, help='Dashboard port')
    parser.add_argument(
        '--download-models',
        action='store_true',
        help='Download pretrained models',
    )
    parser.add_argument(
        '--download-dataset',
        action='store_true',
        help='Download EMBER2024 dataset',
    )
    parser.add_argument('--train', action='store_true', help='Train all models')
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Evaluate on test+challenge sets',
    )
    parser.add_argument('--output-json', metavar='FILE', help='Save scan results to JSON')

    args = parser.parse_args()

    if args.download_models:
        from trainer import download_pretrained_models

        download_pretrained_models()
        return

    if args.download_dataset:
        from trainer import download_dataset

        download_dataset()
        return

    if args.train:
        from trainer import train_all

        train_all()
        return

    if args.evaluate:
        from trainer import evaluate_challenge_set

        evaluate_challenge_set()
        return

    if args.dashboard:
        _print_banner()
        print(f"\n  Dashboard: http://localhost:{args.port}\n")
        from app import app

        app.run(host='0.0.0.0', port=args.port, debug=False)
        return

    if args.scan:
        _print_banner()
        from scanner import Scanner

        scanner = Scanner(method=args.method)
        result = scanner.scan(args.scan)
        _print_scan_result(result)
        if args.output_json:
            with open(args.output_json, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
            print(f"\n  Results saved to {args.output_json}")
        return

    if args.scan_dir:
        _print_banner()
        from scanner import Scanner

        scanner = Scanner(method=args.method)
        results = scanner.scan_directory(args.scan_dir, recursive=args.recursive)
        print(f"\n  Scanned {len(results)} files.")
        malware_count = sum(1 for r in results if r['verdict'] == 'MALWARE')
        suspicious_count = sum(1 for r in results if r['verdict'] == 'SUSPICIOUS')
        unsupported_count = sum(1 for r in results if r['verdict'] == 'UNSUPPORTED')
        clean_count = len(results) - malware_count - suspicious_count - unsupported_count
        print(f"  Malware:    {malware_count}")
        print(f"  Suspicious: {suspicious_count}")
        print(f"  Clean:      {clean_count}")
        print(f"  Unsupported:{unsupported_count}")
        if args.output_json:
            with open(args.output_json, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            print(f"\n  Results saved to {args.output_json}")
        return

    _print_banner()
    parser.print_help()


def _print_banner():
    print(
        """
  ==================================================
    MUNDA - Malware Detector
    Dataset:  EMBER2024 (3.2M files, 6 formats)
    Model:    LightGBM
    Coverage: Windows | Linux | Android | PDF
  ==================================================
"""
    )


def _print_scan_result(result: dict):
    icons = {'MALWARE': '[!]', 'SUSPICIOUS': '[?]', 'CLEAN': '[OK]', 'UNSUPPORTED': '[--]'}
    verdict = result['verdict']
    print(
        f"""
  ---------------------------------------------
    {icons.get(verdict, '[ ]')} {verdict}
  ---------------------------------------------
    File:        {result['filename']}
    Platform:    {result['platform']}
    Size:        {result['size']}
    Confidence:  {result['confidence']}
    Final Score: {_format_score(result['final_score'])}
    LightGBM:    {_format_score(result['lgbm_score'])}
    MalConv2:    {_format_score(result['malconv_score'])}
    Method:      {result['method']}
    SHA256:      {result['sha256'][:36]}
    Scan time:   {result['scan_time_s']}s
"""
    )
    for model, error in result.get('model_errors', {}).items():
        print(f"  Warning: {model} unavailable: {error}")


def _format_score(score) -> str:
    if score is None:
        return 'unavailable'
    return f"{score:.4f}"


def _print_startup_error(message: str) -> None:
    print(f"\n[MUNDA] {message}")
    print("Install dependencies with:")
    print("  python3 -m pip install -r requirements.txt")
    print("Then download the models with:")
    print("  python3 main.py --download-models")


if __name__ == '__main__':
    try:
        main()
    except ModuleNotFoundError as e:
        _print_startup_error(f"Missing Python dependency: {e.name}")
        sys.exit(1)
    except RuntimeError as e:
        _print_startup_error(str(e))
        sys.exit(1)
