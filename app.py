"""
app.py
CrossGuard Web Dashboard - Flask backend.
Run with: python app.py
"""

import os
import sys
import json
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, request, jsonify, render_template, send_from_directory

app = Flask(__name__, template_folder=os.path.dirname(__file__))
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB max upload

# Lazy-load scanner so dashboard starts fast
_scanner = None

def get_scanner():
    global _scanner
    if _scanner is None:
        from scanner import Scanner
        _scanner = Scanner(method='meta')
    return _scanner


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/scan', methods=['POST'])
def scan_file():
    """Scan an uploaded file."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    uploaded = request.files['file']
    if uploaded.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    # Save to temp file
    suffix = Path(uploaded.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        uploaded.save(tmp.name)
        tmp_path = tmp.name

    try:
        scanner = get_scanner()
        result  = scanner.scan(tmp_path)
        # Override filename with original name
        result['filename'] = uploaded.filename
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@app.route('/api/status')
def status():
    """Health check endpoint."""
    return jsonify({
        'status': 'running',
        'scanner_loaded': _scanner is not None,
        'version': '1.0.0',
        'name': 'CrossGuard',
    })


if __name__ == '__main__':
    print("CrossGuard Dashboard starting at http://localhost:5000")
    app.run(debug=False, host='0.0.0.0', port=5000)
