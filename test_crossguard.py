"""
test_crossguard.py
Unit and integration tests for MUNDA.
Run with: python -m pytest tests/ -v
Or:       python tests/test_crossguard.py
"""

import os
import sys
import struct
import tempfile
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ── Helpers to create test files ─────────────────────────

def _make_pe_file(path: str, is_64bit: bool = False) -> str:
    """Create a minimal (but structurally valid) PE file for testing."""
    # DOS header (64 bytes) — e_magic=MZ, e_lfanew=0x40
    dos_header = b'MZ' + b'\x00' * 58 + struct.pack('<I', 0x40)
    # PE signature
    pe_sig = b'PE\x00\x00'
    # COFF header
    machine = 0x8664 if is_64bit else 0x014c
    coff = struct.pack('<HHIIIHH',
        machine,  # Machine
        0,        # NumberOfSections
        0,        # TimeDateStamp
        0,        # PointerToSymbolTable
        0,        # NumberOfSymbols
        0,        # SizeOfOptionalHeader
        0x0002,   # Characteristics (executable)
    )
    content = dos_header + pe_sig + coff
    with open(path, 'wb') as f:
        f.write(content)
    return path


def _make_elf_file(path: str) -> str:
    """Create a minimal ELF file for testing."""
    elf_magic = b'\x7fELF' + b'\x00' * 60
    with open(path, 'wb') as f:
        f.write(elf_magic)
    return path


def _make_pdf_file(path: str) -> str:
    """Create a minimal PDF file for testing."""
    with open(path, 'wb') as f:
        f.write(b'%PDF-1.4\n%comment\n')
    return path


def _make_apk_file(path: str) -> str:
    """Create a minimal APK (ZIP with AndroidManifest.xml) for testing."""
    import zipfile
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('AndroidManifest.xml', '<manifest/>')
        z.writestr('classes.dex', b'\x64\x65\x78\x0a')
    return path


# ── Test: File Identifier ────────────────────────────────

class TestFileIdentifier(unittest.TestCase):

    def setUp(self):
        from file_identifier import identify_file
        self.identify = identify_file
        self.tmpdir = tempfile.mkdtemp()

    def test_pe_win32(self):
        path = _make_pe_file(os.path.join(self.tmpdir, 'test.exe'), is_64bit=False)
        result = self.identify(path)
        self.assertIn(result['file_type'], ['WIN32', 'WIN64', 'DOTNET'])
        self.assertEqual(result['platform'][:7], 'Windows')

    def test_pe_win64(self):
        path = _make_pe_file(os.path.join(self.tmpdir, 'test64.exe'), is_64bit=True)
        result = self.identify(path)
        self.assertIn(result['file_type'], ['WIN32', 'WIN64', 'DOTNET'])

    def test_elf(self):
        path = _make_elf_file(os.path.join(self.tmpdir, 'test.elf'))
        result = self.identify(path)
        self.assertEqual(result['file_type'], 'ELF')
        self.assertEqual(result['platform'], 'Linux')

    def test_pdf(self):
        path = _make_pdf_file(os.path.join(self.tmpdir, 'test.pdf'))
        result = self.identify(path)
        self.assertEqual(result['file_type'], 'PDF')
        self.assertEqual(result['platform'], 'Document')

    def test_apk(self):
        path = _make_apk_file(os.path.join(self.tmpdir, 'test.apk'))
        result = self.identify(path)
        self.assertEqual(result['file_type'], 'APK')
        self.assertEqual(result['platform'], 'Android')

    def test_unknown(self):
        path = os.path.join(self.tmpdir, 'test.bin')
        with open(path, 'wb') as f:
            f.write(b'\xDE\xAD\xBE\xEF' * 64)
        result = self.identify(path)
        self.assertEqual(result['file_type'], 'UNKNOWN')

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            self.identify('/nonexistent/path/file.exe')

    def test_size_reported(self):
        path = _make_pdf_file(os.path.join(self.tmpdir, 'sized.pdf'))
        result = self.identify(path)
        self.assertIn('size_bytes', result)
        self.assertGreater(result['size_bytes'], 0)


# ── Test: Feature Extractor ──────────────────────────────

class TestFeatureExtractor(unittest.TestCase):

    def setUp(self):
        from feature_extractor import extract_features
        self.extract = extract_features
        self.tmpdir = tempfile.mkdtemp()

    def test_vector_shape_elf(self):
        path = _make_elf_file(os.path.join(self.tmpdir, 'test.elf'))
        vec = self.extract(path)
        self.assertEqual(vec.shape, (2568,))
        self.assertEqual(vec.dtype, np.float32)

    def test_vector_shape_pdf(self):
        path = _make_pdf_file(os.path.join(self.tmpdir, 'test.pdf'))
        vec = self.extract(path)
        self.assertEqual(vec.shape, (2568,))

    def test_vector_shape_pe(self):
        path = _make_pe_file(os.path.join(self.tmpdir, 'test.exe'))
        vec = self.extract(path)
        self.assertEqual(vec.shape, (2568,))

    def test_vector_no_nan(self):
        path = _make_elf_file(os.path.join(self.tmpdir, 'nonan.elf'))
        vec = self.extract(path)
        self.assertFalse(np.any(np.isnan(vec)), "Feature vector contains NaN")

    def test_vector_no_inf(self):
        path = _make_pdf_file(os.path.join(self.tmpdir, 'noinf.pdf'))
        vec = self.extract(path)
        self.assertFalse(np.any(np.isinf(vec)), "Feature vector contains Inf")

    def test_values_are_finite_and_useful(self):
        path = _make_elf_file(os.path.join(self.tmpdir, 'range.elf'))
        vec = self.extract(path)
        self.assertTrue(np.all(np.isfinite(vec)))
        self.assertGreater(np.sum(np.abs(vec)), 0)

    def test_different_files_differ(self):
        path1 = _make_elf_file(os.path.join(self.tmpdir, 'a.elf'))
        path2 = _make_pdf_file(os.path.join(self.tmpdir, 'b.pdf'))
        v1 = self.extract(path1)
        v2 = self.extract(path2)
        self.assertFalse(np.allclose(v1, v2), "Different files produced identical vectors")


# ── Test: Ensemble logic (no models needed) ───────────────

class TestEnsembleLogic(unittest.TestCase):

    def _make_mock_ensemble(self, method='weighted'):
        """Build an ensemble with mock models."""
        from ensemble import EnsembleDetector

        class MockLGBM:
            def predict(self, fv, ft): return 0.8

        class MockMalConv:
            def predict(self, fp): return 0.9

        return EnsembleDetector(MockLGBM(), MockMalConv(), method=method)

    def test_weighted_average(self):
        ens = self._make_mock_ensemble('weighted')
        # lgbm=0.8, malconv=0.9 → 0.4*0.8 + 0.6*0.9 = 0.86
        result = ens._combine(0.8, 0.9)
        self.assertAlmostEqual(result, 0.86, places=4)

    def test_simple_average(self):
        ens = self._make_mock_ensemble('average')
        result = ens._combine(0.6, 0.8)
        self.assertAlmostEqual(result, 0.70, places=4)

    def test_verdict_malware(self):
        self.assertEqual(
            self._make_mock_ensemble()._verdict(0.90), 'MALWARE'
        )

    def test_verdict_suspicious(self):
        self.assertEqual(
            self._make_mock_ensemble()._verdict(0.60), 'SUSPICIOUS'
        )

    def test_verdict_clean(self):
        self.assertEqual(
            self._make_mock_ensemble()._verdict(0.10), 'CLEAN'
        )

    def test_confidence_high(self):
        conf = self._make_mock_ensemble()._confidence(0.9, 0.92, 0.91)
        self.assertEqual(conf, 'HIGH')

    def test_confidence_low(self):
        conf = self._make_mock_ensemble()._confidence(0.1, 0.9, 0.5)
        self.assertEqual(conf, 'LOW')


# ── Test: Reporter ───────────────────────────────────────

class TestReporter(unittest.TestCase):

    def _sample_results(self):
        return [
            {'filename': 'virus.exe', 'verdict': 'MALWARE', 'final_score': 0.95,
             'lgbm_score': 0.93, 'malconv_score': 0.97, 'platform': 'Windows',
             'file_type': 'WIN32', 'size': '1.2 MB', 'confidence': 'HIGH',
             'sha256': 'abc' * 21 + 'a', 'scan_time_s': 1.2},
            {'filename': 'app.apk', 'verdict': 'CLEAN', 'final_score': 0.05,
             'lgbm_score': 0.04, 'malconv_score': 0.06, 'platform': 'Android',
             'file_type': 'APK', 'size': '3.4 MB', 'confidence': 'HIGH',
             'sha256': 'def' * 21 + 'd', 'scan_time_s': 0.8},
        ]

    def test_json_report(self):
        from reporter import ReportGenerator
        with tempfile.TemporaryDirectory() as td:
            gen = ReportGenerator(self._sample_results())
            path = os.path.join(td, 'report.json')
            gen.save_json(path)
            import json
            with open(path) as f:
                data = json.load(f)
            self.assertEqual(data['total_scans'], 2)
            self.assertEqual(data['summary']['malware'], 1)

    def test_csv_report(self):
        from reporter import ReportGenerator
        import csv
        with tempfile.TemporaryDirectory() as td:
            gen = ReportGenerator(self._sample_results())
            path = os.path.join(td, 'report.csv')
            gen.save_csv(path)
            with open(path) as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 2)
            self.assertIn('verdict', rows[0])

    def test_text_report(self):
        from reporter import ReportGenerator
        with tempfile.TemporaryDirectory() as td:
            gen = ReportGenerator(self._sample_results())
            path = os.path.join(td, 'report.txt')
            gen.save_text(path)
            with open(path) as f:
                content = f.read()
            self.assertIn('MALWARE', content)
            self.assertIn('CLEAN', content)


# ── Test: Scan Logger ────────────────────────────────────

class TestScanLogger(unittest.TestCase):

    def test_write_and_read(self):
        from logger import ScanLogger
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False, mode='w') as tf:
            tmp_path = tf.name
        try:
            sl = ScanLogger(tmp_path)
            sl.log({'filename': 'test.exe', 'verdict': 'MALWARE',
                    'final_score': 0.9, 'platform': 'Windows'})
            sl.log({'filename': 'clean.pdf', 'verdict': 'CLEAN',
                    'final_score': 0.05, 'platform': 'Document'})
            records = sl.read_all()
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]['verdict'], 'MALWARE')
            self.assertEqual(records[1]['verdict'], 'CLEAN')
        finally:
            os.unlink(tmp_path)

    def test_stats(self):
        from logger import ScanLogger
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False, mode='w') as tf:
            tmp_path = tf.name
        try:
            sl = ScanLogger(tmp_path)
            for v in ['MALWARE', 'CLEAN', 'CLEAN', 'SUSPICIOUS']:
                sl.log({'verdict': v, 'final_score': 0.5, 'platform': 'Windows'})
            stats = sl.stats()
            self.assertEqual(stats['total'], 4)
            self.assertEqual(stats['malware'], 1)
            self.assertEqual(stats['clean'], 2)
            self.assertEqual(stats['suspicious'], 1)
        finally:
            os.unlink(tmp_path)


# ── Test: Metrics ────────────────────────────────────────

class TestMetrics(unittest.TestCase):

    def test_perfect_classifier(self):
        from evaluator import compute_metrics
        y_true   = np.array([1, 1, 0, 0, 1, 0])
        y_scores = np.array([0.99, 0.95, 0.01, 0.02, 0.98, 0.03])
        m = compute_metrics(y_true, y_scores)
        self.assertAlmostEqual(m['roc_auc'], 1.0, places=3)
        self.assertAlmostEqual(m['accuracy'], 1.0, places=3)
        self.assertEqual(m['fp'], 0)
        self.assertEqual(m['fn'], 0)

    def test_random_classifier(self):
        from evaluator import compute_metrics
        np.random.seed(42)
        y_true   = np.random.randint(0, 2, 1000)
        y_scores = np.random.rand(1000)
        m = compute_metrics(y_true, y_scores)
        # Random should be near 0.5
        self.assertGreater(m['roc_auc'], 0.4)
        self.assertLess(m['roc_auc'], 0.6)

    def test_all_keys_present(self):
        from evaluator import compute_metrics
        y_true   = np.array([1, 0, 1, 0])
        y_scores = np.array([0.9, 0.1, 0.8, 0.2])
        m = compute_metrics(y_true, y_scores)
        for key in ['roc_auc', 'pr_auc', 'accuracy', 'precision',
                    'recall', 'f1', 'fpr', 'fnr', 'tp', 'tn', 'fp', 'fn']:
            self.assertIn(key, m, f"Missing key: {key}")


# ── Run all tests ─────────────────────────────────────────

if __name__ == '__main__':
    print("\nMUNDA Test Suite")
    print("=" * 50)
    unittest.main(verbosity=2)
