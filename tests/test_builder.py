import unittest
import importlib


class BuilderTests(unittest.TestCase):
    def test_runtime_rejects_wrong_checksum(self):
        try:
            build = importlib.import_module('tools.build_windows')
        except ImportError:
            self.fail('Costruttore pacchetti non implementato')
        with self.assertRaisesRegex(ValueError, '(?i)(checksum|sha|integrità)'):
            build.verify_runtime(b'not python', 'x64')

    def test_unsupported_architecture_is_explicit(self):
        try:
            build = importlib.import_module('tools.build_windows')
        except ImportError:
            self.fail('Costruttore pacchetti non implementato')
        with self.assertRaises(ValueError):
            build.verify_runtime(b'', 'x86')

class PortableLayoutTests(unittest.TestCase):
    def test_portable_bundle_contains_its_test_dependencies(self):
        import io
        import tempfile
        import zipfile
        from pathlib import Path
        from unittest.mock import patch
        from tools.build_windows import build
        buffer=io.BytesIO()
        with zipfile.ZipFile(buffer,'w') as z:
            z.writestr('python.exe',b'fixture')
            z.writestr('python314._pth','python314.zip\n.\n')
        with tempfile.TemporaryDirectory() as directory, patch('tools.build_windows.verify_runtime'):
            folder, bundle=build('x64',buffer.getvalue(),Path(directory))
            self.assertTrue((folder/'tools'/'build_windows.py').is_file())
            self.assertTrue(bundle.is_file())
