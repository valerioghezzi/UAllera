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
