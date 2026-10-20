"""The public inference fixture must not require private assignments or a model."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
from jamovi_audit.pipeline import audit_zip


class RealModelFixtureTests(unittest.TestCase):
    def test_public_fixture_exercises_word_and_missing_outputs_for_two_people(self):
        spec = importlib.util.spec_from_file_location('ci_real_model', Path(__file__).resolve().parent.parent / 'tools/ci_real_model.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with patch('jamovi_audit.pipeline.load_reference', return_value=None):
            result = audit_zip(module.synthetic_delivery())
        self.assertEqual(2, result['summary']['student_count'])
        self.assertEqual(4, result['summary']['file_count'])
        self.assertFalse(result['summary']['reference_available'])
        for report in result['reports']:
            self.assertTrue(any(f['status'] == 'Mancante' and f.get('part') for f in report['findings']))
        self.assertIn('Persona_01', result['html'])
        self.assertIn('Persona_02', result['html'])

