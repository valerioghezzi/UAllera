"""The public inference fixture must not require private assignments or a model."""
import importlib.util
import io
from contextlib import redirect_stderr
from pathlib import Path
import unittest
from unittest.mock import patch
from jamovi_audit.pipeline import audit_zip


class RealModelFixtureTests(unittest.TestCase):
    def test_real_pipeline_gate_requires_cited_correct_causality_judgments(self):
        spec = importlib.util.spec_from_file_location('ci_real_model', Path(__file__).resolve().parent.parent / 'tools/ci_real_model.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        def report(second='errata', status='complete', findings=True):
            return {'reports': [{'linguistic': {'status': status, 'findings': [{'part': 6, 'proposal': label, 'quote': 'Evidenza sintetica.'}] if findings else []}}
                                for label in ('coerente', second)]}
        module.verify_pipeline(report())
        for bad in (report(second='coerente'), report(status='unavailable'), report(findings=False)):
            with self.subTest(bad=bad), self.assertRaises(AssertionError):
                module.verify_pipeline(bad)

    def test_pipeline_only_mode_preserves_default_full_benchmark(self):
        spec = importlib.util.spec_from_file_location('ci_real_model', Path(__file__).resolve().parent.parent / 'tools/ci_real_model.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertFalse(module.parse_args([]).pipeline_only)
        self.assertTrue(module.parse_args(['--pipeline-only']).pipeline_only)
        for value in ('0', '71'):
            with self.subTest(value=value), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                module.parse_args(['--limit', value])

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

