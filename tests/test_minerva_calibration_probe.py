"""Probe wiring tests only; no native process, downloads or inference."""
from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    from tools import probe_minerva_calibration as probe
except ImportError:
    probe = None


class CalibrationProbeTests(unittest.TestCase):
    def test_frozen_cases_balanced_and_system_examples_have_no_case_ids(self):
        self.assertIsNotNone(probe, 'Separate calibration probe is missing')
        self.assertEqual(Counter(c['expected_assessment'] for c in probe.CASES), {
            'coerente': 2, 'errata': 2, 'incompleta': 2, 'non_valutabile': 2})
        self.assertEqual(len({c['id'] for c in probe.CASES}), 8)
        for case in probe.CASES:
            self.assertTrue(case['criterion'])
            self.assertNotIn(case['id'], probe.SYSTEM_OVERRIDE)
        self.assertEqual(len(probe.EXAMPLES), 4)

    def test_override_system_only_restores_and_gate_is_predeclared(self):
        self.assertIsNotNone(probe)
        original = probe.mr.messages_for
        def validation(config, cases, progress=None):
            self.assertEqual(cases, list(probe.CASES))
            spans = [{'evidence_id': 'CURRENT', 'text': 'risposta'}]
            normal = original(spans, [6], 'criterio')
            changed = probe.mr.messages_for(spans, [6], 'criterio')
            self.assertEqual(changed[1:], normal[1:])
            self.assertIn(probe.SYSTEM_OVERRIDE, changed[0]['content'])
            return {'status': 'complete', 'attempted': 8, 'evaluated': 8,
                    'metrics': {'matching_labels': 7}, 'cases': []}
        with tempfile.TemporaryDirectory() as temp, patch.object(probe.validation, 'run_validation', side_effect=validation), patch.object(probe.validation, 'render_validation', return_value='<html>fixture</html>'):
            result = probe.run_probe({}, Path(temp), {}, 'secret-fixture')
            self.assertFalse(result['passed'])
            self.assertEqual(json.loads((Path(temp)/'Casi_congelati.json').read_text(encoding='utf-8')), list(probe.CASES))
        self.assertIs(probe.mr.messages_for, original)

    def test_failure_restores_monkeypatches_and_redacts_request_response(self):
        self.assertIsNotNone(probe)
        messages, request, assess = probe.mr.messages_for, probe.local_llm._request, probe.validation.assess_relation
        def validation(*args, **kwargs):
            probe.local_llm._request({}, 'POST', '/completion', {'text': 'private-key'})
            raise RuntimeError('private-key must not escape')
        with tempfile.TemporaryDirectory() as temp, patch.object(probe.local_llm, '_request', return_value={'content': 'private-key'}), patch.object(probe.validation, 'run_validation', side_effect=validation):
            with self.assertRaises(RuntimeError):
                probe.run_probe({}, Path(temp), {}, 'private-key')
            text = '\n'.join(p.read_text(encoding='utf-8') for p in Path(temp).glob('*') if p.is_file())
            self.assertNotIn('private-key', text)
            self.assertIn('[redacted]', text)
            self.assertFalse(json.loads((Path(temp)/'Esito_collaudo.json').read_text(encoding='utf-8'))['passed'])
        self.assertIs(probe.mr.messages_for, messages)
        self.assertIs(probe.local_llm._request, request)
        self.assertIs(probe.validation.assess_relation, assess)

    def test_eight_valid_matching_results_pass_the_fixed_gate(self):
        self.assertIsNotNone(probe)
        fixture = {'status': 'complete', 'attempted': 8, 'evaluated': 8,
                   'metrics': {'matching_labels': 8}, 'cases': []}
        with tempfile.TemporaryDirectory() as temp, patch.object(probe.validation, 'run_validation', return_value=fixture), patch.object(probe.validation, 'render_validation', return_value='<html>fixture</html>'):
            result = probe.run_probe({}, Path(temp), {})
        self.assertTrue(result['passed'])
        self.assertEqual(result['checks'], {'attempted_8': True, 'valid_8': True, 'concordant_8': True})


if __name__ == '__main__':
    unittest.main()
