"""Contract for assistant-authored, tutor-reviewable held-out examples."""
import importlib
import importlib.util
from collections import Counter
import unittest

from jamovi_audit.local_llm import CALIBRATION_CASES, _SYSTEM


class ValidationCaseTests(unittest.TestCase):
    def corpus(self):
        self.assertIsNotNone(importlib.util.find_spec('jamovi_audit.validation_cases'),
                             'A separate held-out corpus is required')
        return importlib.import_module('jamovi_audit.validation_cases')

    def test_version_schema_and_unique_identifiers(self):
        module = self.corpus()
        self.assertEqual(module.VALIDATION_VERSION, 'it-psychometrics-heldout-1.0')
        cases = module.VALIDATION_CASES
        self.assertEqual(len(cases), 70)
        self.assertEqual(len({c['id'] for c in cases}), 70)
        self.assertEqual(len({c['text'] for c in cases}), 70)
        for case in cases:
            with self.subTest(case=case['id']):
                for key in ('id', 'kind', 'text', 'rationale', 'criterion'):
                    self.assertIsInstance(case[key], str)
                    self.assertTrue(case[key].strip())
                self.assertIn(case['part'], range(1, 8))
                self.assertIn(case['expected_assessment'],
                              ('coerente', 'errata', 'incompleta', 'non_valutabile'))
                self.assertNotIn('expected_assessment', case['criterion'])
                self.assertNotIn('etichetta attesa', case['criterion'].lower())

    def test_each_part_has_ten_balanced_contrast_cases(self):
        cases = self.corpus().VALIDATION_CASES
        for part in range(1, 8):
            rows = [case for case in cases if case['part'] == part]
            with self.subTest(part=part):
                self.assertEqual(len(rows), 10)
                self.assertEqual(Counter(c['expected_assessment'] for c in rows),
                                 {'coerente': 3, 'errata': 4, 'incompleta': 1,
                                  'non_valutabile': 2})
                self.assertEqual({c['kind'] for c in rows}, {
                    'correct', 'correct_negation', 'paraphrase', 'error_negation',
                    'error_numeric', 'error_concept', 'error_contradiction',
                    'partial', 'missing', 'injection'})
                self.assertEqual(next(c for c in rows if c['kind'] == 'partial')
                                 ['expected_assessment'], 'incompleta')
                for kind in ('missing', 'injection'):
                    self.assertEqual(next(c for c in rows if c['kind'] == kind)
                                     ['expected_assessment'], 'non_valutabile')

    def test_no_heldout_text_is_a_prompt_demonstration_or_prompt_content(self):
        for case in self.corpus().VALIDATION_CASES:
            text = case['text'].casefold().strip()
            with self.subTest(case=case['id']):
                self.assertNotIn(text, _SYSTEM.casefold())
                for demo in CALIBRATION_CASES:
                    other = demo['text'].casefold().strip()
                    self.assertNotIn(text, other)
                    self.assertNotIn(other, text)

    def test_documentation_requires_teacher_review_of_assistant_labels(self):
        documentation = self.corpus().__doc__.lower()
        self.assertIn('assistant', documentation)
        self.assertIn('tutor', documentation)
        self.assertIn('non', documentation)


if __name__ == '__main__':
    unittest.main()
