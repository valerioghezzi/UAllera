import unittest
from unittest.mock import patch
from tools.calibrate_local import run


class CalibrationTests(unittest.TestCase):
    def test_expected_labels_are_not_injected_into_model_context(self):
        case={'id':'private-case','part':2,'kind':'error','text':'Una risposta di prova.','expected_assessment':'errata'}
        response={'status':'complete','findings':[{'part':2,'proposal':'coerente'}],'limits':[]}
        with patch('tools.calibrate_local.assess_relation',return_value=response) as assess:
            result=run({'enabled':True,'model':'local'},[case])
        self.assertNotIn('expected_assessment',str(assess.call_args))
        self.assertEqual(result['matching_labels'],0)
        self.assertEqual(result['citable_results'],1)

    def test_missing_and_contradictory_proposals_do_not_count_as_matches(self):
        case={'id':'case','part':1,'kind':'missing','text':'Titolo.','expected_assessment':'non_valutabile'}
        for findings in [[],[{'part':1,'proposal':'coerente'},{'part':1,'proposal':'errata'}]]:
            with self.subTest(findings=findings),patch('tools.calibrate_local.assess_relation',return_value={'status':'complete','findings':findings,'limits':[]}):
                result=run({'enabled':True,'model':'local'},[case])
                self.assertEqual(result['matching_labels'],0)
                self.assertEqual(result['citable_results'],0)
