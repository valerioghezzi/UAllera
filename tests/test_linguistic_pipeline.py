import io
import unittest
import zipfile
from unittest.mock import patch
from jamovi_audit.pipeline import audit_zip


def delivery():
    b=io.BytesIO()
    with zipfile.ZipFile(b,'w') as z:
        z.writestr('Classe/Esempio/relazione.docx',b'invalid')
    return b.getvalue()


class LinguisticPipelineTests(unittest.TestCase):
    def test_engine_proposals_are_separate_from_deterministic_findings(self):
        proposed={'status':'complete','provider':'ollama','model':'local','findings':[], 'limits':['Proposta del tutor']}
        with patch('jamovi_audit.pipeline.assess_relation',return_value=proposed) as assess:
            result=audit_zip(delivery(),linguistic_config={'enabled':True,'provider':'ollama','model':'local'})
        self.assertEqual(result['reports'][0]['linguistic'],proposed)
        self.assertTrue(assess.called)
        self.assertIn('statistical_context',set(assess.call_args.args[1]))
        self.assertIn('linguistic',result['summary'])

    def test_disabled_engine_is_not_called(self):
        with patch('jamovi_audit.pipeline.assess_relation') as assess:
            result=audit_zip(delivery())
        self.assertFalse(assess.called)
        self.assertEqual(result['summary']['linguistic'],'disabled')
