import io
import unittest
import zipfile
from unittest.mock import patch
from jamovi_audit.pipeline import audit_zip
from jamovi_audit.readers import read_docx


def delivery(names=('Esempio',)):
    b=io.BytesIO()
    with zipfile.ZipFile(b,'w') as z:
        for name in names:
            z.writestr(f'Classe/{name}/relazione.docx',b'invalid')
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

    def test_cancelled_minerva_keeps_successful_evidence_and_never_dispatches_next_student(self):
        cancellation = {'requested': False}
        proposal = {'status':'cancelled','provider':'llamacpp','model':'minerva-7b-instruct-q4-local',
                    'coverage':{'successful_chunks':1,'planned_chunks':3},
                    'findings':[{'part':6,'quote':'Una correlazione non dimostra causalità.',
                                 'proposal':'coerente','location':'Paragrafo 1'}], 'limits':[]}
        def review(*args, **kwargs):
            cancellation['requested'] = True
            return proposal
        with patch('jamovi_audit.pipeline.assess_relation',side_effect=review) as assess, \
             patch('jamovi_audit.pipeline.read_docx',wraps=read_docx) as reader, \
             patch('jamovi_audit.pipeline.compare_submissions') as compare:
            result = audit_zip(delivery(('A_Primo','B_Secondo','C_Terzo')),
                               cancelled=lambda:cancellation['requested'],
                               linguistic_config={'enabled':True,'provider':'llamacpp','model':'minerva-7b-instruct-q4-local'})
        self.assertEqual(assess.call_count,1)
        self.assertEqual(reader.call_count,1)
        compare.assert_not_called()
        self.assertEqual(result['summary']['status'],'cancelled')
        self.assertTrue(result['summary']['incomplete'])
        self.assertEqual(result['summary']['student_count'],1)
        self.assertEqual(result['summary']['planned_student_count'],3)
        self.assertEqual(result['summary']['unexecuted_student_count'],2)
        self.assertEqual(result['summary']['file_count'],1)
        self.assertEqual(result['summary']['comparison_status'],'not_run_cancelled')
        self.assertEqual(result['reports'][0]['linguistic'],proposal)
        self.assertIn('Una correlazione non dimostra causalità.',result['html'])
        self.assertIn('Rapporto incompleto',result['html'])
        self.assertNotIn('B_Secondo',result['html'])
        self.assertNotIn('C_Terzo',result['html'])

    def test_legacy_engine_cancellation_keeps_existing_interrupt_behavior(self):
        cancellation = {'requested':False}
        def review(*args, **kwargs):
            cancellation['requested']=True
            return {'status':'cancelled','findings':[]}
        with patch('jamovi_audit.pipeline.assess_relation',side_effect=review):
            with self.assertRaises(InterruptedError):
                audit_zip(delivery(),cancelled=lambda:cancellation['requested'],
                          linguistic_config={'enabled':True,'provider':'ollama','model':'local'})

    def test_chunk_progress_reaches_pipeline_callback_with_student_context(self):
        events=[]
        def review(*args, **kwargs):
            progress=kwargs.get('progress')
            if progress:
                progress(1,3,'Minerva: sezione 2/3')
            return {'status':'complete','findings':[],'limits':[]}
        with patch('jamovi_audit.pipeline.assess_relation',side_effect=review):
            audit_zip(delivery(),progress=lambda *event:events.append(event),
                      linguistic_config={'enabled':True,'provider':'llamacpp','model':'minerva-7b-instruct-q4-local'})
        self.assertTrue(any(done==1 and total==3 and 'Esempio' in label and 'sezione 2/3' in label
                            for done,total,label in events))
