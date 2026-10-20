import json
import unittest
from unittest.mock import patch


class LinguisticValidationTests(unittest.TestCase):
    def module(self):
        try:
            from jamovi_audit import validation
            return validation
        except ImportError:
            self.fail('Validazione su casi separati non implementata')

    def cases(self):
        return [{'id':str(i),'part':1,'kind':'test','text':'Testo sintetico '+str(i),'criterion':'Interpretare il livello di misura.','rationale':'Nota riservata al docente.','expected_assessment':v} for i,v in enumerate(['errata','coerente','errata','non_valutabile'])]

    def response(self, label=None, status='complete'):
        return {'status':status,'findings':([] if label is None else [{'part':1,'proposal':label,'quote':'Testo sintetico','location':'Paragrafo 1'}]),'limits':[]}

    def test_metrics_do_not_hide_missing_proposals_and_false_reassurance(self):
        m=self.module()
        with patch.object(m,'assess_relation',side_effect=[self.response('errata'),self.response('errata'),self.response(),self.response('coerente')]):
            r=m.run_validation({'enabled':True,'model':'local'},cases=self.cases())
        self.assertEqual(r['planned'],4);self.assertEqual(r['attempted'],4);self.assertEqual(r['evaluated'],3)
        self.assertEqual(r['metrics']['matching_labels'],1)
        self.assertEqual(r['metrics']['error_recall'],.5)
        self.assertEqual(r['metrics']['false_alarm_rate'],1)
        self.assertEqual(r['metrics']['false_reassurance_count'],1)
        self.assertEqual(r['metrics']['valid_coverage'],.75)

    def test_answer_key_is_never_passed_as_context(self):
        m=self.module()
        with patch.object(m,'assess_relation',return_value=self.response('errata')) as call:
            r=m.run_validation({'enabled':True,'model':'local'},cases=self.cases()[:1])
        self.assertNotIn('expected_assessment',str(call.call_args))
        self.assertNotIn('Nota riservata',str(call.call_args))
        self.assertIn('corpus_sha256',r)

    def test_unavailable_engine_stops_and_never_reports_zero_accuracy(self):
        m=self.module()
        with patch.object(m,'assess_relation',return_value=self.response(status='unavailable')) as call:
            r=m.run_validation({'enabled':True,'model':'local'},cases=self.cases())
        self.assertEqual(call.call_count,1)
        self.assertEqual(r['status'],'unavailable')
        self.assertEqual(r['unexecuted'],3)
        self.assertIsNone(r['metrics']['agreement_valid'])
        self.assertIsNone(r['metrics']['agreement_attempted'])
        self.assertIsNone(r['metrics']['error_recall'])
        self.assertEqual(r['evaluated'],0)

    def test_conflicting_predictions_are_not_eligible_as_valid(self):
        m=self.module();response=self.response('errata');response['findings'].append({'part':1,'proposal':'coerente'})
        with patch.object(m,'assess_relation',return_value=response):
            r=m.run_validation({'enabled':True,'model':'local'},cases=self.cases()[:1])
        self.assertEqual(r['evaluated'],0)
        self.assertIsNone(r['cases'][0]['observed'])

    def test_cancel_before_run_preserves_no_fictitious_results(self):
        m=self.module()
        with patch.object(m,'assess_relation') as call:
            r=m.run_validation({'enabled':True,'model':'local'},cases=self.cases(),cancelled=lambda:True)
        self.assertFalse(call.called);self.assertEqual(r['status'],'cancelled');self.assertEqual(r['attempted'],0)

    def test_html_is_self_contained_and_escapes_hostile_content(self):
        m=self.module()
        with patch.object(m,'assess_relation',return_value=self.response('coerente')):
            r=m.run_validation({'enabled':True,'model':'<script>alert(1)</script>'},cases=self.cases()[:1])
        html=m.render_validation(r)
        self.assertNotIn('<script>',html);self.assertIn('&lt;script&gt;',html)
        self.assertIn('Non misura',html);self.assertIn('frode',html.lower())
        self.assertEqual(html.count('<html'),1)
        self.assertIn(r['evaluation_profile'],html)
        self.assertIn('documento intero',html)
