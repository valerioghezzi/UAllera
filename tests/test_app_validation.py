import unittest
import json,time
from unittest.mock import patch
import test_app
from jamovi_audit.validation import run_validation


class AppValidationTests(unittest.TestCase):
    setUp=test_app.LocalAppTests.setUp
    tearDown=test_app.LocalAppTests.tearDown
    request=test_app.LocalAppTests.request
    auth=test_app.LocalAppTests.auth

    def config_headers(self):
        return {**self.auth(),'X-Local-Provider':'ollama','X-Local-Model':'local'}

    def test_model_validation_is_visible_without_selecting_zip(self):
        _,_,body=self.request('GET','/?token='+self.server.token)
        self.assertIn(b'id="validate"',body)
        self.assertIn(b'70',body)

    def test_model_validation_requires_selected_engine(self):
        self.assertEqual(self.request('POST','/api/validate',b'',self.auth())[0],400)

    def test_busy_audit_and_validation_are_mutually_exclusive(self):
        with self.server.state_lock:self.server.state['busy']=True
        self.assertEqual(self.request('POST','/api/validate',b'',self.config_headers())[0],409)

    def test_validation_report_has_distinct_download_name_and_no_students(self):
        case={'id':'H1.01','part':1,'kind':'correct','text':'Testo sintetico.','criterion':'Spiegare.','expected_assessment':'coerente'}
        with patch('jamovi_audit.validation.assess_relation',return_value={'status':'unavailable','findings':[],'limits':[]}):
            result=run_validation({'enabled':True,'model':'local'},cases=[case])
        with patch.object(self.app,'run_validation',return_value=result):
            self.assertEqual(self.request('POST','/api/validate',b'',self.config_headers())[0],202)
            deadline=time.monotonic()+5
            while True:
                _,_,body=self.request('GET','/api/status',headers=self.auth());state=json.loads(body)
                if not state['busy']:break
                if time.monotonic()>deadline:self.fail('Validazione bloccata')
                time.sleep(.01)
        self.assertTrue(state['ready']);self.assertEqual(state['mode'],'validation')
        status,headers,body=self.request('GET','/report',headers=self.auth())
        self.assertEqual(status,200)
        self.assertIn('Validazione_modello.html',headers['Content-Disposition'])
        self.assertIn(b'Validazione linguistica',body)
        self.assertEqual(state['students'],0)


    def test_invalid_timeout_is_rejected_before_starting_job(self):
        headers={**self.config_headers(),'X-Local-Timeout':'0'}
        self.assertEqual(self.request('POST','/api/validate',b'',headers)[0],400)
