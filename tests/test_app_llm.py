import json
from unittest.mock import patch
import unittest
import test_app


class LocalLinguisticAppTests(unittest.TestCase):
    setUp = test_app.LocalAppTests.setUp
    tearDown = test_app.LocalAppTests.tearDown
    request = test_app.LocalAppTests.request
    auth = test_app.LocalAppTests.auth

    def test_page_exposes_optional_local_engine(self):
        _, _, body = self.request('GET', '/?token=' + self.server.token)
        for value in [b'Ollama', b'LM Studio', b'llama.cpp', b'127.0.0.1', b'id="provider"', b'id="model"']:
            self.assertIn(value, body)

    def test_model_listing_requires_session(self):
        self.assertEqual(self.request('GET', '/api/models?provider=ollama')[0], 403)

    def test_unsafe_engine_rejected_before_zip_read(self):
        headers = {**self.auth(), 'X-Local-Provider': 'ollama', 'X-Local-Endpoint': 'https://example.com', 'X-Local-Model': 'local', 'Content-Length': '100'}
        status, _, body = self.request('POST', '/api/scan', b'', headers)
        self.assertEqual(status, 400)
        self.assertIn('error', json.loads(body))

    def test_model_listing_uses_local_adapter(self):
        with patch.object(self.app, 'list_models', return_value=['modello-locale']):
            status, _, data = self.request('GET', '/api/models?provider=ollama', headers=self.auth())
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(data)['models'], ['modello-locale'])

    def test_model_unavailable_is_informative(self):
        with patch.object(self.app, 'list_models', side_effect=RuntimeError('Motore locale spento')):
            status, _, data = self.request('GET', '/api/models?provider=ollama', headers=self.auth())
        self.assertEqual(status, 503)
        self.assertIn('spento', json.loads(data)['error'])
