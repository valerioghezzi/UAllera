import http.client
import importlib
import json
import threading
import unittest


class LocalAppTests(unittest.TestCase):
    def setUp(self):
        try:
            self.app = importlib.import_module('app')
        except ImportError:
            self.fail('Interfaccia locale non implementata')
        self.server = self.app.make_server(0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_address[1]

    def tearDown(self):
        if hasattr(self, 'server'):
            self.server.shutdown()
            self.server.server_close()
            self.thread.join(timeout=3)

    def request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=10)
        conn.request(method, path, body=body, headers=headers or {})
        response = conn.getresponse()
        data = response.read()
        status, rh = response.status, dict(response.getheaders())
        conn.close()
        return status, rh, data

    def auth(self):
        return {'X-Session-Token': self.server.token}

    def test_binds_only_loopback(self):
        self.assertEqual(self.server.server_address[0], '127.0.0.1')

    def test_landing_requires_token(self):
        self.assertEqual(self.request('GET', '/')[0], 403)
        status, headers, body = self.request('GET', '/?token=' + self.server.token)
        self.assertEqual(status, 200)
        self.assertIn(b'Seleziona', body)
        self.assertIn("default-src 'none'", headers['Content-Security-Policy'])

    def test_foreign_origin_is_rejected(self):
        headers = {**self.auth(), 'Origin': 'https://example.com'}
        self.assertEqual(self.request('POST', '/api/cancel', b'', headers)[0], 403)

    def test_foreign_host_is_rejected(self):
        self.assertEqual(self.request('GET', '/api/status', headers={**self.auth(), 'Host': 'example.com'})[0], 403)

    def test_no_report_is_not_a_download(self):
        self.assertEqual(self.request('GET', '/report', headers=self.auth())[0], 404)

    def test_wrong_endpoint_does_not_expose_files(self):
        self.assertEqual(self.request('GET', '/../../app.py', headers=self.auth())[0], 404)

    def test_oversize_request_rejected_before_read(self):
        headers = {**self.auth(), 'Content-Length': str(513 * 1024 * 1024)}
        self.assertEqual(self.request('POST', '/api/scan', b'', headers)[0], 413)

    def test_browser_connection_reset_does_not_emit_server_error(self):
        import os, socket, struct, sys
        started=threading.Event();finished=threading.Event();errors=[]
        base=self.app.Handler
        class ObservedHandler(base):
            def handle(handler):
                started.set()
                try:super().handle()
                finally:finished.set()
        self.server.RequestHandlerClass=ObservedHandler
        self.server.handle_error=lambda request,address:errors.append(type(sys.exc_info()[1]).__name__)
        connection=socket.create_connection(('127.0.0.1',self.port),timeout=3)
        connection.sendall(b'GET /api/status HTTP/1.1')
        self.assertTrue(started.wait(3))
        connection.setsockopt(socket.SOL_SOCKET,socket.SO_LINGER,struct.pack('hh' if os.name=='nt' else 'ii',1,0))
        connection.close()
        self.assertTrue(finished.wait(3))
        self.assertEqual(errors,[])
        self.assertEqual(self.request('GET','/api/status',headers=self.auth())[0],200)

    def test_status_is_json(self):
        status, _, data = self.request('GET', '/api/status', headers=self.auth())
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(data)['busy'], False)


if __name__ == '__main__':
    unittest.main()
