import http.client
import io
import json
import threading
import time
import unittest
import zipfile
import app
from test_pipeline import empty_omv, word


class FullLocalAppTests(unittest.TestCase):
    def test_zip_to_single_report_download_through_real_http(self):
        server=app.make_server(0)
        thread=threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        def request(method,path,body=None):
            conn=http.client.HTTPConnection('127.0.0.1',server.server_address[1],timeout=10)
            conn.request(method,path,body,{'X-Session-Token':server.token})
            r=conn.getresponse();result=(r.status,dict(r.getheaders()),r.read());conn.close()
            return result
        buffer=io.BytesIO()
        with zipfile.ZipFile(buffer,'w') as z:
            z.writestr('Classe/Uno/a.omv',empty_omv());z.writestr('Classe/Uno/a.docx',word())
            z.writestr('Classe/Due/a.omv',b'corrotto')
        try:
            self.assertEqual(request('POST','/api/scan',buffer.getvalue())[0],202)
            deadline=time.monotonic()+20
            while True:
                status,_,data=request('GET','/api/status');state=json.loads(data)
                if not state['busy']:break
                if time.monotonic()>deadline:self.fail('Controllo locale bloccato')
                time.sleep(.02)
            self.assertTrue(state['ready'],state.get('error'))
            self.assertEqual(state['students'],2)
            status,headers,body=request('GET','/report')
            self.assertEqual(status,200)
            self.assertIn('attachment',headers['Content-Disposition'])
            html=body.decode()
            self.assertEqual(html.count('<html'),1)
            self.assertIn('id="student-1"',html);self.assertIn('id="student-2"',html)
            self.assertNotIn(server.token,html)
        finally:
            server.shutdown();server.server_close();thread.join(timeout=3)
