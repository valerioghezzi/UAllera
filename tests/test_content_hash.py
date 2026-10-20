import io
import unittest
import zipfile
from jamovi_audit.readers import read_omv


class ContentHashTests(unittest.TestCase):
    def test_recompression_changes_raw_hash_only(self):
        entries={'meta':b'jamovi-Archive-Version: 11.0\n',
                 'metadata.json':b'{"dataSet":{"rowCount":0,"fields":[]}}',
                 'data.bin':b'', 'xdata.json':b'{}'}
        buffers=[]
        for compression in [zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED]:
            b=io.BytesIO()
            with zipfile.ZipFile(b,'w',compression) as z:
                for k,v in entries.items():z.writestr(k,v)
            buffers.append(read_omv(b.getvalue()))
        self.assertTrue(buffers[0].get('content_sha256'))
        self.assertEqual(buffers[0]['content_sha256'],buffers[1]['content_sha256'])
        self.assertNotEqual(buffers[0]['sha256'],buffers[1]['sha256'])
