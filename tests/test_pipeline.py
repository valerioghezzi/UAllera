import io
import json
import unittest
import zipfile
from pathlib import Path


def archive(entries):
    b = io.BytesIO()
    with zipfile.ZipFile(b, 'w', zipfile.ZIP_DEFLATED) as z:
        for p, content in entries.items():
            z.writestr(p, content)
    return b.getvalue()


def empty_omv():
    return archive({'meta': 'Manifest-Version: 1.0\nData-Archive-Version: 1.0.2\njamovi-Archive-Version: 11.0\nCreated-By: jamovi 2.7.4\n',
                    'metadata.json': json.dumps({'dataSet': {'rowCount': 0, 'columnCount': 0, 'fields': []}}),
                    'data.bin': b'', 'xdata.json': '{}'})


def word(text='Parte 1 Relazione di prova'):
    return archive({'word/document.xml': '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>' + text + '</w:t></w:r></w:p></w:body></w:document>'})


class PipelineTests(unittest.TestCase):
    def call(self, data, **kwargs):
        try:
            from jamovi_audit.pipeline import audit_zip
        except ImportError:
            self.fail('audit_zip non implementata')
        return audit_zip(data, **kwargs)

    def test_one_report_contains_both_students(self):
        data = archive({'Download/Rossi_Maria/Rossi_Maria_relazione.omv': empty_omv(),
                        'Download/Rossi_Maria/Rossi_Maria_relazione.docx': word(),
                        'Download/Bianchi_Luca/Bianchi_Luca_relazione.omv': empty_omv()})
        result = self.call(data)
        self.assertEqual(len(result['reports']), 2)
        self.assertIn('Rossi', result['html'])
        self.assertIn('Bianchi', result['html'])
        self.assertEqual(result['html'].lower().count('<html'), 1)

    def test_corrupt_student_does_not_stop_other_student(self):
        data = archive({'Rossi/relazione.omv': b'broken', 'Bianchi/relazione.omv': empty_omv()})
        result = self.call(data)
        self.assertEqual(len(result['reports']), 2)
        self.assertTrue(any('verific' in f.get('status', '').lower()
                            for r in result['reports'] for f in r['findings']))

    def test_no_consignments_is_explicit_error(self):
        with self.assertRaisesRegex(ValueError, '(?i)(nessun|nessuna|omv)'):
            self.call(archive({'leggi.txt': 'Solo istruzioni'}))

    def test_cancel_before_scan(self):
        with self.assertRaises(InterruptedError):
            self.call(archive({'Rossi/a.omv': empty_omv()}), cancelled=lambda: True)

    def test_repeated_scan_has_same_findings(self):
        data = archive({'Rossi/a.omv': empty_omv()})
        a, b = self.call(data), self.call(data)
        self.assertEqual(a['reports'], b['reports'])

    def test_network_connections_are_not_needed(self):
        import socket
        old = socket.create_connection
        def no_network(*a, **kw):
            raise AssertionError('Tentativo di rete nel controllo')
        socket.create_connection = no_network
        try:
            result = self.call(archive({'Rossi/a.omv': empty_omv()}))
            self.assertEqual(len(result['reports']), 1)
        finally:
            socket.create_connection = old


if __name__ == '__main__':
    unittest.main()
