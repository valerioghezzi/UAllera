"""Original-source provenance: real DOCX XML, without model or service mocks."""
import io
import unittest
import zipfile
from xml.sax.saxutils import escape

from jamovi_audit.readers import read_docx


def document(body, extra=None):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as archive:
        archive.writestr('word/document.xml',
                         '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                         '<w:body>' + body + '</w:body></w:document>')
        for name, content in (extra or {}).items():
            archive.writestr(name, content)
    return stream.getvalue()


def paragraph(text):
    return '<w:p><w:r><w:t xml:space="preserve">' + escape(text) + '</w:t></w:r></w:p>'


def table(content):
    return '<w:tbl><w:tr><w:tc>' + content + '</w:tc></w:tr></w:tbl>'


class DocxEvidenceTests(unittest.TestCase):
    def test_preserves_more_than_128_sources_and_interleaved_tables_losslessly(self):
        expected = ['  È identico 🧪  '] * 130
        body = ''.join(paragraph(text) for text in expected[:65])
        body += table(paragraph('tabella intermedia'))
        body += ''.join(paragraph(text) for text in expected[65:])
        result = read_docx(document(body))
        self.assertFalse(result['errors'])
        sources = result.get('evidence_sources', [])
        self.assertEqual([source['text'] for source in sources],
                         expected[:65] + ['tabella intermedia'] + expected[65:])
        self.assertEqual(len({source['id'] for source in sources}), 131)
        self.assertTrue(all(source['kind'] == 'paragraph' for source in sources))
        self.assertEqual(sources, read_docx(document(body))['evidence_sources'])
        self.assertEqual(result['evidence_coverage']['omissions'], [])
        self.assertEqual(len(result['paragraphs']), 130)
        self.assertEqual(result['tables'][0]['rows'], [['tabella intermedia']])

    def test_nested_tables_and_wrappers_preserve_order_without_cell_duplication(self):
        body = paragraph('inizio')
        body += '<w:sdt><w:sdtPr/><w:sdtContent>' + paragraph('controllo')
        body += table(paragraph('prima') + table(paragraph('annidata')) + paragraph('dopo'))
        body += '</w:sdtContent></w:sdt>'
        body += '<w:customXml>' + paragraph('custom') + '</w:customXml>'
        body += paragraph('fine')
        result = read_docx(document(body))
        sources = result.get('evidence_sources', [])
        self.assertEqual([source['text'] for source in sources],
                         ['inizio', 'controllo', 'prima', 'annidata', 'dopo', 'custom', 'fine'])
        self.assertEqual(len({source['id'] for source in sources}), 7)
        self.assertIn('annidata', sources[3]['location'].lower())
        self.assertEqual(result['evidence_coverage']['omissions'], [])
        # Adding the manifest must not silently change the legacy reader schema.
        self.assertEqual([p['text'] for p in result['paragraphs']], ['inizio', 'fine'])
        self.assertEqual(result['tables'], [])

    def test_paragraph_sources_preserve_tabs_breaks_and_empty_paragraphs(self):
        body = '<w:p><w:r><w:t>È</w:t><w:tab/><w:t>β</w:t><w:br/><w:t>後</w:t><w:cr/></w:r></w:p><w:p/>'
        result = read_docx(document(body))
        self.assertEqual([s['text'] for s in result.get('evidence_sources', [])], ['È\tβ\n後\n', ''])

    def test_long_single_paragraph_is_not_clipped_or_normalized(self):
        text = '  Unicode È\u0300 🧪 < & "  ' * 10000
        result = read_docx(document(paragraph(text)))
        self.assertEqual(result.get('evidence_sources', [{}])[0].get('text'), text)

    def test_excludes_textboxes_and_explicitly_reports_unexamined_content(self):
        body = '<w:p><w:r><w:t>originale</w:t><w:drawing><w:txbxContent>'
        body += paragraph('testo nella forma') + '</w:txbxContent></w:drawing></w:r></w:p>'
        body += '<w:altChunk/>'
        extra = {name: '<w:document/>' for name in
                 ('word/header1.xml', 'word/footer1.xml', 'word/footnotes.xml', 'word/endnotes.xml')}
        result = read_docx(document(body, extra))
        self.assertEqual([s['text'] for s in result.get('evidence_sources', [])], ['originale'])
        coverage = result['evidence_coverage']
        self.assertEqual(coverage['scope'], 'body_paragraphs_and_table_paragraphs_in_document_order')
        self.assertTrue({'drawing', 'altChunk', 'header', 'footer', 'footnotes', 'endnotes'}
                        <= {item['kind'] for item in coverage['omissions']})
        self.assertTrue(all(item['part'] and item['location'] and item['reason'] for item in coverage['omissions']))

    def test_unknown_body_content_is_reported_without_claiming_full_coverage(self):
        result = read_docx(document('<w:unknown>' + paragraph('non supportato') + '</w:unknown>'))
        self.assertEqual(result.get('evidence_sources', []), [])
        self.assertTrue(result.get('evidence_coverage', {}).get('omissions'))

    def test_header_with_custom_part_name_is_identified_by_content_type(self):
        content_types = ('<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                         '<Override PartName="/word/sections/title.xml" '
                         'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.header+xml"/>'
                         '</Types>')
        result = read_docx(document(paragraph('corpo'), {
            '[Content_Types].xml': content_types, 'word/sections/title.xml': '<header/>'}))
        omissions = result.get('evidence_coverage', {}).get('omissions', [])
        self.assertTrue(any(row['kind'] == 'header' and row['part'] == 'word/sections/title.xml'
                            for row in omissions))

    def test_unreadable_word_never_claims_source_coverage(self):
        result = read_docx(b'not a docx')
        self.assertTrue(result['errors'])
        self.assertEqual(result.get('evidence_sources'), [])
        self.assertEqual(result.get('evidence_coverage', {}).get('scope'), 'unavailable')


if __name__ == '__main__':
    unittest.main()
