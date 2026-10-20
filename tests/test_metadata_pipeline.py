"""Synthetic archive tests: source clocks survive read-only end-to-end checks."""
import hashlib
import io
import unittest
from unittest.mock import patch
import zipfile

from jamovi_audit.pipeline import audit_zip
from test_pipeline import empty_omv


def zip_at(entries, clock):
    out = io.BytesIO()
    with zipfile.ZipFile(out, 'w') as archive:
        for name, value in entries.items():
            archive.writestr(zipfile.ZipInfo(name, clock), value)
    return out.getvalue()


def docx_at(core, clock=(2025, 1, 2, 10, 0, 0), text='Osservazione personale'):
    return zip_at({
        'word/document.xml': '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>' + text + '</w:t></w:r></w:p></w:body></w:document>',
        'docProps/core.xml': '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/">' + core + '</cp:coreProperties>',
    }, clock)


class MetadataPipelineTests(unittest.TestCase):
    def test_source_and_internal_clocks_remain_separate_and_anomaly_is_review_only(self):
        core = '<dc:creator>Giulia Esempio</dc:creator><dcterms:created>2026-10-20T12:00:00Z</dcterms:created><dcterms:modified>2026-10-20T11:00:00Z</dcterms:modified>'
        original = zip_at({'Studente/a.docx': docx_at(core)}, (2026, 10, 20, 15, 0, 0))
        digest = hashlib.sha256(original).hexdigest()
        with patch('os.path.getmtime', side_effect=AssertionError('Filesystem timestamps forbidden')), patch('os.path.getctime', side_effect=AssertionError('Filesystem timestamps forbidden')):
            result = audit_zip(original)
        report = result['reports'][0]
        self.assertIn('metadata', report)
        summary = report['metadata']['docx']
        self.assertEqual(summary['authors'][0]['value'], 'Giulia Esempio')
        self.assertEqual(summary['zip_entries']['max'], '2025-01-02T10:00:00')
        self.assertEqual(summary['source_zip_metadata']['iso'], '2026-10-20T15:00:00')
        self.assertEqual(summary['zip_entries']['timezone'], 'unknown')
        anomalies = [f for f in report['findings'] if f.get('id') == 'METADATA.CREATED_AFTER_MODIFIED']
        self.assertEqual(len(anomalies), 1)
        self.assertEqual(anomalies[0]['status'], 'Da verificare')
        self.assertEqual(hashlib.sha256(original).hexdigest(), digest)

    def test_real_distinct_folders_with_shared_author_emit_weak_metadata_pair(self):
        core = '<dc:creator>Giulia Esempio</dc:creator>'
        original = zip_at({'A/a.docx': docx_at(core, text='Una breve osservazione'), 'B/b.docx': docx_at(core, text='Un commento indipendente')}, (2026, 10, 20, 15, 0, 0))
        result = audit_zip(original)
        self.assertEqual([c['kind'] for c in result['comparisons']], ['metadata_autore_condiviso'])
        self.assertIn('debole', result['comparisons'][0]['limits'])
        self.assertIn('diagnostics', vars(result['comparisons']))

    def test_simultaneous_modifications_without_content_match_do_not_emit_pair(self):
        core = '<dcterms:modified>2026-10-20T11:00:00Z</dcterms:modified>'
        original = zip_at({'A/a.docx': docx_at(core, text='Un testo breve'), 'B/b.docx': docx_at(core, text='Altre osservazioni personali')}, (2026, 10, 20, 15, 0, 0))
        result = audit_zip(original)
        self.assertEqual(result['comparisons'], [])
        self.assertTrue(all(r.get('metadata', {}).get('docx', {}).get('dates', {}).get('modified', {}).get('valid') for r in result['reports']))

    def test_exact_omv_match_receives_one_clock_corroboration_and_keeps_diagnostics(self):
        original = zip_at({'A/a.omv': empty_omv(), 'B/b.omv': empty_omv()}, (2026, 10, 20, 15, 0, 0))
        # Reuse exactly one binary document to avoid a possible second-boundary
        # difference in ZIP fixture construction.
        with zipfile.ZipFile(io.BytesIO(original)) as archive:
            omv = archive.read('A/a.omv')
        original = zip_at({'A/a.omv': omv, 'B/b.omv': omv}, (2026, 10, 20, 15, 0, 0))
        result = audit_zip(original)
        kinds = [c['kind'] for c in result['comparisons']]
        self.assertEqual(kinds.count('omv_identico'), 1)
        self.assertEqual(kinds.count('metadata_correlazione_temporale'), 1)
        corroboration = next(c for c in result['comparisons'] if c['kind'] == 'metadata_correlazione_temporale')
        self.assertEqual(corroboration['corroborates'], ['omv_identico'])
        self.assertEqual(corroboration['evidence'][0]['comparison_basis'], 'local_clock_unknown_timezone')
        self.assertIn('pairs_total', result['comparisons'].diagnostics)

    def test_missing_and_malformed_metadata_remain_unknown(self):
        original = zip_at({'A/a.docx': docx_at('<dcterms:created>not-a-date</dcterms:created>'), 'B/b.omv': b'corrupt'}, (2026, 10, 20, 15, 0, 0))
        result = audit_zip(original)
        self.assertTrue(all('metadata' in report for report in result['reports']))
        first = next(r for r in result['reports'] if r['label'] == 'A')
        self.assertFalse(first['metadata']['docx']['dates']['created']['valid'])
        self.assertEqual(first['metadata']['omv']['authors'], [])
        self.assertIsNone(first['metadata']['omv']['zip_entries']['max'])
        self.assertFalse(any(f.get('id') == 'METADATA.CREATED_AFTER_MODIFIED' for report in result['reports'] for f in report['findings']))


if __name__ == '__main__':
    unittest.main()
