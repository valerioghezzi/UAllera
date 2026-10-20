import io
import unittest
import zipfile

try:
    from jamovi_audit import metadata
except ImportError:
    metadata = None


def archive(clock=(2026, 10, 20, 9, 15, 18)):
    out = io.BytesIO()
    with zipfile.ZipFile(out, 'w') as z:
        z.writestr(zipfile.ZipInfo('metadata.json', clock), '{}')
        z.writestr(zipfile.ZipInfo('data.bin', clock), b'')
    return out.getvalue()


def student(key, core=None, zip_entries=None):
    report = {'key': key, 'label': key, 'paths': {'docx': [key + '/relazione.docx'], 'omv': [key + '/relazione.omv']}}
    docs = {'docx': {'metadata': core or {}}, 'omv': {'metadata': {}, 'zip_metadata': zip_entries or {}}}
    return report, docs


class MetadataTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(metadata, 'metadata module required')

    def test_word_aware_dates_retain_offset_and_compare_actual_instants(self):
        summary = metadata.inspect_document_metadata({'metadata': {'created': '2026-10-20T11:00:00+02:00', 'modified': '2026-10-20T10:00:00Z'}}, 'docx')
        self.assertEqual(summary['dates']['created']['timezone'], '+02:00')
        self.assertEqual(summary['dates']['modified']['timezone'], 'UTC')
        self.assertEqual(metadata.metadata_anomalies(summary), [])

    def test_zip_dos_timestamp_has_unknown_timezone_and_two_second_precision(self):
        entries = metadata.extract_zip_metadata(archive())
        value = entries['metadata.json']
        self.assertEqual(value['iso'], '2026-10-20T09:15:18')
        self.assertEqual(value['timezone'], 'unknown')
        self.assertEqual(value['precision_seconds'], 2)
        self.assertEqual(value['source'], 'ZIP entry')
        summary = metadata.inspect_document_metadata({'zip_metadata': entries}, 'omv')
        self.assertEqual(summary['zip_entries']['count'], 2)
        self.assertEqual(summary['zip_entries']['min'], summary['zip_entries']['max'])

    def test_outer_archive_clock_is_preserved_but_not_internal_save_clock(self):
        outer = {'iso': '2026-10-20T09:15:18', 'timezone': 'unknown', 'source': 'ZIP entry', 'precision_seconds': 2, 'valid': True}
        summary = metadata.inspect_document_metadata({'metadata': {}}, 'omv', outer)
        self.assertEqual(summary['source_zip_metadata'], outer)
        self.assertIsNone(summary['zip_entries']['max'])

    def test_close_batch_timestamps_alone_never_flag_pairs(self):
        reports, parsed = [], {}
        for key in ['a', 'b', 'c']:
            report, docs = student(key, {'modified': '2026-10-20T10:00:00Z'}, metadata.extract_zip_metadata(archive()))
            reports.append(report); parsed[key] = docs
        self.assertEqual(metadata.metadata_cross_checks(reports, parsed, []), [])

    def test_same_specific_author_is_a_single_weak_pair_with_explicit_limits(self):
        a, ad = student('a', {'creator': 'Giulia Esempio', 'lastModifiedBy': 'Giulia Esempio'})
        b, bd = student('b', {'creator': ' giulia  esempio '})
        findings = metadata.metadata_cross_checks([a, b], {'a': ad, 'b': bd}, [])
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]['kind'], 'metadata_autore_condiviso')
        self.assertEqual(findings[0]['students'], ['a', 'b'])
        self.assertIn('debole', findings[0]['limits'])
        self.assertIn('modello', findings[0]['limits'])
        self.assertIn('non', findings[0]['limits'].lower())

    def test_generic_default_authors_are_excluded(self):
        for author in ['User', 'Microsoft Office User', 'python-docx', 'Administrator', 'Utente']:
            with self.subTest(author=author):
                a, ad = student('a', {'creator': author}); b, bd = student('b', {'creator': author})
                self.assertEqual(metadata.metadata_cross_checks([a, b], {'a': ad, 'b': bd}, []), [])

    def test_shared_assignment_template_author_is_excluded(self):
        a, ad = student('a', {'creator': 'Tutor Specifico'}); b, bd = student('b', {'creator': 'Tutor Specifico'})
        reference = {'template_metadata': {'creator': 'Tutor Specifico'}}
        self.assertEqual(metadata.metadata_cross_checks([a, b], {'a': ad, 'b': bd}, [], reference), [])

    def test_omv_author_is_compared_when_word_is_missing(self):
        a, ad = student('a'); b, bd = student('b')
        a['paths']['docx'] = []; b['paths']['docx'] = []
        ad['docx'] = None; bd['docx'] = None
        ad['omv']['metadata'] = {'author': 'Giulia Esempio'}
        bd['omv']['metadata'] = {'author': 'Giulia Esempio'}
        findings = metadata.metadata_cross_checks([a, b], {'a': ad, 'b': bd}, [])
        self.assertEqual(len(findings), 1)
        self.assertIn('metadata.json:author', findings[0]['evidence'][0]['location_left'])

    def test_comparison_without_content_evidence_cannot_corroborate_time(self):
        a, ad = student('a', {'modified': '2026-10-20T10:00:00Z'})
        b, bd = student('b', {'modified': '2026-10-20T10:00:00Z'})
        previous = [{'students': ['a', 'b'], 'kind': 'testo_coincidente', 'evidence': []}]
        self.assertEqual(metadata.metadata_cross_checks([a, b], {'a': ad, 'b': bd}, previous), [])

    def test_same_folder_different_versions_does_not_flag_author(self):
        a, ad = student('a', {'creator': 'Giulia Esempio'}); b, bd = student('b', {'creator': 'Giulia Esempio'})
        b['paths'] = a['paths']
        self.assertEqual(metadata.metadata_cross_checks([a, b], {'a': ad, 'b': bd}, []), [])

    def test_copy_match_and_timestamps_add_only_one_corroboration_record(self):
        entries = metadata.extract_zip_metadata(archive())
        a, ad = student('a', {'modified': '2026-10-20T10:00:00Z'}, entries)
        b, bd = student('b', {'modified': '2026-10-20T10:01:00Z'}, entries)
        original = [{'students': ['a', 'b'], 'kind': 'testo_coincidente', 'evidence': [{'component': 'testo'}]}]
        findings = metadata.metadata_cross_checks([a, b], {'a': ad, 'b': bd}, original)
        self.assertEqual(len(original), 1)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]['kind'], 'metadata_correlazione_temporale')
        self.assertEqual(findings[0]['corroborates'], ['testo_coincidente'])
        self.assertEqual({e['comparison_basis'] for e in findings[0]['evidence']}, {'known_timezone', 'local_clock_unknown_timezone'})
        self.assertTrue(all(e['gap_seconds'] <= 120 for e in findings[0]['evidence']))
        self.assertIn('non indipendente', findings[0]['limits'])

    def test_unknown_word_timezones_are_not_assumed_comparable(self):
        a, ad = student('a', {'modified': '2026-10-20T10:00:00'})
        b, bd = student('b', {'modified': '2026-10-20T10:00:00'})
        previous = [{'students': ['a', 'b'], 'kind': 'testo_coincidente'}]
        self.assertEqual(metadata.metadata_cross_checks([a, b], {'a': ad, 'b': bd}, previous), [])

    def test_aware_word_and_unknown_zip_clock_are_never_mixed(self):
        a, ad = student('a', {'modified': '2026-10-20T09:15:18Z'})
        b, bd = student('b', {}, metadata.extract_zip_metadata(archive()))
        previous = [{'students': ['a', 'b'], 'kind': 'omv_identico'}]
        self.assertEqual(metadata.metadata_cross_checks([a, b], {'a': ad, 'b': bd}, previous), [])

    def test_created_after_modified_is_metadata_review_with_clock_import_limit(self):
        summary = metadata.inspect_document_metadata({'metadata': {'created': '2026-10-20T11:00:00Z', 'modified': '2026-10-20T10:00:00Z'}}, 'docx')
        findings = metadata.metadata_anomalies(summary)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]['status'], 'Da verificare')
        self.assertIn('orologio', findings[0]['expected'])
        self.assertIn('importazione', findings[0]['expected'])

    def test_malformed_dates_and_corrupt_zip_are_unknown_without_exception(self):
        summary = metadata.inspect_document_metadata({'metadata': {'created': 'tomorrow', 'modified': '2026-99-77'}}, 'docx')
        self.assertFalse(summary['dates']['created']['valid'])
        self.assertEqual(metadata.metadata_anomalies(summary), [])
        self.assertEqual(metadata.extract_zip_metadata(b'not a zip'), {})

    def test_missing_metadata_is_unknown_without_anomaly(self):
        summary = metadata.inspect_document_metadata({}, 'docx')
        self.assertEqual(summary['authors'], [])
        self.assertIsNone(summary['dates']['created']['iso'])
        self.assertEqual(metadata.metadata_anomalies(summary), [])


if __name__ == '__main__':
    unittest.main()
