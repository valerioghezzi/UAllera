"""Optional development-only generative invariants.

Run with an isolated development environment and requirements-dev.txt. Absence of
Hypothesis is an explicit skip, never represented as executed property coverage.
The deterministic profile runs 60 examples per property without network/models.
"""
import io
import unittest
import zipfile
from xml.sax.saxutils import escape

try:
    from hypothesis import given, settings, strategies as st
except ModuleNotFoundError as exc:
    if exc.name != 'hypothesis':
        raise
    raise unittest.SkipTest('Hypothesis is not installed; optional property tests were NOT executed') from exc

from jamovi_audit.archive import inventory, safe_zip
from jamovi_audit.readers import read_docx, read_omv
from jamovi_audit.report import compare_submissions

PROPERTY = settings(max_examples=60, deadline=None, database=None, derandomize=True)
TEXT = st.text(alphabet="ABCDE abcdefghijklmnopqrstuvwxyz èéàΩ猫<>&\"'\n\t", min_size=1, max_size=160)


def archive(pairs, compression=zipfile.ZIP_STORED):
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', compression) as bundle:
        for name, content in pairs:
            bundle.writestr(zipfile.ZipInfo(name, (2024, 1, 2, 3, 4, 6)), content,
                            compress_type=compression)
    return output.getvalue()


def omv_entries(payload):
    return [('meta', b'jamovi-Archive-Version: 11.0\n'),
            ('metadata.json', b'{"dataSet":{"rowCount":0,"fields":[]}}'),
            ('data.bin', b''), ('xdata.json', b'{}'), ('synthetic.bin', payload)]


class GenerativeInvariants(unittest.TestCase):
    @PROPERTY
    @given(TEXT, st.sampled_from(('utf-8', 'utf-16')))
    def test_word_source_encoding_preserves_exact_evidence_text(self, text, encoding):
        xml = ('<?xml version="1.0" encoding="' + encoding + '"?>'
               '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
               '<w:body><w:p><w:r><w:t xml:space="preserve">' + escape(text) +
               '</w:t></w:r></w:p></w:body></w:document>')
        document = read_docx(archive([('word/document.xml', xml.encode(encoding))]))
        self.assertFalse(document['errors'])
        self.assertEqual(text, document['evidence_sources'][0]['text'])
        self.assertEqual(text, document['paragraphs'][0]['text'])

    @PROPERTY
    @given(st.dictionaries(st.text(alphabet='abcè猫012', min_size=1, max_size=16),
                           st.binary(max_size=96), max_size=12))
    def test_safe_zip_roundtrips_safe_unicode_names_and_arbitrary_bytes(self, generated):
        # Prefix avoids Windows reserved devices; generated names exclude path
        # separators, trailing dots/spaces, and case/NFC collision alternatives.
        entries = {'Classe/doc_' + name + '.bin': content for name, content in generated.items()}
        self.assertEqual(entries, safe_zip(archive(list(entries.items()))))

    @PROPERTY
    @given(st.sampled_from(('../', '/absolute/', 'C:/', 'a/../', 'a\\')),
           st.text(alphabet='abcdef012', min_size=1, max_size=20))
    def test_zip_unsafe_name_prefixes_are_rejected(self, prefix, filename):
        with self.assertRaises(ValueError):
            safe_zip(archive([(prefix + filename + '.docx', b'x')]))

    @PROPERTY
    @given(st.binary(max_size=512))
    def test_content_hash_ignores_zip_order_and_compression_but_not_changed_bytes(self, payload):
        entries = omv_entries(payload)
        first = read_omv(archive(entries))
        reordered = read_omv(archive(list(reversed(entries)), zipfile.ZIP_DEFLATED))
        changed = read_omv(archive(omv_entries(payload + b'\x00')))
        self.assertFalse(first['errors'])
        self.assertFalse(reordered['errors'])
        self.assertEqual(first['content_sha256'], reordered['content_sha256'])
        self.assertNotEqual(first['content_sha256'], changed['content_sha256'])
        self.assertNotEqual(first['sha256'], reordered['sha256'])

    @PROPERTY
    @given(st.lists(st.integers(min_value=0, max_value=999), min_size=1, max_size=15, unique=True))
    def test_inventory_identity_and_pairing_ignore_dictionary_insertion_order(self, identities):
        entries = {f'Classe/Persona_{ident}/{kind}/elaborato.{suffix}': b'x'
                   for ident in identities for kind, suffix in (('Analisi', 'omv'), ('Relazione', 'docx'))}
        first = inventory(entries)
        self.assertEqual(first, inventory(dict(reversed(list(entries.items())))))
        self.assertEqual(len(identities), len(first))
        self.assertEqual(len(first), len({group['key'] for group in first}))
        self.assertTrue(all(len(group['omv']) == len(group['docx']) == 1 for group in first))

    @PROPERTY
    @given(st.lists(st.integers(min_value=0, max_value=2), min_size=2, max_size=6))
    def test_comparison_pair_and_rule_set_is_invariant_under_student_order(self, groups):
        reports = [{'key': str(i), 'label': str(i), 'findings': []} for i in range(len(groups))]
        passages = [
            'La lucciola arancione custodisce silenziosamente ricordi sorprendenti della passeggiata notturna.',
            'Questa biblioteca azzurra nasconde curiosamente avventure straordinarie dietro scaffali polverosi dimenticati.',
            'Una montagna verde raccoglie lentamente frammenti preziosi provenienti dal viaggio immaginario serale.',
        ]
        parsed = {str(i): {'docx': {'paragraphs': [{'index': 1, 'text': passages[group]}]}}
                  for i, group in enumerate(groups)}
        def pairs(order):
            return {(tuple(sorted(match['students'])), match['kind'])
                    for match in compare_submissions(order, parsed)}
        observed = pairs(reports)
        self.assertEqual(observed, pairs(list(reversed(reports))))
        for left, right in (match[0] for match in observed):
            self.assertEqual(groups[int(left)], groups[int(right)])


if __name__ == '__main__':
    unittest.main()
