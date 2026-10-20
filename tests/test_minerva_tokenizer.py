"""Offline graph tests plus opt-in pinned Hugging Face oracle comparisons."""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    from jamovi_audit import minerva_tokenizer as tokenizer
except ImportError:
    tokenizer = None


def graph():
    byte_values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    codes = dict(zip(byte_values, byte_values))
    missing = 256
    for value in range(256):
        if value not in codes:
            codes[value] = missing; missing += 1
    vocab = {'<s>': 0, '</s>': 1, '<unk>': 2}
    for value in range(256):
        if value != 13:  # Official model has no CR byte token/fallback/unknown replacement.
            vocab[chr(codes[value])] = len(vocab)
    merges = [['a', 'b'], ['b', 'c'], ['ab', 'c']]
    for left, right in merges:
        vocab[left + right] = len(vocab)
    added = []
    for content, token_id in (('<s>', 0), ('</s>', 1), ('<unk>', 2), ('<|start_header_id|>', 51200),
                              ('<|end_header_id|>', 51201), ('<|eot_id|>', 51202)):
        added.append(dict(id=token_id, content=content, single_word=False, lstrip=False, rstrip=False, normalized=False, special=True))
    return dict(version='1.0', truncation=None, padding=None, added_tokens=added,
        normalizer={'type': 'Sequence', 'normalizers': [{'type': 'NFKC'}]},
        pre_tokenizer=dict(type='ByteLevel', add_prefix_space=True, trim_offsets=True, use_regex=True),
        decoder=dict(type='ByteLevel', add_prefix_space=True, trim_offsets=True, use_regex=True),
        post_processor=dict(type='TemplateProcessing',
            single=[{'SpecialToken': {'id': '<s>', 'type_id': 0}}, {'Sequence': {'id': 'A', 'type_id': 0}}],
            pair=[{'SpecialToken': {'id': '<s>', 'type_id': 0}}, {'Sequence': {'id': 'A', 'type_id': 0}},
                  {'SpecialToken': {'id': '<s>', 'type_id': 1}}, {'Sequence': {'id': 'B', 'type_id': 1}}],
            special_tokens={'<s>': {'id': '<s>', 'ids': [0], 'tokens': ['<s>']}}),
        model=dict(type='BPE', dropout=None, unk_token=None, continuing_subword_prefix=None,
            end_of_word_suffix=None, fuse_unk=False, byte_fallback=True, ignore_merges=False,
            vocab=vocab, merges=merges))


class TokenizerTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(tokenizer, 'The stdlib official Minerva tokenizer is missing')
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'tokenizer.json'
        self.data = graph()
        self.install(self.data)

    def install(self, value):
        raw = json.dumps(value, ensure_ascii=False).encode()
        self.path.write_bytes(raw)
        for item in (patch.object(tokenizer, 'TOKENIZER_SIZE', len(raw)),
                     patch.object(tokenizer, 'TOKENIZER_SHA256', hashlib.sha256(raw).hexdigest())):
            item.start(); self.addCleanup(item.stop)

    def test_checksum_and_size_are_verified_before_graph_load(self):
        size = self.path.stat().st_size
        self.path.write_bytes(b'bad')
        with self.assertRaisesRegex(ValueError, '(?i)integrit|SHA|dimension'):
            tokenizer.OfficialTokenizer(self.path)
        self.path.write_bytes(b'x' * size)
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            tokenizer.OfficialTokenizer(self.path)

    def test_unsupported_declared_graph_is_rejected(self):
        for change in ('prefix', 'normalizer', 'added_normalized', 'dropout'):
            with self.subTest(change=change):
                data = copy.deepcopy(self.data)
                if change == 'prefix': data['pre_tokenizer']['add_prefix_space'] = False
                if change == 'normalizer': data['normalizer'] = {'type': 'NFC'}
                if change == 'added_normalized': data['added_tokens'][3]['normalized'] = True
                if change == 'dropout': data['model']['dropout'] = .1
                self.install(data)
                with self.assertRaisesRegex(ValueError, 'support|grafo'):
                    tokenizer.OfficialTokenizer(self.path)

    def test_exact_rendered_text_never_gains_an_automatic_bos(self):
        engine = tokenizer.OfficialTokenizer(self.path)
        self.assertEqual([], engine.encode(''))
        self.assertEqual([0], engine.encode('<s>'))
        result = engine.encode('<s><|start_header_id|>system<|end_header_id|>\n\n')
        self.assertEqual(1, result.count(0)); self.assertEqual([0, 51200], result[:2])
        self.assertEqual(51201, result[result.index(51200) + 8])
        self.assertEqual(self.data['model']['vocab']['Ġ'], result[2])
        self.assertEqual(result, tokenizer.encode('<s><|start_header_id|>system<|end_header_id|>\n\n', self.path))

    def test_specials_are_split_before_nfkc_and_fullwidth_forms_stay_ordinary(self):
        engine = tokenizer.OfficialTokenizer(self.path)
        self.assertEqual([51202], engine.encode('<|eot_id|>'))
        self.assertNotIn(51202, engine.encode('＜|eot_id|＞'))
        self.assertNotIn(51200, engine.encode('＜|start_header_id|＞'))
        self.assertNotIn(0, engine.encode('＜s＞'))

    def test_bpe_uses_merge_priority_and_preserves_unicode_byte_encoding(self):
        engine = tokenizer.OfficialTokenizer(self.path)
        vocab = self.data['model']['vocab']
        self.assertEqual([vocab['Ġ'], vocab['abc']], engine.encode('abc'))
        self.assertEqual(engine.encode('é'), engine.encode('e\u0301'))
        self.assertEqual(engine.encode('ff'), engine.encode('ﬀ'))
        self.assertEqual(engine.encode('a\rb'), engine.encode('a') + [vocab['b']])

    def test_unicode9_nfkc_pins_old_normalization_and_reorders_combining_marks(self):
        expected = [('e\u0301', 'é'), ('ＡＢＣ ﬃ ①', 'ABC ffi 1'), ('\u1100\u1161\u11a8', '각'),
                    ('a\u0315\u0300', 'à\u0315'), ('\U0001e030', '\U0001e030'), ('㋿', '㋿'),
                    ('a\u0315\u07fd\u0300', 'a\u0315\u07fd\u0300')]
        for text, result in expected:
            with self.subTest(text=text): self.assertEqual(result, tokenizer._nfkc9(text))

    def test_pre_tokenizer_matches_gpt2_unicode_regex_whitespace_and_contractions(self):
        expected = [(' hello  world\n\n', [' hello', ' ', ' world', '\n\n']),
                    ("I'm I'M we'd!", ['I', "'m", ' I', "'", 'M', ' we', "'d", '!']),
                    ('a\x1cb', ['a', '\x1c', 'b']), ('a\u00a0b', ['a', '\u00a0', 'b']),
                    ('\t\nX', ['\t', '\n', 'X']), (' 123456', [' 123456'])]
        for text, result in expected:
            with self.subTest(text=text): self.assertEqual(result, list(tokenizer._pieces(text)))

    def test_invalid_unicode_and_non_text_fail_closed(self):
        engine = tokenizer.OfficialTokenizer(self.path)
        with self.assertRaises(ValueError): engine.encode('\ud800')
        with self.assertRaises(TypeError): engine.encode(b'bytes')


@unittest.skipUnless(os.environ.get('UALLERA_OFFICIAL_TOKENIZER') and os.environ.get('UALLERA_TOKENIZER_ORACLE'),
                     'Set pinned tokenizer and public HF oracle paths for offline conformance tests')
class OfficialOracleTests(unittest.TestCase):
    def test_all_saved_rendered_prompts_match_hf_ids_exactly(self):
        self.assertIsNotNone(tokenizer)
        engine = tokenizer.OfficialTokenizer(os.environ['UALLERA_OFFICIAL_TOKENIZER'])
        paths = os.environ['UALLERA_TOKENIZER_ORACLE'].split(os.pathsep)
        total = 0
        for path in paths:
            oracle = json.loads(Path(path).read_text())
            self.assertEqual(tokenizer.TOKENIZER_SHA256, oracle['tokenizer_sha256'])
            for case in oracle['cases']:
                with self.subTest(path=Path(path).name, case=case['id']):
                    self.assertEqual(case['hf_ids'], engine.encode(case['prompt']))
                    total += 1
        self.assertGreaterEqual(total, 48)


if __name__ == '__main__':
    unittest.main()
