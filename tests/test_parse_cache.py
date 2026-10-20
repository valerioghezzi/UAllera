"""Bounded in-memory parsing reuse, using the real DOCX and OMV readers."""
import concurrent.futures
import json
import threading
import unittest
from unittest.mock import patch

from jamovi_audit.readers import read_docx, read_omv
from test_pipeline import word
from test_readers import omv

try:
    from jamovi_audit import parse_cache
except ImportError:
    parse_cache = None


class ParsingCacheTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(parse_cache, 'The per-audit parsing cache is missing')
        self.cache = parse_cache.ParsingCache()

    def test_identical_docx_reuses_reader_without_student_metadata_bleed(self):
        calls = []
        def reader(data):
            calls.append(data)
            return read_docx(data)
        data = word('Una relazione condivisa fra due consegne.')
        first = self.cache.read('docx', data, reader)
        self.assertFalse(first['errors'])
        first['paragraphs'][0]['text'] = 'Modifica del primo studente'
        first['metadata']['student'] = 'Primo'
        first['metadata_summary'] = {'student': 'Primo'}
        first['zip_metadata'] = {'path': 'Primo/relazione.docx'}
        second = self.cache.read('docx', data, reader)
        self.assertEqual(1, len(calls))
        self.assertEqual('Una relazione condivisa fra due consegne.', second['paragraphs'][0]['text'])
        self.assertNotIn('student', second['metadata'])
        self.assertNotIn('metadata_summary', second)
        self.assertNotIn('zip_metadata', second)
        self.assertEqual((1, 1), (self.cache.stats()['hits'], self.cache.stats()['misses']))

    def test_identical_omv_returns_independent_nested_columns_and_metadata(self):
        data = omv()
        calls = []
        def reader(raw):
            calls.append(raw)
            return read_omv(raw)
        first = self.cache.read('omv', data, reader)
        first['columns']['N']['data'].append(999)
        first['metadata']['student'] = 'Primo'
        second = self.cache.read('omv', data, reader)
        self.assertEqual([4.4, None], second['columns']['N']['data'])
        self.assertNotIn('student', second['metadata'])
        self.assertEqual(1, len(calls))

    def test_first_return_and_hits_do_not_alias_reader_owned_output(self):
        owned = {'errors': [], 'nested': {'value': ['original']}}
        first = self.cache.read('docx', b'one', lambda _: owned)
        first['nested']['value'].append('first')
        owned['nested']['value'].append('reader mutation')
        second = self.cache.read('docx', b'one', lambda _: self.fail('Cache miss'))
        self.assertEqual(['original'], second['nested']['value'])
        self.assertEqual(['original', 'reader mutation'], owned['nested']['value'])

    def test_reader_kind_and_schema_version_have_separate_keys(self):
        calls = []
        def reader(data):
            calls.append(data)
            return {'value': len(calls)}
        self.assertEqual(1, self.cache.read('omv', b'same', reader)['value'])
        self.assertEqual(2, self.cache.read('docx', b'same', reader)['value'])
        with patch.object(parse_cache, 'CACHE_SCHEMA_VERSION', 'next-schema'):
            self.assertEqual(3, self.cache.read('omv', b'same', reader)['value'])
        self.assertEqual(1, self.cache.read('omv', b'same', reader)['value'])
        self.assertEqual(3, len(calls))

    def test_short_integer_list_cannot_hide_resident_size_behind_small_json(self):
        cache = parse_cache.ParsingCache(max_bytes=4096)
        output = {'values': [0] * 1000}
        self.assertLess(len(json.dumps(output, separators=(',', ':')).encode()), 4096)
        cache.read('omv', b'one', lambda _: output)
        self.assertEqual(0, cache.stats()['entries'])
        self.assertEqual(1, cache.stats()['bypassed'])

    def test_lru_hits_keep_recent_entries_and_entry_budget_is_enforced(self):
        cache = parse_cache.ParsingCache(max_entries=2, max_bytes=8192)
        calls = []
        def reader(data):
            calls.append(data)
            return {'text': data.decode()}
        for data in (b'a', b'b', b'a', b'c', b'a', b'b'):
            self.assertEqual(data.decode(), cache.read('docx', data, reader)['text'])
        self.assertEqual([b'a', b'b', b'c', b'b'], calls)
        self.assertEqual(2, cache.stats()['entries'])
        self.assertEqual(2, cache.stats()['evictions'])
        self.assertEqual(2, cache.stats()['hits'])

    def test_encoded_utf8_budget_is_counted_and_evicts_before_overflow(self):
        parsed = {'text': 'è' * 10}
        encoded_size = len(json.dumps(parsed, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8'))
        probe = parse_cache.ParsingCache()
        probe.read('docx', b'probe', lambda _: parsed)
        retained_charge = probe.stats()['bytes']
        cache = parse_cache.ParsingCache(max_entries=32, max_bytes=retained_charge * 2)
        for data in (b'a', b'b', b'c'):
            cache.read('docx', data, lambda _: parsed)
            self.assertLessEqual(cache.stats()['bytes'], retained_charge * 2)
        self.assertEqual(retained_charge * 2, cache.stats()['bytes'])
        self.assertEqual(encoded_size * 2, cache.stats()['encoded_bytes'])
        self.assertEqual('unique_deepcopy_graph+json+entry_overhead', cache.stats()['bytes_accounting'])
        self.assertEqual(2, cache.stats()['entries'])
        self.assertEqual(1, cache.stats()['evictions'])

    def test_oversized_nonserializable_and_error_outputs_are_not_retained(self):
        cycle = {}; cycle['cycle'] = cycle
        for output in ({'text': 'x' * 1000}, {'object': object()}, cycle,
                       {'value': float('nan')}, {'errors': ['Malformed document']}):
            with self.subTest(kind=type(output).__name__, keys=list(output)):
                cache = parse_cache.ParsingCache(max_bytes=64)
                calls = []
                def reader(data):
                    calls.append(data)
                    return output
                first = cache.read('docx', b'input', reader)
                second = cache.read('docx', b'input', reader)
                self.assertIsNotNone(first); self.assertIsNotNone(second)
                self.assertEqual(2, len(calls))
                self.assertEqual(0, cache.stats()['entries'])
                self.assertEqual(0, cache.stats()['bytes'])
                self.assertEqual(2, cache.stats()['bypassed'])

    def test_integer_keys_tuple_types_and_shared_references_remain_lossless(self):
        shared = ['value']
        output = {1: ('tuple',), 'left': shared, 'right': shared}
        self.cache.read('omv', b'one', lambda _: output)
        result = self.cache.read('omv', b'one', lambda _: self.fail('Cache miss'))
        self.assertEqual(('tuple',), result[1])
        self.assertIs(result['left'], result['right'])
        self.assertIsNot(shared, result['left'])
        result['left'].append('student')
        self.assertEqual(['value'], output['left'])

    def test_reader_exceptions_propagate_without_caching_and_statistics_remain_bounded(self):
        def broken(_):
            raise ValueError('Malformed input')
        for _ in range(2):
            with self.assertRaisesRegex(ValueError, 'Malformed input'):
                self.cache.read('omv', b'bad', broken)
        self.assertEqual(0, self.cache.stats()['entries'])
        self.assertEqual(2, self.cache.stats()['misses'])
        self.assertEqual(2, self.cache.stats()['bypassed'])

    def test_cache_instances_are_isolated_and_stats_are_copies(self):
        other = parse_cache.ParsingCache()
        self.cache.read('docx', b'one', lambda _: {'text': 'one'})
        self.assertEqual(0, other.stats()['entries'])
        stats = self.cache.stats(); stats['hits'] = 900
        self.assertEqual(0, self.cache.stats()['hits'])
        self.assertEqual(32, stats['max_entries'])
        self.assertEqual(32 * 1024 * 1024, stats['max_bytes'])

    def test_disabled_and_invalid_budgets_are_explicit(self):
        for options in ({'max_entries': 0}, {'max_bytes': 0}):
            cache = parse_cache.ParsingCache(**options)
            cache.read('docx', b'one', lambda _: {'text': 'one'})
            self.assertEqual(0, cache.stats()['entries'])
            self.assertEqual(1, cache.stats()['bypassed'])
        for options in ({'max_entries': -1}, {'max_bytes': -1}, {'max_entries': 1.5}, {'max_bytes': True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                parse_cache.ParsingCache(**options)

    def test_kind_and_input_framing_prevents_separator_boundary_collisions(self):
        first = self.cache.read('docx\0suffix', b'input', lambda _: {'value': 'first'})
        second = self.cache.read('docx', b'suffix\0input', lambda _: {'value': 'second'})
        self.assertEqual('first', first['value']); self.assertEqual('second', second['value'])
        self.assertEqual(2, self.cache.stats()['entries'])

    def test_real_reader_errors_are_preserved_and_retried(self):
        for _ in range(2):
            result = self.cache.read('docx', b'broken-zip', read_docx)
            self.assertTrue(result['errors'])
        self.assertEqual(2, self.cache.stats()['bypassed'])
        self.assertEqual(0, self.cache.stats()['entries'])

    def test_simultaneous_misses_publish_one_entry_without_duplicate_byte_accounting(self):
        barrier = threading.Barrier(4)
        output = {'value': 'same'}
        def reader(_):
            barrier.wait(timeout=3)
            return output
        def task(_):
            return self.cache.read('docx', b'one', reader)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(task, range(4)))
        self.assertEqual([output] * 4, results)
        stats = self.cache.stats()
        self.assertEqual(4, stats['misses']); self.assertEqual(1, stats['entries'])
        self.assertEqual(0, stats['evictions'])
        self.assertEqual(len(b'{"value":"same"}'), stats['encoded_bytes'])
        self.assertGreater(stats['bytes'], stats['encoded_bytes'])

    def test_threaded_cache_hits_return_isolated_values_and_consistent_counters(self):
        self.cache.read('docx', b'one', lambda _: {'values': []})
        def task(index):
            parsed = self.cache.read('docx', b'one', lambda _: self.fail('Cache miss'))
            parsed['values'].append(index)
            return parsed['values']
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(task, range(20)))
        self.assertEqual([[index] for index in range(20)], results)
        self.assertEqual(20, self.cache.stats()['hits'])
        self.assertEqual(1, self.cache.stats()['entries'])


if __name__ == '__main__':
    unittest.main()
