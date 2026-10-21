"""Dev-only comparison contract: independent token parity before semantic scoring."""
import copy
import json
from pathlib import Path
import tempfile
import io
import tarfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    from tools import compare_local_models as comparison
except ImportError:
    comparison = None


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(comparison, 'The bounded comparison runner is missing')
        self.case = {'id': 'x', 'part': 6, 'kind': 'correct', 'text': 'La correlazione non prova causalità.',
                     'criterion': 'Distingui associazione e causa.', 'expected_assessment': 'coerente',
                     'rationale': 'PRIVATE EXPECTED REASON'}

    def payload(self, model='qwen3_5_9b'):
        return comparison.prepare_case(self.case, model)

    def response(self, spans, **changes):
        row = {'part': 6, 'assessment': 'coerente', 'evidence_id': spans[0]['evidence_id'], 'reason': 'Distingue i due concetti.'}
        result = {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps({'findings': [row]})}}],
                  'usage': {'prompt_tokens': 3, 'completion_tokens': 24}}
        result.update(changes)
        return result

    def test_prompt_uses_shared_task_without_expected_answer_or_rationale(self):
        spans, payload = self.payload()
        self.assertEqual(payload['messages'], comparison.mr.messages_for(spans, [6], self.case['criterion']))
        sent = json.dumps(payload)
        self.assertNotIn('PRIVATE EXPECTED REASON', sent)
        self.assertNotIn('expected_assessment', sent)
        self.assertEqual(payload['temperature'], 0)
        self.assertEqual(payload['seed'], 0)
        self.assertEqual(payload['max_tokens'], 1024)
        self.assertEqual(payload['chat_template_kwargs'], {'enable_thinking': True, 'reasoning_effort': 'medium'})
        self.assertEqual(payload['response_format']['json_schema']['schema'], comparison.mr._schema(spans, [6]))
        self.assertNotIn('chat_template_kwargs', self.payload('ministral_3_14b_reasoning_2512')[1])

    def test_baseline_evidence_ids_match_the_existing_validation_document(self):
        expected = comparison.mr.segment_document({'paragraphs': [{'index': 0, 'location': 'Paragrafo 1', 'text': self.case['text']}]})['spans']
        self.assertEqual(self.payload()[0], expected)

    def test_full_chat_payload_and_special_token_flags_gate_inference(self):
        spans, payload = self.payload()
        calls = []
        def post(path, body):
            calls.append((path, copy.deepcopy(body)))
            if path == '/apply-template':
                return {'prompt': 'rendered'}
            if path == '/tokenize':
                return {'tokens': [1, 2, 3]}
            return self.response(spans)
        row = comparison.execute_case(self.case, 'qwen3_5_9b', post, lambda body: [1, 2, 3])
        self.assertEqual(calls[0], ('/apply-template', payload))
        self.assertTrue(calls[1][1]['add_special'])
        self.assertTrue(calls[1][1]['parse_special'])
        self.assertEqual(calls[2], ('/v1/chat/completions', payload))
        self.assertEqual(row['status'], 'complete')
        self.assertEqual(row['observed'], 'coerente')
        self.assertFalse(row['native_reasoning_observed'])
        self.assertEqual(len(row['exchanges']), 3)

    def test_token_mismatch_fails_closed_without_semantic_inference(self):
        def post(path, body):
            self.assertNotEqual(path, '/v1/chat/completions')
            return {'prompt': 'rendered'} if path == '/apply-template' else {'tokens': [1, 2, 4]}
        row = comparison.execute_case(self.case, 'qwen3_5_9b', post, lambda body: [1, 2, 3])
        self.assertEqual(row['status'], 'infrastructure_error')
        self.assertEqual(row['error_code'], 'tokenizer_parity_mismatch')
        self.assertIsNone(row['observed'])
        self.assertFalse(row['inference_attempted'])

    def test_context_overflow_is_infrastructure_failure(self):
        def post(path, body):
            return {'prompt': 'rendered'} if path == '/apply-template' else {'tokens': [1] * 3073}
        row = comparison.execute_case(self.case, 'qwen3_5_9b', post, lambda body: [1] * 3073)
        self.assertEqual(row['error_code'], 'context_budget_exceeded')
        self.assertFalse(row['inference_attempted'])

    def test_truncation_malformed_json_foreign_evidence_and_prompt_count_are_rejected(self):
        spans, _ = self.payload()
        good = self.response(spans)
        variants = []
        length = copy.deepcopy(good); length['choices'][0]['finish_reason'] = 'length'; variants.append(length)
        malformed = copy.deepcopy(good); malformed['choices'][0]['message']['content'] = '{'; variants.append(malformed)
        foreign = copy.deepcopy(good)
        value = json.loads(foreign['choices'][0]['message']['content']); value['findings'][0]['evidence_id'] = 'FOREIGN'
        foreign['choices'][0]['message']['content'] = json.dumps(value); variants.append(foreign)
        count = copy.deepcopy(good); count['usage']['prompt_tokens'] = 4; variants.append(count)
        for response in variants:
            with self.subTest(response=response):
                def post(path, body):
                    if path == '/apply-template': return {'prompt': 'rendered'}
                    if path == '/tokenize': return {'tokens': [1, 2, 3]}
                    return response
                row = comparison.execute_case(self.case, 'qwen3_5_9b', post, lambda body: [1, 2, 3])
                self.assertNotEqual(row['status'], 'complete')
                self.assertIsNone(row['observed'])
                self.assertTrue(row['inference_attempted'])

    def test_native_reasoning_is_observed_without_being_required_for_validity(self):
        spans, _ = self.payload()
        result = self.response(spans); result['choices'][0]['message']['reasoning_content'] = 'Ragionamento.'
        def post(path, body):
            if path == '/apply-template': return {'prompt': 'rendered'}
            if path == '/tokenize': return {'tokens': [1, 2, 3]}
            return result
        row = comparison.execute_case(self.case, 'qwen3_5_9b', post, lambda body: [1, 2, 3])
        self.assertEqual(row['status'], 'complete')
        self.assertTrue(row['native_reasoning_observed'])

    def test_seven_parts_cover_exactly_seventy_baseline_and_fourteen_fresh_cases(self):
        fresh = [dict(self.case, id=f'F{part}.{n}', part=part) for part in range(1, 8) for n in range(2)]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'fresh.json'; path.write_text(json.dumps(fresh))
            cases = comparison.load_cases(path)
        selected = [case for part in range(1, 8) for case in comparison.select_cases(cases, part)]
        self.assertEqual(len(selected), 84)
        self.assertEqual(len({c['id'] for c in selected}), 84)
        self.assertEqual(sum(c['dataset'] == 'baseline' for c in selected), 70)
        for part in range(1, 8):
            subset = comparison.select_cases(cases, part)
            self.assertEqual(len(subset), 12)
            self.assertEqual(sum(c['dataset'] == 'fresh' for c in subset), 2)

    def test_aggregate_uses_planned_denominators_and_never_counts_missing_as_wrong_prediction(self):
        plan = [dict(self.case, id='a', dataset='baseline'), dict(self.case, id='b', dataset='fresh')]
        rows = [dict(plan[0], status='complete', observed='coerente', inference_attempted=True)]
        summary = comparison.summarize(plan, rows)
        self.assertEqual(summary['planned'], 2)
        self.assertEqual(summary['missing'], 1)
        self.assertEqual(summary['matching_labels'], 1)
        self.assertEqual(summary['agreement_planned'], .5)
        self.assertEqual(summary['agreement_valid'], 1)
        self.assertEqual(summary['by_dataset']['fresh']['planned'], 1)
        self.assertEqual(summary['by_dataset']['fresh']['missing'], 1)
        self.assertIsNone(summary['by_dataset']['fresh']['agreement_valid'])
        with self.assertRaises(ValueError): comparison.summarize(plan, rows + rows)

    def test_checkpoint_rejects_changes_to_frozen_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); source = root / 'input.json'; source.write_text('{"a":1}')
            out = root / 'output'
            first = comparison.freeze_inputs({'manifest': source}, out)
            self.assertEqual(first, comparison.freeze_inputs({'manifest': source}, out))
            source.write_text('{"a":2}')
            with self.assertRaises(ValueError): comparison.freeze_inputs({'manifest': source}, out)

    def test_request_failure_is_sanitized_and_preserves_partial_exchanges(self):
        def post(path, body): raise RuntimeError('SECRET-CAPABILITY https://signed.invalid/private')
        row = comparison.execute_case(self.case, 'qwen3_5_9b', post, lambda body: [1, 2, 3])
        self.assertEqual(row['status'], 'infrastructure_error')
        self.assertNotIn('SECRET-CAPABILITY', json.dumps(row))
        self.assertNotIn('signed.invalid', json.dumps(row))


class InfrastructureTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(comparison)
        self.assertTrue(hasattr(comparison, 'OwnedEngine'), 'Pinned asset and owned lifecycle support is missing')

    def test_asset_paths_and_revisions_are_pinned_before_any_download(self):
        entry = {'filename': 'tokenizer.json', 'size_bytes': 4, 'sha256': 'a' * 64,
                 'url': 'https://huggingface.co/org/model/resolve/' + 'b' * 40 + '/tokenizer.json'}
        asset = comparison.pinned_asset(entry)
        self.assertEqual(asset.name, 'tokenizer.json')
        for change in ({'filename': '../secret'}, {'url': 'https://huggingface.co/org/model/resolve/main/tokenizer.json'}, {'sha256': 'unknown'}):
            with self.assertRaises(ValueError): comparison.pinned_asset(dict(entry, **change))

    def test_real_disk_guard_reserves_room_and_memory_is_advisory(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(comparison.shutil, 'disk_usage', return_value=(100, 90, 10)):
                with self.assertRaises(ValueError): comparison.resource_guard(Path(temporary), 11)
            with patch.object(comparison.shutil, 'disk_usage', return_value=(10**12, 0, 10**12)):
                result = comparison.resource_guard(Path(temporary), 14_252_845_984)
                self.assertTrue(result['memory_advisory_only'])
                self.assertEqual(result['required_download_bytes'], 14_252_845_984)

    def test_aggregate_missing_parts_is_explicit_and_hash_tampering_is_rejected(self):
        fresh = [dict(id=f'F{p}.{n}', part=p,
                      text='Public synthetic text', expected_assessment='coerente') for p in range(1, 8) for n in range(2)]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); source = root / 'fresh.json'; source.write_text(json.dumps(fresh))
            cases = comparison.load_cases(source)
            out = root / 'output'; out.mkdir()
            result = comparison.aggregate_parts(cases, out, 'qwen3_5_9b', {'manifest': 'abc'})
            self.assertEqual(result['planned'], 84)
            self.assertEqual(result['missing'], 84)
            self.assertEqual(result['missing_parts'], list(range(1, 8)))
            self.assertEqual(result['by_dataset']['baseline']['planned'], 70)
            self.assertEqual(result['by_dataset']['fresh']['planned'], 14)
            comparison.write_json(out / 'part_1.json', {'model': 'qwen3_5_9b', 'input_hashes': {'manifest': 'CHANGED'}, 'part': 1, 'cases': []})
            with self.assertRaises(ValueError): comparison.aggregate_parts(cases, out, 'qwen3_5_9b', {'manifest': 'abc'})

    def test_verified_runtime_archive_can_extract_internal_library_symlinks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); bundle = root / 'runtime.tar.gz'
            with tarfile.open(bundle, 'w:gz') as archive:
                member = tarfile.TarInfo('llama/libllama.so.0'); member.size = 4
                archive.addfile(member, io.BytesIO(b'code'))
                link = tarfile.TarInfo('llama/libllama.so'); link.type = tarfile.SYMTYPE; link.linkname = 'libllama.so.0'
                archive.addfile(link)
            comparison.unpack_runtime(bundle, root / 'extracted')
            self.assertEqual((root / 'extracted/llama/libllama.so').read_bytes(), b'code')
            with tarfile.open(bundle, 'w:gz') as archive:
                link = tarfile.TarInfo('escape'); link.type = tarfile.SYMTYPE; link.linkname = '/outside'
                archive.addfile(link)
            with self.assertRaises((ValueError, tarfile.FilterError)):
                comparison.unpack_runtime(bundle, root / 'rejected')

    def test_qwen_oracle_requests_a_token_list_despite_new_transformers_defaults(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return [1, 2, 3] if kwargs.get('return_dict') is False else {'input_ids': [1, 2, 3]}
        fake = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer()))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root / 'chat_template.jinja').write_text('official')
            with patch.dict('sys.modules', {'transformers': fake}):
                oracle = comparison.official_oracle('qwen3_5_9b', root)
                self.assertEqual(oracle({'messages': []}), [1, 2, 3])

    def test_readiness_requires_health_before_models_alias(self):
        engine = comparison.OwnedEngine(Path('/fake'), Path('/fake'), Path('/fake'), 'test', Path('/unused'), comparison.time.monotonic()+5)
        engine.server = SimpleNamespace(process=SimpleNamespace(poll=lambda: None))
        calls = []
        def request(path, payload=None, timeout=None):
            calls.append(path)
            if path == '/health': return {'status': 'ok' if calls.count('/health') > 1 else 'loading'}
            return {'data': [{'id': 'test'}]}
        engine.request = request
        with patch.object(comparison.time, 'sleep'): engine.wait_ready()
        self.assertEqual(calls, ['/health', '/health', '/v1/models'])

    def test_owned_server_stops_and_log_is_sanitized_even_when_body_raises(self):
        class Process:
            def __init__(self):
                self.stopped = False
                self.stdout = io.BytesIO()
            def poll(self): return 0 if self.stopped else None
            def terminate(self): self.stopped = True
            def wait(self, timeout): return 0
        process = Process()
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            with patch.object(comparison.subprocess, 'Popen', return_value=process), patch.object(comparison.OwnedEngine, 'wait_ready'):
                with self.assertRaises(RuntimeError):
                    with comparison.OwnedEngine(Path('/fake/server'), Path('/fake/model'), Path('/fake/template'), 'qwen3_5_9b', out, comparison.time.monotonic()+20) as engine:
                        secret = engine.api_key
                        raise RuntimeError(secret)
            self.assertTrue(process.stopped)
            for path in out.iterdir(): self.assertNotIn(secret, path.read_text())

    def test_native_log_redacts_capability_across_read_boundaries(self):
        self.assertTrue(hasattr(comparison.OwnedEngine, 'read_native_log'), 'Sanitized native logs are missing')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine = comparison.OwnedEngine(Path('/fake'), Path('/fake'), Path('/fake'), 'test', root, comparison.time.monotonic()+5)
            engine.server = SimpleNamespace(process=SimpleNamespace(stdout=io.BytesIO(b'x'*4090 + engine.api_key.encode() + b' actual loader error\n')))
            engine.read_native_log()
            saved = (root / 'native.log').read_text()
            self.assertNotIn(engine.api_key, saved)
            self.assertIn('[redacted]', saved)
            self.assertIn('actual loader error', saved)


if __name__ == '__main__': unittest.main()
