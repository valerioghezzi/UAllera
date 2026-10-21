"""Development-only, pinned CPU model comparison; never imported by the app.

Prepare once, run --part 1 through --part 7, then --aggregate. Each part has
an inclusive deadline including startup: original uses 840 seconds and 512
reasoning tokens; the separate budget128 profile uses 780 seconds and 128
reasoning tokens, with fresh cases first. The heldout128 profile uses the same
generation settings for only 14 fresh cases, with 180 seconds per part. Failed protocol checks
are infrastructure failures, never semantic predictions. All cases are public
synthetic data. Dependencies for independent tokenizer oracles are development
only: transformers and mistral-common (no model tensors are loaded).
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import re
import secrets
import shutil
import socket
import subprocess
import tarfile
import urllib.request
import urllib.error
import threading
import codecs
from importlib.metadata import version, PackageNotFoundError
from dataclasses import dataclass

import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.prepare_minerva import Asset, _download, _valid_asset, LocalServer
from jamovi_audit import minerva_review as mr
from jamovi_audit.validation_cases import VALIDATION_CASES

MODEL_KEYS = ('qwen3_8_27b', 'ministral_3_14b_reasoning_2512', 'qwen3_5_9b')
LABELS = ('coerente', 'errata', 'incompleta', 'non_valutabile')
CONTEXT_TOKENS = 4096
OUTPUT_TOKENS = 1024
REASONING_TOKENS = 512
REQUEST_SECONDS = 180
PART_SECONDS = 840
PROFILE = 'native-chat-official-token-parity-1'


@dataclass(frozen=True)
class ComparisonProfile:
    key: str
    version: str
    reasoning_tokens: int
    request_seconds: int
    part_seconds: int
    fresh_first: bool = False
    case_scope: str = 'all'


PROFILES = {
    'original': ComparisonProfile('original', PROFILE, REASONING_TOKENS, REQUEST_SECONDS, PART_SECONDS),
    'budget128': ComparisonProfile('budget128', PROFILE + '-budget128', 128, 300, 780, True),
    'heldout128': ComparisonProfile('heldout128', PROFILE + '-heldout128', 128, 300, 180, True, 'fresh'),
}
ORIGINAL = PROFILES['original']


def protocol_settings(profile=ORIGINAL):
    return {'profile_key': profile.key, 'profile_version': profile.version,
            'context_tokens': CONTEXT_TOKENS, 'output_tokens': OUTPUT_TOKENS, 'case_scope': profile.case_scope,
            'reasoning_budget': profile.reasoning_tokens, 'temperature': 0, 'seed': 0,
            'per_request_seconds': profile.request_seconds, 'per_part_seconds': profile.part_seconds,
            'case_order': 'fresh_then_baseline_within_part' if profile.fresh_first else 'baseline_then_fresh_within_part'}


class ProtocolError(ValueError):
    """A stable public error code; never retains native exception text."""


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_bytes(json.dumps(value, ensure_ascii=False, indent=2).encode() + b'\n')
    os.replace(temporary, path)


def load_cases(fresh_path):
    fresh = json.loads(Path(fresh_path).read_text(encoding='utf-8'))
    cases = [dict(c, dataset='baseline') for c in VALIDATION_CASES]
    if len(cases) != 70 or not isinstance(fresh, list) or len(fresh) != 14:
        raise ValueError('Expected exactly 70 baseline and 14 fresh cases')
    cases += [dict(c, dataset='fresh') for c in fresh]
    ids = set()
    for case in cases:
        if (not isinstance(case.get('id'), str) or case['id'] in ids or
                type(case.get('part')) is not int or case['part'] not in range(1, 8) or
                case.get('expected_assessment') not in LABELS or
                not isinstance(case.get('text'), str) or not case['text'].strip()):
            raise ValueError('Invalid or duplicate case')
        ids.add(case['id'])
    for part in range(1, 8):
        subset = select_cases(cases, part)
        if Counter(c['dataset'] for c in subset) != {'baseline': 10, 'fresh': 2}:
            raise ValueError('Every part must contain 10 baseline and 2 fresh cases')
    return cases


def profile_cases(cases, profile=ORIGINAL):
    return [case for case in cases if profile.case_scope == 'all' or case.get('dataset') == 'fresh']


def select_cases(cases, part, profile=ORIGINAL):
    if type(part) is not int or part not in range(1, 8):
        raise ValueError('Part must be 1 through 7')
    selected = [c for c in profile_cases(cases, profile) if c['part'] == part]
    return sorted(selected, key=lambda c: c.get('dataset') != 'fresh') if profile.fresh_first else selected


def prepare_case(case, model):
    if model not in MODEL_KEYS:
        raise ValueError('Unknown comparison model')
    spans = mr.segment_document({'paragraphs': [{'index': 0, 'location': 'Paragrafo 1', 'text': case['text']}]})['spans']
    parts = [case['part']]
    payload = {
        'model': model,
        'messages': mr.messages_for(spans, parts, case.get('criterion', 'Valutare il significato espresso.'), case.get('statistical_context')),
        'temperature': 0, 'seed': 0, 'max_tokens': OUTPUT_TOKENS, 'stream': False,
        'response_format': {'type': 'json_schema', 'json_schema': {
            'name': 'evidence_findings', 'strict': True, 'schema': mr._schema(spans, parts)}},
    }
    if model.startswith('qwen'):
        payload['chat_template_kwargs'] = {'enable_thinking': True, 'reasoning_effort': 'medium'}
    return spans, payload


def execute_case(case, model, post, oracle, profile=ORIGINAL):
    """Save actual request/response bodies; the HTTP capability lives elsewhere."""
    row = dict(case, status='infrastructure_error', observed=None, inference_attempted=False,
               native_reasoning_observed=False, exchanges=[], profile=profile.version, protocol=protocol_settings(profile))
    started = time.monotonic()
    def exchange(path, payload):
        record = {'path': path, 'request': payload}
        row['exchanges'].append(record)
        result = post(path, payload)
        record['response'] = result
        return result
    try:
        spans, payload = prepare_case(case, model)
        expected = oracle(payload)
        if not isinstance(expected, list) or not expected or any(type(t) is not int or t < 0 for t in expected):
            raise ProtocolError('official_tokenizer_invalid')
        rendered = exchange('/apply-template', payload)
        if not isinstance(rendered.get('prompt'), str):
            raise ProtocolError('template_prompt_missing')
        actual = exchange('/tokenize', {'content': rendered['prompt'], 'add_special': True, 'parse_special': True})
        row['official_prompt_tokens'] = expected
        row['engine_prompt_tokens'] = actual.get('tokens')
        row['prompt_token_count'] = len(expected)
        if actual.get('tokens') != expected:
            raise ProtocolError('tokenizer_parity_mismatch')
        row['tokenizer_parity'] = True
        if len(expected) + OUTPUT_TOKENS > CONTEXT_TOKENS:
            raise ProtocolError('context_budget_exceeded')
        row['inference_attempted'] = True
        result = exchange('/v1/chat/completions', payload)
        usage = result.get('usage', {})
        row['usage'] = usage
        if usage.get('prompt_tokens') is not None and usage['prompt_tokens'] != len(expected):
            raise ProtocolError('inference_prompt_count_mismatch')
        row['status'] = 'invalid_response'
        choices = result.get('choices')
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError('Unexpected choice count')
        choice = choices[0]
        row['finish_reason'] = choice.get('finish_reason')
        message = choice.get('message', {})
        reasoning = message.get('reasoning_content') or message.get('reasoning')
        row['native_reasoning_observed'] = bool(isinstance(reasoning, str) and reasoning.strip())
        if choice.get('finish_reason') != 'stop' or result.get('truncated') is True:
            raise ValueError('Incomplete generation')
        rows = mr.validate_response(json.loads(message.get('content', '')), spans, [case['part']])
        row.update(status='complete', observed=rows[0]['assessment'], findings=rows,
                   matches_expected=rows[0]['assessment'] == case['expected_assessment'])
    except ProtocolError as error:
        row.update(status='infrastructure_error', error_code=str(error), observed=None)
    except (ValueError, TypeError, KeyError, IndexError, AttributeError):
        row['error_code'] = 'invalid_response' if row['status'] == 'invalid_response' else 'protocol_response_invalid'
    except Exception:
        row.update(status='infrastructure_error', error_code='request_or_oracle_failed')
    row['seconds'] = round(time.monotonic() - started, 3)
    return row


def summarize(plan, rows):
    by_id = {c['id']: c for c in plan}
    seen = set()
    for row in rows:
        if row.get('id') not in by_id or row['id'] in seen:
            raise ValueError('Duplicate or foreign case result')
        seen.add(row['id'])
        expected = by_id[row['id']]
        if any(row.get(k) != expected.get(k) for k in ('part', 'dataset', 'expected_assessment', 'text', 'criterion')):
            raise ValueError('Result does not match frozen case')
    def metrics(cases, selected):
        valid = [r for r in selected if r.get('status') == 'complete' and r.get('observed') in LABELS]
        matching = sum(r['observed'] == by_id[r['id']]['expected_assessment'] for r in valid)
        errors = [c for c in cases if c['expected_assessment'] == 'errata']
        correct = [c for c in cases if c['expected_assessment'] == 'coerente']
        ratio = lambda n, d: n / d if d else None
        return {
            'planned': len(cases), 'attempted': len(selected), 'missing': len(cases) - len(selected),
            'inference_attempted': sum(bool(r.get('inference_attempted')) for r in selected),
            'valid': len(valid), 'invalid': sum(r.get('status') == 'invalid_response' for r in selected),
            'infrastructure_errors': sum(r.get('status') == 'infrastructure_error' for r in selected),
            'matching_labels': matching, 'agreement_planned': ratio(matching, len(cases)),
            'agreement_valid': ratio(matching, len(valid)), 'valid_coverage': ratio(len(valid), len(cases)),
            'error_recall_planned': ratio(sum(r['observed'] == 'errata' and by_id[r['id']]['expected_assessment'] == 'errata' for r in valid), len(errors)),
            'false_alarm_rate_valid_correct': ratio(sum(r['observed'] == 'errata' and by_id[r['id']]['expected_assessment'] == 'coerente' for r in valid), sum(by_id[r['id']]['expected_assessment'] == 'coerente' for r in valid)),
            'false_reassurance_count': sum(r['observed'] == 'coerente' and by_id[r['id']]['expected_assessment'] != 'coerente' for r in valid),
            'planned_error_cases': len(errors), 'planned_correct_cases': len(correct),
            'native_reasoning_observed': sum(bool(r.get('native_reasoning_observed')) for r in selected),
            'confusion': {label: dict(Counter(r['observed'] for r in valid if by_id[r['id']]['expected_assessment'] == label)) for label in LABELS},
        }
    result = metrics(plan, rows)
    result['status'] = 'complete' if result['valid'] == len(plan) else 'partial'
    result['by_dataset'] = {dataset: metrics([c for c in plan if c['dataset'] == dataset], [r for r in rows if by_id[r['id']]['dataset'] == dataset]) for dataset in ('baseline', 'fresh')}
    result['by_part'] = {str(part): metrics([c for c in plan if c['part'] == part], [r for r in rows if by_id[r['id']]['part'] == part]) for part in range(1, 8)}
    return result


def freeze_inputs(paths, output, protocol=None):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    hashes = {key: hashlib.sha256(Path(path).read_bytes()).hexdigest() for key, path in paths.items()}
    if protocol is not None:
        hashes['protocol'] = hashlib.sha256(canonical(protocol)).hexdigest()
    checkpoint = output / 'input_hashes.json'
    if checkpoint.exists() and json.loads(checkpoint.read_text()) != hashes:
        raise ValueError('Frozen comparison input changed')
    if not checkpoint.exists():
        write_json(checkpoint, hashes)
        if protocol is not None:
            write_json(output / 'frozen_protocol.json', protocol)
        for key, path in paths.items():
            (output / ('frozen_' + key + Path(path).suffix)).write_bytes(Path(path).read_bytes())
    return hashes


def pinned_asset(entry):
    name, url = entry.get('filename'), entry.get('url')
    if (not isinstance(name, str) or Path(name).name != name or name in ('', '.', '..') or '\\' in name or
            not isinstance(url, str) or not re.fullmatch(r'[0-9a-f]{64}', entry.get('sha256', '')) or
            type(entry.get('size_bytes')) is not int or entry['size_bytes'] <= 0):
        raise ValueError('Invalid pinned asset metadata')
    model_url = re.fullmatch(r'https://huggingface\.co/[^/]+/[^/]+/resolve/[0-9a-f]{40}/' + re.escape(name), url)
    runtime_url = url == 'https://github.com/ggml-org/llama.cpp/releases/download/b11756/llama-b11756-bin-ubuntu-x64.tar.gz'
    if not model_url and not runtime_url:
        raise ValueError('Asset URL must use an immutable approved source')
    return Asset(name, url, entry['size_bytes'], entry['sha256'])


def resource_guard(root, required_download_bytes):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(root)[2]
    reserve = 512 * 1024 * 1024
    if free < required_download_bytes + reserve:
        raise ValueError('Insufficient free disk for pinned downloads and reserve')
    memory = None
    try:
        meminfo = Path('/proc/meminfo').read_text()
        memory = int(re.search(r'^MemAvailable:\s+(\d+)', meminfo, re.M).group(1)) * 1024
        limit = Path('/sys/fs/cgroup/memory.max').read_text().strip()
        if limit != 'max':
            current = int(Path('/sys/fs/cgroup/memory.current').read_text())
            memory = min(memory, max(0, int(limit) - current))
    except (OSError, ValueError, AttributeError):
        pass
    return {'free_disk_bytes': free, 'disk_reserve_bytes': reserve, 'required_download_bytes': required_download_bytes,
            'available_memory_bytes': memory, 'memory_advisory_only': True,
            'memory_caveat': 'Weights are a lower bound; actual startup and bounded inference determine fit. No swap performance claim.'}


def asset_plan(manifest, model, root):
    matches = [m for m in manifest.get('models', []) if m.get('id') == model]
    if len(matches) != 1:
        raise ValueError('Model missing or duplicated in manifest')
    spec = matches[0]
    root = Path(root); model_root = root / model
    runtime = dict(manifest['runtime'])
    # Retain the existing, independently pinned CPU engine for every candidate.
    if (runtime.get('tag') != 'b11756' or runtime.get('sha256') !=
            '9def57d9e9b0149979a9ceba40896a4e2bf88c94734069a5567d2f487ceab551'):
        raise ValueError('Unexpected comparison runtime')
    runtime_asset = pinned_asset(runtime)
    weight = pinned_asset(spec['files'][0])
    small = [(model_root / 'original' / f['filename'], pinned_asset(f)) for f in spec['original_files']]
    required = {'chat_template.jinja', 'config.json', 'tokenizer_config.json', 'tokenizer.json'} if model.startswith('qwen') else {'chat_template.jinja', 'tekken.json'}
    if not required <= {a.name for _, a in small}:
        raise ValueError('Manifest lacks official tokenizer files')
    downloads = [(root / runtime_asset.name, runtime_asset), (model_root / weight.name, weight)] + small
    return spec, downloads, root / 'runtime' / 'llama-b11756' / 'llama-server', model_root / weight.name, model_root / 'original'


def unpack_runtime(archive_path, destination):
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path) as archive:
        members = archive.getmembers()
        if len(members) > 4096 or sum(m.size for m in members) > 256 * 1024 * 1024:
            raise ValueError('Runtime archive exceeds extraction bounds')
        if any(not (m.isfile() or m.isdir() or m.issym() or m.islnk()) or Path(m.name).is_absolute() or '..' in Path(m.name).parts for m in members):
            raise ValueError('Runtime archive contains an unsafe member')
        for member in members:
            tarfile.data_filter(member, str(destination))
        archive.extractall(destination, filter='data')


def prepare_assets(manifest, model, root, output):
    spec, downloads, executable, weight, original = asset_plan(manifest, model, root)
    missing = sum(a.size for p, a in downloads if not _valid_asset(p, a))
    resources = resource_guard(root, missing)
    write_json(Path(output) / 'resources_prepare.json', resources)
    for path, asset in downloads:
        print('Preparing pinned asset:', asset.name, flush=True)
        _download(path, asset, None)
    unpack_runtime(downloads[0][0], Path(root) / 'runtime')
    write_json(Path(output) / 'prepared_assets.json', {
        'model': model, 'resources': resources,
        'assets': [{'filename': a.name, 'size_bytes': a.size, 'sha256': a.sha256} for _, a in downloads],
        'runtime_source_commit': manifest['runtime']['source_commit'],
        'scope': 'Pinned public assets only. No inference has been claimed by preparation.'})
    return spec, executable, weight, original


def official_oracle(model, original):
    """Independent official tokenizer implementations; no Torch/model loading."""
    original = Path(original)
    if model.startswith('qwen'):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(original), local_files_only=True, trust_remote_code=False, use_fast=True)
        # Force the separately pinned official file, even if tokenizer_config
        # changes its internal precedence between library versions.
        tokenizer.chat_template = (original / 'chat_template.jinja').read_text(encoding='utf-8')
        def encode(payload):
            return tokenizer.apply_chat_template(payload['messages'], tokenize=True, add_generation_prompt=True,
                                                 enable_thinking=True, reasoning_effort='medium', return_dict=False)
        return encode
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
    from mistral_common.protocol.instruct.request import ChatCompletionRequest
    tokenizer = MistralTokenizer.from_file(str(original / 'tekken.json'))
    def encode(payload):
        request = ChatCompletionRequest(messages=payload['messages'], truncate_for_context_length=False)
        return list(tokenizer.encode_chat_completion(request).tokens)
    return encode


class OwnedEngine:
    """Loopback-only HTTP with an in-memory capability and owned cleanup."""
    def __init__(self, executable, model_path, template, model, output, deadline, profile=ORIGINAL):
        self.executable, self.model_path, self.template = executable, model_path, template
        self.model, self.output, self.deadline = model, Path(output), deadline
        self.profile = profile
        self.api_key = secrets.token_urlsafe(32)
        self.server = None
        self.events = []
        self.peak_rss_bytes = 0
        self.log_thread = None

    def __enter__(self):
        with socket.socket() as reservation:
            reservation.bind(('127.0.0.1', 0)); port = reservation.getsockname()[1]
        self.base_url = f'http://127.0.0.1:{port}'
        threads = min(4, os.cpu_count() or 1)
        command = [str(self.executable), '--model', str(self.model_path), '--host', '127.0.0.1', '--port', str(port),
                   '--ctx-size', str(CONTEXT_TOKENS), '--parallel', '1', '--threads', str(threads),
                   '--threads-batch', str(threads), '--alias', self.model, '--no-warmup', '--jinja',
                   '--chat-template-file', str(self.template), '--reasoning-format', 'deepseek',
                   '--reasoning-budget', str(self.profile.reasoning_tokens), '--offline', '--no-webui', '--api-key', self.api_key]
        try:
            # Native argv and exception reprs can contain the capability. Drain
            # the pipe through a bounded redactor before any bytes reach disk.
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            self.server = LocalServer(process, self.base_url, self.api_key, model=self.model)
            self.log_thread = threading.Thread(target=self.read_native_log, daemon=True)
            self.log_thread.start()
            started = time.monotonic()
            self.wait_ready()
            self.events.append({'event': 'ready', 'seconds': round(time.monotonic() - started, 3), 'threads': threads})
            return self
        except Exception as error:
            self.events.append({'event': 'startup_failed', 'exit_code': self.server.process.poll() if self.server else None})
            self.stop()
            if isinstance(error, ProtocolError) and str(error) == 'part_deadline_reached':
                raise ProtocolError('part_deadline_reached') from None
            raise ProtocolError('owned_engine_startup_failed') from None

    def read_native_log(self):
        cap = 2 * 1024 * 1024
        written = 0
        carry = ''
        decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        self.output.mkdir(parents=True, exist_ok=True)
        try:
            with (self.output / 'native.log').open('wb') as target:
                def save(text):
                    nonlocal written
                    data = text.encode('utf-8')
                    if written < cap:
                        selected = data[:cap-written]
                        target.write(selected); target.flush(); written += len(selected)
                stream = self.server.process.stdout
                while block := stream.read(4096):
                    carry += decoder.decode(block)
                    carry = carry.replace(self.api_key, '[redacted]')
                    # Preserve enough raw characters to catch a key split across
                    # pipe reads. Full occurrences were removed before cutting.
                    cut = max(0, len(carry) - len(self.api_key) + 1)
                    save(carry[:cut]); carry = carry[cut:]
                carry += decoder.decode(b'', final=True)
                save(carry.replace(self.api_key, '[redacted]'))
                if written == cap:
                    target.write(b'\n[Native log size limit reached; remaining output drained.]\n')
        except (OSError, AttributeError):
            self.events.append({'event': 'native_log_unavailable'})

    def sample_memory(self):
        try:
            text = Path(f'/proc/{self.server.process.pid}/status').read_text()
            match = re.search(r'^VmHWM:\s+(\d+)', text, re.M)
            if match: self.peak_rss_bytes = max(self.peak_rss_bytes, int(match.group(1)) * 1024)
        except (OSError, AttributeError, ValueError):
            pass

    def request(self, path, payload=None, timeout=None):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise ProtocolError('part_deadline_reached')
        request_limit = min(timeout or self.profile.request_seconds, self.profile.request_seconds)
        deadline_limited = remaining <= request_limit
        timeout = min(request_limit, remaining)
        body = None if payload is None else canonical(payload)
        request = urllib.request.Request(self.base_url + path, data=body,
                                         headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self.api_key})
        # The endpoint is always this owned loopback subprocess; no proxy, external
        # model provider, URL redirect, or remotely supplied destination is used.
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *args, **kwargs): return None
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        try:
            with opener.open(request, timeout=timeout) as response:
                raw = response.read(8 * 1024 * 1024 + 1)
            if len(raw) > 8 * 1024 * 1024:
                raise ProtocolError('native_response_too_large')
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ProtocolError('native_response_not_object')
            # A native error must not reflect the HTTP capability into artifacts.
            if self.api_key in json.dumps(value):
                raise ProtocolError('native_response_contains_capability')
            return value
        except TimeoutError:
            raise ProtocolError('part_deadline_reached' if deadline_limited else 'request_timeout') from None
        except urllib.error.URLError as error:
            code = ('part_deadline_reached' if deadline_limited else 'request_timeout') if isinstance(error.reason, TimeoutError) else 'native_http_error'
            raise ProtocolError(code) from None
        finally:
            self.sample_memory()

    def wait_ready(self):
        deadline = min(self.deadline, time.monotonic() + 180)
        while time.monotonic() < deadline:
            if self.server.process.poll() is not None:
                raise ProtocolError('owned_engine_stopped')
            try:
                health = self.request('/health', timeout=min(2, max(.01, deadline-time.monotonic())))
                models = self.request('/v1/models', timeout=min(2, max(.01, deadline-time.monotonic()))) if health.get('status') == 'ok' else {}
                if any(row.get('id') == self.model for row in models.get('data', [])):
                    return
            except ProtocolError as error:
                if str(error) == 'part_deadline_reached':
                    raise
            except Exception:
                pass
            time.sleep(.25)
        if time.monotonic() >= self.deadline:
            raise ProtocolError('part_deadline_reached')
        raise ProtocolError('owned_engine_readiness_timeout')

    def stop(self):
        self.sample_memory()
        stopped = self.server.stop() if self.server is not None else True
        if self.log_thread is not None:
            self.log_thread.join(timeout=2)
        self.events.append({'event': 'stopped', 'cleanup_complete': stopped, 'peak_rss_bytes': self.peak_rss_bytes,
                            'exit_code': self.server.process.poll() if self.server else None})
        write_json(self.output / 'engine_lifecycle.json', {'events': self.events, 'native_output': 'bounded to 2 MiB and capability-redacted before writing'})

    def __exit__(self, kind, value, traceback):
        if kind is not None: self.events.append({'event': 'body_failed'})
        self.stop()
        return False


def aggregate_parts(cases, output, model, hashes, profile=ORIGINAL):
    cases = profile_cases(cases, profile)
    output = Path(output); rows = []; present = []; failures = []
    for part in range(1, 8):
        path = output / f'part_{part}.json'
        if not path.exists(): continue
        value = json.loads(path.read_text(encoding='utf-8'))
        if (value.get('model') != model or value.get('part') != part or value.get('input_hashes') != hashes or
                value.get('profile') != profile.version):
            raise ValueError('Part model, identity, or frozen input hashes differ')
        if any(c.get('part') != part or c.get('profile') != profile.version for c in value['cases']):
            raise ValueError('Part contains a foreign case')
        rows += value['cases']; present.append(part)
        if value.get('failure_code'): failures.append({'part': part, 'code': value['failure_code']})
    result = summarize(cases, rows)
    result.update(model=model, profile=profile.version, input_hashes=hashes, present_parts=present,
                  missing_parts=[p for p in range(1, 8) if p not in present], failures=failures,
                  missing_case_ids=[c['id'] for c in cases if c['id'] not in {r['id'] for r in rows}],
                  protocol=protocol_settings(profile),
                  limits=['Public synthetic cases with assistant-authored labels; not independent external validation.',
                          'Missing, invalid, and infrastructure outcomes are not model predictions.',
                          'Agreement and error recall over planned cases retain all planned denominators.',
                          'Native reasoning is observed separately and is not a validity requirement.',
                          'No production behavior, classroom suitability, or authorship detection is certified.'])
    write_json(output / 'summary.json', result)
    return result


def run_part(manifest, model, assets, output, cases, part, hashes, profile=ORIGINAL):
    output = Path(output); path = output / f'part_{part}.json'
    if path.exists():
        raise ValueError('Part already exists; use a new output directory to rerun')
    deadline = time.monotonic() + profile.part_seconds
    selected = select_cases(cases, part, profile)
    result = {'model': model, 'part': part, 'input_hashes': hashes, 'profile': profile.version, 'protocol': protocol_settings(profile),
              'planned': len(selected), 'cases': [], 'status': 'starting'}
    def save():
        result['summary'] = summarize(selected, result['cases'])
        write_json(path, result)
    save()
    fatal = False
    try:
        spec, downloads, executable, weight, original = asset_plan(manifest, model, assets)
        for local, asset in downloads:
            if not _valid_asset(local, asset):
                raise ProtocolError('prepared_asset_missing_or_changed')
        unpack_runtime(downloads[0][0], Path(assets) / 'runtime')
        oracle = official_oracle(model, original)
        part_output = output / f'part_{part}_engine'
        part_output.mkdir(parents=True, exist_ok=True)
        with OwnedEngine(executable, weight, original / 'chat_template.jinja', model, part_output, deadline, profile) as engine:
            result['resources_before_inference'] = resource_guard(assets, 0)
            result['runtime_props'] = engine.request('/props')
            for index, case in enumerate(selected, 1):
                if time.monotonic() >= deadline:
                    result['failure_code'] = 'part_deadline_reached'
                    break
                result['active_case'] = case['id']; save()
                print(f"Part {part}: case {index}/{len(selected)} {case['id']}", flush=True)
                row = execute_case(case, model, engine.request, oracle, profile)
                result['cases'].append(row); result.pop('active_case', None)
                result['peak_rss_bytes'] = engine.peak_rss_bytes
                save()
                if row['status'] == 'infrastructure_error':
                    result['failure_code'] = row['error_code']
                    fatal = row['error_code'] != 'part_deadline_reached'
                    break
            result['status'] = 'complete' if len(result['cases']) == len(selected) else 'partial'
    except ProtocolError as error:
        result.update(status='partial' if str(error) == 'part_deadline_reached' else 'infrastructure_error', failure_code=str(error))
        fatal = str(error) != 'part_deadline_reached'
    except Exception:
        result.update(status='infrastructure_error', failure_code='part_setup_or_execution_failed')
        fatal = True
    finally:
        result['remaining_case_ids'] = [c['id'] for c in selected if c['id'] not in {r['id'] for r in result['cases']}]
        save()
    return 1 if fatal else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--fresh', type=Path, required=True)
    parser.add_argument('--model', choices=MODEL_KEYS, required=True)
    parser.add_argument('--profile', choices=tuple(PROFILES), default='original')
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare', action='store_true')
    modes.add_argument('--part', type=int, choices=range(1, 8))
    modes.add_argument('--aggregate', action='store_true')
    args = parser.parse_args(argv)
    profile = PROFILES[args.profile]
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    try:
        hashes = freeze_inputs({'manifest': args.manifest, 'fresh': args.fresh,
                                'baseline': Path(__file__).resolve().parents[1] / 'jamovi_audit' / 'validation_cases.py',
                                'shared_task': Path(mr.__file__), 'runner': Path(__file__)}, output, protocol_settings(profile))
        cases = profile_cases(load_cases(args.fresh), profile)
        write_json(output / 'planned_cases.json', cases)
        manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
        if args.aggregate:
            summary = aggregate_parts(cases, output, args.model, hashes, profile)
            print('Comparison summary:', json.dumps({k: summary[k] for k in ('status', 'planned', 'attempted', 'valid', 'matching_labels', 'missing')}), flush=True)
            return 0
        dependencies = {}
        for package in ('transformers', 'mistral-common', 'tokenizers', 'jinja2'):
            try: dependencies[package] = version(package)
            except PackageNotFoundError: dependencies[package] = None
        write_json(output / 'environment.json', {'python': sys.version, 'dependencies': dependencies, 'model': args.model, 'protocol': protocol_settings(profile)})
        if args.prepare:
            prepare_assets(manifest, args.model, args.assets.resolve(), output)
            return 0
        return run_part(manifest, args.model, args.assets.resolve(), output, cases, args.part, hashes, profile)
    except Exception:
        # Never stringify subprocess/network exceptions or HTTP authorization.
        write_json(output / 'execution_error.json', {'status': 'infrastructure_error', 'code': 'comparison_setup_failed',
                                                    'scope': 'No exception text or private capability retained.'})
        print('Comparison infrastructure failed; inspect sanitized artifacts.', flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
