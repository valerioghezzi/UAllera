"""Bounded setup and real loopback/subprocess fixtures, without model inference."""
import hashlib
import io
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

try:
    from tools import prepare_minerva as minerva
except ImportError:
    minerva = None


def pe(machine=0x8664):
    data = bytearray(128)
    data[:2] = b'MZ'
    struct.pack_into('<I', data, 0x3c, 64)
    data[64:68] = b'PE\0\0'
    struct.pack_into('<H', data, 68, machine)
    return bytes(data)


def archive(entries=None, machine=0x8664):
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w') as bundle:
        for name, data in (entries or {'llama-server.exe': pe(machine), 'ggml.dll': b'fixture-library'}).items():
            # Bypass ZipInfo's host-dependent filename normalization: these
            # fixtures must preserve the actual untrusted bytes on Windows too.
            info = zipfile.ZipInfo('fixture')
            info.filename = name
            info.orig_filename = name
            bundle.writestr(info, data)
    return output.getvalue()


def asset(name, data):
    return minerva.Asset(name, 'https://public.invalid/' + name, len(data), hashlib.sha256(data).hexdigest())


class SetupFixture:
    def setUp(self):
        self.assertIsNotNone(minerva, 'The portable Minerva installer is missing')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'minerva'
        self.model = b'GGUF fixture only, not inference'
        self.tokenizer = b'official tokenizer fixture'
        self.bundle = archive()
        self.downloads = []
        self.configure()

    def configure(self, bundle=None):
        if bundle is not None:
            self.bundle = bundle
        for patcher in (
            mock.patch.object(minerva, 'MODEL', asset('model.gguf', self.model)),
            mock.patch.object(minerva, 'TOKENIZER', asset('tokenizer.json', self.tokenizer), create=True),
            mock.patch.dict(minerva.RUNTIMES, {'x64': asset('runtime.zip', self.bundle)}),
            mock.patch.object(minerva.urllib.request, 'urlopen', side_effect=self.open_asset),
        ):
            patcher.start(); self.addCleanup(patcher.stop)

    def open_asset(self, request, timeout=None):
        url = request.full_url if hasattr(request, 'full_url') else request
        self.downloads.append(url)
        return io.BytesIO(self.model if url.endswith('model.gguf') else self.tokenizer if url.endswith('tokenizer.json') else self.bundle)


class SetupTests(SetupFixture, unittest.TestCase):
    def test_first_setup_reuses_valid_cache_with_exact_template(self):
        progress = []
        result = minerva.prepare(self.root, 'x64', progress.append)
        self.assertEqual(self.model, result.model_path.read_bytes())
        self.assertEqual(pe(), result.executable_path.read_bytes())
        official = Path(__file__).resolve().parents[2] / 'work/minerva_probe/chat_template.jinja'
        if official.exists():
            self.assertEqual(official.read_bytes(), result.template_path.read_bytes())
        self.assertTrue(progress)
        self.assertEqual(3, len(self.downloads))
        minerva.prepare(self.root, 'x64')
        self.assertEqual(3, len(self.downloads))
        self.assertFalse(list(self.root.rglob('*.part')))

    def test_official_tokenizer_is_pinned_repaired_and_required_offline(self):
        prepared = minerva.prepare(self.root, 'x64')
        tokenizer = self.root / 'models' / 'tokenizer.json'
        self.assertTrue(tokenizer.is_file(), 'Official tokenizer asset was not prepared')
        self.assertEqual(tokenizer.read_bytes(), self.tokenizer)
        tokenizer.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Installazione'):
            minerva._validate_local(prepared, None)
        minerva.prepare(self.root, 'x64')
        self.assertEqual(prepared.tokenizer_path.read_bytes(), self.tokenizer)

    def test_unknown_architecture_has_no_side_effects(self):
        with self.assertRaises(ValueError): minerva.prepare(self.root, 'win32')
        self.assertFalse(self.root.exists()); self.assertFalse(self.downloads)

    def test_wrong_hash_never_publishes_partial_model(self):
        original = minerva.MODEL
        with mock.patch.object(minerva, 'MODEL', minerva.Asset(original.name, original.url, original.size, '0' * 64)):
            with self.assertRaisesRegex(ValueError, 'SHA-256'): minerva.prepare(self.root, 'x64')
        self.assertFalse((self.root / 'models' / original.name).exists())
        self.assertFalse(list(self.root.rglob('*.part')))

    def test_oversize_download_is_bounded(self):
        with mock.patch.object(minerva.urllib.request, 'urlopen', return_value=io.BytesIO(b'x' * 10000)):
            with self.assertRaisesRegex(ValueError, 'dimension|size'): minerva.prepare(self.root, 'x64')
        self.assertFalse((self.root / 'models' / minerva.MODEL.name).exists())
        self.assertFalse(list(self.root.rglob('*.part')))

    def test_download_cancellation_closes_descriptor_and_removes_partial(self):
        descriptors = []
        original = minerva.tempfile.mkstemp
        def track(*args, **kwargs):
            result = original(*args, **kwargs); descriptors.append(result[0]); return result
        def cancel(event):
            if event['stage'] == 'download': raise InterruptedError('cancel')
        with mock.patch.object(minerva.tempfile, 'mkstemp', side_effect=track):
            with self.assertRaises(InterruptedError): minerva.prepare(self.root, 'x64', cancel)
        import os
        for fd in descriptors:
            with self.assertRaises(OSError): os.fstat(fd)
        self.assertFalse(list(self.root.rglob('*.part')))

    def test_download_reads_no_more_than_expected_size_plus_one(self):
        response = io.BytesIO(b'x' * 10000)
        # Keep the fixture inspectable after its response context closes.
        class Response:
            def __enter__(self): return response
            def __exit__(self, *args): return False
        with mock.patch.object(minerva.urllib.request, 'urlopen', return_value=Response()):
            with self.assertRaises(ValueError): minerva.prepare(self.root, 'x64')
        self.assertLessEqual(response.tell(), len(self.model) + 1)

    def test_zip_symlinks_and_duplicate_case_paths_are_rejected(self):
        output = io.BytesIO()
        with zipfile.ZipFile(output, 'w') as bundle:
            bundle.writestr('llama-server.exe', pe())
            link = zipfile.ZipInfo('evil.dll'); link.create_system = 3
            link.external_attr = 0o120777 << 16
            bundle.writestr(link, '../outside')
        for data in (output.getvalue(), archive({'llama-server.exe': pe(), 'LLAMA-SERVER.EXE': pe()})):
            with self.subTest(kind=hashlib.sha256(data).hexdigest()):
                self.configure(data)
                with self.assertRaisesRegex(ValueError, 'ZIP'): minerva.prepare(self.root, 'x64')
                self.assertFalse(list(self.root.rglob('llama-server.exe')))

    def test_extra_dll_is_removed_when_repairing_cached_runtime(self):
        prepared = minerva.prepare(self.root, 'x64')
        extra = prepared.executable_path.with_name('untrusted.dll'); extra.write_bytes(b'added')
        minerva.prepare(self.root, 'x64')
        self.assertFalse(extra.exists()); self.assertEqual(3, len(self.downloads))

    def test_truncated_download_is_not_published(self):
        with mock.patch.object(minerva.urllib.request, 'urlopen', return_value=io.BytesIO(self.model[:-1])):
            with self.assertRaisesRegex(ValueError, 'dimension|size'): minerva.prepare(self.root, 'x64')
        self.assertFalse(list(self.root.rglob('*.part')))

    def test_corrupt_model_replaced_and_dll_repaired_from_cached_zip(self):
        result = minerva.prepare(self.root, 'x64')
        result.model_path.write_bytes(b'bad-cache')
        result.executable_path.with_name('ggml.dll').write_bytes(b'modified')
        minerva.prepare(self.root, 'x64')
        self.assertEqual(self.model, result.model_path.read_bytes())
        self.assertEqual(b'fixture-library', result.executable_path.with_name('ggml.dll').read_bytes())
        self.assertEqual(4, len(self.downloads))

    def test_unsafe_zip_paths_rejected_before_extraction(self):
        paths = ('../escape', '..\\escape', '..\\..\\escape', '/absolute', 'C:/drive', 'C:\\drive', '\\\\server\\share\\evil', 'folder\\escape', 'file:stream', 'NUL', 'folder./x')
        for index, path in enumerate(paths):
            with self.subTest(path=path):
                destination = self.root / f'case-{index}'
                self.configure(archive({'llama-server.exe': pe(), path: b'evil'}))
                with self.assertRaisesRegex(ValueError, 'ZIP'): minerva.prepare(destination, 'x64')
                self.assertFalse((self.root.parent / 'escape').exists())
                self.assertFalse(list(destination.rglob('llama-server.exe')))

    def test_forward_slash_nested_zip_paths_work_under_windows_normalization(self):
        self.configure(archive({'llama-server.exe': pe(), 'backend/helper.dll': b'valid'}))
        with mock.patch.object(zipfile.os, 'sep', '\\'):
            prepared = minerva.prepare(self.root, 'x64')
        self.assertEqual(b'valid', (prepared.executable_path.parent / 'backend/helper.dll').read_bytes())

    def test_windows_filename_normalization_cannot_hide_unsafe_raw_zip_path(self):
        raw_name = 'folder\\escape'
        self.configure(archive({'llama-server.exe': pe(), raw_name: b'evil'}))
        self.assertIn(raw_name.encode(), self.bundle)
        # Reproduce Windows ZipInfo parsing on any development host.
        with mock.patch.object(zipfile.os, 'sep', '\\'):
            with zipfile.ZipFile(io.BytesIO(self.bundle)) as bundle:
                info = bundle.infolist()[1]
                self.assertEqual(raw_name, info.orig_filename)
                self.assertEqual('folder/escape', info.filename)
            with self.assertRaisesRegex(ValueError, 'ZIP'):
                minerva.prepare(self.root, 'x64')
        self.assertFalse(list(self.root.rglob('llama-server.exe')))

    def test_nul_in_raw_zip_name_is_rejected_before_stdlib_truncation(self):
        raw_name = 'ggml.dll\0hidden'
        self.configure(archive({'llama-server.exe': pe(), raw_name: b'evil'}))
        with zipfile.ZipFile(io.BytesIO(self.bundle)) as bundle:
            info = bundle.infolist()[1]
            self.assertEqual(raw_name, info.orig_filename)
            self.assertEqual('ggml.dll', info.filename)
        with self.assertRaisesRegex(ValueError, 'ZIP'):
            minerva.prepare(self.root, 'x64')
        self.assertFalse(list(self.root.rglob('llama-server.exe')))

    def test_wrong_pe_architecture_rejected(self):
        self.configure(archive(machine=0xaa64))
        with self.assertRaisesRegex(ValueError, '(?i)architettura|architecture'): minerva.prepare(self.root, 'x64')
        self.assertFalse(list(self.root.rglob('llama-server.exe')))

    def test_arm64_runtime_normalized(self):
        arm = archive(machine=0xaa64)
        with mock.patch.dict(minerva.RUNTIMES, {'ARM64': asset('arm.zip', arm)}), mock.patch.object(minerva.urllib.request, 'urlopen', side_effect=lambda request, timeout=None: io.BytesIO(self.model if request.full_url.endswith('model.gguf') else self.tokenizer if request.full_url.endswith('tokenizer.json') else arm)):
            result = minerva.prepare(self.root, 'arm64')
        self.assertEqual('ARM64', result.architecture)
        self.assertEqual(pe(0xaa64), result.executable_path.read_bytes())

    def test_symlink_cache_cannot_write_outside_destination(self):
        external = Path(self.temp.name) / 'external'; external.mkdir(); self.root.mkdir()
        try: (self.root / 'models').symlink_to(external, target_is_directory=True)
        except OSError: self.skipTest('Symlink creation unavailable')
        with self.assertRaises(ValueError): minerva.prepare(self.root, 'x64')
        self.assertFalse(list(external.iterdir()))


_FIXTURE_SERVER = r'''
import http.server, json, sys, time
port, key, alias, mode = sys.argv[1:]
if mode == 'exit': sys.exit(17)
if mode == 'timeout': time.sleep(30); sys.exit(0)
class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def do_GET(self):
        if self.headers.get('Authorization') != 'Bearer ' + key:
            self.send_response(401); self.end_headers(); return
        data = json.dumps({'data': [{'id': alias}]}).encode()
        self.send_response(200); self.send_header('Content-Length', str(len(data))); self.end_headers(); self.wfile.write(data)
http.server.HTTPServer(('127.0.0.1', int(port)), Handler).serve_forever()
'''


class LaunchTests(SetupFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.prepared = minerva.prepare(self.root, 'x64')
        self.commands = []; self.processes = []; self.fixture_mode = 'ready'
        self.real_popen = subprocess.Popen
        patcher = mock.patch.object(minerva.subprocess, 'Popen', side_effect=self.spawn_fixture)
        patcher.start(); self.addCleanup(patcher.stop); self.addCleanup(self.cleanup_processes)

    def cleanup_processes(self):
        for process in self.processes:
            if process.poll() is None: process.kill()
            process.wait(timeout=5)

    def spawn_fixture(self, args, **kwargs):
        self.commands.append(args)
        value = lambda flag: args[args.index(flag) + 1]
        process = self.real_popen([sys.executable, '-c', _FIXTURE_SERVER, value('--port'), value('--api-key'), value('--alias'), self.fixture_mode], **kwargs)
        self.processes.append(process)
        return process

    def test_launch_authenticated_offline_loopback_and_context_managed(self):
        with mock.patch.object(minerva.urllib.request, 'urlopen', side_effect=AssertionError('Launch must never download')):
            with minerva.launch(self.prepared, startup_timeout=5) as server:
                self.assertTrue(server.base_url.startswith('http://127.0.0.1:'))
                self.assertGreaterEqual(len(server.api_key), 32)
                self.assertIsNone(server.process.poll())
                args = self.commands[-1]
                for flag, value in (('--host', '127.0.0.1'), ('--ctx-size', '4096'), ('--parallel', '1'), ('--alias', minerva.MODEL_ALIAS)):
                    self.assertEqual(value, args[args.index(flag) + 1])
                for flag in ('--offline', '--no-webui', '--jinja', '--chat-template-file'): self.assertIn(flag, args)
                self.assertNotIn(server.api_key, repr(server))
        self.assertIsNotNone(server.process.poll()); server.stop()

    def test_busy_port_never_adopts_or_terminates_existing_service(self):
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0)); listener.listen()
            with self.assertRaises(OSError): minerva.launch(self.prepared, port=listener.getsockname()[1], startup_timeout=.2)
            self.assertFalse(self.commands); self.assertGreater(listener.fileno(), -1)

    def test_early_exit_reported_without_secret(self):
        self.fixture_mode = 'exit'
        with self.assertRaises(RuntimeError) as failure: minerva.launch(self.prepared, startup_timeout=5)
        self.assertIn('17', str(failure.exception)); self.assertIsNotNone(self.processes[0].poll())
        self.assertNotIn(self.commands[0][self.commands[0].index('--api-key') + 1], str(failure.exception))

    def test_timeout_reaps_spawned_process(self):
        self.fixture_mode = 'timeout'
        with self.assertRaises(TimeoutError): minerva.launch(self.prepared, startup_timeout=.2)
        self.assertIsNotNone(self.processes[0].poll())

    def test_launch_callback_cancellation_reaps_spawned_process(self):
        def cancel(event):
            if event['stage'] == 'loading': raise InterruptedError('cancel')
        with self.assertRaises(InterruptedError): minerva.launch(self.prepared, startup_timeout=5, progress=cancel)
        self.assertIsNotNone(self.processes[0].poll())

    def test_cancellation_during_readiness_wait_reaps_loading_process(self):
        self.fixture_mode = 'timeout'; loading = []
        def cancel(event):
            if event['stage'] == 'loading':
                loading.append(event)
                if len(loading) > 1: raise InterruptedError('cancel while loading')
        with self.assertRaises(InterruptedError):
            minerva.launch(self.prepared, startup_timeout=.3, progress=cancel)
        self.assertIsNotNone(self.processes[0].poll())

    def test_launch_refuses_modified_assets_without_download(self):
        self.prepared.model_path.write_bytes(b'changed'); count = len(self.downloads)
        with self.assertRaises(ValueError): minerva.launch(self.root, 'x64', startup_timeout=.2)
        self.assertEqual(count, len(self.downloads)); self.assertFalse(self.commands)




class CleanupTests(unittest.TestCase):
    def test_cleanup_wait_failure_never_propagates_secret_bearing_argv_and_is_idempotent(self):
        self.assertIsNotNone(minerva)
        process = mock.Mock()
        secret = 'internal-test-capability-never-print'
        process.args = ['llama-server.exe', '--api-key', secret]
        process.poll.return_value = None
        process.wait.side_effect = subprocess.TimeoutExpired(process.args, 5)
        server = minerva.LocalServer(process, 'http://127.0.0.1:18442', secret)
        server.stop()
        server.stop()
        self.assertFalse(server.cleanup_complete)
        process.terminate.assert_called_once(); process.kill.assert_called_once()
        self.assertEqual(2, process.wait.call_count)
        self.assertNotIn(secret, repr(server))

    def test_cleanup_process_race_oserror_is_sanitized(self):
        self.assertIsNotNone(minerva)
        process = mock.Mock(); process.poll.return_value = None
        process.terminate.side_effect = OSError('secret-bearing-native-error')
        process.kill.side_effect = OSError('secret-bearing-native-error')
        process.wait.side_effect = subprocess.TimeoutExpired(['--api-key', 'never-print'], 5)
        server = minerva.LocalServer(process, 'http://127.0.0.1:18442', 'never-print')
        server.stop()
        self.assertFalse(server.cleanup_complete)


if __name__ == '__main__': unittest.main()
