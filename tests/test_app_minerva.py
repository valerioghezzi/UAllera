"""Managed Minerva orchestration tests; no downloads, native runtime or inference."""
import io
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import shutil
import subprocess
import threading
import time
import unittest
from unittest.mock import Mock, patch
import zipfile
import test_app


class ManagedMinervaTests(unittest.TestCase):
    request = test_app.LocalAppTests.request
    auth = test_app.LocalAppTests.auth
    tearDown = test_app.LocalAppTests.tearDown

    def setUp(self):
        test_app.LocalAppTests.setUp(self)
        self.assertTrue(hasattr(self.app, 'prepare_minerva'), 'Managed Minerva integration missing')
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.server.minerva_destination = Path(self.temp.name) / 'model-cache'
        self.server.minerva_architecture = 'x64'
        self.process = Mock(); self.process.poll.return_value = None
        self.engine = self.app.prepare_minerva.LocalServer(self.process, 'http://127.0.0.1:18441', 'private-internal-capability')

    def status(self):
        status, _, body = self.request('GET', '/api/minerva/status', headers=self.auth())
        self.assertEqual(200, status)
        return json.loads(body)

    def wait_ready(self):
        deadline = time.monotonic() + 5
        while True:
            state = self.status()
            if not state['busy']: return state
            if time.monotonic() > deadline: self.fail('Managed setup did not finish')
            time.sleep(.01)

    def activate(self):
        self.server.managed_minerva = self.engine
        with self.server.state_lock:
            self.server.minerva_state.update(ready=True, busy=False, stage='ready')

    def test_page_prominently_defaults_to_minerva_and_discloses_setup_download(self):
        _, _, body = self.request('GET', '/?token=' + self.server.token)
        page = body.decode()
        self.assertIn('Minerva', page); self.assertIn('Sapienza', page)
        self.assertIn('4,48 GB', page); self.assertIn('id="minervastart"', page)
        self.assertIn('<option value="minerva"', page)
        self.assertLess(page.index('value="minerva"'), page.index('value="ollama"'))
        self.assertNotIn('(facoltativa)', page)

    def test_managed_endpoints_require_session(self):
        for method, path in (('GET', '/api/minerva/status'), ('POST', '/api/minerva/start'), ('POST', '/api/minerva/cancel')):
            with self.subTest(path=path): self.assertEqual(403, self.request(method, path)[0])

    def test_start_runs_explicit_background_setup_and_status_never_reveals_secret(self):
        def prepare(destination, architecture, progress):
            progress(dict(stage='download', current=10, total=100, message='Download pubblico'))
            return 'prepared-fixture'
        with patch.object(self.app.prepare_minerva, 'prepare', side_effect=prepare) as setup, patch.object(self.app.prepare_minerva, 'launch', return_value=self.engine) as launch:
            self.assertEqual(202, self.request('POST', '/api/minerva/start', b'', self.auth())[0])
            state = self.wait_ready()
        self.assertTrue(state['ready']); self.assertIs(self.server.managed_minerva, self.engine)
        self.assertEqual(self.server.minerva_destination, setup.call_args.args[0]); launch.assert_called_once()
        self.assertNotIn(self.engine.api_key, json.dumps(state))
        _, _, body = self.request('GET', '/api/status', headers=self.auth())
        self.assertNotIn(self.engine.api_key.encode(), body)

    def test_cancel_stops_setup_before_launch(self):
        started = threading.Event(); release = threading.Event()
        def prepare(destination, architecture, progress):
            started.set(); release.wait(3)
            progress(dict(stage='download', current=1, total=2, message='Download pubblico'))
            return 'fixture'
        with patch.object(self.app.prepare_minerva, 'prepare', side_effect=prepare), patch.object(self.app.prepare_minerva, 'launch') as launch:
            self.assertEqual(202, self.request('POST', '/api/minerva/start', b'', self.auth())[0])
            self.assertTrue(started.wait(2))
            self.assertEqual(200, self.request('POST', '/api/minerva/cancel', b'', self.auth())[0])
            release.set(); state = self.wait_ready()
        self.assertFalse(state['ready']); self.assertIsNone(self.server.managed_minerva)
        self.assertIn('annull', state['message'].lower()); launch.assert_not_called()

    def test_setup_failure_is_sanitized_and_does_not_enable_analysis(self):
        with patch.object(self.app.prepare_minerva, 'prepare', side_effect=OSError('fixture disk error')):
            self.assertEqual(202, self.request('POST', '/api/minerva/start', b'', self.auth())[0])
            state = self.wait_ready()
        self.assertFalse(state['ready']); self.assertIn('disco', state['error'].lower())
        self.assertNotIn('fixture disk error', json.dumps(state))

    def test_managed_scan_refuses_unready_before_upload_is_read(self):
        headers = {**self.auth(), 'X-Local-Provider': 'minerva', 'Content-Length': '100'}
        self.assertEqual(400, self.request('POST', '/api/scan', b'', headers)[0])

    def test_managed_config_comes_only_from_server_and_never_request_credentials(self):
        self.activate(); captured = []
        def audit(data, **kwargs):
            captured.append(kwargs['linguistic_config'])
            return {'html': 'Report locale', 'summary': {'student_count': 0}}
        headers = {**self.auth(), 'X-Local-Provider': 'minerva', 'X-Local-Endpoint': 'https://attacker.invalid',
                   'X-Local-Model': 'forged', 'Authorization': 'Bearer browser-forged', 'X-Local-API-Key': 'forged'}
        with patch.object(self.app, 'audit_zip', side_effect=audit):
            self.assertEqual(202, self.request('POST', '/api/scan', b'fixture-zip', headers)[0])
            deadline = time.monotonic() + 3
            while not captured:
                if time.monotonic() > deadline: self.fail('Audit did not start')
                time.sleep(.01)
        config = captured[0]
        self.assertIs(config['_managed_minerva'], self.engine)
        self.assertEqual(config['endpoint'], self.engine.base_url)
        self.assertEqual(config['model'], self.engine.model)
        self.assertNotIn('forged', repr(config))

    def test_managed_models_use_owned_profile_without_browser_endpoint(self):
        self.activate()
        status, _, body = self.request('GET', '/api/models?provider=minerva&endpoint=https://attacker.invalid', headers={**self.auth(), 'X-Local-Provider': 'minerva'})
        self.assertEqual(200, status); self.assertEqual([self.engine.model], json.loads(body)['models'])
        self.assertNotIn(self.engine.api_key.encode(), body)

    def test_managed_models_need_explicit_managed_provider_header(self):
        self.activate()
        self.assertEqual(400, self.request('GET', '/api/models?provider=minerva', headers=self.auth())[0])

    def test_closing_app_stops_only_owned_engine(self):
        self.activate()
        self.server.server_close()
        self.process.terminate.assert_called_once()

    def test_dead_owned_engine_is_not_ready_and_managed_scan_is_refused(self):
        self.activate(); self.process.poll.return_value = 17
        self.assertFalse(self.status()['ready'])
        self.assertEqual(400, self.request('POST', '/api/scan', b'fixture', {**self.auth(), 'X-Local-Provider': 'minerva'})[0])

    def test_shutdown_during_setup_never_launches_a_late_engine(self):
        started = threading.Event(); release = threading.Event()
        def prepare(destination, architecture, progress):
            started.set(); release.wait(3); return 'fixture'
        with patch.object(self.app.prepare_minerva, 'prepare', side_effect=prepare), patch.object(self.app.prepare_minerva, 'launch') as launch:
            self.assertEqual(202, self.request('POST', '/api/minerva/start', b'', self.auth())[0])
            self.assertTrue(started.wait(2)); self.server.server_close(); release.set()
            self.server.minerva_worker.join(timeout=3)
            self.assertFalse(self.server.minerva_worker.is_alive()); launch.assert_not_called()

    def test_cancelled_audit_keeps_partial_report_downloadable_with_explicit_counts(self):
        result = {'html': '<h1>Rapporto incompleto</h1>', 'summary': dict(student_count=1,
            status='cancelled', incomplete=True, planned_student_count=3, unexecuted_student_count=2)}
        with patch.object(self.app, 'audit_zip', return_value=result):
            self.app.run_job(self.server, b'fixture')
        status, _, body = self.request('GET', '/api/status', headers=self.auth())
        state = json.loads(body)
        self.assertEqual(200, status); self.assertTrue(state['ready']); self.assertFalse(state['busy'])
        self.assertTrue(state['incomplete']); self.assertEqual(3, state['planned_student_count'])
        self.assertEqual(2, state['unexecuted_student_count']); self.assertEqual(1, state['students'])
        self.assertIn('annull', state['message'].lower()); self.assertIn('incompleto', state['message'].lower())
        self.assertEqual(200, self.request('GET', '/report', headers=self.auth())[0])
        with patch.object(self.app, 'audit_zip', return_value={'html': 'complete', 'summary': {'student_count': 4}}):
            self.app.run_job(self.server, b'fixture')
        self.assertFalse(self.server.state['incomplete'])

    def test_start_and_validation_are_mutually_exclusive_with_setup(self):
        with self.server.state_lock: self.server.minerva_state.update(busy=True)
        headers = {**self.auth(), 'X-Local-Provider': 'minerva'}
        self.assertEqual(409, self.request('POST', '/api/minerva/start', b'', self.auth())[0])
        self.assertEqual(400, self.request('POST', '/api/validate', b'', headers)[0])


class ManagedTransportTests(unittest.TestCase):
    def test_real_loopback_auth_survives_normalization_and_never_uses_user_strings(self):
        from tools.prepare_minerva import LocalServer, MODEL_ALIAS
        from jamovi_audit.local_llm import validate_config, _request, LocalEngineError
        observed = []
        key = 'private-fixture-loopback-capability'
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_GET(self):
                observed.append(self.headers.get('Authorization'))
                body = b'{"data": []}'
                self.send_response(200 if observed[-1] == 'Bearer ' + key else 401)
                self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)
        fixture = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        worker = threading.Thread(target=fixture.serve_forever, daemon=True); worker.start()
        self.addCleanup(worker.join, 3); self.addCleanup(fixture.server_close); self.addCleanup(fixture.shutdown)
        alive = [True]
        process = Mock(); process.poll.side_effect = lambda: None if alive[0] else 0
        process.terminate.side_effect = lambda: alive.__setitem__(0, False)
        engine = LocalServer(process, f'http://127.0.0.1:{fixture.server_port}', key)
        self.addCleanup(engine.stop)
        source = dict(enabled=True, provider='llamacpp', endpoint=engine.base_url, model=MODEL_ALIAS,
                      _managed_minerva=engine, api_key='forged', Authorization='Bearer forged')
        config = validate_config(validate_config(source))
        self.assertIs(engine, config['_managed_minerva'])
        self.assertEqual({'data': []}, _request(config, 'GET', '/v1/models'))
        self.assertEqual(['Bearer ' + key], observed)
        external = validate_config(dict(source, _managed_minerva=None, api_key=key, Authorization='Bearer ' + key))
        with self.assertRaises(LocalEngineError): _request(external, 'GET', '/v1/models')
        self.assertIsNone(observed[-1])
        for changes in ({'_managed_minerva': key}, {'endpoint': 'http://127.0.0.1:1'}, {'model': 'forged'}):
            with self.subTest(changes=list(changes)):
                with self.assertRaises(ValueError): validate_config(dict(source, **changes))
        engine.stop(); self.assertFalse(alive[0])
        count = len(observed)
        with self.assertRaises(LocalEngineError): _request(config, 'GET', '/v1/models')
        self.assertEqual(count, len(observed))


@unittest.skipUnless(shutil.which('node'), 'UI check requires development Node runtime')
class ManagedUiStateTests(unittest.TestCase):
    def test_managed_ui_never_enables_analysis_until_ready_or_exposes_credentials(self):
        import app
        source = app.PAGE.split('<script>', 1)[1].split('</script>', 1)[0]
        script = r"""
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
const elements=new Map();const get=id=>{if(!elements.has(id))elements.set(id,{value:'',disabled:true,files:[],hidden:false});return elements.get(id)};
get('provider').value='minerva';get('timeout').value='300';get('zip').files=[{name:'local.zip'}];
const timers=[];const context=vm.createContext({document:{getElementById:get},location:{search:'?token=test'},URLSearchParams,setInterval:callback=>{timers.push(callback);return 1},clearInterval:()=>{},fetch:async()=>({ok:true,json:async()=>({busy:false,ready:false,total:1,current:0,message:'Attesa'})})});
vm.runInContext(JSON.parse(readFileSync(0,'utf8')),context);await vm.runInContext('pollMinerva()',context);
assert.equal(get('start').disabled,true);assert.equal(get('validate').disabled,true);assert.ok(timers.length>0,'Keep polling readiness after startup so a stopped engine can be restarted');
vm.runInContext('minervaReady=true;syncControls()',context);
assert.equal(get('start').disabled,false);assert.equal(get('validate').disabled,false);
const headers=vm.runInContext('engineHeaders()',context);
assert.deepEqual(Object.keys(headers).sort(),['X-Local-Provider','X-Local-Timeout']);
vm.runInContext('minervaReady=false;syncControls()',context);
assert.equal(get('start').disabled,true);
"""
        result = subprocess.run([shutil.which('node'), '--input-type=module', '-e', script], input=json.dumps(source),
                                text=True, capture_output=True, timeout=15)
        self.assertEqual(0, result.returncode, result.stderr)


    def test_cancelled_report_summary_has_processed_planned_and_unexecuted_counts(self):
        import app
        source = app.PAGE.split('<script>', 1)[1].split('</script>', 1)[0]
        script = r"""
import {readFileSync} from 'node:fs';import vm from 'node:vm';import assert from 'node:assert/strict';
const elements=new Map();const get=id=>{if(!elements.has(id))elements.set(id,{value:'',disabled:true,files:[],hidden:true});return elements.get(id)};
const state={busy:false,total:3,done:1,ready:true,mode:'audit',incomplete:true,students:1,planned_student_count:3,unexecuted_student_count:2,message:'Controllo annullato'};
const context=vm.createContext({document:{getElementById:get},location:{search:'?token=test'},URLSearchParams,clearInterval:()=>{},fetch:async()=>({ok:true,json:async()=>state})});
vm.runInContext(JSON.parse(readFileSync(0,'utf8')),context);await vm.runInContext('poll()',context);
assert.match(get('summary').textContent,/incompleto/i);assert.match(get('summary').textContent,/1\/3/);assert.match(get('summary').textContent,/2.*non eseguit/i);
assert.equal(get('download').hidden,false);assert.match(get('resulttitle').textContent,/incompleto/i);
"""
        result = subprocess.run([shutil.which('node'), '--input-type=module', '-e', script], input=json.dumps(source), text=True, capture_output=True, timeout=15)
        self.assertEqual(0, result.returncode, result.stderr)


class MinervaPortableTests(unittest.TestCase):
    def test_bundle_has_architecture_setup_launcher_and_all_python_dependencies(self):
        from tools.build_windows import build
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w') as bundle:
            bundle.writestr('python.exe', b'fixture')
            bundle.writestr('python314._pth', 'python314.zip\n.\n')
        with tempfile.TemporaryDirectory() as directory, patch('tools.build_windows.verify_runtime'):
            for architecture in ('x64', 'ARM64'):
                with self.subTest(architecture=architecture):
                    folder, bundle = build(architecture, buffer.getvalue(), Path(directory))
                    self.assertEqual(architecture, json.loads((folder / 'minerva_config.json').read_text())['architecture'])
                    self.assertTrue((folder / 'tools/prepare_minerva.py').is_file())
                    command = (folder / 'Prepara_Minerva.cmd').read_bytes()
                    self.assertIn(b'runtime\\python.exe', command)
                    self.assertIn(('--arch ' + architecture).encode(), command)
                    self.assertNotIn(b'pip install', command)
                    self.assertTrue(bundle.is_file())


if __name__ == '__main__': unittest.main()
