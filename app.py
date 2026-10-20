"""Desktop entry point: browser interface served exclusively on loopback."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit, parse_qs
from pathlib import Path
import argparse
import json
import platform
import secrets
import threading
import webbrowser
from tools import prepare_minerva
from jamovi_audit.pipeline import audit_zip
from jamovi_audit.local_llm import validate_config, list_models
from jamovi_audit.validation import run_validation, render_validation

MAX_UPLOAD = 512 * 1024 * 1024
CSP = "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; img-src data:; form-action 'self'; frame-ancestors 'none'; base-uri 'none'"
PAGE = '''<!doctype html><html lang="it"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Controllo esercitazioni jamovi</title><style>
*{box-sizing:border-box}body{margin:0;background:#f5f5f5;color:#171717;font:17px/1.6 Verdana,Arial,sans-serif}main{max-width:930px;margin:48px auto;padding:0 24px}h1{font-size:30px;line-height:1.3;margin-bottom:12px}h2{font-size:21px}p{max-width:76ch}section{background:#fff;border:1px solid #999;border-radius:7px;padding:26px;margin:24px 0}.tag{font-size:14px;letter-spacing:.05em}label{display:block;font-weight:bold;margin-bottom:12px}input[type=file]{display:block;max-width:100%;font:inherit;margin:12px 0 22px}button,a.button{font:inherit;display:inline-block;cursor:pointer;border:2px solid #171717;border-radius:4px;padding:11px 18px;background:#fff;color:#171717;text-decoration:none;margin:5px 8px 5px 0}button.primary{background:#171717;color:#fff}input[type=text],input[type=number],select{font:inherit;max-width:100%;width:100%;padding:8px;margin-bottom:15px}button:disabled{opacity:.4;cursor:default}button:focus-visible,input:focus-visible,a:focus-visible{outline:3px solid #555;outline-offset:4px}progress{width:100%;height:24px;accent-color:#222}.muted{font-size:14px;color:#444}.error{border-left:5px solid #111;padding:14px;background:#eee}#download[hidden],#result[hidden],#error[hidden]{display:none}.footer{margin-top:32px} @media(max-width:600px){main{margin:24px auto}h1{font-size:25px}section{padding:18px}button{width:100%}}
</style></head><body><main><div class="tag">ELABORAZIONE SUL PC · SENZA SERVIZI ESTERNI</div><h1>Controllo esercitazioni jamovi</h1><p>Esamina i file jamovi e le relazioni Word nelle cartelle degli studenti. Produce un unico rapporto con il dettaglio delle anomalie per ciascuno.</p>
<section><h2>Minerva · motore linguistico locale</h2><p>Minerva di Sapienza è il motore linguistico principale di UAllera. Elabora le relazioni sul PC e propone osservazioni che richiedono la conferma del tutor.</p><p class="muted">La prima preparazione scarica soltanto il modello pubblico (4,48 GB) e il motore llama.cpp compilato dal sorgente ufficiale. Servono circa 5 GB liberi per i file, oltre alla memoria per avviare il modello. Le consegne non vengono inviate in rete. Dopo la preparazione Minerva funziona offline, senza installare Python, R o Ollama.</p><p role="note"><strong>Collaudo Minerva non superato:</strong> con il tokenizer ufficiale, 13 giudizi concordanti su 70 casi sintetici e nessun errore previsto riconosciuto. Le proposte linguistiche sono sperimentali e richiedono il riscontro del tutor.</p><button class="primary" id="minervastart">Prepara e avvia Minerva</button><button id="minervacancel" disabled>Annulla preparazione</button><p id="minervastatus" role="status" aria-live="polite">Minerva deve essere avviato prima dell’analisi linguistica.</p><progress id="minervaprogress" value="0" max="1" aria-label="Avanzamento preparazione Minerva"></progress><label for="provider">Motore linguistico</label><select id="provider"><option value="minerva" selected>Minerva Sapienza · gestito da UAllera</option><option value="ollama">Ollama · configurazione avanzata</option><option value="lmstudio">LM Studio · configurazione avanzata</option><option value="llamacpp">llama.cpp · configurazione avanzata</option></select><details id="advanced"><summary>Configurazione di altri motori locali</summary><p class="muted">Per un motore avanzato già installato, seleziona il servizio e il modello. Minerva viene preparato direttamente da UAllera.</p><label for="endpoint">Indirizzo locale</label><input id="endpoint" value="http://127.0.0.1:11434" type="text" disabled><label for="model">Nome del modello già installato</label><input id="model" list="models" type="text" disabled><datalist id="models"></datalist><button id="loadmodels" disabled>Elenca modelli locali</button></details><label for="timeout">Tempo massimo di attesa per operazione (secondi)</label><input id="timeout" type="number" min="1" max="900" value="900"><button id="validate" disabled>Verifica modello su 70 casi nuovi</button><p class="muted">La verifica utilizza testi sintetici separati dagli esempi del prompt e può richiedere diversi minuti. Produce un rapporto dedicato.</p><p id="modelstatus" role="status" class="muted">Il profilo Minerva utilizza solo il processo locale avviato da UAllera.</p></section>
<section><label for="zip">Seleziona lo ZIP delle consegne</label><p class="muted">Puoi utilizzare lo ZIP scaricato con tutte le cartelle degli studenti, senza estrarlo a mano. Limite 512 MiB.</p><input type="file" id="zip" accept=".zip,application/zip"><button class="primary" id="start" disabled>Avvia controllo</button><button id="cancel" disabled>Annulla</button><p id="status" role="status" aria-live="polite">In attesa del file.</p><progress id="progress" value="0" max="1" aria-label="Avanzamento del controllo"></progress><p id="error" class="error" role="alert" hidden></p><div id="result" hidden><h2 id="resulttitle">Rapporto pronto</h2><p id="summary"></p><button class="primary" id="download" hidden>Scarica rapporto unico</button><p class="muted">Il rapporto HTML si apre senza connessione ed è stampabile dal browser. Comprende un indice e una sezione per studente.</p></div></section>
<p class="muted">I documenti originali restano intatti. Le interpretazioni aperte e le evidenze di somiglianza richiedono la valutazione del tutor. Il software non certifica chi abbia scritto un testo né l’assenza di AI.</p><div class="footer"><button id="exit">Termina programma</button></div></main><script>
const token=new URLSearchParams(location.search).get('token');const $=id=>document.getElementById(id);let timer=null;let minervaTimer=null;let busy=false;let minervaBusy=false;let minervaReady=false;let downloadName="Rapporto_esercitazioni.html";
function message(t){$('status').textContent=t}function fail(t){$('error').textContent=t;$('error').hidden=false}function auth(){return {'X-Session-Token':token}}
async function api(path,options={}){let r=await fetch(path,{...options,headers:{...auth(),...(options.headers||{})}});if(!r.ok){let d=await r.json().catch(()=>({error:'Errore nella richiesta locale.'}));throw Error(d.error||'Errore nella richiesta locale.')}return r}
function managed(){return $('provider').value==='minerva'}
function syncControls(){let p=$('provider').value;let usable=!!p&&(!managed()||minervaReady);$('start').disabled=busy||!$('zip').files.length||!usable;$('validate').disabled=busy||!usable;$('loadmodels').disabled=busy||!p||managed();$('endpoint').disabled=managed()||!p;$('model').disabled=managed()||!p;$('minervastart').disabled=busy||minervaBusy||minervaReady;$('minervacancel').disabled=!minervaBusy}
function engineHeaders(){if(managed())return {'X-Local-Provider':'minerva','X-Local-Timeout':$('timeout').value};return {'X-Local-Provider':$('provider').value,'X-Local-Endpoint':$('endpoint').value,'X-Local-Model':$('model').value,'X-Local-Timeout':$('timeout').value}}
$('provider').onchange=()=>{let p=$('provider').value;$('model').value='';$('endpoint').value='http://127.0.0.1:'+({ollama:11434,lmstudio:1234,llamacpp:8080}[p]||11434);$('modelstatus').textContent=managed()?'Il profilo Minerva utilizza solo il processo locale avviato da UAllera.':'Avvia il motore avanzato e seleziona un modello locale.';syncControls()};
async function pollMinerva(){try{let s=await(await api('/api/minerva/status')).json();minervaBusy=s.busy;minervaReady=s.ready;$('minervaprogress').max=Math.max(s.total||0,1);$('minervaprogress').value=s.current||0;$('minervastatus').textContent=s.error||s.message;if(!minervaTimer&&typeof setInterval==='function')minervaTimer=setInterval(pollMinerva,1000);syncControls()}catch(e){minervaReady=false;$('minervastatus').textContent=e.message;syncControls()}}
$('minervastart').onclick=async()=>{minervaBusy=true;syncControls();$('minervastatus').textContent='Preparazione e avvio del motore locale…';try{await api('/api/minerva/start',{method:'POST'});await pollMinerva()}catch(e){minervaBusy=false;$('minervastatus').textContent=e.message;syncControls()}};
$('minervacancel').onclick=async()=>{try{await api('/api/minerva/cancel',{method:'POST'});$('minervastatus').textContent='Annullamento preparazione richiesto…'}catch(e){$('minervastatus').textContent=e.message}};
$('loadmodels').onclick=async()=>{try{$('loadmodels').disabled=true;let q=new URLSearchParams({provider:$('provider').value,endpoint:$('endpoint').value});let d=await(await api('/api/models?'+q)).json();$('models').replaceChildren(...d.models.map(m=>{let o=document.createElement('option');o.value=m;return o}));if(d.models.length)$('model').value=d.models[0];$('modelstatus').textContent=d.models.length+' modelli disponibili sul motore locale.'}catch(e){$('modelstatus').textContent=e.message}finally{syncControls()}};
function beginJob(){busy=true;syncControls();$('cancel').disabled=false;$('result').hidden=true;$('download').hidden=true;$('error').hidden=true}
$('validate').onclick=async()=>{if(!managed()&&(!$('provider').value||!$('model').value)){fail('Seleziona un motore e un modello locale.');return}beginJob();message('Verifica linguistica locale sui casi sintetici…');try{await api('/api/validate',{method:'POST',headers:engineHeaders()});timer=setInterval(poll,500);await poll()}catch(e){busy=false;fail(e.message);syncControls();$('cancel').disabled=true}};
$('zip').onchange=()=>{syncControls();$('error').hidden=true;message($('zip').files.length?$('zip').files[0].name:'In attesa del file.')};
async function poll(){try{let s=await(await api('/api/status')).json();busy=s.busy;syncControls();$('progress').max=Math.max(s.total,1);$('progress').value=s.done;message(s.message);$('cancel').disabled=!s.busy;downloadName=s.filename||'Rapporto_esercitazioni.html';if(!s.busy){clearInterval(timer);timer=null;if(s.error)fail(s.error);if(s.ready){$('resulttitle').textContent=s.incomplete?'Rapporto incompleto disponibile':'Rapporto pronto';$('summary').textContent=s.incomplete?'Controllo annullato: rapporto incompleto. '+s.students+'/'+s.planned_student_count+' studenti esaminati; '+s.unexecuted_student_count+' non eseguiti. Consulta il rapporto per le analisi già effettuate.':s.mode==='validation'?'Verifica del modello terminata. Consulta copertura, errori riconosciuti e limiti nel rapporto.':s.students+' studenti esaminati. Consulta il rapporto per completezza, errori e verifiche da approfondire.';$('result').hidden=false;$('download').hidden=false;}}}catch(e){clearInterval(timer);timer=null;busy=false;fail(e.message);syncControls()}}
$('start').onclick=async()=>{let f=$('zip').files[0];if(!f)return;if(f.size>536870912){fail('ZIP superiore al limite di 512 MiB.');return}beginJob();message('Lettura dello ZIP sul PC…');try{await api('/api/scan',{method:'POST',headers:{'Content-Type':'application/zip',...engineHeaders()},body:f});timer=setInterval(poll,500);await poll()}catch(e){busy=false;fail(e.message);syncControls();$('cancel').disabled=true}};
$('cancel').onclick=async()=>{try{await api('/api/cancel',{method:'POST'});message('Annullamento richiesto…')}catch(e){fail(e.message)}};
$('download').onclick=async()=>{try{let r=await api('/report');let blob=await r.blob();let url=URL.createObjectURL(blob);let a=document.createElement('a');a.href=url;a.download=downloadName;a.click();setTimeout(()=>URL.revokeObjectURL(url),30000)}catch(e){fail(e.message)}};
$('exit').onclick=async()=>{try{await api('/api/exit',{method:'POST'});clearInterval(timer);clearInterval(minervaTimer);message('Programma terminato. Puoi chiudere questa scheda.');$('start').disabled=true;$('cancel').disabled=true;$('minervastart').disabled=true;$('minervacancel').disabled=true;$('exit').disabled=true}catch(e){fail(e.message)}};
syncControls();pollMinerva();
</script></body></html>'''


class AuditServer(ThreadingHTTPServer):
    daemon_threads = True

    def server_close(self):
        self.cancel.set()
        self.minerva_cancel.set()
        self.closing.set()
        with self.state_lock:
            engine = self.managed_minerva
            self.managed_minerva = None
            self.minerva_state.update(ready=False)
        if engine is not None:
            engine.stop()
        super().server_close()


def _native_minerva_architecture():
    config = Path(__file__).resolve().parent / 'minerva_config.json'
    if config.is_file():
        architecture = json.loads(config.read_text(encoding='utf-8')).get('architecture')
        if architecture in ('x64', 'ARM64'):
            return architecture
        raise ValueError('Architettura del pacchetto Minerva non valida.')
    return 'ARM64' if platform.machine().lower() in ('arm64', 'aarch64') else 'x64'


def make_server(port=0, *, minerva_destination=None, minerva_architecture=None):
    server = AuditServer(('127.0.0.1', port), Handler)
    server.token = secrets.token_urlsafe(32)
    server.state_lock = threading.Lock()
    server.cancel = threading.Event()
    server.minerva_cancel = threading.Event()
    server.closing = threading.Event()
    server.minerva_destination = Path(minerva_destination) if minerva_destination is not None else Path(__file__).resolve().parent / 'minerva'
    server.minerva_architecture = minerva_architecture or _native_minerva_architecture()
    server.managed_minerva = None
    server.minerva_worker = None
    server.minerva_state = dict(busy=False, ready=False, stage='idle', current=0, total=1,
        message='Prepara e avvia Minerva per l’analisi linguistica locale.', error='')
    server.state = dict(busy=False, done=0, total=1, message='In attesa del file.',
                        error='', ready=False, students=0, incomplete=False, planned_student_count=0, unexecuted_student_count=0, result=None, mode='audit', filename='Rapporto_esercitazioni.html')
    return server


class Handler(BaseHTTPRequestHandler):
    server_version = 'ControlloLocale'
    protocol_version = 'HTTP/1.1'

    def handle(self):
        try:
            super().handle()
        except (ConnectionResetError, BrokenPipeError):
            # A closed browser tab may reset an HTTP/1.1 keep-alive connection.
            # Other server errors retain their normal diagnostic traceback.
            self.close_connection = True

    def log_message(self, *args):
        # Never log the session token or student data.
        pass

    def _reply(self, status, body, content_type='application/json; charset=utf-8', download=False, filename='Rapporto_esercitazioni.html'):
        if not isinstance(body, bytes):
            body = json.dumps(body, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Content-Security-Policy', CSP)
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('X-Content-Type-Options', 'nosniff')
        if download:
            self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(body)

    def _authorize(self):
        expected = f'127.0.0.1:{self.server.server_address[1]}'
        if self.headers.get('Host') != expected:
            self.close_connection = True
            self._reply(403, {'error': 'Host non consentito.'})
            return False
        origin = self.headers.get('Origin')
        if origin and origin != 'http://' + expected:
            self.close_connection = True
            self._reply(403, {'error': 'Origine non consentita.'})
            return False
        query = parse_qs(urlsplit(self.path).query)
        token = self.headers.get('X-Session-Token') or query.get('token', [''])[0]
        if not secrets.compare_digest(token, self.server.token):
            self.close_connection = True
            self._reply(403, {'error': 'Sessione non valida. Apri il collegamento mostrato dal programma.'})
            return False
        return True

    def do_GET(self):
        if not self._authorize():
            return
        path = urlsplit(self.path).path
        if path == '/':
            self._reply(200, PAGE.encode('utf-8'), 'text/html; charset=utf-8')
        elif path == '/api/minerva/status':
            with self.server.state_lock:
                engine = self.server.managed_minerva
                if engine is not None and engine.process.poll() is not None:
                    self.server.minerva_state.update(ready=False, message='Minerva si è fermato. Avvia nuovamente il motore locale.')
                state = dict(self.server.minerva_state)
            self._reply(200, state)
        elif path == '/api/models':
            query = parse_qs(urlsplit(self.path).query)
            config = {k: query.get(k, [''])[0] for k in ('provider', 'endpoint')}
            try:
                if config['provider'] == 'minerva' and self.headers.get('X-Local-Provider') != 'minerva':
                    raise ValueError('Seleziona esplicitamente il profilo Minerva gestito da UAllera.')
                config = self._linguistic_config(config['provider'], config['endpoint'])
                models = [config['model']] if config.get('_managed_minerva') is not None else list_models(config)
                self._reply(200, {'models': models})
            except ValueError as exc:
                self._reply(400, {'error': str(exc)})
            except Exception as exc:
                self._reply(503, {'error': f'Motore locale non disponibile: {exc}'})
        elif path == '/api/status':
            with self.server.state_lock:
                state = {k: v for k, v in self.server.state.items() if k != 'result'}
            self._reply(200, state)
        elif path == '/report':
            with self.server.state_lock:
                result = self.server.state['result']
                filename = self.server.state['filename']
            if result is None:
                self._reply(404, {'error': 'Nessun rapporto pronto.'})
            else:
                self._reply(200, result['html'].encode('utf-8'), 'text/html; charset=utf-8', True, filename)
        else:
            self._reply(404, {'error': 'Risorsa non disponibile.'})

    def _linguistic_config(self, provider, endpoint='', model='', timeout='900'):
        if provider == 'minerva':
            with self.server.state_lock:
                engine = self.server.managed_minerva
                ready = self.server.minerva_state['ready'] and not self.server.minerva_state['busy']
            if not isinstance(engine, prepare_minerva.LocalServer) or not ready or engine.process.poll() is not None:
                raise ValueError('Minerva non è pronto. Utilizza «Prepara e avvia Minerva» prima dell’analisi linguistica.')
            config = validate_config(dict(enabled=True, provider='llamacpp', endpoint=engine.base_url,
                                          model=engine.model, timeout=timeout))
            config['_managed_minerva'] = engine
            return config
        return validate_config(dict(enabled=bool(provider), provider=provider or 'ollama',
                                    endpoint=endpoint, model=model, timeout=timeout))

    def do_POST(self):
        if not self._authorize():
            return
        path = urlsplit(self.path).path
        if path in ('/api/minerva/start', '/api/minerva/cancel'):
            if self.headers.get('Transfer-Encoding') or self.headers.get('Content-Length', '0') != '0':
                self.close_connection = True
                self._reply(400, {'error': 'La preparazione Minerva non accetta documenti o configurazioni dal browser.'})
                return
            if path.endswith('/cancel'):
                self.server.minerva_cancel.set()
                self._reply(200, {'ok': True})
                return
            with self.server.state_lock:
                engine = self.server.managed_minerva
                if self.server.minerva_state['busy'] or self.server.state['busy']:
                    self._reply(409, {'error': 'Una preparazione o un controllo è già in corso.'})
                    return
                if engine is not None and engine.process.poll() is None:
                    self._reply(200, {'ok': True, 'ready': True})
                    return
                self.server.minerva_cancel.clear()
                self.server.minerva_state.update(busy=True, ready=False, stage='starting', current=0, total=1,
                    message='Preparazione e avvio Minerva locale…', error='')
            worker = threading.Thread(target=run_minerva_job, args=(self.server,), daemon=True)
            self.server.minerva_worker = worker
            worker.start()
            self._reply(202, {'ok': True})
            return
        if path == '/api/cancel':
            self.server.cancel.set()
            self._reply(200, {'ok': True})
            return
        if path == '/api/exit':
            self.server.cancel.set()
            self.server.minerva_cancel.set()
            self._reply(200, {'ok': True})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return
        if path not in ('/api/scan', '/api/validate'):
            self._reply(404, {'error': 'Risorsa non disponibile.'})
            return
        try:
            provider = self.headers.get('X-Local-Provider', '')
            linguistic_config = self._linguistic_config(provider,
                self.headers.get('X-Local-Endpoint', ''), self.headers.get('X-Local-Model', ''),
                self.headers.get('X-Local-Timeout', '900'))
            if path == '/api/validate' and not linguistic_config.get('enabled'):
                raise ValueError('Seleziona un motore locale per la verifica linguistica.')
            if linguistic_config.get('enabled') and not linguistic_config.get('model'):
                raise ValueError('Indica un modello locale prima del controllo.')
        except ValueError as exc:
            self.close_connection = True
            self._reply(400, {'error': str(exc)})
            return
        if path == '/api/validate':
            if self.headers.get('Transfer-Encoding') or self.headers.get('Content-Length', '0') != '0':
                self.close_connection = True
                self._reply(400, {'error': 'La verifica del modello non accetta documenti.'})
                return
            with self.server.state_lock:
                if self.server.state['busy']:
                    self.close_connection = True
                    self._reply(409, {'error': 'Un controllo è già in corso.'})
                    return
                self.server.cancel.clear()
                self.server.state.update(busy=True, done=0, total=70, message='Verifica linguistica locale…',
                    error='', ready=False, students=0, incomplete=False, planned_student_count=0, unexecuted_student_count=0, result=None, mode='validation', filename='Validazione_modello.html')
            threading.Thread(target=run_validation_job, args=(self.server, linguistic_config), daemon=True).start()
            self._reply(202, {'ok': True})
            return
        try:
            if self.headers.get('Transfer-Encoding'):
                raise ValueError('Trasferimento non supportato.')
            size = int(self.headers.get('Content-Length', '0'))
            if size <= 0:
                raise ValueError('Seleziona un file ZIP non vuoto.')
        except ValueError as exc:
            self.close_connection = True
            self._reply(400, {'error': str(exc)})
            return
        if size > MAX_UPLOAD:
            self.close_connection = True
            self._reply(413, {'error': 'ZIP superiore al limite di 512 MiB.'})
            return
        with self.server.state_lock:
            if self.server.state['busy']:
                self.close_connection = True
                self._reply(409, {'error': 'Un controllo è già in corso.'})
                return
            self.server.cancel.clear()
            self.server.state.update(busy=True, done=0, total=1, message='Lettura dello ZIP…',
                                     error='', ready=False, students=0, incomplete=False, planned_student_count=0, unexecuted_student_count=0, result=None, mode='audit', filename='Rapporto_esercitazioni.html')
        try:
            self.connection.settimeout(60)
            data = self.rfile.read(size)
            if len(data) != size:
                raise ValueError('Trasferimento locale incompleto.')
        except (OSError, ValueError) as exc:
            with self.server.state_lock:
                self.server.state.update(busy=False, error=str(exc), message='File non acquisito.')
            self.close_connection = True
            self._reply(400, {'error': str(exc)})
            return
        threading.Thread(target=run_job, args=(self.server, data, linguistic_config), daemon=True).start()
        self._reply(202, {'ok': True})


def run_minerva_job(server):
    owned = None
    def progress(event):
        if server.minerva_cancel.is_set() or server.closing.is_set():
            raise InterruptedError('Preparazione annullata')
        # Only installer-owned scalar progress fields enter browser status.
        with server.state_lock:
            server.minerva_state.update(stage=event['stage'], current=event['current'], total=event['total'],
                                        message=event['message'])
    try:
        prepared = prepare_minerva.prepare(server.minerva_destination, server.minerva_architecture, progress)
        if server.minerva_cancel.is_set() or server.closing.is_set():
            raise InterruptedError('Preparazione annullata')
        owned = prepare_minerva.launch(prepared, progress=progress, startup_timeout=600)
        with server.state_lock:
            if server.minerva_cancel.is_set() or server.closing.is_set():
                raise InterruptedError('Preparazione annullata')
            server.managed_minerva = owned
            server.minerva_state.update(busy=False, ready=True, stage='ready', message='Minerva è pronto. Analisi linguistica sul PC; le proposte richiedono la conferma del tutor.', error='')
            owned = None
    except InterruptedError:
        with server.state_lock:
            server.minerva_state.update(busy=False, ready=False, stage='cancelled', message='Preparazione Minerva annullata. Puoi riprovare.', error='')
    except Exception as error:
        if isinstance(error, OSError):
            detail = 'Preparazione non riuscita: verifica connessione, spazio libero su disco e permessi della cartella.'
        elif isinstance(error, ValueError):
            detail = 'Preparazione non riuscita: verifica di integrità o architettura del motore non superata. Riprova la preparazione.'
        else:
            detail = 'Minerva non si è avviato. Verifica la memoria disponibile e riprova; il modello deve essere pronto prima dell’analisi linguistica.'
        with server.state_lock:
            server.minerva_state.update(busy=False, ready=False, stage='error', message='Minerva non è pronto.', error=detail)
    finally:
        if owned is not None:
            owned.stop()


def run_job(server, data, linguistic_config=None):
    def progress(done, total, label):
        with server.state_lock:
            server.state.update(done=done, total=total, message=f'{done}/{total} — {label}')
    try:
        result = audit_zip(data, progress=progress, cancelled=server.cancel.is_set, linguistic_config=linguistic_config)
        with server.state_lock:
            server.state.update(result=result, busy=False, ready=True,
                                students=result['summary']['student_count'],
                                incomplete=bool(result['summary'].get('incomplete') or result['summary'].get('status') == 'cancelled'),
                                planned_student_count=result['summary'].get('planned_student_count', result['summary']['student_count']),
                                unexecuted_student_count=result['summary'].get('unexecuted_student_count', 0),
                                message='Controllo annullato: rapporto incompleto disponibile.' if result['summary'].get('incomplete') or result['summary'].get('status') == 'cancelled' else 'Rapporto pronto.')
    except InterruptedError:
        with server.state_lock:
            server.state.update(busy=False, ready=False, result=None, message='Controllo annullato.')
    except Exception as exc:
        with server.state_lock:
            server.state.update(busy=False, ready=False, result=None,
                                error=f'{type(exc).__name__}: {exc}', message='Controllo interrotto.')


def run_validation_job(server, config):
    def progress(done, total, label):
        with server.state_lock:
            server.state.update(done=done, total=total, message=f'{done}/{total} — {label}')
    try:
        result = run_validation(config, progress=progress, cancelled=server.cancel.is_set)
        html = render_validation(result)
        with server.state_lock:
            server.state.update(result={'html': html, 'summary': {'student_count': 0}}, busy=False,
                ready=True, students=0, incomplete=False, planned_student_count=0, unexecuted_student_count=0, message='Verifica linguistica terminata: ' + result['status'])
    except Exception as exc:
        with server.state_lock:
            server.state.update(busy=False, ready=False, result=None,
                error=f'{type(exc).__name__}: {exc}', message='Verifica linguistica interrotta.')


def main():
    parser = argparse.ArgumentParser(description='Controllo locale delle esercitazioni jamovi')
    parser.add_argument('--port', type=int, default=0)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    server = make_server(args.port)
    url = f'http://127.0.0.1:{server.server_address[1]}/?token={server.token}'
    print('Controllo esercitazioni jamovi — locale e offline', flush=True)
    print(url, flush=True)
    print('Utilizza Termina programma nella pagina oppure Ctrl+C in questa finestra.', flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.cancel.set()
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
