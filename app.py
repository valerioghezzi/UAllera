"""Desktop entry point: browser interface served exclusively on loopback."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit, parse_qs
from pathlib import Path
import argparse
import json
import secrets
import threading
import webbrowser
from jamovi_audit.pipeline import audit_zip
from jamovi_audit.local_llm import validate_config, list_models

MAX_UPLOAD = 512 * 1024 * 1024
CSP = "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; img-src data:; form-action 'self'; frame-ancestors 'none'; base-uri 'none'"
PAGE = '''<!doctype html><html lang="it"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Controllo esercitazioni jamovi</title><style>
*{box-sizing:border-box}body{margin:0;background:#f5f5f5;color:#171717;font:17px/1.6 Verdana,Arial,sans-serif}main{max-width:930px;margin:48px auto;padding:0 24px}h1{font-size:30px;line-height:1.3;margin-bottom:12px}h2{font-size:21px}p{max-width:76ch}section{background:#fff;border:1px solid #999;border-radius:7px;padding:26px;margin:24px 0}.tag{font-size:14px;letter-spacing:.05em}label{display:block;font-weight:bold;margin-bottom:12px}input[type=file]{display:block;max-width:100%;font:inherit;margin:12px 0 22px}button,a.button{font:inherit;display:inline-block;cursor:pointer;border:2px solid #171717;border-radius:4px;padding:11px 18px;background:#fff;color:#171717;text-decoration:none;margin:5px 8px 5px 0}button.primary{background:#171717;color:#fff}input[type=text],select{font:inherit;max-width:100%;width:100%;padding:8px;margin-bottom:15px}button:disabled{opacity:.4;cursor:default}button:focus-visible,input:focus-visible,a:focus-visible{outline:3px solid #555;outline-offset:4px}progress{width:100%;height:24px;accent-color:#222}.muted{font-size:14px;color:#444}.error{border-left:5px solid #111;padding:14px;background:#eee}#download[hidden],#result[hidden],#error[hidden]{display:none}.footer{margin-top:32px} @media(max-width:600px){main{margin:24px auto}h1{font-size:25px}section{padding:18px}button{width:100%}}
</style></head><body><main><div class="tag">ELABORAZIONE SUL PC · SENZA SERVIZI ESTERNI</div><h1>Controllo esercitazioni jamovi</h1><p>Esamina i file jamovi e le relazioni Word nelle cartelle degli studenti. Produce un unico rapporto con il dettaglio delle anomalie per ciascuno.</p>
<section><h2>Valutazione linguistica locale (facoltativa)</h2><p class="muted">Avvia prima il tuo motore e carica un modello sul PC. Le risposte proposte saranno da confermare dal tutor. Nessun modello viene scaricato dal programma.</p><label for="provider">Motore</label><select id="provider"><option value="">Disattivata</option><option value="ollama">Ollama</option><option value="lmstudio">LM Studio</option><option value="llamacpp">llama.cpp</option></select><label for="endpoint">Indirizzo locale</label><input id="endpoint" value="http://127.0.0.1:11434" type="text" disabled><label for="model">Nome del modello già installato</label><input id="model" list="models" type="text" disabled><datalist id="models"></datalist><button id="loadmodels" disabled>Elenca modelli locali</button><p id="modelstatus" role="status" class="muted">Valutazione linguistica disattivata. I controlli dei file restano disponibili.</p></section>
<section><label for="zip">Seleziona lo ZIP delle consegne</label><p class="muted">Puoi utilizzare lo ZIP scaricato con tutte le cartelle degli studenti, senza estrarlo a mano. Limite 512 MiB.</p><input type="file" id="zip" accept=".zip,application/zip"><button class="primary" id="start" disabled>Avvia controllo</button><button id="cancel" disabled>Annulla</button><p id="status" role="status" aria-live="polite">In attesa del file.</p><progress id="progress" value="0" max="1" aria-label="Avanzamento del controllo"></progress><p id="error" class="error" role="alert" hidden></p><div id="result" hidden><h2>Rapporto pronto</h2><p id="summary"></p><button class="primary" id="download" hidden>Scarica rapporto unico</button><p class="muted">Il rapporto HTML si apre senza connessione ed è stampabile dal browser. Comprende un indice e una sezione per studente.</p></div></section>
<p class="muted">I documenti originali restano intatti. Le interpretazioni aperte e le evidenze di somiglianza richiedono la valutazione del tutor. Il software non certifica chi abbia scritto un testo né l’assenza di AI.</p><div class="footer"><button id="exit">Termina programma</button></div></main><script>
const token=new URLSearchParams(location.search).get('token');const $=id=>document.getElementById(id);let timer=null;
function message(t){$('status').textContent=t}function fail(t){$('error').textContent=t;$('error').hidden=false}function auth(){return {'X-Session-Token':token}}
async function api(path,options={}){let r=await fetch(path,{...options,headers:{...auth(),...(options.headers||{})}});if(!r.ok){let d=await r.json().catch(()=>({error:'Errore nella richiesta locale.'}));throw Error(d.error||'Errore nella richiesta locale.')}return r}
function engineHeaders(){return {'X-Local-Provider':$('provider').value,'X-Local-Endpoint':$('endpoint').value,'X-Local-Model':$('model').value}}
$('provider').onchange=()=>{let p=$('provider').value;$('endpoint').disabled=!p;$('model').disabled=!p;$('loadmodels').disabled=!p;$('endpoint').value='http://127.0.0.1:'+({ollama:11434,lmstudio:1234,llamacpp:8080}[p]||11434);$('modelstatus').textContent=p?'Avvia il motore e seleziona un modello locale.':'Valutazione linguistica disattivata.'};
$('loadmodels').onclick=async()=>{try{$('loadmodels').disabled=true;let q=new URLSearchParams({provider:$('provider').value,endpoint:$('endpoint').value});let d=await(await api('/api/models?'+q)).json();$('models').replaceChildren(...d.models.map(m=>{let o=document.createElement('option');o.value=m;return o}));if(d.models.length)$('model').value=d.models[0];$('modelstatus').textContent=d.models.length+' modelli disponibili sul motore locale.'}catch(e){$('modelstatus').textContent=e.message}finally{$('loadmodels').disabled=!$('provider').value}};
$('zip').onchange=()=>{$('start').disabled=!$('zip').files.length;$('error').hidden=true;message($('zip').files.length?$('zip').files[0].name:'In attesa del file.')};
async function poll(){try{let s=await(await api('/api/status')).json();$('progress').max=Math.max(s.total,1);$('progress').value=s.done;message(s.message);$('cancel').disabled=!s.busy;if(!s.busy){clearInterval(timer);timer=null;$('start').disabled=!$('zip').files.length;if(s.error)fail(s.error);if(s.ready){$('summary').textContent=s.students+' studenti esaminati. Consulta il rapporto per completezza, errori e verifiche da approfondire.';$('result').hidden=false;$('download').hidden=false;}}}catch(e){clearInterval(timer);timer=null;fail(e.message);$('start').disabled=false}}
$('start').onclick=async()=>{let f=$('zip').files[0];if(!f)return;if(f.size>536870912){fail('ZIP superiore al limite di 512 MiB.');return}$('start').disabled=true;$('cancel').disabled=false;$('result').hidden=true;$('download').hidden=true;$('error').hidden=true;message('Lettura dello ZIP sul PC…');try{await api('/api/scan',{method:'POST',headers:{'Content-Type':'application/zip',...engineHeaders()},body:f});timer=setInterval(poll,500);await poll()}catch(e){fail(e.message);$('start').disabled=false;$('cancel').disabled=true}};
$('cancel').onclick=async()=>{try{await api('/api/cancel',{method:'POST'});message('Annullamento richiesto…')}catch(e){fail(e.message)}};
$('download').onclick=async()=>{try{let r=await api('/report');let blob=await r.blob();let url=URL.createObjectURL(blob);let a=document.createElement('a');a.href=url;a.download='Rapporto_esercitazioni.html';a.click();setTimeout(()=>URL.revokeObjectURL(url),30000)}catch(e){fail(e.message)}};
$('exit').onclick=async()=>{try{await api('/api/exit',{method:'POST'});clearInterval(timer);message('Programma terminato. Puoi chiudere questa scheda.');$('start').disabled=true;$('cancel').disabled=true;$('exit').disabled=true}catch(e){fail(e.message)}};
</script></body></html>'''


class AuditServer(ThreadingHTTPServer):
    daemon_threads = True


def make_server(port=0):
    server = AuditServer(('127.0.0.1', port), Handler)
    server.token = secrets.token_urlsafe(32)
    server.state_lock = threading.Lock()
    server.cancel = threading.Event()
    server.state = dict(busy=False, done=0, total=1, message='In attesa del file.',
                        error='', ready=False, students=0, result=None)
    return server


class Handler(BaseHTTPRequestHandler):
    server_version = 'ControlloLocale'
    protocol_version = 'HTTP/1.1'

    def log_message(self, *args):
        # Never log the session token or student data.
        pass

    def _reply(self, status, body, content_type='application/json; charset=utf-8', download=False):
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
            self.send_header('Content-Disposition', 'attachment; filename="Rapporto_esercitazioni.html"')
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
        elif path == '/api/models':
            query = parse_qs(urlsplit(self.path).query)
            config = {k: query.get(k, [''])[0] for k in ('provider', 'endpoint')}
            try:
                config = validate_config(dict(config, enabled=True))
                self._reply(200, {'models': list_models(config)})
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
            if result is None:
                self._reply(404, {'error': 'Nessun rapporto pronto.'})
            else:
                self._reply(200, result['html'].encode('utf-8'), 'text/html; charset=utf-8', True)
        else:
            self._reply(404, {'error': 'Risorsa non disponibile.'})

    def do_POST(self):
        if not self._authorize():
            return
        path = urlsplit(self.path).path
        if path == '/api/cancel':
            self.server.cancel.set()
            self._reply(200, {'ok': True})
            return
        if path == '/api/exit':
            self.server.cancel.set()
            self._reply(200, {'ok': True})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return
        if path != '/api/scan':
            self._reply(404, {'error': 'Risorsa non disponibile.'})
            return
        try:
            provider = self.headers.get('X-Local-Provider', '')
            linguistic_config = validate_config({
                'enabled': bool(provider), 'provider': provider or 'ollama',
                'endpoint': self.headers.get('X-Local-Endpoint', ''),
                'model': self.headers.get('X-Local-Model', ''),
            })
            if linguistic_config.get('enabled') and not linguistic_config.get('model'):
                raise ValueError('Indica un modello locale prima del controllo.')
        except ValueError as exc:
            self.close_connection = True
            self._reply(400, {'error': str(exc)})
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
                                     error='', ready=False, students=0, result=None)
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


def run_job(server, data, linguistic_config=None):
    def progress(done, total, label):
        with server.state_lock:
            server.state.update(done=done, total=total, message=f'{done}/{total} — {label}')
    try:
        result = audit_zip(data, progress=progress, cancelled=server.cancel.is_set, linguistic_config=linguistic_config)
        with server.state_lock:
            server.state.update(result=result, busy=False, ready=True,
                                students=result['summary']['student_count'], message='Rapporto pronto.')
    except InterruptedError:
        with server.state_lock:
            server.state.update(busy=False, ready=False, result=None, message='Controllo annullato.')
    except Exception as exc:
        with server.state_lock:
            server.state.update(busy=False, ready=False, result=None,
                                error=f'{type(exc).__name__}: {exc}', message='Controllo interrotto.')


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
