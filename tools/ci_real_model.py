"""Development-only real CPU inference on public synthetic fixtures.

Explicitly downloads pinned public test assets. Never called by the offline app.
Requires Linux x64; no API key or external inference service is involved.
"""
from pathlib import Path
import argparse
import hashlib
import http.client
import io
import json
import platform
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from jamovi_audit import validation, local_llm
from jamovi_audit.pipeline import audit_zip

MODEL_URL = 'https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf'
MODEL_SHA = '6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e'
RUNTIME_URL = 'https://github.com/ggml-org/llama.cpp/releases/download/b11756/llama-b11756-bin-ubuntu-x64.tar.gz'
RUNTIME_SHA = '9def57d9e9b0149979a9ceba40896a4e2bf88c94734069a5567d2f487ceab551'


def download(url, path, expected):
    if path.exists() and hashlib.file_digest(path.open('rb'), 'sha256').hexdigest() == expected:
        return
    digest = hashlib.sha256()
    temporary = path.with_suffix('.part')
    with urllib.request.urlopen(url, timeout=60) as response, temporary.open('wb') as output:
        while chunk := response.read(1024 * 1024):
            digest.update(chunk)
            output.write(chunk)
    if digest.hexdigest() != expected:
        temporary.unlink(missing_ok=True)
        raise RuntimeError('SHA-256 mismatch: public test asset rejected')
    temporary.replace(path)


def archive(entries):
    target = io.BytesIO()
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as output:
        for name, data in entries.items():
            output.writestr(name, data)
    return target.getvalue()


def synthetic_delivery():
    # Deliberately empty OMV: the report must state that saved outputs are missing.
    omv = archive({'meta': 'Manifest-Version: 1.0\nData-Archive-Version: 1.0.2\njamovi-Archive-Version: 11.0\nCreated-By: jamovi 2.7.4\n',
                   'metadata.json': json.dumps({'dataSet': {'rowCount': 0, 'columnCount': 0, 'fields': []}}),
                   'data.bin': b'', 'xdata.json': '{}'})
    common = [
        'Parte 1. Le categorie degli item sono ordinate; distanze uguali non sono garantite.',
        'Parte 2. Ricodifico gli item reverse prima di ottenere la media; i dati mancanti restano distinti dal livello minimo.',
        'Parte 3. La media da sola non descrive la dispersione e non dimostra un effetto soffitto.',
        'Parte 4. I punteggi T interni collocano la persona nel campione disponibile, senza creare norme esterne.',
        'Parte 5. Un valore p piccolo non implica che la differenza sia grande; considero anche d e intervallo di confidenza.',
        '',
        'Parte 7. Con deviazione standard 6 e omega 0,75, SEM = 6 per radice di 0,25 = 3.',
    ]
    entries = {}
    for student, association in [
        ('Persona_01', 'Parte 6. Una correlazione significativa non dimostra causalità: un terzo fattore può influenzare entrambe le variabili.'),
        ('Persona_02', 'Parte 6. Una correlazione significativa dimostra necessariamente che X causa Y.'),
    ]:
        paragraphs = common.copy()
        paragraphs[5] = association
        document = '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>' + ''.join(
            '<w:p><w:r><w:t xml:space="preserve">' + escape(text) + '</w:t></w:r></w:p>' for text in paragraphs) + '</w:body></w:document>'
        word = archive({
            '[Content_Types].xml': '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>',
            '_rels/.rels': '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>',
            'word/document.xml': document,
        })
        entries[f'Classe_demo/{student}/relazione.docx'] = word
        entries[f'Classe_demo/{student}/analisi.omv'] = omv
    return archive(entries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('validation_artifacts'))
    parser.add_argument('--limit', type=int, default=70)
    parser.add_argument('--assets', type=Path, default=Path('.real-model-assets'))
    args = parser.parse_args()
    if platform.system() != 'Linux' or platform.machine().lower() not in ('x86_64', 'amd64'):
        parser.error('Development-only test requires Linux x64')
    if not 1 <= args.limit <= 70:
        parser.error('--limit must be 1..70')
    args.output.mkdir(parents=True, exist_ok=True)
    args.assets.mkdir(parents=True, exist_ok=True)
    model = args.assets / 'qwen2.5-1.5b-instruct-q4_k_m.gguf'
    runtime = args.assets / 'llama-b11756-bin-ubuntu-x64.tar.gz'
    print('Preparing checksum-pinned public model and CPU runtime', flush=True)
    download(MODEL_URL, model, MODEL_SHA)
    download(RUNTIME_URL, runtime, RUNTIME_SHA)
    with tarfile.open(runtime) as bundle:
        bundle.extractall(args.assets / 'runtime', filter='data')
    executable = args.assets / 'runtime/llama-b11756/llama-server'
    command = [str(executable.resolve()), '-m', str(model.resolve()), '--host', '127.0.0.1', '--port', '8199',
               '-c', '16384', '-t', '2', '-tb', '4', '--parallel', '1', '--alias', 'qwen2.5-1.5b-q4-local', '--no-warmup']
    provenance = {'model_url': MODEL_URL, 'model_sha256': MODEL_SHA, 'runtime_url': RUNTIME_URL,
                  'runtime_sha256': RUNTIME_SHA, 'platform': platform.platform(), 'threads': 2,
                  'batch_threads': 4, 'context_tokens': 16384, 'command': command,
                  'scope': 'Public synthetic data only; inference on a literal loopback CPU server.'}
    (args.output / 'Provenienza.json').write_text(json.dumps(provenance, indent=2))
    config = {'enabled': True, 'provider': 'llamacpp', 'endpoint': 'http://127.0.0.1:8199',
              'model': 'qwen2.5-1.5b-q4-local', 'timeout': 900}
    with (args.output / 'llama_server.log').open('w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 120
            while True:
                if process.poll() is not None:
                    raise RuntimeError('Real model server exited during startup')
                try:
                    connection = http.client.HTTPConnection('127.0.0.1', 8199, timeout=2)
                    connection.request('GET', '/health')
                    response = connection.getresponse()
                    response.read()
                    status = response.status
                    connection.close()
                    if status == 200:
                        break
                except OSError:
                    pass
                if time.monotonic() > deadline:
                    raise RuntimeError('Real server readiness timeout')
                time.sleep(.25)
            original = local_llm._request
            def record(config, method, path, payload=None, cancelled=None):
                before = time.monotonic()
                response = original(config, method, path, payload, cancelled)
                if path.endswith('completions'):
                    row = {'seconds': round(time.monotonic() - before, 3), 'request': payload, 'response': response}
                    with (args.output / 'Richieste_risposte.jsonl').open('a') as audit:
                        audit.write(json.dumps(row, ensure_ascii=False) + '\n')
                return response
            local_llm._request = record
            completed = []
            original_assess = validation.assess_relation
            def checkpoint(*a, **kw):
                response = original_assess(*a, **kw)
                completed.append(response)
                (args.output / 'Giudizi_progressivi.json').write_text(json.dumps(completed, ensure_ascii=False, indent=2))
                print('REAL RESPONSE', len(completed), response['status'],
                      [(f['part'], f.get('proposal')) for f in response['findings']], flush=True)
                return response
            validation.assess_relation = checkpoint
            from jamovi_audit.validation_cases import VALIDATION_CASES
            result = validation.run_validation(config, cases=VALIDATION_CASES[:args.limit],
                progress=lambda done,total,label: print(f'{done}/{total}: {label}', flush=True))
            result['runtime'] = provenance
            (args.output / 'Validazione_reale.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
            (args.output / 'Validazione_reale.html').write_text(validation.render_validation(result))
            print('REAL MODEL METRICS', json.dumps({k: result[k] for k in ('status','planned','attempted','evaluated','seconds','metrics')}), flush=True)
            if result['status'] != 'complete' or result['attempted'] != args.limit:
                raise RuntimeError('Incomplete real-model benchmark: inspect report')
            validation.assess_relation = original_assess
            data = synthetic_delivery()
            (args.output / 'Consegne_demo_linguistiche.zip').write_bytes(data)
            report = audit_zip(data, linguistic_config=config,
                progress=lambda done,total,label: print('PIPELINE',done,total,label,flush=True))
            (args.output / 'Rapporto_demo_linguistico.html').write_text(report['html'])
            saved = {k: v for k, v in report.items() if k != 'html'}
            (args.output / 'Rapporto_demo_linguistico.json').write_text(json.dumps(saved, ensure_ascii=False, indent=2, default=str))
            assert len(report['reports']) == 2
            print('FULL DOCUMENT PIPELINE', [(r['label'], r['linguistic']['status']) for r in report['reports']], flush=True)
            for row in report['reports']:
                print('CITABLE FULL DOCUMENT PROPOSALS', row['label'],
                      sum(bool(f.get('proposal')) for f in row['linguistic']['findings']), flush=True)
            assert all(r['linguistic']['status'] == 'complete' for r in report['reports']), 'Full-document model requests did not complete; inspect saved report'
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == '__main__':
    main()
