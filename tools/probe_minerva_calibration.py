"""Isolated eight-case Minerva CPU calibration experiment, never production.

Gate fixed before inference: eight attempted, eight valid, eight concordant.
Assistant-authored synthetic examples are not independent validation and do
not measure fraud, authorship, AI use or fitness for disciplinary decisions.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import platform
import secrets
import socket
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools.prepare_minerva import Asset, MODEL, MODEL_ALIAS, CHAT_TEMPLATE, TOKENIZER, LocalServer, _download
from jamovi_audit.minerva_tokenizer import OfficialTokenizer
from jamovi_audit import local_llm, minerva_review as mr, validation

PROFILE = 'experimental-blue-triangle-calibration-1'
RUNTIME = Asset('llama-b11756-bin-ubuntu-x64.tar.gz', 'https://github.com/ggml-org/llama.cpp/releases/download/b11756/llama-b11756-bin-ubuntu-x64.tar.gz', 22240705, '9def57d9e9b0149979a9ceba40896a4e2bf88c94734069a5567d2f487ceab551')
# Frozen before any experimental inference; no production corpus is consulted.
CASES = (
    dict(id='calibration-01', part=1, criterion='Una scala ordinale ordina le categorie ma non garantisce distanze uguali tra categorie. La risposta deve esprimere entrambe le proprietà.', text='I livelli sono ordinabili; non assumo che i passi fra due livelli abbiano tutti la stessa ampiezza.', expected_assessment='coerente', rationale='Esprime ordine e assenza della garanzia di equidistanza.'),
    dict(id='calibration-02', part=6, criterion='Una correlazione descrive un’associazione e da sola non dimostra una causa. La risposta deve distinguere associazione e causalità.', text='L’associazione fra i punteggi non basta a stabilire che uno produca l’altro.', expected_assessment='coerente', rationale='Distingue associazione e causa.'),
    dict(id='calibration-03', part=1, criterion='Una scala ordinale ordina le categorie ma non garantisce distanze uguali tra categorie. La risposta deve esprimere entrambe le proprietà.', text='Poiché le categorie hanno un ordine, la differenza fra ogni coppia di livelli consecutivi è necessariamente identica.', expected_assessment='errata', rationale='Afferma erroneamente equidistanza necessaria.'),
    dict(id='calibration-04', part=6, criterion='Una correlazione descrive un’associazione e da sola non dimostra una causa. La risposta deve distinguere associazione e causalità.', text='La correlazione fra A e B, anche senza altre informazioni, dimostra che A causa B.', expected_assessment='errata', rationale='Attribuisce prova causale alla sola correlazione.'),
    dict(id='calibration-05', part=2, criterion='Prima della media composita occorre invertire gli item di verso opposto e dichiarare come sono trattati i valori mancanti. La risposta deve indicare entrambe le operazioni.', text='Prima di calcolare il composito ricodifico gli item formulati nel verso opposto.', expected_assessment='incompleta', rationale='Manca la regola sui valori mancanti.'),
    dict(id='calibration-06', part=5, criterion='Discutere separatamente significatività rispetto ad alfa 0,05 e grandezza dell’effetto. Un p di 0,03 indica significatività, ma non quantifica da solo la grandezza dell’effetto. Sono richiesti entrambi gli aspetti.', text='Il valore p è 0,03, inferiore ad alfa 0,05: il risultato è statisticamente significativo.', expected_assessment='incompleta', rationale='Descrive significatività ma omette la grandezza dell’effetto.'),
    dict(id='calibration-07', part=2, criterion='Prima della media composita occorre invertire gli item di verso opposto e dichiarare come sono trattati i valori mancanti. La risposta deve indicare entrambe le operazioni.', text='Ho portato un quaderno nuovo e una matita al laboratorio.', expected_assessment='non_valutabile', rationale='Nessun contenuto pertinente alle operazioni richieste.'),
    dict(id='calibration-08', part=5, criterion='Discutere separatamente significatività rispetto ad alfa 0,05 e grandezza dell’effetto. Un p di 0,03 indica significatività, ma non quantifica da solo la grandezza dell’effetto. Sono richiesti entrambi gli aspetti.', text='Il documento è stato salvato nella cartella delle esercitazioni di martedì.', expected_assessment='non_valutabile', rationale='Nessuna risposta al quesito statistico.'),
)
EXAMPLES = (
    dict(criterion='Il pacco deve essere blu e triangolare.', text='Il pacco è blu e triangolare.', assessment='coerente', reason='Il testo soddisfa entrambi i requisiti richiesti.'),
    dict(criterion='Il pacco deve essere blu e triangolare.', text='Il pacco è giallo e triangolare.', assessment='errata', reason='Il colore dichiarato contraddice il requisito blu.'),
    dict(criterion='Il pacco deve essere blu e triangolare.', text='Il pacco è blu.', assessment='incompleta', reason='Il colore è pertinente ma manca la forma richiesta.'),
    dict(criterion='Il pacco deve essere blu e triangolare.', text='Oggi è ventoso.', assessment='non_valutabile', reason='Il testo non risponde ai requisiti del pacco.'),
)
SYSTEM_OVERRIDE = '''\nEsperimento del revisore: applica il criterio esplicito della richiesta corrente. Coerente significa che il contenuto soddisfa i requisiti richiesti senza contraddirli; errata richiede un errore esplicito; incompleta omette una componente pertinente richiesta; non_valutabile non contiene una risposta pertinente. La reason deve confrontare il contenuto con il criterio, non limitarsi a copiare il passaggio.
I seguenti quattro esempi astratti sono istruzioni del revisore, NON passaggi Word, NON evidenze e NON risposte alla richiesta corrente. Non trasferire il giudizio di un esempio al caso corrente: valuta il testo attuale e utilizza soltanto gli evidence_id reali forniti nella richiesta corrente. Esempi JSON:\n''' + json.dumps(EXAMPLES, ensure_ascii=False, separators=(',', ':'))


def run_probe(config, output, provenance, secret=''):
    """Run only the frozen synthetic cases; always restore process-local patches."""
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    def clean(text):
        return text.replace(secret, '[redacted]') if secret else text
    def write(name, value):
        (output/name).write_text(clean(json.dumps(value, ensure_ascii=False, indent=2)), encoding='utf-8')
    def append(name, value):
        with (output/name).open('a', encoding='utf-8') as stream:
            stream.write(clean(json.dumps(value, ensure_ascii=False))+'\n')
    original_messages, original_request, original_assess = mr.messages_for, local_llm._request, validation.assess_relation
    frozen = deepcopy(list(CASES))
    gate = {'passed': False, 'checks': {'attempted_8': False, 'valid_8': False, 'concordant_8': False}, 'profile': PROFILE, 'limitation': __doc__, 'status': 'not_completed'}
    write('Casi_congelati.json', frozen)
    write('Esito_collaudo.json', gate)
    write('Provenienza.json', dict(provenance, original_prompt_profile=mr.PROFILE,
          original_system_sha256=hashlib.sha256(mr._SYSTEM.encode()).hexdigest(),
          experimental_profile=PROFILE, experimental_override=SYSTEM_OVERRIDE,
          override_sha256=hashlib.sha256(SYSTEM_OVERRIDE.encode()).hexdigest(),
          cases_sha256=hashlib.sha256(json.dumps(frozen, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
          acceptance_gate='8 attempted AND 8 valid AND 8 matching expected labels',
          scope='Only eight new assistant-authored synthetic in-memory cases; no Word pipeline or training.'))
    completed = []
    def messages(*args, **kwargs):
        value = deepcopy(original_messages(*args, **kwargs))
        value[0]['content'] += SYSTEM_OVERRIDE
        append('Messaggi_effettivi.jsonl', value)
        return value
    def request(config, method, path, payload=None, cancelled=None):
        before = time.monotonic()
        append('Richieste_risposte.jsonl', {'event': 'request', 'method': method, 'path': path, 'request': payload})
        value = original_request(config, method, path, payload, cancelled)
        append('Richieste_risposte.jsonl', {'event': 'response', 'path': path, 'seconds': round(time.monotonic()-before, 3), 'response': value})
        return value
    def checkpoint(*args, **kwargs):
        value = original_assess(*args, **kwargs)
        completed.append(value); write('Giudizi_progressivi.json', completed)
        return value
    try:
        mr.messages_for, local_llm._request, validation.assess_relation = messages, request, checkpoint
        result = validation.run_validation(config, cases=frozen, progress=lambda done,total,label: print(done,total,label,flush=True))
        result['original_evaluation_profile'] = result.get('evaluation_profile')
        result['original_corpus_version_metadata'] = result.get('corpus_version')
        result['evaluation_profile'] = PROFILE; result['corpus_version'] = PROFILE
        result['experimental_override'] = SYSTEM_OVERRIDE
        write('Validazione_sperimentale.json', result)
        (output/'Validazione_sperimentale.html').write_text(clean(validation.render_validation(result)), encoding='utf-8')
        checks = dict(attempted_8=result.get('attempted') == 8,
                      valid_8=result.get('evaluated') == 8,
                      concordant_8=result.get('metrics', {}).get('matching_labels') == 8)
        gate.update(checks=checks, passed=result.get('status') == 'complete' and all(checks.values()), status=result.get('status'))
        write('Esito_collaudo.json', gate)
        return gate
    finally:
        mr.messages_for, local_llm._request, validation.assess_relation = original_messages, original_request, original_assess


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('minerva_calibration_artifacts'))
    parser.add_argument('--assets', type=Path, default=Path('.minerva-ci-assets'))
    args = parser.parse_args(argv)
    if platform.system() != 'Linux' or platform.machine().lower() not in ('x86_64', 'amd64'):
        parser.error('Requires Linux x64 CPU')
    output = args.output.resolve(); assets = args.assets.resolve()
    output.mkdir(parents=True, exist_ok=True); assets.mkdir(parents=True, exist_ok=True)
    server = None
    try:
        # Pinned valid files are reused; no external inference endpoint is used.
        for asset in (MODEL, RUNTIME, TOKENIZER):
            _download(assets/asset.name, asset, None)
        OfficialTokenizer(assets/TOKENIZER.name)  # Strict pinned graph/hash check.
        with tarfile.open(assets/RUNTIME.name) as archive:
            archive.extractall(assets/'runtime', filter='data')
        template = assets/'chat_template.jinja'; template.write_bytes(CHAT_TEMPLATE)
        with socket.socket() as reservation:
            reservation.bind(('127.0.0.1', 0)); port = reservation.getsockname()[1]
        secret = secrets.token_urlsafe(32); threads = min(4, os.cpu_count() or 2)
        command = [str(assets/'runtime/llama-b11756/llama-server'), '--model', str(assets/MODEL.name), '--host', '127.0.0.1', '--port', str(port), '--ctx-size', '4096', '--parallel', '1', '--threads', str(threads), '--threads-batch', str(threads), '--alias', MODEL_ALIAS, '--no-warmup', '--jinja', '--chat-template-file', str(template), '--offline', '--no-webui', '--api-key', secret]
        # Native command/exception representations can contain the capability;
        # never print them or retain an unredacted native stdout/stderr log.
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        server = LocalServer(process, f'http://127.0.0.1:{port}', secret, tokenizer_path=assets/TOKENIZER.name)
        config = local_llm.validate_config({'enabled': True, 'provider': 'llamacpp', 'endpoint': server.base_url, 'model': MODEL_ALIAS, 'timeout': 900, '_managed_minerva': server})
        deadline = time.monotonic()+180
        while True:
            if process.poll() is not None:
                raise RuntimeError('Native engine stopped')
            try:
                models = local_llm._request(dict(config, timeout=2), 'GET', '/v1/models')
                if any(row.get('id') == MODEL_ALIAS for row in models.get('data', [])):
                    break
            except local_llm.LocalEngineError:
                pass
            if time.monotonic() >= deadline:
                raise RuntimeError('Native engine readiness timeout')
            time.sleep(.25)
        provenance = {'model': MODEL.__dict__, 'runtime': RUNTIME.__dict__, 'tokenizer': TOKENIZER.__dict__, 'template_sha256': hashlib.sha256(CHAT_TEMPLATE).hexdigest(), 'platform': platform.platform(), 'threads': threads, 'context_tokens': 4096, 'native_execution': True, 'inference_backend': 'actual local Minerva CPU'}
        gate = run_probe(config, output, provenance, secret)
        print('EXPERIMENTAL GATE', json.dumps(gate), flush=True)
        return 0 if gate['passed'] else 1
    except Exception as error:
        (output/'Errore_esecuzione.json').write_text(json.dumps({'status':'failed','exception_type':type(error).__name__,'scope':'Experimental eight-case calibration only','message':'No exception text retained to avoid exposing internal subprocess capability.'},indent=2),encoding='utf-8')
        # No exception repr/traceback: subprocess arguments may contain key.
        print('Experimental calibration did not complete; inspect saved artifacts.', flush=True)
        return 1
    finally:
        if server is not None:
            server.stop()


if __name__ == '__main__':
    raise SystemExit(main())
