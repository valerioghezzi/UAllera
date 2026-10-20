"""Actual CPU Minerva evaluation on public synthetic data; development-only."""
from pathlib import Path
import argparse
import hashlib
import http.client
import json
import os
import platform
import secrets
import socket
import subprocess
import sys
import tarfile
import time

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from tools.prepare_minerva import Asset, MODEL, MODEL_ALIAS, CHAT_TEMPLATE, LocalServer, _download
from tools.ci_real_model import synthetic_delivery, verify_pipeline
from jamovi_audit import local_llm, validation
from jamovi_audit.pipeline import audit_zip
from jamovi_audit.validation_cases import VALIDATION_CASES

RUNTIME=Asset('llama-b11756-bin-ubuntu-x64.tar.gz','https://github.com/ggml-org/llama.cpp/releases/download/b11756/llama-b11756-bin-ubuntu-x64.tar.gz',22240705,'9def57d9e9b0149979a9ceba40896a4e2bf88c94734069a5567d2f487ceab551')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('minerva_artifacts'))
    parser.add_argument('--assets',type=Path,default=Path('.minerva-ci-assets'))
    parser.add_argument('--limit',type=int,default=70)
    args=parser.parse_args()
    if not 1<=args.limit<=70:parser.error('--limit must be 1..70')
    if platform.system()!='Linux' or platform.machine().lower() not in ('x86_64','amd64'):parser.error('Requires Linux x64')
    out=args.output.resolve();assets=args.assets.resolve();out.mkdir(parents=True,exist_ok=True);assets.mkdir(parents=True,exist_ok=True)
    for asset in [MODEL,RUNTIME]:
        print('Preparing pinned public asset',asset.name,flush=True)
        _download(assets/asset.name,asset,None)
    with tarfile.open(assets/RUNTIME.name) as archive:archive.extractall(assets/'runtime',filter='data')
    template=assets/'chat_template.jinja';template.write_bytes(CHAT_TEMPLATE)
    threads=min(4,os.cpu_count() or 2)
    with socket.socket() as reservation:reservation.bind(('127.0.0.1',0));port=reservation.getsockname()[1]
    key=secrets.token_urlsafe(32)
    command=[str(assets/'runtime/llama-b11756/llama-server'),'--model',str(assets/MODEL.name),'--host','127.0.0.1','--port',str(port),'--ctx-size','4096','--parallel','1','--threads',str(threads),'--threads-batch',str(threads),'--alias',MODEL_ALIAS,'--no-warmup','--jinja','--chat-template-file',str(template),'--offline','--no-webui','--api-key',key]
    provenance={'model':MODEL.__dict__,'runtime':RUNTIME.__dict__,'template_sha256':hashlib.sha256(CHAT_TEMPLATE).hexdigest(),'platform':platform.platform(),'threads':threads,'context_tokens':4096,'scope':'Public synthetic data only; literal loopback CPU inference. No student data.'}
    (out/'Provenienza.json').write_text(json.dumps(provenance,indent=2))
    original_request=local_llm._request;original_assess=validation.assess_relation
    logpath=out/'llama_server.log'
    with logpath.open('w') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        server=LocalServer(process,f'http://127.0.0.1:{port}',key)
        config=local_llm.validate_config({'enabled':True,'provider':'llamacpp','endpoint':server.base_url,'model':MODEL_ALIAS,'timeout':900,'_managed_minerva':server})
        try:
            deadline=time.monotonic()+180
            while True:
                if process.poll() is not None:raise RuntimeError('Minerva stopped during loading')
                try:
                    response=original_request(dict(config,timeout=2),'GET','/v1/models')
                    if any(row.get('id')==MODEL_ALIAS for row in response.get('data',[])):break
                except local_llm.LocalEngineError:pass
                if time.monotonic()>deadline:raise RuntimeError('Minerva readiness timeout')
                time.sleep(.25)
            def record(config,method,path,payload=None,cancelled=None):
                before=time.monotonic();response=original_request(config,method,path,payload,cancelled)
                if path in ('/completion','/apply-template'):
                    with (out/'Richieste_risposte.jsonl').open('a') as audit:audit.write(json.dumps({'path':path,'seconds':round(time.monotonic()-before,3),'request':payload,'response':response},ensure_ascii=False)+'\n')
                return response
            local_llm._request=record
            completed=[]
            def checkpoint(*a,**kw):
                result=original_assess(*a,**kw);completed.append(result)
                (out/'Giudizi_progressivi.json').write_text(json.dumps(completed,ensure_ascii=False,indent=2))
                print('REAL RESPONSE',len(completed),result['status'],[(f['part'],f.get('proposal')) for f in result['findings']],flush=True)
                return result
            validation.assess_relation=checkpoint
            result=validation.run_validation(config,cases=VALIDATION_CASES[:args.limit],progress=lambda done,total,label:print(done,total,label,flush=True))
            result['runtime']=provenance
            (out/'Validazione_reale.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));(out/'Validazione_reale.html').write_text(validation.render_validation(result))
            validation.assess_relation=original_assess
            data=synthetic_delivery();(out/'Consegne_demo_linguistiche.zip').write_bytes(data)
            report=audit_zip(data,linguistic_config=config,progress=lambda done,total,label:print('PIPELINE',done,total,label,flush=True))
            (out/'Rapporto_demo_linguistico.html').write_text(report['html']);(out/'Rapporto_demo_linguistico.json').write_text(json.dumps({k:v for k,v in report.items() if k!='html'},ensure_ascii=False,indent=2))
            checks={'all_cases_attempted':result['attempted']==args.limit,'all_responses_interpretable':result['evaluated']==args.limit,'two_word_causality':False}
            try:verify_pipeline(report);checks['two_word_causality']=True
            except AssertionError:pass
            # Engineering test thresholds fixed before this experiment; they do
            # not establish external validity or fitness for disciplinary decisions.
            if args.limit==70:
                metrics=result['metrics'];checks.update(agreement_at_least_85_percent=(metrics['agreement_attempted'] or 0)>=.85,error_recall_at_least_85_percent=(metrics['error_recall'] or 0)>=.85,false_alarm_at_most_5_percent=metrics['false_alarm_rate'] is not None and metrics['false_alarm_rate']<=.05)
            gate={'passed':all(checks.values()),'checks':checks,'limitation':'Synthetic assistant-authored test only; not an independent validation or cheating detector.'}
            (out/'Esito_collaudo.json').write_text(json.dumps(gate,indent=2))
            print('METRICS',json.dumps(result['metrics']),flush=True);print('GATE',json.dumps(gate),flush=True)
            if not gate['passed']:raise RuntimeError('Minerva did not pass this engineering evaluation; inspect actual reports')
        finally:
            local_llm._request=original_request;validation.assess_relation=original_assess
            try:server.stop()
            finally:
                log.flush();log.close()
                if logpath.exists():logpath.write_text(logpath.read_text(errors='replace').replace(key,'[internal capability removed]'))
    # Do not retain the memory-only loopback capability in diagnostic artifacts.
    if logpath.exists():logpath.write_text(logpath.read_text(errors='replace').replace(key,'[internal capability removed]'))

if __name__=='__main__':main()
