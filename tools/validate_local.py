"""Run the held-out synthetic suite on an already running local engine."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from jamovi_audit.validation import run_validation,render_validation
from jamovi_audit.validation_cases import VALIDATION_CASES


def main():
    parser=argparse.ArgumentParser(description='Verifica un modello locale su casi sintetici separati dal prompt')
    parser.add_argument('--provider',choices=['ollama','lmstudio','llamacpp'],required=True)
    parser.add_argument('--model',required=True)
    parser.add_argument('--endpoint',default='')
    parser.add_argument('--timeout',type=float,default=300)
    parser.add_argument('--output',type=Path,default=Path('Validazione_modello.html'))
    parser.add_argument('--limit',type=int,default=len(VALIDATION_CASES),help='Numero di casi iniziali; il rapporto dichiara la selezione')
    args=parser.parse_args()
    if not 1<=args.limit<=len(VALIDATION_CASES):parser.error('--limit non valido')
    if args.output.suffix.lower()!='.html':parser.error('--output deve essere HTML')
    def progress(done,total,label):print(f'{done}/{total}: {label}',flush=True)
    result=run_validation({'enabled':True,'provider':args.provider,'model':args.model,'endpoint':args.endpoint,'timeout':args.timeout},
                          cases=VALIDATION_CASES[:args.limit],progress=progress)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(render_validation(result),encoding='utf-8')
    args.output.with_suffix('.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Rapporto:',args.output.resolve(),flush=True)
    if result['status']=='unavailable':raise SystemExit(2)


if __name__=='__main__':main()
