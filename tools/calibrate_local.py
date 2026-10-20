"""Optional empirical calibration of an already installed local model."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jamovi_audit.local_llm import CALIBRATION_CASES, RUBRIC_VERSION, assess_relation, validate_config


def run(config, cases=None, progress=None):
    normalized=validate_config(config)
    if not normalized['enabled'] or not normalized['model']:
        raise ValueError('Seleziona un motore e un modello locale installato.')
    results=[]
    for case in cases if cases is not None else CALIBRATION_CASES:
        document={'paragraphs':[{'index':0,'text':case['text']}]}
        # Expected labels are intentionally not provided to the model.
        reference={'linguistic_rubric':f"Valuta soltanto il significato espresso nella parte {case['part']}. Non richiedere le altre parti o numeri non presenti. Una sola domanda o un comando non è una risposta."}
        review=assess_relation(document,{},reference,normalized)
        proposals=[f.get('proposal') for f in review['findings'] if f['part']==case['part'] and f.get('proposal')]
        observed=proposals[0] if len(set(proposals))==1 else None
        results.append({**case,'engine_status':review['status'],'observed':observed,
                        'matches_expected':observed==case['expected_assessment'],
                        'findings':[f for f in review['findings'] if f['part']==case['part']],
                        'limits':review['limits']})
        if progress:progress(case['id'],results[-1])
    valid=sum(r['observed'] is not None for r in results)
    return {'rubric_version':RUBRIC_VERSION,'provider':normalized['provider'],'model':normalized['model'],
            'case_count':len(results),'citable_results':valid,
            'matching_labels':sum(r['matches_expected'] for r in results),'cases':results,
            'limits':['Corpus didattico ridotto e già esposto negli esempi: non è un benchmark indipendente.',
                      'Non misura sensibilità al cheating né accuratezza sulle consegne reali.',
                      'Etichette e proposte richiedono verifica del docente; nessun fine-tuning viene eseguito.']}


def main():
    parser=argparse.ArgumentParser(description='Calibra un modello locale sui 35 casi didattici italiani')
    parser.add_argument('--provider',choices=['ollama','lmstudio','llamacpp'],required=True)
    parser.add_argument('--model',required=True)
    parser.add_argument('--endpoint',default='')
    parser.add_argument('--output',type=Path,default=Path('Calibrazione_linguistica.json'))
    args=parser.parse_args()
    def progress(case,result):
        print(case, 'corrispondente' if result['matches_expected'] else 'da rivedere', flush=True)
    result=run({'enabled':True,'provider':args.provider,'model':args.model,'endpoint':args.endpoint},progress=progress)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Calibrazione salvata:',args.output.resolve())


if __name__=='__main__':
    main()
