"""Run the exact same offline checks without opening the interface."""
import argparse
import json
from pathlib import Path
import os
import tempfile
from jamovi_audit.pipeline import audit_zip
from jamovi_audit.diagnostics import forensic_snapshot


def main():
    parser = argparse.ArgumentParser(description='Controllo locale delle esercitazioni jamovi')
    parser.add_argument('zip', type=Path, help='ZIP delle cartelle degli studenti')
    parser.add_argument('-o', '--output', type=Path, default=Path('Rapporto_esercitazioni.html'))
    parser.add_argument('--local-engine', choices=['ollama', 'lmstudio', 'llamacpp'])
    parser.add_argument('--model', default='')
    parser.add_argument('--endpoint', default='')
    parser.add_argument('--forensic',type=Path,help='Esporta anche diagnostica locale JSON con evidenze e copertura')
    parser.add_argument('--no-cache',action='store_true',help='Disabilita il riutilizzo del parsing per confronto diagnostico')
    args = parser.parse_args()
    if args.output.suffix.lower() != '.html':
        parser.error('Il rapporto deve avere estensione .html')
    if args.output.resolve()==args.zip.resolve():parser.error('Il rapporto non può sovrascrivere lo ZIP originale.')
    if args.forensic and (args.forensic.resolve() in (args.zip.resolve(),args.output.resolve()) or args.forensic.suffix.lower()!='.json'):
        parser.error('La diagnostica deve essere un JSON distinto da sorgente e rapporto HTML.')
    if args.zip.stat().st_size > 512 * 1024 * 1024:
        parser.error('ZIP superiore al limite di 512 MiB')
    if args.local_engine and not args.model:
        parser.error('--model è richiesto con --local-engine')
    result = audit_zip(args.zip.read_bytes(), linguistic_config={
        'enabled': bool(args.local_engine), 'provider': args.local_engine or 'ollama',
        'model': args.model, 'endpoint': args.endpoint},use_cache=not args.no_cache)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.rapporto-', suffix='.tmp', dir=args.output.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(result['html'])
        os.replace(temp, args.output)
    finally:
        if Path(temp).exists():
            Path(temp).unlink()
    if args.forensic:
        args.forensic.parent.mkdir(parents=True,exist_ok=True)
        fd, temporary=tempfile.mkstemp(prefix='.diagnostica-',suffix='.tmp',dir=args.forensic.parent)
        try:
            with os.fdopen(fd,'w',encoding='utf-8') as output:
                json.dump(forensic_snapshot(result),output,ensure_ascii=False,indent=2)
            os.replace(temporary,args.forensic)
        finally:
            Path(temporary).unlink(missing_ok=True)
    print(f"Rapporto creato: {args.output.resolve()} — {result['summary']['student_count']} studenti")


if __name__ == '__main__':
    main()
