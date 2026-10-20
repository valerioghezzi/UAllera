"""Run the exact same offline checks without opening the interface."""
import argparse
from pathlib import Path
import os
import tempfile
from jamovi_audit.pipeline import audit_zip


def main():
    parser = argparse.ArgumentParser(description='Controllo locale delle esercitazioni jamovi')
    parser.add_argument('zip', type=Path, help='ZIP delle cartelle degli studenti')
    parser.add_argument('-o', '--output', type=Path, default=Path('Rapporto_esercitazioni.html'))
    parser.add_argument('--local-engine', choices=['ollama', 'lmstudio', 'llamacpp'])
    parser.add_argument('--model', default='')
    parser.add_argument('--endpoint', default='')
    args = parser.parse_args()
    if args.output.suffix.lower() != '.html':
        parser.error('Il rapporto deve avere estensione .html')
    if args.zip.stat().st_size > 512 * 1024 * 1024:
        parser.error('ZIP superiore al limite di 512 MiB')
    if args.local_engine and not args.model:
        parser.error('--model è richiesto con --local-engine')
    result = audit_zip(args.zip.read_bytes(), linguistic_config={
        'enabled': bool(args.local_engine), 'provider': args.local_engine or 'ollama',
        'model': args.model, 'endpoint': args.endpoint})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.rapporto-', suffix='.tmp', dir=args.output.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(result['html'])
        os.replace(temp, args.output)
    finally:
        if Path(temp).exists():
            Path(temp).unlink()
    print(f"Rapporto creato: {args.output.resolve()} — {result['summary']['student_count']} studenti")


if __name__ == '__main__':
    main()
