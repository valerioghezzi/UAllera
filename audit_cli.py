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
    args = parser.parse_args()
    if args.output.suffix.lower() != '.html':
        parser.error('Il rapporto deve avere estensione .html')
    if args.zip.stat().st_size > 512 * 1024 * 1024:
        parser.error('ZIP superiore al limite di 512 MiB')
    result = audit_zip(args.zip.read_bytes())
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
