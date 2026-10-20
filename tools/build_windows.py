"""Build offline Windows distributions with checksum-pinned official CPython."""
from pathlib import Path
import argparse
import hashlib
import io
import json
import shutil
import struct
import urllib.request
import zipfile

VERSION = '3.14.8'
RUNTIMES = {
    'x64': ('amd64', 'a93abe456ab01bd96d7a085b3cdb6566b3063f4241360d114142fbdb07f0a310', 0x8664),
    'ARM64': ('arm64', '155be84ccb57c6331cf0e39001c78a1dfac3be62f403f3ff5f2e29b80dda7ebe', 0xAA64),
}


def verify_runtime(data: bytes, arch: str):
    if arch not in RUNTIMES:
        raise ValueError('Architettura non supportata: ' + arch)
    expected = RUNTIMES[arch][1]
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError('Checksum SHA-256 del runtime diverso dalla fonte ufficiale.')
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        executable = z.read('python.exe')
    pe_offset = struct.unpack_from('<I', executable, 0x3C)[0]
    if executable[pe_offset:pe_offset+4] != b'PE\0\0':
        raise ValueError('Eseguibile Python non valido.')
    machine = struct.unpack_from('<H', executable, pe_offset+4)[0]
    if machine != RUNTIMES[arch][2]:
        raise ValueError('Architettura del runtime non corrispondente.')


def build(arch, runtime_data, destination):
    verify_runtime(runtime_data, arch)
    source = Path(__file__).resolve().parent.parent
    destination = Path(destination).resolve()
    folder = destination / ('UAllera_Windows_' + arch)
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    for filename in ['app.py', 'audit_cli.py', 'README.md']:
        shutil.copy2(source / filename, folder / filename)
    for name in ['jamovi_audit', 'tests', 'tools', 'reference']:
        if (source / name).is_dir():
            shutil.copytree(source / name, folder / name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    runtime = folder / 'runtime'
    runtime.mkdir()
    with zipfile.ZipFile(io.BytesIO(runtime_data)) as z:
        # Archive is pinned byte-for-byte to the published official runtime.
        z.extractall(runtime)
    for p in runtime.glob('python*._pth'):
        p.write_text('python314.zip\n.\n..\n', encoding='utf-8')
    (folder / 'Avvia_UAllera.cmd').write_bytes(
        b'@echo off\r\ncd /d "%~dp0"\r\n"%~dp0runtime\\python.exe" "%~dp0app.py"\r\nif errorlevel 1 pause\r\n')
    (folder / 'minerva_config.json').write_text(json.dumps({'architecture': arch, 'engine': 'Minerva Sapienza',
        'model': 'minerva-7b-instruct-v1.0-q4_k_m.gguf', 'setup_download_bytes': 4478676576}, indent=2), encoding='utf-8')
    (folder / 'Prepara_Minerva.cmd').write_bytes((
        '@echo off\r\ncd /d "%~dp0"\r\n"%~dp0runtime\\python.exe" "%~dp0tools\\prepare_minerva.py" '
        '--destination "%~dp0minerva" --arch ' + arch + '\r\nif errorlevel 1 pause\r\n').encode('utf-8'))
    (folder / 'Collaudo_locale.cmd').write_bytes(
        b'@echo off\r\ncd /d "%~dp0"\r\n"%~dp0runtime\\python.exe" -S -m unittest discover -s tests -v\r\npause\r\n')
    (folder / 'runtime_verificato.json').write_text(json.dumps({
        'python': VERSION, 'architecture': arch,
        'source': f'https://www.python.org/ftp/python/{VERSION}/python-{VERSION}-embed-{RUNTIMES[arch][0]}.zip',
        'sha256': hashlib.sha256(runtime_data).hexdigest(),
        'pe_machine': hex(RUNTIMES[arch][2]),
    }, indent=2), encoding='utf-8')
    bundle = destination / (folder.name + '.zip')
    with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in sorted(folder.rglob('*')):
            if path.is_file():
                z.write(path, folder.name + '/' + path.relative_to(folder).as_posix())
    return folder, bundle


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arch', choices=RUNTIMES, required=True)
    p.add_argument('--runtime-zip', type=Path)
    p.add_argument('--output', type=Path, default=Path('dist'))
    args = p.parse_args()
    if args.runtime_zip:
        data = args.runtime_zip.read_bytes()
    else:
        suffix = RUNTIMES[args.arch][0]
        url = f'https://www.python.org/ftp/python/{VERSION}/python-{VERSION}-embed-{suffix}.zip'
        with urllib.request.urlopen(url, timeout=90) as response:
            data = response.read(32 * 1024 * 1024)
    folder, bundle = build(args.arch, data, args.output)
    print(folder)
    print(bundle)


if __name__ == '__main__':
    main()
