"""Pinned, stdlib-only setup and owned loopback launcher for Sapienza Minerva.

Only ``prepare`` fetches public software/model assets. ``launch`` uses verified
local files and starts no cloud service. Progress receives dictionaries with
``stage``, ``current``, ``total`` and ``message``; raising cancels safely.
"""
from __future__ import annotations

import argparse
import atexit
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import secrets
import shutil
import socket
import stat
import struct
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import zipfile


@dataclass(frozen=True)
class Asset:
    name: str
    url: str
    size: int
    sha256: str


MODEL_ALIAS = 'minerva-7b-instruct-q4-local'
MODEL = Asset(
    'minerva-7b-instruct-v1.0-q4_k_m.gguf',
    'https://huggingface.co/sapienzanlp/Minerva-7B-instruct-v1.0-GGUF/resolve/'
    '9a87b0d5cec8f99fb245b92cc64a9d0394fe38d2/minerva-7b-instruct-v1.0-q4_k_m.gguf',
    4478676576, 'dc3fda6cfad45153e1e0bbe87b3cf421589aaf88086fa048afd619c6e2bfc681',
)
RUNTIMES = {
    'x64': Asset('llama-b11756-bin-win-cpu-x64.zip',
        'https://github.com/ggml-org/llama.cpp/releases/download/b11756/llama-b11756-bin-win-cpu-x64.zip',
        23959301, 'd164195b8f3e92f8fd92bc3a5c0eeaef8707b2f1a979e36360ce8192ebf4fbb3'),
    'ARM64': Asset('llama-b11756-bin-win-cpu-arm64.zip',
        'https://github.com/ggml-org/llama.cpp/releases/download/b11756/llama-b11756-bin-win-cpu-arm64.zip',
        12609658, 'ba47a12a87d45a3a717d717e81334e65b725df051c8adf8f68accd824d05a126'),
}
PE_MACHINES = {'x64': 0x8664, 'ARM64': 0xAA64}
_CHUNK = 1024 * 1024
# Exact bytes from the official tokenizer chat_template.jinja at the pinned revision.
CHAT_TEMPLATE = b"{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"


@dataclass(frozen=True)
class PreparedMinerva:
    destination: Path
    architecture: str
    model_path: Path
    executable_path: Path
    template_path: Path


def _architecture(value):
    aliases = {'x64': 'x64', 'amd64': 'x64', 'arm64': 'ARM64', 'aarch64': 'ARM64'}
    try:
        return aliases[str(value).lower()]
    except KeyError:
        raise ValueError('Architettura Minerva non supportata: ' + str(value)) from None


def _progress(callback, stage, current=0, total=0, message=''):
    if callback:
        callback({'stage': stage, 'current': current, 'total': total, 'message': message})


def _safe_path(root, path):
    """Reject symlinks/reparse points rather than following a cache outside root."""
    relative = path.relative_to(root)
    candidate = root
    for component in relative.parts:
        candidate = candidate / component
        if candidate.is_symlink() or (candidate.exists() and
                getattr(candidate.lstat(), 'st_file_attributes', 0) & 0x400):
            raise ValueError('Percorso locale non sicuro: collegamento nel deposito Minerva.')
    return path


def _valid_asset(path, asset, progress=None):
    if not path.is_file() or path.stat().st_size != asset.size:
        return False
    digest = hashlib.sha256()
    count = 0
    _progress(progress, 'verify', 0, asset.size, 'Verifica ' + asset.name)
    with path.open('rb') as source:
        while block := source.read(_CHUNK):
            digest.update(block)
            count += len(block)
            _progress(progress, 'verify', count, asset.size, 'Verifica ' + asset.name)
    return count == asset.size and digest.hexdigest() == asset.sha256


def _download(path, asset, progress):
    """Stream bounded bytes to the same filesystem; publish only after both checks."""
    if _valid_asset(path, asset, progress):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=asset.name + '.', suffix='.part', dir=path.parent)
    try:
        digest = hashlib.sha256()
        current = 0
        request = urllib.request.Request(asset.url, headers={'User-Agent': 'UAllera-Minerva-Setup/1'})
        with os.fdopen(fd, 'wb') as output:
            _progress(progress, 'download', 0, asset.size, 'Scaricamento pubblico ' + asset.name)
            with urllib.request.urlopen(request, timeout=90) as response:
                while True:
                    # One excess byte detects a server ignoring the pinned expected size.
                    block = response.read(min(_CHUNK, asset.size - current + 1))
                    if not block:
                        break
                    current += len(block)
                    if current > asset.size:
                        raise ValueError('Dimensione del download superiore alla dimensione prevista.')
                    digest.update(block)
                    output.write(block)
                    _progress(progress, 'download', current, asset.size, 'Scaricamento pubblico ' + asset.name)
            if current != asset.size:
                raise ValueError('Dimensione del download incompleta rispetto alla dimensione prevista.')
            if digest.hexdigest() != asset.sha256:
                raise ValueError('SHA-256 diverso dalla fonte ufficiale: ' + asset.name)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _zip_members(bundle):
    members = bundle.infolist()
    if len(members) > 4096 or sum(x.file_size for x in members) > 512 * 1024 * 1024:
        raise ValueError('ZIP runtime troppo grande.')
    seen = set()
    reserved = {'CON', 'PRN', 'AUX', 'NUL'} | {f'{stem}{n}' for stem in ('COM', 'LPT') for n in range(1, 10)}
    for info in members:
        name = info.filename
        parts = name.rstrip('/').split('/')
        mode = info.external_attr >> 16
        if (not name or '\\' in name or ':' in name or name.startswith('/') or
                any(not part or part in ('.', '..') or part.rstrip('. ') != part or
                    part.split('.')[0].upper() in reserved for part in parts) or
                stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR)) or
                name.casefold().rstrip('/') in seen or info.flag_bits & 1):
            raise ValueError('Percorso o voce ZIP runtime non sicuro.')
        seen.add(name.casefold().rstrip('/'))
    return members


def _check_pe(path, architecture):
    with path.open('rb') as executable:
        header = executable.read(64)
        if len(header) != 64 or header[:2] != b'MZ':
            raise ValueError('Eseguibile PE runtime non valido.')
        offset = struct.unpack_from('<I', header, 0x3c)[0]
        if offset > path.stat().st_size - 6:
            raise ValueError('Eseguibile PE runtime non valido.')
        executable.seek(offset)
        signature = executable.read(6)
    if signature[:4] != b'PE\0\0' or struct.unpack_from('<H', signature, 4)[0] != PE_MACHINES[architecture]:
        raise ValueError('Architettura PE del runtime non corrispondente.')


def _runtime_valid(root, folder, archive, architecture):
    if not folder.is_dir():
        return False
    with zipfile.ZipFile(archive) as bundle:
        members = _zip_members(bundle)
        expected = set()
        for info in members:
            path = _safe_path(root, folder / info.filename)
            if info.is_dir():
                if not path.is_dir():
                    return False
                continue
            expected.add(path.relative_to(folder).as_posix())
            if not path.is_file() or path.stat().st_size != info.file_size:
                return False
            with bundle.open(info) as source, path.open('rb') as cached:
                while block := source.read(_CHUNK):
                    if cached.read(len(block)) != block:
                        return False
        # Extra DLLs in the executable directory could alter runtime loading.
        for path in folder.rglob('*'):
            _safe_path(root, path)
            if path.is_file() and path.relative_to(folder).as_posix() not in expected:
                return False
    server = folder / 'llama-server.exe'
    if not server.is_file():
        return False
    _check_pe(server, architecture)
    return True


def _extract_runtime(root, folder, archive, architecture):
    temporary = Path(tempfile.mkdtemp(prefix='.runtime-', dir=root))
    try:
        with zipfile.ZipFile(archive) as bundle:
            members = _zip_members(bundle)  # Validate every path before writing any member.
            for info in members:
                path = temporary / PurePosixPath(info.filename)
                if info.is_dir():
                    path.mkdir(parents=True, exist_ok=True)
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.open(info) as source, path.open('xb') as output:
                        shutil.copyfileobj(source, output, _CHUNK)
        _check_pe(temporary / 'llama-server.exe', architecture)
        if folder.exists():
            shutil.rmtree(folder)
        os.replace(temporary, folder)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


@contextmanager
def _setup_lock(root):
    path = _safe_path(root, root / 'setup.lock')
    with path.open('a+b') as lock:
        if os.name == 'nt':
            import msvcrt
            if path.stat().st_size == 0:
                lock.write(b'0'); lock.flush()
            lock.seek(0)
            try:
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                raise RuntimeError('Preparazione Minerva già in corso.') from None
        else:
            import fcntl
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('Preparazione Minerva già in corso.') from None
        try:
            yield
        finally:
            if os.name == 'nt':
                lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _paths(destination, architecture):
    architecture = _architecture(architecture)
    root = Path(destination).expanduser().resolve()
    return PreparedMinerva(root, architecture, root / 'models' / MODEL.name,
        root / ('runtime-b11756-' + architecture) / 'llama-server.exe', root / 'chat_template.jinja')


def prepare(destination, architecture, progress=None):
    """Install/repair pinned public assets, returning verified local paths.

    The 4.48 GB model is shared between architectures. Successful cached files
    are reused after full SHA-256 verification. Cancellation/errors leave no
    partially published asset; a subsequent call can retry safely.
    """
    prepared = _paths(destination, architecture)
    root = prepared.destination
    root.mkdir(parents=True, exist_ok=True)
    with _setup_lock(root):
        archive = root / 'downloads' / RUNTIMES[prepared.architecture].name
        for path in (prepared.model_path, prepared.executable_path, prepared.template_path, archive):
            _safe_path(root, path)
        _download(prepared.model_path, MODEL, progress)
        _download(archive, RUNTIMES[prepared.architecture], progress)
        if not _runtime_valid(root, prepared.executable_path.parent, archive, prepared.architecture):
            _progress(progress, 'extract', message='Preparazione motore locale Minerva')
            _extract_runtime(root, prepared.executable_path.parent, archive, prepared.architecture)
        if not prepared.template_path.is_file() or prepared.template_path.read_bytes() != CHAT_TEMPLATE:
            fd, temporary = tempfile.mkstemp(prefix='template-', suffix='.part', dir=root)
            try:
                with os.fdopen(fd, 'wb') as output:
                    output.write(CHAT_TEMPLATE); output.flush(); os.fsync(output.fileno())
                os.replace(temporary, prepared.template_path)
            finally:
                if os.path.exists(temporary): os.unlink(temporary)
    _progress(progress, 'ready', message='Minerva locale pronto')
    return prepared


def _validate_local(prepared, progress):
    root = prepared.destination
    archive = root / 'downloads' / RUNTIMES[prepared.architecture].name
    for path in (prepared.model_path, prepared.executable_path, prepared.template_path, archive):
        _safe_path(root, path)
    if (not _valid_asset(prepared.model_path, MODEL, progress) or
            not _valid_asset(archive, RUNTIMES[prepared.architecture], progress) or
            not _runtime_valid(root, prepared.executable_path.parent, archive, prepared.architecture) or
            not prepared.template_path.is_file() or prepared.template_path.read_bytes() != CHAT_TEMPLATE):
        raise ValueError('Installazione Minerva mancante o modificata: eseguire la preparazione locale.')


@dataclass(repr=False)
class LocalServer:
    """Only this subprocess is stopped; never an already running local service."""
    process: subprocess.Popen
    base_url: str
    api_key: str = field(repr=False)
    model: str = MODEL_ALIAS
    cleanup_complete: bool = field(default=False, init=False, repr=False)
    _stop_attempted: bool = field(default=False, init=False, repr=False)

    def __repr__(self):
        return f'LocalServer(base_url={self.base_url!r}, model={self.model!r})'

    def stop(self):
        # TimeoutExpired embeds process.args, which contains the private loopback
        # capability. Cleanup must never propagate or stringify that exception.
        if self._stop_attempted:
            return self.cleanup_complete
        self._stop_attempted = True
        try:
            try:
                running = self.process.poll() is None
            except (OSError, subprocess.SubprocessError):
                running = True
            if running:
                try:
                    self.process.terminate()
                except (OSError, subprocess.SubprocessError):
                    pass
                try:
                    self.process.wait(timeout=5)
                except (OSError, subprocess.SubprocessError):
                    try:
                        self.process.kill()
                    except (OSError, subprocess.SubprocessError):
                        pass
                    try:
                        self.process.wait(timeout=5)
                    except (OSError, subprocess.SubprocessError):
                        pass
            try:
                self.cleanup_complete = self.process.poll() is not None
            except (OSError, subprocess.SubprocessError):
                self.cleanup_complete = False
        finally:
            atexit.unregister(self.stop)
        return self.cleanup_complete

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stop()


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def launch(prepared_or_destination, architecture=None, *, port=0, startup_timeout=120, progress=None):
    """Start verified cached assets, wait for authenticated readiness, return owner.

    No downloads occur here. ``api_key`` is a fresh internal loopback capability,
    never an external provider credential. Keep it in memory and use it as a
    Bearer header for local requests. The object supports ``with`` and ``stop``.
    """
    if not isinstance(port, int) or not 0 <= port <= 65535 or startup_timeout <= 0:
        raise ValueError('Porta o tempo di avvio Minerva non valido.')
    if isinstance(prepared_or_destination, PreparedMinerva):
        # Derive paths again: callers cannot substitute arbitrary executable/model paths.
        prepared = _paths(prepared_or_destination.destination, prepared_or_destination.architecture)
    else:
        prepared = _paths(prepared_or_destination, architecture)
    _validate_local(prepared, progress)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reservation:
        if os.name == 'nt':
            reservation.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        reservation.bind(('127.0.0.1', port))
        port = reservation.getsockname()[1]
    key = secrets.token_urlsafe(32)
    arguments = [str(prepared.executable_path), '--model', str(prepared.model_path),
        '--host', '127.0.0.1', '--port', str(port), '--ctx-size', '4096',
        '--parallel', '1', '--alias', MODEL_ALIAS, '--offline', '--no-webui',
        '--jinja', '--chat-template-file', str(prepared.template_path), '--api-key', key]
    options = {'cwd': str(prepared.executable_path.parent), 'stdin': subprocess.DEVNULL,
        'stdout': subprocess.DEVNULL, 'stderr': subprocess.DEVNULL}
    if os.name == 'nt':
        options['creationflags'] = subprocess.CREATE_NO_WINDOW
    try:
        process = subprocess.Popen(arguments, **options)
    except OSError:
        raise RuntimeError('Impossibile avviare il motore Minerva locale per questa architettura.') from None
    server = LocalServer(process, f'http://127.0.0.1:{port}', key)
    atexit.register(server.stop)
    # Ignore proxy environment and reject redirects: the secret stays on loopback.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    deadline = time.monotonic() + startup_timeout
    try:
        _progress(progress, 'loading', message='Caricamento Minerva locale in memoria')
        while time.monotonic() < deadline:
            # Recheck cooperative cancellation while the native model is loading.
            _progress(progress, 'loading', message='Caricamento Minerva locale in memoria')
            code = process.poll()
            if code is not None:
                raise RuntimeError(f'Il motore Minerva locale è terminato durante l’avvio (codice {code}).')
            request = urllib.request.Request(server.base_url + '/v1/models', headers={'Authorization': 'Bearer ' + key})
            try:
                with opener.open(request, timeout=min(1.0, max(.01, deadline - time.monotonic()))) as response:
                    data = response.read(65537)
                    if len(data) <= 65536:
                        payload = json.loads(data)
                        if any(item.get('id') == MODEL_ALIAS for item in payload.get('data', []) if isinstance(item, dict)):
                            if process.poll() is not None:
                                continue
                            _progress(progress, 'ready', message='Minerva locale avviato')
                            return server
            except (OSError, ValueError, urllib.error.URLError):
                pass
            time.sleep(min(.1, max(0, deadline - time.monotonic())))
        raise TimeoutError('Tempo di caricamento Minerva locale superato.')
    except BaseException:
        server.stop()
        raise


def main():
    parser = argparse.ArgumentParser(description='Prepara il motore linguistico locale Minerva Sapienza.')
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--arch', choices=('x64', 'ARM64'), required=True)
    args = parser.parse_args()
    last = [None]
    def show(event):
        current = (event['stage'], event['current'] // (64 * 1024 * 1024))
        if current != last[0] or event['current'] == event['total']:
            amount = f" {event['current']:,}/{event['total']:,} byte" if event['total'] else ''
            print(event['message'] + amount, flush=True)
            last[0] = current
    try:
        prepared = prepare(args.destination, args.arch, show)
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as error:
        parser.exit(1, 'Preparazione Minerva non riuscita: ' + str(error) + '\n')
    print(prepared.destination)


if __name__ == '__main__':
    main()
