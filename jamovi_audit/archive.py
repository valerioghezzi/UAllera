"""Bounded, in-memory ZIP reading and conservative submission grouping."""
import hashlib
import io
import re
import stat
import unicodedata
import zipfile
from pathlib import PurePosixPath

MAX_FILES = 12000
MAX_FILE_SIZE = 128 * 1024 * 1024
MAX_TOTAL_SIZE = 768 * 1024 * 1024
_RESERVED = re.compile(r'^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)', re.I)


def safe_zip(data: bytes, label: str = '') -> dict[str, bytes]:
    """Read bytes without extraction; reject names unsafe on Windows or POSIX."""
    if not isinstance(data, bytes):
        raise ValueError('ZIP: contenuto non binario')
    result, seen, files, dirs = {}, set(), set(), set()
    total = 0
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            infos = z.infolist()
            if len(infos) > MAX_FILES:
                raise ValueError('ZIP: troppe componenti')
            for info in infos:
                name = info.orig_filename
                stripped = name[:-1] if name.endswith('/') else name
                parts = stripped.split('/')
                if (not stripped or '\\' in name or '\x00' in name or
                    any(p in ('', '.', '..') or p.endswith((' ', '.')) or
                        any(ord(c) < 32 or c in ':<>"|?*' for c in p) or _RESERVED.match(p) for p in parts)):
                    raise ValueError('ZIP: percorso non sicuro: ' + name)
                key = unicodedata.normalize('NFC', stripped).casefold()
                if key in seen:
                    raise ValueError('ZIP: nomi duplicati o collisione Windows: ' + name)
                seen.add(key)
                mode = info.external_attr >> 16
                if stat.S_ISLNK(mode) or (mode and stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR)):
                    raise ValueError('ZIP: collegamento o tipo speciale: ' + name)
                if info.flag_bits & 1:
                    raise ValueError('ZIP: archivio cifrato')
                ancestors = { '/'.join(key.split('/')[:n]) for n in range(1, len(parts)) }
                if ancestors & files or (not info.is_dir() and key in dirs):
                    raise ValueError('ZIP: collisione file/cartella: ' + name)
                dirs.update(ancestors)
                if info.is_dir():
                    if key in files: raise ValueError('ZIP: collisione file/cartella')
                    dirs.add(key)
                    continue
                files.add(key)
                if info.file_size > MAX_FILE_SIZE:
                    raise ValueError('ZIP: componente troppo grande')
                total += info.file_size
                if total > MAX_TOTAL_SIZE:
                    raise ValueError('ZIP: dimensione totale eccessiva')
                content = z.read(info)
                if len(content) != info.file_size: raise ValueError('ZIP: dimensione incoerente')
                result[name] = content
    except (zipfile.BadZipFile, RuntimeError, NotImplementedError, OSError) as e:
        raise ValueError(f'ZIP illeggibile {label}: {e}') from e
    return result


_FORMAT_DIRS = {'analisi','analysis','analyses','relazione','relazioni','report','reports','omv','docx','word','jamovi','elaborati','files','documenti','documents'}


def inventory(entries: dict[str, bytes]) -> list[dict]:
    """Keep directory identities; never resolve versions or unknown folder ambiguity."""
    groups = {}
    for path in sorted(entries, key=lambda p: (p.casefold(),p)):
        p = PurePosixPath(path)
        if '__MACOSX' in p.parts or p.name.startswith(('~$', '._')): continue
        ext = p.suffix.lower()
        if ext not in ('.omv','.docx','.zip'): continue
        parent = p.parent
        while parent.name.casefold() in _FORMAT_DIRS and str(parent.parent) != '.': parent = parent.parent
        key = str(parent)
        flat_label = None
        if key == '.' and ext != '.zip':
            flat_label = re.sub(r'(?i)(?:[_ -]relazione)?(?:[_ -](?:v|versione) ?[0-9]+| ?\([0-9]+\))?$', '', p.stem) or p.stem
            key = './' + flat_label.casefold()
        g = groups.setdefault(key, {'key': hashlib.sha256(key.encode()).hexdigest()[:16], 'label':flat_label or (parent.name if key != '.' else 'Consegne nella radice'), 'omv':[], 'docx':[], 'warnings':[], '_folder':key})
        if ext == '.zip':g['warnings'].append('Archivio annidato non espanso: ' + path)
        else:g[ext[1:]].append(path)
    # A nested ZIP alone remains visible; no silent omission of unhandled submissions.
    result=[]
    for folder,g in sorted(groups.items()):
        for kind in ('omv','docx'):
            if len(g[kind])>1:g['warnings'].append(f'Versioni multiple {kind.upper()}: nessun file scelto automaticamente')
        if not g['omv'] or not g['docx']:
            g['warnings'].append('Coppia incompleta; struttura delle cartelle e associazione da verificare')
        del g['_folder']
        result.append(g)
    return result
