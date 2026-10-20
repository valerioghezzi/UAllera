"""Read-only provenance observations; metadata never establish authorship or copying.

ZIP DOS clocks have no timezone and a two-second resolution. They describe
entry metadata, not a certified save time; filesystem extraction times are never
read. Cross-document clock proximity only supplements an existing content match.
"""
from collections import defaultdict
from datetime import datetime
import io
from itertools import combinations
from pathlib import PurePosixPath
import re
import unicodedata
import zipfile

from .archive import MAX_FILES


CLOSE_SECONDS = 120
_AUTHOR_FIELDS = ('creator', 'lastModifiedBy', 'author', 'lastSavedBy')
_GENERIC_AUTHORS = {
    '', 'user', 'utente', 'admin', 'administrator', 'amministratore', 'author',
    'autore', 'owner', 'default', 'default user', 'windows user', 'utente windows',
    'microsoft', 'microsoft office', 'microsoft office user', 'office', 'office user',
    'microsoft word', 'word', 'python-docx', 'python docx', 'libreoffice',
    'openoffice', 'unknown', 'sconosciuto', 'anonymous', 'anonimo', 'student',
    'studente', 'test', 'jamovi',
}
_LIMITS = (
    'Metadati modificabili e non autenticati: non provano autore, autonomia, '
    'copiatura o direzione degli scambi. Valori mancanti o illeggibili restano '
    'sconosciuti. Gli orari ZIP hanno fuso ignoto e precisione di 2 secondi; '
    'possono derivare da ricompressione, modelli, importazioni o orologi diversi. '
    'Nessun uso delle date del filesystem dopo estrazione.'
)


def _normal(value):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', str(value)).casefold()).strip()


def _date(raw, source):
    """Parse an ISO timestamp without silently supplying a missing timezone."""
    result = {'raw': raw, 'iso': None, 'timezone': 'unknown',
              'precision_seconds': None, 'source': source, 'valid': False}
    if not isinstance(raw, str) or not raw.strip():
        return result
    value = raw.strip()
    # A plain date is not a save time. Reject Python's less specific ISO forms.
    if not re.match(r'^\d{4}-\d{2}-\d{2}[Tt ]\d{2}:\d{2}:\d{2}', value):
        return result
    try:
        parsed = datetime.fromisoformat(value[:-1] + '+00:00' if value.endswith(('Z', 'z')) else value)
    except (ValueError, OverflowError):
        return result
    timezone = 'unknown'
    if parsed.tzinfo is not None and parsed.utcoffset() is not None:
        offset = parsed.utcoffset().total_seconds()
        timezone = 'UTC' if offset == 0 else parsed.strftime('%z')[:3] + ':' + parsed.strftime('%z')[3:]
    fraction = re.search(r':\d{2}[.,](\d+)', value)
    result.update(iso=parsed.isoformat(), timezone=timezone,
                  precision_seconds=10 ** -len(fraction.group(1)) if fraction else 1,
                  valid=True)
    return result


def extract_zip_metadata(data: bytes) -> dict:
    """Read central-directory entry clocks, without extracting or decompressing.

    A corrupt ZIP or excessive directory yields no metadata. Duplicate names are
    not overwritten: safe_zip independently rejects such documents before use.
    """
    if not isinstance(data, bytes):
        return {}
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            infos = archive.infolist()
            if len(infos) > MAX_FILES:
                return {}
            entries = {}
            for info in infos:
                if info.is_dir():
                    continue
                if info.filename in entries:
                    return {}
                raw = list(info.date_time)
                try:
                    iso = datetime(*info.date_time).isoformat()
                except (ValueError, OverflowError, TypeError):
                    iso = None
                entries[info.filename] = {
                    'raw': raw, 'iso': iso, 'timezone': 'unknown',
                    'precision_seconds': 2, 'source': 'ZIP entry',
                    'valid': iso is not None,
                }
            return entries
    except (zipfile.BadZipFile, OSError, ValueError, EOFError, RuntimeError):
        return {}


def inspect_document_metadata(parsed: dict, kind: str, source_zip_metadata=None) -> dict:
    """Return JSON-compatible observations for an individual DOCX or OMV."""
    parsed = parsed if isinstance(parsed, dict) else {}
    core = parsed.get('metadata') or {}
    core = core if isinstance(core, dict) else {}
    source = 'docProps/core.xml' if kind.lower() in ('word', 'docx') else 'metadata.json'
    authors = []
    for field in _AUTHOR_FIELDS:
        value = core.get(field)
        if isinstance(value, str) and value.strip():
            authors.append({'field': field, 'value': value.strip(), 'source': source})
    dates = {}
    for name, aliases in [('created', ('created', 'dateCreated', 'creationDate')),
                          ('modified', ('modified', 'dateModified', 'modificationDate'))]:
        field = next((f for f in aliases if core.get(f) is not None), name)
        dates[name] = _date(core.get(field), source + ':' + field)
    entries = parsed.get('zip_metadata') or {}
    entries = entries if isinstance(entries, dict) else {}
    clocks = []
    for entry in entries.values():
        if not isinstance(entry, dict):
            continue
        value = _date(entry.get('iso'), 'ZIP entry')
        if value['valid'] and value['timezone'] == 'unknown':
            clocks.append(value['iso'])
    summary = {
        'kind': kind, 'authors': authors, 'dates': dates,
        'zip_entries': {
            'count': len(entries), 'valid_timestamp_count': len(clocks),
            'min': min(clocks) if clocks else None,
            'max': max(clocks) if clocks else None,
            'distinct_timestamp_count': len(set(clocks)),
            'source': 'ZIP entry', 'timezone': 'unknown', 'precision_seconds': 2,
            'meaning': 'Orologi delle componenti interne; non certificano il momento del salvataggio.',
        },
        'source_zip_metadata': source_zip_metadata,
        'limits': _LIMITS,
    }
    summary['anomalies'] = metadata_anomalies(summary)
    return summary


def metadata_anomalies(summary: dict) -> list:
    """Find factual internal clock contradictions only for known timezones."""
    dates = summary.get('dates') or {}
    created, modified = dates.get('created') or {}, dates.get('modified') or {}
    if not all(v.get('valid') and v.get('timezone') != 'unknown' for v in (created, modified)):
        return []
    try:
        after = datetime.fromisoformat(created['iso']) > datetime.fromisoformat(modified['iso'])
    except (ValueError, TypeError, KeyError):
        return []
    if not after:
        return []
    source = 'Word' if str(summary.get('kind', '')).lower() in ('word', 'docx') else 'OMV'
    return [{
        'id': 'METADATA.CREATED_AFTER_MODIFIED', 'part': 0, 'source': source,
        'status': 'Da verificare', 'requirement': 'Coerenza interna dei metadati temporali',
        'observed': f"Creazione {created['raw']} successiva alla modifica {modified['raw']} (fusi dichiarati).",
        'expected': 'Verificare orologio, importazione e copia da un modello; i metadati sono modificabili e questa incongruenza non prova copiatura.',
        'location': created.get('source', '') + ' / ' + modified.get('source', ''),
    }]


def _template_authors(reference):
    result = set()
    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in _AUTHOR_FIELDS and isinstance(item, str):
                    result.add(_normal(item))
                elif key == 'authors' and isinstance(item, list):
                    for author in item:
                        if isinstance(author, str):
                            result.add(_normal(author))
                        elif isinstance(author, dict) and isinstance(author.get('value'), str):
                            result.add(_normal(author['value']))
                elif isinstance(item, (dict, list)):
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
    if isinstance(reference, dict):
        visit(reference.get('template_metadata') or {})
    return result


def _document_path(report, kind):
    paths = report.get('paths') or {}
    value = paths.get(kind, [])
    return value if isinstance(value, str) else value[0] if isinstance(value, list) and value else ''


def _different_folders(left, right):
    # Missing provenance cannot establish distinct folders. In a flat archive,
    # keys alone are filename-derived and do not establish folder separation.
    a = _document_path(left, 'docx') or _document_path(left, 'omv')
    b = _document_path(right, 'docx') or _document_path(right, 'omv')
    return bool(a and b and _normal(PurePosixPath(a).parent) != _normal(PurePosixPath(b).parent))


def _time_evidence(left, right, kind, lreport, rreport):
    output = []
    if kind == 'docx':
        lclock, rclock = left['dates']['modified'], right['dates']['modified']
        if not all(v.get('valid') and v.get('timezone') != 'unknown' for v in (lclock, rclock)):
            return []
        lvalue, rvalue = lclock['iso'], rclock['iso']
        basis, component = 'known_timezone', 'Date Word di ultima modifica vicine (fusi dichiarati)'
        llocation, rlocation = lclock['source'], rclock['source']
        precision = max(lclock['precision_seconds'], rclock['precision_seconds'])
    else:
        lvalue, rvalue = left['zip_entries']['max'], right['zip_entries']['max']
        if not lvalue or not rvalue:
            return []
        basis, component = 'local_clock_unknown_timezone', 'Massimi degli orologi ZIP interni OMV vicini (fuso ignoto)'
        llocation = rlocation = 'ZIP interno: massimo timestamp DOS delle componenti'
        precision = 2
    try:
        gap = abs((datetime.fromisoformat(lvalue) - datetime.fromisoformat(rvalue)).total_seconds())
    except (ValueError, TypeError):
        return []
    if gap <= CLOSE_SECONDS:
        output.append({
            'component': component, 'left': lvalue, 'right': rvalue,
            'location_left': _document_path(lreport, kind) + ' / ' + llocation,
            'location_right': _document_path(rreport, kind) + ' / ' + rlocation,
            'gap_seconds': gap, 'precision_seconds': precision,
            'comparison_basis': basis, 'threshold_seconds': CLOSE_SECONDS,
            'meaning': 'Scarto dei soli orologi locali, senza distanza temporale assoluta.' if kind == 'omv' else 'Scarto fra date dichiarate nei metadati, senza cronologia certificata.',
        })
    return output


def metadata_cross_checks(reports: list, parsed: dict, comparisons: list, reference=None) -> list:
    """Return additional weak or corroborating observations, without mutations.

    At most one author record and one temporal record per pair. Multiple content
    comparisons share a single corroboration: timestamps are not counted as new
    independent content evidence. No candidate pair arises from timestamps alone.
    """
    bykey = {r['key']: r for r in reports}
    order = {r['key']: i for i, r in enumerate(reports)}
    docs = {key: {kind: inspect_document_metadata((parsed.get(key) or {}).get(kind), kind)
                  for kind in ('docx', 'omv')} for key in bykey}
    excluded = _GENERIC_AUTHORS | _template_authors(reference)
    author_index = defaultdict(dict)
    for key, documents in docs.items():
        for kind, summary in documents.items():
            for author in summary['authors']:
                value = _normal(author['value'])
                if value in excluded or not any(c.isalpha() for c in value):
                    continue
                author_index[value].setdefault(key, []).append(dict(author, kind=kind))
    author_pairs = defaultdict(list)
    for value, members in sorted(author_index.items()):
        for a, b in combinations(sorted(members, key=order.get), 2):
            if not _different_folders(bykey[a], bykey[b]):
                continue
            la, ra = members[a][0], members[b][0]
            author_pairs[(a, b)].append({
                'component': 'Stesso autore dichiarato nei metadati',
                'left': la['value'], 'right': ra['value'],
                'location_left': _document_path(bykey[a], la['kind']) + ' / ' + la['source'] + ':' + la['field'],
                'location_right': _document_path(bykey[b], ra['kind']) + ' / ' + ra['source'] + ':' + ra['field'],
                'document_frequency': len(members), 'class_size': len(reports),
            })
    output = []
    def emit(pair, kind, evidence, limits, **extras):
        output.append(dict(students=list(pair), labels=[bykey[k].get('label', k) for k in pair],
                           kind=kind, evidence=evidence, evidence_count=len(evidence), limits=limits, **extras))
    for pair, evidence in author_pairs.items():
        emit(pair, 'metadata_autore_condiviso', evidence,
             'Segnale debole: stesso autore dichiarato in cartelle diverse, esclusi valori generici e autori del modello noto. Account condiviso, laboratorio, modello o assistenza possono spiegare la coincidenza. ' + _LIMITS)
    existing = defaultdict(set)
    for comparison in comparisons:
        members = comparison.get('students', [])
        kind = str(comparison.get('kind', ''))
        if (len(members) != 2 or any(k not in bykey for k in members) or
                not kind or kind.startswith('metadata_') or not comparison.get('evidence')):
            continue
        if members[0] == members[1]:
            continue
        pair = tuple(sorted(members, key=order.get))
        existing[pair].add(kind)
    for pair, kinds in sorted(existing.items(), key=lambda item: (order[item[0][0]], order[item[0][1]])):
        a, b = pair
        evidence = []
        for kind in ('docx', 'omv'):
            evidence.extend(_time_evidence(docs[a][kind], docs[b][kind], kind, bykey[a], bykey[b]))
        if evidence:
            emit(pair, 'metadata_correlazione_temporale', evidence,
                 'Corroborazione accessoria non indipendente dei confronti di contenuto già presenti; non aggiunge una seconda prova di copiatura. Soglia descrittiva di vicinanza: 120 secondi. Orologi ZIP senza fuso: si confrontano soltanto le indicazioni locali, non istanti assoluti. Download di gruppo, modelli o ricompressione possono uniformare le date. ' + _LIMITS,
                 corroborates=sorted(kinds))
    return output
