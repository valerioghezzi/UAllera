"""Structured, source-bound evidence; traceability never certifies semantics.

No document is modified or reinterpreted. Only exact cited Word substrings enter
anchors; whole source documents, prompts, engine configuration and credentials
are never copied. Artifact hashes are reader observations, not digital signatures.
"""
import hashlib
import json
import re
from collections import Counter

from functools import lru_cache


class _FrozenFacts(dict):
    """Shared computed facts; callers may copy them but cannot annotate in place."""
    def _reject(self,*args,**kwargs):raise TypeError('Computed provenance facts are read-only.')
    __setitem__=__delitem__=clear=pop=popitem=setdefault=update=__ior__=_reject
    def __deepcopy__(self,memo):return self


@lru_cache(maxsize=64)
def _confidence_facts(checks_tuple, quoted, has_anchor):
    # Only static check names/booleans enter this bounded pool, never student text.
    supported=sum(passed for _,passed in checks_tuple);eligible=len(checks_tuple)
    missing=tuple(name for name,passed in checks_tuple if not passed)
    if not has_anchor and not any(name=='original_source_identity' for name,_ in checks_tuple):
        missing+=('original_source_anchor_not_available',)
    return _FrozenFacts(support_score=supported/eligible if quoted and eligible else None,
        supported_checks=supported,eligible_checks=eligible,
        checks=tuple(_FrozenFacts(name=name,passed=passed) for name,passed in checks_tuple),
        missing=missing,scope='source_traceability_only',semantic_confidence=None,calibration='not_available')


VERSION = '1.0'
_TEXT_FIELDS = ('requirement', 'observed', 'expected', 'location', 'reason', 'explanation', 'title')
_PROTOCOL_FIELDS = ('version', 'expected_model_sha256', 'model_sha256', 'template_sha256',
                    'expected_runtime', 'reported_runtime', 'runtime', 'runtime_verified',
                    'asset_verification', 'context_tokens', 'temperature', 'seed', 'output_tokens',
                    'tokenizer_sha256', 'input_encoding')
_LABELS = ('coerente', 'errata', 'incompleta')


def _scalar(value):
    return value is None or type(value) in (str, int, float, bool)


def _selected(value, fields):
    return {key: value[key] for key in fields if key in value and _scalar(value[key])}


def _protocol(value):
    if not isinstance(value, dict):
        return {}
    result = _selected(value, _PROTOCOL_FIELDS)
    runtime = value.get('reported_runtime')
    if isinstance(runtime, dict):
        result['reported_runtime'] = _selected(runtime, ('version', 'build', 'commit', 'compiler', 'target', 'number'))
    return result


def _sha(document):
    value = document.get('sha256') if isinstance(document, dict) else None
    return value.lower() if isinstance(value, str) and re.fullmatch(r'[0-9a-fA-F]{64}', value) else None


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def _word_sources(document):
    """Mirror the reader's stable IDs, retaining ambiguity rather than guessing."""
    if not isinstance(document, dict):
        return []
    sources = document.get('evidence_sources')
    if sources is None:
        sources = [dict(p, id=f'paragraph/{i}') for i, p in enumerate(document.get('paragraphs', [])) if isinstance(p, dict)]
        for ti, table in enumerate(document.get('tables', [])):
            if not isinstance(table, dict):
                continue
            for ci, cell in enumerate(table.get('cells', [])):
                for pi, p in enumerate(cell.get('paragraphs', [])):
                    if isinstance(p, dict):
                        sources.append(dict(p, id=f'table/{ti}/{ci}/{pi}'))
            if not table.get('cells'):
                for ri, row in enumerate(table.get('rows', [])):
                    for ci, text in enumerate(row):
                        if isinstance(text, str):
                            sources.append({'id': f'rows/{ti}/{ri}/{ci}', 'text': text,
                                            'location': f'Tabella {ti+1}, riga {ri+1}, cella {ci+1}'})
    if not isinstance(sources, list):
        return []
    return [{'id': str(source.get('id', i)), 'text': source['text'],
             'location': str(source.get('location', f'Paragrafo {i+1}'))}
            for i, source in enumerate(sources) if isinstance(source, dict) and isinstance(source.get('text'), str)]


def _artifact_kinds(finding, origin):
    source = finding.get('source', '')
    source = source.casefold() if isinstance(source, str) else ''
    if 'omv' in source and 'word' in source:
        return ('omv', 'docx')
    if origin == 'model' or 'word' in source:
        return ('docx',)
    if 'omv' in source:
        return ('omv',)
    # Unknown provenance must not silently assert that only one artifact matters.
    return ('omv', 'docx')


def _word_anchor(finding, sources, linguistic, word_sha):
    checks = {}
    quote = finding.get('quote')
    if not isinstance(quote, str) or not quote:
        return [], {'exact_quote': False, 'original_source_identity': False, 'source_offsets': False}
    explicit = any(key in finding for key in ('source_id', 'char_start', 'char_end', 'evidence_id'))
    location = finding.get('location')
    if explicit:
        candidates = [s for s in sources if s['id'] == finding.get('source_id')]
        source = candidates[0] if len(candidates) == 1 else None
        start, end = finding.get('char_start'), finding.get('char_end')
    else:
        # Legacy quotes have no identity contract: accept only an unambiguous
        # original node and a single literal occurrence in that node.
        candidates = [s for s in sources if s['location'] == location and quote in s['text']]
        source = (candidates[0] if len(candidates) == 1 and
                  candidates[0]['text'].find(quote, candidates[0]['text'].find(quote) + 1) == -1 else None)
        start = source['text'].find(quote) if source else None
        end = start + len(quote) if source else None
    checks['original_source_identity'] = source is not None
    checks['source_location'] = bool(source and location == source['location'])
    checks['source_offsets'] = bool(source and type(start) is int and type(end) is int
                                    and 0 <= start < end <= len(source['text']))
    checks['exact_quote'] = bool(checks['source_offsets'] and source['text'][start:end] == quote)
    evidence_id = finding.get('evidence_id')
    if explicit or isinstance(linguistic.get('manifest'),dict):
        # A modern evidence contract cannot be downgraded to a legacy literal
        # anchor by removing/nulling its request ID. All bindings remain required.
        manifest = linguistic.get('manifest', {})
        manifest = manifest if isinstance(manifest, dict) else {}
        spans = manifest.get('spans', [])
        spans = spans if isinstance(spans, list) else []
        matches = [span for span in spans if isinstance(span, dict) and span.get('evidence_id') == evidence_id]
        checks['request_evidence_binding'] = bool(isinstance(evidence_id, str) and len(matches) == 1
            and matches[0].get('source_id') == finding.get('source_id')
            and type(matches[0].get('start')) is int and type(matches[0].get('end')) is int
            and matches[0]['start'] == start and matches[0]['end'] == end
            and matches[0].get('location') == location)
        checks['document_hash_binding'] = bool(word_sha and manifest.get('source_document_sha256') == word_sha)
        chunks = linguistic.get('chunks', [])
        chunks = chunks if isinstance(chunks, list) else []
        chunk_number = finding.get('chunk')
        requests = [chunk for chunk in chunks if isinstance(chunk, dict)
                    and type(chunk.get('number')) is int and chunk.get('number') == chunk_number]
        checks['successful_request_binding'] = bool(type(chunk_number) is int and len(requests) == 1
            and requests[0].get('status') == 'complete' and isinstance(requests[0].get('evidence_ids'), list)
            and evidence_id in requests[0]['evidence_ids'])
    if not all(checks.values()):
        return [], checks
    return [{'artifact': 'docx', 'source_id': source['id'], 'start': start, 'end': end,
             'quote': quote, 'location': source['location'], 'evidence_id': evidence_id,
             'validated': True, 'binding': 'evidence_id' if evidence_id is not None else 'unique_literal'}], checks


def build_evidence_bundle(report, documents):
    """Return an independent evidence view of deterministic/model proposals.

    ``support_score`` measures only the checks named beside it. It is None for
    records lacking a quoted observation, where an observation-level score would
    be misleading. Neither evaluated status nor a score measures correctness,
    fraud, authorship, AI use, or the accuracy of the deterministic program.
    """
    if not isinstance(report, dict) or not isinstance(documents, dict):
        raise ValueError('Rapporto e documenti devono essere oggetti strutturati.')
    linguistic = report.get('linguistic', {})
    linguistic = linguistic if isinstance(linguistic, dict) else {}
    artifacts = {kind: _FrozenFacts(kind=kind,sha256=_sha(documents.get(kind))) for kind in ('omv','docx')}
    bindings={}
    sources = _word_sources(documents.get('docx'))
    records, occurrences = [], Counter()
    for origin, findings in (('deterministic', report.get('findings', [])), ('model', linguistic.get('findings', []))):
        if not isinstance(findings, list):
            continue
        for finding in findings:
            if not isinstance(finding, dict):
                continue
            kinds=tuple(_artifact_kinds(finding,origin))
            bound=bindings.setdefault(kinds,tuple(artifacts[kind] for kind in kinds))
            checks = {f'artifact_hash:{artifact["kind"]}': artifact['sha256'] is not None for artifact in bound}
            anchors = []
            if origin == 'model' or 'quote' in finding:
                anchors, anchor_checks = _word_anchor(finding, sources, linguistic, artifacts['docx']['sha256'])
                checks.update(anchor_checks)
            confidence=_confidence_facts(tuple(checks.items()),'quote' in finding,bool(anchors))
            status, proposal = finding.get('status'), finding.get('proposal')
            if status == 'Non applicabile':
                evaluation = 'not_applicable'
            elif origin == 'model':
                evaluation = 'evaluated' if proposal in _LABELS and anchors else 'unverifiable'
            else:
                evaluation = 'evaluated' if status in ('Verificato', 'Errato', 'Mancante') else 'unverifiable'
            record = {'origin': origin, 'finding_id': finding.get('id') if isinstance(finding.get('id'), str) else None,
                      'part': finding.get('part') if type(finding.get('part')) is int else None,
                      'status': status if isinstance(status, str) else None,
                      'source': finding.get('source') if isinstance(finding.get('source'), str) else None,
                      'finding_text': {key: finding[key] for key in _TEXT_FIELDS if isinstance(finding.get(key), str)},
                      'proposal': proposal if isinstance(proposal, str) else None,
                      'artifacts': bound, 'anchors': anchors, 'confidence': confidence, 'evaluation': evaluation}
            base_id = 'EV-' + _digest(record)
            occurrences[base_id] += 1
            record['id'] = base_id + (f'-{occurrences[base_id]}' if occurrences[base_id] > 1 else '')
            record['contradictions'] = []
            records.append(record)
    contradictions = []
    for part in sorted({r['part'] for r in records if r['origin'] == 'model' and r['part'] is not None}):
        rows = [r for r in records if r['origin'] == 'model' and r['part'] == part and r['proposal'] in _LABELS]
        if len({r['proposal'] for r in rows}) > 1:
            ids = sorted(r['id'] for r in rows)
            conflict = {'id': 'CONFLICT-' + _digest(ids), 'kind': 'conflicting_model_assessments',
                        'part': part, 'record_ids': ids, 'resolution': 'unresolved'}
            contradictions.append(conflict)
            for row in rows:
                # One group reference preserves opposing labels in linear space;
                # same-label records are not represented as contradictory pairs.
                row['contradictions'] = [conflict['id']]
    counts = Counter(r['evaluation'] for r in records)
    return {'version': VERSION, 'artifacts': list(artifacts.values()), 'records': records,
            'contradictions': contradictions,
            'coverage': {'total_records': len(records), 'evaluated': counts['evaluated'],
                         'unverifiable': counts['unverifiable'], 'not_applicable': counts['not_applicable'],
                         'anchored_records': sum(bool(r['anchors']) for r in records),
                         'by_origin': dict(Counter(r['origin'] for r in records))},
            'protocol': _protocol(linguistic.get('protocol')),
            'limits': ['Le ancore verificano solo provenienza e citazione letterale, non la correttezza del giudizio.',
                       'Le divergenze fra proposte restano irrisolte e richiedono confronto del tutor.',
                       'Confidenza semantica e calibrazione non disponibili; nessuna stima di frode, paternità o origine IA.',
                       'I rilievi deterministici senza ancore puntuali conservano solo il legame con gli artefatti indicati.']}


def export_evidence_json(bundle):
    """Serialize a generated bundle using a nested allowlist, without mutating it.

    This excludes accidental config/document additions; it is not a PII scrubber
    for the cited text itself. Validation occurs during build, against documents.
    """
    if not isinstance(bundle, dict) or bundle.get('version') != VERSION:
        raise ValueError('Bundle di evidenze non supportato.')
    output = {'version': VERSION, 'records': [], 'artifacts': [], 'contradictions': []}
    for artifact in bundle.get('artifacts', []):
        if isinstance(artifact, dict):
            output['artifacts'].append(_selected(artifact, ('kind', 'sha256')))
    for record in bundle.get('records', []):
        if not isinstance(record, dict):
            continue
        row = _selected(record, ('id', 'origin', 'finding_id', 'part', 'status', 'source', 'proposal', 'evaluation'))
        row['finding_text'] = _selected(record.get('finding_text', {}), _TEXT_FIELDS)
        row['artifacts'] = [_selected(a, ('kind', 'sha256')) for a in record.get('artifacts', []) if isinstance(a, dict)]
        row['anchors'] = [_selected(a, ('artifact', 'source_id', 'start', 'end', 'quote', 'location', 'evidence_id', 'validated', 'binding'))
                          for a in record.get('anchors', []) if isinstance(a, dict) and a.get('validated') is True]
        confidence = record.get('confidence', {})
        row['confidence'] = _selected(confidence, ('support_score', 'supported_checks', 'eligible_checks', 'scope', 'semantic_confidence', 'calibration'))
        row['confidence']['checks'] = [_selected(c, ('name', 'passed')) for c in confidence.get('checks', []) if isinstance(c, dict)]
        row['confidence']['missing'] = [s for s in confidence.get('missing', []) if isinstance(s, str)]
        row['contradictions'] = [s for s in record.get('contradictions', []) if isinstance(s, str)]
        output['records'].append(row)
    for conflict in bundle.get('contradictions', []):
        if isinstance(conflict, dict):
            row = _selected(conflict, ('id', 'kind', 'part', 'resolution'))
            row['record_ids'] = [s for s in conflict.get('record_ids', []) if isinstance(s, str)]
            output['contradictions'].append(row)
    coverage = bundle.get('coverage', {})
    output['coverage'] = _selected(coverage, ('total_records', 'evaluated', 'unverifiable', 'not_applicable', 'anchored_records'))
    output['coverage']['by_origin'] = _selected(coverage.get('by_origin', {}), ('deterministic', 'model'))
    output['protocol'] = _protocol(bundle.get('protocol'))
    output['limits'] = [s for s in bundle.get('limits', []) if isinstance(s, str)]
    return json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
