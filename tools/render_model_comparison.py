"""Bounded offline report for saved model comparisons; never performs inference."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import html
from itertools import combinations
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.compare_local_models import LABELS, summarize
from jamovi_audit.validation_cases import VALIDATION_CASES

MAX_JSON_BYTES = 32 * 1024 * 1024
IDENTITY = ('id', 'part', 'dataset', 'expected_assessment', 'text', 'criterion', 'statistical_context')
PAIR_SETTINGS = ('context_tokens', 'output_tokens', 'reasoning_budget', 'temperature', 'seed')
BASELINE = {c['id']: dict(c, dataset='baseline') for c in VALIDATION_CASES}


def read_json(path, expected_sha256=None):
    """Bound input bytes before decoding or parsing untrusted saved artifacts."""
    path = Path(path)
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError('JSON file exceeds 32 MiB')
    with path.open('rb') as handle:
        data = handle.read(MAX_JSON_BYTES + 1)
    if len(data) > MAX_JSON_BYTES:
        raise ValueError('JSON file exceeds 32 MiB')
    if expected_sha256 is not None and hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError('Frozen JSON SHA-256 differs from saved input hash')
    return json.loads(data)


def identity(case):
    return tuple(case.get(k) for k in IDENTITY)


def validate_plan(plan):
    if not isinstance(plan, list) or not plan:
        raise ValueError('Expected a nonempty planned-case list')
    seen = set()
    for case in plan:
        if (not isinstance(case, dict) or not isinstance(case.get('id'), str) or
                not case['id'] or case['id'] in seen or type(case.get('part')) is not int or
                case['part'] not in range(1, 8) or case.get('dataset') not in ('baseline', 'fresh') or
                case.get('expected_assessment') not in LABELS or
                any(not isinstance(case.get(k), str) or not case[k].strip() for k in ('text', 'criterion'))):
            raise ValueError('Invalid or duplicate planned case')
        seen.add(case['id'])
        if case['dataset'] == 'baseline' and (case['id'] not in BASELINE or identity(case) != identity(BASELINE[case['id']])):
            raise ValueError('Baseline plan differs from the current original corpus')


def validate_rows(plan, rows):
    """Keep genuine classification errors, reject inconsistent saved evidence."""
    summary = summarize(plan, rows)
    planned_by_id = {case['id']: case for case in plan}
    for row in rows:
        if identity(row) != identity(planned_by_id[row['id']]):
            raise ValueError('Result context differs from the frozen case')
        observed = row.get('observed')
        if row.get('status') not in ('complete', 'invalid_response', 'infrastructure_error'):
            raise ValueError('Unknown saved row status')
        if row['status'] == 'complete':
            if observed not in LABELS:
                raise ValueError('Complete row lacks a valid assessment')
            findings = row.get('findings', [])
            if not isinstance(findings, list) or any(not isinstance(f, dict) for f in findings):
                raise ValueError('Invalid saved findings')
            assessments = {f.get('assessment', f.get('proposal')) for f in findings
                           if 'assessment' in f or 'proposal' in f}
            if assessments != {observed}:
                raise ValueError('Observed assessment differs from saved findings')
        elif observed is not None:
            raise ValueError('Unfinished row cannot contain a semantic prediction')
    return summary


def validate_scope(plan, saved, directory):
    """Verify full planned denominators against the pinned fresh snapshot."""
    scope = saved['protocol'].get('case_scope', 'all')
    if scope not in ('all', 'fresh'):
        raise ValueError('Unknown planned case scope')
    hashes = saved.get('input_hashes')
    expected_hash = hashes.get('fresh') if isinstance(hashes, dict) else None
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError('Fresh snapshot requires its frozen input SHA-256')
    frozen = read_json(directory / 'frozen_fresh.json', expected_hash)
    if (not isinstance(frozen, list) or any(not isinstance(c, dict) or c.get('dataset') not in (None, 'fresh') for c in frozen)):
        raise ValueError('Invalid frozen fresh snapshot')
    fresh = [dict(c, dataset='fresh') for c in frozen]
    validate_plan(fresh)
    if len(fresh) != 14 or Counter(c['part'] for c in fresh) != {part: 2 for part in range(1, 8)}:
        raise ValueError('Fresh snapshot must contain two cases per part')
    planned_fresh = {c['id']: c for c in plan if c['dataset'] == 'fresh'}
    frozen_fresh = {c['id']: c for c in fresh}
    if (set(planned_fresh) != set(frozen_fresh) or
            any(identity(planned_fresh[key]) != identity(frozen_fresh[key]) for key in frozen_fresh)):
        raise ValueError('Planned fresh identities differ from the frozen snapshot')
    baseline_ids = {c['id'] for c in plan if c['dataset'] == 'baseline'}
    if scope == 'all' and (len(plan) != 84 or baseline_ids != set(BASELINE)):
        raise ValueError('All-case scope requires 70 original and 14 fresh cases')
    if scope == 'fresh' and (len(plan) != 14 or baseline_ids):
        raise ValueError('Fresh scope requires exactly the 14 fresh cases')


def reason_text(row):
    return '\n'.join(str(f.get('reason', f.get('observed', ''))) for f in row.get('findings', []) if isinstance(f, dict))


def public_row(row):
    """Allowlist report fields; raw exchanges and thought text stay in source JSON."""
    keys = (*IDENTITY, 'rationale', 'status', 'observed', 'seconds', 'finish_reason',
            'usage', 'native_reasoning_observed', 'inference_attempted', 'error_code')
    clean = {k: row.get(k) for k in keys}
    clean['reason'] = reason_text(row)
    return clean


def load_result(name, directory):
    directory = Path(directory)
    saved = read_json(directory / 'summary.json')
    plan = read_json(directory / 'planned_cases.json')
    validate_plan(plan)
    if not isinstance(saved, dict) or not isinstance(saved.get('protocol'), dict):
        raise ValueError('Saved summary lacks protocol metadata')
    validate_scope(plan, saved, directory)
    rows = []
    for part in range(1, 8):
        path = directory / f'part_{part}.json'
        if not path.exists():
            continue
        data = read_json(path)
        if (not isinstance(data, dict) or data.get('part') != part or
                data.get('model') != saved.get('model') or data.get('profile') != saved.get('profile') or
                not isinstance(data.get('cases'), list)):
            raise ValueError('Part metadata differs from summary')
        if 'protocol' in data and data['protocol'] != saved['protocol']:
            raise ValueError('Mixed protocols within one result group')
        if 'input_hashes' in data and 'input_hashes' in saved and data['input_hashes'] != saved['input_hashes']:
            raise ValueError('Mixed frozen inputs within one result group')
        for row in data['cases']:
            if not isinstance(row, dict) or row.get('part') != part:
                raise ValueError('Case is in the wrong part file')
            if 'profile' in row and row['profile'] != saved['profile']:
                raise ValueError('Mixed row profiles')
            if 'protocol' in row and row['protocol'] != saved['protocol']:
                raise ValueError('Mixed row protocols')
        rows.extend(data['cases'])
    recomputed = validate_rows(plan, rows)
    return dict(name=name, model=saved.get('model'), profile=saved.get('profile'),
                protocol=saved['protocol'], source=str(directory.resolve()),
                input_hashes=saved.get('input_hashes', {}), plan=plan,
                rows=[public_row(r) for r in rows], summary=recomputed)


def load_minerva(path):
    saved = read_json(path)
    if not isinstance(saved, dict) or not isinstance(saved.get('cases'), list):
        raise ValueError('Invalid Minerva reference')
    plan = list(BASELINE.values())
    rows = []
    for source in saved['cases']:
        row = dict(source, dataset='baseline', inference_attempted=True)
        row['status'] = ('complete' if row.get('engine_status') == 'complete' else 'infrastructure_error')
        rows.append(row)
    if len(rows) != 70 or {r.get('id') for r in rows} != set(BASELINE):
        raise ValueError('Minerva reference must contain all 70 original cases')
    summary = validate_rows(plan, rows)
    return dict(name='Minerva · riferimento originale', model=saved.get('model'),
                profile=saved.get('evaluation_profile'), protocol={'output_tokens': 768},
                source=str(Path(path).resolve()), input_hashes={'corpus': saved.get('corpus_sha256')},
                plan=plan, rows=[public_row(r) for r in rows], summary=summary)


def paired_fresh(groups):
    """Pair only equal 128-token settings and identical sets of 14 fresh cases."""
    results = []
    for left, right in combinations(groups, 2):
        lp, rp = left['protocol'], right['protocol']
        if (lp.get('reasoning_budget') != 128 or rp.get('reasoning_budget') != 128 or
                any(k not in lp or k not in rp or lp[k] != rp[k] for k in PAIR_SETTINGS)):
            continue
        plans = [{c['id']: c for c in g['plan'] if c['dataset'] == 'fresh'} for g in (left, right)]
        if (any(len(p) != 14 or Counter(c['part'] for c in p.values()) != {i: 2 for i in range(1, 8)} for p in plans) or
                set(plans[0]) != set(plans[1]) or
                any(identity(plans[0][key]) != identity(plans[1][key]) for key in plans[0])):
            continue
        valid = [{r['id']: r for r in g['rows'] if r['status'] == 'complete' and r['observed'] in LABELS and r['id'] in plans[0]}
                 for g in (left, right)]
        common = sorted(set(valid[0]) & set(valid[1]))
        common_plan = [plans[0][key] for key in common]
        results.append(dict(left_name=left['name'], right_name=right['name'],
                            profiles=[left['profile'], right['profile']], planned_common=14,
                            completed_pairs=len(common), paired_case_ids=common,
                            missing_pair_ids=sorted(set(plans[0]) - set(common)),
                            missing_by_group={g['name']: sorted(set(plans[0]) - set(v)) for g, v in zip((left, right), valid)},
                            settings={k: lp[k] for k in PAIR_SETTINGS},
                            part_time_caps_seconds=[lp.get('per_part_seconds'), rp.get('per_part_seconds')],
                            left=summarize(common_plan, [valid[0][key] for key in common]),
                            right=summarize(common_plan, [valid[1][key] for key in common])))
    return results


def escaped(value):
    return html.escape(str(value), quote=True)


def percentage(value):
    return 'non disponibile' if value is None else f'{100 * value:.1f}%'


def metric_cells(metrics):
    numbers = ('planned', 'valid', 'missing', 'infrastructure_errors', 'invalid')
    rates = ('agreement_valid', 'agreement_planned', 'error_recall_planned', 'false_alarm_rate_valid_correct')
    return ''.join('<td>' + escaped(metrics[k]) + '</td>' for k in numbers) + ''.join(
        '<td>' + escaped(percentage(metrics[k])) + '</td>' for k in rates) + '<td>' + escaped(metrics['false_reassurance_count']) + '</td>'


def render_html(groups, pairs):
    out = ['''<!doctype html><html lang="it"><head><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; img-src data:">
<title>Confronto dei modelli locali</title><style>body{max-width:1440px;margin:28px auto;padding:0 18px;font:15px/1.5 Arial,sans-serif;color:#15202b;background:#fff}table{border-collapse:collapse;width:100%;margin:14px 0}th,td{border:1px solid #aaa;padding:8px;vertical-align:top;text-align:left}th{background:#eef2f5}pre{white-space:pre-wrap;overflow-wrap:anywhere}details{border:1px solid #bbb;padding:10px;margin:9px 0}.scroll{overflow-x:auto}@media print{details{break-inside:avoid}body{font-size:10pt}}</style></head><body>
<h1>Confronto dei modelli locali</h1>
<p>Metriche ricalcolate dai casi pianificati e dalle risposte salvate. Ogni gruppo conserva il proprio profilo; i conteggi dichiarati nei riepiloghi originali non determinano i risultati.</p>
<p>Concordanza sui validi = etichette concordanti / risposte complete con etichetta valida. Concordanza sui pianificati = etichette concordanti / tutti i casi pianificati: è una misura conservativa della copertura e della concordanza insieme. Mancanti, errori di infrastruttura e formato non sono predizioni errate. Una divisione senza denominatore è indicata come non disponibile.</p>
<p>Errori riconosciuti sui pianificati = casi attesi errata riconosciuti / tutti i casi attesi errata pianificati. Falsi allarmi sui corretti validi = etichette errata / risposte valide attese coerente. Rassicurazioni erronee = etichette coerente quando l’atteso è diverso.</p>
<div class="scroll"><table><thead><tr><th>Gruppo / insieme</th><th>Pianificati</th><th>Validi</th><th>Mancanti</th><th>Infrastruttura</th><th>Formato</th><th>Concordanza sui validi</th><th>Concordanza sui pianificati</th><th>Errori riconosciuti sui pianificati</th><th>Falsi allarmi sui corretti validi</th><th>Rassicurazioni erronee</th></tr></thead><tbody>''']
    for group in groups:
        for dataset, metrics in [('totale', group['summary']), *group['summary']['by_dataset'].items()]:
            out.append('<tr><th>' + escaped(group['name']) + ' / ' + escaped(dataset) + '</th>' + metric_cells(metrics) + '</tr>')
    out.append('''</tbody></table></div><h2>Limiti di interpretazione</h2>
<p>I casi e le etichette attese sono sintetici, redatti dall’assistente; non costituiscono una validazione esterna indipendente del docente. Questa griglia di valutazione non produce voti, punteggi di frode o giudizi sulla paternità dei testi. Non certifica l’idoneità didattica di un modello.</p>
<p>Quantizzazioni diverse e limiti temporali per parte anche diversi impediscono di interpretare il confronto come una prova controllata dell’hardware. Il riferimento Minerva originale prevedeva 768 token di output; i nuovi profili prevedono 1024 token. Profili e copertura devono accompagnare ogni percentuale.</p>
<p>Il testo grezzo dei ragionamenti interni non è riprodotto. Le richieste e le risposte grezze restano negli artefatti JSON originali; qui sono riportati soltanto gli esiti, i motivi finali e gli indicatori di ragionamento osservato.</p>
<h2>Confronti appaiati sui 14 casi nuovi con ragionamento 128</h2>
<p>Sono ammessi soltanto gruppi con budget di ragionamento 128, identici contesto, temperatura, seme e output, e identiche domande, testi ed etichette dei 14 casi nuovi. Le metriche appaiate riguardano esclusivamente l’intersezione delle risposte complete: una copertura parziale non rappresenta tutti i 14 casi. I limiti temporali restano visibili e possono differire.</p>''')
    if not pairs:
        out.append('<p>Nessuna coppia di gruppi soddisfa i requisiti.</p>')
    for pair in pairs:
        out.append('<h3>' + escaped(pair['left_name']) + ' / ' + escaped(pair['right_name']) + '</h3><p>Coppie complete: ' + escaped(pair['completed_pairs']) + '/14. Limiti per parte, secondi: ' + escaped(pair['part_time_caps_seconds']) + '. Casi fuori dall’intersezione: ' + escaped(', '.join(pair['missing_pair_ids']) or 'nessuno') + '.</p>')
        for side, name in [('left', pair['left_name']), ('right', pair['right_name'])]:
            out.append('<p>' + escaped(name) + ': concordanza nell’intersezione ' + escaped(percentage(pair[side]['agreement_valid'])) + '; errori riconosciuti nell’intersezione ' + escaped(percentage(pair[side]['error_recall_planned'])) + '.</p>')
    for group in groups:
        out.append('<h2>' + escaped(group['name']) + '</h2><p>Modello: ' + escaped(group['model']) + '. Profilo: ' + escaped(group['profile']) + '.</p><p>Artefatti originali: ' + escaped(group['source']) + '</p><pre>' + escaped(json.dumps(group['protocol'], ensure_ascii=False, indent=2)) + '</pre>')
        by_id = {r['id']: r for r in group['rows']}
        for case in group['plan']:
            row = by_id.get(case['id'])
            title = escaped(case['id']) + ' · parte ' + escaped(case['part']) + ' · ' + escaped(row['status'] if row else 'Mancante')
            out.append('<details><summary>' + title + '</summary><p>Atteso: ' + escaped(case['expected_assessment']) + '; osservato: ' + escaped(row['observed'] if row and row['observed'] else 'non disponibile') + '.</p><p><strong>Criterio:</strong> ' + escaped(case['criterion']) + '</p><p><strong>Testo:</strong> ' + escaped(case['text']) + '</p><p><strong>Motivo atteso:</strong> ' + escaped(case.get('rationale', 'non disponibile')) + '</p>')
            if row:
                out.append('<p><strong>Motivo del modello:</strong> ' + escaped(row['reason'] or 'non disponibile') + '</p><p>Secondi: ' + escaped(row.get('seconds')) + '; termine della generazione: ' + escaped(row.get('finish_reason') or 'non disponibile') + '; ragionamento nativo osservato: ' + escaped('sì' if row.get('native_reasoning_observed') else 'no / non registrato') + '.</p><pre>' + escaped(json.dumps({'consumo_token': row.get('usage'), 'codice_errore': row.get('error_code')}, ensure_ascii=False, indent=2)) + '</pre>')
            out.append('</details>')
    return ''.join(out) + '</body></html>'


def main(argv=None):
    parser = argparse.ArgumentParser(description='Render an offline report from saved comparison JSON.')
    parser.add_argument('--result', action='append', default=[], metavar='NAME=DIR')
    parser.add_argument('--minerva', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    groups = []
    names = set()
    for value in args.result:
        if '=' not in value:
            parser.error('--result requires NAME=DIR')
        name, directory = value.split('=', 1)
        if not name.strip() or not directory or name in names:
            parser.error('Result names must be nonempty and unique')
        names.add(name)
        groups.append(load_result(name, directory))
    if args.minerva:
        reference = load_minerva(args.minerva)
        if reference['name'] in names:
            parser.error('Minerva reference name duplicates a result name')
        groups.append(reference)
    if not groups:
        parser.error('At least one result or Minerva reference is required')
    if args.output.suffix.lower() not in ('.html', '.htm'):
        parser.error('--output must have an HTML extension')
    consumed = set()
    for group in groups:
        source = Path(group['source'])
        if source.is_dir():
            files = ('summary.json', 'planned_cases.json', 'frozen_fresh.json',
                     *(f'part_{part}.json' for part in range(1, 8)))
            consumed.update((source / filename).resolve() for filename in files)
        else:
            consumed.add(source.resolve())
    destinations = {args.output.resolve(), args.output.with_suffix('.json').resolve()}
    if destinations & consumed:
        raise ValueError('Report outputs must not overwrite input artifacts')
    pairs = paired_fresh(groups)
    page = render_html(groups, pairs)
    sidecar = dict(schema_version=1, groups=groups, paired_fresh=pairs,
                   limits='Etichette sintetiche redatte dall’assistente; nessuna validazione indipendente del docente; nessun punteggio di paternità o frode.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(page, encoding='utf-8')
    args.output.with_suffix('.json').write_text(json.dumps(sidecar, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
