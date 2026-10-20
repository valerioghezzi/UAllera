"""Local, repeatable linguistic evaluation against a separate synthetic corpus."""
from datetime import datetime, timezone
from collections import Counter
import hashlib
import html
import json
import time
from .local_llm import assess_relation, validate_config, RUBRIC_VERSION

LABELS = ('coerente', 'errata', 'incompleta', 'non_valutabile')
LIMITS = [
    'Casi sintetici redatti dall’assistente: le etichette devono essere approvate dal docente.',
    'Testi separati dalle dimostrazioni del prompt; stessi argomenti e autore. Non è una validazione esterna indipendente.',
    'Non misura la sensibilità alla frode, la paternità del testo o la capacità di scoprire copiature.',
    'Il risultato descrive soltanto questo modello, questa rubrica e questi casi; non certifica le consegne reali.',
    'Una citazione valida attesta la presenza del testo, non la correttezza del ragionamento del modello.',
    'Valuta una domanda isolata con i criteri e cinque esempi della parte pertinente; non misura la revisione di un documento intero.',
]


def _metrics(rows):
    valid = [r for r in rows if r['observed'] in LABELS]
    matching = sum(r['observed'] == r['expected_assessment'] for r in valid)
    completed = [r for r in rows if r['engine_status'] == 'complete']
    errors = [r for r in completed if r['expected_assessment'] == 'errata']
    correct = [r for r in completed if r['expected_assessment'] == 'coerente']
    def ratio(n, d):
        return n/d if d else None
    confusion = {expected: dict(Counter(r['observed'] or 'senza_esito' for r in rows if r['expected_assessment'] == expected)) for expected in LABELS}
    return dict(matching_labels=matching, agreement_valid=ratio(matching, len(valid)),
                agreement_attempted=ratio(matching, len(rows)) if completed else None, valid_coverage=ratio(len(valid), len(rows)),
                error_recall=ratio(sum(r['observed'] == 'errata' for r in errors), len(errors)),
                false_alarm_rate=ratio(sum(r['observed'] == 'errata' for r in correct), len(correct)),
                false_reassurance_count=sum(r['observed'] == 'coerente' and r['expected_assessment'] != 'coerente' for r in rows),
                confusion=confusion)


def run_validation(config, cases=None, progress=None, cancelled=None):
    """Run cases locally. Never send expected labels/rationales to the model."""
    from .validation_cases import VALIDATION_CASES, VALIDATION_VERSION
    normalized = validate_config(config)
    if not normalized['enabled'] or not normalized['model']:
        raise ValueError('Seleziona prima un motore e un modello locale installato.')
    selected = list(VALIDATION_CASES if cases is None else cases)
    if not selected:
        raise ValueError('Il corpus di validazione è vuoto.')
    for case in selected:
        if not isinstance(case, dict) or case.get('expected_assessment') not in LABELS or type(case.get('part')) is not int or not 1 <= case['part'] <= 7 or not isinstance(case.get('text'), str) or not case['text'].strip():
            raise ValueError('Caso di validazione non conforme.')
    corpus_json = json.dumps(selected, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    start = time.monotonic()
    result = dict(status='complete', provider=normalized['provider'], model=normalized['model'],
                  endpoint=normalized['endpoint'], timeout=normalized['timeout'], rubric_version=RUBRIC_VERSION, corpus_version=VALIDATION_VERSION,
                  evaluation_profile=('minerva-single-question-evidence-1.0' if normalized['provider']=='llamacpp' and normalized['model']=='minerva-7b-instruct-q4-local' else 'single-question-1.1'),
                  corpus_sha256=hashlib.sha256(corpus_json.encode()).hexdigest(),
                  started_utc=datetime.now(timezone.utc).isoformat(), planned=len(selected), cases=[], limits=list(LIMITS))
    for case in selected:
        if cancelled and cancelled():
            result['status'] = 'cancelled'
            break
        before = time.monotonic()
        if progress:
            progress(len(result['cases']), len(selected), case['id'])
        # Only question/criterion and the synthetic text are model inputs.
        reference = {'linguistic_rubric': f"Valuta solo la parte {case['part']} rispetto a questo quesito: {case.get('criterion', 'Valutare il significato espresso.')}. Non richiedere altre parti o risultati non forniti. Titoli e comandi non sono risposte."}
        document = {'paragraphs': [{'index': 0, 'location': 'Paragrafo 1', 'text': case['text']}]}
        report = {'findings': [], 'statistical_context': case.get('statistical_context', [])}
        review = assess_relation(document, report, reference, normalized, cancelled=cancelled,
                                 target_part=case['part'], criterion=case.get('criterion', 'Valutare il significato espresso.'))
        findings = [f for f in review.get('findings', []) if f.get('part') == case['part']]
        predictions = {f.get('proposal') for f in findings if f.get('proposal') in LABELS}
        all_citable = bool(findings) and all(f.get('proposal') in LABELS for f in findings)
        observed = next(iter(predictions)) if all_citable and len(predictions) == 1 and review['status'] == 'complete' else None
        result['cases'].append(dict(case, observed=observed, engine_status=review['status'],
                                    matches_expected=observed == case['expected_assessment'],
                                    seconds=round(time.monotonic()-before, 3), findings=findings, limits=review.get('limits', []), protocol=review.get('protocol'), coverage=review.get('coverage')))
        if review['status'] in ('unavailable', 'cancelled'):
            result['status'] = review['status']
            break
    rows = result['cases']
    result.update(attempted=len(rows), evaluated=sum(r['observed'] in LABELS for r in rows),
                  unexecuted=len(selected)-len(rows), seconds=round(time.monotonic()-start, 3), metrics=_metrics(rows))
    result['by_part'] = {str(part): {'attempted': len([r for r in rows if r['part'] == part]),
                                   **_metrics([r for r in rows if r['part'] == part])} for part in range(1, 8)}
    if progress:
        progress(len(rows), len(selected), 'Validazione terminata')
    return result


def render_validation(result):
    """One offline HTML with auditable labels, predictions and all limitations."""
    e = lambda value: html.escape(str(value), quote=True)
    def pct(value):
        return 'Non disponibile' if value is None else f'{100*value:.1f}%'
    metrics = result['metrics']
    out = ['<!doctype html><html lang="it"><head><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'unsafe-inline\'; img-src data:"><title>Validazione linguistica UAllera</title><style>body{max-width:1200px;margin:32px auto;padding:0 20px;font:16px/1.6 Verdana,sans-serif;color:#111;background:white}table{border-collapse:collapse;width:100%;margin:20px 0}td,th{border:1px solid #777;padding:9px;text-align:left;vertical-align:top}th{background:#eee}pre{white-space:pre-wrap;overflow-wrap:anywhere}details{margin:12px 0}h1{font-size:28px}@media print{body{font-size:11pt}tr{break-inside:avoid}}</style></head><body><h1>Validazione linguistica del modello locale</h1>',
           '<p>Prova su casi sintetici separati dagli esempi del prompt. Nessun voto automatico o certificazione di idoneità del modello.</p>',
           f"<p>Motore: {e(result['provider'])}; modello: {e(result['model'])}; stato: {e(result['status'])}. Avvio UTC: {e(result['started_utc'])}.</p>",
           f"<p>Profilo di valutazione: {e(result.get('evaluation_profile', 'documento completo, versione precedente'))}.</p>",
           f"<p>Griglia di valutazione: {e(result['rubric_version'])}; corpus: {e(result['corpus_version'])}. SHA-256 corpus: <code>{e(result['corpus_sha256'])}</code>.</p>",
           f"<p>Casi previsti: {result['planned']}; tentati: {result['attempted']}; con esito citabile: {result['evaluated']}; non eseguiti: {result['unexecuted']}. Durata: {result['seconds']} secondi.</p>",
           '<ul>' + ''.join('<li>'+e(item)+'</li>' for item in result['limits']) + '</ul>',
           '<table><thead><tr><th>Misura</th><th>Valore</th><th>Significato</th></tr></thead><tbody>']
    for label, value, meaning in [
        ('Concordanza sugli esiti citabili', pct(metrics['agreement_valid']), 'Concordanti / esiti citabili; leggere insieme alla copertura.'),
        ('Concordanza sui tentativi', pct(metrics['agreement_attempted']), 'Concordanti / casi tentati; senza esito rimangono nel denominatore.'),
        ('Copertura', pct(metrics['valid_coverage']), 'Esiti citabili / casi tentati.'),
        ('Errori concettuali riconosciuti', pct(metrics['error_recall']), 'Etichetta errata correttamente riconosciuta / casi errati con risposta del motore; non misura frodi.'),
        ('Risposte corrette segnalate errate', pct(metrics['false_alarm_rate']), 'Etichetta errata / casi corretti con risposta del motore.'),
        ('Rassicurazioni erronee', metrics['false_reassurance_count'], 'Proposta coerente quando l’atteso non è coerente.')]:
        out.append(f'<tr><td>{e(label)}</td><td>{e(value)}</td><td>{e(meaning)}</td></tr>')
    out.append('</tbody></table><h2>Dettaglio dei casi tentati</h2>')
    for row in result['cases']:
        out.append(f"<details open><summary>{e(row['id'])} · parte {row['part']} · atteso {e(row['expected_assessment'])} · osservato {e(row['observed'] or 'senza esito')}</summary><p><strong>Quesito:</strong> {e(row.get('criterion',''))}</p><p><strong>Testo:</strong> {e(row['text'])}</p><p><strong>Motivo dell’etichetta attesa:</strong> {e(row.get('rationale','Da controllare con il docente.'))}</p><p>Stato motore: {e(row['engine_status'])}; {row['seconds']} secondi.</p><pre>{e(json.dumps(row['findings'],ensure_ascii=False,indent=2))}</pre><p>{e(' '.join(row['limits']))}</p></details>")
    out.append('<h2>Matrice delle etichette</h2><pre>'+e(json.dumps(metrics['confusion'],ensure_ascii=False,indent=2))+'</pre></body></html>')
    return ''.join(out)
