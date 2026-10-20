"""Read-only local orchestration of assignment checks."""
from pathlib import Path
from collections import Counter
import json
from . import __version__
from .archive import safe_zip, inventory
from .readers import read_omv, read_docx
from .checks import check_submission
from .report import ComparisonResults, compare_submissions, render_report
from .local_llm import assess_relation, validate_config, statistical_context
from .metadata import extract_zip_metadata, inspect_document_metadata, metadata_anomalies, metadata_cross_checks, CLOSE_SECONDS


def load_reference():
    path = Path(__file__).resolve().parent.parent / 'reference' / 'reference.json'
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def audit_zip(data: bytes, progress=None, cancelled=None, linguistic_config=None) -> dict:
    """Audit a class ZIP without modifying any supplied bytes."""
    def check_cancelled():
        if cancelled is not None and cancelled():
            raise InterruptedError('Controllo annullato')

    linguistic_config = validate_config(linguistic_config)
    minerva_profile = (linguistic_config.get('enabled') and linguistic_config['provider'] == 'llamacpp'
                       and linguistic_config['model'] == 'minerva-7b-instruct-q4-local')
    check_cancelled()
    entries = safe_zip(data, 'ZIP delle consegne')
    source_zip_metadata = extract_zip_metadata(data)
    submissions = inventory(entries)
    if not submissions:
        raise ValueError('Nessuna consegna OMV o Word trovata nello ZIP.')
    reference = load_reference()
    reports = []
    parsed = {}
    interrupted = False
    def cancelled_after_minerva_review():
        return (minerva_profile and reports and cancelled is not None and cancelled() and
                reports[-1].get('linguistic', {}).get('status') in ('complete', 'partial', 'cancelled'))
    if reference:
        parsed['assignment_text'] = reference.get('assignment_text', [])
    for index, submission in enumerate(submissions):
        if cancelled_after_minerva_review():
            interrupted = True
            break
        check_cancelled()
        if progress is not None:
            progress(index, len(submissions), submission['label'])
        documents = {}
        for kind, reader in [('omv', read_omv), ('docx', read_docx)]:
            paths = submission.get(kind, [])
            if len(paths) != 1:
                documents[kind] = None
                continue
            try:
                documents[kind] = reader(entries[paths[0]])
            except Exception as exc:
                # A single damaged document must not hide other students.
                documents[kind] = {'errors': [f'Lettura non riuscita: {type(exc).__name__}: {exc}']}
        check_cancelled()
        document_metadata = {}
        for kind in ('omv', 'docx'):
            paths = submission.get(kind, [])
            source_clock = source_zip_metadata.get(paths[0]) if len(paths) == 1 else None
            document = documents[kind]
            if isinstance(document, dict) and len(paths) == 1:
                document['zip_metadata'] = extract_zip_metadata(entries[paths[0]])
            observed = inspect_document_metadata(document, kind, source_clock)
            document_metadata[kind] = observed
            if isinstance(document, dict):
                document['metadata_summary'] = observed
        report = check_submission(submission, documents['omv'], documents['docx'], reference)
        report['metadata'] = document_metadata
        for kind, observed in document_metadata.items():
            for anomaly in metadata_anomalies(observed):
                paths = submission.get(kind, [])
                location = (paths[0] + ' / ' if len(paths) == 1 else '') + anomaly.get('location', '')
                report['findings'].append(dict(anomaly, location=location))
        # Metadata review findings belong to coverage without increasing the
        # count of automatically decided educational requirements.
        counts = Counter(f['status'] for f in report['findings'])
        report['coverage']['total'] = len(report['findings'])
        report['coverage']['by_status'] = dict(counts)
        if linguistic_config.get('enabled'):
            if progress is not None:
                progress(index, len(submissions), submission['label'] + ' — valutazione linguistica locale')
            def linguistic_progress(done, total, label):
                progress(done, total, submission['label'] + ' — ' + label)
            report['linguistic'] = assess_relation(
                documents['docx'], dict(report, statistical_context=statistical_context(documents['omv'])), reference=reference,
                config=linguistic_config, cancelled=cancelled,
                progress=linguistic_progress if progress is not None else None)
            review_status = report['linguistic'].get('status')
            if minerva_profile and (review_status == 'cancelled' or
                    (review_status in ('complete', 'partial') and cancelled is not None and cancelled())):
                interrupted = True
            else:
                check_cancelled()
        reports.append(report)
        parsed[submission['key']] = documents
        if interrupted:
            break
    if cancelled_after_minerva_review():
        interrupted = True
    if interrupted:
        comparisons = ComparisonResults()
        comparisons.diagnostics = {'status': 'not_run_cancelled',
                                   'reason': 'Confronti tra studenti non eseguiti per annullamento.'}
    else:
        check_cancelled()
        comparisons = compare_submissions(reports, parsed)
        check_cancelled()
        comparisons.extend(metadata_cross_checks(reports, parsed, comparisons, reference))
    # Preserve the indexed content-comparison diagnostics carried by the list.
    if hasattr(comparisons, 'diagnostics'):
        comparisons.diagnostics['pairs_flagged_by_rule'] = {
            kind: len({tuple(c['students']) for c in comparisons if c['kind'] == kind})
            for kind in sorted({c['kind'] for c in comparisons})
        }
        comparisons.diagnostics['metadata'] = {
            'close_seconds': CLOSE_SECONDS, 'timestamp_only_candidates': False,
            'zip_timezone': 'unknown', 'zip_precision_seconds': 2,
            'filesystem_times_used': False,
        }
    summary = {'student_count': len(reports),
               'planned_student_count': len(submissions),
               'unexecuted_student_count': len(submissions) - len(reports),
               'status': 'cancelled' if interrupted else 'complete',
               'incomplete': interrupted,
               'comparison_status': 'not_run_cancelled' if interrupted else 'complete',
               'file_count': sum(len(s.get('omv', [])) + len(s.get('docx', [])) for s in submissions[:len(reports)]),
               'version': __version__,
               'reference_available': reference is not None}
    summary['linguistic'] = ('enabled' if linguistic_config.get('enabled') else 'disabled')
    html = render_report(reports, comparisons, summary)
    if progress is not None:
        progress(len(reports), len(submissions), 'Rapporto incompleto disponibile' if interrupted else 'Rapporto pronto')
    return {'reports': reports, 'comparisons': comparisons, 'html': html, 'summary': summary}
