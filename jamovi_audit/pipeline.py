"""Read-only local orchestration of assignment checks."""
from pathlib import Path
import json
from . import __version__
from .archive import safe_zip, inventory
from .readers import read_omv, read_docx
from .checks import check_submission
from .report import compare_submissions, render_report


def load_reference():
    path = Path(__file__).resolve().parent.parent / 'reference' / 'reference.json'
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def audit_zip(data: bytes, progress=None, cancelled=None) -> dict:
    """Audit a class ZIP without modifying any supplied bytes."""
    def check_cancelled():
        if cancelled is not None and cancelled():
            raise InterruptedError('Controllo annullato')

    check_cancelled()
    entries = safe_zip(data, 'ZIP delle consegne')
    submissions = inventory(entries)
    if not submissions:
        raise ValueError('Nessuna consegna OMV o Word trovata nello ZIP.')
    reference = load_reference()
    reports = []
    parsed = {}
    if reference:
        parsed['assignment_text'] = reference.get('assignment_text', [])
    for index, submission in enumerate(submissions):
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
        report = check_submission(submission, documents['omv'], documents['docx'], reference)
        reports.append(report)
        parsed[submission['key']] = documents
    check_cancelled()
    comparisons = compare_submissions(reports, parsed)
    summary = {'student_count': len(reports),
               'file_count': sum(len(s.get('omv', [])) + len(s.get('docx', [])) for s in submissions),
               'version': __version__,
               'reference_available': reference is not None}
    html = render_report(reports, comparisons, summary)
    if progress is not None:
        progress(len(submissions), len(submissions), 'Rapporto pronto')
    return {'reports': reports, 'comparisons': comparisons, 'html': html, 'summary': summary}
