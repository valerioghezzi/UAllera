"""Offline development benchmark of the deterministic audit pipeline.

Uses only synthetic ZIP/OMV/DOCX bytes and the Python standard library. It never
starts a model, downloads assets, or certifies throughput. Example:
    python tools/benchmark_audit.py --sizes 10 100 --seed 1729 --output benchmark.json

Timing covers audit_zip (including report rendering), excludes fixture creation,
and includes tracemalloc overhead. Peak is traced Python allocation, not RSS.
"""
from pathlib import Path
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import platform
import random
import struct
import sys
import time
import tracemalloc
import zipfile
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from jamovi_audit import __version__
from jamovi_audit.pipeline import audit_zip

MAX_STUDENTS = 1000
MAX_SYNTHETIC_ZIP_BYTES = 64 * 1024 * 1024
SCENARIOS = ('unique', 'mixed_duplicates')
FIXTURE_VERSION = 'synthetic-cohort-1'


def _archive(entries):
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for name in sorted(entries):
            info = zipfile.ZipInfo(name, (2024, 1, 2, 3, 4, 6))
            info.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(info, entries[name])
            if output.tell() > MAX_SYNTHETIC_ZIP_BYTES:
                raise ValueError('Synthetic ZIP exceeds the benchmark byte budget')
    data = output.getvalue()
    if len(data) > MAX_SYNTHETIC_ZIP_BYTES:
        raise ValueError('Synthetic ZIP exceeds the benchmark byte budget')
    return data


def _protobuf(number, value):
    def varint(value):
        output = bytearray()
        while value > 127:
            output.append((value & 127) | 128)
            value >>= 7
        output.append(value)
        return bytes(output)
    if isinstance(value, int):
        return varint(number << 3) + varint(value)
    if isinstance(value, float):
        return varint(number << 3 | 1) + struct.pack('<d', value)
    if isinstance(value, str):
        value = value.encode('utf-8')
    return varint(number << 3 | 2) + varint(len(value)) + value


def _student_documents(index, rng):
    # Actual numeric columns and a saved protobuf table exercise both readers.
    rows = 80
    values = [round(rng.uniform(1, 5), 4) for _ in range(rows)]
    ages = [rng.randint(18, 70) for _ in range(rows)]
    fields = [{'name': 'age', 'dataType': 'Integer', 'measureType': 'Continuous'},
              {'name': 'N', 'dataType': 'Decimal', 'measureType': 'Continuous'}]
    mean = sum(values) / rows
    table = _protobuf(1, 'mean') + _protobuf(2, 'Mean') + _protobuf(7, _protobuf(2, mean))
    result = _protobuf(1, 'descriptives') + _protobuf(3, 3) + _protobuf(6, _protobuf(1, table))
    analysis = (_protobuf(2, 1) + _protobuf(3, 'descriptives') + _protobuf(4, 'jmv') +
                _protobuf(7, result) + _protobuf(8, 3))
    omv = _archive({
        'meta': 'jamovi-Archive-Version: 11.0\nCreated-By: synthetic benchmark\n',
        'metadata.json': json.dumps({'dataSet': {'rowCount': rows, 'fields': fields}, 'synthetic_id': index}),
        'data.bin': struct.pack('<' + 'i' * rows, *ages) + struct.pack('<' + 'd' * rows, *values),
        'xdata.json': '{}', '01 descriptives/analysis': analysis,
    })
    # Independent synthetic lexical markers avoid accidentally benchmarking a
    # class-wide prose duplicate. Shared required headings/numbers stay present.
    paragraphs = []
    for part in range(1, 8):
        markers = [''.join(rng.choices('abcdefghijklmnopqrstuvwxyz', k=12)) for _ in range(24)]
        paragraphs.extend([f'Parte {part}',
                           f'Media {mean:.4f} deviazione standard {rng.uniform(.5, 1.5):.4f} campione {rows}.',
                           ' '.join(markers) + '.'])
    body = ''.join('<w:p><w:r><w:rPr><w:rFonts w:ascii="Times New Roman"/>'
                   '<w:sz w:val="24"/></w:rPr><w:t xml:space="preserve">' + escape(text) +
                   '</w:t></w:r></w:p>' for text in paragraphs)
    word = _archive({
        'word/document.xml': '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>' + body + '</w:body></w:document>',
        'word/styles.xml': '<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:docDefaults><w:pPrDefault><w:pPr><w:spacing w:line="480"/></w:pPr></w:pPrDefault></w:docDefaults></w:styles>',
    })
    return omv, word


def synthetic_delivery(student_count, seed=1729, scenario='unique'):
    """Deterministic cohort; mixed mode duplicates students 0/1, 5/6, ... ."""
    if type(student_count) is not int or not 1 <= student_count <= MAX_STUDENTS:
        raise ValueError(f'Student count must be 1..{MAX_STUDENTS}')
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError('Seed must be an integer in 0..4294967295')
    if scenario not in SCENARIOS:
        raise ValueError('Unknown synthetic scenario')
    rng = random.Random(seed)
    entries = {}
    previous = None
    for index in range(student_count):
        current = _student_documents(index, rng)
        if scenario == 'mixed_duplicates' and index % 5 == 1:
            current = previous
        for suffix, content in zip(('omv', 'docx'), current):
            entries[f'Classe_sintetica/Persona_{index:04d}/elaborato.{suffix}'] = content
        previous = current
    return _archive(entries)


def run_benchmarks(sizes=(10, 100), seed=1729, scenarios=SCENARIOS):
    """Return observed measurements, provenance and adjacent-size scaling ratios."""
    sizes = list(sizes)
    scenarios = tuple(scenarios)
    if not sizes or len(sizes) > 8 or len(set(sizes)) != len(sizes):
        raise ValueError('Supply 1..8 distinct cohort sizes')
    if not scenarios or any(s not in SCENARIOS for s in scenarios):
        raise ValueError('Supply known benchmark scenarios')
    if tracemalloc.is_tracing():
        raise RuntimeError('Run the benchmark with no existing tracemalloc session')
    source_hash = hashlib.sha256()
    for path in sorted((ROOT / 'jamovi_audit').glob('*.py')) + [Path(__file__)]:
        source_hash.update(path.name.encode('utf-8'))
        source_hash.update(path.read_bytes())
    reference_path = ROOT / 'reference' / 'reference.json'
    reference_digest = hashlib.sha256(reference_path.read_bytes()).hexdigest() if reference_path.is_file() else None
    result = {'schema_version': 1, 'fixture_version': FIXTURE_VERSION,
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'software_version': __version__, 'source_sha256': source_hash.hexdigest(),
              'python': sys.version, 'platform': platform.platform(), 'machine': platform.machine(),
              'seed': seed, 'linguistic_inference': 'disabled', 'runs': [], 'scaling': [],
              'reference': {'available': reference_digest is not None, 'sha256': reference_digest},
              'measurement': {'scope': 'audit_zip including HTML; fixture generation excluded',
                              'memory': 'peak traced Python allocations; excludes fixture bytes, not process RSS',
                              'repetitions_per_case': 1, 'tracemalloc_overhead_included': True},
              'limits': ['Synthetic workloads and one run per case; not a throughput certification.',
                         'Saved analysis is a small descriptive table; Word uses synthetic lexical markers.',
                         'No model inference, network, downloads, native-memory or Windows build measurement.']}
    for scenario in scenarios:
        previous = None
        for size in sizes:
            data = synthetic_delivery(size, seed=seed, scenario=scenario)
            tracemalloc.start()
            started = time.perf_counter()
            try:
                audited = audit_zip(data, linguistic_config={'enabled': False})
                elapsed = time.perf_counter() - started
                _, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            diagnostics = getattr(audited['comparisons'], 'diagnostics', {})
            cache_stats = audited.get('cache_stats', audited.get('summary', {}).get('cache_stats'))
            run = {'scenario': scenario, 'requested_students': size,
                   'student_count': audited['summary']['student_count'],
                   'reference_available': audited['summary']['reference_available'],
                   'input_bytes': len(data), 'input_sha256': hashlib.sha256(data).hexdigest(),
                   'wall_seconds': elapsed, 'peak_tracemalloc_bytes': peak,
                   'findings': sum(len(r.get('findings', [])) for r in audited['reports']),
                   'comparison_findings': len(audited['comparisons']),
                   'comparison_kinds': dict(Counter(c['kind'] for c in audited['comparisons'])),
                   'comparison_diagnostics': diagnostics, 'cache_stats': cache_stats,
                   'cache_stats_available': cache_stats is not None}
            result['runs'].append(run)
            if previous:
                result['scaling'].append({'scenario': scenario,
                    'from_students': previous['student_count'], 'to_students': run['student_count'],
                    'cohort_ratio': run['student_count'] / previous['student_count'],
                    'wall_ratio': elapsed / previous['wall_seconds'] if previous['wall_seconds'] else None,
                    'traced_peak_ratio': peak / previous['peak_tracemalloc_bytes'] if previous['peak_tracemalloc_bytes'] else None})
            previous = run
            del audited
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sizes', nargs='+', type=int, default=[10, 100], help='1..8 distinct sizes, each 1..1000')
    parser.add_argument('--output', type=Path, default=Path('benchmark_artifacts/benchmark.json'))
    parser.add_argument('--seed', type=int, default=1729)
    args = parser.parse_args(argv)
    if (not 1 <= len(args.sizes) <= 8 or len(set(args.sizes)) != len(args.sizes) or
            any(not 1 <= n <= MAX_STUDENTS for n in args.sizes)):
        parser.error('--sizes requires 1..8 distinct integers in 1..1000')
    if not 0 <= args.seed <= 2**32 - 1:
        parser.error('--seed must be in 0..4294967295')
    return args


def main(argv=None):
    args = parse_args(argv)
    result = run_benchmarks(args.sizes, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for run in result['runs']:
        print(f"{run['scenario']}: {run['student_count']} students, {run['wall_seconds']:.3f}s, "
              f"{run['peak_tracemalloc_bytes']} peak traced bytes, {run['comparison_findings']} comparisons")
    print(f'Observed results: {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
