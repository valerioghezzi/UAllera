"""Configure the pinned Windows engine using structured argv and verify ISA.

No compilation or model execution. CMake cache, effective DEBUG output and
the generated CPU compiler project must all agree before configuration passes.
The workflow checks out the upstream commit; this tool records that expected
pin and does not claim to have independently verified the checkout identity.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

SOURCE_COMMIT = 'd80619397c2ba5cf1366c29f7d43699ca3a159b9'
MAX_LOG_BYTES = 1024*1024
STATIC = {
    'BUILD_SHARED_LIBS': 'OFF', 'CMAKE_MSVC_RUNTIME_LIBRARY': 'MultiThreaded',
    'GGML_BACKEND_DL': 'OFF', 'GGML_OPENMP': 'OFF', 'GGML_OPENMP_FETCH': 'OFF',
    'LLAMA_OPENSSL': 'OFF', 'LLAMA_NATIVE': 'OFF', 'GGML_CPU_ALL_VARIANTS': 'OFF',
}
X64 = {name: 'ON' for name in ('GGML_AVX', 'GGML_AVX2', 'GGML_SSE42', 'GGML_BMI2')}
X64.update({name: 'OFF' for name in ('GGML_AVX_VNNI', 'GGML_AVX512', 'GGML_AVX512_VBMI', 'GGML_AVX512_VNNI', 'GGML_AVX512_BF16')})
ARM = {f'GGML_MACHINE_SUPPORTS_{name}': 'FALSE' for name in ('dotprod', 'i8mm', 'sve', 'sme')}
ARM_OPTIONAL = ('DOTPROD', 'SVE', 'MATMUL_INT8', 'SME', 'FP16_VECTOR_ARITHMETIC')


def configuration_arguments(source, build, architecture):
    if architecture not in ('x64', 'ARM64'):
        raise ValueError('Expected x64 or ARM64')
    flags = dict(STATIC, GGML_NATIVE='OFF' if architecture == 'x64' else 'ON')
    flags.update(X64 if architecture == 'x64' else ARM)
    flags.update({
        'CMAKE_POLICY_DEFAULT_CMP0091': 'NEW', 'LLAMA_BUILD_NUMBER': '11756',
        'LLAMA_BUILD_COMMIT': 'd80619397', 'LLAMA_BUILD_TESTS': 'OFF',
        'LLAMA_BUILD_EXAMPLES': 'OFF', 'LLAMA_BUILD_APP': 'OFF', 'LLAMA_BUILD_COMMON': 'ON',
        'LLAMA_BUILD_TOOLS': 'ON', 'LLAMA_BUILD_SERVER': 'ON', 'LLAMA_BUILD_UI': 'OFF',
        'LLAMA_USE_PREBUILT_UI': 'OFF', 'LLAMA_CONNECT': 'OFF',
    })
    args = ['cmake', '--log-level=DEBUG', '-S', str(Path(source).resolve()), '-B', str(Path(build).resolve()), '-A', architecture]
    for name, value in flags.items():
        kind = 'BOOL' if value in ('ON', 'OFF', 'FALSE') else 'STRING'
        args.append(f'-D{name}:{kind}={value}')
    return args


def _boolean(value):
    normalized = value.strip().upper()
    if normalized in ('ON', 'TRUE', 'YES', 'Y', '1'):
        return True
    if normalized in ('OFF', 'FALSE', 'NO', 'N', '0', '', 'IGNORE', 'NOTFOUND') or normalized.endswith('-NOTFOUND'):
        return False
    raise ValueError('Unrecognized CMake boolean')


def verify_configuration(build, architecture, log):
    if architecture not in ('x64', 'ARM64'):
        raise ValueError('Expected x64 or ARM64')
    build = Path(build); cache = {}
    for line in (build/'CMakeCache.txt').read_text(encoding='utf-8').splitlines():
        if line.startswith(('#', '//')) or '=' not in line or ':' not in line.split('=', 1)[0]:
            continue
        name, value = line.split('=', 1); cache[name.split(':', 1)[0]] = value
    expected = dict(STATIC, GGML_NATIVE='OFF' if architecture == 'x64' else 'ON')
    expected.update(X64 if architecture == 'x64' else ARM)
    for name, value in expected.items():
        if name not in cache:
            raise ValueError('Missing required cache setting: '+name)
        differs = (_boolean(cache[name]) != _boolean(value)) if value in ('ON', 'OFF', 'FALSE') else cache[name] != value
        if differs:
            raise ValueError('Incorrect cache setting: '+name)
    effective = re.findall(r'^\s*(?:--\s*)?GGML_NATIVE\s*:\s*(ON|OFF)\s*$', log, flags=re.MULTILINE)
    wanted = expected['GGML_NATIVE']
    if not effective or any(value != wanted for value in effective):
        raise ValueError('Effective GGML_NATIVE DEBUG output missing or incorrect')
    project = build/'ggml/src/ggml-cpu.vcxproj'
    raw = project.read_bytes(); tree = ET.fromstring(raw)
    compiler = [el for el in tree.iter() if el.tag.rsplit('}', 1)[-1] == 'ClCompile']
    compiler_text = '\n'.join(' '.join(el.itertext()) for el in compiler)
    if re.search(r'AVX512|AVX_VNNI|AVXVNNI|AVX10', compiler_text, flags=re.IGNORECASE):
        raise ValueError('Unexpected AVX512/VNNI/AVX10 compiler option or definition')
    groups = [child for parent in tree.iter() if parent.tag.rsplit('}', 1)[-1] == 'ItemDefinitionGroup'
              for child in parent if child.tag.rsplit('}', 1)[-1] == 'ClCompile']
    if not groups:
        raise ValueError('No compiler configuration groups in CPU project')
    for group in groups:
        runtime = next((child.text for child in group if child.tag.rsplit('}', 1)[-1] == 'RuntimeLibrary'), None)
        if runtime != 'MultiThreaded':
            raise ValueError('Static MultiThreaded CRT missing in compiler configuration')
    if architecture == 'x64':
        for group in groups:
            values = {child.tag.rsplit('}', 1)[-1]: child.text or '' for child in group}
            enum = values.get('EnableEnhancedInstructionSet', '')
            options = values.get('AdditionalOptions', '')
            if enum.lower() not in ('avx2', 'advancedvectorextensions2') and not re.search(r'/arch:AVX2(?:\s|$)', options, re.IGNORECASE):
                raise ValueError('AVX2 is not explicit in every compiler configuration')
            definitions = values.get('PreprocessorDefinitions', '').split(';')
            for required in ('GGML_AVX2', 'GGML_FMA', 'GGML_F16C', 'GGML_BMI2'):
                if not any(value.split('=', 1)[0] == required for value in definitions):
                    raise ValueError('Missing effective CPU definition: '+required)
    else:
        for feature in ARM_OPTIONAL:
            name = 'HAVE_'+feature
            if name not in cache or _boolean(cache[name]):
                raise ValueError('ARM optional extension enabled or unverified: '+name)
        if 'HAVE_FMA' not in cache or not _boolean(cache['HAVE_FMA']):
            raise ValueError('ARM FMA baseline not verified')
        for group in groups:
            text = ' '.join(group.itertext())
            for feature in ARM_OPTIONAL:
                if re.search(r'(?:-D|/D|;)\s*__ARM_FEATURE_'+feature+r'(?:=1)?(?:\s|;|$)', ';'+text):
                    raise ValueError('ARM optional compiler definition enabled: '+feature)
            if '__ARM_FEATURE_FMA=1' not in text:
                raise ValueError('ARM FMA compiler definition missing')
    allowlist = set(expected) | {'HAVE_'+feature for feature in ARM_OPTIONAL} | {'HAVE_FMA'}
    return {'architecture': architecture, 'effective_native': wanted,
            'cpu_profile': 'x64 AVX2/FMA/F16C/BMI2; no AVX512/VNNI' if architecture == 'x64' else 'ARMv8-A/FMA; optional DOTPROD/I8MM/SVE/SME/FP16 arithmetic disabled',
            'verified_cache': {name: cache[name] for name in sorted(allowlist) if name in cache},
            'compiler_configurations': len(groups), 'project_sha256': hashlib.sha256(raw).hexdigest()}


def _run(arguments, log_path):
    captured = bytearray(); truncated = False
    with subprocess.Popen(arguments, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=False) as child:
        try:
            while True:
                chunk = child.stdout.read(4096)
                if not chunk:
                    break
                remaining = MAX_LOG_BYTES-len(captured)
                public = chunk[:remaining]
                if public:
                    captured.extend(public); print(public.decode('utf-8', errors='replace'), end='', flush=True)
                truncated |= len(chunk) > remaining
            returncode = child.wait()
        finally:
            if child.poll() is None:
                child.kill(); child.wait(timeout=5)
            log_path.write_bytes(captured)
    return returncode, captured.decode('utf-8', errors='replace'), truncated


def configure(source, build, architecture, output):
    build = Path(build).resolve(); output = Path(output).resolve()
    if (build/'CMakeCache.txt').exists():
        raise ValueError('Fresh configuration required: existing CMakeCache.txt')
    args = configuration_arguments(source, build, architecture)
    output.parent.mkdir(parents=True, exist_ok=True)
    log_path = output.with_suffix('.log')
    result = {'schema_version': 'uallera-native-configuration-1', 'passed': False,
              'expected_source_commit': SOURCE_COMMIT, 'source_checkout_verified_by': 'calling workflow',
              'arguments': args, 'log_file': log_path.name, 'log_budget_bytes': MAX_LOG_BYTES}
    try:
        code, log, truncated = _run(args, log_path)
        result.update(cmake_returncode=code, log_truncated=truncated)
        if code != 0 or truncated:
            raise ValueError('CMake failed or public log budget exceeded')
        result.update(verify_configuration(build, architecture, log), passed=True)
    except Exception as exc:
        result['error'] = str(exc) if isinstance(exc, ValueError) else 'Native configuration failed'
        raise
    finally:
        output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--build', required=True, type=Path)
    parser.add_argument('--architecture', required=True, choices=('x64', 'ARM64'))
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = configure(args.source, args.build, args.architecture, args.output)
    except Exception:
        print('Native configuration rejected; inspect bounded log and manifest.')
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
