"""Synthetic PE import closure tests: no downloads or native execution."""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
import zipfile

from tools import inspect_native_imports as gate

TOOL = Path(gate.__file__)


def image(imports=(), delay=(), machine=0x8664):
    data = bytearray(0x2000)
    data[:2] = b'MZ'
    struct.pack_into('<I', data, 0x3c, 0x80)
    data[0x80:0x84] = b'PE\0\0'
    struct.pack_into('<HH', data, 0x84, machine, 1)
    struct.pack_into('<H', data, 0x94, 240)
    optional = 0x98
    struct.pack_into('<H', data, optional, 0x20b)
    struct.pack_into('<Q', data, optional+24, 0x140000000)
    struct.pack_into('<I', data, optional+60, 0x200)
    struct.pack_into('<I', data, optional+108, 16)
    struct.pack_into('<IIII', data, optional+240+8, 0x1e00, 0x1000, 0x1e00, 0x200)
    pointer = 0x800
    for names, index, offset, width in [(imports, 1, 0x200, 20), (delay, 13, 0x500, 32)]:
        if not names:
            continue
        struct.pack_into('<II', data, optional+112+index*8, offset+0xe00, (len(names)+1)*width)
        for i, name in enumerate(names):
            if width == 20:
                struct.pack_into('<IIIII', data, offset+i*width, 0, 0, 0, pointer+0xe00, 0)
            else:
                struct.pack_into('<IIIIIIII', data, offset+i*width, 1, pointer+0xe00, 0, 0, 0, 0, 0, 0)
            raw = name.encode('ascii') + b'\0'
            data[pointer:pointer+len(raw)] = raw
            pointer += len(raw)
    return bytes(data)


class NativeGateTests(unittest.TestCase):
    def test_invented_api_set_is_missing_in_normal_and_delay_imports(self):
        invented = 'api-ms-win-core-madeup-l99-99-99.dll'
        for kind in ('imports', 'delay'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                folder = Path(temp)
                (folder/'server.exe').write_bytes(image(**{kind: [invented]}))
                result = gate.inspect(folder, 'x64')
                self.assertFalse(result['ok'])
                self.assertEqual(result['missing_imports'], [{
                    'importer': 'server.exe', 'dll': invented,
                    'kind': 'delay_imports' if kind == 'delay' else 'imports',
                }])

    def test_static_cli_rejects_bundled_crt_and_openmp_normal_and_delay_imports(self):
        for kind in ('imports', 'delay'):
            for name in ('MSVCP140.dll', 'VCRUNTIME140_1.dll', 'libomp.dll'):
                with self.subTest(kind=kind, name=name), tempfile.TemporaryDirectory() as temp:
                    folder = Path(temp)
                    (folder/'server.exe').write_bytes(image(**{kind: [name]}))
                    (folder/name).write_bytes(image())
                    # Ordinary closure accepts a supplied dependency. Static
                    # runtime mode must reject the same actual PE import.
                    self.assertTrue(gate.inspect(folder, 'x64')['ok'])
                    run = subprocess.run([sys.executable, str(TOOL), '--directory', str(folder), '--architecture', 'x64', '--require-static-runtime'], capture_output=True, text=True)
                    self.assertEqual(run.returncode, 1, run.stderr)
                    report = json.loads(run.stdout)
                    self.assertEqual(report['forbidden_runtime_imports'], [{
                        'importer': 'server.exe', 'dll': name,
                        'kind': 'delay_imports' if kind == 'delay' else 'imports',
                    }])
                    self.assertEqual(report['missing_imports'], [])

    def test_static_cli_accepts_only_platform_imports_in_both_tables(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder/'server.exe').write_bytes(image(['KERNEL32.dll'], ['api-ms-win-core-synch-l1-2-0.dll']))
            run = subprocess.run([sys.executable, str(TOOL), '--directory', str(folder), '--architecture', 'x64', '--require-static-runtime'], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertTrue(json.loads(run.stdout)['require_static_runtime'])

    def test_static_runtime_rejects_ucrt_system_imports_and_unreferenced_runtime_files(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder/'server.exe').write_bytes(image(['ucrtbase.dll'], ['api-ms-win-crt-runtime-l1-1-0.dll']))
            (folder/'vcomp140.dll').write_bytes(image())
            run = subprocess.run([sys.executable, str(TOOL), '--directory', str(folder), '--architecture', 'x64', '--require-static-runtime'], capture_output=True, text=True)
            self.assertEqual(run.returncode, 1, run.stderr)
            report = json.loads(run.stdout)
            self.assertEqual({item['dll'].lower() for item in report['forbidden_runtime_imports']}, {'ucrtbase.dll', 'api-ms-win-crt-runtime-l1-1-0.dll'})
            self.assertEqual(report['forbidden_runtime_files'], ['vcomp140.dll'])

    def test_closed_system_and_case_insensitive_bundle_dependencies(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder/'server.exe').write_bytes(image(['KERNEL32.dll', 'WORKER.DLL'], ['api-ms-win-crt-runtime-l1-1-0.dll']))
            (folder/'worker.dll').write_bytes(image(['WS2_32.dll']))
            result = gate.inspect(folder, 'x64')
        self.assertTrue(result['ok'], result)
        self.assertEqual(len(result['files']), 2)
        self.assertEqual(result['missing_imports'], [])

    def test_missing_crt_openmp_and_unknown_dependencies_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder/'server.exe').write_bytes(image(['MSVCP140.dll', 'libomp.dll'], ['mystery.dll']))
            result = gate.inspect(folder, 'x64')
        self.assertFalse(result['ok'])
        self.assertEqual({v['dll'].lower() for v in result['missing_imports']}, {'msvcp140.dll', 'libomp.dll', 'mystery.dll'})

    def test_wrong_architecture_and_malformed_pe_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder/'wrong.dll').write_bytes(image(machine=0xaa64))
            (folder/'broken.exe').write_bytes(b'MZ')
            result = gate.inspect(folder, 'x64')
        self.assertFalse(result['ok'])
        self.assertEqual(len(result['errors']), 2)

    def test_dependency_in_other_folder_is_not_in_loader_search_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder/'server.exe').write_bytes(image(['worker.dll']))
            (folder/'other').mkdir()
            (folder/'other'/'worker.dll').write_bytes(image())
            result = gate.inspect(folder, 'x64')
        self.assertFalse(result['ok'])
        self.assertEqual(result['missing_imports'][0]['dll'], 'worker.dll')

    def test_zip_cli_reports_json_and_rejects_unsafe_or_duplicate_names(self):
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp)/'unsafe.zip'
            with zipfile.ZipFile(archive, 'w') as bundle:
                bundle.writestr('../bad.exe', image())
            self.assertFalse(gate.inspect(archive, 'x64')['ok'])
            with zipfile.ZipFile(archive, 'w') as bundle:
                bundle.writestr('server.exe', image())
                bundle.writestr('SERVER.EXE', image())
            self.assertFalse(gate.inspect(archive, 'x64')['ok'])
            with zipfile.ZipFile(archive, 'w') as bundle:
                bundle.writestr('server.exe', image(['kernel32.dll']))
            output = Path(temp)/'manifest.json'
            result = subprocess.run([sys.executable, str(TOOL), '--zip', str(archive), '--architecture', 'x64', '--output', str(output)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(json.loads(result.stdout)['ok'])
            self.assertEqual(json.loads(output.read_text(encoding='utf-8')), json.loads(result.stdout))

    def test_missing_path_and_empty_bundle_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            self.assertFalse(gate.inspect(Path(temp), 'x64')['ok'])
            self.assertFalse(gate.inspect(Path(temp)/'absent', 'x64')['ok'])

    def test_nested_dll_folder_cannot_imply_windows_loader_search_path(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder/'server.exe').write_bytes(image())
            (folder/'plugins').mkdir()
            (folder/'plugins'/'worker.dll').write_bytes(image(['helper.dll']))
            (folder/'plugins'/'helper.dll').write_bytes(image())
            result = gate.inspect(folder, 'x64')
        self.assertFalse(result['ok'])
        self.assertTrue(any('directory' in error['error'].lower() for error in result['errors']))



if __name__ == '__main__':
    unittest.main()
