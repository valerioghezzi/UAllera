"""Synthetic configuration gates; no CMake, compiler or engine execution."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    from tools import configure_native_engine as configure
except ImportError:
    configure = None


def fixture(folder, architecture='x64', native=None, project_arch='AVX2', optional=False):
    values = {
        'BUILD_SHARED_LIBS': 'OFF', 'CMAKE_MSVC_RUNTIME_LIBRARY': 'MultiThreaded',
        'GGML_BACKEND_DL': 'OFF', 'GGML_OPENMP': 'OFF', 'GGML_OPENMP_FETCH': 'OFF',
        'LLAMA_OPENSSL': 'OFF', 'LLAMA_NATIVE': 'OFF', 'GGML_CPU_ALL_VARIANTS': 'OFF',
        'GGML_NATIVE': native or ('OFF' if architecture == 'x64' else 'ON'),
    }
    if architecture == 'x64':
        values.update({name: 'ON' for name in ['GGML_AVX', 'GGML_AVX2', 'GGML_SSE42', 'GGML_BMI2']})
        values.update({name: 'OFF' for name in ['GGML_AVX_VNNI', 'GGML_AVX512', 'GGML_AVX512_VBMI', 'GGML_AVX512_VNNI', 'GGML_AVX512_BF16']})
    else:
        values.update({f'GGML_MACHINE_SUPPORTS_{name}': 'FALSE' for name in ['dotprod', 'i8mm', 'sve', 'sme']})
        values.update({f'HAVE_{name}': '1' if optional and name == 'SVE' else '' for name in ['DOTPROD', 'SVE', 'MATMUL_INT8', 'SME', 'FP16_VECTOR_ARITHMETIC']})
        values['HAVE_FMA'] = '1'
    (folder/'CMakeCache.txt').write_text('\n'.join(f'{key}:STRING={value}' for key,value in values.items()), encoding='utf-8')
    target = folder/'ggml/src/ggml-cpu.vcxproj'; target.parent.mkdir(parents=True, exist_ok=True)
    definitions = 'GGML_AVX2;GGML_FMA;GGML_F16C;GGML_BMI2' if architecture == 'x64' else '__ARM_FEATURE_FMA=1'
    if project_arch == 'AVX512': definitions += ';GGML_AVX512'
    target.write_text(f'<Project xmlns="http://schemas.microsoft.com/developer/msbuild/2003"><ItemDefinitionGroup><ClCompile><RuntimeLibrary>MultiThreaded</RuntimeLibrary><EnableEnhancedInstructionSet>{project_arch}</EnableEnhancedInstructionSet><AdditionalOptions>/arch:{project_arch} %(AdditionalOptions)</AdditionalOptions><PreprocessorDefinitions>{definitions}</PreprocessorDefinitions></ClCompile></ItemDefinitionGroup></Project>', encoding='utf-8')


class NativeConfigureTests(unittest.TestCase):
    def test_project_dynamic_crt_cannot_hide_behind_static_cache(self):
        self.assertIsNotNone(configure)
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp); fixture(folder)
            project = folder/'ggml/src/ggml-cpu.vcxproj'
            project.write_text(project.read_text(encoding='utf-8').replace('<RuntimeLibrary>MultiThreaded</RuntimeLibrary>', '<RuntimeLibrary>MultiThreadedDLL</RuntimeLibrary>'), encoding='utf-8')
            with self.assertRaises(ValueError): configure.verify_configuration(folder, 'x64', '-- GGML_NATIVE : OFF\n')

    def test_correct_x64_cache_effective_debug_and_project_pass(self):
        self.assertIsNotNone(configure, 'Structured native configurator missing')
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp); fixture(folder)
            result = configure.verify_configuration(folder, 'x64', '-- GGML_NATIVE : OFF\n')
        self.assertEqual(result['architecture'], 'x64')
        self.assertEqual(result['effective_native'], 'OFF')
        self.assertTrue(result['project_sha256'])

    def test_cache_off_cannot_hide_generated_avx512_in_any_configuration(self):
        self.assertIsNotNone(configure)
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp); fixture(folder, project_arch='AVX512')
            with self.assertRaises(ValueError): configure.verify_configuration(folder, 'x64', '-- GGML_NATIVE : OFF\n')

    def test_cache_native_on_and_effective_override_are_rejected(self):
        self.assertIsNotNone(configure)
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp); fixture(folder, native='ON')
            with self.assertRaises(ValueError): configure.verify_configuration(folder, 'x64', '-- GGML_NATIVE : OFF\n')
            (folder/'ggml/src/ggml-cpu.vcxproj').unlink()
            fixture(folder)
            with self.assertRaises(ValueError): configure.verify_configuration(folder, 'x64', '-- GGML_NATIVE : ON\n')

    def test_arm_optional_extensions_are_rejected_and_baseline_passes(self):
        self.assertIsNotNone(configure)
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp); fixture(folder, 'ARM64', project_arch='NotSet', optional=True)
            with self.assertRaises(ValueError): configure.verify_configuration(folder, 'ARM64', '-- GGML_NATIVE : ON\n')
            (folder/'ggml/src/ggml-cpu.vcxproj').unlink()
            fixture(folder, 'ARM64', project_arch='NotSet')
            self.assertEqual(configure.verify_configuration(folder, 'ARM64', '-- GGML_NATIVE : ON\n')['effective_native'], 'ON')

    def test_arguments_are_individual_literal_typed_tokens_and_reuse_is_rejected(self):
        self.assertIsNotNone(configure)
        with tempfile.TemporaryDirectory(prefix='native source ') as temp:
            folder = Path(temp)
            arguments = configure.configuration_arguments(folder/'source', folder/'build', 'x64')
            self.assertIn('-DGGML_NATIVE:BOOL=OFF', arguments)
            self.assertIn('-DLLAMA_NATIVE:BOOL=OFF', arguments)
            self.assertIn(str((folder/'source').resolve()), arguments)
            self.assertNotIn('$native', ' '.join(arguments))
            build = folder/'build'; build.mkdir(); (build/'CMakeCache.txt').write_text('existing', encoding='utf-8')
            with patch.object(configure, '_run') as run:
                with self.assertRaises(ValueError): configure.configure(folder/'source', build, 'x64', folder/'result.json')
                run.assert_not_called()


if __name__ == '__main__': unittest.main()
