"""Bounded Win64 child debugger: exception addresses and executable code only.

No command lines, exception parameters, strings from the child, model data,
registers, stacks or memory dumps are persisted. The CLI launches only a
loopback offline synthetic diagnostic engine, without authentication options.
Primary Win32 contracts:
https://learn.microsoft.com/en-us/windows/win32/api/minwinbase/ns-minwinbase-debug_event
https://learn.microsoft.com/en-us/windows/win32/api/debugapi/nf-debugapi-waitfordebugevent
https://learn.microsoft.com/en-us/windows/win32/api/debugapi/nf-debugapi-continuedebugevent
https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-virtualqueryex
https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-readprocessmemory
"""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import platform
import struct
import subprocess
import time

DWORD = C.c_uint32
WORD = C.c_uint16
PTR = C.c_void_p
SIZE = C.c_size_t
BOOL = C.c_int32
DBG_CONTINUE = 0x00010002
DBG_EXCEPTION_NOT_HANDLED = 0x80010001


class EXCEPTION_RECORD(C.Structure):
    _fields_ = [('ExceptionCode', DWORD), ('ExceptionFlags', DWORD),
                ('ExceptionRecord', PTR), ('ExceptionAddress', PTR),
                ('NumberParameters', DWORD), ('ExceptionInformation', C.c_uint64*15)]


class EXCEPTION_DEBUG_INFO(C.Structure):
    _fields_ = [('ExceptionRecord', EXCEPTION_RECORD), ('dwFirstChance', DWORD)]


class CREATE_PROCESS_DEBUG_INFO(C.Structure):
    _fields_ = [('hFile', PTR), ('hProcess', PTR), ('hThread', PTR),
                ('lpBaseOfImage', PTR), ('dwDebugInfoFileOffset', DWORD),
                ('nDebugInfoSize', DWORD), ('lpThreadLocalBase', PTR),
                ('lpStartAddress', PTR), ('lpImageName', PTR), ('fUnicode', WORD)]


class CREATE_THREAD_DEBUG_INFO(C.Structure):
    _fields_ = [('hThread', PTR), ('lpThreadLocalBase', PTR), ('lpStartAddress', PTR)]


class LOAD_DLL_DEBUG_INFO(C.Structure):
    _fields_ = [('hFile', PTR), ('lpBaseOfDll', PTR), ('dwDebugInfoFileOffset', DWORD),
                ('nDebugInfoSize', DWORD), ('lpImageName', PTR), ('fUnicode', WORD)]


class EXIT_PROCESS_DEBUG_INFO(C.Structure):
    _fields_ = [('dwExitCode', DWORD)]


class DEBUG_UNION(C.Union):
    _fields_ = [('Exception', EXCEPTION_DEBUG_INFO), ('CreateProcessInfo', CREATE_PROCESS_DEBUG_INFO),
                ('CreateThread', CREATE_THREAD_DEBUG_INFO), ('LoadDll', LOAD_DLL_DEBUG_INFO),
                ('ExitProcess', EXIT_PROCESS_DEBUG_INFO), ('raw', C.c_ubyte*160)]


class DEBUG_EVENT(C.Structure):
    _fields_ = [('dwDebugEventCode', DWORD), ('dwProcessId', DWORD), ('dwThreadId', DWORD), ('u', DEBUG_UNION)]


class STARTUPINFO(C.Structure):
    _fields_ = [('cb', DWORD), ('lpReserved', C.c_wchar_p), ('lpDesktop', C.c_wchar_p), ('lpTitle', C.c_wchar_p),
                ('dwX', DWORD), ('dwY', DWORD), ('dwXSize', DWORD), ('dwYSize', DWORD),
                ('dwXCountChars', DWORD), ('dwYCountChars', DWORD), ('dwFillAttribute', DWORD),
                ('dwFlags', DWORD), ('wShowWindow', WORD), ('cbReserved2', WORD),
                ('lpReserved2', PTR), ('hStdInput', PTR), ('hStdOutput', PTR), ('hStdError', PTR)]


class PROCESS_INFORMATION(C.Structure):
    _fields_ = [('hProcess', PTR), ('hThread', PTR), ('dwProcessId', DWORD), ('dwThreadId', DWORD)]


class MEMORY_BASIC_INFORMATION(C.Structure):
    _fields_ = [('BaseAddress', PTR), ('AllocationBase', PTR), ('AllocationProtect', DWORD),
                ('PartitionId', WORD), ('RegionSize', SIZE), ('State', DWORD), ('Protect', DWORD), ('Type', DWORD)]


def _kernel():
    if os.name != 'nt' or C.sizeof(PTR) != 8:
        raise RuntimeError('Requires a 64-bit Windows Python process')
    kernel = C.WinDLL('kernel32', use_last_error=True)
    signatures = {
        'CreateProcessW': ([C.c_wchar_p, C.c_wchar_p, PTR, PTR, BOOL, DWORD, PTR, C.c_wchar_p, C.POINTER(STARTUPINFO), C.POINTER(PROCESS_INFORMATION)], BOOL),
        'WaitForDebugEvent': ([C.POINTER(DEBUG_EVENT), DWORD], BOOL),
        'ContinueDebugEvent': ([DWORD, DWORD, DWORD], BOOL),
        'CloseHandle': ([PTR], BOOL), 'TerminateProcess': ([PTR, DWORD], BOOL),
        'DebugActiveProcessStop': ([DWORD], BOOL), 'WaitForSingleObject': ([PTR, DWORD], DWORD),
        'VirtualQueryEx': ([PTR, PTR, C.POINTER(MEMORY_BASIC_INFORMATION), SIZE], SIZE),
        'ReadProcessMemory': ([PTR, PTR, PTR, SIZE, C.POINTER(SIZE)], BOOL),
        'GetLastError': ([], DWORD), 'GetCurrentProcess': ([], PTR),
        'IsWow64Process2': ([PTR, C.POINTER(WORD), C.POINTER(WORD)], BOOL),
    }
    for name, (arguments, result) in signatures.items():
        function = getattr(kernel, name)
        function.argtypes = arguments; function.restype = result
    return kernel


def image_metadata(path):
    """Read only executable hash/PE headers, never the model path."""
    path = Path(path)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        stream.seek(0); header = stream.read(64)
        if len(header) != 64 or header[:2] != b'MZ':
            raise ValueError('Executable is not a PE image')
        offset = struct.unpack_from('<I', header, 60)[0]
        stream.seek(offset); pe = stream.read(84)
    if len(pe) < 84 or pe[:4] != b'PE\0\0':
        raise ValueError('Executable PE headers are invalid')
    machine = struct.unpack_from('<H', pe, 4)[0]
    return {'sha256': digest, 'architecture': {0x8664: 'x64', 0xAA64: 'ARM64'}.get(machine, hex(machine)),
            'image_size': struct.unpack_from('<I', pe, 24+56)[0]}


def continuation(code, first_chance, startup_seen):
    startup = code == 0x80000003 and bool(first_chance) and not startup_seen
    return (DBG_CONTINUE if startup else DBG_EXCEPTION_NOT_HANDLED, startup)


def exception_record(kernel, process, event, main_base, image_size):
    address = int(event.ExceptionRecord.ExceptionAddress or 0)
    record = {'code': hex(event.ExceptionRecord.ExceptionCode), 'first_chance': bool(event.dwFirstChance),
              'address': hex(address), 'rva': hex(address-main_base) if main_base and main_base <= address < main_base+image_size else None,
              'code_bytes_hex': '', 'code_bytes_requested': 32}
    memory = MEMORY_BASIC_INFORMATION()
    # Restrict the sole memory read to a committed executable page; no stacks,
    # exception parameters, model buffers or arbitrary surrounding memory.
    if kernel.VirtualQueryEx(process, address, C.byref(memory), C.sizeof(memory)):
        if memory.State == 0x1000 and memory.Type == 0x1000000 and memory.Protect & 0xF0 and not memory.Protect & 0x101:
            available = int(memory.BaseAddress or 0)+int(memory.RegionSize)-address
            count = min(32, max(0, available))
            if count:
                buffer = (C.c_ubyte*32)(); read = SIZE()
                if kernel.ReadProcessMemory(process, address, buffer, count, C.byref(read)):
                    record['code_bytes_hex'] = bytes(buffer[:min(count, read.value)]).hex()
    return record


def capture(command, cwd=None, timeout=120, output=None):
    """Debug one owned child; persist no argv or child-controlled text."""
    if not command or not 0 < timeout <= 120:
        raise ValueError('One executable and a timeout in (0, 120] are required')
    command = [str(item) for item in command]
    if any('\0' in item for item in command):
        raise ValueError('Invalid command argument')
    kernel = _kernel()
    metadata = image_metadata(command[0])
    result = {'schema_version': 'uallera-windows-code-exception-1', 'executable': metadata,
              'native_architecture': 'unknown', 'debugger_pointer_bits': C.sizeof(PTR)*8,
              'main_image_base': None, 'exceptions': [], 'returncode': None,
              'timed_out': False, 'errors': [], 'memory_policy': 'At most 32 bytes at each executable exception address; no dumps, strings, stacks, registers, parameters or model data'}
    def save():
        if output:
            Path(output).write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    try:
        process_machine, native_machine = WORD(), WORD()
        if kernel.IsWow64Process2(kernel.GetCurrentProcess(), C.byref(process_machine), C.byref(native_machine)):
            result['native_architecture'] = {0x8664: 'x64', 0xAA64: 'ARM64'}.get(native_machine.value, hex(native_machine.value))
    except AttributeError:
        pass
    startup = STARTUPINFO(); startup.cb = C.sizeof(startup)
    info = PROCESS_INFORMATION(); exited = False; created = False
    base = 0; startup_seen = False; debug_handles = set(); auto_closed = set(); pending = None
    before = time.monotonic()
    save()
    try:
        mutable = C.create_unicode_buffer(subprocess.list2cmdline(command))
        if not kernel.CreateProcessW(command[0], mutable, None, None, False,
                                     0x00000002 | 0x08000000, None, str(cwd) if cwd else None,
                                     C.byref(startup), C.byref(info)):
            raise OSError(int(kernel.GetLastError()), 'CreateProcessW failed')
        created = True
        while time.monotonic()-before < timeout:
            event = DEBUG_EVENT()
            if not kernel.WaitForDebugEvent(C.byref(event), 250):
                code = int(kernel.GetLastError())
                if code in (121, 258):
                    continue
                raise OSError(code, 'WaitForDebugEvent failed')
            status = DBG_CONTINUE; kind = event.dwDebugEventCode
            pending = (event.dwProcessId, event.dwThreadId,
                       DBG_EXCEPTION_NOT_HANDLED if kind == 1 else DBG_CONTINUE)
            if kind == 3:
                base = int(event.u.CreateProcessInfo.lpBaseOfImage or 0)
                result['main_image_base'] = hex(base)
                debug_handles.update(h for h in (event.u.CreateProcessInfo.hProcess, event.u.CreateProcessInfo.hThread) if h)
                if event.u.CreateProcessInfo.hFile:
                    kernel.CloseHandle(event.u.CreateProcessInfo.hFile)
            elif kind == 6 and event.u.LoadDll.hFile:
                kernel.CloseHandle(event.u.LoadDll.hFile)
            elif kind == 1:
                row = exception_record(kernel, info.hProcess, event.u.Exception, base, metadata['image_size'])
                status, initial = continuation(event.u.Exception.ExceptionRecord.ExceptionCode, event.u.Exception.dwFirstChance, startup_seen)
                startup_seen |= initial; row['startup_breakpoint'] = initial
                if len(result['exceptions']) < 64:
                    result['exceptions'].append(row)
                save()
            elif kind == 5:
                result['returncode'] = int(event.u.ExitProcess.dwExitCode)
            if not kernel.ContinueDebugEvent(event.dwProcessId, event.dwThreadId, status):
                raise OSError(int(kernel.GetLastError()), 'ContinueDebugEvent failed')
            pending = None
            if kind == 5:
                exited = True; auto_closed.update(debug_handles)
                break
        if not exited:
            result['timed_out'] = True
    except Exception:
        # No exception repr can leak command arguments or child text.
        result['errors'].append('Debugger operation failed')
    finally:
        if created and not exited:
            if pending:
                kernel.ContinueDebugEvent(*pending)
            if not kernel.TerminateProcess(info.hProcess, 1460):
                result['errors'].append('Owned process termination failed')
            # Termination is itself a debug event. Drain/continue it so the OS
            # closes its process/thread debug handles before closing our own.
            cleanup_deadline = time.monotonic()+2
            while time.monotonic() < cleanup_deadline:
                final = DEBUG_EVENT()
                if not kernel.WaitForDebugEvent(C.byref(final), 50):
                    continue
                kind = final.dwDebugEventCode
                if kind == 3:
                    debug_handles.update(h for h in (final.u.CreateProcessInfo.hProcess, final.u.CreateProcessInfo.hThread) if h)
                    if final.u.CreateProcessInfo.hFile:
                        kernel.CloseHandle(final.u.CreateProcessInfo.hFile)
                elif kind == 6 and final.u.LoadDll.hFile:
                    kernel.CloseHandle(final.u.LoadDll.hFile)
                status = DBG_EXCEPTION_NOT_HANDLED if kind == 1 else DBG_CONTINUE
                if kernel.ContinueDebugEvent(final.dwProcessId, final.dwThreadId, status) and kind == 5:
                    exited = True; auto_closed.update(debug_handles)
                    result['returncode'] = int(final.u.ExitProcess.dwExitCode)
                    break
            if not exited:
                kernel.DebugActiveProcessStop(info.dwProcessId)
                result['errors'].append('Owned process exit was not confirmed')
            kernel.WaitForSingleObject(info.hProcess, 2000)
        for handle in (info.hThread, info.hProcess):
            if handle and handle not in auto_closed:
                kernel.CloseHandle(handle)
        result['seconds'] = round(time.monotonic()-before, 3)
        save()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', required=True, type=Path)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--template', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--timeout', type=float, default=120)
    args = parser.parse_args(argv)
    executable = args.executable.resolve()
    command = [str(executable), '--model', str(args.model.resolve()), '--host', '127.0.0.1', '--port', '0',
               '--ctx-size', '4096', '--parallel', '1', '--alias', 'minerva-7b-instruct-q4-local',
               '--offline', '--no-webui', '--no-warmup', '--jinja', '--chat-template-file', str(args.template.resolve())]
    try:
        result = capture(command, executable.parent, args.timeout, args.output)
    except Exception:
        print('Windows debugger could not initialize.')
        return 1
    print(json.dumps(result, indent=2))
    return 0 if result['returncode'] is not None and not result['errors'] and not result['timed_out'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
