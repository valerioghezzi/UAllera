"""Win64 ABI and debugger protocol fixtures; no native child is executed."""
import ctypes as C
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    from tools import debug_windows_engine as debug
except ImportError:
    debug = None


class WindowsDebuggerTests(unittest.TestCase):
    def test_win64_structure_sizes_and_alignment(self):
        self.assertIsNotNone(debug, 'Isolated Windows debugger is missing')
        self.assertEqual(C.sizeof(debug.EXCEPTION_DEBUG_INFO), 160)
        self.assertEqual(C.sizeof(debug.DEBUG_EVENT), 176)
        self.assertEqual(debug.DEBUG_EVENT.u.offset, 16)
        self.assertEqual(C.sizeof(debug.CREATE_PROCESS_DEBUG_INFO), 72)
        self.assertEqual(C.sizeof(debug.STARTUPINFO), 104)
        self.assertEqual(C.sizeof(debug.MEMORY_BASIC_INFORMATION), 48)

    def test_only_initial_first_chance_breakpoint_is_handled(self):
        self.assertIsNotNone(debug)
        self.assertEqual(debug.continuation(0x80000003, True, False), (debug.DBG_CONTINUE, True))
        for code, first, seen in [(0x80000003, True, True), (0xC000001D, True, False), (0xC000001D, False, True), (0x80000003, False, False)]:
            self.assertEqual(debug.continuation(code, first, seen)[0], debug.DBG_EXCEPTION_NOT_HANDLED)

    def test_exception_records_main_rva_only_executable_code_and_no_parameters(self):
        self.assertIsNotNone(debug)
        class Kernel:
            def VirtualQueryEx(self, process, address, info, size):
                info._obj.BaseAddress = 0x1000
                info._obj.RegionSize = 0x1000
                info._obj.State = 0x1000
                info._obj.Protect = 0x20
                info._obj.Type = 0x1000000
                return size
            def ReadProcessMemory(self, process, address, buffer, size, read):
                self.read_size = size
                C.memmove(buffer, b'\x0f\x0b' + b'\x90'*30, size)
                read._obj.value = size
                return 1
        event = debug.EXCEPTION_DEBUG_INFO()
        event.ExceptionRecord.ExceptionCode = 0xC000001D
        event.ExceptionRecord.ExceptionAddress = 0x1020
        event.ExceptionRecord.ExceptionInformation[0] = 0xDEADBEEF
        event.dwFirstChance = 1
        kernel = Kernel()
        row = debug.exception_record(kernel, 1, event, 0x1000, 0x1000)
        self.assertEqual(row['rva'], '0x20')
        self.assertEqual(row['code'], '0xc000001d')
        self.assertEqual(len(bytes.fromhex(row['code_bytes_hex'])), 32)
        self.assertNotIn('deadbeef', json.dumps(row).lower())
        self.assertEqual(kernel.read_size, 32)
        with patch.object(kernel, 'VirtualQueryEx', return_value=0):
            self.assertEqual(debug.exception_record(kernel, 1, event, 0x1000, 0x1000)['code_bytes_hex'], '')

    def test_fixture_events_continue_to_second_chance_and_close_file_handles(self):
        self.assertIsNotNone(debug)
        kernel = FakeKernel()
        with tempfile.TemporaryDirectory() as temp, patch.object(debug, '_kernel', return_value=kernel), patch.object(debug, 'image_metadata', return_value={'sha256': 'fixture', 'architecture': 'x64', 'image_size': 0x1000}):
            output = Path(temp)/'debug.json'
            result = debug.capture(['synthetic.exe'], temp, 1, output)
            self.assertEqual(json.loads(output.read_text(encoding='utf-8')), result)
        self.assertEqual(result['returncode'], 0xC000001D)
        self.assertEqual([row['first_chance'] for row in result['exceptions']], [True, True, False])
        self.assertEqual(kernel.continued, [debug.DBG_CONTINUE, debug.DBG_CONTINUE, debug.DBG_EXCEPTION_NOT_HANDLED, debug.DBG_EXCEPTION_NOT_HANDLED, debug.DBG_CONTINUE])
        self.assertIn(30, kernel.closed)
        self.assertNotIn('synthetic.exe', json.dumps(result))
        self.assertFalse(kernel.killed)

    def test_timeout_terminates_owned_child_and_drains_exit_event(self):
        self.assertIsNotNone(debug)
        class SlowKernel(FakeKernel):
            def __init__(self):
                super().__init__(); self.events = [3]
            def WaitForDebugEvent(self, event, milliseconds):
                if not self.events:
                    import time
                    time.sleep(.02)
                    return 0
                return super().WaitForDebugEvent(event, milliseconds)
            def GetLastError(self): return 121
            def TerminateProcess(self, process, code):
                self.killed = True; self.events.append(5); return 1
        kernel = SlowKernel()
        with patch.object(debug, '_kernel', return_value=kernel), patch.object(debug, 'image_metadata', return_value={'sha256': 'fixture', 'architecture': 'x64', 'image_size': 0x1000}):
            result = debug.capture(['fixture.exe'], timeout=.01)
        self.assertTrue(result['timed_out'])
        self.assertTrue(kernel.killed)
        self.assertEqual(kernel.events, [])
        self.assertIn(10, kernel.closed)
        self.assertIn(11, kernel.closed)


class FakeKernel:
    def __init__(self):
        self.events = [3, 1, 1, 1, 5]
        self.continued = []; self.closed = []; self.killed = False
        self.exceptions = 0
    def CreateProcessW(self, app, command, pa, ta, inherit, flags, env, cwd, startup, info):
        info._obj.hProcess = 10; info._obj.hThread = 11
        info._obj.dwProcessId = 123; info._obj.dwThreadId = 124
        return 1
    def WaitForDebugEvent(self, event, milliseconds):
        kind = self.events.pop(0); obj = event._obj
        obj.dwDebugEventCode = kind; obj.dwProcessId = 123; obj.dwThreadId = 124
        if kind == 3:
            obj.u.CreateProcessInfo.hFile = 30
            obj.u.CreateProcessInfo.hProcess = 20; obj.u.CreateProcessInfo.hThread = 21
            obj.u.CreateProcessInfo.lpBaseOfImage = 0x1000
        elif kind == 1:
            self.exceptions += 1
            obj.u.Exception.ExceptionRecord.ExceptionCode = 0x80000003 if self.exceptions == 1 else 0xC000001D
            obj.u.Exception.ExceptionRecord.ExceptionAddress = 0x1020
            obj.u.Exception.dwFirstChance = self.exceptions != 3
        else:
            obj.u.ExitProcess.dwExitCode = 0xC000001D
        return 1
    def ContinueDebugEvent(self, pid, tid, status): self.continued.append(status); return 1
    def CloseHandle(self, handle): self.closed.append(handle); return 1
    def VirtualQueryEx(self, *args): return 0
    def GetLastError(self): return 0
    def TerminateProcess(self, *args): self.killed = True; return 1
    def DebugActiveProcessStop(self, pid): return 1
    def WaitForSingleObject(self, *args): return 0


if __name__ == '__main__': unittest.main()
