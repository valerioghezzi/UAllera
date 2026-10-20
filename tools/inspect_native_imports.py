"""Inspect Windows PE dependency closure without loading or executing binaries.

Stdlib-only development/CI gate for Windows 10/11 x64 and ARM64 bundles.
Reads normal and delay-import DLL tables of every EXE/DLL/PYD in a ZIP or
directory. All PE images must occupy one directory; no custom DLL search
paths are assumed. Imports must resolve there or to the explicit Windows
system DLL/API-set policy below. Unknown external libraries, including MSVC/OpenMP runtimes,
fail the gate. --require-static-runtime additionally rejects CRT/OpenMP
imports even when their DLLs are bundled, and rejects bundled runtime DLLs
even if unreferenced. This is filename closure, not an execution/ABI compatibility
test: exports, minimum Windows versions, and dynamic LoadLibrary calls are
not validated. No network, extraction, binary execution or DLL loading.

PE format: https://learn.microsoft.com/en-us/windows/win32/debug/pe-format
Loader: https://learn.microsoft.com/en-us/windows/win32/dlls/dynamic-link-library-search-order
UCRT: https://learn.microsoft.com/en-us/cpp/windows/universal-crt-deployment
"""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import zipfile


ARCHITECTURES = {"x64": 0x8664, "ARM64": 0xAA64}
PE_SUFFIXES = (".exe", ".dll", ".pyd")
MAX_FILES = 4096
MAX_PE_BYTES = 256 * 1024 * 1024
MAX_TOTAL_BYTES = 512 * 1024 * 1024
# Conservative documented platform imports used by the inspected native engine
# and CPython embeddable distributions. No MSVC/OpenMP redist is a system DLL.
SYSTEM_DLLS = frozenset("""
advapi32.dll bcrypt.dll bcryptprimitives.dll cfgmgr32.dll combase.dll
comctl32.dll comdlg32.dll crypt32.dll cryptbase.dll dbghelp.dll dnsapi.dll
dwmapi.dll gdi32.dll imm32.dll iphlpapi.dll kernel32.dll kernelbase.dll
msvcrt.dll ncrypt.dll netapi32.dll ntdll.dll ole32.dll oleaut32.dll
powrprof.dll propsys.dll psapi.dll rpcrt4.dll secur32.dll setupapi.dll
shell32.dll shlwapi.dll user32.dll userenv.dll ucrtbase.dll version.dll
winhttp.dll wininet.dll winmm.dll winspool.drv ws2_32.dll wtsapi32.dll
""".split())
# Finite contracts observed in the official llama.cpp b11756 Windows CPU
# x64/ARM64 and CPython 3.14.8 embeddable PE import tables. Deliberately not
# a wildcard policy or a complete Windows SDK catalogue: a new contract must
# be verified before adding it. UCRT remains forbidden in static-runtime mode.
SYSTEM_API_SETS = frozenset("""
api-ms-win-core-path-l1-1-0.dll
api-ms-win-core-synch-l1-2-0.dll
api-ms-win-crt-conio-l1-1-0.dll
api-ms-win-crt-convert-l1-1-0.dll
api-ms-win-crt-environment-l1-1-0.dll
api-ms-win-crt-filesystem-l1-1-0.dll
api-ms-win-crt-heap-l1-1-0.dll
api-ms-win-crt-locale-l1-1-0.dll
api-ms-win-crt-math-l1-1-0.dll
api-ms-win-crt-private-l1-1-0.dll
api-ms-win-crt-process-l1-1-0.dll
api-ms-win-crt-runtime-l1-1-0.dll
api-ms-win-crt-stdio-l1-1-0.dll
api-ms-win-crt-string-l1-1-0.dll
api-ms-win-crt-time-l1-1-0.dll
api-ms-win-crt-utility-l1-1-0.dll
""".split())
RUNTIME_DLL = re.compile(r"(?:msvcp|msvcr|msvcm|vcruntime|concrt|vcomp|libomp|libiomp|libgomp|iomp|omp|gomp)[a-z0-9_.-]*\.dll\Z")


def _is_runtime_dll(name):
    name = name.lower()
    return bool(RUNTIME_DLL.fullmatch(name) or name in ("ucrtbase.dll", "ucrtbased.dll")
                or name.startswith("api-ms-win-crt-"))


def pe_imports(data):
    """Return PE architecture and normal/delay DLL names; reject malformed data."""
    def unpack(fmt, offset):
        size = struct.calcsize(fmt)
        if offset < 0 or offset + size > len(data):
            raise ValueError("Truncated PE structure")
        return struct.unpack_from(fmt, data, offset)

    def u16(offset):
        return unpack("<H", offset)[0]

    def u32(offset):
        return unpack("<I", offset)[0]

    if data[:2] != b"MZ":
        raise ValueError("DOS header missing")
    header = u32(0x3C)
    if data[header:header + 4] != b"PE\0\0":
        raise ValueError("PE signature missing")
    machine, section_count = unpack("<HH", header + 4)
    optional_size = u16(header + 20)
    optional = header + 24
    magic = u16(optional)
    if magic == 0x20B:
        image_base = unpack("<Q", optional + 24)[0]
        directory_offset, count_offset = 112, 108
    elif magic == 0x10B:
        image_base = u32(optional + 28)
        directory_offset, count_offset = 96, 92
    else:
        raise ValueError("Unsupported PE optional header")
    if optional_size < directory_offset or optional + optional_size > len(data):
        raise ValueError("Truncated PE optional header")
    directory_count = u32(optional + count_offset)
    if directory_count > (optional_size - directory_offset) // 8:
        raise ValueError("PE data directories exceed optional header")
    headers_size = u32(optional + 60)
    if not 0 < section_count <= 96:
        raise ValueError("Invalid PE section count")
    sections = []
    for index in range(section_count):
        offset = optional + optional_size + index * 40
        virtual_size, address, raw_size, pointer = unpack("<IIII", offset + 8)
        if pointer + raw_size > len(data):
            raise ValueError("Truncated PE section")
        sections.append((address, max(virtual_size, raw_size), pointer, raw_size))

    def rva_offset(rva, size=1):
        if 0 <= rva and rva + size <= min(headers_size, len(data)):
            return rva
        for address, span, pointer, raw_size in sections:
            if address <= rva < address + span:
                relative = rva - address
                if relative + size > raw_size or pointer + relative + size > len(data):
                    raise ValueError("RVA outside file-backed PE section")
                return pointer + relative
        raise ValueError("RVA outside PE sections")

    def dll_name(rva):
        raw = bytearray()
        for index in range(512):
            value = data[rva_offset(rva + index)]
            if value == 0:
                break
            raw.append(value)
        else:
            raise ValueError("Unterminated DLL import name")
        name = raw.decode("ascii")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+\.(?:dll|drv)", name, re.IGNORECASE):
            raise ValueError("Invalid DLL import basename")
        return name

    def imported(index, delayed=False):
        if index >= directory_count:
            return []
        rva, size = unpack("<II", optional + directory_offset + index * 8)
        if rva == 0 and size == 0:
            return []
        width = 32 if delayed else 20
        if not rva or size < width or size // width > 4096:
            raise ValueError("Invalid PE import directory size")
        names = []
        for offset in range(0, size - width + 1, width):
            values = unpack("<" + "I" * (width // 4), rva_offset(rva + offset, width))
            if not any(values):
                return sorted(set(names), key=str.lower)
            pointer = values[1] if delayed else values[3]
            if delayed and not values[0] & 1:
                pointer -= image_base
            names.append(dll_name(pointer))
        raise ValueError("Unterminated PE import directory")

    return {
        "machine": hex(machine),
        "architecture": next((name for name, value in ARCHITECTURES.items() if value == machine), "unsupported"),
        "imports": imported(1),
        "delay_imports": imported(13, True),
    }


def _safe_member(name):
    path = PurePosixPath(name)
    if (not name or "\\" in name or ":" in name or "\0" in name
            or path.is_absolute() or any(part in ("", ".", "..") for part in name.rstrip("/").split("/"))):
        raise ValueError("Unsafe ZIP member path")
    return path.as_posix()


def inspect(path, architecture, require_static_runtime=False):
    """Return JSON-compatible manifest; ``ok`` is the fail-closed gate result."""
    if architecture not in ARCHITECTURES:
        raise ValueError("Architecture must be x64 or ARM64")
    path = Path(path)
    result = {
        "schema_version": "uallera-native-imports-1",
        "source": str(path), "expected_architecture": architecture,
        "method": "Static PE normal and delay-import filename closure; no binary execution",
        "platform_policy": "Finite Windows 10/11 system DLL and observed API-set allowlists",
        "allowed_api_sets": sorted(SYSTEM_API_SETS),
        "require_static_runtime": require_static_runtime,
        "limitations": ["No export/symbol or minimum-OS verification", "Dynamic LoadLibrary dependencies are not visible", "No Windows execution or inference performed"],
        "files": [], "missing_imports": [], "forbidden_runtime_imports": [],
        "forbidden_runtime_files": [], "errors": [], "ok": False,
    }
    total = 0
    seen = set()

    def add(name, size, read):
        nonlocal total
        if name.lower() in seen:
            raise ValueError("Case-insensitive duplicate PE path: " + name)
        seen.add(name.lower())
        if len(seen) > MAX_FILES or size > MAX_PE_BYTES or total + size > MAX_TOTAL_BYTES:
            raise ValueError("PE inspection size/file budget exceeded")
        total += size
        try:
            data = read()
            if len(data) != size:
                raise ValueError("PE file size changed during inspection")
            item = {"path": name, "size_bytes": size, "sha256": hashlib.sha256(data).hexdigest()}
            item.update(pe_imports(data))
            result["files"].append(item)
            if item["architecture"] != architecture:
                result["errors"].append({"path": name, "error": "Architecture mismatch", "actual": item["architecture"], "machine": item["machine"]})
        except (ValueError, UnicodeError, OSError, zipfile.BadZipFile) as exc:
            result["errors"].append({"path": name, "error": str(exc)})

    try:
        if path.is_dir():
            for member in sorted(path.rglob("*")):
                if member.name.lower().endswith(PE_SUFFIXES):
                    if member.is_symlink() or not member.is_file():
                        raise ValueError("PE member is not a regular file")
                    add(member.relative_to(path).as_posix(), member.stat().st_size, member.read_bytes)
        elif path.is_file():
            with path.open("rb") as stream:
                result["source_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
            with zipfile.ZipFile(path) as bundle:
                for member in sorted(bundle.infolist(), key=lambda item: item.filename):
                    name = _safe_member(member.orig_filename)
                    if stat.S_ISLNK(member.external_attr >> 16):
                        raise ValueError("ZIP symlink is not supported")
                    if name.lower().endswith(PE_SUFFIXES) and not member.is_dir():
                        add(name, member.file_size, lambda item=member: bundle.read(item))
        else:
            raise ValueError("Input ZIP/directory does not exist")
    except (ValueError, OSError, zipfile.BadZipFile, RuntimeError) as exc:
        result["errors"].append({"path": str(path), "error": str(exc)})
    if not result["files"]:
        result["errors"].append({"path": str(path), "error": "No valid PE images found"})

    directories = {str(PurePosixPath(item["path"]).parent).lower() for item in result["files"]}
    if len(directories) > 1:
        result["errors"].append({"path": str(path), "error": "PE images span multiple directories; executable directory search cannot be established"})

    included = {item["path"].lower(): item for item in result["files"]}
    for item in result["files"]:
        if require_static_runtime and _is_runtime_dll(PurePosixPath(item["path"]).name):
            result["forbidden_runtime_files"].append(item["path"])
        item["dependencies"] = []
        for kind in ("imports", "delay_imports"):
            for dll in item[kind]:
                lower = dll.lower()
                sibling = (PurePosixPath(item["path"]).parent / lower).as_posix().lower()
                target = sibling if sibling in included else lower if lower in included else None
                resolution = "bundled" if target else "windows_system" if lower in SYSTEM_DLLS or lower in SYSTEM_API_SETS else "missing"
                dependency = {"dll": dll, "kind": kind, "resolution": resolution}
                if target:
                    dependency["target"] = included[target]["path"]
                item["dependencies"].append(dependency)
                if resolution == "missing":
                    result["missing_imports"].append({"importer": item["path"], "dll": dll, "kind": kind})
                if require_static_runtime and _is_runtime_dll(lower):
                    result["forbidden_runtime_imports"].append({"importer": item["path"], "dll": dll, "kind": kind})
    result["inspected_bytes"] = total
    result["ok"] = not any(result[field] for field in (
        "errors", "missing_imports", "forbidden_runtime_imports", "forbidden_runtime_files"))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--directory", type=Path)
    source.add_argument("--zip", type=Path)
    parser.add_argument("--architecture", "--arch", required=True, choices=tuple(ARCHITECTURES))
    parser.add_argument("--require-static-runtime", action="store_true",
                        help="Reject CRT/MSVC/OpenMP DLL imports even if bundled, and bundled runtime DLLs")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = inspect(args.directory or args.zip, args.architecture, args.require_static_runtime)
    content = json.dumps(result, indent=2, ensure_ascii=True) + "\n"
    if args.output:
        args.output.write_text(content, encoding="utf-8")
    print(content, end="")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
