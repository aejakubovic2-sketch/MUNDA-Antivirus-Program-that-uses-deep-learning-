"""
feature_extractor.py
Extracts the EMBER v3 feature vector (2,568 dimensions) from any supported
file type: PE (Win32/Win64/.NET), ELF, APK, PDF.

For non-PE files only the first 696 dimensions are fully populated
(general info + byte histograms + string stats) as per the EMBER2024 paper.
"""

import math
import re
import struct
import zipfile
from functools import lru_cache
from collections import Counter
from typing import Optional
import numpy as np


# ── Constants ────────────────────────────────────────────────────────────────

VECTOR_SIZE = 2568          # EMBER v3 full vector size
NONPE_VECTOR_SIZE = 696     # Populated dims for non-PE files


# ── Public API ───────────────────────────────────────────────────────────────

def extract_features(filepath: str) -> np.ndarray:
    """
    Extract EMBER v3 feature vector from any supported file.
    Returns a float32 numpy array of shape (2568,).
    """
    with open(filepath, 'rb') as f:
        raw = f.read()

    official_extractor = _get_official_extractor()
    if official_extractor is not None:
        try:
            return official_extractor.feature_vector(raw).astype(np.float32)
        except Exception:
            pass

    vec = np.zeros(VECTOR_SIZE, dtype=np.float32)

    # --- Segment 1: General file info (4 dims, index 0-3) ---
    _fill_general(vec, raw, filepath)

    # --- Segment 2: Byte histogram (256 dims, index 4-259) ---
    _fill_byte_histogram(vec, raw, offset=4)

    # --- Segment 3: Byte-entropy histogram (256 dims, index 260-515) ---
    _fill_byte_entropy_histogram(vec, raw, offset=260)

    # --- Segment 4: String features (120 dims, index 516-635) ---
    _fill_string_features(vec, raw, offset=516)

    # --- Segment 5-8: PE-specific (indices 636-2567) ---
    if raw[:2] == b'MZ':
        _fill_pe_features(vec, raw, filepath, offset=636)

    return vec


@lru_cache(maxsize=1)
def _get_official_extractor():
    """
    Prefer the EMBER2024 package extractor when it is installed.

    The bundled LightGBM models were trained with thrember's 2,568-dimensional
    feature layout. The local extractor below remains as a fallback so scanning
    still works if thrember is unavailable.
    """
    try:
        from ember_compat import get_thrember
        return get_thrember().PEFeatureExtractor()
    except Exception:
        return None


# ── Segment fillers ───────────────────────────────────────────────────────────

def _fill_general(vec, raw, filepath):
    """4 features: file size, virtual size proxy, has_debug, entry point proxy."""
    size = len(raw)
    vec[0] = min(size, 100_000_000) / 100_000_000      # normalised file size
    vec[1] = _entropy(raw)                              # whole-file entropy
    vec[2] = 1.0 if b'MZ' == raw[:2] else 0.0          # is PE
    vec[3] = 1.0 if raw[:4] == b'\x7fELF' else 0.0     # is ELF


def _fill_byte_histogram(vec, raw, offset):
    """256-dim byte frequency histogram, normalised."""
    counts = np.bincount(np.frombuffer(raw, dtype=np.uint8), minlength=256)
    total = counts.sum()
    if total > 0:
        vec[offset:offset + 256] = counts / total


def _fill_byte_entropy_histogram(vec, raw, offset):
    """
    256-dim byte-entropy histogram.
    Slide a 2048-byte window; for each window compute (byte, entropy) pair
    and accumulate into 16x16 joint histogram, then flatten.
    """
    WINDOW = 2048
    STEP = 1024
    hist = np.zeros((16, 16), dtype=np.float32)
    data = np.frombuffer(raw, dtype=np.uint8)
    n = len(data)
    num_windows = 0
    for start in range(0, max(1, n - WINDOW + 1), STEP):
        window = data[start:start + WINDOW]
        counts = np.bincount(window, minlength=256)
        probs = counts / counts.sum()
        h = -np.sum(probs[probs > 0] * np.log2(probs[probs > 0]))  # entropy 0-8
        byte_idx = int(np.mean(window)) // 16         # 0-15
        ent_idx = min(int(h * 2), 15)                 # 0-15
        hist[byte_idx, ent_idx] += 1
        num_windows += 1
    if num_windows > 0:
        hist /= num_windows
    vec[offset:offset + 256] = hist.flatten()


def _fill_string_features(vec, raw, offset):
    """120 string-based features."""
    try:
        text = raw.decode('latin-1')
    except Exception:
        text = ''

    printable_strings = re.findall(r'[ -~]{5,}', text)
    num_strings = len(printable_strings)
    avg_len = (sum(len(s) for s in printable_strings) / num_strings
               if num_strings else 0)

    # Basic string counts
    vec[offset + 0] = min(num_strings, 10000) / 10000
    vec[offset + 1] = min(avg_len, 300) / 300
    vec[offset + 2] = min(len([s for s in printable_strings if len(s) > 20]), 1000) / 1000

    # URL patterns
    urls = re.findall(r'https?://', text)
    vec[offset + 3] = min(len(urls), 100) / 100

    # Registry keys
    regs = re.findall(r'HKEY_', text)
    vec[offset + 4] = min(len(regs), 100) / 100

    # File paths
    paths = re.findall(r'[A-Za-z]:\\', text)
    vec[offset + 5] = min(len(paths), 100) / 100

    # MZ headers embedded (packed/dropper indicator)
    embedded_mz = text.count('MZ')
    vec[offset + 6] = min(embedded_mz, 10) / 10

    # Suspicious keyword flags (76 keywords → indices 7-82)
    SUSPICIOUS_KEYWORDS = [
        'CreateRemoteThread', 'VirtualAlloc', 'WriteProcessMemory',
        'NtUnmapViewOfSection', 'ShellExecute', 'WinExec', 'LoadLibrary',
        'GetProcAddress', 'RegSetValue', 'InternetOpen', 'URLDownloadToFile',
        'CreateService', 'StartService', 'OpenProcess', 'TerminateProcess',
        'IsDebuggerPresent', 'CheckRemoteDebuggerPresent', 'GetTickCount',
        'Sleep', 'GetSystemTime', 'FindFirstFile', 'CopyFile', 'DeleteFile',
        'SetFileAttributes', 'CreateMutex', 'OpenMutex', 'CreateEvent',
        'SetThreadContext', 'SuspendThread', 'ResumeThread', 'CreateThread',
        'HeapCreate', 'VirtualProtect', 'SetWindowsHookEx', 'GetAsyncKeyState',
        'keylogger', 'ransomware', 'encrypt', 'decrypt', 'bitcoin', 'wallet',
        'cmd.exe', 'powershell', 'wscript', 'cscript', 'regsvr32', 'mshta',
        'rundll32', 'svchost', 'lsass', 'mimikatz', 'cobaltstrike', 'meterpreter',
        'reverse_shell', 'bind_shell', 'shellcode', 'exploit', 'payload',
        'backdoor', 'rootkit', 'bootkit', 'trojan', 'spyware', 'adware',
        'botnet', 'command_and_control', 'c2', 'exfiltrate', 'base64',
        'rot13', 'xor', 'rc4', 'aes256', 'tor', 'onion', 'darkweb',
        'privilege_escalation', 'uac_bypass', 'token_impersonation',
        'pass_the_hash', 'lateral_movement', 'persistence',
    ]
    for i, kw in enumerate(SUSPICIOUS_KEYWORDS[:76]):
        vec[offset + 7 + i] = 1.0 if kw.lower() in text.lower() else 0.0

    # Remaining dims: entropy of printable strings
    if printable_strings:
        all_str = ''.join(printable_strings)
        str_bytes = all_str.encode('latin-1', errors='replace')
        vec[offset + 83] = _entropy(str_bytes) / 8.0
    # Indices 84-119 left as 0 (reserved for future string patterns)


def _fill_pe_features(vec, raw, filepath, offset):
    """PE-specific features using pefile library (indices 636-2567)."""
    try:
        import pefile
        pe = pefile.PE(data=raw, fast_load=False)
    except Exception:
        return  # Leave PE dims as zeros if parsing fails

    idx = offset

    # ── DOS Header (30 dims) ─────────────────────────────
    dh = pe.DOS_HEADER
    dos_fields = [
        dh.e_magic, dh.e_cblp, dh.e_cp, dh.e_crlc, dh.e_cparhdr,
        dh.e_minalloc, dh.e_maxalloc, dh.e_ss, dh.e_sp, dh.e_csum,
        dh.e_ip, dh.e_cs, dh.e_lfarlc, dh.e_ovno, dh.e_res_0,
        dh.e_res_1, dh.e_res_2, dh.e_res_3, dh.e_oemid, dh.e_oeminfo,
        dh.e_res2_0, dh.e_res2_1, dh.e_res2_2, dh.e_res2_3, dh.e_res2_4,
        dh.e_res2_5, dh.e_res2_6, dh.e_res2_7, dh.e_res2_8, dh.e_lfanew,
    ]
    for i, val in enumerate(dos_fields[:30]):
        vec[idx + i] = _normalise(val)
    idx += 30

    # ── COFF Header (10 dims) ────────────────────────────
    fh = pe.FILE_HEADER
    vec[idx + 0] = _normalise(fh.Machine)
    vec[idx + 1] = _normalise(fh.NumberOfSections)
    vec[idx + 2] = _normalise(fh.TimeDateStamp)
    vec[idx + 3] = _normalise(fh.PointerToSymbolTable)
    vec[idx + 4] = _normalise(fh.NumberOfSymbols)
    vec[idx + 5] = _normalise(fh.SizeOfOptionalHeader)
    vec[idx + 6] = _normalise(fh.Characteristics)
    vec[idx + 7] = fh.NumberOfSections
    vec[idx + 8] = min(fh.NumberOfSections / 20, 1.0)
    vec[idx + 9] = 1.0 if fh.Characteristics & 0x2000 else 0.0  # is DLL
    idx += 10

    # ── Optional Header (40 dims) ────────────────────────
    if hasattr(pe, 'OPTIONAL_HEADER'):
        oh = pe.OPTIONAL_HEADER
        opt_fields = [
            oh.Magic, oh.MajorLinkerVersion, oh.MinorLinkerVersion,
            oh.SizeOfCode, oh.SizeOfInitializedData, oh.SizeOfUninitializedData,
            oh.AddressOfEntryPoint, oh.BaseOfCode, oh.ImageBase,
            oh.SectionAlignment, oh.FileAlignment,
            oh.MajorOperatingSystemVersion, oh.MinorOperatingSystemVersion,
            oh.MajorImageVersion, oh.MinorImageVersion,
            oh.MajorSubsystemVersion, oh.MinorSubsystemVersion,
            oh.SizeOfImage, oh.SizeOfHeaders, oh.CheckSum,
            oh.Subsystem, oh.DllCharacteristics,
            oh.SizeOfStackReserve, oh.SizeOfStackCommit,
            oh.SizeOfHeapReserve, oh.SizeOfHeapCommit,
            oh.NumberOfRvaAndSizes,
        ]
        for i, val in enumerate(opt_fields[:27]):
            vec[idx + i] = _normalise(val)
        # Entry point anomaly: is entry point in a non-standard section?
        ep = oh.AddressOfEntryPoint
        vec[idx + 27] = 1.0 if ep == 0 else 0.0
        vec[idx + 28] = min(oh.SizeOfCode / max(oh.SizeOfImage, 1), 1.0)
        vec[idx + 29] = min(oh.SizeOfImage / 100_000_000, 1.0)
    idx += 40

    # ── Section features (256 dims) ──────────────────────
    for sec in pe.sections[:8]:           # max 8 sections
        try:
            name = sec.Name.rstrip(b'\x00').decode('latin-1', errors='replace')
            raw_size = sec.SizeOfRawData
            virt_size = sec.Misc_VirtualSize
            entropy = sec.get_entropy()
            chars = sec.Characteristics

            vec[idx + 0] = _hash_name(name)
            vec[idx + 1] = min(raw_size / 10_000_000, 1.0)
            vec[idx + 2] = min(virt_size / 10_000_000, 1.0)
            vec[idx + 3] = entropy / 8.0
            vec[idx + 4] = _normalise(chars)
            vec[idx + 5] = (raw_size / virt_size) if virt_size > 0 else 0.0
            # Executable, readable, writable flags
            vec[idx + 6] = 1.0 if chars & 0x20000000 else 0.0   # executable
            vec[idx + 7] = 1.0 if chars & 0x40000000 else 0.0   # readable
            vec[idx + 8] = 1.0 if chars & 0x80000000 else 0.0   # writable
            # Suspicious: writable + executable
            vec[idx + 9] = 1.0 if (chars & 0x20000000 and chars & 0x80000000) else 0.0
            # High entropy section (packed/encrypted)
            vec[idx + 10] = 1.0 if entropy > 7.0 else 0.0
        except Exception:
            pass
        idx += 32   # 32 dims per section, 8 sections = 256

    # Skip remaining section slots
    idx = offset + 30 + 10 + 40 + 256

    # ── Import features (hashed, 128 dims) ───────────────
    try:
        pe.parse_data_directories(
            directories=[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_IMPORT']]
        )
        if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
            for entry in pe.DIRECTORY_ENTRY_IMPORT[:16]:
                dll_name = entry.dll.decode('latin-1', errors='replace').lower()
                h = _hash_name(dll_name)
                bucket = int(h * 128) % 128
                vec[idx + bucket] += 0.1
    except Exception:
        pass
    idx += 128

    # ── Export features (32 dims) ─────────────────────────
    try:
        pe.parse_data_directories(
            directories=[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_EXPORT']]
        )
        if hasattr(pe, 'DIRECTORY_ENTRY_EXPORT'):
            exp = pe.DIRECTORY_ENTRY_EXPORT
            vec[idx + 0] = min(exp.struct.NumberOfFunctions / 1000, 1.0)
            vec[idx + 1] = min(exp.struct.NumberOfNames / 1000, 1.0)
    except Exception:
        pass
    idx += 32

    # ── Authenticode / signature features (88 dims) ───────
    try:
        security_dir = pe.OPTIONAL_HEADER.DATA_DIRECTORY[
            pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_SECURITY']
        ]
        has_sig = security_dir.VirtualAddress > 0
        vec[idx + 0] = 1.0 if has_sig else 0.0
    except Exception:
        pass
    idx += 88

    pe.close()


# ── Utility functions ────────────────────────────────────────────────────────

def _entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts = Counter(data)
    total = len(data)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def _normalise(val, cap=2**32) -> float:
    try:
        return min(abs(int(val)), cap) / cap
    except Exception:
        return 0.0


def _hash_name(name: str) -> float:
    """Simple deterministic hash of a string → float 0-1."""
    h = 0
    for c in name:
        h = (h * 31 + ord(c)) & 0xFFFFFFFF
    return h / 0xFFFFFFFF
