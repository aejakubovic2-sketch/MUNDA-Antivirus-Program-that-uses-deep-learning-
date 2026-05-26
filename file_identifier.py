"""
file_identifier.py
Detects the type of a binary file: PE (Win32/Win64/.NET), ELF (Linux),
APK (Android), PDF, macOS Mach-O/DMG/App bundle, or UNKNOWN.
Uses magic bytes — no libraries needed for basic detection.
"""

import os
import zipfile


# Magic byte signatures
MAGIC = {
    b'MZ':         'PE',       # Windows PE (exe, dll, sys)
    b'\x7fELF':    'ELF',      # Linux ELF
    b'%PDF':       'PDF',      # PDF document
}

MACHO_MAGICS = {
    b'\xfe\xed\xfa\xce',
    b'\xce\xfa\xed\xfe',
    b'\xfe\xed\xfa\xcf',
    b'\xcf\xfa\xed\xfe',
    b'\xca\xfe\xba\xbe',
    b'\xbe\xba\xfe\xca',
    b'\xca\xfe\xba\xbf',
}


def identify_file(filepath: str) -> dict:
    """
    Returns a dict with:
      - file_type: 'WIN32' | 'WIN64' | 'DOTNET' | 'ELF' | 'APK' | 'PDF'
                   | 'MACHO' | 'DMG' | 'MACAPP' | 'UNKNOWN'
      - platform:  'Windows' | 'Linux' | 'Android' | 'Document' | 'macOS' | 'Unknown'
      - size_bytes: int
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    if os.path.isdir(filepath):
        if filepath.lower().endswith('.app'):
            return {
                'file_type': 'MACAPP',
                'platform': 'macOS',
                'size_bytes': _directory_size(filepath),
            }
        raise IsADirectoryError(f"Directory is not a supported file target: {filepath}")

    size = os.path.getsize(filepath)
    suffix = os.path.splitext(filepath)[1].lower()

    with open(filepath, 'rb') as f:
        header = f.read(16)

    # --- macOS DMG disk image ---
    if suffix == '.dmg' or _has_udif_trailer(filepath):
        return {'file_type': 'DMG', 'platform': 'macOS', 'size_bytes': size}

    # --- macOS Mach-O executable/library ---
    if header[:4] in MACHO_MAGICS:
        return {'file_type': 'MACHO', 'platform': 'macOS', 'size_bytes': size}

    # --- PDF ---
    if header[:4] == b'%PDF':
        return {'file_type': 'PDF', 'platform': 'Document', 'size_bytes': size}

    # --- ELF (Linux) ---
    if header[:4] == b'\x7fELF':
        return {'file_type': 'ELF', 'platform': 'Linux', 'size_bytes': size}

    # --- APK (Android) = ZIP with AndroidManifest.xml ---
    if header[:2] == b'PK':
        try:
            with zipfile.ZipFile(filepath, 'r') as z:
                names = z.namelist()
            if 'AndroidManifest.xml' in names:
                return {'file_type': 'APK', 'platform': 'Android', 'size_bytes': size}
        except Exception:
            pass

    # --- Windows PE ---
    if header[:2] == b'MZ':
        platform, file_type = _classify_pe(filepath)
        return {'file_type': file_type, 'platform': platform, 'size_bytes': size}

    return {'file_type': 'UNKNOWN', 'platform': 'Unknown', 'size_bytes': size}


def _has_udif_trailer(filepath: str) -> bool:
    """DMG/UDIF images normally end with a 512-byte trailer starting with 'koly'."""
    try:
        if os.path.getsize(filepath) < 512:
            return False
        with open(filepath, 'rb') as f:
            f.seek(-512, os.SEEK_END)
            return f.read(4) == b'koly'
    except OSError:
        return False


def _directory_size(dirpath: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(dirpath):
        for filename in files:
            path = os.path.join(root, filename)
            try:
                total += os.path.getsize(path)
            except OSError:
                pass
    return total


def _classify_pe(filepath: str):
    """
    Distinguishes between Win32, Win64, and .NET PE files.
    """
    try:
        import pefile
        pe = pefile.PE(filepath, fast_load=True)

        # Check for .NET CLR header (data directory index 14)
        pe.parse_data_directories(
            directories=[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_COM_DESCRIPTOR']]
        )
        if (hasattr(pe, 'DIRECTORY_ENTRY_COM_DESCRIPTOR') and
                pe.DIRECTORY_ENTRY_COM_DESCRIPTOR):
            return 'Windows (.NET)', 'DOTNET'

        # Check machine type for 32 vs 64 bit
        machine = pe.FILE_HEADER.Machine
        if machine == 0x8664 or machine == 0xAA64:   # x64 or ARM64
            return 'Windows 64-bit', 'WIN64'
        else:
            return 'Windows 32-bit', 'WIN32'

    except Exception:
        # fallback — just call it a generic PE
        return 'Windows', 'WIN32'


def format_size(size_bytes: int) -> str:
    """Human-readable file size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"
