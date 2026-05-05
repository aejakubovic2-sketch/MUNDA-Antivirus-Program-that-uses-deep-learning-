"""
file_identifier.py
Detects the type of a binary file: PE (Win32/Win64/.NET), ELF (Linux),
APK (Android), PDF, or UNKNOWN.
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


def identify_file(filepath: str) -> dict:
    """
    Returns a dict with:
      - file_type: 'WIN32' | 'WIN64' | 'DOTNET' | 'ELF' | 'APK' | 'PDF' | 'UNKNOWN'
      - platform:  'Windows' | 'Linux' | 'Android' | 'Document' | 'Unknown'
      - size_bytes: int
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    size = os.path.getsize(filepath)

    with open(filepath, 'rb') as f:
        header = f.read(16)

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
