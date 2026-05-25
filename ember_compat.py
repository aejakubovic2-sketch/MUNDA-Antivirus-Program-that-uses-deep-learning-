"""
ember_compat.py
Compatibility helpers for the EMBER2024/thrember package.

Some current signify releases expose SignedPEFile from a submodule instead of
signify.authenticode directly. thrember imports the old location, so we patch
that symbol before importing thrember.
"""

import os
import tempfile
from functools import lru_cache


@lru_cache(maxsize=1)
def get_thrember():
    """Return the thrember module after applying local compatibility patches."""
    os.environ.setdefault(
        'MPLCONFIGDIR',
        os.path.join(tempfile.gettempdir(), 'munda-matplotlib'),
    )

    import signify.authenticode as authenticode
    if not hasattr(authenticode, 'SignedPEFile'):
        from signify.authenticode.signed_file import SignedPEFile
        authenticode.SignedPEFile = SignedPEFile

    import thrember
    return thrember
