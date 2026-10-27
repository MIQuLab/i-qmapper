"""
paths.py — Centralized safe path resolution.

Single source of truth for user-provided file path sanitization. Use this
every time the user supplies a filename (JSON export, PNG export, session
save, layout import, ...).

Why a dedicated module:
  The previous code had two near-identical copies of `_safe_path` in
  ui.py and ui_qpu2.py, both using `str.startswith(cwd)` which has a
  well-known prefix-match bypass:

    cwd = '/home/alice'
    user input = '../alice_evil/data'
    realpath(...) = '/home/alice_evil/data'
    '/home/alice_evil/data'.startswith('/home/alice') == True  # BYPASS

  os.path.commonpath gives the correct answer — it compares path
  *components*, not raw string prefixes:

    commonpath(['/home/alice', '/home/alice_evil/data']) == '/home'
    '/home' != '/home/alice'  →  reject

Usage:
    from .paths import safe_path
    out = safe_path(user_input, default_ext='.json')
    # raises ValueError on unsafe input, otherwise returns absolute path
    with open(out, 'w') as f:
        ...
"""
from __future__ import annotations

import os

# Most modern POSIX filesystems (ext4, APFS, NTFS path-component limit) cap a
# single path component at 255 bytes. We apply the same cap to the raw
# user input to fail early with a clear error instead of producing an
# OSError("File name too long") deep inside open().
_MAX_FILENAME_BYTES = 255


def _has_dotdot_component(p: str) -> bool:
    """Return True iff `..` appears as a SEPARATE path component.

    Splits on both POSIX and Windows separators so that legitimate names
    containing the literal characters '..' (e.g. 'my..file.json') are
    allowed, while a real traversal attempt ('a/../b' or 'a\\..\\b') is
    rejected. The previous `'..' in p` check rejected the legitimate case
    along with the malicious one.
    """
    # Normalize Windows separators to POSIX for the component scan only.
    # The authoritative containment check below still goes through
    # os.path.realpath, which understands the native separator.
    parts = p.replace('\\', '/').split('/')
    return any(part == '..' for part in parts)


def safe_path(raw_path: str, default_ext: str = '') -> str:
    """Sanitize a user-provided filename and return an absolute path under CWD.

    Rules:
      - Empty string rejected.
      - Null bytes ('\\x00') rejected — some filesystems accept them but
        the resulting path is nearly impossible to delete with normal
        tools and is a known smuggling vector against C-level open().
      - Any '..' COMPONENT (separated by '/' or '\\\\') rejected. Plain
        names like 'my..file.json' that happen to contain the literal
        characters are allowed.
      - Absolute paths ('/...') rejected.
      - Home-expansion ('~...') rejected.
      - Path components longer than 255 bytes rejected — matches the
        per-component limit on ext4/APFS/NTFS and fails with a clear
        error instead of an opaque OSError from open().
      - After realpath resolution, the result MUST sit under the current
        working directory. Verified via os.path.commonpath, NOT via
        str.startswith (which is vulnerable to '../alice_evil' when cwd is
        '/home/alice').

    Args:
        raw_path: user-supplied filename (typically from a Text widget).
        default_ext: optional hint string used only in error messages.

    Returns:
        absolute, normalized path string, safe to pass to open().

    Raises:
        ValueError with a human-readable message on any rejection.
    """
    p = raw_path.strip() if isinstance(raw_path, str) else ''
    if not p:
        raise ValueError("Empty path")

    if '\x00' in p:
        raise ValueError("Null byte in path")

    # Fast rejection of obvious traversal attempts. commonpath below is the
    # authoritative check; these are just for clearer error messages.
    if _has_dotdot_component(p) or p.startswith('/') or p.startswith('~'):
        raise ValueError(
            f"Unsafe path: '{p}' — use a simple filename like 'my_file{default_ext}'"
        )

    # Per-component byte length check. Encoded with UTF-8 so multi-byte
    # filenames are measured in bytes (which is what the filesystem cares
    # about), not characters.
    for part in p.replace('\\', '/').split('/'):
        if len(part.encode('utf-8')) > _MAX_FILENAME_BYTES:
            raise ValueError(
                f"Path component too long ({len(part.encode('utf-8'))} bytes "
                f"> {_MAX_FILENAME_BYTES}): '{part[:40]}…'"
            )

    # Resolve to absolute and verify containment via commonpath.
    # realpath follows symlinks — intentional, so a symlink pointing out of
    # CWD is also rejected.
    cwd = os.path.realpath(os.getcwd())
    resolved = os.path.realpath(os.path.join(cwd, p))

    try:
        common = os.path.commonpath([cwd, resolved])
    except ValueError:
        # ValueError is raised by commonpath when paths are on different
        # drives on Windows. That unambiguously means the resolved path
        # is not under CWD. Suppress the internal cause — the clean
        # message is the whole point of raising this.
        raise ValueError(f"Path escapes working directory: '{p}'") from None

    if common != cwd:
        raise ValueError(f"Path escapes working directory: '{p}'")

    return resolved


def resolve_user_save_path(raw_path: str, default_ext: str = '',
                           default_dir: str | None = None) -> str:
    """Resolve a USER-TYPED export destination.

    Unlike `safe_path` (which confines to CWD — the right tool for paths
    derived from untrusted file content), this trusts the local user to
    choose where their own export goes: accepts absolute paths and `~`,
    resolves a bare/relative name against `default_dir` (CWD when None),
    creates the parent directory if missing, and applies `default_ext`
    when no extension is present. Empty input and NUL bytes are still
    rejected.

    Returns an absolute path. Raises ValueError on empty/NUL input or if
    the parent directory cannot be created.
    """
    if not raw_path or not raw_path.strip():
        raise ValueError("No filename given.")
    if '\x00' in raw_path:
        raise ValueError("Filename contains a NUL byte.")
    p = os.path.expanduser(raw_path.strip())
    if default_ext and not os.path.splitext(p)[1]:
        p = p + default_ext
    base = os.path.realpath(default_dir) if default_dir else os.path.realpath(os.getcwd())
    p = p if os.path.isabs(p) else os.path.join(base, p)
    p = os.path.realpath(p)
    parent = os.path.dirname(p)
    try:
        os.makedirs(parent, exist_ok=True)
    except OSError as e:
        raise ValueError(f"Cannot create folder '{parent}': {e}") from e
    return p
