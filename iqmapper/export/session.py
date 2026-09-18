"""
export.session — Session save/restore and style-preset save/restore,
including the path-safety primitives that guard those flows.

Sessions and presets are written under `.iqmapper_sessions/` and
`.iqmapper_styles/` next to this sub-module on disk. Names accepted
through `*_by_name` helpers are validated against `_SAFE_NAME_RE` to
keep them inside the managed directory.
"""
import json
import logging
import os
import re

from . import MAX_IMPORT_BYTES

_log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# SESSION SAVE / RESTORE
# ═══════════════════════════════════════════════════════════════════════

_SESSION_DIR = os.path.join(os.getcwd(), ".iqmapper_sessions")

# Allowed characters in a *session/style preset name* (the "name" argument
# used by save_session/save_style_preset and their _by_name loaders).
# Names are interpolated into filenames inside _SESSION_DIR/_STYLE_DIR, so
# anything that lets the user escape that directory (path separators, `..`,
# absolute prefixes, null bytes) MUST be rejected here — otherwise
# `load_session_by_name("../../etc/passwd")` reads files outside the
# managed dir. The whitelist is intentionally tight (no `/`, no `.`, no
# `\`, no spaces) so it's obvious what kind of names are accepted.
_SAFE_NAME_RE = re.compile(r'^[A-Za-z0-9_\-]+$')


def _validate_internal_name(name, label='session'):
    """Reject names that could traverse outside the managed session/style dir.

    Used by save_session/load_session_by_name/save_style_preset/load_style_by_name.
    For arbitrary user-chosen filesystem paths use `paths.safe_path` instead.
    """
    if not isinstance(name, str) or not _SAFE_NAME_RE.match(name):
        raise ValueError(
            f"Invalid {label} name: must be 1+ characters of "
            f"[A-Za-z0-9_-] (got {name!r})"
        )


def _reject_unsafe_path(path, label='output'):
    """Defence-in-depth path check for save_session_to_path / save_style_to_path.

    These APIs intentionally accept absolute paths (so `safe_path`'s CWD
    constraint can't apply), but still reject the smuggling shapes: null
    bytes (silently truncated on some filesystems → adjacent-file writes)
    and literal `..` components.
    """
    if not isinstance(path, str) or not path:
        raise ValueError(f"Invalid {label} path: empty or non-string")
    if '\x00' in path:
        raise ValueError(f"Invalid {label} path: contains null byte")
    parts = path.replace('\\', '/').split('/')
    if any(p == '..' for p in parts):
        raise ValueError(f"Invalid {label} path: contains '..' component")


def _ensure_private_dir(path):
    """Create directory and best-effort restrict it to owner-only (0o700)."""
    os.makedirs(path, exist_ok=True)
    try:
        os.chmod(path, 0o700)
    except OSError:
        _log.debug("chmod 0o700 failed for %s", path, exc_info=True)


def _open_private(path, mode='w'):
    """open(path, mode) but create the file 0o600 atomically.

    Avoids the umask race where plain open(...,'w') briefly leaves the file
    world-readable on shared hosts before chmod fires. Falls back to plain
    open() where os.open's mode arg is ignored (Windows; single-user threat
    model). Text mode + UTF-8 because callers always write JSON — change the
    signature, not this, if a binary caller ever appears.
    """
    if 'w' not in mode and 'x' not in mode:
        return open(path, mode, encoding='utf-8')
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    fd = os.open(path, flags, 0o600)
    return os.fdopen(fd, mode, encoding='utf-8')


def save_session(state, name, widget_values=None):
    """Save full session to internal dir. Returns filepath."""
    _validate_internal_name(name, 'session')
    _ensure_private_dir(_SESSION_DIR)
    session = state.to_session_dict(widget_values=widget_values)
    fname = f"{name}.qpusession.json"
    path = os.path.join(_SESSION_DIR, fname)
    # `_open_private` already pins encoding='utf-8' so non-ASCII labels
    # (custom circuit names, calibration timestamps) round-trip cleanly
    # regardless of the host platform's default text encoding.
    with _open_private(path, 'w') as f:
        json.dump(session, f, indent=2, ensure_ascii=False)
    return path


def save_session_to_path(state, path, widget_values=None):
    """Save full session to a user-chosen path. Creates parent dirs if needed.

    Convention: callers are expected to have routed `path` through
    `paths.safe_path` (or an equivalent CWD-containment check) before
    handing it here. The defence-in-depth call below catches the
    obvious smuggling shapes (null bytes, literal `..` components) even
    if a future caller forgets — but it does NOT replace `safe_path`,
    because this API intentionally accepts absolute paths.
    """
    _reject_unsafe_path(path, 'session')
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    session = state.to_session_dict(widget_values=widget_values)
    with _open_private(path, 'w') as f:
        json.dump(session, f, indent=2, ensure_ascii=False)
    return path


def load_session(state, file_bytes):
    """
    Load session from file bytes.
    Returns (backend_name, widget_values) — caller must reconnect backend.

    Raises ValueError on any of:
      • file too large (defence against a multi-GB JSON eating the
        kernel before validation runs),
      • JSON parse failure (re-raised from json.loads),
      • payload is not a JSON object (top-level array/scalar rejected),
      • `_version` missing / too old / too new (delegated to
        `state.from_session_dict`, which owns the version policy).
    """
    if len(file_bytes) > MAX_IMPORT_BYTES:
        raise ValueError(
            f"Session file too large: {len(file_bytes) / 1_048_576:.1f} MB "
            f"exceeds the {MAX_IMPORT_BYTES // 1_048_576} MB limit."
        )
    session = json.loads(file_bytes.decode('utf-8'))
    # Top-level must be a dict: a JSON list or scalar would cause `.get`
    # to crash with AttributeError further down. Reject up-front with a
    # user-friendly message.
    if not isinstance(session, dict):
        raise ValueError(
            f"Session payload is not a JSON object "
            f"(got {type(session).__name__})")
    # Version validation happens inside `state.from_session_dict` —
    # single source of truth so the supported-range only needs to be
    # bumped in one place when SessionState gains / drops fields.
    widget_values = state.from_session_dict(session)
    return session.get('backend_name'), widget_values


def list_saved_sessions():
    """Return list of saved session names."""
    if not os.path.exists(_SESSION_DIR):
        return []
    return sorted([
        f.replace('.qpusession.json', '')
        for f in os.listdir(_SESSION_DIR)
        if f.endswith('.qpusession.json')
    ])


def load_session_by_name(state, name):
    """Load a session by name from disk.

    `name` MUST match `_SAFE_NAME_RE` (alphanumeric + `_` + `-`).
    A bare `os.path.join(_SESSION_DIR, name + ext)` with an unvalidated
    name lets `name="../../etc/passwd"` resolve outside the managed
    directory; the validator below blocks that vector up-front.
    """
    _validate_internal_name(name, 'session')
    path = os.path.join(_SESSION_DIR, f"{name}.qpusession.json")
    with open(path, 'rb') as f:
        return load_session(state, f.read())


# ═══════════════════════════════════════════════════════════════════════
# STYLE PRESET SAVE / RESTORE
# ═══════════════════════════════════════════════════════════════════════

_STYLE_DIR = os.path.join(os.getcwd(), ".iqmapper_styles")


def save_style_preset(state, name, widget_values=None):
    """Save visual style preset to internal dir. Returns filepath."""
    _validate_internal_name(name, 'style preset')
    _ensure_private_dir(_STYLE_DIR)
    style = state.to_style_dict(widget_values=widget_values)
    fname = f"{name}.qpustyle.json"
    path = os.path.join(_STYLE_DIR, fname)
    with _open_private(path, 'w') as f:
        json.dump(style, f, indent=2, ensure_ascii=False)
    return path


def save_style_to_path(state, path, widget_values=None):
    """Save visual style preset to a user-chosen path.

    See `save_session_to_path` for the safe-path contract; the same
    rules apply here.
    """
    _reject_unsafe_path(path, 'style preset')
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    style = state.to_style_dict(widget_values=widget_values)
    with _open_private(path, 'w') as f:
        json.dump(style, f, indent=2, ensure_ascii=False)
    return path


def load_style_preset(state, file_bytes):
    """Load style from file bytes. Returns widget_values or None."""
    if len(file_bytes) > MAX_IMPORT_BYTES:
        raise ValueError(
            f"Style preset file too large: {len(file_bytes) / 1_048_576:.1f} MB "
            f"exceeds the {MAX_IMPORT_BYTES // 1_048_576} MB limit."
        )
    style = json.loads(file_bytes.decode('utf-8'))
    if style.get('_type') != 'iqmapper_style':
        raise ValueError("Not a valid I-QMapper style file")
    return state.from_style_dict(style)


def list_saved_styles():
    """Return list of saved style preset names."""
    if not os.path.exists(_STYLE_DIR):
        return []
    return sorted([
        f.replace('.qpustyle.json', '')
        for f in os.listdir(_STYLE_DIR)
        if f.endswith('.qpustyle.json')
    ])


def load_style_by_name(state, name):
    """Load a style preset by name from disk.

    `name` MUST match `_SAFE_NAME_RE`. See `load_session_by_name` for
    the path-traversal rationale.
    """
    _validate_internal_name(name, 'style preset')
    path = os.path.join(_STYLE_DIR, f"{name}.qpustyle.json")
    with open(path, 'rb') as f:
        return load_style_preset(state, f.read())

