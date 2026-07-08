"""
data.py — Data fetching and device graph generation.

Cache: key = last_update_date timestamp (not day, not TTL); per-backend
index file tracks all known calibrations; one API call checks whether
the calibration changed, extraction only on a miss; historical dates
cached permanently. Public API: get_props_for_datetime(),
list_known_calibrations().
"""
import logging
import os
import json as _json
import re
import time
import threading
import tempfile
import numpy as np
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from .constants import QUBIT_PROP_OPTIONS, GATE_PROP_OPTIONS

_log = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_BACKOFF = 1.5

# Whether `backend.properties(datetime=...)` accepts the kwarg —
# probed once per session. qiskit-ibm-runtime ~0.30+ removed the kwarg;
# without this guard, snapshot / scan-day / history-fetch would crash
# with a generic "Fetch failed" banner and no clue why.
_HAS_HISTORICAL_PROPS = None


def _supports_historical_props(backend):
    """Return True if backend.properties(datetime=...) accepts the kwarg.
    Cached globally — qiskit-ibm-runtime version doesn't change mid-session.
    """
    # backend=None: treat as no support WITHOUT caching, so a later call
    # with a real backend can still probe.
    if backend is None:
        return False
    global _HAS_HISTORICAL_PROPS
    if _HAS_HISTORICAL_PROPS is None:
        try:
            import inspect
            sig = inspect.signature(backend.properties)
            _HAS_HISTORICAL_PROPS = 'datetime' in sig.parameters
        except (ValueError, TypeError, AttributeError):
            # signature() can raise on C-level callables/builtins;
            # AttributeError covers backends without this property.
            # Optimistic True so we still attempt the call; if it
            # fails, _retry surfaces the original error.
            _HAS_HISTORICAL_PROPS = True
        if not _HAS_HISTORICAL_PROPS:
            _log.warning(
                "This qiskit-ibm-runtime version does not support "
                "backend.properties(datetime=...). Time-machine features "
                "(Snapshot, Scan day, Multi-day history) will be disabled. "
                "Pin to qiskit-ibm-runtime<0.30 to restore them."
            )
    return _HAS_HISTORICAL_PROPS


_HISTORICAL_DISABLED_MSG = (
    "Historical calibration is unavailable: this qiskit-ibm-runtime "
    "version removed backend.properties(datetime=...). "
    "Pin to qiskit-ibm-runtime<0.30 to use Snapshot / Scan day / Multi-day."
)


def _retry(fn, *args, retries=_MAX_RETRIES, **kwargs):
    """Call fn(*args, **kwargs) with exponential backoff on failure."""
    last_exc = None
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1:
                wait = _RETRY_BACKOFF * (2 ** attempt)
                _log.debug("Retry %d/%d for %s after %.1fs: %s",
                           attempt + 1, retries, fn.__name__, wait, exc)
                time.sleep(wait)
    raise last_exc

_SQ_GATE_ERROR_MAP = {
    'id_error': 'id', 'rx_error': 'rx', 'sx_error': 'sx',
    'rz_error': 'rz', 'x_error': 'x',
}
_TWO_Q_ERROR_GATES = {'cz_error': 'cz', 'rzz_error': 'rzz'}


# ── Property fetchers ────────────────────────────────────────────────

def fetch_qubit_prop(props, qubit_idx, prop_key):
    try:
        n_qubits = len(props.qubits)
        if qubit_idx < 0 or qubit_idx >= n_qubits:
            return np.nan
    except Exception:
        pass
    try:
        if prop_key == 'readout_error':
            return float(props.readout_error(qubit_idx))
        if prop_key == 't1':
            return float(props.t1(qubit_idx)) * 1e6
        if prop_key == 't2':
            return float(props.t2(qubit_idx)) * 1e6
        if prop_key in ('prob_meas0_prep1', 'prob_meas1_prep0'):
            return float(props.qubit_property(qubit_idx, prop_key)[0])
        if prop_key == 'readout_length':
            try:
                return float(props.readout_length(qubit_idx)) * 1e9
            except (AttributeError, TypeError):
                return float(props.qubit_property(qubit_idx, 'readout_length')[0]) * 1e9
        if prop_key in _SQ_GATE_ERROR_MAP:
            return float(props.gate_error(_SQ_GATE_ERROR_MAP[prop_key], [qubit_idx]))
        if prop_key == 'sq_gate_length':
            for gn in ('sx', 'x', 'id'):
                try:
                    return float(props.gate_length(gn, [qubit_idx])) * 1e9
                except Exception:
                    continue
            return np.nan
        if prop_key == 'measure_error':
            return float(props.gate_error('measure', [qubit_idx]))
        if prop_key == 'measure_2_error':
            try:
                return float(props.gate_error('measure_2', [qubit_idx]))
            except Exception:
                return np.nan
    except Exception as exc:
        _log.warning("fetch_qubit_prop failed: qubit=%d prop='%s' — %s", qubit_idx, prop_key, exc)
    return np.nan


def fetch_gate_prop(props, edge, prop_key):
    qubits = list(edge)
    try:
        if prop_key in _TWO_Q_ERROR_GATES:
            return float(props.gate_error(_TWO_Q_ERROR_GATES[prop_key], qubits))
        if prop_key == 'gate_length':
            for gn in ('cz', 'rzz'):
                try:
                    return float(props.gate_length(gn, qubits)) * 1e9
                except Exception:
                    continue
            return np.nan
    except Exception as exc:
        _log.warning("fetch_gate_prop failed: edge=%s prop='%s' — %s", edge, prop_key, exc)
    return np.nan


def _props_shape_matches(qp, gp, n_q, n_e):
    """A cached snapshot is usable only if BOTH qubit_props and gate_props match the
    backend's shape. A qubit-count-only check would accept a snapshot from a
    same-qubit-count but different-topology backend (e.g. Heron r2 vs r3), after
    which gate_props indexing silently runs off the end.
    """
    return (len(next(iter(qp.values()), [])) == n_q and
            len(next(iter(gp.values()), [])) == n_e)


def _extract_last_update(props, *, display=False):
    """Extract last_update_date as a string. Returns None on any failure.

    display=False → UTC-normalised "%Y-%m-%d %H:%M:%S" (for cache keys; always UTC).
    display=True  → original timezone with an offset label, for showing the user.
    A naive datetime is formatted as-is in both modes.
    """
    try:
        dt = props.last_update_date
        if dt is not None:
            if hasattr(dt, 'tzinfo') and dt.tzinfo is not None:
                if display:
                    offset = dt.strftime("%z")  # e.g. "-0500"
                    return dt.strftime(f"%Y-%m-%d %H:%M:%S ({offset})")
                from datetime import timezone
                dt = dt.astimezone(timezone.utc)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        pass
    return None


# ── Timezone helpers ─────────────────────────────────────────────────
# Cache is keyed by UTC (what `last_update_date` ships, stable across
# the user's travel / DST). UI surfaces convert to local time at the
# boundary — otherwise a user at 15:00 CET seeing "23:00 UTC" can't
# tell if the timestamp is past, future, or on a different day. Cache
# code keeps using UTC strings unchanged.

def utc_str_to_local_dt(utc_str):
    """Parse a cached UTC timestamp string ('YYYY-MM-DD HH:MM:SS') into a
    timezone-aware datetime in the system's local timezone. Returns None
    on parse failure."""
    if not utc_str or len(utc_str) < 19:
        return None
    try:
        dt_utc = datetime.strptime(utc_str[:19], "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=timezone.utc)
        return dt_utc.astimezone()
    except Exception:
        return None


def local_dt_to_utc_naive(local_dt):
    """Convert a local-meaning naive datetime (e.g. user typed HH:MM
    intending their wall-clock time) into a naive UTC datetime suitable
    for use as a cache key / API parameter."""
    if local_dt is None:
        return None
    if local_dt.tzinfo is None:
        # Attach the system local TZ so astimezone() does the right thing.
        local_dt = local_dt.replace(tzinfo=datetime.now().astimezone().tzinfo)
    return local_dt.astimezone(timezone.utc).replace(tzinfo=None)


def utc_str_to_local_label(utc_str, ref_date=None):
    """Format a UTC timestamp for display as local-time HH:MM. If the
    local date differs from `ref_date` (the user's DatePicker day),
    append a short date hint so the day boundary is visible — e.g.
    23:30 UTC is 01:30 *the next day* in CET, which a bare HH:MM hides."""
    local_dt = utc_str_to_local_dt(utc_str)
    if local_dt is None:
        # Fall back to the raw HH:MM slice so the caller gets *something*.
        return utc_str[11:16] if utc_str and len(utc_str) > 15 else (utc_str or '?')
    base = local_dt.strftime("%H:%M")
    if ref_date is not None and local_dt.date() != ref_date:
        return f"{base} ({local_dt.strftime('%b %d')})"
    return base


def _audit_nan_fraction(qubit_props_data, gate_props_data, threshold=0.10):
    warnings = []
    for label, data_dict in [("qubit", qubit_props_data), ("gate", gate_props_data)]:
        for key, arr in data_dict.items():
            if not hasattr(arr, '__len__') or len(arr) == 0:
                continue
            nan_count = int(np.sum(np.isnan(arr.astype(float))))
            total = len(arr)
            if nan_count == 0 or nan_count == total:
                continue
            if nan_count / total >= threshold:
                warnings.append(f"{key}: {nan_count}/{total} ({nan_count/total:.0%}) NaN")
    return warnings


# ── Cache system (timestamp-based) ──────────────────────────────────


def _default_cache_dir():
    """Per-user cache directory for qpu_mapper.

    Order: an explicit QPU_MAPPER_CACHE_DIR override, then the OS standard
    (platformdirs), then a stdlib fallback. Examples of the OS standard:
      macOS   ~/Library/Caches/qpu_mapper
      Linux   ~/.cache/qpu_mapper   (or $XDG_CACHE_HOME/qpu_mapper)
      Windows %LOCALAPPDATA%\\qpu_mapper\\Cache
    """
    env = os.environ.get('QPU_MAPPER_CACHE_DIR')
    if env:
        return os.path.expanduser(env)
    try:
        import platformdirs
        return platformdirs.user_cache_dir('qpu_mapper')
    except Exception:
        if os.name == 'nt':
            base = os.environ.get('LOCALAPPDATA') or os.path.expanduser('~')
            return os.path.join(base, 'qpu_mapper', 'Cache')
        xdg = os.environ.get('XDG_CACHE_HOME')
        base = xdg if xdg else os.path.join(os.path.expanduser('~'), '.cache')
        return os.path.join(base, 'qpu_mapper')


_CACHE_DIR = _default_cache_dir()

# One-time migration: if the legacy package-dir cache exists and the new
# per-user cache does not, copy it once so existing calibrations carry
# over. Best-effort — a copy failure must not break startup.
_LEGACY_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".qpu_calibration_cache")
try:
    if os.path.isdir(_LEGACY_CACHE_DIR) and not os.path.exists(_CACHE_DIR):
        import shutil
        shutil.copytree(_LEGACY_CACHE_DIR, _CACHE_DIR)
except Exception:
    _log.debug("Legacy cache migration failed (non-fatal)", exc_info=True)

_cache_dirs_created = set()

# Per-backend lock for the read-modify-write cycle on index.json.
# `_add_to_index` is called from `_save_calibration`, called from
# `_fetch_one` workers in `fetch_calibration_history`'s ThreadPoolExecutor
# — so two threads on the same backend can race the (load → append → save):
#   T1 reads (5) ; T2 reads (5) ; T1 writes (6) ; T2 writes (6) → T1's
#   addition is LOST.
# Locking only the read-modify-write block avoids serialising the
# per-snapshot JSON writes (those are atomic via tempfile + os.replace).
# `_index_locks_lock` is the meta-lock guarding the dict itself
# (allocating a new lock for a never-seen backend).
_index_locks: dict = {}
_index_locks_lock = threading.Lock()


def _get_index_lock(backend_name: str) -> threading.Lock:
    """Return the lock guarding `index.json` for `backend_name`."""
    # Double-checked: fast path is lock-free; meta-lock only fires on
    # the first sighting of a new backend name.
    if backend_name not in _index_locks:
        with _index_locks_lock:
            if backend_name not in _index_locks:
                _index_locks[backend_name] = threading.Lock()
    return _index_locks[backend_name]


def _safe_remove(path):
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def _backend_cache_dir(backend_name):
    # Strip path components from `backend_name` before joining — values
    # like "../../etc/passwd" or "..\\evil" (Windows) would let
    # os.path.join walk out of _CACHE_DIR. The legacy replace("/", "_")
    # only caught POSIX separators; os.path.basename handles backslash +
    # POSIX cross-platform, then `'', '.', '..'` are rejected so nothing
    # in the result can start a traversal.
    safe_name = os.path.basename(backend_name.replace("\\", "/"))
    if safe_name in ('', '.', '..'):
        raise ValueError(
            f"Invalid backend name for cache dir: {backend_name!r}")
    bdir = os.path.join(_CACHE_DIR, safe_name.replace("/", "_"))
    if bdir not in _cache_dirs_created:
        os.makedirs(bdir, exist_ok=True)
        # Restrict to owner-only on Unix so calibration snapshots aren't
        # readable by other users on shared hosts. Best-effort: chmod is
        # a no-op on Windows, and we don't want a permission error to
        # block cache writes.
        for _d in (_CACHE_DIR, bdir):
            try:
                os.chmod(_d, 0o700)
            except OSError:
                _log.debug("chmod 0o700 failed for %s", _d, exc_info=True)
        _cache_dirs_created.add(bdir)
    return bdir


def _sanitize_ts(ts_str):
    """'2026-01-20 14:18:17' → '2026-01-20T14-18-17' (filesystem-safe).

    Whitelist: any residual character that isn't ``[0-9A-Za-z_.\\-T]``
    collapses to ``-`` so a hostile timestamp cannot smuggle path
    separators or ``..`` into the filename component (symmetric with
    the guards in ``_backend_cache_dir``).
    """
    if ts_str is None:
        return 'unknown'
    s = ts_str.strip().replace(' ', 'T').replace(':', '-')
    return re.sub(r'[^0-9A-Za-z_.\-T]', '-', s)


def _cal_path(backend_name, sanitized_ts):
    return os.path.join(_backend_cache_dir(backend_name), f"cal_{sanitized_ts}.json")


def _index_path(backend_name):
    return os.path.join(_backend_cache_dir(backend_name), "index.json")


def _load_index(backend_name):
    path = _index_path(backend_name)
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return _json.load(f).get('calibrations', [])
    except Exception:
        _log.debug("Index load failed for %s", backend_name)
    return []


def _save_index(backend_name, entries):
    path = _index_path(backend_name)
    dirn = _backend_cache_dir(backend_name)
    try:
        fd, tmp = tempfile.mkstemp(dir=dirn, suffix='.tmp')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                _json.dump({'calibrations': entries}, f, indent=2)
            os.replace(tmp, path)
        except Exception:
            _safe_remove(tmp)
            raise
    except Exception as exc:
        _log.warning("Failed to save index for %s: %s", backend_name, exc)


def _add_to_index(backend_name, last_update_str):
    # Held across the whole load+append+save so two workers in
    # fetch_calibration_history can't lose entries to last-writer-wins.
    # See `_get_index_lock` above for the race scenario.
    with _get_index_lock(backend_name):
        entries = _load_index(backend_name)
        san = _sanitize_ts(last_update_str)
        if any(_sanitize_ts(e.get('last_update', '')) == san for e in entries):
            return
        entries.append({
            'last_update': last_update_str,
            'fetched_at': datetime.now(timezone.utc).isoformat(),
        })
        entries.sort(key=lambda e: e.get('last_update', ''))
        _save_index(backend_name, entries)


def _save_calibration(backend_name, last_update_str, qubit_props, gate_props):
    san = _sanitize_ts(last_update_str)
    path = _cal_path(backend_name, san)
    data = {
        'last_update': last_update_str,
        'qubit_props': {k: v.tolist() if hasattr(v, 'tolist') else list(v)
                        for k, v in qubit_props.items()},
        'gate_props': {k: v.tolist() if hasattr(v, 'tolist') else list(v)
                       for k, v in gate_props.items()},
    }
    dirn = _backend_cache_dir(backend_name)
    try:
        fd, tmp = tempfile.mkstemp(dir=dirn, suffix='.tmp')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                _json.dump(data, f)
            os.replace(tmp, path)
        except Exception:
            _safe_remove(tmp)
            raise
    except Exception as exc:
        _log.warning("Failed to save cache %s/%s: %s", backend_name, san, exc)
        return
    _add_to_index(backend_name, last_update_str)
    _log.info("Cached calibration %s: %s", backend_name, last_update_str)


def _load_calibration(backend_name, last_update_str):
    if last_update_str is None:
        return None
    san = _sanitize_ts(last_update_str)
    path = _cal_path(backend_name, san)
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = _json.load(f)
    except Exception as exc:
        _log.warning("Corrupted cache %s — removing: %s", path, exc)
        _safe_remove(path)
        return None
    if 'qubit_props' not in data or 'gate_props' not in data:
        _log.warning("Invalid cache %s — missing keys, removing", path)
        _safe_remove(path)
        return None
    expected_qp = {pk for _, pk in QUBIT_PROP_OPTIONS}
    expected_gp = {pk for _, pk in GATE_PROP_OPTIONS}
    if expected_qp - set(data['qubit_props'].keys()) or \
       expected_gp - set(data['gate_props'].keys()):
        _log.warning("Stale cache %s — removing", path)
        _safe_remove(path)
        return None
    try:
        qp = {k: np.array(v) for k, v in data['qubit_props'].items()}
        gp = {k: np.array(v) for k, v in data['gate_props'].items()}
        return qp, gp
    except Exception as exc:
        _log.warning("Cache array conversion failed %s — removing: %s", path, exc)
        _safe_remove(path)
        return None


# Legacy cache compat (read-only)
def _cache_path(backend_name, date_str):
    return os.path.join(_backend_cache_dir(backend_name), f"{date_str}.json")

def _meta_path(backend_name, date_str):
    return os.path.join(_backend_cache_dir(backend_name), f"{date_str}.meta.json")

def _load_legacy_cache(backend_name, date_str):
    path = _cache_path(backend_name, date_str)
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = _json.load(f)
        if 'qubit_props' in data and 'gate_props' in data:
            return ({k: np.array(v) for k, v in data['qubit_props'].items()},
                    {k: np.array(v) for k, v in data['gate_props'].items()})
    except Exception:
        _log.debug("Cache read failed: %s", path)
    return None


# ── Props resolution ────────────────────────────────────────────────

class _CachedPropsPlaceholder:
    def __init__(self, last_update_str):
        self._last_update_str = last_update_str
        self.last_update_date = None
        try:
            self.last_update_date = datetime.strptime(last_update_str, "%Y-%m-%d %H:%M:%S")
        except Exception:
            _log.debug("CachedPropsPlaceholder date parse failed: %s", last_update_str)
    def readout_error(self, *a, **kw):
        raise RuntimeError("CachedPropsPlaceholder: data in state, don't call methods")
    t1 = t2 = qubit_property = gate_error = gate_length = readout_error


def _extract_and_cache(state, props, coords, edges, backend_name):
    """Extract all properties from a real props object, save to cache."""
    state.qubit_props_data = {
        pk: np.array([fetch_qubit_prop(props, i, pk) for i in range(len(coords))])
        for _, pk in QUBIT_PROP_OPTIONS
    }
    state.gate_props_data = {
        pk: np.array([fetch_gate_prop(props, e, pk) for e in edges])
        for _, pk in GATE_PROP_OPTIONS
    }
    last_upd = _extract_last_update(props)
    if last_upd and backend_name:
        _save_calibration(backend_name, last_upd,
                          state.qubit_props_data, state.gate_props_data)


def _get_current_props(state, backend, prefetched_props=None):
    """Fetch current calibration: 1 API call, use cache if last_update matches.
    If prefetched_props is provided, skips the API call (avoids double fetch).
    """
    props = prefetched_props or _retry(backend.properties)
    if props is None:
        return None
    last_upd = _extract_last_update(props)
    if last_upd:
        cached = _load_calibration(state.active_backend_name, last_upd)
        if cached is not None:
            state.qubit_props_data, state.gate_props_data = cached
            state.initial_props = props  # keep for LUCJ scoring
            _log.info("Cache hit for %s: %s", state.active_backend_name, last_upd)
            return _CachedPropsPlaceholder(last_upd)
    state.initial_props = props
    return props


def get_props_for_datetime(state, backend, cal_datetime):
    """PUBLIC: Fetch calibration for a specific datetime, with cache.
    Cache-first: checks local index before calling the API.
    Returns (placeholder, qubit_props, gate_props) or raises.
    """
    backend_name = state.active_backend_name

    # ── Cache-first: check if we already have a calibration for this time ──
    try:
        day = cal_datetime.date() if hasattr(cal_datetime, 'date') else cal_datetime
        known = list_known_calibrations(backend_name, day)
        if known:
            # Find closest cached calibration to requested time
            req_hm = cal_datetime.hour * 60 + cal_datetime.minute
            best_match = None
            best_diff = float('inf')
            for entry in known:
                lu = entry.get('last_update', '')
                if len(lu) > 11:
                    try:
                        h, m = int(lu[11:13]), int(lu[14:16])
                        diff = abs((h * 60 + m) - req_hm)
                        if diff < best_diff:
                            best_diff = diff
                            best_match = lu
                    except (ValueError, IndexError):
                        pass
            if best_match:
                cached = _load_calibration(backend_name, best_match)
                if cached is not None:
                    _log.info("Cache-first hit for %s @ %s → %s (no API call)",
                              backend_name, cal_datetime, best_match)
                    return _CachedPropsPlaceholder(best_match), cached[0], cached[1]
    except Exception:
        pass  # fall through to API call

    # ── No cache match: call API ──
    if not _supports_historical_props(backend):
        raise RuntimeError(_HISTORICAL_DISABLED_MSG)
    props = _retry(backend.properties, datetime=cal_datetime)
    if props is None:
        raise ValueError(f"No calibration data for {cal_datetime}")
    last_upd = _extract_last_update(props)
    if last_upd:
        cached = _load_calibration(backend_name, last_upd)
        if cached is not None:
            _log.info("Cache hit (post-API) for %s @ %s: %s", backend_name, cal_datetime, last_upd)
            return _CachedPropsPlaceholder(last_upd), cached[0], cached[1]
    # Extract and cache
    n_q = len(state.qubit_ids) if state.qubit_ids else 0
    edges = state.edges_list or []
    qp = {pk: np.array([fetch_qubit_prop(props, i, pk) for i in range(n_q)])
          for _, pk in QUBIT_PROP_OPTIONS}
    gp = {pk: np.array([fetch_gate_prop(props, e, pk) for e in edges])
          for _, pk in GATE_PROP_OPTIONS}
    if last_upd:
        _save_calibration(backend_name, last_upd, qp, gp)
    return _CachedPropsPlaceholder(last_upd or 'unknown'), qp, gp


def list_known_calibrations(backend_name, date=None, by_local_day=True):
    """List all known calibration timestamps for a backend.

    Args:
        date: optional date OR datetime to filter by day. A `datetime`
            is normalised to `.date()` — the previous strict comparison
            silently returned `[]` for any caller that passed a datetime.
        by_local_day: when True (default), filter by the user's LOCAL
            day (each cached UTC ts is converted to local first). Matches
            "today's calibrations" = today on the wall clock, not UTC.
            Pass False for the legacy UTC-prefix filter.

    Returns: [{'last_update': '...', 'fetched_at': '...'}, ...]
    """
    entries = _load_index(backend_name)
    if date is None:
        return entries
    # Accept both `date` and `datetime` — hasattr is enough; isinstance
    # would force the import.
    if hasattr(date, 'date') and callable(date.date):
        date = date.date()
    if not by_local_day:
        date_str = date.strftime("%Y-%m-%d")
        return [e for e in entries if e.get('last_update', '').startswith(date_str)]

    filtered = []
    for e in entries:
        local_dt = utc_str_to_local_dt(e.get('last_update', ''))
        if local_dt is not None and local_dt.date() == date:
            filtered.append(e)
    return filtered


def invalidate_cache_for_date(backend_name, target_date):
    """Remove cached calibration data for a given date from BOTH caches
    — timestamp-indexed (cal_<ts>.json + index.json) and legacy date-keyed
    (<date>.json + <date>.meta.json). Used by the UI Refresh button to
    force a re-fetch.

    Args:
        backend_name: backend identifier (e.g. 'ibm_kingston').
        target_date: datetime.date (or any object with strftime).

    Returns:
        Count of timestamp-indexed entries removed (legacy files dropped
        silently and not counted).
    """
    if not backend_name:
        return 0
    date_prefix = target_date.strftime("%Y-%m-%d")
    removed = 0

    # New cache: drop index entries on target date + their backing files;
    # rewrite the index only if something actually changed.
    entries = _load_index(backend_name)
    keep = []
    for e in entries:
        lu = e.get('last_update', '')
        if lu.startswith(date_prefix):
            _safe_remove(_cal_path(backend_name, _sanitize_ts(lu)))
            removed += 1
        else:
            keep.append(e)
    if len(keep) != len(entries):
        _save_index(backend_name, keep)

    # Legacy cache: best-effort removal of the date-keyed pair.
    _safe_remove(_cache_path(backend_name, date_prefix))
    _safe_remove(_meta_path(backend_name, date_prefix))

    return removed


def scan_day_calibrations(state, backend, scan_date, progress_callback=None):
    """Scan 24 hours of a day to discover + cache every distinct
    calibration. Calls backend.properties(datetime=...) per hour of the
    user's LOCAL day, dedupes by last_update, extracts + caches each.

    Timezone: scan_date is the user's wall-clock day (DatePicker
    semantics). Probe times cover that local day; the API receives UTC
    datetimes computed from the local-tz boundary; the final filter
    keeps only timestamps whose LOCAL date == scan_date. Without this,
    a CET user picking "10 April" would miss late-night-CET cals that
    fall on UTC 9 April, and spuriously include pre-dawn UTC-10 cals
    that are local 11 April.

    Args:
        state: SessionState (backend_name + extraction context).
        backend: IBM BackendV2 instance.
        scan_date: datetime.date, interpreted as local wall-clock.
        progress_callback: optional fn(hour, total, status_msg).

    Returns:
        Sorted list of unique UTC timestamp strings whose local
        equivalent falls on scan_date.
    """
    backend_name = state.active_backend_name
    seen = {}  # last_update → props

    # Bail early on runtime versions that don't accept datetime=… —
    # otherwise we'd hit 24 retries × TypeError before giving up.
    if not _supports_historical_props(backend):
        if progress_callback:
            progress_callback(24, 24, _HISTORICAL_DISABLED_MSG)
        return []

    # UTC range covering scan_date in the user's local tz. local_tz from
    # datetime.now().astimezone() avoids the zoneinfo import.
    local_tz = datetime.now().astimezone().tzinfo
    local_start = datetime(scan_date.year, scan_date.month, scan_date.day,
                            tzinfo=local_tz)
    utc_start = local_start.astimezone(timezone.utc).replace(tzinfo=None)

    for hour_offset in range(24):
        cal_dt = utc_start + timedelta(hours=hour_offset)
        if progress_callback:
            progress_callback(hour_offset, 24,
                f"Scanning {scan_date} h={hour_offset:02d} (local)...")
        try:
            props = _retry(backend.properties, datetime=cal_dt)
            if props is None:
                continue
            last_upd = _extract_last_update(props)
            if last_upd and last_upd not in seen:
                seen[last_upd] = props
        except Exception as exc:
            _log.warning("scan_day h+%d failed: %s", hour_offset, exc)

    # Cache each unique calibration
    n_q = len(state.qubit_ids) if state.qubit_ids else 0
    edges = state.edges_list or []
    for last_upd, props in seen.items():
        cached = _load_calibration(backend_name, last_upd)
        if cached is None:
            try:
                qp = {pk: np.array([fetch_qubit_prop(props, i, pk) for i in range(n_q)])
                      for _, pk in QUBIT_PROP_OPTIONS}
                gp = {pk: np.array([fetch_gate_prop(props, e, pk) for e in edges])
                      for _, pk in GATE_PROP_OPTIONS}
                _save_calibration(backend_name, last_upd, qp, gp)
                _log.info("Scanned & cached: %s @ %s", backend_name, last_upd)
            except Exception as exc:
                _log.warning("scan_day extraction failed for %s: %s", last_upd, exc)

    if progress_callback:
        progress_callback(24, 24, f"Done — {len(seen)} unique calibrations found")

    # Keep only timestamps whose LOCAL date == scan_date. Previous
    # str.startswith on the UTC string was a silent tz bug (see docstring).
    filtered = []
    for ts in seen.keys():
        local_dt = utc_str_to_local_dt(ts)
        if local_dt is not None and local_dt.date() == scan_date:
            filtered.append(ts)
    filtered.sort()
    excluded = len(seen) - len(filtered)
    if excluded > 0:
        _log.info("scan_day %s: excluded %d timestamp(s) outside the local day",
                  scan_date.isoformat(), excluded)

    return filtered


# ── Device graph generation ─────────────────────────────────────────

def generate_device_data(state, vendor, chip=None, cal_date=None, prefetched_props=None):
    if cal_date is not None:
        # Accept both date and datetime
        if hasattr(cal_date, 'hour'):
            cal_datetime = cal_date  # already a datetime
        else:
            cal_datetime = datetime.combine(cal_date, datetime.min.time())
    else:
        cal_datetime = None
    return _generate_ibm(state, cal_datetime, prefetched_props=prefetched_props)


def _generate_ibm(state, cal_datetime, prefetched_props=None):
    from .constants import resolve_coords
    backend = state.active_backend
    state.calibration_data_label = None
    if backend is None:
        raise RuntimeError("No IBM backend connected.")

    try:
        raw_edges = list(backend.coupling_map.get_edges())
    except Exception as exc:
        raise RuntimeError(f"Cannot read coupling map: {exc}") from exc

    try:
        n_qubits = backend.num_qubits
    except Exception as exc:
        n_qubits = max(max(e) for e in raw_edges) + 1 if raw_edges else 0
        _log.warning("backend.num_qubits failed, inferred %d from edges: %s", n_qubits, exc)

    coords, chip_label = resolve_coords(backend)
    # Symmetric check. Was `len(coords) < n_qubits` only — when
    # len(coords) > n_qubits the code emitted q_ids for ALL coords,
    # giving "phantom" NaN-data qubits that were still clickable and
    # addressable by LUCJ scoring. Defensive against degraded-mode
    # backends or a stale constants.py entry.
    if len(coords) != n_qubits:
        raise RuntimeError(
            f"Coordinate count mismatch for '{state.active_backend_name}': "
            f"backend reports {n_qubits} qubits but resolve_coords() returned "
            f"{len(coords)} coordinates for chip '{chip_label}'. "
            f"Check the hardcoded layout in constants.py."
        )

    state.active_chip = chip_label
    state.active_coords = coords
    edges = [e for e in raw_edges if e[0] < len(coords) and e[1] < len(coords)]
    q_ids = [str(i) for i in range(len(coords))]
    bname = state.active_backend_name

    try:
        if cal_datetime is not None:
            ph, qp, gp = get_props_for_datetime(state, backend, cal_datetime)
            state.qubit_props_data, state.gate_props_data = qp, gp
            lu_display = _extract_last_update(ph, display=True) or _extract_last_update(ph)
            state.calibration_data_label = (
                f"{bname} — req: {cal_datetime.strftime('%Y-%m-%d %H:%M')}"
                + (f" · cal: {lu_display}" if lu_display else ""))

        elif state.initial_props is not None:
            props = state.initial_props
            lu = _extract_last_update(props)           # UTC for cache
            lu_display = _extract_last_update(props, display=True) or lu  # local for display
            cached = _load_calibration(bname, lu) if lu else None
            if cached:
                state.qubit_props_data, state.gate_props_data = cached
            else:
                _extract_and_cache(state, props, coords, edges, bname)
            # Intentionally do NOT set initial_props = None — LUCJ
            # scoring needs the raw BackendProperties for .gate_error()
            # and .readout_error(). Cache handles dedup.
            state.calibration_data_label = (
                f"{bname} — {lu_display}" if lu_display else f"{bname} — live")

        else:
            props = _get_current_props(state, backend, prefetched_props=prefetched_props)
            if props is None:
                raise ValueError("backend.properties() returned None")
            lu = _extract_last_update(props)
            lu_display = _extract_last_update(props, display=True) or lu
            if not isinstance(props, _CachedPropsPlaceholder):
                _extract_and_cache(state, props, coords, edges, bname)
            state.calibration_data_label = (
                f"{bname} — {lu_display}" if lu_display else f"{bname} — live")

        nw = _audit_nan_fraction(state.qubit_props_data, state.gate_props_data)
        if nw:
            msg = f"Partial calibration: {'; '.join(nw)}"
            state.props_warning = (state.props_warning + f" | {msg}") if state.props_warning else msg

    except Exception as exc:
        nq, ne = len(coords), len(edges)
        state.qubit_props_data = {pk: np.full(nq, np.nan) for _, pk in QUBIT_PROP_OPTIONS}
        state.gate_props_data = {pk: np.full(ne, np.nan) for _, pk in GATE_PROP_OPTIONS}
        state.calibration_data_label = f"{bname} — NO CALIBRATION DATA"
        state.props_warning = f"Cannot retrieve calibration for '{bname}': {exc}"
        _log.exception("Calibration failed for '%s'", bname)

    x = np.array([c[1] for c in coords]) - min(c[1] for c in coords)
    y = np.array([c[0] for c in coords]) - max(c[0] for c in coords)
    if 'Nighthawk' in chip_label and len(q_ids) > 100:
        y = y * 0.8
    return x, y, edges, q_ids, \
           state.qubit_props_data.get('readout_error', np.zeros(len(q_ids))), \
           state.gate_props_data.get('cz_error', np.zeros(len(edges)))


# ── Calibration history ─────────────────────────────────────────────

def fetch_calibration_history(state, start_date, end_date, progress_callback=None):
    backend = state.active_backend
    bname = state.active_backend_name
    if backend is None or bname is None or not state.qubit_ids:
        return 0

    n_q = len(state.qubit_ids)
    n_e = len(state.edges_list) if state.edges_list else 0
    dates = []
    cur = start_date
    while cur <= end_date:
        dates.append(cur)
        cur += timedelta(days=1)
    total = len(dates)

    state.cal_history = {}
    state.cal_history_dates = []
    uncached = []

    for idx, dt in enumerate(dates):
        ds = dt.strftime("%Y-%m-%d")
        found = False
        # Try new cache (by timestamp)
        day_cals = list_known_calibrations(bname, dt)
        if day_cals:
            cached = _load_calibration(bname, day_cals[-1]['last_update'])
            if cached:
                qp, gp = cached
                if _props_shape_matches(qp, gp, n_q, n_e):
                    state.cal_history[ds] = {'qubit_props': qp, 'gate_props': gp}
                    state.cal_history_dates.append(ds)
                    found = True
        # Try legacy cache
        if not found:
            legacy = _load_legacy_cache(bname, ds)
            if legacy:
                qp, gp = legacy
                if _props_shape_matches(qp, gp, n_q, n_e):
                    state.cal_history[ds] = {'qubit_props': qp, 'gate_props': gp}
                    state.cal_history_dates.append(ds)
                    found = True
        if found:
            if progress_callback:
                progress_callback(idx, total, f"{ds} (cached)")
        else:
            uncached.append(dt)

    fetched = total - len(uncached)
    # If the runtime can't fetch historical props, skip API round-trips
    # and return what's in the cache. Callers (ui.history / ui.time_panel)
    # surface "no data" via len(cal_history_dates).
    if uncached and not _supports_historical_props(backend):
        if progress_callback:
            progress_callback(total, total, _HISTORICAL_DISABLED_MSG)
        state.cal_history_dates.sort()
        return len(state.cal_history_dates)

    def _fetch_one(dt):
        ds = dt.strftime("%Y-%m-%d")
        try:
            props = _retry(backend.properties, datetime=datetime.combine(dt, datetime.min.time()))
            if props is None:
                return ds, None, None
            lu = _extract_last_update(props)
            qp = {pk: np.array([fetch_qubit_prop(props, i, pk) for i in range(n_q)])
                  for _, pk in QUBIT_PROP_OPTIONS}
            gp = {pk: np.array([fetch_gate_prop(props, e, pk) for e in state.edges_list])
                  for _, pk in GATE_PROP_OPTIONS}
            if lu:
                _save_calibration(bname, lu, qp, gp)
            return ds, qp, gp
        except Exception as exc:
            _log.warning("History fetch failed %s/%s: %s", bname, ds, exc)
            return ds, None, None

    _lock = threading.Lock()
    workers = min(4, len(uncached)) if uncached else 1
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_fetch_one, dt): dt for dt in uncached}
        for fut in as_completed(futs):
            fetched += 1
            ds, qp, gp = fut.result()
            if progress_callback:
                progress_callback(fetched - 1, total, ds)
            if qp is not None:
                with _lock:
                    state.cal_history[ds] = {'qubit_props': qp, 'gate_props': gp}
                    state.cal_history_dates.append(ds)

    state.cal_history_dates.sort()
    return len(state.cal_history_dates)


def auto_load_recent_history(state, days=14, min_cached_threshold=5):
    """Pre-populate state.cal_history at connect time.

    Three steps:
      1. Cache-only probe of the last `days` days — no IBM calls here.
      2. If ≥ `min_cached_threshold` snapshots, populate from cache and
         STOP (respects IBM rate limits; spec says "good enough" cache
         skips enrichment).
      3. Otherwise delegate to fetch_calibration_history (cache + IBM).

    Args:
        state: SessionState with active_backend / active_backend_name /
               qubit_ids / edges_list already populated (call AFTER
               _load_device_graph / generate_device_data).
        days: window size, default 14.
        min_cached_threshold: minimum cached snapshots for "skip IBM",
            default 5.

    Returns:
        (n_loaded, source) where source is:
          'noop'  — cal_history_dates was already non-empty; no change.
          'cache' — cache had ≥ threshold; only cached entries loaded.
          'fetch' — cache had < threshold; fetch_calibration_history ran.
    """
    # Spec: don't touch state if the user has already populated
    # cal_history via Multi-day mode before this auto-load fires.
    if state.cal_history_dates:
        return len(state.cal_history_dates), 'noop'

    bname = state.active_backend_name
    backend = state.active_backend
    if backend is None or bname is None or not state.qubit_ids:
        return 0, 'noop'

    n_q = len(state.qubit_ids)
    n_e = len(state.edges_list) if state.edges_list else 0

    # ── Step 1: cache-only probe ─────────────────────────────────────
    # Walk the last `days` days back from today (UTC, matching
    # fetch_calibration_history's scan). Per day: most-recent cached
    # entry (mirrors `day_cals[-1]` there).
    today_utc = datetime.now(timezone.utc).date()
    cached = {}
    for back in range(days):
        d = today_utc - timedelta(days=back)
        ds = d.strftime("%Y-%m-%d")
        day_cals = list_known_calibrations(bname, d)
        if day_cals:
            loaded = _load_calibration(bname, day_cals[-1]['last_update'])
            if loaded:
                qp, gp = loaded
                if _props_shape_matches(qp, gp, n_q, n_e):
                    cached[ds] = {'qubit_props': qp, 'gate_props': gp}
                    continue
        # Legacy by-date cache for older snapshots predating the
        # timestamp-indexed layout. Same shape gate.
        legacy = _load_legacy_cache(bname, ds)
        if legacy:
            qp, gp = legacy
            if _props_shape_matches(qp, gp, n_q, n_e):
                cached[ds] = {'qubit_props': qp, 'gate_props': gp}

    if len(cached) >= min_cached_threshold:
        # Step 2: cache is enough — skip the IBM round-trip.
        state.cal_history = cached
        state.cal_history_dates = sorted(cached.keys())
        return len(state.cal_history_dates), 'cache'

    # ── Step 3: fall through to the regular fetch (cache + IBM) ─────
    start_d = today_utc - timedelta(days=days - 1)
    end_d = today_utc
    n_loaded = fetch_calibration_history(state, start_d, end_d)
    return n_loaded, 'fetch'


def apply_history_snapshot(state, date_str):
    if date_str not in state.cal_history:
        return False
    snap = state.cal_history[date_str]
    qp, gp = snap['qubit_props'], snap['gate_props']
    n_q = len(state.qubit_ids) if state.qubit_ids else 0
    n_e = len(state.edges_list) if state.edges_list else 0
    if len(next(iter(qp.values()), [])) != n_q or len(next(iter(gp.values()), [])) != n_e:
        return False
    state.qubit_props_data = qp
    state.gate_props_data = gp
    state.calibration_data_label = f"{state.active_backend_name} — {date_str} (history)"
    # Reset delta rendering — previous delta arrays referred to the old
    # (qp, gp), so leaving delta_mode True would render a stale delta
    # over the new snapshot. Callers that want delta against this
    # snapshot must re-apply explicitly (see ui.history.py
    # _on_slider_change for the reference pattern).
    state.delta_mode = False
    return True