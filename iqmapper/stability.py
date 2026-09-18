"""
stability.py — Pure (UI-free) helpers for calibration stability analysis.

Two entry points:
  analyze_property_stability(values, kind) — per-series detector. Order-
    independent except for `worst_day_idx`, which references the position
    in the input array.
  get_layout_stability_report(state, circuit_idx, mode, window_days=14)
    — composes the per-series detector over every qubit and gate in a
    layout, using the most recent `window_days` snapshots from
    `state.cal_history`. Returns the dict that drives the Stability
    Check UI panel.

Kept pure so a future "Robust LQS" / CSV exporter can reuse without
dragging UI deps along.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


# "Bad" sample thresholds vs. series median:
#   errors > 2× median → bad
#   times  < 0.5× median → bad
# A series is unstable when ≥ `_min_bad_days(total)` samples are bad.
_ERROR_BAD_FACTOR = 2.0
_TIME_BAD_FACTOR  = 0.5
_MIN_SAMPLES      = 5     # need at least this many non-NaN points to judge


def _min_bad_days(total: int) -> int:
    """Bad-days threshold for "unstable": 2 of ≤14, 3 of >14 —
    avoids flagging a single day-of-the-week dip on 7-day windows while
    still catching short bursts on 14-day ones.
    """
    return 2 if total <= 14 else 3


def analyze_property_stability(values, kind: str) -> dict:
    """Analyze a property time series.

    Args:
        values: numpy array (or array-like) of property values over time.
                Length is the number of snapshots considered; NaN entries
                are skipped (missing calibration days).
        kind:   ``'error'`` (lower is better, e.g. readout_error, cz_error)
                or ``'time'`` (higher is better, e.g. T1, T2).

    Returns dict with:
        unstable        bool — True iff enough days are "bad" (see _min_bad_days)
        score           float ≥ 0 — magnitude for ranking (higher = worse)
        bad_days        int — number of samples flagged bad
        total_days      int — number of non-NaN samples
        median, max, min float — series statistics on non-NaN values
        worst_day_idx   int — index INTO THE ORIGINAL `values` array of the
                              single worst day (max for 'error', min for 'time').
                              -1 if no valid data.

    If the series has fewer than 5 valid (non-NaN) values or the median
    is zero (degenerate — can't form ratios), returns a minimal dict
    ``{'unstable': False, 'score': 0.0, 'bad_days': 0,
       'total_days': <count>, ...}`` with `median`/`max`/`min` set from
    whatever data is available (or `nan` if none) and `worst_day_idx = -1`.
    Callers should treat such series as "not enough data to judge".
    """
    if kind not in ('error', 'time'):
        raise ValueError(
            f"kind must be 'error' or 'time' (got {kind!r})")

    arr = np.asarray(values, dtype=float)
    mask = ~np.isnan(arr)
    valid = arr[mask]
    total = int(valid.size)

    # Degenerate paths: too few samples or zero median (can't form ratios).
    if total < _MIN_SAMPLES:
        return {
            'unstable': False,
            'score': 0.0,
            'bad_days': 0,
            'total_days': total,
            'median': float(np.median(valid)) if total else float('nan'),
            'max':    float(np.max(valid))    if total else float('nan'),
            'min':    float(np.min(valid))    if total else float('nan'),
            'worst_day_idx': -1,
        }

    median = float(np.median(valid))
    if median == 0.0:
        return {
            'unstable': False,
            'score': 0.0,
            'bad_days': 0,
            'total_days': total,
            'median': median,
            'max': float(np.max(valid)),
            'min': float(np.min(valid)),
            'worst_day_idx': -1,
        }

    max_v = float(np.max(valid))
    min_v = float(np.min(valid))

    if kind == 'error':
        # Lower is better → bad day = unusually HIGH value.
        threshold = _ERROR_BAD_FACTOR * median
        bad_mask = mask & (arr > threshold)
        score = max_v / median - 1.0
        # Use the ORIGINAL array (NaN masked) so the returned index is
        # the actual bad-day position, not a re-indexed one.
        safe_arr = np.where(mask, arr, -np.inf)
        worst_idx = int(np.argmax(safe_arr))
    else:  # 'time'
        # Higher is better → bad day = unusually LOW value.
        threshold = _TIME_BAD_FACTOR * median
        bad_mask = mask & (arr < threshold)
        score = 1.0 - (min_v / median)
        safe_arr = np.where(mask, arr, np.inf)
        worst_idx = int(np.argmin(safe_arr))

    bad_days = int(np.sum(bad_mask))
    unstable = bad_days >= _min_bad_days(total)

    return {
        'unstable': unstable,
        'score': float(max(score, 0.0)),  # clip tiny negatives from FP noise
        'bad_days': bad_days,
        'total_days': total,
        'median': median,
        'max': max_v,
        'min': min_v,
        'worst_day_idx': worst_idx,
    }


# ── Layout report ────────────────────────────────────────────────────────

# Hard-coded for v1: qubit→readout_error, gate→cz_error. Module
# constants so a future per-property matrix (Robust LQS) won't need to
# tear through call sites.
_QUBIT_PROP_KEY = 'readout_error'
_GATE_PROP_KEY  = 'cz_error'

_LUCJ_ROLE_KEYS = ('alpha', 'beta', 'other', 'ancilla')


def _collect_layout_qubits(layout: dict, mode: str):
    """Yield (qubit_id, role) for every qubit in the layout. LUCJ walks
    the four group lists in stable order; general walks the flat
    'qubits' list. Duplicates aren't deduped — the Stability UI shows
    a per-role entry, which is more informative.
    """
    if mode == 'lucj':
        for role in _LUCJ_ROLE_KEYS:
            for qid in layout.get(role, []) or []:
                yield qid, role
    else:
        for qid in layout.get('qubits', []) or []:
            yield qid, 'qubit'


def _qubit_id_index_map(state) -> dict:
    """Map qubit_id → property-array index. `state.qubit_ids` is the
    canonical order used by every property array. Materialised once per
    report so the per-qubit/per-gate loop is O(1), not O(n) list.index.
    """
    return {q: i for i, q in enumerate(state.qubit_ids or [])}


def _windowed_dates(state, window_days: int) -> list:
    """Return the most recent `window_days` dates from cal_history_dates.
    Already sorted chronologically (see fetch_calibration_history /
    ui.time_panel._on_intra_scan), so the trailing slice gives the
    recent window. Falls short of the window when fewer dates exist;
    caller surfaces actual count via `snapshots_used`.
    """
    dates = list(state.cal_history_dates or [])
    if window_days <= 0:
        return []
    return dates[-window_days:]


def _series_for_qubit(state, dates, qubit_idx: int, prop_key: str) -> np.ndarray:
    """1-D series of `prop_key` for `qubit_idx` across `dates`. Missing
    snapshots become NaN — analyze_property_stability handles that as
    "no data for that day".
    """
    out = np.full(len(dates), np.nan, dtype=float)
    for i, ds in enumerate(dates):
        snap = (state.cal_history or {}).get(ds, {})
        arr = snap.get('qubit_props', {}).get(prop_key)
        if arr is None:
            continue
        if 0 <= qubit_idx < len(arr):
            out[i] = arr[qubit_idx]
    return out


def _series_for_edge(state, dates, edge_idx: int, prop_key: str) -> np.ndarray:
    """Materialise a 1-D series of `prop_key` for `edge_idx` across `dates`."""
    out = np.full(len(dates), np.nan, dtype=float)
    for i, ds in enumerate(dates):
        snap = (state.cal_history or {}).get(ds, {})
        arr = snap.get('gate_props', {}).get(prop_key)
        if arr is None:
            continue
        if 0 <= edge_idx < len(arr):
            out[i] = arr[edge_idx]
    return out


def get_layout_stability_report(
    state, circuit_idx: int, mode: str, window_days: int = 14,
) -> Optional[dict]:
    """Build a stability report for all qubits and gates in a layout
    (see module docstring for the contract). Returns None when there's
    no calibration history (UI renders the "switch to Multi-day" hint)
    or the layout is empty.
    """
    if not state.cal_history_dates or not state.cal_history:
        return None

    layouts = getattr(state, 'layouts_data', None) or {}
    layout = layouts.get(circuit_idx)
    if not layout:
        return None

    qubit_iter = list(_collect_layout_qubits(layout, mode))
    if not qubit_iter:
        return None

    dates = _windowed_dates(state, window_days)
    if not dates:
        return None

    qid_to_idx = _qubit_id_index_map(state)
    layout_qids = {qid for qid, _role in qubit_iter}

    # ── Qubit analysis ──────────────────────────────────────────────
    qubit_entries = []
    for qid, role in qubit_iter:
        idx = qid_to_idx.get(qid, -1)
        if idx < 0:
            # Qubit not in the active backend's list — likely a stale
            # layout from a different backend. Skip silently.
            continue
        series = _series_for_qubit(state, dates, idx, _QUBIT_PROP_KEY)
        analysis = analyze_property_stability(series, kind='error')
        qubit_entries.append({
            'qid': qid,
            'role': role,
            'analysis': analysis,
            'series_dates': list(dates),
            'series_values': series,
        })

    # ── Gate analysis (edges with BOTH endpoints in the layout) ─────
    gate_entries = []
    for edge_idx, edge in enumerate(state.edges_list or []):
        if len(edge) < 2:
            continue
        u, v = edge[0], edge[1]
        # `edges_list` holds pairs of qubit INDICES (see plot.py); the
        # isinstance fallback defensively handles a non-int entry by
        # treating it as an already-resolved qubit_id.
        u_id = state.qubit_ids[u] if isinstance(u, int) and u < len(state.qubit_ids) else u
        v_id = state.qubit_ids[v] if isinstance(v, int) and v < len(state.qubit_ids) else v
        if u_id not in layout_qids or v_id not in layout_qids:
            continue
        series = _series_for_edge(state, dates, edge_idx, _GATE_PROP_KEY)
        analysis = analyze_property_stability(series, kind='error')
        gate_entries.append({
            'edge': (u_id, v_id),
            'edge_idx': edge_idx,
            'analysis': analysis,
            'series_dates': list(dates),
            'series_values': series,
        })

    # Unstable first (desc by score), then stable (also by score so the
    # "worst stable" surfaces just below the cutoff).
    def _sort_key(entry):
        a = entry['analysis']
        return (0 if a.get('unstable') else 1, -a.get('score', 0.0))
    qubit_entries.sort(key=_sort_key)
    gate_entries.sort(key=_sort_key)

    return {
        'window_days':           window_days,
        'snapshots_used':        len(dates),
        'qubits':                qubit_entries,
        'gates':                 gate_entries,
        'unstable_qubit_count':  sum(1 for e in qubit_entries if e['analysis'].get('unstable')),
        'unstable_gate_count':   sum(1 for e in gate_entries if e['analysis'].get('unstable')),
    }
