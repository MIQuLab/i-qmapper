"""
stability_filter.py — Stability-aware ancilla sweep (pure, synchronous).

Given cached calibration history, score every qubit's volatility, exclude the
volatile ones from the sweep's search space (a hard filter), and run
``ancilla_sweep.sweep_ancilla_layouts`` on the pruned chip — with an automatic
RELAX VALVE that re-admits the least-volatile excluded qubits when the filter
is too strict for ``norb`` to fit.

All functions are pure and offline: history in, dict out. The volatility math
is NOT reimplemented here — it reuses ``stability.analyze_property_stability``
(the same per-series detector the Stability Check panel uses), so the two views
can never diverge. Only the plumbing (assembling a per-qubit series across the
windowed history) lives here.

Cap-pressure note: pruning shrinks the host graph, which usually DECREASES the
number of VF2 embeddings per template — but on a tight fit it can push VF2
toward truncation. ``vf2_truncated`` propagates untouched from the sweep.
"""

import logging

import numpy as np

from .ancilla_sweep import sweep_ancilla_layouts
from .stability import analyze_property_stability

_log = logging.getLogger(__name__)

# Quality floor mirrors the C1 box's device-outlier rule — the `outliers`
# block in ``export/scoring.py`` (readout > mean + sigma*std over the non-NaN
# device readouts, guarded by ``len > OUTLIER_MIN_SAMPLES``). The source
# constants OUTLIER_SIGMA=3.0 / OUTLIER_MIN_SAMPLES=10 live in
# ``export/__init__.py``; we mirror the min-samples guard here to keep this
# lean engine free of the heavy ``export`` package import (the sigma is a
# caller parameter, defaulting to the same 3.0).
_OUTLIER_MIN_SAMPLES = 10

# Volatility = analyze_property_stability's 'error' score = max/median - 1.
# The default threshold flags a qubit whose worst windowed day exceeded ~1.5×
# its own median readout error (score > 0.5) — a clear, repeated instability,
# not day-to-day noise.
_DEFAULT_THRESHOLD = 0.5

# Fraction of the initial excluded set re-admitted per relax step.
_RELAX_BATCH_FRAC = 0.20

_READOUT_KEY = 'readout_error'


def _windowed_dates(history, window_days, today):
    dates = sorted(history.keys())
    if today is not None:
        dates = [d for d in dates if d <= today]
    if window_days and window_days > 0:
        dates = dates[-window_days:]
    return dates


def compute_qubit_volatility(history, window_days, *, today=None):
    """Chip-wide per-qubit volatility from cached history (pure).

    Args:
        history: ``{date_str: {'qubit_props': {'readout_error': array}, ...}}``
            — the shape of ``state.cal_history``.
        window_days: use the most recent this-many dates.
        today: optional upper bound date string; dates after it are ignored.

    Returns ``{qubit_id: volatility_score}`` — higher = more volatile. Reuses
    ``stability.analyze_property_stability(series, 'error')['score']`` per
    qubit (readout_error series). Qubits with fewer than the detector's
    minimum valid samples score 0.0 (not enough data to judge). Empty history
    → ``{}``.
    """
    if not history:
        return {}
    dates = _windowed_dates(history, window_days, today)
    if not dates:
        return {}

    n_qubits = 0
    for d in dates:
        arr = history[d].get('qubit_props', {}).get(_READOUT_KEY)
        if arr is not None:
            n_qubits = max(n_qubits, len(arr))

    volatility = {}
    for q in range(n_qubits):
        series = []
        for d in dates:
            arr = history[d].get('qubit_props', {}).get(_READOUT_KEY)
            if arr is not None and q < len(arr):
                series.append(arr[q])
            else:
                series.append(float('nan'))
        analysis = analyze_property_stability(series, kind='error')
        volatility[q] = analysis['score']
    return volatility


def select_excluded(volatility, *, threshold):
    """Qubit ids with volatility strictly over ``threshold``, sorted MOST
    volatile first (so the relax valve re-admits from the least-volatile
    tail)."""
    over = [(q, v) for q, v in volatility.items() if v > threshold]
    over.sort(key=lambda qv: qv[1], reverse=True)
    return [q for q, _ in over]


def select_quality_floor(current_props_arrays, *, sigma=3.0):
    """Qubit ids whose CURRENT readout_error is a device OUTLIER — greater than
    ``mean + sigma*std`` over the non-NaN device readouts.

    This promotes the C1 Estimated-Layout-Quality box's ">3σ from device mean"
    warning (the ``outliers`` block in ``export/scoring.py``) from a warning to
    a FILTER component. It catches STABLY-terrible qubits — a qubit parked at
    ~73× the device median but flat over time — which the volatility score
    (``max/median − 1`` ≈ 0 when flat) is blind to by construction.

    Args:
        current_props_arrays: either a ``{'readout_error': seq}`` dict
            (``qubit_props`` shape) or a bare per-qubit readout sequence,
            indexed by physical qubit id. "Current" — not historical: a
            stably-terrible qubit is terrible NOW.
        sigma: outlier cutoff in standard deviations (default 3.0, matching
            ``export.OUTLIER_SIGMA``).

    Returns sorted ``list[int]`` of offending qubit ids. Empty when there are
    too few samples to judge (``<= OUTLIER_MIN_SAMPLES``, the same guard C1
    uses) or no valid readouts.
    """
    arr = (current_props_arrays.get(_READOUT_KEY)
           if isinstance(current_props_arrays, dict) else current_props_arrays)
    if arr is None:
        return []
    arr = np.asarray(arr, dtype=float)
    if arr.size <= _OUTLIER_MIN_SAMPLES:
        return []
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return []
    threshold = float(np.mean(valid) + sigma * float(np.std(valid)))
    return sorted(int(q) for q in range(arr.size)
                  if not np.isnan(arr[q]) and arr[q] > threshold)


def _n_qubits_from_history(history):
    """Qubit count from the first history snapshot's readout array."""
    for snap in history.values():
        arr = snap.get('qubit_props', {}).get(_READOUT_KEY)
        if arr is not None:
            return len(arr)
    return 0


def _current_readout_array(props, n_qubits):
    """The CURRENT props' readout_error for every qubit (NaN-safe), so the
    quality floor's device statistics are computed on the epoch being scored,
    not on history."""
    out = []
    for q in range(n_qubits):
        try:
            v = props.readout_error(q)
        except Exception:
            v = None
        out.append(float('nan') if v is None else float(v))
    return out


def sweep_with_stability(norb, backend, props, history, *,
                         window_days=14, threshold=_DEFAULT_THRESHOLD,
                         min_bridges=None, top_n=3, mapping_cap=None,
                         today=None, quality_sigma=3.0):
    """Stability-filtered ancilla sweep with an automatic relax valve.

    1. Build the excluded set = VOLATILITY-excluded (windowed history) ∪
       QUALITY-FLOOR-excluded (current device outliers, F2). The floor is
       computed on the CURRENT props, not history — a stably-terrible qubit
       is terrible NOW, and volatility scoring is blind to it.
    2. Run the sweep excluding those qubits.
    3. If the pruned graph cannot host ``norb`` (the sweep raises — zero
       embeddings, no bridges fit, or template creation fails), re-admit the
       LEAST-bad excluded qubits in batches and retry. RE-ADMISSION ORDER:
       volatility qubits first (least-volatile of them first), quality-floor
       qubits LAST — a stably-terrible outlier is the last thing we want back,
       so floor qubits sit at the HEAD of the list and the tail-first valve
       reaches them only after every volatility qubit has been re-admitted (a
       single batch is capped at the volatility/floor boundary so it can never
       cross early).
    4. Return the sweep dict EXTENDED with a ``stability`` block. When
       ``quality_sigma`` is not None the block also carries
       ``n_floor_excluded`` and ``floor_qubits`` (sorted).

    ``quality_sigma=None`` disables the floor entirely — byte-identical to the
    pre-floor behaviour (no floor keys added). No / empty history AND no floor
    hit → the plain sweep with ``stability={'available': False}``.
    """
    def _plain():
        res = sweep_ancilla_layouts(
            norb, backend, props, min_bridges=min_bridges,
            top_n=top_n, mapping_cap=mapping_cap)
        res['stability'] = {'available': False}
        return res

    if not history:
        return _plain()

    # Anchor the window to the SELECTED date when one is active — dates after
    # `today` are ignored, so the window is [today-window_days, today].
    # today=None keeps the now-anchored latest-window behaviour (live mode).
    volatility = compute_qubit_volatility(history, window_days, today=today)

    # Quality floor on the CURRENT props (independent of history).
    floor_qubits = []
    if quality_sigma is not None:
        n_q = _n_qubits_from_history(history)
        if n_q:
            floor_qubits = select_quality_floor(
                _current_readout_array(props, n_q), sigma=quality_sigma)

    if not volatility and not floor_qubits:
        return _plain()

    # Combined exclusion: floor at the HEAD, volatility (most→least) at the
    # TAIL, so the tail-first relax valve re-admits volatility before floor.
    vol_excluded = select_excluded(volatility, threshold=threshold)
    floor_set = set(floor_qubits)
    vol_only = [q for q in vol_excluded if q not in floor_set]
    floor_head = sorted(floor_set)
    combined = floor_head + vol_only
    n_floor = len(floor_head)
    n_excluded_initial = len(combined)
    batch = max(1, int(n_excluded_initial * _RELAX_BATCH_FRAC))

    current = list(combined)
    n_readmitted = 0
    relaxed = False
    while True:
        try:
            res = sweep_ancilla_layouts(
                norb, backend, props, min_bridges=min_bridges,
                top_n=top_n, mapping_cap=mapping_cap,
                exclude_qubits=set(current) or None)
            break
        except Exception as exc:
            if not current:
                # Even the unfiltered chip can't host norb — a real failure,
                # not a filter-too-strict one. Propagate.
                raise
            relaxed = True
            # Peel the tail (least-volatile first). Cap at the vol/floor
            # boundary so a batch never re-admits a floor qubit while any
            # volatility qubit is still excluded.
            n_vol_left = max(0, len(current) - n_floor)
            peel = min(batch, n_vol_left) if n_vol_left > 0 else min(batch, len(current))
            readmit = current[-peel:]
            current = current[:-peel]
            n_readmitted += len(readmit)
            _log.info("Stability sweep: norb=%d didn't fit with %d excluded "
                      "(%s); re-admitted %d least-bad, retrying",
                      norb, len(current) + len(readmit), exc, len(readmit))

    res['stability'] = {
        'available': True,
        'window_days': window_days,
        'threshold': threshold,
        'n_excluded_initial': n_excluded_initial,
        'n_readmitted': n_readmitted,
        'excluded_final': sorted(current),
        'relaxed': relaxed,
    }
    if quality_sigma is not None:
        floor_final = sorted(q for q in current if q in floor_set)
        res['stability']['n_floor_excluded'] = len(floor_final)
        res['stability']['floor_qubits'] = floor_final
    return res
