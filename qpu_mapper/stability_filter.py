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

from .ancilla_sweep import sweep_ancilla_layouts
from .stability import analyze_property_stability

_log = logging.getLogger(__name__)

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


def sweep_with_stability(norb, backend, props, history, *,
                         window_days=14, threshold=_DEFAULT_THRESHOLD,
                         min_bridges=None, top_n=3, mapping_cap=None,
                         today=None):
    """Stability-filtered ancilla sweep with an automatic relax valve.

    1. Score volatility over the windowed history and build the excluded set.
    2. Run the sweep excluding those qubits.
    3. If the pruned graph cannot host ``norb`` (the sweep raises — zero
       embeddings, no bridges fit, or template creation fails), re-admit the
       LEAST-volatile excluded qubits in batches (``_RELAX_BATCH_FRAC`` of the
       initial set per step, at least 1) and retry, until it fits or nothing
       is left excluded.
    4. Return the sweep dict EXTENDED with a ``stability`` block.

    No / empty history → run the plain sweep and set
    ``stability={'available': False}`` (never raises for lack of history).
    """
    def _plain():
        res = sweep_ancilla_layouts(
            norb, backend, props, min_bridges=min_bridges,
            top_n=top_n, mapping_cap=mapping_cap)
        res['stability'] = {'available': False}
        return res

    if not history:
        return _plain()

    # (F2) Anchor the window to the SELECTED date when one is active — dates
    # after `today` are ignored, so the window is [today-window_days, today].
    # today=None keeps the now-anchored latest-window behaviour (live mode).
    volatility = compute_qubit_volatility(history, window_days, today=today)
    if not volatility:
        return _plain()

    excluded = select_excluded(volatility, threshold=threshold)  # most→least
    n_excluded_initial = len(excluded)
    batch = max(1, int(n_excluded_initial * _RELAX_BATCH_FRAC))

    current = list(excluded)
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
            # Re-admit the least-volatile tail (current is most→least).
            readmit = current[-batch:]
            current = current[:-batch]
            n_readmitted += len(readmit)
            _log.info("Stability sweep: norb=%d didn't fit with %d excluded "
                      "(%s); re-admitted %d least-volatile, retrying",
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
    return res
