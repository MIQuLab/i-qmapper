"""
props_shim.py — Cache-aware props adapter for the multi-QPU scan.

The ancilla sweep (``ancilla_sweep.sweep_ancilla_layouts``) scores layouts
through a ``BackendProperties``-like object, calling EXACTLY two methods:

* ``readout_error(qubit)`` -> float
* ``gate_error(gate_name, (u, v))`` -> float   (the caller tries both edge
  orders, so this need not be direction-aware — but we make it so anyway)

The app's disk cache (``data.py``) does not store ``BackendProperties``
objects; it stores extracted numeric dicts:

* ``qubit_props`` — ``{'readout_error': array[q], 't1': ..., ...}``
  (per-qubit arrays, indexed by physical qubit id)
* ``gate_props``  — ``{'cz_error': array[i], ...}`` (per-EDGE arrays,
  positionally parallel to an edge list)

:class:`CachedPropsShim` adapts those dicts to the two-method surface the
sweep needs. The edge list required to map ``(u, v) -> array index`` is
derived deterministically from the backend coupling map the same way
``data._generate_ibm`` builds ``state.edges_list``
(``[e for e in coupling_map.get_edges() if e[0] < n and e[1] < n]``), so a
snapshot cached by the normal connect flow reconstructs correctly here.

:func:`make_scan_get_props` is the ``get_props`` factory the scan injects:
cache-first per backend name, live-fetch-and-cache on a miss. It is NOT the
state-bound ``data.get_props_for_datetime`` (which is pinned to the *current*
backend's name/qubits/edges and would mis-shape a remote backend) — it uses
the per-backend cache primitives instead, which are keyed correctly.
"""

import logging

_log = logging.getLogger(__name__)

# 2Q gate error keys the disk cache actually carries (see
# constants.GATE_PROP_OPTIONS). IBM Heron/Nighthawk use 'cz'.
_DEFAULT_GATE_KEY = 'cz_error'


class CachedPropsShim:
    """Adapt cached ``(qubit_props, gate_props)`` dicts + an edge list to the
    ``readout_error`` / ``gate_error`` surface the sweep consumes.

    Args:
        qubit_props: ``{'readout_error': sequence indexed by qubit id, ...}``.
        gate_props: ``{'<gate>_error': sequence parallel to `edges`, ...}``.
        edges: list of ``(u, v)`` pairs, positionally parallel to the
            ``gate_props`` arrays.
        gate_key: which ``gate_props`` key holds the 2Q error (default
            ``'cz_error'``).

    Missing lookups raise ``KeyError`` with a clear message; the scan's
    per-backend isolation turns that into a per-backend error string.
    """

    def __init__(self, qubit_props, gate_props, edges, gate_key=_DEFAULT_GATE_KEY):
        if 'readout_error' not in qubit_props:
            raise KeyError("qubit_props has no 'readout_error' entry")
        if gate_key not in gate_props:
            raise KeyError(f"gate_props has no {gate_key!r} entry")
        self._readout = qubit_props['readout_error']
        gate_arr = gate_props[gate_key]
        # frozenset key → direction-insensitive lookup (mirrors the sweep's
        # own edge_err map). Last write wins on duplicate undirected edges.
        self._gate = {}
        for i, (u, v) in enumerate(edges):
            if i < len(gate_arr):
                self._gate[frozenset((int(u), int(v)))] = gate_arr[i]

    def readout_error(self, qubit):
        try:
            return float(self._readout[qubit])
        except (IndexError, KeyError, TypeError) as exc:
            raise KeyError(
                f"no readout_error for qubit {qubit!r}") from exc

    def gate_error(self, gate_name, pair):
        u, v = pair
        key = frozenset((int(u), int(v)))
        try:
            return float(self._gate[key])
        except KeyError as exc:
            raise KeyError(
                f"no gate_error for edge {(u, v)!r}") from exc

    def score_layout(self, layout_qubit_ids):
        """Circuit-level fidelity for a layout on THIS backend, via the shared
        ``export.scoring.estimate_fidelity_core`` — so a scan candidate lands
        on the SAME scale as the local Estimated Layout-Quality box. Returns
        the core dict (``None`` only if the layout is empty)."""
        from .export.scoring import estimate_fidelity_core
        ids = list(layout_qubit_ids)
        if not ids:
            return None
        readout_by_id = {}
        for q in ids:
            try:
                readout_by_id[q] = self._readout[q]
            except (IndexError, KeyError, TypeError):
                pass
        gate_edges = [(u, v, err)
                      for key, err in self._gate.items()
                      for (u, v) in (tuple(key),)]
        return estimate_fidelity_core(ids, readout_by_id, gate_edges,
                                      all_readout=self._readout)


def _backend_edges(backend):
    """Deterministic edge list for a backend, matching data._generate_ibm."""
    raw = list(backend.coupling_map.get_edges())
    n = int(getattr(backend, 'num_qubits', 0) or 0)
    if n <= 0 and raw:
        n = max(max(e) for e in raw) + 1
    return [(u, v) for (u, v) in raw if u < n and v < n]


def make_scan_get_props(state, *, now=None):
    """Build the cache-aware ``get_props(backend)`` callable the scan injects.

    Policy (per backend):
      1. Cache-first: look for today's cached snapshot under the backend's own
         name (``data.list_known_calibrations`` + ``data._load_calibration``)
         and adapt it.
      2. Miss: fetch ``backend.properties()`` live, build the shim directly
         from it (correct regardless of array ordering), and best-effort warm
         the disk cache via the same per-backend extract+save primitives the
         app uses, so the next scan is instant.
      3. Hard failure: raise — the scan isolates it into a per-backend error.

    Args:
        state: SessionState (used only for read-through context; never mutated).
        now: optional datetime for the "today" cache window (injectable for
            tests). Defaults to ``datetime.now()``.
    """
    from datetime import datetime as _dt
    from . import data
    from .auto_lucj import _get_two_q_gate_name

    def _gate_key_for(backend):
        name = _get_two_q_gate_name(backend)
        return f"{name}_error" if name else _DEFAULT_GATE_KEY

    def _from_cache(backend_name, edges, gate_key):
        try:
            day = now or _dt.now()
            known = data.list_known_calibrations(backend_name, day)
        except Exception:
            return None
        if not known:
            return None
        # Newest snapshot for the day.
        ts = sorted(k.get('last_update', '') for k in known)[-1]
        cached = data._load_calibration(backend_name, ts)
        if not cached:
            return None
        qp, gp = cached[0], cached[1]
        key = gate_key if gate_key in gp else _DEFAULT_GATE_KEY
        return CachedPropsShim(qp, gp, edges, gate_key=key)

    def get_props(backend):
        backend_name = getattr(backend, 'name', None)
        edges = _backend_edges(backend)
        gate_key = _gate_key_for(backend)

        shim = _from_cache(backend_name, edges, gate_key)
        if shim is not None:
            _log.debug("Scan props: cache hit for %s", backend_name)
            return shim

        # Live fetch. Build the shim directly from the BackendProperties so
        # the edge mapping is self-consistent regardless of cache ordering.
        props = backend.properties()
        if props is None:
            raise RuntimeError(
                f"backend.properties() returned None for {backend_name}")
        gate_name = _get_two_q_gate_name(backend) or 'cz'
        n = int(getattr(backend, 'num_qubits', 0) or 0)
        qubit_props = {'readout_error': [props.readout_error(q) for q in range(n)]}
        gate_arr = []
        for (u, v) in edges:
            try:
                gate_arr.append(props.gate_error(gate_name, (u, v)))
            except Exception:
                try:
                    gate_arr.append(props.gate_error(gate_name, (v, u)))
                except Exception:
                    gate_arr.append(float('nan'))
        gate_props = {f"{gate_name}_error": gate_arr}
        # Best-effort cache warm — never let a cache write break the scan.
        try:
            lu = data._extract_last_update(props)
            if lu:
                data._save_calibration(backend_name, lu, qubit_props, gate_props)
        except Exception:
            _log.debug("Scan props: cache warm failed for %s",
                       backend_name, exc_info=True)
        return CachedPropsShim(qubit_props, gate_props, edges,
                               gate_key=f"{gate_name}_error")

    return get_props
