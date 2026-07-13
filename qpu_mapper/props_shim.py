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
        vintage: calibration provenance for the UI — ``'live'`` for the active
            backend scored on the session's current props, or the cached
            snapshot's UTC timestamp string for a remote backend. ``None`` when
            unknown.
        n_qubits: the backend's qubit count. When given, every ``qubit_props``
            array MUST have exactly this length (hard invariant below).

    HARD INVARIANTS (raise ValueError). The disk cache is POSITIONAL — it
    stores no edge list — so the gate arrays pair 1:1 with ``edges`` (the
    backend's OWN coupling map) purely by position. If a cached snapshot's
    array lengths don't match this backend's edge/qubit counts, the pairing
    is garbage (a dead coupler could land on the wrong edge, or vanish —
    exactly the 66%-scan vs 0%-live bug). We refuse to construct rather than
    silently mispair; the scan's per-backend isolation turns the ValueError
    into an honest error row.

    Missing lookups raise ``KeyError`` with a clear message.
    """

    def __init__(self, qubit_props, gate_props, edges,
                 gate_key=_DEFAULT_GATE_KEY, vintage=None, n_qubits=None):
        if 'readout_error' not in qubit_props:
            raise KeyError("qubit_props has no 'readout_error' entry")
        if gate_key not in gate_props:
            raise KeyError(f"gate_props has no {gate_key!r} entry")

        edges = list(edges)
        n_edges = len(edges)
        # Every gate array must pair 1:1 with the backend's edges.
        for gk, arr in gate_props.items():
            if len(arr) != n_edges:
                raise ValueError(
                    f"gate_props[{gk!r}] has {len(arr)} values but the "
                    f"backend exposes {n_edges} edges — cached snapshot does "
                    f"not match this backend's coupling map (positional "
                    f"mispairing would hide/relabel gates).")
        # Every qubit array must have one value per qubit.
        if n_qubits is not None:
            for qk, arr in qubit_props.items():
                if len(arr) != n_qubits:
                    raise ValueError(
                        f"qubit_props[{qk!r}] has {len(arr)} values but the "
                        f"backend has {n_qubits} qubits — cached snapshot "
                        f"does not match this backend.")

        self.vintage = vintage
        self.n_qubits = n_qubits
        self._readout = qubit_props['readout_error']
        gate_arr = gate_props[gate_key]
        # Keep the RAW (edges, gate array) so score_layout can reproduce the
        # C1 wrapper's EXACT edge surface — every coupling in the backend's
        # own map (both directions, as edges_list carries them), paired
        # positionally. Deduping here would halve the gate factors and inflate
        # the product (SYM-1).
        self._edges = [(int(u), int(v)) for (u, v) in edges]
        self._gate_arr = gate_arr
        # Deduped map for the direction-insensitive gate_error() the sweep's
        # product_score uses (one value per undirected edge). Lengths are
        # guaranteed equal above — no silent truncation.
        self._gate = {}
        for i, (u, v) in enumerate(self._edges):
            self._gate[frozenset((u, v))] = gate_arr[i]

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
        ``export.scoring.estimate_fidelity_core`` — the SAME edge surface as
        the C1 wrapper (``estimate_circuit_fidelity``): every coupling among
        the layout set, taken from the backend's OWN edge list (both
        directions), so a scan candidate lands on exactly the C1 scale.
        Returns the core dict (``None`` only if the layout is empty)."""
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
        # Full directed surface, positionally paired — mirrors the C1 wrapper's
        # `for edge in edges_list` loop (which counts both directions). NOT the
        # deduped self._gate, which would halve the factors and inflate.
        gate_edges = [(u, v, self._gate_arr[i])
                      for i, (u, v) in enumerate(self._edges)]
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
    """Build the ``get_props(backend)`` callable the scan injects.

    Product decision: scan scoring is ALWAYS LIVE — as if connecting to each
    backend right now. The disk cache is NEVER a scoring source here:
      * active backend → the session's current props (that IS live);
      * remote backend → fetch ``backend.properties()`` live, build the shim
        from the fresh data, and (side effect) warm the disk cache so the
        7/14/30-day HISTORY the stability filter reads stays fed;
      * fetch failure → raise. NO silent fallback to a cached snapshot — a
        stale score presented as current is the exact bug class we killed.

    Args:
        state: SessionState (read-through context; never mutated).
        now: optional datetime stamped into the vintage as the fetch time
            (injectable for tests). Defaults to ``datetime.now()``.
    """
    from datetime import datetime as _dt
    from . import data
    from .auto_lucj import _get_two_q_gate_name

    def _gate_key_for(backend):
        name = _get_two_q_gate_name(backend)
        return f"{name}_error" if name else _DEFAULT_GATE_KEY

    def _live_stamp():
        dt = now or _dt.now()
        return dt.strftime('%H:%M') if hasattr(dt, 'strftime') else str(dt)

    def _from_state():
        """(V1) The active backend is scored on the SESSION'S CURRENT props —
        the same qubit_props_data/gate_props_data/edges_list the C1 box reads —
        so the scan row for the active backend equals the C1 box exactly, live.
        Returns None if the session hasn't loaded calibration yet."""
        qp = getattr(state, 'qubit_props_data', None)
        gp = getattr(state, 'gate_props_data', None)
        edges = getattr(state, 'edges_list', None)
        if not qp or not gp or edges is None or 'readout_error' not in qp:
            return None
        key = _DEFAULT_GATE_KEY if _DEFAULT_GATE_KEY in gp else next(iter(gp))
        n_qubits = len(state.qubit_ids) if getattr(state, 'qubit_ids', None) else None
        return CachedPropsShim(qp, gp, list(edges), gate_key=key,
                               vintage='live', n_qubits=n_qubits)

    def get_props(backend):
        backend_name = getattr(backend, 'name', None)

        # (V1) Active backend → live session props (that IS live).
        if (backend_name is not None
                and backend_name == getattr(state, 'active_backend_name', None)):
            live = _from_state()
            if live is not None:
                _log.debug("Scan props: active backend %s → live state props",
                           backend_name)
                return live

        edges = _backend_edges(backend)

        # Remote backend → ALWAYS live fetch (no cache-first scoring read).
        # Build the shim directly from the BackendProperties so the edge
        # mapping is self-consistent regardless of cache ordering.
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
        # WRITE-ONLY cache warming (history feed) — never gates the score, and
        # never breaks the scan if it fails.
        try:
            lu = data._extract_last_update(props)
            if lu:
                data._save_calibration(backend_name, lu, qubit_props, gate_props)
        except Exception:
            _log.debug("Scan props: cache warm failed for %s",
                       backend_name, exc_info=True)
        # Vintage is 'live · <fetch time>' by construction — all scan rows are
        # freshly fetched, so there is no stale case for the table to badge.
        return CachedPropsShim(qubit_props, gate_props, edges,
                               gate_key=f"{gate_name}_error",
                               vintage=f"live · {_live_stamp()}",
                               n_qubits=n or None)

    return get_props
