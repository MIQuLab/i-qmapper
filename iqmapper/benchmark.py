"""
benchmark.py — opt-in, tiered, candidate-targeted real-time benchmarking.

I-QMapper normally scores layouts on the QPU's *reported* calibration, which
IBM refreshes about once a day. Qubits drift inside that window (transient
two-level-system defects, heating, spectator effects), so a layout chosen at
14:00 can sit on a qubit that degraded at 11:00. IBM's "Real-time benchmarking
for qubit selection" tutorial re-measures T1/T2/readout/1Q-RB/2Q-RB with
`qiskit-experiments` and feeds the fresh numbers back into the properties.

This module packages that idea as a feature that is safe to ship:

Why targeted, not full-chip
    The tutorial's whole-chip suite costs roughly four minutes of QPU time per
    run. That is unaffordable on the Open plan and wasteful even on Premium,
    since a sweep only ever cares about a few dozen qubits. Levels 1 and 2
    measure ONLY the candidate qubits and edges that the sweep actually
    proposed, which is seconds to tens of seconds. Level 3 reproduces the
    tutorial suite for people who want the full-chip diagnostic and know what
    it costs.

Why never automatic
    Benchmarking spends real quantum-computer time and money. Nothing here
    runs on connect, on a timer, or inside the multi-QPU scan — the scan
    touches dozens of backends and an implicit benchmark per backend would be
    catastrophic. A run happens only when a human clicks the button, having
    been shown the cost estimate first. `plan_gate` additionally refuses
    Level 3 (and warns on Level 2) unless the user acknowledges the cost, and
    treats an unknown plan as Open, the conservative case.

What the masks mean
    A benchmark measures *some* elements, not all. `MeasuredEpoch` therefore
    carries `measured_qubit_mask` / `measured_edge_mask`: booleans, one per
    qubit / per edge, True where the value in `qubit_props`/`gate_props` came
    from this run and False where it is still the reported value. Everything
    downstream (heatmap, LQS, re-ranking, export) reads the arrays exactly as
    it reads a reported epoch; the masks exist so the UI can say which numbers
    are fresh, and so a partial failure is visible rather than silent.

On comparing RB error to reported gate error
    Randomized benchmarking returns an error-per-gate averaged over random
    Clifford sequences; the reported `cz_error` is the provider's own
    per-gate estimate from its calibration routine. Both are per-gate error
    rates on the same footing for ranking purposes, and the tutorial swaps one
    for the other exactly this way — but they are different estimators, so an
    absolute value may shift slightly even on a qubit that has not drifted.
    Treat a measured epoch as a better ordering, not as a recalibration.

State mutation note
    `apply_measured_epoch` attaches the epoch to the session as
    `state._measured_epoch` with `setattr`. `SessionState` is a frozen engine
    in this project, so the field is deliberately NOT declared there; it is
    created at runtime and every reader must use `getattr(state,
    '_measured_epoch', None)`. Session save/restore ignores it by design — a
    measured epoch is a live, perishable thing and should not survive a
    reload as if it were calibration.

Failure policy
    Nothing in this module raises on a per-element measurement or fit failure.
    `results_to_epoch` keeps the reported value for that element, leaves its
    mask False, and appends a human-readable notice. Only a missing
    `qiskit-experiments` install (an install-time problem) raises, from
    `make_runtime_runner`.

Import policy
    The heavy runtime imports are main-thread-only by design — see
    `preload_runtime`.
"""
import json as _json
import logging
import os
import tempfile
import time
import warnings
from collections import namedtuple
from types import SimpleNamespace

import numpy as np

_log = logging.getLogger(__name__)


# ── Levels ──────────────────────────────────────────────────────────

#: Level 0 is "use the reported calibration" — the default, and not a plan:
#: it is simply the absence of a benchmark, so no code path corresponds to it.
LEVEL_REPORTED = 0
LEVEL_QUICK_READOUT = 1
LEVEL_TARGETED = 2
LEVEL_FULL = 3

LevelInfo = namedtuple('LevelInfo', 'level name description scope')

LEVELS = {
    LEVEL_QUICK_READOUT: LevelInfo(
        LEVEL_QUICK_READOUT, 'Quick readout',
        'Readout assignment error on the candidate qubits only.',
        'candidate qubits'),
    LEVEL_TARGETED: LevelInfo(
        LEVEL_TARGETED, 'Targeted',
        'Quick readout plus two-qubit randomized benchmarking on the '
        'candidate edges.',
        'candidate qubits + edges'),
    LEVEL_FULL: LevelInfo(
        LEVEL_FULL, 'Full chip',
        'The tutorial suite: T1, T2 (Hahn), readout, 1Q RB and 2Q RB over '
        'the whole device.',
        'whole chip'),
}

#: Dropdown-ready (label, level) pairs.
LEVEL_OPTIONS = [(f'{i} {LEVELS[i].name}', i)
                 for i in (LEVEL_QUICK_READOUT, LEVEL_TARGETED, LEVEL_FULL)]

# Tutorial parameters, kept verbatim so a Level-3 run reproduces its figures.
DEFAULT_SHOTS = 1000
T1T2_DELAYS = [1e-6, 20e-6, 40e-6, 80e-6, 200e-6, 400e-6]
RB_1Q_LENGTHS = [10, 100, 500]
RB_1Q_SAMPLES = 10
RB_2Q_LENGTHS = [10, 50, 100]
RB_2Q_SAMPLES = 10

#: Plausibility bounds. A failed fit does not return an error — it returns a
#: NUMBER, and an absurd one poisons whatever consumes it. A 1Q EPG of 0.71
#: forwarded into `rb_analysis._exclude_1q_error` made it evaluate
#: `(1 - 2*epg) ** n_gate` on a negative base and raise
#: "The uncertainties module does not handle complex results", killing every
#: two-qubit callback of the calibration check. IBM single-qubit gate errors
#: run about 1e-4 to 1e-3, so anything at or above 5e-2 is a failed fit, not
#: a bad qubit.
_EPG_1Q_MAX = 0.05

#: Readout assignment error is a probability; 0.5 is chance, and a symmetric
#: matrix worse than that means the fit, not the qubit, is broken.
_READOUT_MAX = 0.5

#: T1/T2 in microseconds. Superconducting coherence times are hundreds of
#: microseconds at best; 10 ms is a fit that ran away.
_COHERENCE_MAX_US = 10_000.0

#: A saved epoch older than this is offered as history rather than as "reuse".
REUSE_TTL_MINUTES = 60


# ── Cost model ──────────────────────────────────────────────────────
#
# Deliberately a transparent formula over circuit counts rather than a fitted
# model: the point is that a user can read it and see where the number came
# from. The per-circuit constants are back-solved so that the tutorial's
# full-chip suite on a 127-qubit device lands near its quoted ~240 s, which
# makes them accurate to roughly a factor of two and no better. They are
# wall-clock seconds per circuit including the provider's own overhead, not
# pure quantum-execution time.

_SEC_PER_CIRCUIT_1Q = 1.20
_SEC_PER_CIRCUIT_2Q = 1.90
_SEC_PER_T1T2 = 1.20
_SESSION_OVERHEAD_S = 8.0

# The model charges two-qubit RB per EDGE, not per layer, even though
# `build_plan` groups disjoint edges into layers that a provider can execute
# together. That makes the estimate an upper bound rather than a best case,
# which is the right direction to err in when the number is being shown to
# someone about to spend their allowance. The per-qubit scaling on the
# parallel experiments is there for the same reason: transpilation and
# analysis cost really does grow with qubit count even when execution does
# not. Calibration check lives in tests/test_benchmark.py.


# ── Data types ──────────────────────────────────────────────────────

Targets = namedtuple('Targets', 'qubits edges')
ExperimentSpec = namedtuple('ExperimentSpec', 'kind qubits params layer')
BenchmarkPlan = namedtuple('BenchmarkPlan', 'level specs targets shots')
CostEstimate = namedtuple('CostEstimate', 'circuits qpu_seconds text')
Gate = namedtuple('Gate', 'allowed warning')


class MeasuredEpoch:
    """A calibration epoch whose measured elements came from a benchmark run.

    `qubit_props` / `gate_props` are full-width dict-of-arrays in exactly the
    shape `state.qubit_props_data` / `state.gate_props_data` use, so the epoch
    drops into the existing channel untouched. The masks say which entries are
    measured; everything else is the reported value carried through.
    """

    __slots__ = ('qubit_props', 'gate_props', 'measured_qubit_mask',
                 'measured_edge_mask', 'level', 'when', 'notices',
                 'planned_count', 'extra', 'consistency', 'relative')

    def __init__(self, qubit_props, gate_props, measured_qubit_mask,
                 measured_edge_mask, level, when, notices=None,
                 planned_count=None, extra=None, consistency=None,
                 relative=False):
        self.qubit_props = qubit_props
        self.gate_props = gate_props
        self.measured_qubit_mask = np.asarray(measured_qubit_mask, dtype=bool)
        self.measured_edge_mask = np.asarray(measured_edge_mask, dtype=bool)
        self.level = int(level)
        self.when = when                      # epoch seconds (time.time())
        self.notices = list(notices or [])
        # (n_qubits, n_edges) the plan INTENDED to measure. Compared against
        # measured_count so the UI can tell "complete", "partial" and "nothing
        # came back" apart — the last of which must never read as success.
        self.planned_count = tuple(planned_count or (0, 0))
        # Side data the arrays have no column for: per-qubit 1Q EPGs (used to
        # subtract single-qubit error from the 2Q Clifford decay), and
        # anything else diagnostic.
        self.extra = dict(extra or {})
        # Filled by `consistency_stats` after folding: how the measured values
        # compare with the reported ones, per property.
        self.consistency = dict(consistency or {})
        # True when the values were rescaled to the reported medians because a
        # systematic offset was detected (see `apply_relative_mode`).
        self.relative = bool(relative)

    # ── convenience ────────────────────────────────────────────
    @property
    def n_measured_qubits(self):
        return int(self.measured_qubit_mask.sum())

    @property
    def n_measured_edges(self):
        return int(self.measured_edge_mask.sum())

    @property
    def measured_count(self):
        return (self.n_measured_qubits, self.n_measured_edges)

    @property
    def is_empty(self):
        """Nothing was measured. A FAILURE, not a successful no-op."""
        return self.measured_count == (0, 0)

    @property
    def is_complete(self):
        return (self.planned_count != (0, 0)
                and self.measured_count == self.planned_count)

    def coverage_text(self):
        """'120/156 qubits, 210/240 edges' for the status line."""
        pq, pe = self.planned_count
        return (f'{self.n_measured_qubits}/{pq} qubits, '
                f'{self.n_measured_edges}/{pe} edges')

    def age_minutes(self, now=None):
        return ((time.time() if now is None else now) - self.when) / 60.0

    def younger_than(self, minutes, now=None):
        """True when this epoch is fresh enough to offer for reuse."""
        return self.age_minutes(now) < minutes

    def time_label(self):
        return time.strftime('%H:%M', time.localtime(self.when))

    def __repr__(self):                       # pragma: no cover - debug aid
        return (f'<MeasuredEpoch L{self.level} {self.time_label()} '
                f'{self.n_measured_qubits}q/{self.n_measured_edges}e>')


# ── Target selection ────────────────────────────────────────────────

def _normalize_edge(a, b, coupling_edges):
    """Return the edge oriented as the backend's coupling map lists it.

    The sweep works on an undirected view, but the reported gate arrays are
    indexed by `state.edges_list`, whose orientation comes from the device.
    Writing a measured value against the wrong orientation would silently
    update the wrong edge, so every edge goes through here.
    """
    if (a, b) in coupling_edges:
        return (a, b)
    if (b, a) in coupling_edges:
        return (b, a)
    return None


#: Target-selection modes, in the order the UI lists them.
TARGET_CURRENT = 'current'
TARGET_SWEEP = 'sweep'
TARGET_WHOLE = 'whole'
TARGET_CUSTOM = 'custom'

TARGET_LABELS = {
    TARGET_CURRENT: 'Current layout(s)',
    TARGET_SWEEP: 'Sweep candidates (top-K)',
    TARGET_WHOLE: 'Whole QPU',
    TARGET_CUSTOM: 'Custom',
}


def parse_qubit_list(text):
    """Qubit indices out of free text, by the paste box's own rules.

    Deliberately NOT `export.autodetect.parse_auto_text`: that function maps
    the parsed qubits onto circuit layouts as a side effect, which is exactly
    wrong for choosing what to measure. The token rules are the same ones it
    uses — strip brackets, split on commas and whitespace, accept
    non-negative integers, drop everything else — so a list that works in the
    paste box works here.
    """
    if not text:
        return []
    cleaned = (str(text).replace('[', ' ').replace(']', ' ')
               .replace(',', ' ').replace('\n', ' '))
    out = []
    for token in cleaned.split():
        if token.isdigit():
            value = int(token)
            if value not in out:
                out.append(value)
    return out


def _edges_within(qubits, coupling):
    """Coupling edges with BOTH endpoints in `qubits`, device orientation."""
    wanted = set(qubits)
    seen = set()
    edges = []
    for (a, b) in coupling:
        if a in wanted and b in wanted:
            edge = _normalize_edge(a, b, set(coupling))
            if edge is not None and edge not in seen:
                seen.add(edge)
                edges.append(edge)
    return edges


def select_targets(state, sweep_result=None, top_k=5, mode=None, custom=None):
    """Qubits and edges worth measuring, as a `Targets`.

    `mode` picks what "worth measuring" means:

      `current` — every qubit placed on the map, across ALL circuit slots,
          however it got there: hand-placed, auto-placed, or handed over from
          a scan. This is the default, because the thing a user is about to
          run is the thing worth measuring.
      `sweep`   — the union over the top-K candidates of the last sweep, for
          choosing BETWEEN candidates rather than validating one.
      `whole`   — the entire device. Worth knowing: Level 1 does not scale
          with qubit count (readout is two circuits whatever the width), so
          whole-QPU Level 1 is the cheap way to refresh the readout heatmap.
          Levels 2 and 3 scale with edge count and are not cheap.
      `custom`  — an explicit qubit list; edges are the coupling edges with
          both endpoints in it.

    With no `mode`, falls back to the original behaviour: sweep candidates if
    there are any, else the current layout, else empty — and empty means the
    UI disables the button rather than quietly measuring the whole chip.

    Edges always come back in the device's own orientation, because the gate
    arrays are indexed by `state.edges_list` and writing a measured value
    against a flipped edge would silently update the wrong one.
    """
    coupling = [tuple(e) for e in (getattr(state, 'edges_list', None) or [])]

    if mode == TARGET_WHOLE:
        n_q = len(getattr(state, 'qubit_ids', None) or [])
        return Targets(list(range(n_q)), list(coupling))

    if mode == TARGET_CUSTOM:
        qubits = [q for q in parse_qubit_list(custom)
                  if 0 <= q < len(getattr(state, 'qubit_ids', None) or [])]
        return Targets(sorted(qubits), _edges_within(qubits, coupling))

    if mode == TARGET_CURRENT:
        qubits = _qubits_from_current_layout(state)
        return Targets(sorted(qubits), _edges_within(qubits, coupling))

    if mode == TARGET_SWEEP:
        qubits = _qubits_from_sweep(sweep_result, top_k)
        return Targets(sorted(qubits), _edges_within(qubits, coupling))

    # Legacy/auto: sweep candidates, else the current layout, else nothing.
    qubits = _qubits_from_sweep(sweep_result, top_k) \
        or _qubits_from_current_layout(state)
    if not qubits:
        return Targets([], [])
    return Targets(sorted(qubits), _edges_within(qubits, coupling))


def available_target_modes(state, sweep_result=None):
    """The modes the UI should offer, in order, for this session.

    `sweep` appears only once a sweep has actually produced candidates, so the
    dropdown never offers a mode that would resolve to nothing.
    """
    modes = [TARGET_CURRENT]
    if _qubits_from_sweep(sweep_result, 5):
        modes.append(TARGET_SWEEP)
    modes.append(TARGET_WHOLE)
    modes.append(TARGET_CUSTOM)
    return modes


def default_target_mode(state, sweep_result=None):
    """`current` when something is placed, else `sweep`, else `current`."""
    if _qubits_from_current_layout(state):
        return TARGET_CURRENT
    if _qubits_from_sweep(sweep_result, 5):
        return TARGET_SWEEP
    return TARGET_CURRENT


def _qubits_from_sweep(sweep_result, top_k):
    """Union of the top-K sweep candidates' qubits."""
    qubits = set()
    for entry in _flatten_candidates(sweep_result, top_k):
        for key in ('alpha', 'beta', 'ancillas'):
            for q in (entry.get(key) or []):
                try:
                    qubits.add(int(q))
                except (TypeError, ValueError):
                    continue
    return qubits


def _flatten_candidates(sweep_result, top_k):
    """Top-K entries out of either candidate-list or raw-sweep-dict form."""
    if not sweep_result:
        return []
    if isinstance(sweep_result, dict):
        flat = []
        for _k, entries in sorted((sweep_result.get('strata') or {}).items()):
            flat.extend(entries or [])
    else:
        flat = list(sweep_result)
    return [e for e in flat[:max(0, int(top_k))] if isinstance(e, dict)]


def _qubits_from_current_layout(state):
    """Physical qubits painted on the map right now (all circuits)."""
    out = set()
    for d in (getattr(state, 'layouts_data', None) or {}).values():
        if not isinstance(d, dict):
            continue
        for role in ('alpha', 'beta', 'ancilla', 'other'):
            for q in (d.get(role) or []):
                try:
                    out.add(int(q))
                except (TypeError, ValueError):
                    continue
    return out


# ── Plan construction ───────────────────────────────────────────────

def _layer_edges(edges):
    """Group edges into layers that share no qubit, via edge colouring.

    Two-qubit RB on disjoint pairs runs in parallel, which is where most of
    the saving in Level 2 comes from. `rustworkx.graph_bipartite_edge_color`
    gives a proper edge colouring of the TARGET subgraph (not the whole chip),
    so each colour class is a set of vertex-disjoint pairs. Falls back to a
    greedy pass if rustworkx is unavailable or the call fails, and in the
    worst case to one edge per layer, which is correct but slower.
    """
    if not edges:
        return []
    try:
        import rustworkx

        graph = rustworkx.PyGraph()
        index = {}
        for (a, b) in edges:
            for q in (a, b):
                if q not in index:
                    index[q] = graph.add_node(q)
        handles = {}
        for (a, b) in edges:
            handles[graph.add_edge(index[a], index[b], None)] = (a, b)
        coloring = rustworkx.graph_bipartite_edge_color(graph)
        layers = {}
        for edge_handle, color in dict(coloring).items():
            edge = handles.get(edge_handle)
            if edge is not None:
                layers.setdefault(color, []).append(edge)
        if layers:
            return [layers[c] for c in sorted(layers)]
    except Exception:
        _log.debug("edge colouring unavailable; falling back to greedy",
                   exc_info=True)
    return _greedy_layers(edges)


def _greedy_layers(edges):
    """First-fit layering: correct for any graph, just not always minimal."""
    layers = []
    for (a, b) in edges:
        for layer in layers:
            if all(a not in pair and b not in pair for pair in layer):
                layer.append((a, b))
                break
        else:
            layers.append([(a, b)])
    return layers


def build_plan(level, targets, n_qubits, coupling_edges):
    """The experiment specs a run at `level` would execute.

    Returns a `BenchmarkPlan` of provider-agnostic `ExperimentSpec`s — no
    qiskit objects are constructed here, which is what lets the whole planning
    and costing path be tested offline with no quantum stack installed.

    THE STRUCTURE IS COMPOSITE, and that is the point. Each spec becomes ONE
    `ParallelExperiment` covering many qubits or pairs at once, exactly as the
    tutorial does:

      readout     -> one LocalReadoutError over every target qubit: 2 circuits
                     whatever the width.
      t1 / t2hahn -> one ParallelExperiment of per-qubit T1/T2Hahn: one circuit
                     per delay, so 6 each, not 6 x n_qubits.
      rb1q        -> one ParallelExperiment of per-qubit StandardRB:
                     len(lengths) x num_samples = 30 circuits.
      rb2q_layer  -> one ParallelExperiment per edge-colouring LAYER, each
                     layer a set of vertex-disjoint pairs: 30 circuits per
                     layer, and heavy-hex needs about three layers.

    A full-chip Level 3 on 156 qubits is therefore 6 + 6 + 2 + 30 + 3x30 =
    ~134 circuits. The previous version emitted one spec per qubit and per
    EDGE, which on that device meant ~43,400 circuits in a single job and was
    rejected outright by IBM (error 1520, shot limit). Keeping the specs
    composite is what keeps the run submittable; `plan_circuit_count` is the
    check on it.
    """
    level = int(level)
    coupling = [tuple(e) for e in (coupling_edges or [])]

    if level == LEVEL_FULL:
        qubits = list(range(int(n_qubits)))
        edges = list(coupling)
        targets = Targets(qubits, edges)
    else:
        qubits = list(targets.qubits)
        edges = list(targets.edges)

    specs = []

    if level == LEVEL_FULL and qubits:
        specs.append(ExperimentSpec('t1', tuple(qubits),
                                    {'delays': list(T1T2_DELAYS)}, 0))
        specs.append(ExperimentSpec('t2hahn', tuple(qubits),
                                    {'delays': list(T1T2_DELAYS)}, 0))

    if qubits:
        specs.append(ExperimentSpec('readout', tuple(qubits), {}, 0))

    # 1Q RB runs at L2 as well as L3, and BEFORE the two-qubit layers: its
    # per-qubit EPGs are what lets the 2Q analysis subtract single-qubit error
    # from the Clifford decay. Without that subtraction a 2Q Clifford's ~8-10
    # single-qubit gates are all attributed to the two-qubit gate, which on
    # 20 Sept inflated every measured cz_error by roughly 2-3x uniformly.
    if level in (LEVEL_TARGETED, LEVEL_FULL) and qubits:
        specs.append(ExperimentSpec(
            'rb1q', tuple(qubits),
            {'lengths': list(RB_1Q_LENGTHS), 'num_samples': RB_1Q_SAMPLES}, 0))

    if level in (LEVEL_TARGETED, LEVEL_FULL) and edges:
        # One ParallelExperiment per colour class. Pairs inside a layer are
        # vertex-disjoint, so they genuinely run together.
        for layer_idx, layer in enumerate(_layer_edges(edges)):
            flat = tuple(q for pair in layer for q in pair)
            specs.append(ExperimentSpec(
                'rb2q_layer', flat,
                {'pairs': [tuple(p) for p in layer],
                 'lengths': list(RB_2Q_LENGTHS),
                 'num_samples': RB_2Q_SAMPLES}, layer_idx))

    return BenchmarkPlan(level, specs, targets, DEFAULT_SHOTS)


# ── Circuit accounting and the per-job shot budget ──────────────────

#: IBM rejects a job whose total circuit-shots exceed this (error 1520).
_MAX_CIRCUIT_SHOTS_PER_JOB = 10_000_000

#: Submit at 80% of the hard limit; the provider counts a little differently
#: than we do and a rejected job costs a whole round trip.
_BUDGET_MARGIN = 0.8

#: Effective per-job ceiling.
_JOB_SHOT_BUDGET = int(_MAX_CIRCUIT_SHOTS_PER_JOB * _BUDGET_MARGIN)

JobChunk = namedtuple('JobChunk', 'specs circuits shots index')


def _spec_circuits(spec):
    """Circuits ONE spec contributes, at the tutorial's parameterisation.

    Every branch is flat in the number of qubits or pairs, because every spec
    is a ParallelExperiment: widening a spec adds qubits to each circuit, not
    circuits to the job.
    """
    params = spec.params
    if spec.kind in ('t1', 't2hahn'):
        return len(params.get('delays') or [])
    if spec.kind == 'readout':
        return 2                              # |0...0> and |1...1> preparations
    if spec.kind in ('rb1q', 'rb2q_layer', 'rb2q'):
        return (len(params.get('lengths') or [])
                * int(params.get('num_samples') or 0))
    return 1


def plan_circuit_count(plan):
    """Exact number of circuits the plan submits."""
    return sum(_spec_circuits(spec) for spec in plan.specs)


def plan_planned_count(plan):
    """(n_qubits, n_edges) this plan actually intends to MEASURE.

    Not the target counts: a Level 1 plan targets qubits and edges but only
    ever measures readout, so counting its target edges as "planned" would
    make a perfect L1 run look permanently partial. Planned means "a spec in
    this plan writes to that element".
    """
    # Pairs are counted AS GIVEN, not sorted into undirected pairs: a
    # device's gate arrays are indexed by its directed edge list, so
    # (a, b) and (b, a) are two array slots and the measured mask counts
    # both. Deduplicating here made a complete run read "348/176 edges".
    qubits, edges = set(), set()
    for spec in plan.specs:
        if spec.kind in ('readout', 't1', 't2hahn'):
            qubits.update(int(q) for q in spec.qubits)
        elif spec.kind == 'rb2q_layer':
            for pair in (spec.params.get('pairs') or []):
                edges.add(tuple(int(q) for q in pair))
        elif spec.kind == 'rb2q':
            edges.add(tuple(int(q) for q in spec.qubits))
    return (len(qubits), len(edges))


def plan_shots(plan, shots=DEFAULT_SHOTS):
    """Total circuit-shots: the quantity IBM's per-job limit applies to."""
    return plan_circuit_count(plan) * int(shots)


def _split_spec(spec, shots):
    """Break one over-budget spec into several that each fit.

    Only RB specs can be split, by reducing `num_samples` — the samples are
    independent random Clifford sequences, so running them across two jobs is
    the same experiment, just submitted twice. T1/T2/readout are already tiny
    (2-6 circuits) and cannot exceed a budget measured in millions.
    """
    circuits = _spec_circuits(spec)
    if circuits * shots <= _JOB_SHOT_BUDGET:
        return [spec]
    samples = int(spec.params.get('num_samples') or 0)
    lengths = len(spec.params.get('lengths') or [])
    if samples <= 1 or lengths == 0:
        return [spec]                         # indivisible; caller reports it
    per_job = max(1, _JOB_SHOT_BUDGET // (lengths * shots))
    out = []
    remaining = samples
    while remaining > 0:
        take = min(per_job, remaining)
        params = dict(spec.params)
        params['num_samples'] = take
        out.append(spec._replace(params=params))
        remaining -= take
    return out


def split_for_submission(plan, shots=DEFAULT_SHOTS):
    """Group the plan's experiments into jobs that each fit the shot limit.

    Returns a list of `JobChunk`. Every spec ends up in exactly one chunk —
    nothing is dropped, which is the property the tests pin, because silently
    discarding an experiment would look like a partial measurement rather than
    a bug.
    """
    shots = int(shots)
    expanded = []
    for spec in plan.specs:
        expanded.extend(_split_spec(spec, shots))

    chunks = []
    current, current_circuits = [], 0
    for spec in expanded:
        circuits = _spec_circuits(spec)
        if current and (current_circuits + circuits) * shots > _JOB_SHOT_BUDGET:
            chunks.append(JobChunk(tuple(current), current_circuits,
                                   current_circuits * shots, len(chunks)))
            current, current_circuits = [], 0
        current.append(spec)
        current_circuits += circuits
    if current:
        chunks.append(JobChunk(tuple(current), current_circuits,
                               current_circuits * shots, len(chunks)))
    return chunks


def plan_is_submittable(plan, shots=DEFAULT_SHOTS):
    """(ok, reason). False when some chunk still exceeds the hard limit.

    A plan that cannot be made to fit must disable Run rather than be sent and
    rejected: a rejection costs a round trip and, before this version, ended in
    a false success.
    """
    for chunk in split_for_submission(plan, shots):
        if chunk.shots > _MAX_CIRCUIT_SHOTS_PER_JOB:
            _log.warning(
                'one experiment needs %s circuit-shots, over IBM\'s %s '
                'per-job limit, and cannot be split further',
                f'{chunk.shots:,}', f'{_MAX_CIRCUIT_SHOTS_PER_JOB:,}')
            return False, (
                "One experiment exceeds IBM's per-job limit \u2014 reduce "
                "the targets or the level")
    return True, None


def estimate_cost(plan, shots=DEFAULT_SHOTS):
    """Wall-clock QPU seconds a plan is expected to consume.

    Now that every spec is a ParallelExperiment, the driver is the CIRCUIT
    COUNT, not the qubit count: widening a spec puts more qubits into the same
    circuits rather than adding circuits. So the model is flat per circuit,
    with a higher rate for two-qubit RB because those circuits are deeper.

    An ESTIMATE, and a coarse one: back-solved from the tutorial's single
    published figure (~4 min for the full suite) and blind to queueing and
    device load. `circuits` and `shots` on the returned estimate are exact;
    only `qpu_seconds` is a guess.
    """
    circuits = plan_circuit_count(plan)
    seconds = 0.0
    for spec in plan.specs:
        n = _spec_circuits(spec)
        if spec.kind in ('rb2q_layer', 'rb2q'):
            seconds += n * _SEC_PER_CIRCUIT_2Q
        elif spec.kind in ('t1', 't2hahn'):
            seconds += n * _SEC_PER_T1T2
        else:
            seconds += n * _SEC_PER_CIRCUIT_1Q

    seconds *= (shots / float(DEFAULT_SHOTS)) if DEFAULT_SHOTS else 1.0
    seconds += _SESSION_OVERHEAD_S if plan.specs else 0.0

    n_jobs = len(split_for_submission(plan, shots)) if plan.specs else 0
    total_shots = circuits * int(shots)
    return CostEstimate(circuits, seconds,
                        _format_cost(circuits, total_shots, n_jobs, seconds))


def _format_shots(total):
    if total >= 1_000_000:
        return f'{total / 1_000_000:.1f}M shots'
    if total >= 1_000:
        return f'{total / 1_000:.0f}k shots'
    return f'{total} shots'


def _format_cost(circuits, total_shots, n_jobs, seconds):
    """The one line that has to tell the truth before the click.

    Circuits, shots and job count are exact; the seconds are the estimate.
    Showing the job count matters because a run that needs more than one job
    is the situation that used to fail at submission.
    """
    if circuits == 0:
        return 'nothing to measure'
    job_txt = '1 job' if n_jobs <= 1 else f'{n_jobs} jobs'
    if seconds < 90:
        time_txt = f'~{seconds:.0f} s QPU (estimate)'
    else:
        time_txt = f'~{seconds / 60.0:.1f} min QPU (estimate)'
    return (f'~{circuits} circuits · {_format_shots(total_shots)} · '
            f'{job_txt} · {time_txt}')


# ── Plan (subscription) gating ──────────────────────────────────────

# Substrings that mark a subscription with no per-run allowance to protect.
# 'on-prem' joined the list after a live run reported `plan: on-prem` and was
# needlessly gated as if it were Open. Matched case-insensitively as a
# substring, so 'On-Prem', 'premium-plan' and 'IBM Dedicated' all land here.
_UNMETERED_PLANS = ('premium', 'dedicated', 'internal', 'on-prem', 'onprem')


def plan_gate(plan_name, level):
    """Whether `level` may run under subscription `plan_name`.

    An unknown plan is treated as Open. That is the conservative direction:
    the worst case of guessing Open on a Premium account is one extra
    acknowledgement click, whereas guessing Premium on an Open account can
    burn a monthly allowance in a single run.
    """
    level = int(level)
    name = (plan_name or '').strip().lower()
    unmetered = any(p in name for p in _UNMETERED_PLANS)

    if unmetered:
        return Gate(True, None)

    if level == LEVEL_QUICK_READOUT:
        return Gate(True, None)
    if level == LEVEL_TARGETED:
        return Gate(True,
                    'Level 2 spends QPU time from your monthly allowance — '
                    'tick the acknowledgement to continue.')
    if level == LEVEL_FULL:
        return Gate(False,
                    'Level 3 uses ~4 min of QPU \u2014 a large share of an '
                    'Open-plan monthly allowance')
    return Gate(True, None)


def requires_acknowledgement(plan_name, level):
    """True when the UI must see the cost checkbox ticked before running."""
    level = int(level)
    if level == LEVEL_FULL:
        return True                           # always, on every plan
    gate = plan_gate(plan_name, level)
    return gate.warning is not None or not gate.allowed


# ── Execution ───────────────────────────────────────────────────────

_INSTALL_HINT = ('Real-time benchmarking needs qiskit-experiments. '
                 'Install it with:  pip install "i-qmapper[benchmark]"')


def have_qiskit_experiments():
    """Cheap availability probe for the UI, with no import side effects."""
    try:
        import importlib.util
        return importlib.util.find_spec('qiskit_experiments') is not None
    except Exception:                         # pragma: no cover - defensive
        return False


def run_plan(plan, runner):
    """Execute `plan` through the injected `runner` and return raw results.

    `runner` is any callable taking the plan and returning a mapping of raw
    per-experiment results. Everything that touches a network or a quantum
    stack lives behind it, which is what makes the rest of this module — and
    every test — offline. `make_runtime_runner` builds the production one.
    """
    if not plan.specs:
        return {}
    return runner(plan)


def _runtime_api():
    """Resolve the quantum stack, late and in one place.

    Every class the production runner needs comes through here, for two
    reasons. It keeps the imports lazy, so `import iqmapper.benchmark` stays
    free for the majority of users who never benchmark. And it gives the
    contract tests a single seam to monkeypatch, so the runner's call shapes
    can be pinned offline — which is exactly what was missing when the
    `Session(service=...)` signature drifted underneath us and the failure
    only surfaced on a live device.
    """
    try:
        # pandas is pulled in transitively by qiskit_experiments, but it is
        # named explicitly here so that preloading this module genuinely
        # finishes importing it. See preload_runtime() for why that matters.
        import pandas                                  # noqa: F401
        from qiskit_experiments.framework import (BatchExperiment,
                                                  ParallelExperiment)
        from qiskit_experiments.library import (LocalReadoutError, StandardRB,
                                                T1, T2Hahn)
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc

    from qiskit_ibm_runtime import Session, SamplerV2

    return SimpleNamespace(
        Session=Session, SamplerV2=SamplerV2,
        BatchExperiment=BatchExperiment, ParallelExperiment=ParallelExperiment,
        LocalReadoutError=LocalReadoutError, StandardRB=StandardRB,
        T1=T1, T2Hahn=T2Hahn)


#: Resolved runtime stack, cached after the first successful preload.
_RUNTIME_CACHE = [None]


def runtime_ready():
    """True once the quantum stack has been imported in this process.

    The UI uses it to decide whether to show "Loading benchmark libraries…"
    before the first run; later runs skip the message because the preload is
    a no-op.
    """
    return _RUNTIME_CACHE[0] is not None


def preload_runtime():
    """Import the quantum stack now, on the calling thread. Idempotent.

    This exists because of a live crash. `make_runtime_runner` used to resolve
    its imports lazily inside the background worker, so the first benchmark
    run imported qiskit_experiments — and transitively pandas and scipy — on a
    worker thread while the kernel's main thread was still serving widget
    events. Plotly's click handler (`_handler_js2py_pointsCallback`) probes
    `sys.modules['pandas']`, found the half-initialised module the worker had
    just put there, and raised:

        AttributeError: partially initialized module 'pandas'

    Clicking the map during a benchmark was enough to trigger it. Python's
    import machinery publishes a module into `sys.modules` before executing
    it, so any thread reading `sys.modules` mid-import sees a shell — this is
    inherent to importing a large package off the main thread while the main
    thread is live, not a plotly or pandas bug.

    The fix is to do the import on the MAIN thread before the worker starts.
    The UI calls this from its run handler ahead of `spawn_bg`; it blocks for a
    second or two on first use and returns instantly thereafter. The worker
    then only ever finds an already-populated cache and never triggers a
    first-time import.

    Raises ImportError with the install hint when the extra is missing.
    """
    if _RUNTIME_CACHE[0] is None:
        _RUNTIME_CACHE[0] = _runtime_api()
    return _RUNTIME_CACHE[0]


def _reset_runtime_cache():
    """Drop the cached stack. For tests that swap `_runtime_api` out."""
    _RUNTIME_CACHE[0] = None


class BenchmarkRunError(RuntimeError):
    """A job was rejected, errored or cancelled by the provider.

    Carries the provider's own message and the job ids, because "no results
    returned" told the user nothing on the run that motivated this class: IBM
    had rejected the submission outright (error 1520, shot limit) and the
    panel still reported success.
    """

    def __init__(self, message, job_ids=None, chunk=None, expdata=None):
        super().__init__(message)
        self.job_ids = list(job_ids or [])
        self.chunk = chunk
        # Carried so the panel can still write a raw dump on the failure path,
        # which is exactly when the diagnostic artefact matters most.
        self.expdata = list(expdata or [])


#: Analysis-result names the tutorial's experiments produce. Kept in one place
#: because they are the contract between qiskit-experiments and our parser, and
#: a rename there silently emptied every epoch before the integration tests.
RESULT_T1 = 'T1'
RESULT_T2 = 'T2'
RESULT_READOUT_MITIGATOR = 'Local Readout Mitigator'
RESULT_EPG_PREFIXES = ('EPG', 'EPC')


def _build_experiment(spec, api, context=None):
    """One `ExperimentSpec` → ONE composite qiskit-experiments object.

    Every branch returns a single experiment covering all of the spec's qubits
    or pairs at once. `flatten_results=True` is passed explicitly rather than
    left to the installed version's default, so the analysis results land at a
    predictable depth whatever version is installed.
    """
    physical = list(spec.qubits)
    if spec.kind == 't1':
        return api.ParallelExperiment(
            [api.T1((q,), delays=spec.params['delays']) for q in physical],
            flatten_results=True)
    if spec.kind == 't2hahn':
        return api.ParallelExperiment(
            [api.T2Hahn((q,), delays=spec.params['delays']) for q in physical],
            flatten_results=True)
    if spec.kind == 'readout':
        return api.LocalReadoutError(physical)
    if spec.kind == 'rb1q':
        return api.ParallelExperiment(
            [api.StandardRB((q,), lengths=spec.params['lengths'],
                            num_samples=spec.params['num_samples'])
             for q in physical],
            flatten_results=True)
    if spec.kind == 'rb2q_layer':
        pairs = [tuple(p) for p in spec.params['pairs']]
        epg_results = (context or {}).get('epg_results') or {}
        notices = (context or {}).setdefault('notices', []) \
            if context is not None else []
        inner = []
        applied, skipped = 0, []
        for pair in pairs:
            rb = api.StandardRB(pair, lengths=spec.params['lengths'],
                                num_samples=spec.params['num_samples'])
            if apply_epg_1q_options(rb, pair, epg_results, notices):
                applied += 1
            else:
                skipped.append(tuple(pair))
            inner.append(rb)
        # MANDATORY: whether the subtraction happened is the difference
        # between a measured cz_error of 2.9e-3 and 4.0e-3, so it is never
        # left to be inferred from the absence of a message.
        if context is not None:
            tally = context.setdefault('_epg_tally', {'applied': 0,
                                                      'skipped': []})
            tally['applied'] += applied
            tally['skipped'].extend(skipped)
        return api.ParallelExperiment(inner, flatten_results=True)
    if spec.kind == 'rb2q':                   # legacy single-pair spec
        return api.StandardRB(tuple(physical),
                              lengths=spec.params['lengths'],
                              num_samples=spec.params['num_samples'])
    raise ValueError(f'unknown experiment kind: {spec.kind}')


EpgHarvest = namedtuple('EpgHarvest', 'results rejected')


def _plausible_epg(value):
    return (value is not None and np.isfinite(value)
            and 0.0 <= value < _EPG_1Q_MAX)


def harvest_epg_results(entry):
    """Per-qubit lists of the ACTUAL 1Q EPG analysis-result objects.

    Not floats. `rb_analysis._exclude_1q_error` iterates whatever is handed to
    `epg_1_qubit` and reads `.device_components` off each element, so passing
    plain numbers raised

        AttributeError: 'float' object has no attribute 'device_components'

    in every two-qubit callback of the 20 Sept 13:08 run — phase 1 succeeded,
    phase 2 produced no EPG_cz at all, and because the JOBS were DONE nothing
    upstream noticed. The tutorial passes the 1Q RB analysis results
    themselves, so that is what we keep and forward.

    Results whose value is implausible (see `_EPG_1Q_MAX`) are withheld: a
    failed fit returns a number, and forwarding one crashed the whole 2Q
    analysis. Returns `EpgHarvest(results, rejected)`; a rejected qubit is
    treated as having no 1Q EPG at all, so its pairs are simply not
    subtracted — and the tally says which.
    """
    out, values = {}, {}
    items = entry if isinstance(entry, (list, tuple)) else [entry]
    for item in items:
        name = str(getattr(item, 'name', '') or '')
        if not name.upper().startswith('EPG'):
            continue
        qubits = _components_to_qubits(item)
        if len(qubits) != 1:
            continue
        qubit = int(qubits[0])
        try:
            value = _result_value(item)
        except Exception:
            value = None
        out.setdefault(qubit, []).append(item)
        # Track the worst value seen for the qubit: one broken basis-gate fit
        # is enough to poison the subtraction for every pair it touches.
        if value is not None and np.isfinite(value):
            values[qubit] = max(values.get(qubit, value), float(value))
        else:
            values[qubit] = float('nan')

    rejected = {q: v for q, v in values.items() if not _plausible_epg(v)}
    for qubit in rejected:
        out.pop(qubit, None)
    return EpgHarvest(out, rejected)


def apply_epg_1q_options(experiment, pair, epg_results, notices=None):
    """Hand a two-qubit RB analysis the 1Q EPG RESULTS for its two qubits.

    `epg_results` maps qubit -> list of analysis-result objects harvested from
    the phase-1 1Q RB run (see `harvest_epg_results`).

    When a result is missing for either qubit the subtraction is SKIPPED for
    that pair, with a notice, rather than substituted with a fabricated
    stand-in. That is a deliberate choice: the failure mode is deferred to
    analysis time — a wrong shape does not raise when the option is set, it
    raises inside the fit callback after the QPU time has already been spent,
    and `set_options` cannot validate it. A synthetic object that merely looks
    right would risk repeating exactly the failure this function exists to
    fix, so an un-subtracted pair plus a visible notice is the safer trade.

    Returns True when the option was set.
    """
    notices = notices if notices is not None else []
    pair = tuple(pair)
    results = []
    missing = []
    for q in pair:
        got = (epg_results or {}).get(int(q))
        if got:
            results.extend(got)
        else:
            missing.append(int(q))

    if missing:
        return False

    analysis = getattr(experiment, 'analysis', None)
    if analysis is None or not hasattr(analysis, 'set_options'):
        return False
    try:
        analysis.set_options(epg_1_qubit=results)
        return True
    except Exception as exc:                  # noqa: BLE001 - reported
        notices.append(f'{list(pair)}: the installed qiskit-experiments '
                       f'rejected epg_1_qubit ({exc}); not 1Q-subtracted')
        return False


def epg_tally_notice(context):
    """The one line that says whether 1Q subtraction actually happened.

    Absence of a message is not evidence of success: a reanalysis replayed
    from a SINGLE-phase record has no 1Q EPGs to hand the 2Q analyses, and
    produced an un-subtracted cz_error (x2.85 rather than x2.11) with nothing
    on screen to say so. This notice is emitted on every run that contains
    two-qubit RB, whatever the outcome.
    """
    tally = (context or {}).get('_epg_tally')
    if not tally:
        return None
    applied = int(tally.get('applied') or 0)
    skipped = list(tally.get('skipped') or [])
    rejected = dict((context or {}).get('_epg_rejected') or {})
    if not applied and not skipped:
        return None
    text = f'1Q EPG subtraction: applied to {applied} pairs'
    if skipped:
        qubits = sorted({q for pair in skipped for q in pair})
        # Name the implausible fits with their value: "q17: 1Q EPG 0.71
        # implausible" tells the user which qubit to look at, where a bare
        # count does not.
        blamed = [f'q{q}: 1Q EPG {rejected[q]:.2g} implausible'
                  for q in qubits if q in rejected and
                  np.isfinite(rejected[q])]
        blamed += [f'q{q}: 1Q EPG fit failed'
                   for q in qubits if q in rejected and
                   not np.isfinite(rejected[q])]
        if not blamed:
            shown = ', '.join(str(q) for q in qubits[:12])
            more = '' if len(qubits) <= 12 else f', +{len(qubits) - 12} more'
            blamed = [f'qubits {shown}{more}']
        detail = '; '.join(blamed[:6])
        if len(blamed) > 6:
            detail += f'; +{len(blamed) - 6} more'
        text += (f', skipped {len(skipped)} ({detail}) — '
                 f'those edges read high versus reported')
    return text


def collect_analysis_errors(expdata, depth=0):
    """Client-side analysis failures, which `status()` does NOT report.

    A job can be DONE while every fit callback raised: on 20 Sept 13:08 both
    phases reported DONE=12 and `errors()` was None at the top level, yet
    phase 2 produced zero results because each callback had thrown. Silence
    there is what turned a total analysis failure into six bland "no results
    returned" notices. Walks the tree because composite experiments record
    failures on the child that raised.
    """
    found = []
    if expdata is None or depth > 6:
        return found
    for attr in ('analysis_errors', 'errors'):
        getter = getattr(expdata, attr, None)
        if getter is None:
            continue
        try:
            value = getter() if callable(getter) else getter
        except Exception:
            continue
        if value:
            text = str(value).strip()
            if text and text.lower() not in ('none', '[]', '{}'):
                found.append(text)
    try:
        for child in list(expdata.child_data()):
            found.extend(collect_analysis_errors(child, depth + 1))
    except Exception:
        pass
    return found


def summarise_analysis_errors(errors, limit=3):
    """Collapse repeated callback failures into one readable notice."""
    if not errors:
        return None
    counts = {}
    for text in errors:
        head = text.splitlines()[-1][:200] if text else text
        counts[head] = counts.get(head, 0) + 1
    ordered = sorted(counts.items(), key=lambda kv: -kv[1])
    parts = [f'{n}x {head}' if n > 1 else head for head, n in ordered[:limit]]
    extra = ('' if len(ordered) <= limit
             else f' (+{len(ordered) - limit} other error(s))')
    return f'analysis failed: {"; ".join(parts)}{extra}'


def _job_ids_of(expdata):
    try:
        return [str(j) for j in (expdata.job_ids or [])]
    except Exception:
        return []


def _status_name(obj):
    try:
        status = obj.status()
    except Exception:
        return ''
    return getattr(status, 'name', None) or str(status)


def _check_chunk_status(expdata, chunk_index):
    """Raise `BenchmarkRunError` when the provider failed this chunk.

    This is the gap that produced the false success: a rejected submission
    left an errored `ExperimentData` whose `analysis_results()` was simply
    empty, and an empty parse looked exactly like a successful run that
    measured nothing.
    """
    job_ids = _job_ids_of(expdata)
    status = _status_name(expdata).upper()
    if status in ('ERROR', 'CANCELLED'):
        detail = ''
        try:
            errors = expdata.errors()
            if errors:
                detail = str(errors).strip()
        except Exception:
            pass
        raise BenchmarkRunError(
            f'chunk {chunk_index + 1}: job {status.lower()}'
            + (f' — {detail}' if detail else ''),
            job_ids, chunk_index)
    return job_ids


def make_runtime_runner(backend, on_jobs=None, sampler_factory=None):
    """The production runner: qiskit-experiments against a real backend.

    The session is opened as `Session(backend=backend)`. Older
    qiskit-ibm-runtime took a `service=` kwarg as well; 0.48 removed it,
    because a backend object already carries its own service. The call shape
    is pinned by tests/test_benchmark_runtime_contract.py.

    The plan is split into chunks that each fit IBM's per-job circuit-shot
    limit and submitted sequentially INSIDE ONE SESSION, so the whole run
    still costs one reservation. Each chunk's status is checked before its
    results are parsed; a failed chunk raises rather than yielding an empty
    parse that would look like a successful measurement of nothing.

    Args:
        on_jobs: optional ``callable(chunk_index, job_ids, status)`` invoked as
            soon as ids are known, so the panel can persist them for later
            reanalysis even if the process dies mid-run.
        sampler_factory: optional ``callable(session) -> sampler``, for the
            integration tests to substitute a local simulator without touching
            any other part of the path.
    """
    api = preload_runtime()

    def _runner(plan):
        # Two phases inside ONE session: everything that is not two-qubit RB
        # first, then the 2Q layers built with the 1Q EPGs the first phase
        # produced. The ordering is what makes the subtraction possible —
        # a StandardRB analysis cannot be told the single-qubit error after
        # it has been constructed and run.
        has_2q = any(sp.kind in ('rb2q_layer', 'rb2q') for sp in plan.specs)
        if has_2q:
            phase1 = [sp for sp in plan.specs
                      if sp.kind not in ('rb2q_layer', 'rb2q')]
            phase2 = [sp for sp in plan.specs
                      if sp.kind in ('rb2q_layer', 'rb2q')]
            if phase1:
                return _run_phased(plan, phase1, phase2)
        chunks = split_for_submission(plan, plan.shots)
        merged = {}
        notices = []
        collected = []
        spec_offset = 0

        with api.Session(backend=backend) as session:
            sampler = (sampler_factory(session) if sampler_factory is not None
                       else api.SamplerV2(mode=session))
            for chunk in chunks:
                experiments = [(spec, _build_experiment(spec, api))
                               for spec in chunk.specs]
                batch = api.BatchExperiment(
                    [exp for _s, exp in experiments], flatten_results=False)
                with warnings.catch_warnings():
                    # qiskit-experiments' RB analysis divides by a per-Clifford
                    # gate count that is zero for any basis gate the sequences
                    # never use, emitting one RuntimeWarning per such gate.
                    # It is library-side and harmless, but it floods the
                    # notebook. Scoped to that module only: our own warnings,
                    # and every other library's, still surface.
                    warnings.filterwarnings(
                        'ignore', category=RuntimeWarning,
                        module=r'.*qiskit_experiments.*rb_analysis.*')
                    data = batch.run(sampler=sampler, shots=plan.shots)
                job_ids = _job_ids_of(data)
                if on_jobs is not None:
                    try:
                        on_jobs(chunk.index, job_ids, 'submitted')
                    except Exception:
                        _log.debug('job-id callback failed', exc_info=True)
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        'ignore', category=RuntimeWarning,
                        module=r'.*qiskit_experiments.*rb_analysis.*')
                    data.block_for_results()

                try:
                    job_ids = _check_chunk_status(data, chunk.index)
                except BenchmarkRunError as exc:
                    if len(chunks) == 1:
                        exc.expdata = collected + [data]
                        raise
                    # Partial failure: keep what the other chunks produced and
                    # surface this one loudly rather than losing the whole run.
                    notices.append(
                        f'chunk {chunk.index + 1}/{len(chunks)} failed: {exc}')
                    spec_offset += len(chunk.specs)
                    continue
                if on_jobs is not None:
                    try:
                        on_jobs(chunk.index, job_ids, _status_name(data))
                    except Exception:
                        _log.debug('job-id callback failed', exc_info=True)

                for local_idx, (spec, _exp) in enumerate(experiments):
                    idx = spec_offset + local_idx
                    try:
                        merged[idx] = list(_iter_analysis_results(
                            data.child_data(local_idx)))
                    except Exception as exc:   # noqa: BLE001 - per-experiment
                        _log.warning('benchmark: no results for %s %s: %s',
                                     spec.kind, spec.qubits, exc)
                        merged[idx] = None
                collected.append(data)
                spec_offset += len(chunk.specs)

        if notices:
            merged['_notices'] = notices
        # The ExperimentData objects live on the runner, not in the result
        # mapping: `results_to_epoch` iterates that mapping by spec index, and
        # a stray key there would look like an extra experiment.
        _runner.last_expdata = collected
        return merged

    def _run_phased(plan, phase1, phase2, session=None, sampler=None):
        """Phase 1 (readout/T1/T2/1Q RB) then phase 2 (2Q RB), one session.

        Phase 1's per-qubit EPGs are handed to phase 2's analyses so the
        two-qubit numbers are 1Q-subtracted. If phase 1 yields no EPGs the run
        still proceeds — the values are then comparable only in relative
        terms, which `consistency_stats` detects and flags.
        """
        merged = {}
        collected = []
        context = {'epg_results': {}, 'epg_1q': {}, 'notices': []}
        index_of = {id(sp): i for i, sp in enumerate(plan.specs)}
        phase_no = [0]

        def _submit(specs, session, sampler):
            experiments = [(sp, _build_experiment(sp, api, context))
                           for sp in specs]
            batch = api.BatchExperiment([e for _s, e in experiments],
                                        flatten_results=False)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    'ignore', category=RuntimeWarning,
                    module=r'.*qiskit_experiments.*rb_analysis.*')
                data = batch.run(sampler=sampler, shots=plan.shots)
                job_ids = _job_ids_of(data)
                if on_jobs is not None:
                    try:
                        on_jobs(phase_no[0], job_ids, 'submitted')
                    except Exception:
                        _log.debug('job-id callback failed', exc_info=True)
                data.block_for_results()
            _check_chunk_status(data, phase_no[0])
            if on_jobs is not None:
                try:
                    on_jobs(phase_no[0], _job_ids_of(data),
                            _status_name(data))
                except Exception:
                    _log.debug('job-id callback failed', exc_info=True)

            # A DONE job can still have had every analysis callback raise.
            # Surfacing that is the difference between "the fits failed, here
            # is why" and six bland "no results returned" notices.
            failures = collect_analysis_errors(data)
            summary = summarise_analysis_errors(failures)
            if summary:
                context['notices'].append(f'phase {phase_no[0] + 1}: {summary}')
            phase_no[0] += 1
            collected.append(data)
            for local_idx, (sp, _exp) in enumerate(experiments):
                try:
                    merged[index_of[id(sp)]] = list(
                        _iter_analysis_results(data.child_data(local_idx)))
                except Exception as exc:      # noqa: BLE001 - per-experiment
                    _log.warning('benchmark: no results for %s: %s',
                                 sp.kind, exc)
                    merged[index_of[id(sp)]] = None
            return data

        import contextlib as _ctxlib
        # An already-open session is reused (the calibration check drives two
        # arms through ONE session so the whole check costs one reservation);
        # otherwise this opens its own.
        own_session = session is None
        ctx_mgr = (api.Session(backend=backend) if own_session
                   else _ctxlib.nullcontext(session))
        with ctx_mgr as session:
            if sampler is None:
                sampler = (sampler_factory(session)
                           if sampler_factory is not None
                           else api.SamplerV2(mode=session))
            _submit(phase1, session, sampler)

            # Harvest the 1Q EPG RESULT OBJECTS before building the
            # two-qubit experiments: the analysis needs the results, not
            # their values (see harvest_epg_results).
            for sp in phase1:
                if sp.kind != 'rb1q':
                    continue
                entry = merged.get(index_of[id(sp)]) or []
                harvest = harvest_epg_results(entry)
                context['epg_results'].update(harvest.results)
                context.setdefault('_epg_rejected', {}).update(
                    harvest.rejected)
                harvested = {}
                _collect_rb1q(entry, sp, [], harvested)
                context['epg_1q'].update(harvested.get('epg_1q', {}))

            if not context['epg_results']:
                context['notices'].append(
                    'No 1Q results \u2014 two-qubit errors are not '
                    '1Q-subtracted and read high')
            if phase2:
                _submit(phase2, session, sampler)

        tally = epg_tally_notice(context)
        if tally:
            context['notices'].append(tally)
        if context['notices']:
            merged['_notices'] = list(context['notices'])
        _runner.last_expdata = collected
        return merged

    def _run_in_session(plan, session, sampler):
        """Run one plan inside a caller-owned session. Both arms of the
        calibration check go through here, so each gets its own phase-1 1Q RB
        and therefore its own subtraction, while sharing one reservation."""
        phase1 = [sp for sp in plan.specs
                  if sp.kind not in ('rb2q_layer', 'rb2q')]
        phase2 = [sp for sp in plan.specs
                  if sp.kind in ('rb2q_layer', 'rb2q')]
        return _run_phased(plan, phase1, phase2, session, sampler)

    _runner.run_in_session = _run_in_session
    _runner.api = api
    _runner.last_expdata = []
    return _runner


def make_calibration_check_runner(backend, on_jobs=None,
                                  sampler_factory=None):
    """Run both calibration-check arms in ONE session.

    Returns `runner(plans) -> [raw_parallel, raw_sequential]`. Each arm is a
    full plan with its own phase-1 1Q RB, so the subtraction applies to both
    and the two numbers are directly comparable — which is the entire point:
    an estimator bias moves both arms, crosstalk moves only the parallel one.
    """
    inner = make_runtime_runner(backend, on_jobs, sampler_factory)
    api = inner.api

    def _run(plans):
        out = []
        collected = []
        with api.Session(backend=backend) as session:
            sampler = (sampler_factory(session) if sampler_factory is not None
                       else api.SamplerV2(mode=session))
            for plan in plans:
                out.append(inner.run_in_session(plan, session, sampler))
                collected.extend(getattr(inner, 'last_expdata', []) or [])
        _run.last_expdata = collected
        return out

    _run.last_expdata = []
    return _run


def _safe_analysis_results(expdata):
    """`analysis_results()` without tripping the deprecation.

    Leaving `dataframe` unset is deprecated as of qiskit-experiments 0.9, and
    under `-W error` that DeprecationWarning is RAISED. Both the parser and
    the raw dump guard their calls with `except Exception`, so the warning
    was being swallowed and reported as "no results returned" for every
    experiment — a healthy run looking exactly like the failed one. Ask for
    the DataFrame explicitly; older versions have no such keyword, hence the
    TypeError fallback.
    """
    if expdata is None:
        return None
    for attempt in ({'dataframe': True}, {}):
        try:
            return expdata.analysis_results(**attempt)
        except TypeError:
            continue
        except Exception:
            return None
    return None


def _iter_analysis_results(expdata):
    """Yield analysis results from either shape qiskit-experiments returns.

    `ExperimentData.analysis_results()` returns a list of `AnalysisResult`
    objects in older versions and a pandas DataFrame in newer ones (with
    `dataframe=True` becoming the default). Normalising here means the parser
    below never has to know which is installed — and a version bump cannot
    silently empty every epoch again.

    Recurses into `child_data` so a composite experiment's results are found
    whether they were flattened to the top or left nested.
    """
    if expdata is None:
        return

    # Ask for the DataFrame explicitly. Leaving `dataframe` unset is
    # deprecated as of qiskit-experiments 0.9, and under `-W error` that
    # DeprecationWarning is raised as an exception inside the per-experiment
    # guard below — which then reported a perfectly good run as "no results
    # returned" for every experiment. Older versions have no such keyword,
    # hence the TypeError fallback.
    results = _safe_analysis_results(expdata)

    for item in _normalise_results(results):
        yield item

    # Nested composites: look one level down too.
    try:
        children = list(expdata.child_data())
    except Exception:
        children = []
    for child in children:
        for item in _iter_analysis_results(child):
            yield item


def _normalise_results(results):
    """List / DataFrame / None → a flat list of result-like objects."""
    if results is None:
        return []
    # DataFrame: has .itertuples and .columns
    if hasattr(results, 'itertuples') and hasattr(results, 'columns'):
        out = []
        for row in results.itertuples(index=False):
            # The column is `components` in qiskit-experiments 0.10 and
            # `device_components` in the legacy object form. Reading only the
            # latter yielded None for every row, so nothing could be matched
            # to a qubit and every epoch came back empty.
            components = getattr(row, 'components', None)
            if components is None:
                components = getattr(row, 'device_components', None)
            out.append(SimpleNamespace(
                name=getattr(row, 'name', None),
                value=getattr(row, 'value', None),
                device_components=list(components or []),
                quality=getattr(row, 'quality', None)))
        return out
    if isinstance(results, (list, tuple)):
        return list(results)
    return [results]


def _components_to_qubits(item, fallback=()):
    """Physical qubit ids an analysis result refers to.

    `device_components` entries are `Qubit(3)`-style objects or their string
    form; both are handled, because which one appears depends on the version
    and on whether the results came back as a DataFrame.
    """
    raw = getattr(item, 'device_components', None)
    if not raw:
        raw = getattr(item, 'qubits', None)
    out = []
    for component in (raw or []):
        if isinstance(component, (int, np.integer)):
            out.append(int(component))
            continue
        # `Qubit(3)` exposes .index as an int. Careful: str also has an
        # `.index` ATTRIBUTE — the method — so the type check matters.
        index = getattr(component, 'index', None)
        if isinstance(index, (int, np.integer)):
            out.append(int(index))
            continue
        text = str(component)
        digits = ''.join(ch for ch in text if ch.isdigit())
        if digits:
            out.append(int(digits))
    return out or list(fallback)


# ── Results → epoch ─────────────────────────────────────────────────

def _summarise_failures(notices, label, failures, total):
    """One line per failure KIND, never a list of every index.

    The 20 Sept run emitted a notice naming all 156 qubit indices verbatim,
    which is unreadable and hides the one thing that matters — the error. Up
    to five failures are still named individually, because at that size the
    indices are the useful part.
    """
    if not failures:
        return
    if len(failures) <= 5:
        for element, reason in failures:
            notices.append(f'{label} q{element}: {reason}; kept reported value')
        return
    reasons = {}
    for _element, reason in failures:
        reasons[reason] = reasons.get(reason, 0) + 1
    common = max(reasons.items(), key=lambda kv: kv[1])[0]
    notices.append(f'{label}: {len(failures)}/{total} failed — {common}'
                   + ('' if len(reasons) == 1
                      else f' (+{len(reasons) - 1} other error(s))')
                   + '; kept reported values')


def _readout_error_from_matrix(matrix):
    """Symmetrised assignment error, as the tutorial computes it."""
    m = np.asarray(matrix, dtype=float)
    if m.shape != (2, 2):
        raise ValueError(f'assignment matrix must be 2x2, got {m.shape}')
    return float((m[0, 1] + m[1, 0]) / 2.0)


def _result_value(item):
    """Pull a scalar out of whatever shape the analysis result carries."""
    value = getattr(item, 'value', item)
    for attr in ('nominal_value', 'n', 'value'):
        inner = getattr(value, attr, None)
        if inner is not None and not callable(inner):
            value = inner
            break
    return float(value)


def _lookup(results, idx):
    entry = results.get(idx) if isinstance(results, dict) else None
    return entry


def results_to_epoch(raw, plan, reported_qubit_props, reported_gate_props,
                     edges_list, *, when=None):
    """Fold raw runner output into a full `MeasuredEpoch`.

    Starts from copies of the reported arrays, so any element the benchmark
    did not measure — or measured and failed to fit — keeps exactly the value
    it had. Only successfully-fitted elements are overwritten and marked in
    the masks. Every failure becomes a notice; none becomes an exception.

    Units follow the reported arrays: T1/T2 are stored in microseconds (see
    `data.fetch_qubit_prop`), while the experiments report seconds, so those
    are converted here. Readout and gate errors are dimensionless in both.
    """
    when = time.time() if when is None else when
    qubit_props = {k: np.array(v, dtype=float, copy=True)
                   for k, v in (reported_qubit_props or {}).items()}
    gate_props = {k: np.array(v, dtype=float, copy=True)
                  for k, v in (reported_gate_props or {}).items()}

    n_q = len(next(iter(qubit_props.values()), []))
    n_e = len(next(iter(gate_props.values()), []))
    q_mask = np.zeros(n_q, dtype=bool)
    e_mask = np.zeros(n_e, dtype=bool)
    notices = []
    extra = {}

    edge_index = {tuple(e): i for i, e in enumerate(edges_list or [])}
    two_q_key = _two_qubit_key(gate_props)

    for idx, spec in enumerate(plan.specs):
        entry = _lookup(raw, idx)
        # An empty sequence counts as "nothing came back" just as None does:
        # a runner that returns no analysis results must produce a visible
        # notice, never a silent no-op that looks like a successful run.
        if entry is None or (isinstance(entry, (list, tuple)) and not entry):
            notices.append(f'{spec.kind}: no results returned for '
                           f'{len(spec.qubits)} qubit(s), kept reported values')
            continue
        try:
            if spec.kind == 'readout':
                _apply_readout(entry, spec, qubit_props, q_mask, n_q, notices)
            elif spec.kind in ('t1', 't2hahn'):
                _apply_coherence(entry, spec, qubit_props, q_mask, n_q,
                                 notices)
            elif spec.kind == 'rb2q_layer':
                _apply_rb2q_layer(entry, spec, gate_props, e_mask, edge_index,
                                  two_q_key, notices)
            elif spec.kind == 'rb2q':         # legacy single-pair spec
                _apply_rb2q(entry, spec, gate_props, e_mask, edge_index,
                            two_q_key, notices)
            # rb1q is measured at Level 3 for the diagnostic view; it has no
            # single reported counterpart to overwrite (the reported arrays
            # carry per-gate sx/x/rz errors separately), so it is recorded in
            # the notices rather than written into an array.
            elif spec.kind == 'rb1q':
                _collect_rb1q(entry, spec, notices, extra)
        except Exception as exc:              # noqa: BLE001 - never fatal
            _log.warning("benchmark: folding %s %s failed: %s",
                         spec.kind, spec.qubits, exc, exc_info=True)
            notices.append(f'{spec.kind} {list(spec.qubits)}: '
                           f'{exc}; kept reported values')

    run_notices = (raw.get('_notices') if isinstance(raw, dict) else None) or []
    for note in run_notices:
        notices.append(str(note))

    planned = plan_planned_count(plan)
    _assert_attribution(raw, plan, q_mask, e_mask, notices)
    return MeasuredEpoch(qubit_props, gate_props, q_mask, e_mask,
                         plan.level, when, notices, planned_count=planned,
                         extra=extra)


def _assert_attribution(raw, plan, q_mask, e_mask, notices):
    """Never return "nothing measured" when results were in fact present.

    The failure this guards against: the analysis produced 348 EPG_cz plus
    T1/T2/readout for 156 qubits, and the epoch still came back (0, 0) with
    an EMPTY notice list — so the UI reported "no usable measurements" with
    nothing to explain it, while the raw dump held everything. Silence there
    is worse than a wrong number, because it looks like an answer.

    If any analysis result was seen but none was attributed to a target, say
    how many and name the most likely reason.
    """
    found = 0
    for key, entry in (raw or {}).items():
        if not isinstance(key, int):
            continue
        if isinstance(entry, (list, tuple)):
            found += len(entry)
        elif entry is not None:
            found += 1
    if not found:
        return
    if q_mask.any() or e_mask.any():
        return

    if not plan.specs:
        reason = 'the plan being folded has no experiments'
    elif not plan.targets.qubits and not plan.targets.edges:
        reason = 'the plan has no targets to attribute results to'
    else:
        _log.warning(
            'attribution failed: no result qubit/edge matched the plan '
            'targets. Check the edge orientation against state.edges_list '
            'and that the plan matches the run that produced these results.')
        reason = 'plan/run mismatch'
    notices.append(f'Results found but none matched the targets — '
                   f'{reason}')


def _two_qubit_key(gate_props):
    """Which gate array the 2Q RB result should overwrite."""
    for key in ('cz_error', 'rzz_error'):
        if key in gate_props:
            return key
    return None


def _apply_readout(entry, spec, qubit_props, q_mask, n_q, notices):
    arr = qubit_props.get('readout_error')
    if arr is None:
        notices.append('readout: no reported readout_error array to update')
        return
    matrices = _readout_matrices(entry, spec)
    if not matrices:
        notices.append(f'readout: no assignment matrix for '
                       f'{len(spec.qubits)} qubit(s), kept reported values')
        return
    failures = []
    for qubit, matrix in matrices.items():
        if not (0 <= qubit < n_q):
            continue
        try:
            value = _readout_error_from_matrix(matrix)
            if not np.isfinite(value) or not (0.0 <= value <= _READOUT_MAX):
                raise ValueError(
                    f'implausible readout error {value:.3g} '
                    f'(expected 0-{_READOUT_MAX:g})')
            arr[qubit] = value
            q_mask[qubit] = True
        except Exception as exc:              # noqa: BLE001 - per element
            failures.append((qubit, str(exc)))
    _summarise_failures(notices, 'readout', failures, len(matrices))


def _readout_matrices(entry, spec):
    """Map physical qubit → 2x2 assignment matrix from a readout result.

    `LocalReadoutError` returns ONE analysis result for the whole target set,
    whose value is a `LocalReadoutMitigator`. Its `assignment_matrix` is a
    METHOD taking the qubits to report on — reading it as an attribute yields
    the bound method itself, and `float()` on that raised

        float() argument must be a string or a real number, not 'method'

    for all 156 qubits on the 20 Sept run. The same class of bug as reading
    `str.index` as an integer. Attributes are tried too, since older versions
    and some stand-ins expose a plain array.
    """
    out = {}
    items = entry if isinstance(entry, (list, tuple)) else [entry]
    for item in items:
        value = getattr(item, 'value', item)
        qubits = _components_to_qubits(item, fallback=spec.qubits)
        accessor = getattr(value, 'assignment_matrix', None)
        if accessor is None and isinstance(value, dict):
            accessor = value.get('assignment_matrix')

        if callable(accessor):
            # The live shape: one mitigator, queried per qubit.
            for q in qubits:
                try:
                    matrix = np.asarray(accessor([int(q)]), dtype=float)
                except Exception:
                    try:
                        matrix = np.asarray(accessor(int(q)), dtype=float)
                    except Exception:
                        continue
                if matrix.shape == (2, 2):
                    out[int(q)] = matrix
            continue

        if accessor is None:
            continue
        matrix = np.asarray(accessor, dtype=float)
        if matrix.shape == (2, 2) and len(qubits) == 1:
            out[int(qubits[0])] = matrix
        elif matrix.ndim == 3:                # one 2x2 per qubit, stacked
            for pos, q in enumerate(qubits):
                if pos < matrix.shape[0]:
                    out[int(q)] = matrix[pos]
    return out


def _apply_coherence(entry, spec, qubit_props, q_mask, n_q, notices):
    """T1/T2 in seconds from the experiment → microseconds in the arrays."""
    key = 't1' if spec.kind == 't1' else 't2'
    arr = qubit_props.get(key)
    if arr is None:
        notices.append(f'{key}: no reported {key} array to update')
        return
    items = entry if isinstance(entry, (list, tuple)) else [entry]
    if not items:
        notices.append(f'{key.upper()}: no results for {len(spec.qubits)} '
                       f'qubit(s), kept reported values')
        return
    failures = []
    for item in items:
        # Identify the qubit from device_components. Falling back to
        # spec.qubits[0] wrote all 156 of the 20 Sept T1 results onto qubit 0,
        # which is why that run reported "measured qubits: 1".
        qubits = _components_to_qubits(item)
        if not qubits:
            continue
        qubit = int(qubits[0])
        if not (0 <= qubit < n_q):
            continue
        try:
            seconds = _result_value(item)
            micros = seconds * 1e6            # data.py stores µs
            if not np.isfinite(micros) or not (0 < micros < _COHERENCE_MAX_US):
                raise ValueError(
                    f'implausible {key} {micros:.3g} us '
                    f'(expected 0-{_COHERENCE_MAX_US:g})')
            arr[qubit] = micros
            q_mask[qubit] = True
        except Exception as exc:              # noqa: BLE001 - per element
            failures.append((qubit, str(exc)))
    _summarise_failures(notices, key.upper(), failures, len(items))


def _apply_rb2q(entry, spec, gate_props, e_mask, edge_index, two_q_key,
                notices):
    """Two-qubit EPG from RB → the backend's 2Q error array for that edge."""
    pair = tuple(spec.qubits)
    if two_q_key is None:
        notices.append(f'{list(pair)}: no two-qubit error array to update')
        return
    arr = gate_props.get(two_q_key)
    idx = edge_index.get(pair)
    if idx is None:
        idx = edge_index.get((pair[1], pair[0]))
    if idx is None or arr is None or not (0 <= idx < len(arr)):
        notices.append(f'{list(pair)}: edge not in the device map, skipped')
        return
    try:
        epg = _epg_from_rb(entry)
        if not np.isfinite(epg) or epg < 0:
            raise ValueError(f'implausible EPG={epg}')
        arr[idx] = epg
        e_mask[idx] = True
    except Exception as exc:                  # noqa: BLE001 - per element
        notices.append(f'{list(pair)}: RB fit failed ({exc}), '
                       f'kept reported value')


def _apply_rb2q_layer(entry, spec, gate_props, e_mask, edge_index, two_q_key,
                      notices):
    """Fold ONE layer's parallel 2Q RB: many pairs, one experiment.

    Each analysis result names its own qubits through `device_components`, so
    results are matched to edges by those components rather than by position —
    a ParallelExperiment gives no ordering guarantee worth relying on.
    """
    pairs = [tuple(p) for p in (spec.params.get('pairs') or [])]
    if two_q_key is None:
        notices.append(f'{spec.kind}: no two-qubit error array to update')
        return
    arr = gate_props.get(two_q_key)
    if arr is None:
        return

    items = entry if isinstance(entry, (list, tuple)) else [entry]
    by_pair = {}
    for item in items:
        name = str(getattr(item, 'name', '') or '')
        if not name.upper().startswith(RESULT_EPG_PREFIXES):
            continue
        qubits = tuple(sorted(_components_to_qubits(item)))
        if len(qubits) != 2:
            continue
        by_pair.setdefault(qubits, item)

    for pair in pairs:
        item = by_pair.get(tuple(sorted(pair)))
        # `a or b` is wrong here: edge index 0 is falsy, so the first edge
        # of every device silently fell through to the reversed lookup and
        # was reported "not in the device map". Test the sentinel explicitly.
        idx = edge_index.get(pair)
        if idx is None:
            idx = edge_index.get((pair[1], pair[0]))
        if idx is None or not (0 <= idx < len(arr)):
            notices.append(f'{list(pair)}: edge not in the device map, skipped')
            continue
        if item is None:
            notices.append(f'{list(pair)}: no RB result in the layer, '
                           f'kept reported value')
            continue
        try:
            epg = _result_value(item)
            if not np.isfinite(epg) or epg < 0:
                raise ValueError(f'implausible EPG={epg}')
            arr[idx] = epg
            e_mask[idx] = True
        except Exception as exc:              # noqa: BLE001 - per element
            notices.append(f'{list(pair)}: RB fit failed ({exc}), '
                           f'kept reported value')


def _epg_from_rb(entry):
    """Error-per-gate out of an RB analysis result set."""
    items = entry if isinstance(entry, (list, tuple)) else [entry]
    for item in items:
        name = getattr(item, 'name', None)
        if name and str(name).upper().startswith('EPG'):
            return _result_value(item)
    for item in items:                        # fall back to EPC
        name = getattr(item, 'name', None)
        if name and str(name).upper().startswith('EPC'):
            return _result_value(item)
    raise ValueError('no EPG/EPC in RB results')


def _collect_rb1q(entry, spec, notices, extra):
    """Per-qubit 1Q EPGs, and a notice that counts QUBITS.

    `StandardRB` reports one result per (qubit, basis gate) — rz, sx, x — so
    the 20 Sept run saw 468 results for 156 qubits and announced "median EPG
    over 624 qubit(s)". The per-gate values are averaged per qubit here and
    kept in `epoch.extra['epg_1q']`, which is what the 2Q RB analysis needs to
    subtract single-qubit error from the Clifford decay (see Part B).

    These are NOT counted as measured qubit properties: the reported arrays
    carry per-gate errors separately, and overwriting one of them with an RB
    average would mix two estimators in the same column.
    """
    items = entry if isinstance(entry, (list, tuple)) else [entry]
    per_qubit = {}
    for item in items:
        name = str(getattr(item, 'name', '') or '')
        if not name.upper().startswith('EPG'):
            continue
        qubits = _components_to_qubits(item)
        if len(qubits) != 1:
            continue
        try:
            value = _result_value(item)
        except Exception:
            continue
        if np.isfinite(value) and value >= 0:
            per_qubit.setdefault(int(qubits[0]), []).append(float(value))

    if not per_qubit:
        notices.append(f'1Q RB: no usable results for {len(spec.qubits)} '
                       f'qubit(s)')
        return

    averaged = {q: float(np.mean(v)) for q, v in per_qubit.items()}
    # Same bound as the forwarding path: an implausible EPG must not be
    # recorded as if it were a measurement.
    implausible = {q: v for q, v in averaged.items() if not _plausible_epg(v)}
    averaged = {q: v for q, v in averaged.items() if _plausible_epg(v)}
    if implausible:
        notices.append(
            f'1Q RB: {len(implausible)} qubit(s) returned an implausible EPG '
            f'(>= {_EPG_1Q_MAX:g}) and were dropped: '
            + ', '.join(f'q{q}={v:.2g}' for q, v in
                        sorted(implausible.items())[:6]))
    if not averaged:
        notices.append(f'1Q RB: no plausible results for {len(spec.qubits)} '
                       f'qubit(s)')
        return
    extra.setdefault('epg_1q', {}).update(averaged)
    notices.append(
        f'1Q RB: median EPG {np.median(list(averaged.values())):.2e} over '
        f'{len(averaged)} qubit(s)')


# ── Consistency against the reported calibration ───────────────────
#
# The 20 Sept run measured 351 edges at a median 4.03e-3 against a reported
# median 1.43e-3: 2.8x worse across the board, 238 edges more than 2x worse,
# only 4 better. Real drift does not move an entire chip uniformly in one
# direction — that signature is an ESTIMATOR MISMATCH. Believing those
# absolute numbers would have made every layout look bad by the same factor,
# which changes no ranking but corrupts every fidelity estimate shown.
#
# So a measured epoch is now checked against the reported one before it is
# trusted, and when the medians disagree badly the values are rescaled to the
# reported median. Rescaling preserves every per-element RATIO, which is where
# the real signal lives: on 20 Sept the outliers (edge 190: 6.6e-2 -> 0.75, a
# collapsed coupler) were genuine and are exactly what the feature is for.

#: Median ratio outside this band means "do not trust the absolute values".
CONSISTENCY_BAND = (0.5, 2.0)


def _median_ratio_stats(measured, reported, mask):
    measured = np.asarray(measured, dtype=float)
    reported = np.asarray(reported, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    n = min(len(measured), len(reported), len(mask))
    sel = (mask[:n] & np.isfinite(measured[:n]) & np.isfinite(reported[:n])
           & (reported[:n] > 0))
    if not sel.any():
        return None
    m, r = measured[:n][sel], reported[:n][sel]
    # Nothing was actually written to this property: the mask marks elements
    # measured for SOME property, and a device carrying both cz_error and
    # rzz_error will have one of them untouched and identical to reported.
    # Reporting a x1.00 ratio for it is noise that hides the real offender.
    if np.allclose(m, r, rtol=0, atol=0):
        return None
    ratios = m / r
    idx = np.where(sel)[0]
    worst = idx[np.argsort(-ratios)][:5]
    return {
        'n': int(sel.sum()),
        'median_reported': float(np.median(r)),
        'median_measured': float(np.median(m)),
        # Two different statistics, and they differ on skewed data (20 Sept:
        # 2.82 vs 2.43). `ratio_of_medians` is what rescaling corrects, so it
        # is the one the policy uses; `median_ratio` describes the typical
        # element. Both are reported because quoting one as "the" ratio has
        # already caused confusion.
        'ratio_of_medians': float(np.median(m) / np.median(r)),
        'median_ratio': float(np.median(ratios)),
        'frac_worse_2x': float((ratios > 2).mean()),
        'frac_better_2x': float((ratios < 0.5).mean()),
        'n_worse_2x': int((ratios > 2).sum()),
        'n_better_2x': int((ratios < 0.5).sum()),
        'top_degradations': [
            {'index': int(i), 'reported': float(reported[i]),
             'measured': float(measured[i]),
             'ratio': float(measured[i] / reported[i])}
            for i in worst],
    }


def consistency_stats(epoch, reported_qubit_props, reported_gate_props):
    """Compare a measured epoch with the reported calibration, per property.

    Only measured elements are compared — an unmeasured one still holds the
    reported value, so including it would dilute the ratio towards 1 and hide
    exactly the offset this is looking for.
    """
    out = {}
    for key in ('readout_error', 't1', 't2'):
        if key in epoch.qubit_props and key in (reported_qubit_props or {}):
            stats = _median_ratio_stats(epoch.qubit_props[key],
                                        reported_qubit_props[key],
                                        epoch.measured_qubit_mask)
            if stats:
                out[key] = stats
    for key in ('cz_error', 'rzz_error'):
        if key in epoch.gate_props and key in (reported_gate_props or {}):
            stats = _median_ratio_stats(epoch.gate_props[key],
                                        reported_gate_props[key],
                                        epoch.measured_edge_mask)
            if stats:
                out[key] = stats
    return out


def consistency_offenders(stats, band=CONSISTENCY_BAND):
    """Properties whose ratio-of-medians falls outside the trust band."""
    low, high = band
    return {k: v for k, v in (stats or {}).items()
            if not (low <= v['ratio_of_medians'] <= high)}


def consistency_summary(stats, band=CONSISTENCY_BAND):
    """One line per property, for the panel."""
    parts = []
    for key, v in (stats or {}).items():
        flag = '' if (band[0] <= v['ratio_of_medians'] <= band[1]) else ' \u26a0'
        parts.append(f"{key}: x{v['ratio_of_medians']:.2f} median vs reported "
                     f"· {v['n_worse_2x']} of {v['n']} >2x worse{flag}")
    return ' · '.join(parts)


def apply_relative_mode(epoch, reported_qubit_props, reported_gate_props,
                        stats=None, band=CONSISTENCY_BAND):
    """Rescale offending properties so their median matches the reported one.

    Each measured value is multiplied by (reported median / measured median)
    for its property. That leaves every per-element ratio untouched — the
    collapsed coupler stays 11x worse than its neighbours — while removing the
    uniform offset that makes the absolute numbers unusable. Returns the
    number of properties rescaled; mutates `epoch` in place and sets
    `epoch.relative`.
    """
    stats = stats or consistency_stats(epoch, reported_qubit_props,
                                       reported_gate_props)
    offenders = consistency_offenders(stats, band)
    if not offenders:
        return 0

    rescaled = 0
    for key, v in offenders.items():
        factor = v['median_reported'] / v['median_measured'] \
            if v['median_measured'] else None
        if not factor or not np.isfinite(factor):
            continue
        if key in epoch.qubit_props:
            arr = epoch.qubit_props[key]
            mask = epoch.measured_qubit_mask[:len(arr)]
        elif key in epoch.gate_props:
            arr = epoch.gate_props[key]
            mask = epoch.measured_edge_mask[:len(arr)]
        else:
            continue
        arr[:len(mask)][mask] = arr[:len(mask)][mask] * factor
        rescaled += 1
        epoch.notices.append(
            f'{key}: rescaled by x{factor:.3f} to the reported median '
            f'(relative mode; per-element ratios preserved)')
    if rescaled:
        epoch.relative = True
    return rescaled


# ── Calibration check: estimator bias vs crosstalk ─────────────────

def build_calibration_check_plans(targets, n_qubits, coupling_edges,
                                  n_edges=6):
    """Two plans over the same few edges: one parallel, one sequential.

    The 20 Sept offset has two plausible causes that a single run cannot
    distinguish. An estimator mismatch (the 1Q EPGs not being subtracted from
    the Clifford decay) inflates every edge whether or not its neighbours are
    being driven. Crosstalk from running many pairs simultaneously only
    inflates the parallel case. Measuring the same edges both ways separates
    them: if both come back high, it is the estimator; if only the parallel
    run is high, it is crosstalk.

    Returns (parallel_plan, sequential_plan), each ~30 circuits plus the 1Q RB
    they share.
    """
    edges = [tuple(e) for e in (targets.edges or [])][:max(1, int(n_edges))]
    qubits = sorted({q for e in edges for q in e})
    subset = Targets(qubits, edges)

    parallel = build_plan(LEVEL_TARGETED, subset, n_qubits, coupling_edges)

    # Sequential: one pair per layer, so nothing runs alongside anything else.
    seq_specs = []
    for spec in parallel.specs:
        if spec.kind != 'rb2q_layer':
            seq_specs.append(spec)
            continue
        for i, pair in enumerate(spec.params['pairs']):
            params = dict(spec.params)
            params['pairs'] = [tuple(pair)]
            seq_specs.append(ExperimentSpec('rb2q_layer', tuple(pair), params,
                                            len(seq_specs) + i))
    sequential = parallel._replace(specs=seq_specs)
    return parallel, sequential


#: Marks a submission record as a calibration check rather than a benchmark.
RECORD_KIND_BENCHMARK = 'benchmark'
RECORD_KIND_CALIBRATION_CHECK = 'calibration_check'

#: A calibration check needs enough edges for the parallel arm to mean
#: something; below this the two arms are nearly the same experiment.
CALIBRATION_CHECK_EDGES = 6


def calibration_check_table_html(table, esc=str):
    """The verdict table, as the panel renders it under the status."""
    if not table or not table.get('rows'):
        return ("<div style='font-size:13px;color:#e74c3c;'>"
                "Calibration check produced no comparable edges.</div>")

    def _ratio(value):
        return '—' if value is None else f'×{value:.2f}'

    rows = ''.join(
        f"<tr><td style='padding:1px 8px;'>{r['edge'][0]}–{r['edge'][1]}</td>"
        f"<td style='padding:1px 8px;text-align:right;'>{r['reported']:.3e}</td>"
        f"<td style='padding:1px 8px;text-align:right;'>{r['parallel']:.3e}</td>"
        f"<td style='padding:1px 8px;text-align:right;'>"
        f"{r['sequential']:.3e}</td></tr>"
        for r in table['rows'])
    return (
        "<div style='font-size:13px;color:#1a5276;background:#f4f8fb;"
        "padding:6px 10px;border-radius:4px;border:1px solid #d6e4f0;"
        "margin:3px 0;'>"
        "<b>🔬 Calibration check</b><br>"
        "<table style='font-size:12px;border-collapse:collapse;margin:4px 0;'>"
        "<tr><th style='padding:1px 8px;text-align:left;'>edge</th>"
        "<th style='padding:1px 8px;'>reported</th>"
        "<th style='padding:1px 8px;'>parallel</th>"
        "<th style='padding:1px 8px;'>sequential</th></tr>"
        f"{rows}</table>"
        f"median vs reported — parallel {_ratio(table['ratio_parallel'])}, "
        f"sequential {_ratio(table['ratio_sequential'])}<br>"
        f"<b>{esc(table['verdict'])}</b></div>")


def calibration_check_table(edges, reported, parallel_epoch,
                            sequential_epoch, gate_key='cz_error'):
    """Rows of (edge, reported, parallel, sequential) plus the two ratios."""
    rows = []
    par = parallel_epoch.gate_props.get(gate_key)
    seq = sequential_epoch.gate_props.get(gate_key)
    rep = np.asarray(reported, dtype=float)
    index = {tuple(sorted(e)): i for i, e in enumerate(edges)}
    # ONLY the edges this check actually measured. An unmeasured edge still
    # holds its reported value, so including the rest of the device dragged
    # both medians back to the reported one and every check read "x1.00,
    # consistent" however biased the measured edges were.
    par_mask = np.asarray(parallel_epoch.measured_edge_mask, dtype=bool)
    seq_mask = np.asarray(sequential_epoch.measured_edge_mask, dtype=bool)
    par_vals, seq_vals, rep_vals = [], [], []
    for edge, i in sorted(index.items(), key=lambda kv: kv[1]):
        if par is None or seq is None or i >= len(rep):
            continue
        if i >= len(par_mask) or i >= len(seq_mask):
            continue
        if not (par_mask[i] and seq_mask[i]):
            continue
        rows.append({'edge': edge, 'reported': float(rep[i]),
                     'parallel': float(par[i]), 'sequential': float(seq[i])})
        rep_vals.append(rep[i])
        par_vals.append(par[i])
        seq_vals.append(seq[i])

    def _ratio(values):
        if not values or not rep_vals or np.median(rep_vals) == 0:
            return None
        return float(np.median(values) / np.median(rep_vals))

    return {
        'rows': rows,
        'ratio_parallel': _ratio(par_vals),
        'ratio_sequential': _ratio(seq_vals),
        'verdict': _check_verdict(_ratio(par_vals), _ratio(seq_vals)),
    }


def _check_verdict(ratio_parallel, ratio_sequential):
    """Read the two ratios into a one-line conclusion."""
    if ratio_parallel is None or ratio_sequential is None:
        return 'inconclusive (a run produced no usable values)'
    low, high = CONSISTENCY_BAND
    par_off = not (low <= ratio_parallel <= high)
    seq_off = not (low <= ratio_sequential <= high)
    if par_off and seq_off:
        return ('Estimator bias: both arms offset \u2014 single-qubit '
                'error is being counted as two-qubit error')
    if par_off and not seq_off:
        return ('crosstalk: only the parallel run is offset, so driving '
                'neighbouring pairs together is inflating the error')
    if seq_off and not par_off:
        return 'unexpected: the sequential run is offset but the parallel one is not'
    return 'consistent: both runs agree with the reported calibration'


# ── Applying an epoch to the session ────────────────────────────────

def apply_measured_epoch(state, epoch):
    """Install `epoch` as the session's active calibration. True on success.

    Mirrors `data.apply_history_snapshot`: same shape check, same arrays, same
    `delta_mode` reset (a stale delta would render against the previous
    epoch). Also attaches the epoch itself as `state._measured_epoch` so the
    UI can badge measured elements — see the module docstring on why that is a
    runtime attribute rather than a `SessionState` field.
    """
    if epoch is None:
        return False
    n_q = len(state.qubit_ids) if state.qubit_ids else 0
    n_e = len(state.edges_list) if state.edges_list else 0
    qp, gp = epoch.qubit_props, epoch.gate_props
    if len(next(iter(qp.values()), [])) != n_q or \
       len(next(iter(gp.values()), [])) != n_e:
        return False

    state.qubit_props_data = qp
    state.gate_props_data = gp
    mode = ', relative' if getattr(epoch, 'relative', False) else ''
    state.calibration_data_label = (
        f'{state.active_backend_name} — measured {epoch.time_label()} '
        f'(L{epoch.level}{mode}, {epoch.n_measured_qubits}q/'
        f'{epoch.n_measured_edges}e)')
    state.delta_mode = False
    setattr(state, '_measured_epoch', epoch)
    return True


def measured_badge(state, *, qubit=None, edge=None):
    """`" (measured)"` when this element came from a benchmark, else `""`.

    Hover-text helper for the UI. Safe to call on a session that has never
    benchmarked, which is the common case.
    """
    epoch = getattr(state, '_measured_epoch', None)
    if epoch is None:
        return ''
    try:
        if qubit is not None:
            if 0 <= int(qubit) < len(epoch.measured_qubit_mask) \
                    and epoch.measured_qubit_mask[int(qubit)]:
                return ' (measured)'
        if edge is not None:
            if 0 <= int(edge) < len(epoch.measured_edge_mask) \
                    and epoch.measured_edge_mask[int(edge)]:
                return ' (measured)'
    except (TypeError, ValueError):
        return ''
    return ''


# ── Persistence ─────────────────────────────────────────────────────

_MEASURED_PREFIX = 'measured_'


def _measured_dir(backend_name):
    from .data import _backend_cache_dir
    return _backend_cache_dir(backend_name)


def _measured_path(backend_name, when):
    stamp = time.strftime('%Y%m%dT%H%M%S', time.localtime(when))
    return os.path.join(_measured_dir(backend_name),
                        f'{_MEASURED_PREFIX}{stamp}.json')


def _jsonable_consistency(stats):
    out = {}
    for key, value in (stats or {}).items():
        out[key] = {k: (list(v) if isinstance(v, (list, tuple)) else v)
                    for k, v in (value or {}).items()}
    return out


def _jsonable_extra(extra):
    out = {}
    for key, value in (extra or {}).items():
        if isinstance(value, dict):
            out[key] = {str(k): (float(v) if isinstance(v, (int, float))
                                 else v)
                        for k, v in value.items()}
        else:
            out[key] = value
    return out


def save_measured_epoch(backend_name, epoch):
    """Write `epoch` beside the reported calibrations. Path, or None.

    Same atomic temp-file-then-rename dance `data._save_calibration` uses, and
    the same array-of-lists JSON shape plus the benchmark-specific level,
    masks and notices. A save failure is logged and swallowed: losing the
    on-disk copy must not lose the epoch the user just paid for, which is
    already live on `state`.
    """
    try:
        path = _measured_path(backend_name, epoch.when)
        payload = {
            'when': epoch.when,
            'level': epoch.level,
            'notices': list(epoch.notices),
            # Persisted so a saved epoch can be read back for what it IS:
            # without these, a rescaled epoch reloaded as `relative=None`
            # and looked like raw measured values.
            'relative': bool(getattr(epoch, 'relative', False)),
            'planned_count': list(getattr(epoch, 'planned_count', (0, 0))),
            'consistency': _jsonable_consistency(
                getattr(epoch, 'consistency', None)),
            'extra': _jsonable_extra(getattr(epoch, 'extra', None)),
            'qubit_props': {k: np.asarray(v).tolist()
                            for k, v in epoch.qubit_props.items()},
            'gate_props': {k: np.asarray(v).tolist()
                           for k, v in epoch.gate_props.items()},
            'measured_qubit_mask': epoch.measured_qubit_mask.tolist(),
            'measured_edge_mask': epoch.measured_edge_mask.tolist(),
        }
        dirn = _measured_dir(backend_name)
        fd, tmp = tempfile.mkstemp(dir=dirn, suffix='.tmp')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as fh:
                _json.dump(payload, fh)
            os.replace(tmp, path)
        except Exception:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise
        return path
    except Exception as exc:                  # noqa: BLE001 - never fatal
        _log.warning("Failed to save measured epoch for %s: %s",
                     backend_name, exc)
        return None


def list_measured_epochs(backend_name):
    """Saved epoch timestamps for a backend, newest first."""
    try:
        dirn = _measured_dir(backend_name)
        out = []
        for name in os.listdir(dirn):
            if not (name.startswith(_MEASURED_PREFIX)
                    and name.endswith('.json')):
                continue
            try:
                with open(os.path.join(dirn, name), 'r', encoding='utf-8') as f:
                    out.append(float(_json.load(f).get('when')))
            except Exception:
                _log.debug("skipping unreadable measured epoch %s", name)
        return sorted(out, reverse=True)
    except Exception:                         # pragma: no cover - defensive
        return []


def load_measured_epoch(backend_name, when):
    """Read back a saved epoch by its `when`, or None."""
    try:
        path = _measured_path(backend_name, when)
        if not os.path.exists(path):
            return None
        with open(path, 'r', encoding='utf-8') as fh:
            data = _json.load(fh)
        return MeasuredEpoch(
            {k: np.array(v, dtype=float)
             for k, v in data['qubit_props'].items()},
            {k: np.array(v, dtype=float)
             for k, v in data['gate_props'].items()},
            data['measured_qubit_mask'], data['measured_edge_mask'],
            data['level'], float(data['when']), data.get('notices'),
            planned_count=tuple(data.get('planned_count') or (0, 0)),
            extra={k: {int(q): float(x) for q, x in v.items()}
                   if k == 'epg_1q' and isinstance(v, dict) else v
                   for k, v in (data.get('extra') or {}).items()},
            consistency=data.get('consistency'),
            relative=bool(data.get('relative', False)))
    except Exception as exc:                  # noqa: BLE001 - corrupt file
        _log.warning("Failed to load measured epoch %s/%s: %s",
                     backend_name, when, exc)
        return None


def _stamp_for(when):
    return time.strftime('%Y%m%dT%H%M%S', time.localtime(when))


def jobs_path(backend_name, when):
    return os.path.join(_measured_dir(backend_name),
                        f'{_MEASURED_PREFIX}{_stamp_for(when)}_jobs.json')


def raw_path(backend_name, when):
    return os.path.join(_measured_dir(backend_name),
                        f'{_MEASURED_PREFIX}{_stamp_for(when)}_raw.json')


def _write_json(path, payload):
    try:
        dirn = os.path.dirname(path)
        fd, tmp = tempfile.mkstemp(dir=dirn, suffix='.tmp')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as fh:
                _json.dump(payload, fh, default=str)
            os.replace(tmp, path)
        except Exception:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise
        return path
    except Exception as exc:                  # noqa: BLE001 - diagnostic only
        _log.warning('Failed to write %s: %s', path, exc)
        return None


def save_job_record(backend_name, when, plan, shots=None, session_id=None,
                    chunks=None, jobs=None, statuses=None,
                    kind=RECORD_KIND_BENCHMARK, arms=None):
    """Persist what was submitted, as soon as it is known.

    Written at submission rather than at completion, because the run that
    motivated this file never completed: the job was rejected and there was no
    record of what had been sent. With this, a failed or interrupted run can
    still be diagnosed — and, via `reanalyze_from_jobs`, recovered without
    spending QPU time again.
    """
    shots = shots or plan.shots
    payload = {
        'when': when,
        # 'benchmark' or 'calibration_check'. The latter measures the
        # INSTRUMENT, not the device, so it never yields an epoch — and
        # Reanalyze needs to know which it is replaying.
        'kind': kind,
        'arms': arms or [],
        'level': plan.level,
        'shots': shots,
        'circuits': plan_circuit_count(plan),
        'total_shots': plan_shots(plan, shots),
        'session_id': session_id,
        'targets': {'qubits': list(plan.targets.qubits),
                    'edges': [list(e) for e in plan.targets.edges]},
        'specs': [{'kind': s.kind, 'qubits': list(s.qubits),
                   'layer': s.layer, 'params': _jsonable(s.params)}
                  for s in plan.specs],
        'chunks': [{'index': c.index, 'circuits': c.circuits,
                    'shots': c.shots,
                    'kinds': [s.kind for s in c.specs]}
                   for c in (chunks or split_for_submission(plan, shots))],
        'jobs': dict(jobs or {}),
        'statuses': dict(statuses or {}),
    }
    return _write_json(jobs_path(backend_name, when), payload)


def _jsonable(params):
    out = {}
    for key, value in (params or {}).items():
        if isinstance(value, (list, tuple)):
            out[key] = [list(v) if isinstance(v, tuple) else v for v in value]
        else:
            out[key] = value
    return out


def latest_job_record(backend_name):
    """The most recent replayable run for a backend, or None.

    Deliberately NOT "the newest _jobs.json". Records were only written from
    v3.1 onward, and only the non-phased runner wrote them before that — so
    the newest record on disk could be an OLD run while a newer one had left
    only a raw dump. That is what happened on 20 Sept: the panel kept
    replaying the 12:38 single-batch run (one job, no 1Q phase, hence no
    subtraction and a cz_error of x2.85) while the 13:08 two-phase run sat
    beside it, reachable only through its raw dump.

    So: consider every run that has EITHER a job record or a raw dump, take
    the newest by timestamp, and recover the per-phase job ids from the dump
    when the record is absent or single-phase.
    """
    try:
        dirn = _measured_dir(backend_name)
        names = os.listdir(dirn)
    except Exception:
        _log.debug('no cache dir for %s', backend_name, exc_info=True)
        return None

    runs = {}
    for name in names:
        if not name.startswith(_MEASURED_PREFIX):
            continue
        if name.endswith('_jobs.json'):
            stamp = name[len(_MEASURED_PREFIX):-len('_jobs.json')]
            runs.setdefault(stamp, {})['jobs_file'] = name
        elif name.endswith('_rean.json'):
            continue                          # a replay, not a submission
        elif name.endswith('_raw.json'):
            stamp = name[len(_MEASURED_PREFIX):-len('_raw.json')]
            runs.setdefault(stamp, {})['raw_file'] = name
    if not runs:
        return None

    loaded = []
    for stamp in sorted(runs, reverse=True):
        record = _load_run(dirn, stamp, runs[stamp])
        if record:
            loaded.append((stamp, record))

    # Drop replays. A reanalysis dumped before this version was written as a
    # plain `_raw.json`, so it looks like the newest run — but it executed
    # nothing and its job ids belong to an OLDER submission. Any run whose
    # ids are all claimed by something earlier is therefore a replay of it,
    # and replaying the replay would keep resurrecting the wrong run.
    def _ids_of(rec):
        return {j for ids in (rec.get('jobs') or {}).values() for j in ids}

    for index, (stamp, record) in enumerate(loaded):
        ids = _ids_of(record)
        if not ids:
            continue
        older = set()
        for _s, other in loaded[index + 1:]:
            older |= _ids_of(other)
        if ids <= older and older:
            record['_is_replay'] = True
            continue
        return record

    return loaded[0][1] if loaded else None


def _load_run(dirn, stamp, files):
    """One run's record, with phase-grouped job ids where recoverable."""
    record = None
    jobs_file = files.get('jobs_file')
    if jobs_file:
        try:
            with open(os.path.join(dirn, jobs_file), 'r',
                      encoding='utf-8') as fh:
                record = _json.load(fh)
            record['_path'] = os.path.join(dirn, jobs_file)
        except Exception:
            record = None

    raw_file = files.get('raw_file')
    raw_path_full = os.path.join(dirn, raw_file) if raw_file else None
    phases = []
    if raw_path_full:
        try:
            phases = [p for p in job_ids_from_raw(raw_path_full) if p]
        except Exception:
            phases = []

    if record is None:
        # Raw-only run: no spec list, so the caller cannot rebuild the plan.
        # Still worth returning, so the UI can say WHICH run it cannot
        # replay rather than silently reaching for an older one.
        if not phases:
            return None
        return {'_path': raw_path_full, '_stamp': stamp, 'specs': [],
                'jobs': {str(i): ids for i, ids in enumerate(phases)},
                '_raw_path': raw_path_full}

    record['_stamp'] = stamp
    record['_raw_path'] = raw_path_full
    # A record written before the phased runner recorded only one group; the
    # raw dump knows better, so prefer its phase split when it has more.
    recorded = [v for v in (record.get('jobs') or {}).values() if v]
    if len(phases) > len(recorded):
        record['jobs'] = {str(i): ids for i, ids in enumerate(phases)}
        record['_phases_from_raw'] = True
    return record


_MAX_DUMP_ITEMS = 1000


def notices_path(backend_name, when):
    return os.path.join(_measured_dir(backend_name),
                        f'{_MEASURED_PREFIX}{_stamp_for(when)}_notices.json')


def calibration_check_path(backend_name, when):
    return os.path.join(_measured_dir(backend_name),
                        f'{_MEASURED_PREFIX}{_stamp_for(when)}_calcheck.json')


def save_calibration_check(backend_name, when, table):
    """Persist the verdict table beside the run's other artifacts."""
    return _write_json(calibration_check_path(backend_name, when),
                       {'when': when,
                        'rows': [{**r, 'edge': list(r['edge'])}
                                 for r in (table or {}).get('rows', [])],
                        'ratio_parallel': (table or {}).get('ratio_parallel'),
                        'ratio_sequential': (table or {}).get(
                            'ratio_sequential'),
                        'verdict': (table or {}).get('verdict')})


def save_attempt_notices(backend_name, when, notices):
    """Persist an attempt's notices, whatever the outcome.

    Separate from the epoch file because the attempts that most need
    explaining are the ones that produce no epoch to attach notices to.
    """
    return _write_json(notices_path(backend_name, when),
                       {'when': when, 'notices': [str(n) for n in
                                                  (notices or [])]})


def dump_raw_results(expdata_list):
    """Everything an ExperimentData tree holds, as plain JSON-able data.

    Deliberately exhaustive and shape-agnostic: the bug this exists to catch
    was a parser looking in the wrong place, so a dump that only recorded what
    the parser already understood would have been useless. Values longer than
    1000 entries are truncated.
    """
    return {'experiments': [_dump_one(d) for d in (expdata_list or [])]}


def _dump_one(expdata, depth=0):
    if expdata is None or depth > 6:
        return {}
    node = {
        'type': type(expdata).__name__,
        'experiment_type': getattr(expdata, 'experiment_type', None),
        'status': _status_name(expdata),
        'job_ids': _job_ids_of(expdata),
        'results': [],
        'errors': None,
        'children': [],
    }
    try:
        errors = expdata.errors()
        node['errors'] = str(errors) if errors else None
    except Exception:
        pass
    raw = _safe_analysis_results(expdata)
    for item in _normalise_results(raw):
        node['results'].append({
            'name': str(getattr(item, 'name', '') or ''),
            'device_components': [str(c) for c in
                                  (getattr(item, 'device_components', None)
                                   or [])],
            'value': _dump_value(getattr(item, 'value', None)),
        })
    try:
        for child in list(expdata.child_data()):
            node['children'].append(_dump_one(child, depth + 1))
    except Exception:
        pass
    return node


def _dump_value(value):
    try:
        if hasattr(value, 'nominal_value'):
            return float(value.nominal_value)
        if isinstance(value, (int, float, str, bool)) or value is None:
            return value
        if hasattr(value, 'tolist'):
            listed = value.tolist()
            if isinstance(listed, list) and len(listed) > _MAX_DUMP_ITEMS:
                return listed[:_MAX_DUMP_ITEMS] + ['...truncated']
            return listed
        if isinstance(value, (list, tuple)):
            listed = list(value)
            if len(listed) > _MAX_DUMP_ITEMS:
                return listed[:_MAX_DUMP_ITEMS] + ['...truncated']
            return listed
        return str(value)[:500]
    except Exception:
        return '<unrepresentable>'


def reanalysis_raw_path(backend_name, when):
    return os.path.join(_measured_dir(backend_name),
                        f'{_MEASURED_PREFIX}{_stamp_for(when)}_rean.json')


def save_reanalysis_raw(backend_name, when, expdata_list):
    """Dump a REANALYSIS separately from a submission.

    A reanalysis executes nothing; its dump describes jobs that some earlier
    run submitted. Writing it as `_raw.json` made it look like the newest
    run, so the replay target became an artefact of the previous replay.
    """
    return _write_json(reanalysis_raw_path(backend_name, when),
                       dump_raw_results(expdata_list))


def save_raw_results(backend_name, when, expdata_list):
    return _write_json(raw_path(backend_name, when),
                       dump_raw_results(expdata_list))


def diagnose_raw(path_or_dump):
    """Human summary of a raw dump: where the results were, and what failed."""
    dump = path_or_dump
    if isinstance(path_or_dump, str):
        try:
            with open(path_or_dump, 'r', encoding='utf-8') as fh:
                dump = _json.load(fh)
        except Exception as exc:
            return f'Could not read {path_or_dump}: {exc}'

    lines = []
    experiments = (dump or {}).get('experiments') or []
    if not experiments:
        return 'Empty dump: no ExperimentData recorded.'

    from collections import Counter
    names = Counter()
    depth_hits = Counter()
    statuses = Counter()
    errors = []

    def _walk(node, depth):
        statuses[node.get('status') or '?'] += 1
        if node.get('errors'):
            errors.append((depth, node['errors']))
        for result in node.get('results') or []:
            names[result.get('name') or '?'] += 1
            depth_hits[depth] += 1
        for child in node.get('children') or []:
            _walk(child, depth + 1)

    for exp in experiments:
        _walk(exp, 0)

    lines.append(f'{len(experiments)} top-level experiment(s).')
    lines.append('Job statuses: ' + ', '.join(
        f'{k}={v}' for k, v in statuses.most_common()) or 'none')
    if depth_hits:
        lines.append('Analysis results by nesting depth: ' + ', '.join(
            f'depth {d}: {n}' for d, n in sorted(depth_hits.items())))
    else:
        lines.append('NO analysis results at any depth — '
                     'the run produced nothing to parse.')
    if names:
        lines.append('Result names: ' + ', '.join(
            f'{k} x{v}' for k, v in names.most_common(12)))
    if errors:
        lines.append(f'{len(errors)} node(s) reported errors:')
        for depth, text in errors[:5]:
            lines.append(f'  depth {depth}: {str(text)[:300]}')
    return '\n'.join(lines)


# ── Reanalysis from job ids (zero QPU) ──────────────────────────────

def job_ids_from_raw(path_or_dump):
    """Job ids per phase, recovered from a raw dump.

    Needed because `_run_phased` did not call the job-id callback until this
    version, so the 20 Sept 13:08 run left no `_jobs.json` — but the raw dump
    records `job_ids` on every node, so the run is still recoverable.
    Returns a list of lists, one per top-level experiment (phase).
    """
    dump = path_or_dump
    if isinstance(path_or_dump, str):
        with open(path_or_dump, 'r', encoding='utf-8') as fh:
            dump = _json.load(fh)

    def _ids(node):
        out = list(node.get('job_ids') or [])
        for child in node.get('children') or []:
            out.extend(_ids(child))
        return out

    phases = []
    for experiment in (dump or {}).get('experiments') or []:
        seen, ordered = set(), []
        for jid in _ids(experiment):
            if jid not in seen:
                seen.add(jid)
                ordered.append(jid)
        phases.append(ordered)
    return phases


def _initialize_container(experiment, backend, notices=None):
    """Build the ExperimentData the way `BaseExperiment.run()` does.

    A composite experiment's results do not live in one flat container: the
    analysis calls `CompositeAnalysis._component_experiment_data`, which walks
    per-component CHILD containers and the composite metadata that names them.
    `run()` creates all of that up front via
    `BaseExperiment._initialize_experiment_data()`, before a single circuit is
    submitted.

    Reanalysis used to skip that and construct a bare
    `ExperimentData(experiment=batch)`, which allocates no children and no
    component metadata — so the analysis raised

        AnalysisError: Unable to extract component child experiment data

    `_initialize_experiment_data` is private but has been the stable entry
    point for this across versions; the public constructor is kept as a
    fallback, and a notice records when the composite metadata still looks
    absent so the next failure is legible rather than cryptic.
    """
    notices = notices if notices is not None else []
    try:
        experiment.backend = backend
    except Exception:
        pass

    expdata = None
    initializer = getattr(experiment, '_initialize_experiment_data', None)
    if callable(initializer):
        try:
            expdata = initializer()
        except Exception as exc:              # noqa: BLE001 - fall back
            notices.append(f'_initialize_experiment_data failed ({exc}); '
                           f'falling back to a bare container')
    if expdata is None:
        from qiskit_experiments.framework import ExperimentData
        expdata = ExperimentData(experiment=experiment, backend=backend)

    missing = _missing_composite_metadata(expdata, experiment)
    if missing:
        _log.warning(
            'reanalysis container is missing composite metadata (%s); '
            'component results may not be extractable in this '
            'qiskit-experiments version', ', '.join(missing))
        notices.append('Reanalysis container incomplete \u2014 results may '
                       'be partial')
    return expdata


#: Keys `CompositeAnalysis` uses to locate a component's child container. The
#: name has moved between versions, so presence of ANY of them is enough.
_COMPOSITE_METADATA_KEYS = ('component_child_index', 'component_types',
                            'component_metadata')


def _missing_composite_metadata(expdata, experiment):
    """Diagnostic: what a composite container should carry but does not.

    Not a hard check — the exact keys differ by version, and this module is
    written against several. It exists so that if the container is wrong
    again the notice says which piece was absent instead of surfacing a bare
    AnalysisError from deep inside the library.
    """
    if not getattr(experiment, '_experiments', None):
        return []                             # not composite; nothing to carry
    try:
        metadata = dict(expdata.metadata or {})
    except Exception:
        return ['metadata unreadable']
    # Only the METADATA is required at initialisation. qiskit-experiments
    # 0.10 allocates the child containers later, during add_jobs/analysis, so
    # demanding them here produced a false alarm on runs that then worked.
    if any(key in metadata for key in _COMPOSITE_METADATA_KEYS):
        return []
    return ['/'.join(_COMPOSITE_METADATA_KEYS)]


def plan_for_replay(record, backend_name=None, n_qubits=None,
                    coupling_edges=None):
    """The plan to replay, and how it was obtained.

    Returns `(plan, how)` where `how` is 'record' (exact, from the saved spec
    list) or 'reconstructed' (a whole-chip Level 3 plan rebuilt from the
    device, which is fully determined and so safe to recreate). Runs from
    before the spec list was recorded are otherwise unreplayable, and a
    Level 1 or 2 run cannot be reconstructed at all because its targets are
    not recoverable from the dump.
    """
    plan = plan_from_record(record)
    if plan is not None:
        return plan, 'record'

    level = None
    if record:
        level = record.get('level')
    if level is None and backend_name and record and record.get('_stamp'):
        epoch_file = os.path.join(_measured_dir(backend_name),
                                  f"{_MEASURED_PREFIX}{record['_stamp']}.json")
        try:
            with open(epoch_file, 'r', encoding='utf-8') as fh:
                level = _json.load(fh).get('level')
        except Exception:
            level = None

    if int(level or 0) == LEVEL_FULL and n_qubits and coupling_edges:
        return (build_plan(LEVEL_FULL, Targets([], []), n_qubits,
                           coupling_edges), 'reconstructed')
    return None, None


def plan_from_record(record):
    """Rebuild the exact `BenchmarkPlan` a saved submission used.

    Reanalysis must replay the plan that PRODUCED the jobs, not whatever the
    UI's target selector happens to point at now. Using the live selector
    meant that reanalysing with the selector on "Current layout(s)" and
    nothing placed built a plan with ZERO specs: no experiment to rebuild, no
    result to attribute, no notice to emit — a silent (0, 0) that looked like
    "the jobs contained nothing" when the jobs in fact contained everything.
    """
    if not record:
        return None
    specs = []
    for raw_spec in record.get('specs') or []:
        params = dict(raw_spec.get('params') or {})
        if 'pairs' in params:
            params['pairs'] = [tuple(p) for p in params['pairs']]
        specs.append(ExperimentSpec(raw_spec['kind'],
                                    tuple(raw_spec.get('qubits') or []),
                                    params, int(raw_spec.get('layer') or 0)))
    if not specs:
        return None
    targets_raw = record.get('targets') or {}
    targets = Targets(list(targets_raw.get('qubits') or []),
                      [tuple(e) for e in (targets_raw.get('edges') or [])])
    return BenchmarkPlan(int(record.get('level') or LEVEL_FULL), specs,
                         targets, int(record.get('shots') or DEFAULT_SHOTS))


def reanalyze_from_jobs(plan, job_ids, backend, service=None):
    """Rebuild the analysis from already-executed jobs. Costs no QPU time.

    qiskit-experiments runs its fits CLIENT-SIDE: the jobs hold the raw
    counts, and analysis is pure post-processing. So a run whose fits crashed
    — as the 20 Sept two-qubit callbacks did — can be recovered from its job
    ids alone, with the corrected options, without re-executing anything.

    `job_ids` may be a flat list (single-phase) or a list of per-phase lists
    (as `job_ids_from_raw` returns). The phased form matters: the plan is
    submitted as non-2Q first then 2Q, so the 1Q EPG results must be
    harvested from phase 1 and handed to phase 2's analyses exactly as the
    live runner does — reanalysing both phases as one batch would reproduce
    the un-subtracted numbers.
    """
    if not plan.specs:
        raise BenchmarkRunError(
            "Nothing to reanalyse: this run's plan was not recorded")
    api = preload_runtime()
    phases_ids = (list(job_ids) if job_ids and isinstance(job_ids[0], (list, tuple))
                  else [list(job_ids or [])])

    has_2q = any(sp.kind in ('rb2q_layer', 'rb2q') for sp in plan.specs)
    if has_2q and len(phases_ids) > 1:
        groups = [[sp for sp in plan.specs
                   if sp.kind not in ('rb2q_layer', 'rb2q')],
                  [sp for sp in plan.specs
                   if sp.kind in ('rb2q_layer', 'rb2q')]]
    else:
        groups = [list(plan.specs)]
        phases_ids = [[j for ids in phases_ids for j in ids]]

    index_of = {id(sp): i for i, sp in enumerate(plan.specs)}
    context = {'epg_results': {}, 'epg_1q': {}, 'notices': []}
    merged = {}
    collected = []

    for phase, (specs, ids) in enumerate(zip(groups, phases_ids)):
        if not specs:
            continue
        experiments = [(sp, _build_experiment(sp, api, context))
                       for sp in specs]
        batch = api.BatchExperiment([e for _s, e in experiments],
                                    flatten_results=False)
        jobs = []
        for job_id in ids:
            if service is None:
                jobs.append(job_id)
                continue
            try:
                jobs.append(service.job(job_id))
            except Exception as exc:          # noqa: BLE001 - reported
                raise BenchmarkRunError(
                    f'could not retrieve job {job_id}: {exc}', ids) from exc

        expdata = _initialize_container(batch, backend, context['notices'])
        expdata.add_jobs(jobs)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', category=RuntimeWarning,
                module=r'.*qiskit_experiments.*rb_analysis.*')
            batch.analysis.run(expdata, replace_results=True)
            expdata.block_for_results()

        failures = summarise_analysis_errors(collect_analysis_errors(expdata))
        if failures:
            context['notices'].append(f'phase {phase + 1}: {failures}')
        collected.append(expdata)

        for local_idx, (sp, _exp) in enumerate(experiments):
            try:
                merged[index_of[id(sp)]] = list(
                    _iter_analysis_results(expdata.child_data(local_idx)))
            except Exception:
                merged[index_of[id(sp)]] = None

        if phase == 0 and len(groups) > 1:
            for sp in specs:
                if sp.kind != 'rb1q':
                    continue
                entry = merged.get(index_of[id(sp)]) or []
                harvest = harvest_epg_results(entry)
                context['epg_results'].update(harvest.results)
                context.setdefault('_epg_rejected', {}).update(
                    harvest.rejected)
                harvested = {}
                _collect_rb1q(entry, sp, [], harvested)
                context['epg_1q'].update(harvested.get('epg_1q', {}))

    tally = epg_tally_notice(context)
    if tally:
        context['notices'].append(tally)
    elif any(sp.kind in ('rb2q_layer', 'rb2q') for sp in plan.specs):
        context['notices'].append(
            '1Q EPG subtraction: applied to 0 pairs (single-phase run) '
            '\u2014 two-qubit errors read high')
    if context['notices']:
        merged['_notices'] = list(context['notices'])
    return merged, collected


def latest_reusable_epoch(backend_name, ttl_minutes=REUSE_TTL_MINUTES,
                          now=None):
    """The newest saved epoch still inside the reuse TTL, or None.

    Lets the UI offer "reuse the 14:32 measurement" instead of spending QPU
    time again on a sweep the user re-ran two minutes later.
    """
    for when in list_measured_epochs(backend_name):
        epoch = load_measured_epoch(backend_name, when)
        if epoch is not None and epoch.younger_than(ttl_minutes, now):
            return epoch
    return None


# ── Summary line for the UI ─────────────────────────────────────────

def summary_line(targets, cost, plan_name=None):
    """The one-line "what this will cost" string shown before the click."""
    plan_txt = (plan_name or 'unknown').lower()
    return (f'Targets: {len(targets.qubits)} qubits · {len(targets.edges)} '
            f'edges · {cost.text} · plan: {plan_txt}')
