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

_SEC_PER_CIRCUIT_1Q = 0.012
_SEC_PER_CIRCUIT_2Q = 0.035
_SEC_PER_T1T2 = 0.030
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
                 'measured_edge_mask', 'level', 'when', 'notices')

    def __init__(self, qubit_props, gate_props, measured_qubit_mask,
                 measured_edge_mask, level, when, notices=None):
        self.qubit_props = qubit_props
        self.gate_props = gate_props
        self.measured_qubit_mask = np.asarray(measured_qubit_mask, dtype=bool)
        self.measured_edge_mask = np.asarray(measured_edge_mask, dtype=bool)
        self.level = int(level)
        self.when = when                      # epoch seconds (time.time())
        self.notices = list(notices or [])

    # ── convenience ────────────────────────────────────────────
    @property
    def n_measured_qubits(self):
        return int(self.measured_qubit_mask.sum())

    @property
    def n_measured_edges(self):
        return int(self.measured_edge_mask.sum())

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

    if level == LEVEL_FULL and qubits:
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
            return False, (
                f'One experiment needs {chunk.shots:,} circuit-shots, over '
                f"IBM's {_MAX_CIRCUIT_SHOTS_PER_JOB:,} per-job limit, and "
                f'cannot be split further. Reduce the target set or the level.')
    return True, None


def estimate_cost(plan, shots=DEFAULT_SHOTS):
    """Wall-clock QPU seconds a plan is expected to consume.

    An ESTIMATE, and a coarse one: the constants are back-solved from the
    tutorial's single published figure (~240 s for the full suite on 127
    qubits) and take no account of queueing, of how busy the device is, or of
    the provider's batching. Treat it as an order of magnitude that is right,
    not a quote. It is monotonic in plan size, which is the property the UI
    actually depends on.
    """
    circuits = 0
    seconds = 0.0
    for spec in plan.specs:
        n = _spec_circuits(spec)
        circuits += n
        if spec.kind == 'readout':
            # LocalReadoutError prepares |0...0> and |1...1> across every
            # target qubit at once: two circuits, whatever the width. So a
            # whole-QPU Level 1 costs the same as a three-qubit one, which is
            # what makes it the cheap way to refresh the readout heatmap.
            seconds += n * _SEC_PER_CIRCUIT_1Q
        elif spec.kind in ('t1', 't2hahn'):
            # One sequence per delay per qubit, run in parallel across qubits.
            seconds += n * _SEC_PER_T1T2 * max(1, len(spec.qubits))
        elif spec.kind == 'rb2q':
            seconds += n * _SEC_PER_CIRCUIT_2Q
        else:
            seconds += n * _SEC_PER_CIRCUIT_1Q * max(1, len(spec.qubits))

    seconds *= (shots / float(DEFAULT_SHOTS)) if DEFAULT_SHOTS else 1.0
    seconds += _SESSION_OVERHEAD_S if plan.specs else 0.0

    return CostEstimate(circuits, seconds, _format_cost(seconds))


def _format_cost(seconds):
    if seconds < 90:
        return f'~{seconds:.0f} s QPU (estimate)'
    return f'~{seconds / 60.0:.1f} min QPU (estimate)'


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
                    'Level 3 measures the whole chip (~4 min of QPU time). '
                    'On an Open-plan account this can consume a large part of '
                    'your monthly allowance in one run.')
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


def _build_experiment(spec, api):
    """One `ExperimentSpec` → the qiskit-experiments object that runs it."""
    physical = list(spec.qubits)
    if spec.kind == 't1':
        return api.ParallelExperiment(
            [api.T1((q,), delays=spec.params['delays']) for q in physical])
    if spec.kind == 't2hahn':
        return api.ParallelExperiment(
            [api.T2Hahn((q,), delays=spec.params['delays'])
             for q in physical])
    if spec.kind == 'readout':
        return api.LocalReadoutError(physical)
    if spec.kind == 'rb1q':
        return api.ParallelExperiment(
            [api.StandardRB((q,), lengths=spec.params['lengths'],
                            num_samples=spec.params['num_samples'])
             for q in physical])
    if spec.kind == 'rb2q':
        return api.StandardRB(tuple(physical),
                              lengths=spec.params['lengths'],
                              num_samples=spec.params['num_samples'])
    raise ValueError(f'unknown experiment kind: {spec.kind}')


def make_runtime_runner(backend):
    """The production runner: qiskit-experiments against a real backend.

    The session is opened as `Session(backend=backend)`. Older
    qiskit-ibm-runtime took a `service=` kwarg as well; 0.48 removed it,
    because a backend object already carries its own service. Passing it now
    raises `TypeError: Session.__init__() got an unexpected keyword argument
    'service'`, which is what the first live run hit. The call shape is pinned
    by tests/test_benchmark_runtime_contract.py so the next such drift fails
    offline instead of on a device.

    The session is entered with `with`, so it closes on the way out whether
    the run succeeds or raises — a leaked session keeps holding a reservation.

    Resolves through `preload_runtime`, so if the UI already preloaded on the
    main thread this is a cache read. Calling it from a worker without that
    preload would import the stack off-thread — see `preload_runtime` for the
    crash that causes.
    """
    api = preload_runtime()

    def _runner(plan):
        experiments = [(spec, _build_experiment(spec, api))
                       for spec in plan.specs]
        batch = api.BatchExperiment([exp for _s, exp in experiments],
                                    flatten_results=False)

        with api.Session(backend=backend) as session:
            sampler = api.SamplerV2(mode=session)
            data = batch.run(sampler=sampler, shots=plan.shots)
            data.block_for_results()

        out = {}
        for idx, (spec, _exp) in enumerate(experiments):
            try:
                out[idx] = data.child_data(idx).analysis_results()
            except Exception as exc:          # noqa: BLE001 - per-experiment
                _log.warning("benchmark: no results for %s %s: %s",
                             spec.kind, spec.qubits, exc)
                out[idx] = None
        return out

    return _runner


# ── Results → epoch ─────────────────────────────────────────────────

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

    edge_index = {tuple(e): i for i, e in enumerate(edges_list or [])}
    two_q_key = _two_qubit_key(gate_props)

    for idx, spec in enumerate(plan.specs):
        entry = _lookup(raw, idx)
        # An empty sequence counts as "nothing came back" just as None does:
        # a runner that returns no analysis results must produce a visible
        # notice, never a silent no-op that looks like a successful run.
        if entry is None or (isinstance(entry, (list, tuple)) and not entry):
            notices.append(f'{spec.kind} {list(spec.qubits)}: no results '
                           f'returned, kept reported values')
            continue
        try:
            if spec.kind == 'readout':
                _apply_readout(entry, spec, qubit_props, q_mask, n_q, notices)
            elif spec.kind in ('t1', 't2hahn'):
                _apply_coherence(entry, spec, qubit_props, q_mask, n_q,
                                 notices)
            elif spec.kind == 'rb2q':
                _apply_rb2q(entry, spec, gate_props, e_mask, edge_index,
                            two_q_key, notices)
            # rb1q is measured at Level 3 for the diagnostic view; it has no
            # single reported counterpart to overwrite (the reported arrays
            # carry per-gate sx/x/rz errors separately), so it is recorded in
            # the notices rather than written into an array.
            elif spec.kind == 'rb1q':
                _note_rb1q(entry, spec, notices)
        except Exception as exc:              # noqa: BLE001 - never fatal
            _log.warning("benchmark: folding %s %s failed: %s",
                         spec.kind, spec.qubits, exc, exc_info=True)
            notices.append(f'{spec.kind} {list(spec.qubits)}: '
                           f'{exc}; kept reported values')

    return MeasuredEpoch(qubit_props, gate_props, q_mask, e_mask,
                         plan.level, when, notices)


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
        notices.append(f'readout {list(spec.qubits)}: no assignment matrix in '
                       f'the results, kept reported values')
        return
    for qubit, matrix in matrices.items():
        if not (0 <= qubit < n_q):
            continue
        try:
            arr[qubit] = _readout_error_from_matrix(matrix)
            q_mask[qubit] = True
        except Exception as exc:              # noqa: BLE001 - per element
            notices.append(f'q{qubit}: readout fit failed ({exc}), '
                           f'kept reported value')


def _readout_matrices(entry, spec):
    """Map physical qubit → 2x2 assignment matrix from a readout result."""
    out = {}
    items = entry if isinstance(entry, (list, tuple)) else [entry]
    for item in items:
        value = getattr(item, 'value', item)
        matrix = getattr(value, 'assignment_matrix', None)
        if matrix is None and isinstance(value, dict):
            matrix = value.get('assignment_matrix')
        if matrix is None:
            continue
        qubits = (getattr(item, 'qubits', None)
                  or getattr(value, 'qubits', None) or spec.qubits)
        matrix = np.asarray(matrix, dtype=float)
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
        notices.append(f'{key.upper()} {list(spec.qubits)}: no results, '
                       f'kept reported values')
        return
    for item in items:
        qubits = getattr(item, 'qubits', None) or spec.qubits
        if not qubits:
            continue
        qubit = int(qubits[0])
        if not (0 <= qubit < n_q):
            continue
        try:
            seconds = _result_value(item)
            if not np.isfinite(seconds) or seconds <= 0:
                raise ValueError(f'implausible {key}={seconds}')
            arr[qubit] = seconds * 1e6        # data.py stores µs
            q_mask[qubit] = True
        except Exception as exc:              # noqa: BLE001 - per element
            notices.append(f'q{qubit}: {key.upper()} fit failed ({exc}), '
                           f'kept reported value')


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


def _note_rb1q(entry, spec, notices):
    """Record 1Q RB as a notice: diagnostic, with no single array to own."""
    try:
        items = entry if isinstance(entry, (list, tuple)) else [entry]
        values = []
        for item in items:
            name = getattr(item, 'name', None)
            if name and str(name).upper().startswith(('EPG', 'EPC')):
                values.append(_result_value(item))
        if values:
            notices.append(f'1Q RB: median EPG {np.median(values):.2e} over '
                           f'{len(values)} qubit(s)')
    except Exception as exc:                  # noqa: BLE001 - informational
        notices.append(f'1Q RB: could not summarise ({exc})')


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
    state.calibration_data_label = (
        f'{state.active_backend_name} — measured {epoch.time_label()} '
        f'(L{epoch.level}, {epoch.n_measured_qubits}q/'
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
            data['level'], float(data['when']), data.get('notices'))
    except Exception as exc:                  # noqa: BLE001 - corrupt file
        _log.warning("Failed to load measured epoch %s/%s: %s",
                     backend_name, when, exc)
        return None


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
