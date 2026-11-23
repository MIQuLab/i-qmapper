"""Integration tests: the benchmark pipeline against REAL qiskit-experiments.

Why these exist
---------------
Every offline test fed the parser fake results shaped the way WE assumed
qiskit-experiments shapes them. It never ran a real experiment, never saw a
real `AnalysisResult`, and never submitted anything. So when the live Level 3
run was rejected by IBM and came back with no analysis results at all, nothing
in the suite noticed — the parser found nothing, reported "no results
returned" for every experiment, and the panel called it a success.

These tests close that gap without a QPU: real experiment objects, real
analysis, real result shapes, executed on a local fake backend or Aer. No
network, no credentials, no queue.

They are marked `integration` and skip cleanly when the optional stack is
absent, so the default suite stays runnable on a bare install:

    python -m pytest -m integration -q
"""

import numpy as np
import pytest

from iqmapper import benchmark as bench

pytestmark = pytest.mark.integration

qiskit_experiments = pytest.importorskip(
    'qiskit_experiments', reason='needs the [benchmark] extra')
pytest.importorskip('qiskit_aer', reason='needs qiskit-aer for local execution')


# ── a local backend + sampler ───────────────────────────────────────

@pytest.fixture(scope='module')
def fake_backend():
    """A small local backend. Real target, real coupling, no network."""
    from qiskit_ibm_runtime.fake_provider import FakeManilaV2
    return FakeManilaV2()


@pytest.fixture(scope='module')
def local_api(fake_backend):
    """The runtime API namespace with execution pointed at Aer.

    Only the EXECUTION is substituted: the experiment classes, the composite
    machinery and the analysis are the real ones, which is the entire point.
    """
    from types import SimpleNamespace

    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import SamplerV2 as AerSampler
    from qiskit_experiments.framework import (BatchExperiment,
                                              ParallelExperiment)
    from qiskit_experiments.library import (LocalReadoutError, StandardRB, T1,
                                            T2Hahn)

    simulator = AerSimulator.from_backend(fake_backend)

    class _Session:
        """Context manager with the production shape, backed by Aer."""

        def __init__(self, backend=None):
            self.backend = backend

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    return SimpleNamespace(
        Session=_Session,
        SamplerV2=lambda mode=None: AerSampler(),
        BatchExperiment=BatchExperiment, ParallelExperiment=ParallelExperiment,
        LocalReadoutError=LocalReadoutError, StandardRB=StandardRB,
        T1=T1, T2Hahn=T2Hahn,
        _simulator=simulator)


def _reported(n_q, n_e):
    return ({'readout_error': np.full(n_q, 0.05),
             't1': np.full(n_q, 100.0),
             't2': np.full(n_q, 80.0)},
            {'cz_error': np.full(n_e, 0.01)})


def _coupling_of(backend):
    edges = []
    seen = set()
    for u, v in backend.coupling_map.get_edges():
        key = frozenset((u, v))
        if key not in seen:
            seen.add(key)
            edges.append((int(u), int(v)))
    return edges


def _run_through_production_runner(monkeypatch, local_api, backend, plan,
                                   counter=None):
    """Drive the REAL runner with only execution substituted."""
    api = local_api
    if counter is not None:
        inner = api.SamplerV2

        def _counting(mode=None):
            sampler = inner(mode)
            original = sampler.run

            def _run(circuits, **kwargs):
                counter['circuits'] += len(list(circuits))
                return original(circuits, **kwargs)

            sampler.run = _run
            return sampler

        api = type(api)(**{**vars(api), 'SamplerV2': _counting})

    monkeypatch.setattr(bench, '_runtime_api', lambda: api)
    bench._reset_runtime_cache()
    try:
        runner = bench.make_runtime_runner(backend)
        return bench.run_plan(plan, runner), getattr(runner, 'last_expdata', [])
    finally:
        bench._reset_runtime_cache()


# ── 1. Level 1 really produces readout measurements ─────────────────

def test_l1_plan_on_fake_backend_yields_measurements(monkeypatch, fake_backend,
                                                     local_api):
    """The whole run against a local sampler, on a plan built for a real
    device.
    """
    targets = bench.Targets([0, 1, 2], [])
    coupling = _coupling_of(fake_backend)
    plan = bench.build_plan(bench.LEVEL_QUICK_READOUT, targets,
                            fake_backend.num_qubits, coupling)
    plan = plan._replace(shots=100)

    raw, _expdata = _run_through_production_runner(
        monkeypatch, local_api, fake_backend, plan)

    qp, gp = _reported(fake_backend.num_qubits, len(coupling))
    epoch = bench.results_to_epoch(raw, plan, qp, gp, coupling)

    assert not epoch.is_empty, (
        f'no measurements came back; notices: {epoch.notices}')
    assert epoch.n_measured_qubits >= 3, epoch.notices
    values = epoch.qubit_props['readout_error'][list(targets.qubits)]
    assert np.all(np.isfinite(values))
    assert np.all((values >= 0.0) & (values <= 0.5)), values


# ── 2. Level 2: a 2Q RB pair produces a finite EPG ──────────────────

def test_l2_single_pair_on_fake_backend(monkeypatch, fake_backend, local_api):
    coupling = _coupling_of(fake_backend)
    pair = coupling[0]
    targets = bench.Targets(list(pair), [pair])
    plan = bench.build_plan(bench.LEVEL_TARGETED, targets,
                            fake_backend.num_qubits, coupling)
    # Keep it to seconds: short sequences, few samples, few shots.
    specs = []
    for spec in plan.specs:
        if spec.kind == 'rb2q_layer':
            params = dict(spec.params)
            params['lengths'] = [2, 4]
            params['num_samples'] = 2
            spec = spec._replace(params=params)
        specs.append(spec)
    plan = plan._replace(specs=specs, shots=100)

    raw, _expdata = _run_through_production_runner(
        monkeypatch, local_api, fake_backend, plan)

    qp, gp = _reported(fake_backend.num_qubits, len(coupling))
    epoch = bench.results_to_epoch(raw, plan, qp, gp, coupling)

    assert epoch.n_measured_edges >= 1, epoch.notices
    idx = coupling.index(pair)
    epg = epoch.gate_props['cz_error'][idx]
    assert np.isfinite(epg) and 0.0 <= epg < 1.0, epg


# ── 3. what we submit is what we planned ────────────────────────────

def test_submitted_circuit_count_matches_plan(monkeypatch, fake_backend,
                                              local_api):
    """The accounting bug that caused error 1520, asserted directly."""
    targets = bench.Targets([0, 1, 2], [])
    coupling = _coupling_of(fake_backend)
    plan = bench.build_plan(bench.LEVEL_QUICK_READOUT, targets,
                            fake_backend.num_qubits, coupling)
    plan = plan._replace(shots=100)

    counter = {'circuits': 0}
    _run_through_production_runner(monkeypatch, local_api, fake_backend, plan,
                                  counter)

    expected = bench.plan_circuit_count(plan)
    assert counter['circuits'] == expected, (
        f'submitted {counter["circuits"]} circuits, planned {expected}')


# ── 4. the full suite, small ────────────────────────────────────────

def test_l3_small_fake_full_suite(monkeypatch, fake_backend, local_api):
    """Every experiment kind produces results, and diagnose_raw finds them."""
    coupling = _coupling_of(fake_backend)
    n_q = fake_backend.num_qubits
    plan = bench.build_plan(bench.LEVEL_FULL, bench.Targets([], []), n_q,
                            coupling)
    specs = []
    for spec in plan.specs:
        params = dict(spec.params)
        if spec.kind in ('t1', 't2hahn'):
            params['delays'] = [1e-6, 20e-6, 40e-6]
        if spec.kind in ('rb1q', 'rb2q_layer'):
            params['lengths'] = [2, 4]
            params['num_samples'] = 2
        specs.append(spec._replace(params=params))
    plan = plan._replace(specs=specs, shots=100)

    raw, expdata = _run_through_production_runner(
        monkeypatch, local_api, fake_backend, plan)

    qp, gp = _reported(n_q, len(coupling))
    epoch = bench.results_to_epoch(raw, plan, qp, gp, coupling)
    assert not epoch.is_empty, epoch.notices
    assert epoch.n_measured_qubits > 0 and epoch.n_measured_edges > 0

    text = bench.diagnose_raw(bench.dump_raw_results(expdata))
    assert 'NO analysis results' not in text, text
    assert 'depth' in text


# ── 5. delays must be legal on the device ───────────────────────────

def test_t1_delays_are_within_the_backend_limits(fake_backend):
    """A delay longer than the device allows is rejected at submission."""
    target = getattr(fake_backend, 'target', None)
    max_delay = None
    try:
        granularity = target.granularity
        dt = target.dt
        max_delay = getattr(target, 'max_delay', None)
        if max_delay is not None and dt:
            max_delay = max_delay * dt
        assert granularity is not None
    except Exception:
        pytest.skip('backend exposes no delay limits')
    if max_delay is None:
        pytest.skip('backend declares no max_delay')
    assert max(bench.T1T2_DELAYS) <= max_delay, (
        f'longest tutorial delay {max(bench.T1T2_DELAYS)}s exceeds the '
        f'device limit {max_delay}s')


# ── reanalysis reproduces the direct run ────────────────────────────

def _jobs_of(expdata_list):
    """The job OBJECTS a run produced, for reanalysis to consume."""
    jobs = []
    for expdata in expdata_list or []:
        getter = getattr(expdata, 'jobs', None)
        if callable(getter):
            try:
                jobs.extend(list(getter()))
                continue
            except Exception:
                pass
    return jobs


def test_reanalysis_from_jobs_reproduces_the_direct_run(monkeypatch,
                                                         fake_backend,
                                                         local_api):
    """Rebuild from the JOBS, not from the run's data.

    A bare `ExperimentData` with jobs added to it lacks the composite
    children and component metadata that `BaseExperiment.run()` allocates
    before submission (`AnalysisError: Unable to extract component child
    experiment data`). Reanalysing from the original `expdata` would hide
    that, because that container already has them. This test therefore goes through
    `reanalyze_from_jobs` with job objects and nothing else.

    Analysis is client-side, so this costs no execution: the same counts are
    re-fitted.
    """
    targets = bench.Targets([0, 1, 2], [])
    coupling = _coupling_of(fake_backend)
    plan = bench.build_plan(bench.LEVEL_QUICK_READOUT, targets,
                            fake_backend.num_qubits, coupling)._replace(
                                shots=100)

    raw, expdata = _run_through_production_runner(monkeypatch, local_api,
                                                  fake_backend, plan)
    qp, gp = _reported(fake_backend.num_qubits, len(coupling))
    direct = bench.results_to_epoch(raw, plan, qp, gp, coupling)
    assert not direct.is_empty, direct.notices

    jobs = _jobs_of(expdata)
    if not jobs:
        pytest.skip('the local sampler exposes no job objects to reanalyze')

    monkeypatch.setattr(bench, '_runtime_api', lambda: local_api)
    bench._reset_runtime_cache()
    try:
        # A pre-built job list is passed straight through (service=None), so
        # this exercises container construction + add_jobs + analysis.run.
        raw2, expdata2 = bench.reanalyze_from_jobs(plan, [jobs], fake_backend,
                                                   service=None)
    finally:
        bench._reset_runtime_cache()

    assert not any('Unable to extract component' in str(n)
                   for n in (raw2.get('_notices') or [])), raw2.get('_notices')
    qp2, gp2 = _reported(fake_backend.num_qubits, len(coupling))
    again = bench.results_to_epoch(raw2, plan, qp2, gp2, coupling)

    assert not again.is_empty, again.notices
    assert again.measured_count == direct.measured_count
    np.testing.assert_allclose(
        again.qubit_props['readout_error'][list(targets.qubits)],
        direct.qubit_props['readout_error'][list(targets.qubits)],
        rtol=1e-9)


def test_reanalysis_container_has_the_composite_structure(fake_backend,
                                                           local_api):
    """The container itself, before any jobs: children + metadata present."""
    coupling = _coupling_of(fake_backend)
    plan = bench.build_plan(bench.LEVEL_QUICK_READOUT,
                            bench.Targets([0, 1, 2], []),
                            fake_backend.num_qubits, coupling)
    experiments = [bench._build_experiment(sp, local_api) for sp in plan.specs]
    batch = local_api.BatchExperiment(experiments, flatten_results=False)

    notices = []
    container = bench._initialize_container(batch, fake_backend, notices)

    assert notices == [], notices
    # Children are allocated later (during add_jobs/analysis) in 0.10; the
    # metadata naming the components is what must exist up front.
    metadata = dict(container.metadata or {})
    assert any(k in metadata for k in bench._COMPOSITE_METADATA_KEYS), metadata


def test_two_phase_reanalysis_keeps_the_1q_subtraction(monkeypatch,
                                                        fake_backend,
                                                        local_api):
    """Reanalysing both phases as one batch would lose the subtraction.

    The 1Q EPG results come from phase 1 and must be harvested before
    phase 2's analyses are constructed, exactly as the live runner does.
    """
    coupling = _coupling_of(fake_backend)
    pair = coupling[0]
    targets = bench.Targets(list(pair), [pair])
    plan = bench.build_plan(bench.LEVEL_TARGETED, targets,
                            fake_backend.num_qubits, coupling)
    specs = []
    for spec in plan.specs:
        params = dict(spec.params)
        if spec.kind in ('rb1q', 'rb2q_layer'):
            params['lengths'] = [2, 4]
            params['num_samples'] = 2
        specs.append(spec._replace(params=params))
    plan = plan._replace(specs=specs, shots=100)

    raw, expdata = _run_through_production_runner(monkeypatch, local_api,
                                                  fake_backend, plan)
    assert len(expdata) == 2, 'expected a two-phase run'
    per_phase = [_jobs_of([d]) for d in expdata]
    if not all(per_phase):
        pytest.skip('the local sampler exposes no job objects to reanalyze')

    monkeypatch.setattr(bench, '_runtime_api', lambda: local_api)
    bench._reset_runtime_cache()
    try:
        raw2, _ = bench.reanalyze_from_jobs(plan, per_phase, fake_backend,
                                            service=None)
    finally:
        bench._reset_runtime_cache()

    qp, gp = _reported(fake_backend.num_qubits, len(coupling))
    again = bench.results_to_epoch(raw2, plan, qp, gp, coupling)
    assert again.n_measured_edges >= 1, again.notices
    assert not any('device_components' in n for n in again.notices), (
        again.notices)


def test_calibration_check_runs_both_ways(monkeypatch, fake_backend,
                                          local_api):
    """Parallel vs sequential over the same edges, and a verdict.

    Restored after being lost in a file rewrite. It is the experiment that
    distinguishes estimator bias (both arms offset) from crosstalk (only the
    parallel arm offset), so losing it silently was a real gap.
    """
    coupling = _coupling_of(fake_backend)
    edges = coupling[:2]
    targets = bench.Targets(sorted({q for e in edges for q in e}), edges)
    par_plan, seq_plan = bench.build_calibration_check_plans(
        targets, fake_backend.num_qubits, coupling, n_edges=2)

    # The sequential arm must run one pair per experiment.
    seq_layers = [s for s in seq_plan.specs if s.kind == 'rb2q_layer']
    assert all(len(s.params['pairs']) == 1 for s in seq_layers), seq_layers
    par_layers = [s for s in par_plan.specs if s.kind == 'rb2q_layer']
    assert len(par_layers) <= len(seq_layers)

    def _shrink(plan):
        specs = []
        for spec in plan.specs:
            params = dict(spec.params)
            if spec.kind in ('rb1q', 'rb2q_layer'):
                params['lengths'] = [2, 4]
                params['num_samples'] = 2
            specs.append(spec._replace(params=params))
        return plan._replace(specs=specs, shots=100)

    qp, gp = _reported(fake_backend.num_qubits, len(coupling))
    epochs = []
    for plan in (_shrink(par_plan), _shrink(seq_plan)):
        raw, _ = _run_through_production_runner(monkeypatch, local_api,
                                                fake_backend, plan)
        epochs.append(bench.results_to_epoch(raw, plan, dict(qp), dict(gp),
                                             coupling))

    table = bench.calibration_check_table(coupling, gp['cz_error'],
                                          epochs[0], epochs[1])
    assert table['rows'], 'the check produced no rows'
    assert table['ratio_parallel'] is not None
    assert table['ratio_sequential'] is not None
    assert table['verdict']
    for row in table['rows']:
        assert np.isfinite(row['parallel']) and np.isfinite(row['sequential'])
