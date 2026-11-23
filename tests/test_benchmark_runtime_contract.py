"""Contract tests for the production benchmark runner.

These exist because of a specific failure. The first live benchmark run died
on `Session.__init__() got an unexpected keyword argument 'service'`:
qiskit-ibm-runtime 0.48 dropped the `service=` kwarg, and nothing offline
exercised the production runner, so the drift reached a real device before it
reached a test.

Everything here monkeypatches `benchmark._runtime_api` with fakes that record
how they were constructed and called. No quantum stack is imported, no network
is touched — but the exact call shapes the runner depends on are pinned, so
the next signature change fails here instead of on a QPU.
"""
from types import SimpleNamespace

import pytest

from iqmapper import benchmark as bench


# ── fakes ───────────────────────────────────────────────────────────

class _FakeSession:
    """Records its constructor kwargs and its context-manager lifecycle."""

    instances = []

    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)
        self.entered = False
        self.exited = False
        self.exit_exc = None
        _FakeSession.instances.append(self)

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, exc_type, exc, tb):
        self.exited = True
        self.exit_exc = exc_type
        return False                          # never swallow


class _FakeSampler:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)
        _FakeSampler.instances.append(self)


class _FakeData:
    def __init__(self, n_children, results=None):
        self._n = n_children
        self._results = results
        self.blocked = False

    def block_for_results(self):
        self.blocked = True

    def child_data(self, idx):
        if idx >= self._n:
            raise IndexError(idx)
        return SimpleNamespace(
            analysis_results=lambda: (self._results or [f'result-{idx}']))


class _FakeBatch:
    instances = []

    def __init__(self, experiments, **kwargs):
        self.experiments = list(experiments)
        self.kwargs = dict(kwargs)
        self.run_calls = []
        self.raise_on_run = None
        _FakeBatch.instances.append(self)

    def run(self, **kwargs):
        self.run_calls.append(dict(kwargs))
        if self.raise_on_run is not None:
            raise self.raise_on_run
        return _FakeData(len(self.experiments))


class _FakeExperiment:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def _fake_api():
    return SimpleNamespace(
        Session=_FakeSession, SamplerV2=_FakeSampler,
        BatchExperiment=_FakeBatch, ParallelExperiment=_FakeExperiment,
        LocalReadoutError=_FakeExperiment, StandardRB=_FakeExperiment,
        T1=_FakeExperiment, T2Hahn=_FakeExperiment)


@pytest.fixture(autouse=True)
def _clean_runtime_cache():
    """`preload_runtime` caches the resolved stack process-wide, so a fake
    left behind by one test would be handed to the next. Clear it either
    side."""
    bench._reset_runtime_cache()
    yield
    bench._reset_runtime_cache()


@pytest.fixture
def api(monkeypatch):
    for cls in (_FakeSession, _FakeSampler, _FakeBatch):
        cls.instances = []
    monkeypatch.setattr(bench, '_runtime_api', _fake_api)
    return _fake_api()


def _plan(level=bench.LEVEL_TARGETED):
    targets = bench.Targets([0, 1, 2], [(0, 1), (1, 2)])
    return bench.build_plan(level, targets, 8, [(i, i + 1) for i in range(7)])


# ── the call shapes ─────────────────────────────────────────────────

def test_session_is_constructed_with_backend_only(api):
    """`Session(backend=...)` and nothing else.

    qiskit-ibm-runtime 0.48 removed `service=`; the backend carries its own
    service. Any extra kwarg here is the bug that broke the first live run.
    """
    backend = object()
    runner = bench.make_runtime_runner(backend)
    runner(_plan())

    assert len(_FakeSession.instances) == 1
    session = _FakeSession.instances[0]
    assert session.kwargs == {'backend': backend}, (
        f'Session must take backend= only, got {session.kwargs}')
    assert 'service' not in session.kwargs


def test_sampler_is_constructed_with_mode_session(api):
    runner = bench.make_runtime_runner(object())
    runner(_plan())

    assert len(_FakeSampler.instances) == 1
    sampler = _FakeSampler.instances[0]
    assert set(sampler.kwargs) == {'mode'}, sampler.kwargs
    assert sampler.kwargs['mode'] is _FakeSession.instances[0]


def test_batch_is_run_with_sampler_and_shots(api):
    plan = _plan()
    runner = bench.make_runtime_runner(object())
    runner(plan)

    batch = _FakeBatch.instances[0]
    assert len(batch.run_calls) == 1
    call = batch.run_calls[0]
    assert call['sampler'] is _FakeSampler.instances[0]
    assert call['shots'] == 1000 == plan.shots
    assert 'backend' not in call, (
        'the sampler carries the backend now; passing it again is the old API')


def test_results_are_collected_per_experiment(api):
    """Every spec gets an entry, keyed by its index in the PLAN.

    The runner submits in two phases (non-2Q first, so its 1Q EPGs can be fed
    to the 2Q analyses), so a spec's position within its batch is not its
    position in the plan — the mapping must be by plan index.
    """
    plan = _plan()
    runner = bench.make_runtime_runner(object())
    out = runner(plan)

    indices = {k for k in out if isinstance(k, int)}
    assert indices == set(range(len(plan.specs)))
    assert all(out[i] is not None for i in indices)


def test_session_closes_even_when_the_run_raises(api):
    """A leaked session keeps holding a reservation — `with` must cover it."""
    runner = bench.make_runtime_runner(object())

    original_init = _FakeBatch.__init__

    def _exploding_init(self, experiments, **kwargs):
        original_init(self, experiments, **kwargs)
        self.raise_on_run = RuntimeError('device went away')

    _FakeBatch.__init__ = _exploding_init
    try:
        with pytest.raises(RuntimeError, match='device went away'):
            runner(_plan())
    finally:
        _FakeBatch.__init__ = original_init

    assert len(_FakeSession.instances) == 1
    session = _FakeSession.instances[0]
    assert session.entered is True
    assert session.exited is True, 'the session leaked on the failure path'
    assert session.exit_exc is RuntimeError


def test_experiments_are_built_for_every_spec(api):
    """Across both phases, every spec is built exactly once."""
    plan = _plan(bench.LEVEL_FULL)
    runner = bench.make_runtime_runner(object())
    runner(plan)

    built = sum(len(b.experiments) for b in _FakeBatch.instances)
    assert built == len(plan.specs)
    assert all(b.kwargs == {'flatten_results': False}
               for b in _FakeBatch.instances)


def test_two_qubit_rb_is_submitted_after_one_qubit_rb(api):
    """Ordering is the whole point of the phase split.

    The 2Q analyses can only subtract single-qubit error if the 1Q EPGs exist
    when they are CONSTRUCTED, so 1Q RB must be a separate, earlier submission.
    """
    plan = _plan(bench.LEVEL_FULL)
    runner = bench.make_runtime_runner(object())
    runner(plan)

    assert len(_FakeBatch.instances) == 2, 'expected a two-phase submission'
    first, second = _FakeBatch.instances
    assert len(first.experiments) + len(second.experiments) == len(plan.specs)
    # The 2Q layers are all in the SECOND submission.
    n_2q = len([s for s in plan.specs if s.kind == 'rb2q_layer'])
    assert len(second.experiments) == n_2q


def test_runner_is_not_built_when_the_extra_is_missing(monkeypatch):
    def _no_stack():
        raise ImportError(bench._INSTALL_HINT)

    monkeypatch.setattr(bench, '_runtime_api', _no_stack)
    with pytest.raises(ImportError, match=r'i-qmapper\[benchmark\]'):
        bench.make_runtime_runner(object())


# ── plan gating, including the live-observed 'on-prem' ──────────────

@pytest.mark.parametrize('plan_name', [
    'on-prem', 'On-Prem', 'ON-PREM', 'onprem', 'premium', 'Premium',
    'IBM Dedicated', 'internal',
])
def test_unmetered_plans_are_unrestricted(plan_name):
    """A live run reported `plan: on-prem` and was gated as if it were Open."""
    for level in (1, 2, 3):
        gate = bench.plan_gate(plan_name, level)
        assert gate.allowed is True, (plan_name, level)
        assert gate.warning is None, (plan_name, level)


@pytest.mark.parametrize('plan_name', ['open', 'Open Plan', '', None, 'pay-go'])
def test_everything_else_keeps_the_conservative_open_rules(plan_name):
    assert bench.plan_gate(plan_name, 1).allowed is True
    assert bench.plan_gate(plan_name, 1).warning is None
    assert bench.plan_gate(plan_name, 2).warning is not None
    assert bench.plan_gate(plan_name, 3).allowed is False


def test_level3_still_needs_acknowledgement_on_an_unmetered_plan():
    """Unmetered is not unlimited: ~4 minutes of QPU is worth a confirmation."""
    assert bench.requires_acknowledgement('on-prem', 3) is True
    assert bench.requires_acknowledgement('on-prem', 2) is False


# ── preload: the import must happen on the MAIN thread ──────────────

def test_preload_is_idempotent_and_caches(monkeypatch):
    """Only the first call resolves; the rest are cache reads."""
    calls = []

    def _recording_api():
        calls.append(1)
        return _fake_api()

    monkeypatch.setattr(bench, '_runtime_api', _recording_api)

    assert bench.runtime_ready() is False
    first = bench.preload_runtime()
    assert bench.runtime_ready() is True
    second = bench.preload_runtime()
    assert first is second
    assert len(calls) == 1, 'preload_runtime must resolve exactly once'


def test_make_runtime_runner_reads_the_cache_not_the_importer(monkeypatch):
    """Once preloaded, building a runner must not re-import anything."""
    calls = []

    def _recording_api():
        calls.append(1)
        return _fake_api()

    monkeypatch.setattr(bench, '_runtime_api', _recording_api)
    bench.preload_runtime()
    assert len(calls) == 1

    bench.make_runtime_runner(object())
    assert len(calls) == 1, 'the runner re-resolved an already-loaded stack'


def test_a_failed_preload_is_not_cached(monkeypatch):
    """A missing extra must stay reportable, not poison the cache."""
    def _boom():
        raise ImportError(bench._INSTALL_HINT)

    monkeypatch.setattr(bench, '_runtime_api', _boom)
    for _ in range(2):
        with pytest.raises(ImportError, match=r'i-qmapper\[benchmark\]'):
            bench.preload_runtime()
    assert bench.runtime_ready() is False
