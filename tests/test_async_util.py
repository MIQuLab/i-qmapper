"""Regression tests for iqmapper.async_util.

Two invariants pinned here:

1. **Cross-thread marshalling reaches the MAIN loop** (the live empty-map root
   cause). ``marshal_to_main`` captures the main loop + main-thread identity at
   setup and, from a worker, schedules onto THAT loop — it never resolves a
   loop from the calling worker (which has none) and inline-runs the paint on
   the worker. Headless simulates this by running a real asyncio loop on a
   dedicated thread and marshalling from another thread.

2. **Deferred-scheduler exc binding** (historical bug): ``_worker`` schedules
   ``lambda err=exc: _safe(on_error, err)`` — binding ``exc`` as a default so a
   DEFERRED scheduler (which runs the lambda after the except block cleared
   ``exc``) still delivers it. Driven via a deferred marshal_to_main.
"""
import threading

import pytest

import iqmapper.async_util as async_util


@pytest.fixture
def _capture_isolation():
    """Save/restore the captured (main thread, loop, kind) triple, and RESET it
    to the clean (None, None, None) so each test starts fresh — capture-once
    would otherwise let one test's capture short-circuit the next."""
    saved = (async_util._MAIN_THREAD, async_util._MAIN_LOOP,
             async_util._MAIN_LOOP_KIND)
    async_util._MAIN_THREAD = None
    async_util._MAIN_LOOP = None
    async_util._MAIN_LOOP_KIND = None
    try:
        yield
    finally:
        (async_util._MAIN_THREAD, async_util._MAIN_LOOP,
         async_util._MAIN_LOOP_KIND) = saved


def _join(thread):
    thread.join(timeout=2.0)
    assert not thread.is_alive(), "worker thread did not finish"


# ── M3 t1: THE live-shaped regression — worker → captured main loop ─────────

def test_marshal_schedules_onto_captured_loop_from_worker(monkeypatch,
                                                          _capture_isolation):
    """A real asyncio loop runs on a dedicated 'main' thread; marshal_to_main
    invoked from ANOTHER (worker) thread must run the callback ON THE LOOP's
    thread, never inline on the worker. Impossible to fail before this pass —
    nothing simulated the worker-with-no-loop hand-off."""
    import asyncio
    import queue

    loop = asyncio.new_event_loop()
    started = threading.Event()

    def _run_loop():
        asyncio.set_event_loop(loop)
        loop.call_soon(started.set)
        loop.run_forever()

    loop_thread = threading.Thread(target=_run_loop, name='fake-main-loop',
                                   daemon=True)
    loop_thread.start()
    assert started.wait(2.0), "fake main loop did not start"

    # Capture, exactly as init_main_loop would on that thread.
    monkeypatch.setattr(async_util, '_MAIN_THREAD', loop_thread)
    monkeypatch.setattr(async_util, '_MAIN_LOOP', loop)
    monkeypatch.setattr(async_util, '_MAIN_LOOP_KIND', 'asyncio')
    try:
        ran_on = queue.Queue()
        called_from = queue.Queue()

        def _cb():
            ran_on.put(threading.current_thread().ident)

        def _worker():
            called_from.put(threading.current_thread().ident)
            async_util.marshal_to_main(_cb)

        w = threading.Thread(target=_worker, name='worker', daemon=True)
        w.start()
        _join(w)

        landed = ran_on.get(timeout=2.0)
        assert landed == loop_thread.ident            # ran on the MAIN loop
        assert landed != called_from.get_nowait()     # NOT on the worker
    finally:
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(2.0)
        loop.close()


# ── M3 t3: caller already on the captured main thread → inline fast path ────

def test_marshal_inline_on_main_thread(monkeypatch, _capture_isolation):
    # A captured loop that MUST NOT be used from the main-thread caller.
    class _ExplodingLoop:
        def call_soon_threadsafe(self, cb):        # pragma: no cover
            raise AssertionError("must not schedule when already on main")

    monkeypatch.setattr(async_util, '_MAIN_THREAD', threading.current_thread())
    monkeypatch.setattr(async_util, '_MAIN_LOOP', _ExplodingLoop())
    monkeypatch.setattr(async_util, '_MAIN_LOOP_KIND', 'asyncio')

    ran = []
    async_util.marshal_to_main(lambda: ran.append(threading.current_thread().ident))
    assert ran == [threading.current_thread().ident]   # inline, on this thread


# ── M3 t2: headless (no captured loop) → inline, zero warnings ──────────────

def test_init_main_loop_headless_no_loop_no_warning(monkeypatch,
                                                    _capture_isolation):
    import warnings

    # Fake tornado whose IOLoop.current() would emit the Py>=3.10 deprecation
    # with no running loop — init_main_loop must NOT let it escape.
    import sys
    import types

    class _IOLoop:
        @staticmethod
        def current(instance=True):
            import asyncio as _a
            import warnings as _w
            try:
                _a.get_running_loop()
            except RuntimeError:
                _w.warn("There is no current event loop", DeprecationWarning)
            return None                       # no current IOLoop headless

    tornado = types.ModuleType("tornado")
    ioloop = types.ModuleType("tornado.ioloop")
    ioloop.IOLoop = _IOLoop
    tornado.ioloop = ioloop
    monkeypatch.setitem(sys.modules, "tornado", tornado)
    monkeypatch.setitem(sys.modules, "tornado.ioloop", ioloop)

    import asyncio
    prev = _policy_set_loop()
    asyncio.set_event_loop(None)                     # no set loop → tier-2 skips
    try:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            async_util.init_main_loop()
    finally:
        asyncio.set_event_loop(prev)

    assert [str(x.message) for x in rec] == []      # deprecation never escapes
    assert async_util._MAIN_LOOP is None             # no loop captured headless

    # And marshal on this (captured main) thread runs inline.
    ran = []
    async_util.marshal_to_main(lambda: ran.append(1))
    assert ran == [1]


def test_init_main_loop_captures_running_loop(_capture_isolation):
    """When a loop IS running on the calling thread, init captures it as the
    asyncio main loop (the live cell-execution case)."""
    import asyncio

    async def _main():
        async_util.init_main_loop()
        return async_util._MAIN_LOOP, async_util._MAIN_LOOP_KIND

    loop, kind = asyncio.run(_main())
    assert loop is not None and kind == 'asyncio'


# ── N3: capture sources — the live ipykernel case + tornado last resort ─────

def _policy_set_loop():
    import asyncio
    policy = asyncio.get_event_loop_policy()
    return getattr(getattr(policy, '_local', None), '_loop', None)


def test_capture_via_set_loop_when_not_in_coroutine(monkeypatch,
                                                    _capture_isolation):
    """THE live fix: get_running_loop() fails on the main thread (capture runs
    in a widget callback — launch_mapper on Connect — not a coroutine), but the
    thread OWNS a running loop set via set_event_loop. Capture must pick it up
    from the policy thread-local (previously it gave up → None → worker inline)."""
    import asyncio
    loop = asyncio.new_event_loop()
    started = threading.Event()

    def _run():
        loop.call_soon_threadsafe(started.set)
        loop.run_forever()
    t = threading.Thread(target=_run, name='kernel', daemon=True)
    t.start()
    assert started.wait(2.0)

    prev = _policy_set_loop()
    asyncio.set_event_loop(loop)                     # policy._local._loop = loop
    monkeypatch.setattr(asyncio, 'get_running_loop',
                        lambda: (_ for _ in ()).throw(RuntimeError('widget callback')))
    try:
        async_util.init_main_loop()
        assert async_util._MAIN_LOOP is loop            # captured the RUNNING loop
        assert async_util._MAIN_LOOP_KIND == 'asyncio'  # via the set-loop tier
    finally:
        asyncio.set_event_loop(prev)                    # restore prior loop
        loop.call_soon_threadsafe(loop.stop)
        t.join(2.0)
        loop.close()


def test_capture_via_tornado_last_resort(monkeypatch, _capture_isolation):
    """No running loop, no running SET loop, but a current tornado IOLoop
    exists → capture it (kind='tornado')."""
    import asyncio
    import sys
    import types

    monkeypatch.setattr(asyncio, 'get_running_loop',
                        lambda: (_ for _ in ()).throw(RuntimeError('no loop')))
    prev = _policy_set_loop()
    asyncio.set_event_loop(None)                     # no set loop → tier-2 skips

    sentinel = object()

    class _IOLoop:
        @staticmethod
        def current(instance=True):
            return sentinel
    tornado = types.ModuleType('tornado')
    ioloop = types.ModuleType('tornado.ioloop')
    ioloop.IOLoop = _IOLoop
    tornado.ioloop = ioloop
    monkeypatch.setitem(sys.modules, 'tornado', tornado)
    monkeypatch.setitem(sys.modules, 'tornado.ioloop', ioloop)
    try:
        async_util.init_main_loop()
        assert async_util._MAIN_LOOP is sentinel
        assert async_util._MAIN_LOOP_KIND == 'tornado'
    finally:
        asyncio.set_event_loop(prev)


# ── N4: WIRING pin — the real UI build captures, then a worker schedules ────

def _build_ui(monkeypatch):
    import numpy as np

    def _fake_device(state, vendor, chip, date, prefetched_props=None):
        n = 8
        return (list(range(n)), [0.0] * n, [(i, i + 1) for i in range(n - 1)],
                [str(i) for i in range(n)],
                np.array([0.01 * (i + 1) for i in range(n)]),
                np.array([0.001 * (i + 1) for i in range(n - 1)]))

    monkeypatch.setattr("iqmapper.data.generate_device_data", _fake_device)
    from iqmapper.state import SessionState
    from iqmapper.ui.builder import MapperUIBuilder
    state = SessionState()
    state.active_backend = object()
    state.active_backend_name = 'stub'
    state.active_chip = 'chip'
    state.active_service = None
    state.available_backends = []
    state.props_warning = None
    state.initial_props = object()
    state.active_ansatz = 'lucj'
    state.active_vendor = 'IBM'
    state.active_coords = []
    state.custom_colors = []
    state.cal_history_dates = ['_seed_']
    return MapperUIBuilder(state).build()


def test_build_invokes_capture(monkeypatch, _capture_isolation):
    # Reset to a sentinel to PROVE build() re-runs init_main_loop().
    monkeypatch.setattr(async_util, '_MAIN_THREAD', None)
    monkeypatch.setattr(async_util, '_MAIN_LOOP', None)
    _build_ui(monkeypatch)
    assert async_util._MAIN_THREAD is threading.current_thread()   # wiring ran


def test_build_captures_loop_then_worker_schedules(monkeypatch, _capture_isolation):
    """Integration-shaped: build the real UI from within a running loop (so the
    build's init_main_loop captures it), then marshal from a WORKER and assert
    the callback lands on the captured-loop thread — the 'scheduled to captured
    loop' branch, pinned end-to-end through the wiring."""
    import asyncio
    import queue

    async def _main():
        _build_ui(monkeypatch)                       # build() → init_main_loop()
        assert async_util._MAIN_LOOP is asyncio.get_running_loop()

        ran_on = queue.Queue()

        def _cb():
            ran_on.put(threading.current_thread().ident)

        w = threading.Thread(target=lambda: async_util.marshal_to_main(_cb),
                             daemon=True)
        w.start()
        w.join(2.0)
        for _ in range(200):                         # let the loop run the cb
            if not ran_on.empty():
                break
            await asyncio.sleep(0.005)
        assert ran_on.get_nowait() == threading.get_ident()   # ran on main loop

    asyncio.run(_main())


# ── P3: capture-once + main-only — a worker must NEVER overwrite the capture ─

def test_worker_capture_does_not_overwrite_good_capture(monkeypatch,
                                                        _capture_isolation):
    """THE live bug as a fixture: a good main-thread capture must survive a
    later init_main_loop() reached FROM A WORKER (the secondary build
    re-entering build() off-thread). Previously that wrote thread=<worker>,
    loop=None over the real capture → marshal fell back to inline-on-worker."""
    sentinel_loop = object()
    main_t = threading.current_thread()
    monkeypatch.setattr(async_util, '_MAIN_THREAD', main_t)
    monkeypatch.setattr(async_util, '_MAIN_LOOP', sentinel_loop)
    monkeypatch.setattr(async_util, '_MAIN_LOOP_KIND', 'asyncio')

    seen = []

    def _worker():
        async_util.init_main_loop()                 # reached from a worker
        seen.append(threading.current_thread())
    w = threading.Thread(target=_worker, name='Thread-8 (_worker)', daemon=True)
    w.start()
    _join(w)

    assert async_util._MAIN_LOOP is sentinel_loop    # capture preserved, not None
    assert async_util._MAIN_THREAD is main_t         # main-thread identity kept


def test_worker_capture_skipped_when_none_captured(monkeypatch, _capture_isolation):
    """With NO capture yet, a worker call must SKIP — never write None (or the
    worker thread) as the 'main' capture."""
    monkeypatch.setattr(async_util, '_MAIN_THREAD', None)
    monkeypatch.setattr(async_util, '_MAIN_LOOP', None)
    monkeypatch.setattr(async_util, '_MAIN_LOOP_KIND', None)

    def _worker():
        async_util.init_main_loop()
    w = threading.Thread(target=_worker, name='worker', daemon=True)
    w.start()
    _join(w)

    assert async_util._MAIN_LOOP is None
    assert async_util._MAIN_THREAD is None           # worker never became 'main'


def test_secondary_build_on_worker_still_marshals_to_main(monkeypatch,
                                                          _capture_isolation):
    """End-to-end: after a GOOD main capture (as run() produces), a worker that
    re-enters init_main_loop (skipped) and then marshals the paint continuation
    still lands it ON THE CAPTURED LOOP's thread — not inline on the worker."""
    import asyncio
    import queue

    loop = asyncio.new_event_loop()
    started = threading.Event()

    def _run():
        loop.call_soon_threadsafe(started.set)
        loop.run_forever()
    main_like = threading.Thread(target=_run, name='KernelMainLoop', daemon=True)
    main_like.start()
    assert started.wait(2.0)

    # Good capture, exactly as run() on the main thread would produce.
    monkeypatch.setattr(async_util, '_MAIN_THREAD', main_like)
    monkeypatch.setattr(async_util, '_MAIN_LOOP', loop)
    monkeypatch.setattr(async_util, '_MAIN_LOOP_KIND', 'asyncio')
    try:
        ran_on = queue.Queue()

        def _secondary_worker():
            async_util.init_main_loop()              # re-entry on worker → SKIP
            assert async_util._MAIN_LOOP is loop      # capture still good
            async_util.marshal_to_main(
                lambda: ran_on.put(threading.current_thread().ident))
        w = threading.Thread(target=_secondary_worker, name='Thread-8 (_worker)',
                             daemon=True)
        w.start()
        _join(w)

        assert ran_on.get(timeout=2.0) == main_like.ident   # paint on captured loop
    finally:
        loop.call_soon_threadsafe(loop.stop)
        main_like.join(2.0)
        loop.close()


# ── t4: default_spawn_bg deferred-scheduler behaviour (exc binding etc.) ────

def test_on_error_receives_exception_with_deferred_scheduler(monkeypatch,
                                                             _capture_isolation):
    """Deferred marshal: with the OLD code the lambda resolved `exc` after the
    except block cleared it → NameError swallowed in _safe → on_error never
    ran. The `err=exc` default binding fixes it."""
    queued = []
    monkeypatch.setattr(async_util, 'marshal_to_main', queued.append)

    captured = []

    def _boom():
        raise ValueError("kaboom")

    t = async_util.default_spawn_bg(_boom, on_error=lambda e: captured.append(e))
    _join(t)                                     # except block has exited
    assert queued, "on_error callback should have been queued"
    for cb in queued:
        cb()

    assert len(captured) == 1 and isinstance(captured[0], ValueError)
    assert str(captured[0]) == "kaboom"


def test_on_done_receives_result_with_deferred_scheduler(monkeypatch,
                                                         _capture_isolation):
    queued = []
    monkeypatch.setattr(async_util, 'marshal_to_main', queued.append)
    sentinel = object()
    captured = []

    t = async_util.default_spawn_bg(lambda: sentinel,
                                    on_done=lambda r: captured.append(r))
    _join(t)
    for cb in queued:
        cb()
    assert captured == [sentinel]


def test_inline_marshal_delivers_on_error(monkeypatch, _capture_isolation):
    """Inline marshal (headless-style) runs the callback synchronously; on_error
    still fires with the exception."""
    monkeypatch.setattr(async_util, 'marshal_to_main', lambda cb: cb())
    captured = []

    t = async_util.default_spawn_bg(
        lambda: (_ for _ in ()).throw(RuntimeError("inline-boom")),
        on_error=lambda e: captured.append(e))
    _join(t)
    assert len(captured) == 1 and isinstance(captured[0], RuntimeError)
    assert str(captured[0]) == "inline-boom"


def test_callback_exception_is_contained(monkeypatch, _capture_isolation):
    """_safe wraps the user callback so its exception never escapes the worker
    or the scheduler."""
    queued = []
    monkeypatch.setattr(async_util, 'marshal_to_main', queued.append)

    def _raises(_r):
        raise RuntimeError("callback failure — must be contained by _safe")

    t = async_util.default_spawn_bg(lambda: 1, on_done=_raises)
    _join(t)
    for cb in queued:                            # draining must not propagate
        cb()
