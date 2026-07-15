"""
async_util.py — Background-threading helper shared across modules.

`default_spawn_bg` runs a function on a daemon thread and marshals its
completion callback back onto the kernel's main event loop. Extracted from
ui.time_panel.py so non-UI callers (connection, ui.history, ui/builder) can
use it without importing a UI panel module. Imports only stdlib (tornado is a
soft, lazy secondary), so it can never participate in an import cycle with the
rest of the package.

Threading contract (the whole point of this module):
ipywidgets value writes are advertised as thread-safe, but
plotly.FigureWidget.batch_update() — invoked by refresh_plot on the
completion callback — silently drops its comm mutations when run from a
worker thread, giving an empty/ghost map LIVE (invisible headless, which has
no frontend). So the completion callback MUST run on the thread that owns the
kernel loop. We guarantee that by capturing the main loop + main-thread
identity ONCE, on the main thread, at setup time (:func:`init_main_loop`,
called from build()), and scheduling onto THAT captured loop from workers —
never resolving a loop from the calling worker (which has none).
"""
import asyncio
import logging
import threading

_log = logging.getLogger(__name__)


# Captured ONCE on the MAIN thread by init_main_loop(). This (thread, loop)
# pair is the ONLY scheduling state — no per-callsite resolution, no sticky
# "inline" cache that a worker could inherit. `_MAIN_LOOP` stays None headless
# (no running loop) → marshal runs inline, warning-free.
_MAIN_THREAD = None       # threading.Thread that owns the kernel
_MAIN_LOOP = None         # captured asyncio loop or tornado IOLoop (or None)
_MAIN_LOOP_KIND = None    # 'asyncio' | 'tornado' | None


def _capture_main_loop():
    """Return (loop, kind) for the loop owned by THIS (main) thread, or
    (None, None). Tiered, because the live capture happens inside a WIDGET
    CALLBACK (launch_mapper → build), NOT top-level cell code — so
    ``get_running_loop()`` often fails there even though the kernel loop is
    running and owned by this thread.

      1. ``get_running_loop()``  — we're inside a coroutine/callback ON it;
      2. the thread's ALREADY-SET event loop, IF ``is_running()`` — the
         ipykernel case: the thread owns the running kernel loop but we're in a
         widget-callback context, not its coroutine. Read via the policy's
         thread-local so we never CREATE (and leak) a loop when none is set —
         which would trip ResourceWarning under -W error headless;
      3. a current tornado ``IOLoop`` (wraps the same asyncio loop).
    Headless (no running/set loop, no IOLoop) → (None, None) → marshal inline.
    """
    # 1. running loop
    try:
        return asyncio.get_running_loop(), 'asyncio'
    except RuntimeError:
        pass
    # 2. the thread's ALREADY-SET, running event loop (live ipykernel main
    #    thread) — read the policy thread-local, never call get_event_loop()
    #    (which creates+leaks a loop when none is set).
    try:
        policy = asyncio.get_event_loop_policy()
        local = getattr(policy, '_local', None)
        loop = getattr(local, '_loop', None)
        if loop is not None and loop.is_running():
            return loop, 'asyncio'
    except Exception:
        _log.debug("asyncio set-loop probe failed", exc_info=True)
    # 3. tornado IOLoop (optional dependency); probed WITHOUT creating one.
    try:
        import warnings
        from tornado.ioloop import IOLoop
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            loop = IOLoop.current(instance=False)
        if loop is not None:
            return loop, 'tornado'
    except Exception:
        _log.debug("tornado IOLoop probe failed", exc_info=True)
    return None, None


def init_main_loop():
    """Capture the kernel main loop + main-thread identity — CAPTURE-ONCE,
    MAIN-ONLY.

    Call at true startup (run()) and from the UI build. Two guards make it safe
    to call from anywhere, any number of times — critical because the SECONDARY
    build re-enters build() and, if that ever runs on a worker, must NOT clobber
    a good main-thread capture with None (the live empty-map cause: a worker
    capture wrote ``loop=None`` over the real loop → marshal fell back to
    inline-on-worker):

      * if a loop is ALREADY captured → return (never overwrite a good value);
      * if NOT on the main thread → return (a worker owns no loop; capturing
        there would only ever write None). Logged so the live toggle shows it.

    Otherwise capture via the tiered :func:`_capture_main_loop` and log an INFO
    line so the live toggle shows thread + whether a loop was actually found.
    Headless leaves the loop None (and retries on the next main-thread call),
    so :func:`marshal_to_main` runs inline, warning-free.
    """
    global _MAIN_THREAD, _MAIN_LOOP, _MAIN_LOOP_KIND
    if _MAIN_LOOP is not None:
        return                                   # already captured — never redo
    if threading.current_thread() is not threading.main_thread():
        _log.debug("capture skipped: not on main thread (%s)",
                   threading.current_thread().name)
        return
    _MAIN_THREAD = threading.current_thread()
    _MAIN_LOOP, _MAIN_LOOP_KIND = _capture_main_loop()
    _log.info("main loop captured: thread=%s loop=%s kind=%s",
              _MAIN_THREAD.name, _MAIN_LOOP is not None, _MAIN_LOOP_KIND)


def _schedule_on_captured_loop(cb):
    """Thread-safe schedule of ``cb`` onto the CAPTURED main loop."""
    if _MAIN_LOOP_KIND == 'asyncio':
        _MAIN_LOOP.call_soon_threadsafe(cb)
    else:                                    # tornado IOLoop
        _MAIN_LOOP.add_callback(cb)


def marshal_to_main(cb):
    """Run zero-arg ``cb`` on the kernel main loop.

    Resolution (uses the pair captured by :func:`init_main_loop`, NEVER the
    calling thread's context):

      1. already on the captured main thread → run inline (correct + warning-
         free — this is the manual-button and headless-main path);
      2. off-thread with a captured loop → schedule onto THAT loop
         (``call_soon_threadsafe`` / ``add_callback``) so the callback lands on
         the main thread — the live worker→main hand-off;
      3. no captured loop (headless, or init_main_loop never ran) → inline.

    Safe to call from any thread.
    """
    if _MAIN_THREAD is not None and threading.current_thread() is _MAIN_THREAD:
        _log.debug("marshal_to_main: inline (already on main thread)")
        cb()
        return
    if _MAIN_LOOP is not None:
        try:
            _schedule_on_captured_loop(cb)
            _log.debug("marshal_to_main: scheduled to captured loop (%s)",
                       _MAIN_LOOP_KIND)
            return
        except Exception:
            _log.exception("marshal_to_main: scheduling onto the captured loop "
                           "failed; running inline")
    else:
        _log.debug("marshal_to_main: inline (no captured loop)")
    cb()


def default_spawn_bg(fn, on_done=None, on_error=None):
    """Run `fn()` in a daemon thread and marshal its completion callback
    (`on_done` or `on_error`) onto the kernel main loop via marshal_to_main.

    Callers can substitute their own spawn_bg (e.g. queue + Play-timer) if
    they need finer control over scheduling."""
    def _safe(cb, arg):
        try:
            cb(arg)
        except Exception:
            _log.exception("Callback raised")

    def _worker():
        try:
            result = fn()
        except Exception as exc:
            _log.exception("Background op raised")
            if on_error:
                marshal_to_main(lambda err=exc: _safe(on_error, err))
            return
        if on_done:
            marshal_to_main(lambda: _safe(on_done, result))

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return t
