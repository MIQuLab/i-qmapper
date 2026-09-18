"""
ui.benchmark_panel.py — sidebar section "5. Benchmark".

Wires `iqmapper.benchmark` to widgets. The engine stays pure and injectable;
everything network-, thread- or widget-shaped lives here.

It sits in the sidebar rather than in the Auto tab because benchmarking is a
property of the BACKEND, not of the auto search: it has to serve hand-placed
layouts as readily as swept ones, every circuit slot rather than one, the whole
QPU, and — via the QPU selector — the inter-QPU second panel.

Three rules this module exists to enforce, and which its tests pin:

  * A benchmark runs only from `btn_bench_run`. Nothing on the connect path,
    the refresh path or the multi-QPU scan path reaches into `benchmark`.
  * The cost is on screen before the click. The summary line re-computes on
    every change to the level, the target mode, the QPU or the candidate
    list, and the button stays disabled while the plan is empty or an
    acknowledgement the plan gate demands is unticked.
  * Everything operates on the SELECTED QPU. With QPU 2 chosen, the targets,
    the plan, the run, the epoch, the cache file and the figure refresh all
    belong to the secondary session — nothing may leak into the primary.

Threading follows `ui.inter_qpu`: the QPU work goes to a background worker via
`spawn_bg`, and every widget touch happens on the main thread in the
completion callback, because mutating a widget off the kernel thread silently
drops its comm messages.
"""
import logging

from ..html_safe import esc

_log = logging.getLogger(__name__)


def setup_benchmark_panel(ctx):
    """Wire the benchmark section. Returns a dict of helpers for tests."""
    from .. import benchmark as bench

    primary_state = ctx.state
    dd_bench_target = ctx.dd_bench_target
    txt_bench_custom = ctx.txt_bench_custom
    tb_bench_qpu = ctx.tb_bench_qpu
    dd_bench_level = ctx.dd_bench_level
    lbl_bench_summary = ctx.lbl_bench_summary
    chk_bench_ack = ctx.chk_bench_ack
    btn_bench_run = ctx.btn_bench_run
    lbl_bench_status = ctx.lbl_bench_status
    lbl_bench_reuse = ctx.lbl_bench_reuse
    btn_bench_reuse = ctx.btn_bench_reuse

    lucj_all_layouts = ctx.lucj_all_layouts
    refresh = ctx.refresh
    spawn_bg = getattr(ctx, 'bench_spawn_bg', None)
    rescore_cell = getattr(ctx, 'rescore_cell', None)
    notices_label = getattr(ctx, 'lbl_lucj_notices', None)
    # Filled by ui.inter_qpu once QPU 2 exists; None until then.
    qpu2_handles_cell = getattr(ctx, 'qpu2_handles_cell', None)
    chk_inter_qpu = getattr(ctx, 'chk_inter_qpu', None)
    # Injected for tests; production passes None and the real runner is built
    # from the live backend at click time.
    runner_factory = getattr(ctx, 'bench_runner_factory', None)

    _last_epoch = [None]

    # ── which QPU are we acting on ──────────────────────────────
    def _qpu2():
        """QPU 2's live handles, or None when it is not connected."""
        if qpu2_handles_cell is None or qpu2_handles_cell[0] is None:
            return None
        try:
            return qpu2_handles_cell[0]()
        except Exception:
            _log.debug("QPU 2 handles unavailable", exc_info=True)
            return None

    def _qpu2_available():
        return bool(getattr(chk_inter_qpu, 'value', False)) and \
            _qpu2() is not None

    def active_target():
        """(state, refresh_callable) for the QPU the selector points at.

        Everything downstream goes through this, which is what keeps a QPU-2
        benchmark out of the primary session: there is one place that decides
        whose state is being measured.
        """
        if tb_bench_qpu.value == 2:
            handles = _qpu2()
            if handles is not None:
                return handles.state, _make_qpu2_refresh(handles)
        return primary_state, refresh

    def _make_qpu2_refresh(handles):
        """Repaint QPU 2's own figure, not QPU 1's."""
        def _refresh2():
            from .. import plot as _plot
            if handles.fig is not None and handles.widgets is not None:
                _plot.refresh_plot(handles.fig, handles.state, handles.widgets)
        return _refresh2

    def _sync_qpu_selector():
        """Show the QPU toggle only while QPU 2 is actually there."""
        available = _qpu2_available()
        tb_bench_qpu.layout.display = '' if available else 'none'
        if not available and tb_bench_qpu.value != 1:
            tb_bench_qpu.value = 1

    # ── plan (subscription) detection ───────────────────────────
    def _plan_name():
        """Best-effort subscription name, cached on state for the session.

        Read-only and never blocking: qiskit-ibm-runtime has moved this around
        between versions, so every lookup is attempted defensively and any
        failure simply yields None, which `plan_gate` treats as Open.
        """
        state, _r = active_target()
        cached = getattr(state, '_plan_name_cache', None)
        if cached is not None:
            return None if cached == '__none__' else cached
        name = None
        try:
            service = getattr(state, 'active_service', None)
            if service is not None:
                for attr in ('active_account', 'instance'):
                    try:
                        got = getattr(service, attr, None)
                        got = got() if callable(got) else got
                    except Exception:
                        continue
                    if isinstance(got, dict):
                        name = got.get('plan') or got.get('plan_name')
                    elif isinstance(got, str):
                        name = got
                    if name:
                        break
                if not name:
                    try:
                        instances = service.instances()
                        if instances:
                            first = instances[0]
                            name = (first.get('plan')
                                    if isinstance(first, dict) else None)
                    except Exception:
                        pass
        except Exception:
            _log.debug("plan detection failed; assuming Open", exc_info=True)
        setattr(state, '_plan_name_cache', name or '__none__')
        return name

    # ── target modes ────────────────────────────────────────────
    def _sweep_result():
        """Sweep candidates belong to QPU 1's Auto tab; QPU 2 has none."""
        return list(lucj_all_layouts) if tb_bench_qpu.value == 1 else []

    def _sync_target_options():
        """Offer only the modes that would resolve to something."""
        state, _r = active_target()
        modes = bench.available_target_modes(state, _sweep_result())
        options = [(bench.TARGET_LABELS[m], m) for m in modes]
        if list(dd_bench_target.options) != options:
            previous = dd_bench_target.value
            dd_bench_target.options = options
            dd_bench_target.value = (
                previous if previous in modes
                else bench.default_target_mode(state, _sweep_result()))
        txt_bench_custom.layout.display = (
            '' if dd_bench_target.value == bench.TARGET_CUSTOM else 'none')

    def _current_plan():
        state, _r = active_target()
        targets = bench.select_targets(
            state, _sweep_result(), mode=dd_bench_target.value,
            custom=txt_bench_custom.value)
        n_qubits = len(getattr(state, 'qubit_ids', None) or [])
        coupling = getattr(state, 'edges_list', None) or []
        level = dd_bench_level.value or bench.LEVEL_QUICK_READOUT
        return targets, bench.build_plan(level, targets, n_qubits, coupling)

    # ── summary + gating ────────────────────────────────────────
    def refresh_summary(*_a):
        """Recompute the cost line and the button's enabled state."""
        try:
            _sync_qpu_selector()
            if not bench.have_qiskit_experiments():
                lbl_bench_summary.value = (
                    "<div style='font-size:13px;color:#8a4b00;"
                    "background:#fff3cd;padding:4px 8px;border-radius:4px;"
                    "border:1px solid #ffe08a;margin:2px 0;'>"
                    "Install the benchmark extra: "
                    "<code>pip install \"i-qmapper[benchmark]\"</code></div>")
                btn_bench_run.disabled = True
                return
            _sync_target_options()
            targets, plan = _current_plan()
            cost = bench.estimate_cost(plan)
            level = dd_bench_level.value or bench.LEVEL_QUICK_READOUT
            plan_name = _plan_name()
            gate = bench.plan_gate(plan_name, level)
            needs_ack = bench.requires_acknowledgement(plan_name, level)

            extra = ''
            if dd_bench_target.value == bench.TARGET_WHOLE:
                if level == bench.LEVEL_QUICK_READOUT:
                    extra = ("<br><span style='color:#1e8449;'>Readout is two "
                             "circuits whatever the width — whole-QPU Level 1 "
                             "is the cheap heatmap refresh.</span>")
                elif level == bench.LEVEL_FULL:
                    extra = ("<br><span style='color:#e67e22;'>Full-chip suite "
                             "≈ 4 min of QPU</span>")
                    needs_ack = True
            warn = (f"<br><span style='color:#e67e22;'>⚠ "
                    f"{esc(gate.warning)}</span>") if gate.warning else ''

            lbl_bench_summary.value = (
                f"<div style='font-size:13px;color:#1a5276;background:#f4f8fb;"
                f"padding:4px 8px;border-radius:4px;border:1px solid #d6e4f0;"
                f"margin:2px 0;'>{esc(bench.summary_line(targets, cost, plan_name))}"
                f"{extra}{warn}</div>")

            chk_bench_ack.layout.display = '' if needs_ack else 'none'
            btn_bench_run.disabled = bool(
                (not plan.specs) or (needs_ack and not chk_bench_ack.value))
            _refresh_reuse_offer()
        except Exception:
            _log.exception("Benchmark summary refresh failed")
            btn_bench_run.disabled = True

    def _refresh_reuse_offer():
        """Offer a saved epoch — per backend — instead of re-spending QPU."""
        try:
            state, _r = active_target()
            name = getattr(state, 'active_backend_name', None)
            epoch = bench.latest_reusable_epoch(name) if name else None
            _last_epoch[0] = epoch
            if epoch is None:
                lbl_bench_reuse.value = ''
                btn_bench_reuse.disabled = True
                btn_bench_reuse.layout.display = 'none'
                return
            lbl_bench_reuse.value = (
                f"<div style='font-size:13px;color:#1e8449;margin:2px 0;'>"
                f"Reuse (measured {esc(epoch.time_label())}, "
                f"L{epoch.level})</div>")
            btn_bench_reuse.disabled = False
            btn_bench_reuse.layout.display = ''
        except Exception:
            _log.debug("reuse offer unavailable", exc_info=True)

    # ── applying an epoch ───────────────────────────────────────
    def _install_epoch(epoch, *, source):
        """Apply, refresh, re-rank, surface notices. Main thread only."""
        state, refresh_fn = active_target()
        if not bench.apply_measured_epoch(state, epoch):
            lbl_bench_status.value = _err(
                'Measured epoch does not match the current device '
                '(qubit/edge count changed) — reported calibration kept.')
            return False
        try:
            refresh_fn()
        except Exception:
            _log.exception("plot refresh after benchmark failed")

        n_ranked = 0
        # The Auto candidate list belongs to QPU 1; re-ranking it against a
        # QPU-2 epoch would be meaningless (different device, different props).
        if tb_bench_qpu.value == 1 and rescore_cell is not None \
                and rescore_cell[0] is not None:
            try:
                n_ranked = rescore_cell[0]() or 0
            except Exception:
                _log.exception("candidate re-ranking after benchmark failed")

        if notices_label is not None and epoch.notices:
            notices_label.value = (
                "<div style='font-size:13px;color:#8a4b00;background:#fff3cd;"
                "padding:4px 8px;border-radius:4px;border:1px solid #ffe08a;"
                "margin:2px 0;'>⚠ " +
                "<br>".join(esc(n) for n in epoch.notices) + "</div>")

        rank_txt = f" · {n_ranked} candidate(s) re-ranked" if n_ranked else ''
        qpu_txt = f"QPU {tb_bench_qpu.value} · " if _qpu2_available() else ''
        lbl_bench_status.value = (
            f"<div style='font-size:14px;color:#1e8449;background:#eafaf1;"
            f"padding:4px 8px;border-radius:4px;border:1px solid #abebc6;"
            f"margin:3px 0;'>✅ {esc(qpu_txt)}{esc(source)}: measured "
            f"epoch applied ({epoch.n_measured_qubits}q/"
            f"{epoch.n_measured_edges}e, L{epoch.level}){rank_txt}</div>")
        _refresh_reuse_offer()
        return True

    def _err(msg):
        return (f"<div style='font-size:14px;color:#e74c3c;background:#fdedec;"
                f"padding:4px 8px;border-radius:4px;border:1px solid #f5b7b1;"
                f"margin:3px 0;'>❌ {esc(str(msg))}</div>")

    # ── run ─────────────────────────────────────────────────────
    def _on_run(_b=None):
        state, _refresh_fn = active_target()
        try:
            _targets, plan = _current_plan()
        except Exception as exc:              # noqa: BLE001 - reported
            lbl_bench_status.value = _err(exc)
            return
        if not plan.specs:
            lbl_bench_status.value = _err(
                'No qubits to measure for this target selection.')
            return

        level = plan.level
        plan_name = _plan_name()
        needs_ack = (bench.requires_acknowledgement(plan_name, level)
                     or (dd_bench_target.value == bench.TARGET_WHOLE
                         and level == bench.LEVEL_FULL))
        if needs_ack and not chk_bench_ack.value:
            lbl_bench_status.value = _err(
                'Tick "I understand this uses QPU time" to run this level.')
            return

        cost = bench.estimate_cost(plan)
        btn_bench_run.disabled = True

        # Import the quantum stack HERE, on the main thread, before any worker
        # starts. Importing it from the worker publishes half-built modules
        # into sys.modules while the kernel thread is still serving widget
        # events, and a map click mid-import then dies on "partially
        # initialized module 'pandas'". See benchmark.preload_runtime.
        if runner_factory is None:
            if not bench.runtime_ready():
                lbl_bench_status.value = (
                    "<div style='font-size:14px;color:#8a4b00;"
                    "background:#fff3cd;padding:6px 10px;border-radius:4px;"
                    "border:1px solid #ffe08a;margin:3px 0;font-weight:bold;'>"
                    "⏳ Loading benchmark libraries…</div>")
            try:
                bench.preload_runtime()
            except Exception as exc:          # noqa: BLE001 - reported
                btn_bench_run.disabled = False
                lbl_bench_status.value = _err(exc)
                return

        lbl_bench_status.value = (
            f"<div style='font-size:14px;color:#8a4b00;background:#fff3cd;"
            f"padding:6px 10px;border-radius:4px;border:1px solid #ffe08a;"
            f"margin:3px 0;font-weight:bold;'>⏳ Running L{level} "
            f"benchmark ({esc(cost.text)})…</div>")

        reported_q = dict(getattr(state, 'qubit_props_data', {}) or {})
        reported_g = dict(getattr(state, 'gate_props_data', {}) or {})
        edges = list(getattr(state, 'edges_list', None) or [])
        backend_name = getattr(state, 'active_backend_name', None)

        def _work():
            """Background: build the runner and spend the QPU time."""
            if runner_factory is not None:
                runner = runner_factory(state)
            else:
                runner = bench.make_runtime_runner(
                    getattr(state, 'active_backend', None))
            return bench.run_plan(plan, runner)

        def _done(raw):
            """Main thread: fold, apply, persist."""
            btn_bench_run.disabled = False
            try:
                epoch = bench.results_to_epoch(raw, plan, reported_q,
                                               reported_g, edges)
            except Exception as exc:          # noqa: BLE001 - reported
                _log.exception("folding benchmark results failed")
                lbl_bench_status.value = _err(exc)
                return
            if _install_epoch(epoch, source=f'L{level} benchmark') \
                    and backend_name:
                bench.save_measured_epoch(backend_name, epoch)

        def _failed(exc):
            """Main thread: the reported epoch is left exactly as it was."""
            btn_bench_run.disabled = False
            _log.exception("benchmark run failed: %s", exc)
            lbl_bench_status.value = _err(exc)

        if spawn_bg is None:                  # no runner injected: inline
            try:
                _done(_work())
            except Exception as exc:          # noqa: BLE001 - reported
                _failed(exc)
            return
        spawn_bg(_work, on_done=_done, on_error=_failed)

    def _on_reuse(_b=None):
        epoch = _last_epoch[0]
        if epoch is None:
            lbl_bench_status.value = _err('No recent measurement to reuse.')
            return
        _install_epoch(epoch, source='Reused measurement')

    btn_bench_run.on_click(_on_run)
    btn_bench_reuse.on_click(_on_reuse)
    for widget in (dd_bench_level, dd_bench_target, chk_bench_ack,
                   tb_bench_qpu, txt_bench_custom):
        widget.observe(refresh_summary, names='value')
    if chk_inter_qpu is not None:
        chk_inter_qpu.observe(refresh_summary, names='value')
    refresh_summary()

    return {
        'refresh_summary': refresh_summary,
        'run': _on_run,
        'reuse': _on_reuse,
        'plan_name': _plan_name,
        'current_plan': _current_plan,
        'active_target': active_target,
        'qpu2_available': _qpu2_available,
    }
