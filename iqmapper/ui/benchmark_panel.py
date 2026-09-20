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
    chk_bench_absolute = getattr(ctx, 'chk_bench_absolute', None)
    chk_err_threshold = getattr(ctx, 'chk_err_threshold', None)
    txt_g_thr_max = getattr(ctx, 'txt_g_thr_max', None)
    dd_gate_prop = getattr(ctx, 'dd_gate_prop', None)
    lbl_bench_consistency = getattr(ctx, 'lbl_bench_consistency', None)
    lbl_bench_details = getattr(ctx, 'lbl_bench_details', None)
    acc_bench_details = getattr(ctx, 'acc_bench_details', None)
    btn_show_on_map = getattr(ctx, 'btn_bench_show_on_map', None)
    #: (gate_key, threshold, n_edges) the map link would apply.
    _map_target = [None]
    lbl_bench_reuse = ctx.lbl_bench_reuse
    btn_bench_reuse = ctx.btn_bench_reuse
    btn_bench_reanalyze = getattr(ctx, 'btn_bench_reanalyze', None)
    btn_bench_calcheck = getattr(ctx, 'btn_bench_calcheck', None)
    lbl_bench_calcheck = getattr(ctx, 'lbl_bench_calcheck', None)

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

            # A plan that cannot be split under IBM's per-job shot limit must
            # not be submittable: sending it costs a round trip and, before
            # v3, ended in a false success. Computed BEFORE the summary is
            # written so the reason is visible, not just a greyed button.
            ok, reason = bench.plan_is_submittable(plan, plan.shots)
            if not ok:
                extra += (f"<br><span style='color:#e74c3c;'>\u274c "
                          f"{esc(reason)}</span>")

            lbl_bench_summary.value = (
                f"<div style='font-size:13px;color:#1a5276;background:#f4f8fb;"
                f"padding:4px 8px;border-radius:4px;border:1px solid #d6e4f0;"
                f"margin:2px 0;'>{esc(bench.summary_line(targets, cost, plan_name))}"
                f"{extra}{warn}</div>")

            _sync_calcheck(targets, plan_name, level)
            chk_bench_ack.layout.display = '' if needs_ack else 'none'
            btn_bench_run.disabled = bool(
                (not plan.specs) or (not ok)
                or (needs_ack and not chk_bench_ack.value))
            _refresh_reuse_offer()
        except Exception:
            _log.exception("Benchmark summary refresh failed")
            btn_bench_run.disabled = True

    def _calcheck_plans():
        """(parallel, sequential) over the first N target edges, or None."""
        state, _r = active_target()
        targets = bench.select_targets(
            state, _sweep_result(), mode=dd_bench_target.value,
            custom=txt_bench_custom.value)
        if len(targets.edges) < bench.CALIBRATION_CHECK_EDGES:
            return None, None, targets
        par, seq = bench.build_calibration_check_plans(
            targets, len(getattr(state, 'qubit_ids', None) or []),
            list(getattr(state, 'edges_list', None) or []),
            n_edges=bench.CALIBRATION_CHECK_EDGES)
        return par, seq, targets

    def _sync_calcheck(targets, plan_name, level):
        """Cost line and enablement for the check, before the click."""
        if btn_bench_calcheck is None:
            return
        n_edges = len(targets.edges)
        if n_edges < bench.CALIBRATION_CHECK_EDGES:
            btn_bench_calcheck.disabled = True
            if lbl_bench_calcheck is not None:
                lbl_bench_calcheck.value = (
                    f"<div style='font-size:12px;color:#6c757d;margin:2px 0;'>"
                    f"Calibration check needs at least "
                    f"{bench.CALIBRATION_CHECK_EDGES} target edges "
                    f"(have {n_edges}).</div>")
            return
        par, seq, _t = _calcheck_plans()
        if par is None:
            btn_bench_calcheck.disabled = True
            return
        circuits = (bench.plan_circuit_count(par)
                    + bench.plan_circuit_count(seq))
        shots = bench.plan_shots(par, par.shots) + bench.plan_shots(
            seq, seq.shots)
        jobs = (len(bench.split_for_submission(par, par.shots))
                + len(bench.split_for_submission(seq, seq.shots)))
        seconds = (bench.estimate_cost(par).qpu_seconds
                   + bench.estimate_cost(seq).qpu_seconds)
        # The check spends QPU time like any run, so the same gate applies.
        needs_ack = bench.requires_acknowledgement(plan_name,
                                                   bench.LEVEL_TARGETED)
        btn_bench_calcheck.disabled = bool(
            needs_ack and not chk_bench_ack.value)
        if lbl_bench_calcheck is not None:
            lbl_bench_calcheck.value = (
                f"<div style='font-size:12px;color:#1a5276;margin:2px 0;'>"
                f"Calibration check: "
                f"{bench.CALIBRATION_CHECK_EDGES} edges \u00d7 2 arms \u00b7 "
                f"~{circuits} circuits \u00b7 "
                f"{esc(bench._format_shots(shots))} \u00b7 "
                f"{jobs} job(s) \u00b7 ~{seconds:.0f} s QPU (estimate)"
                f"</div>")

    def _on_calcheck(_b=None):
        """Run both arms in one session. Measures the INSTRUMENT.

        No epoch is applied: the check exists to tell whether the numbers a
        benchmark produces are biased, so treating its output as a
        calibration would be circular.
        """
        state, _refresh_fn = active_target()
        par, seq, targets = _calcheck_plans()
        if par is None:
            lbl_bench_status.value = _err(
                f'Calibration check needs at least '
                f'{bench.CALIBRATION_CHECK_EDGES} target edges.')
            return

        plan_name = _plan_name()
        if bench.requires_acknowledgement(plan_name, bench.LEVEL_TARGETED) \
                and not chk_bench_ack.value:
            lbl_bench_status.value = _err(
                'Tick "I understand this uses QPU time" to run the '
                'calibration check.')
            return

        backend_name = getattr(state, 'active_backend_name', None)
        reported_g = dict(getattr(state, 'gate_props_data', {}) or {})
        reported_q = dict(getattr(state, 'qubit_props_data', {}) or {})
        edges = list(getattr(state, 'edges_list', None) or [])
        import time as _time
        started_at = _time.time()
        jobs_seen = {}

        def _record_jobs(phase_index, job_ids, status):
            jobs_seen[str(phase_index)] = list(job_ids)
            if backend_name:
                bench.save_job_record(
                    backend_name, started_at, par, shots=par.shots,
                    jobs=jobs_seen, statuses={str(phase_index): status},
                    kind=bench.RECORD_KIND_CALIBRATION_CHECK,
                    arms=['parallel', 'sequential'])

        btn_bench_calcheck.disabled = True
        lbl_bench_status.value = (
            "<div style='font-size:14px;color:#8a4b00;background:#fff3cd;"
            "padding:6px 10px;border-radius:4px;border:1px solid #ffe08a;"
            "margin:3px 0;font-weight:bold;'>\u23f3 Calibration check: "
            "parallel then sequential\u2026</div>")

        if runner_factory is None:
            try:
                bench.preload_runtime()
            except Exception as exc:          # noqa: BLE001 - reported
                btn_bench_calcheck.disabled = False
                lbl_bench_status.value = _err(exc)
                return

        def _work():
            if runner_factory is not None:
                runner = runner_factory(state)
                return [bench.run_plan(par, runner),
                        bench.run_plan(seq, runner)], []
            runner = bench.make_calibration_check_runner(
                getattr(state, 'active_backend', None), on_jobs=_record_jobs)
            return runner([par, seq]), getattr(runner, 'last_expdata', [])

        def _done(payload):
            btn_bench_calcheck.disabled = False
            raws, expdata = payload
            if backend_name and expdata:
                try:
                    bench.save_raw_results(backend_name, started_at, expdata)
                except Exception:
                    _log.debug('raw dump failed', exc_info=True)
            try:
                par_epoch = bench.results_to_epoch(
                    raws[0], par, dict(reported_q), dict(reported_g), edges,
                    when=started_at)
                seq_epoch = bench.results_to_epoch(
                    raws[1], seq, dict(reported_q), dict(reported_g), edges,
                    when=started_at)
                table = bench.calibration_check_table(
                    edges, reported_g.get('cz_error', []), par_epoch,
                    seq_epoch)
            except Exception as exc:          # noqa: BLE001 - reported
                _log.exception('calibration check folding failed')
                lbl_bench_status.value = _err(exc)
                return

            notices = list(par_epoch.notices) + list(seq_epoch.notices)
            if backend_name:
                bench.save_attempt_notices(backend_name, started_at, notices)
                bench.save_calibration_check(backend_name, started_at, table)
            _render_calcheck(table)
            lbl_bench_status.value = (
                "<div style='font-size:14px;color:#1e8449;background:#eafaf1;"
                "padding:4px 8px;border-radius:4px;border:1px solid #abebc6;"
                "margin:3px 0;'>\u2705 Calibration check complete — the "
                "calibration in use is unchanged.</div>")
            _render_inline_notices(notices)

        def _failed(exc):
            btn_bench_calcheck.disabled = False
            _log.exception('calibration check failed: %s', exc)
            lbl_bench_status.value = _err(exc)

        if spawn_bg is None:
            try:
                _done(_work())
            except Exception as exc:          # noqa: BLE001 - reported
                _failed(exc)
            return
        spawn_bg(_work, on_done=_done, on_error=_failed)

    def _reanalyze_calcheck(state, name, record, job_ids):
        """Replay a calibration check from its jobs. Zero QPU, no epoch."""
        par = bench.plan_from_record(record)
        if par is None:
            lbl_bench_status.value = _err(
                'This calibration-check record has no experiment list, so '
                'its arms cannot be rebuilt.')
            return
        _seq_specs = []
        for spec in par.specs:
            if spec.kind != 'rb2q_layer':
                _seq_specs.append(spec)
                continue
            for pair in spec.params['pairs']:
                params = dict(spec.params)
                params['pairs'] = [tuple(pair)]
                _seq_specs.append(spec._replace(qubits=tuple(pair),
                                                params=params))
        seq = par._replace(specs=_seq_specs)

        reported_q = dict(getattr(state, 'qubit_props_data', {}) or {})
        reported_g = dict(getattr(state, 'gate_props_data', {}) or {})
        edges = list(getattr(state, 'edges_list', None) or [])
        import time as _time
        attempt_at = _time.time()

        # One arm's ids per half of the recorded phases.
        half = max(1, len(job_ids) // 2)
        arms = [job_ids[:half], job_ids[half:] or job_ids[:half]]

        try:
            raw_par, exp_par = bench.reanalyze_from_jobs(
                par, arms[0], getattr(state, 'active_backend', None),
                getattr(state, 'active_service', None))
            raw_seq, exp_seq = bench.reanalyze_from_jobs(
                seq, arms[1], getattr(state, 'active_backend', None),
                getattr(state, 'active_service', None))
            par_epoch = bench.results_to_epoch(raw_par, par, dict(reported_q),
                                               dict(reported_g), edges,
                                               when=attempt_at)
            seq_epoch = bench.results_to_epoch(raw_seq, seq, dict(reported_q),
                                               dict(reported_g), edges,
                                               when=attempt_at)
            table = bench.calibration_check_table(
                edges, reported_g.get('cz_error', []), par_epoch, seq_epoch)
        except Exception as exc:              # noqa: BLE001 - reported
            _log.exception('calibration-check reanalysis failed')
            _persist_attempt(name, attempt_at, exp_par if 'exp_par' in dir()
                             else [], [f'reanalysis failed: {exc}'])
            lbl_bench_status.value = _err(exc)
            return

        notices = list(par_epoch.notices) + list(seq_epoch.notices)
        _persist_attempt(name, attempt_at, list(exp_par) + list(exp_seq),
                         notices)
        if name:
            bench.save_calibration_check(name, attempt_at, table)
        _render_calcheck(table)
        lbl_bench_status.value = (
            "<div style='font-size:14px;color:#1e8449;background:#eafaf1;"
            "padding:4px 8px;border-radius:4px;border:1px solid #abebc6;"
            "margin:3px 0;'>\u2705 Calibration check reanalysed (no QPU) — "
            "the calibration in use is unchanged.</div>")
        _render_inline_notices(notices)

    def _render_calcheck(table):
        if lbl_bench_calcheck is not None:
            lbl_bench_calcheck.value = bench.calibration_check_table_html(
                table, esc)

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
        """Apply, refresh, re-rank, surface notices. Main thread only.

        An epoch that measured NOTHING is a failure, not a successful no-op.
        Applying one overwrites the reported calibration with an identical
        copy and labels it "measured" — which is exactly what happened after
        IBM rejected the 18 Sept submission, and is the single outcome this
        feature must never produce.
        """
        state, refresh_fn = active_target()
        _check_consistency(state, epoch)
        if getattr(epoch, 'is_empty', False):
            lbl_bench_status.value = _empty_sentence()
            _surface_notices(epoch)
            return False
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

        _surface_notices(epoch)

        analysis_failed = [n for n in (getattr(epoch, 'notices', None) or [])
                           if 'analysis failed' in n]
        rank_txt = f" · {n_ranked} candidate(s) re-ranked" if n_ranked else ''
        qpu_txt = f"QPU {tb_bench_qpu.value} · " if _qpu2_available() else ''
        coverage = (epoch.coverage_text() if hasattr(epoch, 'coverage_text')
                    else f'{epoch.n_measured_qubits}q/{epoch.n_measured_edges}e')
        if getattr(epoch, 'is_complete', True) and not analysis_failed:
            lbl_bench_status.value = (
                f"<div style='font-size:14px;color:#1e8449;background:#eafaf1;"
                f"padding:4px 8px;border-radius:4px;border:1px solid #abebc6;"
                f"margin:3px 0;'>✅ {esc(qpu_txt)}{esc(source)}: measured "
                f"{esc(coverage)} (L{epoch.level}){rank_txt}</div>")
        else:
            lbl_bench_status.value = (
                f"<div style='font-size:14px;color:#8a4b00;background:#fff3cd;"
                f"padding:4px 8px;border-radius:4px;border:1px solid #ffe08a;"
                f"margin:3px 0;'>⚠ {esc(qpu_txt)}{esc(source)}: partial — "
                f"measured {esc(coverage)} (L{epoch.level}){rank_txt}"
                + (f"<br>{esc(analysis_failed[0])}" if analysis_failed else '')
                + "</div>")
        _refresh_reuse_offer()
        return True

    def _check_consistency(state, epoch):
        """Compare against the reported calibration; rescale if it is off.

        A measured epoch whose median sits 2.8x away from the reported one
        across 351 edges is not describing drift — a whole chip does not move
        uniformly in one direction. That is an estimator mismatch, and its
        absolute values must not reach the LQS. Relative mode keeps the
        per-element ratios (the collapsed couplers, which ARE real) and
        discards only the offset.
        """
        if lbl_bench_consistency is None:
            return
        try:
            reported_q = getattr(state, '_reported_qubit_props', None) or {}
            reported_g = getattr(state, '_reported_gate_props', None) or {}
            if not reported_q and not reported_g:
                lbl_bench_consistency.value = ''
                return
            stats = bench.consistency_stats(epoch, reported_q, reported_g)
            epoch.consistency = stats
            offenders = bench.consistency_offenders(stats)
            rescale = bool(offenders) and not bool(
                getattr(chk_bench_absolute, 'value', False))
            if rescale:
                bench.apply_relative_mode(epoch, reported_q, reported_g,
                                          stats)
            rescaled = set(offenders) if rescale else set()

            lbl_bench_consistency.value = _result_sentence(epoch, stats,
                                                           rescaled)
            _render_details(stats, rescaled, reported_g)
        except Exception:
            _log.exception('consistency check failed')

    #: Property keys → the words a user recognises.
    _PROP_NAMES = {'readout_error': 'Readout error', 't1': 'T1', 't2': 'T2',
                   'cz_error': 'CZ error', 'rzz_error': 'RZZ error'}

    def _prop_name(key):
        if key in _PROP_NAMES:
            return _PROP_NAMES[key]
        if key.endswith('_error'):
            return f'{key[:-len("_error")].upper()} error'
        return key

    def _join_names(keys):
        # Escape the property names, not the surrounding copy: running the
        # whole sentence through esc() turned every apostrophe into &#x27;.
        names = [esc(_prop_name(k)) for k in keys]
        if len(names) <= 1:
            return names[0] if names else ''
        return ' and '.join([', '.join(names[:-1]), names[-1]])

    def _is_edge_prop(key):
        return key in ('cz_error', 'rzz_error')

    def _result_sentence(epoch, stats, rescaled):
        """One plain sentence. No jargon, no warning glyph, no method talk.

        The panel used to print the engine's own vocabulary — "estimator
        mismatch suspected; measured values applied in RELATIVE mode" — which
        describes the implementation rather than what the user now has. This
        says what happened to their numbers and why, and leaves the figures
        to the Details table.
        """
        in_band = [k for k in stats if k not in rescaled]   # stats order
        parts = ['\u2705 Benchmark applied']

        if not rescaled:
            parts[0] += (" \u2014 measured values are in line with IBM's "
                         "reported calibration.")
        else:
            parts[0] += '.'
            if in_band:
                parts.append(
                    f"{_join_names(in_band)} look like real drift today;")
            # Names and factors MUST share one ordering — `rescaled` is a
            # set, so iterating it for the factors while sorting it for the
            # names could pair "T2 and CZ error" with CZ's factor first.
            # The stats order (readout, t1, t2, 2Q) is the one the Details
            # table uses, so use it here too.
            order = [k for k in stats if k in rescaled]
            factors = ', '.join(
                f"\u00d7{stats[k]['ratio_of_medians']:.1f}" for k in order)
            parts.append(
                f"{_join_names(order)} differ by a constant factor "
                f"({factors}) and are shown on IBM's scale "
                f"\u00b7 README: Relative mode")

        missing_q = max(0, epoch.planned_count[0] - epoch.measured_count[0])
        missing_e = max(0, epoch.planned_count[1] - epoch.measured_count[1])
        if missing_q or missing_e:
            parts.append(f'{missing_q} qubits and {missing_e} edges could '
                         f'not be measured and keep their reported values.')

        return (f"<div style='font-size:13px;color:#1a5276;"
                f"background:#f4f8fb;padding:6px 10px;border-radius:4px;"
                f"border:1px solid #d6e4f0;margin:2px 0;'>"
                f"{' '.join(parts)}</div>")

    def _empty_sentence():
        return ("<div style='font-size:13px;color:#8a4b00;"
                "background:#fff3cd;padding:6px 10px;border-radius:4px;"
                "border:1px solid #ffe08a;margin:2px 0;'>"
                "\u26a0 Nothing usable was measured \u2014 IBM's reported "
                "calibration is still in use.</div>")

    def _details_rows(stats, rescaled):
        rows = []
        for key, value in stats.items():
            is_rescaled = key in rescaled
            ratio = (f"\u00d7{value['ratio_of_medians']:.2f}"
                     + (' \u26a0' if is_rescaled else ''))
            unit = 'edges' if _is_edge_prop(key) else 'qubits'
            count = value['n_worse_2x']
            # A rescaled property with nothing individually out of line has
            # no "changed a lot" story to tell — the whole difference was the
            # constant factor, which the previous column already reports.
            changed = ('\u2014' if (is_rescaled and not count)
                       else f"{count} of {value['n']} {unit} >2\u00d7 worse")
            applied = ('rescaled to IBM\'s scale' if is_rescaled
                       else 'as measured')
            if is_rescaled and key in ('t1', 't2'):
                applied = 'rescaled (different estimator)'
            rows.append({'key': key, 'name': _prop_name(key), 'ratio': ratio,
                         'changed': changed, 'applied': applied,
                         'count': count})
        return rows

    def _render_details(stats, rescaled, reported_g):
        """The numbers, one row per property, collapsed by default."""
        if acc_bench_details is None or lbl_bench_details is None:
            return
        rows = _details_rows(stats, rescaled)
        if not rows:
            lbl_bench_details.value = ''
            return

        body = ''
        for row in rows:
            changed_cell = esc(row['changed'])
            if (_is_edge_prop(row['key']) and row['count']
                    and btn_show_on_map is not None):
                # The count becomes actionable: the button below the table
                # paints exactly these edges on the device map.
                changed_cell = f"{esc(row['changed'])} \u2192"
            body += (
                f"<tr><td style='padding:2px 10px;'>{esc(row['name'])}</td>"
                f"<td style='padding:2px 10px;text-align:right;'>"
                f"{esc(row['ratio'])}</td>"
                f"<td style='padding:2px 10px;'>{changed_cell}</td>"
                f"<td style='padding:2px 10px;'>{esc(row['applied'])}</td>"
                f"</tr>")

        lbl_bench_details.value = (
            "<table style='font-size:12px;border-collapse:collapse;'>"
            "<tr>"
            "<th style='padding:2px 10px;text-align:left;'>Property</th>"
            "<th style='padding:2px 10px;'>vs reported</th>"
            "<th style='padding:2px 10px;text-align:left;'>changed a lot</th>"
            "<th style='padding:2px 10px;text-align:left;'>how applied</th>"
            "</tr>"
            f"{body}</table>"
            "<div style='font-size:11px;color:#6c757d;margin-top:4px;'>"
            "as measured = used directly \u00b7 rescaled = median "
            "aligned to IBM\u2019s, ranking kept</div>")

        _sync_show_on_map(stats, reported_g)

    def _sync_show_on_map(stats, reported_g):
        """Offer the map link only when there are edges worth looking at."""
        if btn_show_on_map is None:
            return
        edge_key = next((k for k in stats
                         if _is_edge_prop(k) and stats[k]['n_worse_2x']), None)
        _map_target[0] = None
        if edge_key is None or chk_err_threshold is None \
                or txt_g_thr_max is None:
            btn_show_on_map.layout.display = 'none'
            return
        median = stats[edge_key]['median_reported']
        _map_target[0] = (edge_key, 2.0 * median,
                          stats[edge_key]['n_worse_2x'])
        btn_show_on_map.description = (
            f"show {stats[edge_key]['n_worse_2x']} edges on map")
        btn_show_on_map.layout.display = ''

    def _on_show_on_map(_b=None):
        """Three writes: pick the 2Q property, set the cap, enable the filter.

        The threshold widgets already drive the Error tab's rendering, so
        this needs no plot change — it just points them at "worse than twice
        IBM's median", which is exactly what the table counted.
        """
        target = _map_target[0]
        if target is None:
            return
        edge_key, cap, _count = target
        try:
            if dd_gate_prop is not None:
                options = [v for _lbl, v in (dd_gate_prop.options or [])
                           if not isinstance(dd_gate_prop.options, dict)]
                if not options or edge_key in options:
                    dd_gate_prop.value = edge_key
            if txt_g_thr_max is not None:
                txt_g_thr_max.value = f'{cap:.3e}'
            if chk_err_threshold is not None:
                chk_err_threshold.value = True
        except Exception:
            _log.exception('show-on-map wiring failed')

    def _surface_notices(epoch):
        _render_inline_notices(getattr(epoch, 'notices', None))
        if notices_label is not None and getattr(epoch, 'notices', None):
            notices_label.value = (
                "<div style='font-size:13px;color:#8a4b00;background:#fff3cd;"
                "padding:4px 8px;border-radius:4px;border:1px solid #ffe08a;"
                "margin:2px 0;'>⚠ " +
                "<br>".join(esc(n) for n in epoch.notices) + "</div>")

    def _render_inline_notices(notices):
        """Show the run's notices under the status, not only in the Auto tab.

        The benchmark section lives in the sidebar; its notices were going to
        the Auto tab's notices bar, which a user running a benchmark is not
        looking at. The reason a run came back empty belongs next to the
        message saying it came back empty.
        """
        if lbl_bench_consistency is None:
            return
        if not notices:
            return
        shown = list(notices)[:6]
        more = ('' if len(notices) <= 6
                else f"<br>\u2026and {len(notices) - 6} more")
        lbl_bench_status.value += (
            "<div style='font-size:12px;color:#6c757d;background:#f8f9fa;"
            "padding:4px 8px;border-radius:4px;border:1px solid #dee2e6;"
            "margin:2px 0;'>"
            + "<br>".join(esc(n) for n in shown) + more + "</div>")

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
        # Kept for the consistency check, which has to compare against what
        # was reported BEFORE the measured epoch replaces it.
        setattr(state, '_reported_qubit_props', reported_q)
        setattr(state, '_reported_gate_props', reported_g)

        import time as _time
        started_at = _time.time()
        job_ids_seen = {}

        def _record_jobs(chunk_index, job_ids, status):
            """Persist ids the moment they exist, not at the end.

            The 18 Sept run left no record of what had been submitted because
            it never completed. With this, an interrupted or rejected run can
            still be diagnosed and — via reanalyze_from_jobs — recovered.
            """
            job_ids_seen[str(chunk_index)] = list(job_ids)
            if backend_name:
                bench.save_job_record(
                    backend_name, started_at, plan, shots=plan.shots,
                    jobs=job_ids_seen, statuses={str(chunk_index): status})

        def _work():
            """Background: build the runner and spend the QPU time."""
            if runner_factory is not None:
                runner = runner_factory(state)
            else:
                runner = bench.make_runtime_runner(
                    getattr(state, 'active_backend', None),
                    on_jobs=_record_jobs)
            result = bench.run_plan(plan, runner)
            return result, getattr(runner, 'last_expdata', [])

        def _done(payload):
            """Main thread: fold, apply, persist. Raw dump always written."""
            btn_bench_run.disabled = False
            raw, expdata = (payload if isinstance(payload, tuple)
                            else (payload, []))
            # The raw dump is saved BEFORE anything can go wrong in folding,
            # and regardless of whether the epoch turns out usable: it is the
            # diagnostic artefact that did not exist on 18 Sept.
            if backend_name and expdata:
                try:
                    bench.save_raw_results(backend_name, started_at, expdata)
                except Exception:
                    _log.debug('raw dump failed', exc_info=True)
            try:
                # `when=started_at` deliberately: the epoch, its _jobs.json and
                # its _raw.json must share one timestamp. On 20 Sept they did
                # not — the dumps were named by submission time (12:38:08) and
                # the epoch by completion time (12:50:10), so the three
                # artifacts of a single run looked like unrelated files and the
                # raw dump appeared to be missing.
                epoch = bench.results_to_epoch(raw, plan, reported_q,
                                               reported_g, edges,
                                               when=started_at)
            except Exception as exc:          # noqa: BLE001 - reported
                _log.exception("folding benchmark results failed")
                lbl_bench_status.value = _err(exc)
                return
            if _install_epoch(epoch, source=f'L{level} benchmark') \
                    and backend_name:
                bench.save_measured_epoch(backend_name, epoch)
            _refresh_reanalyze_offer()

        def _failed(exc):
            """Main thread: the reported epoch is left exactly as it was.

            The raw dump is written here too: a failed run is precisely when
            the diagnostic artefact is most needed.
            """
            btn_bench_run.disabled = False
            expdata = getattr(exc, 'expdata', None)
            if backend_name and expdata:
                try:
                    bench.save_raw_results(backend_name, started_at, expdata)
                except Exception:
                    _log.debug('raw dump on failure path failed',
                               exc_info=True)
            _log.exception("benchmark run failed: %s", exc)
            if isinstance(exc, bench.BenchmarkRunError):
                ids = ', '.join(exc.job_ids) if exc.job_ids else 'none'
                lbl_bench_status.value = _err(
                    f'Benchmark job failed: {exc} (job ids: {ids})')
            else:
                lbl_bench_status.value = _err(exc)
            _refresh_reanalyze_offer()

        if spawn_bg is None:                  # no runner injected: inline
            try:
                _done(_work())
            except Exception as exc:          # noqa: BLE001 - reported
                _failed(exc)
            return
        spawn_bg(_work, on_done=_done, on_error=_failed)

    def _refresh_reanalyze_offer():
        """Show Reanalyze when a submission record exists for this backend."""
        if btn_bench_reanalyze is None:
            return
        try:
            state, _r = active_target()
            name = getattr(state, 'active_backend_name', None)
            record = bench.latest_job_record(name) if name else None
            has_jobs = bool(record and any(record.get('jobs', {}).values()))
            btn_bench_reanalyze.layout.display = '' if has_jobs else 'none'
            btn_bench_reanalyze.disabled = not has_jobs
        except Exception:
            _log.debug('reanalyze offer unavailable', exc_info=True)

    def _on_reanalyze(_b=None):
        """Rebuild the epoch from the last submission's jobs. Zero QPU.

        qiskit-experiments fits client-side, so the jobs already hold
        everything needed: this re-runs the analysis, never the circuits.
        """
        state, _refresh_fn = active_target()
        name = getattr(state, 'active_backend_name', None)
        record = bench.latest_job_record(name) if name else None
        if not record:
            lbl_bench_status.value = _err('No previous submission to reanalyze.')
            return
        # Per-PHASE grouping, not a flat list. `reanalyze_from_jobs` takes
        # the two-phase path (1Q RB first, then 2Q RB built with those EPGs)
        # only when handed a list of lists; flattening collapsed both phases
        # into one batch, so the 2Q analyses were constructed before any 1Q
        # EPG existed and came back UN-subtracted — a measured median of
        # 4.05e-3 where the phased path gives 2.96e-3.
        jobs_by_phase = record.get('jobs') or {}

        def _phase_key(key):
            try:
                return (0, int(key))
            except (TypeError, ValueError):
                return (1, str(key))

        job_ids = [list(jobs_by_phase[k])
                   for k in sorted(jobs_by_phase, key=_phase_key)
                   if jobs_by_phase[k]]
        if not any(job_ids):
            lbl_bench_status.value = _err(
                'The last submission recorded no job ids.')
            return

        # Replay the plan that PRODUCED these jobs, from the submission
        # record. Never `_current_plan()`: the target selector describes what
        # the user would run NEXT, and with it on "Current layout(s)" and
        # nothing placed the rebuilt plan had zero experiments — so nothing
        # was folded, no notice was raised, and the panel reported (0, 0)
        # while the raw dump held 348 EPG_cz and 156 T1/T2/readout.
        # A calibration check replays as TWO arms and yields a table, never
        # an epoch — it measures the instrument, not the device.
        if (record.get('kind') == bench.RECORD_KIND_CALIBRATION_CHECK
                or record.get('arms')):
            _reanalyze_calcheck(state, name, record, job_ids)
            return

        plan, how = bench.plan_for_replay(
            record, name,
            len(getattr(state, 'qubit_ids', None) or []),
            list(getattr(state, 'edges_list', None) or []))
        if plan is None:
            lbl_bench_status.value = _err(
                "Can't rebuild this run's plan (no experiment list "
                "saved). Only runs made with the current version can be "
                "reanalysed.")
            return

        if btn_bench_reanalyze is not None:
            btn_bench_reanalyze.disabled = True
        if runner_factory is None:
            if not bench.runtime_ready():
                lbl_bench_status.value = (
                    "<div style='font-size:14px;color:#8a4b00;"
                    "background:#fff3cd;padding:6px 10px;border-radius:4px;"
                    "border:1px solid #ffe08a;margin:3px 0;font-weight:bold;'>"
                    "\u23f3 Loading benchmark libraries\u2026</div>")
            try:
                bench.preload_runtime()
            except Exception as exc:          # noqa: BLE001 - reported
                if btn_bench_reanalyze is not None:
                    btn_bench_reanalyze.disabled = False
                lbl_bench_status.value = _err(exc)
                return

        reported_q = dict(getattr(state, 'qubit_props_data', {}) or {})
        reported_g = dict(getattr(state, 'gate_props_data', {}) or {})
        edges = list(getattr(state, 'edges_list', None) or [])
        # `_check_consistency` compares against what was reported BEFORE
        # the epoch lands. `_on_run` stashes these; the reanalysis path
        # did not, so the check found nothing to compare with, returned
        # early, and the epoch was applied unrescaled with no
        # consistency line at all.
        setattr(state, '_reported_qubit_props', reported_q)
        setattr(state, '_reported_gate_props', reported_g)

        import time as _time
        attempt_at = _time.time()
        expdata = []
        try:
            raw, expdata = bench.reanalyze_from_jobs(
                plan, job_ids, getattr(state, 'active_backend', None),
                getattr(state, 'active_service', None))
            epoch = bench.results_to_epoch(raw, plan, reported_q, reported_g,
                                           edges, when=attempt_at)
        except Exception as exc:              # noqa: BLE001 - reported
            if btn_bench_reanalyze is not None:
                btn_bench_reanalyze.disabled = False
            _log.exception('reanalysis failed')
            # Persist whatever exists even on the failure path: a reanalysis
            # that leaves nothing behind cannot be diagnosed, which is
            # exactly how the 20 Sept attempt became a dead end.
            _persist_attempt(name, attempt_at, expdata,
                             [f'reanalysis failed: {exc}'])
            lbl_bench_status.value = _err(exc)
            return

        if btn_bench_reanalyze is not None:
            btn_bench_reanalyze.disabled = False
        # Raw dump and notices FIRST, unconditionally — an empty epoch is
        # precisely the case where the artefacts are needed, and the old code
        # returned before writing them.
        _persist_attempt(name, attempt_at, expdata, epoch.notices)
        if _install_epoch(epoch,
                          source=f'Reanalysis of L{plan.level} run (no QPU)') \
                and name:
            bench.save_measured_epoch(name, epoch)

    def _persist_attempt(backend_name, when, expdata, notices):
        """Write the raw dump and the notices for ANY attempt.

        Including one that produced nothing. The 20 Sept reanalysis wrote no
        file at all because the empty-epoch branch returned before the dump,
        so there was no way to tell whether it had failed on the composite
        metadata or on something else.
        """
        if not backend_name:
            return
        try:
            if expdata:
                bench.save_reanalysis_raw(backend_name, when, expdata)
        except Exception:
            _log.debug('raw dump failed', exc_info=True)
        try:
            bench.save_attempt_notices(backend_name, when, notices)
        except Exception:
            _log.debug('notice dump failed', exc_info=True)

    def _on_reuse(_b=None):
        epoch = _last_epoch[0]
        if epoch is None:
            lbl_bench_status.value = _err('No recent measurement to reuse.')
            return
        _install_epoch(epoch, source='Reused measurement')

    btn_bench_run.on_click(_on_run)
    btn_bench_reuse.on_click(_on_reuse)
    if btn_show_on_map is not None:
        btn_show_on_map.on_click(_on_show_on_map)
    if btn_bench_calcheck is not None:
        btn_bench_calcheck.on_click(_on_calcheck)
    if btn_bench_reanalyze is not None:
        btn_bench_reanalyze.on_click(_on_reanalyze)
    _refresh_reanalyze_offer()
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
        'reanalyze': _on_reanalyze,
        'calibration_check': _on_calcheck,
        'calcheck_plans': _calcheck_plans,
        'show_on_map': _on_show_on_map,
        'result_sentence': _result_sentence,
        'render_details': _render_details,
    }
