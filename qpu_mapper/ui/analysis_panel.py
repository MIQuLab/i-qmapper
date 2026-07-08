"""
ui.analysis_panel.py — Export, ranking, fidelity, trend, and stable callbacks.

Extracted from ui.py for maintainability.
"""
import json
import logging
import os
import numpy as np

from ..html_safe import esc
from .status import status_html

_log = logging.getLogger(__name__)


def setup_analysis_callbacks(ctx):
    """Wire analysis/export callbacks. Returns dict of helper functions."""

    state = ctx.state
    export_mod = ctx.export_mod

    fname_json = ctx.fname_json
    btn_save_json = ctx.btn_save_json
    lbl_json_status = ctx.lbl_json_status
    export_format = ctx.export_format
    lbl_fidelity = ctx.lbl_fidelity
    dd_mode = ctx.dd_mode
    num_circuits = ctx.num_circuits
    intra_qpu_mode = ctx.intra_qpu_mode

    txt_inspect_qubit = ctx.txt_inspect_qubit
    btn_inspect = ctx.btn_inspect
    lbl_inspect_result = ctx.lbl_inspect_result

    dd_rank_prop = ctx.dd_rank_prop
    dd_rank_order = ctx.dd_rank_order
    txt_rank_filter = ctx.txt_rank_filter
    rank_table_html = ctx.rank_table_html
    dd_grank_prop = ctx.dd_grank_prop
    dd_grank_order = ctx.dd_grank_order
    txt_grank_filter = ctx.txt_grank_filter
    grank_table_html = ctx.grank_table_html

    txt_trend_qubits = ctx.txt_trend_qubits
    dd_trend_prop = ctx.dd_trend_prop
    btn_trend_plot = ctx.btn_trend_plot
    trend_chart_box = ctx.trend_chart_box

    txt_trend_gates = ctx.txt_trend_gates
    dd_gtrend_prop = ctx.dd_gtrend_prop
    btn_gtrend_plot = ctx.btn_gtrend_plot
    gtrend_chart_box = ctx.gtrend_chart_box

    dd_stable_prop = ctx.dd_stable_prop
    txt_stable_threshold = ctx.txt_stable_threshold
    slider_stable_ratio = ctx.slider_stable_ratio
    btn_stable_find = ctx.btn_stable_find
    stable_html = ctx.stable_html

    dd_gstable_prop = ctx.dd_gstable_prop
    txt_gstable_threshold = ctx.txt_gstable_threshold
    slider_gstable_ratio = ctx.slider_gstable_ratio
    btn_gstable_find = ctx.btn_gstable_find
    gstable_html = ctx.gstable_html

    # Section A widgets (current calibration inspector — opt-in via checkbox).
    chk_inspect_enable        = ctx.chk_inspect_enable
    inspector_body_box        = ctx.inspector_body_box
    dd_inspect_qubit          = ctx.dd_inspect_qubit
    lbl_inspect_qubit_metrics = ctx.lbl_inspect_qubit_metrics
    dd_inspect_gate           = ctx.dd_inspect_gate
    lbl_inspect_gate_metrics  = ctx.lbl_inspect_gate_metrics
    # Section B widgets (stability check — opt-in via chk_stability_enable,
    # gated additionally by cal_history availability).
    chk_stability_enable        = ctx.chk_stability_enable
    stability_body_box          = ctx.stability_body_box
    rb_stability_window         = ctx.rb_stability_window
    dd_unstable_qubit           = ctx.dd_unstable_qubit
    lbl_unstable_qubit_metrics  = ctx.lbl_unstable_qubit_metrics
    dd_unstable_gate            = ctx.dd_unstable_gate
    lbl_unstable_gate_metrics   = ctx.lbl_unstable_gate_metrics
    lbl_stability_note          = ctx.lbl_stability_note
    stability_section_box       = ctx.stability_section_box

    QUBIT_PROP_OPTIONS = ctx.QUBIT_PROP_OPTIONS
    GATE_PROP_OPTIONS = ctx.GATE_PROP_OPTIONS

    _user_save_path = ctx.user_save_path_in_dir

    def _get_export():
        return ctx.get_export()

    # ── Qubit Inspector ───────────────────────────────────────────
    def _on_inspect(b):
        q_id = txt_inspect_qubit.value.strip()
        if not q_id:
            lbl_inspect_result.value = ""
            return
        q_id_map = {q: i for i, q in enumerate(state.qubit_ids)} if state.qubit_ids else {}
        q_idx = q_id_map.get(q_id, -1)
        if q_idx < 0:
            lbl_inspect_result.value = (
                f"<span style='color:#e74c3c;font-size:14px;'>Qubit {esc(q_id)} not found.</span>")
            return

        lines = ["<div style='font-size:14px;background:var(--jp-layout-color1, #fff);padding:8px 10px;"
                 "border-radius:6px;border:2px solid #2980B9;margin:4px 0;'>"]
        lines.append(f"<b style='font-size:16px;color:#2C3E50;'>Qubit {esc(q_id)}</b><br>")

        # All qubit properties
        if state.qubit_props_data:
            for label, key in QUBIT_PROP_OPTIONS:
                arr = state.qubit_props_data.get(key, np.array([]))
                if q_idx < len(arr) and not np.isnan(arr[q_idx]):
                    val = arr[q_idx]
                    lines.append(f"<span style='color:#555;'>{esc(label)}:</span> "
                                 f"<b>{_fmt_value(val, key)}</b><br>")

        # Connected gates — emit every gate property the backend exposes per
        # neighbour (full per-edge card, not just CZ). NaN/missing props are
        # skipped silently per line.
        if state.gate_props_data and state.edges_list:
            # Unique-neighbour map so (u,v) and (v,u) don't double-count.
            neighbor_edges = {}  # other_idx -> list of edge indices
            for gi, edge in enumerate(state.edges_list):
                if q_idx in edge:
                    other = edge[1] if edge[0] == q_idx else edge[0]
                    neighbor_edges.setdefault(other, []).append(gi)

            if neighbor_edges:
                # Header count = number of distinct neighbor qubits.
                lines.append(
                    f"<br><b style='color:#1a5276;'>"
                    f"Connected gates ({len(neighbor_edges)} neighbors):</b><br>"
                )
                # Sort by neighbour index for reproducible UI ordering.
                for other in sorted(neighbor_edges.keys()):
                    other_id = (state.qubit_ids[other]
                                if other < len(state.qubit_ids)
                                else str(other))
                    # First edge index suffices — heavy-hex stores each
                    # unordered pair once, so the choice is unambiguous.
                    gi = neighbor_edges[other][0]
                    # Collect every gate property present for this edge.
                    prop_bits = []
                    for label, key in GATE_PROP_OPTIONS:
                        arr = state.gate_props_data.get(key, np.array([]))
                        if gi < len(arr) and not np.isnan(arr[gi]):
                            prop_bits.append(
                                f"{esc(label)}=<b>{_fmt_value(arr[gi], key)}</b>"
                            )
                    if prop_bits:
                        gate_line = (
                            f"↔ Q{esc(other_id)}: "
                            + " &middot; ".join(prop_bits)
                        )
                        lines.append(gate_line + "<br>")

        # Mini trend if history available
        if state.cal_history_dates and state.cal_history:
            trend = export_mod.get_qubit_trend_data(state, [q_id], 'readout_error')
            if trend['dates'] and trend['series'].get(q_id):
                vals = [v for v in trend['series'][q_id] if not np.isnan(v)]
                if vals:
                    lines.append(f"<br><br><b style='color:#1a5276;'>Readout trend "
                                 f"({len(trend['dates'])} snapshots):</b><br>")
                    lines.append(f"min={min(vals):.4E} · max={max(vals):.4E} · avg={np.mean(vals):.4E}")
                    txt_trend_qubits.value = q_id

        lines.append("</div>")
        lbl_inspect_result.value = "".join(lines)
    btn_inspect.on_click(_on_inspect)

    # ── JSON save ─────────────────────────────────────────────────
    _json_overwrite_pending = [False]

    def _save_json(b):
        try:
            path = _user_save_path(f"{fname_json.value}.json", '.json')
        except ValueError as e:
            lbl_json_status.value = status_html('error', str(e))
            return
        if os.path.exists(path) and not _json_overwrite_pending[0]:
            _json_overwrite_pending[0] = True
            lbl_json_status.value = status_html(
                'warn',
                f"'{os.path.basename(path)}' already exists. "
                f"Click Export again to overwrite.",
                bold=os.path.basename(path))
            return
        _json_overwrite_pending[0] = False
        try:
            ed = _get_export()
            with open(path, 'w') as f:
                json.dump(ed, f, indent=4)
            lbl_json_status.value = status_html(
                'ok', f"JSON saved: {path}", bold=path)
        except Exception as e:
            lbl_json_status.value = status_html('error', f"JSON save error: {e}")

    btn_save_json.on_click(_save_json)
    fname_json.observe(
        lambda c: _json_overwrite_pending.__setitem__(0, False), 'value')
    export_format.observe(lambda _: _get_export(), 'value')

    # ── Fidelity ──────────────────────────────────────────────────
    def _update_fidelity():
        n_circ = num_circuits.value if intra_qpu_mode.value else 1
        parts = []
        for i in range(n_circ):
            f = export_mod.estimate_circuit_fidelity(state, i, dd_mode.value)
            if f is None:
                continue
            name = state.layouts_data[i].get('name', f'C{i+1}')
            fid_pct = f['fidelity'] * 100
            color = '#27ae60' if fid_pct > 90 else '#e67e22' if fid_pct > 70 else '#e74c3c'
            wq = f"Q{esc(f['worst_qubit'][0])}={f['worst_qubit'][1]:.3E}" if f['worst_qubit'][0] else "—"
            wg = f"{esc(f['worst_gate'][0])}={f['worst_gate'][1]:.3E}" if f['worst_gate'][0] else "—"
            parts.append(
                f"<b>{esc(name)}</b>: <span style='color:{color};font-weight:bold;'>"
                f"{fid_pct:.1f}%</span> "
                f"({f['n_qubits']}q × {f['n_gates']}g · "
                f"worst qubit: {wq} · worst gate: {wg})")
            if f.get('outliers'):
                ol_list = ', '.join(f"Q{esc(q)}={e:.3E}" for q, e in f['outliers'])
                parts.append(
                    f"<span style='color:#e74c3c;font-size:17px;'>"
                    f"⚠️ {len(f['outliers'])} outlier(s) in layout (>3σ from device mean): {ol_list}"
                    f" — calibration may be noisy</span>")
        if parts:
            lbl_fidelity.value = (
                f"<div style='font-size:18px;background:#f8f9fa;padding:10px 12px;"
                f"border-radius:4px;border:1px solid #dee2e6;margin:6px 0;line-height:1.8;'>"
                f"📊 <b>Estimated Layout-Quality</b> "
                f"<span style='font-size:14px;color:#888;'>(upper bound — assumes independent errors, no crosstalk)</span>"
                f"<br>{'<br>'.join(parts)}</div>")
        else:
            lbl_fidelity.value = ""
        # Same trigger surface as _update_fidelity (layout/mode/intra-QPU/
        # num_circuits/backend) — refresh the two inspector sections too.
        try:
            _refresh_current_inspector()
        except Exception:
            _log.exception("current inspector refresh failed")
        try:
            _refresh_stability_inspector(rb_stability_window.value)
        except Exception:
            _log.exception("stability inspector refresh failed")

    # ── Section A: current calibration inspector ─────────────────────
    # Two dropdowns + two inline labels. Each option's value is a tuple
    # `('Q', qid, role, ci)` or `('G', edge_idx, ci)` so the selection
    # handler renders metrics without re-walking the layout — the label
    # is what the user reads.

    def _iter_layout_qubits():
        """Yield (circuit_idx, qid, role) for each layout slot in scope.

        Multi-programming aware: walks 1..num_circuits when intra_qpu_mode
        is on, single C1 otherwise. LUCJ mode emits roles
        alpha/beta/other/ancilla; general mode emits role='qubit'.
        """
        n_circ = num_circuits.value if intra_qpu_mode.value else 1
        is_lucj = dd_mode.value == 'lucj'
        for ci in range(n_circ):
            layout = state.layouts_data.get(ci, {})
            if is_lucj:
                for role in ('alpha', 'beta', 'other', 'ancilla'):
                    for qid in layout.get(role, []) or []:
                        yield ci, qid, role
            else:
                for qid in layout.get('qubits', []) or []:
                    yield ci, qid, 'qubit'

    def _iter_layout_edges():
        """Yield (circuit_idx, edge_idx, (u_id, v_id)) for layout edges.

        An edge is "in the layout" iff both endpoints are present in the
        layout's qubit set (for the current mode).
        """
        n_circ = num_circuits.value if intra_qpu_mode.value else 1
        is_lucj = dd_mode.value == 'lucj'
        qubit_ids = state.qubit_ids or []
        for ci in range(n_circ):
            layout = state.layouts_data.get(ci, {})
            if is_lucj:
                qids_in_layout = set()
                for role in ('alpha', 'beta', 'other', 'ancilla'):
                    qids_in_layout.update(layout.get(role, []) or [])
            else:
                qids_in_layout = set(layout.get('qubits', []) or [])
            if not qids_in_layout:
                continue
            for ei, edge in enumerate(state.edges_list or []):
                if len(edge) < 2:
                    continue
                u, v = edge[0], edge[1]
                u_id = qubit_ids[u] if isinstance(u, int) and u < len(qubit_ids) else u
                v_id = qubit_ids[v] if isinstance(v, int) and v < len(qubit_ids) else v
                if u_id in qids_in_layout and v_id in qids_in_layout:
                    yield ci, ei, (u_id, v_id)

    def _qubit_property_value(qid, prop_key):
        """Return the current-snapshot value of `prop_key` for `qid`, or NaN."""
        q_id_map = {q: i for i, q in enumerate(state.qubit_ids or [])}
        idx = q_id_map.get(qid, -1)
        if idx < 0:
            return float('nan')
        arr = state.qubit_props_data.get(prop_key, np.array([]))
        if idx < len(arr):
            return float(arr[idx])
        return float('nan')

    def _gate_property_value(edge_idx, prop_key):
        arr = state.gate_props_data.get(prop_key, np.array([]))
        if 0 <= edge_idx < len(arr):
            return float(arr[edge_idx])
        return float('nan')

    def _empty_dropdown_options(reason: str):
        """Single-entry options list that renders as a disabled hint —
        the empty-state must surface AS an option (not a sibling label)
        so the dropdown is visibly populated with the explanation."""
        return [(reason, None)]

    def _refresh_current_inspector():
        # Section A is opt-in: when unticked, collapse the body and skip
        # the analysis. The checkbox observer re-fires this on tick.
        if not chk_inspect_enable.value:
            inspector_body_box.layout.display = 'none'
            return
        inspector_body_box.layout.display = 'flex'

        # Empty-state gates: no layout / no calibration → disable + hint.
        # _iter_layout_* handles multi-circuit and general/LUCJ shapes so
        # one code path serves all four.
        qubit_rows = list(_iter_layout_qubits())
        if not qubit_rows:
            dd_inspect_qubit.options = _empty_dropdown_options('no layout assigned')
            dd_inspect_qubit.disabled = True
            dd_inspect_gate.options = _empty_dropdown_options('no layout assigned')
            dd_inspect_gate.disabled = True
            lbl_inspect_qubit_metrics.value = ''
            lbl_inspect_gate_metrics.value = ''
            return
        if not state.qubit_props_data:
            dd_inspect_qubit.options = _empty_dropdown_options('no calibration loaded')
            dd_inspect_qubit.disabled = True
            dd_inspect_gate.options = _empty_dropdown_options('no calibration loaded')
            dd_inspect_gate.disabled = True
            lbl_inspect_qubit_metrics.value = ''
            lbl_inspect_gate_metrics.value = ''
            return

        # Build qubit options sorted by readout_error ascending (best first).
        q_entries = []
        for ci, qid, role in qubit_rows:
            ro = _qubit_property_value(qid, 'readout_error')
            q_entries.append((ro, ci, qid, role))
        # NaN goes to the end so users see actual data first.
        q_entries.sort(key=lambda t: (np.isnan(t[0]), t[0]))

        def c_name(ci):
            return state.layouts_data.get(ci, {}).get('name', f'C{ci+1}')
        qubit_options = [
            (f"Q{qid} — ro={_fmt_value(ro, 'readout_error')} ({role}, {c_name(ci)})",
             ('Q', qid, role, ci))
            for ro, ci, qid, role in q_entries
        ]
        dd_inspect_qubit.options = qubit_options
        dd_inspect_qubit.disabled = False
        # Explicitly render metrics for the new top option — setting
        # `.value` to its current value is a no-op for traitlets, so the
        # rendering helper must be invoked directly after rebuilding.
        if qubit_options:
            _render_inspect_qubit_metrics(dd_inspect_qubit.value)
        else:
            lbl_inspect_qubit_metrics.value = ''

        # Build gate options sorted by cz_error ascending.
        g_entries = []
        for ci, ei, (u, v) in _iter_layout_edges():
            cz = _gate_property_value(ei, 'cz_error')
            g_entries.append((cz, ci, ei, (u, v)))
        g_entries.sort(key=lambda t: (np.isnan(t[0]), t[0]))
        gate_options = [
            (f"({u}, {v}) — CZ={_fmt_value(cz, 'cz_error')} ({c_name(ci)})",
             ('G', ei, ci))
            for cz, ci, ei, (u, v) in g_entries
        ]
        dd_inspect_gate.options = gate_options
        dd_inspect_gate.disabled = not gate_options
        if gate_options:
            _render_inspect_gate_metrics(dd_inspect_gate.value)
        else:
            lbl_inspect_gate_metrics.value = (
                "<span style='color:#888;font-size:16px;'>"
                "no gates in layout (single-qubit only)</span>")

    def _render_inspect_qubit_metrics(value):
        # value: ('Q', qid, role, ci) or None.
        if not value or value[0] != 'Q':
            lbl_inspect_qubit_metrics.value = ''
            return
        _, qid, _role, _ci = value
        # readout + T1 + T2 + SX + MEASURE. _fmt_value renders N/A for
        # missing props.
        ro = _qubit_property_value(qid, 'readout_error')
        t1 = _qubit_property_value(qid, 't1')
        t2 = _qubit_property_value(qid, 't2')
        sx = _qubit_property_value(qid, 'sx_error')
        me = _qubit_property_value(qid, 'measure_error')
        lbl_inspect_qubit_metrics.value = (
            f"<span style='color:#555;font-size:16px;'>"
            f"readout={_fmt_value(ro, 'readout_error')} · "
            f"T1={_fmt_value(t1, 't1')} · "
            f"T2={_fmt_value(t2, 't2')} · "
            f"SX={_fmt_value(sx, 'sx_error')} · "
            f"MEASURE={_fmt_value(me, 'measure_error')}</span>"
        )

    def _render_inspect_gate_metrics(value):
        if not value or value[0] != 'G':
            lbl_inspect_gate_metrics.value = ''
            return
        _, ei, _ci = value
        cz  = _gate_property_value(ei, 'cz_error')
        rzz = _gate_property_value(ei, 'rzz_error')
        gl  = _gate_property_value(ei, 'gate_length')
        lbl_inspect_gate_metrics.value = (
            f"<span style='color:#555;font-size:16px;'>"
            f"CZ={_fmt_value(cz, 'cz_error')} · "
            f"RZZ={_fmt_value(rzz, 'rzz_error')} · "
            f"length={_fmt_value(gl, 'gate_length')}</span>"
        )

    dd_inspect_qubit.observe(
        lambda c: _render_inspect_qubit_metrics(c['new']), names='value')
    dd_inspect_gate.observe(
        lambda c: _render_inspect_gate_metrics(c['new']), names='value')
    # Opt-in checkbox tick/untick: re-run refresh so the body populates
    # (and reveals) or collapses cleanly.
    chk_inspect_enable.observe(
        lambda c: _refresh_current_inspector(), names='value')

    # ── Section B: stability check (requires cal_history) ────────────

    # Cache the last computed report so on-select renderers don't
    # recompute — the report carries the per-day series used for
    # "worst day" rendering.
    _stability_cache = {'report': None}

    def _hide_stability_section(note_html: str):
        stability_section_box.layout.display = 'none'
        lbl_stability_note.value = note_html

    def _show_stability_section():
        stability_section_box.layout.display = 'flex'
        lbl_stability_note.value = ''

    def _all_stable_option(qubit_count: int, gate_count: int, kind: str):
        # ipywidgets needs a non-empty options list — a placeholder + a
        # disabled dropdown says "nothing to pick" without breaking the
        # widget visually.
        return [(
            f"✅ all stable in selection "
            f"({qubit_count} qubits, {gate_count} gates analyzed)", None)]

    def _refresh_stability_inspector(window_days: int):
        from ..stability import get_layout_stability_report

        # Section B is opt-in (mirrors Section A). Background auto-load
        # of 14-day history runs at connect, so cal_history is already
        # populated when the user opts in — no second wait.
        if not chk_stability_enable.value:
            stability_body_box.layout.display = 'none'
            return
        stability_body_box.layout.display = 'flex'

        if not state.cal_history_dates:
            _hide_stability_section(
                "<span style='color:#888;font-size:16px;'>"
                "no history fetched — switch to Multi-day mode and refetch "
                "to enable stability check</span>")
            _stability_cache['report'] = None
            return

        n_circ = num_circuits.value if intra_qpu_mode.value else 1
        is_lucj = dd_mode.value == 'lucj'
        mode_arg = 'lucj' if is_lucj else 'general'

        # Multi-programming: run the analysis once per circuit and
        # concatenate the qubit/gate lists, decorating each option text
        # with `(C1, C2…)` so the user can disambiguate.
        reports = []
        for ci in range(n_circ):
            r = get_layout_stability_report(
                state, ci, mode=mode_arg, window_days=window_days)
            if r is not None:
                reports.append((ci, r))

        if not reports:
            _show_stability_section()
            dd_unstable_qubit.options = []
            dd_unstable_qubit.disabled = True
            dd_unstable_gate.options = []
            dd_unstable_gate.disabled = True
            lbl_unstable_qubit_metrics.value = (
                "<span style='color:#888;font-size:16px;'>"
                "no layout assigned</span>")
            lbl_unstable_gate_metrics.value = ''
            _stability_cache['report'] = None
            return

        # All circuits share the same window — `snapshots_used` is
        # identical by construction; pull it from the first report.
        snapshots_used = reports[0][1]['snapshots_used']
        total_qubits = sum(len(r['qubits']) for _, r in reports)
        total_gates  = sum(len(r['gates'])  for _, r in reports)

        # Surface "5/30 days available" when the window exceeds fetched.
        if snapshots_used < window_days:
            header = (
                f"<span style='color:#e67e22;font-size:16px;'>"
                f"⚠️ {snapshots_used}/{window_days} days available "
                f"(fetch more history for a full window)</span>")
        else:
            header = (
                f"<span style='color:#888;font-size:16px;'>"
                f"analysing {snapshots_used} day(s) · "
                f"{total_qubits} qubits, {total_gates} gates in layout(s)</span>")
        lbl_stability_note.value = header
        _show_stability_section()

        # ── Cache for the on-select renderers ────────────────────────
        # Keyed by ('Q', ci, qid) / ('G', ci, edge_idx) so the renderer
        # can look up the entry in O(1).
        q_cache = {}
        g_cache = {}
        for ci, r in reports:
            for entry in r['qubits']:
                q_cache[('Q', ci, entry['qid'])] = entry
            for entry in r['gates']:
                g_cache[('G', ci, entry['edge_idx'])] = entry
        _stability_cache['report'] = {
            'window_days': window_days,
            'snapshots_used': snapshots_used,
            'qubits': q_cache, 'gates': g_cache,
        }

        # ── Unstable qubits dropdown ─────────────────────────────────
        def c_name(ci):
            return state.layouts_data.get(ci, {}).get('name', f'C{ci+1}')
        q_unstable = []
        for ci, r in reports:
            for entry in r['qubits']:
                if entry['analysis'].get('unstable'):
                    q_unstable.append((ci, entry))
        if q_unstable:
            q_options = []
            for ci, entry in q_unstable:
                a = entry['analysis']
                q_options.append((
                    f"Q{entry['qid']} — {a['bad_days']}/{a['total_days']} "
                    f"days bad (score={a['score']:.2f}) · {entry['role']}, {c_name(ci)}",
                    ('Q', ci, entry['qid']),
                ))
            dd_unstable_qubit.options = q_options
            dd_unstable_qubit.disabled = False
            _render_unstable_qubit_metrics(dd_unstable_qubit.value)
        else:
            dd_unstable_qubit.options = _all_stable_option(
                total_qubits, total_gates, 'qubits')
            dd_unstable_qubit.disabled = True
            lbl_unstable_qubit_metrics.value = ''

        # ── Unstable gates dropdown ──────────────────────────────────
        g_unstable = []
        for ci, r in reports:
            for entry in r['gates']:
                if entry['analysis'].get('unstable'):
                    g_unstable.append((ci, entry))
        if g_unstable:
            g_options = []
            for ci, entry in g_unstable:
                a = entry['analysis']
                u, v = entry['edge']
                g_options.append((
                    f"({u}, {v}) — {a['bad_days']}/{a['total_days']} "
                    f"days bad (score={a['score']:.2f}) · {c_name(ci)}",
                    ('G', ci, entry['edge_idx']),
                ))
            dd_unstable_gate.options = g_options
            dd_unstable_gate.disabled = False
            _render_unstable_gate_metrics(dd_unstable_gate.value)
        else:
            dd_unstable_gate.options = _all_stable_option(
                total_qubits, total_gates, 'gates')
            dd_unstable_gate.disabled = True
            lbl_unstable_gate_metrics.value = ''

    def _render_unstable_qubit_metrics(value):
        report = _stability_cache.get('report')
        if not value or not report:
            lbl_unstable_qubit_metrics.value = ''
            return
        entry = report.get('qubits', {}).get(value)
        if entry is None:
            lbl_unstable_qubit_metrics.value = ''
            return
        _render_stability_metrics(entry, lbl_unstable_qubit_metrics,
                                  'readout_error')

    def _render_unstable_gate_metrics(value):
        report = _stability_cache.get('report')
        if not value or not report:
            lbl_unstable_gate_metrics.value = ''
            return
        entry = report.get('gates', {}).get(value)
        if entry is None:
            lbl_unstable_gate_metrics.value = ''
            return
        _render_stability_metrics(entry, lbl_unstable_gate_metrics,
                                  'cz_error')

    def _render_stability_metrics(entry, label_widget, prop_key):
        """Shared formatter for the per-selection stability metrics."""
        a = entry['analysis']
        dates = entry.get('series_dates') or []
        widx = a.get('worst_day_idx', -1)
        if 0 <= widx < len(dates):
            wdate = dates[widx]
            series = entry.get('series_values')
            wval = (float(series[widx])
                    if series is not None and widx < len(series)
                    else float('nan'))
            worst_str = f"worst day: {esc(str(wdate)[:10])} (value={_fmt_value(wval, prop_key)})"
        else:
            worst_str = "worst day: N/A"
        label_widget.value = (
            f"<span style='color:#555;font-size:16px;'>"
            f"max={_fmt_value(a['max'], prop_key)} · "
            f"median={_fmt_value(a['median'], prop_key)} · "
            f"min={_fmt_value(a['min'], prop_key)} · "
            f"{worst_str}</span>"
        )

    dd_unstable_qubit.observe(
        lambda c: _render_unstable_qubit_metrics(c['new']), names='value')
    dd_unstable_gate.observe(
        lambda c: _render_unstable_gate_metrics(c['new']), names='value')
    # Window change re-runs the analysis directly — no need to bounce
    # through _update_fidelity.
    rb_stability_window.observe(
        lambda c: _refresh_stability_inspector(rb_stability_window.value),
        names='value')
    # Opt-in checkbox tick/untick: re-run refresh so the body populates
    # (and reveals) or collapses cleanly.
    chk_stability_enable.observe(
        lambda c: _refresh_stability_inspector(rb_stability_window.value),
        names='value')

    # ── Rankings ──────────────────────────────────────────────────
    def _update_ranking(change=None):
        try:
            filt = txt_rank_filter.value.strip()
            html = export_mod.get_qubit_ranking_html(
                state, dd_rank_prop.value, top_n=0, ascending=dd_rank_order.value,
                filter_text=filt)
            rank_table_html.value = html
        except Exception:
            # Blank the table on failure (UX), but log the traceback so
            # bugs in get_qubit_ranking_html don't vanish silently.
            _log.exception(
                "get_qubit_ranking_html failed (prop=%r, filter=%r)",
                dd_rank_prop.value, txt_rank_filter.value)
            rank_table_html.value = ""
    dd_rank_prop.observe(_update_ranking, 'value')
    dd_rank_order.observe(_update_ranking, 'value')
    txt_rank_filter.observe(_update_ranking, 'value')

    def _update_grank(change=None):
        try:
            filt = txt_grank_filter.value.strip()
            html = export_mod.get_gate_ranking_html(
                state, dd_grank_prop.value, top_n=0, ascending=dd_grank_order.value,
                filter_text=filt)
            grank_table_html.value = html
        except Exception:
            # Same rationale as `_update_ranking`.
            _log.exception(
                "get_gate_ranking_html failed (prop=%r, filter=%r)",
                dd_grank_prop.value, txt_grank_filter.value)
            grank_table_html.value = ""
    dd_grank_prop.observe(_update_grank, 'value')
    dd_grank_order.observe(_update_grank, 'value')
    txt_grank_filter.observe(_update_grank, 'value')

    # ── Trend chart formatting helper ────────────────────────────
    from ..constants import US_QUBIT_PROPS, NS_QUBIT_PROPS, fmt_e, fmt_us, fmt_ns

    def _fmt_value(v, prop_key):
        """Format a value like the hover tooltip on the device map."""
        import math
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return 'N/A'
        if prop_key in US_QUBIT_PROPS:
            return fmt_us(v)
        if prop_key in NS_QUBIT_PROPS or prop_key == 'gate_length':
            return fmt_ns(v)
        return fmt_e(v)

    def _trend_hover_texts(values, prop_key):
        """Build hover text list for a trend series."""
        return [_fmt_value(v, prop_key) for v in values]

    def _trend_yaxis_opts(prop_key):
        """Return Plotly yaxis options matching the property type."""
        if prop_key in US_QUBIT_PROPS:
            return dict(ticksuffix=' µs', exponentformat='none')
        if prop_key in NS_QUBIT_PROPS or prop_key == 'gate_length':
            return dict(ticksuffix=' ns', exponentformat='none')
        return dict(exponentformat='e', tickformat='.2e')

    # ── Qubit Trend chart ─────────────────────────────────────────
    # Plot/Hide uses a separate visibility flag, not `children`, so an
    # error message left in the box doesn't swallow the next click as a
    # "hide" — only a successful render flips the flag. Same pattern in
    # the gtrend / stable / gstable toggles below.
    _trend_visible = [False]

    def _on_trend_plot(b):
        from ipywidgets import HTML as _HTML
        if _trend_visible[0]:
            trend_chart_box.children = []
            btn_trend_plot.description = '📈 Plot Trend'
            _trend_visible[0] = False
            return
        raw = txt_trend_qubits.value.strip()
        if not raw:
            trend_chart_box.children = [_HTML(
                "<span style='color:#e74c3c;font-size:14px;'>Enter qubit IDs (e.g. 0, 5, 47)</span>")]
            return
        qids = [q.strip() for q in raw.replace(';', ',').split(',') if q.strip()]
        if not state.cal_history_dates:
            trend_chart_box.children = [_HTML(
                "<span style='color:#e74c3c;font-size:14px;'>Fetch calibration history first (Range mode)</span>")]
            return
        prop_key = dd_trend_prop.value
        trend = export_mod.get_qubit_trend_data(state, qids, prop_key)
        if not trend['dates']:
            trend_chart_box.children = [_HTML(
                "<span style='color:#888;font-size:14px;'>No data.</span>")]
            return
        import plotly.graph_objects as go
        tfig = go.FigureWidget()
        for qid, values in trend['series'].items():
            hover = _trend_hover_texts(values, prop_key)
            tfig.add_trace(go.Scatter(
                x=trend['dates'], y=values,
                mode='lines+markers', name=f'Q{qid}',
                text=hover, hovertemplate='%{fullData.name}: %{text}<extra></extra>',
                line=dict(width=2), marker=dict(size=5)))
        prop_label = dict(QUBIT_PROP_OPTIONS).get(prop_key, prop_key)
        yaxis_opts = _trend_yaxis_opts(prop_key)
        tfig.update_layout(
            title=f"{prop_label} over time",
            xaxis_title="Date", yaxis_title=prop_label,
            yaxis=yaxis_opts,
            height=300, margin=dict(l=70, r=20, t=40, b=40),
            template='plotly_white', legend=dict(orientation='h', y=-0.2))
        trend_chart_box.children = [tfig]
        btn_trend_plot.description = '🙈 Hide Trend'
        _trend_visible[0] = True
    btn_trend_plot.on_click(_on_trend_plot)

    # ── Gate Trend chart ──────────────────────────────────────────
    _gtrend_visible = [False]

    def _on_gtrend_plot(b):
        from ipywidgets import HTML as _HTML
        if _gtrend_visible[0]:
            gtrend_chart_box.children = []
            btn_gtrend_plot.description = '📈 Plot Gate Trend'
            _gtrend_visible[0] = False
            return
        raw = txt_trend_gates.value.strip()
        if not raw:
            gtrend_chart_box.children = [_HTML(
                "<span style='color:#e74c3c;font-size:14px;'>"
                "Enter gate edges (e.g. (0,1), (5,6))</span>")]
            return
        # Parse "(0,1), (5,6)" → ["(0,1)", "(5,6)"]. Both parens are
        # required; `\(?...\)?` silently accepted mismatched parens (no
        # error feedback for malformed input). NO capture groups —
        # get_gate_trend_data expects strings and runs its own
        # re.findall(r'\d+', estr) on each; (\d+) would yield tuples.
        import re as _re
        edge_strs = _re.findall(r'\(\s*\d+\s*,\s*\d+\s*\)', raw)
        if not edge_strs:
            gtrend_chart_box.children = [_HTML(
                "<span style='color:#e74c3c;font-size:14px;'>"
                "Invalid format. Use (0,1), (5,6)</span>")]
            return
        if not state.cal_history_dates:
            gtrend_chart_box.children = [_HTML(
                "<span style='color:#e74c3c;font-size:14px;'>"
                "Fetch calibration history first</span>")]
            return
        prop_key = dd_gtrend_prop.value
        trend = export_mod.get_gate_trend_data(state, edge_strs, prop_key)
        if not trend['dates']:
            gtrend_chart_box.children = [_HTML(
                "<span style='color:#888;font-size:14px;'>No data.</span>")]
            return
        import plotly.graph_objects as go
        tfig = go.FigureWidget()
        for label, values in trend['series'].items():
            hover = _trend_hover_texts(values, prop_key)
            tfig.add_trace(go.Scatter(
                x=trend['dates'], y=values,
                mode='lines+markers', name=label,
                text=hover, hovertemplate='%{fullData.name}: %{text}<extra></extra>',
                line=dict(width=2), marker=dict(size=5)))
        prop_label = dict(GATE_PROP_OPTIONS).get(prop_key, prop_key)
        yaxis_opts = _trend_yaxis_opts(prop_key)
        tfig.update_layout(
            title=f"{prop_label} over time",
            xaxis_title="Date", yaxis_title=prop_label,
            yaxis=yaxis_opts,
            height=300, margin=dict(l=70, r=20, t=40, b=40),
            template='plotly_white', legend=dict(orientation='h', y=-0.2))
        gtrend_chart_box.children = [tfig]
        btn_gtrend_plot.description = '🙈 Hide Gate Trend'
        _gtrend_visible[0] = True
    btn_gtrend_plot.on_click(_on_gtrend_plot)

    # ── Stable finder: dynamic threshold label ────────────────────
    # T1/T2 flip the comparison (higher is better), so the label flips
    # "Max err" → "Min val" to match. Logic switch lives in
    # export.find_stable_* (driven by constants.HIGHER_IS_BETTER_PROPS);
    # the label here only mirrors it so the UI stays consistent.
    from ..constants import HIGHER_IS_BETTER_PROPS

    def _refresh_stable_label(prop_key, threshold_widget):
        if prop_key in HIGHER_IS_BETTER_PROPS:
            threshold_widget.description = 'Min val:'
            threshold_widget.placeholder = 'e.g. 50 (µs)'
        else:
            threshold_widget.description = 'Max err:'
            threshold_widget.placeholder = 'e.g. 1e-3'

    def _on_stable_prop_change(change):
        _refresh_stable_label(change['new'], txt_stable_threshold)
    dd_stable_prop.observe(_on_stable_prop_change, names='value')
    # Initialise the label to match the dropdown's starting value.
    _refresh_stable_label(dd_stable_prop.value, txt_stable_threshold)

    def _on_gstable_prop_change(change):
        _refresh_stable_label(change['new'], txt_gstable_threshold)
    dd_gstable_prop.observe(_on_gstable_prop_change, names='value')
    _refresh_stable_label(dd_gstable_prop.value, txt_gstable_threshold)

    # ── Stable qubit finder ───────────────────────────────────────
    _stable_visible = [False]

    def _on_stable_find(b):
        if _stable_visible[0]:
            stable_html.value = ''
            btn_stable_find.description = '🔎 Find Stable Qubits'
            _stable_visible[0] = False
            return
        if not state.cal_history_dates:
            stable_html.value = (
                "<span style='color:#e74c3c;font-size:14px;'>"
                "Fetch calibration history first (Range mode).</span>")
            return
        try:
            max_err = float(txt_stable_threshold.value.strip())
        except (ValueError, AttributeError):
            stable_html.value = (
                "<span style='color:#e74c3c;font-size:14px;'>"
                "Invalid threshold value.</span>")
            return
        # IntSlider returns int 0-100, convert to 0.0-1.0 ratio expected
        # by export.find_stable_qubits.
        ratio = slider_stable_ratio.value / 100.0
        html = export_mod.get_stable_qubits_html(
            state, dd_stable_prop.value, max_err, ratio, top_n=0)
        stable_html.value = html
        btn_stable_find.description = '🙈 Hide Stable Qubits'
        _stable_visible[0] = True
    btn_stable_find.on_click(_on_stable_find)

    # ── Stable gate finder ────────────────────────────────────────
    _gstable_visible = [False]

    def _on_gstable_find(b):
        if _gstable_visible[0]:
            gstable_html.value = ''
            btn_gstable_find.description = '🔎 Find Stable Gates'
            _gstable_visible[0] = False
            return
        if not state.cal_history_dates:
            gstable_html.value = (
                "<span style='color:#e74c3c;font-size:14px;'>"
                "Fetch calibration history first.</span>")
            return
        try:
            max_err = float(txt_gstable_threshold.value.strip())
        except (ValueError, AttributeError):
            gstable_html.value = (
                "<span style='color:#e74c3c;font-size:14px;'>"
                "Invalid threshold value.</span>")
            return
        ratio = slider_gstable_ratio.value / 100.0
        html = export_mod.get_stable_gates_html(
            state, dd_gstable_prop.value, max_err, ratio, top_n=0)
        gstable_html.value = html
        btn_gstable_find.description = '🙈 Hide Stable Gates'
        _gstable_visible[0] = True
    btn_gstable_find.on_click(_on_gstable_find)

    return {
        'update_fidelity': _update_fidelity,
        'update_ranking': _update_ranking,
        'update_grank': _update_grank,
    }
