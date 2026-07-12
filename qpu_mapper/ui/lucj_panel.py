"""
ui.lucj_panel.py — LUCJ layout generation callbacks.

Extracted from ui.py for maintainability. All functions receive
a `ctx` SimpleNamespace with shared UI state and widgets.
"""
import contextlib
import logging

from ipywidgets import Dropdown, HTML

from ..constants import hex_to_rgba
from ..html_safe import esc

_log = logging.getLogger(__name__)


def _format_best_per_k(strata):
    """"6→24.1% · 5→24.6% · 4→25.0% · 3→25.3%" — the LQS gradient across
    strata (best product_score per k, k descending). Shows the trade-off
    instead of a single size-biased "winner"."""
    parts = []
    for k in sorted(strata.keys(), reverse=True):
        entries = strata[k]
        if entries:
            parts.append(f"{k}→{entries[0]['product_score'] * 100.0:.1f}%")
    return ' · '.join(parts)


def _format_sweep_status(res):
    """One-line sweep summary (M2): 'Sweep: k=6..3 · 214 embeddings ·
    best per k: ...'. All interpolations esc()'d."""
    k_range = f"{res['k_max_effective']}..{res['min_bridges']}"
    emb = f"{res['n_raw_embeddings']:,}"
    per_k = _format_best_per_k(res.get('strata', {}))
    return (f"Sweep: k={esc(k_range)} · {esc(emb)} embeddings · "
            f"best per k: {esc(per_k)}")


def setup_lucj_callbacks(ctx):
    """Wire all LUCJ-related callbacks. Called from ui.py after widget creation."""

    # Unpack context for readability
    state = ctx.state
    out = ctx.out
    W = ctx.W

    lucj_norb_slider = ctx.lucj_norb_slider
    lucj_ancilla_dd = ctx.lucj_ancilla_dd
    lucj_min_ancilla = ctx.lucj_min_ancilla
    lucj_chk_score = ctx.lucj_chk_score
    btn_lucj_run = ctx.btn_lucj_run
    lbl_lucj_status = ctx.lbl_lucj_status
    lbl_lucj_trunc = ctx.lbl_lucj_trunc
    dd_lucj_layout = ctx.dd_lucj_layout
    lbl_lucj_detail = ctx.lbl_lucj_detail

    lucj_multi_norbs = ctx.lucj_multi_norbs
    lucj_multi_ancillas = ctx.lucj_multi_ancillas
    lucj_buffer_slider = ctx.lucj_buffer_slider
    btn_lucj_multi = ctx.btn_lucj_multi
    lbl_lucj_multi = ctx.lbl_lucj_multi
    lucj_multi_detail_box = ctx.lucj_multi_detail_box

    num_circuits = ctx.num_circuits
    intra_qpu_mode = ctx.intra_qpu_mode

    _lucj_all_layouts = ctx.lucj_all_layouts
    _lucj_multi_results = ctx.lucj_multi_results
    _lucj_multi_norbs_list = ctx.lucj_multi_norbs_list
    _lucj_multi_ancillas_list = ctx.lucj_multi_ancillas_list
    _lucj_multi_dropdowns = ctx.lucj_multi_dropdowns

    def _refresh():
        ctx.refresh()

    def _get_export():
        ctx.get_export()

    # ── Helper: apply candidate to circuit slot ──────────────────
    def _apply_multi_candidate(circ_idx, candidate):
        d = state.layouts_data[circ_idx]
        d['alpha'] = []; d['beta'] = []; d['other'] = []
        d['qubits'] = []; d['ancilla'] = []; d['styles'] = {}
        alpha_color = d['defaults']['fill']
        beta_color = d['defaults']['fill_beta']
        line_only = bool(getattr(ctx, 'chk_line_only', None) and ctx.chk_line_only.value)
        alpha_fill = 'none' if line_only else hex_to_rgba(alpha_color, 1.0)
        beta_fill = 'none' if line_only else hex_to_rgba(beta_color, 1.0)
        # Match manual placement: use the circuit's configured outline
        # (line colour / width / text), not a hardcoded black width-2 —
        # otherwise LUCJ alpha/beta look unlike the ancillas in Line only.
        _line = d.get('current_line', d['defaults']['line'])
        _width = d['sl_w'].value if 'sl_w' in d else 2
        _text = d.get('current_text_color', '#000000')
        d['alpha'] = [str(q) for q in candidate['alpha'] if str(q) in state.qubit_ids]
        d['beta'] = [str(q) for q in candidate['beta'] if str(q) in state.qubit_ids]
        for q in d['alpha']:
            d['styles'][q] = {'fill': alpha_fill,
                              'line': _line, 'width': _width, 'text_color': _text}
        for q in d['beta']:
            d['styles'][q] = {'fill': beta_fill,
                              'line': _line, 'width': _width, 'text_color': _text}

    # ── Apply single layout to circuit 0 ─────────────────────────
    def _apply_lucj_layout(entry):
        state.push_undo()
        d = state.layouts_data[0]
        d['alpha'] = []; d['beta'] = []; d['other'] = []
        d['qubits'] = []; d['ancilla'] = []; d['styles'] = {}
        alpha_color = d['defaults']['fill']
        beta_color = d['defaults']['fill_beta']
        line_only = bool(getattr(ctx, 'chk_line_only', None) and ctx.chk_line_only.value)
        alpha_fill = 'none' if line_only else hex_to_rgba(alpha_color, 1.0)
        beta_fill = 'none' if line_only else hex_to_rgba(beta_color, 1.0)
        # Match manual placement: use the circuit's configured outline
        # (line colour / width / text), not a hardcoded black width-2 —
        # otherwise LUCJ alpha/beta look unlike the ancillas in Line only.
        _line = d.get('current_line', d['defaults']['line'])
        _width = d['sl_w'].value if 'sl_w' in d else 2
        _text = d.get('current_text_color', '#000000')
        d['alpha'] = [str(q) for q in entry['alpha'] if str(q) in state.qubit_ids]
        d['beta'] = [str(q) for q in entry['beta'] if str(q) in state.qubit_ids]
        for q in d['alpha']:
            d['styles'][q] = {'fill': alpha_fill,
                              'line': _line, 'width': _width, 'text_color': _text}
        for q in d['beta']:
            d['styles'][q] = {'fill': beta_fill,
                              'line': _line, 'width': _width, 'text_color': _text}
        # Sweep candidates carry their kept bridges in entry['ancillas'];
        # place them as the ancilla role (green) so a derived k<k_max
        # candidate renders with exactly its own ancilla count. Legacy
        # single-k entries have no 'ancillas' key -> no ancillas placed
        # (unchanged behaviour).
        anc_color = d['defaults']['fill_anc']
        anc_fill = 'none' if line_only else hex_to_rgba(anc_color, 1.0)
        d['ancilla'] = [str(q) for q in entry.get('ancillas', [])
                        if str(q) in state.qubit_ids]
        for q in d['ancilla']:
            d['styles'][q] = {'fill': anc_fill,
                              'line': _line, 'width': _width, 'text_color': _text}
        _refresh()
        _get_export()

    # ── Dropdown layout select ───────────────────────────────────
    def _on_lucj_layout_select(change):
        idx = change['new']
        if idx is None or idx >= len(_lucj_all_layouts):
            return
        entry = _lucj_all_layouts[idx]
        _apply_lucj_layout(entry)
        if 'product_score' in entry:
            # Sweep candidate: multiplicative LQS is the headline; the
            # additive sum (size-biased across k) goes in the detail only.
            k = len(entry['ancillas'])
            lqs = entry['product_score'] * 100.0
            header = (f"<b>k={k} · #{entry['rank_in_stratum']}</b> "
                      f"· LQS {lqs:.1f}% "
                      f"<span style='color:#888;'>(add {entry['additive_score']:.4f}, "
                      f"{esc(str(entry.get('source', '')))})</span>")
            anc = ', '.join(str(q) for q in entry['ancillas'])
            lbl_lucj_detail.value = (
                f"<div style='font-size:14px;background:#f8f9fa;padding:6px 8px;"
                f"border-radius:4px;border:1px solid #dee2e6;margin:3px 0;'>"
                f"{header}<br>"
                f"α: {', '.join(str(q) for q in entry['alpha'])}<br>"
                f"β: {', '.join(str(q) for q in entry['beta'])}<br>"
                f"ancillas: {anc}</div>"
            )
        else:
            sc = f"{entry['score']:.4f}" if entry['score'] is not None else "N/A"
            lbl_lucj_detail.value = (
                f"<div style='font-size:14px;background:#f8f9fa;padding:6px 8px;"
                f"border-radius:4px;border:1px solid #dee2e6;margin:3px 0;'>"
                f"<b>#{entry['rank']}</b> (err={sc})<br>"
                f"α: {', '.join(str(q) for q in entry['alpha'])}<br>"
                f"β: {', '.join(str(q) for q in entry['beta'])}</div>"
            )
    dd_lucj_layout.observe(_on_lucj_layout_select, names='value')

    # ── Dropdown re-seed helper (observer-safe) ──────────────────
    def _seed_layout_dropdown(entries, dd_opts):
        """Store `entries` as the flat candidate list and re-seed
        dd_lucj_layout without firing the select observer, then apply the
        first entry. Shared by the legacy and sweep paths."""
        _lucj_all_layouts.clear()
        _lucj_all_layouts.extend(entries)
        # Suppress observer firing while re-seeding: otherwise setting
        # value=0 fires _on_lucj_layout_select → apply, then we apply again
        # below — double push_undo + redundant refresh. Same unobserve →
        # set → observe dance as _reoptimize_downstream.
        dd_lucj_layout.unobserve(_on_lucj_layout_select, names='value')
        dd_lucj_layout.options = dd_opts
        dd_lucj_layout.value = 0 if dd_opts else None
        dd_lucj_layout.disabled = not dd_opts
        dd_lucj_layout.observe(_on_lucj_layout_select, names='value')
        if entries:
            _apply_lucj_layout(entries[0])

    # ── Legacy single-count search ───────────────────────────────
    def _generate_legacy(norb, requested_ancillas):
        from ..auto_lucj import get_all_zigzag_layouts

        all_layouts, num_ab = get_all_zigzag_layouts(
            num_orbitals=norb,
            backend=state.active_backend,
            score=lucj_chk_score.value,
            props=state.initial_props,
            requested_bridges=requested_ancillas,
        )

        # Surface the geometric-cap clamp: auto_lucj logs a stderr
        # warning when the requested bridges exceed ceil(norb/4), but
        # Jupyter hides stderr — without this the user saw
        # "ancillas=1" with no hint that their "5" was clamped.
        ancilla_note = ""
        if (requested_ancillas is not None
                and requested_ancillas >= 0
                and requested_ancillas != num_ab):
            ancilla_note = (
                f"<br>⚠️ You requested <b>{requested_ancillas}</b> "
                f"ancilla(s) but the zigzag geometry only allows "
                f"<b>{num_ab}</b> for norb={norb} (max one bridge "
                f"every 4 orbitals)."
            )

        dd_opts = []
        for entry in all_layouts:
            sc = f"err={entry['score']:.4f}" if entry['score'] is not None else "?"
            dd_opts.append((
                f"#{entry['rank']} ({sc}) α:[{entry['alpha'][0]}..{entry['alpha'][-1]}]",
                entry['rank'] - 1,
            ))
        _seed_layout_dropdown(all_layouts, dd_opts)

        best = all_layouts[0]
        if best['score'] is None and lucj_chk_score.value:
            lbl_lucj_status.value = (
                f"<div style='font-size:14px;color:#e67e22;background:#fef9e7;padding:4px 8px;"
                f"border-radius:4px;border:1px solid #fdebd0;margin:3px 0;'>"
                f"⚠️ Scoring unavailable (no calibration data). "
                f"<b>{len(all_layouts)}</b> layouts found (unsorted). "
                f"ancillas={num_ab}{ancilla_note}</div>")
        else:
            sc = f"{best['score']:.4f}" if best['score'] is not None else "N/A"
            lbl_lucj_status.value = (
                f"<div style='font-size:14px;color:#1e8449;background:#eafaf1;padding:4px 8px;"
                f"border-radius:4px;border:1px solid #abebc6;margin:3px 0;'>"
                f"✅ <b>{len(all_layouts)}</b> layouts found (norb={norb}, "
                f"ancillas={num_ab}). Best: err={sc}{ancilla_note}</div>")

    # ── Stratified ancilla sweep (new default) ───────────────────
    def _generate_sweep(norb):
        from ..ancilla_sweep import sweep_ancilla_layouts

        # 0 (or blank) → engine default min(3, k_max); else the user floor.
        min_val = lucj_min_ancilla.value
        min_bridges = None if (min_val is None or min_val <= 0) else min_val

        res = sweep_ancilla_layouts(
            norb, state.active_backend, state.initial_props,
            min_bridges=min_bridges)

        strata = res['strata']
        # Flatten k descending, entries in stratum order; the dropdown value
        # is the flat index. Label format is fixed: "k=4 · #1 · LQS 25.3%".
        flat = []
        dd_opts = []
        for k in sorted(strata.keys(), reverse=True):
            for entry in strata[k]:
                lqs = entry['product_score'] * 100.0
                dd_opts.append((
                    f"k={k} · #{entry['rank_in_stratum']} · LQS {lqs:.1f}%",
                    len(flat),
                ))
                flat.append(entry)

        _seed_layout_dropdown(flat, dd_opts)

        # Truncation banner (U5): visible only when the VF2 cap was hit.
        if res['vf2_truncated']:
            lbl_lucj_trunc.value = (
                f"<div style='font-size:14px;color:#e67e22;background:#fef9e7;"
                f"padding:4px 8px;border-radius:4px;border:1px solid #fdebd0;"
                f"margin:3px 0;'>⚠️ Enumeration capped at "
                f"{esc(str(res['n_raw_embeddings']))} mappings — ranking may "
                f"be non-exhaustive.</div>")
        else:
            lbl_lucj_trunc.value = ""

        # One-line summary (M2): per-stratum LQS gradient, NOT a single
        # size-biased "best LQS" (product_score always favours fewer bridges).
        if flat:
            lbl_lucj_status.value = (
                f"<div style='font-size:14px;color:#1e8449;background:#eafaf1;"
                f"padding:4px 8px;border-radius:4px;border:1px solid #abebc6;"
                f"margin:3px 0;'>✅ {_format_sweep_status(res)}</div>")
        else:
            lbl_lucj_status.value = (
                f"<div style='font-size:14px;color:#e67e22;background:#fef9e7;"
                f"padding:4px 8px;border-radius:4px;border:1px solid #fdebd0;"
                f"margin:3px 0;'>⚠️ Sweep produced no candidates.</div>")

    # ── Single circuit LUCJ generate ─────────────────────────────
    def _on_lucj_generate(b):
        if state.active_backend is None:
            lbl_lucj_status.value = "<span style='font-size:14px;color:#e74c3c;'>❌ No backend connected.</span>"
            return
        if hasattr(state.active_backend, 'coords'):
            lbl_lucj_status.value = (
                "<span style='font-size:14px;color:#e74c3c;'>❌ Auto-LUCJ "
                "requires a connected IBM backend "
                "(VF2 layout search needs real calibration data).</span>")
            return
        norb = lucj_norb_slider.value
        btn_lucj_run.disabled = True
        btn_lucj_run.description = '⏳ Generating...'
        lbl_lucj_status.value = ""
        lbl_lucj_trunc.value = ""
        try:
            requested_ancillas = lucj_ancilla_dd.value
            # Exact count picked (>= 0) → legacy single-count search,
            # unchanged. 'Auto' (-1) → stratified sweep (new default).
            if requested_ancillas is not None and requested_ancillas >= 0:
                _generate_legacy(norb, requested_ancillas)
            else:
                _generate_sweep(norb)
        except Exception as e:
            lbl_lucj_status.value = (
                f"<div style='font-size:14px;color:#e74c3c;background:#fdedec;padding:6px 10px;"
                f"border-radius:5px;border:1px solid #f5b7b1;margin:4px 0;'>"
                f"❌ <b>Error:</b> {esc(str(e))}</div>")
        finally:
            btn_lucj_run.disabled = False
            btn_lucj_run.description = '🤖 Generate LUCJ Layout'

    btn_lucj_run.on_click(_on_lucj_generate)

    # ── M1: sweep mode always scores → grey out the score checkbox ────
    def _on_ancilla_mode_change(change=None):
        # 'Auto' (-1) == sweep mode; the engine always computes LQS there,
        # so the legacy "Score layouts" toggle is meaningless.
        sweep_mode = (lucj_ancilla_dd.value is not None
                      and lucj_ancilla_dd.value < 0)
        lucj_chk_score.disabled = sweep_mode
        lucj_chk_score.tooltip = ('Sweep always scores' if sweep_mode
                                  else 'Score layouts (pick lowest error)')
    lucj_ancilla_dd.observe(_on_ancilla_mode_change, names='value')
    _on_ancilla_mode_change()  # set initial state (default -1 → disabled)

    # ── B2/B3/B4: Multi-QPU scan ─────────────────────────────────
    btn_scan_qpus = ctx.btn_scan_qpus
    chk_scan_other_families = ctx.chk_scan_other_families
    lbl_scan_progress = ctx.lbl_scan_progress
    scan_results_box = ctx.scan_results_box
    scan_spawn_bg = ctx.scan_spawn_bg
    scan_marshal = ctx.scan_marshal
    inter_qpu_handoff = ctx.inter_qpu_handoff  # shared cell → stage hook

    _scan = {'progress': {}, 'order': []}
    _STATUS_ICON = {'start': '⏳', 'ok': '✓', 'skipped': '·', 'error': '✗'}

    def _render_progress_line():
        chips = []
        for name in _scan['order']:
            st, detail = _scan['progress'].get(name, ('', ''))
            txt = f"{esc(str(name))} {_STATUS_ICON.get(st, '')}"
            if st == 'error' and detail:
                txt += f" {esc(str(detail))}"
            chips.append(txt)
        lbl_scan_progress.value = (
            ("<div style='font-size:13px;color:#555;margin:3px 0;'>"
             + " · ".join(chips) + "</div>") if chips else "")

    def _on_scan_progress(name, status, detail=''):
        # Called from the WORKER thread — NEVER touch widgets here. Marshal
        # the widget mutation onto the main loop.
        def _apply():
            if name not in _scan['progress']:
                _scan['order'].append(name)
            _scan['progress'][name] = (status, detail)
            _render_progress_line()
        scan_marshal(_apply)

    def _entry_label(entry):
        k = len(entry['ancillas'])
        return (f"k={k} · #{entry['rank_in_stratum']} · "
                f"LQS {entry['product_score'] * 100.0:.1f}%")

    def _render_targets(targets):
        rows = []
        for t in targets:
            if t['eligible']:
                rows.append(f"<span style='color:#1e8449;'>"
                            f"{esc(str(t['name']))} ✓</span>")
            else:
                rows.append(f"<span style='color:#999;'>{esc(str(t['name']))} "
                            f"— {esc(str(t['reason']))}</span>")
        scan_results_box.children = [HTML(
            "<div style='font-size:13px;margin:3px 0;'><b>Targets:</b><br>"
            + "<br>".join(rows) + "</div>")]

    def _stage_handoff(backend_name, candidate):
        hook = inter_qpu_handoff[0] if inter_qpu_handoff else None
        if hook is None:
            lbl_scan_progress.value = (
                "<div style='color:#e67e22;font-size:13px;'>Inter-QPU panel "
                "unavailable — cannot open a remote candidate.</div>")
            return
        # A handoff must never break the scan UI.
        with contextlib.suppress(Exception):
            hook(backend_name, candidate)

    def _render_results(scan_result):
        results = scan_result['results']
        head = ("<table style='font-size:13px;border-collapse:collapse;'>"
                "<tr style='background:#eaf2f8;'>"
                "<th style='padding:3px 8px;text-align:left;'>QPU</th>"
                "<th style='padding:3px 8px;'>family</th>"
                "<th style='padding:3px 8px;'>best LQS</th>"
                "<th style='padding:3px 8px;'>k</th>"
                "<th style='padding:3px 8px;'>cands</th></tr>")
        body = []
        for r in results:
            name = esc(str(r['name']))
            fam = esc(str(r['family'] or '?'))
            if r['error']:
                body.append(
                    f"<tr style='color:#999;background:#f7f7f7;'>"
                    f"<td style='padding:3px 8px;'>{name}</td>"
                    f"<td style='padding:3px 8px;'>{fam}</td>"
                    f"<td colspan='3' style='padding:3px 8px;'>✗ "
                    f"{esc(str(r['error']))}</td></tr>")
            elif r['best_lqs'] is None:
                body.append(
                    f"<tr style='color:#999;'>"
                    f"<td style='padding:3px 8px;'>{name}</td>"
                    f"<td style='padding:3px 8px;'>{fam}</td>"
                    f"<td colspan='3' style='padding:3px 8px;'>no candidates"
                    f"</td></tr>")
            else:
                lqs = esc(f"{r['best_lqs'] * 100.0:.1f}%")
                ncand = sum(len(v) for v in r['sweep']['strata'].values())
                body.append(
                    f"<tr><td style='padding:3px 8px;'>{name}</td>"
                    f"<td style='padding:3px 8px;text-align:center;'>{fam}</td>"
                    f"<td style='padding:3px 8px;text-align:center;'>{lqs}</td>"
                    f"<td style='padding:3px 8px;text-align:center;'>"
                    f"{esc(str(r['best_k']))}</td>"
                    f"<td style='padding:3px 8px;text-align:center;'>"
                    f"{esc(str(ncand))}</td></tr>")
        children = [HTML(head + "".join(body) + "</table>")]

        # Per-QPU stratified dropdowns (same label format as single-circuit).
        for r in results:
            sweep = r.get('sweep')
            if not sweep:
                continue
            strata = sweep['strata']
            flat, dd_opts = [], []
            for k in sorted(strata.keys(), reverse=True):
                for entry in strata[k]:
                    dd_opts.append((_entry_label(entry), len(flat)))
                    flat.append(entry)
            if not dd_opts:
                continue
            children.append(HTML(
                f"<div style='font-weight:bold;margin:6px 0 2px;font-size:13px;'>"
                f"{esc(str(r['name']))} — {_format_best_per_k(strata)}</div>"))
            dd = Dropdown(options=dd_opts, value=None, description='Layout:',
                          layout=W, style={'description_width': '50px'})
            det = HTML("")

            def _make_cb(qpu_name, flatlist, detail_lbl):
                def _cb(change):
                    idx = change['new']
                    if idx is None or idx >= len(flatlist):
                        return
                    entry = flatlist[idx]
                    detail_lbl.value = (
                        f"<div style='font-size:13px;background:#f8f9fa;"
                        f"padding:4px 6px;border-radius:3px;margin:2px 0;'>"
                        f"α: {esc(', '.join(str(q) for q in entry['alpha']))}<br>"
                        f"β: {esc(', '.join(str(q) for q in entry['beta']))}<br>"
                        f"ancillas: "
                        f"{esc(', '.join(str(q) for q in entry['ancillas']))}"
                        f"</div>")
                    # Drive the inter-QPU panel — NEVER touch C1 / current layout.
                    _stage_handoff(qpu_name, entry)
                return _cb
            dd.observe(_make_cb(r['name'], flat, det), names='value')
            children.extend([dd, det])

        scan_results_box.children = children

    def _on_scan_done(scan_result):
        _render_results(scan_result)
        btn_scan_qpus.disabled = False
        btn_scan_qpus.description = '🌐 Scan all QPUs'

    def _on_scan_error(err):
        lbl_scan_progress.value = (
            f"<div style='color:#e74c3c;font-size:13px;'>❌ Scan failed: "
            f"{esc(str(err))}</div>")
        btn_scan_qpus.disabled = False
        btn_scan_qpus.description = '🌐 Scan all QPUs'

    def _on_scan_click(b):
        if state.active_backend is None or state.active_service is None:
            lbl_scan_progress.value = (
                "<span style='color:#e74c3c;font-size:13px;'>❌ Connect a "
                "backend first.</span>")
            return
        from ..qpu_scan import list_scan_targets, scan_qpus, _detect_family
        from ..props_shim import make_scan_get_props

        norb = lucj_norb_slider.value
        min_val = lucj_min_ancilla.value
        min_bridges = None if (min_val is None or min_val <= 0) else min_val
        try:
            family = _detect_family(state.active_backend)
        except Exception:
            family = None

        targets = list_scan_targets(
            state, norb, same_family_as=family,
            include_other_families=chk_scan_other_families.value)
        _render_targets(targets)                # immediate feedback

        eligible = [t for t in targets if t['eligible']]
        if not eligible:
            lbl_scan_progress.value = (
                "<div style='color:#e67e22;font-size:13px;'>No eligible "
                "backends for this norb / family scope.</div>")
            return

        _scan['progress'].clear()
        _scan['order'].clear()
        btn_scan_qpus.disabled = True
        btn_scan_qpus.description = '⏳ Scanning...'
        get_backend = state.active_service.backend
        get_props = make_scan_get_props(state)

        def _work():
            return scan_qpus(eligible, norb, get_backend, get_props,
                             min_bridges=min_bridges, top_n=3,
                             progress_cb=_on_scan_progress)
        scan_spawn_bg(_work, on_done=_on_scan_done, on_error=_on_scan_error)

    btn_scan_qpus.on_click(_on_scan_click)

    # ── Multi-circuit LUCJ generate ──────────────────────────────
    def _on_lucj_multi(b):
        if state.active_backend is None:
            lbl_lucj_multi.value = (
                "<span style='font-size:14px;color:#e74c3c;'>❌ No backend connected.</span>")
            return
        if hasattr(state.active_backend, 'coords'):
            lbl_lucj_multi.value = (
                "<span style='font-size:14px;color:#e74c3c;'>❌ Auto-LUCJ "
                "requires a connected IBM backend.</span>")
            return

        raw = lucj_multi_norbs.value.strip()
        if not raw:
            lbl_lucj_multi.value = (
                "<span style='font-size:14px;color:#e74c3c;'>❌ Enter norb values (e.g. 8, 8)</span>")
            return
        try:
            norb_list = [int(x.strip()) for x in raw.split(',') if x.strip()]
        except ValueError:
            lbl_lucj_multi.value = (
                "<span style='font-size:14px;color:#e74c3c;'>❌ Invalid norb values</span>")
            return
        if len(norb_list) < 1:
            return

        # Per-circuit ancilla counts (parallel to norb). Blank → fall back
        # to the single Ancilla dropdown for every circuit (legacy).
        anc_raw = lucj_multi_ancillas.value.strip()
        ancilla_list = None
        if anc_raw:
            try:
                ancilla_list = [int(x.strip()) for x in anc_raw.split(',') if x.strip()]
            except ValueError:
                lbl_lucj_multi.value = (
                    "<span style='font-size:14px;color:#e74c3c;'>❌ Invalid ancilla values "
                    "(use integers, e.g. 2, 2; -1 = auto)</span>")
                return
            if len(ancilla_list) != len(norb_list):
                lbl_lucj_multi.value = (
                    f"<span style='font-size:14px;color:#e74c3c;'>❌ Ancilla list has "
                    f"{len(ancilla_list)} value(s) but {len(norb_list)} circuit(s). Match them "
                    f"(norb '8, 8' → ancilla '2, 2'), or leave ancilla blank.</span>")
                return

        btn_lucj_multi.disabled = True
        btn_lucj_multi.description = '⏳ Generating...'
        lbl_lucj_multi.value = ""

        try:
            from ..auto_lucj import get_multi_circuit_layouts

            requested_ancillas = lucj_ancilla_dd.value
            results = get_multi_circuit_layouts(
                norb_list=norb_list,
                backend=state.active_backend,
                score=lucj_chk_score.value,
                buffer_qubits=lucj_buffer_slider.value,
                props=state.initial_props,
                requested_bridges=requested_ancillas,
                requested_bridges_list=ancilla_list,
            )

            if not intra_qpu_mode.value and len(norb_list) > 1:
                intra_qpu_mode.value = True
            num_circuits.value = len(norb_list)

            state.push_undo()
            for res in results:
                best = res['all_candidates'][0]
                _apply_multi_candidate(res['circuit_idx'], best)

            _refresh()
            _get_export()

            # Store for re-optimization
            _lucj_multi_results.clear()
            _lucj_multi_results.extend(results)
            _lucj_multi_norbs_list.clear()
            _lucj_multi_norbs_list.extend(norb_list)
            _lucj_multi_ancillas_list.clear()
            _lucj_multi_ancillas_list.extend(ancilla_list if ancilla_list is not None else [])
            _lucj_multi_dropdowns.clear()

            # Build per-circuit dropdowns
            detail_widgets = []
            for res in results:
                cidx = res['circuit_idx']
                cands = res['all_candidates']

                detail_widgets.append(HTML(
                    f"<div style='font-size:14px;font-weight:bold;color:#2C3E50;margin:6px 0 2px 0;'>"
                    f"C{cidx+1} — norb={res['norb']}, "
                    f"{len(cands)} layouts, ancillas={res['num_bridges']}</div>"))

                dd_opts = []
                for c in cands:
                    sc = f"err={c['score']:.4f}" if c['score'] is not None else "?"
                    dd_opts.append((f"#{c['rank']} ({sc}) α:[{c['alpha'][0]}..{c['alpha'][-1]}]",
                                    c['rank'] - 1))

                dd_circ = Dropdown(options=dd_opts, value=0,
                                  description=f'C{cidx+1}:', layout=W,
                                  style={'description_width': '35px'})
                lbl_circ = HTML("")

                def _show_detail(entry, lbl):
                    lbl.value = (
                        f"<div style='font-size:14px;background:#f8f9fa;padding:4px 6px;"
                        f"border-radius:3px;border:1px solid #dee2e6;margin:2px 0;'>"
                        f"α: {', '.join(str(q) for q in entry['alpha'])}<br>"
                        f"β: {', '.join(str(q) for q in entry['beta'])}</div>")

                _show_detail(cands[0], lbl_circ)

                _lucj_multi_dropdowns[cidx] = {
                    'dropdown': dd_circ, 'label': lbl_circ,
                    'candidates': cands,
                }

                def _make_on_change(ci, candidates, label):
                    def _on_change(change):
                        idx = change['new']
                        if idx is None or idx >= len(candidates):
                            return
                        entry = candidates[idx]
                        state.push_undo()
                        _apply_multi_candidate(ci, entry)
                        _refresh()
                        _get_export()
                        _show_detail(entry, label)
                        if ci < len(_lucj_multi_norbs_list) - 1:
                            _reoptimize_downstream(ci)
                    return _on_change

                dd_circ.observe(_make_on_change(cidx, cands, lbl_circ), names='value')
                detail_widgets.extend([dd_circ, lbl_circ])

            lucj_multi_detail_box.children = detail_widgets

            summary_lines = []
            for res in results:
                cands = res['all_candidates']
                sc = f"err={cands[0]['score']:.4f}" if cands[0]['score'] is not None else "unscored"
                summary_lines.append(
                    f"<b>C{res['circuit_idx']+1}</b> norb={res['norb']}: "
                    f"{len(cands)} candidates, best {sc}")

            # Per-circuit clamp note: with mixed norb_list each circuit
            # may hit its own ceil(norb/4) cap. Same rationale as the
            # single-circuit ancilla_note above — auto_lucj's stderr
            # warning is invisible in Jupyter.
            ancilla_notes = []
            if requested_ancillas is not None and requested_ancillas >= 0:
                ancilla_notes = [
                    (f"C{res['circuit_idx']+1} (norb={res['norb']}): "
                     f"requested {requested_ancillas}, got "
                     f"{res['num_bridges']} (max one bridge every 4 orbitals)")
                    for res in results
                    if res['num_bridges'] != requested_ancillas
                ]
            ancilla_warning = ""
            if ancilla_notes:
                ancilla_warning = (
                    "<br>⚠️ <b>Ancilla request clamped:</b><br>"
                    + "<br>".join(f"&nbsp;&nbsp;• {n}" for n in ancilla_notes)
                )

            lbl_lucj_multi.value = (
                f"<div style='font-size:14px;color:#1e8449;background:#eafaf1;padding:8px 10px;"
                f"border-radius:5px;border:1px solid #abebc6;margin:4px 0;line-height:1.7;'>"
                f"✅ <b>{len(results)} circuits placed!</b> "
                f"(buffer={lucj_buffer_slider.value})<br>"
                f"{'<br>'.join(summary_lines)}"
                f"{ancilla_warning}<br>"
                f"<span style='font-size:14px;color:#555;'>Use dropdowns below to switch layouts per circuit.</span></div>")

            ctx.update_view()

        except Exception as e:
            lbl_lucj_multi.value = (
                f"<div style='font-size:14px;color:#e74c3c;background:#fdedec;padding:6px 10px;"
                f"border-radius:5px;border:1px solid #f5b7b1;margin:4px 0;'>"
                f"❌ <b>Error:</b> {esc(str(e))}</div>")
        finally:
            btn_lucj_multi.disabled = False
            btn_lucj_multi.description = '🤖 Generate Multi-Circuit'

    btn_lucj_multi.on_click(_on_lucj_multi)

    # ── Re-optimization (F3) ─────────────────────────────────────
    def _reoptimize_downstream(changed_idx):
        try:
            from ..auto_lucj import reoptimize_after_change

            locked = {}
            for ci in range(changed_idx + 1):
                info = _lucj_multi_dropdowns.get(ci)
                if info is None:
                    continue
                dd = info['dropdown']
                cands = info['candidates']
                sel_idx = dd.value
                if sel_idx is not None and sel_idx < len(cands):
                    full = cands[sel_idx].get('layout_full', cands[sel_idx]['layout'])
                    locked[ci] = set(full)

            new_results = reoptimize_after_change(
                backend=state.active_backend,
                norb_list=_lucj_multi_norbs_list,
                locked_qubits=locked,
                changed_idx=changed_idx,
                buffer_qubits=lucj_buffer_slider.value,
                score=lucj_chk_score.value,
                props=state.initial_props,
                requested_bridges=lucj_ancilla_dd.value,
                requested_bridges_list=(list(_lucj_multi_ancillas_list)
                                        if _lucj_multi_ancillas_list else None),
            )

            for res in new_results:
                cidx = res['circuit_idx']
                cands = res['all_candidates']
                info = _lucj_multi_dropdowns.get(cidx)
                if info is None:
                    continue

                dd = info['dropdown']
                lbl = info['label']
                dd.unobserve_all()

                dd_opts = []
                for c in cands:
                    sc = f"err={c['score']:.4f}" if c['score'] is not None else "?"
                    dd_opts.append((f"#{c['rank']} ({sc}) α:[{c['alpha'][0]}..{c['alpha'][-1]}]",
                                    c['rank'] - 1))
                dd.options = dd_opts
                dd.value = 0
                info['candidates'] = cands

                _apply_multi_candidate(cidx, cands[0])
                lbl.value = (
                    f"<div style='font-size:14px;background:#f8f9fa;padding:4px 6px;"
                    f"border-radius:3px;border:1px solid #dee2e6;margin:2px 0;'>"
                    f"α: {', '.join(str(q) for q in cands[0]['alpha'])}<br>"
                    f"β: {', '.join(str(q) for q in cands[0]['beta'])}<br>"
                    f"<span style='color:#e67e22;'>♻️ re-optimized ({len(cands)} candidates)</span></div>")

                def _make_reopt_cb(ci, candidates, label):
                    def _cb(change):
                        idx = change['new']
                        if idx is None or idx >= len(candidates):
                            return
                        entry = candidates[idx]
                        state.push_undo()
                        _apply_multi_candidate(ci, entry)
                        _refresh()
                        _get_export()
                        label.value = (
                            f"<div style='font-size:14px;background:#f8f9fa;padding:4px 6px;"
                            f"border-radius:3px;border:1px solid #dee2e6;margin:2px 0;'>"
                            f"α: {', '.join(str(q) for q in entry['alpha'])}<br>"
                            f"β: {', '.join(str(q) for q in entry['beta'])}</div>")
                        if ci < len(_lucj_multi_norbs_list) - 1:
                            _reoptimize_downstream(ci)
                    return _cb
                dd.observe(_make_reopt_cb(cidx, cands, lbl), names='value')

            _refresh()
            _get_export()
            lbl_lucj_multi.value = lbl_lucj_multi.value.replace(
                '</div>',
                f'<br><span style="color:#e67e22;">♻️ Re-optimized C{changed_idx+2}+ after C{changed_idx+1} change</span></div>',
                1,
            )

        except Exception as e:
            _log.exception("LUCJ re-optimization failed")
            with out:
                print(f"⚠️ Re-optimization failed: {e!s}")
