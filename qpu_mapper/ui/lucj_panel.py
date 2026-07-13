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


def _entry_lqs(entry):
    """(lqs_value, is_template): the circuit-level LQS (``entry['lqs_circuit']``,
    same scale as the Estimated Layout-Quality box) when available, else the
    internal template heuristic ``product_score`` with the template flag set."""
    val = entry.get('lqs_circuit')
    if val is not None:
        return val, False
    return entry['product_score'], True


def _entry_lqs_pct(entry):
    """('25.3%', is_template) for display."""
    val, is_tmpl = _entry_lqs(entry)
    return f"{val * 100.0:.1f}%", is_tmpl


def _format_best_per_k(strata):
    """"6→24.1% · 5→24.6% · 4→25.0% · 3→25.3%" — the LQS gradient across
    strata (best entry per k, k descending). Uses circuit-level LQS when
    present. Shows the trade-off instead of a single size-biased "winner"."""
    parts = []
    for k in sorted(strata.keys(), reverse=True):
        entries = strata[k]
        if entries:
            val, _ = _entry_lqs(entries[0])
            parts.append(f"{k}→{val * 100.0:.1f}%")
    return ' · '.join(parts)


def _sweep_mode_label(res):
    """(H2) 'direct+derived' when any lower stratum got a direct-VF2 hit,
    else 'derived-only'."""
    kmax = res.get('k_max_effective')
    ps = res.get('per_stratum', {})
    direct = any(v.get('n_direct', 0) > 0 for k, v in ps.items()
                 if kmax is not None and k < kmax)
    return 'direct+derived' if direct else 'derived-only'


def _format_sweep_status(res):
    """One-line sweep summary (M2): 'Sweep: k=6..3 · 214 embeddings ·
    best per k: ...'. All interpolations esc()'d."""
    k_range = f"{res['k_max_effective']}..{res['min_bridges']}"
    emb = f"{res['n_raw_embeddings']:,}"
    per_k = _format_best_per_k(res.get('strata', {}))
    return (f"Sweep: k={esc(k_range)} · {esc(emb)} embeddings · "
            f"best per k: {esc(per_k)}")


def _k_cap_phrase(norb, k_effective):
    """(N1) 'k capped at E (requested R)' when the geometric request
    ceil(norb/4) exceeds what actually fit; '' otherwise. Pure metadata —
    replaces the operator's need to read the repeated stderr warning."""
    request = (norb + 3) // 4
    if k_effective is not None and k_effective < request:
        return f"k capped at {k_effective} (requested {request})"
    return ""


class _ScanNoticeCapture(logging.Handler):
    """(N2) Captures WARNING+ records off the 'qpu_mapper' logger during a
    scan, tagging each with the current backend context. emit() never raises
    out — a logging hiccup must not break the scan."""

    def __init__(self, context_getter):
        super().__init__(level=logging.WARNING)
        self._context_getter = context_getter
        self.records = []            # list of (backend_context, message)

    def emit(self, record):
        try:
            self.records.append((self._context_getter(), record.getMessage()))
        except Exception:
            pass


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
    chk_deep_search = ctx.chk_deep_search
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
            # Sweep candidate: circuit-level LQS (matches the Estimated
            # Layout-Quality box) is the headline; the internal ranking sum
            # (size-biased across k) goes in the detail only.
            k = len(entry['ancillas'])
            pct, is_tmpl = _entry_lqs_pct(entry)
            scale = ('template fidelity (single-pass edges)' if is_tmpl
                     else 'circuit-level (matches Estimated Layout-Quality)')
            header = (f"<b>k={k} · #{entry['rank_in_stratum']}</b> "
                      f"· LQS {pct} "
                      f"<span style='color:#888;'>{scale}; internal ranking "
                      f"sum {entry['additive_score']:.4f}, "
                      f"{esc(str(entry.get('source', '')))}</span>")
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
    # (A2) The LQS shown here is the circuit-level score — the same number
    # the Estimated Layout-Quality box reports. '(template)' flags the
    # internal heuristic used only when circuit scoring is unavailable.
    with contextlib.suppress(Exception):
        dd_lucj_layout.tooltip = ('LQS = circuit-level fidelity, the same '
                                  'number as the Estimated Layout-Quality '
                                  'box. "(template)" = internal heuristic '
                                  'fallback when circuit scoring is '
                                  'unavailable.')
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

    # ── A2: circuit-level LQS for a sweep candidate ──────────────
    def _circuit_score(entry):
        """Score a sweep candidate with the SAME path as the Estimated
        Layout-Quality box: build a scratch layout from the candidate and
        run export.scoring.estimate_circuit_fidelity against the current
        backend's props. Returns a fidelity in [0, 1] or None."""
        from types import SimpleNamespace
        from ..export.scoring import estimate_circuit_fidelity
        layout = {
            'alpha': [str(q) for q in entry['alpha']],
            'beta': [str(q) for q in entry['beta']],
            'other': [],
            'ancilla': [str(q) for q in entry['ancillas']],
        }
        fake = SimpleNamespace(
            layouts_data={0: layout},
            qubit_props_data=getattr(state, 'qubit_props_data', {}) or {},
            gate_props_data=getattr(state, 'gate_props_data', {}) or {},
            qubit_ids=state.qubit_ids,
            edges_list=state.edges_list,
        )
        res = estimate_circuit_fidelity(fake, 0, 'lucj')
        return res['fidelity'] if res else None

    def _annotate_circuit_lqs(res):
        """Fill entry['lqs_circuit'] for any strata entry that lacks it (the
        stability-filter path can't pass circuit_score_fn to the engine)."""
        for entries in res.get('strata', {}).values():
            for e in entries:
                if 'lqs_circuit' not in e:
                    try:
                        e['lqs_circuit'] = _circuit_score(e)
                    except Exception:
                        e['lqs_circuit'] = None

    # ── P1: SINGLE SOURCE of sweep params — local AND scan read this ──
    def _sweep_kwargs_from_ui():
        """The one place sweep parameters are derived from the UI. Both the
        local sweep and every scan per-target call pass exactly this, so the
        two can never diverge again (the tonight bug)."""
        min_val = lucj_min_ancilla.value
        min_bridges = None if (min_val is None or min_val <= 0) else min_val
        return {
            'min_bridges': min_bridges,
            'top_n': 3,
            'hybrid': chk_deep_search.value,
        }

    # Best circuit-level LQS from the most recent LOCAL sweep — quoted as the
    # active backend's reference in the scan table (P2), so the scan never
    # re-searches self yet keeps the reference point.
    _local_best_lqs = [None]

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

        # P1: exact same params the scan uses (min_bridges / top_n / hybrid).
        kwargs = _sweep_kwargs_from_ui()

        # V2: opt-in stability filter (drop-in superset of the plain sweep).
        history = getattr(state, 'cal_history', None)
        if _stability_filter_active() and history:
            from ..stability_filter import sweep_with_stability
            res = sweep_with_stability(
                norb, state.active_backend, state.initial_props, history,
                window_days=rb_stability_window.value,
                threshold=ft_stability_threshold.value,
                min_bridges=kwargs['min_bridges'], top_n=kwargs['top_n'])
            _render_stability_banner(res.get('stability'), norb)
            # The frozen stability engine can't take circuit_score_fn — fill
            # circuit-level LQS on its survivors here (A2).
            _annotate_circuit_lqs(res)
        else:
            res = sweep_ancilla_layouts(
                norb, state.active_backend, state.initial_props,
                circuit_score_fn=_circuit_score,   # A2: aligned circuit LQS
                **kwargs)                          # P1 single source (incl. hybrid)
            if _stability_filter_active():
                # Checkbox on but no history dict → skipped note.
                _render_stability_banner(None, norb)
            else:
                lbl_lucj_stability.value = ""

        strata = res['strata']
        # Flatten k descending, entries in stratum order; the dropdown value
        # is the flat index. Label: "k=4 · #1 · LQS 25.3%" (circuit-level;
        # "(template)" suffix when circuit scoring was unavailable).
        flat = []
        dd_opts = []
        for k in sorted(strata.keys(), reverse=True):
            for entry in strata[k]:
                pct, is_tmpl = _entry_lqs_pct(entry)
                suffix = " (template)" if is_tmpl else ""
                dd_opts.append((
                    f"k={k} · #{entry['rank_in_stratum']} · LQS {pct}{suffix}",
                    len(flat),
                ))
                flat.append(entry)

        # Remember the best circuit-level LQS for the scan's active-row ref.
        circ = [e['lqs_circuit'] for e in flat
                if e.get('lqs_circuit') is not None]
        _local_best_lqs[0] = max(circ) if circ else None

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
            cap = _k_cap_phrase(norb, res.get('k_max_effective'))
            cap_txt = f" · {esc(cap)}" if cap else ""
            mode_txt = f" · {esc(_sweep_mode_label(res))}"
            lbl_lucj_status.value = (
                f"<div style='font-size:14px;color:#1e8449;background:#eafaf1;"
                f"padding:4px 8px;border-radius:4px;border:1px solid #abebc6;"
                f"margin:3px 0;'>✅ {_format_sweep_status(res)}{mode_txt}"
                f"{cap_txt}</div>")
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
    btn_scan_clear = ctx.btn_scan_clear
    lbl_scan_progress = ctx.lbl_scan_progress
    scan_results_box = ctx.scan_results_box
    scan_spawn_bg = ctx.scan_spawn_bg
    scan_marshal = ctx.scan_marshal
    inter_qpu_handoff = ctx.inter_qpu_handoff  # shared cell → stage hook

    # ── Stability filter (V1/V2/V3) ──────────────────────────────
    chk_stability_filter = ctx.chk_stability_filter
    rb_stability_window = ctx.rb_stability_window
    ft_stability_threshold = ctx.ft_stability_threshold
    stability_ctrl_box = ctx.stability_ctrl_box
    lbl_lucj_stability = ctx.lbl_lucj_stability
    chk_scan_stability = ctx.chk_scan_stability

    def _on_stability_toggle(change=None):
        # Reveal the window/threshold row only when the filter is on.
        stability_ctrl_box.layout.display = (
            'block' if chk_stability_filter.value else 'none')
        if not chk_stability_filter.value:
            lbl_lucj_stability.value = ""
    chk_stability_filter.observe(_on_stability_toggle, names='value')
    _on_stability_toggle()

    def _stability_filter_active():
        return bool(chk_stability_filter.value
                    and not chk_stability_filter.disabled)

    def _render_stability_banner(stab, norb):
        """Render res['stability'] (V2): filtered / relaxed / unavailable."""
        if stab is None or not stab.get('available', False):
            lbl_lucj_stability.value = (
                "<div style='font-size:13px;color:#888;margin:3px 0;'>"
                "🛡 no history in range — filter skipped</div>")
            return
        if stab.get('relaxed'):
            lbl_lucj_stability.value = (
                f"<div style='font-size:13px;color:#e67e22;background:#fef9e7;"
                f"padding:4px 8px;border-radius:4px;border:1px solid #fdebd0;"
                f"margin:3px 0;'>🛡 filter too strict for norb="
                f"{esc(str(norb))} — re-admitted "
                f"{esc(str(stab.get('n_readmitted')))} of "
                f"{esc(str(stab.get('n_excluded_initial')))} "
                f"(most-volatile still excluded)</div>")
        else:
            n = len(stab.get('excluded_final', []))
            lbl_lucj_stability.value = (
                f"<div style='font-size:13px;color:#1e8449;background:#eafaf1;"
                f"padding:4px 8px;border-radius:4px;border:1px solid #abebc6;"
                f"margin:3px 0;'>🛡 filtered {esc(str(n))} volatile qubits "
                f"(window {esc(str(stab.get('window_days')))}d)</div>")

    def _cached_history_for(backend_name):
        """Assemble a per-backend history dict from the DISK CACHE ONLY (no
        remote fetch). Empty when the backend has no cached snapshots."""
        from .. import data
        try:
            known = data.list_known_calibrations(backend_name)
        except Exception:
            return {}
        hist = {}
        for entry in known or []:
            ts = entry.get('last_update')
            if not ts:
                continue
            cached = data._load_calibration(backend_name, ts)
            if cached:
                hist[ts] = {'qubit_props': cached[0], 'gate_props': cached[1]}
        return hist

    def _stability_badge(r):
        stab = r.get('stability')
        if not stab:
            return ""
        if not stab.get('available', False):
            return ("<span style='color:#999;'>stability: n/a "
                    "(no cached history)</span>")
        n = len(stab.get('excluded_final', []))
        extra = " (relaxed)" if stab.get('relaxed') else ""
        return (f"<span style='color:#1e8449;'>🛡 −{esc(str(n))} "
                f"qubits{extra}</span>")

    def _scan_circuit_score(props):
        """Per-backend circuit-level scorer: uses the shim's own arrays via
        the shared core, so every scan entry lands on the SAME scale as the
        local C1 box. Returns None when the shim can't score."""
        def _fn(entry):
            qubits = (list(entry['alpha']) + list(entry['beta'])
                      + list(entry['ancillas']))
            if not hasattr(props, 'score_layout'):
                return None
            res = props.score_layout(qubits)
            return res['fidelity'] if res else None
        return _fn

    def _annotate_scan_circuit_lqs(scan_result, get_backend, get_props):
        """(A3/S2) Fill each scan entry's lqs_circuit with THAT backend's
        circuit-level fidelity, so the scan table/dropdowns display the same
        scale as the local sweep. Runs on the worker thread (props are
        cache-first). A per-backend failure leaves lqs_circuit as-is (None →
        the '(template)' fallback still applies to genuine failures only)."""
        for r in scan_result.get('results', []):
            sweep = r.get('sweep')
            if not sweep:
                continue
            try:
                props = get_props(get_backend(r['name']))
            except Exception:
                continue
            # (V2) Surface the calibration provenance the shim used.
            r['vintage'] = getattr(props, 'vintage', None)
            scorer = _scan_circuit_score(props)
            for k, entries in sweep.get('strata', {}).items():
                for e in entries:
                    if e.get('lqs_circuit') is None:
                        try:
                            e['lqs_circuit'] = scorer(e)
                        except Exception as exc:
                            # (F3) Annotation failure is data absence — surface
                            # it in the notices with backend + stratum + reason,
                            # never a silent 0.0.
                            e['lqs_circuit'] = None
                            _log.warning("circuit score unavailable for %s "
                                         "k=%d: %s", r['name'], k, exc)

    def _scan_backends(eligible, norb, get_backend, get_props, use_filter):
        """The SINGLE handler-side scan path. Per target it calls the sweep
        with EXACTLY _sweep_kwargs_from_ui() — the same params as the local
        sweep (incl. hybrid) — so scan and local can't diverge (P1). When
        use_filter is on and the backend has CACHED history, it routes through
        sweep_with_stability instead. Not qpu_scan.scan_qpus, which is frozen
        and can't forward hybrid; this mirrors its contract (per-backend
        isolation, progress_cb, best_lqs sort) plus stability + vintage.
        """
        from ..ancilla_sweep import sweep_ancilla_layouts
        from ..stability_filter import sweep_with_stability
        kwargs = _sweep_kwargs_from_ui()
        window = rb_stability_window.value
        threshold = ft_stability_threshold.value
        results = []
        n_scanned = n_failed = n_skipped = 0
        for t in eligible:
            name = t.get('name') if isinstance(t, dict) else t
            family = t.get('family') if isinstance(t, dict) else None
            _on_scan_progress(name, 'start', 'fetching calibration…')
            try:
                backend = get_backend(name)
                props = get_props(backend)
                hist = _cached_history_for(name) if use_filter else None
                if hist:
                    # sweep_with_stability (frozen) can't take hybrid; it uses
                    # the engine default. Forward the shared min_bridges/top_n.
                    res = sweep_with_stability(
                        norb, backend, props, hist, window_days=window,
                        threshold=threshold,
                        min_bridges=kwargs['min_bridges'],
                        top_n=kwargs['top_n'])
                    stab = res.get('stability', {'available': False})
                else:
                    res = sweep_ancilla_layouts(norb, backend, props, **kwargs)
                    stab = ({'available': False, 'reason': 'no cached history'}
                            if use_filter else None)
            except Exception as exc:
                n_failed += 1
                results.append({'name': name, 'family': family, 'sweep': None,
                                'best_lqs': None, 'best_k': None,
                                'error': str(exc),
                                'stability': ({'available': False}
                                              if use_filter else None)})
                _on_scan_progress(name, 'error', str(exc))
                continue
            best_lqs, best_k = None, None
            for k, entries in res['strata'].items():
                for e in entries:
                    if best_lqs is None or e['product_score'] > best_lqs:
                        best_lqs, best_k = e['product_score'], k
            if best_lqs is None:
                n_skipped += 1
                results.append({'name': name, 'family': family, 'sweep': res,
                                'best_lqs': None, 'best_k': None,
                                'error': None, 'stability': stab})
                _on_scan_progress(name, 'skipped', 'no candidates')
                continue
            n_scanned += 1
            results.append({'name': name, 'family': family, 'sweep': res,
                            'best_lqs': best_lqs, 'best_k': best_k,
                            'error': None, 'stability': stab})
            _on_scan_progress(name, 'ok', f'best LQS {best_lqs * 100:.1f}%')
        ranked = [r for r in results if r['best_lqs'] is not None]
        rest = [r for r in results if r['best_lqs'] is None]
        ranked.sort(key=lambda r: r['best_lqs'], reverse=True)
        return {'results': ranked + rest, 'n_scanned': n_scanned,
                'n_failed': n_failed, 'n_skipped': n_skipped}

    _scan = {'progress': {}, 'order': []}
    _STATUS_ICON = {'start': '⏳', 'ok': '✓', 'skipped': '·', 'error': '✗'}

    def _clear_scan_results():
        # (C2) Full reset — table, per-QPU dropdowns, notices, progress line,
        # and the scan button. No ghost dropdown state survives.
        scan_results_box.children = []
        lbl_scan_progress.value = ""
        _scan['progress'].clear()
        _scan['order'].clear()
        _scan['current'] = None
        _scan['active'] = None
        btn_scan_qpus.disabled = False
        btn_scan_qpus.description = '🌐 Scan all QPUs'
    btn_scan_clear.on_click(lambda b: _clear_scan_results())

    def _render_progress_line():
        chips = []
        for name in _scan['order']:
            st, detail = _scan['progress'].get(name, ('', ''))
            txt = f"{esc(str(name))} {_STATUS_ICON.get(st, '')}"
            if st in ('start', 'error') and detail:
                txt += f" {esc(str(detail))}"
            chips.append(txt)
        lbl_scan_progress.value = (
            ("<div style='font-size:13px;color:#555;margin:3px 0;'>"
             + " · ".join(chips) + "</div>") if chips else "")

    def _on_scan_progress(name, status, detail=''):
        # Called from the WORKER thread — NEVER touch widgets here. Marshal
        # the widget mutation onto the main loop.
        # Track the active backend SYNCHRONOUSLY (before marshaling) so log
        # records emitted by the sweep that follows are attributed correctly.
        if status == 'start':
            _scan['current'] = name
        def _apply():
            if name not in _scan['progress']:
                _scan['order'].append(name)
            _scan['progress'][name] = (status, detail)
            _render_progress_line()
        scan_marshal(_apply)

    def _entry_label(entry):
        k = len(entry['ancillas'])
        pct, is_tmpl = _entry_lqs_pct(entry)
        suffix = " (template)" if is_tmpl else ""
        return f"k={k} · #{entry['rank_in_stratum']} · LQS {pct}{suffix}"

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

    def _vintage_cell(vintage):
        """(L2) Calibration provenance. Scan scoring is always LIVE now, so
        every remote row is 'live · <fetch time>' by construction. The old
        cached-snapshot stale-badge path (utc parse + >24h compare) is gone —
        the scan never scores from a cached snapshot."""
        if not vintage:
            return "<span style='color:#999;'>—</span>"
        return f"<span style='color:#1e8449;'>{esc(str(vintage))}</span>"

    def _render_results(scan_result):
        results = scan_result['results']
        head = ("<table style='font-size:13px;border-collapse:collapse;'>"
                "<tr style='background:#eaf2f8;'>"
                "<th style='padding:3px 8px;text-align:left;'>QPU</th>"
                "<th style='padding:3px 8px;'>family</th>"
                "<th style='padding:3px 8px;'>best LQS</th>"
                "<th style='padding:3px 8px;'>k</th>"
                "<th style='padding:3px 8px;'>cands</th>"
                "<th style='padding:3px 8px;'>stability</th>"
                "<th style='padding:3px 8px;'>calib</th></tr>")
        body = []
        # (P2) Active backend reference row — greyed, quotes the LOCAL number,
        # never re-searched. Keeps the cross-QPU reference point at zero cost.
        active = _scan.get('active')
        if active:
            lb = _local_best_lqs[0]
            lqs_ref = (f"{lb * 100.0:.1f}% (local)" if lb is not None
                       else "run Auto above")
            body.append(
                f"<tr style='color:#888;background:#f0f0f0;'>"
                f"<td style='padding:3px 8px;'>📍 {esc(str(active['name']))}</td>"
                f"<td style='padding:3px 8px;text-align:center;'>"
                f"{esc(str(active['family'] or '?'))}</td>"
                f"<td style='padding:3px 8px;text-align:center;'>"
                f"{esc(lqs_ref)}</td>"
                f"<td colspan='4' style='padding:3px 8px;'>current backend — "
                f"see local results above</td></tr>")
        for r in results:
            name = esc(str(r['name']))
            fam = esc(str(r['family'] or '?'))
            badge = _stability_badge(r)
            vcell = _vintage_cell(r.get('vintage'))
            if r['error']:
                body.append(
                    f"<tr style='color:#999;background:#f7f7f7;'>"
                    f"<td style='padding:3px 8px;'>{name}</td>"
                    f"<td style='padding:3px 8px;'>{fam}</td>"
                    f"<td colspan='5' style='padding:3px 8px;'>✗ "
                    f"{esc(str(r['error']))}</td></tr>")
            elif r['best_lqs'] is None:
                body.append(
                    f"<tr style='color:#999;'>"
                    f"<td style='padding:3px 8px;'>{name}</td>"
                    f"<td style='padding:3px 8px;'>{fam}</td>"
                    f"<td colspan='3' style='padding:3px 8px;'>no candidates"
                    f"</td>"
                    f"<td style='padding:3px 8px;text-align:center;'>"
                    f"{badge}</td>"
                    f"<td style='padding:3px 8px;text-align:center;'>"
                    f"{vcell}</td></tr>")
            else:
                # (F3) Summary best-LQS = MAX circuit LQS over ALL entries,
                # ignoring None (annotation failed) and NOT just the k_max
                # stratum #1 — quoting k_max quoted its 0.0 dead-edge wall
                # while lower strata were fine (SYM-2). Fall back to the
                # product heuristic (flagged) only when NOTHING was scored.
                lqs_vals = [e['lqs_circuit']
                            for _entries in r['sweep']['strata'].values()
                            for e in _entries
                            if e.get('lqs_circuit') is not None]
                if lqs_vals:
                    lqs = esc(f"{max(lqs_vals) * 100.0:.1f}%")
                else:
                    lqs = esc(f"{r['best_lqs'] * 100.0:.1f}% (template)")
                ncand = sum(len(v) for v in r['sweep']['strata'].values())
                cap = _k_cap_phrase(_scan.get('norb', 0),
                                    r['sweep'].get('k_max_effective'))
                cap_html = (f"<br><span style='color:#e67e22;'>{esc(cap)}</span>"
                            if cap else "")
                body.append(
                    f"<tr><td style='padding:3px 8px;'>{name}</td>"
                    f"<td style='padding:3px 8px;text-align:center;'>{fam}</td>"
                    f"<td style='padding:3px 8px;text-align:center;'>{lqs}</td>"
                    f"<td style='padding:3px 8px;text-align:center;'>"
                    f"{esc(str(r['best_k']))}</td>"
                    f"<td style='padding:3px 8px;text-align:center;'>"
                    f"{esc(str(ncand))}</td>"
                    f"<td style='padding:3px 8px;text-align:center;'>"
                    f"{badge}{cap_html}</td>"
                    f"<td style='padding:3px 8px;text-align:center;'>"
                    f"{vcell}</td></tr>")
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

    def _append_notices(notices):
        # (N2) Collapsible "⚠ notices (N)" under the results table:
        # deduplicated messages (message → ×count), grouped by backend
        # context. Hidden when empty.
        if not notices:
            return
        from collections import OrderedDict
        groups = OrderedDict()
        for ctx, msg in notices:
            groups.setdefault(ctx, OrderedDict())
            groups[ctx][msg] = groups[ctx].get(msg, 0) + 1
        parts = []
        for ctx, msgs in groups.items():
            parts.append(f"<b>{esc(str(ctx))}</b>")
            for msg, cnt in msgs.items():
                suffix = f" ×{cnt}" if cnt > 1 else ""
                parts.append(f"&nbsp;&nbsp;{esc(msg)}{esc(suffix)}")
        html = (
            f"<details style='margin:6px 0;font-size:12px;color:#666;'>"
            f"<summary>⚠ notices ({len(notices)})</summary>"
            f"<div style='padding:4px 8px;'>" + "<br>".join(parts) + "</div>"
            f"</details>")
        scan_results_box.children = list(scan_results_box.children) + [HTML(html)]

    def _on_scan_done(payload):
        scan_result, notices = payload
        _render_results(scan_result)
        _append_notices(notices)
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
        from ..qpu_scan import list_scan_targets, _detect_family
        from ..props_shim import make_scan_get_props

        norb = lucj_norb_slider.value
        try:
            family = _detect_family(state.active_backend)
        except Exception:
            family = None

        _clear_scan_results()                   # (C2) drop any stale results
        targets = list_scan_targets(
            state, norb, same_family_as=family,
            include_other_families=chk_scan_other_families.value)

        # (P2) Never re-search the active backend — mark it skipped and quote
        # the local result instead. No sweep, no fetch for it.
        active_name = getattr(state, 'active_backend_name', None)
        _scan['active'] = None
        for t in targets:
            if t.get('name') == active_name:
                t['eligible'] = False
                t['reason'] = 'current backend — see local results above'
                _scan['active'] = {'name': t['name'], 'family': t.get('family')}
        _render_targets(targets)                # immediate feedback

        eligible = [t for t in targets if t['eligible']]
        if not eligible:
            lbl_scan_progress.value = (
                "<div style='color:#e67e22;font-size:13px;'>No eligible "
                "backends for this norb / family scope.</div>")
            return

        _scan['norb'] = norb
        btn_scan_qpus.disabled = True
        btn_scan_qpus.description = '⏳ Scanning...'
        get_backend = state.active_service.backend
        get_props = make_scan_get_props(state)
        use_scan_filter = (_stability_filter_active()
                           and chk_scan_stability.value)

        def _work():
            # (N2/N3) Capture WARNING+ off 'qpu_mapper' for the scan's
            # duration and stop them leaking under the cell (propagate=False).
            # Attach in try, restore BOTH in finally (done + error paths).
            logger = logging.getLogger('qpu_mapper')
            orig_propagate = logger.propagate
            handler = _ScanNoticeCapture(lambda: _scan.get('current') or 'general')
            try:
                logger.addHandler(handler)
                logger.propagate = False
            except Exception:
                pass
            try:
                # (P1) One scan path, one param source — forwards hybrid.
                result = _scan_backends(eligible, norb, get_backend,
                                        get_props, use_scan_filter)
                # (A3/S2) Circuit-level LQS per backend — one scale everywhere.
                _annotate_scan_circuit_lqs(result, get_backend, get_props)
                return (result, list(handler.records))
            finally:
                try:
                    logger.removeHandler(handler)
                    logger.propagate = orig_propagate
                except Exception:
                    pass
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
