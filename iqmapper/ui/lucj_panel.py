"""
ui.lucj_panel.py — LUCJ layout generation callbacks.

Extracted from ui.py for maintainability. All functions receive
a `ctx` SimpleNamespace with shared UI state and widgets.
"""
import contextlib
import logging

from ipywidgets import Button, Dropdown, HBox, HTML, Layout

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


def _k_cap_phrase(norb, k_effective):
    """'k capped at E (requested R)' when the geometric request
    ceil(norb/4) exceeds what actually fit; '' otherwise. Pure metadata —
    replaces the operator's need to read the repeated stderr warning."""
    request = (norb + 3) // 4
    if k_effective is not None and k_effective < request:
        return f"k capped at {k_effective} (requested {request})"
    return ""


#: Message templates kept out of the notices panel. The behaviour behind them
#: is expected and needs no action, so they go to the log file only.
_LOG_ONLY_NOTICES = (
    'Requested %d bridges does not fit on this backend',
)


def _log_only(record):
    msg = record.msg if isinstance(record.msg, str) else ''
    return any(msg.startswith(t) for t in _LOG_ONLY_NOTICES)


class _ScanNoticeCapture(logging.Handler):
    """Captures WARNING+ records off the 'iqmapper' logger during a
    scan, tagging each with the current backend context. Records matching
    `_LOG_ONLY_NOTICES` are skipped (the log file still gets them). emit()
    never raises out — a logging hiccup must not break the scan."""

    def __init__(self, context_getter):
        super().__init__(level=logging.WARNING)
        self._context_getter = context_getter
        self.records = []            # list of (backend_context, message)

    def emit(self, record):
        try:
            if _log_only(record):
                return
            self.records.append((self._context_getter(), record.getMessage()))
        except Exception:
            pass


@contextlib.contextmanager
def _capture_qpu_notices(context_getter):
    """Capture WARNING+ records off the 'iqmapper' logger for the duration of
    a sweep and stop them leaking under the notebook cell (propagate=False).
    Attaches the handler in the try; restores BOTH the handler and propagate in
    finally, so a raising sweep can't leave the logger muted or double-handled.
    Yields the handler — read `handler.records` (a list of (context, message))
    after (or inside) the block.

    Shared by every sweep path: the multi-QPU scan, the local single sweep, and
    the multi-circuit sweep + replan. Restoration is the try/finally guarantee
    the tests pin."""
    logger = logging.getLogger('iqmapper')
    orig_propagate = logger.propagate
    handler = _ScanNoticeCapture(context_getter)
    try:
        logger.addHandler(handler)
        logger.propagate = False
    except Exception:
        pass
    try:
        yield handler
    finally:
        try:
            logger.removeHandler(handler)
            logger.propagate = orig_propagate
        except Exception:
            pass


class _SnapshotProps:
    """UI-layer adapter: dict-array calibration (state.qubit_props_data /
    gate_props_data + edges_list) → the BackendProperties surface the
    auto-LUCJ search calls (``readout_error`` / ``gate_error`` / ``t1`` /
    ``t2``). Lets the search score on the LOADED snapshot — the SAME arrays
    C1 and the error map read — instead of live backend.properties().

    Does NOT touch data.py / auto_lucj.py: it only reshapes what the UI
    already holds. Unknown qubits/edges/keys return NaN, which the search's
    existing NaN-penalty path handles.
    """
    def __init__(self, qubit_props, gate_props, edges):
        self._ro = qubit_props.get('readout_error')
        self._t1 = qubit_props.get('t1')
        self._t2 = qubit_props.get('t2')
        # 2Q error: prefer cz_error, else the first *_error key present.
        self._gkey = ('cz_error' if 'cz_error' in gate_props else
                      next((k for k in gate_props if k.endswith('_error')), None))
        self._garr = gate_props.get(self._gkey) if self._gkey else None
        # Map both edge orders to the positional gate array index.
        self._edge_ix = {}
        for i, e in enumerate(edges):
            u, v = int(e[0]), int(e[1])
            self._edge_ix.setdefault((u, v), i)
            self._edge_ix.setdefault((v, u), i)

    @staticmethod
    def _at(arr, i):
        try:
            return float(arr[i])
        except Exception:
            return float('nan')

    def readout_error(self, qubit):
        return self._at(self._ro, qubit) if self._ro is not None else float('nan')

    def t1(self, qubit):
        # Arrays store µs (constants: 'T1 (µs)'); the search multiplies by 1e6
        # expecting seconds — so hand back seconds.
        v = self._at(self._t1, qubit) if self._t1 is not None else float('nan')
        return v / 1e6

    def t2(self, qubit):
        v = self._at(self._t2, qubit) if self._t2 is not None else float('nan')
        return v / 1e6

    def gate_error(self, gate_name, edge):
        if self._garr is None:
            return float('nan')
        i = self._edge_ix.get((int(edge[0]), int(edge[1])))
        return float('nan') if i is None else self._at(self._garr, i)


def setup_lucj_callbacks(ctx):
    """Wire all LUCJ-related callbacks. Called from ui.py after widget creation."""

    # Unpack context for readability
    state = ctx.state
    out = ctx.out
    W = ctx.W

    lucj_norb_slider = ctx.lucj_norb_slider
    lucj_ancilla_dd = ctx.lucj_ancilla_dd
    lucj_min_ancilla = ctx.lucj_min_ancilla
    lucj_top_n = ctx.lucj_top_n
    lucj_search_cap = ctx.lucj_search_cap
    lucj_chk_score = ctx.lucj_chk_score
    chk_deep_search = ctx.chk_deep_search
    chk_lucj_diverse = ctx.chk_lucj_diverse
    chk_diverse_cap = getattr(ctx, 'chk_diverse_cap', None)
    chk_guided_search = getattr(ctx, 'chk_guided_search', None)
    int_beam_width = getattr(ctx, 'int_beam_width', None)
    int_refine_top = getattr(ctx, 'int_refine_top', None)
    guided_params_box = getattr(ctx, 'guided_params_box', None)
    btn_lucj_run = ctx.btn_lucj_run
    lbl_lucj_status = ctx.lbl_lucj_status
    lbl_lucj_trunc = ctx.lbl_lucj_trunc
    lbl_lucj_notices = ctx.lbl_lucj_notices
    dd_lucj_layout = ctx.dd_lucj_layout
    lbl_lucj_detail = ctx.lbl_lucj_detail

    lucj_multi_norbs = ctx.lucj_multi_norbs
    lucj_multi_ancillas = ctx.lucj_multi_ancillas
    lucj_buffer_slider = ctx.lucj_buffer_slider
    chk_fast_multi = getattr(ctx, 'chk_fast_multi', None)
    int_plans_kept = getattr(ctx, 'int_plans_kept', None)
    int_cands_per_circuit = getattr(ctx, 'int_cands_per_circuit', None)
    ft_boundary_penalty = getattr(ctx, 'ft_boundary_penalty', None)
    joint_params_box = getattr(ctx, 'joint_params_box', None)
    btn_lucj_multi = ctx.btn_lucj_multi
    btn_lucj_multi_sweep = ctx.btn_lucj_multi_sweep
    btn_scan_batch = ctx.btn_scan_batch
    scan_batch_box = ctx.scan_batch_box
    lbl_lucj_multi = ctx.lbl_lucj_multi
    lbl_lucj_multi_notices = ctx.lbl_lucj_multi_notices
    lucj_multi_detail_box = ctx.lucj_multi_detail_box

    set_num_circuits = ctx.set_num_circuits  # count change w/o full view rebuild
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
        # Sweep entries carry per-circuit ancillas; paint them as the ancilla
        # role. Legacy candidates have no 'ancillas' key → nothing changes.
        anc_fill = 'none' if line_only else hex_to_rgba(d['defaults']['fill_anc'], 1.0)
        d['ancilla'] = [str(q) for q in candidate.get('ancillas', [])
                        if str(q) in state.qubit_ids]
        for q in d['ancilla']:
            d['styles'][q] = {'fill': anc_fill,
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

    # ── Calibration epoch: search must use the LOADED snapshot ───
    def _snapshot_label():
        """A display id for the active NON-LIVE snapshot, else None (live)."""
        if getattr(state, 'time_mode', 'live') == 'live':
            return None
        ts = getattr(state, 'point_timestamp', None)
        if isinstance(ts, str) and len(ts) >= 10:
            return ts[:10]
        return getattr(state, 'calibration_data_label', None) or 'loaded snapshot'

    def _local_scoring_props():
        """The props the auto-LUCJ search should score on. Live mode → the live
        BackendProperties (unchanged). A loaded snapshot → an adapter over the
        SAME arrays C1/the error map read, so ranking shares their epoch. Falls
        back to initial_props ONLY if the arrays are absent — a scoring-input
        choice must never break generation."""
        if getattr(state, 'time_mode', 'live') == 'live':
            return state.initial_props
        qp = getattr(state, 'qubit_props_data', None)
        gp = getattr(state, 'gate_props_data', None)
        edges = getattr(state, 'edges_list', None)
        if not qp or not gp or edges is None or 'readout_error' not in qp:
            return state.initial_props
        try:
            return _SnapshotProps(qp, gp, list(edges))
        except Exception:
            return state.initial_props

    def _epoch_suffix():
        """B2: '· scored on <date>' for a non-live snapshot, else ''."""
        lbl = _snapshot_label()
        return f" · scored on {esc(str(lbl))}" if lbl else ""

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
            # Compact counts only — the placed qubit ids live on the map and in
            # the Export tab, so the detail panel need not dump index lists.
            counts = (f"α={len(entry['alpha'])} β={len(entry['beta'])} "
                      f"Anc={len(entry['ancillas'])}")
            lbl_lucj_detail.value = (
                f"<div style='font-size:14px;background:#f8f9fa;padding:6px 8px;"
                f"border-radius:4px;border:1px solid #dee2e6;margin:3px 0;'>"
                f"{header}<br>{counts}</div>"
            )
        else:
            sc = f"{entry['score']:.4f}" if entry['score'] is not None else "N/A"
            counts = f"α={len(entry['alpha'])} β={len(entry['beta'])}"
            lbl_lucj_detail.value = (
                f"<div style='font-size:14px;background:#f8f9fa;padding:6px 8px;"
                f"border-radius:4px;border:1px solid #dee2e6;margin:3px 0;'>"
                f"<b>#{entry['rank']}</b> (err={sc})<br>{counts}</div>"
            )
    # The LQS shown here is the circuit-level score — the same number
    # the Estimated Layout-Quality box reports. '(template)' flags the
    # internal heuristic used only when circuit scoring is unavailable.
    with contextlib.suppress(Exception):
        dd_lucj_layout.tooltip = (
            'LQS = circuit-level fidelity; "(template)" = fallback '
            'heuristic \u00b7 README: LQS')
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

    # ── circuit-level LQS for a sweep candidate ──────────────
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

    # ── Temporal anchoring: one epoch per search ─────────────────────
    def _selected_date_anchor():
        """The selected calibration's date 'YYYY-MM-DD' when a PAST date
        is active, else None (live). Used to anchor the stability window and
        the epoch-mixing header note to the user's chosen date."""
        if getattr(state, 'time_mode', 'live') == 'live':
            return None
        ts = getattr(state, 'point_timestamp', None)
        return ts[:10] if isinstance(ts, str) and len(ts) >= 10 else None

    def _local_scored_vintage():
        """Provenance stamp for the LOCAL Auto sweep, mirroring the scan
        detail's 'live · HH:MM'. A selected PAST date → 'scored on <date>';
        otherwise the live fetch moment → 'scored on live · HH:MM'. Makes the
        two-moments-two-numbers reality visible so a scan-vs-local mismatch
        reads as calibration drift, not a bug. esc()'d value."""
        anchor = _selected_date_anchor()
        if anchor:
            return f"scored on {esc(anchor)}"
        from datetime import datetime as _dt
        return f"scored on live · {_dt.now().strftime('%H:%M')}"

    # ── SINGLE SOURCE of sweep params — local AND scan read this ──
    def _sweep_kwargs_from_ui():
        """The one place sweep parameters are derived from the UI. Both the
        local sweep and every scan per-target call pass exactly this, so the
        two can never diverge again (a divergence here silently skews rankings)."""
        min_val = lucj_min_ancilla.value
        min_bridges = None if (min_val is None or min_val <= 0) else min_val
        top_val = lucj_top_n.value
        cap_val = lucj_search_cap.value
        return {
            'min_bridges': min_bridges,
            'top_n': top_val if (top_val and top_val >= 1) else 3,
            'hybrid': chk_deep_search.value,
            'mapping_cap': cap_val if (cap_val and cap_val >= 1) else None,
            # 0.8 default when ON; None → exact top-N (byte-compatible).
            'diversity_threshold': 0.8 if chk_lucj_diverse.value else None,
            # The cap counts distinct candidates (engine #17). VF2-only —
            # combined_sweep passes it to the exhaustive half only.
            'diverse_cap': bool(getattr(chk_diverse_cap, 'value', True)),
        }

    # ── ONE dispatch point for the search ────────────────────────
    def _guided_on():
        """Guided search checkbox; absent widget → on (the default)."""
        return bool(getattr(chk_guided_search, 'value', True))

    def _guided_params():
        return {
            'beam_width': int(getattr(int_beam_width, 'value', 200) or 200),
            'refine_top': int(getattr(int_refine_top, 'value', 5) or 5),
        }

    def _run_sweep(norb, backend, props, **kwargs):
        """Run the search: exhaustive, plus guided when its checkbox is on.

        `engine_select.combined_sweep` keeps the best of both engines per
        ancilla count and returns the exhaustive sweep's dict shape, so every
        caller below and all the display code downstream is unchanged. It
        routes each engine-specific argument to its own engine.
        """
        from ..engine_select import combined_sweep
        return combined_sweep(norb, backend, props, use_guided=_guided_on(),
                              **kwargs, **_guided_params())

    def _log_reservoir_notices(res):
        """The per-stratum cap notices go to the log file, not the UI."""
        for note in (res or {}).get('reservoir_notices') or []:
            _log.info("%s", note)

    def _source_tag(entry):
        """' · G' (guided only), ' · E+G' (both engines) or ' · E'
        (exhaustive) for a search entry; '' for one with no source."""
        source = entry.get('source')
        if not source:
            return ""
        return {'guided': ' · G', 'both': ' · E+G'}.get(source, ' · E')

    def _on_guided_toggle(_change=None):
        """Beam parameters show only while Guided search is on."""
        if guided_params_box is not None:
            guided_params_box.layout.display = '' if _guided_on() else 'none'

    if chk_guided_search is not None:
        chk_guided_search.observe(_on_guided_toggle, names='value')
        _on_guided_toggle()

    # ── Multi-circuit planner: Joint (default) or Sequential (Fast) ──
    def _selected_planner():
        """'sequential' when "Fast: one circuit at a time" is checked, else
        'joint' (also when the widget is absent)."""
        return 'sequential' if getattr(chk_fast_multi, 'value', False) \
            else 'joint'

    def _joint_params():
        return {
            'beam_width': int(getattr(int_plans_kept, 'value', 3) or 3),
            'branch': int(getattr(int_cands_per_circuit, 'value', 3) or 3),
            'boundary_penalty': float(
                getattr(ft_boundary_penalty, 'value', 0.0) or 0.0),
        }

    def _run_multi(norbs, backend, props, *, sweep_kwargs, planner=None,
                   **seq_kwargs):
        """Place several circuits with `planner`, or the selected one.

        Sequential passes `seq_kwargs` to `multi_sweep.sweep_multi_circuits`
        untouched, so each call site keeps its exact call. Joint plans with
        `joint_plan.plan_multi_circuits`, searching each circuit through
        `_run_sweep` with `sweep_kwargs`, so it uses the combined search.
        """
        if (planner or _selected_planner()) != 'joint':
            from ..multi_sweep import sweep_multi_circuits
            return sweep_multi_circuits(norbs, backend, props, **seq_kwargs)

        from ..joint_plan import plan_multi_circuits
        scorer = seq_kwargs.get('circuit_score_fn')
        base = {k: v for k, v in sweep_kwargs.items() if k != 'min_bridges'}

        def _search(norb, *, exclude_qubits, min_bridges):
            return _run_sweep(norb, backend, props, circuit_score_fn=scorer,
                              exclude_qubits=exclude_qubits,
                              min_bridges=min_bridges, **base)

        return plan_multi_circuits(
            norbs, backend, props, search_fn=_search,
            buffer_qubits=seq_kwargs.get('buffer_qubits', 1),
            exclude_qubits=seq_kwargs.get('exclude_qubits'),
            min_bridges=seq_kwargs.get('min_bridges'),
            top_n=seq_kwargs.get('top_n', 3), circuit_score_fn=scorer,
            **_joint_params())

    def _replan_multi(result, index, new_choice):
        """Replan with the planner that produced `result`."""
        if result.get('planner') == 'joint':
            from ..joint_plan import replan_from
        else:
            from ..multi_sweep import replan_from
        return replan_from(result, index, new_choice)

    def _joint_pending_html(n_circuits):
        """'running…' line for a joint run: the worst-case search count."""
        from ..joint_plan import worst_case_searches
        n = worst_case_searches(n_circuits, _joint_params()['beam_width'])
        return (f"<div style='font-size:13px;color:#555;margin:3px 0;'>"
                f"⏳ joint plan · up to {n} searches</div>")

    def _joint_gain_text(result):
        """' · +X% vs one-by-one', ' · places 3 vs 2 one-by-one', or ''
        (a sequential result, or no gain to report)."""
        if result.get('planner') != 'joint':
            return ""
        greedy, joint = result.get('greedy_total'), result.get('joint_total')
        placed = len(result.get('circuits') or [])
        greedy_placed = result.get('greedy_placed', placed)
        if placed > greedy_placed:
            return f" · places {placed} vs {greedy_placed} one-by-one"
        if greedy and joint is not None:
            gain = (joint / greedy - 1.0) * 100.0
            if round(gain, 1) > 0.0:
                return f" · +{gain:.1f}% vs one-by-one"
        return ""

    def _shared_couplers_html(result):
        shared = result.get('boundary_couplers') or []
        if not shared:
            return ""
        return ("<br>shared couplers: "
                + ", ".join(f"{u}–{v}" for u, v in shared))

    def _on_planner_change(_change=None):
        """Joint-only params show only while Fast is off."""
        if joint_params_box is not None:
            joint_params_box.layout.display = (
                '' if _selected_planner() == 'joint' else 'none')

    if chk_fast_multi is not None:
        chk_fast_multi.observe(_on_planner_change, names='value')
        _on_planner_change()

    # Best circuit-level LQS from the most recent LOCAL sweep — quoted as the
    # active backend's reference in the scan table, so the scan never
    # re-searches self yet keeps the reference point.
    _local_best_lqs = [None]

    # ── Legacy single-count search ───────────────────────────────
    def _generate_legacy(norb, requested_ancillas):
        from ..auto_lucj import get_all_zigzag_layouts

        all_layouts, num_ab = get_all_zigzag_layouts(
            num_orbitals=norb,
            backend=state.active_backend,
            score=lucj_chk_score.value,
            props=_local_scoring_props(),   # loaded snapshot, not live
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
                f"ancillas={num_ab}{ancilla_note}{_epoch_suffix()}</div>")
        else:
            sc = f"{best['score']:.4f}" if best['score'] is not None else "N/A"
            lbl_lucj_status.value = (
                f"<div style='font-size:14px;color:#1e8449;background:#eafaf1;padding:4px 8px;"
                f"border-radius:4px;border:1px solid #abebc6;margin:3px 0;'>"
                f"✅ <b>{len(all_layouts)}</b> layouts found (norb={norb}, "
                f"ancillas={num_ab}). Best: err={sc}{ancilla_note}{_epoch_suffix()}</div>")

    # ── Stratified ancilla sweep (new default) ───────────────────
    def _generate_sweep(norb):
        from ..ancilla_sweep import sweep_ancilla_layouts

        # exact same params the scan uses (min_bridges / top_n / hybrid).
        kwargs = _sweep_kwargs_from_ui()

        # pre-rank on the same epoch as the circuit-score (selected date).
        scoring_props = _local_scoring_props()
        anchor = _selected_date_anchor()          # window/header anchor
        # opt-in stability filter (drop-in superset of the plain sweep).
        history = getattr(state, 'cal_history', None)
        if _stability_filter_active() and history:
            from ..stability_filter import sweep_with_stability
            res = sweep_with_stability(
                norb, state.active_backend, scoring_props, history,
                window_days=rb_stability_window.value,
                threshold=ft_stability_threshold.value,
                min_bridges=kwargs['min_bridges'], top_n=kwargs['top_n'],
                mapping_cap=kwargs['mapping_cap'], today=anchor,
                quality_sigma=_floor_sigma())
            _render_stability_banner(res.get('stability'), norb, anchor=anchor)
            # The frozen stability engine can't take circuit_score_fn — fill
            # circuit-level LQS on its survivors here.
            _annotate_circuit_lqs(res)
        else:
            # Always-on quality floor for the UNfiltered path: outlier qubits
            # are removed up front via exclude_qubits (the filtered path gets
            # the same floor through sweep_with_stability's quality_sigma, so
            # they never both fire on one search — no double-exclusion).
            floor_excl = _floor_exclusions()
            res = _run_sweep(
                norb, state.active_backend, scoring_props,
                circuit_score_fn=_circuit_score,   # aligned circuit LQS
                exclude_qubits=(floor_excl or None),
                **kwargs)                          # single source (incl. hybrid)
            parts = []
            if _stability_filter_active():
                # Checkbox on but no history dict → skipped note.
                parts.append(
                    "<div style='font-size:13px;color:#888;margin:3px 0;'>"
                    "🛡 no history in range — filter skipped</div>")
            if floor_excl:
                parts.append(
                    f"<div style='font-size:13px;color:#1e8449;background:#eafaf1;"
                    f"padding:4px 8px;border-radius:4px;border:1px solid #abebc6;"
                    f"margin:3px 0;'>🛡 floor excluded {len(floor_excl)} "
                    f"outlier qubit(s)</div>")
            lbl_lucj_stability.value = "".join(parts)

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
                    f"k={k} · #{entry['rank_in_stratum']} · LQS {pct}{suffix}"
                    f"{_source_tag(entry)}",
                    len(flat),
                ))
                flat.append(entry)

        # Remember the best circuit-level LQS for the scan's active-row ref.
        circ = [e['lqs_circuit'] for e in flat
                if e.get('lqs_circuit') is not None]
        _local_best_lqs[0] = max(circ) if circ else None

        _seed_layout_dropdown(flat, dd_opts)

        # Truncation banner: visible only when the VF2 cap was hit.
        # Name the REAL cap (mapping_cap) and WHICH strata truncated — not
        # k_max's n_raw_embeddings, which is an embedding count, not the cap.
        # With the diverse reservoir engaged, the ℹ line below already says
        # this stratum's cap was reached AND how much it saw — so repeating
        # it as a warning is noise that also contradicts it ("ranking may be
        # non-exhaustive" reads as a problem; "kept 2,000 diverse candidates
        # of 4,498 seen" reads as what actually happened). Warn only about
        # strata the reservoir did NOT cover.
        _reservoir_ks = {k for k, st in (res.get('reservoir') or {}).items()
                         if st.get('kept')}
        _trunc_ks = sorted(
            (k for k, m in res.get('per_stratum', {}).items()
             if m.get('truncated') and k not in _reservoir_ks), reverse=True)
        if res['vf2_truncated'] and _trunc_ks:
            ks_txt = ', '.join(f"k={k}" for k in _trunc_ks)
            cap = res.get('mapping_cap')
            cap_txt = f"{cap:,}" if isinstance(cap, int) else "the"
            lbl_lucj_trunc.value = (
                f"<div style='font-size:14px;color:#e67e22;background:#fef9e7;"
                f"padding:4px 8px;border-radius:4px;border:1px solid #fdebd0;"
                f"margin:3px 0;'>⚠️ direct search truncated at "
                f"{esc(cap_txt)} mappings for {esc(ks_txt)} — ranking may "
                f"be non-exhaustive.</div>")
        else:
            lbl_lucj_trunc.value = ""

        # One short line: how many layouts the dropdown holds and when they
        # were scored. The dropdown labels carry k, LQS and source.
        _log_reservoir_notices(res)
        if flat:
            # Provenance stamp — mirrors the scan detail's 'live · HH:MM' so the
            # user can SEE this number and a scan row's number were taken at
            # DIFFERENT moments (calibration drift), not that one is wrong.
            vin_txt = f" · {_local_scored_vintage()}"
            lbl_lucj_status.value = (
                f"<div style='font-size:14px;color:#1e8449;background:#eafaf1;"
                f"padding:4px 8px;border-radius:4px;border:1px solid #abebc6;"
                f"margin:3px 0;'>✅ {len(flat)} layout"
                f"{'s' if len(flat) != 1 else ''} found{vin_txt}</div>")
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
        lbl_lucj_notices.value = ""
        try:
            requested_ancillas = lucj_ancilla_dd.value
            # Capture WARNING+ off 'iqmapper' for the sweep so bridge/VF2
            # notices land in the loud notices bar, not under the cell.
            with _capture_qpu_notices(
                    lambda: state.active_backend_name or 'local sweep') as handler:
                # Exact count picked (>= 0) → legacy single-count search,
                # unchanged. 'Auto' (-1) → stratified sweep (new default).
                if requested_ancillas is not None and requested_ancillas >= 0:
                    _generate_legacy(norb, requested_ancillas)
                else:
                    _generate_sweep(norb)
            lbl_lucj_notices.value = _notices_html(handler.records)
        except Exception as e:
            lbl_lucj_status.value = (
                f"<div style='font-size:14px;color:#e74c3c;background:#fdedec;padding:6px 10px;"
                f"border-radius:5px;border:1px solid #f5b7b1;margin:4px 0;'>"
                f"❌ <b>Error:</b> {esc(str(e))}</div>")
        finally:
            btn_lucj_run.disabled = False
            btn_lucj_run.description = '🤖 Generate LUCJ Layout'

    btn_lucj_run.on_click(_on_lucj_generate)

    # ── re-score the current candidates under a new epoch ─────────
    def _rescore_current_candidates():
        """Re-rank the candidate dropdown against the CURRENT epoch.

        Called after `benchmark` installs a measured epoch: the entries were
        ranked on the reported calibration, and the whole point of measuring
        is that the ordering may now be different. Re-scores through the same
        `_circuit_score` path the sweep itself uses, so the numbers stay
        comparable, then re-seeds the dropdown in the sweep's own label
        format. Legacy (explicit-count) entries carry 'score' instead of
        'lqs_circuit' and are left alone — they are not LQS-ranked.

        Returns the number of entries re-scored. Never raises: a benchmark
        that succeeded must not be undone by a re-ranking problem.
        """
        entries = list(_lucj_all_layouts)
        if not entries or not any('lqs_circuit' in e for e in entries):
            return 0
        try:
            for entry in entries:
                if 'lqs_circuit' in entry:
                    try:
                        entry['lqs_circuit'] = _circuit_score(entry)
                    except Exception:
                        entry['lqs_circuit'] = None
            ranked = sorted(
                entries,
                key=lambda e: (e.get('lqs_circuit') is not None,
                               e.get('lqs_circuit') or 0.0),
                reverse=True)
            dd_opts = []
            for i, entry in enumerate(ranked):
                pct, is_tmpl = _entry_lqs_pct(entry)
                suffix = " (template)" if is_tmpl else ""
                k = entry.get('n_ancillas', len(entry.get('ancillas') or []))
                dd_opts.append((
                    f"k={k} · #{entry.get('rank_in_stratum', i + 1)} · "
                    f"LQS {pct}{suffix}{_source_tag(entry)} · measured", i))
            _seed_layout_dropdown(ranked, dd_opts)
            circ = [e['lqs_circuit'] for e in ranked
                    if e.get('lqs_circuit') is not None]
            _local_best_lqs[0] = max(circ) if circ else None
            return len(ranked)
        except Exception:
            _log.exception("Re-ranking candidates under the measured epoch "
                           "failed; the epoch itself is applied")
            return 0

    # Deferred wiring: ui.benchmark_panel calls this after applying a measured
    # epoch. Same single-element-cell pattern as the inter-QPU handoff hook.
    _rescore_cell = getattr(ctx, 'rescore_cell', None)
    if _rescore_cell is not None:
        _rescore_cell[0] = _rescore_current_candidates

    # ── sweep mode always scores → hide the score checkbox ───────
    def _on_ancilla_mode_change(change=None):
        # 'Auto' (-1) == sweep mode; the engine always computes LQS there,
        # so the legacy "Score layouts" toggle is meaningless.
        sweep_mode = (lucj_ancilla_dd.value is not None
                      and lucj_ancilla_dd.value < 0)
        lucj_chk_score.disabled = sweep_mode
        # Hide rather than grey out. 'Auto' is the default, so a permanently
        # disabled checkbox was the first thing every user saw. The widget
        # stays in the tree (the legacy path still reads .value on run) and
        # only its display is toggled.
        lucj_chk_score.layout.display = 'none' if sweep_mode else ''
        if not sweep_mode:
            lucj_chk_score.tooltip = 'Score layouts (pick lowest error)'
    lucj_ancilla_dd.observe(_on_ancilla_mode_change, names='value')
    _on_ancilla_mode_change()  # set initial state (default -1 → hidden)

    # ── Multi-QPU scan ─────────────────────────────────
    btn_scan_qpus = ctx.btn_scan_qpus
    chk_scan_other_families = ctx.chk_scan_other_families
    btn_scan_clear = ctx.btn_scan_clear
    lbl_scan_progress = ctx.lbl_scan_progress
    scan_results_box = ctx.scan_results_box
    scan_spawn_bg = ctx.scan_spawn_bg
    scan_marshal = ctx.scan_marshal
    inter_qpu_handoff = ctx.inter_qpu_handoff  # shared cell → stage hook

    # ── Stability filter ──────────────────────────────
    chk_stability_filter = ctx.chk_stability_filter
    rb_stability_window = ctx.rb_stability_window
    ft_stability_threshold = ctx.ft_stability_threshold
    ft_stability_floor = ctx.ft_stability_floor
    stability_ctrl_box = ctx.stability_ctrl_box

    def _floor_sigma():
        """Quality-floor σ from the UI; 0 → None (floor off). Always on: the
        floor applies to every search path independent of the volatility
        filter (single, multi, scan)."""
        v = getattr(ft_stability_floor, 'value', None)
        return v if (v and v > 0) else None

    def _readout_array(props):
        """Per-qubit CURRENT readout array from a props object — the shim's own
        array when present, else built from readout_error(q)."""
        arr = getattr(props, '_readout', None)
        if arr is not None:
            return arr
        n = getattr(props, 'n_qubits', None) or len(
            getattr(state, 'qubit_ids', None) or [])
        if not n:
            return None
        out = []
        for q in range(n):
            try:
                out.append(props.readout_error(q))
            except Exception:
                out.append(float('nan'))
        return out

    def _floor_exclusions(props=None):
        """Physical qubit ids the always-on quality floor removes — the C1
        device-outlier rule promoted to a factory guard. Computed on `props`
        (per-backend, for the scan) or the local current props. Empty when
        σ=0 or no readout array is available."""
        sigma = _floor_sigma()
        if sigma is None:
            return set()
        from ..stability_filter import select_quality_floor
        if props is None:
            arr = (getattr(state, 'qubit_props_data', None) or {}).get('readout_error')
        else:
            arr = _readout_array(props)
        if arr is None:
            return set()
        return set(select_quality_floor(arr, sigma=sigma))
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

    def _render_stability_banner(stab, norb, anchor=None):
        """Render res['stability']: filtered / relaxed / unavailable.
        `anchor` (a 'YYYY-MM-DD' selected date) names the window's upper bound
        when a past date is active."""
        # "14d before <date>" when a past date anchors the window, else "14d".
        def _window_txt():
            wd = esc(str(stab.get('window_days')))
            return (f"{wd}d before {esc(anchor)}" if anchor else f"{wd}d")
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
            total = len(stab.get('excluded_final', []))
            m_floor = stab.get('n_floor_excluded', 0)      # outliers
            n_vol = total - m_floor
            floor_txt = f" + {esc(str(m_floor))} outlier" if m_floor else ""
            lbl_lucj_stability.value = (
                f"<div style='font-size:13px;color:#1e8449;background:#eafaf1;"
                f"padding:4px 8px;border-radius:4px;border:1px solid #abebc6;"
                f"margin:3px 0;'>🛡 filtered {esc(str(n_vol))} volatile"
                f"{floor_txt} qubits (window {_window_txt()})</div>")

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
        floor_m = r.get('floor_excluded', 0)
        parts = []
        if stab and stab.get('available', False):
            n = len(stab.get('excluded_final', []))
            extra = " (relaxed)" if stab.get('relaxed') else ""
            parts.append(f"<span style='color:#1e8449;'>🛡 −{esc(str(n))} "
                         f"qubits{extra}</span>")
        elif stab and not stab.get('available', False):
            parts.append("<span style='color:#999;'>stability: n/a "
                         "(no cached history)</span>")
        # Always-on floor: shown when it excluded outliers on the UNfiltered
        # path (the filtered −N badge above already counts them).
        if floor_m:
            parts.append(f"<span style='color:#1e8449;'>🛡 floor "
                         f"−{esc(str(floor_m))}</span>")
        return " ".join(parts)

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
        """Fill each scan entry's lqs_circuit with THAT backend's
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
            # Surface the calibration provenance the shim used.
            r['vintage'] = getattr(props, 'vintage', None)
            scorer = _scan_circuit_score(props)
            for k, entries in sweep.get('strata', {}).items():
                for e in entries:
                    if e.get('lqs_circuit') is None:
                        try:
                            e['lqs_circuit'] = scorer(e)
                        except Exception as exc:
                            # Annotation failure is data absence — surface
                            # it in the notices with backend + stratum + reason,
                            # never a silent 0.0.
                            e['lqs_circuit'] = None
                            _log.warning("circuit score unavailable for %s "
                                         "k=%d: %s", r['name'], k, exc)

    def _scan_backends(eligible, norb, get_backend, get_props, use_filter):
        """The SINGLE handler-side scan path. The per-backend isolation /
        progress / best-LQS ranking loop lives ONCE, in qpu_scan.scan_qpus; we
        inject a ``sweep_fn`` closure that carries the only handler-specific
        parts: the SAME params as the local sweep (_sweep_kwargs_from_ui, incl.
        hybrid & cap — so scan and local can't diverge, P1) plus, when the
        filter is on and the backend has CACHED history, the volatility +
        quality-floor filter via sweep_with_stability. The stability block the
        closure attaches to each sweep is lifted to the row level afterwards so
        the badges render unchanged.
        """
        from .. import qpu_scan
        from ..ancilla_sweep import sweep_ancilla_layouts
        from ..stability_filter import sweep_with_stability
        kwargs = _sweep_kwargs_from_ui()
        window = rb_stability_window.value
        threshold = ft_stability_threshold.value
        floor_sigma = _floor_sigma()          # scan inherits the floor

        def _sweep_fn(norb, backend, props, *, min_bridges=None, top_n=3):
            # min_bridges/top_n arrive from scan_qpus (they equal kwargs); the
            # shared UI source also supplies hybrid & mapping_cap.
            name = getattr(backend, 'name', None)
            hist = _cached_history_for(name) if (use_filter and name) else None
            if hist:
                res = sweep_with_stability(
                    norb, backend, props, hist, window_days=window,
                    threshold=threshold, min_bridges=kwargs['min_bridges'],
                    top_n=kwargs['top_n'], mapping_cap=kwargs['mapping_cap'],
                    quality_sigma=floor_sigma)
            else:
                # Always-on quality floor on THIS backend's live props.
                floor_excl = _floor_exclusions(props)
                res = sweep_ancilla_layouts(
                    norb, backend, props,
                    exclude_qubits=(floor_excl or None), **kwargs)
                res['stability'] = ({'available': False,
                                     'reason': 'no cached history'}
                                    if use_filter else None)
                res['floor_excluded'] = len(floor_excl)
            return res

        def _progress(name, status, detail=''):
            # Preserve the 'start' detail (scan_qpus emits it empty).
            if status == 'start' and not detail:
                detail = 'fetching calibration…'
            _on_scan_progress(name, status, detail)

        result = qpu_scan.scan_qpus(
            eligible, norb, get_backend, get_props,
            min_bridges=kwargs['min_bridges'], top_n=kwargs['top_n'],
            progress_cb=_progress, sweep_fn=_sweep_fn)

        # Lift each sweep's stability block to the row level (badges read
        # r['stability']); error rows (sweep=None) reflect filter availability.
        for r in result['results']:
            sweep = r.get('sweep')
            r['stability'] = (sweep.get('stability') if sweep is not None
                              else ({'available': False} if use_filter else None))
            r['floor_excluded'] = (sweep.get('floor_excluded', 0)
                                   if sweep is not None else 0)
        return result

    _scan = {'progress': {}, 'order': []}
    _STATUS_ICON = {'start': '⏳', 'ok': '✓', 'skipped': '·', 'error': '✗'}

    def _clear_scan_results():
        # Full reset — table, per-QPU dropdowns, notices, progress line,
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
        return (f"k={k} · #{entry['rank_in_stratum']} · LQS {pct}{suffix}"
                f"{_source_tag(entry)}")

    def _short_reason(reason):
        """Map a full ineligibility reason to a compact form for the one-line
        summary (the full string lives in the line's tooltip)."""
        r = str(reason or '')
        if r.startswith('current backend'):
            return 'current'
        if r.startswith('different family'):
            lo, hi = r.find('('), r.rfind(')')
            return r[lo + 1:hi] if 0 <= lo < hi else 'other family'
        if r.startswith('needs'):
            return 'too small'
        if r.startswith('family unknown'):
            return 'unknown family'
        return r or 'skipped'

    def _render_targets(targets):
        # One compact line: eligible names joined by ' · ' (+ count), then the
        # skipped ones after an em-dash with a SHORT reason. The full per-
        # backend reasons ride in the title tooltip.
        elig = [t for t in targets if t['eligible']]
        skip = [t for t in targets if not t['eligible']]
        skip_txt = ', '.join(
            f"{esc(str(t['name']))} ({esc(_short_reason(t.get('reason')))})"
            for t in skip)
        if elig:
            names = ' · '.join(esc(str(t['name'])) for t in elig)
            line = f"<b>Targets:</b> {names} ({len(elig)})"
            if skip:
                line += f" — skipped: {skip_txt}"
        else:
            line = "<b>Targets:</b> none"
            if skip:
                line += f" — {skip_txt}"
        tip = esc(' | '.join(
            (f"{t['name']}: eligible" if t['eligible']
             else f"{t['name']}: {t.get('reason')}")
            for t in targets))
        scan_results_box.children = [HTML(
            f"<div title='{tip}' style='font-size:13px;margin:3px 0;'>"
            f"{line}</div>")]

    _HANDOFF_STYLE = {'ok': ('✅', '#1e8449'), 'info': ('📡', '#555'),
                      'warn': ('⚠', '#e67e22'), 'error': ('❌', '#e74c3c')}

    def _handoff_reporter(label):
        """`report(level, text)` writing one short line into `label` — the
        status next to the button that was clicked."""
        def _report(level, text):
            icon, colour = _HANDOFF_STYLE.get(level, _HANDOFF_STYLE['info'])
            label.value = (f"<span style='font-size:13px;color:{colour};'>"
                           f"{icon} {esc(str(text))}</span>")
        return _report

    def _stage_handoff(backend_name, candidate, report):
        """Open a scan pick (one candidate, or a batch list) on QPU 2.

        Every outcome goes to `report`, so it shows next to the button that
        was clicked; a failure is logged as well, never swallowed. A handoff
        failure must not break the scan UI, so nothing is re-raised."""
        hook = inter_qpu_handoff[0] if inter_qpu_handoff else None
        if hook is None:
            report('error', "QPU 2 panel unavailable here — open from "
                            "QPU 1's scan")
            return
        try:
            hook(backend_name, candidate, report=report)
        except Exception as exc:                  # noqa: BLE001 - reported
            _log.warning("Opening %s on QPU 2 failed: %s", backend_name, exc,
                         exc_info=True)
            report('error', f"could not open {backend_name} on QPU 2: {exc}")

    def _vintage_cell(vintage):
        """Calibration provenance. Scan scoring is always LIVE now, so
        every remote row is 'live · <fetch time>' by construction. The old
        cached-snapshot stale-badge path (utc parse + >24h compare) is gone —
        the scan never scores from a cached snapshot."""
        if not vintage:
            return "<span style='color:#999;'>—</span>"
        return f"<span style='color:#1e8449;'>{esc(str(vintage))}</span>"

    def _best_scan_entry(sweep):
        """The single-scan candidate to hand off for a QPU: the entry with the
        highest displayed metric (lqs_circuit, else product_score)."""
        best, best_v = None, None
        for entries in sweep.get('strata', {}).values():
            for e in entries:
                v = e.get('lqs_circuit')
                if v is None:
                    v = e['product_score']
                if best is None or v > best_v:
                    best, best_v = e, v
        return best

    def _scan_best_lqs(r):
        """The single-scan display metric quoted EVERYWHERE for a backend:
        the MAX circuit LQS (lqs_circuit) across ALL its strata, ignoring None
        (annotation failed). Falls back to the product heuristic — flagged
        '(template)' — only when NOTHING was circuit-scored. Returns
        (value, is_template) so the comparison-table cell, the selector label
        and the detail line all quote the SAME number (previously the label
        quoted best_lqs = max product_score, diverging from the lqs_circuit
        table/detail)."""
        sweep = r.get('sweep') or {}
        lqs_vals = [e['lqs_circuit']
                    for entries in sweep.get('strata', {}).values()
                    for e in entries if e.get('lqs_circuit') is not None]
        if lqs_vals:
            return max(lqs_vals), False
        return (r.get('best_lqs') or 0.0), True

    def _scan_best_lqs_str(r):
        """`_scan_best_lqs` formatted as the display percent string."""
        val, is_tmpl = _scan_best_lqs(r)
        return f"{val * 100.0:.1f}%{' (template)' if is_tmpl else ''}"

    def _scan_selector(rows, *, label_fn, detail_fn, open_ok_fn, payload_fn,
                       disabled_tip):
        """One Dropdown over scan rows (already ranked) + one detail line + one
        '🔗 open on QPU 2' button. Selecting a row updates the detail; the open
        button is disabled (with a tooltip) for rows that cannot be opened
        (errors, a batch that does not fully fit, the QPU 1 reference row).
        `disabled_tip` is a string, or a callable(row) returning one."""
        opts = [(label_fn(r), i) for i, r in enumerate(rows)]
        if not opts:
            return []
        dd = Dropdown(options=opts, value=0, description='QPU:', layout=W,
                      style={'description_width': '40px'})
        det = HTML("")
        btn = Button(description='🔗 open on QPU 2', button_style='',
                     layout=Layout(width='180px'))
        status = HTML("")                       # handoff outcome, next to btn
        report = _handoff_reporter(status)

        def _sync(idx):
            r = rows[idx]
            det.value = detail_fn(r)
            ok = open_ok_fn(r)
            btn.disabled = not ok
            tip = disabled_tip(r) if callable(disabled_tip) else disabled_tip
            btn.tooltip = '' if ok else tip

        dd.observe(lambda ch: ch['new'] is not None and _sync(ch['new']),
                   names='value')

        def _open(_b):
            if dd.value is None:
                return
            r = rows[dd.value]
            if not open_ok_fn(r):
                return
            payload = payload_fn(r)
            if payload is not None:
                _stage_handoff(r['name'], payload, report)   # one OR a list

        btn.on_click(_open)
        _sync(0)
        return [dd, det, HBox([btn, status],
                              layout=Layout(align_items='center'))]

    def _scan_detail_line2(r):
        """Second detail line — the per-backend data the removed comparison
        table used to carry: stability badge · k-cap phrase · calibration
        vintage. Segments that don't apply are omitted; '' when none apply.
        All dynamic values are esc()'d inside the reused helpers / here."""
        segs = []
        badge = _stability_badge(r)                 # already-styled, esc'd HTML
        if badge:
            segs.append(badge)
        cap = _k_cap_phrase(_scan.get('norb', 0),
                            (r.get('sweep') or {}).get('k_max_effective'))
        if cap:
            segs.append(f"<span style='color:#e67e22;'>{esc(cap)}</span>")
        vintage = r.get('vintage')
        if vintage:
            segs.append(_vintage_cell(vintage))     # styled 'live · <time>'
        return " · ".join(segs)

    def _render_results(scan_result):
        results = scan_result['results']
        children = []

        # when a PAST date is active locally, the local backend was scored
        # on that date but remote backends are scored LIVE — say so, no silent
        # mixing.
        _anchor = _selected_date_anchor()
        if _anchor:
            children.append(HTML(
                f"<div style='font-size:12px;color:#b9770e;background:#fef9e7;"
                f"padding:3px 8px;border-radius:4px;border:1px solid #fdebd0;"
                f"margin:0 0 4px;'>⏱ local backend scored on "
                f"{esc(_anchor)}; remote backends on live calibrations</div>"))

        # ONE QPU selector — the ONLY results UI now (the comparison table is
        # gone; its per-backend columns moved into the detail's second line).
        # Pick a backend, see its best-per-k detail + stability/cap/calib, open
        # its best candidate on QPU 2. Rows keep the scan_qpus ranking; errored
        # / no-candidate rows are selectable but their open button is disabled.
        def _single_label(r):
            name = esc(str(r['name']))
            if r.get('error'):
                return f"{name} — ✗ error"
            if r.get('best_lqs') is None:
                return f"{name} — no candidates"
            return f"{name} — {_scan_best_lqs_str(r)}"   # max lqs_circuit, tmpl fbk

        def _single_detail(r):
            box = ("<div style='font-size:13px;background:#f8f9fa;padding:4px 6px;"
                   "border-radius:3px;margin:2px 0;'>")
            if r.get('error'):
                return box + f"✗ {esc(str(r['error']))}</div>"
            sweep = r.get('sweep') or {}
            strata = sweep.get('strata', {})
            if not strata or r.get('best_lqs') is None:
                return box + "no candidates</div>"
            best = _best_scan_entry(sweep)
            counts = (f"α={len(best['alpha'])} β={len(best['beta'])} "
                      f"Anc={len(best['ancillas'])}") if best else ""
            # line 1 (as before): per-k bests + counts.
            line1 = f"{esc(_format_best_per_k(strata))}<br>{counts}"
            # line 2 (rescued from the table): stability · cap · calib.
            line2 = _scan_detail_line2(r)
            line2_html = f"<br>{line2}" if line2 else ""
            return box + f"{line1}{line2_html}</div>"

        selector = _scan_selector(
            results,
            label_fn=_single_label,
            detail_fn=_single_detail,
            open_ok_fn=lambda r: not r.get('error') and r.get('best_lqs') is not None,
            payload_fn=lambda r: _best_scan_entry(r.get('sweep') or {}),
            disabled_tip='no candidates to open on QPU 2')
        children.extend(selector)

        scan_results_box.children = children

    def _notices_html(notices):
        """The LOUD "⚠ notices" widget HTML — an amber warning bar with a count
        badge (Y-pass styling), clearly clickable, expanding to the grouped,
        deduplicated messages (message → ×count, grouped by backend context).
        `notices` is a list of (context, message) tuples. Returns '' when empty
        so callers can hide the widget. Shared by scan, local and multi."""
        if not notices:
            return ""
        from collections import OrderedDict
        groups = OrderedDict()
        for bctx, msg in notices:
            groups.setdefault(bctx, OrderedDict())
            groups[bctx][msg] = groups[bctx].get(msg, 0) + 1
        parts = []
        for bctx, msgs in groups.items():
            parts.append(f"<b>{esc(str(bctx))}</b>")
            for msg, cnt in msgs.items():
                suffix = f" ×{cnt}" if cnt > 1 else ""
                parts.append(f"&nbsp;&nbsp;{esc(msg)}{esc(suffix)}")
        # LOUD collapsed header: an amber warning bar with a count badge,
        # clearly clickable. Expanded content keeps the small grey styling.
        return (
            f"<details style='margin:8px 0;'>"
            f"<summary style='cursor:pointer;list-style:none;font-size:14px;"
            f"font-weight:bold;color:#9c6500;background:#fff3cd;"
            f"border:1px solid #ffe08a;border-left:4px solid #f0ad4e;"
            f"border-radius:4px;padding:6px 10px;'>&#9888; notices "
            f"<span style='background:#f0ad4e;color:#ffffff;border-radius:10px;"
            f"padding:1px 8px;font-size:12px;'>{len(notices)}</span>"
            f"</summary>"
            f"<div style='padding:4px 8px;font-size:12px;color:#666;'>"
            + "<br>".join(parts) + "</div>"
            f"</details>")

    def _append_notices(notices):
        # Collapsible loud notices under the scan results. Hidden when empty.
        html = _notices_html(notices)
        if not html:
            return
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

        _clear_scan_results()                   # drop any stale results
        targets = list_scan_targets(
            state, norb, same_family_as=family,
            include_other_families=chk_scan_other_families.value)

        # Never re-search the active backend — mark it skipped and quote
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
            # Capture WARNING+ off 'iqmapper' for the scan's duration and stop
            # them leaking under the cell — the same context every sweep uses.
            with _capture_qpu_notices(
                    lambda: _scan.get('current') or 'general') as handler:
                # One scan path, one param source — forwards hybrid.
                result = _scan_backends(eligible, norb, get_backend,
                                        get_props, use_scan_filter)
                # Circuit-level LQS per backend — one scale everywhere.
                _annotate_scan_circuit_lqs(result, get_backend, get_props)
                return (result, list(handler.records))
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
                    f"<span style='font-size:14px;color:#e74c3c;'>❌ "
                    f"{len(ancilla_list)} ancilla values but "
                    f"{len(norb_list)} circuits — match them or leave "
                    f"ancilla blank.</span>")
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
                props=_local_scoring_props(),   # loaded snapshot, not live
                requested_bridges=requested_ancillas,
                requested_bridges_list=ancilla_list,
            )

            if not intra_qpu_mode.value and len(norb_list) > 1:
                intra_qpu_mode.value = True     # OFF→ON reveal: legitimate rebuild
            # Same treatment as the sweep path: targeted count refresh, no
            # visible Auto-view rebuild (results render in place below).
            set_num_circuits(len(norb_list))

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
                        f"α={len(entry['alpha'])} β={len(entry['beta'])}</div>")

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

            # No ctx.update_view(): the count refresh is targeted (above) and
            # the detail widgets / summary render into persistent widgets.

        except Exception as e:
            lbl_lucj_multi.value = (
                f"<div style='font-size:14px;color:#e74c3c;background:#fdedec;padding:6px 10px;"
                f"border-radius:5px;border:1px solid #f5b7b1;margin:4px 0;'>"
                f"❌ <b>Error:</b> {esc(str(e))}</div>")
        finally:
            btn_lucj_multi.disabled = False
            btn_lucj_multi.description = '🤖 Generate Multi-Circuit'

    btn_lucj_multi.on_click(_on_lucj_multi)

    # ── Multi-Circuit SWEEP (auto-best per circuit + override) ──────
    _multi_sweep = {'result': None, 'replan_in_flight': False, 'dropdowns': {}}

    def _parse_multi_norbs():
        raw = lucj_multi_norbs.value.strip()
        err = ("<span style='font-size:14px;color:#e74c3c;'>❌ {}</span>")
        if not raw:
            return None, err.format('Enter norb values (e.g. 8, 8)')
        try:
            norbs = [int(x.strip()) for x in raw.split(',') if x.strip()]
        except ValueError:
            return None, err.format('Invalid norb values')
        if not norbs:
            return None, err.format('Enter norb values (e.g. 8, 8)')
        return norbs, None

    def _multi_min_bridges(norbs):
        """The ancilla-per-circuit field, read as per-circuit min_bridges for
        the sweep. Blank → engine default (None). A value <= 0 → default for
        that circuit."""
        raw = lucj_multi_ancillas.value.strip()
        err = ("<span style='font-size:14px;color:#e74c3c;'>❌ {}</span>")
        if not raw:
            return None, None
        try:
            vals = [int(x.strip()) for x in raw.split(',') if x.strip()]
        except ValueError:
            return None, err.format('Invalid ancilla values (integers, e.g. 2, 2)')
        if len(vals) != len(norbs):
            return None, err.format(
                f'Ancilla list has {len(vals)} value(s) but {len(norbs)} '
                f'circuit(s). Match them, or leave blank.')
        return [v if v and v >= 1 else None for v in vals], None

    def _multi_stability_exclusions():
        """Hard pre-exclusion set for every circuit: the ALWAYS-ON quality
        floor ∪ (volatility exclusions only when the stability checkbox is on).
        The union is idempotent — a qubit both volatile and an outlier is
        excluded once."""
        excl = set(_floor_exclusions())                 # always-on floor
        if _stability_filter_active():
            history = getattr(state, 'cal_history', None)
            if history:
                from ..stability_filter import (
                    compute_qubit_volatility, select_excluded)
                vol = compute_qubit_volatility(
                    history, rb_stability_window.value,
                    today=_selected_date_anchor())
                excl |= set(select_excluded(
                    vol, threshold=ft_stability_threshold.value))
        return excl or None

    def _apply_multi_sweep(result):
        circuits = result.get('circuits', [])
        if not circuits:
            return
        if not intra_qpu_mode.value and len(circuits) > 1:
            intra_qpu_mode.value = True     # OFF→ON reveal: legitimate rebuild
        # Count change from our own apply → targeted refresh, no view rebuild
        # (the Multi results below render in place; no flicker).
        set_num_circuits(len(circuits))
        state.push_undo()
        for c in circuits:
            _apply_multi_candidate(c['circuit_idx'], c['chosen'])
        _refresh()
        _get_export()

    #: Keys of a multi-circuit result that belong to one placement plan.
    _PLAN_KEYS = ('circuits', 'total_product', 'total_lqs_circuit',
                  'placed_qubits', 'buffer_qubits_used', 'boundary_couplers',
                  'joint_total', 'failed_at')

    def _multi_plans(result):
        """Complete plans (every circuit placed), best first: the planner's
        `plans` when the result carries them, else the result's own plan
        when it is complete. Each item is a dict of `_PLAN_KEYS`."""
        if result.get('plans'):
            return [p for p in result['plans'] if p.get('circuits')
                    and p.get('failed_at') is None]
        if result.get('failed_at') is None and result.get('circuits'):
            return [{k: result.get(k) for k in _PLAN_KEYS}]
        return []

    def _plan_label(n, plan):
        """'Plan 1 · total 48.2% · C1 79.2% + C2 61.0%'."""
        total, tmpl, parts = 1.0, False, []
        for c in plan['circuits']:
            pct, is_tmpl = _entry_lqs_pct(c['chosen'])
            tmpl = tmpl or is_tmpl
            lqs = c['chosen'].get('lqs_circuit')
            total *= lqs if lqs is not None else c['chosen']['product_score']
            parts.append(f"C{c['circuit_idx'] + 1} {pct}")
        return (f"Plan {n} · total {total * 100.0:.1f}%"
                f"{' (template)' if tmpl else ''} · " + " + ".join(parts))

    def _multi_failed_banner(result):
        i = result.get('failed_at')
        if i is None:
            return ""
        return (
            f"<div style='font-size:13px;color:#8a4b00;background:#fff3cd;"
            f"padding:6px 10px;border-radius:4px;border:1px solid #ffe08a;"
            f"margin:4px 0;'>⚠️ Circuit doesn't fit after C1..C{i} — "
            f"reduce norb or ancillas; the ones that fit are placed"
            f"</div>")

    def _multi_override_disabled(disabled):
        for dd in _multi_sweep['dropdowns'].values():
            with contextlib.suppress(Exception):
                dd.disabled = disabled

    def _make_multi_override_cb(i, flat):
        def _cb(change):
            idx = change['new']
            if idx is None or idx >= len(flat):
                return
            _on_multi_override(i, flat[idx])
        return _cb

    def _multi_override_sort_key(e):
        """Order override alternatives by the DISPLAYED metric: circuit LQS
        descending, template (no lqs_circuit) entries sink below all scored
        ones, product_score breaks ties. So the dropdown reads best→worst and
        the preselected chosen sits at/near the top — not mid-list by stratum."""
        lqs = e.get('lqs_circuit')
        return (0 if lqs is not None else 1,
                -(lqs if lqs is not None else 0.0),
                -(e.get('product_score') or 0.0))

    def _on_plan_select(result, plans, j):
        """Apply plan `j` of `result` to every circuit at once."""
        if _multi_sweep['replan_in_flight']:
            return
        new_result = dict(result)
        new_result.update(plans[j])
        new_result['selected_plan'] = j
        _multi_sweep['result'] = new_result
        _apply_multi_sweep(new_result)
        _render_multi_sweep_output(new_result)

    def _plans_dropdown(result):
        """The Plans menu over the complete plans; [] when there are none."""
        plans = _multi_plans(result)
        if not plans:
            return []
        selected = min(result.get('selected_plan', 0), len(plans) - 1)
        dd = Dropdown(options=[(_plan_label(n, p), n - 1)
                               for n, p in enumerate(plans, start=1)],
                      value=selected, description='Plans:',
                      layout=Layout(width=W.width, margin=W.margin,
                                    min_width=W.min_width),
                      style={'description_width': 'initial'})
        dd.observe(lambda ch: ch['new'] is not None and ch['new'] != selected
                   and _on_plan_select(result, plans, ch['new']),
                   names='value')
        _multi_sweep['dropdowns']['plans'] = dd
        return [dd]

    def _build_multi_override_dropdowns(result):
        from ..fit_check import LaterFit
        _multi_sweep['dropdowns'] = {}
        widgets = _plans_dropdown(result)
        fit = LaterFit(result)
        for c in result['circuits']:
            i = c['circuit_idx']
            # Flatten across strata, then rank by the displayed metric (best
            # first) — labels keep "k=… · #… · LQS …%" so no info is lost.
            # An alternative after which a later circuit no longer fits is
            # left out (the chosen layout always stays).
            entries = sorted(
                (e for k in c['strata'] for e in c['strata'][k]),
                key=_multi_override_sort_key)
            flat, opts, hidden = [], [], 0
            for e in entries:
                if e is not c['chosen'] and not fit.later_circuits_fit(i, e):
                    hidden += 1
                    continue
                opts.append((_entry_label(e), len(flat)))
                flat.append(e)
            if not opts:
                continue
            chosen_idx = next((j for j, e in enumerate(flat)
                               if e is c['chosen']), 0)
            dd = Dropdown(options=opts, value=chosen_idx,
                          description=f'C{i + 1} alternatives:', layout=W,
                          style={'description_width': 'initial'})
            dd.observe(_make_multi_override_cb(i, flat), names='value')
            _multi_sweep['dropdowns'][i] = dd
            widgets.append(dd)
            if hidden:
                widgets.append(HTML(
                    f"<div style='font-size:12px;color:#555;margin:0 0 4px;'>"
                    f"{hidden} alternative{'s' if hidden != 1 else ''} "
                    f"hidden: the other circuits would not fit</div>"))
            if any(m.get('truncated') for m in c.get('per_stratum', {}).values()):
                widgets.append(HTML(
                    f"<div style='font-size:12px;color:#e67e22;margin:0 0 4px;'>"
                    f"⚠️ C{i + 1}: search truncated — ranking may be "
                    f"non-exhaustive.</div>"))
        lucj_multi_detail_box.children = widgets

    def _render_multi_sweep_output(result):
        n = len(result['circuits'])
        lbl_lucj_multi.value = (
            f"<div style='font-size:14px;color:#1e8449;background:#eafaf1;"
            f"padding:4px 8px;border-radius:4px;border:1px solid #abebc6;"
            f"margin:3px 0;'>✅ {n} circuit{'s' if n != 1 else ''} placed"
            f"{_joint_gain_text(result)}{_shared_couplers_html(result)}</div>"
            + _multi_failed_banner(result))
        _build_multi_override_dropdowns(result)

    def _on_multi_override(i, choice):
        if _multi_sweep['replan_in_flight']:
            return                                  # in-flight guard
        result = _multi_sweep['result']
        if result is None:
            return
        _multi_sweep['replan_in_flight'] = True
        _multi_override_disabled(True)

        def _work():
            with _capture_qpu_notices(lambda: f'replan C{i + 1}') as handler:
                new_result = _replan_multi(result, i, choice)
            return (new_result, list(handler.records))

        def _done(payload):
            new_result, records = payload
            _multi_sweep['result'] = new_result
            _apply_multi_sweep(new_result)
            _multi_sweep['replan_in_flight'] = False
            _render_multi_sweep_output(new_result)   # rebuilds dropdowns (enabled)
            lbl_lucj_multi_notices.value = _notices_html(records)

        def _err(exc):
            _multi_sweep['replan_in_flight'] = False
            _multi_override_disabled(False)
            lbl_lucj_multi.value = (
                f"<div style='font-size:14px;color:#e74c3c;'>❌ replan failed: "
                f"{esc(str(exc))}</div>")

        scan_spawn_bg(_work, on_done=_done, on_error=_err)

    def _on_lucj_multi_sweep(b):
        if state.active_backend is None:
            lbl_lucj_multi.value = (
                "<span style='font-size:14px;color:#e74c3c;'>❌ No backend "
                "connected.</span>")
            return
        if hasattr(state.active_backend, 'coords'):
            lbl_lucj_multi.value = (
                "<span style='font-size:14px;color:#e74c3c;'>❌ Auto-LUCJ "
                "requires a connected IBM backend.</span>")
            return
        norbs, err = _parse_multi_norbs()
        if err:
            lbl_lucj_multi.value = err
            return
        min_bridges_list, err = _multi_min_bridges(norbs)
        if err:
            lbl_lucj_multi.value = err
            return

        btn_lucj_multi_sweep.disabled = True
        btn_lucj_multi_sweep.description = '⏳ Sweeping...'
        lbl_lucj_multi.value = (_joint_pending_html(len(norbs))
                                if _selected_planner() == 'joint' else "")
        lbl_lucj_multi_notices.value = ""
        kwargs = _sweep_kwargs_from_ui()
        excl = _multi_stability_exclusions()
        scoring_props = _local_scoring_props()
        buffer = lucj_buffer_slider.value

        def _reset_btn():
            btn_lucj_multi_sweep.disabled = False
            btn_lucj_multi_sweep.description = '🧺 Generate Multi-Circuit'

        def _work():
            with _capture_qpu_notices(lambda: 'multi sweep') as handler:
                result = _run_multi(
                    norbs, state.active_backend, scoring_props,
                    sweep_kwargs=kwargs,
                    min_bridges=min_bridges_list, top_n=kwargs['top_n'],
                    buffer_qubits=buffer, exclude_qubits=excl,
                    mapping_cap=kwargs['mapping_cap'], hybrid=kwargs['hybrid'],
                    diversity_threshold=kwargs['diversity_threshold'],
                    circuit_score_fn=_circuit_score)
            return (result, list(handler.records))

        def _done(payload):
            result, records = payload
            _reset_btn()
            _multi_sweep['result'] = result
            _apply_multi_sweep(result)          # targeted count refresh (no rebuild)
            _render_multi_sweep_output(result)  # summary + dropdowns in place
            lbl_lucj_multi_notices.value = _notices_html(records)
            # No ctx.update_view() here: _apply_multi_sweep already did the
            # targeted refresh + _refresh()/_get_export(), and the Multi results
            # render into persistent widgets — a full rebuild would only flicker.

        def _err(exc):
            _reset_btn()
            lbl_lucj_multi.value = (
                f"<div style='font-size:14px;color:#e74c3c;background:#fdedec;"
                f"padding:6px 10px;border-radius:5px;border:1px solid #f5b7b1;"
                f"margin:4px 0;'>❌ <b>Error:</b> {esc(str(exc))}</div>")

        scan_spawn_bg(_work, on_done=_done, on_error=_err)

    btn_lucj_multi_sweep.on_click(_on_lucj_multi_sweep)

    # ── Batch scan: the whole multi-circuit batch on every QPU ─────
    def _batch_backend_exclusions(props, name):
        """Per-backend hard exclusions for a batch sweep: always-on floor on
        THIS backend's props ∪ volatility (when the scan filter is on and the
        backend has cached history)."""
        excl = set(_floor_exclusions(props))
        if _stability_filter_active() and chk_scan_stability.value and name:
            hist = _cached_history_for(name)
            if hist:
                from ..stability_filter import (
                    compute_qubit_volatility, select_excluded)
                vol = compute_qubit_volatility(hist, rb_stability_window.value)
                excl |= set(select_excluded(
                    vol, threshold=ft_stability_threshold.value))
        return excl or None

    def _batch_metric(res):
        """(value, is_template) — batch LQS = total_lqs_circuit, else product."""
        m = res.get('total_lqs_circuit')
        if m is not None:
            return m, False
        return (res.get('total_product') or 0.0), True

    def _batch_line_html(res):
        """Compact per-QPU line: 'C1: k=3 · 72.2% | batch: 72.2%'. A circuit
        that could not fit is labeled inline ('Ci: does not fit'); '(template)'
        marks a product-only metric."""
        parts = []
        for c in res.get('circuits', []):
            ch = c['chosen']
            pct, is_tmpl = _entry_lqs_pct(ch)
            tmpl = ' (template)' if is_tmpl else ''
            parts.append(f"C{c['circuit_idx'] + 1}: k={len(ch['ancillas'])} "
                         f"· {pct}{tmpl}")
        fa = res.get('failed_at')
        if fa is not None:
            # A partial batch has no batch total: it would be a product over
            # fewer circuits than a complete one.
            parts.append(f"C{fa + 1}: does not fit")
            return ' | '.join(parts)
        m, is_tmpl = _batch_metric(res)
        parts.append(f"batch: {m * 100.0:.1f}%{' (template)' if is_tmpl else ''}")
        return ' | '.join(parts)

    def _batch_complete(res, n_circuits):
        """Every circuit of the batch placed."""
        res = res or {}
        return (res.get('failed_at') is None
                and len(res.get('circuits') or []) == n_circuits
                and n_circuits > 0)

    def _rank_batch_rows(rows, n_circuits):
        """Complete batches first, by batch metric; then partial batches (more
        circuits placed first); then backends that were not scanned; errors
        last. A partial batch's metric is a product over fewer circuits, so
        it is never compared with a complete one."""
        def _key(r):
            res = r.get('sweep')
            if r.get('error'):
                return (3, 0, 0.0)
            if not res:
                return (2, 0, 0.0)
            metric, _ = _batch_metric(res)
            if _batch_complete(res, n_circuits):
                return (0, 0, -metric)
            return (1, -len(res.get('circuits') or []), -metric)
        return sorted(rows, key=_key)

    def _local_batch_reference(norbs):
        """The local multi-circuit result, if it was planned on the active
        backend for these norbs; else None."""
        local = _multi_sweep['result']
        if not local:
            return None
        ctx = local.get('_context') or {}
        if ctx.get('backend') is not state.active_backend:
            return None
        if list(ctx.get('norbs') or []) != list(norbs):
            return None
        if not _batch_complete(local, len(norbs)):
            return None
        return local

    def _with_reference_row(results, active_name, local, n_circuits):
        """Scan rows without the active backend, ranked by `_rank_batch_rows`;
        with a complete local result, a reference row for it at its rank."""
        rows = [r for r in results if r.get('name') != active_name]
        if local is not None and active_name:
            metric, _ = _batch_metric(local)
            rows.append({'name': active_name, 'family': None, 'sweep': local,
                         'best_lqs': metric, 'best_k': None, 'error': None,
                         'reference': True})
        return _rank_batch_rows(rows, n_circuits)

    def _render_batch_results(scan_result, active_name=None, norbs=()):
        # ONE QPU selector over the scan_qpus-ranked batches: pick a backend,
        # see its per-circuit detail, open the whole batch on QPU 2. A batch
        # that does not fully fit is selectable but not openable. The active
        # backend is QPU 1: it appears only as a reference row built from the
        # local multi-circuit result, never openable.
        n_circuits = len(norbs)
        local = _local_batch_reference(norbs)
        rows = _with_reference_row(scan_result['results'], active_name,
                                   local, n_circuits)

        def _label(r):
            name = esc(str(r['name']))
            if r.get('reference'):
                name += ' (this QPU)'
            if r.get('error'):
                return f"{name} — ✗ error"
            res = r.get('sweep')
            if not res:
                return f"{name} — skipped"
            if not _batch_complete(res, n_circuits):
                placed = len(res.get('circuits') or [])
                return f"{name} — {placed} of {n_circuits} circuits fit"
            m, is_tmpl = _batch_metric(res)
            return (f"{name} — batch {m * 100.0:.1f}%"
                    f"{' (template)' if is_tmpl else ''}")

        def _detail(r):
            box = ("<div style='font-size:13px;background:#f8f9fa;padding:4px 6px;"
                   "border-radius:3px;margin:2px 0;'>")
            if r.get('error'):
                return box + f"✗ {esc(str(r['error']))}</div>"
            return box + f"{esc(_batch_line_html(r.get('sweep') or {}))}</div>"

        def _open_ok(r):
            res = r.get('sweep') or {}
            return (not r.get('error') and not r.get('reference')
                    and bool(res.get('circuits'))
                    and res.get('failed_at') is None)

        def _payload(r):
            res = r.get('sweep') or {}
            return [c['chosen'] for c in res.get('circuits', [])]

        def _tip(r):
            if r.get('reference'):
                return 'already QPU 1'
            return 'batch does not fully fit on this backend'

        children = _scan_selector(
            rows, label_fn=_label, detail_fn=_detail, open_ok_fn=_open_ok,
            payload_fn=_payload, disabled_tip=_tip)
        if local is None and active_name:
            children.append(HTML(
                "<div style='font-size:13px;color:#555;margin:3px 0;'>"
                "this QPU not included — run Generate Multi-Circuit to "
                "compare</div>"))
        scan_batch_box.children = children

    def _on_scan_batch(b):
        if state.active_backend is None:
            lbl_lucj_multi.value = (
                "<span style='font-size:14px;color:#e74c3c;'>❌ No backend "
                "connected.</span>")
            return
        if state.active_service is None:
            lbl_lucj_multi.value = (
                "<span style='font-size:14px;color:#e74c3c;'>❌ No service "
                "connected — cannot scan other QPUs.</span>")
            return
        norbs, err = _parse_multi_norbs()
        if err:
            lbl_lucj_multi.value = err
            return
        min_bridges_list, err = _multi_min_bridges(norbs)
        if err:
            lbl_lucj_multi.value = err
            return

        btn_scan_batch.disabled = True
        btn_scan_batch.description = '⏳ Scanning batch...'
        scan_batch_box.children = []
        kwargs = _sweep_kwargs_from_ui()
        buffer = lucj_buffer_slider.value
        active_name = getattr(state, 'active_backend_name', None)

        def _batch_sweep_fn(norb, backend, props, *, min_bridges=None, top_n=3):
            excl = _batch_backend_exclusions(props, getattr(backend, 'name', None))
            # One circuit at a time on every QPU: the scan runs a batch per
            # backend, so it keeps the fast sequential plan.
            res = _run_multi(
                norbs, backend, props, sweep_kwargs=kwargs,
                planner='sequential', buffer_qubits=buffer,
                min_bridges=min_bridges_list, top_n=kwargs['top_n'],
                mapping_cap=kwargs['mapping_cap'], hybrid=kwargs['hybrid'],
                diversity_threshold=kwargs['diversity_threshold'],
                # THIS backend's calibration, as in the single-circuit scan.
                circuit_score_fn=_scan_circuit_score(props),
                exclude_qubits=excl)
            # Give scan_qpus a rankable 'strata' surface: the batch metric.
            metric, _ = _batch_metric(res)
            res['strata'] = {0: [{'product_score': metric}]}
            return res

        def _reset():
            btn_scan_batch.disabled = False
            btn_scan_batch.description = '🌐 Scan batch on all QPUs'

        def _work():
            from .. import qpu_scan
            from ..qpu_scan import _detect_family
            from ..props_shim import make_scan_get_props
            try:
                family = _detect_family(state.active_backend)
            except Exception:
                family = None
            targets = qpu_scan.list_scan_targets(
                state, norbs[0], same_family_as=family,
                include_other_families=chk_scan_other_families.value)
            # Never re-search the active backend: it is QPU 1. No sweep, no
            # fetch; the local multi-circuit result stands in for it.
            for t in targets:
                if t.get('name') == active_name:
                    t['eligible'] = False
                    t['reason'] = 'current backend — see local results above'
            get_backend = state.active_service.backend
            get_props = make_scan_get_props(state)
            return qpu_scan.scan_qpus(
                targets, norbs[0], get_backend, get_props,
                min_bridges=kwargs['min_bridges'], top_n=kwargs['top_n'],
                sweep_fn=_batch_sweep_fn)

        def _done(result):
            _reset()
            _render_batch_results(result, active_name, norbs)

        def _err(exc):
            _reset()
            lbl_lucj_multi.value = (
                f"<div style='font-size:14px;color:#e74c3c;'>❌ batch scan "
                f"failed: {esc(str(exc))}</div>")

        scan_spawn_bg(_work, on_done=_done, on_error=_err)

    btn_scan_batch.on_click(_on_scan_batch)

    # ── Re-optimization ─────────────────────────────────────
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
                props=_local_scoring_props(),   # loaded snapshot, not live
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
                            f"α={len(entry['alpha'])} β={len(entry['beta'])}</div>")
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
