"""
ui.lucj_panel.py — LUCJ layout generation callbacks.

Extracted from ui.py for maintainability. All functions receive
a `ctx` SimpleNamespace with shared UI state and widgets.
"""
import logging

from ipywidgets import Dropdown, HTML

from ..constants import hex_to_rgba
from ..html_safe import esc

_log = logging.getLogger(__name__)


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
    lucj_chk_score = ctx.lucj_chk_score
    btn_lucj_run = ctx.btn_lucj_run
    lbl_lucj_status = ctx.lbl_lucj_status
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
        sc = f"{entry['score']:.4f}" if entry['score'] is not None else "N/A"
        lbl_lucj_detail.value = (
            f"<div style='font-size:14px;background:#f8f9fa;padding:6px 8px;"
            f"border-radius:4px;border:1px solid #dee2e6;margin:3px 0;'>"
            f"<b>#{entry['rank']}</b> (err={sc})<br>"
            f"α: {', '.join(str(q) for q in entry['alpha'])}<br>"
            f"β: {', '.join(str(q) for q in entry['beta'])}</div>"
        )
    dd_lucj_layout.observe(_on_lucj_layout_select, names='value')

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
        try:
            from ..auto_lucj import get_all_zigzag_layouts

            requested_ancillas = lucj_ancilla_dd.value
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

            _lucj_all_layouts.clear()
            _lucj_all_layouts.extend(all_layouts)

            # Populate dropdown
            dd_opts = []
            for entry in all_layouts:
                sc = f"err={entry['score']:.4f}" if entry['score'] is not None else "?"
                dd_opts.append((
                    f"#{entry['rank']} ({sc}) α:[{entry['alpha'][0]}..{entry['alpha'][-1]}]",
                    entry['rank'] - 1,
                ))
            # Suppress observer firing while re-seeding: otherwise
            # setting value=0 fires _on_lucj_layout_select → apply, then
            # we apply again below — double push_undo + redundant refresh.
            # Same unobserve → set → observe dance as _reoptimize_downstream.
            dd_lucj_layout.unobserve(_on_lucj_layout_select, names='value')
            dd_lucj_layout.options = dd_opts
            dd_lucj_layout.value = 0
            dd_lucj_layout.disabled = False
            dd_lucj_layout.observe(_on_lucj_layout_select, names='value')

            best = all_layouts[0]
            _apply_lucj_layout(best)

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
                    f"ancillas={num_ab}). Best: err={sc}{ancilla_note}"
                    f"{_epoch_suffix()}</div>")

        except Exception as e:
            lbl_lucj_status.value = (
                f"<div style='font-size:14px;color:#e74c3c;background:#fdedec;padding:6px 10px;"
                f"border-radius:5px;border:1px solid #f5b7b1;margin:4px 0;'>"
                f"❌ <b>Error:</b> {esc(str(e))}</div>")
        finally:
            btn_lucj_run.disabled = False
            btn_lucj_run.description = '🤖 Generate LUCJ Layout'

    btn_lucj_run.on_click(_on_lucj_generate)

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
                props=_local_scoring_props(),   # loaded snapshot, not live
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
