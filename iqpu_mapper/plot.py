"""
plot.py — Plotly figure creation, rendering, and refresh.

Architecture: SINGLE qubit trace [3] containing ALL qubits.
point_inds[0] == global qubit index. No customdata, no two-trace split.

PNG compatibility (kaleido):
  - marker.line.width is scalar (kaleido drops list widths)
  - text array always synced with point count
  - gate hover markers REMOVED in design mode (prevents click interference)
"""
import logging

import numpy as np
import plotly.graph_objects as go
from .constants import (
    COLORSCALE_OPTIONS, US_QUBIT_PROPS, NS_QUBIT_PROPS,
    QUBIT_PROP_OPTIONS, GATE_PROP_OPTIONS,
    fmt_e, fmt_us, fmt_ns, fmt_delta,
    make_color_mapper, make_delta_color_mapper,
    blend_colors, DELTA_COLORSCALE,
)
from .html_safe import esc

_log = logging.getLogger(__name__)

# Module-level lookup dicts (avoid recreating per refresh)
_QUBIT_PROP_LABELS = dict(QUBIT_PROP_OPTIONS)
_GATE_PROP_LABELS = dict(GATE_PROP_OPTIONS)

# ── Edge-width tuning constants (pixel widths in the plot) ──────────
# Floor for error-mode SVG edges — kaleido renders very thin SVG lines
# as nearly invisible at typical export scales.
MIN_EDGE_WIDTH_PX = 2.0

# Delta mode: the divergent palette's near-zero gray band would
# disappear against white — bump the edges.
# Final width = max(base_w + BUMP, FLOOR).
DELTA_EDGE_WIDTH_BUMP_PX = 2
DELTA_EDGE_WIDTH_FLOOR_PX = 4

# Normal error mode: slight bump over the user's base so colour-coded
# errors read clearly without overpowering qubit markers.
ERROR_EDGE_WIDTH_BUMP_PX = 1


def create_figure():
    return go.FigureWidget()


# Stable trace identifiers. The qubit trace, colorbar placeholders and
# legend traces are looked up by .meta — never by fig.data[i] — so
# adding/removing/reordering traces (e.g. the 6 legend traces) cannot
# misroute a marker update to the wrong trace. We used .uid first, but
# Plotly 6.x FigureWidget regenerates .uid to a random UUID on every
# trace mutation, so a custom uid silently becomes None at lookup time.
# .meta is preserved across FigureWidget operations.
_META_QUBITS     = 'qubits'
_META_CB_QUBIT   = 'cb_qubit'
_META_CB_GATE    = 'cb_gate'
_META_LEGEND_FMT = 'legend_{role}'


def _trace_by_meta(fig, meta):
    """Return the first trace whose .meta matches; None if absent.
    (FigureWidget regenerates .uid, so .meta is the stable identifier.)"""
    for tr in fig.data:
        if getattr(tr, 'meta', None) == meta:
            return tr
    return None


def build_device_graph(fig, state, widgets_ref):
    """Trace structure: [0] default edges, [1] highlighted edges,
    [2] gate hover (error mode only), [3] ALL qubit markers (click
    target), [4] qubit colorbar, [5] gate colorbar.
    """
    x, y = state.x_coord, state.y_coord
    edges, q_ids = state.edges_list, state.qubit_ids
    n = len(q_ids)
    cs_plotly = _get_cs_plotly(state)

    edge_x, edge_y = [], []
    for e in edges:
        edge_x.extend([x[e[0]], x[e[1]], None])
        edge_y.extend([y[e[0]], y[e[1]], None])

    q_x = [float(v) for v in x]
    q_y = [float(v) for v in y]

    with fig.batch_update():
        fig.data = []
        fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', hoverinfo='skip', showlegend=False, line=dict(color='#dee2e6', width=3)))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', hoverinfo='skip', showlegend=False, line=dict(color='#000000', width=3)))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=1, opacity=0), hoverinfo='skip', showlegend=False))
        fig.add_trace(go.Scatter(
            x=q_x, y=q_y, mode='markers+text',
            text=[str(q) for q in q_ids], textposition="middle center",
            textfont=dict(color='black', size=widgets_ref.font_size_slider.value),
            marker=dict(size=widgets_ref.size_slider.value, color=['white']*n, line=dict(color=['#dee2e6']*n, width=1.5)),
            showlegend=False, meta=_META_QUBITS))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(colorscale=cs_plotly, showscale=False), hoverinfo='none', showlegend=False, meta=_META_CB_QUBIT))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(colorscale=cs_plotly, showscale=False), hoverinfo='none', showlegend=False, meta=_META_CB_GATE))
        # NOTE: no legend-only traces in the live figure — updating them in
        # the interactive refresh was interfering with the qubit outline
        # repaint. The legend is rendered ONLY by build_static_figure
        # (exports), driven by the same state.legend_config.

    fig.update_layout(
        paper_bgcolor='white', plot_bgcolor='white',
        width=widgets_ref.plot_w_slider.value, height=widgets_ref.plot_h_slider.value,
        showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
        margin=dict(l=20, r=20, t=20, b=110), hovermode="closest",
        font=dict(color="black", size=16),
        hoverlabel=dict(font=dict(size=20)),
        template="plotly_white")


# Legend role render order. Trace lookups (qubits, colorbars, legend
# entries) go through _trace_by_meta — never by index — so trace order
# can change without misrouting marker updates.
_LEGEND_ROLE_ORDER = ('alpha', 'beta', 'other', 'qubits', 'ancilla', 'unselected')


def _legend_default_config():
    """Fallback when state.legend_config is missing (mirrors
    state._default_legend_config so refresh_plot can run standalone)."""
    return {
        'enabled': True,
        'roles': {r: {'show': r != 'unselected', 'label': r}
                  for r in _LEGEND_ROLE_ORDER},
    }


def _role_style(d, role):
    """Dot (fill, line, width) for a role from the circuit's group_styles.
    'unselected' uses the default white / grey node style."""
    if role == 'unselected':
        return 'white', '#dee2e6', 1.5
    gs = d.get('group_styles', {}).get(role, {})
    return gs.get('fill', 'white'), gs.get('line', '#dee2e6'), gs.get('width', 2)


def _role_populated(d, role, n_qubits):
    """True iff the role has at least one qubit assigned in this circuit.
    'unselected' is populated when at least one of the chip's qubits is
    NOT in any of alpha/beta/other/qubits/ancilla."""
    if role == 'unselected':
        assigned = set(map(str, (d.get('alpha', []) or []) + (d.get('beta', []) or [])
                                 + (d.get('other', []) or []) + (d.get('qubits', []) or [])
                                 + (d.get('ancilla', []) or [])))
        return len(assigned) < n_qubits
    return len(d.get(role, []) or []) > 0


def _ensure_visible_outline(line_color, bg_color, width):
    """A line-only qubit (no fill) is visible only through its outline —
    if that outline has too little contrast with the background OR its
    width is 0, the qubit disappears. Return (colour, width) guaranteed
    visible; a sufficiently-contrasting outline is left untouched.
    """
    col = line_color
    try:
        def _lum(h):
            h = str(h).lstrip('#')
            return 0.299 * int(h[0:2], 16) + 0.587 * int(h[2:4], 16) + 0.114 * int(h[4:6], 16)
        if abs(_lum(line_color) - _lum(bg_color)) < 30:
            col = '#333333'
    except Exception:
        pass  # non-hex (named / rgba) — leave as-is rather than guess
    w = float(width or 0)
    return col, (w if w >= 1.0 else 1.5)


def refresh_plot(fig, state, widgets_ref):
    w = widgets_ref
    is_err = (w.view_mode.value == 'error')
    n_q, n_edges = len(state.qubit_ids), len(state.edges_list)
    n_circ = w.num_circuits.value if w.intra_qpu_mode.value else 1

    q_prop_key, g_prop_key = w.dd_qubit_prop.value, w.dd_gate_prop.value
    # Look up by the user-selected key only — never fall back to
    # state.readout_errors / state.gate_errors, which would silently
    # colour the map with readout/cz values even when the user picked
    # T1, gate_length, etc. (a silent misread on a user-visible plot).
    # Empty arrays are fine: make_color_mapper handles len(arr)==0.
    q_arr = np.array(state.qubit_props_data.get(q_prop_key, []), dtype=float)
    g_arr = np.array(state.gate_props_data.get(g_prop_key, []), dtype=float)
    cs_name = state.current_colorscale

    # ── Delta mode override ──────────────────────────────────────
    # Atomic: delta applies to BOTH qubits and gates or neither. A
    # half-applied delta (qubits = normal palette, edges = raw delta
    # values) gives two incompatible colour scales with no warning — only
    # the both-valid render has a coherent interpretation.
    _is_delta = is_err and state.delta_mode and state.delta_qubit_props
    if _is_delta:
        dq = state.delta_qubit_props.get(q_prop_key)
        dg = state.delta_gate_props.get(g_prop_key)
        dq_ok = dq is not None and len(dq) == n_q
        dg_ok = dg is not None and len(dg) == n_edges
        if dq_ok and dg_ok:
            q_arr = np.array(dq, dtype=float)
            g_arr = np.array(dg, dtype=float)
        else:
            _is_delta = False

    # Build the qubit-color mapper ONCE before the per-qubit loop — the
    # previous per-element call re-ran NaN-filter / log10 / min / max for
    # every i, an O(n²) loop that ate ~150 ms per refresh on 156-qubit
    # Heron during slider drag. Bounded indexing
    # (`q_arr[i] if i < len(q_arr) else np.nan`) handles partial calibration
    # — without it, the empty q_arr fix above would IndexError on the
    # first qubit. Mappers render NaN as '#000000' so the visual is
    # "missing data".
    if _is_delta:
        _q_color_for = make_delta_color_mapper(q_arr, q_prop_key)
        colors = [_q_color_for(q_arr[i] if i < len(q_arr) else np.nan) for i in range(n_q)]
    elif is_err:
        _q_color_for = make_color_mapper(q_arr, cs_name, q_prop_key)
        colors = [_q_color_for(q_arr[i] if i < len(q_arr) else np.nan) for i in range(n_q)]
    else:
        colors = [w.pp_bg_fill.value for _ in range(n_q)]
    # In delta mode, force dark borders so qubits are always visible
    l_colors = ['#333333'] * n_q if _is_delta else [w.pp_bg_line.value] * n_q
    l_widths = [2.0] * n_q if _is_delta else [1.5] * n_q
    _gtc = w.global_text_color
    _default_tc = _gtc.value if hasattr(_gtc, 'value') else 'black'
    text_colors = [_default_tc] * n_q

    q_label = _QUBIT_PROP_LABELS.get(q_prop_key, q_prop_key)
    g_label = _GATE_PROP_LABELS.get(g_prop_key, g_prop_key)
    if _is_delta:
        q_label = f"Δ {q_label}"
        g_label = f"Δ {g_label}"

    h_text = []
    for i in range(n_q):
        if is_err:
            v = q_arr[i] if i < len(q_arr) else np.nan
            if _is_delta:
                v_str = fmt_delta(v, q_prop_key)
            else:
                v_str = fmt_us(v) if q_prop_key in US_QUBIT_PROPS else (fmt_ns(v) if q_prop_key in NS_QUBIT_PROPS else (fmt_ns(v) if q_prop_key == 'gate_length' else fmt_e(v)))
            h_text.append(f"<b>Qubit {esc(state.qubit_ids[i])}</b><br>{esc(q_label)}: {v_str}")
        else:
            h_text.append(f"Q: {esc(state.qubit_ids[i])}")

    global_style, all_ancillas, all_others = {}, set(), set()
    # qubit_group_key[q_idx] = (circuit_idx, role) with role in alpha /
    # beta / other / qubits / ancilla. Drives the "same group → coloured
    # edge" check below. Comparing (circuit, role) instead of fills lets
    # Manual mode work too — Manual paints α and β with the same
    # `current_fill`, so the old `s_u['fill'] == s_v['fill']` test wrongly
    # fired on every α-β boundary edge.
    qubit_group_key = {}
    q_id_map = {q_id: i for i, q_id in enumerate(state.qubit_ids)}
    for idx in range(n_circ):
        d = state.layouts_data[idx]
        all_ancillas.update(d['ancilla'])
        all_others.update(d.get('other', []))
        for role in ('alpha', 'beta', 'other', 'qubits', 'ancilla'):
            for q_id in d.get(role, []) or []:
                qi = q_id_map.get(q_id)
                if qi is not None:
                    qubit_group_key[qi] = (idx, role)
        for q_id, s in d['styles'].items():
            q_idx = q_id_map.get(q_id)
            if q_idx is None: continue
            global_style[q_idx] = s
            text_colors[q_idx] = s.get('text_color', _default_tc)
            if is_err:
                if s['fill'] != 'none' and not w.chk_isolate.value:
                    colors[q_idx] = s['fill'] if s['fill'].startswith('rgba') else blend_colors(colors[q_idx], s['fill'])
                # line-only ('none') keeps the error colour; only line applied
                l_colors[q_idx] = s['line']
                # Outline width in error mode — lowered from 4 (defensive
                # against an old Plotly stroke-vanishing bug during
                # transitions) to match design-mode width for visual
                # consistency. Bump back to 3-4 if vanishing strokes return.
                l_widths[q_idx] = s.get('width', 2)
                h_text[q_idx] += f"<br>Sub-Circ: <b>{esc(d['name'])}</b>"
            else:
                if s['fill'] != 'none':
                    colors[q_idx] = s['fill']
                    l_colors[q_idx] = s['line']
                    l_widths[q_idx] = s.get('width', 2)
                else:
                    # Line-only: fill stays the background, so the qubit
                    # shows only via its outline — guarantee visibility.
                    l_colors[q_idx], l_widths[q_idx] = _ensure_visible_outline(
                        s['line'], w.pp_bg_fill.value, s.get('width', 2))

    base_w = w.gate_size_slider.value
    hl_w   = w.gate_hl_w_slider.value
    edge_colors, edge_widths = [], []
    # Same hoist as for qubits: build the gate-color mapper once.
    if is_err:
        _g_color_for = (make_delta_color_mapper(g_arr, g_prop_key) if _is_delta
                        else make_color_mapper(g_arr, cs_name, g_prop_key))
    else:
        _g_color_for = None
    for i, edge in enumerate(state.edges_list):
        u, v = edge
        if is_err:
            val = g_arr[i] if i < len(g_arr) else np.nan
            edge_colors.append(_g_color_for(val))
            # Thicker edges in delta mode so near-zero (gray) gates stay visible.
            edge_widths.append(
                max(base_w + DELTA_EDGE_WIDTH_BUMP_PX, DELTA_EDGE_WIDTH_FLOOR_PX)
                if _is_delta
                else base_w + ERROR_EDGE_WIDTH_BUMP_PX
            )
        else:
            match_col = None
            if w.chk_link_gates.value and u in global_style and v in global_style:
                # Skip gate colouring if BOTH endpoints are "other" qubits
                both_other = (state.qubit_ids[u] in all_others and state.qubit_ids[v] in all_others)
                if not both_other:
                    s_u, s_v = global_style[u], global_style[v]
                    # Highlight only same-group edges (α-α, β-β, …) or
                    # ancilla-adjacent edges; α/β boundary stays grey.
                    # Same-group check compares (circuit, role) tuples
                    # — NOT fill colour (see qubit_group_key comment above).
                    if state.qubit_ids[u] in all_ancillas:
                        match_col = s_u['line']
                    elif state.qubit_ids[v] in all_ancillas:
                        match_col = s_v['line']
                    else:
                        gk_u = qubit_group_key.get(u)
                        gk_v = qubit_group_key.get(v)
                        if gk_u is not None and gk_u == gk_v:
                            match_col = s_u['line']
            edge_colors.append(match_col if match_col else w.pp_gate_col.value)
            edge_widths.append(hl_w if match_col else base_w)

    # ── Threshold dimming (error mode only) ──────────────────────
    _DIM_COL = '#e8e8e8'
    if is_err and w.chk_err_threshold.value:
        # Parse qubit min/max
        try:
            q_lo = float(w.txt_q_thr_min.value) if w.txt_q_thr_min.value.strip() else None
        except (ValueError, TypeError):
            q_lo = None
        try:
            q_hi = float(w.txt_q_thr_max.value) if w.txt_q_thr_max.value.strip() else None
        except (ValueError, TypeError):
            q_hi = None
        # Parse gate min/max
        try:
            g_lo = float(w.txt_g_thr_min.value) if w.txt_g_thr_min.value.strip() else None
        except (ValueError, TypeError):
            g_lo = None
        try:
            g_hi = float(w.txt_g_thr_max.value) if w.txt_g_thr_max.value.strip() else None
        except (ValueError, TypeError):
            g_hi = None

        if q_lo is not None or q_hi is not None:
            for i in range(n_q):
                v = q_arr[i] if i < len(q_arr) else np.nan
                outside = np.isnan(v)
                if not outside and q_lo is not None and v < q_lo:
                    outside = True
                if not outside and q_hi is not None and v > q_hi:
                    outside = True
                if outside:
                    colors[i] = _DIM_COL
                    l_colors[i] = '#d0d0d0'
                    l_widths[i] = 1.0
        if g_lo is not None or g_hi is not None:
            for i in range(len(edge_colors)):
                v = g_arr[i] if i < len(g_arr) else np.nan
                outside = np.isnan(v)
                if not outside and g_lo is not None and v < g_lo:
                    outside = True
                if not outside and g_hi is not None and v > g_hi:
                    outside = True
                if outside:
                    edge_colors[i] = _DIM_COL
                    edge_widths[i] = max(1, base_w - 1)

    # ── Focus mode (error mode only): keep the heatmap on the placed
    # layout, grey the rest. Same dim treatment as the threshold block;
    # the criterion is selection membership (`i in global_style`).
    # Guarded by `global_style` so it no-ops when nothing is placed
    # (otherwise the whole chip would be greyed). Edges stay coloured only
    # when both endpoints are in the selection AND in the same circuit;
    # cross-circuit boundary gates belong to neither layout, so they dim.
    if is_err and w.chk_focus_selected.value and global_style:
        for i in range(n_q):
            if i not in global_style:
                colors[i] = _DIM_COL
                l_colors[i] = '#d0d0d0'
                l_widths[i] = 1.0
        for i, edge in enumerate(state.edges_list):
            u, v = edge
            gk_u = qubit_group_key.get(u)
            gk_v = qubit_group_key.get(v)
            if not (u in global_style and v in global_style
                    and gk_u is not None and gk_v is not None
                    and gk_u[0] == gk_v[0]):
                edge_colors[i] = _DIM_COL
                edge_widths[i] = max(1, base_w - 1)

    # ── Isolate layout (error mode): pure error-heatmap on the placed
    # layout — fill not blended (above), outline stripped — and the rest
    # greyed. One toggle = the paper-style focused view.
    if is_err and w.chk_isolate.value and global_style:
        for i in range(n_q):
            if i in global_style:
                l_widths[i] = 0          # no outline → pure heatmap dot
            else:
                colors[i] = _DIM_COL
                l_colors[i] = '#d0d0d0'
                l_widths[i] = 1.0
        for i, edge in enumerate(state.edges_list):
            u, v = edge
            gk_u = qubit_group_key.get(u)
            gk_v = qubit_group_key.get(v)
            if not (u in global_style and v in global_style
                    and gk_u is not None and gk_v is not None
                    and gk_u[0] == gk_v[0]):
                edge_colors[i] = _DIM_COL
                edge_widths[i] = max(1, base_w - 1)

    with fig.batch_update():
        if is_err:
            # Edges as SVG shapes only (per-edge colour + width). The
            # previous scatter "fallback in case shapes fail" peeked out
            # from under the shape edges as a doubled grey underlay —
            # shapes are reliable in practice, so no underlay.
            shapes = []
            for i, edge in enumerate(state.edges_list):
                u, v = edge
                x0, y0 = float(state.x_coord[u]), float(state.y_coord[u])
                x1, y1 = float(state.x_coord[v]), float(state.y_coord[v])
                ew = max(float(edge_widths[i]), MIN_EDGE_WIDTH_PX)
                shapes.append(dict(
                    type='line', x0=x0, y0=y0, x1=x1, y1=y1,
                    line=dict(color=edge_colors[i], width=ew),
                    xref='x', yref='y', layer='below'))
            fig.layout.shapes = tuple(shapes)
            # Empty both scatter line traces fully — width=0 plus
            # x=(None,) prevents Plotly from drawing stale cached geometry.
            fig.data[0].x = (None,); fig.data[0].y = (None,)
            fig.data[0].line.width = 0
            fig.data[1].x = (None,); fig.data[1].y = (None,)
            fig.data[1].line.width = 0
        else:
            fig.layout.shapes = ()
            default_x, default_y = [], []
            all_same = len(set(zip(edge_colors, edge_widths))) <= 1
            if all_same and edge_colors:
                for edge in state.edges_list:
                    u, v = edge
                    default_x.extend([state.x_coord[u], state.x_coord[v], None])
                    default_y.extend([state.y_coord[u], state.y_coord[v], None])
                fig.data[0].x = tuple(default_x); fig.data[0].y = tuple(default_y)
                fig.data[0].line.color = edge_colors[0]; fig.data[0].line.width = edge_widths[0]
                # Deactivating fig.data[1] also requires resetting its
                # line colour/width — otherwise Plotly's retained values +
                # cached x/y produce a faint second polyline (visible as a
                # lighter stripe over each gate). width=0 + matching colour
                # guarantees nothing draws even with stale geometry.
                fig.data[1].x = (None,)
                fig.data[1].y = (None,)
                fig.data[1].line.color = edge_colors[0]
                fig.data[1].line.width = 0
                # Drop shapes from a previous highlighted-edges render —
                # without this, ghost edges (notably a thin black line)
                # linger from earlier-session layouts.
                fig.layout.shapes = ()
            else:
                # Default-colour edges → fig.data[0] (scatter, single
                # style). ALL highlighted edges → fig.layout.shapes
                # uniformly. Previous code optimised the dominant
                # highlight into fig.data[1] (no SVG outline) and left
                # minority groups as shapes — minorities then showed a
                # faint contour the dominant group didn't (most visible
                # in multi-circuit intra-QPU layouts). Routing every
                # highlight through one path keeps it uniform.
                default_col = w.pp_gate_col.value
                default_wid = base_w

                highlight_shapes = []
                for i, edge in enumerate(state.edges_list):
                    u, v = edge
                    ec, ew = edge_colors[i], edge_widths[i]
                    if ec == default_col and ew == default_wid:
                        default_x.extend([state.x_coord[u], state.x_coord[v], None])
                        default_y.extend([state.y_coord[u], state.y_coord[v], None])
                    else:
                        highlight_shapes.append(dict(
                            type='line',
                            x0=float(state.x_coord[u]),
                            y0=float(state.y_coord[u]),
                            x1=float(state.x_coord[v]),
                            y1=float(state.y_coord[v]),
                            line=dict(color=ec, width=float(ew)),
                            xref='x', yref='y', layer='below',
                        ))

                fig.data[0].x = tuple(default_x) if default_x else (None,)
                fig.data[0].y = tuple(default_y) if default_y else (None,)
                fig.data[0].line.color = default_col
                fig.data[0].line.width = default_wid

                # data[1] unused in this branch — kill stale geometry +
                # style so Plotly doesn't redraw a ghost trace.
                fig.data[1].x = (None,)
                fig.data[1].y = (None,)
                fig.data[1].line.width = 0

                fig.layout.shapes = highlight_shapes

        # Gate hover [2] — only in error mode, REMOVED in design mode
        if is_err:
            gate_h = []
            for i in range(n_edges):
                gv = g_arr[i] if i < len(g_arr) else np.nan
                gv_str = fmt_delta(gv, g_prop_key) if _is_delta else (fmt_ns(gv) if g_prop_key == 'gate_length' else fmt_e(gv))
                # state.qubit_ids may come from untrusted JSON — escape.
                edge = state.edges_list[i]
                try:
                    u_id = state.qubit_ids[edge[0]]
                    v_id = state.qubit_ids[edge[1]]
                    edge_label = f"({esc(u_id)}, {esc(v_id)})"
                except (IndexError, TypeError):
                    edge_label = esc(edge)
                gate_h.append(f"<b>Gate {edge_label}</b><br>{esc(g_label)}: {gv_str}")
            fig.data[2].x = tuple((state.x_coord[e[0]] + state.x_coord[e[1]]) / 2 for e in state.edges_list)
            fig.data[2].y = tuple((state.y_coord[e[0]] + state.y_coord[e[1]]) / 2 for e in state.edges_list)
            fig.data[2].hovertext = tuple(gate_h); fig.data[2].hoverinfo = 'text'
            fig.data[2].marker.size = 20; fig.data[2].marker.opacity = 0
        else:
            fig.data[2].x = (None,); fig.data[2].y = (None,); fig.data[2].hoverinfo = 'skip'

        # Qubit trace — looked up by stable .meta, NEVER by index. The
        # legend traces (added later) would otherwise shift order on a
        # rebuild and a stale fig.data[3] would route the marker update
        # to the wrong trace, leaving the placeholder grey outline in
        # place on the first refresh. (We used .uid first, but
        # FigureWidget regenerates .uid; .meta is preserved.)
        q_trace = _trace_by_meta(fig, _META_QUBITS)
        if q_trace is not None:
            q_trace.mode = 'markers+text' if w.show_numbers.value else 'markers'
            q_trace.hoverinfo = 'text' if is_err else 'none'
            q_trace.text = tuple(str(state.qubit_ids[i]) for i in range(n_q)) if w.show_numbers.value else tuple('' for _ in range(n_q))
            q_trace.hovertext = tuple(h_text)
            q_trace.marker.update(color=list(colors), size=int(w.size_slider.value), line=dict(color=list(l_colors), width=list(l_widths)))
            q_trace.update(textfont=dict(color=list(text_colors), size=int(w.font_size_slider.value)))

        fig.layout.paper_bgcolor = 'white'; fig.layout.plot_bgcolor = 'white'; fig.layout.font.color = 'black'
        _update_colorbars(fig, state, q_arr, g_arr, q_prop_key, g_prop_key, q_label, g_label, is_err, _is_delta)
        # NOTE: no live-legend update — the legend is rendered only by
        # build_static_figure (export). Updating legend traces in the
        # interactive refresh was interfering with the qubit outline
        # repaint, so the live figure has no legend traces at all.


def _update_colorbars(fig, state, q_arr, g_arr, q_prop_key, g_prop_key, q_label, g_label, is_err, is_delta=False):
    cs_pl = _get_cs_plotly(state)
    delta_cs = DELTA_COLORSCALE['plotly']
    cb_qubit = _trace_by_meta(fig, _META_CB_QUBIT)
    cb_gate = _trace_by_meta(fig, _META_CB_GATE)
    if cb_qubit is None or cb_gate is None:
        return
    try:
        if is_err and len(q_arr) > 0 and len(g_arr) > 0:
            valid_q = q_arr[~np.isnan(q_arr)]

            # ── DELTA MODE: divergent colorbar ───────────────────
            if is_delta and len(valid_q) > 0:
                max_abs_q = max(np.max(np.abs(valid_q)), 1e-15)
                cb_qubit.marker.update(
                    cmin=-max_abs_q, cmax=max_abs_q, showscale=True,
                    colorscale=delta_cs, reversescale=False,
                    colorbar=dict(
                        orientation='h', x=0.25, y=-0.08, len=0.42, thickness=18, tickfont=dict(size=13),
                        tickvals=[-max_abs_q, 0, max_abs_q],
                        ticktext=["◀ better", "0", "worse ▶"],
                        title=dict(text=f"<b>{q_label}</b> ({state.delta_date_a} → {state.delta_date_b})<br>▼", side="top")))

                valid_g = g_arr[~np.isnan(g_arr)]
                if len(valid_g) > 0:
                    max_abs_g = max(np.max(np.abs(valid_g)), 1e-15)
                    cb_gate.marker.update(
                        cmin=-max_abs_g, cmax=max_abs_g, showscale=True,
                        colorscale=delta_cs, reversescale=False,
                        colorbar=dict(
                            orientation='h', x=0.75, y=-0.08, len=0.42, thickness=18, tickfont=dict(size=13),
                            tickvals=[-max_abs_g, 0, max_abs_g],
                            ticktext=["◀ better", "0", "worse ▶"],
                            title=dict(text=f"<b>{g_label}</b><br>▼", side="top")))
                else:
                    cb_gate.marker.showscale = False
                return

            # ── NORMAL ERROR MODE ────────────────────────────────
            if len(valid_q) > 0:
                if q_prop_key in US_QUBIT_PROPS or q_prop_key in NS_QUBIT_PROPS:
                    q_min, q_med, q_max = np.min(valid_q), np.median(valid_q), np.max(valid_q)
                    _fmt = fmt_us if q_prop_key in US_QUBIT_PROPS else fmt_ns
                    cb_qubit.marker.update(cmin=q_min, cmax=q_max, showscale=True, colorscale=cs_pl, reversescale=False,
                        colorbar=dict(orientation='h', x=0.25, y=-0.08, len=0.42, thickness=18, tickfont=dict(size=13), tickvals=[q_min, q_max],
                            ticktext=[f"min {_fmt(q_min)}", f"max {_fmt(q_max)}"],
                            title=dict(text=f"<b>{q_label}</b>&nbsp;&nbsp;Median {_fmt(q_med)}<br>▼", side="top")))
                else:
                    vp = valid_q[valid_q > 0]
                    if len(vp) > 0:
                        lr = np.log10(vp + 1e-12); r_min, r_med, r_max = np.min(lr), np.median(lr), np.max(lr)
                        cb_qubit.marker.update(cmin=r_min, cmax=r_max, showscale=True, colorscale=cs_pl, reversescale=False,
                            colorbar=dict(orientation='h', x=0.25, y=-0.08, len=0.42, thickness=18, tickfont=dict(size=13), tickvals=[r_min, r_max],
                                ticktext=[f"min {fmt_e(10**r_min)}", f"max {fmt_e(10**r_max)}"],
                                title=dict(text=f"<b>{q_label}</b>&nbsp;&nbsp;Median {fmt_e(10**r_med)}<br>▼", side="top")))
                    else:
                        cb_qubit.marker.showscale = False
            else:
                cb_qubit.marker.showscale = False
            valid_g = g_arr[~np.isnan(g_arr)]
            if len(valid_g) > 0:
                if g_prop_key == 'gate_length':
                    g_min, g_med, g_max = np.min(valid_g), np.median(valid_g), np.max(valid_g)
                    cb_gate.marker.update(cmin=g_min, cmax=g_max, showscale=True, colorscale=cs_pl, reversescale=False,
                        colorbar=dict(orientation='h', x=0.75, y=-0.08, len=0.42, thickness=18, tickfont=dict(size=13), tickvals=[g_min, g_max],
                            ticktext=[f"min {fmt_ns(g_min)}", f"max {fmt_ns(g_max)}"],
                            title=dict(text=f"<b>{g_label}</b>&nbsp;&nbsp;Median {fmt_ns(g_med)}<br>▼", side="top")))
                else:
                    vp = valid_g[(valid_g > 0) & (valid_g < 1.0)]
                    if len(vp) > 0:
                        lg = np.log10(vp + 1e-12); g_min, g_med, g_max = np.min(lg), np.median(lg), np.max(lg)
                        cb_gate.marker.update(cmin=g_min, cmax=g_max, showscale=True, colorscale=cs_pl, reversescale=False,
                            colorbar=dict(orientation='h', x=0.75, y=-0.08, len=0.42, thickness=18, tickfont=dict(size=13), tickvals=[g_min, g_max],
                                ticktext=[f"min {fmt_e(10**g_min)}", f"max {fmt_e(10**g_max)}"],
                                title=dict(text=f"<b>{g_label}</b>&nbsp;&nbsp;Median {fmt_e(10**g_med)}<br>▼", side="top")))
                    else:
                        cb_gate.marker.showscale = False
            else:
                cb_gate.marker.showscale = False
        else:
            cb_qubit.marker.showscale = False; cb_gate.marker.showscale = False
    except Exception:
        # Bumped from .warning to .exception: the user sees the colorbar
        # silently vanish when this fails, so without a traceback they
        # can't tell missing data from a fmt_us/fmt_ns crash. UI still
        # degrades gracefully via the showscale toggles below.
        _log.exception("Colorbar update failed; hiding both scales")
        cb_qubit.marker.showscale = False; cb_gate.marker.showscale = False


def _update_legend(fig, state, n_q, n_circ):
    """Restyle + toggle the six legend-only traces from state.legend_config
    and the displayed circuits' group_styles. Traces are looked up by .meta
    (never by index) so trace reorders cannot misroute updates.

    A role's entry is shown iff: master enabled is True, that role's `show`
    is True, AND the role is populated anywhere across the visible
    circuits (multi-circuit-aware union — see _role_populated)."""
    cfg = getattr(state, 'legend_config', None) or _legend_default_config()
    enabled = bool(cfg.get('enabled', True))
    roles_cfg = cfg.get('roles', {})

    # Drive the LIVE figure's layout.showlegend off the master toggle so
    # turning it off hides the legend frame entirely (not just the
    # entries). Mirrors build_static_figure, which sets showlegend on the
    # export based on the same config.
    fig.layout.showlegend = enabled

    # Multi-circuit union: a role counts as populated if ANY circuit has it.
    all_assigned = set()
    populated = set()
    for ci in range(n_circ):
        d_ci = state.layouts_data[ci]
        for role in ('alpha', 'beta', 'other', 'qubits', 'ancilla'):
            items = list(map(str, d_ci.get(role, []) or []))
            if items:
                populated.add(role)
                all_assigned.update(items)
    if len(all_assigned) < n_q:
        populated.add('unselected')

    # Styling: take colours from circuit 0's group_styles (canonical role
    # palette — controls already write through SPEC_O/P so this stays in sync).
    d0 = state.layouts_data[0]
    for role in _LEGEND_ROLE_ORDER:
        trace = _trace_by_meta(fig, _META_LEGEND_FMT.format(role=role))
        if trace is None:
            continue   # legend traces not built yet (partial figure / tests)
        fill, line, lw = _role_style(d0, role)
        role_cfg = roles_cfg.get(role, {})
        trace.marker.color = fill
        trace.marker.line.color = line
        trace.marker.line.width = float(lw)
        trace.name = role_cfg.get('label', role)
        trace.showlegend = (
            enabled
            and bool(role_cfg.get('show', False))
            and role in populated
        )


def _get_cs_plotly(state):
    return COLORSCALE_OPTIONS[state.current_colorscale]['plotly']


def apply_zoom(fig, state):
    r = state.get_zoom_range()
    if r is None: return
    x0, x1, y0, y1 = r; state.set_zoom(x0, x1, y0, y1)
    with fig.batch_update():
        fig.layout.xaxis.update(range=[x0, x1], autorange=False)
        fig.layout.yaxis.update(range=[y0, y1], autorange=False, scaleanchor='x')


def do_zoom(fig, state, factor):
    r = state.get_zoom_range()
    if r is None: return
    x0, x1, y0, y1 = r
    xc, xh = (x0+x1)/2, (x1-x0)/2*factor; yc, yh = (y0+y1)/2, (y1-y0)/2*factor
    state.set_zoom(xc-xh, xc+xh, yc-yh, yc+yh); apply_zoom(fig, state)


def reset_zoom(fig, state):
    state.reset_zoom()
    with fig.batch_update():
        fig.layout.xaxis.update(autorange=True); fig.layout.yaxis.update(autorange=True, scaleanchor='x')


# ── Static figure builder (PNG / GIF / any export) ──────────────────

_DIM_COL = '#e8e8e8'
_DIM_LINE = '#d0d0d0'


def _parse_threshold(val_str):
    """Parse a threshold text value. Returns float or None."""
    if not val_str or not val_str.strip():
        return None
    try:
        return float(val_str)
    except (ValueError, TypeError):
        return None


def _is_outside_threshold(val, lo, hi):
    """Check if a value falls outside [lo, hi]. NaN is always outside."""
    if np.isnan(val):
        return True
    if lo is not None and val < lo:
        return True
    if hi is not None and val > hi:
        return True
    return False


def collect_render_opts(widgets_ref):
    """Extract rendering options from widgets into a plain dict — the
    bridge between ipywidgets and the pure-data renderer. Call from
    ui.py, pass to build_static_figure().
    """
    w = widgets_ref
    return {
        'is_error': w.view_mode.value == 'error',
        'qubit_size': int(w.size_slider.value),
        'gate_width': int(w.gate_size_slider.value),
        'gate_hl_width': int(w.gate_hl_w_slider.value),
        'show_numbers': bool(w.show_numbers.value),
        'font_size': int(w.font_size_slider.value),
        'link_gates': bool(w.chk_link_gates.value),
        'bg_fill': str(w.pp_bg_fill.value),
        'bg_line': str(w.pp_bg_line.value),
        'gate_color': str(w.pp_gate_col.value),
        'q_prop_key': w.dd_qubit_prop.value,
        'g_prop_key': w.dd_gate_prop.value,
        'n_circuits': w.num_circuits.value if w.intra_qpu_mode.value else 1,
        'fig_width': int(w.plot_w_slider.value),
        'fig_height': int(w.plot_h_slider.value),
        'threshold_enabled': bool(w.chk_err_threshold.value),
        'focus_selected': bool(w.chk_focus_selected.value),
        'isolate': bool(w.chk_isolate.value),
        'q_thr_min': _parse_threshold(w.txt_q_thr_min.value),
        'q_thr_max': _parse_threshold(w.txt_q_thr_max.value),
        'g_thr_min': _parse_threshold(w.txt_g_thr_min.value),
        'g_thr_max': _parse_threshold(w.txt_g_thr_max.value),
        'text_color': w.global_text_color.value if w.global_text_color is not None else '#000000',
    }


def build_static_figure(state, opts, margin=40, crop_to_selection=False, annotation=None):
    """Single source of truth for static rendering (PNG, GIF frame).
    Handles error / design / delta modes, threshold dimming, circuit
    styles, link gates, line-only mode.

    Args:
        state: SessionState with layout and calibration data.
        opts: dict from collect_render_opts() — pure values, no widgets.
        margin: pixel margin around the figure.
        crop_to_selection: if True, zoom to only styled qubits.
        annotation: optional text overlay (e.g. date label for a GIF frame).

    Returns:
        go.Figure ready for write_image().
    """
    from collections import defaultdict

    n_q = len(state.qubit_ids)
    n_circuits = opts['n_circuits']
    is_err = opts['is_error']
    bg_fill = opts['bg_fill']
    bg_line = opts['bg_line']
    base_w = opts['gate_width']
    hl_w = opts['gate_hl_width']
    q_prop_key = opts['q_prop_key']
    g_prop_key = opts['g_prop_key']

    # ── 1. Collect qubit styles from all active circuits ──────────
    styled = {}
    global_style = {}
    all_ancillas = set()
    all_others = set()
    # See refresh_plot: compare (circuit, role) tuples — not fill colours
    # — so Manual-mode α-β boundary edges don't get coloured.
    qubit_group_key = {}
    q_id_map = {q_id: i for i, q_id in enumerate(state.qubit_ids)}
    for cidx in range(n_circuits):
        d = state.layouts_data[cidx]
        all_ancillas.update(d['ancilla'])
        all_others.update(d.get('other', []))
        for role in ('alpha', 'beta', 'other', 'qubits', 'ancilla'):
            for q_id in d.get(role, []) or []:
                qi = q_id_map.get(q_id)
                if qi is not None:
                    qubit_group_key[qi] = (cidx, role)
        for q_id, s in d['styles'].items():
            idx = q_id_map.get(q_id)
            if idx is not None:
                styled[idx] = s
                global_style[idx] = s

    # ── 2. Build per-qubit colors ─────────────────────────────────
    q_colors = []
    q_line_colors = []
    q_widths = []
    q_text_colors = []

    # Prepare error/delta arrays
    q_arr = g_arr = None
    _is_delta = False

    if is_err:
        # Same fix as refresh_plot — look up the user's selected key
        # only, never silently fall back to readout/cz error. This path
        # feeds PNG + time-lapse GIF too, so wrong colours would land
        # in saved files.
        q_arr = np.array(state.qubit_props_data.get(q_prop_key, []), dtype=float)
        g_arr = np.array(state.gate_props_data.get(g_prop_key, []), dtype=float)

        # Delta mode override
        _is_delta = state.delta_mode and bool(state.delta_qubit_props)
        if _is_delta:
            dq = state.delta_qubit_props.get(q_prop_key)
            if dq is not None and len(dq) == n_q:
                q_arr = np.array(dq, dtype=float)
            else:
                _is_delta = False
            dg = state.delta_gate_props.get(g_prop_key)
            if dg is not None and len(dg) == len(state.edges_list):
                g_arr = np.array(dg, dtype=float)

    _default_tc = opts.get('text_color', '#000000')

    # Build the qubit-color mapper ONCE before the per-qubit loop —
    # avoids n² behaviour over up to 156 qubits per PNG/GIF frame.
    if is_err:
        _q_color_for = (make_delta_color_mapper(q_arr, q_prop_key) if _is_delta
                        else make_color_mapper(q_arr, state.current_colorscale, q_prop_key))
    else:
        _q_color_for = None

    for i in range(n_q):
        tc = str(styled[i].get('text_color', _default_tc)) if i in styled else _default_tc

        if is_err:
            # Bounded indexing (same reason as refresh_plot): empty
            # q_arr would crash a raw q_arr[i]; mapper renders NaN as
            # '#000000'.
            qv = q_arr[i] if i < len(q_arr) else np.nan
            base_col = str(_q_color_for(qv))

            if i in styled:
                s = styled[i]
                if s['fill'] != 'none' and not opts.get('isolate'):
                    base_col = str(s['fill']) if s['fill'].startswith('rgba') else str(blend_colors(base_col, s['fill']))
                q_colors.append(base_col)
                q_line_colors.append(str(s['line']))
                q_widths.append(float(s.get('width', 4)))
            else:
                q_colors.append(base_col)
                # Dark border in delta mode for visibility
                q_line_colors.append('#333333' if _is_delta else str(bg_line))
                q_widths.append(2.0 if _is_delta else 1.5)
        else:
            if i in styled:
                s = styled[i]
                if s['fill'] == 'none':
                    _lc, _lw = _ensure_visible_outline(s['line'], bg_fill, s.get('width', 2))
                    q_colors.append(str(bg_fill))
                    q_line_colors.append(str(_lc))
                    q_widths.append(float(_lw))
                else:
                    q_colors.append(str(s['fill']))
                    q_line_colors.append(str(s['line']))
                    q_widths.append(float(s.get('width', 2)))
            else:
                q_colors.append(str(bg_fill))
                q_line_colors.append(str(bg_line))
                q_widths.append(1.5)

        q_text_colors.append(tc)

    # ── 3. Threshold dimming (qubits) ─────────────────────────────
    q_lo = opts.get('q_thr_min')
    q_hi = opts.get('q_thr_max')
    g_lo = opts.get('g_thr_min')
    g_hi = opts.get('g_thr_max')

    if is_err and opts.get('threshold_enabled') and (q_lo is not None or q_hi is not None):
        for i in range(n_q):
            v = q_arr[i] if i < len(q_arr) else np.nan
            if _is_outside_threshold(v, q_lo, q_hi):
                q_colors[i] = _DIM_COL
                q_line_colors[i] = _DIM_LINE
                q_widths[i] = 1.0

    # Focus (qubits): grey everything not in the selection. Mirrors
    # refresh_plot so PNG/GIF export inherits the same masked heatmap.
    if is_err and opts.get('focus_selected') and global_style:
        for i in range(n_q):
            if i not in global_style:
                q_colors[i] = _DIM_COL
                q_line_colors[i] = _DIM_LINE
                q_widths[i] = 1.0

    # Isolate (mirrors refresh_plot): pure heatmap on placed qubits
    # (no outline), everything else greyed.
    if is_err and opts.get('isolate') and global_style:
        for i in range(n_q):
            if i in global_style:
                q_widths[i] = 0
            else:
                q_colors[i] = _DIM_COL
                q_line_colors[i] = _DIM_LINE
                q_widths[i] = 1.0

    # ── 4. Build edge styles ──────────────────────────────────────
    edge_x, edge_y = [], []
    edge_shapes = []

    # Hoist gate-color mapper outside the per-edge loop (same rationale
    # as the qubit loop above).
    if is_err:
        _g_color_for = (make_delta_color_mapper(g_arr, g_prop_key) if _is_delta
                        else make_color_mapper(g_arr, state.current_colorscale, g_prop_key))
    else:
        _g_color_for = None

    for i, edge in enumerate(state.edges_list):
        u, v = edge
        x0, y0 = float(state.x_coord[u]), float(state.y_coord[u])
        x1, y1 = float(state.x_coord[v]), float(state.y_coord[v])

        if is_err:
            val = g_arr[i] if i < len(g_arr) else np.nan
            ec = _g_color_for(val)
            ew = (
                float(max(base_w + DELTA_EDGE_WIDTH_BUMP_PX, DELTA_EDGE_WIDTH_FLOOR_PX))
                if _is_delta
                else float(base_w + ERROR_EDGE_WIDTH_BUMP_PX)
            )

            # Gate threshold dimming
            if opts.get('threshold_enabled') and (g_lo is not None or g_hi is not None):
                if _is_outside_threshold(val, g_lo, g_hi):
                    ec = _DIM_COL
                    ew = max(1.0, float(base_w - 1))

            # Focus mode: grey edges not fully inside the SAME circuit's
            # selection — a boundary gate between two adjacent circuits has
            # styled endpoints in DIFFERENT circuits and belongs to neither.
            gk_u = qubit_group_key.get(u)
            gk_v = qubit_group_key.get(v)
            if (opts.get('focus_selected') or opts.get('isolate')) and global_style and not (
                    u in global_style and v in global_style
                    and gk_u is not None and gk_v is not None and gk_u[0] == gk_v[0]):
                ec = _DIM_COL
                ew = max(1.0, float(base_w - 1))

            edge_shapes.append(dict(type='line', x0=x0, y0=y0, x1=x1, y1=y1,
                line=dict(color=ec, width=ew), xref='x', yref='y', layer='below'))
        else:
            match_col = None
            if opts['link_gates'] and u in global_style and v in global_style:
                both_other = (state.qubit_ids[u] in all_others and state.qubit_ids[v] in all_others)
                if not both_other:
                    s_u, s_v = global_style[u], global_style[v]
                    if state.qubit_ids[u] in all_ancillas:
                        match_col = s_u['line']
                    elif state.qubit_ids[v] in all_ancillas:
                        match_col = s_v['line']
                    else:
                        gk_u = qubit_group_key.get(u)
                        gk_v = qubit_group_key.get(v)
                        if gk_u is not None and gk_u == gk_v:
                            match_col = s_u['line']

            if match_col:
                edge_shapes.append(dict(type='line', x0=x0, y0=y0, x1=x1, y1=y1,
                    line=dict(color=match_col, width=float(hl_w)),
                    xref='x', yref='y', layer='below'))
            else:
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

    # ── 5. Assemble go.Figure ─────────────────────────────────────
    export_fig = go.Figure()

    # Default edges trace
    if edge_x:
        export_fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y, mode='lines', hoverinfo='skip',
            showlegend=False, line=dict(color=str(opts['gate_color']), width=float(base_w)),
        ))

    # Qubit markers — grouped by line width (kaleido drops list widths,
    # so we emit one scalar-width trace per distinct width).
    width_groups = defaultdict(list)
    for i in range(n_q):
        width_groups[q_widths[i]].append(i)

    mode_str = 'markers+text' if opts['show_numbers'] else 'markers'
    for w_val, indices in width_groups.items():
        export_fig.add_trace(go.Scatter(
            x=[float(state.x_coord[i]) for i in indices],
            y=[float(state.y_coord[i]) for i in indices],
            mode=mode_str,
            text=[str(state.qubit_ids[i]) for i in indices] if opts['show_numbers'] else None,
            textposition='middle center',
            textfont=dict(color=[q_text_colors[i] for i in indices], size=opts['font_size']),
            marker=dict(
                color=[q_colors[i] for i in indices],
                size=opts['qubit_size'],
                line=dict(color=[q_line_colors[i] for i in indices], width=float(w_val)),
            ),
            hoverinfo='none', showlegend=False,
        ))

    # ── Legend (PNG / GIF export inherits the interactive legend) ──
    # Build legend-only traces with the same visibility + styling rules as
    # _update_legend uses for the interactive figure (keep in sync).
    legend_cfg = getattr(state, 'legend_config', None) or _legend_default_config()
    legend_enabled = bool(legend_cfg.get('enabled', True))
    legend_roles_cfg = legend_cfg.get('roles', {})
    _all_assigned = set()
    _populated = set()
    for ci in range(n_circuits):
        d_ci = state.layouts_data[ci]
        for role in ('alpha', 'beta', 'other', 'qubits', 'ancilla'):
            items = list(map(str, d_ci.get(role, []) or []))
            if items:
                _populated.add(role)
                _all_assigned.update(items)
    if len(_all_assigned) < n_q:
        _populated.add('unselected')
    d0 = state.layouts_data[0]
    legend_visible_count = 0
    if legend_enabled:
        for role in _LEGEND_ROLE_ORDER:
            role_cfg = legend_roles_cfg.get(role, {})
            if not role_cfg.get('show', False) or role not in _populated:
                continue
            fill, line, lw = _role_style(d0, role)
            # kaleido constraint: keep marker.line.width a scalar here.
            export_fig.add_trace(go.Scatter(
                x=[None], y=[None], mode='markers',
                marker=dict(size=30, color=fill,
                            line=dict(color=line, width=float(lw))),
                name=role_cfg.get('label', role),
                showlegend=True, hoverinfo='skip',
            ))
            legend_visible_count += 1

    # Layout
    annotations = []
    if annotation:
        annotations.append(dict(
            text=f"<b>{annotation}</b>", xref="paper", yref="paper",
            x=0.02, y=0.98, showarrow=False,
            font=dict(size=18, color="#1a5276"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#dee2e6", borderwidth=1,
        ))

    export_fig.update_layout(
        paper_bgcolor='white', plot_bgcolor='white',
        width=opts['fig_width'], height=opts['fig_height'],
        showlegend=(legend_visible_count > 0),
        legend=dict(
            x=1.02, y=1.0, xanchor='left', yanchor='top',
            bgcolor='rgba(255,255,255,0.85)',
            bordercolor='#cccccc', borderwidth=1,
            font=dict(size=34), itemsizing='constant',
        ),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor='x'),
        margin=dict(l=margin, r=margin, t=margin, b=margin + 90),
        hovermode='closest',
        font=dict(color='black', size=16),
        template='plotly_white',
        shapes=edge_shapes,
        annotations=annotations if annotations else None,
    )

    # ── 6. Crop to selection ──────────────────────────────────────
    if crop_to_selection:
        sel_indices = []
        for cidx in range(n_circuits):
            d = state.layouts_data[cidx]
            for q_id in d['styles']:
                idx = q_id_map.get(q_id)
                if idx is not None:
                    sel_indices.append(idx)
        if sel_indices:
            sel_x = [float(state.x_coord[i]) for i in sel_indices]
            sel_y = [float(state.y_coord[i]) for i in sel_indices]
            x_min, x_max = min(sel_x), max(sel_x)
            y_min, y_max = min(sel_y), max(sel_y)
            pad_x = max((x_max - x_min) * 0.15, 1.5)
            pad_y = max((y_max - y_min) * 0.15, 1.5)
            export_fig.update_layout(
                xaxis=dict(range=[x_min - pad_x, x_max + pad_x], autorange=False),
                yaxis=dict(range=[y_min - pad_y, y_max + pad_y], autorange=False, scaleanchor='x'),
            )

    return export_fig