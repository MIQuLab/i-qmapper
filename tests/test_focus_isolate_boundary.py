"""Focus/Isolate error view: the gate SEPARATING two adjacent circuits must
dim — its endpoints are styled by DIFFERENT circuits, so it belongs to neither
layout.

Scenario (all tests): C1 owns {0,1} with intra-edge (0,1); C2 owns {2,3} with
intra-edge (2,3); a physical boundary edge (1,2) sits between them. Error mode,
per-edge values present. edges_list order → shape index:
  edge 0 = (0,1) intra-C1,  edge 1 = (1,2) BOUNDARY,  edge 2 = (2,3) intra-C2.
"""
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('plotly')

from iqmapper import plot                      # noqa: E402

_STYLE = {'fill': 'rgba(99,110,250,1.0)', 'line': '#000000',
          'width': 2, 'text_color': '#000000'}


def _make_state(single=False):
    from iqmapper.state import SessionState
    s = SessionState()
    s.active_backend_name = 'ibm_test'
    s.active_chip = 'Heron r1'
    s.current_colorscale = 'Viridis'
    s.qubit_ids = ['0', '1', '2', '3']
    s.edges_list = [(0, 1), (1, 2), (2, 3)]        # edge 1 = boundary
    s.x_coord = np.array([0., 1., 2., 3.])
    s.y_coord = np.array([0., 0., 0., 0.])
    s.readout_errors = np.array([0.01, 0.02, 0.03, 0.04])
    s.gate_errors = np.array([0.01, 0.02, 0.03])
    s.qubit_props_data = {'readout_error': s.readout_errors}
    s.gate_props_data = {'cz_error': s.gate_errors}
    s.delta_mode = False

    for i in list(s.layouts_data):                 # clear the default slots
        d = s.layouts_data[i]
        for k in ('alpha', 'beta', 'other', 'qubits', 'ancilla'):
            d[k] = []
        d['styles'] = {}

    d0 = s.layouts_data[0]
    d0['alpha'], d0['beta'] = ['0'], ['1']
    d0['styles'] = {'0': dict(_STYLE), '1': dict(_STYLE)}
    if not single:
        d1 = s.layouts_data[1]
        d1['alpha'], d1['beta'] = ['2'], ['3']
        d1['styles'] = {'2': dict(_STYLE), '3': dict(_STYLE)}
    return s


def _V(x):
    return SimpleNamespace(value=x)


def _wref(*, focus=False, isolate=False, n_circuits=2):
    return SimpleNamespace(
        view_mode=_V('error'), num_circuits=_V(n_circuits),
        intra_qpu_mode=_V(n_circuits > 1),
        dd_qubit_prop=_V('readout_error'), dd_gate_prop=_V('cz_error'),
        size_slider=_V(20), gate_size_slider=_V(3), gate_hl_w_slider=_V(5),
        font_size_slider=_V(12), show_numbers=_V(True),
        plot_w_slider=_V(800), plot_h_slider=_V(600),
        pp_bg_fill=_V('#ffffff'), pp_bg_line=_V('#dddddd'),
        pp_gate_col=_V('#888888'), global_text_color=_V('#000000'),
        chk_link_gates=_V(False), chk_focus_selected=_V(focus),
        chk_isolate=_V(isolate), chk_err_threshold=_V(False),
        txt_q_thr_min=_V(''), txt_q_thr_max=_V(''),
        txt_g_thr_min=_V(''), txt_g_thr_max=_V(''),
    )


def _opts(*, focus=False, isolate=False, n_circuits=2):
    return {
        'is_error': True, 'qubit_size': 20, 'gate_width': 3, 'gate_hl_width': 5,
        'show_numbers': True, 'font_size': 12, 'link_gates': False,
        'bg_fill': '#ffffff', 'bg_line': '#dddddd', 'gate_color': '#888888',
        'q_prop_key': 'readout_error', 'g_prop_key': 'cz_error',
        'n_circuits': n_circuits, 'fig_width': 800, 'fig_height': 600,
        'threshold_enabled': False, 'focus_selected': focus, 'isolate': isolate,
        'q_thr_min': None, 'q_thr_max': None, 'g_thr_min': None, 'g_thr_max': None,
        'text_color': '#000000',
    }


def _edge_colors_interactive(state, w):
    fig = plot.create_figure()
    plot.build_device_graph(fig, state, w)
    plot.refresh_plot(fig, state, w)
    # In error mode each edge is one layout shape, in edges_list order.
    return [sh.line.color for sh in fig.layout.shapes]


def _edge_colors_static(state, opts):
    fig = plot.build_static_figure(state, opts)
    return [sh.line.color for sh in fig.layout.shapes]


# ── 1 + 2. interactive Focus / Isolate ──────────────────────────────────────

def test_focus_dims_boundary_edge_interactive():
    colors = _edge_colors_interactive(_make_state(), _wref(focus=True))
    assert len(colors) == 3
    assert colors[1] == plot._DIM_COL          # (1,2) boundary → dimmed
    assert colors[0] != plot._DIM_COL          # (0,1) intra-C1 → heatmap
    assert colors[2] != plot._DIM_COL          # (2,3) intra-C2 → heatmap


def test_isolate_dims_boundary_edge_interactive():
    colors = _edge_colors_interactive(_make_state(), _wref(isolate=True))
    assert len(colors) == 3
    assert colors[1] == plot._DIM_COL
    assert colors[0] != plot._DIM_COL
    assert colors[2] != plot._DIM_COL


# ── 3. static (PNG) renderer, both modes ────────────────────────────────────

def test_static_dims_boundary_edge():
    for focus, isolate in ((True, False), (False, True)):
        colors = _edge_colors_static(_make_state(),
                                     _opts(focus=focus, isolate=isolate))
        assert len(colors) == 3
        assert colors[1] == plot._DIM_COL, (focus, isolate)
        assert colors[0] != plot._DIM_COL, (focus, isolate)
        assert colors[2] != plot._DIM_COL, (focus, isolate)


# ── 4. single-circuit regression: α-β internal edge stays colored ───────────

def test_single_circuit_unchanged():
    # One circuit owning {0,1}; edge (0,1) is its own α-β gate → NOT dimmed.
    state = _make_state(single=True)
    colors_i = _edge_colors_interactive(state, _wref(focus=True, n_circuits=1))
    assert colors_i[0] != plot._DIM_COL

    colors_s = _edge_colors_static(_make_state(single=True),
                                   _opts(focus=True, n_circuits=1))
    assert colors_s[0] != plot._DIM_COL
