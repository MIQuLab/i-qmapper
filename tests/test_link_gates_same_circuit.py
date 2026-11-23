"""Linked-gate highlighting: an ancilla colours its edges only towards
qubits of its own circuit, in the design view and in the PNG export.

Scenario: C1 owns alpha 0 and ancilla 1, C2 owns alpha 2 and beta 3, on a
line 0-1-2-3. Edge (0,1) is C1's ancilla gate, edge (1,2) joins C1's ancilla
to C2, edge (2,3) is C2's alpha-beta boundary.
"""
import pytest

pytest.importorskip('plotly')

from iqmapper import plot                      # noqa: E402

from test_focus_isolate_boundary import _make_state, _opts, _wref  # noqa: E402

_C1_LINE, _C2_LINE = '#aa0000', '#0000aa'


def _state():
    s = _make_state()
    d0, d1 = s.layouts_data[0], s.layouts_data[1]
    d0['alpha'], d0['beta'], d0['ancilla'] = ['0'], [], ['1']
    for q in ('0', '1'):
        d0['styles'][q]['line'] = _C1_LINE
    for q in ('2', '3'):
        d1['styles'][q]['line'] = _C2_LINE
    return s


def _highlighted(shapes):
    """{(x0, x1): colour} of the highlighted edges on the 0..3 line."""
    return {(round(sh.x0), round(sh.x1)): sh.line.color for sh in shapes}


def test_design_view_highlights_the_ancilla_edge_only_inside_its_circuit():
    w = _wref()
    w.view_mode.value = 'design'
    w.chk_link_gates.value = True
    fig = plot.create_figure()
    state = _state()
    plot.build_device_graph(fig, state, w)
    plot.refresh_plot(fig, state, w)
    assert _highlighted(fig.layout.shapes) == {(0, 1): _C1_LINE}


def test_png_export_highlights_the_ancilla_edge_only_inside_its_circuit():
    opts = _opts()
    opts['is_error'], opts['link_gates'] = False, True
    fig = plot.build_static_figure(_state(), opts)
    assert _highlighted(fig.layout.shapes) == {(0, 1): _C1_LINE}
