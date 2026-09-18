"""PNG export copies the live figure's visible colorbars onto the static
export figure (plot.build_static_figure has no colorbar code).

Offline: never calls write_image (kaleido/chrome unavailable); uses plain
plotly figures and stubs the render/write path.
"""
import types
from types import SimpleNamespace

import pytest

pytest.importorskip('plotly')
import plotly.graph_objects as go     # noqa: E402

from iqmapper import plot            # noqa: E402  (constants + _trace_by_meta only)
from iqmapper.ui.png_export import _append_live_colorbars  # noqa: E402


def _cb_trace(meta, *, showscale, cmin, cmax, title):
    return go.Scatter(
        x=[None], y=[None], mode='markers', meta=meta,
        hoverinfo='none', showlegend=False,
        marker=dict(showscale=showscale, cmin=cmin, cmax=cmax,
                    colorscale='Viridis', colorbar=dict(title=title)))


# ── 1. visible bars are copied with their config + canvas gets bottom room ──

def test_appends_visible_colorbars():
    live = go.Figure()
    live.add_trace(_cb_trace(plot._META_CB_QUBIT, showscale=True,
                             cmin=0.0, cmax=1.0, title='Qubit'))
    live.add_trace(_cb_trace(plot._META_CB_GATE, showscale=True,
                             cmin=2.0, cmax=3.0, title='Gate'))
    export = go.Figure()

    n = _append_live_colorbars(export, live, plot)

    assert n == 2
    assert len(export.data) == 2
    assert sorted(t.marker.cmin for t in export.data) == [0.0, 2.0]
    assert sorted(t.marker.cmax for t in export.data) == [1.0, 3.0]
    assert sorted(t.marker.colorbar.title.text for t in export.data) == ['Gate', 'Qubit']
    assert export.layout.margin.b >= 120


# ── 2. design mode (showscale=False) → nothing copied, margin untouched ─────

def test_hidden_bars_are_not_copied():
    live = go.Figure()
    live.add_trace(_cb_trace(plot._META_CB_QUBIT, showscale=False,
                             cmin=0.0, cmax=1.0, title='Qubit'))
    live.add_trace(_cb_trace(plot._META_CB_GATE, showscale=False,
                             cmin=2.0, cmax=3.0, title='Gate'))
    export = go.Figure()
    margin_before = export.layout.margin.b

    n = _append_live_colorbars(export, live, plot)

    assert n == 0
    assert len(export.data) == 0
    assert export.layout.margin.b == margin_before      # untouched


# ── 3. no live fig / no placeholder traces → no-op, no exception ────────────

def test_missing_live_fig_is_noop():
    export = go.Figure()
    assert _append_live_colorbars(export, None, plot) == 0
    assert len(export.data) == 0

    live = go.Figure()
    live.add_trace(go.Scatter(x=[1], y=[1], meta='qubits'))   # not a CB placeholder
    assert _append_live_colorbars(export, live, plot) == 0
    assert len(export.data) == 0


# ── 4. the export site runs the helper BETWEEN build and write ──────────────

def _walk(node):
    yield node
    for ch in getattr(node, 'children', []):
        yield from _walk(ch)


def test_export_site_invokes_helper(monkeypatch, tmp_path):
    from ipywidgets import Text, Button, HTML, Output, Layout
    from iqmapper.ui import png_export as pe

    events = []

    class _FakeFig:
        def write_image(self, path, scale=None, width=None, height=None):
            events.append('write')

    plot_mod = SimpleNamespace(
        collect_render_opts=lambda w: {'fig_width': 100, 'fig_height': 100},
        build_static_figure=lambda state, opts, margin=0, crop_to_selection=False: (
            events.append('build') or _FakeFig()),
        _META_CB_QUBIT=plot._META_CB_QUBIT, _META_CB_GATE=plot._META_CB_GATE,
        _trace_by_meta=plot._trace_by_meta,
    )
    # Record that the helper ran (its own behavior is covered by tests 1-3).
    monkeypatch.setattr(pe, '_append_live_colorbars',
                        lambda ef, lf, pm: events.append('append') or 0)

    out_path = str(tmp_path / 'shot.png')                # does not exist yet
    ctx = SimpleNamespace(
        state=SimpleNamespace(), widgets_ref=SimpleNamespace(),
        fname_img=Text(value='shot'), btn_save_img=Button(), W=Layout(),
        plot_mod=plot_mod, safe_path=lambda p, *a, **k: p,
        user_save_path=lambda name, ext, default_dir=None: out_path,
        save_dir=lambda: str(tmp_path), out=Output(),
        lbl_png_status=HTML(), fig=go.Figure(),
    )
    popup, _, _ = pe.setup_png_export(ctx)

    btn_confirm = next(w for w in _walk(popup)
                       if isinstance(w, Button) and 'Export now' in (w.description or ''))
    btn_confirm.click()

    # Helper ran, and it ran AFTER build_static_figure and BEFORE write_image.
    assert events == ['build', 'append', 'write']
