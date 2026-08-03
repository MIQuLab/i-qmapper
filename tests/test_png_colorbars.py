"""Regression tests for error-map colorbars in the PNG export.

The bug: `plot.build_static_figure` renders no colorbars. On the interactive
figure the two error-map scales live on placeholder traces
(`_META_CB_QUBIT` / `_META_CB_GATE`) that `plot._update_colorbars` configures
in place, so a PNG built purely from the static renderer came out with the
heatmap but no scales — the reader had colours and no units.

`png_export._append_live_colorbars` copies the visible placeholders from the
live figure onto the export figure and makes room for them at the bottom.

No `write_image`, no kaleido: these tests exercise figure construction only.
"""
import plotly.graph_objects as go

import iqmapper.plot as plot_mod
from iqmapper.ui.png_export import (
    _CB_EXPORT_MIN_BOTTOM_MARGIN,
    _append_live_colorbars,
)


def _live_fig(*, qubit_visible=True, gate_visible=True, include_traces=True):
    """A stand-in for the interactive figure carrying the two placeholders."""
    fig = go.Figure()
    if not include_traces:
        return fig
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers', hoverinfo='none', showlegend=False,
        meta=plot_mod._META_CB_QUBIT,
        marker=dict(
            colorscale='Viridis', showscale=qubit_visible,
            cmin=0.0, cmax=0.25,
            colorbar=dict(title=dict(text='<b>Readout error</b>'),
                          orientation='h', x=0.25, y=-0.08),
        ),
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers', hoverinfo='none', showlegend=False,
        meta=plot_mod._META_CB_GATE,
        marker=dict(
            colorscale='Viridis', showscale=gate_visible,
            cmin=0.001, cmax=0.02,
            colorbar=dict(title=dict(text='<b>Gate error</b>'),
                          orientation='h', x=0.75, y=-0.08),
        ),
    ))
    return fig


def _export_fig(margin_b=40):
    fig = go.Figure()
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=margin_b))
    return fig


def test_visible_colorbars_are_copied_with_their_scale():
    """Both bars copied, preserving cmin/cmax/title — the PNG must show the
    same numeric range as the interactive view, not a rescaled one."""
    export_fig = _export_fig()
    n = _append_live_colorbars(export_fig, _live_fig(), plot_mod)

    assert n == 2
    assert len(export_fig.data) == 2

    q, g = export_fig.data
    assert q.marker.showscale is True
    assert (q.marker.cmin, q.marker.cmax) == (0.0, 0.25)
    assert q.marker.colorbar.title.text == '<b>Readout error</b>'
    assert (g.marker.cmin, g.marker.cmax) == (0.001, 0.02)
    assert g.marker.colorbar.title.text == '<b>Gate error</b>'

    # Copied traces must not draw data points of their own.
    assert q.mode == 'markers' and list(q.x) == [None]
    assert q.showlegend is False


def test_bottom_margin_is_raised_for_the_bars():
    """The bars sit at y=-0.08, below the axes — without extra bottom margin
    they are clipped out of the PNG."""
    export_fig = _export_fig(margin_b=40)
    _append_live_colorbars(export_fig, _live_fig(), plot_mod)
    assert export_fig.layout.margin.b >= _CB_EXPORT_MIN_BOTTOM_MARGIN


def test_generous_existing_margin_is_not_shrunk():
    """A user-chosen margin larger than our minimum must be left alone."""
    export_fig = _export_fig(margin_b=300)
    _append_live_colorbars(export_fig, _live_fig(), plot_mod)
    assert export_fig.layout.margin.b == 300


def test_hidden_colorbars_are_skipped():
    """showscale=False means the bar isn't on screen (non-error mode, or an
    all-NaN array) — nothing should be copied, and no margin bump."""
    export_fig = _export_fig(margin_b=40)
    n = _append_live_colorbars(
        export_fig,
        _live_fig(qubit_visible=False, gate_visible=False),
        plot_mod,
    )

    assert n == 0
    assert len(export_fig.data) == 0
    assert export_fig.layout.margin.b == 40


def test_partially_visible_copies_only_the_visible_bar():
    export_fig = _export_fig()
    n = _append_live_colorbars(
        export_fig, _live_fig(gate_visible=False), plot_mod)

    assert n == 1
    assert len(export_fig.data) == 1
    assert export_fig.data[0].marker.colorbar.title.text == '<b>Readout error</b>'


def test_no_live_figure_is_a_noop():
    """Callers that don't pass ctx.fig still get a PNG — just without bars."""
    export_fig = _export_fig(margin_b=40)
    n = _append_live_colorbars(export_fig, None, plot_mod)

    assert n == 0
    assert len(export_fig.data) == 0
    assert export_fig.layout.margin.b == 40


def test_missing_placeholder_traces_is_a_noop():
    """A live figure without the placeholders (e.g. built before the error
    map existed) must not raise."""
    export_fig = _export_fig(margin_b=40)
    n = _append_live_colorbars(
        export_fig, _live_fig(include_traces=False), plot_mod)

    assert n == 0
    assert len(export_fig.data) == 0


def test_copy_failure_does_not_propagate():
    """A colorbar is cosmetic: if copying one blows up, the export continues."""
    class _Boom:
        _META_CB_QUBIT = plot_mod._META_CB_QUBIT
        _META_CB_GATE = plot_mod._META_CB_GATE

        @staticmethod
        def _trace_by_meta(fig, meta):
            raise RuntimeError("trace lookup exploded")

    export_fig = _export_fig(margin_b=40)
    n = _append_live_colorbars(export_fig, _live_fig(), _Boom)   # must not raise

    assert n == 0
    assert export_fig.layout.margin.b == 40
