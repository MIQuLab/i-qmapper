"""
ui.inter_qpu.py — inter-QPU mode wiring (toggle + connect QPU2).

The user can compare or distribute circuits across two physical QPUs by
flipping the "Inter-QPU" checkbox in the Hardware accordion. Doing so:

  * switches the figure layout from full-width (single QPU) to
    side-by-side (QPU1 + QPU2);
  * adds a `QPU 1 / QPU 2` ToggleButtons strip at the top of the
    sidebar so the user can swap between the two backends' sidebars
    without losing either's settings;
  * unlocks the QPU2 backend dropdown + connect button so the user can
    pick a second backend and instantiate its full panel via
    `ui.build_mapper_ui(s2, is_secondary=True)` — same builder as the
    primary, so QPU 2 has full feature parity (multi-circuit, ranking,
    trend, fidelity, time panel, session save/load, ...). The previous
    "lite" implementation in ui_qpu2.py was a maintenance burden and
    has been removed.

All of this used to live as a >150-line block of nested closures in
ui.py. The behaviour is unchanged — only the location moves. State
that is exclusively inter-QPU (the QPU2 SessionState handle, the
re-parented sidebar body, the saved pre-inter-QPU figure dimensions)
is now scoped inside `setup_inter_qpu`; ui.py only sees the layout
containers it has to mutate.

Public API:
  setup_inter_qpu(ctx) -> refresh_qpu1_label callable

The function wires observers by side effect on the widgets passed in, and
returns the `refresh_qpu1_label` hook so `ui.backend_switch` can refresh
the QPU-1 switcher label after a backend switch.
"""
import contextlib
import logging

from ipywidgets import VBox, HTML, ToggleButtons

from ..constants import hex_to_rgba
from ..html_safe import esc
from ..state import SessionState

_log = logging.getLogger(__name__)


def setup_inter_qpu(ctx):
    """Wire the inter-QPU toggle + connect-QPU2 callbacks.

    Required ctx fields (grouped):

      Modules / classes   : data_mod
      Primary QPU state   : state
      Toggle widgets      : chk_inter_qpu, dd_backend_qpu2,
                            btn_connect_qpu2, lbl_qpu2_status
      Display sliders     : plot_w_slider, plot_h_slider
      Figure label        : lbl_fig1            (QPU1 banner; visibility
                                                 toggled by inter-QPU mode)
      Sidebar / layout    : ui_sidebar          (re-parented when toggling)
                            ui_sidebar_content  (the QPU1 sidebar)
                            _sidebar_body       (the swappable wrapper VBox)
                            fig_box             (QPU1 figure container)
                            _qpu2_sidebar_box   (QPU2 sidebar slot)
                            _qpu2_fig_box       (QPU2 figure slot)
                            full_ui             (top-level HBox)

    Must be called AFTER the layout containers above are created — they
    are mutated inside the toggle handler. Internal state (the QPU2
    SessionState handle, ToggleButtons reference, saved fig dimensions)
    is scoped inside this function via mutable single-element lists.
    """
    state = ctx.state
    chk_inter_qpu = ctx.chk_inter_qpu
    dd_backend_qpu2 = ctx.dd_backend_qpu2
    btn_connect_qpu2 = ctx.btn_connect_qpu2
    lbl_qpu2_status = ctx.lbl_qpu2_status
    plot_w_slider = ctx.plot_w_slider
    plot_h_slider = ctx.plot_h_slider
    lbl_fig1 = ctx.lbl_fig1
    ui_sidebar = ctx.ui_sidebar
    ui_sidebar_content = ctx.ui_sidebar_content
    sidebar_body = ctx.sidebar_body
    fig_box = ctx.fig_box
    qpu2_sidebar_box = ctx.qpu2_sidebar_box
    qpu2_fig_box = ctx.qpu2_fig_box
    full_ui = ctx.full_ui

    # ── Inter-QPU local state ───────────────────────────────────
    # Single-element mutable lists so the nested observers/handlers
    # can rebind without a `nonlocal` declaration. They are scoped to
    # this function — no other UI code reads them.
    _state2 = [None]              # SessionState for QPU 2
    _fig2 = [None]                # FigureWidget for QPU 2
    _sidebar2 = [None]            # QPU 2 sidebar VBox
    _zoom2 = [None]               # QPU 2 zoom toolbar
    _dispose2 = [None]            # callable that unobserves all QPU 2 widgets
    _pre_inter_fig_size = [None, None]  # (width, height) before toggling on
    _qpu_toggle = [None]          # ToggleButtons for QPU1/QPU2 sidebar swap
    _pending_handoff = [None]     # (backend_name, candidate) staged from a scan pick
    _qpu2_content_box = [None]    # Container holding the QPU2 sidebar

    # ── Toggle handler ──────────────────────────────────────────
    def _on_toggle_inter_qpu(change):
        if change['new']:
            dd_backend_qpu2.disabled = False
            btn_connect_qpu2.disabled = False
            if state.available_backends:
                dd_backend_qpu2.options = state.available_backends
            # Save current fig size and shrink for side-by-side display.
            _pre_inter_fig_size[0] = plot_w_slider.value
            _pre_inter_fig_size[1] = plot_h_slider.value
            plot_w_slider.value = 1200
            plot_h_slider.value = 1400
            lbl_fig1.layout.display = 'block'

            # Add QPU switcher toggle at top of sidebar
            _qpu_toggle[0] = ToggleButtons(
                options=[('📡 QPU 1', 'qpu1'), ('📡 QPU 2', 'qpu2')],
                value='qpu1',
                style={'button_width': '140px'},
            )
            _qpu2_content_box[0] = VBox([
                HTML("<div style='font-size:15px;color:#888;padding:20px;'>"
                     "Connect QPU 2 from the Hardware section above.</div>")
            ])

            def _on_qpu_switch(c):
                if c['new'] == 'qpu1':
                    sidebar_body.children = [ui_sidebar_content]
                else:
                    sidebar_body.children = [_qpu2_content_box[0]]
            _qpu_toggle[0].observe(_on_qpu_switch, 'value')

            sidebar_body.children = [ui_sidebar_content]
            ui_sidebar.children = [_qpu_toggle[0], sidebar_body]

            # Layout: sidebar + QPU1 map + QPU2 map
            full_ui.children = [ui_sidebar, fig_box, qpu2_fig_box]
        else:
            dd_backend_qpu2.disabled = True
            btn_connect_qpu2.disabled = True
            lbl_qpu2_status.value = ""
            # Detach QPU 2 observers BEFORE dropping references — otherwise
            # the entire QPU 2 widget graph + FigureWidget stays alive in
            # ipywidgets' internal handler lists and we leak on every
            # inter-QPU on/off cycle.
            if _dispose2[0] is not None:
                try:
                    _dispose2[0]()
                except Exception:
                    # The original bare `pass` here masked the very
                    # leak this block was written to prevent: if the
                    # disposer raised, every QPU2 observer + the
                    # FigureWidget it captured stayed alive in memory
                    # AND we'd silently move on as if cleanup succeeded.
                    # Log the traceback so the leak is visible in the
                    # kernel log (the UI still proceeds to drop the
                    # references below — partial cleanup is better
                    # than no cleanup, and we don't want a disposer
                    # bug to block the inter-QPU toggle).
                    _log.exception(
                        "QPU2 disposer raised; observers may be leaked")
            _dispose2[0] = None
            _state2[0] = None
            _fig2[0] = None
            _sidebar2[0] = None
            _zoom2[0] = None
            qpu2_sidebar_box.children = []
            qpu2_fig_box.children = []
            # Detach the observer from the outgoing toggle BEFORE dropping
            # our reference to it. Without this, repeated inter-QPU
            # on/off cycles accumulate dead observers (the ToggleButtons
            # is no longer in the display tree but the closure registered
            # via .observe keeps it alive in memory along with every
            # captured local). unobserve_all is safe even when the only
            # registered handler was _on_qpu_switch.
            if _qpu_toggle[0] is not None:
                try:
                    _qpu_toggle[0].unobserve_all()
                except Exception:
                    # Same rationale as the disposer block above: if
                    # unobserve_all raises (very unusual for ipywidgets
                    # ToggleButtons but possible if the widget was
                    # already partially torn down) we silently leak
                    # the toggle's observer closures. Log so it's
                    # visible; the reference drop below still proceeds.
                    _log.exception(
                        "QPU toggle unobserve_all raised; observer may be leaked")
            _qpu_toggle[0] = None
            _qpu2_content_box[0] = None
            lbl_fig1.layout.display = 'none'
            # Restore single-sidebar layout.
            sidebar_body.children = [ui_sidebar_content]
            ui_sidebar.children = [sidebar_body]
            full_ui.children = [ui_sidebar, fig_box]
            if _pre_inter_fig_size[0] is not None:
                plot_w_slider.value = _pre_inter_fig_size[0]
                plot_h_slider.value = _pre_inter_fig_size[1]

    chk_inter_qpu.observe(_on_toggle_inter_qpu, names='value')

    # ── Generic widget-tree disposer ───────────────────────────
    # Used to unobserve every callback in QPU 2's full-feature widget
    # tree (built by build_mapper_ui, hundreds of widgets) so the panel
    # can be GC'd cleanly when the user toggles inter-QPU off.
    def _walk_unobserve_all(root):
        seen = set()
        def _visit(w):
            if id(w) in seen:
                return
            seen.add(id(w))
            try:
                w.unobserve_all()
            except Exception:
                # Best-effort teardown: the widget tree can contain
                # non-Widget objects (traitlets edge cases, custom
                # containers) that lack unobserve_all. Failures here
                # only leak a few observer refs — keep walking.
                _log.debug("unobserve_all failed on %r", w, exc_info=True)
            if hasattr(w, 'children'):
                for c in w.children:
                    _visit(c)
        _visit(root)

    # ── QPU-1 toggle-label refresh ──────────────────────────────
    def refresh_qpu1_label():
        """Rebuild the QPU-1 switcher label from the CURRENT backend name.

        The QPU-1/QPU-2 ToggleButtons label for option[0] is built once,
        inside `_on_connect_qpu2`, from `state.active_backend_name` at that
        moment. A later backend switch (`ui.backend_switch`) changes the
        name but has no reference to this closure, so the tab keeps the
        stale name. `ui.backend_switch` calls this hook after every switch.

        No-op when the inter-QPU switcher was never built (`_qpu_toggle[0]`
        is None) or is not the expected two-option strip. option[1] (the
        QPU-2 entry) and the current `.value` are left untouched.
        """
        toggle = _qpu_toggle[0]
        if toggle is None:
            return
        options = list(toggle.options)
        if len(options) != 2:
            return
        # Reassigning .options resets .value to the first option (verified
        # for ipywidgets ToggleButtons even when the old value survives), so
        # capture and restore the user's current selection explicitly.
        saved_value = toggle.value
        new_label = f"📡 {state.active_backend_name or 'QPU 1'}"
        options[0] = (new_label, options[0][1])
        toggle.options = options
        toggle.value = saved_value

    # ── B5: remote-candidate handoff (scan → inter-QPU) ─────────
    handoff_cell = getattr(ctx, 'handoff_cell', None)

    def _apply_candidate_to_state(s2, candidate):
        """Pre-load a scan-selected candidate onto QPU-2's layout state so
        the secondary map renders it from its first paint. Called BEFORE
        build_mapper_ui(s2), so no qubit-id filtering is possible yet — the
        candidate's qubits are valid ids for that backend by construction."""
        d = s2.layouts_data[0]
        d['alpha'] = []; d['beta'] = []; d['other'] = []
        d['qubits'] = []; d['ancilla'] = []; d['styles'] = {}
        defs = d['defaults']
        line = defs['line']
        fills = {'alpha': hex_to_rgba(defs['fill'], 1.0),
                 'beta': hex_to_rgba(defs['fill_beta'], 1.0),
                 'ancilla': hex_to_rgba(defs['fill_anc'], 1.0)}
        d['alpha'] = [str(q) for q in candidate['alpha']]
        d['beta'] = [str(q) for q in candidate['beta']]
        d['ancilla'] = [str(q) for q in candidate.get('ancillas', [])]
        for role in ('alpha', 'beta', 'ancilla'):
            for q in d[role]:
                d['styles'][q] = {'fill': fills[role], 'line': line,
                                  'width': 2, 'text_color': '#000000'}

    def stage_remote_candidate(backend_name, candidate):
        """Scan-side entry point: reveal the inter-QPU panel, prefill the
        QPU-2 backend dropdown, and stage the candidate so it renders as
        soon as the (user-clicked) connect completes."""
        if not chk_inter_qpu.value:
            chk_inter_qpu.value = True          # fires _on_toggle_inter_qpu
        with contextlib.suppress(Exception):
            dd_backend_qpu2.value = backend_name
        _pending_handoff[0] = (backend_name, candidate)
        # (C3) Prominent one-liner explaining the half-open state until the
        # user clicks Connect. Replaced by the connect status once connected.
        staged_html = (
            f"<div style='font-size:14px;color:#8a4b00;background:#fff3cd;"
            f"padding:6px 10px;border-radius:4px;border:1px solid #ffe08a;"
            f"margin:3px 0;font-weight:bold;'>📡 {esc(str(backend_name))} "
            f"staged — click <b>Connect QPU 2</b> to load the device and "
            f"paint the candidate.</div>")
        lbl_qpu2_status.value = staged_html
        # Also surface it inside the revealed QPU-2 sidebar slot, so the
        # prompt is visible wherever the user is looking.
        if _qpu2_content_box[0] is not None:
            with contextlib.suppress(Exception):
                _qpu2_content_box[0].children = [HTML(staged_html)]

    if handoff_cell is not None:
        handoff_cell[0] = stage_remote_candidate

    # ── Connect-QPU2 handler ────────────────────────────────────
    def _on_connect_qpu2(b):
        backend_name = dd_backend_qpu2.value
        if not backend_name:
            lbl_qpu2_status.value = (
                "<span style='font-size:15px;color:#e74c3c;'>"
                "Select a backend</span>"
            )
            return
        if state.active_service is None:
            lbl_qpu2_status.value = (
                "<span style='font-size:15px;color:#e74c3c;'>"
                "No service connected</span>"
            )
            return

        lbl_qpu2_status.value = (
            f"<span style='font-size:15px;color:#e67e22;'>"
            f"⏳ Connecting {esc(backend_name)}...</span>"
        )
        try:
            from ..connection import _check_properties
            from ..constants import resolve_coords
            # Late import to avoid the ui ↔ ui.inter_qpu cycle that would
            # otherwise trigger at module load time.
            from . import build_mapper_ui

            s2 = SessionState()
            s2.active_service = state.active_service
            s2.available_backends = state.available_backends
            s2.active_ansatz = state.active_ansatz
            s2.current_colorscale = state.current_colorscale
            s2.active_vendor = 'IBM'

            new_backend = state.active_service.backend(backend_name)
            props, props_warn = _check_properties(new_backend, backend_name)

            new_coords, chip_label = resolve_coords(new_backend)

            s2.active_backend = new_backend
            s2.active_backend_name = backend_name
            s2.active_chip = chip_label
            s2.initial_props = props
            s2.props_warning = props_warn
            s2.active_coords = new_coords

            # B5: if this connect was staged from a scan pick for this
            # backend, pre-load the chosen candidate so QPU-2 paints it.
            pend = _pending_handoff[0]
            if pend is not None and pend[0] == backend_name:
                with contextlib.suppress(Exception):
                    _apply_candidate_to_state(s2, pend[1])
                _pending_handoff[0] = None

            # Build the FULL secondary UI — feature-parity with QPU 1.
            # build_mapper_ui internally calls _load_device_graph which
            # populates s2.x_coord / qubit_ids / edges_list from props,
            # so we don't need to call data_mod.generate_device_data
            # ahead of time anymore.
            full_ui_2, f2, w2 = build_mapper_ui(
                s2, is_secondary=True, return_widgets=True)
            # Top-level shape: HBox([ui_sidebar, fig_box]).
            # fig_box.children = (lbl_fig1, zoom_bar, fig).
            #
            # We only re-parent `sidebar2`, `zoom2_bar`, and `f2` into the
            # inter-QPU layout. `lbl_fig1` (children[0] of fig_box2) and
            # `fig_box2` itself are left orphan inside `full_ui_2` — they
            # are never displayed (full_ui_2 has no parent), but we keep
            # a reference via the `_dispose2` lambda below so the disposer
            # walks them and clears any observers when inter-QPU is
            # toggled off.
            sidebar2 = full_ui_2.children[0]
            fig_box2 = full_ui_2.children[1]
            zoom2_bar = fig_box2.children[1]

            _state2[0] = s2
            _fig2[0] = f2
            _sidebar2[0] = sidebar2
            _zoom2[0] = zoom2_bar
            # Disposer: walk the full widget tree (sidebar + fig_box) and
            # unobserve_all on each descendant. The tree has ~300 widgets
            # so a generic walker is more robust than enumerating them.
            _dispose2[0] = lambda: _walk_unobserve_all(full_ui_2)

            # Place QPU2 sidebar in the switchable content
            if _qpu2_content_box[0] is not None:
                _qpu2_content_box[0].children = [sidebar2]
            if _qpu_toggle[0] is not None:
                _qpu_toggle[0].options = [
                    (f"📡 {state.active_backend_name or 'QPU 1'}", 'qpu1'),
                    (f"📡 {backend_name}", 'qpu2'),
                ]
                # Show QPU2 sidebar after connect so user sees the new backend
                _qpu_toggle[0].value = 'qpu2'
                sidebar_body.children = [_qpu2_content_box[0]]

            # QPU2 figure banner.
            lbl_fig2 = HTML(
                f"<div style='font-size:14px;font-weight:bold;color:#fff;"
                f"background:#2C3E50;padding:3px 10px;border-radius:4px;"
                f"text-align:center;margin-bottom:2px;'>"
                f"📡 {esc(backend_name)}</div>"
            )
            qpu2_fig_box.children = [lbl_fig2, zoom2_bar, f2]
            # Match inter-QPU fig size by setting the QPU 2 sliders —
            # their observers (wired by build_mapper_ui) propagate the
            # value to f2.update_layout for us. Without this, the sliders
            # in the QPU 2 sidebar stayed at 1700/1650 (default) while
            # f2 was shrunk to 1200/1400, so the next slider drag would
            # snap f2 back to the wrong size.
            # QPU2's fig-size sliders come from its WidgetsRef now — no
            # fragile description-string tree-walk that breaks silently if a
            # slider label is renamed. These are the same slider objects
            # mounted in sidebar2, so setting .value fires their observers
            # (wired by build_mapper_ui) and resizes f2.
            w2.plot_w_slider.value = 1200
            w2.plot_h_slider.value = 1400

            # All interpolated values come from external/IBM data (backend
            # name, chip label, props warnings, calibration label) and a
            # session-loaded session file could carry adversarial strings
            # via state restore — escape every one of them before they hit
            # the HTML widget.
            lbl_qpu2_status.value = (
                f"<div style='font-size:15px;color:#1e8449;background:#eafaf1;"
                f"padding:4px 8px;border-radius:4px;border:1px solid #abebc6;"
                f"margin:3px 0;'>"
                f"✅ Connected to <b>{esc(backend_name)}</b> "
                f"({esc(chip_label)}) — "
                f"{len(s2.qubit_ids)} qubits, {len(s2.edges_list)} gates"
                f"{'<br>⚠️ ' + esc(props_warn) if props_warn else ''}"
                f"{'<br>📅 ' + esc(s2.calibration_data_label or '') if s2.calibration_data_label else ''}"
                f"</div>"
            )
        except Exception as e:
            lbl_qpu2_status.value = (
                f"<span style='font-size:15px;color:#e74c3c;'>"
                f"❌ {esc(str(e))}</span>"
            )

    btn_connect_qpu2.on_click(_on_connect_qpu2)

    # Expose the label-refresh hook so ui.backend_switch can call it after a
    # backend switch (the switcher has no other handle on this closure).
    return refresh_qpu1_label
