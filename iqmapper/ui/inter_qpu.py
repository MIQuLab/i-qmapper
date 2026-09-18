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


def _resync_widget_state(fig):
    """Force ``fig``'s full state into its synced widget traits before the
    figure is first mounted via ``Box.children``.

    plotly 6.x FigureWidget snapshots ``_widget_data``/``_widget_layout`` only
    at construction (empty) and in ``_repr_mimebundle_`` (cell display). A
    figure mounted as a child widget never passes through either, so the
    frontend model renders whatever subset of the incremental
    ``_py2js_addTraces``/``restyle``/``relayout`` messages it managed to apply.
    In the live one-click handoff that subset lost the qubit node trace →
    "edges but no nodes" empty map (the frontend model's ``do_data`` handler is
    a no-op, so this resync repaints nothing once a view exists — it MUST run
    before the mount). Re-snapshotting here makes the trait state
    authoritative, matching what plotly itself does at display time.

    Touches plotly-private attributes, so it is a guarded no-op if they ever
    move; failure must never break the connect (worst case: the pre-fix
    behavior). Returns True if the resync was applied.
    """
    try:
        from copy import deepcopy
        fig._widget_data = deepcopy(fig._data)
        fig._widget_layout = deepcopy(fig._layout_obj._props)
        return True
    except Exception:                    # noqa: BLE001 - logged, never fatal
        _log.warning("widget-state resync skipped (plotly internals moved?)",
                     exc_info=True)
        return False


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
    _w2 = [None]                  # WidgetsRef for QPU 2 (for refresh_plot repaint)
    _sidebar2 = [None]            # QPU 2 sidebar VBox
    _zoom2 = [None]               # QPU 2 zoom toolbar
    _dispose2 = [None]            # callable that unobserves all QPU 2 widgets
    _connect_in_flight = [False]  # race guard: one auto-connect at a time

    # Background runner for the ONE-CLICK auto-connect: the network fetch runs
    # on a worker; default_spawn_bg marshals the completion (the PAINT) back
    # onto the kernel main loop. Injectable via ctx for headless tests.
    _spawn_bg = getattr(ctx, 'spawn_bg', None)
    if _spawn_bg is None:
        from ..async_util import default_spawn_bg as _spawn_bg
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

    # ── remote-candidate handoff (scan → inter-QPU) ─────────
    handoff_cell = getattr(ctx, 'handoff_cell', None)

    def _apply_one_to_slot(d, candidate):
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

    def _apply_candidate_to_state(s2, candidate):
        """Pre-load scan-selected candidate(s) onto QPU-2's layout state so the
        secondary map renders from its first paint. A single candidate lands on
        C1; a LIST (a multi-circuit batch) lands on C1..Cn, one per slot — the
        list form is what the batch-scan handoff sends."""
        cands = candidate if isinstance(candidate, (list, tuple)) else [candidate]
        for i, cand in enumerate(cands):
            d = s2.layouts_data.get(i)
            if d is not None:
                _apply_one_to_slot(d, cand)

    def stage_remote_candidate(backend_name, candidate):
        """Scan-side entry point: reveal the inter-QPU panel, prefill the
        QPU-2 backend dropdown, and stage the candidate so it renders as
        soon as the (user-clicked) connect completes."""
        if not chk_inter_qpu.value:
            chk_inter_qpu.value = True          # fires _on_toggle_inter_qpu
        with contextlib.suppress(Exception):
            dd_backend_qpu2.value = backend_name
        _pending_handoff[0] = (backend_name, candidate)
        # Prominent one-liner explaining the half-open state until the
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

    # The scan handoff now triggers ONE-CLICK connect (stage + auto-
    # connect), defined after the connect handler below. Bound via a trampoline
    # so this assignment can precede _auto_connect's definition.
    if handoff_cell is not None:
        handoff_cell[0] = lambda bn, cand: _auto_connect(bn, cand)

    # ── Connect-QPU2: worker-safe FETCH + main-thread PAINT ──────
    # Every ipywidgets / plotly FigureWidget mutation (build_mapper_ui,
    # _apply_candidate_to_state, refresh_plot, mounting into
    # qpu2_fig_box) MUST run on the kernel's main thread. A FigureWidget
    # mutated from a background worker sends its comm messages on a
    # non-kernel thread and the browser channel silently drops them —
    # the map paints nothing (no error, no exception). This is why the
    # one-click auto-connect produced an empty QPU-2 map while the manual
    # button (which runs in the widget-callback = main thread) always
    # worked. So we fetch (network only, no widgets) on the worker and
    # marshal the PAINT back to the main loop via default_spawn_bg's
    # on_done, matching the "marshal background work back before touching
    # any widget" rule the rest of the codebase already follows.
    def _connect_qpu2_fetch(backend_name):
        """Worker-safe: backend handle + properties + coord resolution.
        Touches NO widgets. Returns the device bundle or raises (the
        caller surfaces the error)."""
        from ..connection import _check_properties
        from ..constants import resolve_coords
        new_backend = state.active_service.backend(backend_name)
        props, props_warn = _check_properties(new_backend, backend_name)
        new_coords, chip_label = resolve_coords(new_backend)
        return new_backend, props, props_warn, new_coords, chip_label

    def _connect_qpu2_paint(backend_name, fetched):
        """Main-thread ONLY: build the full secondary UI, apply any staged
        candidate, repaint, and mount the QPU-2 figure. Returns None on
        success or an error STRING on failure (so the auto path can fall
        back to staged)."""
        new_backend, props, props_warn, new_coords, chip_label = fetched
        try:
            # Late import to avoid the ui <-> ui.inter_qpu cycle that would
            # otherwise trigger at module load time.
            from . import build_mapper_ui

            s2 = SessionState()
            s2.active_service = state.active_service
            s2.available_backends = state.available_backends
            s2.active_ansatz = state.active_ansatz
            s2.current_colorscale = state.current_colorscale
            s2.active_vendor = 'IBM'
            s2.active_backend = new_backend
            s2.active_backend_name = backend_name
            s2.active_chip = chip_label
            s2.initial_props = props
            s2.props_warning = props_warn
            s2.active_coords = new_coords

            # Build the FULL secondary UI — feature-parity with QPU 1.
            # build_mapper_ui internally calls _load_device_graph which
            # populates s2.x_coord / qubit_ids / edges_list from props.
            full_ui_2, f2, w2 = build_mapper_ui(
                s2, is_secondary=True, return_widgets=True)

            # If this connect was staged from a scan pick for THIS backend,
            # paint the chosen candidate now — AFTER the build (which calls
            # clear_all_layouts()).
            pend = _pending_handoff[0]
            paint_err = None
            if pend is not None and pend[0] == backend_name:
                # Narrow try (was a blanket suppress): a candidate-paint failure
                # must never crash the connect, but it must NOT be hidden either.
                # Log it and surface it on the QPU-2 status below.
                try:
                    _apply_candidate_to_state(s2, pend[1])
                    from .. import plot as _plot
                    _plot.refresh_plot(f2, s2, w2)
                except Exception as _pe:            # noqa: BLE001 - reported below
                    paint_err = _pe
                    _log.warning("QPU-2 candidate paint failed for %s: %s",
                                 backend_name, _pe, exc_info=True)
                _pending_handoff[0] = None

            # Re-parent sidebar2 / zoom2_bar / f2 into the inter-QPU layout;
            # lbl_fig1 + fig_box2 stay orphan inside full_ui_2 (never displayed,
            # kept only so _dispose2 can unobserve them on toggle-off).
            sidebar2 = full_ui_2.children[0]
            fig_box2 = full_ui_2.children[1]
            zoom2_bar = fig_box2.children[1]

            _state2[0] = s2
            _fig2[0] = f2
            _w2[0] = w2                 # kept for same-backend repaint
            _sidebar2[0] = sidebar2
            _zoom2[0] = zoom2_bar
            # Reconnect: release the PREVIOUS QPU-2 widget tree before installing
            # the new disposer. Overwriting without disposing leaked the entire
            # prior UI + FigureWidget on every re-connect (observers keep the
            # tree alive in ipywidgets' handler lists).
            if _dispose2[0] is not None:
                try:
                    _dispose2[0]()
                except Exception:                     # noqa: BLE001 - logged, never fatal
                    _log.exception(
                        "QPU2 disposer raised on reconnect; observers may be leaked")
            _dispose2[0] = lambda: _walk_unobserve_all(full_ui_2)

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
            # Resync the full figure state into the synced widget traits so
            # the view created by the mount below renders the complete figure
            # regardless of which incremental messages the frontend applied.
            # Must precede the children assignment (see _resync_widget_state).
            _resync_widget_state(f2)
            qpu2_fig_box.children = [lbl_fig2, zoom2_bar, f2]
            # Match inter-QPU fig size via the QPU 2 sliders (their observers
            # propagate to f2.update_layout).
            w2.plot_w_slider.value = 1200
            w2.plot_h_slider.value = 1400

            # Escape every externally-sourced string before it hits the widget.
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
            if paint_err is not None:
                lbl_qpu2_status.value += (
                    f"<div style='font-size:13px;color:#e67e22;margin:2px 0;'>"
                    f"⚠ candidate paint failed: {esc(str(paint_err))}</div>")
            return None
        except Exception as e:
            lbl_qpu2_status.value = (
                f"<span style='font-size:15px;color:#e74c3c;'>"
                f"❌ {esc(str(e))}</span>"
            )
            return str(e)

    # ── Connect-QPU2 handler (manual button entry) ──────────────
    def _on_connect_qpu2(b):
        # Manual path: validate, then fetch + paint INLINE on the main
        # thread (the widget-callback context that has always worked live).
        # Returns None on success, or an error STRING on failure — so the
        # one-click auto path can tell whether to fall back to staged.
        backend_name = dd_backend_qpu2.value
        if not backend_name:
            lbl_qpu2_status.value = (
                "<span style='font-size:15px;color:#e74c3c;'>"
                "Select a backend</span>"
            )
            return 'Select a backend'
        if state.active_service is None:
            lbl_qpu2_status.value = (
                "<span style='font-size:15px;color:#e74c3c;'>"
                "No service connected</span>"
            )
            return 'No service connected'

        lbl_qpu2_status.value = (
            f"<span style='font-size:15px;color:#e67e22;'>"
            f"⏳ Connecting {esc(backend_name)}...</span>"
        )
        try:
            fetched = _connect_qpu2_fetch(backend_name)
        except Exception as e:                       # noqa: BLE001 - surfaced
            lbl_qpu2_status.value = (
                f"<span style='font-size:15px;color:#e74c3c;'>"
                f"❌ {esc(str(e))}</span>"
            )
            return str(e)
        return _connect_qpu2_paint(backend_name, fetched)

    btn_connect_qpu2.on_click(_on_connect_qpu2)

    # ── one-click auto-connect on a scan-candidate pick ───────
    def _staged_with_error(backend_name, candidate, err):
        """Failure → back to today's STAGED state (re-stage so a manual
        Connect re-applies the candidate) plus an esc()'d one-line error. The
        user loses nothing; one click recovers."""
        stage_remote_candidate(backend_name, candidate)
        lbl_qpu2_status.value = lbl_qpu2_status.value + (
            f"<div style='font-size:13px;color:#e74c3c;margin:2px 0;'>"
            f"❌ auto-connect failed: {esc(str(err))}</div>")

    def _auto_connect(backend_name, candidate):
        """The scan handoff hook (replaces bare staging): stage, then drive the
        SAME Connect-QPU2 handler automatically. Idempotent + race-guarded."""
        # Always stage first (reveal panel, prefill dropdown, preload candidate,
        # staged banner) — identical to the prior staging behaviour and the fallback.
        stage_remote_candidate(backend_name, candidate)

        # Same backend already connected as QPU-2 → skip the reconnect,
        # just apply the candidate and repaint. _state2[0] holds the connected
        # QPU-2 SessionState.
        s2 = _state2[0]
        if (s2 is not None
                and getattr(s2, 'active_backend_name', None) == backend_name):
            _pending_handoff[0] = None
            # Narrow try (was a blanket suppress): repaint failures are logged
            # and surfaced, not hidden — see the first-connect path.
            paint_err = None
            try:
                _apply_candidate_to_state(s2, candidate)
                if _fig2[0] is not None and _w2[0] is not None:
                    from .. import plot as _plot
                    _plot.refresh_plot(_fig2[0], s2, _w2[0])
            except Exception as _pe:                # noqa: BLE001 - reported below
                paint_err = _pe
                _log.warning("QPU-2 same-backend repaint failed for %s: %s",
                             backend_name, _pe, exc_info=True)
            lbl_qpu2_status.value = (
                f"<div style='font-size:14px;color:#1e8449;background:#eafaf1;"
                f"padding:4px 8px;border-radius:4px;border:1px solid #abebc6;"
                f"margin:3px 0;'>✅ Applied to already-connected "
                f"<b>{esc(str(backend_name))}</b>.</div>"
                + ("" if paint_err is None else
                   f"<div style='font-size:13px;color:#e67e22;margin:2px 0;'>"
                   f"⚠ candidate paint failed: {esc(str(paint_err))}</div>"))
            return

        # A connect already in flight → the staging above is enough;
        # don't launch a second connect.
        if _connect_in_flight[0]:
            return

        _connect_in_flight[0] = True
        btn_connect_qpu2.disabled = True          # disable the manual trigger too
        lbl_qpu2_status.value = (
            f"<div style='font-size:14px;color:#8a4b00;background:#fff3cd;"
            f"padding:6px 10px;border-radius:4px;border:1px solid #ffe08a;"
            f"margin:3px 0;font-weight:bold;'>📡 connecting "
            f"{esc(str(backend_name))}…</div>")

        def _finish():
            _connect_in_flight[0] = False
            btn_connect_qpu2.disabled = False

        # default_spawn_bg marshals on_done / on_error onto the kernel main
        # loop, so _paint runs on the MAIN thread even though the fetch ran on
        # a worker — the whole point of the split. The paint (build + apply +
        # refresh + mount) must never touch a FigureWidget off-thread.
        def _paint(fetched):      # runs on the main loop (marshalled)
            _finish()
            err = _connect_qpu2_paint(backend_name, fetched)
            if err:               # paint reported a failure string
                _staged_with_error(backend_name, candidate, err)

        def _error(exc):          # the worker fetch itself raised
            _finish()
            _staged_with_error(backend_name, candidate, exc)

        _spawn_bg(lambda: _connect_qpu2_fetch(backend_name),
                  on_done=_paint, on_error=_error)

    # Expose the label-refresh hook so ui.backend_switch can call it after a
    # backend switch (the switcher has no other handle on this closure).
    return refresh_qpu1_label
