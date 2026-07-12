"""
ui.backend_switch.py — backend dropdown wiring (callback + initial population).

Owns the user flow that fires when the backend dropdown changes:

  1. Save the existing layout (only when the user has armed the
     `keep layout` checkbox) so it can be restored if the new backend's
     topology is compatible.
  2. Refresh `state` with the new backend / chip / props / coords.
  3. Trigger `_load_device_graph(force=True)` so the figure rebuilds
     with the new device graph + calibration. `force=True` is needed
     because this routine itself holds the `_switching_backend` guard,
     which `_load_device_graph` would otherwise refuse to bypass.
  4. If the layout snapshot is compatible (same chip family, same
     qubit count) re-apply it; otherwise log a clear "discarded"
     warning and continue with empty layouts.
  5. Always restore the dropdown to a sensible state in `finally`.

Initial population (so loading the UI doesn't fire a spurious switch
event when the dropdown's options/value are first set) is done with
the standard ipywidgets `unobserve` → assign → `observe` dance and is
also handled here for symmetry.

Public API:
  setup_backend_switch(ctx) -> on_backend_switch_handler

The handler is returned so callers (e.g. tests) can invoke it
explicitly. `ctx.switching_backend` is shared with
`_load_device_graph` — both routines need to see the same flag for
the reentrancy guard to work; we don't own it. The flag is a single-
element list (`[False]`) used as a kernel-main-loop reentrancy guard,
not a thread synchronisation primitive — `threading.Event`'s blocking
semantics would be wrong here.
"""
import contextlib
import logging

from ..html_safe import esc

_log = logging.getLogger(__name__)

# Placeholder string used by the backend dropdown before any list has
# arrived from the service. Centralised here so the magic-string equality
# check stays in sync with the value the dropdown is seeded with.
LOADING_PLACEHOLDER = '(loading...)'

# Whitelist of layout-data keys we round-trip through the
# save-snapshot / restore-snapshot path. Defined at module scope so the
# snapshot writer and the restore reader cannot drift apart silently
# (the original code listed the same six names in two places — adding a
# new field meant editing both).
_LAYOUT_SNAPSHOT_KEYS = ('alpha', 'beta', 'other', 'qubits',
                         'ancilla', 'styles', 'name')


def _snapshot_layout_entry(d: dict) -> dict:
    """Return a shallow-copy-per-key snapshot of one circuit's layout dict.

    Lists and dicts are copied so the snapshot is decoupled from later
    mutations on `state.layouts_data`. Scalars are kept as-is.
    Unknown keys (anything outside _LAYOUT_SNAPSHOT_KEYS) are dropped.
    """
    snap = {}
    for k in _LAYOUT_SNAPSHOT_KEYS:
        if k not in d:
            continue
        v = d[k]
        if isinstance(v, dict):
            snap[k] = v.copy()
        elif isinstance(v, list):
            snap[k] = list(v)
        else:
            snap[k] = v
    return snap


def _classify_switch_error(exc: BaseException, backend_name: str) -> str:
    """Map an exception from the backend-switch flow to a short hint.

    The IBM runtime client doesn't expose distinct exception types for
    "backend gone" vs "auth expired" vs "transient network", so we
    heuristically inspect the error text. The match is best-effort and
    falls back to the raw exception message — it's strictly additive,
    no diagnosis is lost when the hint doesn't match.
    """
    msg = (str(exc) or '').lower()
    if 'not found' in msg or 'unknown backend' in msg or '404' in msg:
        return (f"Backend '{backend_name}' is no longer available "
                f"(may have been retired or hidden by IBM).")
    if 'auth' in msg or '401' in msg or '403' in msg or 'token' in msg:
        return ("Authentication failed (token may have expired). "
                "Re-save your QiskitRuntimeService account and restart the kernel.")
    if 'timeout' in msg or 'timed out' in msg or 'connection' in msg:
        return f"Network error reaching IBM Quantum: {exc}"
    return f"Switch failed: {exc}"


def setup_backend_switch(ctx):
    """Wire dd_backend_switch.observe + populate its initial options.

    Required ctx fields:
      state                 — SessionState (mutated on switch)
      dd_backend_switch     — Dropdown widget
      lbl_switch_status     — HTML status banner (shows progress / errors)
      lbl_calibration_date  — HTML widget cleared on switch
      lbl_props_warning     — HTML widget cleared on switch
      chk_keep_layout       — Checkbox: when True, snapshot+restore layouts
                              if the new chip is compatible
      switching_backend     — single-element list [False] used as guard,
                              SHARED with _load_device_graph
      load_device_graph     — callable(force=bool) that refreshes the
                              figure for the current `state.active_backend`
      refresh, get_export   — UI redraw helpers
      out                   — Output widget (used for traceback dump
                              when the switch fails)

    Returns the on_backend_switch handler in case the caller needs it
    for testing or programmatic invocation.
    """
    state = ctx.state
    dd_backend_switch = ctx.dd_backend_switch
    lbl_switch_status = ctx.lbl_switch_status
    lbl_calibration_date = ctx.lbl_calibration_date
    lbl_props_warning = ctx.lbl_props_warning
    chk_keep_layout = ctx.chk_keep_layout
    switching_backend = ctx.switching_backend  # list[bool] shared w/ load_device_graph
    load_device_graph = ctx.load_device_graph
    refresh = ctx.refresh
    get_export = ctx.get_export
    # Optional: single-element cell holding ui.inter_qpu's refresh hook (or
    # [None] until inter-QPU is wired). Absent entirely in callers/tests that
    # don't build the inter-QPU panel — treat as "no hook".
    refresh_qpu1_label_cell = getattr(ctx, 'refresh_qpu1_label', None)

    def _refresh_qpu1_label():
        """Best-effort refresh of the QPU-1 switcher label after a switch.

        A label refresh must NEVER break a backend switch, so failures are
        swallowed (logged at debug). No-op when the inter-QPU switcher was
        never opened (hook is None).
        """
        hook = (refresh_qpu1_label_cell[0]
                if refresh_qpu1_label_cell else None)
        if hook is None:
            return
        with contextlib.suppress(Exception):
            _log.debug("Refreshing QPU-1 toggle label after backend switch")
            hook()

    def _on_backend_switch(change):
        new_name = change['new']
        if (not new_name
                or new_name == state.active_backend_name
                or new_name == LOADING_PLACEHOLDER):
            return

        # Reentrancy guard: if a switch is already in progress, ignore.
        if switching_backend[0]:
            return
        switching_backend[0] = True

        # Save old chip type for layout compatibility check below.
        old_chip = state.active_chip or ''
        old_n_qubits = len(state.qubit_ids) if state.qubit_ids else 0
        # Snapshot the current layout iff the user wants it preserved.
        # _snapshot_layout_entry handles the per-key shallow copy so the
        # restored layout is decoupled from later in-place mutations.
        saved_layouts = None
        if chk_keep_layout.value:
            try:
                saved_layouts = {
                    'layouts_data': {
                        idx: _snapshot_layout_entry(d)
                        for idx, d in state.layouts_data.items()
                    },
                    'chip': old_chip,
                    'n_qubits': old_n_qubits,
                }
            except Exception as exc:
                _log.warning("Layout snapshot failed: %s", exc)
                saved_layouts = None

        lbl_switch_status.value = (
            f"<div style='font-size:15px;color:#e67e22;background:#fef9e7;"
            f"padding:4px 8px;border-radius:4px;border:1px solid #f5cba7;"
            f"margin:3px 0;'>"
            f"⏳ Switching to <b>{esc(new_name)}</b>... "
            f"retrieving calibration data</div>"
        )
        lbl_calibration_date.value = ""
        lbl_props_warning.value = ""
        # Disable the dropdown during the switch to prevent double-clicks.
        dd_backend_switch.disabled = True
        try:
            # Lazy imports: `connection` and `constants` both pull in
            # qiskit-ibm-runtime / numpy on first import. Keeping them
            # local to the handler also avoids the import-time cost when
            # the module is loaded but the user never switches backends.
            # No real circular-import risk here — `connection.py` itself
            # only imports `constants` and `html_safe`.
            from ..connection import _check_properties
            from ..constants import resolve_coords

            new_backend = state.active_service.backend(new_name)
            props, props_warn = _check_properties(new_backend, new_name)

            # Resolve coords dynamically from the backend.
            new_coords, chip_label = resolve_coords(new_backend)

            # Snapshot for rollback: if _load_device_graph fails below, the
            # except restores these so state doesn't point at a backend whose
            # graph never loaded.
            _prev_refs = (
                state.active_backend, state.active_backend_name,
                state.active_chip, state.initial_props,
                state.props_warning, state.active_coords,
            )
            # ── CRITICAL: update state BEFORE _load_device_graph ─────
            state.active_backend = new_backend
            state.active_backend_name = new_name
            state.active_chip = chip_label
            state.initial_props = props
            state.props_warning = props_warn
            state.active_coords = new_coords

            # Backend name changed — refresh the inter-QPU switcher's QPU-1
            # label so it doesn't keep showing the previous backend.
            _refresh_qpu1_label()

            lbl_switch_status.value = (
                f"<div style='font-size:15px;color:#2471a3;"
                f"background:#eaf2f8;padding:4px 8px;border-radius:4px;"
                f"border:1px solid #aed6f1;margin:3px 0;'>"
                f"⏳ Building error map for <b>{esc(new_name)}</b>...</div>"
            )

            # force=True: we already hold switching_backend, so the
            # default guard inside _load_device_graph would (correctly)
            # reject this call.
            load_device_graph(force=True)

            # ── Layout preservation ──────────────────────────────────
            layout_msg = ""
            if saved_layouts is not None:
                new_n_qubits = len(state.qubit_ids) if state.qubit_ids else 0
                # E.g. "Heron r3" → "Heron"; layouts only restore on the
                # same chip family. Naming convention "<Family> r<rev>"
                # is enforced by constants.resolve_coords — adding a new
                # family with multi-word names (e.g. "Heavy Hex r2")
                # would require updating this split.
                old_chip_type = saved_layouts['chip'].split(' ')[0]
                new_chip_type = chip_label.split(' ')[0]

                if old_chip_type != new_chip_type:
                    layout_msg = (
                        f"<br>⚠️ Layout discarded: <b>{esc(old_chip_type)}</b> "
                        f"layout is incompatible with <b>{esc(new_chip_type)}</b> "
                        f"topology."
                    )
                elif saved_layouts['n_qubits'] != new_n_qubits:
                    layout_msg = (
                        f"<br>⚠️ Layout discarded: qubit count changed "
                        f"({saved_layouts['n_qubits']} → {new_n_qubits})."
                    )
                else:
                    # Compatible — restore the saved layout dicts in place.
                    # The whitelist is centralised in _LAYOUT_SNAPSHOT_KEYS
                    # (used by _snapshot_layout_entry above) so the writer
                    # and reader cannot drift apart anymore.
                    for idx, saved in saved_layouts['layouts_data'].items():
                        if idx in state.layouts_data:
                            d = state.layouts_data[idx]
                            for key in _LAYOUT_SNAPSHOT_KEYS:
                                if key in saved:
                                    d[key] = saved[key]
                    refresh()
                    get_export()
                    layout_msg = "<br>🔒 Layout preserved from previous backend."

            msg = (f"✅ Switched to <b>{esc(new_name)}</b> "
                   f"({esc(chip_label)}){layout_msg}")
            if props_warn:
                msg += f"<br>⚠️ {esc(props_warn)}"
            lbl_switch_status.value = (
                f"<div style='font-size:15px;color:#1e8449;"
                f"background:#eafaf1;padding:4px 8px;border-radius:4px;"
                f"border:1px solid #abebc6;margin:3px 0;'>"
                f"{msg}</div>"
            )
        except Exception as e:
            _log.exception("Backend switch failed")
            # Roll state back to the previous backend (the graph for the new
            # one failed to load — leaving the new refs in place would make
            # every later read inconsistent with the dropdown). Guarded
            # because failures earlier in the try (e.g. fetching the backend
            # object) happen before `_prev_refs` is bound — those leave
            # state untouched, so there is nothing to roll back.
            if '_prev_refs' in locals():
                (state.active_backend, state.active_backend_name,
                 state.active_chip, state.initial_props,
                 state.props_warning, state.active_coords) = _prev_refs
                # State rolled back to the old backend name — refresh the
                # switcher label to match, undoing any success-path relabel.
                _refresh_qpu1_label()
            hint = _classify_switch_error(e, new_name)
            lbl_switch_status.value = (
                f"<div style='font-size:15px;color:#e74c3c;"
                f"background:#fdedec;padding:4px 8px;border-radius:4px;"
                f"border:1px solid #f5b7b1;margin:3px 0;'>"
                f"❌ {esc(hint)}</div>"
            )
            # Roll the dropdown back to the previous selection so the
            # user isn't stuck on a half-switched value.
            dd_backend_switch.value = state.active_backend_name
        finally:
            switching_backend[0] = False
            dd_backend_switch.disabled = (state.active_service is None)

    # ── Initial population ─────────────────────────────────────
    # Set options/value FIRST (no observer yet — there's nothing to
    # silence), then register the observer ONCE at the end.
    #
    # The original ui.py used the `unobserve → assign → observe` dance
    # because the observer had already been registered earlier in the
    # same function. Now that registration only happens here, the
    # unobserve call would raise ValueError: list.remove(x): x not in
    # list (ipywidgets stores handlers in a list and unobserve calls
    # list.remove, which fails when the handler was never added).
    if state.available_backends:
        dd_backend_switch.options = state.available_backends
        dd_backend_switch.value = (
            state.active_backend_name
            if state.active_backend_name in state.available_backends
            else state.available_backends[0]
        )
        dd_backend_switch.disabled = (state.active_service is None)
    else:
        dd_backend_switch.options = [state.active_backend_name or '(unknown)']
        dd_backend_switch.value = state.active_backend_name or '(unknown)'
        dd_backend_switch.disabled = True
    dd_backend_switch.observe(_on_backend_switch, names='value')

    return _on_backend_switch
