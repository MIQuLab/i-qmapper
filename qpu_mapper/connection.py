"""
connection.py — Step-by-step authentication and backend selection.

Panel layout (4 numbered steps + final action):

    [1] Vendor:       [IBM Quantum ▼]                                ✓
    [2] Account name: [optional       ] [Load Account]      ⟳/✓/✗
                      ✅ Account loaded — N backends available.
    [3] Backend:      [(Select backend…) ▼]                          ✓
    [4] Ansatz:       [LUCJ / Chemistry] [General Ansatz]            ✓

                      [   🚀 Connect & Launch   ]

Each step shows a small status icon on the right (empty / ⟳ / ✓ / ✗).
`Connect & Launch` becomes clickable only when all four steps are ✓.
Changing an earlier step resets every later step's status to empty.

Previously the "Load Account" affordance and the post-load "✅ N backends
loaded" status were the SAME widget (a button that mutated its own
description). Users could not tell that re-clicking it was required after
changing the Account name field. The new layout splits them: an explicit
`Load Account` Button drives the action, a separate passive HTML widget
underneath displays the result.

Auto-load: on first display, if the default Qiskit account is detected
the panel auto-triggers Load Account once with an empty account name, so
the typical user lands on Step 2 already ✓ and only needs to pick a
backend before clicking Connect & Launch.
"""

import logging

import ipywidgets as widgets
from IPython.display import display, clear_output

from .constants import resolve_coords
from .html_safe import esc

_log = logging.getLogger(__name__)


# ── Auth/property helpers (unchanged behaviour) ──────────────────────────

def _check_properties(backend, backend_name):
    """Call backend.properties(). Returns (props_or_None, warning_or_None)."""
    try:
        props = backend.properties()
        if props is None:
            return None, (
                f"'{backend_name}' returned no calibration data (backend.properties() is None). "
                f"This usually means calibration is temporarily unavailable. "
                f"Design mode only — error map will be blank."
            )
        return props, None
    except Exception as e:
        return None, (
            f"Could not retrieve calibration data for '{backend_name}': {e}. "
            f"Design mode only — error map will be blank."
        )


def _try_auto_connect(account_name=None):
    """Connect to IBM Quantum via saved credentials.

    Returns (service, backend_names, error_message). On success the third
    element is None; on failure the first two are None/[] and the third is
    a human-readable string suitable for the status banner.
    """
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError:
        return None, [], (
            "qiskit_ibm_runtime is not installed. "
            "Install it with: pip install qiskit-ibm-runtime"
        )

    try:
        if account_name:
            svc = QiskitRuntimeService(name=account_name)
        else:
            svc = QiskitRuntimeService()
    except Exception as e:
        err_str = str(e).lower()
        if 'account' in err_str or 'credential' in err_str or 'token' in err_str:
            label = f" (account '{account_name}')" if account_name else ""
            return None, [], f"IBM Quantum account error{label}: {e}"
        return None, [], f"Could not initialize IBM Quantum service: {e}"

    try:
        backend_names = sorted(b.name for b in svc.backends())
        return svc, backend_names, None
    except Exception as e:
        err_str = str(e).lower()
        if 'auth' in err_str or '401' in err_str or 'token' in err_str:
            return None, [], f"Authentication failed (token may be expired): {e}"
        if 'timeout' in err_str or 'connect' in err_str:
            return None, [], f"Network error connecting to IBM Quantum: {e}"
        return None, [], f"Connected to IBM Quantum but failed to list backends: {e}"


# Vendor dispatch. Extend here for new providers: each loader has the
# same signature as `_try_auto_connect` and returns
# (service, backend_names, error_message_or_None).
_VENDOR_LOADERS = {
    'ibm': _try_auto_connect,
}


# ── Error banners (module-level so they're easy to swap) ─────────────────

_NO_ACCOUNT_MSG = (
    "<div style='font-size:11px; color:#e74c3c; background:#fdedec; "
    "padding:8px 10px; border-radius:5px; border:1px solid #f5b7b1; margin:4px 0;'>"
    "❌ <b>No IBM Quantum saved account found.</b><br>"
    "<span style='color:#555;'>Set up your credentials outside this tool using:</span><br>"
    "<code style='font-size:11px; background:#f8f9fa; padding:2px 4px; border-radius:3px;'>"
    "from qiskit_ibm_runtime import QiskitRuntimeService<br>"
    "QiskitRuntimeService.save_account(channel='ibm_quantum_platform', "
    "token='YOUR_TOKEN', instance='YOUR_CRN', overwrite=True)"
    "</code><br>"
    "<span style='font-size:10px; color:#888;'>"
    "The old <code>channel='ibm_quantum'</code> was removed by IBM after July 2025; "
    "the current channel is <code>'ibm_quantum_platform'</code> (alias <code>'ibm_cloud'</code>). "
    "<code>YOUR_CRN</code> is the IBM Cloud instance CRN from your IBM Cloud dashboard.<br>"
    "Pass <code>name='your-label'</code> as well if you want to save more than one "
    "account and switch between them from the Account field above.<br>"
    "Then restart the kernel and run the mapper again."
    "</span>"
    "</div>"
)


def _make_error_html(error_msg):
    """Build the error banner shown below Step 2 on Load Account failure."""
    err_lower = error_msg.lower()
    is_no_account = any(
        kw in err_lower for kw in ('account', 'credential', 'not installed'))
    if is_no_account:
        return _NO_ACCOUNT_MSG
    return (
        f"<div style='font-size:11px; color:#e74c3c; background:#fdedec; "
        f"padding:8px 10px; border-radius:5px; border:1px solid #f5b7b1; margin:4px 0;'>"
        f"❌ <b>Connection failed:</b> {esc(error_msg)}</div>"
    )


# ── Step status icon machinery ───────────────────────────────────────────

# Tiny HTML for the right-hand status column of each step row. Kept as a
# dict so the same vocabulary is used everywhere — adding a new state
# (e.g. 'warning') is a single entry, not a sprinkle of conditionals.
_STATUS_ICON_HTML = {
    'empty':    "",
    'progress': "<span style='color:#0f62fe;font-size:18px;font-weight:bold;' "
                "title='In progress'>⟳</span>",
    'ok':       "<span style='color:#198754;font-size:18px;font-weight:bold;' "
                "title='Done'>✓</span>",
    'error':    "<span style='color:#dc3545;font-size:18px;font-weight:bold;' "
                "title='Error'>✗</span>",
}


# ── Main UI ──────────────────────────────────────────────────────────────

def _make_connection_ui(on_connect_callback):
    """Display the 4-step connection panel.

    `on_connect_callback` is invoked exactly as before with kwargs:
        vendor, backend_name, backend, chip,
        props, props_warning, ansatz,
        service, available_backends.
    """
    out_status = widgets.Output()

    # ── Title (Carbon-style header) ──────────────────────────────────────
    title = widgets.HTML(
        "<div style='padding:16px 20px 12px 20px; "
        "background:#f4f4f4; border-bottom:1px solid #e0e0e0; "
        "margin:-24px -28px 0 -28px;'>"
        "<h2 style='margin:0; font-size:22px; font-weight:400; "
        "color:#161616; letter-spacing:-0.01em; "
        "font-family: \"IBM Plex Sans\", -apple-system, sans-serif;'>"
        "Interactive QPU Mapper</h2>"
        "<div style='font-size:12px; color:#525252; margin-top:4px;'>"
        "Connect to a backend and configure the layout editor.</div>"
        "</div>"
    )

    # ── Layout constants for the step rows ───────────────────────────────
    _LBL_LAYOUT    = widgets.Layout(width='160px')
    _STATUS_LAYOUT = widgets.Layout(
        width='30px', display='flex',
        justify_content='center', align_items='center',
    )
    _ROW_LAYOUT    = widgets.Layout(align_items='center', margin='14px 0')

    # ── Per-step status state ────────────────────────────────────────────
    # The dict drives both the icon column and the Connect & Launch
    # enable-gate. Mutating it goes through `_set_status` so the icon and
    # the button stay in sync.
    step_status = {1: 'empty', 2: 'empty', 3: 'empty', 4: 'empty'}
    status_w = {
        k: widgets.HTML(value='', layout=_STATUS_LAYOUT)
        for k in (1, 2, 3, 4)
    }

    def _step_label_html(num, text, active=True):
        """Render the '[N] Label:' cell on the left of a step row."""
        text_color = '#161616' if active else '#a8a8a8'
        num_color  = '#0f62fe' if active else '#a8a8a8'
        return (
            f"<div style='font-size:13px;font-weight:600;color:{text_color};'>"
            f"<span style='color:{num_color};'>[{num}]</span> {esc(text)}</div>"
        )

    def _set_status(step, state):
        step_status[step] = state
        status_w[step].value = _STATUS_ICON_HTML[state]
        _refresh_connect_btn()

    def _refresh_connect_btn():
        btn_connect.disabled = not all(s == 'ok' for s in step_status.values())

    # ── Step 1: Vendor ───────────────────────────────────────────────────
    dd_vendor = widgets.Dropdown(
        options=["IBM Quantum"],
        value="IBM Quantum",
        layout=widgets.Layout(flex='1'),
    )
    step1_lbl = widgets.HTML(_step_label_html(1, 'Vendor:'), layout=_LBL_LAYOUT)
    step1_row = widgets.HBox(
        [step1_lbl, dd_vendor, status_w[1]], layout=_ROW_LAYOUT,
    )

    # ── Step 2: Account ──────────────────────────────────────────────────
    txt_account = widgets.Text(
        value='', placeholder='optional', layout=widgets.Layout(flex='1'),
    )
    btn_load_account = widgets.Button(
        description='Load Account', button_style='info',
        layout=widgets.Layout(width='130px'),
    )
    step2_lbl = widgets.HTML(_step_label_html(2, 'Account name:'), layout=_LBL_LAYOUT)
    step2_row = widgets.HBox(
        [step2_lbl, txt_account, btn_load_account, status_w[2]],
        layout=_ROW_LAYOUT,
    )
    # Passive status message under the row (no longer clickable). Indented
    # to align under the text input column.
    lbl_account_status = widgets.HTML(
        "", layout=widgets.Layout(margin='0 0 0 168px'),
    )
    step2_block = widgets.VBox([step2_row, lbl_account_status])

    # ── Step 3: Backend ──────────────────────────────────────────────────
    dd_backend = widgets.Dropdown(
        options=[], disabled=True, layout=widgets.Layout(flex='1'),
    )
    step3_lbl = widgets.HTML(
        _step_label_html(3, 'Backend:', active=False),
        layout=_LBL_LAYOUT,
    )
    step3_row = widgets.HBox(
        [step3_lbl, dd_backend, status_w[3]], layout=_ROW_LAYOUT,
    )

    # ── Step 4: Ansatz ───────────────────────────────────────────────────
    dd_ansatz = widgets.ToggleButtons(
        options=[("LUCJ / Chemistry", "lucj"), ("General Ansatz", "general")],
        value="lucj",
        layout=widgets.Layout(flex='1'),
        style={"button_width": "150px"},
    )
    step4_lbl = widgets.HTML(_step_label_html(4, 'Ansatz:'), layout=_LBL_LAYOUT)
    step4_row = widgets.HBox(
        [step4_lbl, dd_ansatz, status_w[4]], layout=_ROW_LAYOUT,
    )

    # ── Connect & Launch (bottom) ────────────────────────────────────────
    btn_connect = widgets.Button(
        description='🚀 Connect & Launch', button_style='success',
        disabled=True,
        layout=widgets.Layout(width='100%', height='42px', margin='18px 0 0 0'),
    )

    # ── Mutable state captured by the closures below ─────────────────────
    _state = {'service': None}

    # ── Step 1 & 4 start at 'ok' because Dropdown/ToggleButtons already
    #    carry a default value. Steps 2 and 3 stay 'empty' until the user
    #    (or the init auto-trigger) drives them.
    _set_status(1, 'ok')
    _set_status(4, 'ok')

    # ── Step transitions (helpers) ───────────────────────────────────────
    def _activate_step3():
        step3_lbl.value = _step_label_html(3, 'Backend:', active=True)
        dd_backend.disabled = False

    def _deactivate_step3():
        step3_lbl.value = _step_label_html(3, 'Backend:', active=False)
        dd_backend.disabled = True
        dd_backend.options = []

    def _reset_steps_after_vendor():
        """Vendor change invalidates the account session and backend list.

        Step 4 (ansatz) is not touched — the user's ansatz selection
        survives a vendor switch.
        """
        _set_status(2, 'empty')
        _set_status(3, 'empty')
        lbl_account_status.value = ''
        _state['service'] = None
        _deactivate_step3()

    # ── Step 1 observer ──────────────────────────────────────────────────
    def _on_vendor_change(change):
        # Vendor always has a value once the dropdown is constructed, so
        # step 1 stays 'ok'. The new value invalidates step 2/3 state.
        _set_status(1, 'ok')
        _reset_steps_after_vendor()
        txt_account.disabled = False
        txt_account.placeholder = 'optional'
    dd_vendor.observe(_on_vendor_change, names='value')

    # ── Step 2 logic: Load Account ───────────────────────────────────────
    def _on_load_account_click(b, *, _is_auto=False):
        """Triggered by the Load Account button (or the init auto-trigger).

        `_is_auto=True` means the panel called this itself on first display
        as a silent probe — failures are NOT surfaced as an error banner
        in that case (only explicit user clicks surface errors).
        """
        vendor = dd_vendor.value
        btn_load_account.disabled = True
        _set_status(2, 'progress')
        lbl_account_status.value = ''

        # ── IBM: saved-account lookup ────────────────────────────────────
        # Empty Account text → Qiskit default account. Typed name →
        # named saved account (allows users with multiple credentials
        # to switch without editing their Qiskit config). Whitespace is
        # trimmed so accidental leading/trailing spaces don't break the
        # lookup.
        typed = (txt_account.value or '').strip()
        selected = typed if typed else None
        vendor_key = str(vendor).strip().lower().split()[0] if vendor else ''
        loader = _VENDOR_LOADERS.get(vendor_key)
        if loader is None:
            _set_status(2, 'error')
            if not _is_auto:
                lbl_account_status.value = _make_error_html(
                    f"Vendor '{vendor}' is not supported yet. "
                    f"Currently available: IBM Quantum."
                )
            btn_load_account.disabled = False
            return
        service, backend_names, error = loader(account_name=selected)

        if error is not None:
            _set_status(2, 'error')
            if _is_auto:
                # Init probe: stay silent so the panel doesn't open with
                # a scary red error the user never asked for.
                lbl_account_status.value = ''
            else:
                lbl_account_status.value = _make_error_html(error)
            btn_load_account.disabled = False
            return

        _state['service'] = service
        dd_backend.options = backend_names
        _activate_step3()
        # When backend_names is non-empty, ipywidgets auto-selects the
        # first option, so dd_backend.value becomes truthy. We still
        # gate step 3 on the actual value to handle the (degenerate)
        # zero-backends case.
        if dd_backend.value:
            _set_status(3, 'ok')
        else:
            _set_status(3, 'empty')
        _set_status(2, 'ok')
        lbl_account_status.value = (
            f"<div style='font-size:12px;color:#198754;margin:4px 0;'>"
            f"✅ Account loaded — {len(backend_names)} backends available.</div>"
        )
        btn_load_account.disabled = False
    btn_load_account.on_click(_on_load_account_click)

    # ── Step 2.5: typing in Account text invalidates a prior load ────────
    # If the user typed a new name AFTER successfully loading, the
    # currently-cached service is for a different account. Reset step 2
    # to 'empty' so they have to click Load Account again — same for
    # step 3, which depends on the service.
    def _on_account_text_change(change):
        if step_status[2] != 'empty':
            _set_status(2, 'empty')
            _set_status(3, 'empty')
            lbl_account_status.value = ''
            _state['service'] = None
            _deactivate_step3()
    txt_account.observe(_on_account_text_change, names='value')

    # ── Step 3 observer ──────────────────────────────────────────────────
    def _on_backend_change(change):
        _set_status(3, 'ok' if change['new'] else 'empty')
    dd_backend.observe(_on_backend_change, names='value')

    # ── Step 4 observer (ToggleButtons always has a value) ───────────────
    def _on_ansatz_change(change):
        _set_status(4, 'ok')
    dd_ansatz.observe(_on_ansatz_change, names='value')

    # ── Connect & Launch ─────────────────────────────────────────────────
    def on_connect(b):
        btn_connect.disabled    = True
        btn_connect.description = '⏳ Loading calibration...'
        with out_status:
            clear_output()

        vendor       = dd_vendor.value
        backend_name = dd_backend.value
        ansatz       = dd_ansatz.value

        # ── IBM (round-trips IBM Runtime — off the UI thread) ───────────
        # service.backend(name) and backend.properties() each take 0.5–5 s
        # over the network. Inline on the kernel thread would freeze the
        # UI between click and mapper display — the "⏳ Loading…" label
        # never paints because ipywidgets hasn't flushed. default_spawn_bg
        # marshals on_done back to the main loop so display(full_ui)
        # inside on_connect_callback still runs on the kernel thread.
        service = _state['service']

        def _work():
            backend = service.backend(backend_name)
            _, chip = resolve_coords(backend)
            props, props_warning = _check_properties(backend, backend_name)
            return backend, chip, props, props_warning

        def _done(result):
            backend, chip, props, props_warning = result
            with out_status:
                clear_output()
                if props_warning:
                    print(f'[Calibration]  {props_warning}')
                else:
                    print(f"'{backend_name}' — chip: {chip} — calibration data loaded.")
                print('Launching mapper...')
            on_connect_callback(
                vendor=vendor, backend_name=backend_name,
                backend=backend, chip=chip,
                props=props, props_warning=props_warning,
                ansatz=ansatz,
                service=service,
                available_backends=list(dd_backend.options),
            )

        def _err(exc):
            _log.exception('Connect & launch failed')
            btn_connect.disabled    = False
            btn_connect.description = '🚀 Connect & Launch'
            with out_status:
                print(f'Error: {type(exc).__name__} (see kernel logs for details)')

        from .async_util import default_spawn_bg
        default_spawn_bg(_work, on_done=_done, on_error=_err)

    btn_connect.on_click(on_connect)

    # ── Assemble & display ───────────────────────────────────────────────
    ui = widgets.VBox([
        title,
        step1_row,
        step2_block,
        step3_row,
        step4_row,
        btn_connect,
        out_status,
    ], layout=widgets.Layout(
        padding='24px 28px 24px 28px',
        border_left='4px solid #0f62fe',
        border_top='1px solid #e0e0e0',
        border_right='1px solid #e0e0e0',
        border_bottom='1px solid #e0e0e0',
        width='620px',
    ))

    # ── Auto-trigger Load Account on init for the typical IBM user ───────
    # If the default Qiskit account is detected, the panel opens with
    # Step 2 already ✓; otherwise it opens silent and the user clicks
    # Load Account themselves to surface the actual error.
    if dd_vendor.value == 'IBM Quantum':
        _on_load_account_click(None, _is_auto=True)

    display(ui)
