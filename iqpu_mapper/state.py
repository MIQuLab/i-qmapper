"""
state.py — Centralised mutable state for I-QMapper.

One SessionState = one QPU; inter-QPU mode creates multiple instances.
No swap magic, no slot system — each QPU is fully independent.

v5: time/delta/reference fields are dedicated SessionState.__init__
attributes (previously added lazily by
ui.time_panel.install_state_defaults, which caused AttributeError when
the history slider was used before entering error_inspect). Session
version bumped to 5; v3 and v4 sessions still load.
"""
import numpy as np
from .constants import MAX_CIRCUITS, DEFAULT_CIRCUIT_COLORS


def _default_group_styles(circuit_defaults):
    """Build per-group rendering presets from circuit defaults."""
    return {
        'alpha':   {'fill': circuit_defaults['fill'],       'line': circuit_defaults['line'],
                    'opacity': 1.0, 'width': 2, 'text_color': '#000000'},
        'beta':    {'fill': circuit_defaults['fill_beta'],   'line': circuit_defaults['line'],
                    'opacity': 1.0, 'width': 2, 'text_color': '#000000'},
        'other':   {'fill': circuit_defaults.get('fill_other', '#F39C12'), 'line': circuit_defaults['line'],
                    'opacity': 1.0, 'width': 2, 'text_color': '#000000'},
        'qubits':  {'fill': circuit_defaults['fill'],       'line': circuit_defaults['line'],
                    'opacity': 1.0, 'width': 2, 'text_color': '#000000'},
        'ancilla': {'fill': circuit_defaults.get('fill_anc', '#95A5A6'), 'line': '#000000',
                    'opacity': 1.0, 'width': 2, 'text_color': '#000000'},
    }


def _default_legend_config():
    """Global plot-legend config. One entry per role (no per-circuit copy:
    labels like "alpha" are conceptual; the dot *colour* comes from the
    displayed circuit's group_styles, the *label* and *show* flag come
    from here)."""
    return {
        'enabled': False,                         # master on/off (off by default)
        'roles': {
            'alpha':      {'show': True,  'label': 'alpha'},
            'beta':       {'show': True,  'label': 'beta'},
            'other':      {'show': True,  'label': 'other'},
            'ancilla':    {'show': True,  'label': 'ancilla'},
            'qubits':     {'show': True,  'label': 'selected'},    # general mode
            'unselected': {'show': False, 'label': 'unselected'},  # opt-in, both modes
        },
    }


def _make_layouts():
    """Create fresh layouts_data dict."""
    return {
        i: {
            'name': f'C{i+1}',
            'alpha': [], 'beta': [], 'other': [],
            'qubits': [], 'ancilla': [],
            'styles': {},
            'color': DEFAULT_CIRCUIT_COLORS[i]['fill'],
            'defaults': DEFAULT_CIRCUIT_COLORS[i].copy(),
            'group_styles': _default_group_styles(DEFAULT_CIRCUIT_COLORS[i]),
        }
        for i in range(MAX_CIRCUITS)
    }


# Keys included in undo/redo snapshots
_SNAPSHOT_KEYS = ('alpha', 'beta', 'other', 'qubits', 'ancilla',
                  'styles', 'name', 'color', 'group_styles')


class SessionState:
    """Holds all mutable state for ONE QPU. Inter-QPU = multiple instances."""

    def __init__(self):
        # ── Backend connection ────────────────────────────────────
        self.active_backend       = None
        self.active_backend_name  = None
        self.active_chip          = None
        self.active_service       = None
        self.available_backends   = []
        self.props_warning        = None
        self.initial_props        = None
        self.active_ansatz        = 'lucj'
        self.active_vendor        = ''   # 'IBM'
        self.active_coords        = None

        # ── Device graph data ────────────────────────────────────
        self.x_coord        = []
        self.y_coord        = []
        self.edges_list     = []
        self.qubit_ids      = []
        self.readout_errors = []
        self.gate_errors    = []

        # ── Multi-property calibration data ──────────────────────
        self.qubit_props_data = {}
        self.gate_props_data  = {}
        self.calibration_data_label = None

        # Default colorscale. Cividis (perceptually uniform, colorblind-safe)
        # was the previous default — user prefers the IBM palette
        # aesthetically; Cividis is still in the dropdown for those who need it.
        self.current_colorscale = 'Blue→Purple (IBM)'

        # ── Custom palette colors ────────────────────────────────
        self.custom_colors = []

        # ── Legend config ────────────────────────────────────────
        self.legend_config = _default_legend_config()

        # ── Layout data per circuit ──────────────────────────────
        self.layouts_data = _make_layouts()

        # ── Zoom state ───────────────────────────────────────────
        self.zoom = {'x0': None, 'x1': None, 'y0': None, 'y1': None}

        # ── Undo / redo history ──────────────────────────────────
        self._undo_stack = []
        self._redo_stack = []
        self._max_history = 50

        # ── Calibration history ──────────────────────────────────
        self.cal_history = {}
        self.cal_history_dates = []
        self.cal_history_loading = False

        # ── PNG export settings ──────────────────────────────────
        self.png_margin = 40
        self.png_crop_to_selection = False

        # ── Delta error mode ─────────────────────────────────────
        # Two flags with INTENTIONALLY different semantics:
        #   delta_enabled — persistent USER INTENT ("render delta whenever
        #     ref + current snapshot are both available"). Persisted in
        #     the session JSON; set by the ui.time_panel checkbox.
        #   delta_mode    — transient RUNTIME flag (delta_qubit_props /
        #     delta_gate_props are populated and the plot should colour
        #     with the divergent scale right now). Set atomically with
        #     the delta_*_props by anything that recomputes the diff
        #     (ui.time_panel._apply_current_delta,
        #     ui.history._on_slider_change). NOT persisted.
        # Invariant: delta_mode True ⇒ delta_qubit_props non-empty.
        # Code that invalidates the diff (e.g. apply_history_snapshot
        # on a calibration change) must clear delta_mode; the caller
        # may then recompute if delta_enabled is still True.
        self.delta_mode = False
        self.delta_qubit_props = {}
        self.delta_gate_props = {}
        self.delta_date_a = None
        self.delta_date_b = None

        # ── Time navigation + delta reference ────────────────────
        # Read by ui.history, ui.time_panel, ui_calibration. Previously
        # added lazily by install_state_defaults(state), causing
        # AttributeError if history was fetched before entering
        # error_inspect — now always present.
        self.time_mode            = 'live'
        self.delta_enabled        = False
        self.delta_ref_mode       = 'specific'
        self.point_timestamp      = None
        self.intraday_date        = None
        self.intraday_timestamps  = []
        self.intraday_index       = 0
        self.range_start          = None
        self.range_end            = None
        self.range_index          = 0
        # Single source of truth for the "A side" of delta comparisons
        # (ui_calibration.py + ui.time_panel.py both read/write).
        self.ref_timestamp        = None
        self.ref_qubit_props      = None
        self.ref_gate_props       = None
        # Saved live calibration (restored when switching back to Live mode)
        self._live_qubit_props    = None
        self._live_gate_props     = None
        self._live_cal_label      = None

    # ── Layout helpers ───────────────────────────────────────────

    def clear_all_layouts(self):
        for i in range(MAX_CIRCUITS):
            d = self.layouts_data[i]
            d['alpha'] = []; d['beta'] = []; d['other'] = []
            d['qubits'] = []; d['ancilla'] = []; d['styles'] = {}

    def clear_circuit(self, idx):
        if idx not in self.layouts_data:
            return
        d = self.layouts_data[idx]
        d['alpha'] = []; d['beta'] = []; d['other'] = []
        d['qubits'] = []; d['ancilla'] = []; d['styles'] = {}

    # ── Undo / redo ──────────────────────────────────────────────

    def _snapshot_layouts(self):
        """Create a deep copy of layout data for undo/redo."""
        snapshot = {}
        for i, d in self.layouts_data.items():
            snap = {}
            for k in _SNAPSHOT_KEYS:
                v = d.get(k)
                if v is None:
                    continue
                if isinstance(v, list):
                    snap[k] = list(v)
                elif isinstance(v, dict) and k == 'styles':
                    snap[k] = {qk: dict(qv) for qk, qv in v.items()}
                elif isinstance(v, dict) and k == 'group_styles':
                    snap[k] = {gk: dict(gv) for gk, gv in v.items()}
                else:
                    snap[k] = v
            snapshot[i] = snap
        return snapshot

    def _restore_snapshot(self, snapshot):
        """Apply a previously saved snapshot to layouts_data."""
        for i, snap in snapshot.items():
            if i in self.layouts_data:
                for k, v in snap.items():
                    self.layouts_data[i][k] = v

    def push_undo(self):
        self._undo_stack.append(self._snapshot_layouts())
        if len(self._undo_stack) > self._max_history:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def undo(self):
        if not self._undo_stack:
            return False
        self._redo_stack.append(self._snapshot_layouts())
        self._restore_snapshot(self._undo_stack.pop())
        return True

    def redo(self):
        if not self._redo_stack:
            return False
        self._undo_stack.append(self._snapshot_layouts())
        self._restore_snapshot(self._redo_stack.pop())
        return True

    @property
    def has_undo(self):
        return len(self._undo_stack) > 0

    @property
    def has_redo(self):
        return len(self._redo_stack) > 0

    # ── Zoom helpers ─────────────────────────────────────────────

    def reset_zoom(self):
        self.zoom = {'x0': None, 'x1': None, 'y0': None, 'y1': None}

    def get_zoom_range(self):
        if self.zoom['x0'] is not None:
            return (self.zoom['x0'], self.zoom['x1'],
                    self.zoom['y0'], self.zoom['y1'])
        if len(self.x_coord) > 0:
            xmin, xmax = float(min(self.x_coord)), float(max(self.x_coord))
            ymin, ymax = float(min(self.y_coord)), float(max(self.y_coord))
            px = max((xmax - xmin) * 0.15, 1.0)
            py = max((ymax - ymin) * 0.15, 1.0)
            return xmin - px, xmax + px, ymin - py, ymax + py
        return None

    def set_zoom(self, x0, x1, y0, y1):
        self.zoom = {'x0': x0, 'x1': x1, 'y0': y0, 'y1': y1}

    # ── Session save / restore ───────────────────────────────────

    def to_session_dict(self, widget_values=None):
        layouts_ser = {}
        for i, d in self.layouts_data.items():
            layouts_ser[str(i)] = {
                'name': d['name'],
                'alpha': list(d['alpha']),
                'beta': list(d['beta']),
                'other': list(d.get('other', [])),
                'qubits': list(d['qubits']),
                'ancilla': list(d['ancilla']),
                'styles': {k: dict(v) for k, v in d['styles'].items()},
                'color': d.get('color', ''),
                'group_styles': {gk: dict(gv) for gk, gv in d.get('group_styles', {}).items()},
            }

        # v5: persist the reference calibration so delta mode survives
        # reload. qubit_props / gate_props are numpy arrays → list for JSON.
        ref_qp = {}
        if self.ref_qubit_props:
            ref_qp = {k: (v.tolist() if hasattr(v, 'tolist') else list(v))
                      for k, v in self.ref_qubit_props.items()}
        ref_gp = {}
        if self.ref_gate_props:
            ref_gp = {k: (v.tolist() if hasattr(v, 'tolist') else list(v))
                      for k, v in self.ref_gate_props.items()}

        session = {
            '_version': 5,
            'backend_name': self.active_backend_name,
            'chip': self.active_chip,
            'ansatz': self.active_ansatz,
            'colorscale': self.current_colorscale,
            'custom_colors': list(self.custom_colors),
            'zoom': dict(self.zoom),
            'layouts': layouts_ser,
            'qubit_ids': list(self.qubit_ids),
            'edges_list': [list(e) for e in self.edges_list] if self.edges_list else [],
            'x_coord': [float(x) for x in self.x_coord] if len(self.x_coord) > 0 else [],
            'y_coord': [float(y) for y in self.y_coord] if len(self.y_coord) > 0 else [],
            'qubit_props_data': {k: v.tolist() if hasattr(v, 'tolist') else list(v)
                                 for k, v in self.qubit_props_data.items()},
            'gate_props_data': {k: v.tolist() if hasattr(v, 'tolist') else list(v)
                                for k, v in self.gate_props_data.items()},
            'calibration_data_label': self.calibration_data_label,
            # Reference calibration (delta baseline)
            'ref_timestamp': self.ref_timestamp,
            'ref_qubit_props': ref_qp,
            'ref_gate_props': ref_gp,
            'delta_enabled': self.delta_enabled,
            # Legend config (labels + per-role show flags + master enabled).
            'legend_config': {
                'enabled': bool(self.legend_config.get('enabled', False)),
                'roles': {k: dict(v) for k, v in
                          self.legend_config.get('roles', {}).items()},
            },
        }
        if widget_values:
            session['widget_values'] = widget_values
        return session

    # Session format version this build can read. Bump when
    # to_session_dict / from_session_dict change incompatibly. Stays in
    # sync with the `_version` field to_session_dict writes.
    _MAX_SUPPORTED_VERSION = 5

    def from_session_dict(self, session):
        # ── Version validation ────────────────────────────────────
        # Three failure modes surfaced explicitly instead of silently
        # loading a partially-empty state:
        #   1. `_version` missing / not int → reject (almost certainly
        #      not a session).
        #   2. `_version` > _MAX_SUPPORTED_VERSION → reject with upgrade
        #      hint (otherwise v6+ fields would drop silently).
        #   3. `_version` < 3 → reject as legacy (v3 is the earliest
        #      shape this class still handles).
        raw_version = session.get('_version')
        try:
            version = int(raw_version) if raw_version is not None else None
        except (TypeError, ValueError):
            version = None
        if version is None:
            raise ValueError(
                "Session file is missing the `_version` field — "
                "this does not look like an I-QMapper session.")
        if version > self._MAX_SUPPORTED_VERSION:
            raise ValueError(
                f"Session file _version={version} is newer than this "
                f"build supports (max v{self._MAX_SUPPORTED_VERSION}). "
                f"Upgrade iqpu_mapper to read this session, or re-save it "
                f"from the build that created it.")
        if version < 3:
            raise ValueError(
                f"Session file _version={version} is too old — please "
                f"re-save from a newer iqpu_mapper build (v3+).")

        self.active_backend_name = session.get('backend_name')
        self.active_chip = session.get('chip')
        self.active_ansatz = session.get('ansatz', 'lucj')
        # Fallback 'Blue→Purple (IBM)' preserves the visual identity of
        # pre-field sessions. Matches the __init__ default.
        self.current_colorscale = session.get('colorscale', 'Blue→Purple (IBM)')
        self.custom_colors = session.get('custom_colors', [])
        self.zoom = session.get('zoom', {'x0': None, 'x1': None, 'y0': None, 'y1': None})
        self.calibration_data_label = session.get('calibration_data_label')
        self.qubit_ids = session.get('qubit_ids', [])
        self.edges_list = [tuple(e) for e in session.get('edges_list', [])]
        self.x_coord = np.array(session.get('x_coord', []))
        self.y_coord = np.array(session.get('y_coord', []))
        self.qubit_props_data = {k: np.array(v) for k, v in session.get('qubit_props_data', {}).items()}
        self.gate_props_data = {k: np.array(v) for k, v in session.get('gate_props_data', {}).items()}
        self.readout_errors = self.qubit_props_data.get('readout_error', np.array([]))
        self.gate_errors = self.gate_props_data.get('cz_error', np.array([]))
        for i_str, ld in session.get('layouts', {}).items():
            i = int(i_str)
            if i not in self.layouts_data:
                continue
            d = self.layouts_data[i]
            for key in ('name', 'alpha', 'beta', 'other', 'qubits', 'ancilla', 'color'):
                if key in ld:
                    d[key] = ld[key]
            if 'styles' in ld:
                d['styles'] = {k: dict(v) for k, v in ld['styles'].items()}
            # Restore per-group rendering presets (v4+)
            if 'group_styles' in ld:
                for gk, gv in ld['group_styles'].items():
                    if gk in d['group_styles']:
                        d['group_styles'][gk].update(gv)
                    else:
                        d['group_styles'][gk] = dict(gv)
            # v3 sessions won't have group_styles — defaults are fine

        # v5: restore reference calibration if present (v3/v4 sessions won't have it).
        self.ref_timestamp = session.get('ref_timestamp')
        rqp = session.get('ref_qubit_props', {}) or {}
        rgp = session.get('ref_gate_props', {}) or {}
        self.ref_qubit_props = {k: np.array(v) for k, v in rqp.items()} if rqp else None
        self.ref_gate_props = {k: np.array(v) for k, v in rgp.items()} if rgp else None
        self.delta_enabled = bool(session.get('delta_enabled', False))

        # Legend config — merge saved values into defaults so older sessions
        # without this field still load cleanly (same pattern as group_styles).
        saved_legend = session.get('legend_config') or {}
        legend_defaults = _default_legend_config()
        if 'enabled' in saved_legend:
            legend_defaults['enabled'] = bool(saved_legend['enabled'])
        for role, cfg in (saved_legend.get('roles') or {}).items():
            if role in legend_defaults['roles'] and isinstance(cfg, dict):
                legend_defaults['roles'][role].update(cfg)
        self.legend_config = legend_defaults

        return session.get('widget_values')

    # ── Style preset ─────────────────────────────────────────────

    def to_style_dict(self, widget_values=None):
        style = {
            '_type': 'iqpu_mapper_style',
            'colorscale': self.current_colorscale,
            'custom_colors': list(self.custom_colors),
            'circuit_colors': {},
        }
        for i, d in self.layouts_data.items():
            style['circuit_colors'][str(i)] = {
                'color': d.get('color', ''),
                'defaults': dict(d.get('defaults', {})),
                'group_styles': {gk: dict(gv) for gk, gv in d.get('group_styles', {}).items()},
            }
        if widget_values:
            style['widget_values'] = widget_values
        return style

    def from_style_dict(self, style):
        self.current_colorscale = style.get('colorscale', self.current_colorscale)
        self.custom_colors = style.get('custom_colors', self.custom_colors)
        for i_str, cc in style.get('circuit_colors', {}).items():
            i = int(i_str)
            if i in self.layouts_data:
                if 'color' in cc:
                    self.layouts_data[i]['color'] = cc['color']
                if 'defaults' in cc:
                    self.layouts_data[i]['defaults'].update(cc['defaults'])
                # Restore per-group rendering presets
                if 'group_styles' in cc:
                    for gk, gv in cc['group_styles'].items():
                        if gk in self.layouts_data[i]['group_styles']:
                            self.layouts_data[i]['group_styles'][gk].update(gv)
                        else:
                            self.layouts_data[i]['group_styles'][gk] = dict(gv)
        return style.get('widget_values')
