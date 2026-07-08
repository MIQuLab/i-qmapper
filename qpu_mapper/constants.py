"""
constants.py — Shared constants, color utilities, and formatting helpers.

Pure functions — no mutable state, no widgets, no side effects.
"""
import numpy as np
import re
import os
import json


# Backward-compat: WidgetsRef moved to qpu_mapper/ui/widgets_ref.py (it
# was the only non-pure thing in this module). PEP 562 ``__getattr__``
# keeps ``from .constants import WidgetsRef`` working without an eager
# top-of-file import that would close the cycle
# constants → ui → ui.builder → constants (mid-load) → boom.
def __getattr__(name):
    if name == 'WidgetsRef':
        from .ui.widgets_ref import WidgetsRef as _WidgetsRef
        return _WidgetsRef
    raise AttributeError(
        f"module 'qpu_mapper.constants' has no attribute {name!r}")

# ── Coordinate helpers ──────────────────────────────────────────────

def heron_coords():
    """Return list of [y, x] coordinates for IBM Heron 156-qubit heavy-hex topology."""
    b = [[3, 7, 11, 15], [1, 5, 9, 13]]
    return [[-i, c] for i in range(15) for c in (range(16) if i % 2 == 0 else b[(i // 2) % 2])]


def nighthawk_coords():
    """Return list of [y, x] coordinates for IBM Nighthawk 120-qubit topology."""
    return [[-row, col] for row in range(12) for col in range(10)]


# ── Property definitions ────────────────────────────────────────────

QUBIT_PROP_OPTIONS = [
    ('Readout assignment error',   'readout_error'),
    ('T1 (µs)',                    't1'),
    ('T2 (µs)',                    't2'),
    ('Prob meas0 prep1',           'prob_meas0_prep1'),
    ('Prob meas1 prep0',           'prob_meas1_prep0'),
    ('Readout length (ns)',        'readout_length'),
    ('ID error',                   'id_error'),
    ('Single-qubit gate length (ns)', 'sq_gate_length'),
    ('RX error',                   'rx_error'),
    ('√X (SX) error',             'sx_error'),
    ('Z-axis rotation (RZ) error', 'rz_error'),
    ('Pauli-X error',              'x_error'),
    ('MEASURE error',              'measure_error'),
    ('MEASURE_2 error',            'measure_2_error'),
]

GATE_PROP_OPTIONS = [
    ('CZ error',           'cz_error'),
    ('RZZ error',          'rzz_error'),
    ('Gate length (ns)',   'gate_length'),
]

# Properties measured in microseconds (µs) — formatting only, no color inversion
US_QUBIT_PROPS = {'t1', 't2'}

# Qubit properties measured in nanoseconds (linear scale, fmt_ns)
NS_QUBIT_PROPS = {'readout_length', 'sq_gate_length'}

# Higher = better (coherence times). Stable Finder filter asks
# "value >= threshold" for these, "value <= threshold" otherwise.
HIGHER_IS_BETTER_PROPS = {'t1', 't2'}

MAX_CIRCUITS = 4

# Known qubit counts for hardcoded layouts
_HERON_NQUBITS     = len(heron_coords())       # 156
_NIGHTHAWK_NQUBITS = len(nighthawk_coords())   # 120


def resolve_coords(backend):
    """Return (coords, chip_label) for a backend. Reads
    ``backend.processor_type`` (dict like
    ``{'family': 'Heron', 'revision': '3'}``) and verifies against
    ``num_qubits``. Raises NotImplementedError for unsupported topologies.
    """
    proc = getattr(backend, 'processor_type', None)
    if proc is None:
        cfg = getattr(backend, 'configuration', lambda: None)()
        if cfg is not None:
            proc = getattr(cfg, 'processor_type', None)

    family = ''
    revision = '?'
    if isinstance(proc, dict):
        family = proc.get('family', '').lower()
        revision = proc.get('revision', '?')
    elif proc is not None:
        family = str(proc).lower().split('_')[0]

    n_qubits = getattr(backend, 'num_qubits', 0)

    if family == 'heron' and n_qubits == _HERON_NQUBITS:
        return heron_coords(), f"Heron r{revision}"

    if family == 'nighthawk' and n_qubits == _NIGHTHAWK_NQUBITS:
        return nighthawk_coords(), f"Nighthawk r{revision}"

    # Qubit count fallback (processor_type not readable)
    if not family:
        if n_qubits == _HERON_NQUBITS:
            return heron_coords(), "Heron"
        if n_qubits == _NIGHTHAWK_NQUBITS:
            return nighthawk_coords(), "Nighthawk"

    backend_name = getattr(backend, 'name', repr(backend))
    raise NotImplementedError(
        f"Unsupported backend topology: '{backend_name}' "
        f"(family='{family}', {n_qubits} qubits). "
        f"Add coordinate layout for this chip to constants.py."
    )


# ── Colorscale system ───────────────────────────────────────────────

COLORSCALE_OPTIONS = {
    'Blue→Purple (IBM)':   {'plotly': [[0.0,'rgb(49,85,246)'],[1.0,'rgb(245,240,255)']], 'low':(49,85,246),  'high':(245,240,255)},
    'Blue→Red (Heat)':     {'plotly': [[0.0,'rgb(0,90,255)'],[0.5,'rgb(255,235,0)'],[1.0,'rgb(255,0,0)']], 'low':(0,90,255),   'high':(255,0,0)},
    'Green→Red (Traffic)': {'plotly': [[0.0,'rgb(0,200,55)'],[0.5,'rgb(255,205,0)'],[1.0,'rgb(235,28,28)']],  'low':(0,200,55),   'high':(235,28,28)},
    'Viridis':             {'plotly': 'Viridis',  'low':(68,1,84),    'high':(253,231,37)},
    'Plasma':              {'plotly': 'Plasma',   'low':(13,8,135),   'high':(240,249,33)},
    'Cividis':             {'plotly': 'Cividis',  'low':(0,34,78),    'high':(253,231,37)},
    'Greys':               {'plotly': 'Greys',    'low':(255,255,255),'high':(0,0,0)},
    'Inferno':             {'plotly': 'Inferno',  'low':(0,0,4),      'high':(252,255,164)},
    'Magma':               {'plotly': 'Magma',    'low':(0,0,4),      'high':(252,253,191)},
    'Turbo':               {'plotly': 'Turbo',    'low':(48,18,59),   'high':(122,4,3)},
    'Hot':                 {'plotly': 'Hot',      'low':(11,0,0),     'high':(255,255,255)},
    'Electric':            {'plotly': 'Electric', 'low':(0,0,0),      'high':(255,250,220)},
    'Deep Blue→Gold':      {'plotly': [[0.0,'rgb(10,20,80)'],[0.5,'rgb(60,140,190)'],[1.0,'rgb(255,210,50)']],
                            'low':(10,20,80),     'high':(255,210,50)},
}

# Divergent colorscale for delta mode: blue (improved) ↔ gray ↔ red (degraded)
DELTA_COLORSCALE = {
    'plotly': [[0.0, 'rgb(0,100,220)'], [0.5, 'rgb(160,160,160)'], [1.0, 'rgb(220,30,30)']],
    'low': (0, 100, 220),      # blue = improved
    'mid': (160, 160, 160),    # medium gray = no change (visible against white bg)
    'high': (220, 30, 30),     # red = degraded
}

DEFAULT_CIRCUIT_COLORS = {
    0: {'fill': '#636EFA', 'line': '#000000', 'fill_beta': '#3498DB', 'fill_other': '#F39C12', 'fill_anc': '#95A5A6'},
    1: {'fill': '#EF553B', 'line': '#000000', 'fill_beta': '#C0392B', 'fill_other': '#E67E22', 'fill_anc': '#95A5A6'},
    2: {'fill': '#00CC96', 'line': '#000000', 'fill_beta': '#16A085', 'fill_other': '#2ECC71', 'fill_anc': '#95A5A6'},
    3: {'fill': '#AB63FA', 'line': '#000000', 'fill_beta': '#8E44AD', 'fill_other': '#9B59B6', 'fill_anc': '#95A5A6'},
}


# ── Color palette system ────────────────────────────────────────────

PALETTE_FILE = os.path.join(os.getcwd(), 'quantum_mapper_palette.json')
STANDARD_COLORS = [
    # Neutrals
    '#FFFFFF', '#adb5bd', '#343a40', '#000000',
    # Core palette — each visually distinct
    '#636EFA',  # blue
    '#EF553B',  # red
    '#00CC96',  # green
    '#AB63FA',  # purple
    '#FFA15A',  # orange
    '#19D3F3',  # cyan
    '#FFD43B',  # yellow
    '#FF6B9D',  # pink
    '#1ABC9C',  # teal
    '#8B4513',  # brown
]


def load_custom_colors():
    if os.path.exists(PALETTE_FILE):
        try:
            with open(PALETTE_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_custom_colors(colors):
    try:
        with open(PALETTE_FILE, 'w') as f:
            json.dump(colors, f)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("Could not save custom colors to %s: %s",
                                            PALETTE_FILE, e)


# ── Formatting utilities ────────────────────────────────────────────

def _fmt_sci(v, prec=3):
    """Scientific notation with a minimal (no leading-zero) exponent: 1.230E-5.

    Robust at any precision (unlike a literal ``.replace('E-0', ...)``).
    Non-finite values are returned via the default format ('NAN'/'INF').
    """
    if not np.isfinite(v):
        return f"{v:.{prec}E}"
    mant, _, exp = f"{v:.{prec}E}".partition('E')
    e = int(exp)  # strips the sign and any leading zeros
    return f"{mant}E{'+' if e >= 0 else '-'}{abs(e)}"


def fmt_e(val):
    """Format error value in scientific notation."""
    try:
        return _fmt_sci(float(val))
    except Exception:
        return "N/A"


def fmt_us(val):
    """Format microsecond values: 248.3 µs instead of 2.483E+2."""
    try:
        v = float(val)
        if np.isnan(v):
            return "N/A"
        if v >= 1000:
            return f"{v/1000:.1f} ms"
        if v >= 1:
            return f"{v:.1f} µs"
        return f"{v*1000:.1f} ns"
    except Exception:
        return "N/A"


def fmt_ns(val):
    """Format gate length in nanoseconds."""
    try:
        v = float(val)
        if np.isnan(v):
            return "N/A"
        return f"{v:.1f} ns"
    except Exception:
        return "N/A"


def fmt_delta(val, prop_key=None):
    """Format delta value with sign indicator."""
    try:
        v = float(val)
        if np.isnan(v):
            return "N/A"
        sign = "+" if v > 0 else ""
        if prop_key in US_QUBIT_PROPS:
            return f"{sign}{v:.1f} µs"
        if prop_key in NS_QUBIT_PROPS or prop_key == 'gate_length':
            return f"{sign}{v:.1f} ns"
        return f"{sign}{_fmt_sci(v)}"
    except Exception:
        return "N/A"


def hex_to_rgba(hex_val, opacity):
    """Convert hex color to rgba string. Returns black on invalid input."""
    try:
        h = hex_val.lstrip('#')
        if len(h) == 3:
            h = ''.join(c * 2 for c in h)
        if len(h) == 8:
            # #RRGGBBAA — combine the embedded alpha with the requested opacity.
            r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
            return f'rgba({r},{g},{b},{opacity * (int(h[6:8], 16) / 255.0)})'
        if len(h) < 6:
            return f'rgba(0,0,0,{opacity})'
        rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
        return f'rgba({rgb[0]},{rgb[1]},{rgb[2]},{opacity})'
    except (ValueError, TypeError, IndexError):
        return f'rgba(0,0,0,{opacity})'


def blend_colors(c1_str, c2_str, mix_ratio=0.4):
    """Blend two colors (rgb/hex) by ratio. Returns c2 if either input is invalid."""
    def p(c):
        if not c or c == 'none':
            return None
        if c.startswith('#'):
            h = c.lstrip('#')
            if len(h) == 3:
                h = ''.join(ch * 2 for ch in h)
            if len(h) < 6:
                return None
            return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
        nums = [int(x) for x in re.findall(r'\d+', c)[:3]]
        return tuple(nums) if len(nums) >= 3 else None
    try:
        c1, c2 = p(c1_str), p(c2_str)
        if c1 is None:
            return c2_str
        if c2 is None:
            return c1_str
        r1, g1, b1 = c1
        r2, g2, b2 = c2
        return f'rgb({int(r1*(1-mix_ratio)+r2*mix_ratio)},{int(g1*(1-mix_ratio)+g2*mix_ratio)},{int(b1*(1-mix_ratio)+b2*mix_ratio)})'
    except Exception:
        return c2_str


def text_color_for_bg(bg_color_str):
    """Return 'black' or 'white' for maximum contrast against background."""
    try:
        nums = [int(x) for x in re.findall(r'\d+', bg_color_str)[:3]]
        if len(nums) < 3:
            return 'black'
        r, g, b = nums
        lum = (0.299*r + 0.587*g + 0.114*b) / 255
        return 'black' if lum > 0.55 else 'white'
    except Exception:
        return 'black'


def make_color_mapper(arr, colorscale_name, prop_key=None):
    """Build a fast (val) -> color callable. Setup (np.asarray, NaN
    filter, log10, min/max) runs ONCE; the returned closure is O(1)
    per element — hoists the array reduction out of the loop (one
    normalisation instead of per-element).

    Semantics:
      - prop_key in US_QUBIT_PROPS ∪ NS_QUBIT_PROPS → linear on raw.
      - everything else → log10 on values in (0, 1).
      - NaN / out-of-range / missing → '#000000'.
    """
    cs = COLORSCALE_OPTIONS[colorscale_name]
    # Build a norm in [0,1] → 'rgb(...)' function. When the scale has
    # explicit plotly stops (a list), interpolate along ALL of them so
    # node colours follow the full scale and match the colorbar — a plain
    # low→high blend skips the mid stop (e.g. yellow in Green→Red) and
    # gives a muddy brown mid. Named plotly scales (strings) have no
    # explicit stops here, so keep the low→high approximation for those.
    _plotly = cs['plotly']
    _stops = None
    if isinstance(_plotly, list):
        try:
            _stops = []
            for _pos, _col in _plotly:
                _n = [int(x) for x in re.findall(r'\d+', _col)]
                if len(_n) < 3:
                    raise ValueError(f"stop {_col!r} is not rgb(r,g,b)")
                _stops.append((float(_pos), _n[0], _n[1], _n[2]))
        except (ValueError, TypeError):
            # Non-rgb stop (hex / CSS-named): fall back to the low→high
            # path below rather than IndexError when the mapper is built.
            _stops = None
    if _stops:

        def _color_at(norm):
            if norm <= _stops[0][0]:
                _, r, g, b = _stops[0]
                return f'rgb({r},{g},{b})'
            for _i in range(len(_stops) - 1):
                p0, r0, g0, b0 = _stops[_i]
                p1, r1, g1, b1 = _stops[_i + 1]
                if norm <= p1:
                    t = (norm - p0) / ((p1 - p0) + 1e-12)
                    return (f'rgb({int(r0 + (r1 - r0) * t)},'
                            f'{int(g0 + (g1 - g0) * t)},'
                            f'{int(b0 + (b1 - b0) * t)})')
            _, r, g, b = _stops[-1]
            return f'rgb({r},{g},{b})'
    else:
        _r0, _g0, _b0 = cs['low']
        _r1, _g1, _b1 = cs['high']

        def _color_at(norm):
            return (f'rgb({int(_r0 + (_r1 - _r0) * norm)},'
                    f'{int(_g0 + (_g1 - _g0) * norm)},'
                    f'{int(_b0 + (_b1 - _b0) * norm)})')

    try:
        arr_f = np.asarray(arr, dtype=float)
    except (TypeError, ValueError):
        arr_f = np.array([])
    valid_arr = arr_f[~np.isnan(arr_f)] if arr_f.size else arr_f
    if valid_arr.size == 0:
        return lambda val: '#000000'

    is_linear = prop_key in US_QUBIT_PROPS or prop_key in NS_QUBIT_PROPS
    if is_linear:
        mn = float(np.min(valid_arr))
        mx = float(np.max(valid_arr))
        denom = (mx - mn) + 1e-9

        def _mapper(val):
            try:
                v = float(val)
            except (TypeError, ValueError):
                return '#000000'
            if np.isnan(v):
                return '#000000'
            norm = (v - mn) / denom
            if norm < 0.0:
                norm = 0.0
            elif norm > 1.0:
                norm = 1.0
            return _color_at(norm)
        return _mapper

    # Log scale path: only values in (0, 1) are valid.
    valid_pos = valid_arr[(valid_arr > 0) & (valid_arr < 1.0)]
    if valid_pos.size == 0:
        return lambda val: '#000000'
    log_arr = np.log10(valid_pos + 1e-12)
    mn = float(np.min(log_arr))
    mx = float(np.max(log_arr))
    if mx > mn:
        denom = (mx - mn) + 1e-9
        log_well_defined = True
    else:
        denom = 1.0
        log_well_defined = False  # all valid values equal → fixed norm 0.5

    def _mapper(val):
        try:
            v = float(val)
        except (TypeError, ValueError):
            return '#000000'
        if np.isnan(v) or v <= 0 or v >= 1.0:
            return '#000000'
        if log_well_defined:
            log_v = np.log10(v + 1e-12)
            norm = (log_v - mn) / denom
            if norm < 0.0:
                norm = 0.0
            elif norm > 1.0:
                norm = 1.0
        else:
            norm = 0.5
        return _color_at(norm)
    return _mapper


def make_delta_color_mapper(arr, _prop_key=None):
    """Divergent delta colour mapper — blue↔gray↔red around zero,
    normalised by max-abs of the array, with the array reduction hoisted
    out. Returns a callable val → color.

    Uniform-zero arrays (snapshot vs. itself) can't normalise; returns
    light gray '#cccccc' instead of white so the chip stays visible on
    the white background. Fallback: NaN / out-of-range / missing → '#000000'.

    ``_prop_key`` is accepted for signature compatibility with
    make_color_mapper and per-property scaling in the future; it is
    currently unused.
    """
    cs = DELTA_COLORSCALE
    low, mid, high = cs['low'], cs['mid'], cs['high']

    try:
        arr_f = np.asarray(arr, dtype=float)
    except (TypeError, ValueError):
        arr_f = np.array([])
    valid = arr_f[~np.isnan(arr_f)] if arr_f.size else arr_f
    if valid.size == 0:
        return lambda val: '#000000'

    max_abs = float(np.max(np.abs(valid)))
    if max_abs < 1e-15:
        # Uniform-zero delta: use neutral gray for visibility.
        return lambda val: '#cccccc'

    def _mapper(val):
        try:
            v = float(val)
        except (TypeError, ValueError):
            return '#000000'
        if np.isnan(v):
            return '#000000'
        norm = v / max_abs
        if norm < -1.0:
            norm = -1.0
        elif norm > 1.0:
            norm = 1.0
        if norm < 0:
            t = -norm
            r = int(mid[0] + (low[0] - mid[0]) * t)
            g = int(mid[1] + (low[1] - mid[1]) * t)
            b = int(mid[2] + (low[2] - mid[2]) * t)
        else:
            t = norm
            r = int(mid[0] + (high[0] - mid[0]) * t)
            g = int(mid[1] + (high[1] - mid[1]) * t)
            b = int(mid[2] + (high[2] - mid[2]) * t)
        return f'rgb({r},{g},{b})'
    return _mapper


def cat_paws_html():
    """4 cat paw prints — Schrödinger's QPU easter egg."""

    def paw_black(size, rotate=0, opacity=1.0):
        s = size; w = s * 2.2; h = s * 2.6
        col = '#2a2a2a'; stroke = '#111111'
        return (
            f"<svg width='{w:.0f}' height='{h:.0f}' viewBox='0 0 {w:.0f} {h:.0f}' "
            f"style='opacity:{opacity}; transform:rotate({rotate}deg); display:inline-block; vertical-align:middle;'>"
            f"<ellipse cx='{w/2:.1f}' cy='{h*0.62:.1f}' rx='{s*0.62:.1f}' ry='{s*0.52:.1f}' fill='{col}' stroke='{stroke}' stroke-width='1.2'/>"
            f"<ellipse cx='{w/2:.1f}' cy='{h*0.22:.1f}' rx='{s*0.22:.1f}' ry='{s*0.26:.1f}' fill='{col}' stroke='{stroke}' stroke-width='1'/>"
            f"<ellipse cx='{w*0.22:.1f}' cy='{h*0.30:.1f}' rx='{s*0.20:.1f}' ry='{s*0.24:.1f}' fill='{col}' stroke='{stroke}' stroke-width='1'/>"
            f"<ellipse cx='{w*0.78:.1f}' cy='{h*0.30:.1f}' rx='{s*0.20:.1f}' ry='{s*0.24:.1f}' fill='{col}' stroke='{stroke}' stroke-width='1'/>"
            f"<ellipse cx='{w*0.08:.1f}' cy='{h*0.44:.1f}' rx='{s*0.16:.1f}' ry='{s*0.20:.1f}' fill='{col}' stroke='{stroke}' stroke-width='1'/>"
            f"<ellipse cx='{w*0.92:.1f}' cy='{h*0.44:.1f}' rx='{s*0.16:.1f}' ry='{s*0.20:.1f}' fill='{col}' stroke='{stroke}' stroke-width='1'/>"
            f"</svg>"
        )

    def paw_calico(size, rotate=0, opacity=1.0):
        s = size; w = s * 2.2; h = s * 2.6
        cream = '#f5efe0'; orange = '#e8832a'; black = '#2a2a2a'; tan = '#c9a96e'
        stroke_light = '#a07840'; stroke_dark = '#555555'
        return (
            f"<svg width='{w:.0f}' height='{h:.0f}' viewBox='0 0 {w:.0f} {h:.0f}' "
            f"style='opacity:{opacity}; transform:rotate({rotate}deg); display:inline-block; vertical-align:middle;'>"
            f"<ellipse cx='{w/2:.1f}' cy='{h*0.62:.1f}' rx='{s*0.62:.1f}' ry='{s*0.52:.1f}' fill='{cream}' stroke='{stroke_light}' stroke-width='1.2'/>"
            f"<ellipse cx='{w*0.42:.1f}' cy='{h*0.60:.1f}' rx='{s*0.35:.1f}' ry='{s*0.28:.1f}' fill='{orange}' opacity='0.65'/>"
            f"<ellipse cx='{w*0.58:.1f}' cy='{h*0.65:.1f}' rx='{s*0.28:.1f}' ry='{s*0.22:.1f}' fill='{black}' opacity='0.55'/>"
            f"<ellipse cx='{w*0.50:.1f}' cy='{h*0.56:.1f}' rx='{s*0.18:.1f}' ry='{s*0.14:.1f}' fill='{tan}' opacity='0.50'/>"
            f"<ellipse cx='{w/2:.1f}' cy='{h*0.22:.1f}' rx='{s*0.22:.1f}' ry='{s*0.26:.1f}' fill='{black}' stroke='{stroke_dark}' stroke-width='1'/>"
            f"<ellipse cx='{w*0.22:.1f}' cy='{h*0.30:.1f}' rx='{s*0.20:.1f}' ry='{s*0.24:.1f}' fill='{orange}' stroke='{stroke_light}' stroke-width='1'/>"
            f"<ellipse cx='{w*0.78:.1f}' cy='{h*0.30:.1f}' rx='{s*0.20:.1f}' ry='{s*0.24:.1f}' fill='{cream}' stroke='{stroke_light}' stroke-width='1'/>"
            f"<ellipse cx='{w*0.08:.1f}' cy='{h*0.44:.1f}' rx='{s*0.16:.1f}' ry='{s*0.20:.1f}' fill='{black}' stroke='{stroke_dark}' stroke-width='1'/>"
            f"<ellipse cx='{w*0.92:.1f}' cy='{h*0.44:.1f}' rx='{s*0.16:.1f}' ry='{s*0.20:.1f}' fill='{orange}' stroke='{stroke_light}' stroke-width='1'/>"
            f"</svg>"
        )

    z1 = paw_black(16, rotate=-12, opacity=0.90)
    z2 = paw_black(13, rotate=10, opacity=0.85)
    z3 = paw_calico(16, rotate=-8, opacity=0.80)
    z4 = paw_calico(13, rotate=14, opacity=0.75)

    return (
        f"<div style='display:flex; align-items:center; gap:4px; margin:2px 0;'>"
        f"{z1}{z2}{z3}{z4}"
        f"<span style='font-size:9px; color:#adb5bd; margin-left:4px;'></span>"
        f"</div>"
    )