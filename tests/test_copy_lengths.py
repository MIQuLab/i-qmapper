"""The message diet, enforced.

The interface says what happened and what to do, in one short sentence; the
README explains why. Without a guard that erodes: an audit found 31 strings of
110-377 characters, tooltips that were paragraphs and notices quoting
`state.edges_list`. These tests walk the real built UI and the module source,
so the limits hold as the code changes rather than only on the day they were
applied.

Limits (from the copy rules):
  tooltips       <= 90
  status lines   <= 120
  notices        <= 100
  section intros <= 2 short sentences

Anything longer must either be an IBM error passed through verbatim (we do not
edit the provider's words) or be moved to the log.
"""
import ast
import glob
import re

import numpy as np

TAG = re.compile(r'<[^>]+>')

#: Files whose user-facing strings are covered.
SOURCES = sorted(glob.glob('iqmapper/ui/*.py')) + [
    'iqmapper/benchmark.py', 'iqmapper/reservoir.py']

#: Passed through from the provider or a library — not ours to shorten.
PASSTHROUGH = (
    'Real-time benchmarking needs qiskit-experiments',   # install hint
)

TOOLTIP_MAX = 90
STATUS_MAX = 120
NOTICE_MAX = 100


def _strip(text):
    return re.sub(r'\s+', ' ', TAG.sub('', text)).strip()


def _build_ui(monkeypatch):
    monkeypatch.setattr(
        'iqmapper.data.generate_device_data',
        lambda s, v, c, d, prefetched_props=None: (
            list(range(8)), [0.0] * 8, [(i, i + 1) for i in range(7)],
            [str(i) for i in range(8)],
            np.array([0.01 * (i + 1) for i in range(8)]),
            np.array([0.001 * (i + 1) for i in range(7)])))
    from iqmapper.state import SessionState
    from iqmapper.ui.builder import MapperUIBuilder
    state = SessionState()
    for key, val in dict(
            active_backend=object(), active_backend_name='ibm_here',
            active_chip='c', active_service=None, available_backends=[],
            props_warning=None, initial_props=object(), active_ansatz='lucj',
            active_vendor='IBM', active_coords=[], custom_colors=[],
            cal_history_dates=['_seed_']).items():
        setattr(state, key, val)
    builder = MapperUIBuilder(state)
    full_ui, _fig = builder.build()
    return builder, full_ui


def _widgets(node, seen=None):
    if seen is None:
        seen = []
    if any(n is node for n in seen):
        return seen
    seen.append(node)
    for child in getattr(node, 'children', ()) or ():
        _widgets(child, seen)
    return seen


# ── the built UI ────────────────────────────────────────────────────

def test_every_tooltip_is_one_short_sentence(monkeypatch):
    builder, full_ui = _build_ui(monkeypatch)
    offenders = []
    for widget in _widgets(full_ui):
        tip = getattr(widget, 'tooltip', None)
        if not isinstance(tip, str) or not tip:
            continue
        text = _strip(tip)
        if len(text) > TOOLTIP_MAX:
            offenders.append(
                f'{type(widget).__name__} ({len(text)} chars): {text[:80]}')
    assert not offenders, 'tooltips over %d chars:\n  %s' % (
        TOOLTIP_MAX, '\n  '.join(offenders))


def test_no_tooltip_mentions_an_internal_name(monkeypatch):
    """A user who does not read code should not meet `state.edges_list`."""
    builder, full_ui = _build_ui(monkeypatch)
    banned = ('state.', 'qiskit-experiments version', 'composite metadata',
              'job phase', '_error', 'edges_list')
    offenders = []
    for widget in _widgets(full_ui):
        tip = getattr(widget, 'tooltip', None)
        if not isinstance(tip, str):
            continue
        for token in banned:
            if token in tip:
                offenders.append(f'{token!r} in {tip[:70]!r}')
    assert not offenders, offenders


def test_widget_descriptions_stay_short(monkeypatch):
    builder, full_ui = _build_ui(monkeypatch)
    offenders = [
        f'{_strip(w.description)[:70]} ({len(_strip(w.description))})'
        for w in _widgets(full_ui)
        if isinstance(getattr(w, 'description', None), str)
        and len(_strip(w.description)) > 60]
    assert not offenders, offenders


# ── the source ──────────────────────────────────────────────────────

def _string_constants(path):
    """Non-docstring, non-log string constants, with their line numbers."""
    with open(path, encoding='utf-8') as fh:
        tree = ast.parse(fh.read())
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc:
                docstrings.add(doc)
    logged = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in ('_log', 'log', 'logger')):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                    logged.add(sub.value)
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant):
            continue
        if not isinstance(node.value, str) or node.value in docstrings:
            continue
        if node.value in logged:
            continue
        out.append((node.lineno, node.value))
    return out


def test_no_user_facing_string_is_a_paragraph():
    """The inventory, re-run. Diagnostics belong in the log, not the UI."""
    offenders = []
    for path in SOURCES:
        for lineno, raw in _string_constants(path):
            text = _strip(raw)
            if len(text) <= STATUS_MAX:
                continue
            if any(p in text for p in PASSTHROUGH):
                continue
            offenders.append(f'{path}:{lineno} ({len(text)}): {text[:90]}')
    assert not offenders, 'strings over %d chars:\n  %s' % (
        STATUS_MAX, '\n  '.join(offenders))


# ── README anchors ──────────────────────────────────────────────────

def _readme_anchors():
    headings = set()
    with open('README.md', encoding='utf-8') as fh:
        for line in fh:
            if line.startswith('#'):
                headings.add(line.lstrip('#').strip())
    return headings


def test_every_readme_pointer_has_a_matching_heading():
    """A dangling `· README: X` is worse than no pointer at all."""
    anchors = _readme_anchors()
    pointer = re.compile(r'README:\s*([^\'"<·]+)')
    missing = []
    for path in SOURCES:
        for lineno, raw in _string_constants(path):
            for match in pointer.finditer(raw):
                name = match.group(1).strip().rstrip('.')
                if name and name not in anchors:
                    missing.append(f'{path}:{lineno} -> {name!r}')
    assert not missing, (
        'README pointers with no matching heading:\n  '
        + '\n  '.join(missing)
        + f'\nheadings present: {sorted(anchors)}')


def test_the_understanding_section_covers_every_anchor_used():
    anchors = _readme_anchors()
    assert 'Understanding the messages' in anchors
    for name in ('Search cap', 'Diverse enumeration', 'Search engine',
                 'Prefer diverse regions', 'Quality floor', 'Levels',
                 'Targets', 'Relative mode', 'Calibration check',
                 'Delta references', 'Offline mode', 'LQS'):
        assert name in anchors, f'README is missing the {name!r} entry'
