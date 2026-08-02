"""
export.layout_io — Apply a qubit list to a circuit slot, and import
layouts from external files (.json / .npy / .qpy / .qasm).

This sub-module owns the user-facing import surface (`import_layout_file`
and the per-format helpers) plus the small payload-application primitives
shared with the JSON path.
"""
import contextlib
import io
import json
import logging
import os
import re

import numpy as np

from ..constants import MAX_CIRCUITS, hex_to_rgba
from . import MAX_IMPORT_BYTES

_log = logging.getLogger(__name__)


def apply_layout_to_circuit(state, idx, q_list, mode, partition=None):
    """
    Apply a list of qubit indices to a circuit slot.

    For LUCJ with explicit partition: splits per (alpha_len, beta_len, other_len).
    For LUCJ without partition: falls back to midpoint split (legacy behavior,
                                only correct when alpha_len == beta_len and 'other' is empty).
    For general: all go to 'qubits'.
    Rebuilds styles from default circuit colors.

    Args:
        state: SessionState
        idx: circuit slot index
        q_list: iterable of qubit IDs (int or str)
        mode: 'lucj' or 'general'
        partition: optional dict {'alpha_len': int, 'beta_len': int, 'other_len': int}
                   When provided, splits q_list exactly per these counts.
                   If the sizes don't add up to len(q_list_str), falls back to midpoint.

    Always stamps styles — earlier callers left them empty, so qubits
    appeared with no visual fill.
    """
    if idx >= MAX_CIRCUITS or idx not in state.layouts_data:
        return
    d = state.layouts_data[idx]
    d['alpha'] = []; d['beta'] = []; d['other'] = []
    d['qubits'] = []; d['ancilla'] = []; d['styles'] = {}

    q_list_str = [str(q) for q in q_list if str(q) in state.qubit_ids]

    if mode == 'lucj':
        use_partition = False
        if isinstance(partition, dict):
            try:
                a_len = int(partition.get('alpha_len', 0))
                b_len = int(partition.get('beta_len', 0))
                o_len = int(partition.get('other_len', 0))
            except (TypeError, ValueError):
                a_len = b_len = o_len = -1   # force fallback
                _log.warning(
                    "Malformed _lucj_partition %r — midpoint fallback", partition)
            if (a_len >= 0 and b_len >= 0 and o_len >= 0
                    and a_len + b_len + o_len == len(q_list_str)):
                use_partition = True
        elif partition is not None:
            _log.warning(
                "_lucj_partition is %s, expected dict — midpoint fallback",
                type(partition).__name__)

        if use_partition:
            d['alpha'] = q_list_str[:a_len]
            d['beta']  = q_list_str[a_len : a_len + b_len]
            d['other'] = q_list_str[a_len + b_len :]
        else:
            # Legacy midpoint split (only correct when alpha_len == beta_len
            # and 'other' is empty — preserved for backward compat with files
            # that don't have _lucj_partition metadata).
            mid = len(q_list_str) // 2
            d['alpha'] = q_list_str[:mid]
            d['beta']  = q_list_str[mid:]

        alpha_color = d['defaults']['fill']
        beta_color  = d['defaults']['fill_beta']
        other_color = d['defaults'].get('fill_other', '#F39C12')
        line_color  = d['defaults']['line']
        for q in d['alpha']:
            d['styles'][q] = {'fill': hex_to_rgba(alpha_color, 1.0), 'line': line_color,
                              'width': 2, 'text_color': '#000000'}
        for q in d['beta']:
            d['styles'][q] = {'fill': hex_to_rgba(beta_color, 1.0), 'line': line_color,
                              'width': 2, 'text_color': '#000000'}
        for q in d['other']:
            d['styles'][q] = {'fill': hex_to_rgba(other_color, 1.0), 'line': line_color,
                              'width': 2, 'text_color': '#000000'}
    else:
        d['qubits'] = q_list_str
        qc = d['defaults']['fill']
        line_color = d['defaults']['line']
        for q in d['qubits']:
            d['styles'][q] = {'fill': hex_to_rgba(qc, 1.0), 'line': line_color, 'width': 2, 'text_color': '#000000'}


def import_json_layout(state, json_bytes, mode):
    """
    Import a JSON layout file into state.layouts_data.

    Supports multiple formats:
    - {"C1": {"alpha": [...], "beta": [...], "ancilla": [...]}}
    - {"C1": {"qubits": [...], "ancilla": [...]}}
    - {"C1": [0, 1, 2, ...]}  (merged format)

    Returns list of imported circuit names on success, or raises Exception.
    """
    if len(json_bytes) > MAX_IMPORT_BYTES:
        raise ValueError(
            f"JSON layout file too large: {len(json_bytes) / 1_048_576:.1f} MB "
            f"exceeds the {MAX_IMPORT_BYTES // 1_048_576} MB limit."
        )
    data = json.loads(json_bytes.decode('utf-8'))

    for circ_name, payload in data.items():
        idx = _resolve_circuit_index(state, circ_name)
        if idx is None:
            continue
        d = state.layouts_data[idx]
        _clear_circuit_slot(d)

        if isinstance(payload, list):
            _apply_merged_list_payload(state, idx, payload, mode)
        elif isinstance(payload, dict):
            _apply_dict_payload(state, idx, payload, mode)
        # Anything else (e.g. payload is an int) is silently skipped:
        # apply_layout_to_circuit can't make sense of it and we don't
        # want one malformed slot to abort the whole import.

    return list(data.keys())


# ─────────────────────────────────────────────────────────────────────
# import_json_layout helpers
# ─────────────────────────────────────────────────────────────────────

# Default color used for orphan 'other' qubits when the layout file
# omits a per-key colour mapping. Kept here so the two import branches
# (merged-with-partition and split) agree on the fallback.
_DEFAULT_OTHER_COLOR = '#F39C12'
_DEFAULT_ANCILLA_COLOR = '#95A5A6'

# Keys that participate in the split-format layout payload, in the
# order their colour mapping is applied.
_SPLIT_KEYS = ('alpha', 'beta', 'other', 'qubits', 'ancilla')


def _resolve_circuit_index(state, circ_name):
    """Map a payload circuit name like 'C2' or 'circuit 2' to a 0-based
    slot index, or None if no slot is available.

    Tries:
      1. exact match against any existing `state.layouts_data[i]['name']`,
      2. the LAST run of digits in the name → int → 0-based index.
    Out-of-range indices return None so the caller can skip the slot
    instead of raising IndexError.
    """
    for i, d in state.layouts_data.items():
        if d['name'] == circ_name:
            return i
    digits = re.findall(r'\d+', circ_name)
    if not digits:
        return None
    try:
        idx = int(digits[-1]) - 1
    except (ValueError, IndexError):
        return None
    if idx < 0 or idx >= MAX_CIRCUITS:
        return None
    return idx


def _clear_circuit_slot(d):
    """Reset the per-circuit dict before re-populating it from a payload."""
    d['alpha'] = []
    d['beta'] = []
    d['other'] = []
    d['qubits'] = []
    d['ancilla'] = []
    d['styles'] = {}


def _coerce_ints(seq):
    """Drop non-numeric entries from a sequence of mixed-type qubit IDs.

    Mirrors the silent-skip behaviour the merged-format import had since
    the start — a non-numeric token in a merged list is a corrupt entry,
    not a fatal error, so we keep going.
    """
    out = []
    for q in seq:
        # Corrupt (non-numeric) entries in a merged list are silently
        # dropped by design — see the docstring above.
        with contextlib.suppress(TypeError, ValueError):
            out.append(int(q))
    return out


def _to_str_list(state, lst):
    """String-coerce qubit IDs and drop entries not present in
    `state.qubit_ids` (i.e. silently ignore qubits the active backend
    doesn't expose)."""
    return [str(q) for q in lst if str(q) in state.qubit_ids]


def _apply_styles(d, qubit_keys, color):
    """Stamp `d['styles'][q]` for every q in `d[key]` for every
    requested `key`, using `color` as the fill."""
    line_color = d['defaults']['line']
    for key in qubit_keys:
        for q in d.get(key, []):
            d['styles'][q] = {
                'fill': hex_to_rgba(color, 1.0),
                'line': line_color, 'width': 2,
                'text_color': '#000000',
            }


def _apply_merged_list_payload(state, idx, payload, mode):
    """Legacy merged-list format: `{"C1": [0, 1, 2, ...]}`.

    No partition info → LUCJ mode falls back to midpoint split, only
    correct when alpha and beta are balanced and 'other' is empty.
    """
    apply_layout_to_circuit(state, idx, _coerce_ints(payload), mode)


def _apply_dict_payload(state, idx, payload, mode):
    """Dispatch between the two dict shapes:

    • Merged-with-partition (`'qubits' + '_lucj_partition'`, LUCJ only)
    • Split (`alpha`/`beta`/`other`/`qubits`/`ancilla` keys)
    """
    is_merged_with_partition = (
        mode == 'lucj'
        and 'qubits' in payload
        and '_lucj_partition' in payload
    )
    if is_merged_with_partition:
        _apply_merged_partition_payload(state, idx, payload, mode)
    else:
        _apply_split_payload(state, idx, payload)


def _apply_merged_partition_payload(state, idx, payload, mode):
    """Merged-with-partition: 'qubits' holds alpha + beta concatenated;
    the partition lengths split them back. 'other' and 'ancilla' are
    separate top-level fields. Legacy files written before the α/β-only
    fix may still have other_len in the partition and other qubits
    inside 'qubits'; apply_layout_to_circuit handles that via o_len.
    """
    d = state.layouts_data[idx]
    apply_layout_to_circuit(
        state, idx, _coerce_ints(payload['qubits']), mode,
        partition=payload['_lucj_partition'],
    )
    if 'other' in payload:
        d['other'] = _to_str_list(state, payload['other'])
        _apply_styles(d, ('other',),
                      d['defaults'].get('fill_other', _DEFAULT_OTHER_COLOR))
    if 'ancilla' in payload:
        d['ancilla'] = _to_str_list(state, payload['ancilla'])
        # Ancilla uses a fixed black line, not the circuit line colour,
        # so the visual treatment matches what the user picks via the
        # ancilla tool in the map.
        anc_color = d['defaults'].get('fill_anc', _DEFAULT_ANCILLA_COLOR)
        for q in d['ancilla']:
            d['styles'][q] = {
                'fill': hex_to_rgba(anc_color, 1.0),
                'line': '#000000', 'width': 2,
                'text_color': '#000000',
            }


def _apply_split_payload(state, idx, payload):
    """Split format: explicit alpha/beta/other/qubits/ancilla keys."""
    d = state.layouts_data[idx]
    for key in _SPLIT_KEYS:
        if key in payload:
            d[key] = _to_str_list(state, payload[key])
    color_map = {
        'alpha':   d['defaults']['fill'],
        'qubits':  d['defaults']['fill'],
        'beta':    d['defaults']['fill_beta'],
        'other':   d['defaults'].get('fill_other', _DEFAULT_OTHER_COLOR),
        'ancilla': d['defaults'].get('fill_anc', _DEFAULT_ANCILLA_COLOR),
    }
    for key, col in color_map.items():
        _apply_styles(d, (key,), col)


def import_npy_layout(state, npy_bytes, mode, n_circ):
    """
    Import a .npy layout file.

    Supports:
    - 1D array of ints → apply to circuit 0
    - 2D array or array of arrays → apply to circuits 0..N

    Returns number of circuits populated.
    Qubit ID validation is handled by apply_layout_to_circuit (str(q) in state.qubit_ids).
    """
    if len(npy_bytes) > MAX_IMPORT_BYTES:
        raise ValueError(
            f"NPY file too large: {len(npy_bytes) / 1_048_576:.1f} MB "
            f"exceeds the {MAX_IMPORT_BYTES // 1_048_576} MB limit."
        )
    # Security: allow_pickle=False blocks the pickle-RCE vector. A .npy file
    # that contains pickled Python objects is equivalent to pickle.loads() on
    # untrusted bytes — i.e. remote code execution. We only accept numeric arrays.
    try:
        data = np.load(io.BytesIO(npy_bytes), allow_pickle=False)
    except ValueError as e:
        raise ValueError(
            "This .npy file contains pickled Python objects, which is a "
            "remote-code-execution vector. For security we only accept numeric "
            "arrays. Re-save with: np.save(path, arr.astype(int))"
        ) from e
    if not np.issubdtype(data.dtype, np.integer):
        raise ValueError(
            f"Expected integer qubit IDs in .npy, got dtype={data.dtype}"
        )
    count = 0

    # Empty array → nothing to import. Without this guard the `data[0]`
    # access below raises IndexError (a 1-D zero-length array has no
    # element to type-check).
    if data.size == 0:
        return 0

    if data.ndim == 1 and isinstance(data[0], (int, np.integer)):
        apply_layout_to_circuit(state, 0, [int(x) for x in data], mode)
        count = 1
    elif (data.ndim == 1 and isinstance(data[0], (list, np.ndarray))) or data.ndim == 2:
        for i in range(min(len(data), n_circ)):
            apply_layout_to_circuit(state, i, [int(x) for x in data[i]], mode)
            count += 1

    return count
# ── Unified file import (auto-detect format) ────────────────────────

def import_layout_file(state, file_bytes, filename, mode, n_circ):
    """
    Auto-detect file format and import layout.

    Supported:
      .json  — JSON layout (same as import_json_layout)
      .npy   — NumPy array (same as import_npy_layout)
      .qpy   — Qiskit QPY: extracts initial_layout from transpiled circuit
      .qasm  — OpenQASM 2/3: extracts qubit indices from register

    Returns (format_str, detail_str) on success.
    Raises Exception on failure.
    """
    ext = os.path.splitext(filename)[1].lower() if filename else ''

    # Auto-detect by content if extension is ambiguous. Magic-byte checks
    # for text formats use upper() so we match regardless of how the
    # source tool capitalised the header (`OPENQASM`, `OpenQASM`,
    # `openqasm` all appear in the wild). Binary formats (.npy / .qpy)
    # have a fixed magic that's never re-cased.
    if not ext:
        head = file_bytes[:8].upper() if isinstance(file_bytes, (bytes, bytearray)) else b''
        if file_bytes[:1] == b'{' or file_bytes[:1] == b'[':
            ext = '.json'
        elif head[:6] == b'QISKIT':
            ext = '.qpy'
        elif head[:7] == b'OPENQAS':
            ext = '.qasm'
        elif file_bytes[:6] == b'\x93NUMPY':
            ext = '.npy'

    if ext == '.json':
        names = import_json_layout(state, file_bytes, mode)
        return '.json', f"Imported circuits: {', '.join(names)}"

    elif ext == '.npy':
        count = import_npy_layout(state, file_bytes, mode, n_circ)
        return '.npy', f"Imported {count} circuit(s) from NumPy array"

    elif ext == '.qpy':
        layout = _extract_layout_from_qpy(file_bytes)
        if layout:
            apply_layout_to_circuit(state, 0, layout, mode)
            return '.qpy', f"Extracted layout ({len(layout)} qubits) from QPY circuit"
        else:
            raise ValueError("QPY file has no initial_layout or routing information")

    elif ext in ('.qasm', '.qasm2', '.qasm3'):
        layout = _extract_layout_from_qasm(file_bytes)
        if layout:
            apply_layout_to_circuit(state, 0, layout, mode)
            return '.qasm', f"Extracted {len(layout)} qubits from QASM"
        else:
            raise ValueError("Could not extract qubit layout from QASM file")

    else:
        raise ValueError(f"Unsupported file format: '{ext}'. Use .json, .npy, .qpy, or .qasm")


def _extract_layout_from_qpy(file_bytes):
    """Extract initial_layout qubit indices from a QPY file, preserving
    the virtual→physical ordering produced by transpile().

    ⚠️ SECURITY: `qpy.load()` walks a binary format produced by Qiskit
    and reconstructs Python objects (registers, instructions, custom
    gates) by importing the classes they reference. A crafted QPY file
    can therefore trigger arbitrary class imports and their import-time
    side effects — this is a deserialization-of-untrusted-data hazard
    comparable in shape (though narrower in surface) to `pickle.load`.
    Callers pass user-selected uploads directly (no UI confirmation step
    as of v0.3); the accepted risk is bounded by the fact that the user
    chose the file. Do NOT route network-received or third-party bytes
    here.

    Output for a transpiled circuit: list where `result[v]` is the physical
    qubit assigned to virtual qubit `v`. Must preserve virtual order — a
    naive `sorted(get_physical_bits().keys())` returns the SET in
    physical-id order, which for LUCJ interleaves α and β so the import
    re-splits them into the wrong groups.

    Resolution order (tries each, falls through on AttributeError /
    Exception):
      1. circ.layout.initial_index_layout(filter_ancillas=True)
         — modern Qiskit helper (≳ 0.45) that returns the list directly.
      2. Manual reconstruction from circ.layout.input_qubit_mapping
         + initial_layout.get_virtual_bits()
         — covers older Qiskit versions where the helper is absent.
      3. Walk get_virtual_bits() ordered by the virtual qubit's index attr
         — last-resort interpretation when neither shape is available.
      4. list(range(circ.num_qubits))
         — circuit was never transpiled (no layout attached).
    """
    # Defensive size cap (see MAX_IMPORT_BYTES at module top for rationale).
    if len(file_bytes) > MAX_IMPORT_BYTES:
        raise ValueError(
            f"QPY file too large: {len(file_bytes) / 1_048_576:.1f} MB "
            f"exceeds the {MAX_IMPORT_BYTES // 1_048_576} MB limit. "
            f"If this is a legitimate file, raise MAX_IMPORT_BYTES."
        )
    try:
        from qiskit import qpy
        circuits = qpy.load(io.BytesIO(file_bytes))
        if not circuits:
            return None
        circ = circuits[0]
        layout = getattr(circ, 'layout', None)
        if layout is not None:
            # Strategy 1: the modern helper. Returns a list of physical
            # indices in virtual-qubit order, optionally filtering out
            # ancillas Qiskit added during transpile.
            helper = getattr(layout, 'initial_index_layout', None)
            if callable(helper):
                try:
                    return list(helper(filter_ancillas=True))
                except TypeError:
                    # filter_ancillas kwarg may not exist on older releases.
                    try:
                        return list(helper())
                    except Exception as exc:
                        _log.debug("initial_index_layout() failed: %s", exc)

            # Strategy 2: rebuild the order from input_qubit_mapping
            # (which preserves the user's virtual-qubit order) using the
            # initial_layout's virtual_bits map (which holds the physical
            # index for each virtual Qubit object).
            input_map = getattr(layout, 'input_qubit_mapping', None)
            initial = getattr(layout, 'initial_layout', None)
            if input_map is not None and initial is not None:
                try:
                    v_bits = initial.get_virtual_bits()
                    ordered = sorted(input_map.items(), key=lambda kv: kv[1])
                    return [v_bits[q] for q, _ in ordered if q in v_bits]
                except Exception as exc:
                    _log.debug("Manual layout reconstruction failed: %s", exc)

            # Strategy 3: walk virtual_bits in expose order — preserves
            # virtual order via dict insertion order. Handle both dict-like
            # and iterable-of-pairs shapes since Qiskit's `.items()` return
            # type has shifted across releases (dict / OrderedDict / mapping).
            if initial is not None:
                try:
                    v_bits = initial.get_virtual_bits()
                    if hasattr(v_bits, 'items'):
                        return [phys for _, phys in v_bits.items()]
                    # Fall-through for iterable-of-pairs shape.
                    return [phys for _, phys in v_bits]
                except Exception as exc:
                    _log.debug("get_virtual_bits walk failed: %s", exc)

        # Strategy 4: non-transpiled circuit — no layout to preserve.
        return list(range(circ.num_qubits))
    except ImportError as exc:
        raise ImportError("qiskit is required to load QPY files") from exc
    except Exception as exc:
        _log.warning("QPY layout extraction failed: %s", exc)
        return None


# Pre-compiled regex for the comment strip + qreg/qubit lookup. The
# string-replace approach used here is intentionally NOT a full QASM
# parser — that's well outside the scope of this tool. We strip the two
# QASM comment flavours (//... line comments and /*...*/ block comments)
# before the qubit-count regex runs, so a register declaration sitting
# behind a `// qreg q[64];` line comment no longer counts.
_QASM_LINE_COMMENT_RE = re.compile(r'//[^\n]*')
_QASM_BLOCK_COMMENT_RE = re.compile(r'/\*.*?\*/', re.DOTALL)
_QASM2_QREG_RE = re.compile(r'\bqreg\s+\w+\s*\[\s*(\d+)\s*\]\s*;')
_QASM3_QUBIT_RE = re.compile(r'\bqubit\s*\[\s*(\d+)\s*\]\s*\w+\s*;')
# QASM 3 single-qubit declaration: `qubit q;` (no size = one qubit each)
_QASM3_QUBIT_SINGLE_RE = re.compile(r'\bqubit\s+\w+\s*;')


def _strip_qasm_comments(text: str) -> str:
    """Remove `//...` line and `/*...*/` block comments from QASM source.

    The qubit-count regexes below would otherwise match register
    declarations inside commented-out code — a common copy-paste artifact.
    Cheaper than a full parser; sufficient for the layout-extraction job.
    """
    text = _QASM_BLOCK_COMMENT_RE.sub('', text)
    text = _QASM_LINE_COMMENT_RE.sub('', text)
    return text


def _extract_layout_from_qasm(file_bytes):
    """Extract qubit indices from a QASM file.

    QASM 2.0 declares quantum registers with `qreg name[N];`, QASM 3.0
    uses `qubit[N] name;` or `qubit name;` (single-qubit form). A
    circuit may contain MULTIPLE register declarations
    (e.g. `qreg q[3]; qreg a[2];`) — Qiskit concatenates them into a
    single global qubit indexing where the second register's qubits are
    appended after the first. The previous implementation used only
    `matches[0]` and silently truncated multi-register circuits to the
    first register's size. We sum the sizes instead so the returned
    layout has one entry per physical qubit the circuit will need.

    Comments are stripped before matching (see `_strip_qasm_comments`).
    """
    try:
        text = file_bytes.decode('utf-8')
        text = _strip_qasm_comments(text)
        # QASM 2.0: `qreg name[N];`
        matches = _QASM2_QREG_RE.findall(text)
        if matches:
            total = sum(int(m) for m in matches)
            if len(matches) > 1:
                _log.info(
                    "QASM has %d qreg declarations (sizes %s); "
                    "extracted %d qubits as a single global index range.",
                    len(matches), matches, total,
                )
            return list(range(total))
        # QASM 3.0: `qubit[N] name;` — same multi-declaration treatment.
        matches = _QASM3_QUBIT_RE.findall(text)
        if matches:
            total = sum(int(m) for m in matches)
            if len(matches) > 1:
                _log.info(
                    "QASM3 has %d qubit[N] declarations (sizes %s); "
                    "extracted %d qubits as a single global index range.",
                    len(matches), matches, total,
                )
            return list(range(total))
        # QASM 3.0 single-qubit form: `qubit q;` each counts as 1.
        singles = _QASM3_QUBIT_SINGLE_RE.findall(text)
        if singles:
            total = len(singles)
            _log.info("QASM3 has %d single-qubit declarations; "
                      "extracted as a global index range.", total)
            return list(range(total))
        return None
    except Exception as exc:
        _log.warning("QASM layout extraction failed: %s", exc)
        return None
