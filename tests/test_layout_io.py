"""Regression tests for iqmapper.export.layout_io — .npy and .qpy paths.

Historical bug: the module used `io.BytesIO(...)` at two sites without
importing `io`, so the .npy and .qpy import paths NameError'd at call
time. `import io` is now present; these tests exercise both paths so a
future regression that drops the import (or the guards) is caught.

Uses `make_state` / `make_slot` from conftest — the layout helpers only
read a small subset of state attributes, so a SimpleNamespace suffices.
"""
import io

import numpy as np
import pytest

from iqmapper.export.layout_io import (
    _extract_layout_from_qpy,
    import_layout_file,
    import_npy_layout,
)


# ═══════════════════════════════════════════════════════════════════════
# .npy path
# ═══════════════════════════════════════════════════════════════════════

def _npy_bytes(arr):
    """Serialize a numpy array to bytes using the standard .npy container.
    Building the bytes in-memory keeps the test deterministic and offline."""
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def test_npy_happy_path_returns_count(make_state, make_slot):
    """Regression: with the missing-import bug, this call NameError'd
    before it could return anything."""
    st = make_state(qubit_ids=['0', '1', '2'],
                    layouts={0: make_slot()})
    payload = _npy_bytes(np.array([0, 1, 2], dtype=int))
    count = import_npy_layout(st, payload, 'general', n_circ=1)
    assert count == 1


def test_npy_rejects_pickled_object_array(make_state, make_slot):
    """The allow_pickle=False guard is now actually reachable. An
    object-dtype .npy is a pickle-RCE vector and must raise ValueError."""
    st = make_state(qubit_ids=['0', '1', '2'], layouts={0: make_slot()})
    payload = _npy_bytes(np.array([{'a': 1}], dtype=object))
    with pytest.raises(ValueError, match="pickled"):
        import_npy_layout(st, payload, 'general', n_circ=1)


def test_npy_rejects_oversized_payload(make_state, make_slot):
    """MAX_IMPORT_BYTES gate fires before np.load, so a crafted-oversize
    file cannot exhaust memory before validation."""
    st = make_state(qubit_ids=['0', '1'], layouts={0: make_slot()})
    from iqmapper.export import MAX_IMPORT_BYTES
    big = b'\x00' * (MAX_IMPORT_BYTES + 1)
    with pytest.raises(ValueError, match="too large"):
        import_npy_layout(st, big, 'general', n_circ=1)


def test_npy_via_dispatcher(make_state, make_slot):
    """import_layout_file routes .npy to import_npy_layout — same code
    path, exercised through the public dispatcher."""
    st = make_state(qubit_ids=['0', '1'], layouts={0: make_slot()})
    payload = _npy_bytes(np.array([0, 1], dtype=int))
    fmt, detail = import_layout_file(st, payload, 'layout.npy', 'general', n_circ=1)
    assert fmt == '.npy'
    assert 'circuit' in detail.lower()


# ═══════════════════════════════════════════════════════════════════════
# .qpy path — skipped if qiskit isn't installed
# ═══════════════════════════════════════════════════════════════════════

def test_qpy_extract_layout_no_nameerror():
    """Regression: with the missing-import bug this call NameError'd
    on `io.BytesIO(file_bytes)` before qpy.load ever ran."""
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit, qpy

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    buf = io.BytesIO()
    qpy.dump(qc, buf)
    payload = buf.getvalue()

    # The helper returns a layout list or None per its contract; either
    # outcome is acceptable — the point of this test is that it does NOT
    # raise NameError on the io.BytesIO(...) call.
    result = _extract_layout_from_qpy(payload)
    assert result is None or isinstance(result, list)
