
from iqpu_mapper.export.data import get_export_data, _safe_export_ids

_MERGED = "Merged (Single List)"
_SPLIT = "Split (Alpha/Beta/...)"


def test_safe_export_ids_sorts_ints_then_strings():
    assert _safe_export_ids(["30", "10", "20"]) == [10, 20, 30]
    out = _safe_export_ids(["5", "q0", "1"])
    assert out[0] == 1 and out[1] == 5 and "q0" in out      # ints sorted first, strings after


def test_export_split_general_sorted(make_state, make_slot):
    st = make_state(layouts={0: make_slot(name="C1", qubits=["2", "0", "1"])})
    data = get_export_data(st, "general", _SPLIT, 1)
    assert data["C1"]["qubits"] == [0, 1, 2]
    assert data["_metadata"]["backend"] == "fake_backend"
    assert data["_metadata"]["chip"] == "fake_chip"


def test_export_merged_lucj_partition(make_state, make_slot):
    st = make_state(layouts={0: make_slot(name="C1", alpha=["0", "2"], beta=["4", "6"])})
    blk = get_export_data(st, "lucj", _MERGED, 1)["C1"]
    assert blk["qubits"] == [0, 2, 4, 6]                    # alpha then beta, each sorted
    assert blk["_lucj_partition"] == {"alpha_len": 2, "beta_len": 2}


def test_export_lucj_ancilla_sorted_and_positional_indices(make_state, make_slot):
    # ancilla given out of order; exported list is sorted and the indices are
    # positional (4k), so ancilla[k] lines up with LUCJ orbital (4k, 4k).
    st = make_state(
        qubit_ids=[str(i) for i in range(40)],
        layouts={0: make_slot(name="C1", alpha=["0"], beta=["1"],
                              ancilla=["30", "10", "20"])},
    )
    blk = get_export_data(st, "lucj", _SPLIT, 1)["C1"]
    assert blk["ancilla"] == [10, 20, 30]
    assert blk["_lucj_ancilla_indices"] == [[0, 0], [4, 4], [8, 8]]


def test_export_ancilla_warning_uses_sorted_position(make_state, make_slot):
    # Finding #4 regression: a topology-failing ancilla must be labelled with the
    # position/p of its place in the SORTED exported list, not its raw input slot.
    # With no edges, every ancilla fails the alpha/beta-neighbour check and warns.
    st = make_state(
        qubit_ids=[str(i) for i in range(40)],
        edges_list=[],
        layouts={0: make_slot(name="C1", alpha=["0"], beta=["1"],
                              ancilla=["30", "10", "20"])},
    )
    warns = get_export_data(st, "lucj", _SPLIT, 1).get("_warnings", [])
    assert len(warns) == 3
    w10 = next(w for w in warns if "Ancilla 10 " in w)     # ancilla 10 is sorted-position 0
    assert "position 0" in w10 and "p=0" in w10            # NOT position 1 / p=4 (the pre-fix bug)
