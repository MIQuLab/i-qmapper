import json

import pytest

import iqmapper.export.layout_io as lio
from iqmapper.export.layout_io import apply_layout_to_circuit, import_json_layout


def _slots(make_state, make_slot, n=3):
    return make_state(
        qubit_ids=[str(i) for i in range(10)],
        edges_list=[(i, i + 1) for i in range(9)],
        layouts={i: make_slot(name=f"C{i+1}") for i in range(n)},
    )


def test_apply_partition_split_and_styles(make_state, make_slot):
    st = _slots(make_state, make_slot)
    apply_layout_to_circuit(st, 0, [0, 1, 2, 3], "lucj",
                            partition={"alpha_len": 2, "beta_len": 2, "other_len": 0})
    d = st.layouts_data[0]
    assert d["alpha"] == ["0", "1"]
    assert d["beta"] == ["2", "3"]
    assert d["other"] == []
    assert set(d["styles"].keys()) == {"0", "1", "2", "3"}     # styles always stamped


def test_apply_midpoint_fallback_on_partition_mismatch(make_state, make_slot):
    st = _slots(make_state, make_slot)
    apply_layout_to_circuit(st, 0, [0, 1, 2, 3, 4], "lucj",
                            partition={"alpha_len": 1, "beta_len": 1, "other_len": 1})
    d = st.layouts_data[0]
    assert d["alpha"] == ["0", "1"]            # mid = 5 // 2 = 2
    assert d["beta"] == ["2", "3", "4"]


def test_apply_filters_unknown_qubits(make_state, make_slot):
    st = _slots(make_state, make_slot)
    apply_layout_to_circuit(st, 0, [0, 1, 999], "general")     # 999 not in qubit_ids
    assert st.layouts_data[0]["qubits"] == ["0", "1"]


def test_import_json_split_format(make_state, make_slot):
    st = _slots(make_state, make_slot)
    payload = json.dumps({"C1": {"alpha": [0, 1], "beta": [2, 3], "ancilla": [4]}}).encode()
    names = import_json_layout(st, payload, "lucj")
    assert names == ["C1"]
    d = st.layouts_data[0]
    assert d["alpha"] == ["0", "1"] and d["beta"] == ["2", "3"] and d["ancilla"] == ["4"]


def test_import_json_size_cap(make_state, make_slot, monkeypatch):
    # Finding #5 regression: import_json_layout must reject oversized payloads.
    st = _slots(make_state, make_slot)
    monkeypatch.setattr(lio, "MAX_IMPORT_BYTES", 50)
    big = json.dumps({"C1": {"qubits": list(range(100))}}).encode()
    assert len(big) > 50
    with pytest.raises(ValueError, match="too large"):
        import_json_layout(st, big, "general")


# ── M1 hostile-partition regression tests ────────────────────────────
# A partition read from an untrusted layout file must never crash the
# import nor silently produce a wrong (negative-slice) split. Every
# malformed shape falls back to the documented midpoint behaviour.

def test_partition_negative_lengths_summing_correctly_falls_back(make_state, make_slot):
    # a=-1, b=4, o=0 sums to 3 == len(q_list): the old code accepted it
    # and produced a negative-slice split with no error. Must midpoint.
    st = _slots(make_state, make_slot)
    apply_layout_to_circuit(st, 0, [0, 1, 2], "lucj",
                            partition={"alpha_len": -1, "beta_len": 4, "other_len": 0})
    d = st.layouts_data[0]
    assert d["alpha"] == ["0"]                 # mid = 3 // 2 = 1
    assert d["beta"] == ["1", "2"]
    assert d["other"] == []


def test_partition_non_dict_garbage_falls_back(make_state, make_slot):
    st = _slots(make_state, make_slot)
    apply_layout_to_circuit(st, 0, [0, 1, 2, 3], "lucj", partition="garbage")
    d = st.layouts_data[0]
    assert d["alpha"] == ["0", "1"]
    assert d["beta"] == ["2", "3"]


def test_partition_non_numeric_value_falls_back(make_state, make_slot):
    st = _slots(make_state, make_slot)
    apply_layout_to_circuit(st, 0, [0, 1, 2, 3], "lucj",
                            partition={"alpha_len": "abc", "beta_len": 2, "other_len": 0})
    d = st.layouts_data[0]
    assert d["alpha"] == ["0", "1"]
    assert d["beta"] == ["2", "3"]


def test_partition_list_type_falls_back(make_state, make_slot):
    st = _slots(make_state, make_slot)
    apply_layout_to_circuit(st, 0, [0, 1, 2, 3], "lucj", partition=[2, 2, 0])
    d = st.layouts_data[0]
    assert d["alpha"] == ["0", "1"]
    assert d["beta"] == ["2", "3"]


def test_hostile_partition_via_full_json_import(make_state, make_slot):
    # End-to-end: the hostile partition arrives through import_json_layout,
    # exactly as an untrusted .json upload would deliver it.
    st = _slots(make_state, make_slot)
    payload = {"C1": {"qubits": [0, 1, 2],
                      "_lucj_partition": {"alpha_len": -1, "beta_len": 4,
                                          "other_len": 0}}}
    import_json_layout(st, json.dumps(payload).encode(), "lucj")
    d = st.layouts_data[0]
    assert d["alpha"] == ["0"] and d["beta"] == ["1", "2"]
