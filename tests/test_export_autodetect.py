from qpu_mapper.export.autodetect import parse_auto_text, _autodetect_lucj_partition


def _two_chain_state(make_state, make_slot):
    # two disjoint chains on the chip: 0-1-2 and 5-6-7 (no edge between them)
    return make_state(
        qubit_ids=[str(i) for i in range(10)],
        edges_list=[(0, 1), (1, 2), (5, 6), (6, 7)],
        layouts={i: make_slot(name=f"C{i+1}") for i in range(4)},
    )


def test_autodetect_two_components(make_state, make_slot):
    st = _two_chain_state(make_state, make_slot)
    alpha, beta, ok = _autodetect_lucj_partition([0, 1, 2, 5, 6, 7], st)
    assert ok is True
    assert set(alpha) == {"0", "1", "2"}      # component with smallest id -> alpha
    assert set(beta) == {"5", "6", "7"}


def test_autodetect_single_component_fails(make_state, make_slot):
    st = _two_chain_state(make_state, make_slot)
    _a, _b, ok = _autodetect_lucj_partition([0, 1, 2], st)    # one connected chain
    assert ok is False


def test_parse_auto_text_lucj_uses_autodetect(make_state, make_slot):
    st = _two_chain_state(make_state, make_slot)
    count, stats = parse_auto_text(st, "0,1,2,5,6,7", "lucj")
    assert count == 1
    assert stats[0]["used_autodetect"] is True
    assert stats[0]["alpha_len"] == 3 and stats[0]["beta_len"] == 3


def test_parse_auto_text_general_mode(make_state, make_slot):
    st = _two_chain_state(make_state, make_slot)
    count, stats = parse_auto_text(st, "0,1,2", "general")
    assert count == 1
    assert st.layouts_data[0]["qubits"] == ["0", "1", "2"]
    assert stats[0]["used_autodetect"] is False
