import re

from iqpu_mapper.export.codegen import get_qiskit_layout_code


def test_codegen_general_dict_and_flatlist(make_state, make_slot):
    st = make_state(layouts={0: make_slot(name="C1", qubits=["0", "3", "5"])})
    code = get_qiskit_layout_code(st, "general", 1)
    assert "from qiskit.transpiler import Layout" in code
    assert "initial_layout_1 = {0: 0, 1: 3, 2: 5}" in code      # virtual -> physical
    assert "[0, 3, 5]" in code                                  # flat ordered list


def test_codegen_sanitizes_identifier(make_state, make_slot):
    # Finding-class regression: punctuation / accents / leading digit in the
    # circuit name must not produce an un-pasteable identifier.
    st = make_state(layouts={0: make_slot(name="C-1 café", qubits=["0", "1"])})
    code = get_qiskit_layout_code(st, "general", 1)
    m = re.search(r"layout_(\w+) = \[0, 1\]", code)
    assert m is not None
    assert ("layout_" + m.group(1)).isidentifier()


def test_codegen_lucj_concatenates_alpha_beta_other(make_state, make_slot):
    st = make_state(layouts={0: make_slot(name="C1", alpha=["0"], beta=["1"], other=["2"])})
    code = get_qiskit_layout_code(st, "lucj", 1)
    assert "[0, 1, 2]" in code
