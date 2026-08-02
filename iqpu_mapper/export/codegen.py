"""
export.codegen — Generate Qiskit `transpile()` code snippets from the
current layout, suitable for direct paste into a user notebook.
"""


def get_qiskit_layout_code(state, mode, n_circ):
    """Generate Python code snippet for Qiskit transpile() with initial_layout.

    Returns a string of Python code the user can paste directly.
    """
    lines = [
        "from qiskit.transpiler import Layout",
        "",
    ]
    for i in range(n_circ):
        d = state.layouts_data[i]
        if mode == 'lucj':
            qubits = [int(q) for q in d['alpha'] + d['beta'] + d.get('other', []) if q]
        else:
            qubits = [int(q) for q in d.get('qubits', []) if q]
        if not qubits:
            continue
        name = d.get('name', f'C{i+1}')
        # Strip newlines: `name` comes from session JSON (not validated by
        # from_session_dict); a `\n` in the pasted snippet would inject
        # live Python code below what looks like a comment.
        name = str(name).replace('\n', ' ').replace('\r', ' ')
        # Generate initial_layout dict: virtual qubit → physical qubit
        layout_dict = {v: p for v, p in enumerate(qubits)}
        lines.append(f"# {name}: {len(qubits)} qubits")
        lines.append(f"initial_layout_{i+1} = {layout_dict}")
        lines.append("# transpiled = transpile(circuit, backend=backend,")
        lines.append(f"#     initial_layout=list(initial_layout_{i+1}.values()))")
        lines.append("")

    # Also generate a flat list version (most common use)
    import re as _re
    for i in range(n_circ):
        d = state.layouts_data[i]
        if mode == 'lucj':
            qubits = [int(q) for q in d['alpha'] + d['beta'] + d.get('other', []) if q]
        else:
            qubits = [int(q) for q in d.get('qubits', []) if q]
        if not qubits:
            continue
        name = d.get('name', f'C{i+1}')
        # Strip newlines: `name` comes from session JSON (not validated by
        # from_session_dict); a `\n` in the pasted snippet would inject
        # live Python code below what looks like a comment.
        name = str(name).replace('\n', ' ').replace('\r', ' ')
        # Coerce `name` to a valid Python identifier — punctuation, accents,
        # or a leading digit would make the pasted snippet a SyntaxError.
        safe_name = _re.sub(r'[^a-z0-9_]', '_', name.lower())
        safe_name = _re.sub(r'_+', '_', safe_name).strip('_')
        if not safe_name or safe_name[0].isdigit():
            safe_name = f"c{i+1}" if not safe_name else f"c{i+1}_{safe_name}"
        lines.append(f"layout_{safe_name} = {qubits}  # physical qubits, ordered")

    return '\n'.join(lines)
