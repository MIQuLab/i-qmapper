"""Tests for qpu_mapper.multi_sweep — the sequential multi-circuit orchestrator.

Offline, deterministic. Two synthetic hosts:
  * TWO-BLOCK: two disjoint width-8 ladder blocks (ids 0..23 and 24..47), the
    same host used by test_ancilla_sweep — one circuit lands in each block.
  * CONNECTED: the two blocks joined by a single coupling edge, so the buffer
    expansion actually reaches across the boundary (a real buffer test).
"""
from types import SimpleNamespace

import rustworkx

from qpu_mapper.multi_sweep import sweep_multi_circuits, replan_from


def _block(base, edges):
    top = [base + j for j in range(8)]
    bot = [base + 8 + j for j in range(8)]
    brg = [base + 16 + j for j in range(8)]
    for j in range(7):
        edges += [(top[j], top[j + 1]), (bot[j], bot[j + 1])]
    for j in range(8):
        edges += [(top[j], brg[j]), (brg[j], bot[j])]


def _make_backend(link=None):
    edges = []
    _block(0, edges)
    _block(24, edges)
    if link:
        edges.append(link)
    g = rustworkx.PyDiGraph()
    for i in range(48):
        g.add_node(i)
    for u, v in edges:
        g.add_edge(u, v, None)
    return SimpleNamespace(coupling_map=SimpleNamespace(graph=g),
                           operation_names=['cz', 'sx', 'rz', 'x'], name='host')


class _Props:
    def readout_error(self, q):
        return 0.01

    def gate_error(self, name, edge):
        return 0.005


def _qubits(entry):
    return set(entry['alpha']) | set(entry['beta']) | set(entry['ancillas'])


def _neighbors(backend):
    """Undirected adjacency {q: {neighbours}} from the backend coupling graph."""
    adj = {}
    for u, v in backend.coupling_map.graph.edge_list():
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    return adj


# ── two circuits land in their own regions, no overlap, buffer clear ────

def test_two_circuits_disjoint_own_regions():
    bk = _make_backend()
    res = sweep_multi_circuits([8, 8], bk, _Props(), buffer_qubits=1,
                               shuffle_seed=None)
    assert res['failed_at'] is None and len(res['circuits']) == 2
    c0 = _qubits(res['circuits'][0]['chosen'])
    c1 = _qubits(res['circuits'][1]['chosen'])
    assert c0.isdisjoint(c1)                                # no qubit overlap
    # Each circuit sits wholly in one block.
    assert (max(c0) < 24) != (max(c1) < 24)
    # Buffer respected: no circuit-1 qubit is a coupling neighbour of circuit-0.
    adj = _neighbors(bk)
    closed0 = set(c0)
    for q in c0:
        closed0|= adj.get(q, set())
    assert c1.isdisjoint(closed0)


def test_buffer_reaches_across_a_boundary():
    # Blocks joined at 7—24: the buffer of a circuit using the boundary must
    # keep the neighbour clear. Assert the general 1-hop-buffer invariant.
    bk = _make_backend(link=(7, 24))
    res = sweep_multi_circuits([8, 8], bk, _Props(), buffer_qubits=1,
                               shuffle_seed=None)
    assert len(res['circuits']) == 2
    c0 = _qubits(res['circuits'][0]['chosen'])
    c1 = _qubits(res['circuits'][1]['chosen'])
    adj = _neighbors(bk)
    for q in c0:
        assert c1.isdisjoint(adj.get(q, set())), \
            "a circuit-1 qubit is adjacent to circuit-0 (buffer not applied)"


# ── total_product == product of the chosen entries' product_scores ──────

def test_total_product_is_product_of_chosen():
    res = sweep_multi_circuits([8, 8], _make_backend(), _Props(),
                               shuffle_seed=None)
    expect = 1.0
    for c in res['circuits']:
        expect *= c['chosen']['product_score']
    assert abs(res['total_product'] - expect) < 1e-12


# ── later circuit's WHOLE strata avoid the earlier region ∪ buffer ──────

def test_sequential_exclusion_covers_all_strata():
    bk = _make_backend()
    res = sweep_multi_circuits([8, 8], bk, _Props(), buffer_qubits=1,
                               shuffle_seed=None)
    c0 = _qubits(res['circuits'][0]['chosen'])
    adj = _neighbors(bk)
    forbidden = set(c0)
    for q in c0:
        forbidden |= adj.get(q, set())
    # EVERY candidate in circuit 1's strata avoids circuit 0's region + buffer.
    for entries in res['circuits'][1]['strata'].values():
        for e in entries:
            assert _qubits(e).isdisjoint(forbidden)


# ── fit failure is honest — failed_at set, partial returned, no raise ───

def test_fit_failure_sets_failed_at():
    res = sweep_multi_circuits([8, 8, 8], _make_backend(), _Props(),
                               shuffle_seed=None)
    assert res['failed_at'] == 2                 # third circuit cannot fit
    assert len(res['circuits']) == 2             # first two placed
    assert res['total_product'] is not None      # partial product over placed


# ── replan_from overrides a circuit and re-sweeps the tail ──────────────

def test_replan_from_reoptimizes_tail():
    bk = _make_backend()
    res = sweep_multi_circuits([8, 8], bk, _Props(), shuffle_seed=None)
    c1_before = _qubits(res['circuits'][1]['chosen'])

    # An override for circuit 0 in the OTHER block than it originally chose.
    orig0 = _qubits(res['circuits'][0]['chosen'])
    orig0_low = max(orig0) < 24
    other = None
    for entries in res['circuits'][0]['strata'].values():
        for e in entries:
            if (max(_qubits(e)) < 24) != orig0_low:      # the other block
                other = e
                break
        if other:
            break
    assert other is not None, "expected a circuit-0 candidate in the other block"

    res2 = replan_from(res, 0, other)
    # Circuit 0 is now the override; circuit 1 re-swept to the freed region.
    assert res2['circuits'][0]['chosen'] is other
    c0_after = _qubits(res2['circuits'][0]['chosen'])
    c1_after = _qubits(res2['circuits'][1]['chosen'])
    assert c0_after.isdisjoint(c1_after)
    # Circuit 1 moved to the block circuit 0 vacated → different from before.
    assert c1_after != c1_before


# ── knob forwarding — per-circuit min_bridges + stability exclusions ────

def test_min_bridges_list_applies_per_circuit():
    res = sweep_multi_circuits([8, 8], _make_backend(), _Props(),
                               min_bridges=[1, 2], shuffle_seed=None)
    assert min(res['circuits'][0]['strata']) == 1      # floor 1 → k=2,1
    assert min(res['circuits'][1]['strata']) == 2      # floor 2 → k=2 only


def test_stability_exclusions_compose():
    # Pre-exclude a chain qubit in block A; it must appear in NO circuit.
    res = sweep_multi_circuits([8, 8], _make_backend(), _Props(),
                               exclude_qubits={0, 1, 2}, shuffle_seed=None)
    placed = set(res['placed_qubits'])
    assert placed.isdisjoint({0, 1, 2})


# ── determinism — same inputs + seed → identical result ─────────────────

def test_determinism():
    def run():
        return sweep_multi_circuits([8, 8], _make_backend(), _Props(),
                                    shuffle_seed=7)
    a, b = run(), run()

    def sig(r):
        return [(_qubits(c['chosen']), round(c['chosen']['product_score'], 12))
                for c in r['circuits']]
    assert sig(a) == sig(b)
    assert a['placed_qubits'] == b['placed_qubits']
    assert a['buffer_qubits_used'] == b['buffer_qubits_used']
