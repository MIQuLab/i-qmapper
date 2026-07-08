# QPU Mapper

**Interactive visualization and layout selection for quantum-circuit ansätze on real QPU topologies.**

QPU Mapper is a Jupyter/ipywidgets tool for placing and evaluating qubit layouts
(LUCJ ansätze and general circuits) on superconducting QPU topologies — driven by
**real device calibration data**. It renders the chip as an interactive graph with
error heatmaps, lets you assign and style qubit roles, ranks qubits and gates by
fidelity, tracks calibration history with delta views, and exports layouts to JSON,
Qiskit code, PNG, and time-lapse GIF/MP4.

> Today it ships with first-class **IBM Quantum** support (heavy-hex Heron /
> Nighthawk). The backend layer is deliberately small and pluggable — see
> [Extending to other vendors](#extending-to-other-vendors).

<!-- TODO: add a short demo GIF here -->
<!-- ![QPU Mapper demo](docs/demo.gif) -->

---

## Features

- **Live calibration data** from IBM Quantum, with an on-disk JSON cache (atomic writes, per-backend).
- **Interactive chip view** (Plotly): error heatmaps with multiple colour scales, per-qubit/per-edge values on hover.
- **Layout design**: assign qubits to LUCJ roles (α / β / ancilla) or general roles; per-role colours, outlines, and styling; isolate/focus modes.
- **Auto-LUCJ placement** on heavy-hex chips (zig-zag layout + VF2 isomorph search + a mapomatic-style layout scorer with T1/T2 coherence penalties).
- **Rankings**: qubits and gates sorted by error, with search/filter; circuit-fidelity estimate.
- **Calibration history & delta**: fetch past snapshots, compare two dates, animate drift over time.
- **Exports**: layout JSON, Qiskit `transpile()` code, high-res PNG, and time-lapse GIF/MP4. Layout **import** from `.json` / `.npy` / `.qpy` / `.qasm`.
- **Sessions**: save/restore the full layout + styling to a `.qpusession.json` file.

## Requirements

- **Python 3.9+**, running inside **Jupyter Notebook** or **JupyterLab** (it's an ipywidgets app).
- Core: `ipywidgets`, `plotly`, `numpy`, `rustworkx`, `ipython`
- IBM live data: `qiskit`, `qiskit-ibm-runtime`
- PNG export: `kaleido` (Plotly's static-image engine)
- Time-lapse GIF/MP4 (optional): `imageio`, `imageio-ffmpeg`

## Installation

```bash
# clone
git clone https://github.com/mbazayeva/qpu_mapper.git
cd qpu_mapper

# core (IBM live data included via qiskit-ibm-runtime)
pip install -e .

# core + PNG export
pip install -e ".[png]"

# core + PNG + time-lapse GIF/MP4
pip install -e ".[png,timelapse]"
```

To use live IBM data you must have IBM Quantum credentials saved **once**, outside
the tool (the tool never reads or stores your token):

```python
from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(
    channel='ibm_quantum_platform',
    token='YOUR_TOKEN',
    instance='YOUR_CRN',
    overwrite=True,
)
```

## Quickstart

In a notebook cell:

```python
from qpu_mapper import run
run()
```

This opens the sidebar UI. Pick **IBM Quantum**, load your account, choose a backend,
select the ansatz, and click **Connect & Launch**. From there you can place qubits,
switch between design and error-heatmap views, inspect rankings and history, and
export.

---

## How it works

The pipeline is: **connect → fetch calibration → (optionally) generate a layout →
store in `SessionState` → render → edit → export.**

| Layer | Files |
|---|---|
| Entry / routing | `__init__.py` (`run()`), `connection.py` (auth UI), `mapper.py` (vendor routing) |
| State (single source of truth) | `state.py` (`SessionState`) |
| Backend / data & algorithms | `data.py` (calibration + cache + device graph), `auto_lucj.py` (LUCJ + scoring), `stability.py`, `constants.py` (coords / colours) |
| Rendering | `plot.py` (`refresh_plot` interactive ↔ `build_static_figure` export) |
| Export | `export/` (analysis, codegen, autodetect, layout_io, scoring, session, data) |
| UI | `ui/builder.py` (`MapperUIBuilder`) + `ui_*.py` panels + `ui/widgets_ref.py` |

A fuller architecture map for reviewers lives in [`CODE_OVERVIEW.md`](CODE_OVERVIEW.md).

---

## Extending to other vendors

The tool talks to a backend through a **small, explicit data contract**. Adding a
new vendor (Rigetti, IonQ, a real Google/Cirq backend, an internal simulator, …)
means satisfying that contract and wiring it through a handful of integration points.

### What the tool needs from any backend

1. **Topology**
   - **Qubit coordinates** — `(row, col)` positions used to lay out the chip graph.
   - **Coupling edges** — a list of `(u, v)` pairs of **qubit indices**.
2. **Calibration**
   - **Per-qubit properties** — at minimum `readout_error`; optionally `t1`, `t2`, and the other entries in `QUBIT_PROP_OPTIONS`.
   - **Per-edge properties** — a 2-qubit gate error keyed `cz_error` (one value per edge), plus any others in `GATE_PROP_OPTIONS`.

These populate `state.active_coords`, `state.edges_list`,
`state.qubit_props_data`, and `state.gate_props_data` — everything downstream
(rendering, rankings, fidelity) reads from those.

### The integration points

| File | What to add |
|---|---|
| `connection.py` | Add the vendor to the `dd_vendor` dropdown; add a branch in the **Load Account** handler (authenticate / instantiate the backend) and in the **Connect** handler (launch via `on_connect_callback`). |
| `mapper.py` | Coordinate resolution. IBM backends go through `resolve_coords()`. A custom backend can instead carry `.coords` on the object — add an `elif backend is not None and hasattr(backend, 'coords')` branch. |
| `data.py` | `generate_device_data()` dispatches by vendor — add a `_generate_<vendor>(state)` that fills `state.qubit_props_data` / `state.gate_props_data` / coords / edges. A Qiskit-compatible backend can reuse `_generate_ibm()` directly. |
| `constants.py` | If your chip has a fixed topology, add a `<chip>_coords()` function and wire it into `resolve_coords()`'s family / qubit-count dispatch. |

### Recipe A — a Qiskit-compatible vendor (easiest)

If your backend is a Qiskit `BackendV2` exposing `.coupling_map.get_edges()` and
`.properties()`, you mostly **reuse the IBM path**:

1. Add the dropdown option and a connection branch that instantiates your backend.
2. If the chip topology isn't already known to `resolve_coords()`, add a
   `<chip>_coords()` to `constants.py` and a matching branch (keyed on
   `processor_type` family or qubit count).
3. `_generate_ibm()` and `resolve_coords()` then work as-is, since they read the
   standard Qiskit interfaces.

### Recipe B — a custom (non-Qiskit) backend

Model it on the lightweight "backend object carries its own data" pattern. Provide
a class with the contract below:

```python
import numpy as np

class MyVendorBackend:
    name = "mychip_v1"

    def __init__(self):
        # Topology
        self._coords = [(r, c) for r in range(rows) for c in range(cols)]  # (row, col)
        self._edges  = [(0, 1), (1, 2), ...]                              # (u, v) index pairs
        n = len(self._coords)
        # Calibration (orders of magnitude matter; arrays must match sizes)
        self.readout_errors = np.full(n, 0.01)                # len == n_qubits
        self.gate_errors    = np.full(len(self._edges), 0.005)  # len == n_edges (2q error)
        self.t1_us = np.full(n, 100.0)   # optional
        self.t2_us = np.full(n, 80.0)    # optional

    @property
    def coords(self): return self._coords
    @property
    def edges(self):  return self._edges
```

Then wire it:

```python
# data.py — add the dispatch + generator
def _generate_<vendor>(state):
    b = state.active_backend
    n = len(b.coords)
    state.qubit_props_data = {
        'readout_error': b.readout_errors,
        't1': getattr(b, 't1_us', np.zeros(n)),
        't2': getattr(b, 't2_us', np.zeros(n)),
    }
    state.gate_props_data = {'cz_error': b.gate_errors}
```

`mapper.py`'s `hasattr(backend, 'coords')` branch picks up the coordinates, and
`connection.py` instantiates the backend and launches it via `on_connect_callback`.

> If you also want LUCJ/Qiskit-style consumers, expose a `coupling_map` object with
> a `.get_edges()` method returning `(u, v)` tuples (a thin wrapper is enough).

### Things to know

- **LUCJ auto-placement is heavy-hex / IBM-specific** (`auto_lucj.py`). Other
  topologies still get manual placement, error heatmaps, rankings, and exports —
  just not the automatic LUCJ layout.
- **Calibration history & delta** assume a queryable per-timestamp calibration
  source (modelled on IBM's). A custom vendor without that gets a single static
  snapshot unless you implement history fetching for it.
- **Edge convention**: `edges_list` holds **qubit-index** pairs; `plot.py` is the
  authoritative reference for the semantics.

---

## Project layout

```
qpu_mapper/
├── __init__.py            # run() entry point
├── connection.py          # auth / vendor-selection UI
├── mapper.py              # vendor routing → builds the app
├── state.py              # SessionState (single source of truth)
├── data.py               # calibration fetch + cache + device graph
├── auto_lucj.py          # LUCJ layout + VF2 + scoring
├── stability.py          # calibration-stability analysis
├── constants.py          # coordinates, colour scales, formatting
├── plot.py               # interactive + static Plotly rendering
├── export/               # analysis, codegen, autodetect, layout_io, scoring, session, data
├── ui/                   # builder.py (MapperUIBuilder) + widgets_ref.py
└── ui_*.py               # per-panel UI modules
```

## Limitations & caveats

- The **circuit-fidelity estimate assumes independent errors** (no crosstalk, no
  scheduling effects). It is an *upper-bound style* indicator for comparing
  layouts, not a hardware fidelity prediction.
- **`.qpy` import** deserializes a Qiskit binary format; only import `.qpy` files
  you trust (treat like opening any untrusted binary). `.json` / `.npy` / `.qasm`
  imports are parsed safely.
- LUCJ auto-placement targets IBM heavy-hex topologies (see above).

## Roadmap

Planned work includes a routing/SWAP-overhead preview, a
circuit-on-topology overlay, a crosstalk model, and additional ansätze.

## License, Credits & Citing

QPU Mapper is released under the [Apache License 2.0](LICENSE) (see also [NOTICE](NOTICE)).

Parts of this project build on IBM open-source work: the auto-LUCJ layout
scorer was adapted from IBM code now incorporated in
[ffsim](https://github.com/qiskit-community/ffsim)
(Sung *et al.*, [arXiv:2605.03123](https://arxiv.org/abs/2605.03123)),
and the layout-quality estimator implements a cost function in the style of
[mapomatic](https://github.com/qiskit-community/mapomatic)
(Nation & Treinish, *PRX Quantum* **4**, 010327, 2023). The LUCJ ansatz is
due to Motta *et al.*, *Chem. Sci.* **14**, 11213–11227 (2023),
[doi:10.1039/D3SC02516K](https://doi.org/10.1039/D3SC02516K).

If you use QPU Mapper in your research, please cite it — see
[CITATION.cff](CITATION.cff).
