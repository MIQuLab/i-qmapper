# I-QMapper

Interactive, calibration-aware qubit layout selection for IBM Quantum processors.

[![tests](https://github.com/MIQuLab/i-qmapper/actions/workflows/tests.yml/badge.svg)](https://github.com/MIQuLab/i-qmapper/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/i-qmapper.svg)](https://pypi.org/project/i-qmapper/)
[![arXiv](https://img.shields.io/badge/arXiv-2606.27508-b31b1b.svg)](https://arxiv.org/abs/2606.27508)

I-QMapper is a Jupyter application that displays a quantum processor's calibration
data on its physical topology, tracks how that data changes over time, and places
circuits — in particular the LUCJ ansatz used in quantum chemistry — on the qubits
where they are expected to run best. Every placement decision is visible and
reproducible: the device, its calibration history and the candidate layouts are
shown in one view, and each candidate is ranked by an explicit quality score.

![LUCJ layout placed automatically on a 156-qubit IBM Heron processor](docs/img/hero.png)

## Installation

```bash
pip install i-qmapper
```

| Extra | Adds | Install |
|---|---|---|
| `png` | PNG export of the map (kaleido) | `pip install "i-qmapper[png]"` |
| `timelapse` | GIF/MP4 export of calibration history | `pip install "i-qmapper[timelapse]"` |
| `benchmark` | On-device benchmarking (qiskit-experiments) | `pip install "i-qmapper[benchmark]"` |

Extras can be combined, e.g. `"i-qmapper[png,benchmark]"`.

Requirements: Python 3.10 or later, JupyterLab or Jupyter Notebook, and an
[IBM Quantum](https://quantum.ibm.com) account whose credential is saved locally
with `qiskit-ibm-runtime`:

```python
from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(channel="ibm_quantum_platform",
                                  token="YOUR_TOKEN", instance="YOUR_CRN",
                                  set_as_default=True, overwrite=True)
```

I-QMapper reads this credential through `qiskit-ibm-runtime`; it does not store or
transmit the token itself. Several credentials can be saved under different
`name=` values and selected from the application.

## Quick start

```python
from iqmapper import run
run()
```

1. In the sidebar, select the account and the backend, then **Connect & Launch**.
2. The map shows the processor with the selected calibration property as a
   colour scale; hover a qubit or coupler for its values.
3. Choose the LUCJ mode and the number of spatial orbitals, then **Generate LUCJ
   Layout**. The candidates are listed with their quality score (LQS) and drawn
   on the map; the Estimated Layout-Quality panel gives the score of the
   current layout.
4. Use the calibration section to view earlier calibrations and the change
   since a chosen reference, and the stability check to see which qubits have
   remained reliable over the last 7, 14 or 30 days.
5. Export the layout as JSON, Qiskit code, PNG, or save the whole session.

A step-by-step walkthrough is in the [tutorial](TUTORIAL.md).

## Capabilities

| | |
|---|---|
| **Calibration map** | Readout error, two-qubit gate error, T1, T2 and the other reported properties on the device topology, for the current calibration or any cached earlier one. |
| **History and drift** | Intraday snapshots, multi-day history, and difference views against a chosen reference calibration. |
| **Stability check** | Per-qubit reliability across a 7-, 14- or 30-day window, used to exclude qubits that score well today but drift. |
| **Layout quality score (LQS)** | A product-form estimate built from readout and two-qubit gate errors of the qubits and couplers a circuit uses. Used to rank candidates before any QPU time is spent. |
| **Automatic LUCJ placement** | An exhaustive subgraph search and a calibration-guided search run together; for every ancilla count the best candidates of both are kept. |
| **Several circuits on one processor** | Joint placement of multiple circuits with a configurable buffer between them; complete plans are ranked by their combined score. |
| **Several processors** | The same search on every accessible backend, ranked, with a one-step transfer of a layout to a second processor. |
| **On-device benchmarking** (optional) | Re-measurement of readout, gate errors and coherence on the chosen qubits, fed back into the map and the ranking as a separate calibration epoch. Never automatic; cost shown before the run. |
| **Import and export** | Layouts from JSON, `.npy` or QPY; export to PNG, GIF/MP4, JSON, Qiskit code and full session files. |

## Repository layout

| Path | Contents |
|---|---|
| `iqmapper/` | The application: connection and calibration cache, device map, layout search engines, scoring, benchmarking, session state |
| `iqmapper/ui/` | The Jupyter interface (ipywidgets panels) |
| `iqmapper/export/` | Layout quality score, layout import/export, code generation, PNG export |
| `tests/` | Offline test suite; no IBM account required |
| `docs/` | User guide, benchmarking and development documentation |
| `TUTORIAL.md` | Step-by-step walkthrough |
| `CHANGELOG.md` | Release history |

A module-by-module description is in [CODE_OVERVIEW.md](CODE_OVERVIEW.md).

## Related project

| Project | Purpose |
|---|---|
| [ewf-geom-opt](https://github.com/MIQuLab/ewf-geom-opt) | Geometry optimisation with embedded wavefunction (EWF) energies, including quantum sampling of the fragment problems; I-QMapper selects the qubit layouts on which those circuits run |

## Supported processors

The device map covers the 156-qubit IBM Heron (r2, r3) and the 120-qubit IBM
Nighthawk processors. Other backends are listed in the multi-processor scans as
not supported and are not searched.

## Documentation

- [Tutorial](TUTORIAL.md) — from launch to an exported layout.
- [User guide](https://github.com/MIQuLab/i-qmapper/blob/main/docs/user-guide.md) — settings, search options, messages, and the definition of the quality score.
- [Benchmarking](https://github.com/MIQuLab/i-qmapper/blob/main/docs/benchmark.md) — levels, cost, and how measured values are reconciled with the reported calibration.
- [Development](https://github.com/MIQuLab/i-qmapper/blob/main/docs/development.md) — architecture, data locations, tests, and adding a vendor.
- [Changelog](CHANGELOG.md)

## Citing

If you use I-QMapper in your research, please cite:

> M. Bazayeva, K. M. Merz Jr., *I-QMapper: Error-Aware Layout Optimization and
> Device Diagnostics for NISQ Hardware*, arXiv:2606.27508 (2026).
> https://arxiv.org/abs/2606.27508

### Works used

1. M. Motta, K. J. Sung, K. B. Whaley, M. Head-Gordon, J. Shee, *Chem. Sci.*
   **14**, 11213 (2023). The LUCJ ansatz.
2. P. D. Nation, M. Treinish, *PRX Quantum* **4**, 010327 (2023). Cost function
   followed by the layout quality score (mapomatic).
3. K. J. Sung *et al.*, arXiv:2605.03123 (2026). The LUCJ layout scorer was
   adapted from code now part of [ffsim](https://github.com/qiskit-community/ffsim).

## License

Apache License 2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
