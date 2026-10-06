# I-QMapper — Code Overview

A guided tour of the codebase for contributors and reviewers. For user-facing
documentation, see the [README](README.md).

## The big picture

```
IBM Quantum API ──► data.py (fetch + per-user cache)
                         │
                         ▼
state.py  ◄──────  calibration snapshots (live / intraday / multi-day)
   │
   ├──► plot.py (device graph, heatmaps, delta rendering)
   ├──► auto_lucj.py (LUCJ auto-placement + layout error scoring)
   ├──► layout search: ancilla_sweep / guided_search / engine_select
   │      └─► multi_sweep / joint_plan (several circuits), qpu_scan (several QPUs)
   ├──► stability.py (multi-day stability analysis)
   └──► export/ (LQS scoring, layout I/O, codegen, PNG/analysis export)
                         ▲
                         │
ui/ (ipywidgets panels) ─┘   ── every panel reads/writes state, never the API directly
```

Design rules that hold across the codebase:

- **The kernel is the trust boundary.** Credentials live in the user's local
  `QiskitRuntimeService` store; the tool never sees, stores, or transmits a token.
- **Analysis runs on cached snapshots.** Once history is fetched, scoring,
  stability, and delta views cost no API calls and no QPU time.
- **UI panels are thin.** Logic lives in the core/`export` modules; panels wire
  widgets to it. Background work is marshaled back to the main loop before any
  widget is touched (see `async_util.py`).
- **Untrusted input is hardened.** Every import path (JSON / `.npy` / QPY /
  sessions) enforces size caps, validates shapes, and fails with a clear
  message instead of crashing (`allow_pickle=False` on `.npy`, qubit IDs
  filtered against the device, malformed payloads fall back safely).

## Top-level modules

| Module | Role |
|---|---|
| `connection.py` | Account loading UI, the `_VENDOR_LOADERS` dispatch table (extension point for non-IBM providers), backend listing and selection. Supports multiple named `QiskitRuntimeService` accounts. |
| `data.py` | Calibration fetch (live, by timestamp, history ranges) and the per-user on-disk cache (`platformdirs`, atomic writes, per-backend index locking, `IQMAPPER_CACHE_DIR` override). |
| `plot.py` | Device-graph construction and all heatmap/delta rendering (Plotly `FigureWidget`), colorbars, and the export legend. |
| `mapper.py` | Thin entry point: receives the connection parameters from `__init__.py`, builds the `SessionState` and launches the UI. |
| `auto_lucj.py` | LUCJ zig-zag auto-placement on heavy-hex topologies and `lightweight_layout_error_scoring`. Portions adapted from IBM code now incorporated in [ffsim](https://github.com/qiskit-community/ffsim) — see the file header and [NOTICE](NOTICE). |
| `stability.py` | Pure, offline multi-day stability analysis over cached history: which qubits/edges stay good across a 7/14/30-day window. |
| `benchmark.py` | Opt-in real-time benchmarking: re-measures T1/T2, readout, 1Q and 2Q RB on the candidate qubits with `qiskit-experiments` and feeds the result back as a "measured" calibration epoch. Optional `[benchmark]` extra. |
| `state.py` | The single `SessionState`: layouts, colors, zoom, time-navigation indices, undo/redo (capped history), and versioned session (de)serialization. |
| `paths.py` | Path-traversal-safe file resolution (`commonpath` containment, symlink-aware) for every read/write the user can influence. |
| `html_safe.py` | The one `esc()` used by every HTML widget sink. |
| `async_util.py` | `default_spawn_bg`: background threads with callbacks marshaled to the main loop; errors are contained and always reach the `on_error` handler. |
| `_upload.py` | Normalizes ipywidgets `FileUpload` payloads across widget versions into `(bytes, filename)`. |
| `constants.py` | Device coordinate tables (Heron 156q, Nighthawk 120q), color mapping factories, palette persistence. |

## Layout search

All single-circuit engines return the same result dict (strata per ancilla
count, provenance, truncation flags), so the display code never needs to know
which one ran.

| Module | Role |
|---|---|
| `ancilla_sweep.py` | The exhaustive engine: one VF2 solve at the maximum ancilla count, a bridge-subtraction cascade down to the floor, optional direct VF2 per lower count, diversity selection. Capped at 2000 mappings per stratum. |
| `reservoir.py` | Makes the VF2 cap count *distinct* candidates: near-duplicate layouts are discarded without using a slot, so the capped sample spreads across the chip. |
| `guided_search.py` | The guided engine: calibration-led beam search that grows layouts on the best neighbours, then a 1-opt polish against the circuit-level LQS. |
| `engine_select.py` | The hybrid engine: guided for ancilla counts up to a threshold, exhaustive above it, merged into the shared result shape. |
| `stability_filter.py` | Stability-aware sweep: excludes qubits that are volatile across cached history (and statistical outliers), then runs the exhaustive sweep, relaxing the filter when the circuit would no longer fit. |
| `multi_sweep.py` | Sequential multi-circuit placement: each circuit greedily on the best region left, with a buffer around placed ones; replan from any circuit. |
| `joint_plan.py` | Joint multi-circuit planner: a beam over placement plans with an injected single-circuit search, memoised searches and boundary-coupler accounting. Same result shape as `multi_sweep`, never worse than it. |
| `qpu_scan.py` | Multi-QPU scan: lists compatible backends and runs a sweep on each, isolating per-backend failures. |
| `props_shim.py` | Cache-aware properties adapter for the scan: serves a backend's cached calibration arrays through the `readout_error` / `gate_error` / `score_layout` interface the engines read. |
| `evaluation.py` | Offline engine comparison: replays VF2 and guided over cached calibration snapshots and tabulates per-stratum differences (CLI: `python -m iqmapper.evaluation`). |

## `export/` — analysis & I/O

| Module | Role |
|---|---|
| `scoring.py` | `estimate_circuit_fidelity` — the Layout-Quality Score (LQS): product-form fidelity estimate from readout and two-qubit gate errors. Cost function in the style of [mapomatic](https://github.com/qiskit-community/mapomatic) (Nation & Treinish, PRX Quantum 4, 010327, 2023). |
| `layout_io.py` | Layout import from `.json` / `.npy` / `.qpy`, with size caps and hardened parsing; layout export to JSON. |
| `codegen.py` | Generates ready-to-run Qiskit code for the chosen layout. |
| `autodetect.py` | Heuristics that classify uploaded files and detect layout shapes. |
| `session.py` | Session and style-preset save/load (size-capped, versioned). |
| `data.py` | Analysis-export helpers (tabular views of calibration data). |
| `analysis.py` | Per-qubit / per-gate rankings, trends, connectivity and stable-element finders. Pure data + HTML. |

## `ui/` — ipywidgets panels

| Module | Role |
|---|---|
| `builder.py` | The main assembler: constructs every panel, owns the sidebar, wires callbacks, holds the `ctx` namespaces passed to panels. |
| `time_panel.py` | Time navigation: live / point / intraday / multi-day modes, snapshot sliders, delta reference selection. |
| `history.py` | Multi-day history fetch and the time-lapse GIF/MP4 export (optional `imageio` / `imageio-ffmpeg`). |
| `analysis_panel.py` | LQS display, layout rankings, and the Stability Check section. |
| `lucj_panel.py` | LUCJ controls: the single dispatch points for the search engine (`_run_sweep`) and the multi-circuit planner (`_run_multi` / `_replan_multi`), the sweep and scan displays, and replan. |
| `benchmark_panel.py` | Sidebar section "5. Benchmark": targets, cost estimate, run, and applying the measured epoch. |
| `inter_qpu.py` | Multi-backend comparison view. |
| `png_export.py` | Publication PNG export (optional `kaleido`). |
| `session_io.py` | Session save/load UI. |
| `backend_switch.py`, `color_picker.py`, `_sidebar_css.py` | Backend switching, palette editing, styling. |
| `pretty_json.py`, `status.py`, `widgets_ref.py` | JSON preview rendering, the shared inline status banner, typed references to widgets shared between modules. |

**Shared layout `W`.** `builder.py` creates one `Layout` object, `W`, and
passes it to dozens of widgets. Never change it at runtime (e.g.
`.layout.display = 'none'` to hide something): every widget that shares it
changes too, so hiding one hides about 86. Any widget whose layout is changed
at runtime gets its own `Layout(...)`, and its UI test asserts it is not
`W` (`widget.layout is not builder.txt_auto.layout`).

## Tests

`tests/` — deterministic and fully offline (no QPU, no network, no
credentials). A few skip when an optional dependency is missing:
the QPY round-trip without qiskit, and `test_benchmark_integration.py` without
the `[benchmark]` extra. Coverage concentrates on the pure logic: the search
engines and planners (on synthetic lattices from
`rustworkx.generators.heavy_hex_graph` with seeded calibrations), path safety,
state round-trips and undo/redo, layout import hardening (including
hostile-input regressions), scoring, the headless UI, the copy-length guard,
and HTML escaping. Run with:

```bash
pip install -e ".[dev]"
python -m pytest -q
```
