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
| `data.py` | Calibration fetch (live, by timestamp, history ranges) and the per-user on-disk cache (`platformdirs`, atomic writes, per-backend index locking, `IQPU_MAPPER_CACHE_DIR` override). |
| `plot.py` | Device-graph construction and all heatmap/delta rendering (Plotly `FigureWidget`), colorbars, and the export legend. |
| `auto_lucj.py` | LUCJ zig-zag auto-placement on heavy-hex topologies and `lightweight_layout_error_scoring`. Portions adapted from IBM code now incorporated in [ffsim](https://github.com/qiskit-community/ffsim) — see the file header and [NOTICE](NOTICE). |
| `stability.py` | Pure, offline multi-day stability analysis over cached history: which qubits/edges stay good across a 7/14/30-day window. |
| `state.py` | The single `SessionState`: layouts, colors, zoom, time-navigation indices, undo/redo (capped history), and versioned session (de)serialization. |
| `paths.py` | Path-traversal-safe file resolution (`commonpath` containment, symlink-aware) for every read/write the user can influence. |
| `html_safe.py` | The one `esc()` used by every HTML widget sink. |
| `async_util.py` | `default_spawn_bg`: background threads with callbacks marshaled to the main loop; errors are contained and always reach the `on_error` handler. |
| `_upload.py` | Normalizes ipywidgets `FileUpload` payloads across widget versions into `(bytes, filename)`. |
| `constants.py` | Device coordinate tables (Heron 156q, Nighthawk 120q), color mapping factories, palette persistence. |

## `export/` — analysis & I/O

| Module | Role |
|---|---|
| `scoring.py` | `estimate_circuit_fidelity` — the Layout-Quality Score (LQS): product-form fidelity estimate from readout and two-qubit gate errors. Cost function in the style of [mapomatic](https://github.com/qiskit-community/mapomatic) (Nation & Treinish, PRX Quantum 4, 010327, 2023). |
| `layout_io.py` | Layout import from `.json` / `.npy` / `.qpy`, with size caps and hardened parsing; layout export to JSON. |
| `codegen.py` | Generates ready-to-run Qiskit code for the chosen layout. |
| `autodetect.py` | Heuristics that classify uploaded files and detect layout shapes. |
| `session.py` | Session and style-preset save/load (size-capped, versioned). |
| `data.py` | Analysis-export helpers (tabular views of calibration data). |

## `ui/` — ipywidgets panels

| Module | Role |
|---|---|
| `builder.py` | The main assembler: constructs every panel, owns the sidebar, wires callbacks, holds the `ctx` namespaces passed to panels. |
| `time_panel.py` | Time navigation: live / point / intraday / multi-day modes, snapshot sliders, delta reference selection. |
| `history.py` | Multi-day history fetch and the time-lapse GIF/MP4 export (optional `imageio` / `imageio-ffmpeg`). |
| `analysis_panel.py` | LQS display, layout rankings, and the Stability Check section. |
| `lucj_panel.py` | LUCJ-specific controls (auto-placement, ancilla/bridge handling). |
| `inter_qpu.py` | Multi-backend comparison view. |
| `png_export.py` | Publication PNG export (optional `kaleido`). |
| `session_io.py` | Session save/load UI. |
| `backend_switch.py`, `color_picker.py`, `_sidebar_css.py` | Backend switching, palette editing, styling. |

## Tests

`tests/` — 183 tests, deterministic and fully offline (no QPU, no network, no
qiskit required except one optional QPY round-trip that skips when qiskit is
absent). Coverage concentrates on the pure logic: path safety, state
round-trips and undo/redo, upload normalization, layout import hardening
(including hostile-input regressions), scoring, time-status describers, and
HTML escaping. Run with:

```bash
pip install -e ".[dev]"
python -m pytest -q
```
