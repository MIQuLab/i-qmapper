# Changelog

All notable changes to this project are documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Opt-in real-time benchmarking (`iqmapper.benchmark`, extra
  `pip install "i-qmapper[benchmark]"`): three tiers that re-measure the
  candidate qubits and edges with `qiskit-experiments` and feed the result
  back as a "measured" calibration epoch, so the heatmap, LQS, re-ranking and
  export all read fresh numbers. Never automatic, never inside the multi-QPU
  scan, and the QPU cost estimate is shown before the run can be started.
  Lives in the sidebar as "5. Benchmark" with selectable targets (current
  layouts / sweep candidates / whole QPU / a custom qubit list) and, in
  inter-QPU mode, a QPU 1 | QPU 2 selector.
- Stratified ancilla sweep: a single VF2 solve at `k_max` feeding a
  bridge-subtraction cascade down to `k_min`, with a hybrid mode, a shuffled
  search for hard instances, and region-diversity selection across the chip.
- Stability filter with an always-on quality floor, so layouts that score well
  on one calibration epoch but drift across neighbouring ones are rejected.
- Multi-circuit sweep: sequential orchestration over several circuits with
  replan-from-step to recover a run without restarting it.
- Multi-QPU scan across the available backends, with one-click inter-QPU
  handoff that reopens the chosen layout on the second QPU.
- Epoch-consistent scoring: fidelity and ranking read a single pinned
  calibration epoch, so comparisons within a sweep stay commensurable.

## [0.0.5] — 2026-08

First public-ready release.

### Added
- Interactive QPU layout mapper for Jupyter (LUCJ + general circuits) on
  IBM heavy-hex backends: live calibration heatmaps, history/delta views,
  auto-LUCJ placement, layout scoring/rankings, JSON/PNG/code exports,
  time-lapse, session save/restore.
- Layout import from .json / .npy / .qpy  with hardened parsing
  (size caps, allow_pickle=False, partition validation).
- Per-user calibration cache (platformdirs) with atomic writes.
- Test suite (175 tests), Apache-2.0 licensing with IBM attributions.
- Companion paper: "I-QMapper: Error-Aware Layout Optimization and
  Device Diagnostics for NISQ Hardware" (arXiv:2606.27508).

### Security
- Path-traversal-safe file resolution (commonpath containment); HTML
  escaping across all widget sinks; malformed-input fallbacks on all
  untrusted import paths.
