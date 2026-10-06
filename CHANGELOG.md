# Changelog

All notable changes to this project are documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Opt-in real-time benchmarking (`iqmapper.benchmark`, extra
  `pip install "i-qmapper[benchmark]"`): three levels that re-measure the
  candidate qubits and edges with `qiskit-experiments` and feed the result
  back as a "measured" calibration epoch, so the heatmap, LQS, re-ranking and
  export all read fresh numbers. Never automatic, never inside the multi-QPU
  scan, and the QPU cost estimate is shown before the run can be started.
  Lives in the sidebar as "5. Benchmark" with selectable targets (current
  layouts / sweep candidates / whole QPU / a custom qubit list) and, in
  inter-QPU mode, a QPU 1 | QPU 2 selector. The experiment plan is composite
  (one `ParallelExperiment` per kind, edge-coloured layers for two-qubit RB),
  so a full-chip Level 3 stays small enough for IBM to accept, and is split
  across jobs when it would exceed IBM's per-job shot limit. A rejected or
  errored job now reports IBM's own message with the job ids, and a run that
  measured nothing is reported as a failure instead of being applied and
  labelled "measured". Measured values are checked against the reported
  calibration after every run; a systematic offset (median ratio outside
  0.5x–2x)
  is treated as an estimator mismatch and the epoch is applied in relative
  mode, preserving per-element ratios so outlying couplers still stand out.
  Two-qubit RB now subtracts the measured single-qubit error rates, and a
  calibration-check mode separates estimator bias from crosstalk.
- Stratified ancilla sweep (the default, exhaustive engine): one VF2 solve at
  the maximum ancilla count feeding a bridge-subtraction cascade down to the
  minimum, plus a direct search per ancilla count ("Deep search per ancilla
  count", on by default), a shuffled search order so a capped search samples
  the whole chip, and region-diversity selection ("Prefer diverse regions").
- Diverse enumeration (on by default): layouts that are near-copies of one
  already kept no longer use up the search cap, so a capped search spreads
  across the chip instead of filling up with variations of the first region
  it finds.
- Guided search engine (opt-in via "Search engine"; Exhaustive stays the
  default): a calibration-led beam search that grows layouts on the
  best-calibrated qubits, then polishes the finalists with small local moves
  (swap a bridge or chain-end qubit, shift a chain by one) kept only when they
  raise the circuit-level LQS.
- Opt-in stability filter that excludes qubits volatile across the cached
  calibration history, and an always-on quality floor (3σ by default, 0 turns
  it off) that removes outlier qubits from every search.
- Multi-circuit sweep: places several circuits one after another; picking a
  different layout for one circuit re-plans the circuits after it.
- Multi-QPU scan across the available backends, with one-click inter-QPU
  handoff that reopens the chosen layout on the second QPU.
- Epoch-consistent scoring: fidelity and ranking read a single pinned
  calibration epoch, so comparisons within a sweep stay commensurable.
- Hybrid search engine (opt-in; the default stays exhaustive): guided search
  for ancilla counts up to a threshold you set ("Guided up to k"), where the
  exhaustive search hits its cap and only ranks a sample, and the exhaustive
  search above it.
- Joint multi-circuit planner (opt-in; Sequential stays the default): keeps
  several placement plans at once instead of placing each circuit greedily,
  so the first circuit no longer takes a region the next one needed. Never
  worse than the sequential planner, uses the selected search engine, and
  reports couplers shared by two circuits at buffer 0 (with an optional
  penalty to avoid them).
- Engine comparison harness (`python -m iqmapper.evaluation`): replays the
  search engines over the cached calibrations, offline, and ranks with the
  app's real circuit-level LQS rebuilt for every calibration (`--scorer lqs`,
  the default; `proxy` uses the product score). It takes the coupling map
  from a saved session (`--session PATH`, or found in the working folder),
  can add a hybrid-engine column (`--hybrid-k`), and prints the suggested
  "Guided up to k" for each orbital count (`--tol` sets how far guided may
  trail and still count).
- Shorter UI messages: status lines, notices and tooltips say what happened
  in one short sentence, and each is explained in the README section
  "Understanding the messages".

### Fixed
- LQS counted every two-qubit gate twice on backends that list each coupler
  in both directions (CZ devices such as Heron), so their scores were too low
  and could not be compared with ECR devices such as Eagle, which list one
  direction. Each coupler now counts once, in the Layout-Quality box, the
  sweep ranking and the multi-QPU scan alike.
- The guided search could return layouts in which two qubits were joined by
  a coupler the ansatz does not use (for example the two ends of a chain that
  wraps round a hexagon). The LQS counted that coupler as a gate, so such a
  layout was scored, and could be ranked, on a gate the circuit never runs.
  Guided layouts now contain only the ansatz's own couplers, as exhaustive
  ones already did.
- With "Auto-Color Gates" on, an ancilla coloured its edges towards qubits of
  other circuits too, so a coupler between two circuits looked like a gate
  of one of them. An ancilla now colours only edges to qubits of its own
  circuit, on the map and in the PNG export.

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
