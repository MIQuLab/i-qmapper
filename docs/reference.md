# Reference

Explanations of the settings and messages of the I-QMapper interface. A tooltip
that ends with `· Reference: <entry>` refers to the entry of that name on this
page. For a first walkthrough see the [tutorial](../TUTORIAL.md); for the code,
[CODE_OVERVIEW.md](../CODE_OVERVIEW.md).

Contents:
[LQS](#lqs) ·
[Automatic LUCJ placement](#automatic-lucj-placement) ·
[Multi-circuit planner](#multi-circuit-planner) ·
[Scanning other processors](#scanning-other-processors) ·
[Calibration history](#calibration-history) ·
[Offline mode](#offline-mode) ·
[Log file](#log-file) ·
[Benchmarking](#benchmarking)

## LQS

The layout quality score estimates how well a circuit would run on a given set of
qubits:

LQS = ∏ (1 − readout error) over the qubits the circuit uses
    × ∏ (1 − two-qubit gate error) over every coupler joining two of those qubits.

Each coupler is counted once, whichever directions the backend lists it in. The
Estimated Layout-Quality panel shows the same number for each circuit, together
with the qubit and coupler counts, e.g. `C1: 77.2% (18q × 18g · …)`.

Layouts produced by the LUCJ search contain only the couplers the ansatz uses,
so for them the coupler count equals the number of two-qubit gates:
2 × (norb − 1) along the two chains plus 2 per ancilla (16 for norb = 8 with one
ancilla). A `(template)` suffix means the full calculation was unavailable and a
simpler estimate was used.

The score assumes independent errors and ignores crosstalk, circuit depth and
coherence; it is intended for comparing layouts on the same processor, not as an
absolute fidelity prediction.

## Automatic LUCJ placement

The LUCJ ansatz maps to a fixed template: an α chain and a β chain of `norb`
qubits each, joined by ancilla bridges. The search looks for the placement of
this template with the highest LQS on the processor.

### Search engine

Two engines run on every search:

- The **exhaustive** engine enumerates the placements that fit the template and
  scores them afterwards. At low ancilla counts on a large processor the number
  of placements exceeds the search cap, and only a sample is ranked.
- The **guided** engine grows each layout along the best-calibrated qubits and
  refines the finalists with local moves. It is not capped, but as a heuristic it
  can miss placements the exhaustive engine would find.

For every ancilla count the two shortlists are merged, duplicates are removed,
and the final list is ranked by LQS with the region-diversity rule below. The
best layout is therefore never worse than either engine alone. Each entry in
the layout list ends with its origin: `· E` exhaustive, `· G` guided, `· E+G`
both.

**Guided search** (Advanced, on by default) enables the guided engine. With it
off, the search uses the exhaustive engine alone, as in the previous release;
**Beam width** and **Refine top** tune the guided engine and are shown only
while it is on. The multi-processor scan and the stability filter use the exhaustive engine
only.

### Search results

A search ends with one line, e.g. `12 layouts found · scored on live · 20:41`:
the number of layouts in the list and the calibration that scored them. Each
entry carries its ancilla count, rank, LQS and origin.

### Search cap

The exhaustive engine stops after this many candidate placements. On a large
processor the cap is normally reached at low ancilla counts. When it is, the log
file records it; with diverse enumeration off, a warning notes that the ranking
may be non-exhaustive. Raising the cap gives a more thorough but slower search.

### Diverse enumeration

With this option on, the cap counts distinct candidates: a placement that
overlaps one already kept by more than 80 % of its qubits, and does not score
better, is discarded without using a slot, and one that scores better replaces
it. Without it, the cap fills with near-copies of the first region found, and
much of a large processor is never examined. The option only has an effect
where the cap is reached. It is on by default on heavy-hex processors and off on
square-lattice processors (Nighthawk), where the number of placements is far
larger and the option costs minutes for a gain of about 1 % in LQS.

### Prefer diverse regions

The first half of each candidate list is always the best-scoring layouts. The
second half may prefer layouts on other parts of the processor, so that the
list offers alternatives rather than variations of one position. A better
layout is never demoted.

### Quality floor

Qubits whose readout error lies more than the given number of standard
deviations above the device mean are excluded before the search starts. Set to 0
to disable. The purpose is to exclude clearly defective qubits, not to tune the
search.

## Multi-circuit planner

Several circuits are placed together on one processor. Placing them one at a
time, each on the best remaining region, can leave a later circuit a poor region
or none at all. The planner therefore keeps several partial plans: after each
circuit it tries **Candidates per circuit** layouts (default 3) on every plan it
holds and keeps the best **Plans kept** (default 3). It also builds the
one-at-a-time plan and returns whichever is better, so the result is never
worse than sequential placement.

The status line `2 circuits placed · +X% vs one-by-one` compares the two plans'
totals (the product of the circuits' LQS); the gain is omitted when there is
none. If the joint plan fits more circuits than sequential placement could, the
line reads `places 3 vs 2 one-by-one` instead.

**Plans** lists the best complete plans, e.g.
`Plan 1 · total 48.2% · C1 79.2% + C2 61.0%`; selecting one places all its
circuits. Below it, each circuit's alternatives include only layouts after
which every later circuit still fits; the others are counted in one line.

**Buffer qubits** is the number of idle qubits kept between circuits. With a
buffer of 1 or more, circuits never share a coupler. At 0 they may sit side by
side; couplers joining two circuits are listed as `shared couplers: 47–48` and
each one multiplies the plan score by `1 − Boundary penalty`, so a non-zero
penalty makes the planner prefer plans without them when such plans exist.

**Fast: one circuit at a time** (Advanced, off by default) skips joint planning
and places the circuits sequentially with the exhaustive engine, as in the
previous release.

## Scanning other processors

**Scan all QPUs** runs the current single-circuit search on every accessible
backend, scored with that backend's own calibration, and ranks the results.
**Scan batch on all QPUs** does the same for the current set of circuits,
placing them one at a time on each backend. Backends that fit all circuits
rank first by combined score; a backend that fits only some of them reads
`ibm_x — 1 of 2 circuits fit`, carries no total, and cannot be opened.
Backends whose topology the application cannot display are listed as
not supported. The processor currently connected is never searched again; its
local result appears as a reference row.

**Open on QPU 2** connects the selected backend as a second processor and
places the chosen layout or batch on it. The outcome is reported next to the
button.

## Calibration history

Calibrations are cached locally as they are fetched, so browsing earlier
calibrations, difference views and the stability check cost no QPU time.

### Delta references

In the difference view each calibration is compared against a reference.
**First** uses the first calibration in the sequence, showing cumulative drift
from a starting point. **Previous** uses the preceding calibration, showing the
step-by-step change as the slider moves; the first position has no comparison.

### Stability check

For each qubit, the readout and gate errors are evaluated over the selected
window (7, 14 or 30 days). Qubits whose values vary strongly across the window
are flagged, and the stability filter can exclude them from the automatic
search. For each unstable qubit or coupler the report shows the maximum, median
and minimum over the window and the worst day.

## Offline mode

A session restored without a backend connection. Layout editing, scoring
against the saved calibration and export all work. Anything that needs live
data — the difference view, calibration history, refreshing the map — becomes
available once a backend is connected from the Hardware section.

Sessions and style presets are saved under `.iqmapper_sessions/` in the
directory the notebook runs from; **Export Session** writes a session file to a
path of your choice.

## Log file

Technical messages (fallbacks, cap statistics, missing calibration values) are
written to `iqmapper.log` in the user log directory
(`platformdirs.user_log_dir("iqmapper")`, e.g. `~/Library/Logs/iqmapper` on
macOS), each distinct message once per session. They never appear under the
notebook cell.

# Benchmarking

I-QMapper normally ranks layouts on the calibration the provider reports, which
is refreshed about once a day. Qubits drift within that window. The Benchmark
section (sidebar, **5. Benchmark**) re-measures the hardware and feeds the
result back as a *measured* calibration epoch; the map, the LQS, the candidate
ranking and the exports read it exactly as they read the reported one.

Requires the optional dependency:

```bash
pip install "i-qmapper[benchmark]"
```

Benchmarking is never automatic: nothing runs on connect, on a timer or inside
the multi-processor scan. A run starts only from **Run benchmark**, and its
estimated QPU cost is shown before the button is enabled.

## Levels

| Level | Measures | Scope | Approximate QPU time |
|---|---|---|---|
| 1 | Readout assignment error | Target qubits | Seconds |
| 2 | Level 1 + single- and two-qubit randomized benchmarking | Target qubits and couplers | Tens of seconds |
| 3 | T1, T2 (Hahn echo), readout, single- and two-qubit RB | Whole processor | About four minutes |

Level 1 uses two circuits regardless of the number of qubits, so its cost does
not grow with the target; Level 1 on the whole processor is the inexpensive way
to refresh the readout map. Levels 2 and 3 scale with the number of couplers.
Level 3 reproduces the full-chip suite of IBM's "Real-time benchmarking for
qubit selection" tutorial, with the experiments grouped so that the whole
processor is covered by about 134 circuits.

## Targets

What a run measures: the layouts currently placed (every circuit slot), the
last search's best candidates, the whole processor, or an explicit list of
qubits. In inter-QPU mode a **QPU 1 | QPU 2** selector applies the targets, the
run and the resulting epoch to the selected processor.

## Cost

Cost estimates are wall-clock QPU seconds. The plan of the account is detected
where possible and assumed to be the Open plan otherwise; on that assumption
Level 2 warns and Level 3 requires an explicit acknowledgement, since a Level 3
run can consume a large part of a monthly allowance. The estimate is calibrated
from a single published figure and should be read as an order of magnitude.

Jobs that would exceed the provider's per-job shot limit are split
automatically. A rejected or failed job reports the provider's own error text
and the job identifiers. A completed run can be re-analysed later from its job
identifiers at no QPU cost.

## Relative mode

Randomized benchmarking and the provider's calibration are different estimators
of the same quantities, and they do not always agree on the absolute scale. A
two-qubit Clifford contains on average about 1.5 two-qubit gates and 8 to 10
single-qubit gates; the two-qubit RB analysis is therefore given the measured
single-qubit error rates and subtracts them.

After every run the measured values are compared with the reported ones,
property by property, and the result is stated: values in line with the
reported calibration, or values that differ by a constant factor and are shown
on the reported scale. The figures are in the **Details** table. If a property's
median is more than twice, or less than half, the reported one, the difference
is treated as a difference of method rather than a change in the device, and the
epoch is applied in relative mode: each value is rescaled so that the median
matches the reported one, which removes the offset while preserving every ratio
between elements. A coupler that has degraded still appears several times worse
than its neighbours. **Apply absolute values** uses the raw measurements
instead.

## Calibration check

Measures a small set of couplers twice in one session — with all pairs driven
in parallel, then one pair at a time — and compares both against the reported
values:

- both arms offset in the same way: estimator bias (single-qubit error
  attributed to the two-qubit gate);
- only the parallel arm offset: crosstalk between simultaneously driven pairs;
- both close to the reported values: the measurements agree.

The check does not change the calibration in use.

## What is applied

Measured values replace reported ones only for the elements actually measured;
all other elements keep their reported values, and the calibration label names
the epoch in use. A run reports what it measured (`measured 156/156 qubits`,
or `partial` when some elements failed), and a run that measured nothing is
reported as a failure and leaves the reported calibration in place. An
experiment or fit that fails falls back to the reported value for that element
with a notice.
