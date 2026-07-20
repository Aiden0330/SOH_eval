# Standalone Guide Task 6 Report

## Scope And Baseline

- Baseline: `6a28950b923db4bb4ff4ffd23e4cb013e33db6b6`.
- Delivery: standalone Chinese HTML, focused guide tests, and this verification report.
- The five embedded screenshot payloads, Task 4 Native CSS, and Task 5 interaction
  behavior were retained. Existing exact payload-hash tests remained green.

## TDD Record

### Fact And Freeze RED

The first focused run after adding Task 6 contracts produced:

```text
42 passed, 8 failed
```

The failures were exactly the new requirements:

- missing `#source-boundary`;
- missing `#delivery-checklist`;
- target/current end-to-end flow not separated;
- Metric Worker and publication policy claims not scoped by layer;
- target outcome vocabulary incorrect;
- `ic_dva`, `capacity_time`, and dataset integration claims inaccurate;
- battery-dashboard ownership inaccurate;
- generation activation assertions contradicted the implementation.

After the HTML corrections, focused GREEN was:

```text
50 passed
```

### Print RED

Real Chrome print-to-PDF exposed a defect not visible from `scrollWidth` or static
computed-style checks: the PDF had 12 pages, but pages 2 through 12 were blank.
The screen layout left `.guide-shell` at `height: 100vh; overflow: hidden`.

A focused regression assertion for the print override was added first:

```text
1 failed
```

The print rule was then changed to:

```css
.guide-shell { display: block; height: auto; overflow: visible; }
```

The targeted test passed, and the final focused suite remained at 50 passed.

## Fact Evidence

### Current Code And Runtime

- `src/sohbench/api_server.py` exposes exactly 9 GET and 11 POST routes. The
  document table is parsed and compared with the source AST.
- `frontend/app/index.html` exposes the five current modes: training, inference,
  models, native dashboard, and diagnostics.
- `fabric/views.py` strips the target from candidate train/test inputs.
  `runner.py` fits FeatureState on the training fold and computes metrics
  internally from held-out truth.
- Current Phase 0 runs training/frozen inference, metric calculation, evidence
  writing, and publication in the same Python or trusted local process. User
  confirmation, Adaptation service separation, exclusive Metric Worker truth
  access, PolicySpec gating, review, and signed publication are labeled target
  architecture.
- Current publication provides artifact integrity, not reviewed-verdict or
  PolicySpec eligibility gating.
- Target outcomes are limited to `complete`, `partial`, `failed`, `timed_out`,
  `cancelled`, and `invalid`. Current Phase 0 implements the first three as its
  aggregate subset.

### Features And Data

- Current `ic_dva` candidates are only `ic_peak_ah_per_v` and
  `ic_peak_voltage_v`. DVA, area, and shape are described only as method
  background or target capability.
- Current `capacity_time` still includes `cycle_fraction`. The guide explicitly
  records this as an open FeaturePolicy gap and does not claim complete future
  information isolation.
- The local `D:\battery_SOH\data\processed\*_features.pkl` acceptance snapshot
  contained:

```text
calce_cs2_features.pkl
hust_features.pkl
nasa_features.pkl
oxford_features.pkl
rwth_isea_public_features.pkl
smoke_eval_demo_features.pkl
snl_lfp_features.pkl
uconn_features.pkl
```

The HTML labels this a dynamic local snapshot and makes no permanent dataset
count claim.

- UConn-ISU-ILCC LFP/Gr is described as integrated. ISU-ILCC Figshare NMC/Gr is
  described as research-only and not integrated.
- `frontend/app/battery-dashboard` has only `README.md` tracked. The guide
  describes it as a transactional sync target whose generated payload is not
  committed and whose availability depends on manifest/runtime status.

### Publication Semantics

- Directory staging is validated and moved to an immutable generation before
  `CURRENT.json` is replaced.
- A pointer failure does not activate the new generation; the prior pointer and
  last-known-good remain readable. A definitely unreferenced generation may be
  cleaned up.
- Compatibility-view synchronization happens after pointer activation. Its
  failure does not invalidate the active generation.
- Standalone packaging has no generation or `CURRENT.json`: it validates a
  temporary HTML and atomically replaces one output file, preserving the prior
  file on failure.

## Browser Acceptance

Method: installed Chrome 150, headless DevTools Protocol, original absolute
`file://` URL, no local server and no installed dependency. Screenshots and the
printed PDF were visually inspected in addition to rectangle checks.

### Desktop 1280x720

- `documentElement scrollWidth/clientWidth`: `1265/1265`.
- Column rects: nav `0..240`, main `240..1005`, quick rail `1005..1265`.
- Key rect overlap pairs: `0`.
- H1 rect: `x=272`, `width=686`, `right=958`; scroll/client width `686/686`.
- Long-path overflow findings: `0`; critical-element overflow findings: `0`.
- All five images were complete. Natural sizes were four at `1440x800` and the
  native dashboard at `1110x800`.

### Mobile 390x844

- `documentElement scrollWidth/clientWidth`: `390/390`.
- H1 rect: `x=16`, `width=358`, `height=66`, `right=374`.
- Key rect overlap pairs: `0`.
- Long-path overflow findings: `0`; critical-element overflow findings: `0`.
- Product images rendered at width `322`; all retained nonzero natural sizes.
- The right rail was hidden, the nav toggle was visible, and chapter quick
  sources were visible.

### Interactions

- Search `EvidencePack` matched overview, architecture, contracts, frontend, and
  evidence. Clear restored all 13 chapters.
- Clicking `#evidence` while search was active cleared the query, restored all
  13 chapters, and set the hash to `#evidence`.
- `all/manager/developer/operator` hid zero chapters. Highlight counts were
  `0/5/12/9`.
- All 13 primary anchors produced the expected hash and a visible target.
- Chapter collapse and re-expand both succeeded.
- Clipboard success reported `已复制`. With Clipboard API rejection and
  `execCommand` failure, fallback reported `已选中，请按 Ctrl+C` with selection
  length 96.
- Scroll-spy moved main from `scrollTop=24770` to the contracts
  `offsetTop=10294`; target top became 0, active nav became `#contracts`, and
  the quick rail showed the contracts summary.

### Progressive And Offline States

- JavaScript-disabled reload: 13 visible chapters, 13 working internal anchors,
  no enhanced details wrapper, five complete images, width `390/390`.
- Reduced motion: html/main scroll behavior `auto`; sampled transition
  durations were all `0s`.
- Console errors: `0`.
- HTTP(S) requests after file navigation and interaction: `0`.

### Print

- Interactive chrome was `display:none`; all 13 chapters remained visible and
  table headers used `table-header-group`.
- All four diagrams reported `break-inside: avoid`.
- Final Chrome PDF: 37 pages; rendered nonblank pages: `37/37`.
- Visual inspection found the four critical diagrams complete and unclipped on
  pages 3, 4, 5, and 13.

## Verification

- Validator: exit 0, 0 issues.
- Focused pytest with current worktree `PYTHONPATH`: 50 passed.
- Inline JavaScript `node --check`: exit 0.
- Forbidden/content-boundary scan: pass after stripping embedded image data
  URIs; no unfinished markers, document metadata dates, timestamp IDs, external
  loadable assets, U+FFFD/known mojibake, invalid target outcome, or superseded
  generation statement.
- Full pytest with current worktree `PYTHONPATH`: exit 0 in about 458 seconds.
  `1095 collected`, `1090 passed`, `5 skipped`, `12 warnings`.
- The warnings are existing overflow-path RuntimeWarnings and chemistry-out
  single-dataset confound warnings exercised by tests.

The first full-suite attempt without `PYTHONPATH` stopped during collection with
22 import errors because the shared editable virtual environment resolved
`sohbench` from `D:\SOH_eval\src`. Diagnostic imports proved the mismatch. The
final command placed this worktree's absolute `src` first and completed
successfully.

## Residual Risks

- `capacity_time.cycle_fraction` remains an implementation FeaturePolicy gap.
- Independent workers, full target outcomes, reviewed PolicySpec eligibility,
  and signed publication remain target architecture, not current capability.
- The browser acceptance is an authoring-time CDP run; focused pytest retains
  the key search/anchor/clipboard and static delivery regressions.

---

## Independent Review Remediation

The first Task 6 delivery was independently reviewed as NOT APPROVED because its
target verdict lane used a non-authoritative vocabulary and several important
fact directions were protected only by loose prose assertions.

### Authoritative Contract

The redesign spec defines the exact target values as:

```text
outcome = complete / partial / failed / timed_out / cancelled / invalid
verdict = pass / warn / fail / not_applicable / not_reviewed
```

The HTML marks the two target lanes with `data-state-lane` and every value with
`data-state-value`. The first remediation parser asserted the exact ordered
attribute collections, including count and missing attributes, but did not bind
those attributes to each node's visible text. The second review remediation
closes that gap by capturing normalized visible text and requiring every
`(data-state-value, visible text)` pair to equal the authoritative value.

The state lane, fallback text, and current-to-target explanation now use the
authoritative verdict vocabulary. The legacy target verdict words were removed
from the deliverable.

### Structured Fact Direction

Machine-readable attributes now freeze these relationships:

- UConn-ISU-ILCC LFP/Gr: `integrated`.
- ISU-ILCC Figshare NMC/Gr: `research-only-not-integrated`.
- `capacity_time`: `open-gap`, with `cycle_fraction` as the current risk feature.
- `ic_dva`: current columns are exactly `ic_peak_ah_per_v` and
  `ic_peak_voltage_v`.
- Metric truth: current `in-process`; target `exclusive-metric-worker`.

The first remediation asserted attribute direction and rejected representative
reversed prose claims globally. That was not sufficient to prevent a marked
element's local visible text from contradicting its attribute. The second review
remediation binds each dataset span's complete normalized text and the full
`capacity_time` row text to the structured contract. The delivery checklist now
says there are three audience reading paths plus one “all” switch, instead of
calling the four controls five paths.

### Remediation TDD

Initial focused RED after adding the structural contracts:

```text
49 passed, 4 failed
```

The four failures were the expected missing checklist wording, structured state
lanes, authoritative verdict explanation, and structured fact markers.

After the HTML update, one test remained overbroad because it rejected the
legitimate current execution-coverage term `succeeded` outside the target lane:

```text
52 passed, 1 failed
```

The global prohibition was narrowed to legacy target values; exact state-lane
parsing remains responsible for rejecting `succeeded` in a target lane. Final
focused GREEN:

```text
53 passed
```

### Remediation Verification

- Validator: exit 0, 0 issues.
- Focused guide pytest: 53 passed.
- Inline JavaScript `node --check`: exit 0.
- `git diff --check`: exit 0.
- Sanitized forbidden scan: no unfinished markers, metadata labels, legacy
  target verdicts, old checklist wording, or representative reversed claims.
- Unicode marker scan: no U+FFFD or known mojibake.
- The full suite was intentionally not rerun for this remediation, per the
  review instruction; final whole-branch verification remains responsible for
  that run.

## Second Independent Review Remediation

### Visible-Text Binding Gap

The second review found one Important issue: structured attributes could remain
correct while the visible state, dataset, or FeaturePolicy text was reversed.
The earlier attribute-only and global-prose checks would still pass for such a
local mutation.

`StructuredStateLaneParser` now captures normalized visible text for every
`state-node`. The shared fact-contract helper retains the exact ordered outcome
and verdict collections and additionally requires each visible label to equal
its own `data-state-value`.

`StructuredFactParser` now captures the complete normalized text of both marked
dataset spans and every marked feature row. The shared helper requires:

- UConn-ISU-ILCC LFP/Gr to say it is currently integrated.
- ISU-ILCC Figshare NMC/Gr to say it is research-only and currently not
  integrated.
- The full `capacity_time` row to say that the current feature set includes
  `cycle_fraction`, the FeaturePolicy gap remains open and must be closed, and
  complete future-information isolation cannot yet be claimed.

### Visible-Text TDD

Four mutation-negative cases were added without changing any structured
attribute: `complete` displayed as `succeeded`, each dataset relationship
reversed independently, and the open gap displayed as closed. All four call the
same fact-contract helper.

Focused RED before parser binding:

```text
53 passed, 4 failed
```

Each failure was the expected `DID NOT RAISE AssertionError`, proving that the
old parser accepted the contradictory visible text. Focused GREEN after binding:

```text
57 passed
```

### Second Review Verification

- Validator: exit 0, 0 issues.
- Focused guide pytest with the current worktree `PYTHONPATH`: 57 passed.
- Inline JavaScript `node --check`: exit 0.
- `git diff --check`: exit 0.
- The full suite was not rerun, as requested; final whole-branch verification
  remains responsible for that run.
