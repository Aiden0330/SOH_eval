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
- A failure before the pointer file's `os.replace` does not activate the new
  generation; the prior pointer and last-known-good remain active. A definitely
  unreferenced generation may be cleaned up.
- A directory durability/fsync failure after pointer replacement may return an
  error even though the new pointer is already active. The operator must re-read
  `CURRENT.json` before retrying or recording the active generation.
- Compatibility-view synchronization happens after pointer activation. Its
  failure does not invalidate the active generation.
- Standalone packaging has no generation or `CURRENT.json`: it validates a
  temporary HTML before replacement. Validation, temporary-write, and other
  pre-replace failures preserve the prior file; a post-replace directory fsync
  error is ambiguous and requires re-reading the file digest.

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

## Final Whole-Branch Review Remediation

### RED Tests

The first focused command after adding the final-review regressions was:

```powershell
$env:PYTHONPATH = "$(Get-Location)\src"
python -m pytest -q tests/test_design_user_guide.py
```

It exited 1 with 36 expected failures and 56 passes across the then-collected
92 cases. The RED failures covered:

- current task `verdict` values, leaderboard named boolean checks, and the
  target verdict enum not being structurally separated;
- current/integrated IC contexts still naming DVA, plus unqualified user
  confirmation in visual/fallback representations;
- overbroad directory and standalone publication failure guarantees;
- missing strict base64, image magic/dimension, screenshot-set, sidecar resource,
  and active JavaScript load-site validation;
- screenshot embedding accepting truncated image bytes and bypassing candidate
  validation/shared atomic publication;
- missing pre-replace failure preservation and post-replace fsync ambiguity
  injection coverage;
- missing `命令`/`输出` fields in chapter quick sources and premature `.js`
  activation on initialization failure.

Self-review then added two malformed-attribute mutations. Their targeted RED was:

```text
10 passed, 2 failed
```

Both failures were expected `AttributeError` paths for bare `rel`/`src`
attributes, proving the validator needed to return issues rather than crash.

### Implementation Decisions

- Bound current task verdicts to exactly `not_reviewed / fail`, bound current
  leaderboard `verdicts` to the named boolean checks `generalization`,
  `calibration`, and `physical_consistency`, and retained the exact target enum
  `pass / warn / fail / not_applicable / not_reviewed`.
- Replaced every current/integrated DVA claim with current IC peak or IC
  `dQ/dV` curve wording. DVA remains only method background or target scope.
- Marked user confirmation as target in both five-layer and workflow visual and
  accessible fallback representations.
- Added a strict standalone resource allowlist, exact five-shot set, canonical
  base64 decoding, PNG/JPEG/WebP magic and structural dimension validation,
  positive declared dimensions, CSS resource rejection, and active JavaScript
  load-site rejection. Literal localhost commands in code blocks remain valid.
- Validated input capture bytes before embedding, validated the complete staged
  candidate, and reused `publish_text_file_atomic`. Pre-replace failures preserve
  the old file; post-replace directory fsync failure remains truthfully ambiguous.
- Added all five quick-source fields to every chapter, delayed `.js` activation
  until initialization completes, and persisted 390 px/no-enhancement/image
  browser smoke coverage with installed Chrome.
- Retained all five approved embedded screenshot payloads and their exact
  SHA-256 contracts; the guide image payload lines were not regenerated.

### Focused GREEN

The required commands were rerun from the worktree with the current `src` first:

```text
python -m pytest -q tests/test_design_user_guide.py
  exit 0; 94 tests passed
python scripts/validate_design_user_guide.py docs/soh-evaluation-system-design-and-user-guide.html
  exit 0; 0 issues
python -m py_compile scripts/validate_design_user_guide.py scripts/embed_design_user_guide_screenshots.py
  exit 0
node frontend/app/dashboard-core.test.mjs
  exit 0
node frontend/app/chinese-ui.test.mjs
  exit 0
node frontend/app/native-dashboard.test.mjs
  exit 0; native dashboard delivery contract passed
git diff --check
  exit 0; only Git autocrlf advisory warnings were emitted
```

The focused Chrome tests additionally proved:

- 390 px `scrollWidth === clientWidth`;
- all five embedded images completed with nonzero natural width;
- the no-enhancement state retained 13 visible chapters and 13 valid anchors;
- an injected initialization exception left `.js` inactive and the native
  mobile TOC visible.

The full Python suite was intentionally not run, per the final-review brief.

### Minor Findings

No Minor finding was intentionally left open. Existing authoring-time print
evidence remains recorded above; no new print runtime dependency was added.

### Commit And Files

- Implementation commit: `208bbca`.
- Changed files:
  `docs/soh-evaluation-system-design-and-user-guide.html`,
  `scripts/validate_design_user_guide.py`,
  `scripts/embed_design_user_guide_screenshots.py`,
  `tests/test_design_user_guide.py`, and
  `.superpowers/sdd/standalone-guide/task-6-report.md`.

## Final Whole-Branch Review Remediation Round 2

### Round 2 RED

After adding the required attribute, script-context and image-decoding
mutations, the focused command was:

```powershell
$env:PYTHONPATH = "$(Get-Location)\src"
python -m pytest -q tests/test_design_user_guide.py
```

It exited 1 with 17 expected failures and 94 passes. The failures proved that
the previous validator:

- discarded ordered duplicate attributes and accepted a first sidecar `src`
  followed by an embedded `src`;
- accepted case-insensitive duplicate names, inline `style`, `onload`, `ping`
  and an unknown resource-bearing attribute;
- accepted `window["fetch"](...)` because script safety depended on spelling
  patterns;
- accepted a CRC-correct PNG with an invalid zlib stream directly, in a
  complete guide and through screenshot embedding;
- accepted JPEG and WebP through both the validator and helper; and
- did not bind declared dimensions to decoded dimensions.

Self-review added two positive-approval mutations for a structurally valid,
dimension-matching but unapproved PNG. With the approval check temporarily
absent, the targeted run exited 1 with exactly 2 failures. Restoring the
approval check made both pass.

### Implementation Choice

- `GuideParser` now preserves each ordered attribute tuple, derives effective
  values with browser-significant first-occurrence semantics and reports
  duplicate names case-insensitively.
- The fixed guide uses explicit global/per-tag attribute allowlists. Event
  attributes, `style`, `ping`, unknown attributes and bare non-boolean forms
  are rejected. The two intentional boolean forms, `hidden` and `open`,
  remain allowed.
- The exact single inline script body is frozen at SHA-256
  `96540c4320d4e03999c29fdacfe2133257e21c167d290fb3923e908715469b26`.
  The existing load-site patterns remain diagnostic defense, not the proof of
  script approval.
- Image support is intentionally narrowed to PNG. PNG validation now
  decompresses the complete concatenated IDAT stream, validates decoded row
  size and filter bytes, and checks payload dimensions against declared
  dimensions.
- The five browser-proven screenshots are frozen by
  `data-shot -> PNG MIME / dimensions / SHA-256`. A well-formed replacement
  requires an explicit approval-contract update.
- Screenshot embedding still validates input bytes and the complete staged
  candidate before `os.replace`; therefore the same attribute, script and
  image approval gates preserve the old destination on rejection.
- The installed-Chrome smoke additionally records HTTP(S) performance resource
  entries and captures `error`/`unhandledrejection`; both enhanced and
  no-enhancement mobile runs observed empty lists.

The guide HTML and all five approved screenshot payloads remained byte-for-byte
unchanged.

### Round 2 GREEN

The mutation-only rerun passed all 17 initial RED cases. The final required
focused verification from the worktree produced:

```text
python -m pytest -q tests/test_design_user_guide.py
  exit 0; 113 tests passed
python scripts/validate_design_user_guide.py docs/soh-evaluation-system-design-and-user-guide.html
  exit 0; 0 issues
python -m py_compile scripts/validate_design_user_guide.py scripts/embed_design_user_guide_screenshots.py
  exit 0
node frontend/app/dashboard-core.test.mjs
  exit 0
node frontend/app/chinese-ui.test.mjs
  exit 0
node frontend/app/native-dashboard.test.mjs
  exit 0; native dashboard delivery contract passed
git diff --check
  exit 0; only Git autocrlf advisory warnings were emitted
```

The full Python suite was intentionally not run, per the Round 2 brief.

### Round 2 Commit And Files

- Implementation commit: `f95f1c3`.
- Changed implementation/test files:
  `scripts/validate_design_user_guide.py`,
  `scripts/embed_design_user_guide_screenshots.py`, and
  `tests/test_design_user_guide.py`.
- Report append:
  `.superpowers/sdd/standalone-guide/task-6-report.md`.

## Final Whole-Branch Review Remediation Round 3

### Round 3 RED

The reviewer mutation was reproduced directly against the Round 2 validator:
both remote and relative CSS `image-set(...)` declarations returned zero
issues. After adding validator and screenshot-helper regressions, this targeted
command was run before production changes:

```powershell
$env:PYTHONPATH = "$(Get-Location)\src"
python -m pytest -q tests/test_design_user_guide.py -k "css_image_set_resources or exactly_one_approved_inline_style_body or css_image_set_without_replacing_html"
```

It exited 1 with exactly 5 failures:

- remote `image-set("http://127.0.0.1:9/sidecar.png" 1x)` produced no
  validator issue;
- relative `image-set("./sidecar.png" 1x)` produced no validator issue;
- a second harmless `<style>` body produced no validator issue; and
- both screenshot-helper candidates were accepted instead of raising before
  replacement.

The helper fixtures started from a distinct old destination with an invalid
training placeholder. This proved that accepting the candidate would replace
bytes, while rejection must preserve the old destination byte-for-byte.

### Round 3 Implementation Choice

`GuideParser` now records each complete inline style body independently. The
validator requires exactly one style body whose SHA-256 is:

```text
7bd17690456794cd7b62360858891a3fb0ddbed604af10972b39b0d6c8f6b965
```

This is the CSS analogue of the Round 2 approved script digest. The existing
`url(...)` and `@import` checks remain useful diagnostics, but safety no longer
depends on enumerating CSS resource-function spellings. Any future CSS change,
including `image-set`, requires an explicit approval-contract update.

The screenshot helper was not changed. Its existing complete-candidate
validation automatically inherits the approved style contract before
`os.replace`, and both new helper tests prove rejection leaves the old file
unchanged.

The historical Task 4 plan remains immutable execution history. Its WebP/JPEG
helper mapping, `.webp` command examples and WebP preference are superseded by
the Round 2 PNG-only helper/validator contract and must not be used as current
instructions.

The final HTML, approved style body, inline script and all five screenshot
payloads were unchanged.

### Round 3 GREEN

The targeted rerun passed all 5 Round 3 mutations. Final focused verification
from the worktree produced:

```text
python -m pytest -q tests/test_design_user_guide.py
  exit 0; 118 tests passed
python scripts/validate_design_user_guide.py docs/soh-evaluation-system-design-and-user-guide.html
  exit 0; 0 issues
python -m py_compile scripts/validate_design_user_guide.py scripts/embed_design_user_guide_screenshots.py
  exit 0
node frontend/app/dashboard-core.test.mjs
  exit 0
node frontend/app/chinese-ui.test.mjs
  exit 0
node frontend/app/native-dashboard.test.mjs
  exit 0; native dashboard delivery contract passed
git diff --check
  exit 0; only Git autocrlf advisory warnings were emitted
```

The full Python suite was not run; Round 3 requested focused verification.

### Round 3 Commit And Files

- Implementation commit: `b7ecee7`.
- Changed implementation/test files:
  `scripts/validate_design_user_guide.py` and
  `tests/test_design_user_guide.py`.
- Audit updates:
  `.superpowers/sdd/standalone-guide/task-6-report.md` and
  `.superpowers/sdd/progress.md`.

## Final Whole-Branch Review Remediation Round 4

### Verification Provenance Before Round 4

The Round 3 statement that the full Python suite was not run is historical:
it describes the Round 3 remediation agent's focused gate before the audit
commit. The controller subsequently ran the full Python suite at HEAD
`9b0f567`; it reached 100% and exited 0. Round 4 does not replace or
reinterpret that controller evidence.

### Round 4 RED

The six merged independent-review findings were checked against the current
code before editing. The generic standalone builder's candidate callback only
checks the doctype, standalone marker and closing HTML shell, while the guide
screenshot helper calls the complete guide validator before replacement.
`GuideParser` also stored IDs only in a set, every dynamic chapter summary used
the same visible accessible name, and print CSS left `.mode-guide` under the
screen `break-inside: avoid` rule.

The following test selection was run after adding the Round 4 regressions and
before production changes:

```powershell
$env:PYTHONPATH = "$(Get-Location)\src"
python -m pytest -q tests/test_design_user_guide.py -k "print_restores_filtered_content or dynamic_chapter_summaries or duplicate_nonempty_ids or duplicate_id_without_replacing_html or delivery_checklist or metric_truth_and_publication_claims or publication_claims_are_scoped"
```

It exited 1 with exactly 7 expected failures:

- no print override allowed long `.mode-guide` content to paginate;
- all 13 generated chapter summary accessible names were identical;
- the validator accepted a duplicate nonempty `overview` ID;
- the screenshot helper replaced its candidate instead of preserving the old
  file when the candidate had that duplicate ID;
- the delivery checklist lacked the controller full-suite provenance;
- the contracts chapter conflated single-file and directory publication; and
- the packaging chapter attributed the guide's stronger candidate gates to
  the generic standalone builder.

### Round 4 Implementation

- Single-file publication is now described as same-directory temporary write,
  file fsync, candidate validation, `os.replace`, and directory fsync. It is
  explicitly separate from directory generations, `CURRENT.json`, and the
  last-known-good active generation.
- The generic `build_standalone_frontend.py` path is scoped to structural
  shell checks. The guide's screenshot embed/validator path retains its
  stronger semantic, resource, image, script, style, and approval checks
  before replacement.
- `GuideParser` now counts each nonempty ID while retaining the ID membership
  set. Duplicate values produce one sorted `duplicate-id` issue, and the
  unchanged screenshot helper inherits this complete-candidate rejection.
- Each generated chapter summary keeps the compact visible text but receives
  an `aria-label` containing its one-based chapter number and H2 text. A Node
  contract executes the real function against all 13 headings and proves the
  accessible names are unique and descriptive.
- Print CSS deliberately overrides `.mode-guide` with
  `break-inside: auto !important` and
  `page-break-inside: auto !important`, while smaller figures, tables and
  product shots keep their avoid rules.
- The delivery checklist now identifies the controller full-suite evidence at
  `9b0f567` and labels the earlier Round 3 no-full-suite statement as the
  historical remediation-agent gate.

The approved inline style SHA-256 is now:

```text
6d90e5c537533a9e6570a82923d8d52b8e47df50ea501ca6a3fcb6b210150f35
```

The approved inline script SHA-256 is now:

```text
4bb54b9106beca74188540499121301774b8d6a1fe1b56d2060664394ca391fa
```

All five approved screenshot data URIs and hashes were unchanged.

### Round 4 GREEN

The identical seven-test selection passed 7/7. Pre-commit focused verification
then produced:

```text
python -m pytest -q tests/test_design_user_guide.py
  exit 0; 121 tests passed
python scripts/validate_design_user_guide.py docs/soh-evaluation-system-design-and-user-guide.html
  exit 0; 0 issues
python -m py_compile scripts/validate_design_user_guide.py scripts/embed_design_user_guide_screenshots.py
  exit 0
node frontend/app/dashboard-core.test.mjs
  exit 0
node frontend/app/chinese-ui.test.mjs
  exit 0
node frontend/app/native-dashboard.test.mjs
  exit 0; native dashboard delivery contract passed
inline guide JavaScript piped to node --check -
  exit 0
git diff --check
  exit 0; only Git autocrlf advisory warnings were emitted
```

The full Python suite was not rerun during Round 4, as requested. The current
full-suite evidence remains the controller run at `9b0f567`.

### Round 4 Commit And Files

- Implementation commit: `e3221d1`.
- Changed implementation/test files:
  `docs/soh-evaluation-system-design-and-user-guide.html`,
  `scripts/validate_design_user_guide.py`, and
  `tests/test_design_user_guide.py`.
- Audit updates:
  `.superpowers/sdd/standalone-guide/task-6-report.md` and
  `.superpowers/sdd/progress.md`.

## Final Frozen Gate

Round 4 was followed by a controller-owned frozen verification at
`9b34ae0`. The deliverable HTML, validator, helper, and tests did not change
during this gate.

- The full Python suite reached 100% and exited 0. Standard output contained
  only the existing chemistry-confound and non-finite metric protection
  warnings; standard error was empty.
- The focused guide suite passed 121 tests. The validator reported zero
  issues; Python compilation, all three Node suites, inline JavaScript syntax,
  and the whole-branch diff check exited 0.
- Chromium QA passed at 1280 x 720 and 390 x 844 with no horizontal overflow,
  all 13 chapters and five approved screenshots present, zero HTTP requests,
  and zero console or page errors. The no-JavaScript view retained all
  chapters, anchors, and screenshots.
- Print QA produced 31 intact A4 pages. The long mode guides paginate across
  pages while screenshots and diagrams remain unsplit; the formerly sparse
  continuation pages contain continuous content.
- Two independent final reviewers rechecked their prior findings and returned
  `APPROVED`, with no remaining Critical, Important, or Minor findings.

The visible HTML contains no commit hash, document version, or publication
date. Commit identifiers remain only in this internal audit report.
