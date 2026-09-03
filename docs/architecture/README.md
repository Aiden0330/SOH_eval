# Architecture notes — index

**Status (2026-09-02):** every document in this directory is **historical**. They record how the
`sohbench` workbench was built phase by phase (June–August 2026, mostly in Chinese) and are kept as
the audit trail behind the current code. The workbench is frozen at tag `v0-workbench`
(ADR-001, `docs/adr/2026-09-02-ADR-001-kernel-extraction-and-agent-platform.md`); the rules that
outlive it are pinned in `docs/superpowers/specs/2026-09-02-soh-kernel-invariants.md`, and the
extraction work is tracked in `docs/superpowers/plans/2026-09-02-kernel-extraction.md`.

| Document | Phase | What it records |
|---|---|---|
| `soh-evaluation-phase-1.5-hardening.md` | 1.5 | Evaluation-case workbench hardening |
| `soh-evaluation-phase-2-closure.md` | 2 | Case closure and reconciliation loop |
| `soh-evaluation-phase-2.1-controlled-retest.md` | 2.1 | Controlled retests, mockup fusion |
| `soh-evaluation-phase-2.2-golden-case.md` | 2.2 | Real golden case (NASA) |
| `soh-evaluation-phase-2.3a-real-package-preflight.md` | 2.3A | Real preflight, training, immutable algorithm packages |
| `soh-evaluation-phase-2.3b-batch-regression.md` | 2.3B | Batch regression, real retests, cross-page consistency |
| `soh-evaluation-phase-2.4-case-pipeline.md` | 2.4 | Case pipeline and candidate artifact convergence |
| `soh-evaluation-phase-3.0-automated-regression-gate.md` | 3.0 | Automated regression gate (release gate script) |
| `soh-evaluation-phase-3.1-native-evidence-canvas.md` | 3.1 | Native evidence canvas, iframe removal |
| `soh-evaluation-phase-3.1.1-production-stability.md` | 3.1.1 | Production stabilisation gate (now the CI Windows job) |
| `soh-evaluation-phase-3.2-saved-combination-library.md` | 3.2 | Saved-combination library and generalisation checks |
| `soh-evaluation-phase-3.3-mechanism-method-packages.md` | 3.3 | Mechanism features, grey-box methods, PINN method packages |
| `soh-evaluation-phase-3.4-guided-closure-and-method-runtime.md` | 3.4 | Guided closure and the three-family method runtime |
| `soh-evaluation-phase-3.9-local-data-connectivity.md` | 3.9 | Local data migration (strategy B) and three-page connectivity |
| `soh-evaluation-phase-4.0-context-binding-and-visualization-audit.md` | 4.0 | Case context binding and legacy visualisation audit |
| `soh-evaluation-label-safety-v1.1-affine-reconstruction.md` | — | Label safety v1.1: affine label reconstruction detection (still normative; superseded in form by Spec 01 §2) |
| `soh-evaluation-evidence-deep-fusion-refactor.md` | — | Evaluation page × detail dashboard fusion proposal (v1) |
| `soh-evaluation-deep-fusion-refactor-v3.md` | — | Evaluation page × detail dashboard fusion proposal (v3) |

The phase 3.0 gate script these documents describe is retired (2026-09-03): its checks are
`tests/workbench/test_release_gates.py` and `tests/kernel/test_golden_invariants.py`.
`phase22_golden` and `phase24_pipeline` remain the workbench's measured-case release pipeline
(behind `scripts/build_phase22_golden_case.py` and `scripts/run_phase24_pipeline_acceptance.py`).
