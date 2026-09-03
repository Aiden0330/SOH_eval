# ADR-001 — Extract the SOH kernel from `sohbench`; build the agent platform in a new repository

- **Date:** 2026-09-02
- **Status:** Proposed (awaiting owner decision)
- **Deciders:** project owner
- **Inputs:** *Agentic AutoML for Battery SOH* research report (2026-09-02, `projects/autoML/agentic-automl-soh-report.html`); repository state at HEAD `732923a` (`feat/phase2-dl`, 93 commits, ~20.5 k lines in `src/sohbench`, 414 tests passing on macOS); import-graph analysis of `src/sohbench` (2026-09-02).

## Context

`sohbench` was built in ten weeks as a benchmark harness plus a human-driven workbench (jobs, cases, evidence pages, trusted plugins). Along the way it accumulated the assets that matter most for an agent-driven AutoML tool and that the agent literature lacks: hard label-safety gates, leakage-safe leave-one-domain protocols, a nested inner/outer search with a constraint oracle, adaptation operators with declared information channels, and immutable provenance.

The same repository also carries an architecture that would constrain an agent platform: a 3.5 k-line `evaluation_service` that owns orchestration, a closed candidate menu in `llm_policy`, in-process execution of uploaded code (`trusted_plugins`), phase-numbered release scripts, Windows-first paths, and a frontend coupling that shapes what a "run" is.

The import-graph analysis shows the two halves are already separable:

| Bucket | Modules | Lines | Dependency status |
|---|---|---|---|
| Kernel | contracts, metrics, metadata, registry, feature_safety, raw_windows, method_contracts, preprocessing, qc, data, protocols, tasks, acceptance, conformal, features, featuresets, window_bank, report, runner, adaptation, estimators/* | ≈ 4,900 | one inverted import to break: `adaptation → evaluation_service`; everything else imports only kernel modules |
| Search | agent_search, llm_policy | ≈ 1,270 | depends on `evaluation_service` for preflight and transforms; to be re-implemented on the kernel API |
| Workbench | evaluation_service, api/*, api_server, inference, model_store, trusted_plugins, evaluation_cases, review_workflow, case_evidence, case_events, frontend_publish, battery_dashboard_import, battery_soh_plugins, cli, paths | ≈ 11,900 | imports the kernel freely; must never be imported by it |
| Phase scripts | phase22_golden, phase24_pipeline, phase30_release_gate | ≈ 1,600 | encode invariants as scripts; to be converted into tests |

## Decision

1. **Extract a kernel package `sohkernel` inside this repository** (`src/sohkernel`), containing the kernel bucket above plus the trusted-inference label gate now inside `model_store` and the prediction-record schema now inside `inference`. The kernel is data-first (DataFrames/Parquet in, JSON manifests out) and has no dependency on FastAPI, the frontend, jobs or cases. Its tests stay next to the data they need (`data/manifest.json`, the standing leak inventory).
2. **Freeze the workbench.** `sohbench` keeps working by importing from `sohkernel` through thin re-export shims. No new features land in the workbench until the platform's first milestone decides its future.
3. **Build the agent platform in a new repository** (`projects/soh-agent`, package `sohagent`) that depends on `sohkernel` (editable install from the sibling folder at first, a versioned wheel once the interface has been stable for one milestone). The platform owns candidates-as-programs, the search engine, the sandbox worker, memory, budgets, roles and the claim flow.
4. **Direction rule.** `sohagent → sohkernel` and `sohbench → sohkernel` are the only allowed directions. `sohkernel` never imports `sohbench` or `sohagent`. An import-direction test enforces this in CI.
5. **Data stays here.** The data root is exposed through `SOHBENCH_DATA_DIR`; Windows default paths are removed.

## Consequences

**Positive.** The ten weeks of leak-defense and protocol work are preserved with their tests; the platform starts with a clean architecture and history; the kernel API becomes explicit because it is consumed across a package boundary; the workbench keeps running for its current users.

**Negative / risks.** Two packages to keep in sync while the kernel API settles (mitigated by editable install and by freezing the workbench); the estimator registry and feature-set registry become shared state between packages (mitigated by making registration explicit in the kernel API, not import-time side effects); the phase release gates lose their scripts (mitigated by converting each invariant into a kernel or workbench test before deletion).

## Alternatives considered

- **Continue in place, add the agent platform inside `sohbench`.** Rejected: the search would inherit the closed menu, the service-centric orchestration and in-process execution; the direction rule cannot be enforced inside one package.
- **Rewrite everything from scratch.** Rejected: re-deriving label safety and protocols would repeat the August 2026 leak episode without the regression net that caught it.

## Branch decisions executed (2026-09-02)

- `master` fast-forwarded to `feat/phase2-dl` (49 commits, no divergence) and tagged **`v0-workbench`**; extraction work continues on branch **`kernel-extraction`** from that tag.
- `claude/gifted-burnell-c55ed8` deleted (no commits ahead).
- `codex/verification-fabric-phase0` (103 ahead / 33 behind; 82 files, +30 k lines) archived as tag **`archive/verification-fabric-phase0`**. A cherry-pick dry run of the 16 kernel-relevant commits produced conflicts in every case: the branch rebuilt `runner.py` and `report.py` around a `sohbench/fabric` subpackage (feature pipelines, outcomes, failures, evidence, specs) that mainline never had. **Decision: no cherry-picks; port by task.** The ideas worth porting, with their reference commits, are now tasks K7b and K7d–K7h in the extraction plan:
  - seed propagation into estimators (`_instantiate_estimator`: `random_state` / `set_seed`) — `d6b1efa`, `f86b753`;
  - feature pipelines fitted inside evaluation folds with causal isolation — `1fdf6ed`, `320b250`, `fabric/feature_pipeline.py`;
  - explicit failure and partial outcomes; failed units kept out of evidence — `45ed3a3`, `32d6884`, `fabric/outcomes.py`, `fabric/failures.py`;
  - trajectory metrics aggregated per asset and ambiguous cycle identities rejected — `4a29255`, `b65a429`;
  - EOL fold metric omitted when unavailable — `8ecb62a`;
  - asset-level confidence intervals in reports — `f0e8d4d`.
  Workbench-only material (run reservations, evidence-pack dual writes, Windows alias hardening, inference ranking ties) is not ported. The branch itself is kept until its stale Windows worktree reference is pruned (`git worktree prune && git branch -D codex/verification-fabric-phase0`), which the owner can run at any time since the tag preserves it.

## Follow-ups

- Extraction plan: `docs/superpowers/plans/2026-09-02-kernel-extraction.md`.
- Kernel invariants spec: `docs/superpowers/specs/2026-09-02-soh-kernel-invariants.md`.
- Platform architecture spec: `projects/soh-agent/docs/specs/agentic-automl-platform.md`.
- First platform milestone (M1): reproduce the 2026-08-27 LFP agent search (inner calce_cs2 + snl_lfp, outer hust) through the new candidate model inside a sandbox, matching `runs/agent_search/lfp_search_report.json` within seed tolerance.
