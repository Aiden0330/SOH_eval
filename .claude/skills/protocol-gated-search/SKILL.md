---
name: protocol-gated-search
description: Build a search/AutoML/agent loop that selects algorithms, features, or parameters WITHOUT corrupting the benchmark — nested inner/outer evaluation, hard admission gates, single-shot held-out claims. Use when automating model/feature/hyperparameter selection for any estimation tool, adding an LLM agent to a training pipeline, or adding per-dataset adaptation at inference.
---

# Protocol-gated search (safe AutoML / agent loops)

Reference implementation: `sohbench.agent_search` + design doc
`docs/superpowers/specs/2026-08-26-agent-guided-search-design.md`.

## The two failure modes to design against

1. **The optimizer finds the leak.** Leaky/degenerate configurations score
   best by construction; a search will select them. Admission gates
   (label-safety, coverage rules, schema/protocol feasibility) must run
   BEFORE scoring as hard constraints — excluded candidates never execute,
   and their exclusion reasons go in the report.
2. **Winner's curse.** N configurations scored on the same validation data
   make the winner's score optimistically biased. Separate:
   - **inner loop** — data the search may see freely (score under the
     benchmark's own protocol, e.g. per-dataset LOCO folds); rank by
     *worst-case* across inner datasets, not the mean;
   - **outer loop** — one or more datasets never touched during search;
     evaluate ONLY the final top-k, ONCE, under the cross-domain protocol
     (LODO). Report all k outcomes, not just the winner. The report carries
     an explicit `integrity` block stating the separation.

## Structure

- `Policy.propose(history) -> candidates` — the only pluggable brain. Ship
  a deterministic grid/heuristic policy first; an LLM policy plugs into the
  same protocol later. LLMs do the *structural* moves (diagnose failure
  modes, pick feature families per chemistry); continuous hyperparameters
  belong to a classical inner optimizer the policy calls as a tool.
- **Constraint oracle** = the pipeline's existing preflight/admission
  check, so search and production enforce identical rules. Never
  reimplement gates inside the search.
- Scoring uses the pipeline's own runner/protocol machinery — never a
  custom split.

## Phase discipline (training vs adaptation)

- Training-phase search may tune anything, scored on held-out folds.
- Inference/adaptation-phase actions are restricted to information
  legitimately available at deployment: unlabeled target-domain feature
  statistics (alignment, re-normalization), raw signal shape (window
  re-fitting by coverage target), and small labeled calibration sets if
  the deployment genuinely provides them (few-parameter output
  recalibration only). Pre-register each adaptation operator's information
  channel; audit its outputs like any feature.

## Provenance

Every decision the search makes (chosen features, window bounds, adaptation
operators) must land in the model/run manifest so downstream verdicts are
auditable and future rule changes can invalidate affected artifacts.
