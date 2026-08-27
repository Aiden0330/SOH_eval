---
name: feature-window-calibration
description: Calibrate physically-parameterized features (voltage/SOC window bounds, segment selections, band-pass limits) per dataset from raw data. Use when onboarding a new dataset or cell type to an estimation tool, when a windowed feature's coverage looks near-total, or when designing partial-window features for a new chemistry or acquisition protocol.
---

# Per-dataset calibration of windowed features

Fixed global window bounds are wrong across acquisition protocols: the same
LFP chemistry needed [3.30, 3.52] V at 5C fast-charge (hust) but
[3.30, 3.38] V at 0.5C (snl) and [3.30, 3.37] V at 1C RPT (uconn), because
overpotential moves the plateau. Calibrate per dataset, from raw curves.

## 1. Choose the integration segment deliberately

Pick ONE consistently-acquired segment per dataset (e.g. the CC charge
phase, or the reference-test charge) — never mix segments. Identify it from
the raw data's phase/state labels, not by heuristics on the signal.

## 2. Scan candidate bounds empirically

For a grid of [lo, hi] bounds, on a sample of cells × cycles compute:

- **coverage** = window integral / full-quantity per cycle → per-dataset
  median (robust to outliers);
- **R²(feature, label-proxy)** — the informativeness/leakage tradeoff.

Target the domain's accepted band (battery partial windows: 40–50% DOD;
hard bound < 90% — beyond that the integral IS the calibration quantity).
Prefer bounds centered on the physically meaningful region (LFP: plateau).
Reject any choice whose R² vs the label approaches the affine-leak
threshold — a window can be too informative to be legal.

## 3. Rebuild with a verifying pipeline

Recompute the feature from raw for every processed row via a script that:

- verifies the raw↔processed **join** before writing (cross-check an
  independent quantity, e.g. full capacity, on sampled rows; reconstruct
  cumulative indices where raw resets per session) and **refuses to write**
  below a match-rate floor (≥95%);
- re-runs the leak audit on the rebuilt column and refuses on violations
  or warnings;
- backs up originals, then updates in place;
- updates every contract surface: the dataset's window metadata file, the
  data manifest's digests, and the standing leak-inventory pins (with a
  dated review note including measured before/after coverage).

## 4. Record the calibration as provenance

The chosen bounds are model-relevant decisions: write them into dataset
contracts and (for searchable windows) model manifests, so inference can
re-derive the feature identically and audits can trace the choice. Old
artifacts built on the previous bounds have different feature semantics —
add them to the invalidation sweep.
