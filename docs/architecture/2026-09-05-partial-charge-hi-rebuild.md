# Partial-charge health indicators — rebuild 2026-09-05

K12 (Wave 6 plan §4): `sohkernel.health_indicators` computed from the raw charge curves and written into the processed tables; the charge columns are audited under the label-safety DOD rule on every rebuild.

| Dataset | Chemistry | Windows (V) | Rows | Span matched | Span coverage (median of rated) | Audit |
|---|---|---|---|---|---|---|
| uconn | LFP | 3.30–3.33, 3.33–3.35, 3.35–3.38, 3.38–3.40 | 2855 | 2855 (100.0%) | 0.6161 | passed · feature column 'hi_span_dq_ah' integrates ≈62% of rated capacity on 'UCONN' — wider than the typical 40–50% DOD partial window (LFP: plateau region); allowed below 90% |
| calce_cs2 | LCO | 3.85–3.90, 3.90–3.95, 3.95–4.00, 4.00–4.05 | 15411 | 11840 (76.8%) | 0.4367 | passed |
| snl_lfp | LFP | 3.30–3.33, 3.33–3.35, 3.35–3.38, 3.38–3.40 | 64905 | 64898 (100.0%) | 0.5577 | passed |
| snl_nmc | NMC | 3.70–3.80, 3.80–3.90, 3.90–4.00, 4.00–4.10 | 14283 | 14283 (100.0%) | 0.4001 | passed |
| snl_nca | NCA | 3.70–3.80, 3.80–3.90, 3.90–4.00, 4.00–4.10 | 12080 | 10954 (90.7%) | 0.4553 | passed |
| hust | LFP | 3.30–3.33, 3.33–3.35, 3.35–3.38, 3.38–3.40 | 146122 | 146122 (100.0%) | 0.0495 | passed |

## Availability per column (fraction of rows with a value)

| Dataset | hi_dq_w1_ah | hi_dq_w2_ah | hi_dq_w3_ah | hi_dq_w4_ah | hi_span_dq_ah | hi_dt_w1_s | hi_dt_w2_s | hi_dt_w3_s | hi_dt_w4_s | hi_span_dt_s | hi_ic_peak_ah_per_v | hi_ic_peak_v | hi_v_q25 | hi_v_q50 | hi_v_q75 | hi_dvdq_v_per_ah | hi_i_mean_a | hi_i_cv |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| uconn | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| calce_cs2 | 0.77 | 0.79 | 0.80 | 0.80 | 0.77 | 0.77 | 0.79 | 0.80 | 0.80 | 0.77 | 0.81 | 0.81 | 0.77 | 0.77 | 0.77 | 0.77 | 0.77 | 0.80 |
| snl_lfp | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| snl_nmc | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| snl_nca | 0.91 | 0.96 | 0.97 | 0.97 | 0.91 | 0.91 | 0.96 | 0.97 | 0.97 | 0.91 | 0.98 | 0.98 | 0.91 | 0.91 | 0.91 | 0.91 | 0.91 | 0.97 |
| hust | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
