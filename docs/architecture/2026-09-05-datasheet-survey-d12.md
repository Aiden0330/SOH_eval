# D12 — datasheet survey of the nine onboarded datasets (2026-09-05)

The owner delegated the `VERIFY` fields of the existing datasheets to an agent survey (Wave 6 plan §0, answer 5). Rule kept
from Spec 06: a field is filled only from a primary source — the record page or archive readme, the peer-reviewed paper
that describes the experiment, or **the raw archive itself**, measured. Where no primary source exists the field says so and
stays `VERIFY`. Chemistry signatures are untouched (D4/D1); the sheets record the survey in their `status`.

## What the survey used

| Evidence | How |
|---|---|
| Raw archives on disk | `scripts/measure_raw_provenance.py` lists every raw file with size and sha256, folds them into one `raw_sha256`, and measures the logging interval (median per-cycle time step of the charge curves through `sohkernel.raw_windows.iter_charge_curves`; direct `.mat` reads for nasa and oxford). Output: `data/datasheets/_provenance.json`, rendered into each sheet's `sampling`, `digests.raw` and `provenance`. |
| Record pages | Mendeley (hust), UConn Digital Commons (uconn), ORA readme on disk (oxford), RWTH Publications (rwth), BatteryArchive study page and the BatteryLife mirror README (snl), CALCE Battery Data page (calce), the PCoE README inside the archive (nasa). |
| Measured protocol values | charge current and cut-off voltages read from one raw cycle per dataset (cell and cycle named in the sheet). |

## Resolved

| Dataset | Field | Value | Source |
|---|---|---|---|
| all nine | `digests.raw`, `provenance.raw_files` | sha256 per raw file and the folded digest | measured |
| hust | `sampling` | 5 s | measured (record: none stated) |
| hust | charge stages | CC at 5 C (5.50 A) to ≈3.54 V, then a CC-CV stage; four CC discharge stages; 1.94 V reached | measured from `our_data/1-1.pkl` |
| hust | licence | CC BY 4.0 | Mendeley record page |
| snl_lfp, snl_nmc, snl_nca | `sampling` | 120 s | measured |
| snl_* | cut-offs | LFP 0.5 C to 3.60 V, discharge to 2.0 V; NMC 4.20 / 2.0 V; NCA 4.20 / 2.5 V | measured, consistent with JES 167 120532 |
| snl_* | licence | no licence text; attribution terms (BatteryArchive "double attribution", BatteryLife mirror README) | study page, mirror README |
| uconn | `sampling` | 1 Hz | record ("sampled at 1Hz") and measured 1.00 s |
| uconn | RPT charge | 1 C (1.20 A) to 3.60 V; RPT every ~100 cycles (~3.5 days) | measured; record |
| calce_cs2 | `sampling` | 30 s (Arbin xlsx) | measured |
| calce_cs2 | licence, equipment, cut-offs | citation terms quoted; Arbin BT2000 (CADEX for CS2_8, CS2_21); 0.5 C to 4.20 V, 2.7 V discharge cut-off | CALCE page; measured |
| oxford | `sampling` | 1 s (time stored as a fraction of a day) | measured; readme |
| oxford | `windows` | none calibrated (characterisation cycles only) — explicit, no longer `VERIFY` | this survey |
| nasa | protocol | 1.5 A CC to 4.2 V, CV to 20 mA; 2 A discharge; room temperature | README.txt inside the archive (2008-09-30) |
| nasa | `sampling` | variable, ≈9 s median | measured |
| rwth_isea_public | `sampling` | not applicable (per-cycle capacity histories; no time series ingested) | mirror `.mat` inspected |
| rwth_isea_public | provenance, reference test | mirror `.mat` under `work/public_sources/`; RPT wording from the record and the mirror README | record page; GitHub API |

## Still `VERIFY`, and why

| Dataset | Field | Why |
|---|---|---|
| nasa | cathode, anode, chemistry signature | no primary source names the chemistry (PCoE description and the archive README both say only "Li-ion 18650, 2 Ah"); secondary literature is split |
| hust | cell model | the record page names none; the EES paper's text was not fetchable; BatteryLife/BatteryML call the cells A123 18650 LFP (secondary) |
| uconn | test temperatures | the record names "11 unique conditions" without listing them |
| rwth_isea_public | licence, cycling profile (C-rates), temperature | all three sit in `Content_RWTH-2021-04545.pdf`, which the publication server serves as a download only; the GitHub mirror has no LICENSE. **The owner opens the PDF to close them** |

## For the owner

The three open items above are one PDF (RWTH) and two facts that only the papers' full texts hold (hust cell model, uconn
temperatures). Everything else on the nine sheets now traces to a record page, a readme or a measurement that a script
can repeat.
