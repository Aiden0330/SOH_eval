# Datasheet signing review and the redistribution decision (2026-09-07)

The owner's instruction of 2026-09-07 was three words: push the branch, keep the non-commercial sets and MATR internal
only and non-redistributable, sign. This record is the owner-delegated review behind the signatures on the ten Wave 6
datasheets (`scripts/draft_datasheets.py`, regenerated sheets under `data/datasheets/`) and what it found.

## 1. The redistribution decision

`source.redistribution` is a new datasheet field. Five sheets carry the owner's decision verbatim:

| set | licence found by the survey | decision |
|---|---|---|
| `turku_rc_nca`, `turku_rc_nmc`, `turku_rc_lfp` | CC BY-NC 3.0 on the FairData record (Metax re-read 2026-09-07); the Data in Brief article is CC BY 4.0 | internal evaluation only, non-redistributable |
| `field_lfp_28` | CC BY-NC 4.0 (Zenodo record and README) | internal evaluation only, non-redistributable (in addition: unlabeled, never training or claims) |
| `matr` | no licence text; data.matr.io asks to cite the papers | internal evaluation only, non-redistributable |

"Internal evaluation only, non-redistributable" means: the derived tables, curves and window banks stay in this
repository's data directory; no released artifact, claim pack or leaderboard record redistributes them; the set is cited
with its record and paper wherever a result depends on it. The five CC BY 4.0 sets (`tongji_*`, `forklift_lfp180`,
`home_storage_lfp`) say "redistributable with attribution under CC BY 4.0". The licence table of
`scripts/build_data_manifest.py` carries the same words; the committed `data/manifest.json` is not rebuilt by this task
(its digests are pinned by suite v2), so its licence strings update at the next manifest rebuild (K12b).

## 2. The signatures

`signed.by` = "owner-delegated review 2026-09-07 (Claude Fable 5.1) on the owner's instruction of 2026-09-07 …" on
eight sheets: `tongji_nca`, `tongji_ncm`, `tongji_blend`, `forklift_lfp180`, `matr`, `turku_rc_lfp`, `field_lfp_28`,
`home_storage_lfp`. For each, the chemistry evidence the data session quoted from the primary source was re-read against
the record metadata fetched on 2026-09-07 (Zenodo API records 6405084, 12091223, 13715694; the Metax record of the
FairData dataset; the archive README of `field_lfp_28`), and the cathode written into `chemistry.cathode` is the kernel's
label (`metadata.DATASET_CHEMISTRY`): the authors' electrode compositions stay in `chemistry.evidence` (`tongji_ncm` is
the authors' "NCM", `tongji_blend` the 42/58 NCM–NCA blend). Everything the sources do not state stays open with the
reason: the anode of the field sets and of the Turku LFP cells, the forklift cell model, the field temperatures, the
time zones, the raw digests and provenance (the D12 measurement ran for the nine older sets only), the forklift
checksum, the MATR checksums and two of its verification notes, and the home-storage label-rule notes.

`sohkernel.datasheets.chemistry_verified` changed with this review: the anode may stay unverified. The kernel's chemistry
label, the partial-window spans and the lcho folds key on the cathode, and several primary sources name the cathode only;
an open anode no longer makes a sheet unverified, while an open cathode still does (pinned in the tests).

## 3. The two sheets that stay unsigned: the Turku Samsung cells

The article (Heydarzadeh et al., Data in Brief 60, 111531; Europe PMC full text re-read 2026-09-07) assigns NCA to the
four Samsung ICR18650-26J cells and NMC to the two Samsung INR21700-40T cells, citing the Samsung specification for the
40T. The manufacturer's documents read differently:

- **INR21700-40T.** Samsung SDI's safety data sheet (2019-01-01, rev. 01) lists the cathode as "lithiated metal oxide
  (Cobalt, Nickel)" and the composition as lithium nickel oxide 25–35 %, cobalt lithium dioxide 1–5 %, aluminium lithium
  oxide 0.1–1 %, with no manganese compound anywhere in the table. That is the composition of an NCA cell.
- **ICR18650-26J.** Samsung SDI's specification ICR18650-26J1 v1.0 records the chemistry as a tick box (LCO / LMO / NCA /
  NCM) that the text export does not resolve; vendor databases (Cell Saviors, DNK Power) list the 26J as NMC (LiNiCoMnO2).

So the article's labels may be swapped or simply wrong for both cells. The two sheets stay unsigned with the cathode
marked open and both sources quoted (`chemistry.cathode`, `chemistry.evidence`, `known_issues`); the kernel labels
(`turku_rc_nca` = NCA, `turku_rc_nmc` = NMC) follow the article until the owner decides, and no chemistry-transfer claim
may cite either set meanwhile. The lcho preflight warns on them, as it should. Resolving it is the owner's: a Samsung
SDI specification that states the cathode in readable text, or the authors, settles it; if the labels change, the two
datasets are renamed (their names carry the chemistry) and the suite v2 tasks `cross-chem-lfp2nca-turku` and any record
citing them are re-read. `turku_rc_lfp` (JGNE JGPFR26650, 3.2 V nominal, 3.6 V charge cut-off) is consistent and signed.

## 4. Tests

`tests/kernel/test_datasheets.py`: the verified-and-signed set now holds sixteen sheets; the two Turku Samsung sheets are
pinned unsigned with the cathode open; the five internal-only sheets carry the decision in `source.redistribution` and
`source.license`; the CC BY sets say so; the anode rule of `chemistry_verified` has its own case.
