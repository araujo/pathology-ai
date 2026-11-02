# Report: SPEC-008 Phase A, MSI Label Source Audit

**Date:** 2026-11-02
**Phase:** A (read-only discovery)
**Companion documents:** `008-cohort-discovery.md`, `008-split-integrity-audit.md`, `../architecture/008-real-cohort-integration.md`

**Status legend**

| Status | Meaning |
|---|---|
| VERIFIED | Observed directly in files during this audit and reproducible with the commands in §9 |
| INFERRED | A conclusion drawn from verified observations but not directly proven |
| NOT VERIFIED | A claim from documentation or legacy reports that this audit did not check |
| BLOCKED | Could not be checked because access, data or authority is missing |

No label was changed, adjudicated or written anywhere. The audit only read files.

---

## 1. Summary

| Finding | Status |
|---|---|
| Four MSI label sources exist locally, in two physical copies. The four sources agree on **every** one of the 820 labelled slides. There are **0** label conflicts and **0** split conflicts. | VERIFIED |
| The four sources are **not independent**. `surgen_manifest.csv` and `SurGen_msi_*` are concatenations of `SR386_msi_*` and `SR1482_*_msi`, and all of them trace to one upstream publication (`CraigMyles/SurGen-Dataset`). Their agreement shows the copies are consistent. It does **not** show the labels are clinically correct. | VERIFIED (lineage), INFERRED (single origin) |
| The SPEC-007 "contradictory label" finding was comparing **different targets**. It set the MSI manifest (`label=0`) against `SR1482_*_kras.csv` (KRAS mutation). For MSI on its own, there is no contradiction. | VERIFIED |
| The combined `MMR_MSI` target is **heterogeneous**. In SR386 the upstream task is `MMR_LOSS` (`"MMR Loss"`/`"No MMR Loss"`). In SR1482 it is `MMR_MSI` (`"MSI/dMMR"`/`"MSS/pMMR"`). The combined `SurGen` CSVs merge both under one binary label. | VERIFIED (code constants), INFERRED (assay difference) |
| The original clinical label tables (assay, date, per-case source) are **not available locally**. Labels can be traced to the published split CSVs, not to clinical records. | BLOCKED |
| 23 of 843 cases (4 SR386, 19 SR1482) have no MSI label in any source. | VERIFIED |
| 177 SR1482 second slides (`_02`) are not referenced by any label file of any target. | VERIFIED |
| Potentially eligible for SPEC-008 (one `_01` slide per labelled case): **820 slides from 820 cases, 79 positive (9.6 %)**. | VERIFIED (counts); eligibility still needs owner decisions D-1 to D-4 |

---

## 2. Label sources

All paths are relative to `../digital-pathology/data/metadata/` unless stated otherwise. That directory is git-ignored in the legacy repository, so the files have no git history of their own. Their version is fixed here by SHA-256.

| # | Source | Rows | Digest (SHA-256) | Lineage | Status |
|---|---|---:|---|---|---|
| S1 | `SurGen/MMR_MSI/dataset_csv/SR386_msi_{train,validate,test}.csv` | 255 / 84 / 84 | train `634aa4f5…e168`, validate `c63c3638…bbb2`, test `3357089…046c0b` | Byte-identical to upstream `CraigMyles/SurGen-Dataset@e56ca8c` `reproducibility/dataset_csv/` (local clone `~/projects/SurGen-Dataset`, upstream commit `561de1d` 2025-01-20 "code init"). Also byte-identical to `../digital-pathology/reproducibility/dataset_csv/` (git-tracked, commit `a5c7752`). | VERIFIED |
| S2 | `…/SR1482_{train,validate,test}_msi.csv` | 235 / 81 / 81 | train `a64c6a18…31d1`, validate `f1f71a3a…6b02`, test `4b88a1fc…b78c` | Same as S1 | VERIFIED |
| S3 | `…/SurGen_msi_{train,validate,test}.csv` | 490 / 165 / 165 | train `6656c466…e6f9`, validate `86054594…8dd8`, test `97740d89…0c19` | Same as S1. Same cases, labels and splits as the union of S1 and S2. | VERIFIED |
| S4 | `surgen_manifest.csv` (+ `_train/_validate/_test`) | 820 (490 / 165 / 165) | `9b945ddb…9b8c` (full); train `94274012…21d1`, validate `bb96d1c1…37d4`, test `832605d9…0c40` | **Legacy-derived**, not upstream. Built from S3 by adding `case_id_norm` and `czi_filename`. The per-split files equal the filtered full file. A byte-identical copy exists at `SurGen/MMR_MSI/`. | VERIFIED (content), INFERRED (derivation) |
| S5 | `SurGen/MMR_MSI/smoke/smoke_{train,val}.csv` | 2 + 2 | `a5c3f553…6867` (both files identical) | Legacy smoke subset: 2 SR386 train slides | VERIFIED |
| S6 | Historical test predictions `…/experiments/SurGen/MMR_MSI/baseline_uni_transformer_full_224_nonoverlap/eda/merged_predictions_with_labels.csv` | 165 | `8f7a9af0…71be` | A `true_label` joined from S4 for the test split | VERIFIED |
| S7 | Non-MSI targets: `SR386_{kras,braf,nras,ras,5y_sur}_*`, `SR1482_*_{kras,braf,nras}` | 33 files | see `008-cohort-discovery.md` §2 | Upstream (same lineage as S1) | VERIFIED; **not MSI sources** |
| S8 | Original clinical/molecular tables behind the CSVs (EBI BioStudies S-BIAD1285 metadata, paper supplement) | — | — | Not present on this host. S3 access to `s3://digital-pathology-training-545053092614` failed (expired SSO token). | BLOCKED |

**Copies.** Every file under `data/metadata/dataset_csv/` is byte-identical to its counterpart under `data/metadata/SurGen/MMR_MSI/dataset_csv/` (36 of 36). Every top-level manifest is byte-identical to the one under `SurGen/MMR_MSI/` (4 of 4). VERIFIED.

**Source version.** The upstream label files have been unchanged since `561de1d` (2025-01-20). The exception is `13c1589` (2025-05-04), which added SR386 KRAS/NRAS files and does not touch MSI. VERIFIED from upstream git log.

## 3. Schemas, identifiers and vocabularies

| Source | Columns | `case_id` form | `slide_id` form | Label vocabulary |
|---|---|---|---|---|
| S1 SR386 | `case_id,slide_id,label` | `SR386_40X_HE_T<n>`, **unpadded**, no slide suffix (e.g. `T1`) | Mixed. Train: = case_id. Validate/test: `slide_<i>` positional index. | `0`, `1` |
| S2 SR1482 | same | `SR1482_40X_HE_T<nnn>_01`, **padded, with slide suffix** (a slide id, not a case id) | `slide_<i>` | `0`, `1` |
| S3 SurGen | same | Both forms above | Both forms. `slide_<i>` restarts at 0 in each sub-cohort block: **81 duplicate `slide_id` values** in validate and in test. | `0`, `1` |
| S4 manifest | `split,case_id,case_id_norm,label,czi_filename` | As S3. `case_id_norm` = `<cohort>_40X_HE_T<nnn>` (padded, no suffix). | — | `0`, `1` |

**Label meaning.** The CSVs hold integers only. Upstream `reproducibility/datasets/dataset_constants.py` documents the encodings:

| Upstream task | Cohort | Encoding | Status |
|---|---|---|---|
| `MMR_LOSS` | SR386 | `"MMR Loss": 1, "No MMR Loss": 0` | VERIFIED (constant) |
| `MMR_MSI` | SR1482 | `"MSI/dMMR": 1, "MSS/pMMR": 0` | VERIFIED (constant) |
| `MMR_MSI` | SurGen | `"MSI/dMMR": 1, "MSS/pMMR": 0` | VERIFIED (constant) |

- The SR386 MSI files therefore encode **MMR protein loss**, presumably by immunohistochemistry. They are used as MSI/dMMR in the combined task. INFERRED.
- MMR loss (dMMR) and MSI are concordant in most colorectal cancers, but they are different assays. The assay, panel and date behind each label are NOT VERIFIED (S8 BLOCKED).
- The positive class is `1` in both cohorts. The README of the legacy repository says `MSS/pMMR=0, MSI/dMMR=1`. VERIFIED.

## 4. Label conflict matrix (MSI)

The matching key is `(cohort, integer T-number)`. Cohort qualification is mandatory, because **276 T-numbers occur in both SR386 and SR1482** (e.g. `SR386_…_T198` is label 0 in train, while `SR1482_…_T198_01` is label 0 in test). A key without the cohort would merge different patients. VERIFIED.

| Pair compared | Cases in both | Same label | Different label | Same split | Different split |
|---|---:|---:|---:|---:|---:|
| S4 manifest vs S3 SurGen | 820 | 820 | **0** | 820 | **0** |
| S4 vs S1 (SR386) | 423 | 423 | **0** | 423 | **0** |
| S4 vs S2 (SR1482) | 397 | 397 | **0** | 397 | **0** |
| S3 vs S1 ∪ S2 | 820 | 820 | **0** | 820 | **0** |
| S5 smoke vs S4 | 2 | 2 | **0** | — (smoke "val" reuses 2 train slides) | — |
| S6 historical `true_label` vs S4 | 165 | 165 | **0** | 165 (all test) | **0** |

**Conflicting MSI records: 0.** VERIFIED.

### 4.1 Correction of the SPEC-007 finding

SPEC-007 discovery §4 reported that the SurGen CSVs "disagree for the same slides". The rows it cited were these:

| Slide | S4 (MSI) | `SR1482_*_kras.csv` (KRAS) |
|---|---|---|
| SR1482 T002_01 | 0 (train) | 0 (train) |
| SR1482 T005_01 | 0 (train) | 1 (train) |
| SR1482 T008_01 | 0 (train) | 1 (train) |
| SR1482 T368_01 | 0 (train) | 1 (train) |
| SR1482 T198_01 | 0 (test) | 1 (test) |

The KRAS file is a different target, so different values are expected. **This is not a label contradiction.** VERIFIED. SPEC-008 §1 inherits the claim, and the owner should note the correction (decision D-0).

The underlying lesson still applies: legacy files do not name their target in a column. A file's meaning comes only from its filename, so the target has to be declared explicitly in the Phase B adapter.

## 5. Missing labels and coverage

The cohort reference is the upstream segmentation parameter lists (`segment_params_SR386.csv`, 427 rows; `segment_params_SR1482.csv`, 593 rows). Together they cover all 1,020 WSIs and 843 cases, matching the upstream README ("1,020 WSIs from 843 cases"). VERIFIED.

| Cohort | Cases | Slides | Cases with MSI label | Cases without MSI label | Slides without MSI label |
|---|---:|---:|---:|---:|---:|
| SR386 | 427 | 427 (`_01` only) | 423 | 4: T004, T208, T429, T430 | 4 |
| SR1482 | 416 | 593 (416 `_01` + 177 `_02`) | 397 | 19 (list below) | 19 `_01` + 177 `_02` = 196 |
| **Total** | **843** | **1,020** | **820** | **23** | **200** |

- **SR1482 cases without an MSI label:** T007, T032, T058, T059, T088, T109, T119, T141, T149, T157, T163, T214, T228, T237, T298, T305, T334, T342, T356.
- **Six of them have other targets.** T032, T109 and T149 appear in BRAF/KRAS/NRAS train files. T214, T305 and T356 appear in KRAS/NRAS train files. Their MSI label is absent, not unknown-by-format. VERIFIED.
- **The four SR386 cases have no label of any target.** VERIFIED.
- The reason labels are missing is not documented locally. NOT VERIFIED.
- **No label file of any target references a `_02` slide** (0 rows). Whether a `_02` slide inherits its case's MSI status is a scientific decision (D-2). INFERRED: MSI/MMR status is normally a property of the tumour or patient, but `_02` slides in SR1482 may be metastatic or other-site blocks (upstream README: SR1482 "including metastatic lesions (liver, lung, peritoneum)").

## 6. Duplicates and ambiguous mappings

| Check | Result | Status |
|---|---|---|
| Duplicate `case_id` within any MSI file | 0 | VERIFIED |
| Duplicate case across partitions within a cohort | 0 | VERIFIED |
| Duplicate `czi_filename` in S4 | 0 | VERIFIED |
| Duplicate `slide_id` in S3 validate / test | **81 each** (`slide_0…slide_80` used by both sub-cohort blocks) | VERIFIED; **hazard** (see split audit §4) |
| `slide_id` usable as an identifier | **No.** `slide_<i>` is a positional index (legacy `docs/surgen_data_model.md` says so explicitly). | VERIFIED |
| SR386 `case_id` → WSI file | Needs zero-padding (`T1` → `T001_01.czi`). S4 `czi_filename` does this for all 423. | VERIFIED |
| SR1482 `case_id` → WSI file | `case_id` already contains `_01`. | VERIFIED |
| Cross-cohort T-number collisions | 276 | VERIFIED; handled by cohort-qualified keys |
| Label value types | Only `0`/`1` in MSI files. `FAIL` appears once in `SR386_braf_train.csv` (not MSI). | VERIFIED |

## 7. Class balance (MSI, `_01` slides = cases)

| Split | SR386 neg/pos | SR1482 neg/pos | Total | Positive % |
|---|---|---|---:|---:|
| train | 235 / 20 | 205 / 30 | 490 | 10.2 % |
| validate | 78 / 6 | 73 / 8 | 165 | 8.5 % |
| test | 78 / 6 | 72 / 9 | 165 | 9.1 % |
| **all** | 391 / 32 | 350 / 47 | **820** | **9.6 %** |

VERIFIED. The held-out test has **15 positive cases**, so any test-set AUROC carries a wide interval (see `008-cohort-discovery.md` §7).

## 8. Eligibility (provisional, not applied)

The counts below apply SPEC-008 §6.3 categories to the observed data. **No record has been assigned a status.** Assignment happens in Phase B after the owner decides.

| Proposed status | Slides | Basis | Depends on |
|---|---:|---|---|
| `eligible` (candidate) | 820 | `_01` slide with a consistent MSI label in S1–S4 and a known split | D-1 (accept upstream labels as the authority), D-3 (combined target) |
| `excluded_missing_label` | 23 | Case without any MSI label | — |
| `pending_review` / `excluded_missing_label` | 177 | SR1482 `_02` slides, no slide-level label | D-2 |
| `excluded_conflict` | 0 | No MSI conflict | — |
| `excluded_missing_group` | 0 | Group = cohort-qualified case, derivable for all slides | D-4 |
| `excluded_duplicate` | 0 locally | WSI content duplicates cannot be checked without the WSIs | BLOCKED (AC-008) |
| `excluded_missing_wsi` | unknown | 16 of 820 eligible WSIs are on this host | BLOCKED (storage) |
| `excluded_quality` | unknown | Not assessed | Phase C/D |

## 9. Reproducing this audit

- **Tool.** The audit used a standalone, read-only Python script (standard library `csv` and `hashlib` only) in the session scratchpad. It was not added to the repository.
- **Logic.** Load each CSV, normalise identifiers to `(cohort, int(T))`, and compare labels and splits pairwise.
- **Committing it.** A version of this audit should become the Phase B cohort adapter's import check (architecture §5), not a one-off script.
- **Verifying a source.** `shasum -a 256 <file>` against §2.

## 10. Owner decisions arising from this audit

| ID | Decision |
|---|---|
| D-0 | Acknowledge that the SPEC-007 "contradiction" was a cross-target comparison (§4.1), and close it as a non-issue for MSI. |
| D-1 | **Label authority.** Accept the upstream published SurGen split CSVs (S1/S2 at upstream `e56ca8c`, pinned by SHA-256) as the authoritative MSI source, **or** require the original clinical tables (S8) first. The labels cannot be traced further locally. |
| D-2 | **`_02` slides.** Exclude the 177 SR1482 `_02` slides (as upstream does), or let them inherit the case label as additional same-group slides. |
| D-3 | **Target definition.** Accept a combined target, "SR386 MMR loss ∪ SR1482 MSI/dMMR = positive", as one endpoint, or restrict the primary analysis to one sub-cohort. Either way, record both source vocabularies in label provenance. |
| D-4 | **Group identity.** Accept `(cohort, T-number)` as the patient/group identifier. This also means accepting that SR386 and SR1482 case numbers refer to disjoint patients (supported by 427 + 416 = 843 cases in the upstream README, but not independently verifiable). |
