# Report: SPEC-008 Phase A, Split Integrity Audit

**Date:** 2026-11-02
**Phase:** A (read-only discovery)
**Companion documents:** `008-cohort-discovery.md`, `008-label-source-audit.md`, `../architecture/008-real-cohort-integration.md`

Status legend: **VERIFIED** / **INFERRED** / **NOT VERIFIED** / **BLOCKED** (see the label audit).

No split was created, changed or repaired.

---

## 1. Summary

| Question | Answer | Status |
|---|---|---|
| Do patient identifiers exist? | Yes, implicitly: `(cohort, T-number)`. There is no separate patient column. The slide suffix (`_01`, `_02`) distinguishes slides of one case. | VERIFIED (structure), INFERRED (that a case is one patient) |
| Are they reliable? | Internally consistent across all sources once qualified by cohort. Unqualified T-numbers collide 276 times across SR386 and SR1482. | VERIFIED |
| Does a patient have several slides? | SR386: never (427 cases, 427 slides). SR1482: **177 of 416 cases** have `_01` and `_02`. | VERIFIED |
| Does any patient occur in more than one partition? | **No.** In the labelled split (820 `_01` slides, 820 cases) each case is in exactly one partition. No `_02` slide is assigned anywhere. | VERIFIED |
| Is the legacy split leakage-safe as published? | **Yes at case level for the 820 `_01` slides.** It becomes unsafe only if `_02` slides are added without inheriting their case's partition. | VERIFIED (case level) |
| Is it consistent across targets? | Yes. Every KRAS/BRAF/NRAS/RAS/5-year-survival file puts each case in the same partition as the MSI split (0 disagreements over 3,161 rows). | VERIFIED |
| Is WSI content duplicated across partitions? | Not checkable for the cohort. The WSIs are not local and the S3 inventory is BLOCKED. Locally: 17 unique WSIs, 0 content duplicates across slide ids. | BLOCKED (cohort), VERIFIED (local) |
| Are the legacy split *assignments* trustworthy? | The membership is consistent. **How** it was generated (stratification, randomisation, seed, whether patient level) is not documented locally. | NOT VERIFIED |

**Verdict:**
- The published SurGen MSI split can be **preserved** for the 820 `_01` slides, with `(cohort, T-number)` as the group key (ADR-008 D4).
- No remediation is required **unless** the owner adds `_02` slides (D-2) or rejects the group key (D-4).

## 2. Partition membership

| Partition | Slides (= cases) | SR386 | SR1482 | SR1482 cases that also have a `_02` slide | Positives |
|---|---:|---:|---:|---:|---:|
| train | 490 | 255 | 235 | 102 | 50 |
| validate | 165 | 84 | 81 | 30 | 14 |
| test | 165 | 84 | 81 | 37 | 15 |
| **total** | **820** | **423** | **397** | **169** | **79** |

- The remaining 8 of the 177 `_02` slides belong to SR1482 cases without an MSI label.
- Partition sizes are 59.8 / 20.1 / 20.1 %.
- Positive rates are 10.2 / 8.5 / 9.1 %. INFERRED: this is consistent with a label-stratified split, but the generating procedure is not documented.
- The legacy README says "489 train". The file has 490 data rows, so that is a documentation off-by-one, not a data defect. VERIFIED.

## 3. Group (patient) integrity checks

| Check | Result | Status |
|---|---|---|
| Cases with slides in more than one partition (`_01` scope) | 0 | VERIFIED |
| Cases with slides in more than one partition if `_02` inherits the case partition | 0 (by construction) | INFERRED |
| Cases with slides in more than one partition if `_02` is assigned independently | Up to 177 cases at risk | INFERRED; **must not happen** |
| Same `(cohort, T)` listed twice within or across MSI files | 0 | VERIFIED |
| Slide assigned to two partitions in one source | 0 | VERIFIED |
| Partition disagreement between sources (S1–S4, S6) | 0 | VERIFIED |
| Partition disagreement between targets for the same case | 0 / 3,161 rows | VERIFIED |
| SR386 and SR1482 patients are disjoint | 427 + 416 = 843, matching the upstream "843 cases". No shared patient key exists to test further. | INFERRED / NOT VERIFIED |
| Same patient under two T-numbers (e.g. re-accessioned) | Not checkable without clinical identifiers | BLOCKED |
| Identifiers contain PII | No. Identifiers are study codes (`SR<study>_40X_HE_T<n>_<slide>`). | INFERRED |

## 4. Every condition that could cause leakage

| # | Condition | Present now? | Consequence | Mitigation in Phase B |
|---|---|---|---|---|
| L1 | Group key without cohort (`T198` in both cohorts) | Latent | Two different patients merged into one group. This over-constrains the split but does not leak. Splitting by an unqualified key **inside** one cohort is safe. | Group id = `surgen:<cohort>:T<nnn>`. The importer rejects unqualified keys. |
| L2 | Adding SR1482 `_02` slides with an independent split | Not present (no `_02` assigned) | Same patient in train and test, up to 177 cases | `_02` inherits the case's partition, or is excluded (D-2). SPEC-007 `GROUP_LEAKAGE` enforces this. |
| L3 | Joining on legacy `slide_id` (`slide_<i>`) | **Present in legacy code.** `scripts/pipeline/split_membership.py` keys membership on `slide_id`, and `SurGen_msi_validate/test.csv` each repeat 81 `slide_id` values, so `slide_0` exists in several splits and cohorts. | Wrong membership, or a crash | Never read `slide_id`. Use `czi_filename` / cohort-qualified case. |
| L4 | Treating SR386 as one slide per case | Holds today (427 = 427) | — | Assert it at import. Fail closed if a second SR386 slide appears. |
| L5 | The same WSI content under two names | Unknown for the cohort | Duplicate content across partitions | Phase B: SHA-256 per WSI from the S3/EBI inventory (or size + ETag first). Phase C/D: also the embedding-store content digest. Duplicate → `excluded_duplicate` + report. |
| L6 | Legacy preprocessing tuned on test slides | **Present in legacy.** `process_list_test.csv` holds per-slide hand-tuned segmentation (`sthresh`, `a_t`, …) for the 165 test slides, and 349 of 820 rows differ from upstream `segment_params_SurGen.csv`. | Test-informed preprocessing (ADR-008 D9) | New tissue detection must use one fixed, slide-independent configuration, or tuning only on train/validate. Legacy per-slide parameters are not imported. |
| L7 | Exploratory use of test labels | **Present in legacy.** `eda/` holds FP/FN/borderline analyses of test predictions. | Any later choice informed by them compromises the held-out claim | The SPEC-008 protocol must be frozen before the test set is evaluated. The report must disclose that the historical test set has been seen before. |
| L8 | Pilot or tuning on test-partition slides | Risk. Local fixture F-CZI-01 = SR1482 T198_01 is in **test**, and 5 of 16 local labelled slides are test (T004, T035, T108, T198, SR386 T568). | Preprocessing tuned on test images | The Phase C pilot uses train/validate slides only, or uses test slides strictly for throughput/integrity, with no parameter change from them. Recorded either way. |
| L9 | Regenerating the split | Not present | Hidden re-splitting | Import the upstream membership as an external split (SPEC-007 `import_split`), pin its digest, never regenerate. |
| L10 | Selection or thresholds on test | Not present in the new framework (SPEC-007 AC-016). Historical threshold 0.5 was fixed. | — | Validation-only selection, already enforced |

## 5. Legacy split trustworthiness

| Aspect | Assessment | Status |
|---|---|---|
| Origin | Published by the dataset authors with the dataset. It is the standard split for SurGen benchmarks. | VERIFIED (upstream repository) |
| Integrity | Internally consistent across 4 sources and 9 targets, case-disjoint | VERIFIED |
| Generation procedure | Undocumented locally (seed, stratification variables, unit) | NOT VERIFIED |
| Suitability as an external held-out test | It is a random internal partition of one dataset, not an external cohort. The historical test set has been used for evaluation and EDA (L7). | VERIFIED (history) |
| Cohort mixing | Each partition contains both sub-cohorts in similar proportion | VERIFIED |

## 6. Remediation

None is required for the `_01` scope. **No replacement split is proposed.**

If the owner chooses to include `_02` slides (D-2):
- Each `_02` slide joins its case's existing partition.
- This produces a **new split version**, because the membership changes.
- SPEC-007 validation must re-confirm 0 group overlap.
- The 8 `_02` slides of unlabelled cases stay excluded.

## 7. Owner decisions arising from this audit

| ID | Decision |
|---|---|
| D-2 | Exclude or include the SR1482 `_02` slides (inheriting partition). |
| D-4 | Accept `(cohort, T-number)` as the group identifier. |
| D-5 | Accept the upstream MSI split as the SPEC-008 split, knowing that its test partition has been used historically (L7). The alternative is a pre-registered decision to report it as a *reused benchmark test set* rather than a fresh held-out set. |
| D-6 | Confirm that the Phase C pilot draws only from train/validate (L8). |
