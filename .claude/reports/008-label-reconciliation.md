# Report: SPEC-008 Phase B, MSI Label Reconciliation

**Date:** 2026-11-02
**Prompt:** `.claude/prompts/008B-cohort-reconciliation-and-manifest-construction.md`
**Companion:** `008-phase-b-implementation.md` (code, tests, gates). Phase A inputs: `008-cohort-discovery.md`, `008-label-source-audit.md`, `008-split-integrity-audit.md`.

**Outcome:** Cohort manifest `surgen-msi@1` was built from pinned sources.

| Slides | Status |
|---:|---|
| 1,020 | recorded |
| **820** | **provisionally eligible** |
| 200 | excluded |
| 0 | conflicts |
| 0 | pending review |

Eligibility stays **provisional** until every WSI has been located, hashed and checked for duplicate content. None of that was done in Phase B.

Status legend:
- **VERIFIED**: produced or checked in this phase and reproducible with the commands in §8
- **INFERRED**: a conclusion drawn from verified facts
- **NOT VERIFIED**: claimed elsewhere and not checked here
- **BLOCKED**: could not be checked

> No label was adjudicated, inferred or changed. No split was generated or repaired. No WSI was read and no model was run. No clinical or performance claim is made.

---

## 1. Owner decisions applied

The repository owner (Gustavo Medeiros de Araujo) confirmed decisions D-0 to D-6 and D-8 on 2026-11-02 by answering "Approve all" in the Phase B session.
- Each decision is recorded in `projects/cohorts/surgen-msi/cohort.yaml` with its approver, date and reference.
- The same record is copied into the manifest.

| ID | Decision | How it is enforced |
|---|---|---|
| D-0 | Close the SPEC-007 "MSI label contradiction" as a cross-target (MSI vs KRAS) comparison error | Recorded. Each source declares its target, and a source whose target is not accepted is rejected (`COHORT_RECONCILIATION`). |
| D-1 | The published upstream MSI CSVs, pinned by SHA-256, are the label authority | The two per-sub-cohort upstream sources have role `authority`. A slide with only corroborating labels would become `pending_review`. |
| D-2 | Exclude SR1482 `_02` slides from the primary analysis | Slide policy `include_indices: ["01"]` |
| D-3 | Combined MSI/dMMR endpoint, keeping each source cohort's label meaning | Target `msi_dmmr`. It accepts the source targets `MMR_LOSS` and `MMR_MSI`, and each record keeps the target its authority names. |
| D-4 | Group key = cohort-qualified T-number | `surgen:{cohort}:T{case:03d}`. The configuration loader rejects a group template that leaves out the cohort. |
| D-5 | Keep the upstream split; the test partition is a reused benchmark | `external@1`, `test_role: reused_benchmark` |
| D-6 | The future extraction pilot uses only train/validation slides | `pilot_partitions: [train, validation]`, recorded in the manifest split |
| D-8 | Legacy-embedding comparability arm is documented, not imported | Recorded. No embedding was imported. |
| D-7, D-9, D-10 | Scale; compute/storage/download route; UNI and SurGen terms | **Pending.** They gate `extraction`/`pilot`, not reconciliation. |

## 2. Source identities (VERIFIED)

The upstream root is a local clone of `github.com/CraigMyles/SurGen-Dataset` at commit `e56ca8cd4d2768ecb0eccd4e8cb269f7e460de4e`.
- The MSI files have not changed since `561de1d` (2025-01-20).
- Every file was re-hashed on read and matched its pin. A mismatch fails before parsing (`COHORT_SOURCE_INTEGRITY`).

| Source id | Role | Target (as named upstream) | File | Rows | SHA-256 |
|---|---|---|---|---:|---|
| upstream-roster-sr386 | roster | — | `segment_params_SR386.csv` | 427 | `c4b9657437b13ec57aaa130485abf56d8d74f1b242938df04d063a0c13e76d8b` |
| upstream-roster-sr1482 | roster | — | `segment_params_SR1482.csv` | 593 | `7bd751cc40223796bb18ae3731521fce60e7e382edd6efeb09b53f9f26ef35b7` |
| upstream-sr386-msi | authority | MMR_LOSS | `SR386_msi_train.csv` | 255 | `634aa4f5e8fa9dfd80847d1839cf0740549f590772a5c31bbeb957ac7a78e168` |
| | | | `SR386_msi_validate.csv` | 84 | `c63c3638f85cb2bcf5ef9d0e364f452abe630db4b965c389ddce855df2d1bbb2` |
| | | | `SR386_msi_test.csv` | 84 | `3357089409034fdfd3b84d0452dcf3917908938769729e406d814a7513046c0b` |
| upstream-sr1482-msi | authority | MMR_MSI | `SR1482_train_msi.csv` | 235 | `a64c6a18ba5bf3d8258e3b74a0ef08e78da458271830f53cffe1e3226ca531d1` |
| | | | `SR1482_validate_msi.csv` | 81 | `f1f71a3a268a35d6e33c74b23d30e3141cdaa7f1d5b412b6decfe9d066066b02` |
| | | | `SR1482_test_msi.csv` | 81 | `4b88a1fce560df85baeabb64bc86c03c9f243d5a632ed2eb09be68292e76b78c` |
| upstream-surgen-msi | corroborating | MMR_MSI | `SurGen_msi_train.csv` | 490 | `6656c4663f69c6e28bd7b4e3cb8ad627eb869209af6789a6b8932b6d3780afe5` |
| | | | `SurGen_msi_validate.csv` | 165 | `8605459457096ca32ed3c0eaa7587819f070effb39d72435eae46a12ef515dd8` |
| | | | `SurGen_msi_test.csv` | 165 | `97740d89417b6e6a58c84c064f2db198a79edf30b3b9b33d0112bf8e1d30cc19` |
| legacy-manifest | corroborating | MMR_MSI | `surgen_manifest.csv` (legacy `data/metadata/`, git-ignored) | 820 | `9b945ddbdecc2eee9a90f18172b553d675288e0fd1dd77145dd40fcd34939b8c` |

**Identifier columns read:**
- Upstream sources: `case_id`.
- Legacy manifest: `czi_filename`, with the partition taken from `split`.
- Rosters: `slide_id`, which in those files holds the `.czi` file name.
- The upstream `slide_id` column (a positional `slide_<i>` index) is **never read**.

**Values:**
- Label vocabulary in every source: `"0"` → `negative`, `"1"` → `positive`.
- Partition names: `train` / `validate` / `test` → SPEC-007 `train` / `validation` / `test`.

## 3. Reconciliation result (VERIFIED)

Every one of the 820 eligible slides has **three** agreeing candidates:
- its sub-cohort's authority file
- the combined upstream file
- the legacy manifest

There are 0 label conflicts, 0 partition conflicts and 0 resolutions. There are also 0 case-level labels on a multi-slide case.
- SR386 ids are unpadded and carry no slide suffix (`SR386_40X_HE_T1`).
- They are attached to the single roster slide of their case (`SR386_40X_HE_T001_01`), which is allowed only because every SR386 case has exactly one slide.

### 3.1 Eligibility counts

| Status | Reason code | SR386 | SR1482 | Total |
|---|---|---:|---:|---:|
| `eligible` (provisional) | — | 423 | 397 | **820** |
| `excluded_missing_label` | `no_label`: no source labels the slide or case | 4 | 19 | **23** |
| `excluded_missing_label` | `slide_policy`: `_02` slide, excluded by D-2, unlabelled | 0 | 177 | **177** |
| `excluded_conflict` | — | 0 | 0 | 0 |
| `excluded_missing_group` | — | 0 | 0 | 0 |
| `excluded_duplicate` | not assessable (no WSI inventory) | — | — | 0 recorded |
| `excluded_missing_wsi` | not assessable (no WSI inventory) | — | — | 0 recorded |
| `excluded_quality` | not assessed (Phase C/D) | — | — | 0 recorded |
| `pending_review` | — | 0 | 0 | 0 |
| **Records** | | **427** | **593** | **1,020** |

There are 843 groups (cases): SR386 427, SR1482 416.

### 3.2 Exclusions by original split

Excluded slides carry no partition in the manifest (`unassigned`), because no source assigns them one. For context only:

| Excluded group | Count | Case's partition in the labelled split |
|---|---:|---|
| `no_label` | 23 | none (no source labels the case) |
| `slide_policy` (`_02`) | 177 | the `_01` slide of the same case is in **train 102 / validation 30 / test 37**; 8 belong to unlabelled cases |

`no_label` slides:
- SR386: T004, T208, T429, T430.
- SR1482: T007, T032, T058, T059, T088, T109, T119, T141, T149, T157, T163, T214, T228, T237, T298, T305, T334, T342, T356.
- All are `_01` slides. These are exactly the Phase A lists.

### 3.3 Eligible slides by partition, sub-cohort and label

| Partition | SR386 neg / pos | SR1482 neg / pos | Slides = groups | Positive |
|---|---|---|---:|---:|
| train | 235 / 20 | 205 / 30 | 490 | 50 (10.2 %) |
| validation | 78 / 6 | 73 / 8 | 165 | 14 (8.5 %) |
| test | 78 / 6 | 72 / 9 | 165 | 15 (9.1 %) |
| **total** | 391 / 32 | 350 / 47 | **820** | **79 (9.6 %)** |

Label semantics are kept per record (`source_targets`):
- The 423 SR386 records carry `MMR_LOSS`.
- The 397 SR1482 records carry `MMR_MSI`.

## 4. Imported split (VERIFIED)

| Item | Value |
|---|---|
| Strategy | `external@1` (SPEC-007), imported from the partitions stated by the authority and corroborating sources, which all agree |
| Assignment digest (canonical sorted `(slide_id, partition)` list) | `0c8c94fc9d73ff7ff87f55c57ef9ce8a8a1fc6dd07e64534daf8f7514d065616` |
| `split.csv` (SPEC-007 `slide_id,partition` table, 820 rows) SHA-256 | `48105e865e68cd33f7f0c06fe8350015a8352793a3d45730e8f8fb629701f8cf` |
| Group overlap between partitions | **0**. Each eligible group has one slide, and no group spans partitions or sub-cohorts. |
| Cross-cohort T-number collisions | 276 in the data. All are kept apart by the qualified key (e.g. `surgen:SR386:T198` in train, `surgen:SR1482:T198` in test). |
| Test role | `reused_benchmark`: used historically for evaluation and EDA (D-5) |
| Pilot partitions | train, validation (D-6) |
| `split.csv` loads with SPEC-007 `load_split_table` | yes: 490 / 165 / 165 |

**Status of the SPEC-007 `SplitManifest`:**
- A `SplitManifest` (`pathology-ai.mil-split/1`) needs a dataset manifest, which needs embedding bags, so it cannot exist before Phase D.
- `application.cohort_service.import_external_split` performs that import through SPEC-007 `import_split`.
- It first refuses any dataset that does not cover exactly the eligible slides, with the same labels and groups.
- The path is tested end to end on synthetic bags (§ companion report).

## 5. Manifest (VERIFIED)

| Item | Value |
|---|---|
| Location | `projects/cohorts/surgen-msi/manifests/v1/` |
| Schema | `pathology-ai.cohort/1` |
| Manifest digest (canonical body) | `854f009fdb30e41304b9959a53bc3fd2e0beede091bd4f47061af931383a5a3c` |
| Manifest id | `pai-cohort-854f009fdb30e41304b9959a53bc3fd2` |
| `cohort-manifest.json` file SHA-256 | `f7748332c6ce866e857ad0e724cd9e24e23182f7092a2752803239da143bd084` |
| `reconciliation.csv` SHA-256 (one row per slide with every candidate's `source:file:row=raw`) | `69c3728c64ede8bc106e9ff0def9e879ab57636c836c4212cd3a361c06241781` |
| Configuration `cohort.yaml` SHA-256 | `2871ef23323e6375985fdc99bf45a479175dcbf5318b1c3c4c426b3a6348379d` |
| Reproducibility | A rebuild from the pinned sources gives the identical digest (opt-in test, run in this phase) |

**Contents:**
- Study codes only: slide, case and file name.
- No names, dates or clinical fields.

**Write-once behaviour:**
- Writing the same content again is accepted.
- Writing different content is refused (`EXPERIMENT_CONFLICT`).
- A membership change needs `version: "2"` in a new directory.

## 6. WSI availability and integrity: NOT VERIFIED

| Gate | State |
|---|---|
| `wsi_integrity` (present, size, SHA-256) | **Open for all 820.** No inventory was run. Phase A found 16 of 820 eligible WSIs locally; S3 was BLOCKED. |
| `wsi_duplicate_content` | **Open for all 820.** It cannot be assessed without digests of every WSI (AC-008). |
| Quality (`excluded_quality`) | Not assessed |

**Contract support:**
- The manifest contract already handles inventory evidence: `missing` → `excluded_missing_wsi`, `unreadable` → `excluded_quality`, a shared digest → `excluded_duplicate` for every slide that has it.
- Closing these gates produces manifest **v2**.

Until then, the SPEC-007 hand-over refuses the records unless `allow_provisional=True` is passed explicitly.

## 7. What this reconciliation does not establish

- **Clinical correctness of the labels.** The agreement is between copies of one publication. The clinical tables behind them are BLOCKED (Phase A S8).
- **The assay behind each label.** SR386 labels encode MMR protein loss and SR1482 labels encode MSI/dMMR. The combined endpoint mixes two assays (INFERRED).
- **Patient disjointness across sub-cohorts.** It is INFERRED from 427 + 416 = 843, and cannot be verified without clinical identifiers.
- **Split generation.** How the split was produced (seed, stratification) is undocumented (NOT VERIFIED). Its integrity is verified; its procedure is not.

## 8. Reproducing

```bash
uv run pathology-ai-cohort reconcile --config projects/cohorts/surgen-msi/cohort.yaml \
    --source-root upstream=~/projects/SurGen-Dataset --source-root legacy=../digital-pathology \
    --out projects/cohorts/surgen-msi/manifests/v1      # idempotent: identical content is accepted
uv run pathology-ai-cohort verify --manifest projects/cohorts/surgen-msi/manifests/v1/cohort-manifest.json
PATHOLOGY_AI_SURGEN_UPSTREAM=~/projects/SurGen-Dataset PATHOLOGY_AI_SURGEN_LEGACY=../digital-pathology \
    uv run pytest tests/integration/test_surgen_cohort_manifest.py -rs
```

The sibling repositories were only read. Their `git status` is unchanged: `../digital-pathology` 4 lines, `~/projects/SurGen-Dataset` 5 lines (same as Phase A).
