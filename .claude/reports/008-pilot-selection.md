# Report: SPEC-008 Phase C1, Extraction Pilot Selection

**Date:** 2026-11-02
**Prompt:** PROMPT-008C1
**Companions:**
- `008-wsi-inventory.md`: the inventory used here
- `008-phase-c1-readiness.md`: scale, procedure, go/no-go

**Outcome:**

| Item | Result |
|---|---|
| Selection | `pai-pilot-5da7534bcaa4d30c93f4eae703b3d479`: 8 slides, 4 strata × 2 |
| Status | **`requires_acquisition`**: every stratum is filled from train/validation; 5 of the 8 images must be downloaded first |
| Test slides selected | **0**. The 5 locally verified test slides were not used. |
| Shortfalls | 0 |
| Pending decisions carried in the selection | D-7 (scale), D-9 (compute/storage/download), D-10 (UNI and SurGen terms), as recorded in manifest v1's ledger |
| Current decision state | **D-7 approved 2026-11-02 (0.5 µm/px ± 15 %)**; D-9 and D-10 pending (readiness report §0) |

> **Update, 2026-11-02:** D-7 was approved after this selection was made: 0.5 µm/px ± 15 %, `uni-official@1`, 224 × 224 RGB.
> - The selection is unchanged, and it does not depend on scale.
> - Its frozen `pending_decisions` still lists D-7, because it derives from the immutable manifest v1.
> - The pilot runs at 0.5 µm/px only: ≈ 99 k patches, ≈ 48 min of MPS. The 1.0 µm/px column in §3 is kept as the evidence that was considered.

The selection is deterministic: re-running it from the same manifest, criteria and inventory gives the same document. `verify_selection` re-derives it, and a regression test does so on every test run.

> The pilot is a technical check of the extraction configuration: memory, throughput, output integrity, resume and cost. It is not a scientific sample. No pilot output may inform labels, thresholds, hyperparameters or model selection (ADR-008 D9).

---

## 1. Criteria

Configuration: `projects/cohorts/surgen-msi/pilot.yaml` (SHA-256 `333fe05f…5f3f`). Schema: `pathology-ai.pilot-config/1`.

| Criterion | Value | Why |
|---|---|---|
| Plan id | `surgen-msi-pilot-c1` | |
| Seed | `20261102` | Recorded; ranking is `SHA-256("<seed>:<slide id>")` |
| Partitions | `train`, `validation` | D-6. The selector refuses any partition outside the manifest's approved `pilot_partitions`, and always refuses `test`. |
| Strata | `cohort × label` | Both sub-cohorts (SR386, SR1482) and both classes (MSI-negative, MSI-positive) |
| Per stratum | 2 | 8 slides: replication within each stratum, at a cost the pilot can afford (readiness §1) |
| Preference | `verified_images` | Within a stratum, locally content-verified slides come first, then the seeded rank. This cuts downloads; the bias is acceptable for a technical pilot (§5). |
| Authorising decision | D-6 | It must be approved and gate `pilot`. A pending decision (e.g. D-10) is refused as the authority. |

**Selection rules (generic, `domain/pilot.py`):**
- Strata are the product of field values over **all** eligible records. A class that exists only in test therefore appears as a **shortfall**, never as a substitution from another stratum or partition.
- At most one slide per group (patient).
- Inputs are bound by digest: cohort `854f009f…5a3c`, inventory `783f44e2…36e2`.

## 2. Strata (VERIFIED)

| Stratum | Candidates in train + validation | Eligible in test (never used) | Selected | Already verified locally | Shortfall |
|---|---:|---:|---:|---:|---:|
| SR1482 · negative | 278 | 72 | 2 | 2 | 0 |
| SR1482 · positive | 38 | 9 | 2 | 0 | 0 |
| SR386 · negative | 313 | 78 | 2 | 1 | 0 |
| SR386 · positive | 26 | 6 | 2 | 0 | 0 |

Candidate counts equal the manifest's partition × sub-cohort × label counts (`008-label-reconciliation.md` §3.3). For example, SR1482 positive = 30 train + 8 validation.

## 3. Selected slides (VERIFIED)

Patch counts are for the SPEC-006 UNI pipeline's tissue and plan settings (readiness §1).
- **Measured** counts come from local tissue detection and patch planning: no pixel extraction, no UNI.
- **Estimated** counts scale the slide's legacy Zenodo row count by the ratio measured on the 11 local train/validation slides (0.878 at 1.0 µm/px; 3.50 at 0.5 µm/px).

| # | Slide | Group | Partition | Label | Image | Bytes | Patches @1.0 | Patches @0.5 |
|---:|---|---|---|---|---|---:|---:|---:|
| 1 | SR1482_40X_HE_T368_01 | surgen:SR1482:T368 | train | negative | verified (`9f35f73d…`) | 848,763,072 | 599 measured | 2,484 measured |
| 2 | SR1482_40X_HE_T156_01 | surgen:SR1482:T156 | validation | negative | verified (`2363380e…`) | 579,305,120 | 501 measured | 2,013 measured |
| 3 | SR1482_40X_HE_T224_01 | surgen:SR1482:T224 | train | **positive** | to acquire | — | ≈ 3.8 k est. (legacy 4,380) | ≈ 15.3 k est. |
| 4 | SR1482_40X_HE_T130_01 | surgen:SR1482:T130 | train | **positive** | to acquire | — | ≈ 3.4 k est. (legacy 3,926) | ≈ 13.7 k est. |
| 5 | SR386_40X_HE_T558_01 | surgen:SR386:T558 | validation | negative | verified (`619200db…`) | 2,670,401,216 | 4,085 measured | 16,171 measured |
| 6 | SR386_40X_HE_T440_01 | surgen:SR386:T440 | train | negative | to acquire | — | ≈ 2.4 k est. (legacy 2,767) | ≈ 9.7 k est. |
| 7 | SR386_40X_HE_T489_01 | surgen:SR386:T489 | train | **positive** | to acquire | — | ≈ 5.6 k est. (legacy 6,398) | ≈ 22.4 k est. |
| 8 | SR386_40X_HE_T044_01 | surgen:SR386:T044 | validation | **positive** | to acquire | — | ≈ 4.3 k est. (legacy 4,933) | ≈ 17.3 k est. |

| Pilot total | Patches @1.0 µm/px | Patches @0.5 µm/px |
|---|---:|---:|
| Measured (3 local slides) | 5,185 | 20,668 |
| Estimated (5 slides to acquire) | ≈ 19.7 k (per-slide range 14.6–23.5 k) | ≈ 78 k (range 58–94 k) |
| **Total** | **≈ 25 k** | **≈ 99 k** |

Partition mix: 6 train and 2 validation slides. Every group is distinct, and every slide is `provisional` in `surgen-msi@1` (image gates open), which is expected before the pilot.

## 4. Missing requirements (instead of substitution)

The pilot cannot run until these five images exist locally and are content-verified. **No test slide and no other stratum was used to avoid this.**

| Slide | Public source (from the legacy download lists, VERIFIED present) |
|---|---|
| SR1482_40X_HE_T224_01 | `https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/285/S-BIAD1285/Files/SR1482_WSIs/SR1482_40X_HE_T224_01.czi` |
| SR1482_40X_HE_T130_01 | `…/Files/SR1482_WSIs/SR1482_40X_HE_T130_01.czi` |
| SR386_40X_HE_T440_01 | `…/Files/SR386_WSIs/SR386_40X_HE_T440_01.czi` |
| SR386_40X_HE_T489_01 | `…/Files/SR386_WSIs/SR386_40X_HE_T489_01.czi` |
| SR386_40X_HE_T044_01 | `…/Files/SR386_WSIs/SR386_40X_HE_T044_01.czi` |

| Requirement | Detail | Status |
|---|---|---|
| Download route approved | EBI HTTPS (public) or S3 (needs SSO re-authentication) | **D-9 pending** |
| Data terms recorded | SurGen terms (SPEC-003 lists F-CZI-01 as CC BY 4.0; not confirmed for this study in Phase A) and UNI terms (CC-BY-NC-ND 4.0, gated) for this study | **D-10 pending** |
| Disk | About 22 GB estimated (INFERRED from about 1 MB per legacy row on the local slides; range about 15–25 GB). 58 GiB free. | OK if the estimate holds |
| Transfer time | About 10 h at the 0.6 MB/s observed by the legacy pipeline from EBI (NOT VERIFIED for this host) | Acceptable |
| Verification after download | Re-run `pathology-ai-cohort inventory` with the download directory as a root. All 8 must be `verified`, with no new duplicate content. Then re-run the selection: it must reproduce the same 8 slides with status `ready`. | Procedure defined (readiness §3) |

Re-running the selection after the downloads changes its inventory digest and preference ranks. Within each stratum, the 5 slides to acquire are the top seeded-rank candidates among the slides not verified locally. Once verified, they compete only with other verified slides:
- the SR1482 negatives are unaffected;
- in SR386 negative, T558 and T440 are the only verified slides;
- in the positive strata, the downloaded slides are the only verified ones.

The same 8 slides are therefore expected, now `ready` (INFERRED from the ranking rule). The re-run is still required, and it is recorded as a new selection id.

## 5. Biases and limits of this selection

- **Local preference.** The 3 local negatives come from slides the legacy project cached for its own diagnostics (`spec0245_corpus`, `train/`, `validate/`). This is fine for a resource and integrity check. It is not a representative scientific sample.
- **Size spread.** The selection includes T558: 2.67 GB, and 16,171 patches at 0.5 µm/px, the largest measured plan among the 11 local train/validation slides apart from T008. The largest local train slide (T008, 6.0 GB, 19,142 patches) was not selected.
  - Memory is bounded by batch size, not by slide size (SPEC-005 AC-002), so slide size drives time, not peak memory.
  - Adding T008 as an explicit stress slide would be an owner choice: a new plan id, not an edit to this one.
- **Positives come only from download.** They depend on D-9 and D-10.
- **Two slides per stratum** gives no statistical power and is not meant to. Pilot outputs are integrity and resource measurements only.

## 6. Tests

| File | Tests | Covers |
|---|---:|---|
| `tests/unit/test_pilot_selection.py` | 23 | determinism and seed sensitivity; rank order; no test slides; test partition refused; partitions outside the approved pilot partitions refused; shortfall without substitution; partition strata; verified-first preference; preference needs an inventory; acquisition list; one slide per group; authorising decision must be approved and gate `pilot`; pending decisions recorded; inventory from another manifest refused; forged, re-inventoried or re-cohorted selections refused; invalid criteria |
| `tests/integration/test_cohort_inventory_pilot.py` | 3 | CLI inventory → pilot on a synthetic cohort (a verified test slide is never chosen); CLI refuses a test partition (`PILOT_SELECTION`); **SurGen regression**: the committed selection reproduces from the committed manifest, criteria and inventory |

## 7. Files written

| Path | SHA-256 |
|---|---|
| `projects/cohorts/surgen-msi/pilot.yaml` | `333fe05fa510af95ef7d9774b9047f39a7e012fa7a8c14442a51bc26057c5f3f` |
| `projects/cohorts/surgen-msi/pilots/pai-pilot-5da7534bcaa4d30c93f4eae703b3d479/pilot-selection.json` | `bc320a1b502e816ec5d0f04835b7cd97a5071ed9c4b026e00d6868268d464754` |

Selection digest: `5da7534bcaa4d30c93f4eae703b3d479e2e7bca13fa4849821260b5be1d0daf0`.

## 8. Reproducing

```bash
uv run pathology-ai-cohort pilot \
    --manifest projects/cohorts/surgen-msi/manifests/v1/cohort-manifest.json \
    --criteria projects/cohorts/surgen-msi/pilot.yaml \
    --inventory projects/cohorts/surgen-msi/inventories/pai-wsi-inventory-783f44e2ea71bead23ce78c80f069e12/wsi-inventory.json \
    --out projects/cohorts/surgen-msi/pilots
```
