# Report: SPEC-008 Phase C1, Extraction Pilot Readiness

**Date:** 2026-11-02
**Prompt:** PROMPT-008C1 (WSI Inventory and Extraction Pilot Readiness)
**Companions:**
- `008-wsi-inventory.md`: inventory adapter and the scoped local inventory
- `008-pilot-selection.md`: the deterministic pilot plan

> **Update, 2026-11-02 (after review): D-7 approved.** The owner approved an extraction scale of **0.5 µm/px ± 15 %**, using the existing `uni-official@1` profile and 224 × 224 RGB model input. §0 records the approval and its governance effects. The sections below are updated accordingly; the 1.0 µm/px figures are kept as the evidence that was considered. D-9 and D-10 remain pending.

> **Update, 2026-11-02 (PROMPT-008D9): D-9 approved, planning only.** EBI HTTPS acquisition, local Mac with MPS, the 8 pilot slides, `uni-official@1` at 0.5 µm/px, fp32, batch 16. It authorizes no download or execution. The directory plan, readiness checks and fail-closed preflight are in `008-d9-infrastructure-readiness.md` and `008-phase-c2-execution-plan.md`. G9 below is resolved; G10–G12 remain open.

**Outcome:**

| Item | Result |
|---|---|
| Phase C1 engineering | **Complete** |
| WSI inventory adapter, scoped report | Done. 17 slides verified locally, 0 duplicates. Cohort gates still open (by design). |
| Pilot selection plan | Done. 8 slides from train/validation, 0 test, 0 shortfall. **5 images to acquire.** |
| Scale and compute comparison (0.5 vs 1.0 µm/px) | Done. Patch planning was **measured** on 11 local slides; cohort figures are **estimates**. |
| Extraction procedure and acceptance criteria | Documented (§3) |
| D-7 extraction scale | **Approved 2026-11-02: 0.5 µm/px ± 15 %**, `uni-official@1`, 224 × 224 RGB (§0) |
| **Go/no-go for Phase C2 (pilot execution)** | **NO-GO.** Blocked by D-9 and D-10, and by acquiring the 5 images (§4–§5). |
| SPEC-008 | Remains **BLOCKED**. AC-008 is open; AC-009 is not started. |

Constraints honoured:
- No WSI was downloaded, and no image file was modified.
- UNI was not executed. No MIL training was run.
- No full-cohort processing took place.
- Split membership is unchanged (`0c8c94fc…5616`).
- D-7 was approved by the owner, not inferred. D-9 and D-10 were not approved.
- Manifest v1 was neither modified nor regenerated (§0).
- Sibling repositories were not changed (`git status`: 4 and 5 lines, as in Phases A and B).
- No Git commit was made.

**What touched real images:**
1. Read-only SHA-256 hashing of the 32 local copies (17 slides).
2. Tissue detection and patch planning (`extract.materialize_pixels=false`: no patch pixels decoded) on the **11 local train/validation slides**, to measure patch counts at both scales.

No test-partition slide was used for any measurement.

Legend:
- **MEASURED**: produced on this host in C1
- **PRIOR**: measured in an earlier phase, with its source given
- **ESTIMATE**: extrapolated; the assumption is stated

---

## 0. D-7 approval and its governance effects

| Item | Record |
|---|---|
| Decision | D-7: extraction scale **0.5 µm/px ± 15 %**. A stored pyramid level within tolerance is read and nothing is resampled. Existing `uni-official@1` profile; 224 × 224 RGB model input; pipeline `wsi-patch-embedding-uni@1.0.0` as committed (`plan.target_spacing_um` 0.5, `embed.model_spacing_um` 0.5, tolerance 0.15). |
| Not approved | 1.0 µm/px, a dual-binding pilot, and any change to the encoder recipe or pipeline file |
| Approver / date / reference | Gustavo Medeiros de Araujo (repository owner) / 2026-11-02 / owner instruction after the PROMPT-008C1 review |
| Gates | `extraction` (unchanged) |
| Effect on SurGen CZI (MEASURED, 11 slides) | Level 2 (downsample 4), 0.44496 µm/px, −11.0 % (inside the 15 % tolerance); patch field of view 99.7 µm |
| Ledger | `projects/cohorts/surgen-msi/cohort.yaml` (D-7 entry); ADR-008 D7 and its decision record; SPEC-008 §9 note |

**Manifests (unchanged):**
- `manifests/v1/` keeps the same three files, byte for byte (`f7748332…d084`, `48105e86…f8cf`, `69c3728c…1781`).
- The v1 digest is `854f009f…5a3c` and the split assignment digest is `0c8c94fc…5616`.
- v1 records D-7 as **pending**, which was true when it was built. It is not edited.

**Configuration:**
- `cohort.yaml` now declares **version "2"**.
  - Old SHA-256: `2871ef23…379d`.
  - New SHA-256: `4e54a9ee…fb69`.
- A manifest embeds its decision ledger. Keeping version "1" would allow two different documents named `surgen-msi@1`.
- **Manifest v2 was not built.**

**Tests (verified):**
- **v1 still reproduces.** The opt-in rebuild test rebuilds v1 from the current configuration with exactly the documented delta reverted (version "1", D-7 pending), and obtains the identical digest.
- **Only the ledger changes.** The current definition yields the same records, split, counts and sources. `test_configuration_differs_from_v1_only_by_the_d7_approval` runs without data and checks this on every test run.
- **v1 cannot be overwritten.** A reconcile aimed at a copy of `manifests/v1/` is refused with `EXPERIMENT_CONFLICT`, and the files are unchanged.

**Documents bound to v1 (unchanged and still valid):**
- The local inventory `pai-wsi-inventory-783f…` and the pilot selection `pai-pilot-5da7…` are bound to the v1 digest.
- The selection's frozen `pending_decisions` field still lists D-7, because it is derived from the v1 ledger. The current state is D-7 approved, with D-9 and D-10 pending.
- Building manifest v2 changes the cohort digest. The inventory's identity source and the selection are bound to that digest, so both would then be re-taken against v2:
  - the slide membership is identical;
  - the same 8 slides are expected (INFERRED from the ranking rule);
  - the result would carry new ids.
- **Recommendation:** decide whether C2 runs on v1 (citing the D-7 record) or on an explicitly built v2. Both are owner steps; neither was taken here.

## 1. Scale and compute comparison

### 1.1 The two bindings (evidence for D-7; 0.5 µm/px approved)

Both bindings use the **same** SPEC-006 profile, `uni-official@1`: RGB, ImageNet normalisation, official transform, no resampling, fp32, checkpoint `56ef09b4…fb40`. They differ only in the explicit physical-scale binding: `plan.target_spacing_um` and `embed.model_spacing_um`, tolerance 0.15.

- **The encoder recipe is not changed.**
- **No pipeline file was edited.** `pipelines/wsi-patch-embedding-uni.v1.yaml` still binds 0.5 µm/px.
- The binding is part of `EncoderIdentity`, so stores of the two bindings have different identities. SPEC-007's compatibility key rejects any mixture (AC-011).

| | **0.5 µm/px** (**approved, D-7**) | **1.0 µm/px** (considered, not approved) | Legacy SurGen (reference only) |
|---|---|---|---|
| Stored level used on SurGen CZI | level 2 (downsample 4) | level 3 (downsample 8) | level 0, resampled |
| Effective spacing | 0.44496 µm/px, −11.0 % (MEASURED, 11/11 slides) | 0.88992 µm/px, −11.0 % (MEASURED, 11/11) | nominal 1.0 (≈ 2013 px → 224 with `INTER_AREA`) |
| Patch field of view | 99.7 µm | 199.3 µm | ≈ 224 µm |
| Channel order / resampling | RGB / none | RGB / none | BGR / `INTER_AREA` |
| Relation to UNI's documented pretraining ("20×", no µm/px stated) | Operator interpretation of 20× (SPEC-006 O1) | About 10×: outside the documented magnification. Effect on embedding quality NOT VERIFIED. | — |
| Reproduces the legacy embeddings | No | **No** (11 % finer, RGB, no resampling, other tissue detector) | — |
| Change needed | none | two explicit overrides, or (recommended) a new versioned pipeline file after D-7 approval | — |

**Scientific trade-off considered for D-7:**
- 0.5 µm/px is closer to UNI's documented pretraining magnification. It was approved.
- 1.0 µm/px is 4× cheaper and nearer the historical field of view, but is not the legacy scale.
- A historical comparison stays descriptive either way (ADR-008 D11).

**Consequences of the approved scale:**
- About 4× the patches, encode time and store volume of 1.0 µm/px (§1.3).
- The Phase D storage need (≈ 52 GB) is now firm, which makes the D-9 storage location a hard requirement (B6).

### 1.2 Patch planning, measured on 11 local train/validation slides

Settings: the UNI pipeline's `tissue` and `plan` stages (saturation-Otsu detector; 224 px; `min_tissue_fraction` 0.5; `min_acquired_fraction` 0.9; `max_patches` 100,000). Only the scale binding was overridden.

The legacy count is the row count of the slide's Zenodo store (`features/.zarray` shape, read as JSON).

| Slide | Partition | @0.5: candidates → selected | @1.0: candidates → selected | Legacy rows | 0.5 / 1.0 | 1.0 / legacy |
|---|---|---:|---:|---:|---:|---:|
| SR1482 T002_01 | train | 7,854 → 1,815 | 1,925 → 446 | 516 | 4.07 | 0.86 |
| SR1482 T005_01 | train | 34,773 → 12,370 | 8,600 → 3,121 | 3,442 | 3.96 | 0.91 |
| SR1482 T008_01 | train | 41,410 → 19,142 | 10,302 → 4,808 | 5,794 | 3.98 | 0.83 |
| SR1482 T231_01 | train | 8,400 → 998 | 2,100 → 250 | 346 | 3.99 | 0.72 |
| SR1482 T348_01 | train | 14,378 → 2,039 | 3,549 → 504 | 682 | 4.05 | 0.74 |
| SR1482 T368_01 | train | 14,850 → 2,484 | 3,690 → 599 | 776 | 4.15 | 0.77 |
| SR1482 T371_01 | train | 14,110 → 1,449 | 3,486 → 348 | 481 | 4.16 | 0.72 |
| SR1482 T085_01 | validation | 4,940 → 706 | 1,235 → 165 | 252 | 4.28 | 0.65 |
| SR1482 T156_01 | validation | 10,375 → 2,013 | 2,542 → 501 | 624 | 4.02 | 0.80 |
| SR1482 T424_01 | validation | 11,305 → 1,362 | 2,772 → 337 | 472 | 4.04 | 0.71 |
| SR386 T558_01 | validation | 32,175 → 16,171 | 7,952 → 4,085 | 3,895 | 3.96 | 1.05 |
| **Total** | | **60,549** | **15,164** | **17,280** | **3.99** | **0.878** |

Pooled ratio of 0.5 to legacy: **3.50**.

**Planning cost (MEASURED):**
- 0.4–2.3 s wall per slide.
- Peak RSS ≤ 216 MiB per process.
- `max_patches` is never reached: the largest plan is 19,142 patches.

**What the measurements change:**
- The 0.5/1.0 ratio is 4, as Phase A assumed.
- But the SPEC-004 detector with `min_tissue_fraction` 0.5 selects fewer patches than the legacy CLAM pipeline did. At 1.0 µm/px it selects 0.88× the legacy rows, even with an 11 % smaller field of view.
- Phase A's 0.5 µm/px estimate (14.5 M patches) was therefore about 15 % high.

**Representativeness:** 10 of the 11 slides are SR1482. The only SR386 slide (T558) has a ratio of 1.05, so the SR386 ratio may be higher (INFERRED). The pilot's 4 SR386 slides will measure this.

### 1.3 Cohort-wide extrapolation (ESTIMATE)

Assumptions:
- Legacy row total for the 820 eligible slides = **3.61 M** (PRIOR, Phase A).
- Scaled by the pooled ratios measured above; ranges use the per-slide extremes (0.65–1.05).
- UNI encode time per patch: PRIOR, SPEC-006 §9, measured on F-CZI-01 with 178 patches at batch 16. MPS **29 ms**, CPU **53 ms**. Sustained throughput and thermal behaviour are NOT VERIFIED.
- Store size: 4,128 B per row (1,024 × fp32 + 4 × float64 level-0 rectangle).
- WSI volume: 820 × 1.60 GB (mean of the 17 local files).
- Free disk: 58 GiB (MEASURED).

| Quantity (820 slides) | **0.5 µm/px** | **1.0 µm/px** | Phase A said (0.5 / 1.0) |
|---|---:|---:|---|
| Patches | **≈ 12.6 M** (9.4–15.1 M) | **≈ 3.2 M** (2.4–3.8 M) | 14.5 M / 3.61 M |
| Embedding stores | ≈ 52 GB (39–62 GB) | ≈ 13 GB (10–16 GB) | 60 GB / 15 GB |
| UNI encode, MPS | ≈ 102 h = **4.2 days** (3.2–5.1 d) | ≈ **25 h** (19–31 h) | 4.9 d / 29 h |
| UNI encode, CPU | ≈ 186 h = 7.8 days | ≈ 47 h | 8.9 d / 53 h |
| Per-slide fixed cost (checkpoint load and hash about 8 s, PRIOR; planning ≤ 2.3 s, MEASURED) | ≈ 2.3 h | ≈ 2.3 h | — |
| UNI encode, one CUDA GPU | NOT VERIFIED | NOT VERIFIED | — |
| WSI volume | ≈ 1.3 TB | ≈ 1.3 TB | 1.2–1.5 TB |
| Fits the 58 GiB free disk? | Stores alone ≈ 52 GB (up to 62): **no margin** | Stores ≈ 13 GB: yes, with per-slide WSI streaming | — |

### 1.4 Pilot cost (8 slides; 3 measured + 5 estimated, see the pilot report §3)

| | 0.5 µm/px | 1.0 µm/px |
|---|---:|---:|
| Patches | ≈ 99 k | ≈ 25 k |
| UNI encode, MPS / CPU | ≈ 48 min / 88 min | ≈ 12 min / 22 min |
| Stores | ≈ 0.41 GB | ≈ 0.10 GB |
| Downloads (5 slides) | ≈ 22 GB, ≈ 10 h at 0.6 MB/s (INFERRED; transfer rate NOT VERIFIED) | same |

The pilot is affordable on this host at **either** scale.

**With D-7 approved,** the pilot runs at 0.5 µm/px only: ≈ 99 k patches, ≈ 48 min of MPS (≈ 88 min CPU) and ≈ 0.41 GB of stores. The dual-binding option is withdrawn.

Pilot stores carry the Phase D identity, so they are reusable in Phase D (`resume_verified`) if they pass A1–A11.

## 2. Pilot plan (summary)

- Selection: `pai-pilot-5da7534bcaa4d30c93f4eae703b3d479`.
- **Already verified locally (3):** SR1482 T368 (train, negative), SR1482 T156 (validation, negative), SR386 T558 (validation, negative).
- **To acquire (5):** SR1482 T224 and T130 (train, positive); SR386 T440 (train, negative); SR386 T489 (train, positive); SR386 T044 (validation, positive).
- The selection has 0 test slides and 0 shortfalls. Details are in `008-pilot-selection.md`.

## 3. Future extraction procedure (Phase C2 pilot, then Phase D)

Nothing in this section has been executed. Every step writes new, versioned artifacts and never modifies an existing one.

### 3.0 Preconditions (all required)

| # | Precondition |
|---|---|
| P1 | ✅ D-7 approved 2026-11-02: 0.5 µm/px ± 15 %, `uni-official@1`, 224 × 224 RGB |
| P2 | D-9 approved: download route (EBI HTTPS or S3), staging directory, compute device (MPS / CPU / CUDA host), storage location for stores |
| P3 | D-10 approved: UNI CC-BY-NC-ND 4.0 non-commercial terms cover this study; SurGen data terms recorded |
| P4 | Owner approval to start Phase C2 |
| P5 | `uv run pytest`, `ruff`, `mypy` green on the commit/tree used |

### 3.1 Download and verification

1. Download only the slides named in the selection's `to_acquire()` list into a **new staging directory** outside the repository and outside the existing roots, e.g. `<D-9 location>/surgen/wsi/`.
   - Use a resumable client writing `<name>.part`, then rename after a complete transfer. Never write into the legacy caches or `~/Downloads`.
2. Record per file:
   - URL;
   - HTTP `Content-Length`, `ETag` and `Last-Modified`, as remote metadata only. **An ETag is never accepted as a SHA-256.**
3. Inventory the staging directory together with the existing roots:
   `pathology-ai-cohort inventory --root pilot-wsi=<staging> --root … --pattern '*.czi' --out projects/cohorts/surgen-msi/inventories`
4. Accept only if:
   - all 8 pilot slides are `verified`;
   - each size equals `Content-Length`;
   - no `inconsistent_copies` and no `duplicate_content_groups`;
   - `unmatched` contains no partial files.
5. Make the verified downloads read-only (`chmod a-w`).
6. Re-run `pathology-ai-cohort pilot` with the new inventory. It must select the same 8 slides with status `ready`, under a new selection id that records the new inventory digest.

**Trust on first download:** if EBI publishes no checksum, the first SHA-256 is the content identity. Where possible, confirm it against a second copy (S3, or a repeat download).

### 3.2 Readability and geometry

- `pathology-ai-imaging-smoke --file <czi> --image-id <id>` per slide. Record:
  - backend `czi-pylibczirw`;
  - level-0 spacing 0.1112 µm/px;
  - stored downsamples including 4 and 8 (MEASURED on 11 slides; required on all 8).
- **Image id.** `ImageId` accepts lowercase only, so use the lowercase canonical slide id (e.g. `sr1482_40x_he_t368_01`). Record the slide id → image id mapping in the run ledger (§3.6).

### 3.3 Tissue detection and patch planning

- These run inside `pathology-ai-embed` (stages `tissue`, `plan`, `extract`) with the UNI pipeline configuration. Do not use separate per-run tuning.
- **Accept** when:
  - the effective spacing is within ±15 % (level 2 for 0.5; level 3 for 1.0);
  - `not_selected_budget` = 0;
  - `selected` > 0 and `selected` < `max_patches`;
  - re-planning gives the identical `plan_digest`.
- Planning parameters are **never** tuned on pilot results without a new, approved pipeline version. Pilot slides are train/validation only, but tuning would still be selection on the cohort.

### 3.4 UNI embedding generation

```bash
uv run pathology-ai-embed --file <czi> --image-id <lowercase slide id> \
    --output-dir <D-9 store location>/<binding>/ \
    --checkpoint uni=../digital-pathology/data/checkpoints/huggingface/uni.pth \
    --override embed.device=mps \
    --manifest-out <run ledger dir>/<slide>.manifest.json
# D-7: the committed pipeline already binds 0.5 um/px +-15%. No scale, profile, channel or
# resampling override is allowed; a run that passes one is outside D-7 and must be refused.
```

- Use fp32 and batch 16, fixed: both are part of the identity.
- Do not change `batch_size` between a run and its resume. A different value opens a different store (`INCOMPATIBLE`).

### 3.5 Provenance and checkpoint verification

- The adapter re-hashes the checkpoint on every load and refuses anything except `56ef09b44a25dc5c7eedc55551b3d47bcd17659a7a33837cf9abc9ec4e2ffb40` (SPEC-006). Weights are never downloaded.
- Each store's identity records:
  - profile `uni-official@1`;
  - scale binding and tolerance;
  - torch and timm versions;
  - device, precision and batch size;
  - the encoder licence notice.
- Each store manifest records the store content digest and `metrics.encoder_resources` (parameter bytes, device peak, timings).
- The pilot report must quote, per slide:
  - the inventory SHA-256 of the WSI;
  - the plan digest;
  - the store id and content digest;
  - the cohort manifest digest and the selection id.

### 3.6 Resume and retry

- `recovery: resume_verified` (pipeline default): re-running the identical command on the same output directory resumes from verified chunks only, and reaches the same content digest (SPEC-005 AC-010, SPEC-006 AC-015).
- **A corrupt chunk fails the run.** Only an explicit `embed.recovery=discard_corrupt` recomputes. Its use must be recorded.
- **Retry policy (proposed):**
  - Up to 2 automatic retries per slide for transient failures (I/O, device).
  - Then mark the slide `failed` in the run ledger and continue.
  - A failed slide in Phase D means a new cohort manifest version (`excluded_quality`). It is never silently dropped. `import_external_split` refuses partial coverage.
- Downloads resume from `.part` files. A finished file is never overwritten.
- **Run ledger:** one append-only CSV per run, with columns:
  - slide id and image id;
  - WSI SHA-256;
  - plan digest;
  - store id and content digest;
  - patches;
  - wall time and peak RSS;
  - device peak;
  - status and attempt number.

### 3.7 Resource monitoring

- **Per slide:**
  - `/usr/bin/time -l` (wall, peak RSS);
  - `encoder_resources` (device peak `driver_allocated` on MPS, timings);
  - free disk before and after;
  - ms per patch.
- **Stop thresholds (proposed):**
  - free disk < 10 GB;
  - peak RSS > 4 GiB (the embed stage declares `memory_mb: 4096`);
  - throughput more than 2× slower than §1.3 assumes.
- **After the pilot:** recompute §1.3 from measured ms per patch, patches per slide by sub-cohort, and store bytes. Phase D needs the updated figures to fit the D-9 budget.

### 3.8 Pilot acceptance criteria (AC-009)

| ID | Criterion |
|---|---|
| A1 | All 8 pilot WSIs `verified` (SHA-256, size = Content-Length), with no duplicate content and no inconsistent copies |
| A2 | Every slide opens. Stored levels and spacing as planned (≤ 15 % error, no resampling). |
| A3 | Plan digests are reproducible on re-plan |
| A4 | Each store is finalized and `verify()` passes. Rows = selected patches, dim 1,024, no NaN/Inf. Row order = patch manifest order. |
| A5 | Re-embedding one slide into a fresh directory (same device and batch) gives the identical content digest |
| A6 | An interrupted run (killed mid-slide) resumes to the identical digest, reading only the missing chunks |
| A7 | Peak RSS within the 4 GiB stage budget. Measured ms per patch and bytes per row reported, and the cohort projection (§1.3) updated. |
| A8 | All pilot stores of one binding share one SPEC-007 compatibility key. A store of the other binding is rejected when mixed. |
| A9 | Checkpoint digest, profile, scale binding and licence notice are present in every store's provenance |
| A10 | The ledger shows only selection slides. `verify_selection` passes, and no test-partition image was opened. |
| A11 | Re-inventory after the pilot gives identical SHA-256 values for every source WSI (never modified) |

Passing A1–A11 produces `008-embedding-pilot.md`. Phase D starts only after the owner approves it.

## 4. Blockers

| # | Blocker | Blocks | Needed |
|---|---|---|---|
| B1 | ~~D-7 extraction scale undecided~~ | — | **Resolved 2026-11-02:** 0.5 µm/px ± 15 % (§0) |
| B2 | **D-9** compute, storage and download route undecided | C2 downloads, D | Owner decision. S3 needs SSO re-authentication; EBI is public but its rate is untested. |
| B3 | **D-10** UNI terms and SurGen data terms not confirmed | C2 and D (the decision gates `pilot` and `extraction`) | Owner confirmation |
| B4 | No MSI-positive WSI on this host; 5 pilot images to acquire (about 22 GB) | C2 | B2, B3, then §3.1 |
| B5 | Cohort-wide image gates open for 820 (AC-008) | D (manifest v2), E | Full inventory of all 1,020 images; storage for about 1.3 TB, or streaming per slide |
| B6 | Storage for Phase D at the approved 0.5 µm/px: stores ≈ 52 GB (up to 62 GB) vs 58 GiB free | D | D-9 storage location |
| B7 | Throughput at scale is unmeasured (prior figure is 178 patches) | D planning | The C2 pilot |

## 5. Go / no-go checklist for Phase C2

| # | Item | State |
|---|---|---|
| G1 | Cohort manifest `surgen-msi@1` verified, split unchanged | ✅ |
| G2 | Pilot partitions approved (D-6), selection excludes test | ✅ |
| G3 | Deterministic pilot selection recorded (seed, criteria, sources) | ✅ `pai-pilot-5da7534b…` |
| G4 | Inventory adapter (read-only, SHA-256, ETag ≠ SHA-256) | ✅ |
| G5 | UNI checkpoint present and pinned | ✅ (PRIOR; re-hashed on every load) |
| G6 | Extraction procedure and acceptance criteria defined | ✅ §3 |
| G7 | Pilot affordable on this host | ✅ ≈ 48 min MPS at 0.5 µm/px, ≈ 0.41 GB of stores, ≈ 22 GB downloads |
| G8 | D-7 scale | ✅ approved 2026-11-02 (0.5 µm/px ± 15 %) |
| G9 | **D-9 compute, storage and download route** | ❌ pending |
| G10 | **D-10 UNI and SurGen terms** | ❌ pending |
| G11 | All 8 pilot images verified locally | ❌ 3 / 8 |
| G12 | Owner approval to start C2 | ❌ |

**Verdict: NO-GO.** Engineering readiness is complete and G8 is resolved. Execution waits for G9–G12.

## 6. Validation

| Check | Result |
|---|---|
| Full test suite | **695 passed, 53 skipped** (after the D-7 update; 694 before it, plus the new data-free D-7 test). The skips are the 52 pre-existing opt-in real-WSI, UNI and CUDA tests, plus the SurGen rebuild, which is opt-in. |
| SurGen rebuild (opt-in, run with both roots bound) | 3 passed. The v1 digest `854f009f…5a3c` is reproduced from the current configuration with the D-7 delta reverted; the current definition gives identical records and split. |
| New tests | 62. Unit: 21 inventory + 23 pilot. Contract: 10. Integration: 3. Architecture: 5. |
| `ruff check`, `ruff format --check` | clean (208 files) |
| `mypy --strict` | clean (152 files) |
| Architecture | 4 new generic modules pass the cohort/target/model/vendor name ban. The inventory adapter opens files `rb` only. Layer rules hold. |
| Write-once documents | Inventory and pilot are accepted on identical re-write and refused on different content; a tamper fails the digest |
| Manifest v1 files | SHA-256 unchanged: `f7748332…d084`, `48105e86…f8cf`, `69c3728c…1781` |

## 7. Changes

**Added:**
- Code:
  - `domain/wsi_inventory.py`
  - `domain/pilot.py`
  - `application/wsi_inventory_service.py`
  - `infrastructure/wsi_inventory.py`
- Tests:
  - `tests/unit/test_wsi_inventory.py`
  - `tests/unit/test_pilot_selection.py`
  - `tests/contract/test_local_wsi_inventory.py`
  - `tests/integration/test_cohort_inventory_pilot.py`
- SurGen configuration and documents:
  - `projects/cohorts/surgen-msi/pilot.yaml`
  - `projects/cohorts/surgen-msi/inventories/pai-wsi-inventory-783f…/wsi-inventory.json`
  - `projects/cohorts/surgen-msi/pilots/pai-pilot-5da7…/pilot-selection.json`
- Reports: the three Phase C1 reports.

**Modified (additive only):**
- `domain/errors.py`: 2 error classes.
- `ports/__init__.py`: the `WSIObjectSource` port.
- `cli.py`: `inventory` and `pilot` subcommands and their notices.
- `tests/cohort_support.py`: synthetic pilot cohort.
- `tests/architecture/test_dependency_rules.py`: 4 modules added to the generic list, plus the read-only rule.
- `README.md`: a C1 section.

**Unchanged:**
- SPEC-001…007 code, `pipelines/`, `cohort.yaml` and `manifests/v1/`.
- SPEC-008 and ADR-008 status (Proposed).

**D-7 governance update (after review, 2026-11-02):**
- `projects/cohorts/surgen-msi/cohort.yaml`: D-7 approved (approver, date, reference); `version: "2"`; header note on versions.
- `.claude/adr/008-real-cohort-validation-architecture.md`: D7 approval note and a decision record table.
- `.claude/specs/008-real-cohort-integration-and-scientific-validation.md`: decision-status note in §9.
- `tests/integration/test_surgen_cohort_manifest.py`:
  - the v1 rebuild now reverts the documented delta, and checks that the current definition changes nothing but the ledger;
  - new data-free test `test_configuration_differs_from_v1_only_by_the_d7_approval`.
- `README.md`: v1 is verified, not rewritten; a new version goes to a new directory.
- This report and `008-pilot-selection.md`.

Not changed: `manifests/v1/`, the inventory and pilot documents, `pipelines/`, and all code.

**Data terms (D-10):** the inventory records file names, sizes and SHA-256 values of study images. It holds no clinical data. Confirm the SurGen terms before committing or redistributing `inventories/` and `pilots/`, as for `manifests/v1/`.

## 8. Owner decisions requested

| ID | Decision | Evidence in this report |
|---|---|---|
| ~~D-7~~ | **Approved 2026-11-02:** 0.5 µm/px ± 15 %, `uni-official@1`, 224 × 224 RGB | §0 |
| D-9 | Download route (EBI HTTPS vs S3 after SSO), staging and store locations, device (MPS for the pilot; CUDA host or about 1–4 days of MPS for Phase D) | §1.3–§1.4, §3.1 |
| D-10 | UNI terms cover this study; SurGen data terms recorded | §4 B3 |
| C2 | Approval to execute the pilot (§3) on selection `pai-pilot-5da7534b…` | §2, §5 |
| v1 / v2 | Run C2 on manifest v1 (citing the D-7 record), or first build manifest v2 explicitly. v2 changes only the ledger; it would require a re-taken inventory and selection. | §0 |

**STOP.** Phase C1 is complete. Phase C2 waits for the owner's explicit approval.
