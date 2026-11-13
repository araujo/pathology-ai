# Report: SPEC-008 Technical End-to-End Smoke Test (`smoke_e2e`)

**Date:** 2026-11-11
**Authorization:** D-12, recorded in `projects/cohorts/surgen-msi/cohort.yaml` on 2026-11-08 (uncommitted)
**Plan:** `.claude/reports/008-smoke-e2e-plan.md`
**Run:** `smoke-20261111T130922Z`, under `~/pathology-ai-data/smoke_e2e/`
**Outcome:** **PASS.** Every stage ran, and every artifact was read back and verified:

WSI → tissue → patches → UNI → MIL training → checkpoint → evaluation → inference.

> **Engineering validation only.** Three training slides and two validation slides, both negative, cannot support any predictive claim. The scores and metrics below show only that the code paths run. They are **not** clinical, biomarker or scientific evidence (ADR-008 D9).

---

## 1. Scope and conditions

| Item | Value |
|---|---|
| WSIs | 5 (§2). No download. 2 read from the C2 pins in `surgen-msi/wsi/`, 3 from the legacy cache, all read-only |
| Protocol | `wsi-patch-embedding-uni@1.0.0` (`uni-official@1`): 0.5 µm/px ± 15 %, RGB 224 × 224, fp32, batch 16 |
| Overrides (D-12) | `plan.max_patches`, `extract.max_patches` and `embed.max_patches` = 256; `plan.selection=uniform_stride`; `embed.device=mps` |
| MIL | `attention-mil@1` (33,906 parameters); CPU; 3 epochs; seed 13; train bags capped at 128 instances (`sha256-rank@1`); checkpoint chosen on `validation_loss` |
| Host | Apple M4 Pro, 24 GiB, macOS 27.2.1, on AC power, run under `caffeinate` |
| Software | torch 2.14.1, timm 1.0.30, pylibCZIrw 6.1.0; repository at `8df84d0` plus the uncommitted D-11/D-12 changes |
| Phase C2 | Paused and untouched (§8) |

## 2. Inputs (Step 0)

| Slide | Partition (v1) | Label (v1) | Source | SHA-256 (verified before and after the run) |
|---|---|---|---|---|
| SR1482_40X_HE_T368_01 | train | negative | legacy | `9f35f73d…d78d8e` |
| SR1482_40X_HE_T224_01 | train | positive | EBI pin (trust on first download) | `3865a5ff…646a6b` |
| SR1482_40X_HE_T130_01 | train | positive | EBI pin (trust on first download) | `acb3d20c…931720` |
| SR1482_40X_HE_T156_01 | validation | negative | legacy | `2363380e…8f4f20` |
| SR386_40X_HE_T558_01 | validation | negative | legacy | `619200db…05bee2` |

- UNI checkpoint SHA-256: `56ef09b4…2ffb40` (= pin).
- Labels and partitions come from pilot selection `pai-pilot-5da7534b…` and match `manifests/v1/split.csv`, checked by the Step 5 script.

## 3. Results by stage

### 3.1 Readability (Step 1)

- All 5 slides read with backend `pylibczirw 6.1.0`.
- Level-0 spacing is 0.11123–0.11124 µm/px.
- Every pyramid includes downsample 4.

### 3.2 Tissue and patch planning (Step 2: run twice, no pixels)

| Slide | Candidates | Eligible | Selected | Capped (`not_selected_budget`) | Level / spacing | Plan digest, identical on both runs |
|---|---|---|---|---|---|---|
| T368 | 14,850 | 2,484 | 256 | 2,228 | 2 / 0.4449 µm/px | `009a3c8a…` ✔ |
| T224 | 32,208 | 14,594 | 256 | 14,338 | 2 / 0.4450 | `2c97a373…` ✔ |
| T130 | 39,435 | 14,394 | 256 | 14,138 | 2 / 0.4450 | `25514108…` ✔ |
| T156 | 10,375 | 2,013 | 256 | 1,757 | 2 / 0.4450 | `49179a17…` ✔ |
| T558 | 32,175 | 16,171 | 256 | 15,915 | 2 / 0.4450 | `1a6939f3…` ✔ |

- The spacing error is 11.0 % (within ±15 %), with no resampling.
- **Cross-check against C1:** T558 has 16,171 eligible patches and T156 has 2,013, exactly the counts C1 measured. Apart from the cap, planning is identical to the approved protocol.
- Peak RSS ≤ 203 MB; 1.3 s per planning run on average.

### 3.3 UNI embeddings (Step 3, MPS)

| Slide | Store id | Rows × dim | Content digest | Wall | Encode | Peak RSS | Peak GPU (MPS) memory |
|---|---|---|---|---|---|---|---|
| T368 | `pai-emb-b39d0c41…` | 256 × 1024 | `ebf52e12…` | 17.7 s | 7.30 s | 1.50 GiB | 2.04 GiB |
| T224 | `pai-emb-4fbf166a…` | 256 × 1024 | `6432ad65…` | 19.4 s | 7.28 s | 1.55 GiB | 2.04 GiB |
| T130 | `pai-emb-b83c034f…` | 256 × 1024 | `771176b6…` | 19.7 s | 7.37 s | 1.54 GiB | 2.04 GiB |
| T156 | `pai-emb-71f2a279…` | 256 × 1024 | `bfcc1fbf…` | 16.8 s | 7.30 s | 1.52 GiB | 2.04 GiB |
| T558 | `pai-emb-2875aae7…` | 256 × 1024 | `eb0e0214…` | 19.0 s | 7.33 s | 1.55 GiB | 2.04 GiB |

- Encode throughput is **28.5 ms/patch**, consistent with the 29 ms prior.
- The rest of each slide's wall time is the UNI checkpoint load and region reads.
- On every slide the embed plan digest equals the Step 2 digest.
- The encoder block records:
  - profile `uni-official@1` and model version `mass100k.b55a5ec`;
  - the checkpoint SHA-256;
  - batch 16, 0.5 µm/px ± 0.15, and CLS-token output;
  - float32 at 1024 dimensions, on device `mps`.
- `deterministic_algorithms` is recorded as `false`. Bitwise UNI reproducibility on MPS was not tested here; that is C2 criterion A5.

### 3.4 Store read-back (Step 4)

Each of the 5 stores was reopened read-only, its chunks re-hashed, and every row read:
- 256 rows × 1024 dimensions, float32, all finite, 256 distinct rows.
- The content digest agrees three ways: re-hashed store = finalized manifest = embed summary.
- Store size: 6.8 MB in total.

### 3.5 MIL dataset and split (Step 5)

| Item | Value |
|---|---|
| Dataset | `surgen-msi-smoke-e2e@1`, digest `8361ffe7…`. 5 eligible slides, policy `exact@1` |
| Compatibility key | **One key for all 5 stores:** `pai-embkey-789d01e2…`. Feature dim 1024 |
| Split | `pai-split-39ac5911…`, `external@1`. Train 3 slides / 3 groups (1 negative, 2 positive); validation 2 / 2 (2 negative); test 0. No group overlap |

### 3.6 Training and checkpoint (Step 6)

| Epoch | Train loss | Validation loss | Selected |
|---|---|---|---|
| 0 | 0.784 | **0.720** | ✔ |
| 1 | 0.567 | 0.871 | |
| 2 | 0.443 | 0.986 | |

- Checkpoint `pai-milckpt-67a5ca69…`, digest `67a5ca69…`, `weights.bin` SHA-256 `cebe6980…a636` (re-hashed independently: matches).
- Format `pathology-ai.mil-checkpoint/1`; 144 KB.
- Training took 1.05 s for 15 bag loads and 2,688 instances (1.8 s wall, 0.30 GB RSS).
- 3 epochs ran, with no early stop.
- Training loss falls while validation loss rises, as expected: the training set is 2/3 positive and the validation set is all negative. The loop, the selection on validation loss and the checkpointing all work.

### 3.7 Evaluation (Step 7: validation partition, checkpoint reloaded from disk)

- Loss 0.720, the same as the selection value.
- Confusion matrix (rows = true, columns = predicted, order negative/positive): `[[1, 1], [0, 0]]`.
- Accuracy 0.5; precision for "positive" 0.0.
- **Undefined metrics, with the reasons the code reports:**

| Metric | Reason |
|---|---|
| Balanced accuracy | needs at least two classes with true slides (1) |
| Recall (positive) | no slide is truly "positive" |
| F1 (positive) | needs defined precision and recall |
| AUROC | needs positive and negative slides (0 positive, 2 negative) |
| AUPRC | needs at least one positive slide |

### 3.8 Inference (Step 8: all 5 slides, checkpoint reloaded from disk)

Predictions digest `c3d599f8…`, decision rule argmax over uncalibrated sigmoid scores:

| Slide | Predicted | Score (negative / positive) | Instances attended |
|---|---|---|---|
| T130 | positive | 0.390 / 0.610 | 256 |
| T156 | negative | 0.511 / 0.489 | 256 |
| T224 | positive | 0.433 / 0.567 | 256 |
| T368 | negative | 0.521 / 0.479 | 256 |
| T558 | positive | 0.464 / 0.536 | 256 |

- Scores sum to 1 for every slide.
- The prediction file records the compatibility key, the store and embedding identities, the bag-selection digest, attention weights (marked "pooling weights, not validated localisation") and the limitations.
- Three of the five slides were in training, so these scores mean nothing. They show only that inference runs.

### 3.9 Repeatability (Step 9: same seed, separate checkpoint directory)

The second training run produced:
- the same experiment identity `pai-milexp-ce55fd77…`;
- the same checkpoint digest;
- a **byte-identical `weights.bin`**;
- the same validation predictions digest `e51b91f2…`;
- **byte-identical prediction files** (`cmp`).

## 4. Timing and resources

| Step | Wall time |
|---|---|
| 0: pre-checks (re-hash ≈ 18 GB) | 46 s |
| 1: readability | 4 s |
| 2: planning, 2 × 5 runs | 13 s |
| 3: UNI on MPS, 5 × 256 patches | 93 s |
| 4: store read-back | < 1 s |
| 5: dataset and split | 1 s |
| 6–9: train, evaluate and predict, twice | 11 s |
| **Total** | **≈ 2.8 min** (plan estimate: 10–15 min) |

- Peak memory: 1.55 GiB RSS and 2.04 GiB on the GPU (UNI); ≤ 0.32 GiB RSS for MIL.
- Disk: 7.9 MB.
- Network: none.

## 5. Artifact integrity

| Artifact | Check | Result |
|---|---|---|
| Source WSIs | SHA-256 before and after the run | unchanged (5/5) |
| UNI checkpoint | SHA-256 = pin | ✔ |
| Embedding stores | re-hash + finalized manifest + row read | ✔ (5/5) |
| MIL dataset and split | written once; split validated against the dataset digest | ✔ |
| MIL checkpoint | reloaded by evaluate and predict; `weights.bin` re-hashed | ✔ |
| Predictions | written once; digest identical across runs | ✔ |
| Run evidence | `env.txt`, `steps.sha256` (step scripts, env, sources, config, `cohort.yaml`), `logs/` (including `/usr/bin/time -l`) | in the run directory |

## 6. Failures and deviations

- **No stage failed, and no gate or integrity check failed.**
- Deviations from the plan text (procedure only, not protocol):
  - The plan's per-command steps 6–9 ran as a single script, `step6.sh`.
  - Lookups use `sources.tsv` because the system bash 3.2 has no associative arrays.
  - The repeat run's output files carry the tag `-repeat` (a cosmetic slip in the script).
- **Provenance of the overrides.** Each embedding manifest names the pipeline `wsi-patch-embedding-uni@1.0.0` and its plan digest. It does **not** list the override values (`max_patches=256`, `selection=uniform_stride`, `device=mps`) in readable form. The cap is evidenced by:
  - the plan digest, identical to the Step 2 digest of the capped plan;
  - `patch_counts` in each embed summary (`selected` 256, `not_selected_budget` > 0);
  - `device=mps` in the encoder resources;
  - the exact commands, in the step scripts hashed in `steps.sha256` in the run directory.

  A reader of a store alone could mistake it for an uncapped protocol store. The compatibility key does not tell them apart either: see plan §3, as corrected on 2026-11-13.
- Not tested here:
  - UNI bitwise reproducibility and interrupted-run recovery on MPS (C2 criteria A5 and A6).
  - Uncapped stores.
  - The AUROC and AUPRC code paths on real data. The plan's §6 synthetic-label variant was not run.

## 7. Conclusions (technical)

1. The pipeline runs end to end on real SurGen CZI slides at the approved 0.5 µm/px UNI protocol. It produces 1024-dimensional float32 embeddings, a single compatible feature space, a validation-selected attention-MIL checkpoint, and slide-level predictions.
2. Every artifact was read back and verified. MIL training and inference on CPU are bit-reproducible with a fixed seed.
3. Planning reproduces C1's measured eligible-patch counts, and UNI throughput matches the prior estimate.
4. The full C2 pilot (about 100k patches) should take roughly 50 min of UNI time, as planned.

## 8. State after the run

- **Phase C2 is still paused.** 2 of 5 downloads are pinned; the partial `SR386_40X_HE_T440_01` file (0.80 GB) is untouched. Nothing under `surgen-msi/` was modified, and the binding still hashes to `14fd2818…`.
- The power guard was not used and has not been fixed.
- Uncommitted changes: `cohort.yaml` (D-11, D-12), the ledger test, `projects/smoke-e2e/mil-smoke.yaml`, and the plan and this report.
- Phase D was not started.
