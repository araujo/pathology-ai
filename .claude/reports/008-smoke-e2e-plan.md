# Plan: SPEC-008 Technical End-to-End Smoke Test (`smoke_e2e`)

**Date:** 2026-11-08
**Status:** **Prepared, not executed.** Needs owner authorization (§7). Nothing has been downloaded, embedded or trained.
**Purpose:** prove that the pipeline runs end to end:

WSI → tissue → patches → UNI embeddings → MIL training → checkpoint → inference.

It is not meant to show predictive performance. No metric from this test is clinical or scientific evidence.

---

## 1. Data: the 5 WSIs that are available now

| Slide | Source | Verified | Partition (v1) | Label (v1) |
|---|---|---|---|---|
| SR1482_40X_HE_T368_01 | legacy, read-only | inventory SHA-256 `9f35f73d…` | train | negative |
| SR1482_40X_HE_T224_01 | EBI, pinned in `wsi/` | `3865a5ff…` (trust on first download) | train | positive |
| SR1482_40X_HE_T130_01 | EBI, pinned in `wsi/` | `acb3d20c…` (trust on first download) | train | positive |
| SR1482_40X_HE_T156_01 | legacy, read-only | inventory SHA-256 `2363380e…` | validation | negative |
| SR386_40X_HE_T558_01 | legacy, read-only | inventory SHA-256 `619200db…` | validation | negative |

- Labels and partitions come from the approved pilot selection `pai-pilot-5da7534b…` and agree with `manifests/v1/split.csv` (checked).
- No test-partition slide is used. No slide is moved between partitions.
- Every one of the 5 is used. This is already the minimum: the training set needs both classes (1 negative, 2 positive), and a validation set is needed for checkpoint selection.

**Limits of the labels:**
- Supervised training is technically possible.
- Validation holds only negatives. AUROC, AUPRC, balanced accuracy and the positive-class metrics will be *undefined*; the metrics code reports a reason instead of failing.
- The checkpoint is chosen on `validation_loss`, which is defined.
- **No synthetic labels are needed.** If the owner prefers an AUROC to exercise as well, §6 gives a clearly labelled synthetic-label variant.

## 2. What already exists (reused, unchanged)

| Stage | Component | Notes |
|---|---|---|
| Readability | `pathology-ai-imaging-smoke` | CZI backend, spacing |
| Tissue and patches | `pipelines/wsi-patch-embedding-uni.v1.yaml` (bound file, read only) | 0.5 µm/px ± 15 %, 224 px, saturation-Otsu tissue |
| UNI embeddings | `pathology-ai-embed` + `uni-official@1` | fp32, batch 16, `resume_verified` store |
| Store checks | `NumpyEmbeddingStoreProvider.verify()` | content digest, dim, dtype |
| Dataset and split | `pathology-ai-mil dataset`, `split` (`external@1`) | Validates labels and groups; refuses group leakage |
| Training and checkpoint | `pathology-ai-mil train` (`attention-mil`) | Validation-selected checkpoint (`pathology-ai.mil-checkpoint/1`); test-isolation guard |
| Evaluation | `pathology-ai-mil evaluate --partition validation` | Undefined metrics carry reasons |
| Inference | `pathology-ai-mil predict --slides <csv>` | Checks compatibility key and feature dim |

**Missing functionality: none that blocks the test.** The only new files are:
- `projects/smoke-e2e/mil-smoke.yaml`: MIL config, already written and parse-checked with `load_experiment_config`. 3 epochs, CPU, seed 13, train bags capped at 128 instances.
- At run time, in the output directory: `labels.csv` (slide, store id, v1 label, group), `split.csv` (v1 partitions) and `predict.csv`.
- A run script under the output directory, written only when execution is approved.

## 3. Patch cap (deterministic) and protocol fidelity

| Setting | Approved UNI protocol | `smoke_e2e` |
|---|---|---|
| Scale | 0.5 µm/px ± 15 % | **same** |
| Patch | RGB 224 × 224 | **same** |
| Tissue detector and thresholds | pipeline values | **same** |
| Precision, batch | fp32, 16 | **same** |
| Device | MPS (override) | **same** |
| `plan.max_patches`, `extract.max_patches`, `embed.max_patches` | 100,000 | **256** |
| `plan.selection` | `fail_closed` | **`uniform_stride`** (deterministic, evenly spaced over the eligible patches) |

- Only how many patches are selected changes; how they are made and encoded does not.
- **Compatibility key (corrected 2026-11-13).** The compatibility key does *not* separate smoke stores from C2 stores, contrary to an earlier version of this plan.
  - Under `exact@1` the key (`domain/mil.py`, `EmbeddingCompatibility`) binds the encoder and weights, input spec, model transforms, precision and device. It does not bind the patch budget or selection policy.
  - Smoke stores and C2 MPS stores therefore **may share a compatibility key**. The smoke run's key was `pai-embkey-789d01e2…`.
  - That is by design: the key identifies a feature space, not a patch sample.
- **Experiment isolation is enforced by:**
  - explicit store identities: every MIL labels table names its `store_id`s, and a dataset reads only those;
  - dataset selection: a dataset manifest and its digest fix the slides and stores of an experiment;
  - separate output directories: smoke stores live under `~/pathology-ai-data/smoke_e2e/`, never under the C2 store root.
- Where a capped store came from is visible in its plan digest and `patch_counts` (`not_selected_budget` > 0), not in its compatibility key.
- MIL adds its own cap: `bags.train.max_instances: 128` (SHA-256-ranked, seeded).

**Option B, not recommended:** no patch cap, so the stores are exactly the approved protocol. That needs about 60k patches, roughly 30 min of UNI work, and does not save C2 any work, because C2 writes its own stores.

## 4. Layout and isolation

```
~/pathology-ai-data/smoke_e2e/<SMOKE_RUN_ID>/        (outside surgen-msi/: never seen by C2 gates G3/G8/G9)
  stores/                    UNI stores (5)
  mil/{config,labels,split,predict}.*   dataset.json  split.json
  checkpoints/  eval-validation/  predictions/
  logs/  env.txt  report.json
```

- The WSIs are only read: the 2 pinned files in `surgen-msi/wsi/` (read-only and `uchg`) and the 3 legacy files.
- Nothing is cloned or copied.
- Nothing in `surgen-msi/` (binding, D-11, manifest v1, selection, `wsi/`, staging, C2 run directory) is written.

## 5. Execution steps (after approval)

Each step runs as a script under `bash -euo pipefail` with `caffeinate`, on AC power.

0. **Pre-checks.**
   - AC power; ≥ 10 GB free.
   - No C2 process running. The C2 download stays paused during the smoke test, so the two never compete for disk or CPU.
   - `shasum -c SHA256SUMS` on the 2 pins; SHA-256 of the 3 legacy files; UNI checkpoint SHA-256.
1. **Readability:** `pathology-ai-imaging-smoke` on all 5 (backend, level-0 spacing 0.1112 µm/px).
2. **Tissue and plan, no pixels:** `pathology-ai-preprocess` with the §3 overrides, twice per slide. Accept: identical `plan_digest` both times, level 2 (0.445 µm/px), selected = min(256, eligible).
3. **UNI:** `pathology-ai-embed --pipeline wsi-patch-embedding-uni@1.0.0` with the §3 overrides and `embed.device=mps`. Record `/usr/bin/time -l`.
4. **Store checks:** 5 finalized stores; rows = selected; dim **1024**; float32; finite; content digests agree; one compatibility key.
5. **Labels and split:** generate `labels.csv` and `split.csv` from the pilot selection and the store ids, then run `pathology-ai-mil dataset` and `split`. Accept: no group leakage, and the class counts re-derive (train 1 negative / 2 positive; validation 2 negative).
6. **Train:** `pathology-ai-mil train`. Accept: a checkpoint whose experiment identity re-derives; per-epoch losses are finite.
7. **Reload and evaluate:** `pathology-ai-mil evaluate --partition validation`. Accept: runs; undefined metrics carry reasons.
8. **Inference:** `pathology-ai-mil predict` on all 5 slides. Accept: 5 predictions with scores in [0, 1] that sum to 1.
9. **Repeatability:** repeat steps 6–8 with the same seed. Accept: identical checkpoint tensors (or a recorded tolerance, if MPS/CPU nondeterminism shows up) and identical predictions.
10. **Report:** `.claude/reports/008-smoke-e2e-report.md` with digests, dimensions, patch counts, timings and resources. Every metric is labelled **"technical smoke test, not evidence"**.

## 6. Alternative: synthetic-label variant (only if the owner wants it)

- Same stores. A second labels file with `label_source: SYNTHETIC` assigns labels by a seeded hash so that validation holds both classes.
- It exercises the AUROC and AUPRC code paths.
- Every output is marked SYNTHETIC, and it is never run on or mixed with the real-label run.
- The `pathology-ai-mil synthetic` generator, which needs no WSI at all, is already covered by the test suite.

## 7. Authorization needed

- D-11 covers only the C2 pilot, and it excludes MIL training. D-9 says MIL training is not authorized. The smoke test therefore needs a **separate owner decision**.
- Proposed text:
  > Technical end-to-end smoke test authorized on the 5 available pilot WSIs (listed in 008-smoke-e2e-plan.md §1). UNI at the approved protocol with a 256-patch uniform_stride cap, MIL training on CPU for 3 epochs, checkpoint and inference. Outputs are kept in `smoke_e2e/` under the owner's control (D-10). No metric is reported as evidence. Not Phase C2, not Phase D.
- Where to record it, the owner's choice:
  - (a) A `D-12` entry in `cohort.yaml`. This also requires updating `test_configuration_differs_from_v1_only_by_the_recorded_approvals`. The binding is not affected.
  - (b) An approval record in the smoke output directory and in the report. This leaves the cohort ledger untouched.
- D-10 covers the licence: non-commercial use, and UNI embeddings and UNI-derived weights stay under the owner's control.

## 8. Estimates

| Item | Estimate | Basis |
|---|---|---|
| Steps 0–2 (hashing ≈ 18 GB + planning) | ≈ 3–4 min | hashing at disk speed; planning ≤ 2.3 s per slide (C1, measured) |
| Step 3: UNI, 5 × 256 patches | ≈ 2–3 min | 29 ms per patch (prior estimate) + ≈ 12 s checkpoint load per slide |
| Steps 4–9: MIL | ≈ 1–2 min | 1,280 instances × 1024 dims; CPU |
| **Total** | **≈ 10–15 min** | |
| Peak memory | ≈ 1.1 GiB RSS + ≈ 2.1 GiB on the GPU (MPS) | SPEC-006 prior |
| Disk | ≈ 6 MB of stores, < 50 MB in total | 1,280 × 1024 × 4 B |
| Network | none | — |

## 9. Related findings to resolve before C2 resumes

- **C2 is not running.**
  - `step1-r3` exited 2 on 2026-11-06 at 21:12Z: DNS to `ftp.ebi.ac.uk` failed while power flapped between AC and battery.
  - The tool used up its 5 attempts and stopped cleanly: lock released, 2 of 5 downloads pinned, 0.80 GB of 2.27 GB of `SR386_40X_HE_T440_01` kept.
  - The Mac was on battery when checked on 2026-11-08.
- **The power-guard script is defective.** It is operator tooling, not bound.
  - Its "safe point" test rejected any curl whose arguments contained the substring `--head`.
  - `c2_wsi.get()` always passes `--header If-Range: …`, so the guard **could never fire**. My earlier claim that it protected the run was wrong.
  - Fix: compare whole argument tokens. Re-test with a dummy process before using it again.
