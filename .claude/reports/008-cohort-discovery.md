# Report: SPEC-008 Phase A, Cohort Discovery

**Date:** 2026-11-02
**Prompt:** `.claude/prompts/008-real-cohort-integration-and-scientific-validation.md` (Phase A only)

**Authoritative inputs.** All present:
- SPEC-008 and ADR-008 (both *Proposed*)
- Reports 005, 006 (integration + discovery) and 007 (training + discovery)
- Architecture notes 003–007

**Companion documents:**
- `008-label-source-audit.md`
- `008-split-integrity-audit.md`
- `../architecture/008-real-cohort-integration.md`

**Phase A outcome:**

| Item | Result |
|---|---|
| Discovery | Complete |
| Data or code modified | None |
| Expensive work run | None |
| Scientific validation | **BLOCKED** pending the owner decisions in §10 and the storage/compute in §8 |

Status legend:
- **VERIFIED**: observed directly in this audit
- **INFERRED**: a conclusion drawn from verified facts
- **NOT VERIFIED**: claimed by documentation or legacy reports and not checked here
- **BLOCKED**: could not be checked (access, data or authority missing)

> No clinical, diagnostic or performance claim is made in this report. Historical metrics are
> quoted as found and are not reproduced or endorsed.

---

## 1. Sources inspected (read-only)

| Location | What it is | Access | State before = after |
|---|---|---|---|
| `pathology-ai` (this repository) | SPEC-001…007 framework. No real cohort data. | read | unchanged (only the four Phase A documents were added) |
| `../digital-pathology` | Legacy SurGen pipeline. `data/` is git-ignored and holds metadata, a WSI cache, the UNI checkpoint and historical experiments. | read | HEAD `c85df8a`, 4 status lines (both unchanged) |
| `~/projects/SurGen-Dataset` | Clone of upstream `github.com/CraigMyles/SurGen-Dataset` @ `e56ca8c` (2025-11-02). Holds pre-encoded UNI embeddings (untracked). | read | HEAD `e56ca8c`, 5 status lines (both unchanged) |
| `~/projects/digital-pathology` | An older copy of the legacy repository. The cached WSIs and checkpoint are byte-identical to `../digital-pathology`. | read | unchanged |
| `~/Downloads/*.czi` | Ad-hoc SurGen WSI downloads (used by SPEC-003…007) | read | unchanged |
| `s3://digital-pathology-training-545053092614` | Legacy training bucket (account 545053092614) | **BLOCKED**: all three AWS profiles have expired SSO tokens. No listing was obtained. | — |
| EBI BioStudies S-BIAD1285 / Zenodo 14047723 / Hugging Face | Public origins of the WSIs, embeddings and the upstream model | Not contacted (no downloads in Phase A) | — |

- `../hal-viewer` was not needed. Its HEAD (`e248202`) and 9 status lines are unchanged.
- The only process executed against sibling data was read-only file I/O: `csv`, `shasum`, and `zarr.open_group(mode="r")` through the legacy virtualenv interpreter.

## 2. Inventory

### 2.1 Cohort

These facts are VERIFIED against the upstream README and the segmentation lists.

| Item | Value |
|---|---|
| Dataset | SurGen: colorectal cancer H&E WSIs, Zeiss CZI, 40× |
| Level-0 spacing | 0.1112 µm/px (upstream README). Consistent with the 2013-px level-0 stride for 224 µm in the embeddings (224 / 0.1112 = 2014). |
| Sub-cohorts | **SR386:** 427 WSIs, 427 cases, primary CRC. **SR1482:** 593 WSIs, 416 cases, includes metastatic lesions. |
| Total | 1,020 WSIs, 843 cases |
| MSI-labelled | 820 cases, each with one `_01` slide: 490 train / 165 validate / 165 test, 79 positive |
| Without an MSI label | 23 cases; 177 `_02` slides referenced by no label file |
| Public origin | WSIs: EBI BioStudies S-BIAD1285. UNI embeddings: Zenodo record 14047723. Splits and labels: upstream repository. Article: GigaScience, doi 10.1093/gigascience/giaf086. |

### 2.2 Metadata and label files

Full digests and lineage are in the label audit §2.

| File group | Count | Role | Status |
|---|---:|---|---|
| `SR386_msi_*`, `SR1482_*_msi`, `SurGen_msi_*` | 9 | MSI labels + split (upstream) | VERIFIED byte-identical to upstream |
| `surgen_manifest*.csv` | 4 (+4 copies) | Legacy join table: case ↔ CZI ↔ label ↔ split | VERIFIED consistent with upstream |
| Other targets (KRAS, BRAF, NRAS, RAS, 5Y_SUR) | 24 | Not MSI. Used only to cross-check split consistency. | VERIFIED |
| `segment_params_{SR386,SR1482,SurGen}.csv` | 3 | Upstream CLAM segmentation parameters, covering all 1,020 WSIs | VERIFIED |
| `SurGen/MMR_MSI/process_list_*.csv` | 3 | Legacy per-slide **retuned** segmentation for the 820 MSI slides. 349 rows differ from upstream. | VERIFIED |
| `SurGen/MMR_MSI/download_*.txt` | 3 | EBI FTP URLs for the 820 slides (165 / 490 / 165) | VERIFIED |
| `smoke/*.csv` | 3 | 2-slide smoke subset | VERIFIED |

### 2.3 WSIs on this host

- **17 unique files, about 27 GB.** Duplicated copies across the legacy caches and `~/Downloads` were matched by SHA-256.
- **SHA-256 computed for every file.** VERIFIED.
- **No content duplicate** between different slide ids. VERIFIED.

| Slide | Bytes | SHA-256 (prefix) | MSI split / label |
|---|---:|---|---|
| SR1482 T002_01 | 547,312,640 | `b014a4fa6d6fbe8b` | train / 0 |
| SR1482 T004_01 | 2,098,256,352 | `0674f10ab356d743` | test / 0 |
| SR1482 T005_01 | 2,868,092,160 | `1695b3136bd67493` | train / 0 |
| SR1482 T005_02 | 1,980,508,160 | `9177084e9f4fec41` | — (no label; case T005 is train) |
| SR1482 T008_01 | 5,994,180,704 | `c529cf8a9bd3f285` | train / 0 |
| SR1482 T035_01 | 768,698,912 | `9c774a0707d48d6c` | test / 0 |
| SR1482 T085_01 | 284,587,392 | `1cd01239e959c354` | validate / 0 |
| SR1482 T108_01 | 263,222,720 | `3ea74daccff7373c` | test / 0 (13 stale `.part*` download fragments sit beside it) |
| SR1482 T156_01 | 579,305,120 | `2363380e036443e6` | validate / 0 |
| SR1482 T198_01 (F-CZI-01) | 77,963,616 | `8634d0551cb7c137` | test / 0 |
| SR1482 T231_01 | 344,106,944 | `2c7b13287f9c09f6` | train / 0 |
| SR1482 T348_01 | 871,981,280 | `fcf206c31908e6bf` | train / 0 |
| SR1482 T368_01 | 848,763,072 | `9f35f73dfc7e250e` | train / 0 |
| SR1482 T371_01 | 632,211,360 | `18cdb48667cceb77` | train / 0 |
| SR1482 T424_01 | 439,698,464 | `bfb7b1517f2024d0` | validate / 0 |
| SR386 T558_01 | 2,670,401,216 | `619200db97a683bc` | validate / 0 |
| SR386 T568_01 | 6,001,653,152 | `452139549b7d3e94` | test / 0 |

**Consequences** (all VERIFIED):
- Only **16 of the 820** eligible WSIs are local (2 %).
- **Every one is MSI-negative.**
- 5 are in the test partition.
- A positive-containing pilot needs downloads.
- F-CZI-01 (77 MB) is much smaller than the other SR1482 files. Whether it is the complete original or a reduced copy is NOT VERIFIED. Comparing its digest with the EBI file would settle it.

### 2.4 Embedding stores

| Store | Content | Provenance | Status |
|---|---|---|---|
| `~/projects/SurGen-Dataset/embeddings/*.zarr` | **1,020 zarr v2 groups**, one per WSI including `_02`. `features` float32 [N, 1024] and `coords` int64 [N, 2]. N = 66–11,118 (median 4,744); 4,631,924 rows in total, about 16.7 GB. 0 unreadable, rows = coords for all, no NaN in samples. **No attributes**: no encoder id, checkpoint, spacing or channel order is recorded. | Upstream README links Zenodo 14047723 as "pre-encoded UNI embeddings". The legacy guide (`docs/SurGen_Reproduction_Guide.md`) says these files are that download. mtimes are 2024-11-10. | Content VERIFIED; Zenodo origin INFERRED (no checksum manifest compared) |
| `…/embeddings/uni/` | 165 symlinks into the above. This is the legacy test layout: 81 SR1482 names plus 84 SR386 names in unpadded form (e.g. `SR386_40X_HE_T166.zarr`). | Legacy | VERIFIED (not a separate store) |
| Geometry of those stores | Modal coordinate step **2013 level-0 px** on 12 sampled stores (6 per sub-cohort), 0 duplicate coordinates. This means 224 px at 1.0 µm/px, non-overlapping, on a 0.1112 µm/px scan. | — | VERIFIED (sample); cohort-wide INFERRED |
| Encoder recipe of those stores | UNI checkpoint `56ef09b4…fb40`. Level-0 ROI of about 2013 px → `cv2.INTER_AREA` to 224 → ImageNet normalisation applied to **BGR** pixels (no RGB conversion). | Legacy `.claude/reports/SPEC-024.5-report.md` §19–20: 16 rows of T368 re-embedded at cosine ≥ 0.9995 with BGR and 0.70 with RGB. Matches `uni-legacy-surgen@1` in `plugins/models/uni.py`. | NOT VERIFIED here (legacy evidence only) |
| `pathology-ai` stores | Only fixture bags from SPEC-005/006 (F-CZI-01, F-SVS-01) | — | VERIFIED |
| Legacy `data/cache/spec0245_features/*.npy` | 39 files: 13 SR1482 slides × 3 tissue presets | Legacy SPEC-024.5 diagnostics | VERIFIED (not a training store) |

### 2.5 Model checkpoints

| Checkpoint | Size | SHA-256 | Notes | Status |
|---|---:|---|---|---|
| UNI `../digital-pathology/data/checkpoints/huggingface/uni.pth` | 1,213,527,781 | `56ef09b44a25dc5c7eedc55551b3d47bcd17659a7a33837cf9abc9ec4e2ffb40` | Pinned by SPEC-006. License CC-BY-NC-ND 4.0, gated. Owner decision O2 (SPEC-006) on whether existing terms acceptance covers this use is still open. | VERIFIED (digest, per SPEC-006) |
| Legacy MIL `…/baseline_uni_transformer_full_224_nonoverlap/best_model_val_auroc_epoch_2_auroc_0.8914.pth` | 128,284,154 | `a4e5fbdf318fb848d3a5276c5c2307fd5d3eb73f947c1487f68e31f338364bab` | Transformer MIL selected on validation AUROC 0.8914 at epoch 2. A byte-identical copy sits in a mistyped sibling directory (`…nonoverla`). | VERIFIED |
| Legacy MIL `…best_model_val_loss_epoch_5_loss_0.2290.pth` | 128,283,866 | `bfa433b7b1ac2354dea41ae84946d6c50141c256a391f533e341ba016eb85f00` | Validation-loss checkpoint, unused by the summary | VERIFIED |
| Upstream author model `~/projects/SurGen-Dataset/models/best_model_val_auroc_epoch_29_auroc_0.9191.pth` | 128,278,121 | `baaf49d1282d2541…` | INFERRED to be `craigmyles/surgen_msi_classification` from Hugging Face. **Not used.** SPEC-008 trains its own models. | INFERRED |

These are pickled `torch.save` files. They were hashed, never loaded, so no unsafe deserialization took place.

## 3. Historical preprocessing and experiments (A4)

| Aspect | Historical (legacy / upstream) | Source | Status |
|---|---|---|---|
| WSI resolution handling | Level-0 read. Region size = 224 × 1.0 / 0.1112 ≈ 2013 px, resized to 224 with `cv2.INTER_AREA`. Target 1.0 µm/px (about 10×). | `reproducibility/README.md`, `full_uni.yaml` (`target_mpp: 1.0`), SPEC-006 discovery | VERIFIED (config), VERIFIED (geometry from coords) |
| Patch generation | CLAM-derived contour segmentation at level 6 with per-slide parameters, 224 px, step 224 (no overlap). Upstream parameters ≠ legacy retuned parameters (349 / 820 differ). | segment/process lists | VERIFIED |
| UNI preprocessing | BGR uint8 → ToTensor → ImageNet mean/std, no centre crop. **Not** the official UNI recipe (RGB, 20×). | SPEC-006 discovery §4; legacy SPEC-024.5 | VERIFIED (code reading in SPEC-006), NOT VERIFIED (re-execution here) |
| Embedding extraction | fp32, CUDA `DataParallel`, zarr per WSI, no identity metadata | SPEC-006 discovery | VERIFIED (code) |
| Which embeddings the historical run used | The Zenodo stores via the symlink layout | Legacy guide "Path A"; `merged_predictions` `match_type` = `embedding_id` / `case_id` | INFERRED |
| MIL architecture | Transformer MIL: linear 1024 → d, ReLU, `nn.TransformerEncoder` (2 layers, 2 heads, dropout 0.15), mean pooling, linear head (1 logit) | `transformer_main_multiclass.py`, `baseline_uni_surgen.env` | VERIFIED (code + config) |
| Training config | Adam lr 1e-4, batch 1, 200 epochs, AMP, BCE with `pos_weight` = neg/pos (`balance_classes`), seed 1 (default), all patches (no cap), threshold 0.5 | `baseline_uni_surgen.env`, `full_uni.yaml`, script defaults | VERIFIED (config). Actual run arguments NOT VERIFIED (no run log or W&B export locally). |
| Model selection | Best validation AUROC. **Selected at epoch 2 of 200.** | Checkpoint filename | VERIFIED |
| Historical test metrics (local artifact) | n = 165 slides = 165 cases; 15 positive / 150 negative. **AUROC 0.8284**, AUPRC 0.3077, accuracy 0.8182, precision 0.2727, recall 0.600. Confusion at 0.5: TP 9, FP 24, FN 6, TN 126. Per sub-cohort AUROC: SR386 0.848 (n = 84, 6 pos), SR1482 0.798 (n = 81, 9 pos). | `eda/metrics_summary.csv`; **recomputed here from `merged_predictions_with_labels.csv`**, which equals `test_predictions.csv` (165 ids) | VERIFIED |
| Uncertainty of that AUROC | **Descriptive only.** A case-level percentile bootstrap (2,000 replicates, seed 20261102, all replicates defined) gives roughly 0.72–0.91. | Computed here | VERIFIED (descriptive) |
| Other historical numbers | Legacy guide: AUROC 0.8282, accuracy 0.8121, precision 0.2647 (another run). SPEC-008 §4: "around 0.81". The upstream paper's figure was not checked. | docs | NOT VERIFIED |

## 4. Scientific comparability assessment

Comparing the planned SPEC-008 run with the historical baseline:

| Dimension | Historical | Planned SPEC-008 (default) | Comparable? |
|---|---|---|---|
| Cohort / cases | 820 `_01` slides | Same if D-1/D-2 keep upstream scope | Yes (if D-2 = exclude) |
| Target | Combined SR386 MMR loss + SR1482 MSI/dMMR | Same unless D-3 restricts it | Yes (same heterogeneity) |
| Split | Upstream split | Same (preserved) | Yes |
| Test history | Test set used for evaluation and EDA | Same test set | **Reused benchmark**, not fresh (D-5) |
| Scale | 1.0 µm/px (about 10×) | `uni-official@1` binding 0.5 µm/px ± 15 % (operator interpretation of "20×") | **No** |
| Channel order / resize | BGR, INTER_AREA | RGB, no resampling (stored level) | **No** |
| Tissue detection | CLAM, per-slide retuned | SPEC-004 detector, one configuration | **No** |
| Encoder weights | UNI `56ef09b4…` | Same | Yes |
| MIL | Transformer MIL | Mean pooling + gated attention MIL (SPEC-007). No transformer MIL exists, and SPEC-008 forbids new architectures. | **No** |
| Training | 200 epochs, pos_weight, all patches | SPEC-007 config (to be fixed in Phase E) | Partly |
| Threshold | 0.5 fixed | Validation-derived (SPEC-008 §11) | **No** (threshold metrics only) |
| Metric unit | Slide = case | Slide = case (if D-2 = exclude); otherwise case-level aggregation is needed | Yes / depends |

**Verdict:**
- **Only a descriptive comparison is defensible** (ADR-008 D11).
- The AUROC difference would mix at least four changes: scale, channel order, tissue detection and MIL architecture.
- A controlled comparison would need either (a) SPEC-008 MIL models trained on the historical Zenodo embeddings, under a recorded legacy identity, or (b) a legacy-equivalent extraction. Option (b) is blocked by design (`uni-legacy-surgen@1` is BLOCKED in SPEC-006).
- Option (a) is offered as owner decision D-8.

## 5. Are the existing UNI embeddings reusable?

| Use | Answer | Reason |
|---|---|---|
| As SPEC-006 `wsi-patch-embedding-uni` output | **No** | Different profile (BGR, 1.0 µm/px, INTER_AREA, CLAM tissue). No identity, digest or provenance metadata. ADR-006 D7 forbids sharing an identity with `uni-official@1`. Mixing would violate SPEC-008 §9 and AC-011. |
| As a separately identified *legacy-reference* bag set for a secondary comparability arm | **Possible, with approval** | It would need: an importer (zarr is not a dependency of this repository); an identity declaring `uni-legacy-surgen@1` as *imported, not executed*; per-store content digests; and a verified Zenodo checksum. It must never be mixed with new bags. That is guaranteed by the SPEC-007 compatibility key. |
| As evidence of patch counts / volume | **Yes** | Used in §8 |

## 6. Infrastructure readiness (A5)

| Item | State | Status |
|---|---|---|
| Host | Apple M4 Pro, 12 CPU cores, 16-core GPU (Metal 4 / MPS), 24 GiB RAM. No CUDA. | VERIFIED |
| Free disk | **58 GiB** free of 460 GiB (87 % used) | VERIFIED |
| WSI storage access | 16 / 820 local. S3 BLOCKED (expired SSO). EBI FTP public but not tested. The legacy SPEC-024.5 report observed about 0.6 MB/s from EBI. | VERIFIED / BLOCKED / NOT VERIFIED |
| UNI checkpoint | Local, pinned, verified (SPEC-006) | VERIFIED |
| UNI license | CC-BY-NC-ND 4.0, gated. Owner confirmation O2 is still open. | open |
| SurGen data license | Not recorded locally (BioStudies terms) | NOT VERIFIED |
| Pipelines | `wsi-patch-embedding-uni@1.0.0` (CPU default, MPS optional). SPEC-007 MIL (CPU/MPS). | VERIFIED (SPEC-006/007) |
| Measured UNI throughput | CPU about 53 ms/patch, MPS about 29 ms/patch (encode only, batch 16). Wall time includes about 7 s of checkpoint load and detection per slide. | VERIFIED (SPEC-006 §9) |
| CUDA | Not available. NOT VERIFIED by SPEC-006. | NOT VERIFIED |

## 7. Statistical feasibility

- **Test set.** 165 cases, 15 positive. A case-level bootstrap is possible because every replicate has both classes in practice (2,000 / 2,000 defined on the historical predictions).
- **Expected precision.** Intervals will be about ±0.09–0.10 AUROC wide (historical example: 0.72–0.91).
- **Validation.** 165 cases, 14 positive. Threshold selection and early stopping on 14 positives are noisy, and AUROC-based selection at epoch 2 (historical) is a symptom of this.
- **Limits on interpretation.** Differences between mean pooling and attention MIL smaller than about 0.1 AUROC will not be distinguishable. The report must say so (AC-015/016).

## 8. Volume, compute and storage estimates

These figures are INFERRED and must be confirmed by the Phase C pilot (ADR-008 D8).

| Quantity | 1.0 µm/px (legacy scale) | 0.5 µm/px (`uni-official@1` default) |
|---|---:|---:|
| Patches, 820 `_01` slides | 3.61 M (measured from the Zenodo stores: train 2.18 M, validate 0.70 M, test 0.74 M) | about 14.5 M (×4 for half the spacing; actual depends on the SPEC-004 detector and `min_tissue_fraction` 0.5) |
| + 169 labelled `_02` slides (if D-2 includes them) | +0.92 M | +about 3.7 M |
| Embedding storage (fp32, 4 KiB/row + coords) | about 15 GB | about 60 GB |
| UNI encode, MPS (29 ms/patch) | about 29 h | **about 4.9 days** continuous |
| UNI encode, CPU (53 ms/patch) | about 53 h | about 8.9 days |
| UNI encode, one CUDA GPU (assumed 300–600 patch/s, ViT-L fp32) | about 2–3 h | about 7–13 h (NOT VERIFIED) |
| WSI storage, 820 slides | About 1.2–1.5 TB, extrapolated from the 17 local files (mean about 1.6 GB). This assumes the sample is representative (NOT VERIFIED). | same |
| WSI transfer at 0.6 MB/s (EBI, observed in legacy) | about 25 days | same |

**Conclusion:**
- Cohort-wide extraction is **not feasible on this host** as it stands: 58 GiB free versus about 1.3 TB of WSIs, plus days of MPS time.
- Streaming per slide (download → embed → delete) reduces disk to roughly the largest WSI (6 GB) plus the stores. Transfer and compute still dominate.
- A CUDA host with S3/EBI bandwidth is the realistic route, and needs approval.

## 9. Blockers to scientific validation

| # | Blocker | Kind | Unblocked by |
|---|---|---|---|
| B1 | Label authority: only published split CSVs; clinical source tables unavailable | Governance | D-1 |
| B2 | Target heterogeneity (MMR loss vs MSI/dMMR) | Scientific | D-3 |
| B3 | `_02` slide policy, group key | Scientific | D-2, D-4 |
| B4 | Test set previously used (evaluation + EDA) | Scientific | D-5 |
| B5 | Extraction scale (0.5 µm/px binding is an operator interpretation) | Scientific | D-7 (SPEC-006 O1) |
| B6 | WSI access: S3 credentials expired, 16 / 820 local, 58 GiB free | Infrastructure | S3 re-authentication or EBI download plan, storage |
| B7 | Compute: no CUDA; MPS about 5 days at 0.5 µm/px | Infrastructure | D-9 |
| B8 | UNI license confirmation (SPEC-006 O2) and SurGen data terms | Governance | D-10 |
| B9 | WSI content duplicate check across the cohort (AC-008) | Data | B6 |
| B10 | No local MSI-positive WSI for a representative pilot | Data | Pilot selection after B6 |

## 10. Owner decisions required

The first five are restated from the label and split audits.

| ID | Decision | Recommendation |
|---|---|---|
| D-0 | Close the SPEC-007 "label contradiction" as a cross-target comparison | Close |
| D-1 | Authoritative MSI source: upstream split CSVs pinned by digest, or wait for clinical tables | Accept upstream CSVs, with their digests and upstream commit recorded as provenance. Note that labels are unverifiable beyond publication. |
| D-2 | SR1482 `_02` slides | Exclude for the primary analysis (matches the historical baseline and keeps slide = case). Optionally run a secondary analysis with inherited partition and patient-level aggregation. |
| D-3 | Combined target vs per-sub-cohort | Primary: combined (as published), with per-sub-cohort results reported as secondary |
| D-4 | Group key `(cohort, T-number)` | Accept |
| D-5 | Test set status | Accept, and report it explicitly as a previously used benchmark split |
| D-6 | Pilot slides from train/validate only | Accept |
| D-7 | Extraction scale: confirm 0.5 µm/px ± 15 % or choose 1.0 µm/px | Owner/scientific call. 1.0 µm/px is 4× cheaper and closer to the historical baseline, but needs a new profile (binding change only, still RGB / official recipe). |
| D-8 | Secondary comparability arm on the imported historical Zenodo embeddings (separate identity, never mixed) | Optional. It isolates "new MIL vs legacy MIL" from "new preprocessing". |
| D-9 | Compute and storage: CUDA host or accept multi-day MPS; download route (S3 vs EBI); storage location for about 60 GB of stores | Required before Phase D |
| D-10 | Confirm UNI terms (non-commercial research) cover this study, and record the SurGen data terms | Required before Phase C |

## 11. Proposed Phase B (not started)

Phase B is detailed in `../architecture/008-real-cohort-integration.md`. It is reconciliation only: no extraction, no training.

1. **Generic cohort contracts.** Add `pathology-ai.cohort/1` to `domain/cohort.py`: source records with digest and row, normalised candidate labels, conflict and resolution records, the eligibility statuses of SPEC-008 §6.3, and a content-addressed, write-once manifest. This is standard-library only and has no SurGen terms.
2. **Generic declarative CSV label import.** The column mapping, identifier regex, group template, vocabulary map and target name are declared in YAML. Add a cohort service that reconciles several sources and fails closed on conflicts.
3. **SurGen cohort definition, configuration only.** Put it under `projects/cohorts/surgen-msi/`: the source list with pinned SHA-256s and the upstream commit; ID normalisation (cohort-qualified, zero-padded); vocabularies (SR386 `MMR_LOSS`, SR1482 `MMR_MSI`); and the decisions D-1 to D-4 as explicit policy fields.
4. **Outputs.**
   - Reconciliation report: `008-label-reconciliation.md`.
   - Cohort manifest v1: 820 eligible (expected) and 200 excluded, with reasons by original split.
   - Import of the upstream split as a SPEC-007 external split, validated for `GROUP_LEAKAGE`.
5. **WSI inventory step, once access exists.** Name, size and ETag/SHA-256 per WSI, plus CZI metadata (spacing, levels) read lazily. Fills `excluded_missing_wsi` and `excluded_duplicate`.
6. **Tests.** Contract and property tests on synthetic CSV fixtures. These include the observed hazards: cross-cohort T collisions, `slide_<i>` duplicates, unpadded IDs, cross-target files mistaken for MSI, conflicting sources, and `_02` inheritance. **No real labels are committed to the repository.**

**STOP.** Phase A is complete. Phase B waits for the owner's decisions.
