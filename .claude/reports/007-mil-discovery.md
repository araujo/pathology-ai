# Report — SPEC-007 Phase A: MIL Discovery

**Date:** 2026-10-23
**Prompt:** `.claude/prompts/007-mil-training-and-slide-level-classification.md`
**Authoritative inputs:**
- SPEC-007 (`.claude/specs/007-mil-training-and-slide-level-classification.md`, *Proposed*)
- ADR-007 (`.claude/adr/07-mil-training-architecture.md`, *Proposed*). The file is named `07-…`, not `007-…`. The content is complete, so this naming slip is not a blocker.
- Reports 005 and 006, architecture documents 005 and 006
- SPEC/ADR-001…006

**Verdict:**
- **No substantive architectural conflict.** Implementation proceeds through Phases B–E.
- Three governance adjustments are needed. All are additive and listed in §8.

> Nothing in this report is evidence of clinical validity, biomarker accuracy or any
> diagnostic property. It describes contracts, data availability and risks.

---

## 1. Existing embedding manifests and durable stores (SPEC-005/006)

**Store layout.** A finalized store is a directory `<root>/<store_id>/` (`infrastructure/embedding_store.py`) containing:

| File | Content |
|---|---|
| `header.json` | `EmbeddingStoreHeader`: run identity, layout, and the row → patch records |
| `embeddings.npy` | `(rows, dim)` rows |
| `level0.npy` | `(rows, 4)` level-zero rectangles |
| `chunks/*.json` | One SHA-256 commit record per chunk |
| `manifest.json` | The completion marker. It is written last. |

Every JSON document is `{"body", "digest"}`.

**What MIL can reuse unchanged:**

| Need | Existing contract | Notes |
|---|---|---|
| Open a store read-only by id | `NumpyEmbeddingStoreProvider.open_existing(store_id)` | Path-traversal ids are rejected. |
| Finalized-only check | `NumpyEmbeddingStore.verify()` + `finalized_manifest()` | Re-hashes every chunk. Raises `EMBEDDING_STORE_INCOMPLETE` or `EMBEDDING_STORE_CORRUPT`. |
| Rows | `read_rows(start, stop)` on a read-only memmap | Bounded by the requested range. |
| Patch identity | `EmbeddingRecord`: `row`, `patch_id`, `plan_index`, `level_rect`, `level0` | `level0.npy` is row-aligned and verified against the header. |
| Encoder identity | `EncoderIdentity`: component, model, weights checksum, sorted parameters, output dim and dtype | UNI parameters include the profile, the scale binding (`model_spacing_um`, tolerance), `batch_size` and the torch/timm versions. |
| Model input | `ModelInputSpec` (size, channels, dtype, normalization, required spacing and tolerance), `TransformChain`, `Precision`, `device` | — |
| Bag identity | `EmbeddingManifest.manifest_digest`; `embedding_identity = H(run_identity, content_digest)` | — |

**Gaps:**

1. The `EmbeddingStore` port does not expose `finalized_manifest()`. Only the concrete NumPy store does.
   - MIL adds its own read-only port, `EmbeddingBagSource`. The SPEC-005 port is unchanged, so `FaultyStore` and the existing conformance tests are unaffected.
2. There is no parser from a manifest body back to `EmbeddingManifest`.
   - `header_from_body` already rebuilds the run identity (encoder, input spec, transforms, precision, device).
   - The manifest body adds `content_digest`, `resolution` and `status`.
   - MIL rebuilds the header, verifies the store, and takes the manifest digest as `digest(body)`. That is sufficient, and no parser is needed.
3. **Transform chains differ by source format.**
   - CZI declares `channel_reorder(bgr→rgb)`; SVS declares `drop_alpha(rgba→rgb)`.
   - Both produce the identical declared model input.
   - A compatibility key that bound the whole chain would make CZI and SVS bags of the *same* encoder incompatible.
   - **Decision:** the compatibility key binds the *model-input part* of the chain (cast, scale, resample, normalize, layout) and excludes declared source channel adaptations. The input spec itself (channel order, range, normalization, required spacing and tolerance) is still bound. This is recorded as policy `exact@1`.

## 2. UNI output contract and model provenance (SPEC-006)

- **UNI bags.** They are `float32`, with `dim = 1024` and one row per selected patch. The real fixtures gave CZI 178 rows and SVS 41 rows.
- **Identity.** The encoder identity carries the pinned weights digest `sha256:56ef09b4…fb40`, profile `uni-official@1`, the operator scale binding of 0.5 µm/px ± 15 %, `batch_size` and the runtime versions.
- **Consequence for MIL.** Bags embedded with another batch size, another torch version or another device have different identities.
  - Under the default policy they are **incompatible**. This is intended: ADR-006 D3 says batch invariance is not guaranteed.
  - A separate policy may relax *device only*. It is explicit and recorded.
- **Encoder independence.** MIL code never reads encoder names. It reads `output_dim` and compares compatibility keys, which satisfies SPEC-007 P2 and AC-001.

## 3. Legacy MIL implementation (`../digital-pathology`, read-only)

Inspected, not modified, and not copied:
- `scripts/pipeline/{mil_models,bag_loader,training_run,metrics,split_membership,training_config}.py`
- `reproducibility/models/owkin/.../abmil.py`

**Reusable patterns (concepts only):**
- Validation-only checkpoint selection, with the test set untouched during training.
- A fail-closed dataset compatibility gate before any model is built.
- Bags that keep `patch_indices` into the original artifact after subsampling.
- A per-split case-leakage check.
- Attention MIL that returns its genuine attention weights.
- A single architecture switch point (`build_model`) that never branches on task or dataset.

**Known limitations that are not carried over:**

| Legacy behaviour | Risk | SPEC-007 design |
|---|---|---|
| `training_run_id = uuid4()` in the model id | Model ids are non-deterministic, so retraining can never be recognised as the same experiment | The experiment identity is a digest of its declared inputs |
| Compatibility = encoder *name* + feature dimension | Different weights or preprocessing with the same name would mix silently | A compatibility key over the full encoder identity, input spec and precision |
| Batch size fixed at 1, no masks | No padding semantics to test | Padded batches + valid-instance masks; padding invariance is tested |
| Binary only, fixed threshold 0.5 | — | Binary (BCE, one logit) and multiclass (CE). The decision rule is recorded explicitly, with no threshold tuning. |
| Undefined AUROC → `NaN` → `-inf` for selection; `zero_division=0` for precision | Undefined metrics silently become numbers | `MetricValue(value=None, reason=…)`; an undefined metric is never selectable |
| `sklearn` / `pandas` metrics | Heavy dependencies | Pure-Python slide-level metrics (small n), hand-verified |
| Vendored `ABMIL` loaded via a `sys.path` hack | Hidden coupling | Independent PyTorch modules under `plugins/mil/` |
| `torch.save` of a pickled state dict | Unsafe deserialization | Pickle-free checkpoint: raw tensor bytes + a JSON index, both SHA-256-verified |
| Split membership read from CSVs; a group check exists, but missing groups are not fatal | Leakage when `case_id` is absent | A missing group fails closed (`MISSING_GROUP`) |

## 4. Real labeled multi-slide data

**Real slides on this host:**

| Source | Slides |
|---|---|
| F-CZI-01 (SR1482_T198_01) | 1 |
| F-SVS-01 (CMU-1-Small-Region) | 1 |
| `~/Downloads/SR1482_40X_HE_T00{2,5,5,8}_0x.czi`, `T368_01` | 5 (0.5–6 GB each) |
| UNI checkpoint (pinned) | available |

**Labels.** The legacy repository holds SurGen CSVs (`data/metadata/surgen_manifest*.csv` and `data/metadata/dataset_csv/SR1482_*_{kras,braf,nras,msi}.csv`). They **disagree** for the same slides:
- `surgen_manifest.csv` lists T002, T005, T008, T368 and T198 with `label=0`.
- `SR1482_train_kras.csv` lists T368, T005 and T008 as `1` and T002 as `0`.
- `SR1482_test_kras.csv` lists T198 as `1`.

The label semantics, source, version and patient grouping cannot be established from these files. **They are not used.**
- The prompt forbids inferring labels or clinical targets, and SPEC-007 forbids hardcoded biomarkers.
- No real labeled cohort is therefore available.
- Real-data evidence is limited to **loading and compatibility** of unlabeled UNI bags (AC-018).
- All training and evaluation evidence is synthetic. **Scientific validation gap:** no real-cohort result exists.

## 5. Patient/group leakage risks

| Risk | Observation | Mitigation |
|---|---|---|
| Several slides per patient | Local example: `T005_01` and `T005_02` (same case, two slides) | Group-aware splits. Group overlap between partitions raises `GROUP_LEAKAGE`. |
| Missing group metadata | The legacy fallback treats each slide as a case | `MISSING_GROUP` for any supervised record without a group. No silent slide-level fallback. |
| One store referenced by two slide ids | The same embeddings would appear in two partitions | `DUPLICATE_SLIDE` on duplicate store ids within a dataset |
| Re-splitting between training and evaluation | — | The split manifest is fingerprinted and bound into the experiment and checkpoint. Evaluation refuses another split digest. |
| Test-driven selection | — | Training receives only train and validation bags. Test bags are read only after the checkpoint is persisted. Tests instrument reads. |

## 6. Packaging and architecture constraints

**Layering** (`tests/architecture/test_dependency_rules.py`):
- `domain`, `ports` and `application` use the standard library only (no numpy).
- `infrastructure` may use numpy and yaml. `plugins` may use numpy.
- `torch` and `timm` are scoped to `plugins/models/`.
- The torch runtime must be imported lazily.

**Consequences for MIL:**
- Dataset, split, selection and metrics contracts live in `domain` (pure Python).
- The training loop is orchestrated by an application service that handles arrays and tensors only as opaque `object`s, through ports.
- Padding and masking (numpy) live in `plugins/mil`.
- The PyTorch models live in `plugins/mil` and are lazily imported, which requires extending the torch scope (§8).

**Other constraints:**
- `CORE_FORBIDDEN` already bans `kras`, `uni`, `colorectal` and `abmil` in the core. The attention model is named `attention-mil`.
- The base wheel (numpy + pyyaml) must build datasets and splits without torch. Training without torch raises `MODEL_RUNTIME_UNAVAILABLE`.
- No second `PipelineRunner`. ADR-007 D8 permits a dedicated, dataset-level training service.

## 7. Conflicts with accepted ADRs

| ADR | Check | Result |
|---|---|---|
| 001 (contracts, fail-closed, provenance) | New typed errors, content-addressed identities | Compatible |
| 002 (root-level packages, explicit wheel list) | `plugins/mil` sits inside `plugins`; no new root package | Compatible |
| 005 D6/D7 (durable store, finalized marker) | MIL reads only verified, finalized stores, read-only | Compatible |
| 005 D8 (paths never in identities) | Datasets reference stores by `store_id`; roots are runtime arguments | Compatible |
| 006 D3 (torch optional, scoped) | MIL torch code is lazily imported. The scope rule gains `plugins/mil/` (additive) | Compatible, with the rule extension |
| 006 D9 (batch-invariance not assumed) | Strict compatibility keeps the encoder `batch_size` binding | Compatible |

No conflict requires stopping.

## 8. Governance adjustments (additive)

1. **Torch scope.** `SCOPED_THIRD_PARTY["torch"]` is extended from `plugins/models/` to `plugins/models/` **and** `plugins/mil/`. The lazy-import rule admits `plugins/mil/torch_models.py`.
2. **Generic-module vocabulary.** A new architecture test applies the forbidden-vocabulary checks to the generic MIL modules. It covers disease, biomarker and encoder names.
3. **New console script.** `pathology-ai-mil` is added. There is no new dependency, and no new extra: `embedding` already provides torch.

## 9. Minimum new contracts

**domain:**
- `domain/mil.py`: slide labels, dataset manifest, compatibility, split, bag selection.
- `domain/mil_training.py`: model spec, training config, experiment identity, checkpoint descriptor, prediction manifest.
- `domain/metrics.py`: slide-level metrics with explicit undefined values.

**ports:**
- `EmbeddingBagSource`
- `MILModelFactory`
- `MILModel`
- `MILBatchBuilder`
- `MILArtifactStore`

**application:**
- `mil_dataset.py`: dataset and split services.
- `mil_training.py`: training, evaluation and prediction services.
- `MILArchitectureRegistry`.

**infrastructure:**
- `mil_bags.py`: verified bag source over the NumPy store.
- `mil_artifacts.py`: JSON documents and pickle-free checkpoints.
- `mil_config.py`: YAML experiment config and CSV labels.

**plugins:**
- `plugins/mil/`: descriptors, numpy batching, lazily imported torch models.
- `plugins/synthetic/mil.py`: deterministic synthetic bag writer.

**composition:**
- `bootstrap.build_mil()`
- `cli.mil_main` (`pathology-ai-mil`)
