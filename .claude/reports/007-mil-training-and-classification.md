# Report — SPEC-007 MIL Training & Slide-Level Classification

**Date:** 2026-10-23
**Prompt:** `.claude/prompts/007-mil-training-and-slide-level-classification.md`
**Authoritative inputs:**
- SPEC-007 and ADR-007 (both *Proposed*). ADR-007's file is `.claude/adr/07-mil-training-architecture.md`.
- Reports 005 and 006, SPEC/ADR-001…006.

**Companion documents:**
- `007-mil-discovery.md` (Phase A)
- `../architecture/007-mil-training-and-classification.md`

**Outcome:**

| Verdict | Acceptance criteria |
|---|---|
| PASS | 20 of 20 |

**Real data status:**
- Real-data evidence covers loading, masking and compatibility of unlabeled bags from the two real fixtures, including real UNI bags (AC-018). No real labelled cohort exists on this host, so every training and evaluation result is **synthetic**. This is a scientific validation gap (§11).
- SPEC-007 and ADR-007 are left *Proposed*.
- No commits were created. The sibling repositories are unchanged.

> Nothing in this report is clinical, diagnostic or biomarker performance. Metrics are computed
> on seeded synthetic cohorts with an injected signal. They show that the training, evaluation
> and prediction path is correct. They say nothing about any disease, tissue or target.

---

## 1. Implementation summary

| Phase | Delivered |
|---|---|
| A | Discovery report: contracts, legacy assessment, data availability, leakage risks, governance. No conflict was found. |
| B | Labels, dataset manifests, compatibility policy and key, group-aware generated and imported splits, verified bag loading, deterministic subsampling, padding and masks, budgets |
| C | `mean-pooling@1` and `attention-mil@1` (gated attention, masked softmax), a registry, binary (BCE) and multiclass (CE) heads, configurable feature and hidden dimensions |
| D | Experiment identity, training with validation-only early stopping, pickle-free verified checkpoints, slide-level metrics, test isolation, prediction manifests, the `pathology-ai-mil` CLI |
| E | 129 new tests, real UNI bag tests, regression and wheel checks (§9) |

**Main modules:**
- `domain/mil.py`, `domain/mil_training.py`, `domain/metrics.py`
- `ports` (8 new protocols and payloads)
- `application/mil_dataset.py`, `application/mil_training.py`, `MILArchitectureRegistry`
- `infrastructure/mil_bags.py`, `mil_artifacts.py`, `mil_config.py`, `canonical.py`
- `plugins/mil/{__init__,batching,torch_models}.py`, `plugins/synthetic/mil.py`
- `bootstrap.build_mil`, `cli.mil_main`
- `projects/experiments/mil-synthetic-{binary,multiclass}.yaml`

**Architectural decisions and deviations:**

| # | Decision | Rationale |
|---|---|---|
| D1 | The compatibility key excludes source channel adaptations (`channel_reorder`, `drop_alpha`) and binds everything else, including UNI's `batch_size`, runtime versions and the scale binding. | CZI (BGR) and SVS (RGBA) bags of one encoder reach the identical declared input. ADR-006 D3 says batch invariance is not guaranteed. |
| D2 | Two named policies: `exact@1` (default, binds device) and `exact-any-device@1` (explicit opt-in). | SPEC-007 §6.5 says relaxation must be explicit and recorded. Report 006 measured CPU vs MPS agreement within tolerance only. |
| D3 | Deterministic ranks are `sha256(seed, …)`, never a PRNG. | Splits and selections are identical across platforms and Python versions. A pinned test covers this. |
| D4 | Checkpoints are raw tensor bytes plus a JSON index with SHA-256s; there is no `torch.save`. | No pickle, a byte-stable format, and corruption detected per tensor. |
| D5 | The training service is a dataset-level application service that orchestrates ports. There is no new runner. | ADR-007 D8; application stays free of numpy and torch. |
| D6 | Evaluation and inference run on CPU. | That is where the reproducibility contract is exact. GPU inference is deferred. |
| D7 | The torch scope rule is extended to `plugins/mil/`. Only `plugins/mil/torch_models.py` imports torch, lazily. | ADR-006 D3. |
| D8 | `NumpyEmbeddingStore.read_selected` is added. | It reads selected rows without private access. The change is additive. |

Deviations from the prompt: none substantive. Mid-training resume is not implemented; see §11.

## 2. Dataset and split contracts

**`SlideLabel`.**
- Fields: slide id, `TaskRef`, value, source and source version, status (`eligible` / `excluded` / `unlabeled`), optional group, cohort and site, and a reason (required when excluded).
- Class names are free configuration. A test uses `"Mutant / wild type?"` as an opaque class name.

**`pathology-ai.mil-dataset/1`.**
- Records are sorted by slide.
- Each record holds: store id, embedding manifest digest, embedding identity, compatibility key, encoder component, weights checksum, input-spec fingerprint, feature dimension and dtype, patch count, label, status, label source and version, group, cohort and site.
- The manifest is fingerprinted. A test shows its digest is independent of store location: the stores are copied elsewhere and the digest is equal.

**`pathology-ai.mil-split/1`.**
- Fields: dataset ref and digest, strategy, grouping field, seed, fractions, and per partition the slide ids, group count and class counts. It also records the group overlap, which must be empty, and its own digest.
- **Generation.** `group-stratified@1`.
- **Import.** `external@1` preserves the supplied membership.
- **Validation.** `validate_split` re-derives everything.

Example (synthetic binary CLI run, 119 slides, 60 groups):

| Partition | Slides | Groups | class-a | class-b |
|---|---|---|---|---|
| train | 72 | 37 | 38 | 34 |
| validation | 24 | 12 | 12 | 12 |
| test | 23 | 11 | 12 | 11 |

## 3. MIL model architectures

| | `mean-pooling@1` | `attention-mil@1` |
|---|---|---|
| Aggregation | masked mean of the valid instances | `ReLU(Linear(D,H))` → gated tanh attention → masked softmax → weighted sum |
| Head | dropout + `Linear(D, out)` | dropout + `Linear(H, out)` |
| Hyperparameters (defaults) | `dropout=0.0` | `hidden_dim=128`, `attention_dim=128`, `gated=true`, `dropout=0.0` |
| Attention output | none | per-instance weights: sum 1 on valid instances, **exactly 0** on padding |
| Parameters (D=32, binary; H=32, A=16) | 33 | 2,162 |
| Parameters (D=1024, binary, defaults) | 1,025 | 164,482 |

`out` is 1 for binary tasks (BCE with logits) and C for multiclass tasks (cross-entropy).

Model tests (`tests/unit/test_mil_models.py`, 20 tests):

| Property | Result |
|---|---|
| Output shapes | Both architectures, binary and multiclass |
| Masked-mean correctness | Matches the numpy mean of the valid rows |
| Attention normalization | Sums to 1 over valid instances; exactly 0 on padding |
| Padding invariance | A bag alone vs padded to 50 rows agrees within 1e-6, for logits and attention |
| Gradient propagation | Every parameter changes after one step, and the loss decreases |
| Seeded initialization | Same seed gives identical bytes; different seeds differ; the global RNG is untouched |
| State round-trip | Exact |
| Incompatible states | Other task, missing tensor or other dimension → `CHECKPOINT_INCOMPATIBLE` |
| Hyperparameters | Typos and type errors are rejected |
| Unknown architecture | `EXECUTION_UNSUPPORTED` |

## 4. Training and evaluation results (SYNTHETIC)

**Data.** CLI defaults (`pathology-ai-mil synthetic` then `run`):
- Seeded noise, with a fraction 0.3 of the instances shifted by 1.5 along a class direction.
- Dimension 32; 8–64 instances per bag; groups of 1–3 slides.
- Data type: **synthetic**. These numbers are not clinical performance.

**Binary (attention MIL, `projects/experiments/mil-synthetic-binary.yaml`).** Training ran 15 epochs and stopped early on validation loss (patience 8). The best epoch was 6.

| Partition | Accuracy | Balanced acc. | Precision | Recall | F1 | AUROC | AUPRC | Confusion `[[TN,FP],[FN,TP]]` |
|---|---|---|---|---|---|---|---|---|
| validation | 0.917 | 0.917 | 0.857 | 1.000 | 0.923 | 0.917 | 0.818 | [[10,2],[0,12]] |
| test | 0.696 | 0.701 | 0.643 | 0.818 | 0.720 | 0.841 | 0.894 | [[7,5],[2,9]] |

**Multiclass (mean pooling, `projects/experiments/mil-synthetic-multiclass.yaml`).** 179 slides. Training ran all 40 epochs; the best epoch was 39.

| Partition | Accuracy | Balanced acc. | Macro P | Macro R | Macro F1 | OvR AUROC | AUPRC |
|---|---|---|---|---|---|---|---|
| validation | 0.861 | 0.861 | 0.870 | 0.861 | 0.861 | 0.948 | unavailable: "reported for binary tasks only" |
| test | 0.943 | 0.944 | 0.944 | 0.944 | 0.944 | 0.996 | unavailable (same reason) |

**Test-suite thresholds** (larger cohorts, `tests/integration/test_mil_training.py`):
- Binary attention: test accuracy ≥ 0.75 and AUROC ≥ 0.8.
- Multiclass mean pooling: test accuracy ≥ 0.8.
- Both pass.

**Metric correctness** (`tests/unit/test_mil_metrics.py`, 15 tests):
- Hand-computed binary and multiclass reports.
- AUROC against the brute-force pairwise definition, with ties, on 5 seeds.
- AUPRC against the threshold definition, on 5 seeds.
- Undefined metrics carry explicit reasons and are never 0. Examples: AUROC with one class; precision of a class never predicted; macro values naming the undefined classes.
- F1 is 0 when precision and recall are both 0.

## 5. Checkpoint and prediction identities

**Binary CLI run** (identical from the source checkout and from the installed wheel):

| Item | Value |
|---|---|
| Dataset digest | `3d189623ba7e7b5a…` |
| Split digest | `a9f73165f31c9539…` |
| Compatibility key | `pai-embkey-dd80b08e3e9350c678d90e7874125c14` |
| Experiment identity | `pai-milexp-86a2fcad9bddbbb2662345d7707ed740` |
| Checkpoint | `pai-milckpt-189c1f932ac5f5ec0476babff379f50c`, digest `189c1f93…` |
| Weights SHA-256 | `4973af207c7b9342…` |
| Validation predictions `content_digest` | `b2c92e984a4640b4…` |
| Test predictions `content_digest` | `8e62b299344d89e5…` |
| Runtime bound into the identity | `device=cpu`, `runtime.torch=2.14.1` |

**Multiclass CLI run:**
- Experiment `pai-milexp-ed87a7244fb099b6714468ed82872daa`, checkpoint `pai-milckpt-4a365dcebdb9e9189b831a6b3198d00f`.
- Test predictions `13c47d0be3bb287a…`.

**Each slide prediction records:**
- `slide_id` and `image_id`.
- `predicted_class` (argmax).
- Raw `logits` and uncalibrated `scores`, with `score_kind` sigmoid or softmax.
- `scores_calibrated=false` and `calibrated_probabilities=null`. The constructor refuses anything else.
- `store_id`, `embedding_manifest_digest` and `embedding_identity`.
- The selection: original and selected counts, algorithm, seed, epoch and indices digest.
- For attention MIL, an attention summary: digest and top-5 patch ids, `semantics=attention_weight`, with a notice that it is not localisation.

**Each prediction manifest records** the experiment, checkpoint id and digest, model spec, task and classes, positive class, decision rule, purpose and partition, dataset and split digests, compatibility key, bag policy, override flag and device.

**Integrity and reproducibility tests:**

| Test | Evidence |
|---|---|
| Round-trip and idempotent save | `test_save_load_round_trip_and_identity` |
| Identity binds every input | `test_identity_binds_everything_the_result_depends_on` (dataset, split, key, seed, runtime, hyperparameters) |
| Corruption | flipped, truncated and extended weights → `CHECKPOINT_INTEGRITY`; tampered descriptor → `CHECKPOINT_INTEGRITY`; another experiment's descriptor moved in → `CHECKPOINT_INTEGRITY` |
| No overwrite | another result for the same experiment → `EXPERIMENT_CONFLICT`; unrelated files → `EXPERIMENT_CONFLICT` |
| Interrupted write | weights without a descriptor is not a checkpoint; it is replaced on save |
| End to end | a corrupt checkpoint blocks both evaluation and reuse (`test_corrupt_checkpoint_blocks_reuse_and_evaluation`) |
| Reproducibility | two in-process runs, and a fresh subprocess, give equal weights SHA, checkpoint digest and test prediction digest (`test_training_and_prediction_are_reproducible`, `test_fresh_processes_reproduce_checkpoints_and_predictions`) |
| Reload | a new service instance reuses the checkpoint with **0 rows read**; `predict` after reload gives logits identical to `evaluate`; repeated `predict` gives an equal digest (`test_rerun_reuses_the_verified_checkpoint_and_reload_reproduces`) |
| CLI idempotence | re-running `run` reuses everything; prediction digests are equal |

**Reproducibility contract.**
- **Exact** (bit-identical weights and predictions) for identical seed, data, configuration, torch version, device type (CPU) and thread count on the same platform.
- **Tolerance only** across devices or torch versions. Both are bound into the experiment identity, so a change gives a new experiment and never silently overwrites.

## 6. Group leakage validation

| Test | Result |
|---|---|
| `test_generated_splits_never_share_a_group[0..11]` | 12 seeds on a 28-slide, 15-group, 3-class cohort: pairwise group overlap is empty; every eligible slide is assigned once; `validate_split` passes |
| `test_external_split_is_preserved_and_leakage_is_rejected` | Imported membership is identical; moving one slide of a multi-slide group → `GROUP_LEAKAGE` |
| `test_supervised_slides_need_a_group_but_unlabeled_ones_do_not` | Eligible without a group → `MISSING_GROUP`; unlabeled without a group is accepted |
| `test_split_validation_fails_closed` | Another dataset → `INVALID_SPLIT` ("never regenerated"); tampered group count → `INVALID_SPLIT`; non-empty overlap → `GROUP_LEAKAGE`; unlabeled member → `INVALID_SPLIT` |
| `test_one_store_cannot_back_two_slides` | `DUPLICATE_SLIDE` (the same embeddings cannot appear twice) |
| `test_test_partition_is_never_read_during_training` | An instrumented source shows that `train` described and read only train and validation stores; the guard raises `TEST_ISOLATION` on a test store |
| `test_test_labels_do_not_influence_the_model` | Flipping every test label changes the dataset digest and the experiment id, but leaves the weights SHA and history identical |
| `test_evaluation_never_resplits_or_mixes_embeddings` | Evaluation with another split → `INVALID_SPLIT`; before training → `CHECKPOINT_MISSING` |

## 7. Real UNI bag compatibility

**Setup.**
- The fixtures are F-CZI-01 (SR1482_T198_01) and F-SVS-01 (CMU-1-Small-Region). UNI was run with the pinned checkpoint `56ef09b4…fb40`, profile `uni-official@1`, on CPU, through the unchanged SPEC-006 pipeline.
- **No label is attached to either slide.** No real labelled cohort exists, and the legacy CSVs conflict (discovery §4).

| Check (`tests/integration/test_real_uni_mil.py`) | Result |
|---|---|
| Verified, finalized loading | CZI **178 × 1024**, SVS **41 × 1024**, float32. No fixed patch count is assumed. |
| Transforms differ, key equal | CZI `channel_reorder`, SVS `drop_alpha`; one `exact@1` compatibility key |
| Unlabeled dataset | `{eligible: 0, excluded: 0, unlabeled: 2}`; labels and groups are `None`; patch counts recorded |
| Padding and mask | Batch `(2, 178, 1024)`; mask sums `[178, 41]`; rows bit-identical to the store |
| Mixing refused | A model trained on synthetic 32-d embeddings refuses the UNI bags: `EMBEDDING_INCOMPATIBLE: the model was trained on embeddings …` |
| Technical forward pass | Untrained, seeded `attention-mil` at D=1024 (164,482 parameters): finite logits; attention sums 1 per bag; padded batch 1,458,532 B; forward 0.9 ms (CPU) |
| Torch-fixture variant | The same checks on real-slide `torch-fixture` bags (32-d) pass without UNI |

Data type: **real** embedding bags, used for loading and compatibility only. Classification on real bags was not evaluated.

## 8. Memory and runtime measurements

Host: Apple M4 Pro, 24 GiB, macOS arm64, Python 3.12, torch 2.14.1, numpy 2.5.4, CPU.

**Pipeline runs:**

| Run | Measurement |
|---|---|
| Binary CLI `run`, source (119 slides, D=32) | Training 1.12 s for 15 epochs: 1,440 bags and 52,260 instances loaded; max padded batch 33,024 B. Test evaluation 0.012 s. Peak RSS 299 MiB (whole CLI, incl. torch import). |
| Binary CLI `run`, installed wheel (cold) | Training 7.1 s (first torch use in a fresh venv); peak RSS 340 MiB; digests identical |
| Multiclass CLI `run` (179 slides) | Training 1.30 s for 40 epochs: 5,760 bags, 187,000 instances |
| Real UNI, two bags, D=1024 | Padded batch 1.39 MiB; untrained attention forward 0.9 ms |

**Large-bag probe** (synthetic bag of 20,000 × 1024 float32, 78 MiB on disk):

| Measurement | Value |
|---|---|
| Verification (`describe`: full re-hash) | 1.88 s |
| Selection (`sha256-rank@1`) | 61 ms |
| `max_instances=512`: traced peak | **4.0 MiB** (padded batch 2.0 MiB) |
| `max_instances=4096`: traced peak | **32.2 MiB** (padded batch 16.0 MiB) |

Host memory scales with the selected instance count, not with the bag size. Only one padded batch is moved to the device at a time.

**Budget enforcement.** A budget below the worst-case padded batch raises `MIL_BUDGET_EXCEEDED` **before any row is read**: `rows_read == 0` and no checkpoint is written (`test_budgets_fail_before_any_row_is_read`).

**Caveats.**
- `tracemalloc` does not count memory-mapped pages: OS page cache holds the touched rows.
- Synthetic store writing is fsync-bound: 10.8 s for 313 chunks.

## 9. Packaging and regression results

**Quality gates (2026-10-23):**

```bash
uv run ruff check .              # All checks passed
uv run ruff format --check .     # 178 files already formatted
uv run mypy                      # Success: no issues found in 136 source files (strict)
uv run pytest -q -W error -rs    # 586 passed, 52 skipped (no fixtures / no UNI configured)
PATHOLOGY_AI_WSI_FIXTURE_CZI=… PATHOLOGY_AI_WSI_FIXTURE_SVS=… PATHOLOGY_AI_UNI_CHECKPOINT=… \
  uv run pytest -q -W error -rs  # 635 passed, 3 skipped
uv build                         # sdist + wheel OK
```

**Skips and baseline.**
- The 3 skips with fixtures are pre-existing: two `UNSUPPORTED` reader-conformance cases and CUDA.
- Baseline before SPEC-007: 461 passed and 48 skipped without fixtures; 506 passed and 3 skipped with fixtures. **All 506 still pass.**

**New tests: 129**

| Suite | Tests | Needs |
|---|---|---|
| `tests/unit/test_mil_contracts.py` | 51 | no torch |
| `tests/unit/test_mil_metrics.py` | 15 | no torch |
| `tests/unit/test_mil_models.py` | 20 | torch |
| `tests/contract/test_mil_checkpoints.py` | 10 | no torch |
| `tests/integration/test_mil_training.py` | 14 | torch (one runs with torch blocked) |
| `tests/integration/test_real_uni_mil.py` | 4 | fixtures; 3 need UNI |
| Architecture rules | 15 | — |

The architecture rules cover the MIL vocabulary over 12 generic modules, MIL torch isolation, and 2 planted violations.

**SPEC-001…006 artifacts are unchanged:**
- Demo stdout SHA-256: `11a211dec82f27688eebaad3907a0d19791b19daf2feb50d5cd3cfaca89df3da`.
- Reference-encoder CLI on SVS: `e98540cb…`, `pai-embid-27bd8f5b…`, `pai-emb-7c6c0dc2…`, `b494540e…`.
- Reference-encoder CLI on CZI: `56441dbc…`, `pai-embid-d3b2649c…`, `pai-emb-7543b229…`, `5221186b…`.
- All are identical to reports 005 and 006.

**Base wheel** (numpy + pyyaml only; `import torch` fails):
- `pathology-ai-mil synthetic`, `dataset` and `split` succeed.
- `train` exits 3 with `MODEL_RUNTIME_UNAVAILABLE … install 'pathology-ai[embedding]'`.
- `build_mil_architectures()` lists both architectures without importing torch.
- The same behaviour is asserted in-repo by `test_datasets_and_splits_work_without_torch_and_training_names_the_extra`.

**Full wheel `[embedding]`.** `pathology-ai-mil run` reproduces every digest of the source checkout: dataset, split, experiment, checkpoint, weights and both prediction manifests.

**Wheel contents.** The wheel contains every new module. It contains no `tests/` and no `projects/`.

**Changed existing files** (all additive):
- `domain/errors.py`: 12 MIL error types.
- `domain/embedding.py`: `embedding_identity()` function; the property is unchanged.
- `ports/__init__.py`.
- `application/registries.py`: `MILArchitectureRegistry`.
- `infrastructure/embedding_store.py`: `read_selected`.
- `bootstrap.py`: `MILServices`, `build_mil`, `build_mil_architectures`.
- `cli.py`: `mil_main`.
- `pyproject.toml`: the `pathology-ai-mil` script. No new dependency and no new extra.
- `README.md`.
- `tests/architecture/test_dependency_rules.py`: torch scope extended to `plugins/mil/`; MIL rules.

**Repository independence.**
- `../digital-pathology` HEAD is still `c85df8a`, with 4 status lines. `../hal-viewer` HEAD is still `e248202`, with 9 status lines.
- Legacy code was read, never copied.
- No commits were created.

## 10. Acceptance criteria evidence matrix

"Data" says whether the evidence used synthetic or real data.

| AC | Verdict | Data | Evidence | Remaining limitations |
|---|---|---|---|---|
| **AC-001** Task independence | **PASS** | n/a | `test_generic_mil_code_is_task_disease_and_encoder_agnostic` (12 modules; disease, biomarker, tissue, dataset and encoder names, incl. `kras`, `uni`, `abmil`, `braf`, `msi`, `cancer`). The existing core rule covers domain, ports and application. Class names are opaque (`test_class_names_are_free_configuration`). | Vocabulary rules are pattern-based. |
| **AC-002** Dataset manifest | **PASS** | synthetic | `test_dataset_manifest_is_validated_and_fingerprinted` (schema, sorted records, digest stable and location-independent, sensitive to labels); `test_documents_round_trip_and_are_never_overwritten` (edited document → digest error) | — |
| **AC-003** Label integrity | **PASS** | synthetic | `test_invalid_duplicate_missing_and_foreign_labels_are_rejected`, `test_malformed_label_records_are_rejected` (7 cases), `test_invalid_tasks_are_rejected` (9), `test_one_store_cannot_back_two_slides`; the CSV loader requires an explicit status | Label *correctness* (semantics) cannot be verified by software. |
| **AC-004** Leakage-safe splits | **PASS** | synthetic | §6: 12-seed property test, external leakage → `GROUP_LEAKAGE`, missing group → `MISSING_GROUP`, tampered or overlapping split rejected | Group identifiers are only as good as the supplied metadata. |
| **AC-005** Immutable split provenance | **PASS** | synthetic | `test_splits_are_deterministic_and_recorded`, `test_selection_is_platform_independent` (pinned ranks), external split preserved, `validate_split` refuses other datasets; `write_once` refuses other content; the experiment binds the split digest | — |
| **AC-006** Embedding compatibility | **PASS** | synthetic + **real UNI** | Encoder variant → `EMBEDDING_INCOMPATIBLE` naming "encoder parameters"; device bound unless `exact-any-device@1`; accepted key enforced; source adaptations excluded only; a synthetic model refuses real UNI bags | Cross-device bit-equality is not claimed. |
| **AC-007** Variable-length bags | **PASS** | synthetic + **real UNI** | `test_masks_and_padding_are_validated`; real CZI/SVS bags 178/41 padded with mask sums `[178, 41]` and rows bit-identical | — |
| **AC-008** Resource-bounded loading | **PASS** | synthetic | `test_subsampling_is_deterministic_bounded_and_recorded`, `test_batch_budget_is_checked_before_loading`, `test_budgets_fail_before_any_row_is_read` (0 rows read), `test_subsampling_is_recorded_and_policy_overrides_are_explicit`; probe in §8 | Page-cache use is not budgeted; 10⁵-instance bags are extrapolated from the 20k probe. |
| **AC-009** Mean pooling baseline | **PASS** | synthetic | `test_mean_pooling_averages_valid_instances_only`, padding invariance, gradients; multiclass end to end with test accuracy ≥ 0.8 | — |
| **AC-010** Attention MIL | **PASS** | synthetic + real (forward only) | `test_attention_is_normalized_over_valid_instances_and_zero_on_padding`, padding invariance, gradients; binary end to end; real-UNI forward pass | Attention is not validated as localisation (ADR-007 D12). |
| **AC-011** Binary classification | **PASS** | synthetic | `test_binary_attention_mil_end_to_end` (train → persisted checkpoint → test metrics and complete prediction manifest); CLI run §4 | No real labelled cohort. |
| **AC-012** Multiclass classification | **PASS** | synthetic | `test_multiclass_mean_pooling_end_to_end` (with class weighting); CLI run §4 | No real labelled cohort. |
| **AC-013** Reproducible training | **PASS** | synthetic | In-process, fresh-process and installed-wheel runs give equal weights SHA, checkpoint digest and prediction digests (§5) | Exact on CPU with the same torch version and thread count; GPU is unverified. |
| **AC-014** Checkpoint integrity | **PASS** | synthetic | §5: 10 contract tests plus the end-to-end corrupt-checkpoint test; incompatible loads (`require_accepts`, `load_state`) rejected | No mid-training resume. |
| **AC-015** Evaluation correctness | **PASS** | synthetic (hand-computed) | §4: 15 metric tests; undefined metrics carry reasons; an undefined selection metric → `MODEL_SELECTION` without a checkpoint | Small-n metrics have wide uncertainty; no confidence intervals. |
| **AC-016** Test-set isolation | **PASS** | synthetic | `test_test_partition_is_never_read_during_training`, `test_test_labels_do_not_influence_the_model`, evaluation needs a persisted checkpoint, decision rule is argmax with no threshold tuning | — |
| **AC-017** Prediction provenance | **PASS** | synthetic | Prediction fields asserted in `test_binary_attention_mil_end_to_end`; `policy_overridden` recorded; runtime differences recorded as limitations | — |
| **AC-018** Real UNI compatibility | **PASS** | **real UNI** | §7: 4 tests on real F-CZI-01 and F-SVS-01 bags | Loading and compatibility only; no real labels, so no real classification. |
| **AC-019** Regression preservation | **PASS** | real + synthetic | All 506 pre-existing tests pass; demo and SPEC-005 reference digests are byte-identical (§9) | — |
| **AC-020** Packaging and governance | **PASS** | n/a | Base and full wheel checks, optional extra, architecture rules (torch scope, lazy import, MIL vocabulary), `.claude/` documents, SPEC and ADR left *Proposed*, sibling repositories untouched, no commits | — |

## 11. Remaining limitations and scientific validation gaps

1. **No real labelled cohort (scientific gap).**
   - Every training and evaluation result is synthetic.
   - No accuracy, AUROC or any other metric has been measured on a real slide-level target.
   - The legacy SurGen label CSVs contradict each other for the same slides and were not used.
   - A real experiment needs a curated label table with source, version and patient groups. It needs owner decisions on the target, cohort and acceptance metrics, and enough groups for a meaningful held-out test.
2. **No clinical validity.**
   - Nothing here supports diagnostic, prognostic or biomarker claims.
   - Attention weights are pooling weights, not validated localisation.
   - Scores are uncalibrated. Calibration is out of scope (ADR-007, Deferred).
3. **Small-sample statistics.** There are no confidence intervals, no cross-validation and no repeated-split estimates. Single-split metrics on tens of slides are highly variable: compare validation 0.917 with test 0.696 in the binary synthetic run.
4. **No mid-training resume.**
   - An interrupted training run writes nothing and restarts from epoch 0.
   - A completed run is reused after verification.
   - Optimizer state is not checkpointed.
5. **GPU.**
   - Training accepts a `cuda` or `mps` device, but only CPU was exercised.
   - Evaluation and inference are CPU-only.
   - CUDA remains unavailable on this host (as in SPEC-006 AC-012).
6. **Strict compatibility by default.**
   - UNI bags embedded with another `batch_size`, torch/timm version or device are incompatible under `exact@1`.
   - This is intended (ADR-006 D3), but a real cohort must be embedded under one configuration.
   - Cross-configuration equivalence would need a new, separately approved policy.
7. **Integrity window.** Stores are verified once per process. Modifications during a run are detected only by the next verification.
8. **Scale.**
   - Only real bags of 178 and 41 instances were exercised.
   - Large bags are covered by a 20k-instance synthetic probe. 10⁵ instances is extrapolated.
   - Full verification re-hashes each store, about 1.9 s per 78 MiB.
9. **Deferred by the ADR.** Transformer MIL, multilabel and survival targets, distributed training, hyperparameter optimization, foundation-model fine-tuning and viewer heatmaps.
10. **Owner decisions.**
    - Accept or amend SPEC-007 and ADR-007, including D1 (source adaptations excluded from the compatibility key) and D2 (`exact-any-device@1` as the only relaxation).
    - Rename `.claude/adr/07-mil-training-architecture.md` to `007-…` if consistency is wanted.
