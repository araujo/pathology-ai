# 007 — MIL Training & Slide-Level Classification: Architecture

**Status:** Implemented. SPEC-007 and ADR-007 remain *Proposed* until the owner accepts them.
**Date:** 2026-10-23
**Related:**
- `../specs/007-mil-training-and-slide-level-classification.md`
- `../adr/07-mil-training-architecture.md`
- `../reports/007-mil-discovery.md`
- `../reports/007-mil-training-and-classification.md`
- `005-streaming-embedding-pipeline.md`
- `006-foundation-model-integration.md`

---

## 1. Shape

```text
finalized embedding stores (SPEC-005/006, unchanged)        <root>/<store_id>/…
        │  read-only, by store id (paths never enter identities)
        ▼
EmbeddingBagSource ── NumpyBagSource (infrastructure)
        │  describe(): verify() + finalized manifest → EmbeddingBagInfo (run identity, digests, records)
        │  read(indices): selected rows only (memmap fancy index)
        ▼
MILDatasetService (application) ── labels (CSV) + slide→store map
        │  build_dataset_manifest (domain): labels × verified bags × compatibility policy
        ▼
MILDatasetManifest  pathology-ai.mil-dataset/1   (digest)
        │  generate_split / import_split / validate_split (domain)
        ▼
SplitManifest       pathology-ai.mil-split/1     (digest, group-disjoint)
        │
        ▼
MILTrainingService (application; no PipelineRunner)
   experiment() → ExperimentSpec (identity = digest of everything results depend on)
   train()      → _PartitionGuard(train ∪ validation stores only)
                  BagPolicy.select (sha256-rank@1) → source.read → MILBatchBuilder (pad + mask)
                  MILModel.train_batch / evaluate_batch   ◄── MILArchitectureRegistry
                  early stopping on validation → best state → MILCheckpointStore.save
   evaluate()   → load verified checkpoint → partition predictions + ClassificationReport
   predict()    → load verified checkpoint → compatibility check → PredictionManifest
        │
        ▼
CheckpointDescriptor pathology-ai.mil-checkpoint/1  +  weights.bin (pickle-free)
PredictionManifest   pathology-ai.slide-prediction/1
```

The existing WSI pipeline, `PipelineRunner`, `embed.patches@1` and the embedding store are
unchanged. Only one method was added to the store: `read_selected`.

## 2. Layers

| Layer | Module | Content |
|---|---|---|
| domain | `domain/mil.py` | `MILTask`, `SlideLabel`/`LabelStatus`, `CompatibilityPolicy` (`exact@1`, `exact-any-device@1`), `EmbeddingCompatibility`, `EmbeddingBagInfo`, `MILDatasetRecord`/`MILDatasetManifest`, `build_dataset_manifest`, `Partition`, `SplitFractions`, `SplitManifest`, `generate_split`, `import_split`, `validate_split`, `BagPolicy`, `BagSelection`, `batch_order`, `rank` |
| domain | `domain/mil_training.py` | `MILArchitectureDescriptor`, `MILModelSpec`, `LossKind`, `OptimizerConfig`, `EarlyStopping`, `TrainingConfig`, `ExperimentSpec`, `EpochRecord`/`TrainingHistory`, `TensorEntry`, `index_tensors`, `CheckpointDescriptor`, `SelectionSummary`, `AttentionSummary`, `SlidePrediction`, `PredictionManifest` |
| domain | `domain/metrics.py` | `MetricValue` (value or explicit reason), `ClassificationReport`, `auroc`, `average_precision`, `classification_report`, `argmax` |
| domain | `domain/errors.py` | `MIL_DATASET_INVALID`, `INVALID_LABEL`, `DUPLICATE_SLIDE`, `MISSING_GROUP`, `INVALID_SPLIT`, `GROUP_LEAKAGE`, `EMBEDDING_INCOMPATIBLE`, `EMPTY_BAG`, `MIL_BUDGET_EXCEEDED`, `EXPERIMENT_CONFLICT`, `MODEL_SELECTION`, `TEST_ISOLATION` |
| ports | `ports/__init__.py` | `EmbeddingBagSource`, `MILBatch`, `MILOutput`, `MILBatchBuilder`, `ModelTensor`, `MILModel`, `MILModelFactory`, `MILCheckpointStore` |
| application | `application/mil_dataset.py` | `MILDatasetService` (build, re-verify, split, import) |
| application | `application/mil_training.py` | `MILTrainingService` (experiment, train, evaluate, predict), `_PartitionGuard` |
| application | `application/registries.py` | `MILArchitectureRegistry` (fail-closed `resolve`) |
| infrastructure | `infrastructure/mil_bags.py` | `NumpyBagSource` |
| infrastructure | `infrastructure/mil_artifacts.py` | `write_once`, `read_document` (digest + round trip), `DirectoryCheckpointStore` |
| infrastructure | `infrastructure/mil_config.py` | YAML experiment configuration (strict keys), label and split CSV tables |
| infrastructure | `infrastructure/canonical.py` | Generic canonical-JSON → frozen dataclass decoder |
| infrastructure | `infrastructure/embedding_store.py` | `NumpyEmbeddingStore.read_selected` (additive) |
| plugins | `plugins/mil/__init__.py` | Descriptors `mean-pooling@1` and `attention-mil@1`; `TorchMILFactory` (lazy torch) |
| plugins | `plugins/mil/batching.py` | `NumpyBatchBuilder` (zero padding + bool mask) |
| plugins | `plugins/mil/torch_models.py` | `MeanPoolingMIL`, `AttentionMIL`, `TorchMILModel` (imports torch) |
| plugins | `plugins/synthetic/mil.py` | Deterministic synthetic cohorts written as finalized stores |
| composition | `bootstrap.py`, `cli.py` | `build_mil`, `build_mil_architectures`; `pathology-ai-mil {synthetic,dataset,split,train,evaluate,predict,run}` |
| definitions | `projects/experiments/mil-synthetic-{binary,multiclass}.yaml` | Example configurations, kept byte-identical to the CLI generator by a test |

`domain`, `ports` and `application` use the standard library only. The torch dependency lives
in exactly one module, `plugins/mil/torch_models.py`, and is imported lazily.

## 3. Identities

| Artifact | Identity | Binds | Never binds |
|---|---|---|---|
| Embedding compatibility key | `pai-embkey-…` = digest(`EmbeddingCompatibility`) | policy; encoder component, model, weights checksum, parameters (incl. profile, scale binding, batch size, runtime versions), output dim and dtype; input-spec fingerprint; model-side transforms; precision; device (`exact@1`) | source channel adaptations (`channel_reorder`, `drop_alpha`) |
| Dataset | digest(`MILDatasetManifest`) | dataset id and version, task and classes, policy, key, every record (slide, store id, embedding manifest digest, embedding identity, patch count, label, status, source, group, cohort, site) | store paths |
| Split | digest / `pai-split-…` | dataset digest, strategy, seed, fractions, membership, group counts, class counts | — |
| Experiment | `pai-milexp-…` = digest(`ExperimentSpec`) | name, task, model spec, training config (seed, optimizer, early stopping, bag policies, clipping, weighting, device), dataset digest, split digest, policy, key, loss, runtime (`device`, `runtime.torch`) | checkpoint root, timestamps, thread count |
| Checkpoint | digest / `pai-milckpt-…` | experiment, tensor index (per-tensor SHA-256), weights SHA-256, history, parameter count, limitations | — |
| Prediction manifest | `content_digest` | experiment, checkpoint id and digest, model, task and classes, decision rule, purpose, partition, dataset and split digests, key, bag policy, override flag, device, every slide prediction | — |

## 4. Contracts in detail

### 4.1 Labels and datasets

- **Label status.** `eligible` needs a label, `unlabeled` has none, and `excluded` needs a reason. An empty cell is never a class. A status is never inferred: the CSV `status` column is required.
- **Label validity.** A label must belong to the task's vocabulary and to its `TaskRef`.
- **Duplicates.** Duplicate slide ids are rejected, and so is one store backing two slides.
- **Missing groups.** An eligible record without `group_id` raises `MISSING_GROUP`. Unlabeled records may lack a group.
- **Store verification.** Every store is verified at build time. `MILDatasetService.verify` re-checks stores before training and evaluation. It reports a replaced or re-embedded store as no longer matching.

### 4.2 Splits

**`group-stratified@1`:**
- Groups are stratified by the sorted label set they contain.
- Within a stratum, groups are ordered by `sha256(seed, "group", group_id)`.
- Each group goes to the partition furthest below its target slide count.
- The result is identical on every platform: no PRNG is involved.

**`external@1`:**
- The membership is preserved exactly.
- It is rejected on group leakage (`GROUP_LEAKAGE`) or on unlabeled or excluded members.

**Validation.** `validate_split` re-derives the summaries against the dataset. A split made for another dataset digest raises `INVALID_SPLIT`, with "never regenerated".

### 4.3 Bags

**`BagPolicy`.** It sets `max_instances` (or unbounded), `seed`, `max_batch_bytes` and the algorithm `sha256-rank@1`.

**Selection.** A bag larger than the limit keeps the `max_instances` indices with the smallest `sha256(seed, slide, epoch, index)`, in plan order.
- Training uses `epoch`, which gives a fresh deterministic subsample per epoch.
- Evaluation and inference use `epoch=None`, so the selection is stable.
- Every prediction records the original and selected counts, the algorithm, the seed and the digest of the indices.

**Budget.** `check_batch` computes the worst-case padded batch (`B × N × (4D + 1)`) before any row is read. Each built batch is checked again.

**Batching.** `NumpyBatchBuilder` zero-pads to the longest selected bag in the batch and sets `mask = True` exactly for real instances. Only one batch is ever on the device.

### 4.4 Models

| Architecture | Forward |
|---|---|
| `mean-pooling@1` | `Σ(x·m)/Σm` → dropout → `Linear(D, outputs)` |
| `attention-mil@1` | `h = ReLU(Linear(D, H))`; `a = tanh(V h) ⊙ σ(U h)` (gated) or `tanh(V h)`; `s = w·a`; `s[¬m] = −∞`; `α = softmax(s)`; `z = Σ α h` → dropout → `Linear(H, outputs)` |

- **Outputs.** `outputs` is 1 for binary tasks (`bce-with-logits`) and the number of classes for multiclass tasks (`cross-entropy`). `class_weighting` sets a positive-class weight for binary tasks and per-class weights `N/(k·n_c)` for multiclass tasks.
- **Hyperparameters.** They are validated against the descriptor's keys and types. A typo is an error, never a default.
- **Attention.** Weights are exactly 0 on padding and sum to 1 over valid instances. They are recorded as `attention_weight` summaries (digest and top 5 patch ids), never as localisation or explanation.

### 4.5 Training and selection

**Determinism.**
- Initialization runs under `fork_rng`, seeded with the experiment seed.
- The model keeps a private generator state for dropout.
- Every pass runs under `use_deterministic_algorithms(True)`, and the previous setting is restored afterwards.
- Training order uses `batch_order(seed, epoch)`.

**Selection.**
- Validation runs after each epoch.
- The selection metric is `validation_loss` (min, the default) or one of `accuracy`, `balanced_accuracy`, `precision`, `recall`, `f1`, `auroc` or `auprc`.
- An undefined value is never an improvement. If no epoch is selectable, training raises `MODEL_SELECTION` and writes no checkpoint.
- Patience and `min_delta` give early stopping. The best state is kept in memory.

**Checkpoint.**
- The selected checkpoint is saved before `train` returns.
- If a checkpoint for the experiment identity already exists, it is verified and reused, and nothing is retrained.
- A different result for the same identity raises `EXPERIMENT_CONFLICT`.
- Resuming from the middle of training is not implemented (§7).

**Test isolation.**
- During `train`, `_PartitionGuard` admits only train and validation stores. Any other read raises `TEST_ISOLATION`.
- Test metrics come only from `evaluate`, which needs the persisted checkpoint.
- There is no threshold tuning: the decision rule is `argmax`.

### 4.6 Checkpoints

- **Layout.** `<root>/<experiment_identity>/weights.bin` holds raw little-endian tensors sorted by name. `checkpoint.json` is written last, through a temporary file and an atomic rename.
- **No pickle.** Pickle is never written or read.
- **Verification on load.**
  - The descriptor digest and its round trip are checked.
  - The experiment directory name must match the descriptor.
  - The file size, the whole-file SHA-256 and each tensor's SHA-256 are checked.
  - Names, dtypes and shapes must match the module exactly, and the state is loaded with `strict=True`.
- **Compatibility on load.** `require_accepts` refuses another compatibility key, feature dimension or task vocabulary.

### 4.7 Predictions

**Content of each record:**
- Raw `logits`.
- Uncalibrated `scores`: sigmoid for binary tasks, giving `(1−p, p)`; softmax for multiclass tasks.
- `scores_calibrated = false` and `calibrated_probabilities = null`. The constructor refuses anything else: no calibration exists in SPEC-007.
- Store id, embedding manifest digest and embedding identity.
- The selection summary and, for attention models, the attention summary.

**Runtime and policy:**
- Evaluation and inference run on CPU, where the reproducibility contract is exact.
- If the inference runtime differs from the training runtime, this is recorded as a limitation.
- An explicit bag-policy override is recorded as `policy_overridden: true`.

## 5. Failure modes (all typed, all fail closed)

| Condition | Error |
|---|---|
| Label outside the vocabulary, or of another task; malformed label record | `INVALID_LABEL` / `MIL_DATASET_INVALID` |
| Duplicate slide, or one store for two slides | `DUPLICATE_SLIDE` |
| Supervised slide without a group | `MISSING_GROUP` |
| Two partitions share a group | `GROUP_LEAKAGE` |
| Split of another dataset, tampered summaries, unlabeled member | `INVALID_SPLIT` |
| Mixed encoder, weights, parameters, input contract, transforms, precision or device; unaccepted key; model trained on other embeddings | `EMBEDDING_INCOMPATIBLE` |
| Store not finalized, interrupted or unknown | `EMBEDDING_STORE_INCOMPLETE` |
| Store bytes or records corrupt | `EMBEDDING_STORE_CORRUPT` |
| Empty bag or selection | `EMPTY_BAG` |
| Padded batch above budget | `MIL_BUDGET_EXCEEDED` (before any row is read) |
| No selectable epoch | `MODEL_SELECTION` |
| Test-store read during training | `TEST_ISOLATION` |
| Checkpoint absent | `CHECKPOINT_MISSING` |
| Checkpoint corrupt or tampered | `CHECKPOINT_INTEGRITY` |
| Checkpoint architecture, dimension or task mismatch | `CHECKPOINT_INCOMPATIBLE` |
| Rewrite of a document or checkpoint with other content | `EXPERIMENT_CONFLICT` |
| Torch not installed | `MODEL_RUNTIME_UNAVAILABLE` (names `pathology-ai[embedding]`) |
| Unknown architecture | `EXECUTION_UNSUPPORTED` |

## 6. Adding an architecture

1. Add a `MILArchitectureDescriptor`, with its name, version, hyperparameter defaults and `produces_attention`, in a plugin module.
2. Implement an `nn.Module` with `forward(x, mask) -> (logits, attention | None)`. It must be mask-aware.
3. Register a factory in `bootstrap.build_mil_architectures()`.

Nothing in `domain`, `ports`, `application` or the training loop changes. Transformer MIL is
deferred (ADR-007 D7).

## 7. Known limits

- **No mid-training resume.** An interrupted training run leaves no checkpoint and is rerun from the start. A completed run is reused.
- **Evaluation and inference are CPU-only.** Training accepts `cuda` or `mps` as a configured device. Only CPU is verified, so GPU determinism is not claimed.
- **Store integrity is checked once per process.** A store modified on disk while training reads it is not detected until the next verification.
- **Metadata is all in memory.** Bag selection hashes up to N indices per bag per epoch (about 61 ms for 20,000 instances). The dataset manifest holds per-slide metadata in memory.
- **Not exercised on real labelled data.** Real labelled cohorts, calibration, multilabel and survival targets, Transformer MIL and distributed training are out of scope.
