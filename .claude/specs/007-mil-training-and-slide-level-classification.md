# SPEC-007 — Multiple Instance Learning & Slide-Level Classification

**Status:** Proposed  
**Priority:** P0  
**Dependencies:** SPEC-001 through SPEC-006  
**Repository:** `pathology-ai`

## 1. Context

SPEC-006 successfully integrated real UNI embeddings into the existing WSI pipeline.

The platform can now produce persistent, traceable embedding bags from real CZI and SVS slides.

SPEC-007 introduces slide-level learning and prediction using Multiple Instance Learning (MIL).

The implementation must remain independent of disease, tissue type, biomarker, and embedding encoder.

KRAS or any other clinical target may be configured as an experiment, but must not be hardcoded into the framework.

## 2. Objectives

1. Consume finalized embedding bags from SPEC-005/006.
2. Associate embedding bags with validated slide-level labels.
3. Support leakage-safe training, validation, and test splits.
4. Implement generic MIL dataset and batch contracts.
5. Establish a simple pooling-based baseline.
6. Implement trainable attention-based MIL.
7. Support binary and multiclass classification.
8. Provide deterministic training and checkpointing.
9. Produce reproducible evaluation metrics.
10. Generate versioned slide-level prediction artifacts.
11. Preserve encoder, patch, label, split, and model provenance.
12. Support future MIL architectures without modifying the core.

## 3. Scope

### In scope

- Finalized embedding bag loading.
- Dataset manifest and label validation.
- Slide-level labels and group identifiers.
- Train/validation/test split manifests.
- Patient/group leakage prevention.
- Configurable feature dimensions.
- Variable-length MIL bags.
- Mean-pooling baseline.
- Attention MIL.
- Binary classification.
- Multiclass classification.
- Cross-entropy/BCE losses as appropriate.
- Training and evaluation loops.
- Checkpoint integrity and reproducibility.
- Slide-level prediction manifests.
- Synthetic dataset tests.
- Real UNI embedding compatibility tests.
- CLI and existing pipeline architecture integration.

### Out of scope

- Clinical validation.
- Real biomarker accuracy claims.
- Foundation-model fine-tuning.
- Patch-level supervised segmentation.
- Multilabel and survival tasks.
- Transformer MIL.
- Distributed training.
- Hyperparameter optimization.
- Production deployment.
- Heatmap rendering or viewer overlays.
- Calibration guarantees without adequate validation data.

## 4. Architectural Principles

### P1 — Task-agnostic core

Disease names, biomarker names, and class labels are dataset or experiment configuration, not domain branches.

### P2 — Encoder-independent MIL

MIL consumes validated embedding bags with declared dimensions and provenance.

No UNI-specific assumptions are permitted inside MIL model implementations.

### P3 — Validated datasets

A dataset is a versioned association of slide identities, embedding stores, labels, and grouping metadata.

Unlabeled slides may be used for inference, but not supervised training.

### P4 — Leakage prevention

Train, validation, and test partitions must not share patient/group identifiers when grouping is available.

For biomedical supervised evaluation, missing required grouping metadata must fail closed rather than silently using slide-level random splits.

### P5 — Split immutability

A split manifest is generated or imported once, validated, fingerprinted, and reused.

No hidden re-splitting between training and evaluation.

### P6 — Variable-length bags

MIL models accept different patch counts per slide.

Bag loading and batching must preserve valid-instance masks and patch identity.

### P7 — Reproducible scientific artifacts

Every trained model and prediction binds the dataset, split, encoder provenance, architecture, hyperparameters, random seed, software versions, and checkpoint digest.

### P8 — Evaluation isolation

The test partition must not influence model selection, early stopping, threshold tuning, or hyperparameter decisions.

## 5. Architecture

```text
Finalized Embedding Stores
    |
    v
Dataset Manifest + Label Registry
    |
    v
Dataset Validation
    |
    v
Group-aware Split Manifest
    |
    v
MIL Bag Dataset
    |
    +-- Bag embeddings [N, D]
    +-- Patch identities
    +-- Coordinates
    +-- Slide target
    +-- Valid-instance mask
    |
    v
MIL Model Registry
    |
    +-- Mean Pooling Baseline
    +-- Attention MIL
    |
    v
Training Service
    |
    +-- Optimizer / Scheduler
    +-- Loss
    +-- Validation
    +-- Early Stopping
    +-- Checkpoint
    |
    v
Evaluation Service
    |
    +-- Per-class metrics
    +-- Confusion matrix
    +-- AUROC / AUPRC where defined
    |
    v
Prediction Service
    |
    +-- Scores
    +-- Predicted class
    +-- Prediction provenance
```

## 6. Dataset Contracts

### 6.1 SlideLabel

A slide-level label record contains:

- Stable slide identifier.
- Task identifier.
- Label value.
- Label source and version.
- Optional patient/group identifier.
- Optional cohort/site metadata.
- Label status and eligibility.

The label schema must support arbitrary configured class names.

### 6.2 MILDatasetManifest

Versioned dataset contract:

`pathology-ai.mil-dataset/1`

Each record references:

- Slide identity.
- Finalized embedding store.
- Embedding manifest digest.
- Encoder and preprocessing identity.
- Feature dimension.
- Patch count.
- Label, when available.
- Patient/group identifier.
- Cohort metadata, where available.

The manifest must validate that all stores are finalized and have matching integrity information.

### 6.3 SplitManifest

Versioned split contract:

`pathology-ai.mil-split/1`

The split manifest contains:

- Dataset digest.
- Split strategy.
- Grouping field.
- Random seed.
- Train slide IDs.
- Validation slide IDs.
- Test slide IDs.
- Class distribution per split.
- Group overlap validation.
- Split digest.

Splits must be reproducible and immutable for a given experiment.

A supplied external split must be preserved and validated rather than silently regenerated.

### 6.4 MILBag

A bag contains:

- Slide ID.
- Embeddings `[N, D]`.
- Valid-instance mask.
- Stable patch IDs.
- Canonical spatial coordinates.
- Optional slide label.
- Provenance references.

The loader must not assume a fixed number of patches per slide.

### 6.5 Feature Compatibility

A supervised MIL experiment must explicitly declare its accepted embedding identity or compatibility policy.

Feature dimensions alone do not establish compatibility.

Mixing embeddings from different encoders, checkpoints, preprocessing profiles, or physical-scale configurations must fail unless a separately approved compatibility policy explicitly permits it.

## 7. MIL Model Contracts

Introduce a framework-neutral MIL model specification and a PyTorch-backed model adapter.

The contract must expose:

- Architecture identifier and version.
- Accepted feature dimension.
- Task type.
- Number of output classes.
- Forward output shape.
- Optional attention outputs.
- Model configuration.
- Checkpoint identity.

### Mean Pooling Baseline

Implement mean aggregation over valid instances followed by a trainable linear classifier.

It serves as a simple scientific and architectural baseline.

### Attention MIL

Implement a standard attention-based aggregation model:

1. Project patch embeddings into a configurable hidden representation.
2. Compute attention logits per instance.
3. Mask invalid instances.
4. Normalize attention weights over valid instances.
5. Aggregate weighted instance representations.
6. Produce slide-level logits.

Attention weights are model-derived signals and must not be described as clinically validated explanations.

## 8. Training

Implement a supervised training service with:

- Explicit experiment configuration.
- Deterministic seeding.
- Configurable epochs.
- Optimizer and learning rate.
- Batch size.
- Optional gradient clipping.
- Validation after each epoch.
- Early stopping based only on validation.
- Best-checkpoint selection.
- Training history.
- Failure-safe checkpoint writes.

The service must support variable-length bags through padding and masks or an equivalent safe strategy.

No training loop may depend on a specific cancer type.

## 9. Evaluation

Required metrics:

### Binary classification

- Confusion matrix.
- Accuracy.
- Balanced accuracy.
- Precision.
- Recall.
- F1.
- AUROC, when defined.
- AUPRC, when defined.

### Multiclass classification

- Confusion matrix.
- Accuracy.
- Balanced accuracy.
- Macro precision.
- Macro recall.
- Macro F1.
- Per-class precision, recall, and F1.
- One-vs-rest AUROC, where defined.

Undefined metrics must be reported as unavailable with an explicit reason, not silently replaced with zero.

All metrics must be calculated at slide level.

## 10. Prediction Artifact

Versioned schema:

`pathology-ai.slide-prediction/1`

Each prediction records:

- Slide ID.
- Task and class vocabulary.
- Predicted class.
- Per-class scores.
- Optional calibrated probabilities, only if calibrated.
- MIL model architecture and version.
- Model checkpoint digest.
- Embedding manifest identity.
- Experiment identity.
- Relevant inference configuration.
- Optional attention reference.
- Prediction content digest.

Raw model scores and probabilities must be distinguished.

## 11. Checkpoints and Provenance

A model checkpoint includes:

- Model state.
- Architecture specification.
- Feature dimension.
- Task specification.
- Class vocabulary.
- Training configuration.
- Dataset and split digests.
- Embedding compatibility policy.
- Best validation metric.
- Random seed.
- Runtime versions.
- Checkpoint checksum.

Checkpoint loading must reject incompatible architecture, feature dimension, or task vocabulary.

Training checkpoints must not silently overwrite a different experiment.

## 12. Resource Management

MIL bag loading must be bounded.

For very large bags, support an explicit deterministic maximum patch count or chunked loading policy.

Any subsampling must record:

- Original patch count.
- Selected patch count.
- Selection algorithm.
- Seed.
- Selected patch indices or equivalent reproducible selection reference.

The test and inference policies must be explicit and must not silently differ from the trained model's expectations.

The implementation must not assume all slide embeddings fit into GPU memory.

## 13. Integration

Reuse the existing application services, registry patterns, and artifact provenance infrastructure.

Do not create a second generic `PipelineRunner`.

Training may use a dedicated application service and CLI because it is a dataset-level iterative workflow, rather than a single-image preprocessing stage.

Prediction should consume existing finalized embedding stores and produce versioned prediction artifacts.

The existing WSI pipeline must remain unchanged.

## 14. Validation Strategy

### Synthetic tests

Create a deterministic, multi-slide labeled dataset with:

- Multiple patients/groups.
- Variable bag sizes.
- Binary and multiclass targets.
- Controlled class separability.
- Explicit train/validation/test splits.

Use this dataset to verify correctness of the complete training and evaluation flow.

### Contract tests

- Invalid labels.
- Duplicate slide IDs.
- Missing grouping metadata.
- Split overlap.
- Group leakage.
- Incompatible embedding identities.
- Non-finalized stores.
- Corrupt embedding stores.
- Feature dimension mismatch.
- Empty bags.
- Mask correctness.

### Model tests

- Mean pooling.
- Attention normalization.
- Padding invariance.
- Output shape.
- Gradient propagation.
- Deterministic initialization.
- Checkpoint compatibility.

### Integration tests

- Training on synthetic embeddings.
- Evaluation on held-out synthetic slides.
- Prediction artifact generation.
- Loading existing real UNI embedding bags.
- Installed-wheel execution.

## 15. Acceptance Criteria

**AC-001 — Task independence**

No cancer-specific, biomarker-specific, or encoder-specific branches exist in generic MIL code.

**AC-002 — Dataset manifest**

Versioned labeled dataset manifests are validated and fingerprinted.

**AC-003 — Label integrity**

Invalid, duplicate, missing, or incompatible labels are rejected.

**AC-004 — Leakage-safe splits**

Patient/group leakage is prevented and tested.

**AC-005 — Immutable split provenance**

Train/validation/test membership is deterministic and recorded.

**AC-006 — Embedding compatibility**

Incompatible embedding identities cannot be silently mixed.

**AC-007 — Variable-length bags**

The loader supports variable patch counts and correct masks.

**AC-008 — Resource-bounded loading**

Bag loading and deterministic subsampling obey configured resource limits.

**AC-009 — Mean pooling baseline**

A trainable mean-pooling classifier operates correctly.

**AC-010 — Attention MIL**

Attention aggregation and masks are correct and trainable.

**AC-011 — Binary classification**

Binary training, prediction, and evaluation work end to end.

**AC-012 — Multiclass classification**

Multiclass training, prediction, and evaluation work end to end.

**AC-013 — Reproducible training**

Repeated controlled CPU training runs reproduce checkpoints and metrics within the documented reproducibility contract.

**AC-014 — Checkpoint integrity**

Model checkpoints are versioned, validated, and reject incompatible loads.

**AC-015 — Evaluation correctness**

Slide-level metrics are correct, and undefined metrics are explicitly identified.

**AC-016 — Test-set isolation**

The test split does not influence training or model selection.

**AC-017 — Prediction provenance**

Predictions contain the required model, embedding, task, and experiment identities.

**AC-018 — Real UNI compatibility**

The loader successfully consumes finalized UNI embedding bags from SPEC-006 without assuming a fixed patch count.

**AC-019 — Regression preservation**

SPEC-001 through SPEC-006 tests and artifacts remain valid.

**AC-020 — Packaging and governance**

CLI, installed-wheel execution, optional dependencies, architectural boundaries, and reports comply with repository governance.

## 16. Implementation Phases

### Phase A — Discovery

Inspect SPEC-005/006 embedding contracts and the legacy MIL implementation in `../digital-pathology` read-only.

Determine the minimum new contracts and identify dataset availability, leakage risks, and architecture compatibility.

### Phase B — Dataset Foundation

Implement label contracts, dataset manifests, split validation, and MIL bag loading.

### Phase C — MIL Models

Implement mean pooling and attention MIL behind reusable model contracts.

### Phase D — Training and Evaluation

Implement supervised training, checkpointing, metrics, and prediction artifacts.

### Phase E — Validation

Run synthetic end-to-end tests, existing real UNI bag compatibility tests, regression gates, and installed-wheel checks.

## 17. Deliverables

- `domain/` — task, dataset, split, and prediction contracts.
- `ports/` — MIL model and dataset interfaces.
- `application/` — training, evaluation, and prediction services.
- `plugins/mil/` — pooling and attention implementations.
- `infrastructure/` — checkpoint and dataset persistence.
- `pipelines/` or `projects/` — example experiment configurations.
- `.claude/architecture/007-mil-training-and-classification.md`
- `.claude/reports/007-mil-discovery.md`
- `.claude/reports/007-mil-training-and-classification.md`

## 18. Definition of Done

SPEC-007 is complete when:

1. AC-001 through AC-020 have reproducible evidence.
2. Binary and multiclass MIL training work on synthetic labeled datasets.
3. Group-aware splitting prevents leakage.
4. Embedding compatibility is validated.
5. Checkpoints and prediction artifacts are reproducible and integrity-checked.
6. Existing real UNI bags can be loaded successfully.
7. Existing platform behavior remains compatible.
8. No clinical performance claims are made from synthetic datasets.
9. Missing real labeled cohorts are documented as a scientific validation gap, not concealed by synthetic tests.