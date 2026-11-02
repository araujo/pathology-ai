# ADR-007 — MIL Training and Slide-Level Classification Architecture

**Status:** Proposed  
**Date:** 2026-10-21  
**Related:** SPEC-007

## Context

The platform can produce persistent embeddings from real WSIs through a model-independent foundation-model pipeline.

MIL introduces dataset-level training, evaluation, checkpointing, and slide-level inference.

These workflows must not introduce disease-specific assumptions or weaken reproducibility and leakage-prevention guarantees.

## Decisions

### D1 — MIL consumes finalized embedding bags

MIL never requires raw WSI pixels during training.

The embedding store and its manifest are the input boundary.

### D2 — Task semantics are configuration

Class vocabularies, biomarkers, disease labels, and target definitions belong in dataset and experiment configuration.

Generic model code remains task-independent.

### D3 — Dataset and split manifests are immutable scientific artifacts

Dataset identity binds slide IDs, labels, embedding identities, and grouping metadata.

Split identity binds group-aware partition membership and strategy.

### D4 — Group leakage prevention is mandatory

Where patient-level or other required grouping is part of the experiment, train, validation, and test partitions must not share groups.

Missing required grouping information causes a validation failure.

### D5 — Embedding compatibility is enforced

Matching vector dimensions is insufficient.

Different encoders, weights, preprocessing profiles, or scale configurations are incompatible by default.

### D6 — Variable-length MIL bags are supported

Bag lengths are not fixed.

Padding and valid-instance masks must be correctly handled.

### D7 — Mean pooling and attention MIL form the initial model set

Mean pooling provides a baseline.

Attention MIL provides a trainable weighted aggregation architecture.

Transformer MIL is deferred.

### D8 — Training is a dedicated application workflow

A dataset-level training service is permitted.

It must reuse platform registries, artifact identities, and provenance.

No competing general-purpose pipeline runner is introduced.

### D9 — Test data is isolated

Validation may guide model selection and early stopping.

Test data must be evaluated only after model selection is complete.

### D10 — Checkpoints and predictions are versioned artifacts

Every checkpoint and prediction must bind model, task, dataset, split, encoder, and relevant execution configuration.

### D11 — Resource limits are explicit

Large bags require bounded loading or deterministic patch selection.

Subsampling must be reproducible and recorded.

### D12 — Attention is not clinical explanation

Attention scores may support research visualization but are not validated evidence of pathology localization.

### D13 — Scientific validation is separate

Successful training on synthetic data does not establish biomarker accuracy or clinical usefulness.

### D14 — Existing repository governance applies

All governance artifacts live under `.claude/`.

Sibling repositories remain read-only.

No Git commits are created without authorization.

## Consequences

### Benefits

- Task-independent MIL.
- Encoder-independent slide-level classification.
- Reproducible datasets and splits.
- Leakage-safe evaluation.
- Reusable prediction artifacts.
- Foundation for later clinical experiments.

### Trade-offs

- Additional dataset metadata requirements.
- Strict compatibility rules.
- More complex training provenance.
- Resource management for large bags.
- Scientific evaluation remains dataset-dependent.

## Deferred

- Transformer MIL.
- Multilabel and survival prediction.
- Distributed training.
- Foundation-model fine-tuning.
- Calibration.
- Clinical validation.
- Viewer heatmap integration.

## Acceptance

Accept this ADR after Phase A confirms compatibility with the existing platform contracts.

SPEC-007 remains Proposed until its acceptance criteria have reproducible implementation evidence.