# PROMPT-007 — MIL Training & Slide-Level Classification

You are implementing SPEC-007 in the `pathology-ai` repository.

## Authoritative Documents

Read:

1. `.claude/specs/007-mil-training-and-slide-level-classification.md`
2. `.claude/adr/007-mil-training-architecture.md`
3. `.claude/reports/006-foundation-model-integration.md`
4. `.claude/reports/005-streaming-patch-to-embedding-pipeline.md`
5. Relevant accepted SPECs and ADRs 001–006.

If required authoritative documents are missing, stop and request them.

## Phase A — Discovery

Before implementation:

- Inspect the existing embedding manifests and durable stores.
- Inspect the UNI output contracts and model provenance.
- Inspect the legacy MIL training and evaluation implementation in `../digital-pathology`, strictly read-only.
- Identify reusable patterns and known legacy limitations.
- Determine whether real labeled multi-slide datasets are available.
- Identify patient/group leakage risks.
- Verify packaging and architecture constraints.
- Identify any conflicts with accepted ADRs.

Write:

`.claude/reports/007-mil-discovery.md`

Stop only for substantive architectural conflicts.

Do not infer missing patient labels or clinical targets.

## Phase B — Dataset and Split Contracts

Implement:

- Slide labels.
- Dataset manifests.
- Group-aware split manifests.
- Dataset and split identity validation.
- Finalized embedding-store loading.
- Encoder compatibility enforcement.
- Variable-length MIL bags.
- Padding and valid-instance masks.
- Deterministic patch subsampling and resource budgets.

Fail closed on incompatible or corrupted artifacts.

Do not introduce hardcoded disease names.

## Phase C — MIL Models

Implement:

- Mean-pooling baseline.
- Attention MIL.
- Model registry.
- Binary and multiclass classification heads.
- Mask-aware aggregation.
- Configurable feature dimensions.

Use the existing optional PyTorch runtime where appropriate.

Do not implement Transformer MIL in this specification.

## Phase D — Training, Evaluation and Prediction

Implement:

- Reproducible training configuration.
- Optimizer and loss handling.
- Validation and early stopping.
- Checkpoint persistence and integrity.
- Slide-level metrics.
- Test-set isolation.
- Prediction manifests.
- CLI entry points.
- Experiment provenance.

The training workflow may have its own application service, but must not create a second generic pipeline runner.

## Phase E — Validation

Execute:

- Ruff.
- Mypy strict.
- Full pytest suite.
- Existing SPEC-001 through SPEC-006 regression tests.
- Synthetic binary training tests.
- Synthetic multiclass training tests.
- Group-leakage rejection tests.
- Embedding compatibility tests.
- Checkpoint corruption tests.
- Resume/reload tests.
- Prediction reproducibility tests.
- Real UNI embedding loading tests.
- Installed-wheel smoke tests.

For each AC-001 through AC-020, provide:

- PASS, FAIL, BLOCKED, or NOT VERIFIED.
- Test evidence.
- Data type used: synthetic or real.
- Remaining limitations.

Do not report synthetic classification accuracy as clinical performance.

## Mandatory Constraints

- Task-agnostic and tumor-agnostic framework.
- No UNI-specific MIL logic.
- No hardcoded KRAS assumptions.
- No patient/group leakage.
- No silent embedding mixing.
- No test-set-driven model selection.
- No unbounded GPU bag loading.
- No modification of sibling repositories.
- No copying legacy source code.
- No Git commits.
- No unsupported scientific or clinical claims.

## Deliverables

Write:

- `.claude/architecture/007-mil-training-and-classification.md`
- `.claude/reports/007-mil-discovery.md`
- `.claude/reports/007-mil-training-and-classification.md`

The final report must include:

1. Implementation summary.
2. Dataset and split contracts.
3. MIL model architectures.
4. Training and evaluation results.
5. Checkpoint and prediction identities.
6. Group leakage validation.
7. Real UNI bag compatibility.
8. Memory and runtime measurements.
9. Packaging and regression results.
10. AC-001 through AC-020 evidence matrix.
11. Remaining limitations and scientific validation gaps.

Keep SPEC-007 and ADR-007 Proposed until owner acceptance.

**Begin with Phase A. If no architectural conflict exists, implement all non-blocked phases and produce the final report.**