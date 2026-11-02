# SPEC-005 — Streaming Patch-to-Embedding Pipeline

**Status:** Proposed  
**Priority:** P0  
**Dependencies:** SPEC-001, SPEC-002, SPEC-003, SPEC-004  
**Target repository:** `pathology-ai`

## 1. Context

SPEC-004 introduced real WSI tissue candidate detection, deterministic patch planning, real patch extraction, canonical level-zero coordinates, and versioned patch manifests.

However, its materialized extraction mode retains selected patch pixels in an in-memory artifact payload. This is appropriate for bounded extraction and testing, but not for scalable whole-slide embedding generation.

The legacy `digital-pathology` implementation demonstrates a proven conceptual pattern: store patch coordinates, lazily read pixel regions during embedding inference, process bounded batches, and incrementally persist embeddings.

SPEC-005 will implement this pattern within the existing `pathology-ai` architecture while addressing legacy limitations around memory accounting, reader lifecycle, implicit preprocessing, embedding identity, incomplete stores, and recovery.

## 2. Objectives

1. Convert a SPEC-004 patch plan into a persistent embedding bag without materializing all patch pixels.
2. Support bounded-memory processing independent of total patch count.
3. Introduce model-independent batch encoding contracts.
4. Preserve explicitly selected stored image resolutions.
5. Validate and record every model input transformation.
6. Persist embeddings incrementally with verifiable completeness.
7. Support safe restart and chunk-level recovery.
8. Preserve deterministic patch-to-embedding association.
9. Execute through the existing `PipelineRunner`.
10. Provide a testable reference encoder without requiring PyTorch or model weights.

## 3. Scope

### In scope

- Lazy patch reading from an existing `OpenedImage`.
- Consumption of SPEC-004 `PatchPlan` and patch manifests.
- Bounded batch production and encoding.
- Model input contracts and transformation validation.
- Batch-oriented encoder port.
- Durable embedding store abstraction.
- NumPy-backed persistent storage.
- Chunk-level commit and restart semantics.
- Embedding provenance and integrity validation.
- Deterministic synthetic encoder.
- Integration with `PipelineRunner`.
- CZI and SVS integration tests.
- Resource and failure-injection tests.

### Out of scope

- Training or fine-tuning foundation models.
- MIL training and inference.
- Slide-level classification.
- Clinical validation.
- Production UNI inference or model-weight distribution.
- Legacy embedding numerical parity.
- Universal image resolution.
- Distributed execution.
- Multi-GPU orchestration.
- Persistent storage of all patch pixels.
- Importing existing legacy Zarr embedding bags.

## 4. Architectural Principles

### P1 — Patch plans are the streaming boundary

The pipeline consumes a SPEC-004 patch plan and associated metadata. Pixel arrays are read only when needed for a batch.

`materialize_pixels=false` is the preferred upstream execution mode.

No whole-slide embedding workflow may depend on an artifact containing every patch's pixels.

### P2 — Bounded pixel memory

Pixel buffering must be bounded by configured batch size, in-flight work, and explicit byte budgets.

No intermediate collection of patch pixels may grow with the total number of patches.

Metadata and coordinate structures may scale with patch count, subject to separate documented limits.

### P3 — Reader independence

Embedding operations use the existing image reader abstractions.

Domain and application layers must not import CZI, OpenSlide, or other format-specific implementations.

### P4 — Native resolution preservation

The source reader reports the actual stored levels and pixel spacing.

Patch extraction uses an explicitly selected stored level or an explicitly requested spacing resolved to a stored level.

There is no universal target spacing and no implicit resampling.

### P5 — Model-specific preprocessing

Input size, channel order, normalization, physical scale, interpolation, precision, and tensor layout are model-contract concerns.

Any transformation must be explicit, validated, versioned, and recorded.

### P6 — Persistent artifact correctness

A partially written embedding store must never be reported as a valid completed artifact.

A store is valid only after all expected rows are verified and an explicit final commit succeeds.

### P7 — Reproducibility

Embedding identity must bind the patch plan, patch manifest, model identity, model weights or synthetic encoder implementation identity, input contract, transformations, output content, and relevant execution configuration.

## 5. Target Pipeline

```text
OpenedImage
    |
    v
SPEC-004 tissue.detect@1
    |
    v
SPEC-004 patch.plan@1
    |
    v
PatchPlan + PatchManifest
(materialize_pixels=false)
    |
    v
embed.patches@1
    |
    +-- Capability and model-input validation
    |
    +-- LazyPatchSource
    |       |
    |       +-- read_region for selected patch
    |       +-- preserve canonical patch identity
    |
    +-- Explicit TransformChain
    |
    +-- BoundedBatchProducer
    |
    +-- EmbeddingEncoder.encode_batch
    |
    +-- EmbeddingStore.write_chunk
    |
    +-- EmbeddingStore.commit_chunk
    |
    +-- EmbeddingStore.finalize
    |
    v
EmbeddingManifest + DurableStoreReference
```

The batching loop lives inside a single `embed.patches@1` stage.

No streaming edge type or second orchestration engine is introduced.

## 6. Domain Contracts

### 6.1 ModelInputSpec

A model input specification declares:

- Expected input width and height.
- Channel count and channel order.
- Input dtype and numeric range.
- Tensor layout.
- Normalization parameters, when required.
- Pixel spacing requirements or explicit acceptance of variable spacing.
- Physical-scale tolerance.
- Input dimension divisibility constraints, where relevant.
- Supported inference precision.

Unknown physical spacing is not silently interpreted as a default.

A model requiring known spacing must reject unknown spacing.

### 6.2 TransformChain

A validated, ordered transformation sequence.

Examples:

- `channel_reorder`
- `resize` or `resample`
- `cast`
- `scale`
- `normalize`
- `layout_convert`

Each transformation must declare its parameters, implementation identifier, and version.

Transforms must not silently change spatial geometry, channel meaning, or numeric interpretation.

The transformation sequence must be included in provenance and execution identity.

The baseline synthetic encoder must be operable without any spatial resampling.

### 6.3 BatchPolicy

A typed batch policy declares:

- Maximum patches per batch.
- Maximum input batch bytes.
- Maximum in-flight batches or equivalent bounded work.
- Maximum output batch bytes.
- Prefetch policy.
- Device and precision configuration when applicable.

The effective memory budget must account for decoded pixels, transformed arrays, queued batches, and encoder outputs.

A batch exceeding its budget must fail with a typed error rather than silently exceeding configured limits.

### 6.4 EmbeddingRecord

Each embedding row is associated with:

- Stable `patch_id`.
- Original patch-plan index.
- Canonical level-zero coordinates.
- Source stored level.
- Effective pixel spacing and physical field of view, when known.
- Embedding row index.
- Embedding dimension and dtype.

Association must not depend solely on positional coincidence.

### 6.5 EmbeddingManifest

Versioned schema: `pathology-ai.embedding-manifest/1`.

The manifest records:

- Source image and reader provenance.
- Patch-plan and patch-manifest identities.
- Patch count.
- Ordered patch identifiers and their mapping to stored rows.
- Encoder implementation identity.
- Model and weights identity where applicable.
- Model input specification.
- Applied transformations.
- Input resolution and effective physical field of view.
- Embedding dimension, dtype, and row count.
- Store format, schema, and reference.
- Chunk completion and integrity information.
- Final content digest.
- Completion status.
- Reproducibility configuration.

Runtime-local paths and volatile timestamps must not contaminate stable scientific identity.

## 7. Ports

### 7.1 EmbeddingEncoder

Introduce a batch-oriented encoder port exposing:

- Stable encoder identity.
- `ModelInputSpec`.
- Expected output dimension and dtype.
- `encode_batch(batch, context)`.
- Explicit resource lifecycle where required.

The port must not require PyTorch types.

Backend-specific adapters may convert arrays into tensors internally.

### 7.2 EmbeddingStore

Introduce a durable store port supporting:

- Creation or validated reopening of a store.
- Metadata and identity validation.
- Incremental chunk writes.
- Chunk integrity verification.
- Chunk commit.
- Inspection of committed ranges.
- Finalization after complete coverage.
- Reopening and validation of finalized stores.

The store must reject overlapping, missing, corrupted, or incompatible writes unless explicitly handled by a safe recovery policy.

### 7.3 LazyPatchSource

Provide a lazy patch-reading abstraction over an existing `OpenedImage` and selected patch records.

The implementation must:

- Read pixels only when requested.
- Preserve patch identity and plan order.
- Respect source validity and coordinate contracts.
- Avoid caching an unbounded number of patches.
- Close any resources it owns.
- Never close a caller-owned image handle unexpectedly.

A separate public port is optional if the existing reader contracts provide sufficient abstraction.

## 8. Reference Storage Implementation

Implement a NumPy-backed durable embedding store using `.npy` arrays and a versioned JSON sidecar.

Use `numpy.lib.format.open_memmap` or an equivalent documented NumPy mechanism.

The storage layout must preserve:

- Embedding rows.
- Stable patch identifiers or a verifiable index mapping.
- Canonical coordinates.
- Chunk completion state.
- Artifact provenance.
- Content integrity.

Writes must be incremental.

The store must distinguish:

- Created.
- In progress.
- Interrupted.
- Finalized.
- Corrupt or incompatible.

Chunk commits must be crash-consistent: a chunk is considered committed only after its data and integrity metadata are durably recorded in a recoverable order.

The implementation must document filesystem assumptions and must not claim transactional guarantees it cannot establish.

Finalization must verify complete expected coverage and integrity.

Zarr is deferred to an optional future adapter behind the same port.

## 9. Reference Encoder

Implement a deterministic NumPy-only synthetic embedding encoder.

Requirements:

- Fixed, declared input specification.
- Configurable embedding dimension.
- Deterministic output for identical input bytes and encoder configuration.
- No trained model weights.
- No external downloads.
- No dependence on PyTorch.
- No claim of meaningful medical representation.

The reference encoder is an architectural test fixture, not a scientific feature extractor.

## 10. Resolution and Transformation Semantics

Keep these concepts separate:

1. Source resolution.
2. Selected stored extraction resolution.
3. Patch physical field of view.
4. Model-required input geometry and scale.
5. Applied transformations.

The pipeline must validate the selected extraction geometry against the encoder's declared requirements before reading patch pixels.

If requirements cannot be satisfied by the selected level and declared transforms, fail with a typed error such as `RESOLUTION_MISMATCH`.

No transformation is permitted merely because two image formats report different native spacing.

The framework must support distinct resolution profiles for distinct datasets, experiments, and encoders.

Compatibility with legacy SurGen embeddings must be represented as a separate, explicitly versioned preprocessing profile, not as the default behavior.

## 11. PipelineRunner Integration

Register `embed.patches@1` through the existing operation registry.

Use the existing open-image execution context introduced in SPEC-004.

The operation consumes patch metadata and writes batches to the durable store.

The stage returns only:

- A durable store reference.
- A completed embedding manifest.
- Bounded execution metrics.

Do not return all patch pixels or all embedding vectors through the in-memory artifact store.

Changes to `PipelineRunner`, if necessary, must be additive and preserve existing execution fingerprints when the new functionality is unused.

## 12. Failure and Recovery

The pipeline must handle:

- Interrupted execution between batches.
- Interrupted chunk writes.
- Reader failures.
- Encoder failures.
- Store write failures.
- Incorrect or unavailable model capabilities.
- Incompatible resume configuration.
- Corrupt persisted chunks.
- Missing source data.
- Resource-budget violations.

Recovery must resume only from verified committed chunks.

An interrupted store must not be treated as finalized.

Repeated execution against an already finalized compatible artifact must be safe and must not silently overwrite valid data.

## 13. Validation Strategy

### Unit tests

- Model input contract validation.
- Resolution and channel compatibility.
- Transform ordering and identity.
- Batch-size and byte-budget enforcement.
- Stable patch-to-row association.
- Embedding identity.
- Manifest canonicalization.

### Store contract tests

- Incremental writes.
- Reopen.
- Finalization.
- Missing chunks.
- Overlapping writes.
- Corrupt chunks.
- Interrupted writes.
- Resume from verified commits.
- Incompatible run identities.

### Integration tests

- Existing synthetic pipeline.
- Real CZI fixture.
- Real SVS fixture.
- Installed-wheel execution.
- Identical output identities across repeated runs.

### Resource tests

- Instrument the number of live patch arrays and in-flight batches.
- Verify bounded pixel retention for different patch counts.
- Measure peak process memory where practical.
- Record GPU memory separately when GPU support is introduced.

### Failure-injection tests

Simulate failures during reading, encoding, writing, committing, and finalization.

Verify that no partial artifact is accepted as complete.

## 14. Acceptance Criteria

**AC-001 — Lazy Patch Consumption**

Embedding extraction consumes SPEC-004 patch plans and reads patch pixels on demand. It does not require materialized pixel artifacts.

**AC-002 — Bounded Pixel Memory**

Pixel retention is bounded by explicit batch and in-flight budgets, with automated tests demonstrating that the number of retained pixel arrays does not grow with total patch count.

**AC-003 — Reader Independence**

The same embedding pipeline executes through existing reader abstractions without format-specific branches in domain or application code.

**AC-004 — Explicit Resolution Semantics**

Source resolution, stored extraction level, effective field of view, and model-required geometry are distinguished and recorded. No universal spacing is imposed.

**AC-005 — Model Input Contract**

Input shape, channels, dtype, scale, normalization, and precision requirements are validated before pixel reads.

**AC-006 — Explicit Transformations**

Every channel conversion, resizing operation, normalization step, and layout conversion is declared and recorded. No silent resampling is permitted.

**AC-007 — Batch Encoder Port**

A model-independent encoder port accepts batches and validates output shape, dtype, and finite values.

**AC-008 — Incremental Persistent Storage**

Embeddings are written incrementally to a durable store. The full embedding matrix is not materialized as an intermediate output.

**AC-009 — Chunk Commit Integrity**

A partially written or corrupted chunk cannot be treated as committed or finalized.

**AC-010 — Safe Recovery**

Interrupted runs resume only from verified committed chunks and reject incompatible configuration or identity.

**AC-011 — Patch-to-Embedding Traceability**

Every embedding row maps deterministically to a stable patch ID, patch-plan index, and canonical spatial coordinates.

**AC-012 — Embedding Manifest**

A versioned manifest records source, plan, encoder, input contract, transforms, output schema, store reference, integrity, and completion state.

**AC-013 — Deterministic Reference Encoder**

A NumPy-only reference encoder produces stable embeddings across repeated executions and fresh processes.

**AC-014 — PipelineRunner Integration**

The complete patch-to-embedding workflow runs through the existing `PipelineRunner`, without a second orchestration engine.

**AC-015 — Real WSI Validation**

The pipeline is validated against the existing CZI and SVS fixtures with exact patch identity and row-count checks.

**AC-016 — Resource Safety**

Configured batch and in-flight budgets are enforced, reader resources are correctly managed, and failures do not leak owned resources.

**AC-017 — Existing Behavior Preservation**

SPEC-001 through SPEC-004 regression tests remain green, and the existing synthetic demo remains byte-identical.

**AC-018 — Packaging and Optional Dependencies**

The base package remains usable without PyTorch, timm, Zarr, or model weights. The reference embedding pipeline runs with base dependencies.

**AC-019 — Failure-Injection Evidence**

Automated tests cover interruption and corruption at critical read, encode, write, commit, and finalize boundaries.

**AC-020 — Reproducibility Evidence**

Repeated runs and installed-wheel executions produce matching stable identities and content digests under equivalent configurations.

## 15. Implementation Phases

### Phase A — Architecture and Contract Assessment

Review existing SPEC-004 contracts and the discovery report.

Identify minimal additions and any architectural conflicts.

Stop for owner approval if a proposed change contradicts accepted ADRs.

### Phase B — Domain and Port Contracts

Implement model input specifications, batch policies, embedding records, manifests, encoder and store contracts.

### Phase C — Durable Storage and Streaming

Implement the NumPy embedding store, lazy patch source, bounded batching, integrity checks, and recovery.

### Phase D — Reference Encoder and Runner Integration

Implement the deterministic NumPy encoder and `embed.patches@1`.

Integrate the pipeline without altering existing orchestration semantics.

### Phase E — Real WSI Validation and Reporting

Run CZI/SVS tests, failure injection, resource tests, installed-wheel verification, regression gates, and reproducibility checks.

## 16. Deliverables

- `domain/` embedding and model-input contracts.
- `ports/` encoder and store contracts.
- `application/` embedding orchestration.
- `infrastructure/` NumPy embedding store.
- `plugins/embedding/` reference encoder and operation.
- `pipelines/wsi-patch-embedding.v1.yaml`.
- Unit, contract, integration, and failure-injection tests.
- Architecture documentation.
- Acceptance-criteria evidence.
- Final implementation report under `.claude/reports/`.

## 17. Definition of Done

SPEC-005 is complete only when:

1. AC-001 through AC-020 are supported by reproducible evidence.
2. Both existing real WSI fixtures can produce valid, persistent embedding bags using the reference encoder.
3. Pixel memory is bounded by explicit policies.
4. Interrupted and corrupted stores are detected and safely handled.
5. Resolution and transformation provenance are complete.
6. Existing behavior and packaging remain compatible.
7. The installed package reproduces expected artifacts.
8. No clinical or model-quality claims are made from synthetic embeddings.