# SPEC-006 — Foundation Model Integration & Real Embedding Extraction

**Status:** Proposed  
**Priority:** P0  
**Dependencies:** SPEC-001 through SPEC-005  
**Repository:** `pathology-ai`

## 1. Context

SPEC-005 implemented a model-independent, bounded-memory, resumable pipeline for extracting and persisting patch embeddings from whole-slide images.

The existing pipeline includes:

- Real WSI reading through format-independent adapters.
- Deterministic tissue detection and patch planning.
- Pixel-free patch manifests.
- Lazy region reads and bounded batching.
- Explicit model input specifications and transformation chains.
- Batch-oriented `EmbeddingEncoder` contracts.
- Incremental embedding persistence with chunk integrity, recovery, and reproducible identities.

The existing NumPy reference encoder validates infrastructure but does not produce medically meaningful representations.

SPEC-006 introduces real pretrained foundation-model inference, starting with UNI, without introducing model-specific assumptions into the domain, application, or pipeline orchestration layers.

The architecture must support future encoder adapters such as UNI2-h, Virchow, and PLIP without modifying the generic streaming pipeline.

## 2. Objectives

1. Integrate a real PyTorch-based pathology foundation model.
2. Implement UNI as the first production-capable encoder adapter.
3. Load pretrained checkpoints from explicit local paths.
4. Validate model identity, architecture, weights, and input requirements.
5. Reuse SPEC-005 streaming, memory budgets, and durable embedding storage.
6. Support CPU and CUDA inference.
7. Make model-specific preprocessing explicit, reproducible, and independently versioned.
8. Generate real embedding bags from CZI and SVS images.
9. Preserve patch identity, coordinates, and physical resolution provenance.
10. Support future foundation models through a plugin/registry mechanism.

## 3. Scope

### In scope

- Optional PyTorch/timm runtime.
- Model registry and encoder factory.
- UNI adapter.
- Explicit local checkpoint resolution.
- Checkpoint SHA-256 verification.
- Model architecture and output-dimension validation.
- CPU and CUDA device handling.
- Bounded inference batches.
- Model-specific normalization, channel order, and input size.
- Explicit physical-scale requirements.
- Real CZI and SVS embedding extraction.
- Deterministic inference controls and reproducibility reporting.
- Model-specific error handling.
- Integration tests with lightweight local model fixtures.
- Optional real UNI integration tests when licensed weights are available.

### Out of scope

- Foundation-model training or fine-tuning.
- Automatic model downloads.
- Interactive model authentication.
- UNI2-h, Virchow, or PLIP implementations.
- MIL training.
- Slide-level classification.
- Clinical validation.
- Multi-GPU distributed inference.
- Model serving.
- Legacy numerical parity as a mandatory requirement.

## 4. Architectural Principles

### P1 — The framework remains model-independent

UNI-specific architecture construction, checkpoint handling, output extraction, and preprocessing profiles belong in a dedicated plugin.

Generic layers must not contain UNI-specific branches.

### P2 — Model contracts are authoritative

Every encoder declares:

- Input dimensions.
- Channel order.
- Input dtype and numeric range.
- Tensor layout.
- Normalization parameters.
- Physical-scale requirements or explicit variable-scale acceptance.
- Supported devices and precision.
- Output embedding dimension.
- Architecture identity.
- Weights identity.

### P3 — No implicit image transformations

Do not silently resize, resample, reorder channels, normalize, or reinterpret image spacing.

All transformations must be declared, validated, versioned, and included in artifact provenance.

### P4 — Native image resolution remains independent

A WSI's source resolution and selected stored extraction level remain properties of the image and patch plan.

A foundation model's required input scale is a separate contract.

A mismatch must fail unless an explicitly declared and supported transformation resolves it.

### P5 — Streaming remains authoritative

The adapter consumes batches from SPEC-005.

It must not load all patches or embeddings into memory.

### P6 — Model weights are controlled artifacts

Weights must be available locally before execution.

No runtime network access, authentication, or downloads are permitted.

The model's identity includes the checkpoint content digest.

### P7 — Scientific claims require separate validation

Successful inference establishes technical functionality, not clinical performance, biomarker prediction accuracy, or diagnostic validity.

## 5. Proposed Architecture

```text
PipelineRunner
    |
    v
embed.patches@1
    |
    +-- PatchPlan + Pixel-free Manifest
    |
    +-- Model Registry
    |       |
    |       +-- Synthetic NumPy Encoder (SPEC-005)
    |       |
    |       +-- UNI PyTorch Encoder (SPEC-006)
    |       |
    |       +-- Future Encoders
    |
    +-- ModelInputSpec validation
    |
    +-- LazyPatchSource
    |
    +-- Declared TransformChain
    |
    +-- Bounded BatchPolicy
    |
    +-- EmbeddingEncoder.encode_batch
    |       |
    |       +-- PyTorch inference_mode
    |       +-- Device execution
    |       +-- Output validation
    |
    +-- EmbeddingStore
    |       |
    |       +-- Incremental chunk commits
    |       +-- Resume and corruption checks
    |
    v
Real EmbeddingManifest + StoreReference
```

The existing `embed.patches@1` operation remains the integration point.

Do not introduce a separate foundation-model pipeline runner.

## 6. Foundation Model Registry

Introduce or extend a registry supporting:

- Encoder identifier.
- Adapter implementation and version.
- Architecture identifier.
- Local checkpoint path.
- Expected checkpoint checksum.
- Input contract.
- Output contract.
- Supported device types.
- Supported precision.
- Model-specific configuration.
- Optional licensing or access metadata.

Model resolution must fail closed when the requested adapter or checkpoint is unavailable.

The registry must not download weights.

The local checkpoint path is operational configuration; the checkpoint digest is part of the scientific artifact identity.

## 7. UNI Adapter

Implement a UNI encoder using the architecture and loading behavior verified against the actual checkpoint and authoritative model documentation.

The legacy `digital-pathology` implementation may be inspected read-only for compatibility evidence, but it must not be treated as the authoritative specification of the official UNI preprocessing recipe.

### Required behavior

- Construct the correct UNI architecture.
- Load a local checkpoint with strict compatibility checks.
- Verify expected checkpoint SHA-256.
- Reject missing, corrupt, or incompatible checkpoints.
- Switch the model to evaluation mode.
- Execute inference without gradient tracking.
- Produce one embedding row per input patch.
- Validate output dimension and dtype.
- Return a NumPy-compatible batch to the generic embedding pipeline.
- Release model resources appropriately.

### Initial expected configuration

The legacy implementation reports:

- `timm` ViT-L/16.
- Input geometry of 224 × 224 pixels.
- Embedding dimension of 1024.
- ImageNet-style normalization.
- A legacy BGR input path.

These are compatibility observations, not an independently verified official UNI contract.

Before finalizing the production UNI profile, verify the official model card, architecture configuration, preprocessing recipe, weights format, and license terms.

## 8. Preprocessing Profiles

Support two independently identified profiles.

### Profile A — Official UNI

A model-specific profile based on verified authoritative UNI documentation.

It declares:

- Input geometry.
- Channel order.
- Normalization.
- Physical-scale requirements.
- Resampling requirements, if any.
- Transform implementation versions.
- Supported precision.

If the official recipe cannot be verified, this profile must remain unavailable rather than being guessed.

### Profile B — Legacy SurGen UNI Compatibility

An explicitly named and versioned compatibility profile based on the legacy experiment.

It may declare:

- BGR channel order.
- Legacy ImageNet normalization.
- The historical 1.0 µm/px target.
- Historical level-zero region geometry.
- INTER_AREA resampling to 224 × 224.
- Legacy checkpoint identity.

The profile must not be presented as the official UNI preprocessing recipe.

Legacy numerical parity is not required for SPEC-006. If the exact resampling operation is unsupported, reject the profile with a clear capability error rather than silently approximating it.

Embeddings from the official and legacy profiles must have distinct identities and must not be treated as interchangeable.

## 9. Device and Precision Management

### CPU

Required for all generic adapter tests.

### CUDA

Required when a compatible CUDA runtime is available.

The adapter must detect requested versus available device capabilities and fail clearly when CUDA is requested but unavailable.

### MPS

Optional, not required for SPEC-006 acceptance.

### Precision

FP32 is the default.

FP16 and BF16 may be added as explicit opt-in configurations only if their support, numerical behavior, and provenance are validated.

No implicit automatic mixed precision.

### Resource management

The adapter must account for:

- Model parameter memory.
- Model activations.
- Input and output tensors.
- CPU/GPU transfers.
- Temporary preprocessing buffers.
- In-flight batch limits.

The SPEC-005 memory budget must remain enforced for host-side buffers.

GPU peak memory must be measured separately when CUDA is available.

## 10. Determinism and Reproducibility

The adapter must record:

- Model identifier.
- Adapter version.
- Architecture.
- Checkpoint SHA-256.
- Preprocessing profile.
- Transform chain.
- Input resolution and field of view.
- Device.
- Precision.
- Runtime library versions.
- Embedding shape and dtype.
- Output digest.

Tests must distinguish exact reproducibility from numerical tolerance.

Do not assume GPU inference is bit-identical across hardware, CUDA versions, or batch sizes.

Where exact batch invariance cannot be guaranteed, include batch configuration in execution identity and document the reproducibility scope.

## 11. Real WSI Integration

Run the same pipeline against the existing CZI and SVS fixtures.

For each:

1. Read the WSI through the existing adapter.
2. Select the extraction level explicitly.
3. Generate the SPEC-004 patch plan.
4. Produce a pixel-free manifest.
5. Validate the UNI model input contract.
6. Read patches lazily.
7. Apply only declared transformations.
8. Run batched UNI inference.
9. Write the embedding bag incrementally.
10. Finalize and validate the artifact.

Use fixture-compatible preprocessing configurations. Do not silently force both WSIs to a universal extraction spacing.

If the official UNI profile requires a transformation not yet implemented, report the capability gap rather than silently substituting another method.

## 12. Validation Strategy

### Unit tests

- Registry resolution.
- Missing model.
- Missing checkpoint.
- Checksum mismatch.
- Architecture mismatch.
- Output-dimension mismatch.
- Input contract mismatch.
- Device selection.
- Precision handling.
- Preprocessing profile identity.

### Adapter tests

- Small local PyTorch model fixture.
- Deterministic CPU inference.
- Batch shape and dtype validation.
- Correct eval and inference modes.
- No gradients.
- Resource cleanup.
- Output identity changes when checkpoint changes.

### Integration tests

- Existing synthetic encoder regression.
- End-to-end streaming with a lightweight torch adapter.
- CZI and SVS processing with local fixture weights.
- Optional real UNI checkpoint tests.

### Recovery tests

- Encoder exception.
- Device failure.
- Invalid output.
- Interrupted write.
- Resume with matching identity.
- Rejection of incompatible checkpoint or preprocessing configuration.

### Packaging tests

- Base install without torch/timm.
- Optional ML installation.
- Missing optional dependency errors.
- Installed-wheel inference.

## 13. Acceptance Criteria

**AC-001 — Model-independent architecture**

Foundation-model integration does not introduce UNI-specific behavior into domain, ports, application, or the generic runner.

**AC-002 — Optional ML runtime**

The base installation remains functional without torch or timm.

**AC-003 — Model registry**

Encoders are selected by stable identifiers and validated configurations.

**AC-004 — Local checkpoint control**

No checkpoint is downloaded or authenticated during pipeline execution.

**AC-005 — Checkpoint integrity**

Missing, corrupt, checksum-mismatched, or architecture-incompatible weights are rejected.

**AC-006 — UNI architecture**

The UNI adapter uses a verified architecture definition and validates its output contract.

**AC-007 — Official preprocessing profile**

The official UNI profile is backed by verified model documentation. Unverified requirements cause a clear blocking status.

**AC-008 — Legacy compatibility separation**

Legacy SurGen preprocessing is independently identified and cannot be silently substituted for the official profile.

**AC-009 — Explicit resolution handling**

Extraction resolution, model input scale, and declared transforms are validated separately.

**AC-010 — Batch inference**

The adapter processes bounded batches and emits one validated embedding row per patch.

**AC-011 — CPU support**

Real PyTorch inference succeeds on CPU using local fixture weights.

**AC-012 — CUDA support**

CUDA inference succeeds when compatible hardware is available; otherwise the test is explicitly skipped and the capability remains unverified.

**AC-013 — Device and precision provenance**

Device, precision, relevant runtime versions, and resource configuration are recorded.

**AC-014 — Persistent embedding integration**

Real model outputs are written through the existing SPEC-005 `EmbeddingStore`.

**AC-015 — Resume integrity**

Interrupted runs resume safely without mixing model or preprocessing identities.

**AC-016 — CZI integration**

The existing CZI fixture can produce a valid persistent embedding bag with a real torch encoder; real UNI is tested when weights are available.

**AC-017 — SVS integration**

The existing SVS fixture can produce a valid persistent embedding bag with a real torch encoder; real UNI is tested when weights are available.

**AC-018 — Reproducibility**

Repeated CPU executions with identical weights and configurations produce stable outputs within a documented reproducibility contract.

**AC-019 — Regression preservation**

SPEC-001 through SPEC-005 quality gates and behavior remain valid.

**AC-020 — Packaging and governance**

Optional dependencies, architectural boundaries, implementation evidence, and final reports comply with repository governance.

## 14. Implementation Phases

### Phase A — Foundation Model Discovery

Before implementation:

- Inspect existing SPEC-005 encoder and transform contracts.
- Inspect the legacy UNI adapter read-only.
- Verify official UNI model documentation.
- Verify model-weight license and access conditions.
- Identify the exact checkpoint architecture and state-dictionary layout.
- Compare official and legacy preprocessing.
- Identify required transformations not yet supported.
- Confirm optional dependency compatibility with Python 3.12.

Produce a discovery report.

Do not implement unverified model-specific assumptions.

### Phase B — Generic Runtime

Implement optional ML dependencies, model registry, checkpoint validation, device selection, and reusable torch encoder infrastructure.

### Phase C — UNI Plugin

Implement the verified UNI adapter and independently versioned preprocessing profiles.

### Phase D — Integration

Connect the UNI adapter to the existing `embed.patches@1` operation and durable embedding store.

### Phase E — Validation

Execute quality gates, lightweight fixture tests, available real UNI tests, resource measurements, and full regression checks.

## 15. Required Deliverables

- Generic model registry and torch runtime.
- UNI encoder plugin.
- Explicit official and legacy profile definitions.
- Local checkpoint verification.
- CPU and available CUDA tests.
- CZI/SVS embedding integration tests.
- Model-specific provenance.
- `.claude/architecture/006-foundation-model-integration.md`.
- `.claude/reports/006-foundation-model-discovery.md`.
- `.claude/reports/006-foundation-model-integration.md`.

## 16. Definition of Done

SPEC-006 is complete when:

1. AC-001 through AC-020 have documented outcomes.
2. The generic torch adapter works end to end.
3. UNI architecture and preprocessing have been verified against authoritative documentation.
4. Model-specific configuration cannot be silently substituted.
5. The existing streaming and persistent storage guarantees remain intact.
6. Both real WSI formats work through the same pipeline.
7. Existing tests and packaging remain green.
8. Any untested real UNI checkpoint or CUDA execution is explicitly marked unverified rather than claimed as passing.
9. No unsupported clinical or biological performance claims are made.

**A fully validated real UNI checkpoint run requires the authorized weights to be available locally. If they are unavailable, the implementation may be technically ready, but real UNI execution remains an explicit outstanding validation gate.**