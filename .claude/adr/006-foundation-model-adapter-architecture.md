# ADR-006 — Foundation Model Adapter Architecture

**Status:** Proposed  
**Date:** 2026-10-21  
**Related specification:** SPEC-006  
**Repository:** `pathology-ai`

## 1. Context

SPEC-005 introduced a model-independent streaming embedding pipeline with lazy WSI patch reading, bounded batches, explicit transformation contracts, persistent embedding storage, chunk integrity, and safe recovery.

Its NumPy reference encoder validates the infrastructure but does not produce scientifically meaningful features.

SPEC-006 introduces pretrained pathology foundation models, starting with UNI.

The challenge is to support real model inference without coupling the framework to UNI, PyTorch, a particular staining protocol, a particular disease, or a universal physical image resolution.

## 2. Decision

### D1 — Foundation models are independent plugins

Each foundation model is implemented as a plugin behind the existing `EmbeddingEncoder` port.

Model-specific architecture construction, checkpoint loading, output interpretation, and preprocessing profiles remain inside the corresponding plugin.

Generic domain and application layers must not contain model-name-specific branches.

### D2 — Reuse SPEC-005 streaming infrastructure

All real model inference must use the existing `embed.patches@1` operation and `PipelineRunner`.

The implementation must preserve:

- Lazy patch reads.
- Bounded memory.
- Explicit transformations.
- Incremental persistent writes.
- Patch-to-embedding traceability.
- Chunk integrity and recovery.

A separate inference orchestrator is prohibited.

### D3 — PyTorch and timm remain optional

Introduce an optional dependency group, such as `[embedding]`, for supported model runtimes.

The base package must continue to operate without PyTorch, timm, CUDA, or downloaded weights.

### D4 — Model checkpoints are controlled artifacts

Checkpoints are loaded from explicitly configured local paths.

The runtime must not download model weights or initiate authentication.

Checkpoint SHA-256, architecture identity, adapter version, and relevant loading configuration must be recorded in embedding provenance.

Missing, corrupt, or incompatible checkpoints must fail closed.

### D5 — Preprocessing is model-specific and explicit

Every model must declare its required:

- Input shape and layout.
- Channel order.
- Input numeric range.
- Normalization.
- Physical image scale, where applicable.
- Resampling method, where applicable.
- Supported precision.
- Output embedding dimension.

Every applied transformation must be declared, versioned, validated, and included in the embedding identity.

### D6 — No universal physical resolution

The source WSI's native spacing, selected stored extraction level, extracted field of view, and model-required input scale are separate concepts.

No global µm/px default is permitted.

No implicit resampling is permitted.

If a model requires a scale or transformation that cannot be satisfied, inference must fail with a typed capability or resolution error.

### D7 — Official UNI and legacy SurGen UNI are distinct profiles

The official UNI preprocessing profile must be verified against authoritative model documentation.

The legacy SurGen compatibility profile must preserve explicitly documented legacy behavior without presenting it as the official recipe.

The two profiles must have different identities and must never be treated as interchangeable.

Exact numerical legacy parity is deferred unless separately approved.

### D8 — CPU is the baseline; CUDA is optional at execution

CPU inference is required.

CUDA execution is supported when compatible hardware and dependencies are available.

MPS is optional.

FP32 is the default precision. Reduced precision requires explicit configuration and validation.

### D9 — Reproducibility is scoped to runtime conditions

Model weights, preprocessing, precision, device, runtime versions, and relevant batch settings must be captured in provenance.

The framework must not claim bit-identical results across different hardware or library versions without evidence.

If an encoder cannot guarantee batch-size invariance, batch configuration must participate in the execution identity.

### D10 — Resource accounting remains explicit

Host-side streaming budgets from SPEC-005 remain enforced.

The model adapter must separately report model parameter memory, relevant temporary allocations, and measured device memory when available.

Model inference must not bypass the established batch limits.

### D11 — Licensing and model documentation are gates

Before enabling an official model adapter, verify the architecture, preprocessing recipe, checkpoint format, and applicable license or access conditions.

No weights are bundled or redistributed without authorization.

Unverified model assumptions must not be silently filled using legacy behavior.

### D12 — Technical validation is not clinical validation

Successful model inference establishes technical correctness of integration only.

Scientific performance, clinical validity, biomarker prediction, and generalization require separate experiments and evaluation.

### D13 — Repository governance is preserved

- All engineering governance artifacts belong under `.claude/`.
- `../digital-pathology` and `../hal-viewer` remain strictly read-only.
- No legacy source code is copied.
- No Git commits are created without explicit authorization.

## 3. Consequences

### Positive

- Reusable model integration architecture.
- Independent encoder adapters.
- Explicit scientific preprocessing contracts.
- Reproducible model and checkpoint provenance.
- Compatibility with future UNI2-h, Virchow, PLIP, and other encoders.
- Direct integration with the future MIL pipeline.

### Trade-offs

- Additional model-specific verification.
- Larger optional runtime dependencies.
- Hardware-dependent numerical behavior.
- More complex memory accounting.
- Separate preprocessing configurations for official and legacy experiments.

## 4. Deferred Decisions

- Additional foundation-model adapters.
- Fine-tuning.
- Multi-GPU inference.
- Distributed embedding extraction.
- Remote model registries.
- Automatic model downloads.
- Legacy numerical parity.
- Clinical validation.

## 5. Acceptance Conditions

ADR-006 can be accepted after the architecture assessment confirms compatibility with SPEC-005 and verifies that no generic layer must become UNI-specific.

SPEC-006 implementation completion requires separate evidence for the real UNI checkpoint, when available.

An unavailable checkpoint must remain an explicitly reported validation gap.