# ADR-005 — Streaming Embedding Architecture and Resolution Contracts

**Status:** Proposed  
**Date:** 2026-10-20  
**Related:** SPEC-005  
**Repository:** `pathology-ai`

## Context

The legacy Digital Pathology pipeline uses lazy patch reads and incremental embedding writes, but has several limitations: implicit resampling, implicit color semantics, positional-only row association, unverified reader concurrency, and incomplete artifact validation.

SPEC-004 provides deterministic patch plans and canonical spatial identities but does not offer scalable persistent embedding extraction.

We need a reusable, model-independent architecture for producing embedding bags from real WSIs without unbounded pixel retention.

## Decisions

### D1 — Coordinates and manifests form the streaming boundary

The embedding pipeline consumes patch plans and metadata rather than materialized patch-pixel collections.

Patch pixels are decoded only when needed.

### D2 — Batching remains inside the embedding operation

Use a fused `embed.patches@1` operation under the existing `PipelineRunner`.

Do not introduce streaming DAG edges or a second orchestration engine.

### D3 — Native stored resolutions are preserved

Do not impose a universal target microns-per-pixel value.

Stored level selection is explicit and reader-neutral.

Any model-required resampling must be explicitly configured, validated, and recorded.

### D4 — Model preprocessing is a contract

Model input geometry, channel order, normalization, dtype, precision, and physical scale are declared by the encoder.

No hidden transformations are permitted.

### D5 — Encoder interfaces are batch-oriented and framework-neutral

Define an `EmbeddingEncoder` port without PyTorch-specific types.

Concrete implementations may use NumPy, PyTorch, or other runtimes.

### D6 — Durable embedding storage uses a port

Define an `EmbeddingStore` contract with incremental writes, chunk commits, integrity checks, resume, and finalization.

The initial adapter uses NumPy `.npy` memory-mapped storage with JSON metadata.

Zarr may be added later without changing domain contracts.

### D7 — Artifact completion is explicit and verifiable

Preallocated array shape does not imply completeness.

Only verified committed chunks contribute to finalization.

Partial artifacts must fail validation.

### D8 — Embedding identity binds scientific inputs

Artifact identity includes patch identity, encoder identity, model-weight identity when applicable, model input contract, transformations, and embedding content integrity.

Runtime-local paths and timestamps do not influence stable scientific identity.

### D9 — Resource budgets are enforced

The system must bound pixel buffers and in-flight work, not merely report peak memory after execution.

Memory regression and failure-injection tests are mandatory.

### D10 — Reader concurrency is conservative

Do not share a native reader handle across forked processes.

Begin with a single reader or bounded, independently opened readers where concurrency is demonstrably safe.

Do not assume thread safety of libCZI or OpenSlide handles.

### D11 — A deterministic reference encoder precedes real foundation models

SPEC-005 validates the infrastructure with a NumPy-only synthetic encoder.

Production UNI integration and model-specific scientific validation belong to a subsequent specification.

### D12 — Legacy compatibility is an explicit profile

Legacy BGR, ImageNet normalization, and INTER_AREA resampling must never become universal defaults.

Any legacy-compatible profile is independently versioned and carries a distinct embedding identity.

### D13 — Dependencies remain optional

The base package must not require PyTorch, timm, model weights, or Zarr.

Model-specific dependencies are introduced through optional adapters and explicit packaging decisions.

### D14 — Failure recovery is identity-bound

Resume requires matching patch plan, encoder, transforms, output schema, and execution identity.

Incompatible or corrupt partial stores are rejected.

### D15 — Existing architecture remains authoritative

Reuse SPEC-001 through SPEC-004 contracts and the existing `PipelineRunner`.

No format-specific business logic is introduced into the core.

Sibling repositories remain read-only and are not runtime dependencies.

## Consequences

### Benefits

- Bounded pixel memory.
- Format-neutral execution.
- Reproducible embeddings.
- Durable intermediate results.
- Explicit resolution and color semantics.
- Recovery from interruptions.
- Independent encoder adapters.
- A clean foundation for MIL and slide-level inference.

### Trade-offs

- More metadata and integrity bookkeeping.
- Additional filesystem recovery complexity.
- Potentially lower throughput with conservative reader concurrency.
- Separate model-specific adapter work.
- NumPy storage is less flexible than Zarr for some remote or distributed storage scenarios.

## Deferred Decisions

- Zarr adapter implementation.
- UNI and other production encoder integrations.
- Multi-GPU and distributed scheduling.
- Object-store-native persistence.
- Numerical reproduction of legacy embeddings.
- Import of legacy embedding artifacts.

## Validation

ADR-005 is satisfied when SPEC-005 AC-001 through AC-020 pass, with particular attention to bounded memory, crash recovery, manifest integrity, and explicit resolution semantics.