# PROMPT-005 — Streaming Patch-to-Embedding Pipeline

**Target file:** `.claude/prompts/005-streaming-patch-to-embedding-pipeline.md`

## Role

You are the implementation engineer for SPEC-005 in the `pathology-ai` repository.

Your responsibility is to implement a scalable, deterministic, model-independent streaming pipeline that consumes real WSI patch plans and produces durable embedding artifacts.

Follow the established architecture, accepted ADRs, and repository conventions.

## Authoritative Documents

Read these files before making changes:

1. `.claude/specs/005-streaming-patch-to-embedding-pipeline.md`
2. `.claude/adr/005-streaming-embedding-architecture.md`
3. `.claude/architecture/005-legacy-streaming-and-embedding-assessment.md`
4. SPEC-001 through SPEC-004 and their accepted ADRs.

If SPEC-005 or ADR-005 is missing, stop and request the original document. Do not reconstruct it from this prompt.

## Phase A — Architecture Assessment

Inspect the existing implementation and verify:

- How SPEC-004 exposes `PatchPlan`, `PatchManifest`, and `OpenedImage`.
- Whether patch plans can be consumed with `materialize_pixels=false`.
- How the `PipelineRunner` passes open-image context and per-run configuration.
- Existing artifact identities and canonical coordinate contracts.
- Existing model, encoder, operation, and storage abstractions.
- Dependency boundaries and packaging constraints.
- How the proposed embedding workflow can be integrated without a second orchestration engine.

Reuse existing contracts where possible.

Produce a concise assessment identifying reusable components, necessary additions, and risks.

**Stop for owner approval only if a substantive architectural conflict or missing authoritative document is discovered.**

## Phase B — Domain Contracts and Ports

Implement the necessary model-independent contracts:

- `ModelInputSpec`
- `TransformChain`
- `BatchPolicy`
- `EmbeddingRecord`
- `EmbeddingManifest`
- `EmbeddingEncoder`
- `EmbeddingStore`

Ensure that:

- Patch identity and canonical coordinates are preserved.
- Model requirements are separate from reader capabilities.
- Input resolution, channel order, shape, dtype, normalization, and precision are explicit.
- No universal physical resolution is imposed.
- No resampling or BGR/RGB conversion occurs implicitly.
- Output dimensions, dtypes, and finite values are validated.
- Existing domain abstractions are extended rather than duplicated.

## Phase C — Streaming and Durable Storage

Implement lazy patch consumption and bounded batch processing.

### Streaming

- Read pixels from `OpenedImage` only when needed.
- Consume SPEC-004 patch plans and manifests without materializing all patch pixels.
- Preserve deterministic patch order and stable patch IDs.
- Enforce explicit batch-size and byte budgets.
- Keep in-flight work bounded.
- Avoid unbounded queues and collections of pixel arrays.
- Do not assume native image-reader handles are thread-safe or fork-safe.

### Persistent storage

Implement a NumPy-backed `EmbeddingStore` using `.npy` and versioned JSON metadata.

Required behavior:

- Incremental chunk writes.
- Integrity verification.
- Explicit chunk commits.
- Reopening and validating partial stores.
- Safe resume from verified chunks.
- Rejection of incompatible run identities.
- Finalization only after complete verified coverage.
- Stable embedding content digests.
- Persistent mapping between patch IDs, coordinates, and embedding rows.

Document filesystem durability assumptions.

Do not treat the shape of a preallocated array as evidence of successful completion.

Do not introduce Zarr, HDF5, or additional dependencies in this phase.

## Phase D — Encoder and Pipeline Integration

Implement a deterministic NumPy-only reference encoder.

This encoder is an architectural test fixture, not a medical foundation model.

Requirements:

- Fixed and validated input contract.
- Configurable embedding dimension.
- Deterministic output for identical inputs.
- No external model downloads.
- No PyTorch dependency.
- No clinically meaningful output claims.

Register:

`embed.patches@1`

Integrate the operation with the existing `PipelineRunner`.

Create:

`pipelines/wsi-patch-embedding.v1.yaml`

The stage must return only a durable store reference, completed manifest, and bounded metrics.

Do not return all embeddings or patch pixels through the in-memory artifact store.

Preserve existing runner behavior and fingerprints when the new operation is unused.

## Phase E — Validation and Evidence

Execute all quality gates:

- `ruff`
- `mypy --strict`
- Full `pytest` suite
- Existing SPEC-001 through SPEC-004 regression tests
- CZI integration tests
- SVS integration tests
- Streaming and resource-bound tests
- Failure-injection and corruption tests
- Resume and finalization tests
- Fresh-process reproducibility tests
- Installed-wheel smoke tests
- Synthetic demo byte-identity verification

Validate that:

1. Pixel retention does not grow with the total number of patches.
2. Embeddings are written incrementally.
3. Patch-to-embedding association is exact.
4. Interrupted stores cannot be accepted as complete.
5. Corrupt chunks are detected.
6. Resume never silently mixes incompatible identities.
7. Resolution and transformation provenance are complete.
8. Equivalent executions produce stable digests.
9. Both CZI and SVS work through the same embedding operation.
10. The base package remains functional without ML framework dependencies.

Report AC-001 through AC-020 individually, with evidence.

## Scientific and Architectural Constraints

- Preserve explicitly selected native/stored resolution.
- Do not impose a universal microns-per-pixel target.
- Do not silently resample images.
- Do not silently change channel order.
- Preserve canonical level-zero coordinates.
- Keep encoder preprocessing model-specific.
- Do not reproduce legacy preprocessing bugs.
- Do not copy code from `../digital-pathology`.
- Do not modify `../digital-pathology` or `../hal-viewer`.
- Do not introduce a second pipeline runner.
- Do not add dependencies without approval.
- Do not implement MIL or slide-level classification.
- Do not download or integrate UNI weights yet.
- Do not create Git commits.

## Deliverables

Produce or update:

- `domain/` — embedding and input contracts.
- `ports/` — encoder and persistent storage interfaces.
- `application/` — embedding orchestration.
- `infrastructure/` — NumPy embedding store.
- `plugins/embedding/` — reference encoder and operation.
- `pipelines/wsi-patch-embedding.v1.yaml`.
- `.claude/architecture/005-streaming-embedding-pipeline.md`.
- `.claude/reports/005-streaming-patch-to-embedding-pipeline.md`.

The final report must include:

- Implementation summary.
- Architectural decisions and deviations.
- Acceptance-criteria matrix (AC-001–AC-020).
- Actual CZI and SVS execution evidence.
- Patch counts and embedding dimensions.
- Batch sizes and memory measurements.
- Persistent artifact sizes and digests.
- Recovery and corruption test results.
- Reproducibility evidence.
- Packaging and regression results.
- Remaining limitations.

## Completion Rule

SPEC-005 is complete only when all 20 acceptance criteria have verifiable evidence and the complete quality-gate suite passes.

Do not mark SPEC-005 or ADR-005 as Accepted before validation.

**Begin with Phase A. If no architectural conflict exists, proceed through Phases B–E and produce the final implementation report.**