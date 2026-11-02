# ADR 004 — Real WSI Preprocessing Architecture

**Status:** Accepted (approved 2026-10-16)  
**Project:** Pathology AI  
**Related SPEC:** `../specs/004-real-wsi-preprocessing-and-patch-extraction.md`  
**Preserves:** ADR-001, ADR-002 and ADR-003

## 1. Context

SPEC-003 introduced real WSI readers and an ingestion workflow.

The framework now needs reusable preprocessing and patch-extraction operations that execute through the existing typed pipeline engine.

The legacy Digital Pathology implementation provides useful tissue detection and patch-generation techniques, but its implementation is coupled to specific readers, coordinate assumptions and experimental workflows.

## 2. Decisions

### D1 — Separate Detection, Planning and Extraction

Tissue detection, patch planning and patch extraction are independent operations with typed inputs and outputs.

Each can be replaced or extended without modifying the other stages.

### D2 — Reader-Neutral Operations

Preprocessing depends on image and spatial contracts, not on CZI, SVS or a concrete reader library.

### D3 — Acquisition Validity Is Not Tissue

Acquired-data coverage and tissue detection remain separate masks or measurements.

Unknown validity must be handled through an explicit policy.

### D4 — Canonical Level-Zero Patch Coordinates

Every patch is identified by its canonical level-zero geometry.

Native CZI and legacy padded coordinates are represented through explicit transforms.

### D5 — Explicit Resolution Semantics

Patch dimensions, physical spacing, read level, resampling and effective output resolution are separate concepts.

No implicit resizing or pixel-spacing assumptions are allowed.

### D6 — Model-Independent Patch Artifacts

Patch extraction produces reusable image artifacts.

Model-specific color normalization, resizing and intensity normalization belong to later model-bound preprocessing.

### D7 — Deterministic Plans and Manifests

Patch plans, rejection decisions, identities and manifests must be deterministic under fixed inputs and configuration.

### D8 — Bounded Execution

Whole-slide level-zero decoding is prohibited as a routine preprocessing strategy.

Region reads, patch counts, batches and output budgets must be bounded.

### D9 — Existing Pipeline Engine

Real WSI preprocessing must use `PipelineRunner`.

A separate workflow engine is not permitted.

### D10 — Evidence-Based Legacy Compatibility

Legacy coordinate behavior must be assessed and translated explicitly.

Equivalent behavior must be independently tested rather than assumed.

### D11 — No Scientific Overclaiming

Tissue candidate detection is an engineering preprocessing step.

It is not tumor detection, diagnosis or clinical validation.

### D12 — Cross-Format Conformance

Every WSI adapter must satisfy a shared set of reader and preprocessing contracts for the capabilities it claims to support.

### D13 — Capability-Gated Preprocessing

Pipeline compatibility is determined by declared and verified capabilities, not file extension or backend identity.

### D14 — Evidence-Based Format Support

Support claims are scoped to tested formats, fixtures, operations and backend versions.

### D15 — Semantic Equivalence, Not Universal Pixel Equality

Cross-backend equivalence requires consistent geometry, metadata, quality decisions and provenance semantics. Pixel equality is required only where technically justified.

## 3. Alternatives

### A — Reuse the Legacy Pipeline Directly

**Rejected.** Creates runtime coupling and carries over legacy coordinate assumptions.

### B — Implement One Monolithic WSI Processor

**Rejected.** Makes detector, patch strategy and storage difficult to replace.

### C — Precompute Entire Full-Resolution WSI

**Rejected.** Creates unacceptable memory and scalability risks.

### D — Typed Modular Operations Through PipelineRunner

**Accepted.** Preserves the platform's extensibility and reproducibility requirements.

## 4. Consequences

### Benefits

- Supports multiple WSI formats.
- Allows multiple tissue detectors.
- Allows different patching strategies.
- Provides a stable handoff for embedding models.
- Makes coordinate and acquisition-validity semantics testable.
- Preserves scientific provenance.

### Risks

- Low-resolution tissue masks can misrepresent fine boundaries.
- Unknown acquisition coverage can affect patch decisions.
- Pixel-based and physical-size patches may not be interchangeable.
- Repeated region reads may be expensive.
- Storing many patches can increase storage cost.
- Legacy patch coordinates can be incompatible.

### Mitigations

- Record mask resolution and effective scale.
- Separate acquired-data and tissue thresholds.
- Reject unsupported physical-resolution requests.
- Bound read and storage budgets.
- Support manifests independently of pixel materialization.
- Require explicit legacy coordinate transforms.

## 5. Deferred Decisions

This ADR does not choose:

- A foundation embedding model.
- A universal tissue detector.
- A distributed patch store.
- A mandatory patch size.
- A universal physical resolution.
- A training framework.
- A CT/MRI preprocessing implementation.
- An API contract revision.
- A DPA integration strategy.

## 6. Status

**Accepted (approved 2026-10-16)** after the Phase A assessment (`../reports/004-real-wsi-preprocessing-and-patch-extraction.md`). D12–D15 were incorporated from the PROMPT-004 cross-format addendum; D1–D11 are unchanged.

Approved implementation decisions:

| # | Decision |
|---|---|
| R1 | NumPy-only deterministic saturation/Otsu baseline detector; replaceable through a detector port. |
| R2 | Minimal optional `OpenedImage` execution context on `ExecutionRequest` and `StageContext`; deterministic cleanup; no backend-specific imports in the runner. Real backends are also registered in the existing `ReaderRegistry` by format code. |
| R3 | A preprocessing-only `TaskDefinition`; existing task semantics preserved. |
| R4 | Patch budgets fail closed by default; explicit deterministic selection policies are allowed and recorded in provenance. |
| R5 | In-memory pixel artifacts plus deterministic JSON manifests; artifact identity is distinguished from durable pixel persistence. |
| R6 | Patch level and resolution configurable per execution; the same stored level is never assumed across formats. |
| R7 | Legacy T198 Zarr coordinates read strictly read-only for coordinate-bridge verification. |
| R8 | Additive `native_padded_origin` metadata in the CZI adapter; canonical level-zero semantics and Imaging API v1 unchanged. |