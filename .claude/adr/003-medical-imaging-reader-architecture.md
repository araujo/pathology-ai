# ADR 003 — Medical Imaging Reader Architecture

**Status:** Accepted (approved 2026-10-16)  
**Project:** Pathology AI  
**Related SPEC:** `../specs/003-medical-imaging-formats-and-readers.md`  
**Preserves:** ADR-001 and ADR-002

## 1. Context

Pathology AI currently provides modality-independent imaging contracts and synthetic execution.

The next architectural capability is reading real medical image files without coupling the domain or pipeline engine to a specific vendor format or decoding library.

Medical imaging formats vary significantly in their metadata, spatial semantics, multiresolution structure and access methods.

A single file extension does not establish all the capabilities needed by a scientific pipeline.

## 2. Decision

### D1 — Port-Based Reader Architecture

Real image readers will be infrastructure adapters implementing domain-owned imaging ports.

Reader implementations must be replaceable without modifying domain entities or pipeline definitions.

### D2 — Separate Modality, Format and Reader Capability

The platform will distinguish:

- Imaging modality.
- Storage format.
- Spatial representation.
- Reader backend.
- Supported operations.

Reader selection must use actual capabilities and file inspection, not extension alone.

### D3 — Level-Zero WSI Coordinates

The canonical spatial reference for WSI remains level-zero pixel coordinates.

Region reads at other pyramid levels must retain sufficient information to map back to level zero.

Physical spacing must be explicit and nullable when unknown.

### D4 — Explicit Pixel Semantics

The reader must report channel order, pixel type and decoding transformations.

Model-specific preprocessing remains outside the reader.

Silent RGB/BGR conversion, normalization or resizing is prohibited.

### D5 — Explicit No-Data Semantics

Valid pixels, background, unacquired regions and out-of-bounds coordinates must not be conflated.

If the file or backend does not provide enough information to distinguish them, the adapter must report the limitation.

### D6 — Bounded and Lazy Access

Readers must support bounded region access where the backend permits it.

Full-slide decoding must not be required for every region request.

Resource limits and cleanup are mandatory.

### D7 — Evidence-Based Format Support

A format is considered supported only after the relevant operations have been tested against actual fixtures.

Documentation-only support must be reported separately.

### D8 — Independent Imaging API

Reader capabilities must remain independent of the future DPA integration.

Existing Imaging API v1 contracts must be preserved.

Incompatible changes require a separate versioning decision.

### D9 — Preserve Root-Level Layout

Reader adapters and supporting services must reside under the root-level `infrastructure/`, `plugins/` or `application/` packages.

The `pathology_ai/` namespace must not return.

### D10 — No Scientific Inference Claims

Successfully decoding a real pathology image is not evidence that a model can produce scientifically valid predictions for that image.

Image ingestion and model validation are separate capabilities.

## 3. Alternatives Considered

### A — One Hardcoded Reader in the Core

**Rejected.** Couples the domain to a format and decoding library.

### B — Extension-Only Reader Selection

**Rejected.** Extensions do not guarantee format, pyramid structure or reader capabilities.

### C — Universal Conversion Before Reading

**Deferred.** Conversion can be useful for ingestion and caching, but must not be mandatory or erase source metadata.

### D — Port-Based Reader Registry

**Accepted.** Allows independent adapters, explicit capabilities and incremental format support.

## 4. Consequences

### Positive

- Additional formats can be introduced without redesigning pipelines.
- Spatial and channel semantics remain explicit.
- Real-image ingestion can be tested independently of model inference.
- DPA integration can use stable imaging contracts.
- Scientific provenance includes reader behavior.

### Risks

- Reader libraries may require native dependencies.
- Different backends may interpret metadata differently.
- Some formats lack reliable spacing or no-data information.
- Generic root-level package names may collide with third-party packages.
- Very large images require careful memory management.

### Mitigations

- Isolate optional dependencies in reader adapters.
- Test against real fixtures.
- Preserve raw metadata when appropriate.
- Reject unsupported operations.
- Enforce bounded reads.
- Track adapter/backend versions.
- Maintain import and packaging tests.

## 5. Non-Decisions

This ADR does not select a universal reader backend.

It does not require support for every WSI format.

It does not define a CT/MRI execution architecture.

It does not authorize model training, clinical inference or DPA modifications.

These decisions will be addressed in later specifications.

## 6. Status

**Status:** Accepted (approved 2026-10-16). Initial backends: `pylibczirw` 6.1.0 (CZI) and OpenSlide (Aperio SVS), each an isolated optional adapter.

Approval authorizes implementation according to SPEC-003, subject to the documented scope and acceptance criteria.