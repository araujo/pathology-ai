# 003 — Medical Imaging Formats & Readers

**Status:** Accepted (approved 2026-10-16)  
**Project:** Pathology AI  
**Repository:** `pathology-ai`  
**Priority:** P0  
**Type:** Feature / Imaging Infrastructure  
**Depends on:** 001 — Platform Foundations; 002 — Repository Structure & Packaging  
**Related ADR:** `003-medical-imaging-reader-architecture.md`

## 1. Context

Pathology AI is an independent medical imaging AI framework designed to support multiple imaging modalities, scientific tasks, datasets, model architectures and research projects.

SPEC-001 established the platform foundation, including domain contracts, ports, synthetic pipelines, spatial references, model-bound preprocessing, scientific validation and provenance.

SPEC-002 moved the framework modules directly to the repository root, preserving behavior and establishing the approved physical layout.

The platform currently supports synthetic imaging execution and contract-only volumetric workflows. It does not yet read real medical image files.

SPEC-003 introduces the first real medical image reader capability, focusing on Whole Slide Images (WSI), while preserving the modality-independent architecture.

## 2. Objectives

1. Establish an extensible medical imaging reader architecture.
2. Implement at least one real WSI reader adapter.
3. Investigate CZI, SVS and pyramidal TIFF compatibility using actual library capabilities.
4. Expose image metadata and spatial geometry through existing domain contracts.
5. Provide bounded, on-demand image region reading.
6. Support multiresolution pyramid information when available.
7. Handle background, missing pixels and no-data regions explicitly.
8. Preserve image channel order and color-space semantics.
9. Integrate a real reader with the existing pipeline infrastructure.
10. Maintain independent Imaging API contracts for future DPA integration.
11. Preserve the existing synthetic pipelines and tests.
12. Keep CT/MRI execution outside the scope of this specification.

## 3. Architectural Principles

### 3.1 Modality and Format Independence

Image modality and storage format must remain separate concepts.

Examples:

- WSI stored as CZI.
- WSI stored as SVS.
- WSI stored as pyramidal TIFF.
- WSI stored as DICOM.
- CT stored as DICOM.
- MRI stored as DICOM or NIfTI.

A file extension must not be treated as proof of modality, image dimensions, spatial calibration or reader compatibility.

### 3.2 Readers Are Infrastructure Adapters

Image readers must implement domain-owned ports.

The domain must not depend on OpenSlide, Bio-Formats, libvips, tifffile, AICSImageIO, or any specific reader library.

The reader implementation belongs in the root-level `infrastructure/` or `plugins/` hierarchy, depending on the established dependency rules.

### 3.3 Explicit Spatial Semantics

All returned image regions must have explicit spatial references.

For WSI, canonical coordinates remain in level-zero pixel space.

A reader must distinguish:

- Level-zero dimensions.
- Pyramid level dimensions.
- Level-specific downsampling.
- Pixel spacing and physical units.
- Region origin and extent.
- Channel interpretation.
- Missing or unsupported spatial metadata.

No physical pixel spacing may be invented when the file does not provide it.

### 3.4 No Silent Image Transformation

Readers must not silently perform:

- BGR-to-RGB conversion.
- RGB-to-BGR conversion.
- Stain normalization.
- Color normalization.
- Tissue segmentation.
- Model-specific resizing.
- Model-specific intensity normalization.

Reader-level decoding must be distinguished from model-bound preprocessing.

Any unavoidable decoder conversion must be explicit and recorded in metadata and provenance.

### 3.5 Fail Closed on Unsupported Capabilities

Unsupported formats, dimensions, pyramid operations, pixel types and spatial transformations must raise typed errors.

The platform must not return apparently valid image data with fabricated metadata.

## 4. Target Architecture

```text
pathology-ai/
├── domain/
│   ├── image.py
│   ├── geometry.py
│   ├── modality.py
│   └── ...
├── ports/
├── application/
│   ├── ...
│   └── imaging_service.py
├── infrastructure/
│   └── imaging/
│       ├── registry.py
│       ├── detection.py
│       └── ...
├── plugins/
│   ├── synthetic/
│   └── imaging/
│       └── <selected-reader-adapter>/
├── presentation/
│   └── api/
├── pipelines/
├── projects/
├── tests/
└── .claude/
```

The exact module names may be adjusted after repository inspection.

The required architectural boundary is that reader implementations remain replaceable without changes to domain models or scientific pipelines.

No `pathology_ai/` wrapper directory may be introduced.

## 5. Reader Contracts

### 5.1 Metadata Contract

A real image reader must expose, where supported:

- Canonical image identity.
- Detected file format.
- Modality and spatial kind.
- Image dimensions.
- Pyramid levels and downsampling factors.
- Channel count and interpretation.
- Pixel data type.
- Pixel spacing, including units and source.
- Scene/series selection.
- Spatial orientation, where applicable.
- Reader backend identity and version.
- Relevant acquisition metadata.
- Explicitly missing metadata.

Metadata should be obtained without decoding the entire image whenever the backend supports this.

### 5.2 Region Reading Contract

The reader must support bounded region access using an explicit coordinate reference.

A region request must specify:

- Image identity.
- Scene or series when applicable.
- Requested pyramid level or resolution.
- Region origin.
- Region dimensions.
- Coordinate frame.
- Expected channel representation, if applicable.

The response must include:

- Pixel payload or a typed pixel buffer reference.
- Actual returned dimensions.
- Channel order.
- Pixel data type.
- Coordinate reference.
- Spatial transform back to level-zero coordinates.
- Valid-data information, where available.
- Reader and decoding provenance.

Requests must have configurable limits to prevent uncontrolled memory allocation.

### 5.3 Valid-Data and No-Data Semantics

A reader must distinguish among:

1. Valid acquired image pixels.
2. Background pixels represented in the file.
3. Missing or unacquired image regions.
4. Areas outside the image bounds.

These categories must not be silently collapsed into a single background value.

If the source format cannot distinguish them, the reader must report that limitation.

For example, an all-white region is not automatically an unacquired region.

### 5.4 Multi-Scene and Multi-Channel Support

The reader architecture must accommodate:

- Multiple scenes.
- Multiple series.
- Multiple channels.
- Z stacks.
- Time points.

SPEC-003 does not require full execution for every combination.

An adapter must either expose a supported selection mechanism or reject unsupported configurations explicitly.

It must never silently choose an arbitrary scene, channel, Z plane or time point.

## 6. Backend Evaluation

Before choosing the first reader implementation, evaluate relevant candidate libraries.

Candidates may include:

- OpenSlide.
- tifffile.
- pyvips/libvips.
- Bio-Formats or a compatible adapter.
- Other libraries identified through repository and fixture inspection.

The evaluation must document:

- Supported formats.
- Pyramid access.
- Random region access.
- Metadata availability.
- Scene/channel handling.
- Installation and native dependencies.
- Licensing and deployment considerations.
- Memory behavior.
- Known limitations.
- Compatibility with available real fixtures.

Do not claim support based solely on a library's documentation.

Distinguish:

- Documented capability.
- Successfully tested capability.
- Unsupported capability.
- Untested capability.

## 7. Real-Image Implementation Scope

### 7.1 Minimum Executable Reader

Implement at least one real WSI reader adapter that can:

1. Open a supported real image.
2. Read metadata.
3. Enumerate pyramid levels when available.
4. Read a bounded region.
5. Return a spatially referenced pixel payload.
6. Close resources correctly.
7. Reject unsupported operations with typed errors.

The reader must be selected through the platform's registry or capability-resolution mechanism.

Scientific pipelines must not instantiate a concrete backend directly.

### 7.2 Format Priorities

Prioritize evaluation in this order:

1. CZI, because existing Digital Pathology experiments use this format.
2. SVS, as a common pathology WSI format.
3. Pyramidal TIFF / OME-TIFF.

This is an evaluation priority, not a promise that all three formats will be implemented.

A format is considered executable only after an actual compatible fixture passes the acceptance tests.

If CZI is not practical for the first adapter, select a supported alternative and document why.

### 7.3 Fixture Requirements

Use de-identified, synthetic-format or openly licensed image fixtures.

Do not assume the existence of specific patient images or access to private cloud buckets.

Fixtures must be:

- Identifiable by source and format.
- Small enough for repeatable tests where possible.
- Accompanied by expected metadata.
- Suitable for testing bounded region reads.
- Covered by documented redistribution or access rules.

Large fixtures must not be committed to Git without explicit approval.

If no real fixture is available, do not claim that a real format has been validated.

## 8. Integration with Existing Pipelines

Introduce a real-image ingestion path compatible with existing ports and pipeline execution contracts.

The first integration may be a dedicated imaging smoke workflow rather than full MIL inference.

At minimum:

```text
Real WSI file
    ↓
Reader selection
    ↓
Metadata extraction
    ↓
Bounded region read
    ↓
Spatially referenced pixel payload
    ↓
Pipeline-compatible image artifact
```

A full trained-model inference workflow is not required.

The synthetic pipelines from SPEC-001 must remain operational and unchanged in behavior.

Do not use synthetic model outputs to claim meaningful predictions on real pathology images.

## 9. Imaging API Readiness

The existing Imaging API v1 contracts must be reviewed against the reader capabilities.

Verify representation of:

- Canonical image identity.
- Source format and modality.
- Pyramid metadata.
- Region requests.
- Channel order.
- Physical spacing.
- Reader capabilities.
- Unsupported operations.

Do not change an existing API v1 contract incompatibly.

If new information cannot be represented, document a proposed future contract extension rather than silently changing the schema.

No HTTP server or DPA integration is required.

## 10. Provenance and Reproducibility

Each real-image read must record:

- Reader adapter identity.
- Reader backend and version.
- Image identity.
- Source content identity or documented integrity mechanism.
- Scene/series selection.
- Requested coordinates and resolution.
- Actual returned coordinates and dimensions.
- Decoder transformations.
- Pixel representation.
- Metadata limitations.

Do not require a full-file hash for every region read if it would force reading a multi-gigabyte WSI.

Use an explicit integrity strategy and state its guarantees.

## 11. Testing Strategy

### Unit Tests

- Format detection.
- Reader registration and selection.
- Unsupported format handling.
- Coordinate conversion.
- Region bounds.
- Pyramid level validation.
- Resource limits.
- Missing metadata.
- Channel order.
- Resource cleanup.

### Contract Tests

- Reader-port conformance.
- Image descriptor serialization.
- Region request/response structure.
- Coordinate and spacing semantics.
- Error taxonomy.

### Integration Tests

Using real, redistributable or externally provided fixtures:

- Open a real image.
- Read metadata.
- Read at least two distinct regions.
- Verify returned dimensions.
- Verify level-zero coordinate mapping.
- Verify channel semantics.
- Verify bounded memory behavior.
- Verify resource closure.

### Negative Tests

- Unsupported extension or content.
- Corrupted image.
- Invalid scene.
- Invalid pyramid level.
- Negative or out-of-bounds coordinates.
- Oversized region request.
- Missing physical spacing.
- Unsupported Z/time/channel selection.
- Incorrect coordinate frame.
- Missing no-data metadata.
- Decoder failure.

All existing tests must continue to pass.

## 12. Acceptance Criteria

**AC-001 — Reader Architecture**  
A real imaging reader can be registered without changing the domain or pipeline engine.

**AC-002 — Real Image Open**  
At least one supported real WSI format can be opened using an actual test fixture.

**AC-003 — Metadata Extraction**  
The reader returns validated image dimensions, pyramid information where available, channels and spatial metadata without inventing unavailable values.

**AC-004 — Bounded Region Read**  
The reader returns at least two bounded regions with correct dimensions and coordinate references.

**AC-005 — Spatial Consistency**  
Returned regions map correctly to level-zero coordinates, including non-integer downsampling when applicable.

**AC-006 — Channel Integrity**  
Channel order and decoder transformations are explicit and tested.

**AC-007 — No-Data Semantics**  
Missing-data, background and out-of-bounds semantics are distinguished where the format supports them; limitations are explicit otherwise.

**AC-008 — Capability Resolution**  
Unsupported formats, operations, scenes and dimensions are rejected deterministically.

**AC-009 — Resource Safety**  
Oversized requests are rejected, file handles are closed and region reading does not require unconditional full-slide decoding.

**AC-010 — Pipeline Integration**  
A real-image ingestion smoke workflow produces a pipeline-compatible artifact with provenance.

**AC-011 — API Contract Stability**  
Existing Imaging and Analysis API v1 contracts remain backward-compatible.

**AC-012 — Regression Protection**  
All existing synthetic pipeline, architecture, contract and integration tests continue to pass.

**AC-013 — Repository Independence**  
No runtime dependency on `digital-pathology` or `hal-viewer` is introduced.

**AC-014 — Evidence-Based Format Support**  
The implementation report distinguishes tested formats from documented or untested formats and records exact fixture evidence.

**AC-015 — Root-Level Architecture**  
All new code follows the SPEC-002 root-level module structure; `pathology_ai/` is not recreated.

## 13. Out of Scope

SPEC-003 does not include:

- Real trained-model inference.
- Training or evaluation.
- MIL architecture changes.
- CT/MRI volume reading.
- DICOM series reconstruction.
- MRI orientation normalization.
- Production API deployment.
- DPA viewer modifications.
- Automatic tissue segmentation.
- Model-specific stain normalization.
- Clinical validation.
- Whole-slide batch processing at scale.
- Distributed image serving.

## 14. Deliverables

Required implementation deliverables:

- Real WSI reader adapter.
- Reader registration and selection.
- Metadata extraction.
- Bounded region reading.
- Spatial mapping.
- Reader provenance.
- Imaging smoke workflow.
- Unit, contract, integration and negative tests.
- Backend evaluation matrix.
- Fixture inventory.

Required governance files:

```text
.claude/
├── specs/
│   └── 003-medical-imaging-formats-and-readers.md
├── prompts/
│   └── 003-medical-imaging-formats-and-readers.md
├── adr/
│   └── 003-medical-imaging-reader-architecture.md
├── architecture/
│   ├── 003-reader-capability-matrix.md
│   └── 003-imaging-fixture-inventory.md
└── reports/
    └── 003-medical-imaging-formats-and-readers.md
```

## 15. Definition of Done

SPEC-003 is complete when:

1. At least one real WSI fixture is successfully opened.
2. Metadata and bounded region reads are tested.
3. Spatial and channel semantics are verified.
4. Unsupported capabilities fail explicitly.
5. A real-image ingestion workflow executes.
6. Existing synthetic pipelines remain functional.
7. Imaging API contracts remain stable.
8. No sibling repository is modified.
9. The report provides reproducible evidence.

**Status:** Accepted on 2026-10-16. Approved decisions: initial CZI backend `pylibczirw` 6.1.0 (optional extra `pathology-ai[czi]`, LGPL-3.0 approved for research and prototyping; commercial redistribution needs a separate compliance review); optional OpenSlide adapter (`pathology-ai[openslide]`) validated against the CC0 `CMU-1-Small-Region.svs` fixture. Implementation report: `../reports/003-medical-imaging-formats-and-readers.md`.