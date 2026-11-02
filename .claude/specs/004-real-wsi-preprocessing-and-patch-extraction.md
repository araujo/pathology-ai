# 004 — Real WSI Preprocessing & Patch Extraction

**Status:** Accepted (approved 2026-10-16)  
**Project:** Pathology AI  
**Priority:** P0  
**Type:** Imaging Pipeline / Preprocessing  
**Depends on:** 001 — Platform Foundations; 002 — Repository Structure & Packaging; 003 — Medical Imaging Formats & Readers  
**Related ADR:** `004-real-wsi-preprocessing-architecture.md`  
**Amended by:** PROMPT-004 cross-format addendum (AC-017, AC-018, §18), incorporated 2026-10-16

## 1. Context

Pathology AI now has an extensible framework, a typed pipeline engine, and real WSI readers implemented in SPEC-003.

The following capabilities are already available:

- Root-level domain, ports, application, infrastructure, plugins and presentation modules.
- Typed pipeline definitions and execution.
- CZI reading through `pylibczirw`.
- SVS reading through OpenSlide.
- Image metadata, physical spacing and stored pyramid levels.
- Explicit native-to-level-zero coordinate transformations.
- Bounded region reading.
- Valid-data reporting and ingestion provenance.
- Synthetic image-processing pipelines.

However, real WSI ingestion is currently separate from `PipelineRunner`. The existing synthetic tissue and patch operations cannot process real image inputs.

SPEC-004 establishes the first executable real-WSI preprocessing and patch-extraction pipeline.

The output must be model-independent, spatially accurate, reproducible and suitable for downstream embedding models.

## 2. Objectives

1. Introduce reader-agnostic real WSI preprocessing operations.
2. Detect candidate tissue regions without hardcoding cancer type, stain, scanner or model.
3. Preserve the distinction between acquired tissue, acquired background and unacquired image regions.
4. Generate deterministic patch plans with explicit coordinate semantics.
5. Extract bounded image patches using SPEC-003 readers.
6. Produce reusable patch artifacts and manifests.
7. Execute the real preprocessing workflow through the existing `PipelineRunner`.
8. Preserve provenance and reproducibility across readers and projects.
9. Support configurable patch sizes, resolutions, overlap and quality filters.
10. Establish a clean handoff contract for future embedding extraction.
11. Preserve all existing synthetic pipeline behavior.
12. Keep the implementation independent from legacy repositories.

## 3. Scope

### In Scope

- WSI tissue candidate detection.
- Configurable preprocessing and quality gates.
- Deterministic patch-grid planning.
- Physical-resolution-aware patch selection.
- Lazy patch extraction.
- Acquired-data coverage checks.
- Model-independent patch representation.
- Patch manifests and storage.
- Real execution through `PipelineRunner`.
- CZI and SVS integration tests.
- Coordinate compatibility assessment with legacy Digital Pathology.
- Technical performance and resource-safety evidence.

### Out of Scope

- Foundation-model inference.
- UNI, Virchow, CONCH or other embedding models.
- MIL training or inference.
- Clinical tumor detection.
- Disease-specific segmentation.
- Stain normalization tied to a particular model.
- Dataset labeling.
- Patient-level prediction.
- DPA integration.
- Distributed preprocessing orchestration.
- CT/MRI volumetric patch execution.
- Modification of the legacy repositories.

## 4. Architectural Principles

### 4.1 Reader Independence

Preprocessing operations consume domain-owned image abstractions and reader ports.

No preprocessing operation may directly import `pylibczirw`, OpenSlide or other format-specific libraries.

### 4.2 Separate Acquisition Coverage and Tissue Detection

Acquisition coverage and tissue detection represent different facts.

Acquisition coverage answers:

> Was this part of the image actually acquired?

Tissue detection answers:

> Does this acquired region appear to contain candidate tissue according to a specified algorithm?

A white pixel may represent acquired glass or background. A black pixel may represent unacquired CZI padding. Neither is sufficient evidence by itself.

### 4.3 Explicit Algorithm Semantics

The first tissue detector may use classical image processing, such as HSV saturation and morphological filtering.

However, the framework must not assume that this detector is universally valid for every modality, stain or tissue type.

The detector must declare its supported input representation and configuration.

### 4.4 Canonical Spatial Reference

Patch coordinates are stored in canonical level-zero pixel space.

Native CZI coordinates, legacy padded coordinates and pyramid-level coordinates must be handled through explicit transformations.

### 4.5 Separate Patch Planning from Pixel Extraction

Planning determines which patches should exist.

Extraction reads and materializes pixels for the selected patches.

These must be independently testable operations.

### 4.6 Model-Independent Output

The patch artifact must not contain implicit assumptions about the future embedding model.

Model-specific preprocessing belongs to the embedding stage, not to generic WSI extraction.

### 4.7 Reproducible Selection

Identical image identity, content integrity, pipeline version and configuration must produce the same patch plan and manifest.

Any stochastic selection must use an explicit seed and record the algorithm and seed.

## 5. Target Architecture

```text
Real WSI ImageSource
        |
        v
ImageBackendRegistry
        |
        v
WSI Metadata + Spatial Geometry
        |
        v
Tissue Detection
        |
        +----> Tissue Candidate Mask
        |
        v
Patch Planning
        |
        +----> PatchPlan
        |
        v
Patch Quality & Coverage Filtering
        |
        v
Patch Extraction
        |
        +----> Patch Pixels / References
        |
        v
PatchManifest + ArtifactStore
        |
        v
Embedding-Ready Dataset
```

All processing stages must be composable through the existing typed pipeline engine.

## 6. Tissue Detection

### 6.1 Detector Port

Define or extend an appropriate domain-owned port for tissue detection.

A detector must accept:

- Spatially referenced image pixels.
- Channel and pixel representation.
- Valid-data mask or validity metadata.
- Detection configuration.
- Spatial transform.

It must return:

- Tissue candidate mask or spatial regions.
- Detector identity and version.
- Parameters.
- Mask resolution and coordinate frame.
- Explicit limitations.

### 6.2 Baseline Detector

Implement at least one classical, deterministic tissue candidate detector for supported color WSI inputs.

Potential operations include:

- Explicit BGR/RGBA to a common working representation.
- HSV or comparable color-space transformation.
- Saturation or optical-intensity thresholding.
- Morphological filtering.
- Minimum-region filtering.

The selected algorithm must be documented and tested.

It must be described as **candidate tissue detection**, not tumor segmentation or clinical tissue classification.

### 6.3 Acquired-Data Awareness

Detection must exclude unacquired regions.

If valid-data information is unavailable, the detector must not silently assume complete acquisition coverage.

The caller must either select an explicitly documented policy for unknown validity or receive a typed unsupported-capability error.

### 6.4 Tissue Mask Provenance

Every mask must record:

- Source image identity.
- Source integrity token.
- Reader and detector versions.
- Scene and level.
- Spatial transform.
- Parameters.
- Input channel representation.
- Valid-data handling.
- Mask digest.

## 7. Patch Planning

### 7.1 Patch Specification

A patch specification must support:

- Patch width and height.
- Patch units: pixels or physical dimensions.
- Target resolution or pixel spacing.
- Grid origin.
- Stride or overlap.
- Minimum tissue fraction.
- Minimum acquired-data fraction.
- Edge policy.
- Maximum patch count or resource budget.
- Deterministic ordering.

Patch geometry must be expressed explicitly.

### 7.2 Resolution Selection

When physical resolution is requested:

- Use source pixel spacing and stored pyramid metadata.
- Select an appropriate supported read level.
- Record the actual effective spacing.
- Reject requests that cannot meet a configured tolerance.

Do not silently substitute a different physical resolution.

When physical spacing is missing, physical-unit requests must fail explicitly unless a separately approved calibration is provided.

### 7.3 Patch Grid

Patch planning must:

- Produce stable ordering.
- Avoid duplicate coordinates.
- Respect level-zero image bounds.
- Handle non-divisible image dimensions.
- Support explicit edge policies.
- Handle non-integer downsampling.
- Remain independent of native CZI offsets.

### 7.4 Selection and Filtering

Patch selection must consider tissue and acquisition coverage separately.

For each candidate patch, record:

- Tissue fraction.
- Acquired-data fraction.
- Inclusion or rejection decision.
- Rejection reasons.
- Requested and effective spatial geometry.

The system must distinguish an empty result caused by no detected tissue from one caused by missing acquisition coverage or an unsupported input.

### 7.5 Large-WSI Safety

Patch planning must not materialize the full-resolution image.

It must support bounded planning memory and configurable patch limits.

A maximum patch count must not be implemented by silently truncating a plan without recording that behavior.

## 8. Patch Extraction

### 8.1 Lazy Extraction

Extract selected patches through the registered real-image backend.

Do not decode the entire WSI as an intermediate step.

### 8.2 Pixel Representation

Every patch must record:

- Patch identity.
- Image identity.
- Scene.
- Canonical level-zero rectangle.
- Requested read level and rectangle.
- Effective physical spacing, if known.
- Width and height.
- Pixel dtype.
- Channel order.
- Valid-data semantics.
- Tissue and acquired-data fractions.
- Extraction provenance.

### 8.3 Image Transformations

Any resampling, channel conversion or intensity transformation must be explicit.

Reader decoding, generic extraction transformations and model-bound preprocessing must be distinguishable.

### 8.4 Resource Limits

Enforce:

- Maximum patch dimensions.
- Maximum pixels per request.
- Maximum number of extracted patches.
- Maximum in-memory batch size.
- Configurable output storage budget.

Errors must be typed and observable.

## 9. Patch Manifest

Introduce a versioned, serializable patch manifest.

The manifest must contain:

- Manifest schema version.
- Image identity and integrity token.
- Pipeline identity and version.
- Tissue detector identity and configuration.
- Patch planning configuration.
- Reader identity and version.
- Spatial coordinate convention.
- Selected patch count.
- Rejected candidate count and reasons.
- Patch identities.
- Patch geometry.
- Patch quality metadata.
- Artifact references.
- Provenance digest.

The manifest must be independent of the future embedding model.

Manifest ordering and serialization must be deterministic.

## 10. Storage and Artifact Semantics

Use existing artifact-store ports where possible.

Do not assume that every patch must remain in memory or be stored as an individual file.

The implementation must distinguish:

- Patch plan metadata.
- Patch pixel payload.
- Persistent artifact identity.
- Artifact storage location.

A content-addressed artifact ID must not be presented as proof of persisted pixel bytes unless the storage backend actually stores those bytes.

The initial implementation may use a local or in-memory storage adapter, provided the limitations are explicit.

## 11. PipelineRunner Integration

The real preprocessing workflow must execute through the existing `PipelineRunner`.

It must use typed operations and artifacts rather than a parallel orchestration engine.

At minimum, implement the logical stages:

1. Real image source resolution.
2. Tissue candidate detection.
3. Patch planning and filtering.
4. Patch extraction.
5. Patch manifest creation.

Stages may be combined when justified by the existing operation model, but the boundaries must remain independently testable.

A real CZI or SVS must be processable through a registered pipeline definition.

The existing synthetic workflows must remain unchanged.

Do not hardcode a specific WSI backend in the runner.

## 12. Legacy Digital Pathology Assessment

Inspect `../digital-pathology` read-only before implementation.

Focus on:

- `reproducibility/wsi_core/WholeSlideImage.py`.
- Tissue segmentation and HSV handling.
- CLAM-style contour extraction and patch generation.
- Existing patch coordinate conventions.
- `scripts/pipeline/czi_coverage.py`.
- `tests/pipeline/czi_fixtures.py`.
- Existing patch manifests and storage formats.
- Patch size, resolution and overlap configuration.
- Existing preprocessing tests.

Record which behaviors are:

- Reused.
- Adapted.
- Rejected.
- Not yet implemented.

### 12.1 Coordinate Compatibility

SPEC-003 identified a difference between legacy padded-image coordinates and the new canonical image coordinates.

For fixture F-CZI-01, the difference is 17 pixels in X.

The patch implementation must:

- Use canonical level-zero coordinates internally.
- Preserve native CZI origin.
- Provide an explicit conversion mechanism for legacy coordinates.
- Test the conversion on F-CZI-01.
- Avoid assuming that the offset is constant across slides.

Do not silently import legacy patch coordinates into the new coordinate frame.

## 13. Validation and Tests

### Unit Tests

- Detector configuration.
- Channel conversion.
- Tissue mask geometry.
- Valid-data exclusion.
- Patch-grid generation.
- Edge policies.
- Stride and overlap.
- Physical-resolution selection.
- Non-integer downsampling.
- Deterministic ordering.
- Filtering and rejection reasons.
- Patch identity stability.
- Manifest serialization.

### Contract Tests

- Detector port.
- Patch specification.
- Patch plan.
- Patch artifact.
- Patch manifest.
- Typed pipeline operations.
- Spatial metadata.

### Integration Tests

- Real CZI preprocessing.
- Real SVS preprocessing.
- At least two patches from each available real fixture.
- End-to-end execution through `PipelineRunner`.
- Pixel correctness against the underlying reader.
- Provenance and manifest verification.
- Legacy coordinate conversion.
- Synthetic pipeline regression.

### Negative Tests

- Missing physical spacing.
- Unsupported channel representation.
- Unknown acquisition coverage.
- No detected tissue.
- All regions rejected.
- Out-of-bounds geometry.
- Unsupported pyramid resolution.
- Excessive patch count.
- Excessive memory budget.
- Invalid manifest.
- Unsupported reader capability.

Real-fixture tests remain opt-in through the environment variables introduced in SPEC-003.

## 14. Acceptance Criteria

**AC-001 — Reader-Agnostic Preprocessing**  
Tissue detection and patch planning use format-independent contracts.

**AC-002 — Real Tissue Candidate Detection**  
At least one deterministic detector runs on a real WSI fixture, with explicit channel and acquisition-validity handling.

**AC-003 — Tissue Mask Geometry**  
Tissue masks have validated spatial transforms and map correctly to level-zero coordinates.

**AC-004 — Deterministic Patch Planning**  
Identical inputs and configuration produce identical patch plans and ordering.

**AC-005 — Resolution Semantics**  
Pixel and physical-resolution requests have explicit, tested behavior, including missing-spacing failures.

**AC-006 — Tissue and Acquisition Filtering**  
Patch decisions independently account for tissue fraction and acquired-data fraction.

**AC-007 — Real Patch Extraction**  
Selected patches are extracted through real readers with correct dimensions, pixels and spatial metadata.

**AC-008 — Resource Safety**  
Planning and extraction enforce bounded memory, request sizes and patch-count budgets.

**AC-009 — Patch Manifest**  
A versioned, deterministic manifest describes selected and rejected patches, artifact references and provenance.

**AC-010 — Artifact Integrity**  
Patch identities and payload persistence semantics are correct and testable.

**AC-011 — PipelineRunner Execution**  
A real WSI preprocessing pipeline executes through the existing typed pipeline engine.

**AC-012 — Legacy Coordinate Compatibility**  
The legacy padded-coordinate transformation is explicit, independently tested and does not assume a universal offset.

**AC-013 — Existing Behavior Preservation**  
All SPEC-001 through SPEC-003 tests and synthetic pipelines continue to pass.

**AC-014 — Repository Independence**  
No runtime dependency or modification is introduced in `digital-pathology` or `hal-viewer`.

**AC-015 — Scientific Semantics**  
Candidate tissue masks and patch-selection scores are not presented as tumor probabilities or clinical predictions.

**AC-016 — Reproducibility Evidence**  
The implementation report includes configuration, fixture identity, patch counts, spatial examples, manifest digests and repeatability checks.

**AC-017 — Cross-Format Reader Conformance**  
The real WSI preprocessing pipeline must be independent of storage format and backend identity. CZI and SVS adapters must satisfy the same applicable preprocessing and patch-extraction contracts. Backend limitations must be declared and tested, rather than handled through implicit format-specific behavior.

**AC-018 — Capability-Gated Execution**  
Before executing preprocessing, the system must verify that the reader provides the capabilities required by the configured pipeline. Unsupported or unknown capabilities must produce explicit decisions or typed errors, never silent assumptions.

## 15. Implementation Phases

### Phase A — Assessment and Design

- Inspect the current platform and SPEC-003 readers.
- Inspect the legacy preprocessing and patching implementation read-only.
- Identify reusable operations and contracts.
- Establish baseline tests.
- Propose detector, planner, extractor and manifest design.
- Identify fixture limitations.
- Request approval before implementation if SPEC/ADR remain Proposed.

### Phase B — Domain and Application Contracts

- Implement detector and patch-domain contracts.
- Implement patch planning and manifest structures.
- Define provenance and artifact semantics.

### Phase C — Real Preprocessing and Extraction

- Implement baseline tissue detection.
- Implement acquisition-aware patch filtering.
- Implement bounded extraction through existing readers.
- Implement legacy coordinate conversion.

### Phase D — Pipeline Integration

- Register typed operations.
- Define real-WSI pipeline YAML.
- Execute through `PipelineRunner`.
- Preserve synthetic pipeline behavior.

### Phase E — Verification and Report

- Run all checks.
- Validate real fixtures.
- Verify deterministic manifests and patch identities.
- Compare synthetic baseline.
- Verify wheel packaging.
- Verify sibling repositories remain unchanged.
- Produce the implementation report.

## 16. Deliverables

```text
.claude/
├── specs/
│   └── 004-real-wsi-preprocessing-and-patch-extraction.md
├── prompts/
│   └── 004-real-wsi-preprocessing-and-patch-extraction.md
├── adr/
│   └── 004-real-wsi-preprocessing-architecture.md
├── architecture/
│   ├── 004-preprocessing-pipeline.md
│   ├── 004-legacy-patch-compatibility.md
│   └── 004-reader-conformance-matrix.md
└── reports/
    └── 004-real-wsi-preprocessing-and-patch-extraction.md
```

## 17. Definition of Done

SPEC-004 is complete only when:

1. A real WSI passes through the existing pipeline engine.
2. Candidate tissue regions are detected with explicit semantics.
3. A deterministic patch plan is generated.
4. Selected patches are extracted from the real image.
5. A versioned patch manifest is produced.
6. Coordinates and provenance are preserved.
7. Resource budgets are enforced.
8. Existing synthetic behavior remains unchanged.
9. All acceptance criteria have reproducible evidence.
10. No sibling repository is modified.
11. The same preprocessing operations pass a shared reader-conformance suite on every available backend, with PASS, FAIL, UNSUPPORTED and NOT_TESTED kept distinct (AC-017).
12. Capability gating rejects or explicitly decides every unsupported or unknown capability before preprocessing starts (AC-018).

## 18. Cross-Format Compatibility (Addendum)

Incorporated from the PROMPT-004 cross-format addendum. The full addendum text is preserved in PROMPT-004 §10.

### 18.1 Format-Neutral Pipeline Contract

The preprocessing pipeline consumes canonical image capabilities, not file extensions or backend names. The following properties must be explicitly represented or validated:

1. Spatial dimensionality.
2. Scene and plane selection.
3. Canonical coordinate frame.
4. Image dimensions and pyramid levels.
5. Pixel spacing and spacing availability.
6. Channel order, dtype and pixel representation.
7. Region-reading capability.
8. Acquisition-validity semantics.
9. Supported read levels and resampling behavior.
10. Reader-specific limitations.

No tissue detector or patch planner may branch directly on `.czi`, `.svs`, `pylibczirw` or `openslide`. Backend-specific behavior remains inside the reader adapter or a clearly defined capability-specific abstraction.

### 18.2 Capability Negotiation

Before preprocessing begins, validate whether the selected reader satisfies the pipeline's requirements, for example:

- A detector requiring RGB-compatible color data rejects unsupported channel representations or uses an explicit conversion.
- A physical-size patch request rejects missing physical spacing.
- A pipeline requiring reliable acquisition coverage does not silently treat `NOT_REPORTED` as fully acquired.
- A requested scene or plane is supported explicitly.
- A requested pyramid resolution is available or satisfies a declared resampling policy.

Use typed errors with actionable diagnostic information. Do not create a parallel capability registry if the SPEC-003 contracts can be extended compatibly.

### 18.3 Reader Conformance Tests

A reusable reader-conformance suite verifies, where supported: content-based reader resolution, metadata consistency, canonical coordinate mapping, bounded region reads, correct patch dimensions, channel-order integrity, physical-spacing behavior, valid-data reporting, resource-limit enforcement, deterministic patch planning, patch provenance and handle cleanup.

Results distinguish `PASS` (tested and verified), `FAIL` (tested and incorrect), `UNSUPPORTED` (explicitly outside the backend's capabilities) and `NOT_TESTED` (no suitable fixture or test execution). Unsupported capability and missing evidence are never reported as successful format support.

### 18.4 Cross-Backend Equivalence

Where equivalent image content is available through different readers or formats, verify that canonical patch coordinates are equivalent, requested physical regions correspond within declared tolerances, patch-grid ordering is deterministic, tissue and acquired-data thresholds have consistent semantics, and patch manifests use the same schema. Record differences caused by decoding, resampling, channel representation or compression.

Byte-identical pixels across lossy formats or different decoders are not required. Use exact comparisons for geometry and deterministic metadata, and documented tolerances for numerical image comparisons. Cross-backend pixel equivalence is not claimed merely because two unrelated fixtures pass the same tests.

### 18.5 Fixtures and Future Formats

Use the opt-in fixtures `PATHOLOGY_AI_WSI_FIXTURE_CZI` and `PATHOLOGY_AI_WSI_FIXTURE_SVS`, retaining integrity checks and explicit skip reasons. The SVS fixture has a single resolution level and does not validate multilevel SVS behavior. Generated fixtures may be used for coordinate transforms, missing spacing, channel representations and error behavior.

Document the expected integration path for multilevel SVS, pyramidal TIFF, OME-TIFF, NDPI, OME-Zarr and DICOM WSI (adapter responsibilities, likely metadata differences, minimum fixture requirements) in `../architecture/004-reader-conformance-matrix.md`. Do not implement new readers, add dependencies for these formats, or claim support for untested formats.

## 19. Status

**Status:** Accepted on 2026-10-16, after the Phase A assessment (`../reports/004-real-wsi-preprocessing-and-patch-extraction.md`). Approved implementation decisions R1–R8 are recorded in ADR-004 §6.
