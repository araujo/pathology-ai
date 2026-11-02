# PROMPT 004 — Implement Real WSI Preprocessing & Patch Extraction

**Repository:** `pathology-ai`  
**SPEC:** `.claude/specs/004-real-wsi-preprocessing-and-patch-extraction.md`  
**ADR:** `.claude/adr/004-real-wsi-preprocessing-architecture.md`  
**Report:** `.claude/reports/004-real-wsi-preprocessing-and-patch-extraction.md`  
**Addendum:** cross-format compatibility and reader conformance, integrated verbatim in §10 (2026-10-16)

## Role

You are Claude Code acting as a senior medical imaging engineer, computer vision architect and Python software engineer.

Implement real WSI preprocessing and patch extraction for Pathology AI.

Use the existing architecture, ports, pipeline engine and real-image readers.

Do not build a parallel framework.

## 1. Required Reading

Read the relevant SPECs, ADRs and implementation reports for 001, 002 and 003.

Read:

- `.claude/specs/004-real-wsi-preprocessing-and-patch-extraction.md`
- `.claude/adr/004-real-wsi-preprocessing-architecture.md`
- `.claude/architecture/001-platform-boundaries.md`
- `.claude/architecture/003-reader-capability-matrix.md`
- `.claude/architecture/003-imaging-fixture-inventory.md`

Read `.claude/CLAUDE.md` if present.

If SPEC-004 or ADR-004 is Proposed, complete Phase A only. Do not implement until approval is granted.

## 2. Phase A — Assessment

Inspect:

1. Existing `PipelineRunner`, typed operations and artifact store.
2. Existing imaging service and real reader ports.
3. Existing spatial geometry and validity contracts.
4. Existing synthetic tissue and patch operations.
5. Existing provenance and fingerprint contracts.
6. Existing API v1 schemas.
7. Existing test and packaging conventions.
8. Available real WSI fixtures.

Inspect `../digital-pathology` read-only.

Pay special attention to:

- `reproducibility/wsi_core/WholeSlideImage.py`.
- Existing HSV-based tissue detection.
- CLAM-style tissue segmentation.
- Patch-grid and contour strategies.
- `scripts/pipeline/czi_coverage.py`.
- Generated CZI fixtures.
- Patch-coordinate conventions.
- Patch metadata and storage.
- Existing scientific preprocessing parameters.

Produce a concise implementation plan that identifies:

- Proposed domain contracts.
- Proposed detector interface.
- Proposed patch planner.
- Proposed extractor.
- Proposed artifact and manifest storage semantics.
- Proposed typed pipeline operations.
- Proposed YAML pipeline definition.
- Legacy coordinate bridge.
- Test fixtures and expected coverage.
- Risks and unresolved decisions.

Record baseline tests and synthetic demo output.

Do not write implementation code in Phase A.

## 3. Phase B — Contracts

After approval:

- Implement reader-independent tissue-mask contracts.
- Implement patch specifications and plans.
- Implement patch metadata and manifests.
- Define explicit validity and quality semantics.
- Preserve level-zero coordinate conventions.
- Preserve existing provenance behavior.

## 4. Phase C — Preprocessing

Implement:

- A deterministic baseline tissue detector.
- Explicit channel conversion where required.
- Acquired-data-aware tissue masking.
- Configurable morphological operations where justified.
- Tissue-fraction calculation.
- Acquired-data-fraction calculation.
- Deterministic patch planning.
- Bounded lazy patch extraction.
- Legacy coordinate translation.

Avoid hardcoding stain, cancer, model or scanner assumptions into domain contracts.

## 5. Phase D — Pipeline Integration

Register real preprocessing operations in the existing pipeline infrastructure.

Create a real-WSI pipeline definition.

Ensure:

- Execution through `PipelineRunner`.
- Typed inputs and outputs.
- Explicit operation versions.
- Deterministic configuration.
- No backend-specific imports in the runner.
- No changes to synthetic pipeline behavior.
- No trained-model dependency.

## 6. Phase E — Verification

Run:

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest -q -W error -rs
uv build
```

With fixture variables configured, execute real CZI and SVS integration tests.

Validate:

- Detector behavior.
- Coordinate mapping.
- Patch planning.
- Patch extraction.
- Artifact identities.
- Manifest serialization.
- Repeatability.
- Memory and patch-count budgets.
- Installed-wheel behavior.
- Synthetic regression.
- Legacy repository status.

## 7. Reporting

Create:

`.claude/reports/004-real-wsi-preprocessing-and-patch-extraction.md`

Include:

1. Executive outcome.
2. Architecture and changed files.
3. Legacy implementation assessment.
4. Tissue detection design and limitations.
5. Patch planning and filtering.
6. Coordinate compatibility.
7. Extraction and storage.
8. PipelineRunner execution evidence.
9. Real CZI and SVS fixture evidence.
10. Determinism and provenance.
11. Resource limits.
12. Tests and packaging.
13. AC-001 through AC-016.
14. Known limitations.
15. Recommendations for SPEC-005.

For each acceptance criterion, report PASS, PARTIAL, FAIL or NOT_RUN with evidence.

## 8. Hard Constraints

- Do not modify `digital-pathology`.
- Do not modify `hal-viewer`.
- Do not create `pathology_ai/`.
- Do not introduce reader-specific dependencies into the core.
- Do not duplicate the pipeline engine.
- Do not silently convert BGR/RGB.
- Do not equate acquisition validity with tissue.
- Do not assume universal pixel spacing.
- Do not use legacy patch coordinates without explicit transformation.
- Do not load entire level-zero WSI into memory.
- Do not introduce a trained model.
- Do not claim tumor detection or clinical validity.
- Do not modify API v1 incompatibly.
- Do not commit changes unless explicitly requested.

## 9. Completion

The implementation is complete only when a real WSI passes through `PipelineRunner`, generates a deterministic patch manifest and selected patch artifacts, and all acceptance criteria are supported by reproducible evidence.

If a material architectural conflict is found, stop and request clarification rather than silently changing the approved architecture.

**Final response must identify the executed real pipeline, fixture results, patch counts, manifest identities, test results, limitations and report path.**

## 10. Addendum — Cross-Format Compatibility & Reader Conformance

The addendum below is preserved verbatim (headings demoted two levels). Its SPEC and ADR updates (§9, §10, §12) have been applied: AC-017/AC-018 and §18 in SPEC-004, D12–D15 in ADR-004, and `.claude/architecture/004-reader-conformance-matrix.md`.

### PROMPT-004 Addendum — Cross-Format Compatibility & Reader Conformance

**Repository:** `pathology-ai`  
**SPEC:** `.claude/specs/004-real-wsi-preprocessing-and-patch-extraction.md`  
**ADR:** `.claude/adr/004-real-wsi-preprocessing-architecture.md`  
**Purpose:** Ensure that SPEC-004 preprocessing and patch extraction remain independent of WSI storage formats and reader backends.

#### 1. Objective

Extend the SPEC-004 design and acceptance criteria to establish verifiable cross-format compatibility.

The same preprocessing and patch-planning operations must work across different WSI readers whenever those readers expose equivalent capabilities and image semantics.

This is an architectural conformance requirement, not a request to implement every imaging format.

#### 2. Required Assessment

Before changing the design, inspect:

- SPEC-003 reader ports and backend capability declarations.
- `ImageBackendRegistry` and `ImagingService`.
- `ImageDescription`, `PixelRegion` and `ValidDataReport`.
- Native-to-level-zero coordinate transformations.
- Stored pyramid levels and effective spacing.
- Channel order and decoder transformations.
- Acquisition-validity semantics.
- SPEC-004 detector, patch planner and extraction contracts.
- Existing real CZI and SVS fixtures.

Inspect `../digital-pathology` read-only for relevant preprocessing and coordinate-handling lessons.

Identify any assumptions that would make SPEC-004 work only with CZI or only with OpenSlide.

#### 3. Format-Neutral Pipeline Contract

The preprocessing pipeline must consume canonical image capabilities, not file extensions or backend names.

The following properties must be explicitly represented or validated:

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

No tissue detector or patch planner may branch directly on `.czi`, `.svs`, `pylibczirw` or `openslide`.

Backend-specific behavior must remain inside the reader adapter or a clearly defined capability-specific abstraction.

#### 4. Capability Negotiation

Before preprocessing begins, validate whether the selected reader satisfies the pipeline's requirements.

Examples:

- A detector requiring RGB-compatible color data must reject unsupported channel representations or use an explicit conversion.
- A physical-size patch request must reject missing physical spacing.
- A pipeline requiring reliable acquisition coverage must not silently treat `NOT_REPORTED` as fully acquired.
- A requested scene or plane must be supported explicitly.
- A requested pyramid resolution must be available or satisfy a declared resampling policy.

Use typed errors with actionable diagnostic information.

Do not create a parallel capability registry if the SPEC-003 contracts can be extended compatibly.

#### 5. Reader Conformance Tests

Create a reusable reader-conformance test suite.

It must verify, where supported:

- Content-based reader resolution.
- Metadata consistency.
- Canonical coordinate mapping.
- Bounded region reads.
- Correct patch dimensions.
- Channel-order integrity.
- Physical-spacing behavior.
- Valid-data reporting.
- Resource-limit enforcement.
- Deterministic patch planning.
- Patch provenance.
- Handle cleanup.

The suite must distinguish:

- `PASS`: tested and verified.
- `FAIL`: tested and incorrect.
- `UNSUPPORTED`: explicitly outside the backend's capabilities.
- `NOT_TESTED`: no suitable fixture or test execution.

Unsupported capability and missing evidence must never be reported as successful format support.

#### 6. Cross-Backend Equivalence

Where equivalent image content is available through different readers or formats, verify that:

- Canonical patch coordinates are equivalent.
- Requested physical regions correspond to the same image area within declared tolerances.
- Patch-grid ordering is deterministic.
- Tissue and acquired-data thresholds have consistent semantics.
- Patch manifests use the same schema.
- Differences caused by decoding, resampling, channel representation or compression are recorded.

Do not require byte-identical pixels across lossy formats or different decoders.

Use exact comparisons for geometry and deterministic metadata where possible, and documented tolerances for numerical image comparisons.

Do not claim cross-backend pixel equivalence merely because two unrelated fixtures pass the same tests.

#### 7. Existing Fixtures

Use the existing opt-in fixtures:

- `PATHOLOGY_AI_WSI_FIXTURE_CZI`
- `PATHOLOGY_AI_WSI_FIXTURE_SVS`

Retain fixture integrity checks and explicit skip reasons.

The existing SVS fixture has only one resolution level. It does not validate multilevel SVS behavior.

Do not expand claims beyond the fixture evidence.

Generated fixtures may be used to test coordinate transforms, missing spacing, channel representations and error behavior.

#### 8. Future Formats

Document the expected integration path for:

- Multilevel SVS.
- Pyramidal TIFF.
- OME-TIFF.
- NDPI.
- OME-Zarr.
- DICOM WSI.

For each, identify the expected adapter responsibilities, likely metadata differences and minimum fixture requirements.

Do not implement new readers in this addendum.

Do not add dependencies for these formats.

Do not claim support for untested formats.

#### 9. SPEC-004 Updates

Add an acceptance criterion:

**AC-017 — Cross-Format Reader Conformance**

The real WSI preprocessing pipeline must be independent of storage format and backend identity.

CZI and SVS adapters must satisfy the same applicable preprocessing and patch-extraction contracts.

Backend limitations must be declared and tested, rather than handled through implicit format-specific behavior.

Add another acceptance criterion:

**AC-018 — Capability-Gated Execution**

Before executing preprocessing, the system must verify that the reader provides the capabilities required by the configured pipeline.

Unsupported or unknown capabilities must produce explicit decisions or typed errors, never silent assumptions.

Update the Definition of Done accordingly.

#### 10. ADR-004 Updates

Extend ADR-004 with the following decisions:

**D12 — Cross-Format Conformance**

Every WSI adapter must satisfy a shared set of reader and preprocessing contracts for the capabilities it claims to support.

**D13 — Capability-Gated Preprocessing**

Pipeline compatibility is determined by declared and verified capabilities, not file extension or backend identity.

**D14 — Evidence-Based Format Support**

Support claims are scoped to tested formats, fixtures, operations and backend versions.

**D15 — Semantic Equivalence, Not Universal Pixel Equality**

Cross-backend equivalence requires consistent geometry, metadata, quality decisions and provenance semantics. Pixel equality is required only where technically justified.

Preserve the existing ADR decisions and numbering.

#### 11. Implementation Boundaries

- Do not modify `digital-pathology` or `hal-viewer`.
- Do not create a new ADR unless an actual architectural conflict is identified.
- Do not create a second pipeline engine.
- Do not create a second reader registry unnecessarily.
- Do not hardcode CZI or SVS in preprocessing domain logic.
- Do not introduce unsupported-format dependencies.
- Do not weaken SPEC-003 validation.
- Do not change Imaging API v1 incompatibly.
- Do not make Git commits unless requested.

#### 12. Deliverables

Update:

- `.claude/specs/004-real-wsi-preprocessing-and-patch-extraction.md`
- `.claude/adr/004-real-wsi-preprocessing-architecture.md`
- `.claude/prompts/004-real-wsi-preprocessing-and-patch-extraction.md`

Create or extend:

- `.claude/architecture/004-reader-conformance-matrix.md`

In the implementation report, include:

- Reader conformance results by backend.
- Capability-gating tests.
- Cross-format assumptions found and eliminated.
- Unsupported and untested operations.
- AC-017 and AC-018 evidence.
- Remaining fixture coverage gaps.

#### 13. Execution Rule

If SPEC-004 and ADR-004 are still Proposed, incorporate these requirements into the Phase A assessment and proposed design only.

Do not begin implementation until owner approval.

If implementation has already been approved, incorporate the requirements without reopening unrelated architectural decisions.

If a material conflict with SPEC-003 or ADR-003 is discovered, report it and request a decision before changing those accepted contracts.