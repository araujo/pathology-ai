# PROMPT 003 — Implement Medical Imaging Formats & Readers

**Project:** Pathology AI  
**Repository:** `pathology-ai`  
**Related SPEC:** `.claude/specs/003-medical-imaging-formats-and-readers.md`  
**Related ADR:** `.claude/adr/003-medical-imaging-reader-architecture.md`  
**Output Report:** `.claude/reports/003-medical-imaging-formats-and-readers.md`

## Role

You are Claude Code acting as a senior medical imaging software engineer and Python architect.

Your task is to implement the first real medical image reader capability for Pathology AI while preserving its modular, modality-extensible architecture.

Do not assume that CZI, SVS or TIFF are supported until tested.

Do not claim real model inference or scientific validation.

## 1. Required Reading

Read:

1. `.claude/CLAUDE.md`, if present.
2. `.claude/specs/001-platform-foundations.md`
3. `.claude/adr/001-platform-architecture.md`
4. `.claude/reports/001-platform-foundations.md`
5. `.claude/specs/002-repository-structure-and-packaging.md`
6. `.claude/adr/002-root-level-package-layout.md`
7. `.claude/reports/002-repository-structure-and-packaging.md`
8. `.claude/specs/003-medical-imaging-formats-and-readers.md`
9. `.claude/adr/003-medical-imaging-reader-architecture.md`
10. `.claude/architecture/001-platform-boundaries.md`

Treat SPEC-003 as authoritative.

If the SPEC or ADR is still Proposed, complete Phase A only and request approval before writing code.

## 2. Phase A — Repository and Backend Assessment

Before implementation:

- Inspect the current imaging domain contracts.
- Inspect the imaging-related ports.
- Inspect synthetic reader behavior.
- Inspect the existing pipeline engine.
- Inspect Imaging API v1 contracts.
- Inspect current test conventions.
- Inspect packaging and optional dependency patterns.
- Inspect available real image fixtures.
- Evaluate candidate reader libraries.

Produce a backend comparison covering:

- CZI.
- SVS.
- Pyramidal TIFF / OME-TIFF.
- Available libraries and their actual capabilities.
- Pyramid support.
- Region reading.
- Metadata.
- Native dependencies.
- Installation constraints.
- Licensing.
- Test fixtures.

Do not invent benchmark results or compatibility evidence.

If no suitable real fixture is available, identify the blocker and request an approved fixture source before claiming AC-002 or AC-004.

## 3. Phase B — Implement Reader Infrastructure

After approval:

### 3.1 Reader Registration

Implement reader registration and capability resolution using existing ports or minimal compatible extensions.

Do not bind the pipeline engine to a concrete reader library.

### 3.2 Metadata Extraction

Implement metadata extraction for the selected backend.

Preserve:

- Image identity.
- Format.
- Modality.
- Dimensions.
- Pyramid levels.
- Channel order.
- Pixel type.
- Physical spacing.
- Scene/series selection.
- Backend identity.

Missing metadata must remain missing.

### 3.3 Region Reading

Implement bounded region reading.

Validate:

- Coordinate frame.
- Pyramid level.
- Origin.
- Width and height.
- Image bounds.
- Resource limits.
- Scene selection.
- Channel selection.

Return explicit pixel and spatial metadata.

Do not silently change channel order or normalize pixels.

### 3.4 Resource Safety

Use deterministic resource cleanup.

Prevent unbounded full-slide reads.

Raise typed errors for unsupported operations and corrupt files.

### 3.5 Real-Image Smoke Workflow

Implement:

```text
Real image
    ↓
Reader registry
    ↓
Metadata
    ↓
Bounded region
    ↓
Spatial pixel artifact
    ↓
Provenance
```

This workflow must not require a trained model.

### 3.6 Contract Preservation

Keep existing Imaging API v1 and Analysis API v1 schemas backward-compatible.

If a new field is needed but cannot be introduced compatibly, document the future API change rather than silently changing v1.

## 4. Phase C — Tests and Validation

Run:

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest -q -W error
uv build
```

Also:

- Test at least one actual WSI fixture.
- Read at least two distinct regions.
- Verify dimensions and level-zero mapping.
- Verify channel semantics.
- Test invalid requests and corrupted inputs.
- Test cleanup and memory limits.
- Test installed-package imports.
- Verify synthetic pipeline regression.
- Verify API contract stability.
- Verify no changes to sibling repositories.

Record exact commands, fixture identities and observed results.

## 5. Phase D — Report

Create:

`.claude/reports/003-medical-imaging-formats-and-readers.md`

Include:

1. Executive summary.
2. Backend evaluation.
3. Tested format matrix.
4. Reader implementation.
5. Fixture inventory.
6. Metadata evidence.
7. Region-reading evidence.
8. Spatial mapping evidence.
9. Channel and no-data semantics.
10. Provenance evidence.
11. API compatibility.
12. Tests and packaging results.
13. Acceptance criteria AC-001 through AC-015.
14. Limitations and deferred formats.
15. Legacy repository protection.
16. Recommendations for subsequent specifications.

Use PASS, PARTIAL, FAIL or NOT_RUN for every acceptance criterion.

## 6. Hard Constraints

- Do not modify `digital-pathology`.
- Do not modify `hal-viewer`.
- Do not create `pathology_ai/`.
- Do not hardcode CZI or any specific vendor in the domain.
- Do not hardcode colorectal cancer, H&E, KRAS, UNI or MIL into imaging core.
- Do not infer physical spacing from image dimensions.
- Do not silently convert BGR/RGB.
- Do not equate background pixels with missing acquired data.
- Do not download patient images or large datasets without approval.
- Do not commit large medical image files.
- Do not claim untested formats as supported.
- Do not claim scientific or clinical validity.
- Do not introduce a production HTTP server.
- Do not modify API v1 incompatibly.
- Do not commit changes unless requested.

## 7. Completion Criteria

Complete the implementation only when at least one real WSI format is verified with a real fixture, the reader is replaceable, spatial metadata and bounded region access are correct, synthetic regressions pass, and the implementation report records reproducible evidence.

If a required fixture or dependency is unavailable, stop and report the blocker instead of fabricating successful validation.

**Final response must identify the exact reader backend, tested format, fixture, test results, remaining limitations and report path.**