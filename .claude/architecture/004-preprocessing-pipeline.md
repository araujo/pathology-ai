# 004 — Real-WSI Preprocessing Pipeline

**Status:** Implemented for SPEC-004 (Accepted 2026-10-16). Last verified 2026-10-18.
**Related:** `../specs/004-real-wsi-preprocessing-and-patch-extraction.md`, `../adr/004-real-wsi-preprocessing-architecture.md`, `004-legacy-patch-compatibility.md`, `004-reader-conformance-matrix.md`, `../reports/004-real-wsi-preprocessing-and-patch-extraction.md`

## 1. Flow

```text
ImageSource (path + declared modality)
  └─ PreprocessingService.run                       application/preprocessing_service.py
       └─ ImagingService.open  (content-based backend resolution, integrity, guaranteed close)
            └─ PipelineRunner.run(ExecutionRequest(source=<OpenedImage>, config_overrides=...))
                 ├─ validate: binding + per-run override check (existing keys, same type)
                 ├─ capability decision: no model → SUPPORTED_UNVALIDATED, research use
                 ├─ executability: operations, reader for format, opened source present
                 ├─ gate (AC-018): every CapabilityGated op's requirements vs the opened image
                 └─ stages (topological order)
                      tissue  tissue.detect@1   → mask.tissue_2d   (TissueMask)
                      plan    patch.plan@1      → patch_plan       (PatchPlan)
                      extract patch.extract@1   → patch_manifest   (PatchManifestPayload)
                                                   + image.region_2d per selected patch
```

The only pipeline definition is `pipelines/wsi-tissue-patch-extraction.v1.yaml`. The same file serves every backend; the per-image stored read level is resolved, never assumed.

## 2. Layers and files

| Layer | File | Content |
|---|---|---|
| domain | `capabilities.py` | `PreprocessingRequirements`, `CapabilityCheck` (satisfied / decided_by_policy / unsupported / unknown), `ValidityPolicy`, `require_capabilities` |
| domain | `tissue.py` | `MaskGrid` (level-zero geometry of a mask raster), `TissueMask` (tissue + acquired, provenance, invariants) |
| domain | `patching.py` | `PatchSpec`, `resolve_patch_geometry`, `plan_candidates`, `decide`, `apply_selection`, `PatchPlan` |
| domain | `manifest.py` | `PatchRecord`, `PatchManifest` (schema `pathology-ai.patch-manifest/1`), `PixelStorage` |
| domain | `frames.py` | `FrameTranslation`, `padded_origin_translation`, `NATIVE_PADDED_ORIGIN` |
| domain | `errors.py`, `kinds.py`, `task.py` | new typed errors; `patch_plan`, `patch_manifest` kinds; `TaskType.PREPROCESSING` / `OutputKind.PATCH_MANIFEST` |
| ports | `__init__.py` | `StageContext.source/pipeline`, `CapabilityGated`, `TissueDetector`, `DetectorInput`, `DetectorResult`, `PatchManifestPayload` |
| application | `runner.py` | optional `source` and `config_overrides`, `gate()`, manifest aux artifacts (additive) |
| application | `preprocessing_service.py` | `WSI_PREPROCESSING_TASK`, `PreprocessingService` |
| plugins | `preprocessing/{arrays,detection,operations}.py` | NumPy-only implementation |
| plugins | `imaging/czi/reader.py` | additive `native.padded_origin` acquisition fact |
| presentation | `api/preprocessing_v1.py`, `contracts/preprocessing/v1/patch-manifest.schema.json` | canonical manifest JSON, verified parser, JSON Schema |
| composition | `bootstrap.py`, `cli.py` | wiring; `pathology-ai-preprocess` |

No core module names a format or backend (architecture tests). `plugins/preprocessing` imports only NumPy and first-party domain/ports.

## 3. Tissue detection

- **Mask source:** the finest stored level with at most `max_mask_pixels` (default 2²⁰) pixels, read in tiles of at most `tile_px`² (default 2048²). If no stored level is that small, the coarsest level is read in tiles and block-averaged by the smallest integer factor that fits. Only acquired pixels are averaged; a cell counts as acquired when at least 50% of it is acquired. Decoding is capped by `max_decoded_pixels` (default 64 Mpx), otherwise `PREPROCESSING_BUDGET_EXCEEDED`.
- **Detector `saturation_otsu` v1:**
  1. Explicit RGB view: `bgr→rgb` reverses the channel axis; `rgba→rgb` drops alpha, which the reader already reported as validity.
  2. Saturation = (max − min)·255 // max.
  3. Otsu threshold over acquired cells only, floored at `min_threshold` (default 8).
  4. Optional opening/closing (`open_radius`, `close_radius`, default 0).
  5. `tissue &= acquired`.
- **Accepted inputs:** RGB, BGR and RGBA uint8. Grayscale and multichannel are rejected by the gate.
- **Recorded on the mask:** detector component, parameters, derived threshold, channel view, grid, validity semantics and policy, area-weighted tissue and acquired fractions, SHA-256 mask digest, capability checks, limitations, and decoded pixel count.

## 4. Planning

- **Read level:** exactly one of `level` or `target_spacing_um` must be set (sentinels `-1` and `0.0` mean unset). A target spacing picks the closest stored level within `spacing_tolerance`; otherwise `UNSUPPORTED_RESOLUTION`. Resampling is never performed.
- **Units:** pixel units need no spacing when the level is explicit. Micrometre units and target spacings require spacing; the gate rejects images without it.
- **Grid:** row-major, starting at `grid_origin`, stepping by stride, never padded.
  - `inside_only` drops partial patches.
  - `align_to_edge` adds one flush patch per axis, which overlaps its neighbour.
  - The candidate count is checked against `max_candidates` before any candidate is materialised.
- **Fractions:** tissue and acquired fractions are exact area-weighted means over the mask grid. The two thresholds are judged independently, and both rejection reasons are recorded.
- **Budget:** `max_patches` with `selection`:
  - `fail_closed` (default) raises `PATCH_BUDGET_EXCEEDED`;
  - `row_major_first` and `uniform_stride` keep a deterministic subset and mark the rest `not_selected_budget`, recorded in the plan's selection note.
- **Empty results** raise a specific error: `NO_ACQUIRED_DATA`, `NO_TISSUE_DETECTED`, or `NO_PATCHES_SELECTED` (every candidate was rejected).

## 5. Extraction and storage

- **Reads:** one bounded `read_region` per selected patch, in plan order and in batches. Budgets are checked before the first read: `max_patches`, `max_batch_bytes`, `max_output_bytes`.
- **Pixels** stay in the reader's native representation. No conversion, resizing or normalisation is applied; that is model-bound and belongs to a later stage.
- **Each `PatchRecord`** carries:
  - patch id and plan index;
  - level-zero and read-level rectangles;
  - effective spacing;
  - dtype, channel order and samples per pixel;
  - validity semantics;
  - planned tissue and acquired fractions, plus the acquired fraction actually read;
  - the read-provenance digest and decoder transformations;
  - the artifact id.
- **Pixel artifact identity:** `ArtifactIdentity.derive(image.region_2d, {"region": provenance}, (image artifact,))`. This is the same rule as `ImagingService.ingest`.
- **Storage:** `PixelStorage.IN_MEMORY` (process-local store) or `NOT_MATERIALIZED` (`materialize_pixels: false`). The manifest states that artifact ids do not prove persistence.

## 6. Identity and determinism

- **Runner artifacts** (mask, plan, manifest, patches) derive from stage, operation, effective config (overrides included), seed and lineage. Timestamps never enter them.
- **`plan_digest`** = SHA-256 of the canonical plan.
- **`manifest_digest`** = SHA-256 of the canonical manifest.
- **`provenance_digest`** = SHA-256 of what produced the manifest: image, integrity, scene, pipeline, reader, backend, detector and its parameters, mask, planner, extractor and plan.
- **Overrides** enter the run fingerprint only when present, so runs without overrides keep their previous fingerprint.
