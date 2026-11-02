# 004 — Reader Conformance Matrix (SPEC-004 AC-017 / AC-018)

**Status:** Results as of 2026-10-18 (macOS arm64, Python 3.12.13, pylibczirw 6.1.0, openslide-python 1.4.6 with OpenSlide 4.0.1).
**Suite:** `tests/contract/test_reader_conformance.py` (shared checks), plus capability gating in `tests/integration/test_preprocessing_pipeline.py` and `tests/unit/test_preprocessing_contracts.py`.
**Related:** `003-reader-capability-matrix.md`, `003-imaging-fixture-inventory.md`, `004-preprocessing-pipeline.md`

## Labels

- **PASS**: tested and verified.
- **FAIL**: tested and incorrect. There are none.
- **UNSUPPORTED**: explicitly outside the subject's capabilities. The test skips with a reason starting `UNSUPPORTED:`, or asserts a typed rejection.
- **NOT_TESTED**: no suitable fixture or test execution.

Unsupported capabilities and missing evidence are not counted as format support.

## Subjects

| Subject | What it is |
|---|---|
| `gen-czi` | CZI written by pylibCZIrw at test time: 4×3 tiles of 256 px, lossless, native origin (-300, 200), one missing tile, 0.5 µm/px. A synthetic format fixture, not a real WSI. |
| `f-czi-01` | Real Zeiss CZI (SPEC-003 inventory). JPEG XR, Bgr24, 7 stored levels. |
| `f-svs-01` | Real Aperio SVS (SPEC-003 inventory). JPEG, **single level**. |
| `array-rgba` | In-memory reference implementation of the `OpenedImage` port, with the same content as `gen-czi` in RGBA. Not a format. |

## 1. Shared conformance checks

| Check | gen-czi | f-czi-01 | f-svs-01 | array-rgba |
|---|---|---|---|---|
| Content-based reader resolution (renamed file with an unknown extension) | PASS | PASS | PASS | UNSUPPORTED (no file) |
| Metadata consistency (level 0, capability channel order, spacing vs missing metadata, validity reported, single plane) | PASS | PASS | PASS | PASS |
| Canonical coordinate mapping (coarsest level → level 0) | PASS | PASS (ds 64) | PASS (ds 1) | PASS |
| Out-of-bounds rejected; `ReadLimits` enforced | PASS (OOB; limit covered by SPEC-003 spy tests) | PASS (limit + OOB) | PASS (OOB; image < limit) | PASS (OOB) |
| Valid-data report with a mask of the region's shape | PASS (coverage) | PASS (coverage) | PASS (alpha) | PASS (alpha) |
| Pipeline: patch dimensions = (h, w, samples) | PASS (4 patches) | PASS (134) | PASS (25) | PASS (4) |
| Channel-order integrity (record = description = region) | PASS (BGR) | PASS (BGR) | PASS (RGBA) | PASS (RGBA) |
| Patch provenance (read digest, lineage to the image artifact, level-0 = mapped level rect) | PASS | PASS | PASS | PASS |
| Physical spacing semantics (effective = spacing × ds; error within tolerance) | PASS (level 0, 0%) | PASS (level 2, 11.0%) | PASS (level 0, 0.2%) | PASS |
| Deterministic planning (mask, plan and manifest digests equal across runs) | PASS | PASS | PASS | PASS |
| Handle cleanup (fd count back to baseline after a pipeline run) | PASS | PASS | PASS | UNSUPPORTED (no handle) |

Counts: **35 passed, 2 skipped (UNSUPPORTED)** with both fixture variables set. Without them, the f-czi-01 and f-svs-01 columns are **NOT_TESTED**, with explicit skip reasons.

## 2. Capability gating (AC-018)

Every case raises `PREPROCESSING_CAPABILITY` before any pixel is read, or records an explicit decision.

| Situation | Result | Evidence |
|---|---|---|
| Grayscale or multichannel input | `channel_representation unsupported`; zero region reads | `test_unsupported_channel_representation_is_rejected_before_any_read` |
| Validity `NOT_REPORTED` with `require_reported` (default) | `acquisition_validity unsupported`; zero reads | `test_unknown_validity_requires_an_explicit_policy` |
| Validity `NOT_REPORTED` with explicit `assume_acquired` | `decided_by_policy`, recorded in the manifest; acquired fractions `None`; plan states that the filter was not applied | same test |
| Missing spacing with `target_spacing_um` or micrometre units | `physical_spacing unsupported` | `test_missing_spacing_fails_for_physical_requests_but_not_for_explicit_levels` |
| Missing spacing with explicit `level` and pixel units | runs; effective spacing `None`; limitation recorded | same test |
| Z or T > 1 | `single_plane unsupported` | `test_multi_plane_and_missing_region_support_are_gated` |
| Reader without region reads | `region_reads unsupported` | same test |
| No stored level within tolerance | `UNSUPPORTED_RESOLUTION` (no resampling) | `test_unsupported_resolution_is_rejected`, unit tests |
| Gated pipeline without an opened source | `EXECUTION_UNSUPPORTED` (`source:opened_image`) | `test_gated_pipeline_without_source_is_not_executable` |
| Opened source does not match the requested image | `EXECUTION_UNSUPPORTED` | `test_closed_or_mismatched_source_is_rejected` |
| Several scenes without an explicit choice | rejected at open by the reader (SPEC-003, unchanged); the gate records the opened scene | SPEC-003 tests |

## 3. Cross-backend equivalence (D15)

`test_same_content_is_equivalent_across_backends` runs identical lossless content through `gen-czi` (BGR, acquisition coverage, negative native origin) and `array-rgba` (RGBA, transparency). Result: **PASS**. The following are identical:

- tissue and acquired masks, mask digest and derived threshold;
- every candidate, decision, reason and fraction;
- counts (4 selected);
- patch pixels after the explicit `bgr→rgb` and `rgba→rgb` views.

The recorded differences are channel order, validity semantics and reader identity.

Exact pixel equality is justified only because both paths are lossless. **F-CZI-01 and F-SVS-01 are unrelated slides.** Between them, only schema, ordering, gating and threshold semantics are compared; no pixel equivalence is claimed. One pipeline definition serves both, resolving level 2 for CZI and level 0 for SVS.

## 4. Assumptions found and eliminated

| Assumption | Where it would have been | Resolution |
|---|---|---|
| A fixed stored level exists in every format | a pipeline with `level: 2` | target spacing plus tolerance, resolved per image; `level` is overridable per run |
| BGR input | legacy `cv2.COLOR_BGR2HSV` | explicit RGB view per declared channel order |
| Black means no data | legacy CZI handling | validity comes from the reader's mask; Otsu uses acquired cells only |
| Alpha is pixel data | SVS RGBA | alpha is validity; dropped only in the detector's explicit RGB view |
| A pyramid exists | single-level SVS | tiled block reduction within a decoded-pixel budget |
| Native and padded frames equal level 0 | legacy coordinates | explicit `FrameTranslation` from reported origins |
| `NOT_REPORTED` means fully acquired | any future reader | `require_reported` by default; `assume_acquired` only explicitly, and recorded |

## 5. Future formats: integration path

None of these is implemented, depended on or claimed. Each needs an `ImageBackend` adapter in its own `plugins/imaging/<name>/` subpackage, with an optional extra, and must pass this suite before a claim is made.

| Format | Adapter responsibilities | Likely metadata differences | Minimum fixtures |
|---|---|---|---|
| Multilevel SVS | already covered by the OpenSlide adapter; verify stored levels and non-integer downsamples | `level_downsamples` like 4.0003; JPEG 2000 variants | OpenSlide `CMU-1.svs` (CC0, multilevel) plus a JPEG 2000 SVS |
| Pyramidal TIFF | OpenSlide `generic-tiff` or tifffile; claim only tested vendors | spacing from resolution tags (unit conversion); may be absent | OpenSlide `Generic-TIFF/CMU-1.tiff`; a generated tifffile pyramid without resolution tags |
| OME-TIFF | tifffile adapter; map OME-XML series and resolutions; pick channels and planes explicitly | `PhysicalSizeX/Y` with units; multi-channel fluorescence (`MULTICHANNEL`, which the detector rejects); Z/C/T planes | a brightfield RGB OME-TIFF pyramid; a fluorescence multi-channel sample to prove rejection |
| NDPI | OpenSlide `hamamatsu`; verify level geometry and offsets | `hamamatsu.XOffsetFromSlideCentre`; possibly no-data regions | OpenSlide `CMU-1.ndpi` (CC0) |
| OME-Zarr | zarr adapter; multiscale datasets as stored levels; chunked region reads | `coordinateTransformations` scale per level; axes order; may be non-RGB | a small OME-NGFF v0.4 RGB pyramid; a sample with a translation transform |
| DICOM WSI | wsidicom or OpenSlide `dicom`; series/instance selection; tiled sparse frames | `TotalPixelMatrixOrigin`, sparse tiling (no-data), ICC profiles, optical paths | a public DICOM WSI with sparse tiling (TCIA or the OpenSlide test data) |

## 6. Coverage gaps

- Only one real CZI and one single-level SVS are tested. Multilevel SVS, multi-scene real CZI and non-integer pyramid ratios on real files are NOT_TESTED.
- Real-file `NOT_REPORTED` validity is NOT_TESTED: both real adapters report validity. Gating is covered with the in-memory reader.
- No real same-content pair across formats exists. Equivalence is shown on generated lossless content only.
