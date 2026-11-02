# Report 003 — Medical Imaging Formats & Readers

**Date:** 2026-10-16
**SPEC:** `../specs/003-medical-imaging-formats-and-readers.md` (Accepted 2026-10-16)
**ADR:** `../adr/003-medical-imaging-reader-architecture.md` (Accepted 2026-10-16)
**Prompt:** `../prompts/003-medical-imaging-formats-and-readers.md`
**Companion documents:** `../architecture/003-reader-capability-matrix.md`, `../architecture/003-imaging-fixture-inventory.md`
**Repository:** `pathology-ai`, branch `main`. **No commits were made** (none were requested). All files are still untracked, so evidence comes from captured command output and hashes, not from `git diff`.

## 1. Executive summary

Pathology AI can now read real whole-slide images through two replaceable, optional reader adapters. Both are selected by file content, not by extension.

| Adapter | Backend | Format tested | Real fixture |
|---|---|---|---|
| `czi-pylibczirw` | pylibczirw 6.1.0 | Zeiss CZI (JPEG XR, Bgr24, 7 stored levels) | F-CZI-01: SurGen `SR1482_40X_HE_T198_01.czi` (CC BY 4.0) |
| `openslide` | openslide-python 1.4.6 / OpenSlide 4.0.1 | Aperio SVS (JPEG, single level) | F-SVS-01: OpenSlide `CMU-1-Small-Region.svs` (CC0) |

What the readers do:

- **Metadata.** They return level-0 dimensions, the pyramid levels that are actually stored, physical spacing (nullable), channel order, pixel type, the scene and plane structure, and backend identity.
- **Regions.** They return bounded regions with explicit level-0 and native-frame coordinates, plus a valid-data report.
- **Errors.** Out-of-bounds, oversized and unsupported requests are rejected with typed errors before anything is decoded.
- **Ingestion.** A model-free smoke workflow (`ImagingService.ingest`, CLI `pathology-ai-imaging-smoke`) turns a real image into content-addressed `image.region_2d` artifacts plus an `IngestionRecord`.

Test results:

- **Without fixture variables:** 235 passed and 10 skipped. Every skip carries an explicit reason.
- **With `PATHOLOGY_AI_WSI_FIXTURE_CZI` and `PATHOLOGY_AI_WSI_FIXTURE_SVS` set:** all **245 pass**.
- ruff, `ruff format --check`, mypy (strict) and `uv build` are clean.
- **Unchanged:** all 166 pre-existing tests, the synthetic demo output (byte-identical), and the Imaging and Analysis API v1 contract files (SHA-256 identical).

**Not claimed:**

- general SVS or CZI compatibility beyond the operations and fixtures listed in §3;
- pyramidal TIFF or OME-TIFF support;
- any model inference, scientific validation or clinical validity.

## 2. Backend evaluation

The full matrix is in `../architecture/003-reader-capability-matrix.md`. Summary:

| Backend | CZI (F-CZI-01) | SVS (F-SVS-01) | Pyramid | Regions | Spacing | Native dependency | Licence |
|---|---|---|---|---|---|---|---|
| pylibczirw 6.1.0 | **TESTED** | n/a | Derived from stored subblock ratios | Yes, ROI + zoom | `Scaling/Distance` (m) | Self-contained wheel (also pulls in `cmake`) | LGPL-3.0 |
| OpenSlide 4.0.1 | **NO**: `JPEG XR compression is not supported` | **TESTED** | Stored levels | Yes | `openslide.mpp-x/y` | `openslide-bin` wheel | LGPL-2.1 |
| libvips 8.18.7 / pyvips | **NO**: same error, because it loads CZI through OpenSlide | not tried | — | — | — | Homebrew libvips | LGPL-2.1 / MIT |
| tifffile | **NO**: not a TIFF file | not tried | — | — | — | Pure Python | BSD-3 |

Why the backends were chosen:

- **pylibczirw** was chosen first because it is the only probed backend that decodes the available CZI files.
- **OpenSlide** was added second, with owner approval and a CC0 fixture. It shows that the reader can be swapped against a genuinely different library.

**Licensing.** pylibczirw's LGPL-3.0 licence is approved for research and prototyping. **Commercial redistribution requires a separate compliance review** (recorded in `pyproject.toml`, the plugin docstring, the README and the SPEC status line). Both libraries are optional extras, so the base wheel contains and installs neither.

## 3. Tested format matrix

| Operation | CZI F-CZI-01 | CZI generated (`tmp_path`) | SVS F-SVS-01 |
|---|---|---|---|
| Content detection, extension ignored | PASS | PASS | PASS |
| Metadata: dimensions, levels, spacing, channels | PASS | PASS (incl. missing spacing) | PASS |
| Native origin ↔ level 0 | PASS (-262063, 144720) | PASS (-500, 300) and per-scene | n/a (no native offset) |
| Stored levels vs decoder zoom | PASS (7 levels, independent cross-check) | PASS (1 level despite `zoom` working) | PASS (1 level) |
| ≥2 distinct bounded regions, byte-equal to raw backend | PASS | PASS | PASS |
| Level-1 ↔ level-0 mapping | PASS (MAE 1.58 aligned vs 7.15 shifted) | — | n/a (single level) |
| Channel order preserved | PASS (BGR, raw-equal) | PASS (known BGR values) | PASS (RGBA, raw-equal) |
| Valid-data mask | PASS (corner `NONE_ACQUIRED`, overview 64.7%) | PASS (gap mask exact) | PASS (alpha, all acquired) |
| Out-of-bounds rejected before decoding | PASS | PASS (spy) | PASS |
| Oversized rejected before decoding | covered by generated | PASS (spy) | covered by generated |
| Scenes | single | PASS (2 scenes, explicit choice required) | single |
| Z/T > 1, Gray8, channel subsets rejected | — | PASS | PASS (channels, z/t, scene) |
| Corrupt or truncated input | — | PASS | — |
| Handle cleanup (fd count) | PASS | PASS | PASS |

**Untested, and therefore not claimed:**

- multi-level SVS;
- JPEG 2000 SVS;
- sparse slides with alpha holes;
- CZI pixel types other than Bgr24;
- real multi-scene CZI;
- non-integer pyramid ratios on a real file (the domain mapping is unit-tested with ds 1.5 and 4.0002);
- generic TIFF, OME-TIFF and OME-Zarr.

## 4. Reader implementation

```text
domain/reading.py                 RegionReadRequest, ReadLimits, validate_region, NativeFrameOffset,
                                  NoDataSemantics/DataValidity/ValidDataReport, SourceIntegrity,
                                  ImageDescription, RegionReadProvenance, PixelRegion, IngestionRecord
domain/errors.py                  + ImageReadError family (8 typed errors, see §12)
domain/kinds.py                   + IMAGE_REGION_2D ("image.region_2d")
ports/__init__.py                 + ImageSource, SourceProbe, OpenedImage, ImageBackend, SourceFingerprinter;
                                    ReaderCapabilities + supports_scenes, detects_content (defaulted)
application/imaging_service.py    ImageBackendRegistry (content-based resolution), ImagingService (open/describe/ingest)
infrastructure/imaging/integrity.py  PartialContentFingerprinter (size + sha256 head/tail; optional full)
plugins/imaging/common.py         coverage_mask, ClosableHandle, backend version lookup (numpy only)
plugins/imaging/czi/              CziBackend / CziImage (lazy pylibCZIrw import)
plugins/imaging/openslide/        OpenSlideBackend / OpenSlideImage (lazy openslide import)
presentation/api/imaging_v1.py    + region_request_from_dict, reader_diagnostics, region_summary
bootstrap.py                      + build_image_backends, build_imaging_service, Platform.imaging
cli.py                            + pathology-ai-imaging-smoke (imaging_smoke_main)
pyproject.toml                    + extras czi/openslide, dev group installs both, mypy overrides, script
```

### 4.1 Registration and capability resolution

`ImageBackendRegistry` keys backends by name. `resolve(source)` asks every backend to `probe` the file content:

- **Exactly one** recognising and available backend: it is selected.
- **More than one:** `AMBIGUOUS_READER`. The caller can name a backend explicitly instead.
- **Recognised, but the backend is not installed:** `READER_BACKEND_UNAVAILABLE`, naming the extra to install.
- **None:** `UNSUPPORTED_IMAGE_FORMAT`.

The pipeline engine (`application/runner.py`) and its `ReaderRegistry` are unchanged. Neither knows about these backends.

### 4.2 Metadata

`ImageDescription` wraps the existing `ImageAsset` and keeps its identity, format, modality, geometry and channel order. Around it, it records:

- the scene and scene count;
- the native offset;
- the backend pixel type, verbatim, and the array dtype;
- samples per pixel and the C/Z/T sizes;
- where the spacing came from and how the levels were established;
- the no-data semantics;
- decoder transformations;
- a small allow-list of non-identifying acquisition facts;
- `missing_metadata`;
- source integrity.

The modality is always **declared by the caller** in `ImageSource`; a CT declaration on a WSI is rejected. Spacing is `None` whenever the file lacks it. `ImageDescription` refuses absent spacing unless `physical_spacing` is listed in `missing_metadata`.

### 4.3 Region reading

Requests are made in the `wsi_level_pixel` frame with integer coordinates. Any other frame raises `COORDINATE_FRAME_MISMATCH`. Negative origins and zero sizes are rejected when the request is constructed.

`validate_region` then checks, in order:

1. the level exists;
2. the region fits that level's grid;
3. the region maps inside the level-0 bounds.

After that, `ReadLimits` checks size. Only then does the adapter call the backend.

Every result carries:

- the pixels;
- the requested and returned level rectangles, and the level-0 rectangle;
- the downsample and native region;
- shape, dtype and channel order;
- decoder transformations and metadata limitations;
- a `ValidDataReport`.

### 4.4 Resource safety

- Handles close deterministically (`ClosableHandle`, idempotent `close()`, context manager). `ImagingService.open` closes them in a `finally` block.
- Using a handle after close raises `READER_CLOSED`.
- Default limits match the API v1 region maximum: 4096×4096 and 16 Mi pixels.
- Neither adapter ever decodes a full level-0 slide. Reads are ROI-bounded. F-CZI-01's level 0 would be about 1.59 GB as BGR; the probe measured about 62 MB RSS (PROBED, not an automated test).

## 5. Fixture inventory

The details are in `../architecture/003-imaging-fixture-inventory.md`.

| ID | Size | SHA-256 | Source / licence | Provenance evidence |
|---|---|---|---|---|
| F-CZI-01 | 77,963,616 | `8634d055…e8be4` | EMBL-EBI BioImage Archive S-BIAD1285 (SurGen), **CC BY 4.0** | The archive file list contains `SR1482_WSIs/SR1482_40X_HE_T198_01.czi` at exactly this size. GigaScience doi:10.1093/gigascience/giaf086 states de-identification and ethics approvals. |
| F-SVS-01 | 1,938,955 | `ed92d5a9…02c8a7` | OpenSlide test data, **CC0-1.0** | SHA-256 equals OpenSlide's `index.json`. Downloaded 2026-10-16 to `~/pathology-ai-fixtures/openslide/`, outside the repository. |

The real-fixture tests fail if a configured file's size or SHA-256 differs. They skip, with a reason, if the variable is unset. No image file exists anywhere in the working tree (`find . -iname '*.czi' -o -iname '*.svs' -o -iname '*.tif*' -o -iname '*.ndpi'` finds nothing).

## 6. Metadata evidence (F-CZI-01, F-SVS-01)

`uv run pathology-ai-imaging-smoke --file …/SR1482_40X_HE_T198_01.czi --image-id f-czi-01 --modality wsi …` reports:

```text
format zeiss-czi; level 0 40470 x 13109; native_origin [-262063, 144720]
levels (downsample): 1, 2, 4, 8, 16, 32, 64  -> (632 x 204) at level 6
pixel_spacing_um 0.11123987969912645 (x = y); spacing_source file:ImageDocument/Metadata/Scaling/Items/Distance
pixel_type Bgr24, dtype uint8, samples 3, planes c=1 z=1 t=1, scene 0 of 1
backend pylibczirw 6.1.0; acquisition: ZEN 3.1 (blue edition) 3.1.0.00011, JpgXr, 8-bit
missing_metadata []
```

The level-0 size 40470×13109 equals the file's own `Information/Image SizeX/SizeY`.

The levels come from stored subblock ratios. The test cross-checks them against an independent enumeration in the test itself: ratios {1, 2, 4, 8, 16, 32, 64} and 216 level-0 subblocks (`only_layer0=True`).

SVS: 2220×2967, 1 level, 0.499 µm/px, RGBA, objective power 20. The backend is reported as `openslide-python 1.4.6+lib4.0.1`.

## 7. Region-reading evidence

| Fixture | Request (level, x, y, w, h) | Level-0 rect | Native region | Shape | Validity | Check |
|---|---|---|---|---|---|---|
| F-CZI-01 | 0, 36224, 4288, 512, 384 | (36224, 4288, 512, 384) | (-225839, 149008, 512, 384) | 384×512×3 | all acquired | byte-equal to raw `pylibCZIrw.read` |
| F-CZI-01 | 0, 20000, 6000, 512, 384 | (20000, 6000, 512, 384) | (-242063, 150720, 512, 384) | 384×512×3 | all acquired | byte-equal; differs from the first region |
| F-CZI-01 | 1, 10000, 3000, 256, 192 | (20000, 6000, 512, 384) | (-242063, 150720, 512, 384) | 192×256×3 | all acquired | zoom 0.5 from stored level 1 |
| F-CZI-01 | 6, 0, 0, 632, 204 | (0, 0, 40448, 13056) | (-262063, 144720, 40448, 13056) | 204×632×3 | partial, 0.647 | coverage mask |
| F-SVS-01 | 0, 0, 0, 256, 256 / 0, 1000, 1500, 300, 200 | same | same | ×4 (RGBA) | all acquired | byte-equal to raw `OpenSlide.read_region` |

The artifact IDs from the installed-wheel run (§12.3) are `pai-art-fe38f763…` and `pai-art-0edcdcc3…`. Repeating an ingestion with the same inputs gives identical artifact identities (tested).

## 8. Spatial mapping evidence

- **Native ↔ level 0:** `native = level0 + origin` (`NativeFrameOffset`), checked in three places:
  - unit round-trip with origin (-262063, 144720);
  - generated CZI with origin (-500, 300), plus a second scene at (100, -50);
  - F-CZI-01, where the returned pixels are byte-equal to raw reads at the expected native ROI.
- **Level ↔ level 0:** a level-1 read of `(x//2, y//2, 256, 256)` reports the same level-0 rectangle as a level-0 read of `(x, y, 512, 512)`. Their pixels agree: the 2×2 mean of level 0 against level 1 gives MAE **1.58**. Shifting by 8 px raises it to **7.15**. The test asserts aligned < 3.0 < shifted.
- **Non-integer downsample:** unit-tested on the domain geometry (ds 1.5 and 4.0002, including bounds). No real fixture has non-integer ratios.
- **Bounds:** right, bottom, per-level and missing-level violations are rejected (unit and fixture tests). A spy confirms the decoder is never called.

## 9. Channel and no-data semantics

**Channels.**

- CZI `Bgr24` is returned as BGR, `ChannelOrder.BGR`. A generated CZI written with (B, G, R) = (10, 120, 230) reads back exactly that. F-CZI-01 output is byte-identical to the raw backend.
- OpenSlide returns RGBA, `ChannelOrder.RGBA`, with alpha kept. openslide-python's conversion from premultiplied ARGB to straight RGBA is recorded as a decoder transformation.
- Nothing converts channels silently. Explicit conversion stays in model-bound preprocessing (`TransformStep`, SPEC-001).
- Channel subsets are rejected rather than guessed.

**No-data.** Each reader declares its `NoDataSemantics`:

- `ACQUISITION_COVERAGE` (CZI): a mask computed from level-0 subblock rectangles, tested at pixel centres at the returned resolution.
- `BACKEND_TRANSPARENCY` (OpenSlide): alpha > 0.
- `NOT_REPORTED` (available for future readers): must be `UNKNOWN`, with a stated reason.

The four categories in SPEC §5.3 are handled as follows:

| Category | Handling |
|---|---|
| (1) Valid acquired pixels | mask True |
| (2) Background represented in the file (glass) | **also acquired**. Not distinguished from tissue; every region states this limitation. |
| (3) Unacquired in-bounds | mask False. The CZI fill is `(0,0,0)` at level 0 but near-white (>250) in stored pyramid levels. Both were observed on F-CZI-01, so only the mask is reliable. |
| (4) Out of bounds | rejected, never padded |

## 10. Provenance evidence

Each region's `RegionReadProvenance` records:

- image id, reader and backend component versions;
- the source integrity token and scene;
- the requested and returned level rectangles, the level-0 rectangle and the downsample;
- the native region, shape, dtype and channel order;
- decoder transformations and metadata limitations;
- validity and acquired fraction.

Provenance is canonicalisable and drives the content-addressed `ArtifactIdentity`. Each `IngestionRecord` adds:

- the description digest;
- region provenance and artifact identities;
- the environment;
- timestamps;
- a fixed usage statement: *"image ingestion only: no model inference was performed; decoding an image is not evidence of scientific or clinical validity"*.

**Integrity strategy.** `size+sha256-head-tail/v1` hashes the size plus the first and last 1 MiB:

- **Detects:** truncation, replacement, and header or trailing-directory changes.
- **Does not detect:** edits strictly in the middle of the file. That limit is written into `SourceIntegrity.guarantees` and unit-tested.

`PartialContentFingerprinter(full=True)` gives a full SHA-256 when needed. The real-fixture tests use the full SHA-256.

## 11. API compatibility

- **All 30 Imaging and Analysis API v1 schema and fixture files have identical SHA-256** before and after (`diff` against the baseline is empty).
- `image_descriptor()` is unchanged. A descriptor built from a real CZI and from a generated CZI both validate against `image-descriptor.schema.json`:
  - `format: zeiss-czi`, `channel_order: bgr`;
  - missing spacing becomes `null`;
  - access is `regions: true, thumbnail: false`.
- `region_request_from_dict` maps a v1 region request onto `RegionReadRequest`. It is tested with the existing valid fixture and with a wrong frame.
- `ReaderCapabilities` gained two fields with defaults (`supports_scenes`, `detects_content`). This is a Python-level addition only; no wire format changed.

**Proposed future contract extensions** (v2 or additive v1.x; not implemented, and needing a versioning decision):

1. Region request: `scene`, `z`, `t`, `channels`. Today a v1 request cannot address a multi-scene image, so such images are rejected.
2. Descriptor: `native_origin`, `scene` / `scene_count`, `pixel_type` / `dtype`, `samples_per_pixel`, `spacing_source`, `missing_metadata`, `reader` and `backend` identity, `no_data` semantics.
3. Region response metadata: `level0` rectangle, `validity`, `acquired_fraction`, and an optional valid-data mask encoding.
4. Error responses using the new error codes.

Until then, `imaging_v1.reader_diagnostics` and `region_summary` expose this information, explicitly marked as **not** part of the v1 contract.

## 12. Tests and packaging results

### 12.1 Commands (2026-10-16, macOS arm64, Python 3.12.13, uv 0.11.17)

| Command | Result |
|---|---|
| `uv sync` | rc 0. Installed pylibczirw 6.1.0, openslide-python 1.4.6, openslide-bin 4.0.1.2, pillow 12.3.0, plus deps (incl. cmake 4.4.4) |
| `uv run ruff check .` | All checks passed |
| `uv run ruff format --check .` | 91 files already formatted |
| `uv run mypy` (strict) | Success: no issues found in 75 source files |
| `uv run pytest -q -W error -rs` | **235 passed, 10 skipped**. The 10 skips are the real-fixture tests, each printing "real fixture not configured: set PATHOLOGY_AI_WSI_FIXTURE_…". |
| same, with `PATHOLOGY_AI_WSI_FIXTURE_CZI` and `PATHOLOGY_AI_WSI_FIXTURE_SVS` set | **245 passed** |
| `uv build` | `pathology_ai-0.1.0-py3-none-any.whl` (67 entries) and `.tar.gz` |

### 12.2 Test inventory

There are 245 collected tests:

- **166 pre-existing:** every original ID is still collected.
- **74 in new files:**

  | File | Tests |
  |---|---|
  | `tests/unit/test_reading_contracts.py` | 23 |
  | `tests/unit/test_imaging_registry.py` | 10 |
  | `tests/integration/test_czi_reader.py` | 19 |
  | `tests/integration/test_real_wsi_fixtures.py` | 10 |
  | `tests/contract/test_reader_contracts.py` | 12 |

- **5 new architecture cases:** 4 planted violations for backend imports, and `test_reader_backends_are_imported_only_by_their_adapters`.

**Changes to existing tests.** `tests/architecture/test_dependency_rules.py` changed in three ways:

- `importlib.import_module("…")` literals now count as imports;
- `pylibCZIrw` and `openslide` are allowed only inside their own adapter subpackages;
- the core forbidden terms gained zeiss, aperio, openslide and pylibczirw.

No existing assertion was weakened or removed.

### 12.3 Packaging and optional-dependency isolation

These runs used clean venvs, with the cwd outside the checkout:

- **Base wheel, no extras.**
  - Installed state: `pylibCZIrw`, `openslide` and `PIL` are absent. `plugins.imaging.czi`, `plugins.imaging.openslide` and `bootstrap` import fine, from `site-packages`. Both backends report `available() == False`.
  - Smoke on F-CZI-01: exit 3, `READER_BACKEND_UNAVAILABLE … install pathology-ai[czi]`.
  - Smoke on F-SVS-01: exit 3, `UNSUPPORTED_IMAGE_FORMAT`.
  - `pathology-ai-demo --root …` output is byte-identical to the pre-SPEC-003 baseline.
- **`pathology-ai[czi] @ file://…whl`.** pylibCZIrw is present and openslide absent. Smoke on F-CZI-01 exits 0: 40470×13109, 7 levels, origin [-262063, 144720], two regions as in §7. F-SVS-01 is still unsupported, as expected without the `openslide` extra.
- **Wheel metadata:** base `Requires-Dist` is only `numpy` and `pyyaml`. `Provides-Extra: czi` brings `pylibczirw<7,>=6.1.0`. `Provides-Extra: openslide` brings `openslide-python<2,>=1.4.6` and `openslide-bin<5,>=4.0.1`. The entry points are `pathology-ai-demo` and `pathology-ai-imaging-smoke`.

### 12.4 Regression

- Synthetic demo from the checkout: byte-identical to the baseline captured before any SPEC-003 change. sha256 prefix `11a211dec82f2768` for both.
- All three synthetic projects and the volumetric `EXECUTION_UNSUPPORTED` path are unchanged.
- `ExecutionFingerprint` inputs are unchanged. The tracked environment packages were deliberately not extended, so existing fingerprints are stable.

## 13. Acceptance criteria

| AC | Status | Evidence |
|---|---|---|
| AC-001 Reader architecture | **PASS** | Two backends register through `ImageBackendRegistry` with no change to `application/runner.py` or the pipeline definitions. The format-neutral contracts added to `domain/reading.py` name no vendor (core-forbidden-terms test). Each backend is confined to its own subpackage (AST rule). |
| AC-002 Real image open | **PASS** | F-CZI-01 (CZI) and F-SVS-01 (SVS) open through content-based resolution (§6, `test_real_wsi_fixtures.py`). |
| AC-003 Metadata extraction | **PASS** | Dimensions, stored levels (cross-checked independently), spacing with its source, channels, pixel type and planes (§6). Missing spacing stays `None` and is listed (generated CZI). |
| AC-004 Bounded region read | **PASS** | Two distinct regions per real fixture with exact shapes, coordinates, and byte equality to the raw backend (§7). |
| AC-005 Spatial consistency | **PASS** | Native ↔ level-0 transforms (3 checks) and level-1 ↔ level-0 pixel agreement (§8). Non-integer downsample is verified in the domain only, because no real fixture has one. |
| AC-006 Channel integrity | **PASS** | BGR (CZI) and RGBA (SVS) preserved and reported. Known-value BGR test; raw-equality tests; decoder transformations recorded (§9). |
| AC-007 No-data semantics | **PASS** | CZI coverage mask (gap test plus real overview at 64.7%, with black versus white fill observed). SVS alpha mask. Out of bounds is rejected. Background versus tissue is reported as a limitation (§9). |
| AC-008 Capability resolution | **PASS** | Unknown content, ambiguity, missing backend, unknown or implicit scene, Z/T > 1, unsupported pixel type, channel subsets, wrong frame and missing level all raise typed errors (unit and integration tests). |
| AC-009 Resource safety | **PASS** | Oversized and out-of-bounds requests are rejected before decoding (spy tests). File-descriptor counts return to baseline (real and generated fixtures, error path, 20 repeated opens). Reads are ROI-bounded. Peak memory is PROBED only (about 62 MB versus about 1.59 GB for a full level-0 decode). |
| AC-010 Pipeline integration | **PASS** | `ImagingService.ingest` produces `DataValue(image.region_2d)` artifacts in the shared `ArtifactStore` plus an `IngestionRecord`, from the real CZI, including from the installed wheel (§10, §12.3). It is deliberately not run through `PipelineRunner`, whose synthetic operations require the synthetic reader (§14). |
| AC-011 API contract stability | **PASS** | 30 contract and fixture files are SHA-256 identical. Real descriptors validate against v1. Extensions are proposed, not applied (§11). |
| AC-012 Regression protection | **PASS** | All 166 prior tests pass. The demo is byte-identical (checkout and installed wheel). |
| AC-013 Repository independence | **PASS** | No import of, runtime dependency on, or shared storage with `digital-pathology` or `hal-viewer` (architecture tests). Fixtures are found only through environment variables. |
| AC-014 Evidence-based format support | **PASS** | The capability matrix separates TESTED, PROBED, DOC, NO and UNTESTED, with fixture IDs, sizes and SHA-256 (§3, §5). |
| AC-015 Root-level architecture | **PASS** | New code is under `domain/`, `ports/`, `application/`, `infrastructure/imaging/`, `plugins/imaging/`, `presentation/api/`. `pathology_ai/` does not exist (test plus `ls`). The wheel `only-include` list is unchanged. |

## 14. Limitations and deferred formats

**Formats**

- **Untested, so not claimed:** multi-level SVS, JPEG 2000 SVS, other OpenSlide vendors, generic pyramidal TIFF, OME-TIFF, OME-Zarr, DICOM WSI, real multi-scene CZI, and CZI pixel types other than Bgr24.
- **Rejected by design:** multi-plane CZI (C/Z/T > 1) and channel subsets.

**Features not implemented**

- Thumbnails: `access.thumbnail` is false.
- `ImageTileProvider` and `ImageMetadataProvider` (the SPEC-001 ports) are not implemented by the real adapters.
- `PipelineRunner` cannot execute pipelines on real images yet, because the synthetic tissue and patch operations require `SyntheticSlideReader`.

**Spatial and pixel limits**

- Non-integer CZI pyramid ratios would round native ROIs to integers. Both the requested and the returned rectangles are recorded, but this has not been exercised on a real file.
- The CZI coverage mask is evaluated at pixel centres at the returned resolution. It is not an exact area fraction at coarse levels; the legacy implementation computes exact area (§15).
- Background (glass) is not distinguished from tissue. Tissue detection is out of scope.

**Evidence limits**

- There is no automated peak-memory test; the memory evidence comes from a probe and from the bounded-request design.
- The partial integrity fingerprint does not detect edits in the middle of a file (documented guarantee).

**Dependencies and licensing**

- The pylibczirw 6.1.0 wheel pulls in `cmake` as a declared dependency.
- LGPL-3.0: commercial redistribution needs a compliance review.

## 15. Digital Pathology Reference Assessment

`../digital-pathology` (HEAD `c85df8a966e8`) was used **read-only** as a technical reference. Only `git status`, `git rev-parse`, `grep` and file reads were run in it. Nothing was imported, copied wholesale, or shared at runtime.

### 15.1 Source files inspected

| File | Relevance |
|---|---|
| `reproducibility/wsi_core/WholeSlideImage.py` | pylibCZIrw initialisation, offset, level dimensions, mpp, region reads, BGR→HSV |
| `scripts/pipeline/czi_coverage.py` | level-0 subblock coverage from the raw ZISRAW directory; no-data semantics (SPEC-024.5) |
| `tests/pipeline/czi_fixtures.py` | generated CZI mosaics with tissue, glass and no-data tiles |
| `scripts/analysis_api/wsi_resolution.py` | slide-location resolution (no reader logic) |
| `reproducibility/README.md`, `.claude/CLAUDE.md`, `docs/*.md`, `.claude/reports/SPEC-024.5-report.md` | dataset provenance mentions (S-BIAD1285) |
| `pyproject.toml`, `reproducibility/environment/requirements.txt` | `pylibczirw>=3.5.2` |

### 15.2 Reused or adapted

| Legacy solution | Adaptation in pathology-ai |
|---|---|
| `pylibCZIrw` as the CZI backend (`CziReader`, `read(roi, zoom, plane)`) | Same library, behind the `ImageBackend` port, lazily imported, optional extra |
| mpp = `Scaling/Items/Distance` value × 1e6 | Same unit conversion. X and Y are matched by `@Id`; 0, absent or non-finite values mean unknown, never assumed |
| Coverage from level-0 subblocks (`czi_coverage.py`) and the lesson that "level 0 decodes no-data as black while pyramid tiles bake white" | Coverage taken from pylibCZIrw's subblock enumeration instead of parsing the raw directory, and exposed as `NoDataSemantics.ACQUISITION_COVERAGE`. The black/white lesson was **independently re-verified** on F-CZI-01 (`test_czi_unacquired_areas_are_flagged_at_every_level`). |
| Generated CZI fixtures (`write_mosaic_czi`) | Same technique (`tests/imaging_support.write_czi`), extended to negative origins, multiple scenes, Z stacks, Gray8 and missing spacing |
| BGR awareness (`cv2.COLOR_BGR2HSV`) | Kept as a reported `ChannelOrder.BGR`; no conversion inside the reader |

### 15.3 Differences required by the new architecture

| Legacy behaviour | pathology-ai behaviour | Reason |
|---|---|---|
| Offset and extent from `total_bounding_rectangle` (includes padded pyramid subblocks: origin -262080, 40512×13120 on F-CZI-01) | Level 0 = `scenes_bounding_rectangle_no_pyramid` (origin -262063, 40470×13109, which equals `SizeX/SizeY`) | The padded extent is not image data. **Consequence:** legacy zero-based patch coordinates for this slide are offset by **(+17, 0)** from pathology-ai level-0 coordinates. Any future DPA or legacy coordinate bridge must translate through native coordinates. |
| Ten levels invented as `1/2**k`, regardless of what is stored | Only stored subblock ratios become levels (7 on F-CZI-01; 1 on unpyramided files) | SPEC/ADR D3: decoder zoom is not a resolution level |
| Downsample estimated from invented dimensions | Downsample is the exact stored ratio. Level dimensions are `floor(level0 / ratio)` | Explicit spatial semantics |
| No bounds or scene validation; padding accepted | `REGION_OUT_OF_BOUNDS`, explicit scene required, unknown scene rejected | pylibCZIrw silently pads and silently "reads" non-existent scenes (observed) |
| Coverage as an exact area fraction per square patch (`covered_fraction`) | Boolean mask at pixel centres at the returned resolution, plus its mean | A per-pixel mask suits arbitrary regions and levels; exact area is not reproduced (§14) |
| Raw ZISRAW directory parser (`pyramid_type == 0`, stored size equals logical size) | pylibCZIrw `enumerate_subblocks` (logical/stored ratio 1, plus scene `S`) | Avoids a second hand-written binary parser. Cross-checked against the library's `only_layer0=True` (216 subblocks), **not** against the legacy parser, which was not executed. |
| Slide identity from `manifest.csv` paths and S3 URIs | Caller-supplied `ImageId` plus `SourceIntegrity`; the path never appears in ids or API documents | Repository independence; no-paths contract |
| Configured through the experiment YAML and `scripts/` imports | Composition through `bootstrap.build_imaging_service` | No legacy coupling |

### 15.4 Tests demonstrating equivalent behaviour (verified independently in pathology-ai)

- **No-data fill (black at level 0, white in the pyramid):** `test_czi_unacquired_areas_are_flagged_at_every_level` (F-CZI-01).
- **Coverage from level-0 subblocks:** `test_gap_is_reported_as_unacquired_not_as_black_tissue` (generated mosaic, the same approach as legacy `czi_fixtures`).
- **mpp extraction:** `test_czi_is_selected_by_content_and_described` (0.11124 µm) and `test_metadata_native_origin_and_single_stored_level`.
- **BGR decoding:** `test_bgr_is_preserved_and_mapped_from_level0`.

### 15.5 Legacy claims versus independent verification

| Statement | Verified in legacy (by its own reports or tests) | Verified here |
|---|---|---|
| pylibCZIrw decodes SurGen CZI (JPEG XR) | Yes (pipeline use) | Yes (F-CZI-01) |
| Level-0 no-data is black, pyramid fill is white | Yes (SPEC-024.5) | Yes, independently |
| Ten 2^k levels exist | Assumed, not verified | **Contradicted**: 7 stored levels on F-CZI-01 |
| `total_bounding_rectangle` is the image extent | Assumed | **Contradicted**: it includes pyramid padding |
| S-BIAD1285 licence and de-identification | Not documented in the legacy repository | Established from the archive (CC BY 4.0) and the publication |

### 15.6 Remaining discrepancies and unsupported capabilities

- Exact-area coverage fraction (legacy `covered_fraction`) is not provided.
- There is no tissue segmentation or patch extraction; legacy CLAM-style segmentation is out of scope.
- The (+17, 0) coordinate difference against legacy patch coordinates for F-CZI-01 is documented but not yet bridged. The offset differs per slide.
- Legacy S3 materialisation and manifest resolution have no counterpart (out of scope).

## 16. Legacy repository protection

| Repository | HEAD before | HEAD after | `git status --porcelain` |
|---|---|---|---|
| `digital-pathology` | `c85df8a966e8cc063a61c509d038f1b1cae89c3a` | same | identical (`diff` empty) |
| `hal-viewer` | `e248202983bc6cdb497ed00ef0168e6b47338e09` | same | identical (`diff` empty) |

Only read-only commands were run in `digital-pathology`. `hal-viewer` was not opened beyond `git rev-parse` and `git status`.

## 17. Recommendations for subsequent specifications

1. **Imaging API v1.1 or v2:** add scene, plane and channel selection, reader diagnostics, and valid-data metadata (§11). Decide whether additive optional fields count as v1-compatible.
2. **Real-image pipeline execution:** introduce reader-agnostic tissue and patch operations that consume `OpenedImage` and `PixelRegion`, so `PipelineRunner` can run on real slides. Keep model-bound channel conversion explicit (BGR sources against RGB models).
3. **Coordinate bridge to legacy and DPA:** carry native origins in artifacts, and verify the per-slide offset between legacy padded coordinates and level-0 coordinates before reusing any legacy patch coordinates or embeddings.
4. **Fixture expansion (needs approval):**
   - a multi-level SVS (OpenSlide `CMU-1.svs`, CC0);
   - a generic or OME-TIFF fixture;
   - a real multi-scene CZI;
   - optionally the S-BIAD1285 OME-Zarr rendition, as a second route to the same slide for cross-backend agreement tests.
5. **Exact coverage and memory tests:** add area-exact coverage where patch-level validity gates need it, and an automated peak-memory check, for example a subprocess RSS bound.
6. **Licensing:** run the LGPL-3.0 compliance review before any commercial distribution that bundles pylibczirw.
7. **Thumbnails:** implement them via the smallest stored level, under the same limits.

## Appendix: reproduction

```bash
uv sync
uv run ruff check . && uv run ruff format --check . && uv run mypy
uv run pytest -q -W error -rs                       # fixture tests skipped with reasons
PATHOLOGY_AI_WSI_FIXTURE_CZI=~/Downloads/digital-pathology-test/SR1482_40X_HE_T198_01.czi \
PATHOLOGY_AI_WSI_FIXTURE_SVS=~/pathology-ai-fixtures/openslide/CMU-1-Small-Region.svs \
  uv run pytest -q -W error -rs                     # 245 passed
uv build
uv run pathology-ai-imaging-smoke --file ~/Downloads/digital-pathology-test/SR1482_40X_HE_T198_01.czi \
  --image-id f-czi-01 --modality wsi --region 0,36224,4288,512,384 --region 1,10000,3000,256,192
```
