# 003 — Reader Capability Matrix

**Status:** Implemented for SPEC-003 (Accepted 2026-10-16). Last verified 2026-10-16.
**Related:** `../specs/003-medical-imaging-formats-and-readers.md`, `../adr/003-medical-imaging-reader-architecture.md`, `003-imaging-fixture-inventory.md`, `../reports/003-medical-imaging-formats-and-readers.md`

## Labels

Every cell uses one of five labels:

- **TESTED**: asserted by an automated test in this repository against the fixture named in the cell. Results are reproducible with the commands in the report.
- **PROBED**: observed by hand in a throwaway probe venv on this machine, without an automated test.
- **DOC**: documented by the library but not exercised here.
- **NO**: observed to fail, or not offered.
- **UNTESTED**: no fixture or probe.

## Fixtures

Two kinds of fixture are cited:

- **F-CZI-01** and **F-SVS-01** are real whole-slide images.
- **Gen-CZI** files are written by pylibCZIrw in `tmp_path` at test time. They contain synthetic content and are decoded by the real decoder, but they are not real WSIs.

## 1. Adapters shipped

| Adapter | Package | Optional extra | Backend (tested version) | Format codes claimed | Licence |
|---|---|---|---|---|---|
| `czi-pylibczirw` | `plugins/imaging/czi/` | `pathology-ai[czi]` | pylibczirw 6.1.0 | `zeiss-czi` | LGPL-3.0. Approved for research and prototyping; commercial redistribution needs a separate compliance review. |
| `openslide` | `plugins/imaging/openslide/` | `pathology-ai[openslide]` | openslide-python 1.4.6, OpenSlide 4.0.1 (openslide-bin 4.0.1.2) | `aperio-svs` only | LGPL-2.1 (openslide-bin bundles further permissive licences) |

The `czi` extra pulls in `cmake` as a declared runtime dependency of pylibczirw 6.1.0. It is not used at runtime by this adapter.

## 2. CZI: pylibczirw 6.1.0

| Capability | Status and evidence |
|---|---|
| Content detection | **TESTED** (Gen-CZI, F-CZI-01). Detection uses the `ZISRAWFILE` magic and works without pylibCZIrw installed. The extension is ignored. |
| Open / close / file handles | **TESTED** (Gen-CZI, F-CZI-01). The open file-descriptor count returns to its baseline after `close()`, after a context-manager error, and after 20 consecutive opens. Use after close raises `READER_CLOSED`. |
| Level-0 frame | **TESTED** (F-CZI-01). The frame is the scene's `scenes_bounding_rectangle_no_pyramid` = (-262063, 144720, 40470×13109), which equals `Information/Image SizeX/SizeY`. The padded `total_bounding_rectangle` (40512×13120) is **not** the image extent. |
| Native ↔ level-0 transform | **TESTED** (Gen-CZI with origin (-500, 300); F-CZI-01 with origin (-262063, 144720)). The relation is `native = level0 + origin` (`NativeFrameOffset`). Returned pixels are byte-identical to the raw `read(roi=native)`. |
| Pyramid levels | **TESTED**. On F-CZI-01 the stored subblock ratios are {1, 2, 4, 8, 16, 32, 64}, cross-checked by an independent enumeration in the test. On Gen-CZI, which has no stored pyramid, the adapter reports 1 level even though `zoom=0.5` decodes. |
| Level ≥1 spatial mapping | **TESTED** (F-CZI-01). Level-0 pixels averaged 2×2 match level 1 at the mapped area with MAE 1.58, against 7.15 when shifted by 8 px. |
| Non-integer downsample | **TESTED** in the domain (unit test, ds 1.5 and 4.0002). **UNTESTED** on a CZI fixture, because none has non-integer ratios. |
| Bounded region read | **TESTED**: two distinct 512×384 regions on F-CZI-01, plus several on Gen-CZI. |
| Channel order | **TESTED**. `Bgr24` decodes to uint8 `(h, w, 3)` in **BGR** order. Gen-CZI writes known (B, G, R) = (10, 120, 230) and reads back the same; on F-CZI-01 the output is identical to the raw backend. No conversion happens. |
| Other pixel types | **NO** (by design). `Gray8` is rejected with `UNSUPPORTED_READ_OPERATION` (Gen-CZI). Other types (Bgr48, Gray16, Bgra32, …) are **UNTESTED** and rejected. |
| Physical spacing | **TESTED**. F-CZI-01 gives 0.11123987969912646 µm (from `Scaling/Items/Distance`, metres). Gen-CZI without spacing gives `None`, listed in `missing_metadata`; pylibCZIrw writes 0 for unknown values, and the adapter treats 0 as unknown. |
| Scenes | **TESTED** (Gen-CZI, 2 scenes). The caller must choose a scene explicitly when there are several. An unknown scene is rejected; pylibCZIrw itself silently "reads" non-existent scene 5. Each scene has its own native origin. |
| C / Z / T > 1 | **NO** (by design). Gen-CZI with Z=2 is rejected; the adapter never picks a plane implicitly. |
| Channel subset selection | **NO**. Rejected with `UNSUPPORTED_READ_OPERATION`. |
| Out-of-bounds | **TESTED** (Gen-CZI, F-CZI-01). Rejected with `REGION_OUT_OF_BOUNDS` before the decoder is called, verified by a spy. pylibCZIrw itself pads with black. |
| Unacquired areas inside bounds | **TESTED**. Gen-CZI with a 32 px gap gives `NONE_ACQUIRED` or `PARTIALLY_ACQUIRED` with an exact mask. On F-CZI-01, the level-0 corner reads black and is `NONE_ACQUIRED`; the level-6 overview is 64.7% acquired, with near-white (>250) fill baked into the pyramid in unacquired areas. |
| Background versus tissue | **NO**. Imaged glass is acquired data; the reader does not segment tissue. This limitation is reported in every region. |
| Corrupt input | **TESTED**. A truncated Gen-CZI file and a file with a valid magic followed by garbage both raise `CORRUPT_IMAGE`. |
| Resource limits | **TESTED**. Requests over `ReadLimits` (default 4096×4096, matching the API v1 maximum) raise `REGION_LIMIT_EXCEEDED` before decoding. The payload size is exactly w·h·3 bytes. |
| Missing extra | **TESTED**. Without pylibCZIrw the file is still recognised and fails with `READER_BACKEND_UNAVAILABLE`, naming the `pathology-ai[czi]` extra; checked by a unit test and by a clean base-wheel install. |
| Peak memory | **PROBED** only: about 62 MB RSS to open F-CZI-01 and run 3 reads in a probe venv. No automated memory test exists; bounds are enforced by the request limits. |

## 3. Aperio SVS: OpenSlide 4.0.1

| Capability | Status and evidence (F-SVS-01 only) |
|---|---|
| Content detection | **TESTED**. `detect_format` = `aperio` is claimed. Other vendors are not claimed; for example, OpenSlide detects F-CZI-01 as `zeiss` but cannot decode it. |
| Metadata | **TESTED**: 2220×2967, 1 stored level, 0.499 µm/px from `openslide.mpp-x/y`, objective power 20. |
| Multi-level pyramid | **UNTESTED**. F-SVS-01 has a single level. |
| Bounded regions | **TESTED**. Two distinct regions, plus one edge region, are byte-identical to raw `read_region`. |
| Channel order | **TESTED**. Output is RGBA; alpha is kept. openslide-python's un-premultiplication is recorded as a decoder transformation. |
| No-data | **TESTED**: in-bounds regions are `ALL_ACQUIRED`, with an alpha-based mask. Alpha=0 holes in sparse slides are **UNTESTED**, because F-SVS-01 has none. |
| Out-of-bounds | **TESTED**: rejected before decoding. OpenSlide itself returns alpha=0 padding (**PROBED**). |
| Cleanup | **TESTED**: the file-descriptor count returns to baseline. |

**Not claimed:** general SVS compatibility. One small, single-level JPEG SVS has been tested. JPEG 2000 SVS, multi-level SVS and other Aperio variants are untested.

## 4. Formats evaluated and not supported

| Format | Status |
|---|---|
| CZI with OpenSlide 4.0.1 or libvips 8.18.7 | **NO** (PROBED): `JPEG XR compression is not supported` |
| CZI with tifffile | **NO** (PROBED): not a TIFF file |
| Generic pyramidal TIFF / OME-TIFF | **UNTESTED**: no fixture. OpenSlide's `generic-tiff` vendor and tifffile are DOC only. |
| Other OpenSlide vendors (Hamamatsu, MIRAX, Philips, DICOM, …) | **UNTESTED**: not claimed by the adapter |
| CZI as OME-Zarr (S-BIAD1285 also publishes it) | **UNTESTED**: no reader |
| DICOM WSI, CT/MRI volumes | Out of scope (SPEC-003 §13) |

## 5. Phase A correction

The Phase A version of this matrix recorded `total_bounding_rectangle` (40512×13120 at x=-262080) as the level-0 box. That rectangle includes padded pyramid subblocks. The image's level-0 data rectangle is `total/scenes_bounding_rectangle_no_pyramid` (40470×13109 at x=-262063), which matches the file's own `SizeX/SizeY` metadata. The implementation and its tests use the latter.
