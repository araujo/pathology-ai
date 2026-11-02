# 004 — Legacy Patch Coordinate Compatibility

**Status:** Implemented and verified on F-CZI-01, 2026-10-18.
**Related:** `004-preprocessing-pipeline.md`, `../reports/003-medical-imaging-formats-and-readers.md` §15.3, `../reports/004-real-wsi-preprocessing-and-patch-extraction.md` §6

## 1. Frames

| Frame | Origin (native CZI pixels) | Used by |
|---|---|---|
| Canonical level zero | `scenes_bounding_rectangle_no_pyramid` top-left (`ImageDescription.native_offset`) | pathology-ai everywhere |
| Legacy padded | `total_bounding_rectangle` top-left. This rectangle includes padded pyramid subblocks. | legacy `WholeSlideImage.offset`, `create_patches.py` coordinates, `dataset_h5.py` reads, `czi_coverage.py` tiles |

The translation follows from the two origins:

```
legacy = native - padded_origin
level0 = native - level0_origin
level0 = legacy + (padded_origin - level0_origin)
```

`domain.frames.FrameTranslation.from_native_origins` computes the offset per image. `padded_origin_translation(description, frame)` takes the padded origin from the reader's `native.padded_origin` acquisition fact; it raises `MISSING_SPATIAL_METADATA` when the reader reports none. The translation never clips: `place()` classifies each translated rectangle as `inside`, `crosses_bounds` or `outside`, and only `inside` rectangles get level-zero coordinates.

## 2. F-CZI-01 evidence

| Item | Value |
|---|---|
| Level-zero origin | (-262063, 144720) |
| Padded origin (`native.padded_origin`) | (-262080, 144720) |
| Translation | level0 = legacy + **(-17, 0)**. In SPEC-003 terms, legacy = level0 + (17, 0). |
| Legacy coordinates | 66 patch top-lefts. Source: the legacy `spec019_real_validation` bag `coords` (identical in the UNI embedding store), read read-only and saved as numbers to `tests/fixtures/legacy/f-czi-01-legacy-patch-coords.json`. |
| Legacy footprint | 2013 level-0 px = `int(224 · 1.0 / 0.11123987969912646)` |
| Placement | **58 inside, 8 crossing bounds**. Legacy x = 0 begins 17 px left of the image. Legacy y = 12078 + 2013 overruns height 13109, because legacy `use_padding=True` allows edge patches. |
| Pixel check | For 2 inside patches, a raw pylibCZIrw read at `roi = padded_origin + legacy` (exactly the legacy `dataset_h5` read) is **byte-identical** to the adapter's level-zero read at the translated rectangle. |
| Negative check | Using legacy coordinates directly as level-zero coordinates reads different pixels (MAE 9.99 on a 2013² patch); the test asserts inequality. |

Test: `tests/integration/test_real_wsi_preprocessing.py::test_legacy_padded_coordinates_translate_explicitly` (opt-in fixture).

## 3. What is not assumed

- **The offset is not constant.** It depends on each slide's two origins. A unit test uses a second pair of origins with a different offset.
- **Legacy padded patches are not imported** into pathology-ai plans. Crossing patches stay flagged and are never clipped or padded.
- **Legacy patch geometry differs** from the default SPEC-004 pipeline (2013 px at level 0, resized to 224 by the model stage). Reproducing the legacy patch set is not a goal; the bridge only translates coordinates.
- **Only CZI is covered.** OpenSlide reports no native frame, so a padded-frame translation is unavailable for SVS, by design.
