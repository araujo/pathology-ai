# 003 — Imaging Fixture Inventory

**Status:** SPEC-003, 2026-10-16. No image file is committed to Git, and none is copied into the repository.
**Related:** `003-reader-capability-matrix.md`, `../specs/003-medical-imaging-formats-and-readers.md` §7.3, `../reports/003-medical-imaging-formats-and-readers.md`

## 1. Real fixtures (opt-in)

Tests read real fixtures through environment variables. When a variable is unset, the dependent tests are **skipped** and the reason is printed under `-ra`. When a variable is set, the file's size **and** full SHA-256 must match the values below, or the test **fails**; it is never skipped.

### F-CZI-01

| Field | Value |
|---|---|
| Format | Zeiss CZI (ZISRAW). JPEG XR, `Bgr24`, 1 scene, C/Z/T = 1 |
| Local path (outside repo) | `~/Downloads/digital-pathology-test/SR1482_40X_HE_T198_01.czi` |
| Environment variable | `PATHOLOGY_AI_WSI_FIXTURE_CZI` |
| Size | 77,963,616 bytes |
| SHA-256 | `8634d0551cb7c137ef9dc87f50a69f71a261c3e7817fc6a56aa8e329a13e8be4` (computed locally) |
| Source | SurGen dataset, EMBL-EBI BioImage Archive **S-BIAD1285** (DOI 10.6019/S-BIAD1285), released 2024-07-24. The archive's `wsi_file_paths.json` lists `SR1482_WSIs/SR1482_40X_HE_T198_01.czi` with size **77,963,616**, the same as the local file. The archive publishes no checksum. |
| Licence | **CC BY 4.0**, from the study metadata (`License: CC BY 4.0`). Attribution: cite the SurGen GigaScience article, doi:10.1093/gigascience/giaf086, and the archive DOI. |
| De-identification | The publication states that the data was de-identified ("removal or redaction of potentially identifiable information, including dates of diagnosis, date of death, and treatment details"). It records ethics approval (St Andrews CS16553; Lothian NRS BioResource REC 20/ES/0061 and 13/ES/0126). The adapter also surfaces no user, date or name metadata from the file. |
| How identity was established | Exact size match against the archive listing, plus a local SHA-256 pinned in the tests. The legacy `digital-pathology` repository names the archive (S-BIAD1285) but contains no licence or de-identification evidence; the evidence above was taken from the archive and the publication. |
| Expected metadata (asserted) | Level 0: 40470×13109. Native origin (-262063, 144720). Stored levels 1, 2, 4, 8, 16, 32, 64. 0.11123987969912646 µm/px. 216 level-0 subblocks. Overview 64.7% acquired. |

### F-SVS-01

| Field | Value |
|---|---|
| Format | Aperio SVS. JPEG, single pyramid level, 2220×2967 |
| Local path (outside repo) | `~/pathology-ai-fixtures/openslide/CMU-1-Small-Region.svs` |
| Environment variable | `PATHOLOGY_AI_WSI_FIXTURE_SVS` |
| Size | 1,938,955 bytes |
| SHA-256 | `ed92d5a9f2e86df67640d6f92ce3e231419ce127131697fbbce42ad5e002c8a7`. This equals the value in OpenSlide's `index.json`, which was downloaded alongside it. |
| Source | `https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/CMU-1-Small-Region.svs`, downloaded 2026-10-16 with owner approval |
| Licence | **CC0-1.0** (OpenSlide test-data `index.json`) |
| Expected metadata (asserted) | 2220×2967. 1 level. 0.499 µm/px. Objective power 20. RGBA. |

## 2. Synthetic-format fixtures (always run)

The CZI adapter tests (`tests/integration/test_czi_reader.py`, `tests/contract/test_reader_contracts.py`) write tiny CZI files with pylibCZIrw into `tmp_path`. The pattern was adapted from `digital-pathology/tests/pipeline/czi_fixtures.py`. The files cover:

- negative native origins;
- tile gaps (unacquired areas);
- known BGR values;
- two scenes;
- Z=2;
- `Gray8`;
- missing spacing;
- truncated and garbage files.

These files contain no patient data and are not counted as real WSIs for AC-002.

## 3. Reproducing

```bash
export PATHOLOGY_AI_WSI_FIXTURE_CZI=~/Downloads/digital-pathology-test/SR1482_40X_HE_T198_01.czi
export PATHOLOGY_AI_WSI_FIXTURE_SVS=~/pathology-ai-fixtures/openslide/CMU-1-Small-Region.svs
uv run pytest -q -W error -rs
# Fetch F-SVS-01 (CC0) again if needed:
curl -sSLO https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/CMU-1-Small-Region.svs
# F-CZI-01: download SR1482_WSIs/SR1482_40X_HE_T198_01.czi from S-BIAD1285 (CC BY 4.0)
```

## 4. Missing fixtures

| Format | Status |
|---|---|
| Multi-level SVS | none. Candidate: OpenSlide `CMU-1.svs` (CC0, about 170 MB), needs approval |
| Generic pyramidal TIFF / OME-TIFF | none. Candidates: OpenSlide `Generic-TIFF/CMU-1.tiff`, or a pyramid generated with tifffile |
| Multi-scene real CZI | none. Generated 2-scene CZI only |
| CZI with non-integer pyramid ratios | none |
