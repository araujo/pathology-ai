# Report 004 — Real WSI Preprocessing & Patch Extraction

**SPEC:** `../specs/004-real-wsi-preprocessing-and-patch-extraction.md` (Accepted 2026-10-16, AC-001–AC-018)
**ADR:** `../adr/004-real-wsi-preprocessing-architecture.md` (Accepted 2026-10-16, D1–D15, R1–R8)
**Architecture notes:** `../architecture/004-preprocessing-pipeline.md`, `../architecture/004-legacy-patch-compatibility.md`, `../architecture/004-reader-conformance-matrix.md`
**Implementation date:** 2026-10-18. macOS arm64, Python 3.12.13, uv 0.11.17, numpy 2.5.4, pylibczirw 6.1.0, openslide-python 1.4.6 with OpenSlide 4.0.1.
**Phase A assessment:** preserved verbatim in Appendix A.

---

## 0. Governance repair (R0)

Phase A found that the three 004 governance files held the wrong content. The repair was:

1. Every file was backed up before any replacement.
2. The original SPEC-004 was supplied by the owner as `.claude/specs/004-original.md` and restored to `specs/004-…md`.
3. AC-017 and AC-018 were appended. The DoD gained items 11–12. §18 (cross-format compatibility) and §19 (status) were added. `004-reader-conformance-matrix.md` was added to the deliverables.
4. The ADR text was moved from the misplaced spec file into `adr/004-…md`, with D12–D15 and the approved R1–R8 added.
5. The addendum was integrated verbatim into PROMPT-004 as §10, with its headings demoted.

Verification was scripted:

- the **AC-001–AC-016 block is byte-identical** to the owner's file;
- the ACs are numbered AC-001…AC-018 and the decisions D1…D15, both contiguous;
- every original line of the spec, ADR and prompt is present, apart from the intended status and header lines;
- every addendum line is present in the prompt.

SPEC-004 and ADR-004 are marked **Accepted (approved 2026-10-16)**. The owner's source file `.claude/specs/004-original.md` was left in place.

---

## 1. Executive outcome

Real whole-slide images now pass through the **existing `PipelineRunner`**. The pipeline is `wsi-tissue-patch-extraction@1.0.0` with three typed, versioned operations:

- `tissue.detect@1`
- `patch.plan@1`
- `patch.extract@1`

It produces a candidate-tissue mask, a deterministic patch plan, real patch pixel artifacts and a versioned, schema-validated patch manifest. No model is involved.

| Fixture | Reader | Mask | Read level | Candidates | Selected | Manifest digest |
|---|---|---|---|---|---|---|
| F-CZI-01 (CZI, 40470×13109) | czi-pylibczirw / pylibczirw 6.1.0 | stored level 5, 1264×409, 32 px cells | 2 (0.44496 µm/px) | 468 | **134** | `6360e193e8d82b4b36814b0b7326bb83eba1fc9b43ad8886d08f0bc547ae9fe8` |
| F-SVS-01 (SVS, 2220×2967) | openslide / 1.4.6+lib4.0.1 | level 0 block-reduced ×3, 740×989 | 0 (0.499 µm/px) | 88 | **25** | `bcca2a48e6b465c77acdf38254ffb5d3788c61291c96bacfd121fcf5afe2a45b` |

The same pipeline definition serves both formats. The read level is resolved per image from the 0.5 µm/px target, never assumed. Repeated runs, and runs from the installed wheel, reproduce these digests exactly.

**Test and build results:**

- 342 tests in total (96 new).
- **340 passed and 2 skipped** with both real fixtures. The 2 skips are explicit UNSUPPORTED cases of the in-memory reference reader.
- 306 passed and 36 skipped without fixtures. Every skip is "fixture not configured" or UNSUPPORTED.
- ruff, ruff format and strict mypy are clean. `uv build` succeeds.
- The synthetic demo output is byte-identical to the baseline.

All 18 acceptance criteria are **PASS** with the evidence below.

The candidate-tissue detector is a colour heuristic. It is **not** tumour detection, it is not scientifically validated, and nothing here supports a clinical claim.

---

## 2. Architecture and changed files

### 2.1 Added

| Path | Purpose |
|---|---|
| `domain/capabilities.py` | `PreprocessingRequirements`, `CapabilityCheck`, `ValidityPolicy`, `require_capabilities` (AC-018) |
| `domain/tissue.py` | `MaskGrid`, `TissueMask` |
| `domain/patching.py` | `PatchSpec`, resolution, grid, decisions, selection, `PatchPlan` |
| `domain/manifest.py` | `PatchRecord`, `PatchManifest`, `PixelStorage` |
| `domain/frames.py` | `FrameTranslation`, `padded_origin_translation` |
| `application/preprocessing_service.py` | `WSI_PREPROCESSING_TASK` (R3), `PreprocessingService` |
| `plugins/preprocessing/{__init__,arrays,detection,operations}.py` | NumPy-only implementation (R1) |
| `presentation/api/preprocessing_v1.py`, `presentation/api/contracts/preprocessing/v1/patch-manifest.schema.json` | canonical manifest JSON, verified parser, JSON Schema |
| `pipelines/wsi-tissue-patch-extraction.v1.yaml` | real-WSI pipeline definition |
| `tests/preprocessing_support.py` | in-memory `OpenedImage` (spy-capable) and reader registration |
| `tests/unit/test_preprocessing_contracts.py` (20), `tests/unit/test_tissue_detection.py` (13), `tests/integration/test_preprocessing_pipeline.py` (20), `tests/integration/test_real_wsi_preprocessing.py` (6, opt-in), `tests/contract/test_reader_conformance.py` (37) | new tests |
| `tests/fixtures/legacy/f-czi-01-legacy-patch-coords.json` | 66 legacy coordinates (numbers only) with source and extraction provenance (R7) |
| `.claude/architecture/004-*.md` | three architecture notes |

### 2.2 Changed (all additive)

| Path | Change |
|---|---|
| `application/runner.py` | Optional `ExecutionRequest.source: OpenedImage` and `config_overrides` (R2, R6). Overrides are validated: the key must exist and keep its type. A capability gate runs before any stage. `source:opened_image` is reported as missing for gated pipelines. `PatchManifestPayload` auxiliary artifacts are stored. Overrides enter `config_digest` only when present. No backend import. |
| `ports/__init__.py` | `StageContext.source/pipeline`; `CapabilityGated`, `TissueDetector`, `DetectorInput`, `DetectorResult`, `PatchManifestPayload` |
| `domain/errors.py` | `PREPROCESSING_CAPABILITY`, `UNSUPPORTED_RESOLUTION`, `PATCH_BUDGET_EXCEEDED`, `PREPROCESSING_BUDGET_EXCEEDED`, `NO_PATCHES_SELECTED`, `NO_ACQUIRED_DATA`, `NO_TISSUE_DETECTED`, `INVALID_MANIFEST` |
| `domain/kinds.py` | `patch_plan`, `patch_manifest` |
| `domain/task.py` | `TaskType.PREPROCESSING` and `OutputKind.PATCH_MANIFEST`. Never serialized by the Analysis API, because a preprocessing run returns no `PredictionResult`. |
| `plugins/imaging/czi/reader.py` | Acquisition fact `native.padded_origin` (R8). The canonical frame and Imaging API v1 are unchanged. |
| `bootstrap.py` | Registers the real backends in the existing `ReaderRegistry` by format code; adds the three operations, the detector, the task and `Platform.preprocessing_runs` |
| `cli.py`, `pyproject.toml` | `pathology-ai-preprocess` entry point |

Unchanged:

- the synthetic operations and pipelines;
- all API v1 schema files;
- the wheel `only-include` list;
- the dependency list (NumPy only, no OpenCV or scipy).

---

## 3. Legacy implementation assessment

The detailed table is in Appendix A §3. Classification under SPEC-004 §12:

| Legacy behaviour | Classification |
|---|---|
| Separating no-data, tissue coverage and glass (`patch_validity.py`) | **Adapted**: independent acquired and tissue fractions, both recorded with their reasons |
| Saturation-labelled glass vs stain (`saturation_labelled_hue_split/v1`) | **Adapted**: Otsu on saturation over acquired cells is the baseline. The legacy split for this slide was 21 at level 6; the new threshold is 26 at level 5. |
| BGR→HSV hue threshold with per-slide `sthresh`, median blur, cv2 contour and hole filters | **Rejected**: stain- and cv2-specific (R1). Hole and minimum-area filtering is **not yet implemented**. |
| Padded `total_bounding_rectangle` frame and `use_padding=True` | **Rejected** as a frame. **Bridged** explicitly (§6). |
| 2013 px level-0 read then `INTER_AREA` resize to 224 | **Rejected** for generic extraction (D6). Model-bound resizing is deferred to the embedding stage. |
| Exact-area coverage per patch | **Adapted**: exact area-weighted fractions on the mask grid (§5) |
| Zarr/H5 patch stores, S3 materialisation | **Not yet implemented** (R5) |

---

## 4. Tissue detection design and limitations

The design is in `004-preprocessing-pipeline.md` §3.

**Explicit semantics recorded on every mask:**

- channel view (`bgr->rgb` or `rgba->rgb`, detector-internal);
- validity semantics and policy;
- derived threshold;
- grid geometry and source level;
- decoded pixel count;
- SHA-256 mask digest;
- capability checks;
- the notice "candidate tissue… not tumour detection".

**Measured on the real fixtures:**

| | F-CZI-01 | F-SVS-01 |
|---|---|---|
| Mask source | stored level 5 (ds 32), 516,976 px decoded in tiles | level 0 (single level), 6,586,740 px decoded in 2048² tiles, block factor 3 |
| Grid edge adjustment (level-0 px) | (+22, +21): last column and row stretched to the image edge | (0, 0) |
| Otsu / threshold | 26 / 26 | 58 / 58 |
| Tissue fraction of image | 0.2604 | 0.2870 |
| Acquired fraction | 0.6453 (scanner coverage) | 1.0 (alpha) |

**Limitations:**

- Saturation is a brightfield colour cue. Pale or faint stains, pen marks, bubbles and coloured debris can be misclassified.
- There is no hole or minimum-region filtering.
- Mask cells at ds 32 blur boundaries finer than about 3.6 µm on F-CZI-01.
- A block-reduced cell counts as acquired at ≥ 50% coverage.
- No qualitative or quantitative comparison with expert annotation was done (out of scope; AC-015).

---

## 5. Patch planning and filtering

The default configuration in the YAML is:

- 256×256 px patches, stride equal to size;
- `target_spacing_um` 0.5 with `spacing_tolerance` 0.15;
- `inside_only`;
- `min_tissue_fraction` 0.5, `min_acquired_fraction` 0.9;
- `require_reported` validity;
- `fail_closed` selection with `max_patches` 512.

| | F-CZI-01 | F-SVS-01 |
|---|---|---|
| Resolved level and effective spacing | 2 / 0.44496 µm/px (11.0% from target; no resampling) | 0 / 0.499 µm/px (0.2%) |
| Grid | 39 × 12 = 468 (remainder strips: 133 px right, 205 px bottom, at level 2) | 8 × 11 = 88 |
| Selected | 134 | 25 |
| Rejected: insufficient tissue | 334 | 63 |
| Rejected: insufficient acquired data | 191, all also low in tissue | 0 |
| Not selected (budget) | 0 | 0 |

**Patch geometry examples (canonical level-zero):**

- F-CZI-01, first selected patch: `pai-patch-c359c7b347e16db7c3b03bdb2a3a9b57`. Level-2 rect (9728, 0, 256×256), level-0 rect (38912, 0, 1024×1024). Tissue 0.7295, acquired 0.9727 (planned) and 0.9727 (read).
- F-SVS-01, first selected patch: `pai-patch-439c49d697c6e71107de968b284badaa`. Level-0 rect (1024, 256, 256×256). Tissue 0.5469, acquired 1.0.

**Behaviour covered by tests:**

- Row-major order, unique positions, nothing out of bounds, non-divisible remainders.
- `align_to_edge` flush patches.
- Stride and overlap, grid origin.
- Non-integer downsample (1.5).
- Independent rejection reasons.
- `fail_closed`, `row_major_first` and `uniform_stride`.
- `max_candidates` checked before materialising.
- Distinct empty-result errors: no tissue, no acquired data, all rejected. These are recorded as FAILED runs with those error codes.

---

## 6. Coordinate compatibility

Details are in `004-legacy-patch-compatibility.md`.

- **Padded origin:** the CZI adapter reports (-262080, 144720) for F-CZI-01. The level-zero origin is (-262063, 144720). The translation is level0 = legacy + **(-17, 0)**, computed from the two origins, never assumed.
- **Legacy coordinates (R7):** the 66 coordinates were read with the legacy environment's own zarr, `mode='r'`, with bytecode writing disabled. The legacy `git status --porcelain --ignored` hash was `7d01186a…` before and after.
- **Placement:** 58 inside and 8 crossing bounds. Legacy x = 0 begins 17 px left of the image; legacy y = 12078 + 2013 px exceeds height 13109.
- **Pixel proof:** the legacy-style raw read (`roi = padded origin + legacy`, zoom 1) is **byte-identical** to the adapter's read at the translated level-zero rectangle, for 2 patches. Untranslated reuse reads different pixels (MAE 9.99) and is asserted unequal.
- **Not universal:** a unit test shows a different origin pair giving a different offset. OpenSlide has no native frame, so the bridge is unavailable for SVS, by design.

---

## 7. Extraction and storage

- **Lazy reads:** one bounded `read_region` per selected patch at the plan level, in batches of 32. Before the first read, the extractor checks `max_patches` (512), `max_batch_bytes` (64 MiB) and `max_output_bytes` (256 MiB).
- **Native representation:**
  - F-CZI-01 patches are BGR uint8 (256, 256, 3).
  - F-SVS-01 patches are RGBA uint8 (256, 256, 4).
  - Decoder transformations are recorded per patch.
  - No conversion, resizing or normalisation is applied (D6).
- **Pixel correctness:** for 3 patches per fixture (first, middle, last), the extracted pixels and the full read provenance equal an independent direct read through the reader.
- **Storage:**
  - `pixel_storage = in_memory`. Pixel artifacts are `image.region_2d` values in the process-local `InMemoryArtifactStore`, listed in `PipelineRun.artifacts` (137 for CZI: mask, plan, manifest and 134 patches; 28 for SVS).
  - The manifest states that **artifact ids are not evidence of persisted pixel bytes**.
  - `materialize_pixels: false` produces a manifest-only result (`not_materialized`), with no patch reads.
- **Size:** the manifest is canonical JSON, 126,428 bytes for CZI and 25,922 bytes for SVS. It contains no file paths. It validates against `patch-manifest.schema.json` with 0 errors and round-trips through `manifest_from_dict`, which rejects tampered digests, missing fields and unknown schema versions with `INVALID_MANIFEST`.

---

## 8. PipelineRunner execution evidence

```text
PreprocessingService.run → ImagingService.open (content probe, partial-content fingerprint)
  → PipelineRunner.run(ExecutionRequest(pipeline=wsi-tissue-patch-extraction@1.0.0,
        task=wsi-preprocessing@1, models={}, source=<OpenedImage>, research_use_acknowledged=True))
  → validate → capability SUPPORTED_UNVALIDATED ("no predictive model in pipeline")
  → usage RESEARCH_UNVALIDATED → executability → gate → tissue → plan → extract
  → PipelineRun SUCCEEDED; handle closed by the service (`with` block; fd count back to baseline)
```

- **Stage versions** recorded in the run: `operation tissue.detect@1 0.1.0`, `operation patch.plan@1 0.1.0`, `operation patch.extract@1 0.1.0`, and the reader component.
- **Bounded reads:** a spy test on F-CZI-01 shows level 0 is **never** read and no request exceeds 2048². Mask-source reads total exactly 1264 × 409 px, and there are exactly 134 level-2 patch reads.
- **CLI from the installed wheel:** `pathology-ai-preprocess` (with `PATHOLOGY_AI_ROOT`) reproduces the digests in §1.

---

## 9. Real CZI and SVS fixture evidence

**Fixtures:** the same files as SPEC-003. Size and SHA-256 are enforced by `fixture_path`.

- F-CZI-01 `8634d055…be4`
- F-SVS-01 `ed92d5a9…8c7`

| Test (`tests/integration/test_real_wsi_preprocessing.py`) | Result |
|---|---|
| `test_czi_runs_through_the_pipeline_engine`: mask level and geometry, threshold, fractions, level 2, 468/134/334/191, first patch geometry, pixel equality | PASS |
| `test_czi_manifest_is_repeatable_and_serializable`: mask, plan and manifest digests and patch artifact ids equal across runs; JSON round trip | PASS |
| `test_czi_reads_are_bounded_and_handles_released`: spy, file descriptors | PASS |
| `test_legacy_padded_coordinates_translate_explicitly` | PASS |
| `test_svs_runs_through_the_pipeline_engine`: block factor 3, 740×989, alpha validity, 88/25, RGBA patches, pixel equality, repeat digest | PASS |
| `test_same_pipeline_definition_serves_both_formats`: same pipeline, schema and spec; levels 2 vs 0 | PASS |

The SVS fixture is **single-level**. Multilevel SVS behaviour is not validated (§14).

---

## 10. Determinism and provenance

| Identity | F-CZI-01 | F-SVS-01 |
|---|---|---|
| mask digest | `5558100635544cac28335029cd492afcda5d85cfab17309330fc521a7b3f5420` | `297eddb662cc30e65effe830e130384eff402dedd0c6c3e7d2aa94dab2747a9a` |
| plan digest | `e65bde7799aa13e323d6553b6864b7d97a55456c4893bfab8a91d9de330b955e` | `91f1ed0977e61c9e63ed519e982a6199167ff8a42ba97b9ccc21ce8d25eec4de` |
| manifest digest | `6360e193e8d82b4b36814b0b7326bb83eba1fc9b43ad8886d08f0bc547ae9fe8` | `bcca2a48e6b465c77acdf38254ffb5d3788c61291c96bacfd121fcf5afe2a45b` |
| provenance digest | `0782d0fa162cdce933b77eb439158cbf089c5eedca9ddade5cff512f557465b9` | `128e5d64dceb8c8eb05bd8d89b3f1b673f17094d8159efa40382d460f8354da1` |
| first patch artifact | `pai-art-20e440ceba1b2e3776bf8e5b24a17df1` | `pai-art-652171bebb69bc5079b6d9e8a12c75df` |

**Repeatability evidence:**

- Identical across repeated runs in one process and in fresh processes.
- Identical before and after a memory-only refactor of block reduction.
- Identical between the source checkout and the installed wheel.

**What the identities depend on:**

- Digests contain no timestamps. They depend on the image identity and integrity token, components, configuration and plan.
- A different image id changes every identity (test).
- An override changes both the run fingerprint and the manifest digest (test).
- Runs without overrides keep their previous fingerprints. Evidence: the synthetic demo hash `11a211de…` is unchanged.

---

## 11. Resource limits

| Limit | Enforcement | Evidence |
|---|---|---|
| Request size | `ReadLimits` in the reader. Mask reads are tiled at ≤ `tile_px`². | spy tests (in-memory and F-CZI-01); SPEC-003 limit tests |
| Decoded pixels for detection | `max_decoded_pixels` (64 Mpx) → `PREPROCESSING_BUDGET_EXCEEDED` before reading | unit and integration tests |
| Mask size | `max_mask_pixels` (1 Mpx) → coarser level or block factor | unit tests; real fixtures |
| Candidates | `max_candidates` → `PATCH_BUDGET_EXCEEDED` before materialising | tests |
| Selected patches | `max_patches` (plan) with a recorded policy; `max_patches` (extract) | tests |
| Batch bytes / output bytes | `max_batch_bytes`, `max_output_bytes`, checked before the first patch read | tests (no patch read happens) |
| Full level-0 decode | never as a single read; only tiled when it is the only stored level | spy tests |

**Measured** in a fresh process, wall time and memory for the pipeline run only:

| | Time | tracemalloc peak | max RSS (start → end) | Retained patch bytes |
|---|---|---|---|---|
| F-CZI-01 | 2.07 s | 59.0 MiB | 45 → 142 MiB | 25.1 MiB |
| F-SVS-01 | 0.35 s | 108.6 MiB | 45 → 272 MiB | 6.2 MiB |

For comparison, a full level-0 decode of F-CZI-01 would need about 1.59 GB. The SVS peak is dominated by 2048² RGBA tiles (decoder plus copies) and can be reduced with `tile_px`. Peak memory is measured, not asserted by an automated test.

---

## 12. Tests and packaging

**Commands (2026-10-18):**

```bash
uv sync                                   # OK
uv run ruff check .                       # All checks passed
uv run ruff format --check .              # 114 files already formatted
uv run mypy                               # Success: no issues found in 92 source files
uv run pytest -q -W error -rs             # 306 passed, 36 skipped (fixtures not configured / UNSUPPORTED)
PATHOLOGY_AI_WSI_FIXTURE_CZI=… PATHOLOGY_AI_WSI_FIXTURE_SVS=… \
  uv run pytest -q -W error -rs           # 340 passed, 2 skipped (UNSUPPORTED: in-memory reference)
uv build                                  # sdist + wheel OK
```

**Baseline before SPEC-004:** 235 passed and 10 skipped without fixtures; 245 passed with fixtures. All of these still pass.

**Wheel checks:**

- The wheel contains the new domain, plugin and presentation modules and `patch-manifest.schema.json`. It contains no `pipelines/`, `projects/` or `tests/`.
- Clean venv with `[czi,openslide]`: `pathology-ai-preprocess` on both fixtures gives exit code 0 and the same manifest digests. Its manifests validate (0 schema errors) and round-trip.
- Base wheel without extras: exit code 3, `READER_BACKEND_UNAVAILABLE … install pathology-ai[czi]`.
- `pathology-ai-demo` from the wheel produces output with SHA-256 `11a211de…`, identical to the baseline.

**Repository independence:**

- `digital-pathology` HEAD `c85df8a9…` with status hash (including ignored files) `7d01186a…` is unchanged.
- `hal-viewer` HEAD `e2482029…` with 9 status lines is unchanged.
- `pathology_ai/` does not exist.

---

## 13. Acceptance criteria

| AC | Result | Evidence |
|---|---|---|
| AC-001 Reader-agnostic preprocessing | **PASS** | Operations use only `OpenedImage` and domain contracts. The core-forbidden-terms and layer AST tests pass. The same pipeline passes the conformance suite on 4 subjects (§13.1). |
| AC-002 Real tissue candidate detection | **PASS** | `saturation_otsu` v1 on F-CZI-01 (BGR, coverage validity) and F-SVS-01 (RGBA, alpha validity), with explicit channel views and validity handling (§4) |
| AC-003 Tissue mask geometry | **PASS** | `MaskGrid` unit tests (stored, block-reduced, non-integer ds, clamped and stretched edges). The real CZI grid is asserted: 1264×409, 32 px cells, edge adjustment (22, 21). Exact area fractions are tested. |
| AC-004 Deterministic patch planning | **PASS** | Row-major unique grid. Identical plans and digests across runs, processes, fixtures and the wheel (§10). |
| AC-005 Resolution semantics | **PASS** | Pixel vs micrometre units, target spacing with tolerance, `UNSUPPORTED_RESOLUTION`, missing-spacing failure (gate plus domain), explicit level without spacing; recorded effective spacing and relative error |
| AC-006 Tissue and acquisition filtering | **PASS** | Independent thresholds and reasons (unit). On F-CZI-01, 191 candidates were rejected for acquisition. NOT_REPORTED validity requires an explicit policy. |
| AC-007 Real patch extraction | **PASS** | 134 CZI and 25 SVS patches extracted through the real readers. Shape, dtype and channel order are correct; pixels and provenance equal direct reads (§7). |
| AC-008 Resource safety | **PASS** | All budgets in §11 are enforced before decoding, with typed errors. Spy-verified request bounds. Peak memory is measured (not asserted). |
| AC-009 Patch manifest | **PASS** | `pathology-ai.patch-manifest/1`: selected records, rejection counts by reason, artifact references, provenance and manifest digests. Canonical JSON, JSON Schema, verified parser, tamper rejection. |
| AC-010 Artifact integrity | **PASS** | Content-addressed pixel artifact ids are stable across runs and lineage-linked to the image. `pixel_storage` states in-memory, non-durable storage; manifest-only mode has no artifact references; validation forbids contradictions. |
| AC-011 PipelineRunner execution | **PASS** | §8. Registered pipeline YAML; no runner backend imports; a FAILED run is recorded for empty results. |
| AC-012 Legacy coordinate compatibility | **PASS** | Explicit translation from reported origins: (-17, 0) on F-CZI-01. 66 legacy coordinates placed (58 / 8). Byte-identical pixels for 2 translated patches; naive use proven wrong; a non-universal offset is unit-tested. |
| AC-013 Existing behaviour preservation | **PASS** | All 245 prior tests pass (with fixtures). Synthetic projects run unchanged (test). The demo is byte-identical in the checkout and the wheel. |
| AC-014 Repository independence | **PASS** | No import of or runtime reference to legacy repositories (architecture tests). Sibling repositories are unchanged (§12). Legacy data is only a numbers-only fixture with provenance. |
| AC-015 Scientific semantics | **PASS** | Mask and manifest notices; "candidate tissue" wording; `research_unvalidated` usage mode; no probabilities, scores or predictions are produced (`prediction is None`). |
| AC-016 Reproducibility evidence | **PASS** | Configuration (§5), fixture identity (§9), counts (§1, §5), spatial examples (§5, §6), digests (§10), repeatability (§10), commands (§12) |
| AC-017 Cross-format reader conformance | **PASS** (scoped to the tested subjects) | `test_reader_conformance.py`: 35 PASS and 2 UNSUPPORTED across gen-czi, F-CZI-01, F-SVS-01 and array-rgba. Exact cross-backend equivalence on identical lossless content. Matrix: `004-reader-conformance-matrix.md`. |
| AC-018 Capability-gated execution | **PASS** | The runner gate checks every gated operation before any stage. Grayscale, unreported validity, missing spacing, multi-plane, missing region support, missing source and mismatched source each raise a typed error with zero pixel reads where applicable. `assume_acquired` is an explicit, recorded decision. |

### 13.1 Reader conformance summary (addendum §12)

- **Results by backend:** gen-czi 9/9 PASS; F-CZI-01 9/9 PASS; F-SVS-01 9/9 PASS; array-rgba 7 PASS and 2 UNSUPPORTED (no file, no handle). The cross-backend equivalence test passes. Without fixture variables, the F-CZI-01 and F-SVS-01 columns are NOT_TESTED.
- **Capability-gating tests:** 11 situations; see the matrix §2.
- **Assumptions found and eliminated:**
  - a fixed stored level;
  - BGR input;
  - black meaning no data;
  - alpha treated as pixel data;
  - a pyramid always existing;
  - native or padded frames equal to level 0;
  - `NOT_REPORTED` meaning fully acquired.

  Details are in the matrix §4.
- **Unsupported or untested operations:**
  - grayscale and multichannel detection (unsupported);
  - Z/T > 1 (unsupported);
  - resampling (not performed);
  - real-file `NOT_REPORTED` validity (untested);
  - multilevel SVS (untested);
  - multi-scene real CZI (untested).
- **Fixture coverage gaps:** see the matrix §6.

---

## 14. Known limitations

1. **Detector quality is unvalidated.** Saturation and Otsu is a brightfield heuristic with no hole or minimum-region filtering. It is likely to misbehave on pale stains, pen marks and artefacts.
2. **Fixture coverage:**
   - only one real CZI and one **single-level** SVS;
   - multilevel SVS, JPEG 2000 SVS, multi-scene real CZI and non-integer pyramid ratios on real files are untested;
   - no real same-content pair across formats exists.
3. **Persistence:** pixels live only in memory (R5), and the batch budget does not reduce total retention. `max_output_bytes` bounds it. No on-disk patch store exists.
4. **Resolution:** without resampling, the effective spacing can differ from the target by up to the tolerance (11% on F-CZI-01), and physical patch size differs across fixtures: 113.9 µm for CZI vs 127.7 µm for SVS. Both are recorded.
5. **Mask resolution:** the CZI mask uses ds 32 cells. Edge cells are stretched by up to 22 px.
6. **Memory:** there is no automated peak-memory assertion. SVS tiles of 2048² RGBA cause about 270 MiB peak RSS.
7. **Config sentinels:** stage config holds only scalars, so "unset" uses sentinels (`level: -1`, `target_spacing_um: 0.0`, stride `0.0`).
8. **Legacy bridge:** the bridge covers CZI padded frames only. Legacy patch geometry (2013 px resized to 224) is not reproduced.
9. **Remaining duplicate:** `.claude/specs/004-original.md` still exists alongside the restored SPEC. It can be removed at the owner's discretion.

---

## 15. Recommendations for SPEC-005

1. **Embedding handoff:** consume `patch_manifest` plus the `image.region_2d` artifacts. Bind a model-specific preprocessing contract there (channel conversion, resize to model input, normalisation), using the existing `PreprocessingContract.resolve_channel_conversion`.
2. **Durable patch store:** add a plugin (for example `.npy` or Zarr chunks) whose manifest records storage location and byte digests, switching `pixel_storage` to a durable value only when bytes are verified on disk.
3. **Detector evaluation:** compare candidate masks against a small annotated set before any scientific use. Consider an optional cv2/scikit-image detector plugin for hole and region filtering, as a separate dependency decision.
4. **Fixtures:** obtain a multilevel SVS (OpenSlide `CMU-1.svs`, CC0) and a pyramidal TIFF to extend the conformance matrix. Consider a same-content multi-format pair, for example converting a CC0 slide losslessly.
5. **Resampling policy:** if a model needs an exact physical spacing, add an explicit, versioned resampling step at the model-bound stage, never in generic extraction.
6. **Typed stage config:** consider optional or nullable config values in the pipeline schema to replace the sentinels.

---

## Appendix A — Phase A assessment (2026-10-16, before approval; preserved verbatim, headings demoted)

**Phase:** A (assessment only). No implementation code has been written.
**Date:** 2026-10-16
**SPEC / ADR status:** both **Proposed**. Per PROMPT-004 §1 and the addendum's §13, implementation (Phases B–E) waits for owner approval.

---

### 0. Blocking document issue

The three governance files do not contain what their names say:

| File | Actual content |
|---|---|
| `.claude/specs/004-real-wsi-preprocessing-and-patch-extraction.md` | **ADR-004** text ("# ADR 004 — Real WSI Preprocessing Architecture", D1–D11, Proposed) |
| `.claude/adr/004-real-wsi-preprocessing-architecture.md` | **PROMPT-004 Addendum** (cross-format conformance, AC-017/018, D12–D15) |
| `.claude/prompts/004-…md` | PROMPT-004 (correct) |

**The SPEC-004 text itself is missing.** That includes AC-001 to AC-016, which the final report must score. The addendum asks for edits to the SPEC, ADR and prompt files. I did not edit them: they are Proposed, and overwriting the misplaced content would destroy the only copy of the ADR.

**Owner action needed:** restore SPEC-004. Then move the ADR text into `adr/004-…`, and move the addendum into `prompts/` (or append it to PROMPT-004). Without the SPEC acceptance criteria, this plan is traced to the ADR decisions (D1–D15) and to the prompt's §4–§6 instead.

---

### 1. Baseline (before any change)

| Check | Result |
|---|---|
| `uv sync` | OK (30 packages) |
| `ruff check .` / `ruff format --check .` | pass / 95 files formatted |
| `mypy` (strict) | no issues, 75 files |
| `pytest -q -W error -rs`, no fixture variables | **235 passed, 10 skipped** (all skips are explicit "real fixture not configured") |
| Same, with `PATHOLOGY_AI_WSI_FIXTURE_CZI` and `PATHOLOGY_AI_WSI_FIXTURE_SVS` set | **245 passed, 0 skipped** |
| `pathology-ai-demo` stdout SHA-256 | `11a211dec82f27688eebaad3907a0d19791b19daf2feb50d5cd3cfaca89df3da`, identical on 2 runs (83 lines) |
| `digital-pathology` HEAD | `c85df8a966e8cc063a61c509d038f1b1cae89c3a`. 4 untracked entries, all pre-existing. Read-only commands only. |
| `hal-viewer` HEAD | `e248202983bc6cdb497ed00ef0168e6b47338e09`. 9 status lines, pre-existing. Not opened. |

---

### 2. Current-state findings

#### 2.1 Pipeline engine
- `PipelineRunner` (`application/runner.py`) validates, authorises, executes stages in topological order, derives content-addressed `ArtifactIdentity` from stage config and lineage, and records a `PipelineRun` even when a run fails. Stage config values are scalars only (`str | int | float | bool`).
- With no predictive model, the capability decision is `SUPPORTED_UNVALIDATED`, so a preprocessing-only run requires `research_use_acknowledged=True`. This is acceptable and honest.
- `ExecutionRequest` requires a `TaskDefinition` even when no model runs. A preprocessing-only task, or an optional task, is needed (**decision R3**).
- **Gap (blocking for Phase D):** the runner resolves an `ImageReader` from `ReaderRegistry` by format code, and passes that unopened reader to stages. Real backends (`ImageBackend`) are opened only by `ImagingService`, from an `ImageSource` path, and the resulting `OpenedImage` handle never reaches `StageContext`. Report 003 §14 already notes that `PipelineRunner` cannot run on real images.

#### 2.2 Readers and capabilities
- `CziBackend` and `OpenSlideBackend` already satisfy the `ImageReader` protocol (`component`, `capabilities`). They can be registered in `ReaderRegistry` under `zeiss-czi` and `aperio-svs` without touching the synthetic reader.
- `ImageDescription` already states channel order, dtype, samples per pixel, scene, Z/T planes, spacing and its source, `NoDataSemantics`, decoder transformations and missing metadata. `PixelRegion` carries `ValidDataReport`, with a mask for both adapters. These cover all ten properties in addendum §3, so **no second capability registry is needed**.
- Channel orders differ: CZI returns **BGR uint8 ×3**; SVS returns **RGBA uint8 ×4**, where alpha is the validity signal.
- No-data semantics differ: CZI uses `ACQUISITION_COVERAGE` (level-zero subblocks); SVS uses `BACKEND_TRANSPARENCY`. `NOT_REPORTED` is possible for future readers.
- `ReadLimits` (default 4096²) is enforced before any decoding.

#### 2.3 Synthetic operations (must stay unchanged)
`tissue.threshold@1` and `patch.grid@1` (`plugins/synthetic/operations.py`):
- They hard-require `SyntheticSlideReader`.
- They use mean intensity below 220.
- They require a pyramid level that exactly matches the model contract's spacing.
- They emit `PatchSet` in the model's channel order.

These operations will not be modified. Real operations get new names (§3.6).

#### 2.4 Architecture constraints that shape the design
- `plugins` may import only **numpy**. Any OpenCV, scipy or scikit-image detector is a new dependency (**decision R1**).
- `domain`, `ports` and `application` must not contain `czi`, `zeiss`, `aperio`, `openslide`, `pylibczirw`, `h&e`, `surgen`, and similar terms (AST and regex tests).
- `infrastructure` may import only `yaml`, so a pixel-writing patch store would have to be a plugin (**decision R5**).

#### 2.5 Fixture probe (throwaway script in the session scratchpad; not repository code)
I ran a numpy-only saturation + Otsu detector through the existing adapters:

| Fixture | Mask source | Acquired | Otsu(sat) | Tissue / image | Tissue / acquired | Time |
|---|---|---|---|---|---|---|
| F-CZI-01 | stored level 6, 632×204, ds 64 | 0.647 | 23 | 0.304 | 0.470 | 0.10 s |
| F-SVS-01 | only level 0, 2220×2967, ds 1 | 1.000 | 63 | 0.279 | 0.279 | 0.28 s |

Interpretation:
- The legacy saturation split on the same CZI slide and level was 21 (`t198_provenance.json`), which is consistent with 23.
- The baseline works without OpenCV.
- F-SVS-01 has no pyramid, so a whole-level read would be a full level-zero decode. That is small for this fixture, but forbidden as a strategy (D8). The detector therefore needs a **tiled block-reduction** path (§3.2).

---

### 3. Legacy assessment (`../digital-pathology`, read-only)

| Item | Legacy behaviour | Disposition |
|---|---|---|
| Tissue segmentation (`WholeSlideImage.segmentTissue`) | BGR→HSV, median-blurred **hue** > `sthresh`, closing, `findContours` with RETR_CCOMP, contour and hole area filters (`a_t`, `a_h` × reference patch area), `max_n_holes` | Not reused. Hue thresholds are stain-specific: the per-slide presets used 130–143. Contour/hole logic depends on cv2. |
| Threshold derivation (`tissue_segmentation.py`) | `saturation_labelled_hue_split/v1`: Otsu on saturation labels glass versus stained, then a hue cut | The **saturation-Otsu labelling** idea is adopted as the stain-neutral baseline. The hue cut is not adopted. |
| SR1482 preset (T198) | seg_level 6, mthresh 9, close 4, a_t 10, a_h 5, max_n_holes 24, `four_pt`, `use_padding=True`; sthresh 135 (CSV) or 143 (derived) | Recorded as a reference configuration only |
| Patch grid (`process_contour`) | Per-contour bounding box. Level-zero footprint `int(224·1.0/mpp)` = **2013 px** for F-CZI-01. `four_pt` eligibility. Padding allows patches past the image edge. | The planner uses a single global grid, never pads, and records the footprint explicitly. Padding becomes a rejection reason. |
| Patch read (`dataset_h5.py`) | `read(roi=offset+coord, zoom=1)`, then `cv2.resize(INTER_AREA)` to 224, BGR kept | This is model-bound resizing, so it is out of scope (D6). The planner records the footprint and the effective spacing instead. |
| Validity (`patch_validity.py`) | `nodata_fraction`, `in_image_fraction`, `tissue_coverage` capped at `1 - nodata`, `glass_fraction`; statuses VALID / NO_DATA / EXCESSIVE_NO_DATA | The **separation of semantics is adopted** (D3) under new neutral names (§4.1) |
| Coordinate frame | `total_bounding_rectangle` origin (padded): legacy = native − (−262080, 144720) | Bridge required (§4.7). For F-CZI-01, **legacy = level0 + (17, 0)**. |
| Existing legacy output for F-CZI-01 | `experiments/runs/spec019_real_validation/output/bags.zarr/SR1482_40X_HE_T198_01/coords` (66×2 int64, blosc-lz4). SPEC-024.5 run: 60 candidates, 59 retained. | This gives real evidence for the bridge, if decoding a blosc chunk read-only is approved (**R7**) |

---

### 4. Proposed design

#### 4.1 Domain contracts (`domain/tissue.py`, `domain/patching.py`; numpy-free, objects typed `object`)
- **`MaskGrid`**: level-zero geometry of a raster mask.
  - Fields: `origin` (always 0,0), `downsample_x/y` (level-zero px per mask px), `width`, `height`, `source` (`STORED_LEVEL(i)` or `BLOCK_REDUCED(level, factor)`).
  - Defines `to_level0(i, j)` and `footprint(rect) -> fractional cell coverage`.
- **`TissueMask`** (payload of `mask.tissue_2d`):
  - `grid: MaskGrid`, `tissue: object` (bool array), `acquired: object | None`, `validity_semantics: NoDataSemantics`.
  - `detector: ComponentVersion`, `parameters` (canonical tuple), `derived` (for example the Otsu threshold), `channel_view: TransformStep`, `limitations`.
  - Invariant: `tissue ⊆ acquired` whenever `acquired` is known. Acquired is never inferred from tissue.
- **`PatchSpec`**:
  - `level: int` (a stored level), `size_px`, `stride_px` (at that level).
  - Optional `target_spacing_um` with `spacing_tolerance`.
  - `min_tissue_fraction`, `min_acquired_fraction`, `unknown_validity_policy` (`REJECT` | `ASSUME_ACQUIRED_RECORDED`), `max_patches`, `selection` (§6, R4).
- **`PatchCandidate` / `PatchDecision`**:
  - `index` (row-major), `grid_xy`, `level_rect`, `level0: Level0Rect`.
  - `tissue_fraction`, `acquired_fraction | None`.
  - `status` ∈ {`SELECTED`, `REJECTED_TISSUE`, `REJECTED_ACQUISITION`, `REJECTED_VALIDITY_UNKNOWN`, `NOT_SELECTED_BUDGET`}.
- **`PatchPlan`** (payload of the new kind `patch_plan`): `spec`, image identity and integrity token, mask artifact id, ordered decisions, counts.
  - Deterministic `plan_digest`, computed with `domain.hashing.digest`.
- **`PatchRecord` / `PatchManifest`**:
  - Per selected patch: plan index, level-zero rect, level rect, the read's `RegionReadProvenance`, effective spacing (or `None` plus a limitation), pixel artifact id, measured acquired fraction of the actual read.
  - `manifest_digest` excludes timestamps.
  - A schema-versioned serializer `pathology-ai.patch-manifest/1` lives in `presentation/api`. It is **additive**; Imaging API v1 is not changed.
- **`PreprocessingRequirements`** (AC-018): required channel representations, `needs_regions`, `needs_physical_spacing`, `validity_policy`, `max_planes` (Z/T = 1), scene explicitness.
  - `check(description, capabilities) -> CapabilityCheck` gives per-requirement PASS / UNSUPPORTED / UNKNOWN with reasons.
  - The `require_*` form raises a new typed `PreprocessingCapabilityError` (code `PREPROCESSING_CAPABILITY`).

#### 4.2 Detector interface (port) and baseline
- **Port:** `TissueDetector.detect(image: OpenedImage, description, config) -> TissueMask`.
- **Baseline `tissue.saturation_otsu@1`** (numpy only):
  - **Mask source:** the coarsest stored level whose size is within `max_mask_pixels` (default 2²⁰ ≈ 1 M) is read in bounded tiles. If even that level is too large, as for single-level SVS, level 0 is read in bounded tiles and **block-mean reduced** by an integer factor. That factor is recorded in `MaskGrid.source`. A whole level is never decoded in one read when it exceeds `ReadLimits`.
  - **Channel view:** explicit and recorded.
    - `bgr→rgb`: index reversal.
    - `rgba→rgb`: drop alpha, after alpha has been taken as the acquired mask.
    - `grayscale` / `multichannel`: rejected (UNSUPPORTED), never guessed.
  - **Score:** saturation = (max − min)/max, uint8.
  - **Threshold:** Otsu over **acquired** pixels only, then `max(otsu, min_threshold)`, configurable. Both the threshold value and its source are recorded.
  - **Morphology:** optional, off by default. Binary opening and closing with a square kernel, implemented in numpy.
  - Optional small-component removal is deferred to R1. It needs connected-components labelling, which is not in numpy.
  - **Output:** `tissue = above_threshold ∧ acquired`.
  - **Limitations recorded:** brightfield, colour-based candidate tissue only, not histology, mask resolution.

#### 4.3 Patch planner (`patch.plan@1`, pure function plus a thin operation)
- One global grid at a **stored** level, starting at (0,0), stepping `stride_px`. Only full in-bounds patches are generated, so there is never padding.
- Physical spacing:
  - If `target_spacing_um` is set, it needs reported spacing (otherwise `MissingSpatialMetadataError`).
  - It also needs a stored level within tolerance (otherwise a typed error). No resampling is done (D5).
  - When it is unset, pixel-based patches are allowed, and the effective spacing is recorded as `None` if unknown.
- Fractions are computed by **exact area-weighted overlap** on the mask grid, using an integral image. This is the legacy "exact area" approach generalised to any mask resolution.
- Ordering is row-major (y, then x). Indices are stable for a fixed geometry and spec.
- The budget (`max_patches`, `max_candidates`) is enforced deterministically (R4).

#### 4.4 Extractor (`patch.extract@1`)
- Lazy and bounded: one `read_region` per selected patch at the plan level, in plan order, in batches (`batch_size`). `ReadLimits` applies to every read.
- Pixels are kept in the **reader's native representation** (BGR or RGBA) with full `RegionReadProvenance`. No conversion, resizing or normalisation happens here (D6).
- The acquired fraction of the actual read is re-measured from `ValidDataReport` and compared to the plan value. A mismatch is recorded, never hidden.
- Output budgets: `max_patches` and `max_output_bytes`, checked before reading.

#### 4.5 Artifact and storage semantics
- Each patch's pixels become an `image.region_2d` artifact. Its identity is derived from the image identity plus the patch's read provenance (the same rule as `ImagingService.ingest`), so identities are repeatable across runs.
- The mask, plan and manifest are separate artifacts with lineage `image → mask → plan → manifest`. They go through the existing runner, so they are stored in the shared `ArtifactStore` and listed in `PipelineRun.artifacts`.
- The manifest is complete without pixels: `extract_pixels=false` produces the manifest only.
- Disk persistence is limited to the JSON manifest export. A pixel store on disk is R5.

#### 4.6 Pipeline integration (minimal, additive runner change)
1. `bootstrap` also registers the real backends in `ReaderRegistry`. They are keyed by format code, so the synthetic reader is unaffected.
2. Add optional `source: OpenedImage | None = None` to both `ExecutionRequest` and `StageContext`. These are port types, so the runner gets no backend imports. The runner passes `source` through unchanged and adds a `PreprocessingRequirements` check before execution when the request carries a source.
3. A new `application/preprocessing_service.py`:
   - opens the image through `ImagingService.open` (content-based resolution, fingerprint, guaranteed close);
   - builds `ImageAsset` from `description.asset`;
   - runs `runner.run(...)` inside the `with` block;
   - returns the outcome plus the manifest.

   There is no second engine.
4. New operations in `plugins/preprocessing/` (numpy only, reader-neutral, no format branching): `tissue.saturation_otsu@1`, `patch.plan@1`, `patch.extract@1`. Each has an explicit `ComponentVersion`.
5. Pipeline file `pipelines/wsi-tissue-patch-extraction.v1.yaml`:

   ```yaml
   schema_version: pathology-ai.pipeline/1
   pipeline_id: wsi-tissue-patch-extraction
   version: "1.0.0"
   inputs: [{name: image, kind: image.pyramidal_2d, spatial_kind: pyramidal_2d}]
   stages:
     - id: tissue
       operation: {name: tissue.saturation_otsu, version: "1"}
       inputs: [{name: image, kind: image.pyramidal_2d, spatial_kind: pyramidal_2d}]
       outputs: [{name: mask, kind: mask.tissue_2d}]
       config: {max_mask_pixels: 1048576, min_threshold: 8, open_px: 0, close_px: 0}
     - id: plan
       operation: {name: patch.plan, version: "1"}
       inputs: [{name: image, kind: image.pyramidal_2d}, {name: mask, kind: mask.tissue_2d}]
       outputs: [{name: plan, kind: patch_plan}]
       config: {level: 2, size_px: 256, stride_px: 256, min_tissue_fraction: 0.5,
                min_acquired_fraction: 0.9, unknown_validity: reject, max_patches: 512,
                selection: row_major}
     - id: extract
       operation: {name: patch.extract, version: "1"}
       inputs: [{name: image, kind: image.pyramidal_2d}, {name: plan, kind: patch_plan}]
       outputs: [{name: manifest, kind: patch_manifest}]
       config: {batch_size: 32, max_output_bytes: 268435456, extract_pixels: true}
   edges: [[$input.image, tissue.image], [$input.image, plan.image], [tissue.mask, plan.mask],
           [$input.image, extract.image], [plan.plan, extract.plan]]
   outputs: {manifest: extract.manifest, plan: plan.plan, mask: tissue.mask}
   ```

   Patch level and size are configuration, not domain constants. The same file must run unchanged on CZI and SVS; for SVS this means `level: 0`, so the level is overridable or a parameterised variant (R6).
6. CLI: `pathology-ai-preprocess --image PATH --pipeline …` prints the manifest summary. Optional.

#### 4.7 Legacy coordinate bridge
- A domain `FrameTranslation(dx, dy)` with `legacy_to_level0` / `level0_to_legacy` and a declared `source` string.
- It is built from **two explicit native origins**: the legacy origin (the padded bounding-rectangle origin) and `description.native_offset`.
- The CZI adapter would report its padded origin as a non-identifying acquisition fact (`native_padded_origin`). This is an additive adapter change; the frame selection from SPEC-003 does not change.
- The bridge never guesses. Without a legacy origin it raises `MissingSpatialMetadataError`.
- Legacy padded patches that cross the image edge are reported as out-of-bounds. They are not clipped.
- Evidence, if R7 is approved: map the 66 legacy T198 coordinates and assert that every one lands at level0 = legacy − (17, 0), in bounds or explicitly flagged.

---

### 5. Tests and expected coverage

| Area | Tests |
|---|---|
| Contracts (unit) | `MaskGrid` mapping, including non-integer factors. `TissueMask` invariants (tissue ⊆ acquired; UNKNOWN validity carries no fake mask). `PatchSpec` validation. `PatchPlan` digest stability. Manifest canonical JSON round-trip. |
| Detector (unit, synthetic arrays) | Known saturation fields with an exact expected mask. BGR and RGB of the same scene give identical masks. RGBA alpha becomes the acquired mask. Grayscale and multichannel are rejected. Otsu uses acquired pixels only, so a black no-data region does not shift the threshold. Morphology on and off. |
| Planner (unit) | Exact area fractions at edges and with non-divisible mask factors. No out-of-bounds or padded patches. Row-major determinism. Every rejection reason. Budget truncation. Physical spacing missing or mismatched raises typed errors. |
| Capability gating (AC-018) | Generated CZI files (Gray8, Z=2, missing spacing, 2 scenes without explicit choice). A fake reader with `NOT_REPORTED` under `reject` and under `assume` policies. Each case gives the expected typed error or recorded decision. |
| Reader conformance (AC-017) | A parametrised suite `tests/contract/test_reader_conformance.py` run against `{generated CZI, F-CZI-01, F-SVS-01}`. Each check reports PASS / FAIL / UNSUPPORTED / NOT_TESTED. Results are written to `.claude/architecture/004-reader-conformance-matrix.md`. |
| Runner (integration) | Real CZI and SVS run through `PipelineRunner`. Repeated runs give identical plan, manifest and artifact ids. Run provenance lists the operation versions. Synthetic pipelines and the demo hash stay unchanged. |
| Resources | A spy reader records every request: none exceeds `ReadLimits`, and level 0 is never requested at full size. Patch and byte budgets. File-descriptor count back to baseline. tracemalloc peak bound for the extract stage. |
| Packaging | Wheel install runs the real pipeline with `PATHOLOGY_AI_ROOT`. The base wheel without extras reports `READER_BACKEND_UNAVAILABLE`. |
| Legacy bridge | A unit test with constructed origins. F-CZI-01 test with the 66 legacy coordinates if R7 is approved; otherwise reported NOT_RUN. |

**Cross-backend equivalence (D15):** F-CZI-01 and F-SVS-01 are unrelated slides, so **no pixel equivalence can be claimed**. Equivalence will be tested for schema, ordering, gating and threshold semantics. Same-content equivalence is possible only with a generated pair (the same synthetic RGB written as a CZI and decoded through the adapter, versus an in-memory reader). A real same-content CZI/SVS pair does not exist; the gap is recorded.

---

### 6. Risks and decisions needed from the owner

| # | Decision | Recommendation |
|---|---|---|
| **R0** | Restore SPEC-004 and fix the file mix-up (§0) | Required before approval |
| **R1** | Detector dependencies: numpy-only, or add OpenCV/scikit-image for median blur, connected components and contours | **numpy-only** for the baseline. Defer contour or hole filtering to an optional detector plugin. |
| **R2** | Runner change: add optional `source: OpenedImage` to `ExecutionRequest` and `StageContext`, and register real backends in `ReaderRegistry` | **Approve.** It is additive and needs no backend import. The alternative (a reader wrapper per run) hides lifecycle problems. |
| **R3** | Task for preprocessing-only runs | Add a catalog `TaskDefinition` "wsi-preprocessing" with no model output, rather than making `task` optional |
| **R4** | Over-budget selection policy | Default: **fail closed** (`PATCH_BUDGET_EXCEEDED`). Optional explicit `selection: row_major_first` or `uniform_stride` (deterministic) recorded in the plan. No random sampling. |
| **R5** | Pixel persistence | In-memory artifacts plus a JSON manifest in SPEC-004. A disk pixel store is deferred to SPEC-005, or added as a plugin writing `.npy`. |
| **R6** | Patch level for the reference pipeline | CZI (0.111 µm/px) at level 2 is about 0.445 µm/px; SVS at level 0 is 0.499 µm/px. Use pixel-based patches with recorded effective spacing, and make `level` (or `target_spacing_um` with a tolerance) a run-time override, so one YAML serves both. Legacy 1.0 µm/px has no stored level on F-CZI-01, so it is rejected, not resampled. |
| **R7** | Read the legacy T198 `coords` zarr chunk (blosc-lz4) read-only, using the legacy venv or a scratch-only `numcodecs` | Approve for bridge evidence. Nothing is written in the legacy repository. |
| **R8** | Additive CZI adapter metadata (`native_padded_origin`) needed by the bridge | Approve. It is additive, and SPEC-003 frame selection is unchanged. |

**Residual risks:**
- A mask at downsample 64 misses fine boundaries (recorded in `MaskGrid`).
- The saturation baseline is a brightfield colour heuristic: it will mis-handle pale or faint staining, pen marks and bubbles. This is documented and not claimed as tissue segmentation quality (D11).
- The SVS fixture is single-level, so multilevel SVS stays untested.
- Many small reads on JPEG XR CZI may be slow. Batching and timing will be reported.

---

### 7. Next step

Phase A is complete. After R0 and approval of R1–R8, or alternatives, Phases B–E will be implemented as above. This report will then be extended with sections 1–15 and AC-001…AC-018 evidence.
