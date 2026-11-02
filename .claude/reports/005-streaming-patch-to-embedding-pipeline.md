# Report — SPEC-005 Streaming Patch-to-Embedding Pipeline

**Date:** 2026-10-20
**Prompt:** `.claude/prompts/005-streaming-patch-to-embedding-pipeline.md`
**Authoritative inputs:** SPEC-005, ADR-005 (both *Proposed*), `005-legacy-streaming-and-embedding-assessment.md`, SPEC/ADR-001…004
**Outcome:**
- All 20 acceptance criteria have automated evidence.
- The full quality-gate suite passes.
- SPEC-005 and ADR-005 are left as *Proposed*. Acceptance is the owner's decision.

---

## 1. Summary

A SPEC-004 patch plan with a pixel-free manifest (`materialize_pixels: false`) now becomes a durable,
verifiable, resumable embedding bag. The new pipeline is `pipelines/wsi-patch-embedding.v1.yaml`,
which runs tissue → plan → extract (no pixels) → `embed.patches@1` through the existing
`PipelineRunner`.

`embed.patches@1` works in this order:

1. Before any pixel read, it validates the encoder's `ModelInputSpec` against the selected stored
   level. Only declared transforms are allowed.
2. Before any pixel read, it plans an explicit memory budget.
3. It reads patches lazily, one decoded region at a time.
4. It encodes them in bounded batches.
5. It commits each batch to a NumPy `.npy` store as a SHA-256-verified chunk.
6. It finalizes only after complete, re-verified coverage.

The stage returns a manifest, a store reference and bounded metrics. It never returns pixels or
embedding rows.

| | F-CZI-01 (SurGen CZI) | F-SVS-01 (CMU-1-Small-Region SVS) |
|---|---|---|
| Reader | `czi-pylibczirw` (BGR, uint8) | `openslide` (RGBA, uint8) |
| Extraction level / spacing | level 2 (ds 4), 0.44496 µm/px (11.0 % off 0.5) | level 0, 0.499 µm/px |
| Patch / field of view | 256 px / 113.9 µm | 256 px / 127.7 µm |
| Declared transforms | `channel_reorder(bgr→rgb)`, cast, scale, normalize, layout | `drop_alpha(rgba→rgb)`, cast, scale, normalize, layout |
| Resampling | none | none |
| Patches (= rows) | 134 of 468 candidates | 25 of 88 candidates |
| Embedding | 64 × float32 | 64 × float32 |
| Batches / chunks | 5 × ≤32 | 1 × 25 |
| Content digest | `56441dbcf66fa696abf34e026d1a508d72fb6d090a299d28f497c6e09cb8bfd1` | `e98540cb395828f92d0da1a42d2b8bca9bc71438a034e844c0b83d81a05fe3f9` |
| Embedding identity | `pai-embid-d3b2649cc67819acbfd14ac84f4c1676` | `pai-embid-27bd8f5b331c3f3d51bc3e1df65244bd` |
| Store id | `pai-emb-7543b229a7c8aebff371649fc31ac58d` | `pai-emb-7c6c0dc2502a1661259e13d24ff54c5c` |
| Manifest digest | `5221186beaff3b27…` | `b494540ecd2a8f1b…` |

Digests are identical across in-process runs, fresh CLI processes and the installed wheel (§8).
They were observed on macOS arm64 with NumPy 2.5.4. Tests compare digests between runs rather
than pinning them, because float rounding may differ across platforms.

The reference encoder is a synthetic NumPy fixture. Its embeddings carry no biological or clinical
meaning, and nothing in this report is evidence about tissue.

---

## 2. Phase A assessment (before implementation)

**Documents.** SPEC-005, ADR-005 and the legacy assessment were all present. No architectural
conflict was found, so implementation proceeded without stopping.

| Question | Finding |
|---|---|
| SPEC-004 exposure | `PatchPlan` holds every candidate and its decision. `PatchManifest` holds the selected `PatchRecord`s: `patch_id`, `plan_index`, `level_rect`, `level0`, effective spacing. `OpenedImage.read_region` is the only pixel access. |
| `materialize_pixels: false` | Supported. It yields `PixelStorage.NOT_MATERIALIZED` and no artifacts, and needs no region capability. |
| Runner context | `StageContext.source` carries the opened image. Per-run configuration is `config_overrides`. `StageContext.extras` already existed but was always empty. |
| Identities | Canonical level-zero `Level0Rect`; `patch_id = short_digest(image, integrity, scene, level_rect, level0)`; `plan_digest`; `manifest_digest`. |
| Model abstractions | `Model.predict(DataValue)` handles one value at a time. `PreprocessingContract` requires a target spacing for 2D models. `TransformStep(name, params)` exists. |
| Constraints | Base dependencies are numpy and pyyaml. The `infrastructure` layer was limited to `yaml` by the AST rule. |
| Integration without a second engine | One fused stage plus a thin `EmbeddingService`. The store location travels through an additive `ExecutionRequest.extras` field. |

**Reusable, unchanged:**
- `PatchPlan`, `PatchManifest` and `PatchRecord`
- `ResolvedPatchGeometry`
- `OpenedImage`
- `PreprocessingRequirements` and the gate
- `TransformStep`
- `ComponentVersion`, `ModelRef`, `ArtifactIdentity` and `digest`
- the config helpers
- `ImagingService`

**Additions:**
- domain embedding contracts
- encoder and store ports
- the `.npy` store
- the plugin
- the service
- the CLI
- one runner field

**Risks identified:**
- Wiring through `model_slot` would force `PreprocessingContract` semantics (see D2).
- `open_memmap` can mutate truncated files. This was found during implementation and fixed (§7).
- Encoder-internal memory was not visible to the budget. This was found and fixed (D6).

---

## 3. Architectural decisions and deviations

| # | Decision | Rationale / ADR link |
|---|---|---|
| D1 | `ExecutionRequest.extras` is forwarded to the existing `StageContext.extras` and excluded from the fingerprint. | The store location is runtime-local (SPEC-005 §6.5, ADR-005 D8). The change is additive, and fingerprints are unchanged (`test_extras_do_not_change_fingerprints`, demo hash). |
| D2 | Encoders are selected by name in the stage config, through `EmbeddingEncoderFactory`, not through the runner's `model_slot`. | `model_slot` binding would require a complete `PreprocessingContract` with a mandatory `target_pixel_spacing_um`, which contradicts ADR-005 D3. `Model.predict` is per-value. This follows the SPEC-004 detector precedent. |
| D3 | `DeclaredTransform` = existing `TransformStep` + versioned implementation `ComponentVersion`. `TransformStep` itself is not modified. | Extend, do not duplicate. Changing `TransformStep` would change existing contract fingerprints and break demo byte-identity. |
| D4 | Channel conversions are declared in the stage (`bgr->rgb,rgba->rgb`) and selected per source order. | One pipeline definition serves CZI (BGR) and SVS (RGBA) with no format branch. Each conversion is explicit and recorded. Undeclared orders fail at the gate. |
| D5 | `embed.patches` refuses manifests with materialized pixels. | P1: no embedding workflow may depend on a pixel-bearing artifact. Fails closed. |
| D6 | `EmbeddingEncoder.working_bytes(batch)` was added to the port and is counted in the budget. The reference encoder computes per sample. | A traced-memory test exposed about 12 MiB of float64 batch temporaries outside the plan. Per-sample computation bounds them and guarantees batch invariance. |
| D7 | Store chunk = batch. `chunk_rows` is part of the store layout, not of the scientific identity. | Resume with a different batch size at the same location is refused (`INCOMPATIBLE`). Other locations produce identical content. |
| D8 | `prefetch: none` is the only policy; one batch in flight; one reader handle on the calling thread. | ADR-005 D10. Threaded prefetch is deferred until reader concurrency is shown to be safe. |
| D9 | `TaskType.FEATURE_EXTRACTION` / `OutputKind.EMBEDDING_BAG` were added. | The run is not a prediction and not preprocessing. Existing enum values are unchanged. |
| D10 | Infrastructure may import `numpy` (architecture rule updated). | Needed by the required `.npy` store. NumPy is already a base dependency, so no dependency was added. |
| D11 | A finalized store is reused only when its manifest digest equals the one the run would produce. Otherwise the run fails with `INCOMPATIBLE`. | Never overwrite valid data silently (SPEC-005 §12). |

**Deviations from the prompt or spec:**
- None substantive.
- `LazyPatchSource` is a plugin class, not a public port. SPEC-005 §7.3 allows this.
- No JSON Schema file was added for the embedding manifest (see §11).

---

## 4. Acceptance criteria

| AC | Verdict | Evidence |
|---|---|---|
| **AC-001** Lazy patch consumption | **PASS** | The upstream manifest is `not_materialized` and the artifact store holds no pixel arrays (`test_patch_plan_to_durable_embedding_bag`, `test_no_pixels_or_embedding_rows_pass_through_the_artifact_store`). Materialized manifests are refused (`test_materialized_manifests_are_refused`). Each patch is read exactly once, on demand (16 and 169 reads; CZI `region_reads=134`). |
| **AC-002** Bounded pixel memory | **PASS** | `test_pixel_retention_does_not_grow_with_patch_count`: with weak-reference tracking, at most **1** decoded patch array is alive for both N=16 and N=169, and the budget is identical. `test_traced_peak_memory_does_not_scale_with_patches`: the traced peak grows by 0.1 MiB from N=16 to N=169, far below the 29 MiB that retaining the patches would need, and stays under the planned working set (§6). |
| **AC-003** Reader independence | **PASS** | One operation and pipeline serve CZI and SVS (`test_one_operation_serves_both_formats`). Format-specific pixel semantics become declared transforms. The AST rules forbid vendor or format names and imports in domain, ports and application; that test is green. |
| **AC-004** Explicit resolution semantics | **PASS** | `ResolutionRecord` records the source spacing, extraction level and spacing, the requested-spacing error, the field of view, the model input, the required spacing and the resampling. Values are asserted on both fixtures. No default spacing exists: the reference spec declares `accepts_variable_spacing` explicitly, and a required spacing rejects unknown spacing (`test_required_spacing_*`, `test_unknown_spacing_needs_explicit_acceptance`). |
| **AC-005** Model input contract | **PASS** | `ModelInputSpec` validates shape, channels, order, dtype, range, layout, normalization, spacing, divisor and precision (`test_invalid_model_input_specs_are_rejected`, 9 cases). A mismatch fails before any patch read: `test_resolution_mismatch_fails_before_any_patch_read` and `test_budget_violation_fails_before_any_patch_read` both assert zero patch reads and no store directory. |
| **AC-006** Explicit transformations | **PASS** | Each transform is a `DeclaredTransform` with parameters and a versioned implementation, stored in the manifest and the run identity. Undeclared reorders or alpha drops raise `CHANNEL_ORDER_MISMATCH`. An undeclared size change raises `RESOLUTION_MISMATCH`. Misordered or incomplete chains are rejected. BGR input with a declared reorder gives rows identical to RGB input, but a distinct identity (`test_bgr_slide_uses_the_declared_reorder`). |
| **AC-007** Batch encoder port | **PASS** | `EmbeddingEncoder` uses no framework types. `validate_output` checks shape, dtype, finiteness and the output budget. Injected NaN, wrong-dimension and wrong-dtype outputs raise `EMBEDDING_OUTPUT_INVALID`, and the batch is never committed (`test_encoder_failures_never_commit_the_batch[nan,dim,dtype]`). |
| **AC-008** Incremental persistent storage | **PASS** | `test_writes_are_incremental_and_committed_one_chunk_at_a_time`: one commit record appears per chunk, after its write. The loop never holds more than one output batch. Rows reach the caller only through `read_rows` on the durable store. |
| **AC-009** Chunk commit integrity | **PASS** | A preallocated full-shape array reads as `created` and cannot be finalized (`test_preallocated_shape_is_not_completion`). Flipped bytes, edited records, unparsable records, truncated data, tampered manifests, and a manifest with a missing chunk are all reported `corrupt` (store contract tests). |
| **AC-010** Safe recovery | **PASS** | Runs interrupted at the read, encode, write, commit and finalize boundaries resume from verified chunks only, reading exactly the missing patches, to the same content digest (§7). A different layout at the same location, or a renamed store, raises `INCOMPATIBLE`. Corruption fails by default; only the explicit `discard_corrupt` recomputes. |
| **AC-011** Patch-to-embedding traceability | **PASS** | Each `EmbeddingRecord` holds the row, `patch_id`, `plan_index`, `level_rect`, `level0` and field of view. Records equal the patch-manifest order and IDs on both fixtures. Each commit binds the chunk's patch-ID digest. `level0.npy` is row-aligned. Spot-checked rows are **bit-identical** to encoding the reader's patch directly (`test_rows_map_exactly_to_their_patches`, `_assert_rows_match_reader` on CZI and SVS). |
| **AC-012** Embedding manifest | **PASS** | `pathology-ai.embedding-manifest/1` records the source, reader and backend, plan and patch-manifest digests and schema, encoder, input spec, transforms, resolution, precision and device, output schema, records, store descriptor and chunk digest, content digest, `finalized` status, limitations and notice. It contains no paths or timestamps. A CLI-written manifest re-digests exactly. |
| **AC-013** Deterministic reference encoder | **PASS** | NumPy-only, PCG64-seeded and computed per sample. Content is identical across batch sizes 1, 8 and 32, repeated runs and fresh processes (`test_digests_are_stable_across_runs_and_batch_sizes`, `test_fresh_processes_reproduce_the_digests`). The weights checksum is recorded. There are no downloads and no torch. |
| **AC-014** PipelineRunner integration | **PASS** | `embed.patches@1` is registered in the existing `OperationRegistry` and gated by `CapabilityGated`. It runs in the runner's topological order, and run provenance is recorded, including `FAILED` runs. The only change is the additive `extras` field. |
| **AC-015** Real WSI validation | **PASS** | CZI: 134 rows, 64 dimensions, 5 chunks. SVS: 25 rows, 1 chunk. Both match the patch-manifest IDs and order and the reader pixels exactly, and both are reproducible. An injected CZI failure at patch 70 resumes from chunks {0, 1} and reads only 70 patches to the identical digest (`tests/integration/test_real_wsi_embedding.py`, 6 tests). |
| **AC-016** Resource safety | **PASS** | Batch, input, output, in-flight and memory budgets are enforced before reads (`test_each_budget_fails_closed`), and actual sizes are rechecked at run time. Encoders are closed on failure (`all(e.closed …)`). The caller's image handle is never closed by the operation, and the service releases it: the file-descriptor count is unchanged after CZI runs. |
| **AC-017** Existing behaviour preservation | **PASS** | All 340 pre-existing tests pass, including SPEC-001…004 and the CZI/SVS preprocessing tests. Demo stdout SHA-256 is `11a211de…df3da` in the checkout and from the wheel, identical to the SPEC-004 baseline. |
| **AC-018** Packaging and optional dependencies | **PASS** | A base wheel (numpy and pyyaml only; no torch, timm, zarr, h5py, cv2 or readers) runs the reference embedding pipeline to the same digest as the source checkout, and imports no ML module (§9). |
| **AC-019** Failure-injection evidence | **PASS** | Tests inject faults at: read (reader exception); encode (exception, NaN, dimension, dtype); write; commit (crash before the record); finalize (crash before the manifest); and corruption of bytes, records, manifest and file length. No partial artifact is ever reported `finalized` (§7). |
| **AC-020** Reproducibility evidence | **PASS** | In-process runs, two fresh subprocesses, fresh CLI processes and the installed wheel produce identical content, run identity, embedding identity and manifest digests on synthetic, CZI and SVS inputs (§8). |

---

## 5. Real WSI execution evidence

These are CLI reports from the installed wheel (`pathology-ai-embed`, default configuration). They
are identical to the source-checkout reports.

**F-CZI-01.**
- Plan and selection:
  - Plan digest `6d339bc0…2d99`; patch-manifest digest `36668d13…f743`.
  - 468 candidates; 134 selected; 334 rejected, of which 191 for acquisition and 334 for tissue.
    These are the same counts as SPEC-004.
  - The plan digest differs from the SPEC-004 extraction pipeline's (`e65bde77…955e`) because
    this pipeline declares larger `max_patches` and `max_candidates` budgets.
- Rows and coordinates:
  - First row: `level0 = (38912, 0, 1024, 1024)`.
  - Rows 0, 67 and 133 are bit-identical to direct `read_region` → declared chain → encoder.
- Execution metrics:
  - `region_reads=134`, `max_live_regions=1`, `batches_encoded=5`.
- Timing and memory:
  - 2.55 s wall time for the whole pipeline.
  - Peak RSS 150.6 MiB. SPEC-004 extraction alone was 142 MiB, dominated by mask decoding.

**F-SVS-01.**
- Plan digest `2644d7d6…7085`; patch-manifest digest `f5fd4298…ef1b`.
- Selection: 88 candidates, 25 selected.
- Transforms: `drop_alpha` declared and recorded; a limitation notes that alpha validity came from
  the plan.
- Execution metrics: `region_reads=25`, `max_live_regions=1`.
- Timing and memory:
  - 0.67 s wall time.
  - Peak RSS 274 MiB, dominated by 2048² RGBA mask tiles as in SPEC-004 (272 MiB).

---

## 6. Batch sizes and memory

Traced memory (`tracemalloc`) was measured from the moment `embed.patches` opens its store, so it
covers the embedding loop rather than detection or planning:

| Input | Batch | Planned working set | Traced embed-stage peak |
|---|---|---|---|
| synthetic, N = 16 | 8 | 11.44 MiB | 9.04 MiB |
| synthetic, N = 169 | 8 | 11.44 MiB | 9.14 MiB |
| synthetic, N = 16 | 32 | 17.44 MiB | 15.03 MiB |
| synthetic, N = 169 | 32 | 29.45 MiB | 27.10 MiB |
| F-CZI-01, N = 134 | 32 | 29.45 MiB | 27.08 MiB |
| F-SVS-01, N = 25 | 25 | 25.01 MiB | 21.78 MiB |

- The peak follows the batch size, not N.
- The planned budget is an upper bound in every case.
- The difference at batch 32 between N=16 and N=169 is the batch itself: 16 rows versus 32 rows.

**CZI budget (default):**

| Component | Size |
|---|---|
| Model-input batch | 25,165,824 B (32 × 3 × 256 × 256 × 4) |
| Output batch | 8,192 B |
| Scratch | 5,701,632 B (decoded patch + 3 float patch temporaries + encoder 3 MiB) |
| Working set | 30,875,648 B, against a 128 MiB budget |

---

## 7. Recovery and corruption results

| Injected at | Store afterwards | Resume reads | Final content |
|---|---|---|---|
| Reader, patch 20 of 169 (batch 8) | `interrupted`, chunks {0, 1} | 153 | = clean |
| Encoder exception, batch 2 | `interrupted`, {0, 1}; encoder closed | 153 | = clean |
| Encoder NaN / wrong dimension / wrong dtype, batch 2 | `EMBEDDING_OUTPUT_INVALID`; {0, 1} | 153 | = clean |
| Store write, chunk 1 | {0}; chunk 1 missing | 161 | = clean |
| Commit crash (rows written, no record), chunk 1 | {0}; chunk 1 missing | 161 | = clean |
| Crash before finalize | every chunk committed, state `interrupted`, no manifest | **0** | = clean |
| One byte flipped in a finalized store | `corrupt`; default resume raises `EMBEDDING_STORE_CORRUPT` | 1 with `discard_corrupt` | = clean |
| Real CZI reader failure at patch 70 | `interrupted`, {0, 1} | 70 | = clean |

The store contract tests (18) also cover:
- Overlapping, repeated or unwritten commits raise `CONFLICT`.
- Wrong-shaped or non-finite rows are refused.
- A torn temporary record is ignored.
- Edited, unparsable or foreign records are `corrupt`.
- A truncated `.npy` file is `corrupt` and is never zero-extended.
- A tampered manifest, or a manifest without full coverage, is `corrupt`.
- Records without a header are `corrupt`.
- A layout change or renamed store raises `INCOMPATIBLE`.
- Path-traversal store IDs are rejected.
- Finalize refuses a manifest whose digest or identity does not match the store.

---

## 8. Reproducibility

| Comparison | Result |
|---|---|
| Two in-process runs, different output directories (CZI, SVS, synthetic) | Identical content, run identity, store ID and manifest digest |
| Batch sizes 1, 8 and 32 (synthetic, N=169) | Identical content and run identity |
| Two fresh `python -c` processes (synthetic) | Identical, and equal to in-process (`214965a9…568a` content) |
| Fresh `pathology-ai-embed` processes (CZI, SVS) | Identical content, embedding identity and manifest digest; the written manifest re-digests exactly |
| Installed wheel with `[czi,openslide]` vs source CLI | All six digests equal for both fixtures |
| Base wheel (no extras) vs source, synthetic | Content `214965a9…568a`, equal |

**Persistent artifacts from the wheel run:**

| Store | File | Size | SHA-256 |
|---|---|---|---|
| CZI, `pai-emb-7543…` | `embeddings.npy` | 34,432 B | `fb3dfa1e…5821` |
| | `header.json` | 74,665 B | `611f724c…4066` |
| | `manifest.json` | 76,494 B | `b49bda7c…1d2d` |
| | `level0.npy` | 4,416 B | — |
| | 5 chunk records | — | — |
| | directory total | 216 KiB | — |
| SVS, `pai-emb-7c6c…` | `embeddings.npy` | 6,528 B | `1783d041…1023` |
| | `header.json` | 15,543 B | `08757da2…0565` |
| | `manifest.json` | 17,310 B | `b8180aed…8e9c` |
| | directory total | 52 KiB | — |

---

## 9. Packaging and regression

**Commands (2026-10-20):**

```bash
uv run ruff check .                 # All checks passed
uv run ruff format --check .        # 134 files already formatted
uv run mypy                         # Success: no issues found in 105 source files (strict)
uv run pytest -q -W error -rs       # 386 passed, 42 skipped (fixtures not configured)
PATHOLOGY_AI_WSI_FIXTURE_CZI=… PATHOLOGY_AI_WSI_FIXTURE_SVS=… \
  uv run pytest -q -W error -rs     # 426 passed, 2 skipped (pre-existing UNSUPPORTED skips)
uv build                            # sdist + wheel OK
```

**Baseline before SPEC-005:** 340 passed and 2 skipped with fixtures. All 340 still pass.

**New tests (86):**

| Suite | Tests |
|---|---|
| Unit contracts | 35 |
| Store contract | 18 |
| In-memory pipeline | 26 |
| Real fixtures | 6 |
| Pipeline YAML round-trip | 1 (picked up automatically) |

**Wheel checks:**

- **Contents.** The wheel contains `domain/embedding.py`, `infrastructure/embedding_store.py`,
  `plugins/embedding/*` and `application/embedding_service.py`. It contains no `pipelines/`,
  `projects/` or `tests/`.
- **Full install `[czi,openslide]`.** `pathology-ai-embed` with `PATHOLOGY_AI_ROOT` exits 0 on
  both fixtures, with digests equal to the source checkout. `pathology-ai-demo` produces SHA-256
  `11a211de…df3da`.
- **Base install** (numpy 2.5.4, pyyaml 6.0.3 and pathology-ai only):
  - It runs the reference embedding pipeline on a synthetic slide to the source digest.
  - It imports no torch, timm, zarr, h5py, cv2 or reader module.
  - Run on a CZI file, it exits 3 with `READER_BACKEND_UNAVAILABLE … install pathology-ai[czi]`.

**Changed existing files** (all additive):
- `domain/errors.py`, `domain/task.py`
- `ports/__init__.py`
- `application/runner.py` (`extras`)
- `bootstrap.py` (registration, `Platform.embedding_runs`)
- `cli.py` (`embed_main`)
- `pyproject.toml` (the `pathology-ai-embed` script; no dependency added)
- `tests/architecture/test_dependency_rules.py` (numpy allowed in infrastructure)
- `README.md`

**Repository independence:**
- `../digital-pathology` HEAD is still `c85df8a`, with 4 status lines. It was not opened during
  implementation.
- `../hal-viewer` HEAD is still `e248202`, with 9 status lines.
- No legacy code was copied. No commits were created.

---

## 10. Validation checklist (prompt Phase E)

1. **Pixel retention is flat in N.** True: 1 live decoded patch at both N=16 and N=169; traced
   peak 9.04 vs 9.14 MiB.
2. **Writes are incremental.** True: one commit record per chunk, after its write.
3. **Association is exact.** True: by patch ID and plan index, with bit-identical rows.
4. **Interrupted stores are never complete.** True: `interrupted` until `manifest.json` exists and
   re-verifies.
5. **Corruption is detected.** True: SHA-256 per chunk, document digests, and `.npy` header and
   size checks.
6. **Resume never mixes identities.** True: identity, layout and mapping are compared, and a
   mismatch raises `INCOMPATIBLE`.
7. **Provenance is complete.** True: `ResolutionRecord` and `TransformChain` are recorded with
   implementation versions.
8. **Digests are stable.** True (§8).
9. **CZI and SVS go through the same operation.** True.
10. **The base package works without ML.** True (§9).

---

## 11. Remaining limitations

- **Encoder in run provenance.** The encoder's `ComponentVersion` is bound into the run identity,
  store header and manifest. It does not appear in `PipelineRun.stage_versions` or the execution
  fingerprint's components, which list the operation only, as for SPEC-004 detectors. A future
  additive runner hook could let operations report sub-components.
- **Batch invariance.** It is required of encoders and verified for the reference encoder.
  GPU or framework encoders may not be bit-exact across batch sizes. Such adapters must either
  guarantee invariance or bind batch size into their identity.
- **No concurrency.** There is no prefetch, no reader threads and no multi-process reading.
  Throughput is bounded by sequential decode and encode: about 52 patches/s on CZI including
  detection.
- **Durability.** Guarantees depend on local POSIX semantics. There is no writer lock: concurrent
  writers to one store are unsupported. Object stores and network filesystems are out of scope.
- **Metadata scales with N.** Manifests and headers are JSON (about 550 B per patch). This is
  fine for 10⁵ patches and is bounded by `max_patches`, but not compact. A columnar mapping or
  Zarr adapter is deferred, behind the same port.
- **No JSON Schema.** No schema or presentation serializer was added for
  `pathology-ai.embedding-manifest/1`. The canonical JSON is written by the store and the CLI.
- **Resampling.** Only integer-factor area resampling is implemented. Other declared methods are
  rejected rather than approximated. Legacy SurGen parity (2013 → 224 `INTER_AREA`) remains out
  of scope.
- **Fixtures.** Both real fixtures yield 134 and 25 patches. Large-N behaviour is evidenced on
  synthetic slides (169 patches) and by the budget algebra, not on a full-size slide at 10⁵
  patches.
- **No UNI or foundation-model adapter, no MIL.** These are out of scope by design. The reference
  encoder has no scientific meaning.
