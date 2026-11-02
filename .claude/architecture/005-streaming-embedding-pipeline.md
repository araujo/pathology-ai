# 005 — Streaming Patch-to-Embedding Pipeline: Architecture

**Status:** Implemented. SPEC-005 and ADR-005 remain *Proposed* until the owner accepts them.
**Date:** 2026-10-20
**Related:** `../specs/005-streaming-patch-to-embedding-pipeline.md`, `../adr/005-streaming-embedding-architecture.md`, `005-legacy-streaming-and-embedding-assessment.md`, `004-preprocessing-pipeline.md`

---

## 1. Shape of the pipeline

```text
PipelineRunner (unchanged engine; one additive field: ExecutionRequest.extras)
│
├─ tissue.detect@1 ─► patch.plan@1 ─► patch.extract@1 (materialize_pixels: false)
│                         │                 │  PatchManifest, no pixels
│                         └────────┬────────┘
│                                  ▼
└─ embed.patches@1  (CapabilityGated)
     1. check: manifest belongs to the plan and image; pixel_storage = not_materialized
     2. compose and validate TransformChain against ModelInputSpec       ◄ no pixel read yet
     3. BatchPolicy.plan() → BatchBudget                                  ◄ no pixel read yet
     4. EmbeddingRunIdentity → store_id; EmbeddingStoreProvider.open()     (create / resume)
     5. for each chunk not yet verified:
          for each record: LazyPatchSource.borrow → apply_chain → batch buffer slot
          EmbeddingEncoder.encode_batch → validate rows → write_chunk → commit_chunk
     6. verify() → EmbeddingManifest → finalize(manifest)
     ▼
   EmbeddingBagPayload(manifest, DurableStoreReference, EmbeddingMetrics)
```

The batch loop lives inside one stage operation (ADR-005 D2). There is no streaming edge, no
second engine and no new orchestration concept. `EmbeddingService` mirrors
`PreprocessingService`. It opens the image, passes the handle and a store provider to the runner,
and closes the handle afterwards.

## 2. Layers

| Layer | Module | Content |
|---|---|---|
| domain | `domain/embedding.py` | `ModelInputSpec`, `Normalization`, `PixelState`, `DeclaredTransform`, `TransformChain`, `build_transform_chain`, `validate_model_input`, `ResolutionRecord`, `BatchPolicy` and `BatchBudget`, `EncoderIdentity`, `EmbeddingRecord`, `EmbeddingRunIdentity`, `StoreLayout`, `EmbeddingStoreHeader`, `ChunkCommit`, `StoreInspection`, `StoreVerification`, `StoreDescriptor`, `DurableStoreReference`, `EmbeddingMetrics`, `EmbeddingManifest` |
| domain | `domain/errors.py` | `RESOLUTION_MISMATCH`, `MODEL_INPUT_CONTRACT`, `EMBEDDING_BUDGET_EXCEEDED`, `EMBEDDING_OUTPUT_INVALID`, `EMBEDDING_STORE_{INCOMPLETE,CORRUPT,INCOMPATIBLE,CONFLICT}` |
| domain | `domain/task.py` | `TaskType.FEATURE_EXTRACTION`, `OutputKind.EMBEDDING_BAG` |
| ports | `ports/__init__.py` | `EmbeddingEncoder`, `EmbeddingEncoderFactory`, `EmbeddingStore`, `EmbeddingStoreProvider`, `EncodeContext`, `EmbeddingBagPayload`, `EMBEDDING_STORE_EXTRA` |
| application | `application/runner.py` | `ExecutionRequest.extras` is forwarded to the existing `StageContext.extras` and excluded from the fingerprint |
| application | `application/embedding_service.py` | `EmbeddingService`, `WSI_EMBEDDING_TASK` |
| infrastructure | `infrastructure/embedding_store.py` | `NumpyEmbeddingStoreProvider`, `NumpyEmbeddingStore` |
| plugins | `plugins/embedding/` | `transforms.py` (NumPy step implementations), `reference.py` (reference encoder), `streaming.py` (`LazyPatchSource` and the bounded loop), `operations.py` (`embed.patches@1`) |
| composition | `bootstrap.py`, `cli.py` | registers the operation and the `reference-projection` encoder; `pathology-ai-embed` |
| definitions | `pipelines/wsi-patch-embedding.v1.yaml` | tissue → plan → extract (no pixels) → embed |

Architecture rule change: `infrastructure` may import `numpy`. It is already a base dependency,
and the `.npy` store needs it. `domain`, `ports` and `application` remain free of NumPy and of
vendor or model names (AST-enforced).

## 3. Resolution and transformation semantics

Five concepts are kept apart, and each is recorded in `EmbeddingManifest.resolution`
(`ResolutionRecord`):

| Concept | Source | Recorded as |
|---|---|---|
| Source resolution | reader (`ImageDescription`) | `source_spacing_um`, `source_spacing_source` |
| Extraction resolution | patch plan (explicit level, or target spacing resolved to a stored level) | `extraction_level`, `extraction_downsample`, `extraction_spacing_um`, `requested_spacing_um`, `spacing_relative_error` |
| Patch field of view | derived | `patch_px`, `field_of_view_um` (`None` when spacing is unknown) |
| Model-required input | encoder `ModelInputSpec` | `model_input_px`, `model_required_spacing_um`, `model_accepts_variable_spacing` |
| Applied transforms | `TransformChain` | `resampling`, plus the full chain in `manifest.transforms` |

How the chain is built (`build_transform_chain`):

1. **Channel.** A conversion is added only if the stage declares it in
   `channel_conversions` (e.g. `"bgr->rgb,rgba->rgb"`). Otherwise the result is
   `CHANNEL_ORDER_MISMATCH`. The gate also checks this before any stage runs, using
   `PreprocessingRequirements.channel_orders`.
2. **Numeric.** `cast`, `scale` and `normalize` follow from the encoder's declared dtype,
   `scaled_range` and `normalization`.
3. **Spatial.** If the patch size differs from the model input, a `resample` step is added only
   when declared (`resample: area`, integer factors only). Otherwise the result is
   `RESOLUTION_MISMATCH`.
4. **Layout.** `hwc → chw` when the encoder declares CHW.

The chain is simulated on a `PixelState` while it is built and again in
`validate_model_input`, which checks the geometry, channels, dtype, range, normalization, layout,
precision and spacing. A required spacing rejects unknown spacing
(`MISSING_SPATIAL_METADATA`) and out-of-tolerance spacing (`RESOLUTION_MISMATCH`). Nothing
imposes a universal µm/px. Each step is a `DeclaredTransform`, which pairs the existing
`TransformStep` with a versioned `ComponentVersion` implementation. The implementation must
match at application time.

A future legacy-compatible profile, such as legacy BGR, ImageNet normalization or area
resampling, would be a separate encoder spec plus separate declared conversions. It produces a
different run identity by construction (ADR-005 D12).

## 4. Memory bounds

`BatchPolicy.plan()` runs before any read and fails with `EMBEDDING_BUDGET_EXCEEDED`. The working
set is:

```text
working_set = max_in_flight_batches × model_input_batch_bytes
            + output_batch_bytes
            + scratch (one decoded patch + 3 float patch temporaries + encoder.working_bytes(batch))
```

The implementation enforces this structurally:

- `LazyPatchSource.borrow` keeps one decoded region alive at a time.
- One preallocated model-input buffer of `batch_patches` rows is reused for every batch.
- One output batch exists at a time and is committed before the next batch is read.
- `prefetch: none` is the only policy. Reading happens on the calling thread with the caller's
  handle (ADR-005 D10).
- Actual buffer and output sizes are checked against the plan at run time.

What scales with the number of patches N: `PatchManifest` and `EmbeddingRecord` metadata, the
store header and the manifest JSON (about 550 B per patch), and the on-disk rows. Pixel memory
does not scale with N. Measured evidence is in the report (§6).

## 5. Durable store (`numpy-npy-directory/1`)

```text
<output-dir>/<store_id>/            store_id = pai-emb-<digest of EmbeddingRunIdentity>
  header.json          run identity, layout, row → patch records          (written once, last at creation)
  embeddings.npy       (rows, dim) float32, preallocated, written via open_memmap
  level0.npy           (rows, 4) float64 level-zero x, y, w, h, row-aligned
  chunks/NNNNNNNN.json one commit per chunk: range, SHA-256 of bytes, patch-id digest
  manifest.json        final EmbeddingManifest — the completion marker
```

- **Documents.** Every JSON document is `{"body", "digest"}`, so torn or edited documents are
  detected.
- **Chunk write order.** Rows are written into the map, then `flush()` (msync), then the data
  file is synced (`F_FULLFSYNC` on macOS). The commit record goes to a temporary file, which is
  synced and then renamed. Finally the directory is synced.
- **Inspection.** The store re-hashes every committed chunk against its record. The `.npy`
  header and the exact file size are checked before mapping. A writable `open_memmap` would
  otherwise zero-extend a truncated file without failing.
- **States.** `created`, `in_progress`, `interrupted`, `finalized`, `corrupt`, `incompatible`
  (the last is raised on open). A preallocated array is never evidence of completion.
- **Finalize.** All chunks are re-verified, then a content digest is computed:

  ```text
  SHA-256("pathology-ai.embedding-content/1\n" + "dtype:rowsxdim\n" + rows in order)
  ```

  The manifest is checked against the store and then written atomically. Finalizing again with
  the identical manifest is a no-op. Finalizing with a different manifest is a conflict.
- **Recovery.** `resume_verified` (default) resumes from verified chunks, and any corruption
  fails the run. `discard_corrupt` (explicit) deletes the commit records that fail verification,
  and an invalid manifest with them, then recomputes those chunks.
- **Compatibility.** An existing header with a different run identity, layout (`chunk_rows`) or
  patch mapping raises `EMBEDDING_STORE_INCOMPATIBLE`. It is never resumed or overwritten.
- **Filesystem assumptions.** The store assumes a local POSIX filesystem: rename within a
  directory is atomic, and `fsync` on a file and on its directory makes them durable. There is
  one writer per store and no lock. Network filesystems and object stores are unsupported.
  SHA-256 detects corruption but does not authenticate anyone. The store makes no transactional
  claim beyond these assumptions.

## 6. Identities

| Identity | Binds | Excludes |
|---|---|---|
| `EmbeddingRunIdentity.value` / `store_id` | image id and integrity, scene, plan digest, patch-manifest digest, ordered patch ids, encoder (component, weights checksum, parameters, output schema), `ModelInputSpec`, `TransformChain`, precision, device | batch size, budgets, paths, timestamps |
| `EmbeddingManifest.content_digest` | the stored bytes, in row order | — |
| `EmbeddingManifest.embedding_identity` | run identity + content digest | — |
| `EmbeddingManifest.manifest_digest` | everything in the manifest, including pipeline and chunk layout | paths, timestamps, metrics |
| run fingerprint (`ExecutionFingerprint`) | unchanged: `ExecutionRequest.extras` is not part of it | the store location |

Encoders must be batch-invariant: a row depends only on its patch. The reference encoder
computes per sample. Content digests are therefore independent of batch size. The test suite
checks batch sizes 1, 8 and 32.

## 7. Encoder port

`EmbeddingEncoder` exposes `identity` (`EncoderIdentity`), `input_spec`,
`encode_batch(batch, EncodeContext)`, `working_bytes(batch)` and `close()`. It uses no framework
types. `EmbeddingEncoderFactory` exposes `name`, `input_spec(config)` (used by the gate without
loading the encoder) and `create(config)`. Encoders are selected by name through the stage
config `encoder`, following the precedent of the SPEC-004 tissue detectors. The runner's
`model_slot` binding is not used, for two reasons:

1. It imposes `PreprocessingContract` semantics, where `target_pixel_spacing_um` is mandatory.
   That contradicts ADR-005 D3.
2. `Model.predict` is single-value.

The encoder's identity enters the run identity, the store header and the manifest. It is not
added to `PipelineRun.stage_versions`, which matches the detector precedent (see the report's
limitations).

`reference-projection@1` takes a fixed 256×256 RGB float32 CHW input, normalized with
mean 0.5 and std 0.5, and accepts variable or unknown spacing. It pools on an 8×8 grid per
channel, adds the per-channel std, and applies a fixed PCG64-seeded projection (seed
20261020) followed by `tanh`. `embedding_dim` is configurable from 1 to 4096. The weights
checksum is the SHA-256 of the projection matrix. It is a test fixture with no medical meaning.

## 8. Configuration (`embed` stage)

| Key | Default in YAML | Meaning |
|---|---|---|
| `encoder` | `reference-projection` | registered encoder factory |
| `embedding_dim` | 64 | encoder parameter |
| `channel_conversions` | `bgr->rgb,rgba->rgb` | declared conversions; `none` declares none |
| `resample` | `none` | `none` or `area` (integer factor only) |
| `batch_size` | 32 | max patches per batch = store chunk size |
| `max_input_batch_bytes` | 32 MiB | model-input batch limit |
| `max_output_batch_bytes` | 1 MiB | encoder output batch limit |
| `memory_budget_bytes` | 128 MiB | whole working-set limit |
| `max_in_flight_batches` | 1 | must be 1 with `prefetch: none` |
| `prefetch` | `none` | only policy offered |
| `device` / `precision` | `cpu` / `fp32` | part of the run identity |
| `max_patches` | 100000 | metadata bound for one run |
| `recovery` | `resume_verified` | or `discard_corrupt` |

The output location is not a configuration key. It is supplied per run (`--output-dir`,
`EmbeddingService.run(..., output_dir)`) and travels in `ExecutionRequest.extras`.
