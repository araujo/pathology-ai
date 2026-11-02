# 005 — Legacy Streaming, Embedding Pipeline and Native Resolution: Discovery Assessment

**Status:** Discovery only (read-only). No SPEC-005 design is approved and no code was written.
**Date:** 2026-10-18
**Inputs:** `../digital-pathology` (HEAD `c85df8a`, read-only), `pathology-ai` working tree (SPEC-001…004 state)
**Related:** `../specs/004-real-wsi-preprocessing-and-patch-extraction.md`, `../adr/004-real-wsi-preprocessing-architecture.md`, `004-preprocessing-pipeline.md`, `004-legacy-patch-compatibility.md`

## Evidence labels

| Label | Meaning |
|---|---|
| **VERIFIED** | Read in source at the cited path and line. The control flow was traced end to end. |
| **INFERRED** | Follows from verified code plus library defaults or arithmetic, but was not executed or measured here. |
| **NOT VERIFIED** | Plausible, but no evidence was gathered (needs execution, a model card, or a license check). |

Legacy paths are relative to `../digital-pathology/`. `R/` = `reproducibility/`, `S/` = `scripts/`.

---

## 1. Executive findings

1. **The legacy pipeline streams by design, but only between two stages.** Patching writes **coordinates only** (`int64[N,2]`), not pixels. Embedding re-reads each patch from the CZI on demand inside a PyTorch `Dataset.__getitem__`, inside `DataLoader` workers. Patch pixels are never persisted. (VERIFIED: `R/wsi_core/WholeSlideImage.py:486-552`, `R/datasets/dataset_h5.py:132-174`)
2. **During embedding, pixel memory is bounded by batch × workers × prefetch, not by patch count.** Embeddings are written incrementally, one batch at a time, into a preallocated Zarr array. (VERIFIED: `R/extract_features.py:93-115`, `R/utils/file_utils.py:94-142`. The prefetch bound is INFERRED from the PyTorch default `prefetch_factor=2`.)
3. **Three steps still scale with N.** None is a pixel buffer:
   - patch-candidate generation builds the full bounding-box grid per contour and pickles it into an `mp.Pool`;
   - all coordinates are concatenated before a single write;
   - after each slide, the whole feature matrix is loaded (`features[:]`) to write a `.pt` copy.

   At `N × 1024 × 4 B` these stay small, but they are not bounded by batch size. (VERIFIED: `WholeSlideImage.py:658-671, 527-550`, `extract_features.py:463-470`)
4. **The legacy pipeline does not preserve native resolution.** It always reads at level 0 (`zoom=1.0`) a region of `int(patch_size·target_mpp/native_mpp)` px, then resamples with `cv2.INTER_AREA` to `target_patch_size`.
   - For the SR1482 UNI contract this is 2013 px → 224 px at a nominal 1.0 µm/px, about a 9× downsample.
   - The resample is set by configuration, but it is **not recorded** in the embedding artifact.

   (VERIFIED: `dataset_h5.py:151-166`, `S/run_pipeline.py:641-662`, `tasks/preprocessing/surgen_sr1482_uni_v1.json`)
5. **Encoder input is BGR.** `pylibCZIrw` returns `Bgr24` in B,G,R memory order, and `Whole_Slide_Bag_FP` never converts it. The legacy SPEC-024.5 contract states BGR as the only supported encoder input. UNI therefore received BGR, ImageNet-normalized tensors for training and inference. (VERIFIED: `.venv/.../pylibCZIrw/czi.py:162`, `S/pipeline/preprocessing_config.py:46-53`)
6. **Every encoder gets the same transform:** `ToTensor` + ImageNet mean/std, fp32, no autocast. That includes PLIP, a CLIP model with different normalization constants. (VERIFIED: `dataset_h5.py:32-48`, `extract_features.py:49,72`. That PLIP expects CLIP statistics is NOT VERIFIED here.)
7. **SPEC-004 in pathology-ai does not stream.** `patch.extract@1` collects **every** selected patch's pixels in one list. It returns them in a single `PatchManifestPayload`, and the runner puts all of them into the in-memory artifact store. Retention is capped by `max_output_bytes` (256 MiB, fail-closed), not bounded by batch size. (VERIFIED: `plugins/preprocessing/operations.py:300-444`, `application/runner.py:419-427`) **SPEC-005 must not consume SPEC-004 pixel artifacts for whole slides.**
8. **What carries over cleanly from legacy:**
   - coordinates-only patch plans with lazy pixel reads at embedding time;
   - batch-chunked, preallocated, positionally written embedding stores;
   - content-addressed preprocessing and embedding identities (SPEC-024.5);
   - fail-closed binding of a model to its preprocessing contract.

   pathology-ai already has the domain hooks for these: `PreprocessingContract`, `TransformStep`, `resolve_channel_conversion`, `materialize_pixels: false`, and `EMBEDDING_BAG`.

---

## 2. Actual legacy execution flow

### 2.1 Entry points

| Entry | Role | Evidence |
|---|---|---|
| `S/run_pipeline.py` `stage_create_patches` → subprocess `R/create_patches.py` | batch patching over a flat CZI symlink dir | `run_pipeline.py:429-526` |
| `S/run_pipeline.py` `stage_extract_embeddings` → subprocess `R/extract_features.py` | batch embedding; validates outputs afterwards | `run_pipeline.py:569-718` |
| `S/parallel_preprocess.py` | per-slide fan-out: ThreadPool of subprocesses, GPU round-robin, status CSV | `parallel_preprocess.py:217-458` |
| `S/analysis_api/artifact_resolution.py` `_regenerate_bag` | inference path: contract → create_patches → validity gate → extract_features | `artifact_resolution.py:1-40, 578-590, 740-794` |

Each stage is a separate **subprocess** invocation of the unchanged `reproducibility/` scripts. Orchestration passes files between stages (VERIFIED).

### 2.2 Trace: WSI → embedding

```
CZI file
 │ R/wsi_core/WholeSlideImage.py  WholeSlideImage.__init__            :86-104
 │   pyczi.CziReader(path)                                          :111
 │   level_dim = 10 synthetic power-of-two levels (not stored levels):114-136
 │   offset = total_bounding_rectangle[0:2]  (legacy padded frame)    :92
 │   retrieve_mpp = Scaling/Items/Distance[0] × 1e6  (X only)         :100
 ▼
segmentTissue(seg_level=6)                                            :199-297
 │   wsi.read(roi=full rect, zoom=1/64) → ndarray BGR uint8 (whole slide @ 1/64)
 │   BGR→HSV, median blur on H, threshold, close, findContours
 │   contours scaled to level 0; img released at function exit
 ▼
process_contours → process_contour (per contour)                      :486-552, 583-702
 │   patch_size_l0 = int(patch_size·target_mpp/retrieve_mpp)          :597-600
 │   full bbox meshgrid of candidates (np)                            :658-661
 │   mp.Pool(min(cpu,4)).starmap(isInContours…)  (one pool per contour):663-671
 │   np.concatenate(all coords) → save_hdf5_coords(..., mode='w')     :528-550
 ▼
patches/<stem>.zarr   coords int64[N,2] (single chunk), attrs patch_size=<L0 footprint>, patch_level=0
 │   (SPEC-024.5 inference path only) patch_validity.measure → retained rows only
 │   S/pipeline/patch_validity.py:319-356, artifact_resolution.py:578-590
 ▼
R/extract_features.py __main__                                        :356-470
 │   load_models(...) → .to(device), DataParallel if >1 GPU, .eval()  :400-406
 │   per slide: wsi = pyczi.CziReader(czi)                            :436
 │   compute_w_loader(...)                                            :48-136
 │     Whole_Slide_Bag_FP(coords file, wsi)                           dataset_h5.py:50-108
 │     DataLoader(bs, num_workers, pin_memory=True, collate_features) :75-76
 │     for batch, coords in loader:                                   :93
 │        batch.to(device, non_blocking) ; model(batch).cpu().numpy() :100-110
 │        save_hdf5(<feat>/h5_files/<model>/<stem>.zarr, mode='a', batch_id) :115
 │   features = file['features'][:] → torch.save(pt_files/<stem>.pt) :463-470
 ▼
embeddings/h5_files/<encoder>/<stem>.zarr  features float32[N,D], coords int64[N,2]
embeddings/pt_files/<stem>.pt               float32[N,D]   (completion marker in effect)
```

### 2.3 `Whole_Slide_Bag_FP.__getitem__` (where pixels are allocated)

`R/datasets/dataset_h5.py:132-174`, VERIFIED:

1. It opens the coordinate store **for every item** (`h5py.File` or `zarr.open`) and reads `coords[idx]`.
2. `img = np.array(wsi.read(roi=(offset+coord, patch_size_l0, patch_size_l0), zoom=1.0))`. This decodes a full-resolution BGR uint8 region.
3. `cv2.resize(img, target_patch_size, INTER_AREA)`, when `target_patch_size` is set. The orchestrators always set it to `patching.patch_size`.
4. `eval_transforms(pretrained=True)`: `ToTensor` (HWC uint8 → CHW float32 /255), then `Normalize(ImageNet)`.
5. `.unsqueeze(0)`, returned together with the coordinate.

`collate_features` uses `torch.cat` on the images and `np.vstack` on the coordinates (`R/utils/utils.py:40-43`).

---

## 3. Streaming and batching implementation

| Aspect | Legacy behaviour | Label |
|---|---|---|
| Mechanism | Map-style `Dataset` over the coordinate array, sequential sampler (no shuffle), `DataLoader` worker processes | VERIFIED |
| Lazy pixels | Yes. Each patch is decoded only inside `__getitem__` | VERIFIED |
| Memory mapping | No. Zarr and HDF5 are used only for coordinates and features. No mmap of pixels | VERIFIED |
| Patch caching | None. No pixel cache. The pylibCZIrw subblock cache is left at default (`SubBlockCacheOptions.Clear()`, no cache options passed) | VERIFIED (default construction); effect NOT VERIFIED |
| Batch size | CLI default 256. Configs and orchestrators use 64 (`experiments/configs/encoder/uni.yaml`, `artifact_resolution.py:760-781`) | VERIFIED |
| Workers | CLI default 16. `parallel_preprocess` uses 4. `analysis_api` uses 4 | VERIFIED |
| Prefetch | PyTorch default `prefetch_factor=2` per worker | INFERRED |
| Pinned memory | `pin_memory=True` on CUDA only | VERIFIED |
| Incremental writes | Yes. One `save_hdf5` call per batch writes `[batch_id·bs : +len]` | VERIFIED |
| Slide parallelism | One subprocess per slide (parallel_preprocess). GPUs round-robin via `GpuScheduler` | VERIFIED |
| Worker lifecycle | Workers live for one `DataLoader` iteration, one slide. There is no explicit close of `CziReader` | VERIFIED (no close call) |
| Reader shared into workers | The `CziReader` native handle is created in the parent and inherited by forked workers. Safety of concurrent reads through a forked libCZI handle was not established. Under `spawn` (macOS default) it would need to be picklable | NOT VERIFIED |

**A DataLoader is present, but that alone does not prove end-to-end streaming.** End-to-end streaming does hold here for one reason: the upstream artifact contains no pixels and the downstream write is per batch. The non-streaming steps are listed in §4.3.

---

## 4. CPU/GPU memory lifecycle

### 4.1 Stage table (SR1482 UNI contract: native 0.11124 µm/px, target 1.0 µm/px, patch 224, bs 64)

| Stage | Type / shape | Lazy or materialized | Location | Copies / conversions | Lifetime | Persistence | Peak risk |
|---|---|---|---|---|---|---|---|
| CZI handle | `CziReader` | lazy | CPU native | — | per slide | — | fd per worker (NOT VERIFIED) |
| Segmentation image | uint8 `(H/64, W/64, 3)` BGR | materialized, whole slide at zoom 1/64 | CPU | BGR→HSV, median, threshold | `segmentTissue` scope | contours in memory | guarded by `w·h > 1e8` abort (`create_patches.py:245-249`) |
| Candidate grid | int64 `(n_bbox, 2)` + Python list of tuples | materialized **per contour** | CPU, 4 pool procs | pickled per task (contour, holes, check fn) | per contour | — | **O(bbox candidates)**. Grows with smaller footprint (native res) (INFERRED) |
| Coordinates | int64 `(N, 2)` | materialized, then one write | CPU | concatenate | per slide | Zarr, single chunk | O(N), small |
| Region read | uint8 `(2013, 2013, 3)` BGR ≈ 11.6 MiB | per item | CPU worker | `np.array(...)` copy of the bitmap | item scope | none | × workers |
| Resized | uint8 `(224, 224, 3)` ≈ 147 KiB | per item | CPU worker | `cv2.resize INTER_AREA` | item scope | none | — |
| Tensor | float32 `(1, 3, 224, 224)` ≈ 588 KiB | per item | CPU worker | `ToTensor` (/255, HWC→CHW); `Normalize` makes a non-in-place copy | until collate | none | — |
| Batch | float32 `(64, 3, 224, 224)` ≈ 36.75 MiB | per batch | CPU shared memory | `torch.cat`; pinned copy | prefetch queue | none | **≈ workers × 2 × 36.75 MiB** (16 workers ≈ 1.15 GiB; 4 workers ≈ 294 MiB) (INFERRED) |
| Device batch | same | per batch | GPU | H2D `non_blocking` | loop iteration | none | + ViT-L activations (no_grad) |
| Model | ViT-L/16, about 303 M params, fp32 ≈ 1.2 GB | resident | GPU (or CPU) | `DataParallel` replicas when >1 GPU | process | — | fixed |
| Embedding batch | float32 `(64, 1024)` = 256 KiB | per batch | GPU → CPU | `.cpu().numpy()` (synchronous) | iteration | Zarr chunk `(64, 1024)` | — |
| Final load | float32 `(N, 1024)` | materialized | CPU | `[:]`, then `torch.from_numpy` | end of slide | `.pt` | **O(N)**, small (N = 5 000 ≈ 20 MB) (INFERRED) |

### 4.2 Device and precision

- **Device:** `cuda` if available, otherwise `cpu`. MPS is never used (`extract_features.py:29`). GPU pinning uses `CUDA_VISIBLE_DEVICES` from the orchestrators (VERIFIED).
- **Precision:** fp32 throughout. There is no `autocast` and no `.half()`. `torch.no_grad()` is used, not `inference_mode` (VERIFIED).

### 4.3 Is memory bounded by batch size?

**Partly.** Embedding pixel memory is bounded by `batch × workers × prefetch` (INFERRED from the verified code plus PyTorch defaults). Three steps scale with patch or candidate count:

- per-contour candidate generation and pickling;
- coordinate concatenation;
- the final `features[:]` load.

All three are coordinate- or embedding-sized, not pixel-sized. No step scales pixel memory with N. **Bounded execution is not claimed as measured.** No RSS or GPU memory measurement of the legacy pipeline was performed in this discovery.

---

## 5. Persistent storage architecture

| Artifact | Format | Layout | Writer | Label |
|---|---|---|---|---|
| Patch coordinates | Zarr v2 DirectoryStore (h5 optional) | `coords int64[N,2]`; Zarr chunk = whole array; attrs `patch_size` (L0 footprint), `patch_level`, `downsample`, `level_dim`, `name`, `save_path` | `save_hdf5_coords` (`file_utils.py:18-91`), one write | VERIFIED |
| Validated patches (SPEC-024.5) | Zarr | `retained/coords`, `candidate_index`, per-candidate validity arrays | `write_validated_patch_artifact` (`patch_validity.py:319-356`) | VERIFIED |
| Embeddings | Zarr v2 | `features float32[N,D]`, `coords int64[N,2]`, preallocated to `sample_count`, chunks `(batch_size, ·)`, **no attrs** | `save_hdf5` (`file_utils.py:94-142`) per batch | VERIFIED |
| Embedding copy | torch `.pt` | float32 `[N,D]` | `extract_features.py:470` | VERIFIED |
| Identity manifests | JSON | `patch_artifact.json`, `embedding_artifact.json`. `embedding_artifact_id = sha256(patch_artifact_id, encoder, weights sha256, encoder input, features hash)` | `artifact_resolution.py:189-248, 640-651` | VERIFIED |

**Coordinate ↔ embedding association.** Rows are aligned by **position only** (`batch_id·batch_size`), with the coordinates copied alongside. This is correct only because the sampler is sequential and `drop_last` is false. There is no patch ID (VERIFIED). The coordinates are in the **legacy padded frame**. `004-legacy-patch-compatibility.md` gives the per-slide translation (F-CZI-01: −17 px in X).

**Resume after interruption** (VERIFIED):

- Resume works per slide only.
- `extract_features.py:419-421` skips a slide when `<stem>.pt` exists. The `.pt` is written last, so it acts as the completion marker.
- The "in progress" check at `:424-429` is broken: its `continue` applies to the inner `for` loop, so it never skips.
- A rerun after a crash rewrites the preallocated Zarr in place (`mode='a'`). No resume happens inside a slide.
- `parallel_preprocess` also requires a `SUCCESS` status row, written atomically with tmp + replace (`:131-139`).

**Risk** (INFERRED): `run_pipeline.py:672-711` and `parallel_preprocess.is_embedding_valid` check only shape and non-emptiness. A crashed, preallocated Zarr has full shape, and its unwritten chunks read as the fill value. Either validator alone would accept it.

---

## 6. Foundation-model preprocessing requirements

| Encoder | Construction | Dim | Input in legacy | Label |
|---|---|---|---|---|
| UNI | `timm vit_large_patch16_224, img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True`, strict load | 1024 | 224² (after INTER_AREA), BGR, ImageNet mean/std, fp32 | VERIFIED (`extract_features.py:224-255`) |
| UNI2-h | `vit_huge_patch14_224` | 1536 | same transform | VERIFIED construction; matching the official recipe NOT VERIFIED |
| Virchow | `vit_huge_patch14_224, mlp_ratio=5.3375` | 1280 | same; pooled output | VERIFIED construction. The official card's SwiGLU/SiLU layers and CLS+mean-token embedding are NOT VERIFIED |
| PLIP | HF `CLIPVisionModelWithProjection` | 512 | ImageNet-normalized | VERIFIED. That CLIP normalization is required is NOT VERIFIED |
| ResNet50 / CTransPath / Owkin iBOT | repo-local | 2048 / 768 / 768 | same | VERIFIED |

`S/pipeline/encoder_registry.py` defines a `BaseEncoder` with `preprocess` (Resize + CenterCrop + ImageNet) and `encode_batch`. **The extraction path does not use it.** Only `resolve_checkpoint` and `KNOWN_ENCODER_DIMS` are referenced (VERIFIED by grep: `get_encoder` has no callers).

**Model-specific vs generic.**

| | What belongs here |
|---|---|
| Model-specific | input size (and divisibility by the ViT patch size), channel order, mean/std, interpolation, expected physical scale, output token pooling, precision tolerance, weights identity |
| Generic pipeline | coordinate iteration, lazy region read, batching, device transfer, incremental write, row ↔ patch association, resume, provenance |

Legacy mixes the two: `Whole_Slide_Bag_FP` owns both the resize and the normalization.

**UNI physical scale.** UNI's published pretraining scale (about 20×, 0.5 µm/px) differs from the legacy 1.0 µm/px contract. NOT VERIFIED here. It is relevant to the decision in §12.

---

## 7. Resolution and physical-scale handling

### 7.1 Legacy behaviour (VERIFIED unless noted)

| Concept | Legacy |
|---|---|
| Pyramid level | Always 0 for patching and embedding. `--patch_level` was removed (`create_patches.py:318-320, 399`). Segmentation and visualization use `zoom=1/2^k` decoder scaling on **synthetic** power-of-two levels (`WholeSlideImage.py:114-148`), not stored pyramid levels |
| µm/px | `Distance[0]` (X) × 1e6, assumed isotropic. No fallback if missing |
| Patch dims | `patch_size` means **model pixels at `target_mpp`**. The level-0 footprint is `int(patch_size·target_mpp/mpp)`, which truncates (2013 px → 223.93 µm, not 224 µm). The stored attr `patch_size` is the footprint, so the same name has two meanings |
| Field of view | Implicit: footprint × native mpp. Not recorded in the embedding artifact |
| Resizing | `cv2.resize(..., INTER_AREA)` to `target_patch_size`, which equals `patching.patch_size`. Effective input spacing = 2013 · 0.11124 / 224 ≈ 0.99996 µm/px |
| Model input resolution | Not declared per model. It comes from the patching config (`run_pipeline.py:641-662`). Without `--target_patch_size`, 2013 px reaches the ViT and fails the divisible-by-16 check (comment at `run_pipeline.py:641-645`) |

### 7.2 pathology-ai today (VERIFIED)

- `resolve_patch_geometry` (`domain/patching.py:154-214`) picks a **stored** level, either explicit or nearest within `spacing_tolerance`. It never resamples, and it records the effective spacing.
- The CZI adapter reports only stored subblock ratios as levels (F-CZI-01: 1, 2, 4, 8, 16, 32, 64).
- `PreprocessingContract` (`domain/preprocessing.py`) declares `model_channel_order`, `interpolation`, `target_pixel_spacing_um`, `patch_size_px`, and ordered `transforms`. `resolve_channel_conversion` refuses any undeclared BGR/RGB swap.
- `missing_fields` makes `target_pixel_spacing_um` mandatory for `PYRAMIDAL_2D` contracts. That is acceptable as a **model** requirement, but it must not become an extraction default.

### 7.3 Proposed separation (for SPEC-005; not implemented)

| # | Concept | Owner | Content | Default |
|---|---|---|---|---|
| 1 | **Source resolution** | reader → `ImageDescription` | level-0 spacing (x, y) or unknown; stored levels and downsamples | as reported; never synthesized |
| 2 | **Extraction resolution** | patch plan (SPEC-004 `PatchSpec`) | the selected stored level and its effective spacing | **an explicitly selected stored level**. No resampling, no implicit level |
| 3 | **Effective patch FOV** | derived, recorded per manifest | `width_px × eff_spacing_x`, `height_px × eff_spacing_y` µm, or "pixels only" when spacing is unknown | derived |
| 4 | **Model-required input** | model's `PreprocessingContract` / a new model input spec | input px (with divisibility), required spacing or "any", tolerance, channel order, dtype, normalization | none. Must be declared per model |
| 5 | **Explicit transforms** | ordered `TransformStep` chain bound to the run | e.g. `channel_reorder(bgr→rgb)`, `resample(method=area, from_px, to_px, from_um, to_um)`, `normalize(mean, std)` | empty |

**Validation rule.** At gate time, before any read:

- Compose (2) with (5). The result must satisfy (4) within tolerance.
- If spacing or size mismatch and no declared `resample` covers it, fail with a typed error, for example `RESOLUTION_MISMATCH`.
- Every applied transform, with its parameters and the implementing component version, goes into the embedding provenance and the run fingerprint.

Configuration layers (dataset → experiment → model → run override) choose (2) and (5). The core never branches on format.

---

## 8. Reusable patterns (adapt conceptually; copy no code)

| Pattern | Legacy evidence | SPEC-005 target |
|---|---|---|
| Coordinates-only plan; pixels read lazily at embedding time; patch pixels never stored | `process_contours`, `Whole_Slide_Bag_FP` | Embed from the SPEC-004 `PatchPlan` / manifest produced with `materialize_pixels: false` |
| Bounded producer → consumer: read/preprocess in parallel, infer per batch, write per batch | `compute_w_loader` | a single `embed` stage operation with a bounded queue |
| Preallocated store, chunk = batch, positional writes | `save_hdf5` (zarr branch) | `EmbeddingStore` port with `open(header) / write(rows, start) / commit()` |
| Completion marker written last | `.pt` skip | explicit commit record. Uncommitted stores are never treated as valid |
| Atomic status file (tmp + replace) | `parallel_preprocess.save_status` | infrastructure store commit |
| Content-addressed preprocessing contract, fail-closed model binding | `preprocessing_config.py` | already mirrored by `PreprocessingContract`; extend with a model input spec |
| Declared encoder input (channel order / interpolation / normalization) | `SUPPORTED_ENCODER_INPUT` | model contract fields + `TransformStep`s |
| Embedding identity = (patch artifact, encoder, weights sha256, encoder input, features hash) | `artifact_resolution.py:189-248` | `ArtifactIdentity.derive` with weights checksum (`ModelArtifact.checksum`) |
| Invariants `len(coords) == len(features)` and exact coordinate equality, no reconciliation | `embedding_artifact.py:93-141` | manifest validation |
| Pre-embedding validity gate (no-data, tissue) | `patch_validity.py` | already in SPEC-004 (acquired/tissue fractions) |
| Checkpoint resolution fails fast on an explicit missing path | `encoder_registry.resolve_checkpoint:101-110` | infrastructure model registry |

---

## 9. Patterns that should not be reused

| Anti-pattern | Evidence | Why |
|---|---|---|
| Resampling hidden inside the dataset, decided by the patching `patch_size` | `dataset_h5.py:161-166`, `run_pipeline.py:641` | silent, conflates grid size with model input size |
| One `patch_size` name for both model pixels and level-0 footprint; `int()` truncation of the FOV | `WholeSlideImage.py:597, 679` | ambiguous geometry |
| Synthetic power-of-two "levels"; decoder zoom used as resolution | `WholeSlideImage.py:114-148` | contradicts SPEC-003 stored-level semantics |
| X-only spacing assumed isotropic | `WholeSlideImage.py:100` | anisotropy is lost |
| Uniform ImageNet normalization for every encoder | `dataset_h5.py:32-48` | model-specific requirement treated as generic |
| Implicit BGR into RGB-pretrained models | `dataset_h5.py:158`; contract `:46-53` | allowed only as an explicit, declared transform |
| Positional row alignment without patch IDs | `file_utils.py:132-139` | breaks if sampling order changes |
| Embedding store without metadata (encoder, preprocessing, spacing, frame) | `embedding_artifact.py:5-15` | provenance lives outside the data |
| Shape-only validity checks on preallocated stores | `run_pipeline.py:672-711`, `parallel_preprocess.py:158-174` | accepts partial outputs |
| Broken in-progress skip | `extract_features.py:424-429` | dead logic |
| Coordinate store reopened per item | `dataset_h5.py:136-141` | I/O overhead per patch |
| `mp.Pool` per contour, `close()` without `join()`, contour pickled per candidate | `WholeSlideImage.py:663-670` | lifecycle and CPU cost; O(candidates) |
| Native reader handle inherited by forked workers | `extract_features.py:436` → `DataLoader` | unverified safety; reopen per worker instead |
| Runtime `login()` / download inside the pipeline | `extract_features.py:237-246` | non-reproducible, interactive |
| `nn.DataParallel` | `extract_features.py:403-404` | legacy multi-GPU approach; one device per run is simpler and fits the runner |
| Subprocess chaining of scripts as orchestration | `run_pipeline.py`, `parallel_preprocess.py` | would form a second orchestration engine (forbidden by ADR-004 D9) |
| Legacy padded coordinates used as level 0 | `004-legacy-patch-compatibility.md` | use the explicit `FrameTranslation` |

Unrelated latent bug seen in passing: `SlideDataset.__len__` returns `len(int)` (`dataset_h5.py:214-215`).

---

## 10. Gaps between SPEC-004 and the legacy pipeline

| # | Gap | SPEC-004 (VERIFIED) | Legacy | Consequence for SPEC-005 |
|---|---|---|---|---|
| G1 | Streaming | Extraction keeps all selected pixels and returns them in one payload. `batch_size` only chunks the loop; retention is capped at 256 MiB | pixels never retained | embed directly from the plan; never route whole-slide pixels through `PatchManifestPayload` |
| G2 | Runner granularity | `op.run` returns a complete mapping. Every output is `put` into the artifact store. No iterator or streaming edge exists | n/a | the batch loop must live **inside** one stage operation; its output is a manifest plus a durable store reference, not arrays |
| G3 | Durable storage | `InMemoryArtifactStore` only (R5). No disk pixel or embedding store | Zarr on disk | needs a durable `EmbeddingStore` adapter (format is a decision, §12) |
| G4 | Encoder port | `Encoder` is a marker on `Model.predict(DataValue, ctx)`, one value at a time | batch `model(tensor)` | needs a batch-oriented encoder port with an input spec |
| G5 | ML runtime | deps are only `numpy` and `pyyaml`. No torch, timm or zarr | torch, timm, zarr, cv2 | optional extra needed (decision) |
| G6 | Resampling | never performed | INTER_AREA from level 0 | a declared `resample` transform is needed to reproduce legacy embeddings |
| G7 | Channel order | explicit; `bgr→rgb` only by declared transform | BGR fed implicitly | the legacy-compatible UNI profile must declare `channel_order: bgr` |
| G8 | Patch identity | `patch_id` + plan index + manifest digest | positional only | key embedding rows by `patch_id`, store the plan index, verify against the manifest digest |
| G9 | Coordinate frame | canonical level 0 | legacy padded | the translation exists; legacy embeddings are importable only via `FrameTranslation` (not proposed for SPEC-005) |
| G10 | Resume | none (in memory) | per-slide `.pt` marker | chunk-level commit log (proposal) |
| G11 | Validity thresholds | tissue and acquired fractions | `min_tissue_coverage 0.2`, `max_nodata 0.5` | equivalent concept; values are per experiment, not core |

---

## 11. Proposed SPEC-005 boundaries (for owner review; not implemented)

```
                    existing PipelineRunner (unchanged engine; additive only)
 ┌───────────────────────────────────────────────────────────────────────────────────┐
 │ tissue.detect@1 → patch.plan@1 → patch.extract@1(materialize_pixels=false)        │
 │                         │                     │ PatchManifest (no pixels)          │
 │                         └──────────┬──────────┘                                    │
 │                                    ▼                                               │
 │  embed.patches@1  (CapabilityGated: resolution + channel + input-spec check)       │
 │   ┌──────────────────────────────────────────────────────────────────────────┐    │
 │   │ PatchSource  ─ bounded queue ─▶ TransformChain ─▶ batch ─▶ EmbeddingEncoder│   │
 │   │ (OpenedImage.read_region,       (declared steps   (≤ max    (batch in,      │   │
 │   │  plan order, ≤ K in flight)      only, recorded)   bytes)    float rows out)│   │
 │   │                                                         │                  │   │
 │   │                                    EmbeddingStore.write(rows, start) ◀─────┘   │
 │   │                                    commit(chunk) … finalize(manifest digest)    │
 │   └──────────────────────────────────────────────────────────────────────────┘    │
 │        output: embedding_bag (EmbeddingManifest + store reference)               │
 └───────────────────────────────────────────────────────────────────────────────────┘
```

| Layer | Proposed content |
|---|---|
| `domain/` | Value objects:<br>• `SourceResolution`, `ExtractionResolution`, `PatchFieldOfView`<br>• `ModelInputSpec`: px, divisor, spacing or any, tolerance, channel order, dtype, normalization<br>• `TransformChain` validation (pure)<br>• `EmbeddingRecord` and `EmbeddingManifest`: schema, patch ids, plan digest, encoder identity and weights checksum, transforms, dtype, dim, store reference, commit state<br>• `BatchPolicy`: batch size, max batch bytes, max in-flight, precision<br>• typed errors such as `RESOLUTION_MISMATCH` and `EMBEDDING_STORE_INCOMPLETE` |
| `ports/` | • `EmbeddingEncoder` (batch protocol + `ModelInputSpec` + component)<br>• `EmbeddingStore` (open / write / commit / committed ranges / finalize)<br>• optionally `PatchSource`, if not kept inside the operation |
| `application/` | • `EmbeddingService`, the analogue of `PreprocessingService`: opens the image and runs through `PipelineRunner`<br>• `gate()` extended through `CapabilityGated` |
| `infrastructure/` | • durable `EmbeddingStore` adapter(s)<br>• weights checksum resolution (no runtime download) |
| `plugins/` | • `plugins/embedding/`: NumPy synthetic encoder for tests (no new deps)<br>• torch runtime and UNI adapter behind an optional extra (pending approval)<br>• explicit transform implementations |
| `pipelines/` | `wsi-patch-embedding.v1.yaml`, plus model/experiment-level resolution profiles (e.g. a "legacy SR1482 UNI" profile with a declared resample and BGR) |

**Invariants to carry into SPEC-005:**

- Pixel memory is bounded by `max_in_flight × patch_bytes + batch_bytes`. Assert this with an automated peak-memory test, not only by measurement.
- No stage retains O(N) pixels. O(N) coordinates and embeddings are allowed and must be declared.
- No resampling unless declared and validated.
- Embedding rows ↔ patch ids are verified against the manifest digest.
- An uncommitted store is never reported as valid.
- One device per run.

---

## 12. Risks and decisions requiring owner approval

| # | Decision / risk | Options | Recommendation |
|---|---|---|---|
| O1 | ML runtime dependencies | torch + timm as optional extra `[embedding]`; or defer | optional extra, never in the base install. Needs approval (no new deps without it) |
| O2 | Durable embedding format | (a) NumPy `.npy` via `open_memmap` (no new dep); (b) Zarr v2 (legacy-compatible, new dep); (c) HDF5 | (a) for SPEC-005; Zarr as an optional adapter later |
| O3 | Channel order for UNI | legacy-compatible BGR (needed to reuse legacy SurGen MIL models) vs RGB (UNI pretraining, NOT VERIFIED) | support both as declared profiles; embeddings from the two are **not interchangeable** and must carry distinct identities |
| O4 | Legacy-parity resampling | reproducing 2013 → 224 INTER_AREA bit-for-bit needs OpenCV (new dep). NumPy area averaging is exact only for integer factors (8.987 is not) | decide whether legacy numerical parity is a SPEC-005 goal; if not, use a declared NumPy/torch resample with documented tolerance |
| O5 | Default extraction resolution | user policy: explicitly selected stored resolution; no universal spacing | require an explicit level or a matched target spacing (no implicit level 0). Note: level 0 at 224 px on F-CZI-01 is a 24.9 µm FOV |
| O6 | Where batching lives | fused `embed.patches` stage (no runner change) vs a streaming edge type in the runner | fused stage; runner changes, if any, strictly additive |
| O7 | Parallel reading | DataLoader-style processes (per-worker reopen) vs threads vs single-thread prefetch | start with bounded threads plus per-thread handles; measure before adding processes. libCZI concurrency is NOT VERIFIED |
| O8 | Precision | fp32 default; bf16/fp16 autocast opt-in | fp32 default; reduced precision recorded and tolerance-tested |
| O9 | Devices | CUDA / CPU / MPS | CPU + CUDA required; MPS opt-in for development |
| O10 | Licensing | legacy `reproducibility/` derives from CLAM (README) and Owkin `rl_benchmarks`; the repo has **no LICENSE file**. UNI / Virchow weights are gated with their own terms | copy no legacy code; license review of model weights before any adapter ships (NOT VERIFIED) |
| O11 | Resume granularity | per slide vs per chunk | per chunk with a commit log; partial stores fail validation |
| R1 | Peak-memory regression | SPEC-004 has no automated memory assertion (report §11) | SPEC-005 acceptance criterion with an automated bound |
| R2 | Model-scale mismatch | the legacy UNI contract is 1.0 µm/px; UNI pretraining scale may differ (NOT VERIFIED) | record as a scientific limitation; per-model profile, not core |

---

## Appendix A — Evidence table

| # | Claim | Code path | Label |
|---|---|---|---|
| E1 | Patching stores coordinates only | `R/wsi_core/WholeSlideImage.py:486-552, 583-702` | VERIFIED |
| E2 | Footprint = `int(patch_size·target_mpp/mpp)` at level 0 | `WholeSlideImage.py:597-600` | VERIFIED |
| E3 | Synthetic 10 power-of-two levels | `WholeSlideImage.py:114-148` | VERIFIED |
| E4 | mpp from `Distance[0]` only | `WholeSlideImage.py:100` | VERIFIED |
| E5 | Candidate grid + `mp.Pool` per contour | `WholeSlideImage.py:658-671` | VERIFIED |
| E6 | Lazy read at `zoom=1.0`, INTER_AREA resize, ImageNet normalization | `R/datasets/dataset_h5.py:32-48, 132-174` | VERIFIED |
| E7 | Coordinate store reopened per item | `dataset_h5.py:136-141` | VERIFIED |
| E8 | DataLoader with workers and `pin_memory`; batch loop; per-batch write | `R/extract_features.py:72-115` | VERIFIED |
| E9 | Preallocated Zarr, chunk = batch, positional writes | `R/utils/file_utils.py:94-142` | VERIFIED |
| E10 | Final `features[:]` → `.pt` | `extract_features.py:463-470` | VERIFIED |
| E11 | `.pt` skip; broken in-progress skip | `extract_features.py:419-429` | VERIFIED |
| E12 | fp32, `no_grad`, CUDA/CPU, DataParallel | `extract_features.py:29, 98-110, 400-406` | VERIFIED |
| E13 | UNI timm construction | `extract_features.py:224-255`; `S/pipeline/encoder_registry.py:204-232` | VERIFIED |
| E14 | `BaseEncoder.preprocess` unused by extraction | grep `get_encoder` (no callers) | VERIFIED |
| E15 | `Bgr24` = B,G,R memory order | `.venv/lib/python3.12/site-packages/pylibCZIrw/czi.py:162` | VERIFIED |
| E16 | Contract declares BGR / INTER_AREA / imagenet | `S/pipeline/preprocessing_config.py:46-53, 163-169`; `tasks/preprocessing/surgen_sr1482_uni_v1.json` | VERIFIED |
| E17 | Orchestrator sets `target_patch_size = patching.patch_size` | `S/run_pipeline.py:641-662`; `S/parallel_preprocess.py:285-298`; `S/analysis_api/artifact_resolution.py:760-781` | VERIFIED |
| E18 | Shape-only embedding validation | `run_pipeline.py:672-711`; `parallel_preprocess.py:158-174` | VERIFIED |
| E19 | Embedding Zarr carries no metadata | `S/pipeline/embedding_artifact.py:5-15` | VERIFIED |
| E20 | Embedding identity hash inputs | `artifact_resolution.py:30-40, 189-248, 640-651` | VERIFIED |
| E21 | Pre-embedding no-data/tissue gate | `S/pipeline/patch_validity.py`; `S/pipeline/czi_coverage.py` | VERIFIED |
| E22 | Prefetch buffer ≈ workers × 2 batches | PyTorch default `prefetch_factor` | INFERRED |
| E23 | Forked workers share the CziReader safely | — | NOT VERIFIED |
| E24 | Virchow / PLIP official preprocessing differs from legacy | model cards not consulted | NOT VERIFIED |
| E25 | SPEC-004 extraction retains all pixels in one payload | `plugins/preprocessing/operations.py:300-444`; `application/runner.py:419-427` | VERIFIED |
| E26 | SPEC-004 never resamples; stored-level selection | `domain/patching.py:154-214` | VERIFIED |
| E27 | Channel reorder only via declared transform | `domain/preprocessing.py` `resolve_channel_conversion` | VERIFIED |
| E28 | pathology-ai base deps: numpy, pyyaml | `pyproject.toml` | VERIFIED |

## Appendix B — Constraint compliance

- `../digital-pathology` and `../hal-viewer` were only read. No writes, no commands that modify files, no commits.
- No implementation code. No changes to SPEC-001…004. No dependencies added. No Git commits.
- No universal physical resolution is proposed. Resampling is permitted only as a declared, validated, recorded transform.
- Bounded memory is claimed only where the data path was traced, and labelled INFERRED where it rests on library defaults. Nothing was measured.
