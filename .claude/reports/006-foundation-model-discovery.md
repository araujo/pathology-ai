# Discovery — SPEC-006 Foundation Model Integration (Phase A)

**Date:** 2026-10-21
**Prompt:** `.claude/prompts/006-foundation-model-integration-and-real-embeddings.md`
**Authoritative inputs:** SPEC-006, ADR-006 (both *Proposed*), SPEC/ADR-005, report 005, architecture 005, SPEC/ADR-001…004. All present.
**Outcome:** No substantive architectural conflict. The dependency group named by ADR-006 D3 is added as an optional extra. Implementation proceeds. Two items need owner decisions and are flagged in §9; neither blocks the generic work.

Verification status labels: **VERIFIED** (read in the cited source or measured here), **NOT VERIFIED** (no authoritative source available), **INFERRED** (follows from verified facts, stated as such).

---

## 1. Existing SPEC-005 contracts (VERIFIED, read in this repository)

| Contract | Location | What SPEC-006 reuses |
|---|---|---|
| `EmbeddingEncoder` | `ports/__init__.py` | Framework-neutral port: `identity`, `input_spec`, `encode_batch(batch, EncodeContext)`, `working_bytes(batch)`, `close()`. A torch adapter fits behind it unchanged. |
| `EmbeddingEncoderFactory` | `ports/__init__.py` | `name`, `input_spec(config)` (used by the capability gate without loading), `create(config)`. |
| `ModelInputSpec` | `domain/embedding.py` | Geometry, channel order, float dtype, `scaled_range`, layout, `Normalization`, required spacing **or** explicit variable/unknown spacing, tolerance, divisor, precisions. Sufficient to express both UNI profiles. |
| `TransformChain` / `build_transform_chain` | `domain/embedding.py`, `plugins/embedding/transforms.py` | `channel_reorder`, `drop_alpha`, `cast`, `scale`, `resample(area, integer factor only)`, `normalize`, `layout`. Each step versioned. |
| `BatchPolicy` / `BatchBudget` | `domain/embedding.py` | Host budget = in-flight input batches + output batch + scratch, where scratch includes `encoder.working_bytes(batch)`. |
| `EncoderIdentity` | `domain/embedding.py` | `component`, `model`, `weights_checksum`, `synthetic`, sorted `parameters`, `output_dim`, `output_dtype`. Bound into `EmbeddingRunIdentity` → `store_id`. |
| `EmbeddingStore` | `infrastructure/embedding_store.py` | Chunk = batch; SHA-256 per chunk; resume from verified chunks; `INCOMPATIBLE` on identity/layout change. |
| `embed.patches@1` | `plugins/embedding/operations.py` | Encoders selected by stage config `encoder` from a name → factory mapping. Calls `factory.create(cfg)` and always `encoder.close()`. |
| Encoder factory registration | `bootstrap.py` | `EmbeddingOperation({name: factory})`. No registry object with metadata exists yet. |
| `PipelineRunner` integration | `application/runner.py`, `application/embedding_service.py` | Store location travels in `ExecutionRequest.extras`, outside fingerprints. |

Observations relevant to SPEC-006:

1. **No generic layer needs a model name.** UNI fits entirely behind `EmbeddingEncoderFactory`. The AST rules already forbid `\buni\b` in `domain`, `ports` and `application`.
2. **Batch invariance.** SPEC-005 requires encoders to be batch-invariant or bind batch size into identity (report 005 §11). A torch adapter cannot guarantee invariance across hardware, so it will bind the configured batch size into its identity.
3. **Encoder-side provenance.** There is no channel for an encoder to report runtime resources (parameter bytes, device peak memory, timings) or encoder-specific limitations. An additive, defaulted channel is needed (§8).
4. **Checkpoint path.** A path in stage config would enter the execution fingerprint (config is fingerprinted). SPEC-006 §6 says the path is operational and the digest is scientific. The path must therefore be supplied at composition time, not in stage config.

## 2. Legacy UNI implementation (read-only inspection of `../digital-pathology`)

Repository state, unchanged by this work: `../digital-pathology` HEAD `c85df8a`, 4 status lines; `../hal-viewer` HEAD `e248202`, 9 status lines. No legacy code was copied.

| Aspect | Legacy behaviour | Evidence |
|---|---|---|
| Architecture | `timm.create_model("vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True)` | `reproducibility/extract_features.py:251-252` (VERIFIED) |
| Checkpoint load | `model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=True)`. The file is `uni.pth`, auto-downloaded from `MahmoodLab/UNI` when absent. There is no digest check. | `extract_features.py:224-255` (VERIFIED) |
| Eval / gradients | `models[name].eval()`; forward under `torch.no_grad()` | `extract_features.py:98, 405` (VERIFIED) |
| Region geometry | Reads at level 0, `zoom=1.0`. The region size is computed from `target_mpp=1.0`: 224 px at 1.0 µm/px is about 2013 level-0 px on F-CZI-01. | `reproducibility/datasets/dataset_h5.py:155-160`; contract `surgen-sr1482-uni-224px-1mpp-v1` (VERIFIED) |
| Resampling | `cv2.resize(img, (224,224), interpolation=cv2.INTER_AREA)` on **uint8**. The output is rounded to uint8. 2013 → 224 is a non-integer factor (8.987). | `dataset_h5.py:166` (VERIFIED) |
| Channel order | pylibCZIrw `Bgr24` is passed through unconverted, so the encoder receives **BGR**. The contract pins `channel_order: BGR`. | `scripts/pipeline/preprocessing_config.py:46-53`; architecture 005 E15/E16 (VERIFIED) |
| Normalization | `ToTensor()` (÷255) then `Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))`. The RGB-ordered ImageNet constants are applied to B, G, R positions in that order. | `dataset_h5.py:32-47`; contract `normalization: imagenet` (VERIFIED) |
| Precision / device | fp32; CUDA with `DataParallel` when available | `extract_features.py:400-406` (VERIFIED) |
| Association | Positional only, padded legacy frame | architecture 005 §… (VERIFIED) |

## 3. Official UNI documentation (verified 2026-10-21)

Sources consulted:

- **S1:** the Hugging Face model card `huggingface.co/MahmoodLab/UNI`, public part. Public API: `sha b55a5ec6cade1a39edfe6534189a9b8ca7a022f0`, `license: cc-by-nc-nd-4.0`, `gated: auto`, `library_name: timm`.
- **S2:** the authors' GitHub repository `github.com/mahmoodlab/UNI`. `README.md` and `uni/get_encoder/get_encoder.py` were fetched at HEAD `42715efc11722a496e0a67f3369505a8f277206c` into the session scratchpad, not into the repository.
- **S3:** the public file listing (`/api/models/MahmoodLab/UNI/tree/main`).

| Item | Official statement | Source | Status |
|---|---|---|---|
| Architecture | ViT-L/16 trained with DINOv2. Code: `timm.create_model('vit_large_patch16_224', img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True)` | S1, S2 `get_encoder.py` (`uni_kwargs`) | VERIFIED |
| Checkpoint format | `pytorch_model.bin`, a plain state dict loaded with `torch.load(..., map_location="cpu")` and `load_state_dict(strict=True)` | S2 `get_encoder.py` | VERIFIED |
| Checkpoint size | 1,213,527,781 bytes | S3 | VERIFIED |
| Checkpoint SHA-256 | Masked in public metadata (gated repo) | S3 | NOT VERIFIED from a public source. See §6 for local evidence. |
| Preprocessing | `Resize(224)`, `CenterCrop(224)`, `ToTensor()`, `Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))` on a PIL RGB image | S2 README (lines 414-419), `get_eval_transforms` + `get_encoder(center_crop=True, which_img_norm='imagenet')` | VERIFIED |
| Channel order | RGB (PIL `Image.open` → `ToTensor`) | S2 README example | VERIFIED |
| Hub `pretrained_cfg` (`resolve_data_config`) | The model card's alternative path reads the gated `config.json`. Its interpolation and `crop_pct` cannot be read without authentication. | S1 | NOT VERIFIED |
| Output | Features `[1, 1024]`; downstream use is "on the class token" | S1, S2 | VERIFIED |
| Magnification / scale | Pretraining data is "Mass-100K … at 20× resolution", made of 256² and 512² images. **No µm/px value is stated.** | S1 | Scale VERIFIED only as "20×"; µm/px NOT VERIFIED |
| Licence | CC-BY-NC-ND 4.0, non-commercial academic research only. Gated access (registration with an institutional email). "You agree not to distribute, publish or reproduce a copy of the models." Derivative models and datasets created from UNI are covered. | S1, S2 README:466 | VERIFIED |

Local measurements with the real checkpoint (§6), torch 2.14.1, timm 1.0.30, macOS arm64 (M4 Pro):

- The state dict holds 342 float32 tensors with 303,350,784 parameters. Shapes: `cls_token (1,1,1024)`, `pos_embed (1,197,1024)`, `patch_embed.proj.weight (1024,3,16,16)`, `blocks.*.ls{1,2}.gamma`, `norm.*`. There is no head.
- `load_state_dict(strict=True)` returns "All keys matched".
- `model(x)` = `forward_head(forward_features(x), pre_logits=True)` = the normalized CLS token `(B, 1024) float32` (`global_pool='token'`).
- Rows at batch 1 and batch 8 are bit-identical on this CPU. They are also bit-identical at 1 and 8 threads.
- **Caution:** timm's built-in default `pretrained_cfg` for `vit_large_patch16_224` is mean/std 0.5, bicubic, `crop_pct` 0.9. That is **not** UNI's documented recipe, so the adapter must never derive preprocessing from timm defaults.

## 4. Official vs legacy preprocessing

| Step | Official UNI (S1/S2) | Legacy SurGen | Same? |
|---|---|---|---|
| Source geometry | Not specified. Any image is resized to a shorter side of 224, then center-cropped. | Level-0 region of about 2013 px for 1.0 µm/px | — |
| Physical scale | "20×" (training data), no µm/px | 1.0 µm/px target (about 10× on a 40× scan) | **No**. Legacy is about half the documented magnification (INFERRED from 20× ≈ 0.5 µm/px convention, NOT VERIFIED). |
| Resize | `Resize(224)`: torchvision bilinear, antialiased (PIL) | `cv2.INTER_AREA` on uint8, then rounded to uint8 | **No** |
| Crop | `CenterCrop(224)` (identity on a 224² input) | none | identity on 224² only |
| Channel order | RGB | BGR (unconverted pylibCZIrw output) | **No** |
| Scaling | ÷255 (`ToTensor`) | ÷255 (`ToTensor`) | yes |
| Normalization | ImageNet mean/std in R, G, B order | the same constants applied in B, G, R order | **No** (as a function of colour) |
| Layout / dtype | CHW float32 | CHW float32 | yes |
| Output | CLS token, 1024 | CLS token, 1024 | yes |

The legacy recipe is therefore **not** scientifically equivalent to the official one, and the two profiles must not share an identity (ADR-006 D7).

## 5. Transform capability gaps against SPEC-005

| Needed by | Transformation | SPEC-005 support | Consequence |
|---|---|---|---|
| Official, when patch ≠ 224 px | torchvision bilinear antialiased `Resize(224)` | **Not implemented** (only integer-factor area averaging) | The official profile runs only on **224 × 224 patches extracted at a stored level**, where `Resize(224)` and `CenterCrop(224)` are identities. Any declared resampling is rejected rather than substituted with `area`. |
| Official | RGB from BGR (CZI) / RGBA (SVS) | `channel_reorder`, `drop_alpha` | supported, declared |
| Official | ÷255, ImageNet normalize, CHW | `cast`, `scale`, `normalize`, `layout` | supported. Note: `scale` multiplies by float32(1/255), while `ToTensor` divides by 255; results may differ by at most 1 ulp (numerical tolerance, not a recipe change). |
| Legacy | `INTER_AREA` uint8 2013 → 224 (non-integer, rounded) | **Not implemented** | The legacy profile is **defined, identified and rejected with a typed capability error**. It is never approximated by float integer-factor area averaging. |
| Legacy | RGBA → BGR (SVS) | not a declared conversion | rejected with `CHANNEL_ORDER_MISMATCH` (legacy was CZI-only) |

## 6. Local checkpoint availability

| Item | Finding |
|---|---|
| File | `../digital-pathology/data/checkpoints/huggingface/uni.pth`. It is git-ignored in the legacy repository, and a byte-identical copy exists under `~/projects/digital-pathology/…`. It is **used in place, read-only, through an explicit path**. It is not copied into this repository. |
| Size | 1,213,527,781 B, equal to the public size of `pytorch_model.bin` (S3) |
| SHA-256 (computed) | `56ef09b44a25dc5c7eedc55551b3d47bcd17659a7a33837cf9abc9ec4e2ffb40` |
| Download provenance | `huggingface_hub` metadata next to the file (`.cache/huggingface/download/pytorch_model.bin.metadata`) records commit `b55a5ec6cade1a39edfe6534189a9b8ca7a022f0` (the current public repo `sha`) and etag `56ef09b4…fb40`. For LFS files the etag is the content SHA-256, and it equals the computed digest. |
| Authorization | The file was obtained through the owner's gated Hugging Face access for the legacy project (INFERRED from the metadata; the access grant itself is NOT VERIFIED here). Use is limited to local non-commercial research inference. Nothing is redistributed, and embeddings derived from UNI inherit the licence restrictions on derivatives. |

## 7. Optional runtime on Python 3.12 (VERIFIED)

| Package | Version installed | Licence | Notes |
|---|---|---|---|
| torch | 2.14.1 (macOS arm64 wheel, CPU + MPS) | BSD-3-Clause | `torch.load(weights_only=True)` loads the UNI state dict. CUDA is unavailable on this host. |
| timm | 1.0.30 | Apache-2.0 | It pulls in `torchvision`, `huggingface_hub` and `safetensors` as dependencies. Only `timm.create_model(..., pretrained=False)` is used. The adapter also sets `HF_HUB_OFFLINE=1` before importing timm. |
| numpy | 2.5.4 | BSD | unchanged base dependency |

Measured on CPU (8 threads):
- ViT-L/16 runs at about 52 ms per patch at batch 8 to 16.
- Activation peak is about 14.4 MiB per sample (231 MiB at batch 16, 453 MiB at batch 32), above 1.16 GiB of fp32 parameters.

MPS runs batch 16 in 0.45 s, with max |CPU − MPS| = 3.5e-5. MPS is optional and not an acceptance target.

## 8. Planned additive changes (no conflict with accepted or proposed ADRs)

1. `domain/foundation.py`: generic model-runtime contracts. These cover `DeviceKind`, `CheckpointExpectation` and `VerifiedCheckpoint` (digest only, no path), `PreprocessingProfile` (status `available` or `blocked`), `EncoderDescriptor` and `EncoderRuntimeReport`. The module has no model names.
2. `domain/errors.py`: `MODEL_RUNTIME_UNAVAILABLE`, `CHECKPOINT_MISSING`, `CHECKPOINT_INTEGRITY`, `CHECKPOINT_INCOMPATIBLE`, `DEVICE_UNAVAILABLE`, `PREPROCESSING_PROFILE_UNAVAILABLE`, `MODEL_INFERENCE`.
3. `ports`: `EmbeddingEncoderFactory.descriptor` and `EmbeddingEncoder.runtime_report()`. The report's limitations enter the manifest. Its resources enter `EmbeddingMetrics.encoder_resources`, which is a defaulted field outside every digest.
4. `application/registries.py`: `EncoderRegistry`, keyed by stable name, with validated descriptors. It fails closed.
5. `plugins/models/`: the generic local-checkpoint resolver and SHA-256 verifier (no torch), a lazily imported torch runtime (device, precision, strict state-dict loading, `TorchEncoder` lifecycle), and a lightweight torch fixture encoder.
6. `plugins/models/uni/`: the UNI adapter with `uni-official@1` and `uni-legacy-surgen@1`.
7. `pyproject.toml`: the optional extra `embedding = ["torch>=2.4,<3", "timm>=1.0.9,<2"]`, which ADR-006 D3 names explicitly. The architecture rule scopes torch and timm to `plugins/models/`.
8. Composition: checkpoint paths are bound at composition time. The sources are `build_platform(checkpoints=…)`, the CLI flag `--checkpoint NAME=PATH`, and the environment variable `PATHOLOGY_AI_CHECKPOINTS`. Paths never enter stage config, fingerprints or identities.

## 9. Decisions requiring the owner (non-blocking)

| # | Question | Interim handling |
|---|---|---|
| O1 | **Physical scale of the official profile.** UNI documents "20×" but no µm/px. | The official profile **requires** an explicit `model_spacing_um` and `model_spacing_tolerance` in the stage config. Without them the profile fails with `PREPROCESSING_PROFILE_UNAVAILABLE`. The value is recorded in the encoder identity as an operator binding, not as an official fact. The shipped UNI pipeline declares 0.5 µm/px ± 15 % and labels it an experiment-level interpretation of "20×". The owner should confirm or replace it. |
| O2 | **Use of the legacy-downloaded checkpoint.** | It is used read-only in place, for local research validation, with the digest pinned. The owner should confirm that this access is covered by their UNI terms acceptance. |

## 10. Blocked or unverifiable items

- The hub `pretrained_cfg` (gated) is NOT VERIFIED. The profile follows the authors' explicit transform (S2) instead.
- The public UNI checkpoint SHA-256 is masked. The pin rests on local download metadata plus the public size and commit.
- CUDA is unavailable on this host, so CUDA execution will be reported NOT VERIFIED.
- Legacy `INTER_AREA` uint8 resampling is unsupported. The legacy profile is identified, but execution is blocked.
