# 006 — Foundation Model Integration: Architecture

**Status:** Implemented. SPEC-006 and ADR-006 remain *Proposed* until the owner accepts them.
**Date:** 2026-10-21
**Related:** `../specs/006-foundation-model-integration-and-real-embeddings.md`, `../adr/006-foundation-model-adapter-architecture.md`, `../reports/006-foundation-model-discovery.md`, `005-streaming-embedding-pipeline.md`

---

## 1. Shape

```text
PipelineRunner (unchanged)
└─ embed.patches@1 (unchanged loop; two additive calls: descriptor-aware factory, runtime_report)
     │  stage config: encoder, profile, model_spacing_*, device, precision, batch_size, ...
     ▼
   EncoderRegistry (application)  ── name → EmbeddingEncoderFactory + EncoderDescriptor
     ├─ reference-projection   NumPy fixture (SPEC-005)
     ├─ torch-fixture          tiny CNN, random local weights (tests)
     └─ uni                    ViT-L/16, uni-official@1 (available), uni-legacy-surgen@1 (blocked)
             │ create(config)
             ▼
   plugins/models/checkpoints  CheckpointLocator (name → local path, operational)
                               expected_checkpoint (pin) → verify_checkpoint (SHA-256, size, stat)
             ▼  (lazy import; MODEL_RUNTIME_UNAVAILABLE without torch)
   plugins/models/torch_runtime
        select_device / select_dtype       explicit, fail closed, no fallback, no AMP
        load_module                        meta build → weights_only load → key/shape/dtype check
                                           → strict assign → no meta left → stat recheck → eval,
                                           requires_grad(False) → device/dtype
        probe_output                       one zero batch: (1, output_dim), finite
        TorchEncoder                       inference_mode, typed MODEL_INFERENCE, host rows,
                                           working_bytes (host), runtime_report (resources), close
```

There is no second runner, no model-specific branch in `domain`, `ports`, `application`, the
operation or the streaming loop. AST tests enforce three things:

- `torch` and `timm` imports are scoped to `plugins/models/`.
- The torch runtime is imported lazily.
- No model names appear in the generic modules.

## 2. Layers

| Layer | Module | Content |
|---|---|---|
| domain | `domain/foundation.py` | `DeviceKind`, `parse_device`, `CheckpointExpectation`, `VerifiedCheckpoint` (digest, no path), `ProfileStatus`, `PreprocessingProfile`, `EncoderDescriptor`, `EncoderRuntimeReport` |
| domain | `domain/errors.py` | `MODEL_RUNTIME_UNAVAILABLE`, `CHECKPOINT_{MISSING,INTEGRITY,INCOMPATIBLE}`, `DEVICE_UNAVAILABLE`, `PREPROCESSING_PROFILE_UNAVAILABLE`, `MODEL_INFERENCE` |
| domain | `domain/embedding.py` | `EmbeddingMetrics.encoder_resources` (defaulted; outside every digest) |
| ports | `ports/__init__.py` | `EmbeddingEncoderFactory.descriptor`, `EmbeddingEncoder.runtime_report()` |
| application | `application/registries.py` | `EncoderRegistry` (a `Mapping` the operation consumes; fail-closed `resolve`) |
| plugins | `plugins/embedding/operations.py` | merges `runtime_report().limitations` into the manifest and `.resources` into the metrics; adds a generic "pretrained encoder: technical integration only" limitation |
| plugins | `plugins/models/` | `checkpoints.py` (stdlib), `torch_runtime.py`, `fixture.py` + `fixture_net.py`, `uni.py` |
| composition | `bootstrap.py`, `cli.py` | registry wiring, `build_platform(checkpoints=…)`, `$PATHOLOGY_AI_CHECKPOINTS`, `pathology-ai-embed --checkpoint NAME=PATH --pipeline ID@VER` |
| definitions | `pipelines/wsi-patch-embedding-uni.v1.yaml` | the same stages as `wsi-patch-embedding`, with 224 px patches and the `uni` encoder |
| packaging | `pyproject.toml` | extra `embedding = ["torch>=2.4,<3", "timm>=1.0.9,<2"]`, and timm mypy override |

## 3. Identities

| What | Where | Enters identity? |
|---|---|---|
| Checkpoint path | `CheckpointLocator` (composition) | **No.** It never appears in stage config, fingerprints, manifests or identities. |
| Checkpoint SHA-256 | `EncoderIdentity.weights_checksum` | **Yes**, via the run identity, `store_id` and manifest. |
| Architecture, profile ref, scale binding, output (`cls_token`), checkpoint format | `EncoderIdentity.parameters` | **Yes** |
| Runtime versions (`runtime.torch`, `runtime.timm`) | `EncoderIdentity.parameters` | **Yes** (ADR-006 D9: a library upgrade is a new execution) |
| Configured `batch_size` | `EncoderIdentity.parameters` | **Yes**, for torch encoders, because torch does not guarantee batch invariance across hardware |
| Device, precision | `EmbeddingRunIdentity` (SPEC-005) | **Yes** |
| Parameter bytes, device peak, timings, threads, platform | `EmbeddingMetrics.encoder_resources` | **No** |
| Licence, scale binding and numeric-tolerance statements | `EmbeddingManifest.limitations` | in the manifest digest, deterministic |

Consequences:
- `uni-official@1` and `uni-legacy-surgen@1` cannot share a store or an embedding identity.
- Different weights cannot share a store, and neither can different devices.
- An interrupted store is resumed only under the identical identity. A run with any other identity opens a different store and leaves the interrupted one untouched.

## 4. Checkpoint control

1. **Binding.** The operator binds `NAME=PATH` through `build_platform(checkpoints=…)`, `--checkpoint`, or `$PATHOLOGY_AI_CHECKPOINTS` (entries separated by `os.pathsep`). Explicit bindings override the environment. Hub IDs and URLs are rejected (`CHECKPOINT_MISSING`).
2. **Pin.** The adapter pins the digest (UNI: `56ef09b4…fb40`, 1,213,527,781 B). Otherwise the stage config must declare `checkpoint_sha256` (torch-fixture). If both are given they must agree. Unpinned weights are never loaded.
3. **Verify.** The verifier checks the file exists and is a regular file, then checks its size (if pinned), then hashes the full file with SHA-256 *before* deserialization. It then records the stat.
4. **Load.** The checkpoint is read with `torch.load(weights_only=True)`. It must be a flat `str → Tensor` mapping. Missing, unexpected, mis-shaped or wrong-dtype keys raise `CHECKPOINT_INCOMPATIBLE`, with the offending names listed. The stat is rechecked after loading, and a replaced file raises `CHECKPOINT_INTEGRITY`.
5. **Network.** Nothing is downloaded. `timm.create_model(..., pretrained=False)` builds on `meta`, and `HF_HUB_OFFLINE=1` is set before timm is imported. A test blocks `socket` during loading and inference.

Filesystem assumption: one local writer, as for the SPEC-005 store. SHA-256 detects substitution and corruption. It does not authenticate the publisher.

## 5. Preprocessing profiles (UNI)

| | `uni-official@1` | `uni-legacy-surgen@1` |
|---|---|---|
| Status | available | **blocked** (`PREPROCESSING_PROFILE_UNAVAILABLE`) |
| Basis | authors' README and `get_encoder` (verified) | legacy SurGen contract (read-only inspection) |
| Patch | 224 × 224 at a stored level. `Resize(224)` and `CenterCrop(224)` are identities. | about 2013 level-0 px, `cv2.INTER_AREA` on uint8 down to 224 |
| Resampling | none; any declared method is refused | not implemented, so the profile is refused (never approximated) |
| Channels | RGB (declared `bgr->rgb` and `rgba->rgb` conversions) | BGR |
| Scaling / normalization | /255, ImageNet mean/std in R, G, B | /255, ImageNet constants in B, G, R positions |
| Scale | operator binding `model_spacing_um` ± `model_spacing_tolerance`, required, no default | 1.0 µm/px |
| Precision | fp32 only | fp32 |

Selecting a profile is mandatory (`profile:`), and no default exists. A blocked or unknown profile fails before any pixel is read, at the capability gate (`input_spec`), and again in `create`.

## 6. Memory and devices

The host budget (SPEC-005 `BatchPolicy`) stays authoritative. `TorchEncoder.working_bytes(b)` reports host memory:

- **On CPU:** `b × activation_bytes_per_sample + 2 × b × output_dim × 4`.
- **On an accelerator:** only the output copies. Activations live in device memory, which is reported as `device_peak_bytes`.

For ViT-L/16, `vit_activation_bytes` gives about 20.4 MiB per sample. The measured value is about 14.4 MiB per sample.

Parameter memory is reported separately (`parameter_bytes`). It is not part of the per-batch budget, because it is constant for the run. Model memory is about 1.16 GiB fp32 for UNI. CPU RSS for a full UNI run is about 1.7 GiB.

Devices are `cpu`, `cuda[:i]` and `mps`. A requested device must be supported by the descriptor and available, otherwise the run fails with `DEVICE_UNAVAILABLE` before any pixel is read. There is no fallback. Precision is explicit, there is no autocast, and UNI and the fixture validate fp32 only.

## 7. Lifecycle

`embed.patches@1` already closed encoders in `finally`. `TorchEncoder.close()` drops the module, collects garbage and empties the CUDA/MPS cache. It is idempotent. Later `encode_batch` calls raise `MODEL_INFERENCE`. Every forward pass runs under `torch.inference_mode()`. The module is in `eval()` with `requires_grad=False`. A failing forward pass, for example a device error or out-of-memory, raises `MODEL_INFERENCE`. The batch is never committed, and the store remains `interrupted` and resumable.

## 8. Adding another encoder (UNI2-h, Virchow, PLIP …)

1. Add `plugins/models/<model>.py` with a descriptor, a checkpoint pin, profiles, `input_spec(config)` and `create(config)` that builds a `TorchEncoder` through `torch_runtime`.
2. Register it in `bootstrap.py`.
3. Add a pipeline definition with its explicit patch geometry, scale binding and conversions.

No change to `domain`, `ports`, `application`, the operation or the streaming loop is needed. A transform the model requires that is not implemented, such as a bilinear resize, must be added to the declared transform vocabulary (domain + NumPy implementation, versioned), never inside the adapter.
