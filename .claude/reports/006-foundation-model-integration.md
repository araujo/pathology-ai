# Report — SPEC-006 Foundation Model Integration & Real Embedding Extraction

**Date:** 2026-10-21
**Prompt:** `.claude/prompts/006-foundation-model-integration-and-real-embeddings.md`
**Authoritative inputs:** SPEC-006 and ADR-006 (both *Proposed*), SPEC/ADR-005, report 005, architecture 005, SPEC/ADR-001…004
**Companion documents:**
- `006-foundation-model-discovery.md` (Phase A)
- `../architecture/006-foundation-model-integration.md`

**Outcome:**

| Verdict | Acceptance criteria | Notes |
|---|---|---|
| PASS | 19 | AC-007 passes with an explicit operator scale binding (owner decision O1). |
| NOT VERIFIED | 1 (AC-012, CUDA) | No CUDA hardware is available. |

- **Real UNI.** It was executed end to end on both real fixtures with the authorized local checkpoint (SHA-256 `56ef09b4…fb40`), on CPU and on MPS (optional).
- **Proposed documents.** SPEC-006 and ADR-006 are left as *Proposed*.
- **Commits.** None were created.

> Technical integration only. Nothing in this report is evidence of clinical validity,
> biomarker accuracy, diagnostic use or biological meaning of any embedding.

---

## 1. Implementation summary

The existing `embed.patches@1` operation now runs real pretrained encoders. Nothing else changed in how it runs: there is no new runner, no new loop and no new store.

| Component | Where | What it does |
|---|---|---|
| Generic contracts | `domain/foundation.py` | Encoder descriptor, checkpoint expectation and verified checkpoint, preprocessing profile (available or blocked), device kinds, runtime report |
| Typed errors | `domain/errors.py` | Seven new codes |
| Registry | `application/registries.py` | `EncoderRegistry`: stable names, validated descriptors, fail-closed resolution |
| Local checkpoints | `plugins/models/checkpoints.py` | Binding of checkpoint files and SHA-256 verification. Standard library only. |
| Torch runtime | `plugins/models/torch_runtime.py` | Lazily imported. Covers device and precision selection, safe and strict loading, `TorchEncoder` lifecycle and inference, resource reporting. |
| Test encoder | `plugins/models/fixture.py` | `torch-fixture`: a lightweight local torch encoder with seeded random weights |
| UNI adapter | `plugins/models/uni.py` | Verified ViT-L/16, pinned checkpoint, class-token output. Profiles `uni-official@1` (available) and `uni-legacy-surgen@1` (blocked). |
| Composition | `bootstrap.py`, `cli.py` | `build_platform(checkpoints=…)`, `$PATHOLOGY_AI_CHECKPOINTS`, `pathology-ai-embed --checkpoint NAME=PATH --pipeline ID@VERSION` |
| Pipeline | `pipelines/wsi-patch-embedding-uni.v1.yaml` | Same stages as `wsi-patch-embedding`, with 224 px patches and `uni-official@1` |
| Packaging | `pyproject.toml` | Optional extra `embedding = ["torch>=2.4,<3", "timm>=1.0.9,<2"]`, which ADR-006 D3 names explicitly |

## 2. Verified model documentation and licensing

The full evidence is in the discovery report §3. In summary:

- **Architecture.** `timm vit_large_patch16_224` with `img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True`. The checkpoint is loaded with strict `load_state_dict` from `pytorch_model.bin`.
  - Sources: the Hugging Face model card (repository sha `b55a5ec`) and the authors' `get_encoder.py` at commit `42715ef`. Status: VERIFIED.
- **Preprocessing.** `Resize(224)`, then `CenterCrop(224)`, then `ToTensor`, then ImageNet mean/std, on an RGB image.
  - Source: the authors' README and `get_eval_transforms`. Status: VERIFIED.
  - The gated hub `pretrained_cfg` could not be read: NOT VERIFIED.
- **Output.** The class token, 1024 values (VERIFIED). Measured: `forward_features(x)[:, 0]` is identical to the pooled output.
- **Scale.** The documentation says "20×" and gives no µm/px. The µm/px value is NOT VERIFIED.
- **Licence.** CC-BY-NC-ND 4.0. Access is gated. Use is limited to non-commercial academic research. Weights may not be redistributed, and derivatives are covered (VERIFIED).
  - No weights are bundled or copied. The checkpoint is read in place.
  - The UNI licence statement is written into every UNI manifest's `limitations`.

## 3. Architectural decisions and deviations

| # | Decision | Rationale |
|---|---|---|
| D1 | Checkpoint paths are bound at composition time (`build_platform`, CLI, environment variable), not in the stage config. | Stage config is fingerprinted, and the path is operational (SPEC-006 §6). The digest goes into `EncoderIdentity.weights_checksum`. |
| D2 | `EmbeddingEncoderFactory.descriptor` and `EmbeddingEncoder.runtime_report()` were added to the ports, with `EmbeddingMetrics.encoder_resources` defaulted. | Registry metadata is available without loading. Limitations enter the manifest, which is deterministic. Measurements enter only the metrics. The reference encoder reports nothing, so the SPEC-005 artifacts are byte-identical (§11). |
| D3 | Torch encoders bind the configured `batch_size` and the `torch`/`timm` versions into their identity. | ADR-006 D9: invariance across batch size is measured on this CPU but not guaranteed. |
| D4 | The model is built on `meta`, keys, shapes and dtypes are checked, then the tensors are assigned strictly. | This gives typed, listed mismatch errors before any allocation. It avoids random initialization of 303 M parameters and roughly halves peak memory. |
| D5 | Only `torch.load(weights_only=True)` is used. | Arbitrary pickled objects are refused. A test covers this. |
| D6 | The UNI official profile needs an explicit `model_spacing_um` and `model_spacing_tolerance`. | The authors document "20×" only. A missing binding raises `PREPROCESSING_PROFILE_UNAVAILABLE`, which is the "clear blocking status" of AC-007. The pipeline YAML declares 0.5 ± 15 % and labels it an experiment-level interpretation (**owner decision O1**). |
| D7 | The official profile refuses any declared resampling. | The official resize is torchvision bilinear with antialiasing. Area averaging is not the same recipe and is never substituted. |
| D8 | The legacy profile is defined and identified, then refused. | The legacy uint8 `INTER_AREA` with a non-integer factor (2013 → 224) is not implemented. SPEC-006 §8 says to reject it rather than approximate it. |
| D9 | `HF_HUB_OFFLINE=1` is set before timm is imported, and `pretrained=False` is passed explicitly. | Defence in depth for "no runtime network" (P6). |
| D10 | `torch` and `timm` imports are scoped to `plugins/models/`, the runtime is imported lazily, and generic modules contain no model names. These rules are enforced by AST tests. | AC-001 and AC-002 |

**Deviations from the prompt or spec:**
- None substantive.
- The UNI adapter is a module (`plugins/models/uni.py`), not a package.
- No separate pipeline YAML ships for `torch-fixture`. Tests derive one from `wsi-patch-embedding`, because overrides cannot add configuration keys.

## 4. Model registry and adapter structure

| Name | Adapter | Architecture | Checkpoint | Devices | Precision | Profiles | Batch-invariant |
|---|---|---|---|---|---|---|---|
| `reference-projection` | `encoder/reference-projection@1` | pooled statistics + random projection | none (synthetic) | cpu | fp32 | `reference@1` | yes |
| `torch-fixture` | `encoder/torch-fixture@1` | conv(8,s8)–conv(4,s4)–avgpool–linear–tanh | `checkpoint_sha256` declared in the stage config | cpu, cuda, mps | fp32 | `fixture@1` | no |
| `uni` | `encoder/uni@1` | `timm/vit_large_patch16_224(...)`, output = class token | pinned `56ef09b4…fb40`, 1,213,527,781 B | cpu, cuda, mps | fp32 | `uni-official@1`, `uni-legacy-surgen@1` (blocked) | no |

Resolution is fail-closed. Each of the following raises a typed error:

| Failure | Error |
|---|---|
| Unknown name | `EXECUTION_UNSUPPORTED` |
| No checkpoint bound | `CHECKPOINT_MISSING` |
| Torch is absent | `MODEL_RUNTIME_UNAVAILABLE` |
| Profile blocked or missing | `PREPROCESSING_PROFILE_UNAVAILABLE` |
| Requested device is unavailable | `DEVICE_UNAVAILABLE` |

How to add another encoder is described in the architecture document §8.

## 5. Official vs legacy preprocessing

| | `uni-official@1` | `uni-legacy-surgen@1` |
|---|---|---|
| Patch | 224² at a stored level (`Resize` and `CenterCrop` are identities) | about 2013 level-0 px, resized to 224 with uint8 `INTER_AREA` |
| Resampling | none (any declared method is refused) | **unsupported, so the profile is refused** |
| Channels | RGB, via declared `bgr->rgb` / `rgba->rgb` | BGR |
| Normalization | ImageNet in R, G, B order | ImageNet constants applied in B, G, R order |
| Scale | operator binding (default in the UNI pipeline: 0.5 µm/px ± 15 %) | 1.0 µm/px |
| Status | available | blocked |
| Input-spec fingerprint | differs from legacy even at an equal scale (`test_official_and_legacy_profiles_have_distinct_identities`) | — |

**Known numeric difference.**
- The declared `scale` step multiplies by float32(1/255), whereas `ToTensor` divides by 255. This can differ by at most 1 ulp.
- This is a numerical tolerance, not a change of recipe. It is recorded in each UNI manifest's limitations.

## 6. Checkpoint integrity evidence

**Real checkpoint.**
- `../digital-pathology/data/checkpoints/huggingface/uni.pth`, read in place.
- SHA-256 `56ef09b44a25dc5c7eedc55551b3d47bcd17659a7a33837cf9abc9ec4e2ffb40`, which equals the LFS etag in the `huggingface_hub` download metadata.
- Size 1,213,527,781 B and commit `b55a5ec`, both equal to the public metadata.
- Strict load: 342 tensors, 303,350,784 fp32 parameters.

**Automated rejection tests.**

| Case | Error | Test |
|---|---|---|
| Missing file, directory, unbound name | `CHECKPOINT_MISSING` | `test_missing_checkpoint_is_rejected`, `test_unbound_encoder_fails_closed`, `test_missing_checkpoint_fails_closed_and_is_recorded` (run `FAILED`, provenance recorded, 0 patch reads) |
| Remote reference (`hf-hub:`, `https://`, `s3://`) | `CHECKPOINT_MISSING` | `test_remote_references_are_never_resolved` |
| One byte flipped, truncated | `CHECKPOINT_INTEGRITY` | `test_corrupt_or_truncated_checkpoint_is_rejected` |
| Wrong digest; nothing deserialized (`torch.load` never called) | `CHECKPOINT_INTEGRITY` | `test_checksum_mismatch_is_rejected_before_deserialization` |
| Non-UNI file bound as `uni` | `CHECKPOINT_INTEGRITY` | `test_uni_rejects_a_checkpoint_that_is_not_the_pinned_one` |
| Stage config pins a different digest than the adapter | `CHECKPOINT_INTEGRITY` | `test_expected_digest_comes_from_the_pin_or_config_never_neither` |
| File replaced between hash and load | `CHECKPOINT_INTEGRITY` | `test_a_checkpoint_replaced_after_hashing_is_rejected` |
| Garbage bytes, pickled object, nested dict | `CHECKPOINT_INCOMPATIBLE` | `test_unreadable_or_unsafe_checkpoints_are_rejected` |
| Missing, unexpected or wrong-dtype keys | `CHECKPOINT_INCOMPATIBLE` (keys listed) | `test_architecture_mismatch_is_rejected` |
| Output-dimension mismatch | `CHECKPOINT_INCOMPATIBLE` | `test_output_dimension_mismatch_is_rejected` |
| Fixture weights loaded as UNI (missing `blocks.0.attn…`) | `CHECKPOINT_INCOMPATIBLE` | `test_uni_architecture_mismatch_is_rejected_without_real_weights` |
| ViT-L keys with wrong shapes | `CHECKPOINT_INCOMPATIBLE` | `test_uni_rejects_wrong_shapes_in_a_vit_l_layout` |

## 7. CZI and SVS execution results

### Real UNI, `uni-official@1`, CPU, batch 16, `pathology-ai-embed --pipeline wsi-patch-embedding-uni@1.0.0`

| | F-CZI-01 | F-SVS-01 |
|---|---|---|
| Extraction | level 2 (ds 4), 0.44496 µm/px (11.0 % within the 15 % binding) | level 0, 0.499 µm/px |
| Patch / FOV / model input | 224 px / 99.7 µm / 224², no resampling | 224 px / 111.8 µm / 224², no resampling |
| Declared transforms | `channel_reorder(bgr→rgb)`, cast, scale, normalize(ImageNet), layout | `drop_alpha(rgba→rgb)`, cast, scale, normalize(ImageNet), layout |
| Rows × dim | **178 × 1024** float32, 12 chunks | **41 × 1024** float32, 3 chunks |
| Region reads | 178 (each patch once) | 41 |
| Content digest | `4c1afe8f96aebf75…` | `2c124f1493269c2b…` |
| Embedding identity | `pai-embid-d84cafd86eceebea3169d8dc6db0633f` | `pai-embid-b49093b73bf42fc6775c49d2bb63e5ba` |
| Store id | `pai-emb-a382ad4df7a568e35dd880d4a096c13d` | `pai-emb-93cd13a063d640ed743ec42920e1fc02` |
| Store size | 968 KiB | 236 KiB |

Tests:
- `test_real_uni_on_czi` and `test_real_uni_on_svs_is_reproducible_and_resumable`:
  - Rows equal the patch manifest IDs and order.
  - The **first stored chunk is bit-identical** to encoding the reader's patches directly with a fresh UNI encoder (same batch).
  - Rows are finite. The manifest carries the pin, profile, scale binding, licence and 20× limitation.
- `test_legacy_profile_is_refused_on_czi`: `PREPROCESSING_PROFILE_UNAVAILABLE`, and no store is created.

### Torch fixture (random weights), 256 px default pipeline

| Fixture | Rows | Dimension | Transforms | Notes |
|---|---|---|---|---|
| F-CZI-01 | 134 | 32 | `channel_reorder` first | — |
| F-SVS-01 | 25 | 32 | `drop_alpha` first | — |

Both are traceable, and the first chunk is bit-identical to direct encoding. For CZI, a repeated run gives an equal content digest. Tests: `test_czi_torch_fixture_embedding_bag`, `test_svs_torch_fixture_embedding_bag`.

## 8. Real UNI execution status

**Executed and validated technically** on both fixtures:

- **CPU, through three routes:** the pytest integration tests, the source CLI, and the installed wheel.
- **MPS:** CZI only (optional).
- **Resume:** an injected forward-pass failure on batch 1 (SVS) leaves the store `interrupted` with chunk 0. The resumed run reads only the missing patches and reaches the identical content digest.

**Not executed:**
- CUDA (no hardware).
- Reduced precision (not offered).

Scientific validity of the embeddings was not assessed, and is out of scope.

## 9. CPU/CUDA memory and timing

Host: Apple M4 Pro, 24 GiB, macOS 27.0.1 arm64, torch 2.14.1, timm 1.0.30, 8 torch threads.

| Run | Wall (CLI, incl. detection) | Encode time | Peak RSS | Host working set (budgeted) | Device peak |
|---|---|---|---|---|---|
| UNI CPU, CZI, 178 patches | 15.3–17.2 s | 9.1–9.7 s (about 53 ms/patch) | 1.71–1.75 GiB | 336.9 MiB (budget 512 MiB) | n/a |
| UNI CPU, SVS, 41 patches | 6.2–6.3 s | 2.1–2.5 s | 1.60–1.75 GiB | 337.5 MiB | n/a |
| UNI MPS, CZI, 178 patches | 12.9 s | 5.2 s | 1.10 GiB | 11.2 MiB (activations on device) | 2,093 MiB (`driver_allocated`) |
| UNI CUDA | NOT VERIFIED | — | — | — | — |

Components of the CPU numbers:
- Model parameter memory is 1,213,403,136 B, reported as `parameter_bytes`.
- Checkpoint load takes about 7.7 s, including SHA-256 of 1.2 GB.
- Activations were measured at 14.4 MiB per sample (231 MiB at batch 16). The budget bound is 20.4 MiB per sample (`vit_activation_bytes`), so the bound is conservative.

## 10. Reproducibility

**Reproducibility contract.**
- Results are **exact (bit-identical)** for identical weights, profile, scale binding, device type, precision, configured batch size, and torch/timm versions on the same platform.
- Across devices they agree within a **numerical tolerance**, and the run identity differs by construction.

| Comparison | Result |
|---|---|
| Two fresh CLI processes, UNI CPU (CZI, SVS) | Content, embedding identity, run identity and manifest digest identical; manifests byte-identical (`cmp`) |
| Installed wheel `[embedding,czi,openslide]` vs source CLI, UNI CPU | All digests equal on both fixtures |
| Two fresh `python -c` processes, torch fixture | Equal, and equal to the in-process digests (`test_fresh_processes_reproduce_cpu_torch_embeddings`) |
| In-process repeat, UNI SVS | Equal manifest digest |
| Batch 1 vs 8, UNI on CPU (probe) | Bit-identical on this CPU. Still bound into identity, because invariance is not guaranteed in general. |
| Batch 1 vs 4, torch fixture | Equal within atol 1e-6 (asserted as tolerance, not identity) |
| CPU vs MPS, UNI CZI, 178 rows | max \|Δ\| = 9.4e-5, min cosine 0.99999988; different run identity |
| 1 vs 8 torch threads, UNI (probe) | Bit-identical on this CPU (threads recorded in the metrics, not the identity) |

## 11. Packaging and regression

**Quality gates (2026-10-21):**

```bash
uv run ruff check .              # All checks passed
uv run ruff format --check .     # 151 files already formatted
uv run mypy                      # Success: no issues found in 116 source files (strict)
uv run pytest -q -W error -rs    # 461 passed, 48 skipped (no fixtures / no UNI configured)
PATHOLOGY_AI_WSI_FIXTURE_CZI=… PATHOLOGY_AI_WSI_FIXTURE_SVS=… PATHOLOGY_AI_UNI_CHECKPOINT=… \
  uv run pytest -q -W error -rs  # 506 passed, 3 skipped
uv build                         # sdist + wheel OK
```

The three skips with fixtures are:
- two pre-existing `UNSUPPORTED` reader-conformance cases;
- CUDA.

**Baseline before SPEC-006:**
- 386 passed and 42 skipped without fixtures.
- All 386 tests still pass.

**New tests: 81**

| Suite | Tests | Needs |
|---|---|---|
| `tests/unit/test_foundation_runtime.py` | 24 | no torch for most tests |
| `tests/unit/test_torch_encoders.py` | 26 | torch; the CUDA and MPS tests are device-gated |
| `tests/integration/test_foundation_pipeline.py` | 13 | torch |
| `tests/integration/test_real_wsi_foundation.py` | 5 | fixtures; UNI tests need the checkpoint |
| Architecture rules | 13 | planted torch/timm violations, generic-module naming (8 modules), lazy runtime import |

**SPEC-001…005 artifacts are unchanged.**
- Demo stdout SHA-256 is `11a211de…df3da` in the source checkout, the base wheel and the full wheel.
- Reference-encoder CLI on CZI: content `56441dbc…bfd1`, `pai-embid-d3b2649c…`, store `pai-emb-7543b229…`, manifest `5221186b…`.
- Reference-encoder CLI on SVS: content `e98540cb…f3f9`, `pai-embid-27bd8f5b…`, store `pai-emb-7c6c0dc2…`, manifest `b494540e…`.
- All of these are identical to report 005.

**Base wheel (numpy and pyyaml only).**
- `torch` and `timm` are not importable.
- The registry lists all three encoders.
- `uni` and `torch-fixture` raise `MODEL_RUNTIME_UNAVAILABLE … install 'pathology-ai[embedding]'`.
- No ML module is imported.
- The reference pipeline still runs (`test_base_runtime_works_without_torch_and_reports_the_missing_extra`, which blocks torch in a subprocess).

**Full wheel.** The real UNI CLI runs from the wheel, with the checkpoint bound through `$PATHOLOGY_AI_CHECKPOINTS`, and its digests equal the source checkout's.

**Changed existing files.** All changes are additive:
- `domain/errors.py`, `domain/embedding.py` (metrics field)
- `ports/__init__.py`
- `application/registries.py`
- `plugins/embedding/reference.py`, `plugins/embedding/operations.py`
- `bootstrap.py`, `cli.py`
- `pyproject.toml`
- `README.md`
- `tests/embedding_support.py`, `tests/architecture/test_dependency_rules.py`

**Repository independence.**
- `../digital-pathology` HEAD is `c85df8a` with 4 status lines, unchanged. `../hal-viewer` HEAD is `e248202` with 9 status lines, unchanged.
- The legacy checkpoint was read, not modified or copied.
- No legacy code was copied.
- No commits were created.

## 12. Acceptance criteria evidence matrix

"Weights" says whether the evidence used fixture weights or real UNI weights.

| AC | Verdict | Weights | Evidence |
|---|---|---|---|
| **AC-001** Model-independent architecture | **PASS** | n/a | AST rules: no `uni`/vendor/model names in domain, ports or application (existing rule), nor in 8 generic runtime and embedding modules (`test_generic_model_runtime_names_no_model`). `torch`/`timm` are scoped to `plugins/models/`. The runtime is imported lazily (`test_model_runtime_is_imported_lazily`). There is no change to the runner or the streaming loop. |
| **AC-002** Optional ML runtime | **PASS** | n/a | The base wheel has no torch or timm, and the demo hash is unchanged. `MODEL_RUNTIME_UNAVAILABLE` names the extra. `test_base_runtime_works_without_torch_and_reports_the_missing_extra`. |
| **AC-003** Model registry | **PASS** | n/a | `EncoderRegistry` with validated descriptors. The tests cover fail-closed resolution, duplicates and mislabelled descriptors (`test_registry_*`, `test_platform_registers_encoders_without_importing_torch`). |
| **AC-004** Local checkpoint control | **PASS** | fixture + real | Remote references are rejected. `pretrained=False`, `HF_HUB_OFFLINE=1`, and sockets are blocked during loading and inference (`test_loading_and_inference_use_no_network`). Real UNI was loaded only from the bound local file. |
| **AC-005** Checkpoint integrity | **PASS** | fixture + synthetic files | §6 (12 rejection paths). |
| **AC-006** UNI architecture | **PASS** | **real UNI** | The verified timm configuration is strictly loaded, with all 342 keys matched. `num_features == 1024`, `global_pool == "token"`, and the load-time probe returns (1, 1024) finite values. The first chunk is bit-identical to a direct forward pass on CZI and SVS. |
| **AC-007** Official preprocessing profile | **PASS** (scale: operator-bound) | **real UNI** | Geometry, colour, normalization and output come from the authors' documentation (§2). The unverified µm/px raises `PREPROCESSING_PROFILE_UNAVAILABLE` unless explicitly bound (`test_official_profile_requires_an_explicit_scale_binding`). The unverifiable hub `pretrained_cfg` is documented as NOT VERIFIED. Owner decision O1 is open. |
| **AC-008** Legacy compatibility separation | **PASS** | real UNI (refusal) | Distinct profile ID, input-spec fingerprint, channel order and scale. The profile is blocked with a reason, there is no default profile, and no fallback occurs (`test_legacy_profile_is_identified_but_blocked`, `test_no_profile_is_chosen_by_default`, `test_legacy_profile_is_refused_on_czi`). |
| **AC-009** Explicit resolution handling | **PASS** | real UNI | `ResolutionRecord` keeps extraction (level 2, 0.445 µm/px / level 0, 0.499 µm/px), FOV, model input (224²) and required spacing (0.5 ± 15 %) separate. Any resampling is refused for the official profile (`test_official_profile_refuses_any_declared_resampling`). No universal µm/px exists. |
| **AC-010** Batch inference | **PASS** | fixture + real | Batches are bounded by `BatchPolicy` (16 for UNI), with one row per patch (178 and 41). Shape, dtype and finiteness are validated by `validate_output`. Input batches are validated (`test_input_batches_are_validated`, 5 cases). |
| **AC-011** CPU support | **PASS** | fixture + **real UNI** | CPU inference with fixture weights (unit and integration tests) and with real UNI (§7). Eval mode, no gradients and `inference_mode` (`test_model_is_in_eval_mode_without_gradients`). |
| **AC-012** CUDA support | **NOT VERIFIED** | — | No CUDA on this host. `test_cuda_inference_matches_cpu_within_tolerance` is explicitly skipped ("AC-012 unverified"). A CUDA request fails with `DEVICE_UNAVAILABLE` before any patch read (`test_unavailable_device_fails_before_any_patch_read`). MPS was verified as an optional extra (§9). |
| **AC-013** Device and precision provenance | **PASS** | fixture + real | Device and precision are in the run identity. Torch and timm versions, batch size, profile and scale binding are in `EncoderIdentity`. Parameter bytes, threads, platform, timings and device peak are in `metrics.encoder_resources`. Unsupported precision raises `DEVICE_UNAVAILABLE`, and fp32 is the only precision offered. |
| **AC-014** Persistent embedding integration | **PASS** | fixture + real | Written through the unchanged SPEC-005 `NumpyEmbeddingStore`: chunks, SHA-256 and a finalized manifest. Rows are read back only from the store. |
| **AC-015** Resume integrity | **PASS** | fixture + **real UNI** | Injected inference errors and NaN outputs leave the store `interrupted` at {0}. The failed batch is never committed. Resume reads 8 patches (fixture) or one chunk (UNI SVS) and reaches the identical digest. Other weights open a different store and never touch the interrupted one (`test_resume_with_other_weights_does_not_touch_the_interrupted_store`). |
| **AC-016** CZI integration | **PASS** | fixture + **real UNI** | 134 × 32 (fixture). 178 × 1024 (UNI). §7. |
| **AC-017** SVS integration | **PASS** | fixture + **real UNI** | 25 × 32 (fixture). 41 × 1024 (UNI). §7. |
| **AC-018** Reproducibility | **PASS** | fixture + **real UNI** | Exact across fresh processes, the CLI and the installed wheel on CPU. Tolerance-scoped across devices and batch sizes. §10. |
| **AC-019** Regression preservation | **PASS** | n/a | All 386 pre-existing tests pass. SPEC-005 reference digests and the demo hash are byte-identical (§11). |
| **AC-020** Packaging and governance | **PASS** | n/a | Optional extra, base and full wheel checks, architecture rules, discovery, architecture and report documents under `.claude/`, sibling repositories untouched, no commits. SPEC-006 and ADR-006 remain *Proposed*. |

## 13. Remaining limitations and blockers

1. **CUDA unverified (AC-012).** It must be run on CUDA hardware: `uv run pytest tests/unit/test_torch_encoders.py -k cuda`, plus a UNI run with `--override embed.device=cuda`.
2. **Owner decision O1, physical scale.** The UNI pipeline declares 0.5 µm/px ± 15 % as its interpretation of "20×". This must be confirmed or replaced per experiment. The value is part of the embedding identity.
3. **Owner decision O2, checkpoint authorization.**
   - The checkpoint is the legacy project's authenticated download, used in place.
   - The owner should confirm that their UNI terms acceptance covers this use.
   - UNI-derived embeddings fall under CC-BY-NC-ND 4.0 derivative terms and must not be redistributed or used commercially.
4. **Official resize not implemented.**
   - The official profile works only where 224 px patches exist at a stored level whose spacing matches the binding.
   - A bilinear antialiased `resize` transform would need to be added to the declared transform vocabulary to support other geometries.
5. **Legacy profile blocked.**
   - uint8 `INTER_AREA` with a non-integer factor is not implemented, and legacy numerical parity remains deferred, as SPEC-006 allows.
   - The legacy BGR profile also has no RGBA→BGR conversion for SVS.
6. **Hub `pretrained_cfg` unverified.** The profile follows the authors' explicit transform instead. If the gated `config.json` differs, for example in `crop_pct`, a new profile version would be needed.
7. **Batch invariance not guaranteed.**
   - Torch identities bind batch size and runtime versions, so changing either produces a new store rather than mixing.
   - The measured invariance on this CPU is not generalized.
8. **Reduced precision.** fp16 and bf16 are not offered (SPEC-006 §9).
9. **Throughput.**
   - UNI CPU runs at about 53 ms per patch, sequential, on one reader thread (ADR-005 D10). The 10⁵-patch scale is extrapolated, not measured.
   - Checkpoint hashing costs about 4 s per run.
10. **Pin provenance.** The public SHA-256 of the UNI checkpoint is masked by Hugging Face. The pin rests on the local download metadata plus the matching public size and commit.
11. **Scientific claims.** No scientific or clinical validation was performed. The embeddings carry no claim of biological or diagnostic meaning.
