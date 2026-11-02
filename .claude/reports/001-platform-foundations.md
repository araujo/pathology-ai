# Report 001 — Platform Foundations

**Date:** 2026-10-15
**SPEC:** `../specs/001-platform-foundations.md` (Accepted 2026-10-15)
**ADR:** `../adr/001-platform-architecture.md` (Accepted 2026-10-15)
**Repository:** `pathology-ai`, git initialised on branch `main`. **No commits have been made** (none were requested), so there is no commit SHA. All files are untracked in the working tree.

## 1. Executive outcome

The minimal foundation is implemented and passes every local check:

- `ruff check` and `ruff format --check` are clean.
- `mypy --strict` reports no issues in 58 files (package and tests).
- **149 pytest tests pass, 0 fail, 0 skipped**, including when run with `-W error`, so no warnings were raised.

The same suite also passes in a copy of the repository that has **no sibling repositories** next to it.

**What runs:**

- One reusable WSI pipeline (tissue → patches → encoder → MIL-like aggregator) runs for two different projects (`patologia/cancer/pulmao` and `patologia/cancer/coloretal`) with different synthetic datasets and model artifacts.
- A structurally different direct image→segmentation pipeline runs for `patologia/pele/psoriase`.
- Every run records provenance and a reproducibility fingerprint.
- Capability resolution separates scoped scientific validation from technical compatibility.

**What is contract-only:**

- The CT/MRI volumetric pipeline validates its representation. Execution is explicitly reported as `EXECUTION_UNSUPPORTED`.
- The Imaging API v1 and Analysis API v1 exist as JSON Schemas, serializers and fixtures. There is no server.
- DPA v1 compatibility is an inventory only, with status `PENDING`. **No DPA compatibility is claimed.**

**All executable models are synthetic.** They are untrained, deterministic demonstrations with no medical meaning. They aren't evidence of parity with Digital Pathology.

## 2. Acceptance criteria

| AC | Status | Evidence |
|---|---|---|
| AC-001 Independent execution | **PASS** | The suite and the demo run in an isolated copy with no `digital-pathology` or `hal-viewer` beside it (149 passed, demo exit 0). Supporting tests: `tests/architecture/test_dependency_rules.py::test_no_legacy_imports_anywhere` and `::test_no_runtime_reference_to_legacy_repositories`. There are no GPU, download or dataset requirements. |
| AC-002 Dataset reuse | **PASS** | `tests/integration/test_synthetic_flows.py::test_two_projects_execute_the_same_pipeline_definition` shows the same pipeline ref and digest with different datasets and fingerprints. `tests/unit/test_config_adapter.py::test_cancer_projects_share_one_pipeline_definition` checks the configs. Inputs: `projects/patologia/cancer/{pulmao,coloretal}/project.yaml` and `pipelines/wsi-patch-embed-aggregate.v1.yaml`. |
| AC-003 Model extensibility | **PASS** | `test_synthetic_flows.py::test_new_model_plugin_registers_and_runs_without_core_changes` defines a new max-pool aggregator inside the test module, registers it and runs it. No domain or pipeline file changes. mypy checks it structurally against the `Model` protocol. |
| AC-004 Pipeline flexibility | **PASS** (WSI/MIL and 2D segmentation executed; volumetric representation only) | `::test_direct_segmentation_pipeline_runs_without_patches_or_mil`, `::test_volumetric_pipeline_validates_but_reports_unsupported_execution`, `::test_volumetric_pipeline_accepts_mri_too`, `::test_wsi_image_rejected_by_volumetric_pipeline`. Pipeline definitions: `pipelines/*.yaml`. |
| AC-005 Imaging abstraction | **PASS** | `pathology_ai/domain/geometry.py`. Tests: `tests/unit/test_ids_geometry.py` (level↔level-0 conversion, voxel→LPS mm, frames cannot be swapped, 2D/3D geometry cross-attachment rejected, missing spacing/affine not fabricated). Schema: `imaging/v1/image-descriptor.schema.json`, with WSI, CT and MRI-4D fixtures. |
| AC-006 Typed outputs | **PASS** | `pathology_ai/domain/prediction.py` (six output kinds). `tests/contract/test_api_contracts.py::test_all_typed_outputs_and_3d_evidence_serialize`. `tests/unit/test_validation_prediction_job.py::test_typed_outputs_are_not_forced_into_classification`. |
| AC-007 Validation awareness | **PASS** | `pathology_ai/domain/validation.py`. Tests: `::test_validated_requires_exact_scope_match`, `::test_validation_does_not_transfer_across_scope` (5 cases), `::test_validation_scope_bound_to_model_and_preprocessing_version`, `test_synthetic_flows.py::test_scoped_validation_record_yields_validated_only_for_its_scope`, `::test_unvalidated_use_requires_research_opt_in`. |
| AC-008 Reproducibility | **PASS** | `pathology_ai/domain/provenance.py` and `application/runner.py`. Tests: `::test_execution_records_full_provenance`, `::test_reruns_are_deterministic_across_platform_instances`, `::test_changing_preprocessing_changes_artifact_identities`, `::test_failed_execution_is_recorded`. |
| AC-009 DPA readiness | **PARTIAL** | The Imaging and Analysis v1 contracts are documented and tested independently of DPA (`tests/contract/test_api_contracts.py`, 29 tests). The DPA v1 compatibility adapter is **not implemented** and its inventory shows open gaps (`../architecture/dpa-v1-compatibility-inventory.md`). That matches the SPEC wording, "interfaces documented and testable", but it is not integration. |
| AC-010 Dependency isolation | **PASS** | `tests/architecture/test_dependency_rules.py`: layer rules across 6 layers, framework-free core, forbidden hardcoded terms, and 6 planted-violation checks. |
| AC-011 Negative tests | **PASS** | Incompatible stage contracts: `test_kind_mismatch_detected`, `test_cycle_detected`, `test_wrong_feature_dimension`, `test_model_stage_arity`. Missing preprocessing: `test_missing_preprocessing`, `test_incomplete_preprocessing_contract`, `test_missing_2d/volume_transform_fields_fail_closed`, `test_preprocessing_bound_to_model`. Unsupported inputs: `test_unsupported_model_input`, `test_unsupported_modality`, `test_wrong_image_dimensions`. Invalid spatial references: `test_evidence_geometry_out_of_bounds_rejected`, `test_frames_cannot_be_swapped`, `test_prediction_rejects_evidence_for_other_image`. Also covered: BGR/RGB (`test_bgr_reader_fails_closed_then_runs_with_explicit_transform`), unlabelled assets (`test_unlabelled_images_run_inference_but_block_supervised_use`), research opt-in, evidence semantics (`test_attention_cannot_be_presented_as_probability`), and 12 invalid API fixtures. |
| AC-012 No legacy changes | **PASS** | Before and after this work: `digital-pathology` is at `main@c85df8a` with 4 status entries, and `hal-viewer` is at `main@e248202` with 9 status entries. The entries are identical and existed before (untracked `.claude/` documents and DPA sqlite data). I only read legacy files and never ran write commands against them. |

## 3. Architecture and module boundaries

The full description is in `../architecture/001-platform-boundaries.md`. In summary:

```text
pathology_ai/
  domain/          stdlib only: ids, errors, modality, geometry, image, kinds, task, preprocessing,
                   model, dataset, validation, pipeline, pipeline_validation, prediction,
                   provenance, project, job, hashing
  ports/           typing.Protocol interfaces (domain only)
  application/     registries, runner, project_service, analysis_service, model_operations
  infrastructure/  YAML config adapter, clocks, environment probe, job executors
  plugins/synthetic/  numpy: reader, tissue/patch ops, encoder, aggregator, segmenter, catalog
  presentation/api/   imaging_v1, analysis_v1 serializers + JSON Schema contracts
  bootstrap.py, cli.py   composition root
pipelines/   3 versioned YAML pipeline documents (adapter input, not the engine)
projects/    patologia/cancer/{pulmao,coloretal}, patologia/pele/psoriase
```

| Component | Implemented | Contract-only / declared |
|---|---|---|
| Domain contracts and validation | ✓ | |
| In-memory registries and stores | ✓ (in-process only) | Persistent stores |
| Runner, project runner, async job service | ✓ (inline or deferred executor) | Distributed or worker executors |
| Synthetic WSI reader and plugins | ✓ synthetic only | Real WSI readers |
| Volumetric pipeline | Validation only | Series reader, reconstruction, normalisation, 3D model |
| `PreprocessingStrategy`, `PatchStrategy`, `VolumeProvider`, `ImageMetadataProvider` ports | | Declared, no adapters |
| Imaging and Analysis API v1 | Schemas, serializers, fixtures | HTTP server |
| DPA v1 adapter | | Inventory only |

The deviation from the SPEC §5 package list is explained in the boundaries document, §2.

## 4. Examples

**One WSI pipeline reused by different projects.** The run is `uv run pathology-ai-demo`, using the first `test` image of each project.

| Namespace | Pipeline | Bound models | Dataset | Fingerprint |
|---|---|---|---|---|
| `patologia/cancer/pulmao` | `wsi-patch-embed-aggregate@1.0.0` | `synthetic-patch-encoder`, `synthetic-aggregator-lung` | `synthetic-lung-slides@0.1.0` | `pai-fp-05e0480b394e11fb874c3099a94f6cc5` |
| `patologia/cancer/coloretal` | `wsi-patch-embed-aggregate@1.0.0` | `synthetic-patch-encoder`, `synthetic-aggregator-colorectal` | `synthetic-colorectal-slides@0.1.0` | `pai-fp-7c764da5c48906780bc236e8936742d4` |
| `patologia/pele/psoriase` | `wsi-direct-segmentation@1.0.0` (non-MIL) | `synthetic-direct-segmenter-skin` | `synthetic-skin-slides@0.1.0` | `pai-fp-f565b8ddc13220b378dee739d14baeac` |

Fingerprints were identical across repeated runs on this machine. They include environment information (Python version, platform, package versions), so they differ across environments by design.

The two cancer projects share the pipeline but bind **different aggregator artifacts**. Sharing a pipeline doesn't mean a model transfers between diseases. Every result is `SUPPORTED_UNVALIDATED` / `research_unvalidated`, because there are no validation records.

**Volumetric pipeline** (`volume-segmentation-3d@1.0.0`, synthetic CT/MRI metadata-only volume):

- Validation passes when the budget includes a GPU.
- `run` raises `EXECUTION_UNSUPPORTED`, missing `model_plugin:synthetic-volume-segmenter@1.0.0`, `operation:volume.reconstruct_series@1`, `operation:volume.spatial_normalize@1` and `reader:synthetic-volume/volume_3d`.
- With the default CPU budget, validation reports `RESOURCE_UNAVAILABLE`.

**Programmatic construction.** `tests/conftest.py::build_wsi_pipeline` builds the WSI pipeline in Python. `test_yaml_is_only_an_adapter_for_programmatic_pipelines` asserts that the YAML-loaded definition equals it.

## 5. API contracts, geometry and DPA gaps

Schemas live in `pathology_ai/presentation/api/contracts/`. Operation lists are in each `operations.json`.

- **Imaging v1** (`pathology-ai.imaging/1.0`): `image-descriptor`, `region-request`, `thumbnail-request` and `slice-request`.
  - The descriptor covers canonical `image_id`, modality code plus spatial kind, format, channel order, checksum, and nullable tissue/disease context.
  - Geometry is either a 2D pyramid (`wsi_level0_pixel`, levels, nullable µm spacing) or a volume (`volume_voxel_index` plus `patient_lps_mm`, shape, nullable spacing, origin and direction, time points, series UID).
  - Access flags say which operations a deployment can serve.
- **Analysis v1** (`pathology-ai.analysis/1.0`): `submit-analysis-request` (project_id + image_id, no ML details), `analysis-job` (five states, with conditional rules for prediction and error), `capability-list` (three states, with a research-acknowledgement flag), and `analysis-result`.
  - Results carry typed outputs as a discriminated union, and evidence with an explicit frame and semantics.
  - Schema rules ensure attention and saliency are never probabilities, and that a localized probability has a validation record.
  - Capability status and usage mode must be consistent, and INCOMPATIBLE results are forbidden.
  - The full provenance block is included.
- **Tests:** 8 valid and 12 invalid fixtures, plus serialized output from real runs, all validated with `jsonschema` Draft 2020-12.

**DPA v1:** see `../architecture/dpa-v1-compatibility-inventory.md`. I verified the fields against DP's `schemas.py`, `job.py` and `routes.py`, and against DPA's DP client and result adapter. The main gaps are:

- `canonical_slide_id` + `source` locator vs `image_id`.
- DPA requires an integer `task_version`.
- `tumor_type` is required but is unavailable without fabricating it.
- DPA requires `patch_index`, `score` and `rank` on every evidence item.
- v1 has no field for validation or research-only status, or for evidence semantics.
- v1 can't represent segmentation, detection, volumetric outputs or provenance.
- Pathology AI evidence IDs aren't DP-computed.

## 6. Scientific caveats and unsupported modalities

- All models, datasets, labels and images are **synthetic** (`synthetic-*` / `syn-*` IDs). Pixels are procedurally generated. No patient data was processed and no models were downloaded.
- Classification scores are marked `scores_are_calibrated_probabilities: false`. Attention is emitted only as `attention_weight` evidence.
- No `ValidationRecord` ships with the platform. The `VALIDATED` path is exercised only in tests, with records flagged `synthetic=True`.
- **Unsupported for execution:** CT, MRI, DICOM, NIfTI, NRRD and MetaImage readers; real WSI vendor formats; stain normalisation other than `none`; resampling when no pyramid level matches the contract spacing (fails closed); training and evaluation.
- Passing these tests is **not** scientific or clinical validation, and not parity with Digital Pathology.

## 7. Validation commands and results

Environment: macOS arm64, Python 3.12.13, uv 0.11.17. Packages: numpy 2.5.4, pyyaml 6.0.3, pytest 9.1.1, ruff 0.17.1, mypy 2.4.0, jsonschema 4.26.0.

| Command | Result |
|---|---|
| `uv sync` | Resolved 23 packages |
| `uv run ruff check .` | All checks passed |
| `uv run ruff format --check .` | 62 files already formatted |
| `uv run mypy` (strict; `pathology_ai`, `tests`) | Success: no issues found in 58 source files |
| `uv run pytest -q -W error` | **149 passed**, 0 failed, 0 skipped, 0 warnings |
| By suite | unit 81 · contract 29 · architecture 19 · integration 20 |
| `uv run pathology-ai-demo` | exit 0 (3 projects executed, volumetric `EXECUTION_UNSUPPORTED`) |
| Isolated copy (no sibling repos): `uv sync && uv run pytest -q && uv run pathology-ai-demo` | 149 passed, demo exit 0 |
| `git -C ../digital-pathology rev-parse --short HEAD; git status --porcelain \| wc -l` | `c85df8a`, 4 (unchanged) |
| `git -C ../hal-viewer rev-parse --short HEAD; git status --porcelain \| wc -l` | `e248202`, 9 (unchanged) |

**Remaining risks:**

1. Stores are in-memory only, so runs are lost when the process ends.
2. `ResourceBudget` defaults to 0 GPUs and 4096 MB. It is a declared check, not real resource measurement.
3. Environment details go into the fingerprint, so identical runs on different hosts get different fingerprints. This is intended, but a "same code and data" key may also be wanted later.
4. The capability check doesn't apply scoped validation to feature extractors. They are checked only for technical compatibility.
5. The DPA inventory reflects the source as read on 2026-10-15 and will drift.

## 8. Recommendations for SPEC-002 and later (not implemented here)

1. **SPEC-002 real 2D readers:** one WSI reader adapter (format chosen by SPEC), implementing `ImageTileProvider` and `ImageMetadataProvider`, with physical-spacing and no-data-region tests on de-identified or public slides.
2. **Persistence:** file-backed or database-backed `ArtifactStore`, `ProvenanceStore` and dataset registry, with content-addressed artifacts.
3. **Preprocessing strategies:** implement `PreprocessingStrategy` and `PatchStrategy` with resampling and interpolation, and test them against reference fixtures.
4. **DPA v1 adapter SPEC:** resolve the decisions in inventory §5, then add contract tests that replay through DPA's parsing rules and shadow-compare against DP. Changing DPA or hal-viewer would need separate approval.
5. **Scientific parity SPEC:** DP as oracle, with model and preprocessing parity tests and a validation-record workflow.
6. **Volumetric SPEC:** a series/volume reader, orientation and spacing normalisation, and a VolumeProvider adapter.
7. **Training/evaluation SPEC:** task-specific losses, metrics and protocols, kept out of the shared core.

## 9. Files

**Modified:**

- `.claude/specs/001-platform-foundations.md`: status set to Accepted.
- `.claude/adr/001-platform-architecture.md`: status set to Accepted.

**Created:**

- Repository setup: `pyproject.toml`, `uv.lock`, `.gitignore`, `.python-version`, `README.md`.
- Package: `pathology_ai/**` (43 Python modules plus 10 JSON contract and operation files).
- Definitions: `pipelines/*.yaml` (3), `projects/**/project.yaml` (3).
- Tests: `tests/**` (9 test modules, `conftest.py`, 20 JSON fixtures).
- Governance: `.claude/architecture/001-platform-boundaries.md`, `.claude/architecture/dpa-v1-compatibility-inventory.md`, and this report.

**Deviations from the SPEC:**

- The package layout puts the §5 sub-areas in `domain` modules (see the boundaries document, §2).
- `.claude/skills/` was not created, as instructed.
- The volumetric pipeline isn't attached to a project; the SPEC requires only the three listed projects.
