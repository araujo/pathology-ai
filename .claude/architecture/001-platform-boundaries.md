# 001 — Platform Boundaries

**Status:** Implemented for SPEC 001 (2026-10-15); physical layout updated for SPEC 002 (2026-10-16); real-image readers added for SPEC 003 (2026-10-16)
**Related:** `../specs/001-platform-foundations.md`, `../adr/001-platform-architecture.md`, `../specs/002-repository-structure-and-packaging.md`, `../adr/002-root-level-package-layout.md`, `dpa-v1-compatibility-inventory.md`

Since SPEC 002 the layers are root-level packages (`domain`, `ports`, `application`, `infrastructure`, `plugins`, `presentation`) next to `bootstrap.py` and `cli.py`. There is no `pathology_ai` package. The distribution is still named `pathology-ai`. The logical rules below are unchanged.

## 1. Layers and allowed dependencies

```text
                 ┌──────────────────────────────┐
                 │ bootstrap.py / cli.py        │  composition root (may import every layer)
                 └──────────────┬───────────────┘
       ┌───────────────┬────────┴────────┬──────────────────┐
       ▼               ▼                 ▼                  ▼
 presentation    infrastructure       plugins/*        (tests)
 (api v1)        (yaml, clock, env,   (synthetic:
       │          executors)           numpy)
       │               │                 │
       └──────┬────────┘                 │
              ▼                          │
         application ────────────────────┤
              │                          │
              ▼                          ▼
            ports ──────────────────► domain   (stdlib only)
```

| Layer | May import (internal) | Third-party allowed |
|---|---|---|
| `domain` | — | none |
| `ports` | `domain` | none |
| `application` | `domain`, `ports` (the release version lives in `application/version.py`) | none |
| `infrastructure` | `domain`, `ports`, `application` | `yaml` |
| `plugins` | `domain`, `ports` | `numpy`; `pylibCZIrw` only in `plugins/imaging/czi/`, `openslide` only in `plugins/imaging/openslide/` (SPEC 003; lazy `importlib.import_module` calls count) |
| `presentation` | `domain`, `ports`, `application` | none |
| `bootstrap.py`, `cli.py` | everything | — |

These rules are checked by `tests/architecture/test_dependency_rules.py`, which reads imports from the AST. The same file also checks the following:

- No layer imports a module that the table above doesn't allow.
- No layer imports the composition root (`bootstrap`, `cli`) or the removed `pathology_ai` namespace, and the `pathology_ai/` directory does not exist.
- The wheel's explicit `only-include` list matches exactly the six layer packages plus `bootstrap.py` and `cli.py`, so `tests`, `pipelines`, `projects` and `.claude` are never installed.
- First-party top-level names don't shadow standard-library modules.
- `domain`, `ports` and `application` import no ML or imaging framework (numpy, torch, pydantic, fastapi, yaml, PIL, cv2, pandas).
- No module imports legacy code (`digital_pathology`, `hal_viewer`, `scripts`, `pipeline`, `analysis_api`).
- No package module refers to the `digital-pathology` or `hal-viewer` paths.
- The core layers don't hardcode any of these terms: CZI, KRAS, UNI, colorectal/coloretal, H&E/hematoxylin, ABMIL, SurGen, Zeiss, Aperio, OpenSlide, pylibCZIrw.
- Each optional reader backend is imported, eagerly or lazily, only by its own adapter subpackage.
- There is no `docs/` directory.
- Twelve planted bad imports are each detected, which shows the rule is not passing vacuously.

## 2. How the package differs from SPEC §5

SPEC §5 proposed separate packages: `imaging/`, `preprocessing/`, `representation/`, `learning/`, `inference/`, `orchestration/` and `provenance/`. In 001 these concepts are pure contracts, so they live as modules in `domain/`:

- `geometry`, `modality` and `image` cover imaging.
- `preprocessing` covers preprocessing.
- `pipeline` and `pipeline_validation` cover orchestration contracts.
- `provenance` covers provenance.

Execution lives in `application/runner.py`. The concrete readers, operations and models live in `plugins/`. SPEC §5 allows this refinement as long as the dependency rules hold.

Training (`learning/`) isn't implemented. The only supervised-use contract is `require_labels`, exposed through `ProjectRunner.supervised_examples`.

## 3. Key domain contracts

| Concept | Module | Notes |
|---|---|---|
| Typed IDs | `domain/ids.py` | One class per kind. Lowercase, no paths, at most 128 characters. |
| Modality vs format | `domain/modality.py` | `ImageModality(code, spatial_kinds)` and an open `ImageFormat`. `require_declared_modality` never infers a modality from a format. |
| 2D pyramidal geometry | `domain/geometry.py` | Level-0 canonical frame, level↔level-0 conversion, optional µm spacing that is never fabricated. |
| 3D/4D volume geometry | `domain/geometry.py` | Voxel index frame and a patient LPS mm frame via spacing, origin and direction. The affine transform needs complete metadata. |
| Image asset | `domain/image.py` | Has no label field. Tissue/disease context is optional (null means unknown). Modality must allow the geometry's spatial kind. |
| Preprocessing | `domain/preprocessing.py` | Versioned and fingerprinted. Required fields are listed per spatial kind. BGR/RGB is only reordered by an explicitly declared transform. |
| Pipelines | `domain/pipeline.py` | A DAG of `StageDefinition`s with typed `PortSpec`s and model slots. The schema version is `pathology-ai.pipeline/1`. |
| Validation | `domain/pipeline_validation.py` | Structural checks and binding checks. Issues are returned in a sorted, deterministic order. |
| Scientific validation | `domain/validation.py` | `ValidationScope` covers dataset, modality, tissue, disease, task, model version, preprocessing version and evaluation protocol. Capability states are `VALIDATED`, `SUPPORTED_UNVALIDATED` and `INCOMPATIBLE`. Research use must be opted into explicitly. |
| Outputs and evidence | `domain/prediction.py` | Six output types in a discriminated union. Evidence carries its frame and a semantics value. Attention and saliency can't be marked as probabilities. A localized probability needs a validation record. |
| Provenance | `domain/provenance.py` | `ExecutionFingerprint` is separate from `PipelineRun.run_id`. `ArtifactIdentity` is content-addressed. IDs use the `pai-…` prefix. |
| Jobs | `domain/job.py` | States are PENDING, RUNNING, COMPLETED, FAILED and CANCELLED. Transitions are guarded. |

## 4. Order of execution checks (`application/runner.py`)

Every check fails closed.

1. **Binding validation.** Structure, kinds, feature dimensions, modality, image dimensions, model input and output, preprocessing (missing, mismatched, incomplete), channel order, pixel spacing, upstream encoder, and resources.
2. **Capability.** Each predictive model is resolved against the scoped `ValidationRecord`s, then `authorize_use` runs.
3. **Executability.** Any missing operations, model plugins or reader raise `ExecutionUnsupportedError` with the sorted list of what's missing.
4. **Execution.** Stages run in a deterministic topological order. Every edge value gets a content-addressed identity derived from the stage, operation, configuration, model checksum, preprocessing fingerprint, seed and lineage.
5. **Provenance.** A `PipelineRun` is always recorded, including for failed executions.

## 5. Implemented vs contract-only

| Component | Status |
|---|---|
| Domain contracts, validation, capability resolution | Implemented and tested |
| In-process registries, artifact store and provenance store | Implemented (in-memory only) |
| Pipeline runner, project runner, analysis job service | Implemented (in-process, synchronous or deferred executor) |
| YAML adapters for pipelines and projects | Implemented |
| Synthetic slide reader, tissue and patch operations, encoder, MIL-like aggregator, direct segmenter | Implemented. Synthetic only, with no medical meaning. |
| `wsi-patch-embed-aggregate` and `wsi-direct-segmentation` pipelines | Executable with synthetic plugins |
| `volume-segmentation-3d` pipeline | Contract only. It validates, but execution reports `EXECUTION_UNSUPPORTED`. |
| Real WSI readers (SPEC 003) | `plugins/imaging/czi` (pylibczirw, `pathology-ai[czi]`) and `plugins/imaging/openslide` (Aperio SVS only, `pathology-ai[openslide]`). Selected by content through `application/imaging_service.ImageBackendRegistry`. Tested formats and limits: `003-reader-capability-matrix.md`. |
| Real-image ingestion smoke workflow | `ImagingService.ingest` and `pathology-ai-imaging-smoke`: metadata, bounded regions, `image.region_2d` artifacts and an `IngestionRecord`. No model is run. Not wired into `PipelineRunner`, whose synthetic operations still require the synthetic reader. |
| CT/MRI/DICOM/NIfTI readers, other WSI formats | Not implemented |
| `PreprocessingStrategy`, `PatchStrategy`, `VolumeProvider`, `ImageMetadataProvider` ports | Declared only. No adapter implements them yet; synthetic operations use the reader directly. |
| Imaging API v1 and Analysis API v1 | JSON Schemas, serializers and fixtures only. There is no HTTP server. |
| DPA v1 compatibility adapter | Not implemented. See the inventory, where status is `PENDING`. |
| Training, evaluation, persistence, deployment | Not implemented |

## 6. Packaging and resource loading (SPEC 002)

| Resource | Where it lives | How it is found |
|---|---|---|
| Framework code | Root-level packages plus `bootstrap.py` and `cli.py`, listed explicitly in `[tool.hatch.build.targets.wheel] only-include` | Normal imports (`from domain.pipeline import ...`). No `sys.path` manipulation. |
| Imaging and Analysis API v1 JSON Schemas | `presentation/api/contracts/{imaging,analysis}/v1/` (inside the wheel) | `importlib.resources`, independent of the working directory |
| Pipeline and project YAML | `pipelines/`, `projects/` at the repository root (**not** in the wheel) | `bootstrap.resolve_definitions_root`: an explicit `root` / `--root PATH`, then `$PATHOLOGY_AI_ROOT`, then the source checkout that contains `bootstrap.py`. The current working directory is never used. A configured location without both directories is an error. |
| Release version | `application/version.py` | Read by hatch for the distribution version and imported by the runner. It feeds the `package` component of the execution fingerprint. |

Supported invocations:

```bash
uv run pathology-ai-demo                          # source checkout
pathology-ai-demo --root /path/to/definitions     # installed wheel
PATHOLOGY_AI_ROOT=/path/to/definitions pathology-ai-demo
```

An installed CLI with no root exits with status 2 and an error naming both options.

**Name-collision risk (ADR-002):** `domain`, `application`, `plugins`, `cli` and `bootstrap` are generic top-level names. Another installed distribution that ships one of them would conflict with this one. Check this when adding dependencies.

