# 001 — Pathology AI Platform Foundations

**Status:** Accepted (approved 2026-10-15)  
**Project:** Pathology AI  
**Repository:** `pathology-ai`  
**Type:** Architecture and Foundation Implementation  
**Priority:** P0  
**References:** Digital Pathology Architecture Assessment (SPEC-025), existing Digital Pathology pipeline, DPA integration contracts

## 1. Context

Pathology AI is a new, independent, extensible AI platform for medical imaging.

The existing `digital-pathology` repository contains a scientifically reproducible research pipeline centered on Whole Slide Images (WSI), patch extraction, feature embeddings, Multiple Instance Learning (MIL), and slide-level predictions.

Although this workflow is useful as a reference, Pathology AI must not inherit assumptions that restrict the platform to:

- Colorectal cancer or KRAS prediction.
- One tissue type or disease.
- CZI images.
- A particular encoder, such as UNI.
- Attention-based MIL.
- Binary classification.
- Rectangular spatial evidence.
- A single imaging modality.

The platform must support reusable pipelines across diseases, tissues, datasets, tasks, and model architectures.

It must also be designed for future integration with the Digital Pathology Analyzer (DPA), without requiring changes to DPA during this implementation.

## 2. Objectives

1. Establish an independent Python repository and modular-monolith architecture.
2. Define stable domain contracts for datasets, projects, tasks, models, pipelines, images, predictions, and evidence.
3. Support reusable pipeline definitions without duplicating code for each disease or dataset.
4. Allow new model architectures through plugins and typed interfaces.
5. Support multiple imaging modalities through extensible imaging contracts.
6. Separate imaging access, preprocessing, training, inference, and presentation.
7. Prepare versioned Imaging and Analysis API contracts for future DPA integration.
8. Establish scientific reproducibility, artifact identity, provenance, and validation requirements.
9. Create repository governance, automated tests, and a minimal executable architectural demonstration.

## 3. Non-Goals

This specification does not require:

- Implementing all supported medical imaging formats.
- Implementing production CT or MRI pipelines.
- Training a clinically validated model.
- Migrating the existing Digital Pathology codebase.
- Replacing the existing DPA backend.
- Modifying `hal-viewer`.
- Implementing a production WSI viewer.
- Implementing clinical decision support.
- Claiming scientific validation for models that have not been independently evaluated.

## 4. Architectural Principles

### 4.1 Dataset Independence

A compatible pipeline definition must be reusable with different datasets and project configurations.

Example:

```text
patologia/cancer/pulmao
patologia/cancer/coloretal
patologia/pele/psoriase
```

Each project can select its own dataset, labels, model artifacts, validation protocol, and preprocessing configuration.

Changing the dataset must not require copying or editing the pipeline implementation.

Changing the dataset alone does not imply that the same trained model is scientifically valid for the new disease.

### 4.2 Model Agnosticism

The platform must not impose MIL, patch extraction, or embeddings as mandatory processing steps.

Supported architectural patterns must include:

- Patch encoder followed by MIL.
- Direct image classification.
- Patch-level classification.
- Image segmentation.
- Object detection.
- Volumetric segmentation.
- Volumetric classification.
- Future multimodal models.

Model plugins must declare their input requirements, output types, dependencies, preprocessing contracts, and compatibility constraints.

### 4.3 Modality Extensibility

The platform must distinguish image formats from imaging modalities.

Examples:

- WSI: CZI, SVS, NDPI, TIFF, DICOM WSI.
- CT: DICOM series, NIfTI, NRRD, MetaImage.
- MRI: DICOM series, NIfTI, NRRD, MetaImage.

A file extension must not be treated as sufficient evidence of imaging modality.

The architecture must support both 2D pyramidal images and 3D/4D image volumes.

### 4.4 Scientific Integrity

Preprocessing must be explicitly bound to the model artifact or validated model configuration.

The platform must not silently change:

- Color channel ordering.
- Stain normalization.
- Resolution or physical spacing.
- Pixel interpolation.
- Intensity normalization.
- Image orientation.
- Patch size and sampling strategy.
- Coordinate transforms.

A model must fail closed when a required preprocessing or input compatibility contract is unavailable.

### 4.5 Independent Repository

Pathology AI must have no runtime, source-code import, or shared mutable artifact dependency on `digital-pathology`.

The existing Digital Pathology implementation remains a scientific reference and potential parity oracle.

Patterns may be reused conceptually, but implementations must be independently owned and tested.

## 5. Proposed Architecture

Use a modular monolith with dependency inversion.

```text
pathology-ai/
├── pathology_ai/
│   ├── domain/
│   ├── ports/
│   ├── application/
│   ├── imaging/
│   │   ├── wsi/
│   │   └── volumetric/
│   ├── preprocessing/
│   ├── representation/
│   ├── learning/
│   ├── inference/
│   ├── orchestration/
│   ├── provenance/
│   ├── infrastructure/
│   └── presentation/
│       └── api/
├── pipelines/
├── projects/
├── tests/
├── pyproject.toml
└── .claude/
    ├── adr/
    ├── architecture/
    ├── specs/
    ├── prompts/
    ├── reports/
    └── skills/
```

The exact Python package boundaries may be refined during implementation, provided the architectural dependency rules remain intact.

### 5.1 Domain Layer

The domain layer defines concepts and invariants without depending on frameworks or infrastructure.

Required domain concepts:

- `ProjectDefinition`
- `DatasetDefinition`
- `DatasetVersion`
- `ImageAsset`
- `ImageModality`
- `SpatialReference`
- `TaskDefinition`
- `OutputSchema`
- `PipelineDefinition`
- `PipelineRun`
- `ModelDefinition`
- `ModelArtifact`
- `PreprocessingContract`
- `PredictionResult`
- `SpatialEvidence`
- `ValidationRecord`
- `ArtifactIdentity`

Tissue and disease context must be optional at image ingestion time.

An unknown-tissue image must be ingestible without assigning an unsupported disease label.

### 5.2 Ports and Adapters

The domain must own interfaces for:

- `ImageReader`
- `ImageMetadataProvider`
- `ImageTileProvider`
- `VolumeProvider`
- `PreprocessingStrategy`
- `PatchStrategy`
- `Encoder`
- `Model`
- `DatasetRepository`
- `ModelRegistry`
- `PipelineRegistry`
- `ArtifactStore`
- `JobExecutor`
- `GroundTruthSource`

Infrastructure adapters implement these interfaces.

The domain must never import a concrete reader, encoder, model, storage implementation, or API framework.

### 5.3 Imaging Subsystem

Define a modality-aware imaging abstraction.

For WSI, the contract must represent:

- Native dimensions.
- Pyramidal levels.
- Physical pixel spacing when available.
- Channels and image metadata.
- Tile or region retrieval.
- Coordinate conversion to canonical level-zero coordinates.
- Reader capabilities.
- Scanner no-data regions when supported.

For CT/MRI, the contract must represent:

- Volumetric dimensions.
- Voxel spacing.
- Orientation and origin.
- Coordinate transforms.
- Series identity.
- Channel or acquisition information.
- Volume and slice retrieval.

Missing metadata must remain explicitly missing rather than being fabricated.

### 5.4 Pipeline Composition

A pipeline is a versioned directed graph of typed processing stages.

A stage must declare:

- Input contract.
- Output contract.
- Required capabilities.
- Configuration schema.
- Artifact dependencies.
- Version.
- Determinism and reproducibility information.

Example WSI pipeline:

```text
Image
  → Tissue Detection
  → Patch Extraction
  → Encoder
  → Embedding Bag
  → MIL Model
  → Slide Classification
```

Example CT pipeline:

```text
DICOM Series
  → Volume Reconstruction
  → Spatial Normalization
  → 3D Segmentation Model
  → Segmentation Mask
```

These pipelines share orchestration and provenance services but not necessarily preprocessing stages or model implementations.

### 5.5 Dataset and Project Registry

Projects are configurations and optional extensions, not forks of the platform.

Example:

```text
projects/
└── patologia/
    ├── cancer/
    │   ├── pulmao/
    │   └── coloretal/
    └── pele/
        └── psoriase/
```

A project configuration references:

- Dataset version.
- Task definition.
- Pipeline version.
- Model definition.
- Training configuration.
- Evaluation protocol.
- Intended use.
- Validation scope.

Dataset and project identities must be stable and independent of display names or filesystem paths.

### 5.6 Model and Task Registry

The model registry must describe:

- Model family and architecture.
- Expected input kind.
- Input shape or embedding dimension.
- Output schema.
- Required preprocessing contract.
- Compatible modalities.
- Validated tissue/disease/task scope.
- Model weights and version.
- Software and environment dependencies.
- Scientific validation status.

Capability discovery must distinguish:

- `VALIDATED`
- `SUPPORTED_UNVALIDATED`
- `INCOMPATIBLE`

A model must not be presented as clinically valid merely because it can execute.

### 5.7 Training and Inference

Training and inference must be independent application services that use registered pipelines and models.

They must support different task types, including:

- Binary classification.
- Multiclass classification.
- Multilabel classification.
- Regression.
- Segmentation.
- Detection.
- Quantification.

Task-specific loss functions, metrics, postprocessing, and evaluation protocols must not be hardcoded into the shared core.

### 5.8 Prediction and Evidence

Prediction results must be typed.

The initial domain model must distinguish:

- Slide/image-level classification.
- Continuous scores.
- Region-level predictions.
- Detection geometries.
- Segmentation references.
- Volumetric evidence.

Spatial evidence must include its coordinate reference system and provenance.

MIL attention scores must not be represented as localized disease probabilities unless separately validated for that interpretation.

## 6. Future DPA Integration

Pathology AI must be designed to expose two independent integration capabilities.

### 6.1 Imaging API

The future Imaging API must support:

- Image discovery.
- Image identity.
- Metadata and modality.
- WSI pyramids and tiles.
- Thumbnails.
- Physical image dimensions.
- Volumetric image metadata.
- Volume or slice references.
- Imaging capabilities.

### 6.2 Analysis API

The future Analysis API must support:

- Task and model capability discovery.
- Asynchronous analysis submission.
- Job status and errors.
- Prediction retrieval.
- Typed evidence.
- Artifact references.
- Scientific validation status.
- Model and preprocessing provenance.

### 6.3 Compatibility Boundary

The existing DPA consumes versioned Digital Pathology analysis contracts.

Pathology AI must prepare an explicit compatibility adapter for the existing DPA-facing API.

Future native contracts may support additional imaging modalities and output types.

The architecture must not assume that current DPA visualization capabilities already support CT, MRI, volumetric overlays, or arbitrary prediction schemas.

**No changes to `hal-viewer` are authorized by this specification.**

## 7. Provenance and Reproducibility

Every executed pipeline must be traceable to:

- Dataset version and split.
- Image/source identity.
- Task definition.
- Pipeline graph and stage versions.
- Preprocessing contract.
- Model architecture and checkpoint.
- Software and dependency versions.
- Execution configuration and random seeds.
- Produced artifacts and their lineage.

Artifact identities must distinguish incompatible preprocessing and model configurations.

Scientific validation records must be separate from successful execution records.

## 8. Foundation Implementation Scope

The first implementation must deliver:

1. Python project and dependency management.
2. Package structure and architecture dependency rules.
3. Typed domain contracts.
4. Port interfaces.
5. In-memory registries.
6. A minimal pipeline graph and stage composition mechanism.
7. A synthetic or deterministic demonstration pipeline.
8. Example project configurations for multiple diseases.
9. Imaging metadata abstractions for 2D and volumetric images.
10. Typed prediction and evidence schemas.
11. Future-facing API contract definitions without requiring production API deployment.
12. Automated unit, contract, and architecture tests.
13. Governance artifacts and implementation report.

The implementation must not require GPU hardware, model downloads, external datasets, or access to the existing Digital Pathology repository to execute its foundation tests.

## 9. Acceptance Criteria

### AC-001 — Independent Execution

The foundation installs, imports, and runs without access to `digital-pathology` or `hal-viewer`.

### AC-002 — Dataset Reuse

Two distinct project configurations can execute the same pipeline definition using different datasets without modifying pipeline code.

### AC-003 — Model Extensibility

A new synthetic model plugin can be registered and executed without modifying the domain layer or existing pipeline implementations.

### AC-004 — Pipeline Flexibility

The framework can represent both a WSI embedding/MIL workflow and a volumetric segmentation workflow, without requiring either to execute real medical inference.

### AC-005 — Imaging Abstraction

The contracts distinguish 2D pyramidal image coordinates from volumetric physical/voxel coordinates.

### AC-006 — Typed Outputs

Classification, segmentation, detection, and volumetric evidence can be represented without forcing all results into a single classification schema.

### AC-007 — Validation Awareness

Capability resolution differentiates scientifically validated models from technically supported but unvalidated combinations.

### AC-008 — Reproducibility

A pipeline execution records dataset, model, preprocessing, stage versions, and execution identity.

### AC-009 — DPA Readiness

Imaging and Analysis API interfaces are documented and testable independently of the DPA implementation.

### AC-010 — Dependency Isolation

Architecture tests reject domain imports from infrastructure, presentation, or concrete plugins.

### AC-011 — Negative Tests

Tests verify failure for incompatible stage contracts, missing required preprocessing, unsupported model inputs, and invalid spatial references.

### AC-012 — No Legacy Changes

No modifications are made to `digital-pathology` or `hal-viewer`.

## 10. Deliverables

Required repository artifacts:

```text
.claude/
├── specs/
│   └── 001-platform-foundations.md
├── prompts/
│   └── 001-platform-foundations.md
├── adr/
│   └── 001-platform-architecture.md
├── architecture/
│   └── 001-platform-boundaries.md
└── reports/
    └── 001-platform-foundations.md
```

The implementation report must include:

- Files created and modified.
- Architecture and contract decisions.
- Tests executed and results.
- Demonstration commands and outputs.
- Requirements implemented versus deferred.
- Known limitations.
- Deviations from this specification.
- Recommendations for the next specification.

## 11. Definition of Done

The foundation is complete only when the acceptance criteria have executable evidence, all required tests pass, the architectural boundaries are documented, and the implementation report is produced.

Passing synthetic foundation tests must not be reported as scientific or clinical validation.

**Status:** Accepted on 2026-10-15; implementation in progress (see `../reports/001-platform-foundations.md`).