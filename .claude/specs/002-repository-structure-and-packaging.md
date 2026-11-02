# 002 — Repository Structure & Packaging Refactor

**Status:** Accepted (approved 2026-10-16)  
**Project:** Pathology AI  
**Repository:** `pathology-ai`  
**Priority:** P0  
**Type:** Architecture Refactoring  
**Depends on:** 001 — Platform Foundations  
**Related ADR:** `002-root-level-package-layout.md`

## 1. Context

The Pathology AI foundation was implemented under specification `001-platform-foundations.md`.

The implementation established a modular, model-agnostic and modality-extensible architecture with:

- Domain entities and validation contracts.
- Ports and in-memory registries.
- Pipeline definitions and execution services.
- Synthetic WSI classification and segmentation pipelines.
- Volumetric pipeline contracts.
- Imaging and Analysis API schemas.
- Scientific provenance and reproducibility tracking.
- Automated unit, contract, architecture and integration tests.

The SPEC-001 implementation report records 149 passing tests, zero failures, and successful execution of three synthetic projects.

However, the implementation introduced an additional Python package directory, `pathology_ai/`, containing the framework's core modules.

The project has adopted a different repository organization: **all first-party framework modules must reside directly under the repository root**.

This specification corrects the physical layout without redesigning the architecture or introducing new functionality.

## 2. Problem Statement

Current structure:

```text
pathology-ai/
├── pathology_ai/
│   ├── domain/
│   ├── ports/
│   ├── application/
│   ├── infrastructure/
│   ├── plugins/
│   ├── presentation/
│   ├── bootstrap.py
│   └── cli.py
├── pipelines/
├── projects/
├── tests/
├── .claude/
└── pyproject.toml
```

Required structure:

```text
pathology-ai/
├── domain/
├── ports/
├── application/
├── infrastructure/
├── plugins/
├── presentation/
│   └── api/
├── pipelines/
├── projects/
├── tests/
├── bootstrap.py
├── cli.py
├── .claude/
├── pyproject.toml
├── uv.lock
└── README.md
```

The new structure must not retain `pathology_ai/` as a second copy, wrapper package, compatibility namespace, or runtime dependency.

## 3. Objectives

1. Move all first-party framework modules from `pathology_ai/` to the repository root.
2. Preserve architectural separation between domain, ports, application, infrastructure, plugins and presentation.
3. Update Python imports and package configuration.
4. Preserve the existing pipeline definitions, project configurations and synthetic execution behavior.
5. Preserve Imaging and Analysis API contracts.
6. Preserve provenance and reproducibility semantics.
7. Maintain the existing command-line demonstration.
8. Validate installation and execution from outside the repository working directory.
9. Update tests and architecture documentation.
10. Ensure that no changes are made to `digital-pathology` or `hal-viewer`.

## 4. Architectural Decision

### 4.1 Root-Level Modules

The following modules must be located directly under the repository root:

```text
domain/
ports/
application/
infrastructure/
plugins/
presentation/
```

The following files must also reside at the repository root:

```text
bootstrap.py
cli.py
```

The following existing directories remain at the root:

```text
pipelines/
projects/
tests/
.claude/
```

### 4.2 Python Import Strategy

The refactor must use explicit absolute imports between first-party packages.

Examples:

```python
from domain.pipeline import PipelineDefinition
from domain.validation import ValidationRecord
from ports.model import Model
from application.runner import PipelineRunner
```

Imports must not reference the former `pathology_ai` package.

The refactor must avoid:

- `sys.path` manipulation in production code.
- Importing modules through filesystem-relative paths.
- Maintaining duplicate modules for backward compatibility.
- Creating an artificial replacement namespace solely to preserve the previous structure.
- Depending on the current working directory for module discovery.

Relative imports within a package may be retained where appropriate, but cross-package imports must be explicit and stable.

### 4.3 Packaging

The Python distribution may continue to be named `pathology-ai`.

The distribution name and import package names are separate concepts.

The project must use an explicit packaging configuration that installs the root-level packages correctly.

The implementation must demonstrate that imports work after installation in an isolated environment, including when the current working directory is outside the repository.

Package discovery must be explicitly controlled so that configuration, governance, tests and project datasets are not unintentionally packaged as importable framework modules.

### 4.4 Dependency Direction

The refactor must preserve the existing dependency rules:

```text
domain
  ↑
ports
  ↑
application
  ↑
infrastructure / plugins / presentation
```

The diagram represents allowed inward dependencies, not mandatory imports between every layer.

Rules:

- `domain` must remain independent of infrastructure, plugins, presentation and ML frameworks.
- Ports must not depend on concrete adapters.
- Application services must depend on interfaces rather than concrete plugins.
- Infrastructure and plugins may implement domain-owned ports.
- Presentation must not contain scientific or model-specific business logic.
- The composition root may assemble concrete implementations.

The physical location of a module must not weaken these rules.

## 5. Scope of Work

### 5.1 Repository Inventory

Before modifying files, inspect and record:

- All Python modules under `pathology_ai/`.
- All imports referencing `pathology_ai`.
- Package discovery configuration.
- CLI entry points.
- Tests and fixtures.
- JSON Schema resources.
- YAML pipeline and project definitions.
- Runtime resource-loading mechanisms.
- Documentation references.
- Provenance and fingerprint generation code.

Create a migration inventory identifying each file's old and new path.

### 5.2 Physical Migration

Move the existing implementation to the new root-level layout.

The migration must preserve file contents wherever possible.

Avoid rewriting domain entities, pipeline execution logic, model interfaces, or serializers unless required for import/package compatibility.

The migration must not create parallel implementations.

### 5.3 Import Migration

Update all first-party imports to the new root-level packages.

Examples:

```python
# Before
from pathology_ai.domain.pipeline import PipelineDefinition

# After
from domain.pipeline import PipelineDefinition
```

Apply equivalent changes to:

- Application services.
- Infrastructure adapters.
- Synthetic plugins.
- API serializers.
- Bootstrap and CLI.
- Unit tests.
- Contract tests.
- Architecture tests.
- Integration tests.

Import migration must be deterministic and auditable.

### 5.4 Packaging and Installation

Update `pyproject.toml` and related packaging configuration.

Requirements:

- Continue using Python 3.12 or newer.
- Continue using uv and hatchling unless a concrete incompatibility is demonstrated.
- Explicitly include framework packages.
- Preserve `pathology-ai-demo`.
- Support isolated installation.
- Avoid relying on editable-install side effects.
- Verify that installed modules resolve from the installed environment rather than an accidental checkout on `sys.path`.

A wheel must be built and inspected.

The implementation must verify the installed distribution using a clean environment and a working directory outside the repository.

### 5.5 Pipeline and Project Compatibility

The following existing pipelines must continue to work:

```text
pipelines/wsi-patch-embed-aggregate.v1.yaml
pipelines/wsi-direct-segmentation.v1.yaml
pipelines/volume-segmentation-3d.v1.yaml
```

The following projects must remain valid:

```text
projects/patologia/cancer/pulmao/
projects/patologia/cancer/coloretal/
projects/patologia/pele/psoriase/
```

The exact pipeline filenames and versions must be confirmed against the repository inventory before any edits.

Project definitions must not be rewritten to compensate for the new package layout unless they contain an explicit module reference requiring migration.

### 5.6 Runtime Resource Loading

Review how the application locates:

- Pipeline YAML files.
- Project configuration files.
- Imaging API JSON Schemas.
- Analysis API JSON Schemas.
- API fixtures.
- Plugin registrations.

Resource discovery must not depend on running the CLI from the repository root.

If resource files are required by an installed distribution, they must be packaged or resolved through an explicit configured resource location.

Do not hardcode developer-machine paths.

### 5.7 API Contract Preservation

The existing Imaging API and Analysis API JSON Schemas must remain semantically unchanged.

The migration must not change:

- API version identifiers.
- Field names.
- Required properties.
- Typed output variants.
- Geometry conventions.
- Evidence semantics.
- Capability status values.
- Job state values.
- Validation scope requirements.

Where possible, assert byte-level equality for JSON Schema and fixture files.

If a packaging-related resource location changes, record it separately from contract content.

### 5.8 Provenance and Reproducibility

The refactor must preserve the meaning of execution fingerprints and artifact identities.

A package-path migration must not silently alter scientific input contracts or output semantics.

If the existing fingerprint includes code location or module names, investigate whether identical synthetic runs produce different fingerprints after the refactor.

Do not simply hardcode old fingerprints or remove code identity from provenance to force equality.

Document any unavoidable identity change and distinguish it from a change in model behavior.

### 5.9 Documentation and Governance

Update the following existing documents where paths or examples become incorrect:

```text
README.md
.claude/architecture/001-platform-boundaries.md
.claude/architecture/dpa-v1-compatibility-inventory.md
```

The original SPEC-001 report must remain an accurate historical record.

Do not rewrite historical evidence to pretend the root-level layout existed during SPEC-001.

Record the migration in the SPEC-002 report.

Do not create a `docs/` directory.

## 6. Out of Scope

This specification does not authorize:

- Implementing real CZI, SVS, TIFF or DICOM readers.
- Implementing CT or MRI execution.
- Adding model training or evaluation.
- Implementing new model architectures.
- Changing pipeline semantics.
- Changing dataset formats.
- Introducing persistent storage.
- Building a production API server.
- Implementing the DPA compatibility adapter.
- Modifying the DPA.
- Modifying the legacy Digital Pathology repository.
- Introducing a microservices architecture.
- Renaming the project or distribution.

## 7. Acceptance Criteria

### AC-001 — Required Root Layout

All framework modules reside directly under the repository root.

The former `pathology_ai/` directory no longer exists.

### AC-002 — Import Migration

No production or test imports depend on `pathology_ai`.

All imports resolve using the new package layout.

### AC-003 — Packaging Integrity

The project builds an installable wheel containing the required framework modules and runtime resources.

No accidental dependency on the source checkout is required.

### AC-004 — External Working Directory

The CLI and core imports function from a working directory outside the repository after a clean installation.

### AC-005 — Existing Test Suite

All 149 tests reported by SPEC-001 continue to pass, unless a test must be updated solely because of the changed import path.

No existing behavioral coverage may be deleted to achieve a passing result.

### AC-006 — Synthetic Pipeline Parity

The two synthetic cancer projects continue to execute the same WSI pipeline.

The synthetic psoriasis segmentation pipeline continues to execute.

Outputs and validation states remain equivalent to the pre-refactor baseline.

### AC-007 — Volumetric Contract Preservation

The CT/MRI pipeline continues to validate structurally and continues to report unsupported execution rather than performing unimplemented inference.

### AC-008 — API Contract Preservation

Imaging and Analysis API schemas and fixtures retain their existing content and behavior.

Existing contract tests pass.

### AC-009 — Architecture Rules

Dependency-direction tests pass using the new package paths.

New architecture tests detect violations introduced through root-level imports.

### AC-010 — Provenance Integrity

The report compares pre- and post-refactor fingerprints and explains any differences.

Pipeline and model identities must not change merely because Python source files moved.

### AC-011 — Repository Independence

Pathology AI remains executable without sibling `digital-pathology` or `hal-viewer` repositories.

### AC-012 — Legacy Repository Protection

Neither `digital-pathology` nor `hal-viewer` is modified.

### AC-013 — Documentation Accuracy

Current architecture documentation reflects the new layout, while historical reports remain unchanged.

### AC-014 — No Functional Scope Expansion

The refactor introduces no new medical imaging readers, trained models, inference algorithms or API services.

## 8. Required Validation

At minimum, execute:

```bash
uv sync

uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest -q -W error

uv run pathology-ai-demo
uv build
```

Additionally:

1. Inspect the wheel contents.
2. Install the wheel into a clean environment.
3. Change to a directory outside the repository.
4. Import the installed root-level packages.
5. Execute the installed CLI.
6. Run representative synthetic pipeline checks.
7. Compare outputs and fingerprints against the SPEC-001 baseline.
8. Verify API Schema and fixture preservation.
9. Verify no unexpected changes to sibling repositories.

If the installed CLI requires explicit project configuration paths, document the supported invocation and test it from an external working directory.

All validation commands and their outputs must be recorded.

## 9. Implementation Sequence

### Phase A — Baseline and Plan

- Inspect repository and package structure.
- Record current tests, CLI outputs, fingerprints and contract hashes.
- Identify package resources and entry points.
- Propose the migration mapping.
- Confirm compatibility risks.

### Phase B — Structural Migration

- Move modules.
- Update imports.
- Update package configuration.
- Repair resource loading and CLI installation.
- Preserve behavior.

### Phase C — Regression Validation

- Run the full test suite.
- Build and inspect the wheel.
- Test installation outside the checkout.
- Compare outputs and contracts.
- Run architecture tests.
- Confirm repository isolation.

### Phase D — Documentation and Report

- Update current architecture documentation.
- Record changes, deviations and evidence.
- Produce the implementation report.

## 10. Deliverables

Required files:

```text
.claude/
├── specs/
│   └── 002-repository-structure-and-packaging.md
├── prompts/
│   └── 002-repository-structure-and-packaging.md
├── adr/
│   └── 002-root-level-package-layout.md
└── reports/
    └── 002-repository-structure-and-packaging.md
```

The report must include:

1. Executive summary.
2. Before/after repository tree.
3. File migration inventory.
4. Packaging configuration changes.
5. Test results.
6. Wheel inspection results.
7. External-directory installation evidence.
8. Synthetic output and fingerprint comparison.
9. API contract preservation evidence.
10. Architecture test evidence.
11. Legacy repository protection evidence.
12. Known limitations and remaining risks.

## 11. Definition of Done

This specification is complete when:

- All framework modules reside at the repository root.
- The old `pathology_ai/` directory is removed.
- Imports and packaging function correctly.
- Existing synthetic pipelines retain their behavior.
- API contracts remain unchanged.
- Tests and isolated installation pass.
- Scientific provenance remains traceable.
- No legacy repository is modified.
- The implementation report documents the results.

**The purpose of SPEC-002 is structural correction, not functional expansion.**

**Status:** Accepted on 2026-10-16; implementation report: `../reports/002-repository-structure-and-packaging.md`.