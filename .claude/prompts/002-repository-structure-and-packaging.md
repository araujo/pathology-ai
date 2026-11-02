# PROMPT 002 — Repository Structure & Packaging Refactor

**Project:** Pathology AI  
**Repository:** `pathology-ai`  
**Related SPEC:** `.claude/specs/002-repository-structure-and-packaging.md`  
**Related ADR:** `.claude/adr/002-root-level-package-layout.md`  
**Output Report:** `.claude/reports/002-repository-structure-and-packaging.md`

## Role

You are Claude Code acting as a senior Python software architect and implementation engineer.

Your responsibility is to refactor the existing Pathology AI repository structure without changing the framework's scientific, functional, or integration behavior.

The owner has explicitly decided that the framework's first-party modules must live directly under the `pathology-ai` repository root.

The current intermediate `pathology_ai/` directory must be eliminated.

This is an approved architectural direction, but implementation must still follow the approval status and execution requirements of the authoritative SPEC and ADR.

## 1. Read Before Acting

Read the following files in order:

1. `.claude/CLAUDE.md`, if present.
2. `.claude/specs/001-platform-foundations.md`
3. `.claude/adr/001-platform-architecture.md`
4. `.claude/reports/001-platform-foundations.md`
5. `.claude/specs/002-repository-structure-and-packaging.md`
6. `.claude/adr/002-root-level-package-layout.md`
7. `.claude/architecture/001-platform-boundaries.md`
8. `.claude/architecture/dpa-v1-compatibility-inventory.md`

Treat SPEC-002 as the authoritative implementation scope.

Treat ADR-002 as the authoritative decision about physical package layout.

Do not silently reinterpret or expand either document.

If SPEC-002 or ADR-002 is still marked `Proposed`, perform the inspection and planning phase only. Request approval before modifying code.

## 2. Mandatory Target Structure

The target repository structure is:

```text
pathology-ai/
├── domain/
├── ports/
├── application/
├── infrastructure/
├── plugins/
│   └── synthetic/
├── presentation/
│   └── api/
├── pipelines/
├── projects/
├── tests/
├── bootstrap.py
├── cli.py
├── .claude/
│   ├── adr/
│   ├── architecture/
│   ├── specs/
│   ├── prompts/
│   └── reports/
├── pyproject.toml
├── uv.lock
└── README.md
```

The following structure is forbidden after migration:

```text
pathology-ai/
└── pathology_ai/
    ├── domain/
    ├── ports/
    ├── application/
    ├── infrastructure/
    ├── plugins/
    └── presentation/
```

Do not retain `pathology_ai/` as a wrapper, alias, compatibility namespace, or duplicate implementation.

## 3. Phase A — Baseline and Migration Plan

Before modifying any files:

### 3.1 Inspect the repository

Inventory:

- All modules under `pathology_ai/`.
- Python package initializers.
- Imports referencing `pathology_ai`.
- `pyproject.toml` build configuration.
- Hatchling package discovery.
- CLI entry points.
- Test imports and fixtures.
- Architecture dependency tests.
- JSON Schema resources.
- Pipeline YAML files.
- Project YAML files.
- Plugin registration and discovery.
- Resource-loading paths.
- Provenance and fingerprint generation.

Do not assume that moving directories alone will preserve the installation behavior.

### 3.2 Record the baseline

Run and record:

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest -q -W error
uv run pathology-ai-demo
```

Record:

- Test counts and failures.
- CLI outputs.
- Pipeline identifiers.
- Model identifiers.
- Prediction outputs.
- Validation states.
- Execution fingerprints.
- Artifact identities.
- API contract hashes.
- Package entry points.

Do not modify source code before capturing this baseline.

If the baseline differs from the SPEC-001 report, document the difference rather than assuming the previous report is still current.

### 3.3 Inspect sibling repositories

Read-only inspection is permitted for:

- `../digital-pathology`
- `../hal-viewer`

Record their current Git status and commit identifiers, if available.

Do not modify, clean, reset, stage, or commit anything in those repositories.

### 3.4 Present the migration plan

Provide:

1. Old-to-new file mapping.
2. Packaging strategy for root-level Python packages.
3. Resource-loading strategy.
4. Import migration strategy.
5. Identified risks.
6. Baseline results.

Stop and request clarification if a material conflict with the approved SPEC or ADR is discovered.

## 4. Phase B — Structural Migration

After approval:

### 4.1 Move framework modules

Move:

```text
pathology_ai/domain/          → domain/
pathology_ai/ports/           → ports/
pathology_ai/application/     → application/
pathology_ai/infrastructure/  → infrastructure/
pathology_ai/plugins/         → plugins/
pathology_ai/presentation/    → presentation/
pathology_ai/bootstrap.py     → bootstrap.py
pathology_ai/cli.py           → cli.py
```

Preserve implementation content wherever possible.

Do not rewrite existing algorithms or domain models.

Do not leave duplicate modules behind.

### 4.2 Update imports

Replace old package imports with root-level imports.

Example:

```python
# Before
from pathology_ai.domain.pipeline import PipelineDefinition

# After
from domain.pipeline import PipelineDefinition
```

Use stable absolute imports across first-party packages.

Relative imports within an individual package are permitted.

Do not use production `sys.path` manipulation to make imports work.

Do not introduce a compatibility layer that restores the old `pathology_ai` namespace.

### 4.3 Update Python packaging

Continue using:

- Python 3.12+
- uv
- hatchling
- pytest
- ruff
- mypy

The distribution may retain the name `pathology-ai`.

Configure package discovery explicitly to include the framework's root-level packages.

Avoid accidentally installing `tests`, `projects`, `pipelines`, or governance directories as Python packages.

Ensure required runtime resources are included in the wheel.

Keep the existing `pathology-ai-demo` console command operational.

The implementation must work after a non-editable wheel installation from a directory outside the source checkout.

### 4.4 Preserve configuration and resource loading

Verify that the application can locate:

- Pipeline definitions.
- Project configurations.
- Imaging API schemas.
- Analysis API schemas.
- Runtime contract resources.

Do not assume that the current working directory is the repository root.

Do not hardcode local developer paths.

If pipeline/project definitions remain external to the wheel, require explicit configuration paths and document the supported usage.

### 4.5 Preserve pipeline behavior

The following existing behaviors must remain unchanged:

- Lung and colorectal synthetic projects reuse the same WSI pipeline.
- Psoriasis executes its direct segmentation pipeline.
- CT/MRI pipeline definitions remain structurally valid.
- Unsupported volumetric execution still fails explicitly.
- Model registration and capability resolution work.
- Scientific validation scope remains enforced.
- Research-only execution requires explicit opt-in.
- Attention evidence is not misrepresented as a calibrated probability.

Do not introduce new model families, datasets, preprocessing algorithms, readers, or training functionality.

### 4.6 Preserve API contracts

Do not change the meaning or version of existing Imaging and Analysis API contracts.

Preserve:

- Canonical image identities.
- Imaging modality descriptors.
- 2D and 3D spatial references.
- Prediction output types.
- Evidence semantics.
- Validation states.
- Job states.
- Provenance fields.

Compare schema and fixture hashes against the baseline.

The DPA compatibility inventory may have its file-path references updated, but its compatibility claims must not be expanded.

### 4.7 Preserve provenance

Compare pre- and post-refactor execution fingerprints and artifact identities.

If any identity changes:

1. Identify the exact cause.
2. Determine whether the change reflects source-layout information or a genuine scientific input change.
3. Avoid hiding the change through hardcoded values.
4. Document the difference and its impact.

Do not modify fingerprint algorithms merely to force equality.

## 5. Phase C — Validation

Execute all existing checks:

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest -q -W error
uv run pathology-ai-demo
uv build
```

### 5.1 Packaging verification

- Inspect the generated wheel.
- Confirm the required root-level packages are included.
- Confirm no obsolete `pathology_ai` package remains.
- Confirm runtime JSON resources are present.
- Install the wheel in a clean environment.
- Execute imports from outside the checkout.
- Execute the installed CLI from outside the checkout.
- Verify that the CLI can resolve runtime resources through supported configuration.

Do not treat an editable installation as sufficient evidence.

### 5.2 Architecture verification

Verify:

- Domain imports remain infrastructure-independent.
- Application services continue to depend on ports.
- Concrete plugins do not leak into domain contracts.
- No cross-repository imports exist.
- No runtime reference to the old `pathology_ai` namespace remains.

Update architecture tests to cover the new import paths.

### 5.3 Behavioral regression

Compare:

- Project execution outcomes.
- Prediction values.
- Validation states.
- Pipeline IDs and versions.
- Model IDs and versions.
- Artifact identities.
- Fingerprints.
- API Schema hashes.

A change in package location must not be treated as permission to change scientific behavior.

### 5.4 Repository isolation

Confirm that the test suite and demo work without sibling repositories.

Recheck `digital-pathology` and `hal-viewer` against the previously recorded baseline.

Report any differences accurately.

## 6. Phase D — Documentation and Report

Update current architecture references:

- `README.md`
- `.claude/architecture/001-platform-boundaries.md`
- `.claude/architecture/dpa-v1-compatibility-inventory.md`, only where needed

Do not rewrite the historical SPEC-001 implementation report.

Create:

`.claude/reports/002-repository-structure-and-packaging.md`

The report must contain:

1. Executive outcome.
2. Acceptance criteria AC-001 through AC-014, each with status and evidence.
3. Before/after repository structure.
4. File migration inventory.
5. Packaging and entry-point changes.
6. Resource-loading strategy.
7. Test results and exact commands.
8. Wheel contents and external-installation evidence.
9. API contract hash comparison.
10. Synthetic prediction and provenance comparison.
11. Architecture dependency validation.
12. Legacy repository status comparison.
13. Known limitations and unresolved risks.
14. Deviations from SPEC-002, if any.

Use `PASS`, `PARTIAL`, `FAIL`, or `NOT_RUN` for each acceptance criterion.

## 7. Hard Constraints

- Do not modify `digital-pathology`.
- Do not modify `hal-viewer`.
- Do not introduce new imaging readers.
- Do not implement CT/MRI inference.
- Do not implement training.
- Do not change existing prediction semantics.
- Do not alter Imaging/Analysis API contract versions.
- Do not create a new `docs/` directory.
- Do not create a `pathology_ai` compatibility package.
- Do not duplicate framework implementations.
- Do not delete tests to make the suite pass.
- Do not commit changes unless explicitly requested.

## 8. Completion

The task is complete only when the new root-level layout works as an installed Python distribution, existing tests pass, synthetic pipeline behavior is preserved, API contracts remain stable, and the report contains verifiable evidence.

If the migration cannot satisfy these requirements without architectural compromises, stop and report the blocker rather than introducing an undocumented workaround.

**Final response must include:**

- What was migrated.
- Whether `pathology_ai/` was fully removed.
- Test totals and failures.
- Wheel and external-installation results.
- Whether synthetic predictions and fingerprints were preserved.
- Whether DPA-facing contracts changed.
- Any remaining risks.
- Exact report path.

Do not claim that real medical imaging support, scientific model validation, or DPA runtime compatibility was achieved by this structural refactor.