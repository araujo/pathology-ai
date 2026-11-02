# ADR 002 — Root-Level Framework Package Layout

**Status:** Accepted (approved 2026-10-16)  
**Project:** Pathology AI  
**Repository:** `pathology-ai`  
**Related SPEC:** `../specs/002-repository-structure-and-packaging.md`  
**Supersedes:** The physical package layout proposed in SPEC-001 and implemented under ADR-001  
**Preserves:** The logical architecture and dependency rules of ADR-001

## 1. Context

Pathology AI is an independent, extensible medical imaging AI framework.

ADR-001 established a modular-monolith architecture based on:

- Pure domain contracts.
- Ports and adapters.
- Composable pipelines.
- Independent imaging modalities.
- Pluggable models.
- Scientific provenance.
- Versioned Imaging and Analysis API contracts.
- Future integration with DPA.

The implementation of SPEC-001 introduced an intermediate Python package directory:

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

This structure is technically valid in Python, but it does not match the repository organization selected by the project owner.

The owner requires the framework modules to be located directly under the `pathology-ai` repository root.

This ADR records that decision and its implications for packaging, imports, architecture and maintenance.

## 2. Decision

### D1 — Root-Level Framework Modules

All first-party framework modules must be placed directly under the repository root.

The required layout is:

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
└── README.md
```

The `pathology_ai/` wrapper directory must be removed.

The project will not maintain a duplicate implementation or compatibility package under the old namespace.

### D2 — Distribution and Import Names Are Independent

The Python distribution name remains:

`pathology-ai`

The Python import names become the root-level packages, such as:

```python
from domain.pipeline import PipelineDefinition
from ports.model import Model
from application.runner import PipelineRunner
```

The repository name is not used directly as a Python import package.

### D3 — Explicit Package Discovery

The build system must explicitly include first-party root-level packages.

Package discovery must not accidentally expose tests, governance files or student project configurations as Python packages.

Runtime resources required by installed code must be packaged or resolved through documented configuration.

A clean, non-editable wheel installation is required to verify this decision.

### D4 — Preserve Dependency Inversion

The logical architecture from ADR-001 remains unchanged.

Dependencies must point inward toward domain contracts and ports.

Domain modules must remain independent of:

- Infrastructure implementations.
- API frameworks.
- Concrete plugins.
- NumPy.
- PyTorch.
- External ML libraries.

Application services must depend on abstractions rather than concrete adapters.

The composition root is responsible for wiring implementations.

### D5 — No Working-Directory Dependency

The installed framework must function when invoked outside the repository root.

Runtime imports and required packaged resources must not depend on the current working directory.

Production `sys.path` manipulation is prohibited.

### D6 — Behavioral and Contract Preservation

The structural refactor must preserve:

- Existing synthetic pipeline behavior.
- Project configuration semantics.
- Model registration.
- Scientific validation rules.
- Prediction and evidence semantics.
- Imaging and Analysis API schemas.
- Reproducibility and provenance semantics.

The migration does not authorize functional changes.

### D7 — No Legacy Compatibility Namespace

The project is still in its initial development phase and has no established external consumers of the `pathology_ai` import namespace.

Therefore, the old package namespace will be removed rather than deprecated.

If external consumers are discovered during implementation, the migration must stop for an explicit compatibility decision.

### D8 — No Changes to Other Repositories

The refactor is limited to `pathology-ai`.

No modifications are authorized in:

- `digital-pathology`
- `hal-viewer`

The future DPA integration remains based on versioned API contracts, not Python imports.

## 3. Alternatives Considered

### Alternative A — Keep `pathology_ai/`

**Rejected.**

This is a conventional Python package layout, but it conflicts with the explicitly selected repository organization.

### Alternative B — Use `src/pathology_ai/`

**Rejected.**

A `src/` layout can provide strong packaging isolation, but it introduces an additional directory hierarchy and does not meet the owner's root-level requirement.

### Alternative C — Root-Level Packages

**Accepted.**

The framework modules are directly visible under the repository root.

This layout requires explicit package discovery, import discipline and architecture tests.

### Alternative D — Root-Level Code with `sys.path` Manipulation

**Rejected.**

It may work during local development but creates fragile installation and runtime behavior.

### Alternative E — Keep Both Layouts

**Rejected.**

Maintaining duplicate modules or a compatibility wrapper would create unnecessary complexity, ambiguous import resolution and additional maintenance obligations.

## 4. Consequences

### Positive

- Repository organization matches the project owner's decision.
- Core framework components are immediately visible at the repository root.
- No redundant `pathology_ai/` directory.
- Existing architectural modules remain independently organized.
- Future imaging readers, models and pipelines can follow the same physical layout.
- No cross-repository dependency is introduced.

### Negative

- Root-level import names such as `domain`, `application` and `plugins` are generic and may collide with third-party packages.
- The build configuration requires explicit maintenance.
- Import paths change across the existing codebase.
- Console entry points and resource resolution require regression testing.
- Future developers must understand that the distribution name and import names differ.

### Mitigations

- Use explicit package discovery.
- Maintain strict dependency tests.
- Verify wheel contents.
- Test imports in isolated environments.
- Test execution from outside the repository.
- Detect package-name collisions during dependency upgrades.
- Avoid ambiguous import aliases or dynamic path manipulation.

## 5. Scientific and Integration Implications

This ADR does not change:

- Medical imaging preprocessing.
- Model training or inference.
- Scientific validation status.
- Dataset identity.
- Pipeline semantics.
- Spatial coordinate systems.
- Prediction meaning.
- DPA API contracts.

The existing Digital Pathology implementation remains an independent scientific reference.

The DPA remains unchanged.

Pathology AI must continue to prepare separate Imaging API and Analysis API contracts for future integration.

## 6. Implementation Requirements

SPEC-002 must demonstrate:

1. Complete removal of the `pathology_ai/` directory.
2. Successful imports from root-level packages.
3. Correct wheel contents.
4. Non-editable installation from an external working directory.
5. Successful execution of existing synthetic projects.
6. Preservation of API contracts.
7. Preservation or documented explanation of fingerprint differences.
8. Passing unit, integration, contract and architecture tests.
9. No changes to legacy repositories.

## 7. Scope Boundary

This ADR governs the physical organization and packaging of the Pathology AI framework.

It does not authorize:

- New imaging readers.
- New model architectures.
- CT/MRI runtime execution.
- Real model training.
- Persistent infrastructure.
- API server deployment.
- DPA migration.
- Scientific parity claims.

These require separate specifications and approval.

## 8. Status and Approval

**Status:** Accepted (approved 2026-10-16)

Once approved, this ADR supersedes only the physical directory-layout decision from the initial foundation.

All other architectural decisions from ADR-001 remain in force.