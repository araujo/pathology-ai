# PROMPT 001 — Implement Pathology AI Platform Foundations

You are Claude Code acting as a senior software architect and implementation engineer in the **new, independent** repository `pathology-ai`.

## Read first

1. `.claude/specs/001-platform-foundations.md` — authoritative scope and acceptance criteria.
2. `.claude/adr/001-platform-architecture.md` — architectural decisions and non-decisions.
3. `.claude/CLAUDE.md` and any repository rules, if present.
4. The Digital Pathology SPEC-025 architecture assessment and SPEC-024.5 preprocessing report **only if accessible as reference**. Do not claim to have read inaccessible files; ask for them if critical.

## Mandatory workflow

**Phase A — Inspect and plan, before edits:** list current repository structure, available runtimes, tests and conventions. Identify missing prerequisites, risks and ambiguous choices. Present a concise implementation plan mapped to AC-001 through AC-012. If SPEC/ADR approval is not recorded, stop after the plan and request approval. Do not create code on a presumed approval.

**Phase B — Implement the approved minimal foundation:**

- Establish Python package and dependency/test tooling according to confirmed repository conventions; avoid inventing an incompatible toolchain.
- Implement pure domain entities, IDs, typed errors, coordinate frames and validation/capability records.
- Implement explicit ports and in-process registries for readers, stages, models, datasets, artifacts and provenance. Keep domain/application imports independent of concrete plugins.
- Define a versioned typed DAG pipeline schema and deterministic compatibility validation (cycle detection, input/output kinds, modality, preprocessing binding and resource requirements).
- Implement a small synthetic in-memory WSI flow using fake reader/patcher/encoder/MIL-like stage/prediction; it is a contract test, **not** a medically valid model.
- Add a structurally different pipeline schema (direct image→segmentation) and a 3D CT/MRI volume schema; these may be non-executable but must validate their representations and report unsupported execution explicitly.
- Provide `projects/patologia/cancer/pulmao`, `projects/patologia/cancer/coloretal`, `projects/patologia/pele/psoriase` examples sharing the appropriate pipeline definition, with clearly synthetic dataset and model IDs.
- Define independent versioned Imaging API and Analysis API schemas/fixtures; include image identity, 2D/3D geometry, typed results, model validation scope, evidence semantics, and async job states. Do **not** build a production server or assert DPA wire parity.
- Write a read-only DPA v1 compatibility inventory with verified fields and unresolved gaps; if actual DPA contract sources are inaccessible, mark the inventory as pending verification rather than inventing field-level parity.
- Implement reproducibility fingerprint interfaces that include reader, code/plugin, model, preprocessing, environment and dataset versions. Do not reuse DP's legacy IDs.
- Add unit, contract, architecture and synthetic integration tests. Use negative tests for wrong dimensions, missing transforms, unsupported modality, cycles, unlabeled assets, unvalidated research opt-in and evidence semantics.

**Phase C — Validate:** run format/lint/type/test commands supported by the project. Record exact commands, counts and failures. Distinguish tested behavior from deferred functionality. Never call a schema-only reader or API an implemented integration.

## Hard prohibitions

- Do not modify `digital-pathology` or `hal-viewer`.
- Do not import or copy legacy DP implementation as the new runtime.
- Do not hardcode CZI, colorectal, UNI, attention MIL, H&E or KRAS into core domain abstractions.
- Do not require a task/model to ingest an unlabeled image.
- Do not assume DICOM means CT; do not treat WSI pixel coordinates as 3D voxel coordinates.
- Do not silently swap BGR/RGB or claim scientific parity without tests.
- Do not download real models, process patient data or add clinical performance claims.
- Do not create a new `docs/` folder; place governance/architecture under `.claude/`.
- Do not implement CT/MRI readers, real model training, full DPA integration or deployment in 001.

## Required report

Write `.claude/reports/001-platform-foundations.md` containing:

1. Executive outcome and commit/branch identifiers, if available.
2. AC-001 through AC-012 status table (`PASS`, `PARTIAL`, `FAIL`, `NOT_RUN`) with evidence links and exact file paths.
3. Architecture and module boundaries; implemented versus schema-only components.
4. Examples showing one WSI pipeline reused by different project configs and a non-MIL pipeline represented.
5. Imaging/Analysis API contract inventory, geometry and DPA v1 compatibility gaps.
6. Scientific caveats, explicit unsupported modalities and research-only status.
7. Exact validation commands, test totals, warnings and remaining risks.
8. Follow-up recommendations for SPEC-002 and later phases; no unapproved expansion of scope.

## Completion criterion

Return a concise summary with what actually runs, what is only defined by contract, any blockers and the report path. Never state “fully generic” or “DPA compatible” without corresponding executed evidence.