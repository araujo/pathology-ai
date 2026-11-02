# PROMPT-008 — Real Cohort Integration & Scientific Validation

You are implementing SPEC-008 in the `pathology-ai` repository.

## Authoritative Documents

Read:

1. `.claude/specs/008-real-cohort-integration-and-scientific-validation.md`
2. `.claude/adr/008-real-cohort-validation-architecture.md`
3. `.claude/reports/007-mil-training-and-classification.md`
4. `.claude/reports/006-foundation-model-integration.md`
5. `.claude/reports/005-streaming-patch-to-embedding-pipeline.md`
6. Relevant previous SPECs, ADRs, and architecture reports.

If an authoritative file is missing, report it before proceeding.

## Mandatory Execution Boundary

**Execute Phase A only in this invocation.**

Do not proceed to cohort reconciliation, embedding extraction, or real MIL training until the owner has reviewed the discovery report and explicitly authorized the next phase.

## Phase A — Read-Only Cohort Discovery

### A1. Inspect available data sources

Inspect the `pathology-ai` repository and the sibling `../digital-pathology` repository strictly read-only.

Identify:

- SurGen WSI locations.
- Metadata and label CSVs.
- Existing train/validation/test assignments.
- Patient and slide identifiers.
- WSI-to-label mapping.
- Duplicate or inconsistent records.
- Existing embedding stores.
- Existing model checkpoints.
- Historical SurGen experimental results.

If S3 access is configured, use only read-only inventory operations.

Do not download the full cohort.

Do not copy or modify sibling repository files.

### A2. Audit labels

Compare all relevant MSI label sources.

Report:

- Source names and versions.
- Row counts.
- Label vocabularies.
- Matching identifiers.
- Contradictory labels.
- Missing labels.
- Duplicate rows.
- Unknown or ambiguous mappings.
- Number of potentially eligible slides.

Do not resolve conflicts automatically.

### A3. Audit patient groups and splits

Determine:

- Whether patient identifiers exist and are reliable.
- Whether a patient has multiple slides.
- Whether the same patient occurs across partitions.
- Whether WSI files or slide identities are duplicated.
- Whether legacy split assignments are trustworthy.

Identify every condition that could cause leakage.

### A4. Inspect previous preprocessing and experiments

Document:

- Historical WSI resolution handling.
- Patch generation.
- UNI preprocessing.
- Embedding extraction.
- MIL architecture.
- Training configuration.
- Historical metrics and sample counts.

Separate verified observations from assumptions.

### A5. Assess infrastructure readiness

Determine:

- Available compute and GPU configuration.
- WSI storage accessibility.
- Existing embeddings.
- Checkpoint availability and license limitations.
- Expected extraction volume.
- Rough compute and storage requirements.
- Required approvals.

Do not execute expensive operations.

### A6. Produce discovery deliverables

Write:

- `.claude/reports/008-cohort-discovery.md`
- `.claude/reports/008-label-source-audit.md`
- `.claude/reports/008-split-integrity-audit.md`
- `.claude/architecture/008-real-cohort-integration.md`

Each report must distinguish:

- VERIFIED.
- INFERRED.
- NOT VERIFIED.
- BLOCKED.

Provide an inventory table, label conflict matrix, split audit, scientific comparability assessment, and explicit owner decisions.

## Quality Gates

Run lightweight static and contract checks only if needed.

Do not run full WSI embedding, training, or GPU workloads.

Do not change existing SPEC-001 through SPEC-007 implementation code during Phase A.

## Constraints

- Read-only discovery of source data.
- No label adjudication.
- No split modification.
- No full cohort extraction.
- No training.
- No model downloads.
- No modification of sibling repositories.
- No Git commits.
- No unsupported scientific claims.

## Final Response

Summarize:

1. What data actually exists.
2. Which labels can be trusted.
3. Which labels conflict.
4. Whether the existing splits are leakage-safe.
5. Whether the existing UNI embeddings are reusable.
6. Which infrastructure is ready.
7. What is blocking scientific validation.
8. Proposed Phase B implementation.
9. Owner decisions required.

**STOP after Phase A. Wait for explicit owner approval before continuing.**