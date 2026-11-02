# PROMPT-008B — Cohort Reconciliation and Manifest Construction

Implement Phase B of SPEC-008 in the `pathology-ai` repository.

## Approved scientific decisions for Phase B

Use the following proposed policy, subject to owner confirmation:

- D-0: Close the SPEC-007 MSI label contradiction as a cross-target comparison error.
- D-1: Use the published upstream MSI CSVs, pinned by SHA-256, as the study label authority.
- D-2: Exclude SR1482 `_02` slides from the primary analysis.
- D-3: Use the published combined MSI/dMMR endpoint, preserving source-cohort label semantics.
- D-4: Use cohort-qualified T-number as the grouping key.
- D-5: Preserve the upstream train/validation/test split and classify test as a reused benchmark.
- D-6: Reserve train/validation slides for the future extraction pilot.
- D-8: Document a separate legacy embedding comparability arm; do not import embeddings during Phase B.

Do not treat these decisions as approved unless the repository owner explicitly confirms them.

## Implementation

After approval:

1. Implement generic cohort and label reconciliation contracts.
2. Implement declarative CSV source mappings.
3. Create the SurGen-specific cohort configuration under `projects/cohorts/surgen-msi/`.
4. Pin authoritative source digests and upstream revision.
5. Normalize identifiers without using legacy positional `slide_id`.
6. Create the versioned cohort manifest covering all 1,020 WSI records.
7. Record the 820 provisionally eligible `_01` slides and 200 excluded slides, subject to WSI integrity checks.
8. Import the original split through SPEC-007's external split contract.
9. Validate zero group overlap and consistent labels.
10. Add synthetic contract tests covering the observed data hazards.

## Required reports

- `.claude/reports/008-label-reconciliation.md`
- `.claude/reports/008-phase-b-implementation.md`

Report:

- Exact source identities and digests.
- Eligibility counts.
- Exclusion reasons.
- Split digest and group integrity.
- Contract validation results.
- Unverified WSI availability and content integrity.
- Remaining blockers for Phase C.

## Constraints

- No WSI extraction.
- No model training.
- No automatic clinical label adjudication.
- No split regeneration.
- No changes to sibling repositories.
- No Git commits.
- No clinical performance claims.

Run appropriate unit, integration, type-checking and regression tests.

Stop after Phase B and wait for approval before starting Phase C.