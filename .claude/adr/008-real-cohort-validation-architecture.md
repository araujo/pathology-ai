# ADR-008 — Real Cohort Integration & Scientific Validation Architecture

**Status:** Proposed  
**Date:** 2026-11-02  
**Related:** SPEC-008

## Context

The platform now supports real WSI preprocessing, foundation-model embedding extraction, MIL training, evaluation, and slide-level prediction.

The first real supervised study requires cohort-specific data reconciliation and rigorous scientific evaluation.

Legacy SurGen label sources have reported contradictions, and the historical experiment is not automatically equivalent to the new pipeline.

## Decisions

### D1 — Cohort adapters are project-specific

SurGen metadata parsing, label normalization, and source mapping belong under `projects/` or a dedicated cohort adapter.

Generic framework layers must not contain SurGen or MSI-specific branches.

### D2 — Clinical labels are not inferred

Conflicting labels must be recorded and excluded until explicitly resolved by an authorized source.

No majority vote, file precedence, or performance-based label selection is allowed without an approved policy.

### D3 — Patient/group leakage is a hard gate

Training cannot proceed unless the approved split is validated for patient/group isolation.

### D4 — Existing splits are preserved unless invalid

Historical split membership is imported and validated.

Any required split remediation produces a new version and requires approval.

### D5 — Cohort manifests are immutable

Source identities, eligibility decisions, label provenance, exclusions, and approved split assignments are fingerprinted.

### D6 — Reuse the existing embedding and MIL infrastructure

SPEC-006 handles embedding extraction.

SPEC-007 handles supervised MIL training, evaluation, and prediction.

No new model architecture or competing training framework is introduced.

### D7 — Physical scale is an explicit experiment decision

The UNI model documentation does not establish a universal µm/px.

The experiment must explicitly approve its extraction scale and tolerance.

**Approved for SurGen MSI (D-7, 2026-11-02):** 0.5 µm/px ± 15 %.
- Scale: a stored pyramid level within tolerance, no resampling.
- Profile: the existing `uni-official@1` preprocessing profile, with 224 × 224 RGB model input.
- Pipeline: bound by `wsi-patch-embedding-uni@1.0.0`, unchanged.
- On SurGen CZI this selects level 2: 0.44496 µm/px, −11.0 %, measured in Phase C1.
- The 1.0 µm/px alternative is not approved.
- See the decision record below.

### D8 — Pilot before full extraction

A representative real-WSI pilot is required before large-scale computation.

Its outputs must demonstrate correctness, memory behavior, throughput, and reproducibility.

### D9 — Test isolation is mandatory

The test partition cannot influence preprocessing tuning, model selection, thresholds, or hyperparameters.

Any exploratory use of test labels must be documented and may invalidate a claim of independent held-out evaluation.

### D10 — Statistical uncertainty respects patient groups

Bootstrap intervals and relevant uncertainty calculations use patient/group resampling when slides are clustered.

### D11 — Historical comparisons are conditional

Legacy and new metrics may be directly compared only when cohort, target, split, and evaluation definitions are compatible.

Otherwise comparisons are descriptive.

### D12 — Scientific conclusions are limited

This retrospective experiment does not establish clinical utility, generalizability, or diagnostic safety.

### D13 — Model and data licensing remain enforced

UNI model restrictions and cohort data permissions must be respected.

Restricted checkpoints, sensitive data, and derived artifacts must not be redistributed without authorization.

### D14 — Governance and repository boundaries remain unchanged

- All engineering governance artifacts reside under `.claude/`.
- Sibling repositories are read-only.
- No legacy source code is copied.
- No Git commits without authorization.

## Consequences

### Benefits

- Traceable real-world labels.
- Leakage-safe evaluation.
- Reproducible cohort experiments.
- Scientifically interpretable uncertainty.
- Reusable infrastructure for future biomarkers and cohorts.

### Trade-offs

- Manual resolution may be required.
- Some slides may be excluded.
- Full extraction may require substantial compute.
- Historical results may not be directly comparable.
- Scientific validation may remain blocked despite completed engineering work.

## Deferred

- External cohort validation.
- Prospective studies.
- Clinical deployment.
- Foundation-model fine-tuning.
- Multi-site generalization.
- Diagnostic claims.

## Decision record (SPEC-008 owner decisions)

The authoritative ledger is `projects/cohorts/surgen-msi/cohort.yaml`. Each approved entry carries its approver, date and reference.

| ID | Decision | Status | Date |
|---|---|---|---|
| D-0 … D-6, D-8 | Phase B label, target, group, split and pilot-partition decisions | Approved | 2026-11-02 |
| D-7 | Extraction scale 0.5 µm/px ± 15 %, `uni-official@1`, 224 × 224 RGB input | **Approved** (repository owner, after the Phase C1 readiness review) | 2026-11-02 |
| D-9 | Phase C2 pilot infrastructure: EBI HTTPS acquisition of the 5 pending pilot WSIs; local Mac with Apple MPS; scope = the 8 slides of `pai-pilot-5da7534b…`; `uni-official@1`, 0.5 µm/px ± 15 %, RGB 224 × 224, fp32, batch 16. **Planning only:** authorizes no download, UNI run, training or Phase C2 execution. Phase D compute/storage not covered. | **Approved** (repository owner, PROMPT-008D9) | 2026-11-02 |
| D-10 | UNI licence coverage and SurGen data permissions | Pending | — |

**Effect of the D-7 approval on manifests:**
- Cohort manifest `surgen-msi@1` is immutable. It keeps its historical ledger, in which D-7 is pending.
- Because a manifest embeds its decision ledger, the cohort configuration now declares version "2".
- Manifest v2 is built only explicitly, into a new directory.
- D-7 gates extraction only. Slide membership, labels and the split do not change: the opt-in rebuild test verifies this.

**Effect of the D-9 approval on manifests:**
- The configuration stays at version "2". Manifest v2 has not been built, so no `surgen-msi@2` document can disagree with the ledger.
- The v1 reproduction reverts both D-7 and D-9 to their pending v1 entries.
- After v2 is built, any further ledger change needs version "3".
- Approving D-9 leaves D-10 as the only pending `extraction`/`pilot` decision. Phase C2 still needs a separate, explicit execution authorization; the preflight gates in `.claude/reports/008-phase-c2-execution-plan.md` fail closed without it.

## Acceptance

Accept this ADR after Phase A confirms the source and governance assumptions.

Any label adjudication, split remediation, or scale change requiring owner authorization must be approved before execution.

SPEC-008 remains Proposed until scientific validation is completed.