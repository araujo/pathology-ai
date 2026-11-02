# SPEC-008 — Real Cohort Integration & Scientific Validation

**Status:** Proposed  
**Priority:** P0  
**Dependencies:** SPEC-001 through SPEC-007  
**Initial study:** SurGen — MSI prediction from H&E whole-slide images  
**Repository:** `pathology-ai`

## 1. Context

SPEC-007 established a generic MIL training, evaluation, and prediction framework.

Its implementation was validated on synthetic labeled cohorts and on unlabeled real UNI embedding bags.

SPEC-008 introduces the first supervised experiment using a real pathology cohort.

The initial study is MSI classification using SurGen WSIs.

Historical SurGen experiments provide a reference, but their data definitions, label files, patient identifiers, split assignments, and preprocessing configurations must be independently audited before reuse.

The SPEC-007 report identified contradictory legacy label CSVs. This is a blocking data-governance concern, not a discrepancy that may be resolved by choosing whichever source produces better model performance.

## 2. Objectives

1. Inventory the real SurGen cohort and metadata.
2. Establish authoritative label and patient identity sources.
3. Detect and resolve label conflicts through an auditable process.
4. Preserve and validate existing train/validation/test assignments where possible.
5. Prevent patient-level and duplicate-slide leakage.
6. Produce a versioned curated cohort manifest.
7. Generate real UNI embeddings for the eligible cohort.
8. Train mean-pooling and attention-MIL models on real labels.
9. Evaluate generalization on a held-out test partition.
10. Calculate uncertainty intervals and compare with the historical baseline.
11. Produce reproducible scientific reports and artifacts.
12. Keep cohort-specific logic outside the generic framework core.

## 3. Scope

### In scope

- Read-only cohort discovery.
- WSI inventory and checksum/provenance capture.
- Metadata and label reconciliation.
- Patient/group identity validation.
- Split audit and preservation.
- Eligibility and exclusion records.
- Real WSI embedding orchestration.
- Resource and cost estimation before extraction.
- Mean-pooling and attention-MIL training.
- Validation-only model selection.
- Held-out test evaluation.
- Bootstrap confidence intervals.
- Class imbalance analysis.
- Historical baseline comparison.
- Scientific reproducibility documentation.
- Cohort-specific experiment configuration.

### Out of scope

- Clinical deployment.
- Clinical diagnostic claims.
- Foundation-model fine-tuning.
- New foundation-model architectures.
- New MIL architectures.
- Hyperparameter optimization over the test set.
- Prospective clinical validation.
- External cohort validation.
- Viewer heatmap integration.
- Automatic correction of contradictory clinical labels.
- Redistribution of restricted model weights or patient data.

## 4. Research Question

Can a MIL classifier trained on UNI embeddings extracted from SurGen H&E WSIs distinguish MSI-positive from MSI-negative slides on an independently held-out test partition?

### Primary endpoint

Slide-level AUROC for MSI classification.

### Secondary endpoints

- AUPRC.
- Sensitivity.
- Specificity.
- Precision.
- F1.
- Balanced accuracy.
- Confusion matrix.
- Prediction score distributions.
- 95% confidence intervals for selected metrics.

### Reference

Historical SurGen UNI experiments reported AUROC around 0.81.

This value is a historical reference, not a validated performance target for the new pipeline.

Any comparison must account for differences in cohort, labels, splits, patch extraction, preprocessing, model architecture, and evaluation.

## 5. Cohort Discovery

Before modifying data or executing expensive workloads, inspect:

- WSI storage locations and file formats.
- Number of available WSIs.
- WSI-to-slide mapping.
- Slide-to-patient mapping.
- Existing metadata CSVs.
- Existing label CSVs.
- Existing split files.
- Duplicate files or identifiers.
- Missing and ambiguous identifiers.
- Historical preprocessing configurations.
- Historical MSI target definitions.
- Patient grouping semantics.

Generate an inventory with source paths, record counts, identifiers, checksums where practical, and unresolved issues.

All discovery is read-only.

## 6. Label Reconciliation

### 6.1 Authoritative label policy

Each MSI label must have a traceable source.

When multiple sources disagree, the system must record the conflict and exclude the affected record until an authorized owner approves a resolution.

Never silently prefer:

- A newer file.
- A larger file.
- The majority label.
- A historical training label.
- A label that improves model metrics.

### 6.2 Reconciliation record

Each reconciliation record includes:

- Slide ID.
- Patient/group ID.
- Source file and source row.
- Original label value.
- Normalized candidate label.
- Conflict status.
- Resolution status.
- Approved resolution, if any.
- Resolver identity or authority reference.
- Resolution rationale.
- Source version and digest.

### 6.3 Eligibility

Possible statuses:

- `eligible`
- `excluded_conflict`
- `excluded_missing_label`
- `excluded_missing_group`
- `excluded_duplicate`
- `excluded_missing_wsi`
- `excluded_quality`
- `pending_review`

Exclusions must be counted and reported by reason and original split.

A record may become eligible only after satisfying all required validation gates.

## 7. Cohort Manifest

Introduce a cohort-specific, versioned manifest:

`pathology-ai.cohort/1`

Each eligible record references:

- WSI source identity.
- Slide ID.
- Patient/group ID.
- Curated MSI label.
- Label provenance.
- Existing split assignment, if available.
- WSI metadata and integrity evidence.
- Eligibility status.
- Exclusion or resolution history.

The manifest must be immutable and content-addressed.

Patient identifiers must be pseudonymized where necessary, with no unnecessary personally identifiable information in reports.

## 8. Split Integrity

Prefer preserving the existing SurGen split assignments when they are valid.

Validate:

- No patient/group overlaps between train, validation, and test.
- No duplicated WSI content across partitions.
- No contradictory slide assignments.
- No unlabeled eligible training records.
- No hidden resplitting.

If the historical split violates group isolation, do not silently repair it.

Produce a proposed remediation plan and request approval before creating a replacement split.

All split changes require a new manifest version and must be documented.

## 9. Embedding Protocol

Use the existing SPEC-006 `wsi-patch-embedding-uni` pipeline.

The extraction configuration must explicitly declare:

- UNI checkpoint and checksum.
- Preprocessing profile.
- Extraction spacing and tolerance.
- Patch size.
- Tissue detection and selection.
- Channel transformations.
- Batch size.
- Device and precision.
- Patch selection policy.
- Embedding store location.
- Resource limits.

### Critical requirements

- Do not silently reproduce legacy preprocessing.
- Do not mix incompatible embedding identities.
- Do not use unapproved scale assumptions.
- Do not download model weights.
- Do not overwrite existing stores.
- Preserve safe resume and SHA-256 verification.
- Keep model license restrictions visible in artifacts.

Before cohort-wide execution, run a small representative pilot with approved resources.

**Decision status:** the SurGen extraction scale was approved on 2026-11-02 (D-7: 0.5 µm/px ± 15 %, `uni-official@1`, 224 × 224 RGB input). See ADR-008 D7 and its decision record. D-9 was approved on 2026-11-02 for the Phase C2 pilot only (EBI HTTPS acquisition, local Mac with MPS, fp32, batch 16), as a planning decision that authorizes no download or execution. Licensing/data permissions (D-10) remain pending, and Phase C2 execution needs an explicit owner authorization.

The pilot must verify memory, throughput, output integrity, and cost assumptions.

## 10. Training Protocol

Train at least:

1. Mean-pooling MIL baseline.
2. Attention MIL.

Both models must consume the same curated dataset, split, and embedding compatibility policy.

Training configuration must record:

- Architecture and hyperparameters.
- Random seeds.
- Optimizer and learning rate.
- Class imbalance handling.
- Epoch budget.
- Early stopping.
- Patch subsampling policy.
- Batch size.
- Runtime versions.
- Selection metric.

Model selection uses validation data only.

The test partition must remain untouched until the model and evaluation protocol are frozen.

## 11. Statistical Evaluation

### Primary analysis

Evaluate AUROC on the held-out test set.

Report:

- Point estimate.
- 95% confidence interval.
- Number of slides and patients.
- Number of positive and negative cases.
- Number of excluded records.
- Whether the estimate is slide-level or patient-level.

### Uncertainty

Use a reproducible bootstrap procedure appropriate to the patient/group structure.

Resample patients/groups, not individual slides independently, when patients contribute multiple slides.

Record:

- Bootstrap seed.
- Number of replicates.
- Resampling unit.
- Interval method.
- Undefined replicate handling.

Do not calculate a misleading interval if the test cohort is too small or lacks sufficient positive/negative groups.

### Secondary metrics

Report AUPRC, sensitivity, specificity, precision, recall, F1, balanced accuracy, and confusion matrix.

Threshold selection must use validation data only.

### Baseline comparison

Compare against historical SurGen results only when the underlying task and evaluation definitions are sufficiently comparable.

Otherwise provide a descriptive comparison with explicit caveats.

## 12. Scientific Reporting

Produce a structured experiment report with:

1. Research question and cohort description.
2. Label provenance and reconciliation.
3. Inclusion and exclusion flow.
4. Patient-level split validation.
5. WSI and patch statistics.
6. Embedding model and preprocessing.
7. MIL model configurations.
8. Training and validation behavior.
9. Held-out test results.
10. Confidence intervals.
11. Class imbalance analysis.
12. Baseline comparability.
13. Limitations and potential biases.
14. Reproducibility information.
15. Unresolved scientific questions.

No claim of clinical utility may be made solely from this retrospective experiment.

## 13. Acceptance Criteria

**AC-001 — Cohort inventory**

WSI and metadata sources are inventoried with counts and provenance.

**AC-002 — Label source audit**

All candidate label sources and contradictory records are identified.

**AC-003 — Conflict-safe reconciliation**

Unresolved conflicting labels cannot enter supervised training.

**AC-004 — Patient identity integrity**

Required grouping identifiers are validated and duplicate/group conflicts are detected.

**AC-005 — Cohort manifest**

A versioned curated cohort manifest is generated with eligibility and exclusion reasons.

**AC-006 — Split audit**

Existing splits are checked for leakage and contradictions without silent modification.

**AC-007 — Split provenance**

Approved split membership is immutable, versioned, and reproducible.

**AC-008 — WSI integrity**

Missing, duplicate, corrupt, or unreadable WSIs are identified before cohort-wide embedding.

**AC-009 — Embedding pilot**

A representative real-WSI pilot validates the extraction configuration and resource assumptions.

**AC-010 — Real UNI embeddings**

All eligible, approved slides have verified compatible UNI embedding bags.

**AC-011 — Embedding compatibility**

Mixed or incompatible embedding configurations are rejected.

**AC-012 — Real binary training**

Mean pooling and attention MIL train on curated real MSI labels.

**AC-013 — Validation-only selection**

Hyperparameters, checkpoint selection, and classification thresholds are determined without test-set influence.

**AC-014 — Held-out test evaluation**

The frozen models are evaluated on the approved held-out test partition.

**AC-015 — Statistical uncertainty**

Appropriate 95% confidence intervals are reported or their absence is justified.

**AC-016 — Class imbalance**

Positive/negative prevalence and relevant metric limitations are documented.

**AC-017 — Historical comparison**

Comparison with legacy SurGen performance is qualified by documented methodological differences.

**AC-018 — Reproducibility**

Cohort, split, embedding, model, and prediction artifacts have stable provenance and reproducibility evidence.

**AC-019 — Regression preservation**

Existing SPEC-001 through SPEC-007 behavior remains intact.

**AC-020 — Scientific reporting and governance**

Reports are complete, data restrictions are respected, and unresolved scientific questions remain explicit.

## 14. Execution Phases

### Phase A — Read-only discovery

Inventory data, audit labels and splits, inspect historical results, and identify blockers.

### Phase B — Cohort reconciliation

Create conflict reports and a proposed curated cohort.

Stop for owner approval if clinical label resolution or split changes are required.

### Phase C — Embedding pilot

Validate the real UNI extraction configuration on a small approved sample.

### Phase D — Cohort embedding

Extract and verify embeddings for all approved eligible slides.

### Phase E — Real MIL training

Train mean-pooling and attention MIL using frozen curated labels and splits.

### Phase F — Statistical evaluation

Evaluate on the held-out test partition, compute uncertainty, and compare with historical results.

### Phase G — Scientific report

Produce the final scientific and technical evidence report.

## 15. Deliverables

- `.claude/architecture/008-real-cohort-integration.md`
- `.claude/reports/008-cohort-discovery.md`
- `.claude/reports/008-label-reconciliation.md`
- `.claude/reports/008-embedding-pilot.md`
- `.claude/reports/008-scientific-validation.md`
- Cohort-specific manifests and experiment configurations under `projects/`.
- Reusable cohort validation components only where justified.

## 16. Definition of Done

SPEC-008 is complete only when:

- All 20 acceptance criteria have supporting evidence.
- Required label and split decisions are approved.
- Real embeddings are generated for the eligible cohort.
- Real MIL models are trained and evaluated.
- The scientific report includes appropriate uncertainty and limitations.

If the cohort is blocked by unresolved labels, permissions, missing files, or insufficient compute, the implementation may be technically complete but SPEC-008 must remain **BLOCKED**, not PASS.