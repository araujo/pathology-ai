# ADR 001 — Independent, Modular, Modality-Extensible Architecture

**Status:** Accepted (approved 2026-10-15)  
**Date:** 2026-10-15  
**Decision owners:** Pathology AI maintainers  
**Related SPEC:** `../specs/001-platform-foundations.md`

## Context

The existing `digital-pathology` research implementation supports a specific SurGen/KRAS WSI workflow (CZI, patch extraction, UNI embeddings, MIL and prediction). Its architecture assessment recommended an independent greenfield platform to prevent vendor, model-family and task assumptions from becoming global constraints.

The new platform must enable students to reuse pipelines for different diseases and datasets; support alternative models (segmentation, detection, direct-pixel, 3D); eventually ingest CT/MRI; and publish image and analysis data to DPA without changing DPA now.

## Decision

1. **Independent greenfield repository `pathology-ai`.** No runtime/code/storage dependency on DP or DPA. DP is a read-only scientific reference and future parity oracle.

2. **Modular monolith with inward dependencies.** Pure domain contracts and ports; application services; infrastructure/reader/model plugins; separate public API adapters.

3. **Composable typed pipelines.** A pipeline is a versioned DAG of stages with declared input/output kinds. WSI patch→embedding→MIL is one reusable template, not the mandatory execution path.

4. **Project/dataset/task/model separation.** Disease paths are organizational namespaces; datasets and trained artifacts are bound by configuration. Model applicability and preprocessing compatibility are verified, not inferred from a disease folder.

5. **Modality-aware imaging abstractions.** WSI pyramids and 3D/4D volumes have distinct spatial contracts. Readers are plugins; file extensions alone do not select clinical modality.

6. **Model-bound preprocessing and evidence semantics.** Color/normalization/geometry assumptions are versioned with model artifacts. Attention weights cannot be promoted to tumor localization or probabilities without validation.

7. **Versioned Imaging API and Analysis API.** They are independent of the DPA implementation. A future v1 compatibility adapter must pass contract and shadow tests; a native richer schema can represent 2D and 3D. No DPA changes in this ADR's implementation phase.

8. **Scientific validation as a first-class capability.** A model is usable only under its declared modality/task/preprocessing/validation constraints; unvalidated research use is explicit and distinguished from validated use.

9. **Content and execution provenance.** Include dataset, code, environment, reader/plugin, preprocessing, model and geometry versions. Do not silently reuse incompatible artifacts.

10. **Incremental delivery.** Foundations first; actual readers, real ML, parity and DPA integration in separate approved increments.

## Rejected alternatives

- **Extend the DP codebase in place:** fastest apparent reuse but perpetuates legacy CZI/UNI/MIL coupling and risks research reproducibility.
- **Extract a shared runtime core from DP:** creates dependency and migration complexity before generic boundaries are proven.
- **One universal pipeline for WSI, CT and MRI:** ignores 2D vs 3D semantics, modality-specific preprocessing and model requirements.
- **Microservices from day one:** adds deployment, transaction and distributed debugging costs without established scaling needs.
- **Bind the core to DPA response schemas:** couples scientific domain models to a current viewer and prevents richer modalities.

## Consequences

**Benefits:** independent lifecycle; reusable disease-agnostic workflows; new models/readers through ports; DPA-ready APIs; testable scientific provenance.

**Costs:** duplicate some infrastructure patterns; up-front contract design; real scientific parity requires dedicated later validation; new formats require reader-specific testing and privacy/security controls.

**Risks:** abstraction overreach, misleading “supports all models” claims, silent coordinate mistakes, and false scientific generalization.

Mitigate through minimal executable slices, explicit capability states, negative contract tests and separate parity gates.

## Non-decisions

No guarantee that one model transfers across cancers; no clinical validation; no CT/MRI implementation or viewer; no DPA cutover; no storage/vendor/cloud selection; no migration of DP legacy artifacts.

These require subsequent ADRs/SPECs and evidence.

## Validation

Architecture/import tests; typed pipeline compatibility tests; synthetic WSI workflow; 2D/3D API fixture checks; unlabeled ingestion contract; negative tests for mismatched modality, preprocessing, geometry and unsupported models.

Later gates: DP scientific parity, DPA v1 contract compatibility, shadow validation and explicitly approved cutover.