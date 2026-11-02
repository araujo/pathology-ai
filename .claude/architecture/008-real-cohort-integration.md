# 008: Real Cohort Integration, Proposed Architecture

**Status:** Proposed (Phase A output). Nothing in this document is implemented. SPEC-008 and ADR-008 remain *Proposed*.
**Date:** 2026-11-02

**Related:**
- `../specs/008-real-cohort-integration-and-scientific-validation.md`
- `../adr/008-real-cohort-validation-architecture.md`
- `../reports/008-cohort-discovery.md`
- `../reports/008-label-source-audit.md`
- `../reports/008-split-integrity-audit.md`
- `007-mil-training-and-classification.md`
- `006-foundation-model-integration.md`

**Design rule (ADR-008 D1):**
- No SurGen, MSI or cohort name appears in `domain`, `ports`, `application`, `infrastructure` or `plugins`.
- Everything SurGen-specific is configuration under `projects/`.

---

## 1. Shape

```text
label source files (read-only, outside the repo)      upstream CSVs pinned by SHA-256
        │
        ▼
CohortSourceSpec (YAML, projects/cohorts/<cohort>/)  ── declares columns, id regex, group template,
        │                                               target + vocabulary per source, policies
        ▼
CohortSourceReader (infrastructure)                    per-row: source, row number, raw value, digest
        │      fails closed on: digest mismatch, unknown column, unparsable id, unknown vocabulary
        ▼
reconcile() (domain, pure)                             candidates per slide → conflict status
        │      never chooses between disagreeing sources; applies only owner-approved resolutions
        ▼
WSI inventory (infrastructure, optional input)         name, size, SHA-256/ETag, reader metadata
        │
        ▼
CohortManifest  pathology-ai.cohort/1  (domain)        records + exclusions + resolutions + split
        │      content-addressed, write-once (reuses SPEC-005/007 write_once + canonical JSON)
        ├──► MIL labels table   ──► SPEC-007 MILDatasetService.build (unchanged)
        ├──► external split     ──► SPEC-007 import_split / validate_split (unchanged, GROUP_LEAKAGE)
        └──► extraction worklist ──► SPEC-006 wsi-patch-embedding-uni (unchanged), Phase C/D
```

Phases E and F reuse SPEC-007 training, evaluation and prediction unchanged.

Two pieces are added later:
- **Patient-resampling bootstrap.** Generic pure code, in Phase F.
- **Validation-derived threshold.** Phase F. It extends the SPEC-007 decision rule, and the rule stays recorded.

## 2. Components to add in Phase B

| Layer | Module (proposed) | Responsibility | Dependencies |
|---|---|---|---|
| domain | `domain/cohort.py` | `SourceRecord`, `LabelCandidate`, `ConflictStatus`, `Resolution` (resolver, authority, rationale, date), `EligibilityStatus` (the eight SPEC-008 §6.3 values), `CohortRecord`, `CohortManifest` (`pathology-ai.cohort/1`, digest), and `reconcile()`. `reconcile()` is pure and deterministic. It raises on unknown statuses and never selects among conflicting values. | stdlib |
| domain | `domain/statistics.py` (Phase F) | Group-resampling bootstrap with an explicit seed, replicate count, unit, interval method, and an undefined-replicate policy and count. Refuses to report an interval below a declared minimum per class. | stdlib |
| application | `application/cohort_service.py` | Orchestrates: read sources → reconcile → join inventory → manifest → derive the MIL label table and external split. Owns no cohort knowledge. | domain, ports |
| ports | `CohortSourceReader`, `WSIInventory` protocols | Read declared sources; list WSIs with integrity evidence | — |
| infrastructure | `infrastructure/cohort_sources.py` | Declarative CSV reader: column map, regex normalisation, vocabulary map, per-file SHA-256 pin. Every error names the file, row and value. | stdlib csv, yaml |
| infrastructure | `infrastructure/wsi_inventory.py` | Local directory inventory (size, SHA-256, streamed). An S3 listing (size + ETag) is optional and read-only, used only if credentials are bound. | stdlib (+ optional boto3, not added without approval) |
| presentation/cli | `pathology-ai-cohort` (`audit`, `reconcile`, `manifest`, `export-mil`) | Thin CLI. Writes manifests and reports to an operator-chosen output directory outside the repository. | — |
| projects | `projects/cohorts/surgen-msi/cohort.yaml` | SurGen definition, configuration only (§3) | — |

**Layering rules.**
- These follow `tests/architecture/test_dependency_rules.py`.
- The new modules are standard-library only in `domain` and `application`, and must pass that existing test unchanged.
- Nothing new imports torch.

## 3. SurGen as configuration (sketch)

The values are those observed in Phase A. The policy fields hold **owner decisions** and stay `pending` until approved. With any field pending, `reconcile` produces `pending_review` and the export to MIL refuses to run.

```yaml
# projects/cohorts/surgen-msi/cohort.yaml  (sketch, not created in Phase A)
cohort_id: surgen-msi
version: "1"
target: {name: msi_dmmr, classes: [negative, positive], positive_class: positive}
identity:
  slide_pattern: '^(?P<cohort>SR386|SR1482)_40X_HE_T(?P<case>\d+)(?:_(?P<slide>\d{2}))?(?:\.czi)?$'
  slide_id: '{cohort}_40X_HE_T{case:03d}_{slide|01}'
  group_id: 'surgen:{cohort}:T{case:03d}'        # D-4
sources:
  - id: upstream-sr386-msi
    files: {train: SR386_msi_train.csv, validation: SR386_msi_validate.csv, test: SR386_msi_test.csv}
    sha256: {train: 634aa4f5…, validation: c63c3638…, test: 33570894…}
    provenance: {repository: CraigMyles/SurGen-Dataset, commit: e56ca8c, path: reproducibility/dataset_csv}
    columns: {id: case_id, label: label}       # slide_id column is ignored (positional index)
    source_target: MMR_LOSS                     # D-3: recorded, not renamed
    vocabulary: {"0": negative, "1": positive}
  - id: upstream-sr1482-msi
    # … same shape, source_target: MMR_MSI
  - id: upstream-surgen-msi                     # redundant copy; must agree, else conflict
  - id: legacy-manifest                         # legacy derivative; must agree, else conflict
policies:
  label_authority: pending                      # D-1
  secondary_slides: pending                     # D-2: exclude | inherit_case_partition
  combined_target: pending                      # D-3
  split: {kind: imported, from_sources: true, test_previously_used: true}   # D-5
```

**Paths.** Source files are bound at run time with `--source-root`, in the same way checkpoints are bound in SPEC-006. Paths never enter digests. File digests do.

## 4. Manifest contract `pathology-ai.cohort/1`

| Field | Content |
|---|---|
| `cohort_id`, `version`, `target` | From configuration |
| `sources[]` | id, file name, SHA-256, row count, provenance (repository/commit or "operator-supplied") |
| `policies` | The resolved policy values and their approval references |
| `records[]` | slide id, group id (pseudonymous study code), WSI identity (name, size, SHA-256 or `null` + reason), label (normalised), source target, `label_provenance[]` (source, row, raw value), original split, eligibility status, exclusion reason, resolution history |
| `counts` | By status × original split × class (the SPEC-008 inclusion/exclusion flow) |
| `digest` | SHA-256 of the canonical JSON without the digest. Write-once. A changed membership produces a new version and never overwrites. |

**Privacy.** Records carry only study codes (`SR386_40X_HE_T001_01`). No dates, names or clinical fields are imported.

## 5. Validation gates (fail closed)

| Gate | Error | Covers |
|---|---|---|
| A source file digest differs from the pin | new `COHORT_SOURCE_INTEGRITY` (no existing code fits) | AC-002, AC-018 |
| An unparsable id, a cross-cohort-ambiguous key, or an unpadded id that cannot be normalised | `INVALID_LABEL` | Label audit §6 |
| Sources disagree on label or split for one slide | Record → `excluded_conflict`; never resolved without a `Resolution` | AC-003 |
| A conflict carries an unapproved `Resolution` (no authority/rationale) | `INVALID_LABEL` | AC-003 |
| A supervised record has no group | `MISSING_GROUP` (SPEC-007) | AC-004 |
| One group appears in two partitions | `GROUP_LEAKAGE` (SPEC-007) | AC-006 |
| A second slide's partition differs from its case | `GROUP_LEAKAGE` | Split audit L2 |
| Two slide ids share one WSI digest | `excluded_duplicate` for both + report | AC-008 |
| Export with any `pending` policy | `INVALID_CONFIGURATION` (existing `ConfigurationError`) | ADR-008 acceptance |
| A legacy `slide_id` (`slide_<i>`) used as a key | Not representable: the reader never reads it | Split audit L3 |

## 6. Reuse of existing contracts (unchanged)

| Need | Existing component |
|---|---|
| Labels, groups, statuses for MIL | `SlideLabel`, `LabelStatus`, `infrastructure/mil_config.load_labels` (the cohort service writes its label table) |
| Split import and leakage validation | `MILDatasetService.import_split`, `validate_split` |
| Embedding compatibility | SPEC-007 compatibility key. It refuses mixed identities, so any imported legacy bags (D-8) can never mix with `uni-official@1` bags. |
| Extraction | `wsi-patch-embedding-uni@1.0.0` with the approved scale binding (D-7) |
| Write-once JSON + canonical digest | `infrastructure/canonical.py`, SPEC-005/007 write-once helpers |

## 7. Later-phase design notes (for approval, not built)

**Pilot (Phase C):**
- About 8–12 train/validate slides, including MSI-positive ones and both sub-cohorts, spanning small (0.26 GB) to large (6 GB) WSIs.
- Selection is deterministic: `sha256(seed, slide_id)` rank within strata.
- Measures: wall time, patches per second, peak RSS/device memory, store digest stability on repeat, and resume after interruption.

**Cohort embedding (Phase D):**
- A per-slide work queue driven by the manifest: fetch → verify WSI digest → embed → verify store → optionally delete the WSI.
- It is resumable and never overwrites. A failure moves the record to `excluded_quality` with the reason, and the manifest records a new version.

**Statistics (Phase F):**
- Case-level bootstrap. Case = slide while `_02` slides are excluded.
- Seed, B = 2,000, percentile interval, and the count of undefined replicates are reported.
- The threshold is chosen on validation by a pre-declared rule (e.g. Youden J or a fixed sensitivity), then frozen before test.

**Optional legacy-reference arm (D-8):**
- A read-only zarr bag importer in a separate plugin with an optional `zarr` extra.
- Its identity is `uni-legacy-surgen@1/imported`, with per-store digests.
- It is used only for a separately labelled comparability analysis.

## 8. Open questions

These are restated from the reports and must be answered before Phase B is closed: D-0 to D-10 (`../reports/008-cohort-discovery.md` §10).
