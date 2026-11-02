# Report: SPEC-008 Phase B, Implementation

**Date:** 2026-11-02
**Prompt:** `.claude/prompts/008B-cohort-reconciliation-and-manifest-construction.md`
**Companion:** `008-label-reconciliation.md` (sources, counts, split, manifest digests)

**Outcome:**

| Item | Result |
|---|---|
| Phase B engineering | Complete |
| Cohort manifest `surgen-msi@1` | Written. 820 provisionally eligible, 200 excluded. |
| Owner decisions D-0 to D-6, D-8 | Approved 2026-11-02 and recorded |
| Scientific validation (SPEC-008) | **BLOCKED** (§6) |
| Phase C | **Not started.** It waits for owner approval. |

Constraints honoured:
- No WSI was read, extracted or downloaded.
- No model was trained or run.
- No label was adjudicated.
- No split was generated.
- Sibling repositories were not modified.
- No Git commit was made.

---

## 1. What was built

| Layer | Module | Responsibility |
|---|---|---|
| domain | `domain/cohort.py` | `pathology-ai.cohort/1` contracts: `Decision`, `CohortTarget`, `SlidePolicy`, `SplitPolicy`, `CohortDefinition`, `SourceFile`, `RosterEntry`, `LabelCandidate`, `WSIEvidence`, `Resolution`, `CohortRecord`, `CohortSplit`, `StatusCount`, `CohortManifest`; pure `reconcile()` / `build_manifest()`. Standard library only. |
| domain | `domain/errors.py` | `COHORT_INVALID`, `COHORT_SOURCE_INTEGRITY`, `COHORT_SOURCE_FORMAT`, `COHORT_RECONCILIATION` |
| ports | `CohortSourceReader` | `read() -> CohortObservations` |
| application | `application/cohort_service.py` | `CohortService.reconcile`; SPEC-007 hand-over: `slide_labels`, `split_assignment`, `import_external_split` |
| infrastructure | `infrastructure/cohort_sources.py` | YAML config (`pathology-ai.cohort-config/1`), identifier grammar, digest-pinned CSV reader, write-once outputs (manifest JSON, SPEC-007 `split.csv`, `reconciliation.csv`) |
| presentation | `cli.py` → `pathology-ai-cohort reconcile|verify` | Thin CLI that prints JSON summaries |
| projects | `projects/cohorts/surgen-msi/cohort.yaml` | Everything SurGen-specific: sources, pins, grammar, target, decisions, limitations |
| projects | `projects/cohorts/surgen-msi/manifests/v1/` | `cohort-manifest.json`, `split.csv`, `reconciliation.csv` |

**Generic core (ADR-008 D1):**
- A new architecture test forbids cohort, target and assay names in the three generic modules (`SR386`, `SR1482`, `MMR`, `dMMR`, `MSS`, plus the existing MIL list).
- Existing SPEC-001 to SPEC-007 code is unchanged.
- Additions only: error classes, one port, one CLI entry point and a README section.

## 2. Reconciliation rules (fail closed)

| Rule | Enforcement |
|---|---|
| A source file differs from its pin | `COHORT_SOURCE_INTEGRITY` before any row is parsed |
| An identifier does not match the grammar (e.g. positional `slide_<i>`) | `COHORT_SOURCE_FORMAT` with file and line |
| Two spellings of one id in a file (`T1` and `T001`) | `COHORT_SOURCE_FORMAT` ("repeats") |
| A value outside the declared vocabulary | `COHORT_SOURCE_FORMAT` with file, line and value |
| A source target not accepted by the cohort target | `COHORT_RECONCILIATION` |
| A label for a slide that is not in the roster | `COHORT_RECONCILIATION` |
| A case-level label on a case with more than one roster slide | `COHORT_RECONCILIATION`. Inheritance is a decision, never implied. |
| Sources disagree on label or partition, including within one source | `excluded_conflict`, unless an approved `Resolution` (resolver, authority, rationale, date) exists |
| Only corroborating sources label a slide | `pending_review` |
| Any decision gating `reconciliation` is pending | Every would-be-eligible slide becomes `pending_review`; the manifest refuses eligible records |
| A group spans sub-cohorts (unqualified key) | `COHORT_RECONCILIATION` |
| A group's slides span partitions | `GROUP_LEAKAGE`. The split is never repaired. |
| Group or slide template omits the cohort or slide field | Configuration error at load |
| A split strategy other than `external@1` | Refused: cohort splits are imported, never generated |
| WSI evidence: missing / unreadable / shared digest | `excluded_missing_wsi` / `excluded_quality` / `excluded_duplicate` |
| WSI not verified | Eligible but **provisional** (`outstanding_gates: wsi_integrity, wsi_duplicate_content`). SPEC-007 hand-over refuses these unless they are accepted explicitly. |
| An edited manifest (counts, split, records or digest) | Refused on construction or read: counts and split re-derive from the records |
| A rewrite with different content | `EXPERIMENT_CONFLICT`. A new version needs a new directory. |

**SPEC-007 hand-over.** `import_external_split` requires the MIL dataset's supervised slides to be exactly the eligible slides, with the same labels and groups. It then calls the unchanged SPEC-007 `import_split` (`external@1`, `GROUP_LEAKAGE`).

## 3. Validation results

| Check | Result |
|---|---|
| Full test suite | **633 passed, 52 skipped**. The skips are the pre-existing opt-in real-WSI, UNI and CUDA tests, which need images or GPU that Phase B forbids. |
| New tests | 47: 21 unit (domain), 17 contract (CSV sources), 4 integration (SPEC-007 hand-over on synthetic bags), 2 SurGen regression, 3 architecture |
| Live rebuild from pinned sources (opt-in, run) | Identical manifest digest `854f009f…5a3c` |
| `ruff check`, `ruff format --check` | clean |
| `mypy --strict` (144 files) | clean |
| Sibling repositories | `git status` unchanged (4 and 5 lines, as in Phase A) |

**Synthetic hazards covered.** The fixtures use invented identifiers and no real labels:
- cross-cohort case-number collisions (qualified vs unqualified keys)
- repeated positional `slide_<i>` columns
- unpadded ids
- a cross-target file substituted for a label file (pin) or declared under another target
- conflicting copies (label, partition, both, within one source)
- approved and invalid resolutions
- corroboration-only labels
- `_02` slides: excluded by policy, case-label inheritance refused, leakage when included in another partition
- pending decisions
- missing and duplicate WSI evidence
- tampered and rewritten outputs
- partial or relabelled MIL datasets at hand-over

## 4. Contract validation on the real cohort

All values are VERIFIED, with details in the companion report.

| Measure | Value |
|---|---|
| Records | 1,020 (843 groups) |
| Eligible | 820 (provisional), with 3 agreeing candidates each |
| Excluded | 23 `no_label` + 177 `_02` by policy |
| Conflicts | 0 |
| Pending review | 0 |
| Split | train / validation / test = 490 / 165 / 165 slides, the same number of groups, positives 50 / 14 / 15 |
| Group overlap | 0 |
| Split assignment digest | `0c8c94fc…5616` |

## 5. Not verified

- **WSI availability, content integrity and duplicate content.** No inventory was run: 16 of 820 eligible WSIs are local, and S3 was BLOCKED in Phase A. All 820 gates remain open (AC-008).
- **Label correctness beyond the publication** (clinical tables unavailable), and the assay behind each label.
- **How the upstream split was generated.**
- **SurGen data terms (D-10).** The manifest records publicly published study codes and labels. Confirm the terms before committing or redistributing `manifests/v1/`.

## 6. Remaining blockers for Phase C

| # | Blocker | Needed |
|---|---|---|
| C1 | WSI access for the pilot slides: S3 SSO expired; EBI FTP untested | Re-authenticate, or approve an EBI download plan (D-9) |
| C2 | No MSI-positive WSI is local, so a representative pilot needs downloads | Pilot selection from train/validation (D-6), then download |
| C3 | Extraction scale is undecided | D-7 |
| C4 | UNI licence coverage and SurGen data terms | D-10 (gates the pilot) |
| C5 | Compute and storage: no CUDA, 58 GiB free | D-9 |
| C6 | WSI inventory to close `wsi_integrity` / `wsi_duplicate_content`, producing manifest v2 | Per-WSI size and SHA-256 (local or S3 ETag + hash). The contract exists; the inventory adapter is not built. |

Each item maps to a SPEC-008 acceptance criterion:

| Criterion | Status |
|---|---|
| AC-001, 002, 003, 004, 005, 006, 007 | Evidence provided |
| AC-008 | **Open** (C6) |
| AC-009 onward | Not started |

**STOP.** Phase B is complete. Phase C waits for the owner's approval.
