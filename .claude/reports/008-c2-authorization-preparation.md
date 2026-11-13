# Report: SPEC-008 Phase C2, Authorization Preparation

**Date:** 2026-11-02
**Prompt:** PROMPT-008C2-AUTH
**Outcome:** D-10 approved and recorded. Binding generated. **Phase C2 execution is NOT authorized yet**: the owner's bound ledger entry (§4) is still to be recorded. Nothing was downloaded and UNI was not run.

---

## 1. D-10

Recorded in `projects/cohorts/surgen-msi/cohort.yaml` as approved (gates `pilot`, `extraction`), citing the owner's confirmation and `008-d10-license-evidence.md`.
- UNI: checkpoint obtained through the owner's own individually approved Hugging Face access, registered with an institutional email. No account name, credential or email address is recorded.
- Use: non-commercial academic research, covering the SurGen MSI/dMMR experiment. The owner is the only operator.
- Non-distribution: UNI weights, UNI embeddings and UNI-derived MIL weights never leave the owner's control. Results are published as aggregates only.
- SurGen: the 5 approved WSIs may be downloaded by EBI HTTPS when Phase C2 requires it. Attribution obligations recorded.
- First-download provenance accepted, with its limits stated in the ledger and in the plan §2.2: a locally computed SHA-256 proves the bytes did not change after the pin, not that they equal what the authors deposited.
- Still OPEN: the institutional ethics position (evidence S-e). It blocks publication, not the pilot.

D-10 records licence compliance only. Its summary does not contain the C2 authorization phrase, and a test checks that.

## 2. What did not change

- Manifest v1, its split and reconciliation files: byte-identical (G1). It still records D-10 as pending.
- The opt-in rebuild from the pinned upstream and legacy sources reproduces v1 (digest `854f009f…5a3c`) after reverting D-7, D-9 and D-10 to pending.
- Pilot selection `pai-pilot-5da7534b…`, its inventory and `pilot.yaml`: byte-identical (G2).
- Cohort membership, labels and the split: unchanged.
- No sibling repository was modified.

## 3. Checks

| Check | Result |
|---|---|
| `ruff check .`, `ruff format --check .` | clean |
| `mypy` (project config), `mypy --strict` on both tools and their tests | clean |
| `pytest` | 719 passed, 53 skipped (opt-in real-WSI, UNI, CUDA and rebuild tests) |
| Opt-in v1 rebuild (`test_surgen_cohort_manifest.py` with both roots bound) | 3 passed |

## 4. Binding

`projects/cohorts/surgen-msi/c2-pilot/c2-binding.json`, SHA-256:

```
14fd2818ceec5c424f17700473d07c2c713567066ab866ceb72ff377bf0d3039
```

- Regenerating it from the tree gives identical bytes. Every file hash in it was re-checked with `shasum`.
- A simulated authorization naming this digest, in a scratch copy of the repository, made G6 PASS. Appending one byte to the plan in that copy made G6 FAIL.

Ledger entry for the owner to add after review (the id is the owner's choice):

```yaml
  - id: <owner-chosen id>
    summary: >-
      Phase C2 execution authorized for pilot selection pai-pilot-5da7534b on manifest
      surgen-msi@1: EBI download of the 5 pending WSIs and UNI extraction per D-7/D-9,
      exactly as bound by c2-binding sha256 14fd2818ceec5c424f17700473d07c2c713567066ab866ceb72ff377bf0d3039.
    status: approved
    gates: [pilot, extraction]
    approver: Gustavo Medeiros de Araujo (repository owner)
    approved_on: "YYYY-MM-DD"
    reference: PROMPT-008C2-AUTH review
```

`cohort.yaml` is not bound, so adding this entry keeps the binding valid. The test `test_configuration_differs_from_v1_only_by_the_recorded_approvals` must be updated in the same change, because v1 must still reproduce by reverting only D-7, D-9 and D-10.

## 5. Preflight on this host (download stage, read-only)

| Gate | Result |
|---|---|
| G1–G5, G7–G11, G13 | PASS. Free disk 59.5 GB; planned writes 26.0 GB; reserve 10 GB. The 3 legacy sources and the UNI checkpoint match their pins. MPS available (torch 2.14.1). |
| G6 | FAIL: no Phase C2 authorization yet (expected) |
| G12 | FAIL: running on battery. Connect AC power before any step. |

`~/pathology-ai-data` does not exist yet. The plan §0.2 creates it.
