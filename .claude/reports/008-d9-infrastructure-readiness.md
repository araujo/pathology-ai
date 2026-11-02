# Report: SPEC-008 D-9, Pilot Infrastructure Readiness

**Date:** 2026-11-02
**Prompt:** PROMPT-008D9 (Pilot Infrastructure Preparation)
**Phase:** C2 infrastructure readiness. Planning only.
**Companions:**
- `008-phase-c2-execution-plan.md`: preflight gates, acquisition procedure and the exact pilot commands
- `008-phase-c1-readiness.md`: scale evidence, extraction procedure, acceptance criteria A1–A11
- `008-pilot-selection.md`: the 8-slide selection `pai-pilot-5da7534b…`

**Outcome:**

| Item | Result |
|---|---|
| D-9 recorded | **Done.** Approved 2026-11-02 by the repository owner, planning-only scope, in the ledger (`cohort.yaml`), ADR-008, SPEC-008 §9 and README |
| Manifest v1 and the split | **Unchanged.** Files byte-identical; v1 still rebuilds to `854f009f…5a3c` with D-7 and D-9 reverted |
| Directory plan | Proposed (§2). Nothing was created. |
| Mac readiness | **Ready, with one operational fix:** the Mac is on battery and sleeps after 1 min (§3) |
| Acquisition and extraction procedures | Prepared, syntax- and lint-checked, **not executed** (execution plan §2–§3) |
| Preflight gates | Prepared and run read-only. **NO-GO**: D-10, Phase C2 authorization and AC power fail (§5) |
| **Phase C2** | **Not started.** Waits for D-10, an explicit C2 authorization and owner review of this report |

**Constraints honoured:**
- No WSI was downloaded, and no HTTP request was sent.
- UNI was not loaded or run. No tensor was created on MPS: only the `torch.backends.mps` capability flags were read.
- No MIL training was run.
- No WSI file was modified. Every WSI read was a read-only SHA-256.
- Manifest v1 and the split are unchanged (`0c8c94fc…5616`).
- No directory was created outside the repository. Sibling repositories were only read.
- No Git commit or staging.

Status legend:
- **MEASURED**: checked on this host in this task
- **PRIOR**: measured in an earlier phase, with its source
- **ESTIMATE**: extrapolated; the assumption is stated
- **NOT VERIFIED**: not checked

---

## 1. D-9 record

| Field | Value |
|---|---|
| Decision | D-9: Phase C2 pilot infrastructure |
| WSI acquisition route | EBI HTTPS (BioStudies S-BIAD1285, `https://ftp.ebi.ac.uk/…`) |
| Compute | Local Mac (Apple M4 Pro, 24 GiB) |
| Accelerator | Apple MPS (`--override embed.device=mps`) |
| Scope | The 8 slides of selection `pai-pilot-5da7534b…`: 3 already verified locally, 5 to download |
| Embedding protocol | `uni-official@1`, 0.5 µm/px ± 15 % (D-7), RGB 224 × 224, fp32, batch size 16 |
| Approver / date | Gustavo Medeiros de Araujo (repository owner) / 2026-11-02 |
| Reference | PROMPT-008D9 |
| Gates | `extraction` (unchanged from the pending entry) |
| **Not authorized** | Downloading WSIs, running UNI, MIL training, starting Phase C2 |
| Not covered | Phase D compute and storage (blocker B6 of C1 remains) |

### 1.1 Where it is recorded

| File | Change |
|---|---|
| `projects/cohorts/surgen-msi/cohort.yaml` | D-9 entry is now `approved`, with approver, date and reference. The summary states the route, compute, scope, protocol and the execution restrictions. A comment states the planning-only scope. The header explains the versioning. |
| `.claude/adr/008-real-cohort-validation-architecture.md` | Decision record row for D-9; "Effect of the D-9 approval on manifests" |
| `.claude/specs/008-real-cohort-integration-and-scientific-validation.md` | §9 decision-status note |
| `README.md` | v1 is reproduced with D-7 **and** D-9 reverted |
| `tests/integration/test_surgen_cohort_manifest.py` | The v1 reproduction reverts both approvals. The data-free test, renamed `test_configuration_differs_from_v1_only_by_the_recorded_approvals`, pins the D-9 record: approved, gates, date, reference, protocol terms, and the planning-only wording. Only D-10 remains pending. |

Storage paths are **not** in the ledger. The cohort configuration binds sources by name at run time and never stores a path. The directory plan (§2) lives in the reports only.

### 1.2 Versioning

- `cohort.yaml` stays at **version "2"**.
  - The reason for v2 was that a manifest embeds its ledger, so two different documents must not share one version name.
  - Manifest v2 has **not** been built. So no `surgen-msi@2` document exists that could disagree with the new ledger.
  - Bumping to "3" would leave a version that was never built.
- **Rule recorded in `cohort.yaml`:** once `manifests/v2` is built, any further ledger change needs version "3".
- **Manifest v1 is untouched.** Its ledger still shows D-7 and D-9 as pending, as they were when it was built.

| File | SHA-256 |
|---|---|
| `manifests/v1/cohort-manifest.json` | `f7748332…d084` (unchanged) |
| `manifests/v1/split.csv` | `48105e86…f8cf` (unchanged) |
| `manifests/v1/reconciliation.csv` | `69c3728c…1781` (unchanged) |
| `cohort.yaml` | `4e54a9ee…fb69` → **`8b482073…eca0`** |

- **Opt-in rebuild** (both roots bound): 3 passed. v1 rebuilds to `854f009f…5a3c`. The current definition gives identical records, split, counts and sources.
- **Effect on the selection:** `pai-pilot-5da7…` is bound to v1 and is unchanged. Its frozen `pending_decisions` still lists D-7 and D-9, as derived from the v1 ledger.

## 2. Local directory plan (proposed, not created)

**Location.** One new data root, outside Git: **`~/pathology-ai-data/surgen-msi`**, referred to as `$PAI_DATA`.

Why this location, rather than an existing root:

| Candidate | Suitable? | Reason |
|---|---|---|
| `~/pathology-ai-fixtures` (existing root of this project) | No | It holds CC0 reader fixtures (SPEC-003). Mixing in cohort data would blur licensing and fixture pinning. The new root follows the same naming convention. |
| `../digital-pathology/data` (legacy root, 18 GB) | No | Sibling repository. Must not be modified. |
| `~/digital-pathology-data/raw_wsi` | No | Legacy pipeline data |
| `~/Downloads` | No | Not a controlled location. C1 already ruled it out for new writes. |
| External volume | None mounted | Only `Macintosh HD` is present |
| **`~/pathology-ai-data/surgen-msi`** | **Yes** | Same APFS volume as the repository and the legacy cache. This allows atomic `mv` from staging and zero-copy APFS clones. It is outside every Git tree. |

**Layout:**

```text
$PAI_DATA = ~/pathology-ai-data/surgen-msi          (mode 700; outside Git)
├── staging/ebi/                  download staging: <slide>.czi.part, .headers, .transfer.json, curl logs
├── wsi/                          verified source WSIs: exactly the 8 pilot .czi + SHA256SUMS
│                                 (files and directory a-w after population)
├── embeddings/
│   ├── uni-official@1-0.5um-mps/ embedding stores: one <store_id>/ per slide (the SPEC-007 store root)
│   └── _verification/<run_id>/   A5 fresh re-embed and A6 interrupt/resume stores (never a store root)
├── runs/c2-pilot/
│   ├── acquisition/              transfers.jsonl (append-only download provenance)
│   └── <run_id>/                 run manifests: embed manifests, smoke JSON, run-ledger.csv, env.json
├── logs/c2-pilot/<run_id>/       stdout/stderr, /usr/bin/time -l output, disk and power snapshots
└── reports/c2-pilot/<run_id>/    machine-readable pilot metrics (JSON/CSV) for the narrative report
```

**What goes in the repository (small text only):**
- the new inventory under `projects/cohorts/surgen-msi/inventories/`;
- the new pilot selection under `projects/cohorts/surgen-msi/pilots/`;
- the narrative report `.claude/reports/008-embedding-pilot.md`.

The D-10 caveat from C1 applies before any of these are committed: they name study images.

**Rules:**
- **Downloads.** A `.part` file is moved into `wsi/` only after its size and SHA-256 check. The move is atomic on one volume, so no space is ever doubled.
- **The 3 local slides** enter `wsi/` as **APFS clones** (`cp -c`) of the legacy cache files.
  - A clone uses no extra space.
  - `chmod a-w` on the clone leaves the legacy file untouched.
  - The clone is re-hashed and must match the inventory.
  - Fallback if cloning fails: bind the legacy cache as a read-only inventory root and pass its paths to `--file`. A full 4.1 GB copy is not needed.
- **Store separation.** The A5 and A6 verification stores live outside the store root. They can never be mixed into a SPEC-007 dataset.
- **Recovery.** Nothing in `$PAI_DATA` is deleted by the procedures. Recovery is by resume, never by overwrite.

## 3. Mac readiness (MEASURED, read-only)

| Check | Result | Status |
|---|---|---|
| macOS | 27.2 (build 26B123), `arm64` | ✅ |
| Hardware | Apple M4 Pro, 12 cores | ✅ |
| RAM | 24 GiB unified. System-wide free 55 % at check time. | ✅ PRIOR need: peak RSS 1.10 GiB plus 2.09 GiB device (SPEC-006 §9, MPS) |
| Disk (Data volume, APFS) | **58.1 GiB (62.4 GB) free** of 494 GB; 87 % used | ✅ for the pilot (§4); ❌ for Phase D (C1 B6) |
| External volumes | None | — |
| Python / uv | CPython 3.12.13 (`.python-version` 3.12); uv 0.11.17 | ✅ |
| PyTorch / timm / NumPy | torch 2.14.1, timm 1.0.30, numpy 2.5.4: the same versions as the SPEC-006 runs | ✅ |
| MPS | `is_built()` True, `is_available()` True. Flags only; no tensor allocated. | ✅ |
| UNI checkpoint | `../digital-pathology/data/checkpoints/huggingface/uni.pth`, 1,213,527,781 B, SHA-256 **`56ef09b44a25dc5c7eedc55551b3d47bcd17659a7a33837cf9abc9ec4e2ffb40` = pin** | ✅ |
| CZI reader | pylibczirw 6.1.0 (`czi-pylibczirw`) | ✅ |
| OpenSlide (not needed for CZI) | openslide-python 1.4.6, openslide-bin 4.0.1.2 | ✅ |
| Dev tools | ruff 0.17.1, mypy 2.4.0, pytest 9.1.1 | ✅ |
| Download client | curl 8.7.1 (SecureTransport) | ✅ |
| 3 local pilot WSIs | Re-hashed in the legacy cache: T368 `9f35f73d…`, T156 `2363380e…`, T558 `619200db…`. All equal the inventory. | ✅ |
| **Power** | **On battery (92 %). System sleep after 1 min**, prevented at check time by another process | ❌ **Must be on AC**, with `caffeinate`, for downloads and embedding |
| Environment | `VIRTUAL_ENV` points to a pyenv interpreter. uv ignores it and warns. `PATHOLOGY_AI_CHECKPOINTS` is unset; the plan passes `--checkpoint`. | ⚠ harmless; `unset VIRTUAL_ENV` in the run shell |
| EBI reachability and rate | Not contacted (no HTTP allowed) | NOT VERIFIED |
| EBI published checksums | Not checked | NOT VERIFIED (trust on first download, §6) |

**Notes:**
- The checkpoint file is owner-writable (`rw-r--r--`). It sits in a sibling repository, so it was not changed. The UNI adapter re-hashes it on every load and refuses any other digest, and the preflight hashes it too.
- The legacy WSI copies are also owner-writable. They are never used in place; read-only clones are used instead (§2).

## 4. Resource estimates (8-slide pilot at 0.5 µm/px)

| Quantity | Value | Basis |
|---|---:|---|
| Downloads (5 WSIs) | **≈ 22 GB** (15–25 GB) | ESTIMATE, C1 pilot report §4 (about 1 MB per legacy row) |
| Clones of the 3 local WSIs | 0 B extra (4.10 GB logical) | APFS clone |
| Patches | ≈ 99 k (20,668 measured + ≈ 78 k estimated) | C1 §1.4 |
| Embedding stores | ≈ 0.41 GB (4,128 B/row) | C1 §1.3 |
| A5 and A6 verification stores | ≤ 0.1 GB (T156: 2,013 rows; T558: 16,171 rows) | ESTIMATE |
| Manifests, logs, reports | ≤ 0.2 GB | ESTIMATE |
| **Planned writes (budget)** | **26 GB** (25 GB downloads + 1 GB outputs) | Upper range |
| Free after the pilot | ≈ 36 GB, above the 10 GB reserve | 62.4 GB − 26 GB |
| Transfer time | ≈ 10 h at 0.6 MB/s; ≈ 1.2 h at 5 MB/s | NOT VERIFIED (legacy observation) |
| SHA-256 hashing | ≈ 15 s for 27 GB | MEASURED rate, about 2.3 GB/s in C1 |
| Tissue and planning | ≤ 2.3 s per slide; ≤ 216 MiB RSS | C1 MEASURED |
| UNI encode, MPS | **≈ 48 min** (29 ms/patch) + ≈ 1 min checkpoint loads | PRIOR, 178 patches. Sustained and thermal behaviour NOT VERIFIED. |
| A5 + A6 extra encode | ≈ 9 min | ESTIMATE |
| Peak memory | ≈ 1.1 GiB RSS + ≈ 2.1 GiB device (stage budget 4 GiB RSS) | PRIOR |
| **Wall time, after downloads** | **≈ 1–1.5 h** | ESTIMATE |

## 5. Preflight gate status (MEASURED, read-only run)

The preflight (`c2_preflight.py`, execution plan §1) was run **read-only** with `--stage download`.

| Gate | Check | Now |
|---|---|---|
| G1 | Manifest v1 files byte-identical | ✅ PASS |
| G2 | Approved cohort and the 8-slide selection (`pai-pilot-5da7…`, `pilot.yaml`, D-6) | ✅ PASS |
| G3 | No test-partition slide; no unexpected file in `wsi/` or staging | ✅ PASS |
| G4 | D-7 and D-9 approved, with the protocol terms | ✅ PASS |
| G5 | **D-10 licensing approved** | ❌ **FAIL** |
| G6 | **Explicit Phase C2 execution authorization** | ❌ **FAIL** (none given; `D-9` is refused as one) |
| G7 | Free disk ≥ planned writes + 10 GB reserve | ✅ PASS: 62.4 GB free, 26.0 GB planned |
| G8 | WSI content integrity (the 3 local sources now; all 8 before extraction) | ✅ PASS |
| G9 | Source WSIs immutable (`a-w`) | ✅ PASS (nothing downloaded yet; enforced before extraction) |
| G10 | UNI checkpoint SHA-256 = pin | ✅ PASS |
| G11 | MPS available | ✅ PASS |
| G12 | **AC power** | ❌ **FAIL** (battery) |

**Result: `preflight[download]: NO-GO (3 gate(s) not passed)`.**

The `--stage extract` run also fails closed:
- G8 and G9 are UNRESOLVED, because `wsi/` does not exist.
- G5, G6 and G12 fail as above.

## 6. Blockers

| # | Blocker | Blocks | Needed |
|---|---|---|---|
| B2 | ~~D-9 undecided~~ | — | **Resolved 2026-11-02** (pilot scope, planning only) |
| B3 | **D-10:** UNI CC-BY-NC-ND 4.0 terms and SurGen data terms not confirmed | All of C2 (G5) | Owner decision |
| B8 | **No Phase C2 execution authorization** | All of C2 (G6) | Owner ledger entry, worded as in the execution plan §1.2 |
| B9 | Mac on battery, 1-min system sleep | Downloads and embedding (G12) | Operational: AC power and `caffeinate` |
| B4 | 5 pilot WSIs not on this host | Extraction | Run the acquisition after B3 and B8 |
| B10 | EBI rate and checksum publication unknown | Download time; second-source confirmation | First transfer (measured), or S3 after SSO |
| B11 | Manifest v1 or v2 for C2 (open since C1) | Inventory and selection binding | Owner choice. **Default in the plan: v1**, citing the D-7 and D-9 ledger entries. v2 would require a re-taken inventory and selection. |
| B6 | Phase D storage (≈ 52 GB of stores vs 58 GiB free) | Phase D | Not covered by D-9 |

**Trust on first download.** If EBI publishes no checksum, the SHA-256 computed at first download is the content identity. The pilot report must say so. A second copy (S3 or a repeat download) would confirm it, but costs a second transfer. That is an owner choice and is not in the default plan.

## 7. Validation

| Check | Result |
|---|---|
| Full test suite | **695 passed, 53 skipped**, the same counts as C1. The renamed D-9 test replaces the D-7 one. The skips are the pre-existing opt-in real-WSI, UNI and CUDA tests, plus the SurGen rebuild. |
| `ruff check`, `ruff format --check` | Clean (210 files), after a one-line docstring fix in the edited test |
| `mypy --strict` | Clean (152 files) |
| SurGen v1 opt-in rebuild (both roots bound) | 3 passed. Digest `854f009f…5a3c` reproduced with D-7 and D-9 reverted. |
| Preflight script | `ruff check` (E, F, W, I, B, UP) and `ruff format --check` clean; `mypy --strict` clean. Run read-only: NO-GO, as expected (§5). |
| Acquisition script | `bash -n` clean on the system bash 3.2. Refuses to start without `C2_AUTH`. Not run against the network. |

**Unchanged files and directories (verified):**
- Manifest v1: all three SHA-256 values unchanged.
- `~/pathology-ai-data` does not exist.
- Sibling repositories: `git status` shows 4 and 5 lines, as in Phases A–C1.

## 8. Changes

**Modified:**
- `projects/cohorts/surgen-msi/cohort.yaml`: D-9 entry and comments; header versioning note
- `tests/integration/test_surgen_cohort_manifest.py`: v1 reproduction reverts D-7 and D-9; the D-9 record is pinned
- `.claude/adr/008-real-cohort-validation-architecture.md`: decision record and manifest effect
- `.claude/specs/008-real-cohort-integration-and-scientific-validation.md`: §9 status note
- `.claude/reports/008-phase-c1-readiness.md`: update banner pointing here
- `README.md`: one sentence

**Added:**
- this report
- `.claude/reports/008-phase-c2-execution-plan.md`

**Unchanged:**
- `manifests/v1/`, `pilot.yaml`, the inventory and selection documents
- `pipelines/` and all code
- sibling repositories and every WSI file

**STOP.** Infrastructure preparation is complete. Phase C2 waits for D-10, an explicit execution authorization and owner review.
