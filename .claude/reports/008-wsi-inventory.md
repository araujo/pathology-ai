# Report: SPEC-008 Phase C1, WSI Inventory

**Date:** 2026-11-02
**Prompt:** PROMPT-008C1 (WSI Inventory and Extraction Pilot Readiness)
**Companions:**
- `008-pilot-selection.md`: pilot selection
- `008-phase-c1-readiness.md`: scale and compute comparison, extraction procedure, go/no-go

**Outcome:**

| Item | Result |
|---|---|
| Generic WSI inventory adapter (local filesystem first) | Built and tested |
| Scoped local inventory of SurGen WSIs on this host | Run. **17 slides content-verified**, 0 duplicate content, 0 inconsistent copies. |
| Eligible slides verified locally | 16 of 820. All MSI-negative; 5 are test slides. |
| Cohort-wide gates `wsi_integrity` / `wsi_duplicate_content` | **Still open for all 820** (deliberately unchanged) |
| Cohort manifest `surgen-msi@1` | Unchanged: digest `854f009f…5a3c`, split digest `0c8c94fc…5616` |

Constraints honoured:
- No WSI was downloaded, decoded or modified.
- No model was run.
- No split membership changed.
- Sibling repositories were only read.
- No Git commit was made.

Status legend:
- **VERIFIED**: measured in this phase and reproducible with §8
- **INFERRED**: a conclusion drawn from verified facts
- **NOT VERIFIED**: not checked
- **BLOCKED**: could not be checked

---

## 1. What was built

All modules are cohort- and format-agnostic. The architecture test `test_generic_cohort_code_names_no_cohort_or_target` now covers them, and forbids SurGen, sub-cohort, target, model and vendor names.

| Layer | Module | Responsibility |
|---|---|---|
| domain | `domain/wsi_inventory.py` | `pathology-ai.wsi-inventory/1`: `InventoryRoot`, `ObservedObject`, `SlideInventoryRecord`, `DuplicateContent`, `WSIInventory`; pure `build_inventory()`, `derive_slides()`, `cohort_wsi_evidence()` |
| domain | `domain/errors.py` | `WSI_INVENTORY_INVALID`, `PILOT_SELECTION` (both subclasses of `COHORT_INVALID`) |
| ports | `WSIObjectSource` | `roots`, `patterns`, `observe()` |
| application | `application/wsi_inventory_service.py` | `WSIInventoryService` (sources → inventory against a manifest); `scoped_summary()` (joins observations with eligibility, partition and label) |
| infrastructure | `infrastructure/wsi_inventory.py` | `LocalWSIObjectSource` (read-only hashing); `ObjectListingSource` (saved `list-objects-v2` JSON, metadata only); write-once inventory and pilot documents; pilot criteria loader |
| presentation | `cli.py` → `pathology-ai-cohort inventory` / `pilot` | Thin subcommands that print JSON summaries |
| projects | `projects/cohorts/surgen-msi/inventories/<id>/wsi-inventory.json` | The scoped local inventory below |

### 1.1 Identity

- Canonical slide identities come from the cohort manifest. Each record's declared image file name (`record.wsi.file_name`) maps an object to its slide.
- No identifier grammar or cohort pattern is involved. An object whose file name matches no record is listed as `unmatched`, never guessed.
- Every record is an expected image, including excluded ones, because duplicate content can hide among exclusions.

### 1.2 Evidence levels (never promoted)

| Verification | Applies to | Carries | Meaning |
|---|---|---|---|
| `content_verified` | local, `readable` | size + **SHA-256 of the full file**, read in this inventory | Content identity established |
| `metadata_only` | remote, `listed` | size and an opaque `remote_checksum` (`s3-etag`) | Listed only. The ETag is **not** a SHA-256 (multipart, SSE and store-specific). It is never compared with a content digest and never used for duplicate detection. |
| `failed` | local, `unreadable` | no digest; a note (read error, empty file, changed while read) | Present but not verifiable |

The domain refuses construction of:
- a remote object with a SHA-256;
- a local object with a remote checksum;
- a failed read with a digest;
- an absolute path, or a path containing `..`.

A remote object becomes content-verified only after it has been fetched and hashed locally.

### 1.3 Per-slide availability

| Availability | Rule |
|---|---|
| `verified` | At least one content-verified copy, and all verified copies agree on SHA-256 and size |
| `inconsistent_copies` | Verified copies of one identity disagree |
| `unreadable` | Local copies exist, none readable |
| `remote_only` | Listed remotely only |
| `not_found_in_scope` | No copy under the inventoried roots. **Not** evidence that the image is missing. |

**Duplicate content:** identical SHA-256 under **different** slide identities. Copies of the same identity are not duplicates.

### 1.4 Read-only guarantees

| Guarantee | Enforcement |
|---|---|
| Files are only opened `rb` | Architecture test `test_wsi_inventory_source_opens_files_read_only` (AST: the only `open` mode is `"rb"`; no `write_*`, `unlink`, `rename`, `replace`, `touch`) |
| Nothing under a root changes | Contract test on a tree made read-only: size, mtime, mode and content identical before and after |
| Symbolic links are never followed | Contract test: a file link and a directory link pointing outside the root are both skipped |
| An object that changes while being hashed gets no digest | Contract test with a simulated concurrent writer → `unreadable`, "changed while it was read" |
| Absolute paths never enter a document | Roots are bound at run time (`NAME=PATH`). Objects are `root:relative/path`. |

### 1.5 Documents

- `WSIInventory` re-derives its slides, duplicate groups and unmatched list from its observations on construction. An edited document is refused.
- The inventory is written once under `<inventory id>/wsi-inventory.json`. Identical content is accepted; different content is refused (`EXPERIMENT_CONFLICT`). A tampered file fails its integrity digest.

### 1.6 Closing cohort gates (guarded)

`cohort_wsi_evidence(inventory, manifest)` turns an inventory into the `WSIEvidence` that a new manifest version consumes. It refuses when:
- the inventory was taken against another manifest; or
- any slide is `not_found_in_scope` or `remote_only` (inventory not *complete*).

The reason: closing `wsi_duplicate_content` for one slide needs the content of every other slide.

Tests show that a complete inventory:
- closes both gates (no provisional records);
- turns shared content into `excluded_duplicate`;
- turns unreadable copies into `excluded_quality`;
- keeps eligible membership otherwise identical.

## 2. Scoped local inventory (VERIFIED)

| Item | Value |
|---|---|
| Inventory id | `pai-wsi-inventory-783f44e2ea71bead23ce78c80f069e12` |
| Inventory digest | `783f44e2ea71bead23ce78c80f069e125ed8254eb49fc2f8828872ec16a836e2` |
| File SHA-256 | `4d8b53a5152e01787ca9400ad68533ec310595afb6f42d0a5e28c8413ac91343` |
| Identity source | `cohort:surgen-msi@1#854f009f…5a3c` |
| Roots | `downloads` (`~/Downloads`); `legacy-cache` (`../digital-pathology/data/cache/digital_pathology/SurGen`); `legacy-cache-copy` (`~/projects/digital-pathology/data/cache/digital_pathology/SurGen`, a separate copy with different inodes) |
| Pattern | `*.czi` |
| Objects hashed | 32 (47.9 GB read; 27.3 GB unique content) |
| Wall time | 21 s (about 2.3 GB/s, Apple M4 Pro SSD) |
| Symbolic links skipped / unmatched objects | 0 / 0 |

The 13 stale `.part*` download fragments next to T108 (Phase A) do not match `*.czi`. They are outside the inventory by pattern.

### 2.1 Slides found

All 17 SHA-256 values equal the Phase A digests, recomputed independently. Every identity with several copies has byte-identical copies.

| Slide | Cohort status | Partition | Label | Bytes | SHA-256 (prefix) | Copies |
|---|---|---|---|---:|---|---:|
| SR1482 T002_01 | eligible | train | negative | 547,312,640 | `b014a4fa6d6fbe8b` | 1 |
| SR1482 T005_01 | eligible | train | negative | 2,868,092,160 | `1695b3136bd67493` | 1 |
| SR1482 T008_01 | eligible | train | negative | 5,994,180,704 | `c529cf8a9bd3f285` | 1 |
| SR1482 T231_01 | eligible | train | negative | 344,106,944 | `2c7b13287f9c09f6` | 2 |
| SR1482 T348_01 | eligible | train | negative | 871,981,280 | `fcf206c31908e6bf` | 2 |
| SR1482 T368_01 | eligible | train | negative | 848,763,072 | `9f35f73dfc7e250e` | 3 |
| SR1482 T371_01 | eligible | train | negative | 632,211,360 | `18cdb48667cceb77` | 2 |
| SR1482 T085_01 | eligible | validation | negative | 284,587,392 | `1cd01239e959c354` | 2 |
| SR1482 T156_01 | eligible | validation | negative | 579,305,120 | `2363380e036443e6` | 2 |
| SR1482 T424_01 | eligible | validation | negative | 439,698,464 | `bfb7b1517f2024d0` | 2 |
| SR386 T558_01 | eligible | validation | negative | 2,670,401,216 | `619200db97a683bc` | 2 |
| SR1482 T004_01 | eligible | **test** | negative | 2,098,256,352 | `0674f10ab356d743` | 2 |
| SR1482 T035_01 | eligible | **test** | negative | 768,698,912 | `9c774a0707d48d6c` | 2 |
| SR1482 T108_01 | eligible | **test** | negative | 263,222,720 | `3ea74daccff7373c` | 2 |
| SR1482 T198_01 (F-CZI-01) | eligible | **test** | negative | 77,963,616 | `8634d0551cb7c137` | 1 |
| SR386 T568_01 | eligible | **test** | negative | 6,001,653,152 | `452139549b7d3e94` | 2 |
| SR1482 T005_02 | excluded_missing_label (`slide_policy`, D-2) | — | — | 1,980,508,160 | `9177084e9f4fec41` | 3 |

Full digests and copy locations are in the inventory document.

### 2.2 Mapping to the cohort

| | Train | Validation | Test | Excluded | Total |
|---|---:|---:|---:|---:|---:|
| Verified locally, negative | 7 | 4 | 5 | 1 (`_02`) | 17 |
| Verified locally, positive | **0** | **0** | **0** | — | **0** |
| Not found in scope (eligible) | 483 | 161 | 160 | — | 804 |
| Not found in scope (excluded) | — | — | — | 199 | 199 |

- **Sub-cohorts.** SR386 has 2 local slides (T558 validation, T568 test). SR1482 has 15.
- **Pilot pool.** The partitions the pilot may use (D-6) hold **11** locally verified slides, all negative.
- **Test slides.** The 5 local test slides were inventoried, which records integrity, but no later step may use them (§ pilot report).

## 3. Global integrity gates: preserved

| Gate | Before C1 | After C1 | Why |
|---|---|---|---|
| `wsi_integrity` | open for 820 | **open for 820** | The inventory is scoped (`complete = false`, 1,003 `not_found_in_scope`) |
| `wsi_duplicate_content` | open for 820 | **open for 820** | Needs digests of all 1,020 images; 17 are known |
| Manifest | `surgen-msi@1` | `surgen-msi@1`, byte-identical | No v2 was built. `cohort_wsi_evidence` refuses this inventory (tested on the synthetic analogue). |

**Duplicate content among the 17** (VERIFIED): none. This says nothing about the other 1,003 images.

**Limitation (INFERRED):** the guard sits in `cohort_wsi_evidence`. Calling `domain.cohort.build_manifest` directly with hand-made `WSIEvidence` for a subset of slides would still close both gates for that subset, which is Phase B behaviour and unchanged. Phase D must build manifest v2 only through `cohort_wsi_evidence`. A follow-up could move the "duplicate gate needs full coverage" rule into `domain/cohort.py` itself. That would change a Phase B contract, so it was not done here.

## 4. Remote sources

| Source | State in C1 |
|---|---|
| S3 `digital-pathology-training-545053092614` | **Not attempted.** Phase A found all three SSO profiles expired. No listing exists to import. |
| EBI BioStudies S-BIAD1285 (HTTPS/FTP) | **Not contacted** (no downloads). Public URLs for all 820 labelled slides are in the legacy `data/metadata/SurGen/MMR_MSI/download_{train,validate,test}.txt` (490 / 165 / 165 lines, VERIFIED). |
| Published per-file checksums from EBI | NOT VERIFIED. If none exist, the SHA-256 recorded at first download becomes the content identity ("trust on first download"). A second independent copy (S3, or a second download) would be needed to confirm it. |

`ObjectListingSource` can import a saved S3 listing as metadata-only evidence as soon as credentials are restored. It does not close any gate.

## 5. Tests added

| Kind | File | Tests |
|---|---|---:|
| unit | `tests/unit/test_wsi_inventory.py` | 21 |
| contract | `tests/contract/test_local_wsi_inventory.py` | 10 |
| integration | `tests/integration/test_cohort_inventory_pilot.py` (CLI end-to-end plus SurGen regression on the committed documents) | 3 |
| architecture | 4 modules added to the generic-cohort rule; read-only `open` rule | 5 |

The pilot tests are listed in the companion report. The full validation results are in the readiness report §6.

**Synthetic hazards covered:**
- agreeing and disagreeing copies
- cross-identity duplicates
- duplicates among excluded records
- an ETag equal to a real digest (no promotion, no duplicate)
- remote objects carrying a SHA-256 (refused)
- unreadable, empty and concurrently modified files
- file and directory symlinks
- absolute and `..` paths
- undeclared roots
- unmatched names
- an inventory from another manifest
- edited and rewritten documents
- partial coverage offered as cohort evidence

## 6. Not verified

- Whether F-CZI-01 (T198, 78 MB) is the complete EBI original. It is a test slide and is not needed by the pilot.
- Image *readability* beyond bytes. The inventory hashes; it does not decode. Decoding is checked by the pilot (§ readiness report, acceptance criteria).
- The 1,003 images not on this host.

## 7. Files written

| Path | SHA-256 |
|---|---|
| `projects/cohorts/surgen-msi/inventories/pai-wsi-inventory-783f44e2ea71bead23ce78c80f069e12/wsi-inventory.json` | `4d8b53a5…1343` |

## 8. Reproducing

```bash
uv run pathology-ai-cohort inventory \
    --manifest projects/cohorts/surgen-msi/manifests/v1/cohort-manifest.json \
    --root downloads=$HOME/Downloads \
    --root legacy-cache=../digital-pathology/data/cache/digital_pathology/SurGen \
    --root legacy-cache-copy=$HOME/projects/digital-pathology/data/cache/digital_pathology/SurGen \
    --pattern '*.czi' --out projects/cohorts/surgen-msi/inventories
uv run pytest tests/unit/test_wsi_inventory.py tests/contract/test_local_wsi_inventory.py \
    tests/integration/test_cohort_inventory_pilot.py
```

A re-run on unchanged files reproduces the same inventory id. The write-once check accepts it.
