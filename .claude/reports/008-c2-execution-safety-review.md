# Report: SPEC-008 Phase C2, Execution Safety Review

**Date:** 2026-11-02
**Prompt:** PROMPT-008D10 (Part B)
**Reviewed:** `008-phase-c2-execution-plan.md` as prepared by PROMPT-008D9 (the embedded `c2_preflight.py`, the embedded `c2_acquire_ebi.sh` and the step commands)
**Outcome:** 11 defects found. 7 of them would have stopped or corrupted the pilot. All are fixed in versioned, tested tools, and the plan is revised. **Phase C2 is still NOT authorized.**

**Constraints honoured:**
- No request was sent to EBI by the tools or the tests. All transfer tests use a local HTTPS mock. (Part A read one BioStudies metadata document; see the licence report.)
- No WSI was downloaded, copied, cloned or modified. UNI was not run. Nothing was trained.
- `~/pathology-ai-data` was not created.
- Manifest v1, the split, `cohort.yaml`, the inventory and the selection are byte-identical.
- Sibling repositories were only read.
- No Git commit or staging.

---

## 1. Findings

Severity:
- **Blocking**: the pilot cannot complete.
- **Integrity**: wrong data or a skipped gate is possible without any error.
- **Robustness**: recovery or evidence is incomplete.

| # | Area (prompt item) | Defect in the D-9 version | Severity | Fix | Test |
|---|---|---|---|---|---|
| F1 | SHA256SUMS permissions (1) | After the first download the script ran `chmod a-w SHA256SUMS`. Download 2 then moved its file into `wsi/` and failed on `>> SHA256SUMS` (permission denied). Every later run stopped with "exists in wsi/ without a pin": a dead end that needed manual repair. Step 1b appended to the same read-only file. | Blocking | Pins are written by staging `SHA256SUMS.next` and **renaming** it into place, which needs only the directory write bit. The file stays `a-w`. | `test_five_sequential_downloads_are_pinned_sealed_and_recorded` |
| F2 | Staging allowlist vs curl files (2) | G3 accepted `.part`, `.headers` and `.transfer.json` only. The script also wrote `<name>.curl.log` and `<name>.headers.get.<n>`. After **any** first attempt, every re-run of the script (which starts with preflight) was NO-GO, so resume was impossible. The files were never removed, so `preflight[extract]` could never pass either. | Blocking | One regular expression in the preflight, `STAGING_ARTIFACT`, is also enforced by the tool: `Layout.artifact` refuses to create any name it does not match. After a slide is pinned, its artifacts are moved to `acquisition/<slide>/`. At the extract stage, staging must be empty. | `test_every_staging_artifact_name_is_allowlisted_by_preflight`, `test_resume_across_runs_passes_preflight_staging_allowlist`, `test_preflight_extract_stage_refuses_leftover_staging` |
| F3 | Source identity (2, 3) | `--max-redirs 0` without `-L` does not fail on a 3xx. curl exits 0 with the redirect body. HEAD then reported that body's Content-Length, GET fetched the same body, sizes matched, and an HTML page would have been pinned as a WSI (trust on first download). | Integrity | HEAD must be **200**. A fresh GET must be 200 with the full Content-Length. The file must start with the CZI magic `ZISRAWFILE`. | `test_redirect_is_refused_and_never_stored`, `test_not_a_czi_is_refused` |
| F4 | Resume and range validation (3) | Resume relied on a HEAD taken before the GET, with no `If-Range` and no check of `Content-Range`. A change between HEAD and GET, or a 206 at the wrong offset, would have appended wrong bytes. The size would still match, so the wrong file would have been pinned. | Integrity | GET carries `If-Range` (strong ETag, else Last-Modified). A resumed answer must be 206 with exactly `bytes <have>-<len−1>/<len>` and unchanged validators. Otherwise the bytes of that answer are **truncated away**, keeping the valid prefix, and the run stops. A missing validator or `Accept-Ranges` refuses the resume. After completion, 64 KiB on each side of every resume offset is **re-fetched and compared**. | `test_interrupted_transfer_resumes_with_if_range_and_verified_seam`, `test_invalid_resume_answer_is_refused_…` (ignored range, shifted range), `test_remote_change_between_attempts_is_refused`, `test_corrupted_bytes_at_a_resume_seam_are_detected` |
| F5 | Pipelines and `tee` (7) | Step blocks were pasted into an interactive shell without `pipefail`. `acquire 2>&1 \| tee` returned tee's status, so a refused acquisition went on to step 1b. `preflight --stage extract \| tee` **hid a NO-GO**, and extraction would have started. `cmp <(jq -r …) <(jq -r …)` compared `null` with `null` and passed when a digest was missing. | Integrity | Every step is a script run with `bash -euo pipefail`. `jq -e` is used and the values are compared as non-empty strings. The step scripts are hashed into `env.txt`. | Plan §0.1 and steps 3, 5, 6 (procedure; not executed) |
| F6 | Selection id after re-inventory (5) | The selection id digests the inventory digest and each slide's availability, so re-inventory **always** gives a new id. The authorization named `pai-pilot-5da7534b`, and no gate related the new document to it. The plan inventoried 4 roots, `~/Downloads` included. Because verified slides rank first, one more SurGen file there can **displace an approved slide**. Even with the same 8 slides, the order inside SR386/negative changes (T558 and T440 swap) once all 8 are verified. | Integrity | New gate **G13**: the re-taken selection must reproduce (`verify_selection`) from manifest v1, `pilot.yaml` and its inventory. The inventory's only root must be `pilot-wsi`. The slides must match the approved ones field by field as a set (identity, group, partition, label, stratum, rank key), status `ready`, and every SHA-256 equal to `SHA256SUMS`. Step 1d inventories `pilot-wsi` only. | `test_retaken_selection_over_the_pilot_root_is_the_approved_pilot`, `test_an_extra_verified_slide_elsewhere_changes_the_retaken_selection`, `test_g13_is_unresolved_without_the_retaken_documents` |
| F7 | Authorization binding (6) | The authorization was an id in the ledger. The tools were copied out of a Markdown file into `$PAI_DATA/tools`, unversioned and editable after approval. Nothing tied the approval to the code, the URLs, the pipelines or the plan that would run. | Integrity | `c2-binding.json` lists the SHA-256 of manifest v1, `pilot.yaml`, the approved selection and inventory, both pipelines, both tools and the plan, plus the UNI pin, the local sources and the 5 URLs. The ledger entry must name `c2-binding sha256 <hex>` and both gates. G6 re-hashes everything and refuses a preflight that is not running from the bound repository path. | `test_c2_authorization_is_bound_to_exact_artifact_digests`, `test_binding_names_every_executable_input` |
| F8 | Immutable WSIs (4) | Files and `wsi/` were only `a-w`. The script and step 1b reopened `wsi/` for writing during later downloads, and `rm -f` deletes an `a-w` file in a writable directory. G9 was skipped at the download stage, so files already pinned were not checked on a resume. | Robustness | Each pinned WSI also gets `chflags uchg` (no write, rename or unlink until the flag is cleared on purpose). `wsi/` is writable only inside the pin step. G9 checks pinned files at both stages. | `test_sealed_requires_the_immutable_flag_for_wsis` and the five-download test |
| F9 | APFS cloning (4) | Step 1b pinned whatever hash the clone had and compared it with the inventory only at G8, after everything else. `[[ -e ]] && continue` kept an existing **unpinned** clone (for example a partial one) forever. | Robustness | `clone-local` clones into staging (`cp -c`, which fails rather than falling back to a byte copy), hashes the clone, **refuses unless it equals the inventory SHA-256**, then uses the same pin sequence. | `test_clone_local_verifies_against_the_inventory_and_leaves_sources_untouched`, `test_clone_with_wrong_content_is_refused` |
| F10 | Provenance of 8 WSIs (9) | Only the 5 downloads had records. The 3 clones had none. A download's record was appended **after** the pin, and the "already verified" path never wrote it, so a crash in between lost it for good. HEAD headers were overwritten on each attempt. | Robustness | `transfers.jsonl` and `local-clones.jsonl` each hold one record per slide, written once even across crashes. Attempt numbers continue across runs. Every header file is archived. `c2_wsi.py provenance` writes a write-once `wsi-provenance.json` for all 8, cross-checked against `SHA256SUMS`, file sizes and Content-Length. | `test_provenance_covers_all_eight_and_is_write_once`, `test_crash_before_archive_recovers_without_duplicate_provenance` |
| F11 | Repeated preflight and recovery (8) | Each crash window (between move, pin, `chmod` and provenance) left a state the script refused. Two concurrent runs could interleave appends. Attempt counters restarted at 1. | Robustness | The pin sequence is journal → `a-w` → `link(2)` → rename `SHA256SUMS` → unlink the temporary → `uchg`. Every step is re-checked on re-run. G8 accepts an unpinned WSI only while its hash journal is in staging. An exclusive lock stops a concurrent run; a stale lock is reported, never broken automatically. The tests also found that a `SHA256SUMS.next` left behind by a crash blocked the next pin; that is fixed. | `test_crash_between_link_and_pin_recovers_without_duplicate_pins`, `test_concurrent_run_is_refused` |

Minor changes:
- G6 now requires the C2 entry to gate both `pilot` and `extraction`, as the plan's template already did.
- The acquisition no longer uses the unpinned system `python3`. It runs in the project environment.

## 2. Tools after the revision

| File | Lines of concern |
|---|---|
| `projects/cohorts/surgen-msi/c2-pilot/c2_preflight.py` | G1–G13, `bound_files`, `binding_document`, `check_binding`, `retaken_problems`, `STAGING_ARTIFACT` |
| `projects/cohorts/surgen-msi/c2-pilot/c2_wsi.py` | `download`, `get_problem`, `verify_content`, `pin`, `complete`, `acquire`, `clone_local`, `provenance` |
| `tests/integration/test_c2_pilot_tools.py` | 24 tests (§3) |
| `pyproject.toml` | `mypy_path` so the tools are type-checked through their tests |

`c2-binding.json` is **not** written. Writing it is part of the owner's authorization (plan §1.2), because it must hash the final reviewed tree.

## 3. Test method

- **Mock EBI.** A threaded HTTPS server on 127.0.0.1 with a self-signed certificate for `ftp.ebi.ac.uk`. It serves the 5 allowlisted paths with 300 kB synthetic CZI-like files, with ETag, Last-Modified, `Accept-Ranges`, `Range` and `If-Range` (RFC 9110 semantics).
- **Real client.** The tests run the system curl 8.7.1 with `--cacert <test cert>` and `--connect-to ftp.ebi.ac.uk:443:127.0.0.1:<port>`. The URL, Host header and SNI are the production ones, and nothing resolves to EBI. The extra curl arguments come in through a `Settings` object in the test; production code has no environment hook that could redirect it.
- **Injected faults (one per request):**
  - connection dropped after N bytes;
  - Range ignored (200);
  - Range shifted by one byte;
  - corrupted first byte of a 206;
  - 301 on HEAD;
  - ETag changed between runs;
  - HTML body;
  - crash (raised interrupt) between link and pin, and before archive.
- **Platform.** The tests are skipped off macOS or without curl/openssl, because they use APFS clones and `chflags`. A fixture clears the immutable flags afterwards.

## 4. Validation

| Check | Result |
|---|---|
| New tests | **24 passed** |
| Full suite | **719 passed, 53 skipped** (695 before plus the 24 new; the skips are the same pre-existing opt-in real-WSI, UNI, CUDA and rebuild tests) |
| `ruff check .`, `ruff format --check .` | Clean |
| `mypy` (project configuration, 153 files) and `mypy --strict` on both tools | Clean |
| Manifest v1 | `f7748332…`, `48105e86…`, `69c3728c…` unchanged; `cohort.yaml` `8b482073…eca0` unchanged |
| Sibling repositories | `git status` shows 4 and 5 lines, as in Phases A–D9; SurGen clone still at `e56ca8cd…` |

## 5. Read-only preflight on this host (2026-11-02)

The data root does not exist; nothing was created.

| Stage | Result | Gates not passed |
|---|---|---|
| `download` (with `--c2-authorization D-9`) | **NO-GO** | G5 (D-10 pending); G6 (D-9 refused as an authorization); G12 (battery). G8 PASS: the 3 legacy sources match the inventory. G10 PASS: checkpoint = pin. |
| `extract` | **NO-GO** | G5, G6, G8 (no WSIs), G9 (UNRESOLVED, no `wsi/`), G12, G13 (UNRESOLVED, no re-taken documents) |

## 6. Residual risks (not fixed by code)

| Risk | Why it remains | Mitigation |
|---|---|---|
| Trust on first download | The S-BIAD1285 study metadata has no checksums (licence report §3.1). TLS and the seam checks protect the transfer, not the publisher's original bytes. | Owner decision O-5 (§7) |
| EBI server behaviour unknown | Whether EBI sends a strong ETag, Last-Modified and `Accept-Ranges: bytes` is not verified. Without them a resume is **refused** (fail closed), and a re-download needs the operator to move the `.part` aside. | The first HEAD record shows it. If needed, the owner accepts restarting a slide from zero. |
| SurGen release changed on 2026-08-03 (OME-Zarr added) | Not known whether the CZI files changed. The 3 local slides are checked against the C1 inventory, but the 5 downloads have no earlier reference. | ETag and Last-Modified are recorded. A second source would confirm (O-5). |
| `uchg` seals outlive the pilot | Deleting the data later needs `chflags -R nouchg`, on purpose | Retention decision O-9 |
| Tools are macOS-specific | `cp -c`, `chflags`, `pmset`, `caffeinate` | D-9 fixes the host as the local Mac |
| The binding includes the plan | Any edit to the plan, even a typo fix, invalidates an authorization | Intended. Re-bind and re-authorize. |

## 7. Remaining owner decisions

| # | Decision | Blocks | Recommendation |
|---|---|---|---|
| O-1 | **D-10** (licences): confirm UNI access (account, institutional email, approval, declared use), non-commercial purpose, single operator, no distribution of UNI derivatives, attribution | G5, so all of C2 | Approve only if item 1 of licence report §5 can be confirmed |
| O-2 | **Phase C2 authorization**: generate and commit `c2-binding.json`, add the bound ledger entry (plan §1.2), update the data-free ledger test | G6, so all of C2 | After O-1 and after review of this revision |
| O-3 | Institutional ethics position for secondary use of public, de-identified data | Publication (possibly use) | Record it in D-10 or as a separate note |
| O-4 | Distribution policy for UNI embeddings and UNI-based MIL weights | Any sharing | Do not distribute (licence report §2.3) |
| O-5 | Accept trust on first download for the 5 WSIs, or require a second source (S3 after SSO, or a repeat download compared by SHA-256) | Strength of the provenance claim | Accept for the pilot and state it in the pilot report; reconsider for Phase D |
| O-6 | Manifest v1 or v2 for C2 (blocker B11, open since C1) | Inventory and selection binding | Keep v1 (the plan and G13 assume it) |
| O-7 | Commit the new tools, tests and reports; later, the re-taken inventory and selection documents that step 1 writes into the repository | Binding (G6 hashes the working tree; a commit makes it reviewable) | Commit before generating the binding |
| O-8 | Operational: AC power during download and extraction | G12 | — |
| O-9 | Retention and clean-up of the sealed WSIs and stores after the pilot | Disk for Phase D (B6) | Decide with the Phase D storage plan |

**STOP.** Nothing was executed. D-10 and the Phase C2 authorization remain with the owner.
