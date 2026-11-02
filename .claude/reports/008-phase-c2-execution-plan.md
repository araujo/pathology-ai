# Plan: SPEC-008 Phase C2, Pilot Execution

**Date:** 2026-11-02 (revised 2026-11-02 by PROMPT-008D10; D-10 recorded by PROMPT-008C2-AUTH)
**Prompt:** PROMPT-008D9 (Pilot Infrastructure Preparation); safety review PROMPT-008D10
**Status:** **Prepared. D-10 approved (2026-11-02). Phase C2 execution NOT yet authorized (§1.2). Nothing in this document has been executed.**
**Companions:**
- `008-d9-infrastructure-readiness.md`: D-9 record, directory plan, Mac readiness, estimates, blockers
- `008-phase-c1-readiness.md` §3: procedure rationale and acceptance criteria A1–A11
- `008-c2-execution-safety-review.md`: the defects found in the D-9 version of this plan and their fixes
- `008-d10-license-evidence.md`: the evidence behind D-10 (approved 2026-11-02)

This plan runs the 8-slide pilot `pai-pilot-5da7534bcaa4d30c93f4eae703b3d479`:
- on the local Mac with Apple MPS (D-9);
- with `uni-official@1` at 0.5 µm/px ± 15 %, RGB 224 × 224, fp32, batch 16 (D-7, D-9);
- after acquiring the 5 pending WSIs by EBI HTTPS (D-9).

Every step **fails closed**. Any non-zero exit stops the run. A stopped step is resumed by re-running the same command, never by editing or deleting outputs.

**Before any command:**
1. D-10 is approved (done, 2026-11-02).
2. The owner has recorded the Phase C2 authorization, bound to `c2-binding.json` (§1.2).
3. `preflight[download]` prints `GO`.

**What changed in the PROMPT-008D10 revision** (details in the safety review):
- The preflight and the acquisition are no longer code blocks to copy. They are versioned, tested files in `projects/cohorts/surgen-msi/c2-pilot/`, run in place from the repository. The bash acquisition script is replaced by `c2_wsi.py`.
- The authorization binds the SHA-256 of every file the run depends on, this plan included (G6).
- The re-taken selection is checked for equivalence with the approved one (new gate G13), because its id always changes.
- Every step runs as a script under `bash -euo pipefail`, so a failure inside `| tee` stops the run.

---

## 0. Shell setup

### 0.1 Session environment (every session)

Save this once as `$PAI_DATA/c2-env.sh` (it holds no secret and is not bound). Every step script sources it.

```bash
# c2-env.sh: sourced by every step script of this plan
cd ~/ufsc/pathology-ai
unset VIRTUAL_ENV                       # use the project .venv (uv warns otherwise)
export PAI_DATA=~/pathology-ai-data/surgen-msi
export LEGACY=../digital-pathology/data/cache/digital_pathology/SurGen   # read only
export UNI_CKPT=../digital-pathology/data/checkpoints/huggingface/uni.pth   # read only
export STORES=$PAI_DATA/embeddings/uni-official@1-0.5um-mps
export RUN_ID=<set once, e.g. c2-pilot-20261103T090000Z; never changed on resume>
export RUN=$PAI_DATA/runs/c2-pilot/$RUN_ID LOGS=$PAI_DATA/logs/c2-pilot/$RUN_ID
export C2_AUTH=<owner decision id>      # see §1.2
export TOOLS=projects/cohorts/surgen-msi/c2-pilot
export V1=projects/cohorts/surgen-msi/manifests/v1/cohort-manifest.json
SLIDES=(SR1482_40X_HE_T368_01 SR1482_40X_HE_T156_01 SR1482_40X_HE_T224_01 SR1482_40X_HE_T130_01
        SR386_40X_HE_T558_01 SR386_40X_HE_T440_01 SR386_40X_HE_T489_01 SR386_40X_HE_T044_01)
COMMON=(--data-root "$PAI_DATA" --checkpoint "$UNI_CKPT" --legacy-root "$LEGACY"
        --c2-authorization "$C2_AUTH")
```

**How to run a step.** Copy the step's block verbatim to `$RUN/steps/<step>.sh` and run:

```bash
caffeinate -dimsu bash -euo pipefail "$RUN/steps/<step>.sh"   # no sleep while it runs (G12)
```

- `-e` stops at the first failing command, and `pipefail` makes `cmd | tee log` fail when `cmd` fails. In the D-9 version a NO-GO piped into `tee` was silently ignored.
- Never paste a block into an interactive shell: a failing line there does not stop the following lines.
- Step 7 records `shasum -a 256 $RUN/steps/*.sh`, so the commands actually run are part of the evidence.

### 0.2 Create the directory plan (first session only)

```bash
mkdir -p -m 700 ~/pathology-ai-data
mkdir -p -m 700 $PAI_DATA/{staging/ebi,embeddings/_verification,runs/c2-pilot/acquisition,logs/c2-pilot,reports/c2-pilot}
mkdir -p -m 700 "$STORES" "$RUN/steps" "$LOGS" "$PAI_DATA/reports/c2-pilot/$RUN_ID"
touch ~/pathology-ai-data/.metadata_never_index          # keep Spotlight off the WSIs
# wsi/ is created by c2_wsi.py (0700 while a pin is written, 0500 otherwise).
# Nothing is copied to $PAI_DATA/tools: the tools run from the repository, where G6 can bind them.
```

## 1. Preflight gates

### 1.1 Gates

`projects/cohorts/surgen-msi/c2-pilot/c2_preflight.py` is read-only. It writes nothing, sends no request, creates no tensor and loads no model.

A gate that cannot be evaluated is **UNRESOLVED**, which counts as a failure. The exit status is 0 only when all 13 gates pass.

| Gate | Requirement | `download` stage | `extract` stage |
|---|---|---|---|
| G1 | Manifest v1 files byte-identical (3 SHA-256 pins) | ✔ | ✔ |
| G2 | Approved cohort and pilot selection: selection file SHA-256 `bc320a1b…`, `pilot.yaml` `333fe05f…`, cohort `854f009f…`, exactly the 8 slides, authorized by D-6 | ✔ | ✔ |
| G3 | No test-partition slides: selection partitions ⊆ {train, validation}. Nothing in `wsi/` except the 8 files and `SHA256SUMS`. Download stage: staging holds only the artifacts `c2_wsi.py` writes for pilot slides (the same regular expression is enforced by the tool, so the two cannot diverge). Extract stage: staging is **empty**. | ✔ | ✔ |
| G4 | D-7 and D-9 approved in the ledger, with the protocol terms (`EBI HTTPS`, `MPS`, `uni-official@1`, `fp32`, `batch size 16`) | ✔ | ✔ |
| G5 | **D-10 approved**, gating `pilot` and `extraction` | ✔ | ✔ |
| G6 | **Explicit Phase C2 authorization bound to artifact digests**: `--c2-authorization ID` names an approved ledger decision gating `pilot` and `extraction` whose summary contains `Phase C2 execution authorized`, `pai-pilot-5da7534b` and `c2-binding sha256 <64 hex>`. `c2-binding.json` must hash to that value, and every file it lists must still hash to its recorded value. The preflight must itself be the bound repository file. | ✔ | ✔ |
| G7 | Free disk − planned writes ≥ **10 GB reserve**. Planned = 1 GB outputs, plus at the download stage 25 GB minus bytes already on disk | ✔ | ✔ |
| G8 | WSI content integrity. Both stages: `SHA256SUMS` well formed, pins only pilot files, agrees with the inventory for the 3 local slides, and every pinned file re-hashes to its pin. Download stage: the 3 legacy sources equal the inventory SHA-256; a WSI without a pin is accepted only while its hash journal is in staging (interrupted pin). Extract stage: exactly 8 pins and 8 files. | ✔ | ✔ |
| G9 | Immutable sources: every pinned WSI is `a-w` with the user-immutable flag (`chflags uchg`), `SHA256SUMS` is `a-w`; at the extract stage `wsi/` itself has no write bit | ✔ | ✔ |
| G10 | UNI checkpoint SHA-256 = `56ef09b4…fb40` | ✔ | ✔ |
| G11 | MPS built and available (capability flags only) | ✔ | ✔ |
| G12 | AC power (`pmset -g batt`) | ✔ | ✔ |
| G13 | **Re-taken selection equals the approved pilot** (`--retaken-selection`, `--retaken-inventory`): it reproduces from manifest v1, `pilot.yaml` and its inventory; the inventory's only root is `pilot-wsi`; same cohort and criteria; the same 8 slides with the same group, partition, label, stratum and rank key (order-insensitive); status `ready`; every SHA-256 equals `SHA256SUMS` | (deferred) | ✔ |

**Further stop conditions during the run** (proposed in C1 §3.7):
- free disk < 10 GB (re-checked before every transfer attempt and every slide);
- peak RSS > 4 GiB;
- throughput more than 2× slower than 29 ms/patch.

### 1.2 Form of the Phase C2 authorization

D-9 itself is refused as an authorization: it gates only `extraction`, and its summary says that it authorizes no execution.

The owner, when ready to authorize (D-10 is approved):
1. Generates the binding from the reviewed tree and commits it:
   ```bash
   uv run --offline python $TOOLS/c2_preflight.py --print-binding > $TOOLS/c2-binding.json
   shasum -a 256 $TOOLS/c2-binding.json        # -> <binding sha256>
   ```
   The binding lists the SHA-256 of manifest v1, `pilot.yaml`, the approved selection and inventory, both pipelines, `c2_preflight.py`, `c2_wsi.py` and **this plan**, plus the UNI checkpoint pin, the 3 local sources and the 5 URLs.
2. Adds an entry like the one below to `cohort.yaml`, choosing the id:
   ```yaml
     - id: <owner-chosen id>
       summary: >-
         Phase C2 execution authorized for pilot selection pai-pilot-5da7534b on manifest
         surgen-msi@1: EBI download of the 5 pending WSIs and UNI extraction per D-7/D-9,
         exactly as bound by c2-binding sha256 <binding sha256>.
       status: approved
       gates: [pilot, extraction]
       approver: ...
       approved_on: "YYYY-MM-DD"
       reference: ...
   ```

Consequences:
- Any later edit to a bound file, including a typo fix in this plan, fails G6. The owner then re-generates the binding and records a new authorization; the old one is never edited.
- `cohort.yaml` is not bound (the authorization lives in it). G4–G6 check its decisions instead.
- It is a ledger change after D-9, still within the unbuilt version "2". The versioning rule in `cohort.yaml` applies.
- The data-free test `test_configuration_differs_from_v1_only_by_the_recorded_approvals` will then fail by design (v1 must reproduce from the ledger by reverting only D-7, D-9 and D-10). It must be updated in the same change.

### 1.3 Tools

| File | Role | Tests |
|---|---|---|
| `c2_preflight.py` | Gates G1–G13 (read-only); `--print-binding` | `tests/integration/test_c2_pilot_tools.py` |
| `c2_wsi.py` | `acquire` (5 EBI downloads), `clone-local` (3 APFS clones), `provenance` (all 8). Runs `preflight[download]` first. | same file (local HTTPS mock of `ftp.ebi.ac.uk`) |

Both are checked by `ruff`, `ruff format` and `mypy --strict` (`mypy_path` in `pyproject.toml`).

## 2. Acquisition (EBI HTTPS, 5 files)

### 2.1 Allowlist

The URLs below are the only acquisition targets (`c2_wsi.ALLOW`, also recorded in the binding). Source: legacy `data/metadata/SurGen/MMR_MSI/download_{train,validate}.txt` (VERIFIED, read locally).

| Slide | Partition | Label | URL |
|---|---|---|---|
| SR1482_40X_HE_T224_01 | train | positive | `https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/285/S-BIAD1285/Files/SR1482_WSIs/SR1482_40X_HE_T224_01.czi` |
| SR1482_40X_HE_T130_01 | train | positive | `https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/285/S-BIAD1285/Files/SR1482_WSIs/SR1482_40X_HE_T130_01.czi` |
| SR386_40X_HE_T440_01 | train | negative | `https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/285/S-BIAD1285/Files/SR386_WSIs/SR386_40X_HE_T440_01.czi` |
| SR386_40X_HE_T489_01 | train | positive | `https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/285/S-BIAD1285/Files/SR386_WSIs/SR386_40X_HE_T489_01.czi` |
| SR386_40X_HE_T044_01 | validation | positive | `https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/285/S-BIAD1285/Files/SR386_WSIs/SR386_40X_HE_T044_01.czi` |

### 2.2 `c2_wsi.py acquire`

| Requirement | How | Test |
|---|---|---|
| Only the 5 approved files | Hard-coded allowlist; the URL must lie under the S-BIAD1285 base and name `<slide>.czi` | binding test |
| Unexpected source identity refused | HTTPS only (`--proto =https`). HEAD must answer **200**; a 3xx is refused (curl without `-L` returns a redirect body with exit 0, so the D-9 script would have stored and pinned it) | redirect test |
| `.part` temporaries | `staging/ebi/<slide>.czi.part`; headers per attempt (`.aNNN.head`, `.aNNN.get`), transfer record (`.transfer.json`), hash journal (`.verified.json`), seam buffers. All allowlisted by G3. | G3 allowlist tests |
| Interrupted transfer recovery | Up to 5 attempts per run; any later run resumes from the `.part`. Attempt numbers continue across runs, so no header file is overwritten. | resume tests |
| Range validation | Resume = `curl --continue-at <size>` plus `If-Range: <strong ETag>` (else Last-Modified). The answer must be **206** with `Content-Range: bytes <size>-<len−1>/<len>` and unchanged validators; a fresh GET must be 200 with the full Content-Length. Otherwise the bytes of that answer are **truncated away** and the run stops, keeping the valid prefix. | ignore-range, shifted-range tests |
| Seam verification | After completion, 64 KiB on each side of every resume offset is re-fetched by range and compared byte for byte. This catches a 206 with valid headers and wrong bytes, which size checks cannot. | corrupted-seam test |
| Remote change detected | Content-Length, ETag, Last-Modified from the first HEAD are kept. A difference on any later HEAD or GET **refuses**; the `.part` is kept for the operator. | remote-change test |
| Content type | The file must start with the CZI magic `ZISRAWFILE` | not-a-CZI test |
| SHA-256 | Computed locally over the complete file (trust on first download); the ETag is recorded with `etag_is_digest: false` and never compared with a digest | — |
| Never overwrite verified files | Pin sequence: journal → `chmod a-w` → `link(2)` into `wsi/` (fails rather than replaces) → `SHA256SUMS` rebuilt in staging and **renamed** into place → unlink the temporary → `chflags uchg`. Each step is re-checked on re-run. | five-download, crash-recovery tests |
| Concurrency | An exclusive lock file in `runs/c2-pilot/acquisition/`; a stale lock is reported, never broken automatically | lock test |
| Disk reserve | Before each attempt: free − remaining bytes ≥ 10 GB | — |
| Provenance | Append-only `transfers.jsonl` (one record per slide, written once, even across crashes): URL, Content-Length, ETag, Last-Modified, first-seen and completion time, attempts, resume offsets, verified seams, SHA-256 and its basis, size, curl version, host, platform, D-9, selection id, C2 authorization id and binding SHA-256. Every header file is archived under `acquisition/<slide>/`. | five-download test |
| Gated | Runs `preflight[download]` first and stops unless it prints GO | — |

**First-download provenance (accepted by D-10, with its limits).** EBI publishes no checksum for S-BIAD1285 files, so the SHA-256 of each of the 5 downloads is computed locally and pinned at first download.
- It proves: the bytes analysed later are the bytes received; nothing changed them after the pin.
- It does not prove: that the received bytes equal what the authors deposited. HTTPS to `ftp.ebi.ac.uk`, the `ZISRAWFILE` magic, the full Content-Length, the If-Range and seam checks and step 2 readability reduce, but do not remove, that gap.
- The recorded ETag and Last-Modified identify the server object at that time only. They are not digests and are never used as one.
- The 3 local slides are different: they are checked against the C1 inventory SHA-256 before they are pinned.
- The pilot report marks every downloaded slide "trust on first download".

`clone-local` uses the same pin sequence for the 3 local slides. The clone is made with `cp -c` (`clonefile(2)`: no data copied, fails rather than falling back to a copy across volumes), hashed, and **refused unless it equals the inventory SHA-256** before it is pinned. A record goes to `local-clones.jsonl`. `provenance` then writes the write-once `wsi-provenance.json`: one record per pilot WSI, with its source kind, cross-checked against `SHA256SUMS` and the file sizes.

## 3. Pilot commands

All steps are run as scripts (§0.1).

### Step 1. Assemble and verify the 8 WSIs

```bash
source "$PAI_DATA/c2-env.sh"
# 1a. Gate, then acquire the 5 approved WSIs. c2_wsi.py runs preflight[download] itself.
uv run --offline python $TOOLS/c2_wsi.py acquire "${COMMON[@]}" 2>&1 | tee -a $LOGS/acquire.log

# 1b. Clone the 3 local slides (APFS clone; the legacy files are only read).
uv run --offline python $TOOLS/c2_wsi.py clone-local "${COMMON[@]}" 2>&1 | tee -a $LOGS/clone.log

# 1c. Provenance for all 8, then an independent re-hash of every pin.
uv run --offline python $TOOLS/c2_wsi.py provenance --data-root "$PAI_DATA"
(cd $PAI_DATA/wsi && shasum -a 256 -c SHA256SUMS) | tee $RUN/sha256-check.txt      # 8 x OK

# 1d. Inventory over the pilot directory ONLY. A second root (legacy caches, ~/Downloads) could
#     add a content-verified slide that displaces an approved one in the verified-first ranking.
uv run pathology-ai-cohort inventory --manifest $V1 --root pilot-wsi=$PAI_DATA/wsi \
    --pattern '*.czi' --out projects/cohorts/surgen-msi/inventories > $RUN/inventory.json
INV=$(jq -er .written $RUN/inventory.json)

# 1e. Re-take the selection with that inventory. Its id WILL differ from pai-pilot-5da7534b…
#     (the id digests the inventory and per-slide availability); G13 checks equivalence.
uv run pathology-ai-cohort pilot --manifest $V1 --criteria projects/cohorts/surgen-msi/pilot.yaml \
    --inventory "$INV" --out projects/cohorts/surgen-msi/pilots > $RUN/pilot-selection.json
SEL=$(jq -er .written $RUN/pilot-selection.json)
printf 'RETAKEN_INVENTORY=%s\nRETAKEN_SELECTION=%s\n' "$INV" "$SEL" > $RUN/retaken.env

# 1f. Extraction gate (all 13 gates; G8 re-hashes, G9 checks seals, G13 checks equivalence).
uv run --offline python $TOOLS/c2_preflight.py --stage extract "${COMMON[@]}" \
    --retaken-selection "$SEL" --retaken-inventory "$INV" | tee $RUN/preflight-extract.txt
```

Accept: 1a–1c exit 0; `shasum -c` prints 8 × OK; 1f prints `preflight[extract]: GO`. The inventory and selection documents written in 1d–1e are new files in the repository; nothing existing is changed.

### Step 2. CZI readability and geometry

```bash
source "$PAI_DATA/c2-env.sh"
for s in "${SLIDES[@]}"; do
  id=$(tr '[:upper:]' '[:lower:]' <<<"$s")                # ImageId is lowercase
  uv run pathology-ai-imaging-smoke --file $PAI_DATA/wsi/$s.czi --image-id $id --modality wsi \
      --region 2,0,0,224,224 > $RUN/smoke-$s.json 2> $LOGS/smoke-$s.err
done
#   Accept (A2): backend czi-pylibczirw; level-0 spacing 0.1112 um/px; stored downsamples
#   include 4 (and 8). Record the slide id -> image id mapping in the run ledger.
```

### Step 3. Tissue detection and patch planning (no pixels, no UNI)

`pathology-ai-preprocess` runs `wsi-tissue-patch-extraction@1.0.0`. Its `tissue` stage equals the UNI pipeline's. Its `plan` and `extract` stages differ only in the values overridden below (verified by diffing the two pipeline files). With these overrides the plan configuration equals `wsi-patch-embedding-uni@1.0.0`, and no pixel is decoded.

```bash
source "$PAI_DATA/c2-env.sh"
UNI_PLAN=(--override plan.width=224 --override plan.height=224
          --override plan.max_patches=100000 --override plan.max_candidates=1000000
          --override extract.max_patches=100000 --override extract.materialize_pixels=false)
for s in "${SLIDES[@]}"; do
  id=$(tr '[:upper:]' '[:lower:]' <<<"$s")
  for k in 1 2; do                                         # twice: A3 plan-digest reproducibility
    uv run pathology-ai-preprocess --file $PAI_DATA/wsi/$s.czi --image-id $id "${UNI_PLAN[@]}" \
        > $RUN/plan-$s-$k.json 2> $LOGS/plan-$s-$k.err
  done
  a=$(jq -er .plan_digest $RUN/plan-$s-1.json); b=$(jq -er .plan_digest $RUN/plan-$s-2.json)
  [[ -n $a && $a == "$b" ]]                    # jq -e: a missing digest fails, never "null"=="null"
done
#   Accept (A3): level 2 (0.44496 um/px, within 15%); counts.not_selected_budget = 0;
#   0 < counts.selected < 100,000; identical plan_digest on both runs. After step 4, the
#   embed summary's plan_digest is compared with these; a difference is reported as a finding.
#   No planning parameter is ever tuned on pilot results.
```

### Step 4. UNI embeddings on MPS

```bash
source "$PAI_DATA/c2-env.sh"
for s in "${SLIDES[@]}"; do
  id=$(tr '[:upper:]' '[:lower:]' <<<"$s")
  df -k $PAI_DATA | tail -1 > $LOGS/disk-before-$s.txt
  /usr/bin/time -l uv run pathology-ai-embed --file $PAI_DATA/wsi/$s.czi --image-id $id \
      --pipeline wsi-patch-embedding-uni@1.0.0 \
      --output-dir $STORES --checkpoint uni=$UNI_CKPT \
      --override embed.device=mps \
      --manifest-out $RUN/embed-$s.manifest.json \
      > $RUN/embed-$s.summary.json 2> $LOGS/embed-$s.time.txt
  df -k $PAI_DATA | tail -1 > $LOGS/disk-after-$s.txt
done
#   D-7/D-9: device=mps is the ONLY override. fp32 and batch 16 are the pipeline values and are
#   never overridden. Any scale, profile, channel, resampling or batch override is outside the
#   approval. On a failure, re-run the same command (resume_verified). At most 2 retries per
#   slide; then mark the slide `failed` in the run ledger.
```

### Step 5. Verify the finalized stores (A4, A8, A9)

```bash
source "$PAI_DATA/c2-env.sh"
uv run python - $STORES <<'PY' > $RUN/store-verification.jsonl
import json, sys
from pathlib import Path
import numpy as np
from infrastructure.embedding_store import NumpyEmbeddingStoreProvider
root = Path(sys.argv[1]); provider = NumpyEmbeddingStoreProvider(root)
for d in sorted(p for p in root.iterdir() if p.is_dir()):
    store = provider.open_existing(d.name)          # read-only
    manifest = store.finalized_manifest()           # re-hashes every chunk; raises if incomplete
    v = store.verify()
    rows = store.read_rows(0, v.rows)
    print(json.dumps({"store_id": d.name, "rows": v.rows, "dim": int(rows.shape[1]),
                      "dtype": str(rows.dtype), "finite": bool(np.isfinite(rows).all()),
                      "content_digest": v.content_digest,
                      "manifest_content_digest": manifest.get("content_digest")}))
PY
#   Accept: 8 stores; rows = selected patches; dim 1024; float32; finite; digests agree.
#   A8: all 8 embedding_identity values share one SPEC-007 compatibility key.
#   A9: checkpoint digest, profile, scale binding and licence notice present in every
#   embed-*.summary.json `encoder` block.
```

### Step 6. Deterministic recovery and integrity (A5, A6, A11)

```bash
source "$PAI_DATA/c2-env.sh"
V=$PAI_DATA/embeddings/_verification/$RUN_ID
# A5: re-embed one slide into a fresh directory; the content digest must be identical.
uv run pathology-ai-embed --file $PAI_DATA/wsi/SR1482_40X_HE_T156_01.czi \
    --image-id sr1482_40x_he_t156_01 --pipeline wsi-patch-embedding-uni@1.0.0 \
    --output-dir $V/a5 --checkpoint uni=$UNI_CKPT --override embed.device=mps \
    > $RUN/a5.summary.json
same() { local x y; x=$(jq -er .content_digest "$1"); y=$(jq -er .content_digest "$2")
          [[ -n $x && $x == "$y" ]] || { echo "STOP: $1 and $2 differ ($x vs $y)" >&2; return 1; }; }
same $RUN/a5.summary.json $RUN/embed-SR1482_40X_HE_T156_01.summary.json

# A6: interrupt a long slide mid-run, then resume with the identical command.
cmd=(uv run pathology-ai-embed --file $PAI_DATA/wsi/SR386_40X_HE_T558_01.czi
     --image-id sr386_40x_he_t558_01 --pipeline wsi-patch-embedding-uni@1.0.0
     --output-dir $V/a6 --checkpoint uni=$UNI_CKPT --override embed.device=mps)
"${cmd[@]}" > $LOGS/a6-first.out 2>&1 & pid=$!
sleep 180; kill -INT $pid; wait $pid || true                # about 6,000 of 16,171 rows done
"${cmd[@]}" > $RUN/a6.summary.json                          # resumes from verified chunks
same $RUN/a6.summary.json $RUN/embed-SR386_40X_HE_T558_01.summary.json

# A11: the sources were never modified.
source $RUN/retaken.env
(cd $PAI_DATA/wsi && shasum -a 256 -c SHA256SUMS)
uv run --offline python $TOOLS/c2_preflight.py --stage extract "${COMMON[@]}" \
    --retaken-selection "$RETAKEN_SELECTION" --retaken-inventory "$RETAKEN_INVENTORY" \
    | tee $RUN/preflight-post.txt
```

### Step 7. Throughput, memory, disk and provenance

```bash
source "$PAI_DATA/c2-env.sh"
{ uname -a; sw_vers; sysctl -n machdep.cpu.brand_string hw.memsize;
  uv run python -c 'import torch,timm,numpy;print(torch.__version__,timm.__version__,numpy.__version__)';
  git rev-parse HEAD; git status --short | head -50; shasum -a 256 projects/cohorts/surgen-msi/cohort.yaml;
  shasum -a 256 projects/cohorts/surgen-msi/c2-pilot/c2-binding.json $RUN/steps/*.sh;
} > $RUN/env.txt
# Run ledger (one row per slide; append-only CSV):
#   slide_id,image_id,wsi_sha256,plan_digest,store_id,content_digest,patches,wall_s,
#   peak_rss_bytes,device_peak_bytes,ms_per_patch,store_bytes,status,attempt
# Sources: SHA256SUMS and acquisition/wsi-provenance.json; embed-*.summary.json (plan_digest, embeddings.store_id, content_digest,
# patch_counts, metrics.encoder_resources); *.time.txt ("maximum resident set size", "real");
# disk-before/after-*.txt; `du -sk $STORES/<store_id>`.
# Write the metrics as JSON/CSV under $PAI_DATA/reports/c2-pilot/$RUN_ID/ and recompute the
# C1 §1.3 cohort projection from measured ms/patch, patches per slide by sub-cohort, and bytes/row.
```

### Step 8. Pilot report

- Write `.claude/reports/008-embedding-pilot.md`. It covers A1–A11 with evidence, the run ledger, the measured resources and the updated Phase D projection.
- It quotes for every slide:
  - the WSI SHA-256 and its source, marked "trust on first download" where that applies;
  - the plan digest;
  - the store id and content digest;
  - the cohort digest `854f009f…5a3c`;
  - the re-taken selection id, stated as equivalent to `pai-pilot-5da7534b…` by G13 (not equal).
- It reports no model-quality or label result (ADR-008 D9).
- Then **STOP** for owner review. Phase D needs a separate approval.

## 4. Order and expected duration

| Step | Duration (ESTIMATE) | Network | Writes |
|---|---|---|---|
| 1a–1f | ≈ 10 h at 0.6 MB/s (≈ 1.2 h at 5 MB/s) + minutes | EBI only | ≈ 22 GB WSIs, KB of metadata |
| 2–3 | ≈ 1 min | none | JSON |
| 4 | ≈ 50 min MPS | none | ≈ 0.41 GB |
| 5–6 | ≈ 15 min | none | ≤ 0.1 GB |
| 7–8 | report writing | none | text |

## 5. What this plan does not do

- No test-partition slide is opened. G3 and the selection refuse it.
- No full-cohort processing. No manifest v2 build.
- No change to `pipelines/`, the split, or any existing manifest, inventory or selection. New documents only (the re-taken inventory and selection, and the run outputs).
- No MIL training or evaluation.
- No modification of the legacy caches, the checkpoint or sibling repositories.
- No tuning of tissue or planning parameters on pilot results.

## 6. Preparation checks

| Check | Result |
|---|---|
| `c2_preflight.py`, `c2_wsi.py` | `ruff check`, `ruff format --check`, `mypy --strict` clean |
| Mock-based tests | `tests/integration/test_c2_pilot_tools.py`: 24 passed. Real curl over real TLS against a local mock of `ftp.ebi.ac.uk` (`--connect-to`, self-signed certificate). No request left the host; no real WSI was read or written. Covers: 5 sequential pins, resume with If-Range and seam check, resume across runs with G3, ignored and shifted ranges, remote change, corrupted seam, redirect, non-CZI, crash between link and pin, crash before archive, lock, staging allowlist, APFS clone with inventory check, write-once provenance, binding tamper detection, re-taken selection equivalence and displacement. |
| Full suite | 719 passed, 53 skipped (the pre-existing opt-in real-WSI, UNI, CUDA and rebuild tests) |
| Read-only preflight on this host | see the safety review §5 |
| Commands in §3 | Flags checked against `cli.py`; the `written` key of the inventory and pilot summaries locates the new documents. **Not executed.** |

**STOP.** Execution requires D-10, the bound Phase C2 authorization (§1.2) and a GO from both preflight stages.
