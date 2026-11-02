# Plan: SPEC-008 Phase C2, Pilot Execution

**Date:** 2026-11-02
**Prompt:** PROMPT-008D9 (Pilot Infrastructure Preparation)
**Status:** **Prepared. NOT authorized. Nothing in this document has been executed.**
**Companions:**
- `008-d9-infrastructure-readiness.md`: D-9 record, directory plan, Mac readiness, estimates, blockers
- `008-phase-c1-readiness.md` §3: procedure rationale and acceptance criteria A1–A11

This plan runs the 8-slide pilot `pai-pilot-5da7534bcaa4d30c93f4eae703b3d479`:
- on the local Mac with Apple MPS (D-9);
- with `uni-official@1` at 0.5 µm/px ± 15 %, RGB 224 × 224, fp32, batch 16 (D-7, D-9);
- after acquiring the 5 pending WSIs by EBI HTTPS (D-9).

Every step **fails closed**. Any non-zero exit stops the run. A stopped step is resumed by re-running the same command, never by editing or deleting outputs.

**Before any command:**
1. D-10 is approved.
2. The owner has recorded the Phase C2 authorization (§1.2).
3. `preflight[download]` prints `GO`.

---

## 0. Shell setup (every session)

Run every block in **bash** (`bash -l`), on AC power.

```bash
cd ~/ufsc/pathology-ai
unset VIRTUAL_ENV                       # use the project .venv (uv warns otherwise)
export PAI_DATA=~/pathology-ai-data/surgen-msi
export LEGACY=../digital-pathology/data/cache/digital_pathology/SurGen   # read only
export UNI_CKPT=../digital-pathology/data/checkpoints/huggingface/uni.pth   # read only
export STORES=$PAI_DATA/embeddings/uni-official@1-0.5um-mps
export RUN_ID=c2-pilot-$(date -u +%Y%m%dT%H%M%SZ)    # set once; reuse the same value to resume
export RUN=$PAI_DATA/runs/c2-pilot/$RUN_ID LOGS=$PAI_DATA/logs/c2-pilot/$RUN_ID
export C2_AUTH=<owner decision id>      # see §1.2
export PREFLIGHT=$PAI_DATA/tools/c2_preflight.py   # copied from §1.3 (see §0.1)
export V1=projects/cohorts/surgen-msi/manifests/v1/cohort-manifest.json
SLIDES=(SR1482_40X_HE_T368_01 SR1482_40X_HE_T156_01 SR1482_40X_HE_T224_01 SR1482_40X_HE_T130_01
        SR386_40X_HE_T558_01 SR386_40X_HE_T440_01 SR386_40X_HE_T489_01 SR386_40X_HE_T044_01)
caffeinate -dimsu -w $$ &               # no sleep while this shell lives (AC power required, G12)
```

### 0.1 Create the directory plan (first session only)

```bash
mkdir -p -m 700 ~/pathology-ai-data
mkdir -p -m 700 $PAI_DATA/{staging/ebi,wsi,embeddings/_verification,runs/c2-pilot/acquisition,logs/c2-pilot,reports/c2-pilot,tools}
mkdir -p -m 700 "$STORES" "$RUN" "$LOGS" "$PAI_DATA/reports/c2-pilot/$RUN_ID"
touch ~/pathology-ai-data/.metadata_never_index          # keep Spotlight off the WSIs
# Save §1.3 as $PAI_DATA/tools/c2_preflight.py and §2.2 as $PAI_DATA/tools/c2_acquire_ebi.sh
```

## 1. Preflight gates

### 1.1 Gates (Task 6)

`c2_preflight.py` is read-only. It writes nothing, sends no request, creates no tensor and loads no model.

A gate that cannot be evaluated is **UNRESOLVED**, which counts as a failure. The exit status is 0 only when all 12 gates pass.

| Gate | Requirement | `download` stage | `extract` stage |
|---|---|---|---|
| G1 | Manifest v1 files byte-identical (3 SHA-256 pins) | ✔ | ✔ |
| G2 | Approved cohort and pilot selection: selection file SHA-256 `bc320a1b…`, `pilot.yaml` `333fe05f…`, cohort `854f009f…`, exactly the 8 slides, authorized by D-6 | ✔ | ✔ |
| G3 | No test-partition slides: selection partitions ⊆ {train, validation}. Nothing in `wsi/` except the 8 files and `SHA256SUMS`. Nothing in staging except the artifacts of the 5 approved downloads. | ✔ | ✔ |
| G4 | D-7 and D-9 approved in the ledger, with the protocol terms (`EBI HTTPS`, `MPS`, `uni-official@1`, `fp32`, `batch size 16`) | ✔ | ✔ |
| G5 | **D-10 approved**, gating `pilot` and `extraction` | ✔ | ✔ |
| G6 | **Explicit Phase C2 authorization**: `--c2-authorization ID` names an approved ledger decision gating `pilot` whose summary contains `Phase C2 execution authorized` | ✔ | ✔ |
| G7 | Free disk − planned writes ≥ **10 GB reserve**. Planned = 1 GB outputs, plus at the download stage 25 GB minus bytes already on disk | ✔ | ✔ |
| G8 | WSI content integrity. Download stage: the 3 legacy sources equal the inventory SHA-256. Extract stage: `SHA256SUMS` lists exactly the 8 files, agrees with the inventory for the 3 local slides, and every file re-hashes to its pin. | ✔ | ✔ |
| G9 | Immutable sources: every file in `wsi/`, and `wsi/` itself, has no write bit | (deferred) | ✔ |
| G10 | UNI checkpoint SHA-256 = `56ef09b4…fb40` | ✔ | ✔ |
| G11 | MPS built and available (capability flags only) | ✔ | ✔ |
| G12 | AC power (`pmset -g batt`) | ✔ | ✔ |

**Further stop conditions during the run** (proposed in C1 §3.7):
- free disk < 10 GB (re-checked before every file and every slide);
- peak RSS > 4 GiB;
- throughput more than 2× slower than 29 ms/patch.

### 1.2 Form of the Phase C2 authorization

D-9 itself is refused as an authorization: it gates only `extraction`, and its summary says that it authorizes no execution.

The owner adds an entry like the one below to `cohort.yaml`, choosing the id:

```yaml
  - id: <owner-chosen id>
    summary: >-
      Phase C2 execution authorized for pilot selection pai-pilot-5da7534b on manifest
      surgen-msi@1: EBI download of the 5 pending WSIs and UNI extraction per D-7/D-9.
    status: approved
    gates: [pilot, extraction]
    approver: ...
    approved_on: "YYYY-MM-DD"
    reference: ...
```

Consequences of adding it:
- It is a ledger change after D-9, still within the unbuilt version "2". The versioning rule in `cohort.yaml` applies.
- The data-free test `test_configuration_differs_from_v1_only_by_the_recorded_approvals` will then fail by design. It must be updated in the same change, to add the new id to the documented delta.

### 1.3 `c2_preflight.py`

```python
"""SPEC-008 Phase C2 pilot preflight (PROMPT-008D9). Read-only; fails closed.

Every gate must PASS for exit status 0. A gate that cannot be evaluated is UNRESOLVED and
counts as a failure. Nothing is written, downloaded, loaded onto a device or executed.

    uv run --offline python c2_preflight.py --stage download|extract \
        --repo . --data-root ~/pathology-ai-data/surgen-msi \
        --checkpoint ../digital-pathology/data/checkpoints/huggingface/uni.pth \
        [--legacy-root ../digital-pathology/data/cache/digital_pathology/SurGen] \
        [--c2-authorization <owner decision id>]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

GB = 10**9
RESERVE_BYTES = 10 * GB  # minimum free disk after every planned write (Task 6)
DOWNLOAD_BUDGET_BYTES = 25 * GB  # upper estimate for the 5 pending WSIs (C1: 15-25 GB)
OUTPUT_BUDGET_BYTES = 1 * GB  # stores 0.41 GB + A5/A6 copies + manifests and logs

COHORT_DIR = Path("projects/cohorts/surgen-msi")
MANIFEST_V1 = {
    "cohort-manifest.json": "f7748332c6ce866e857ad0e724cd9e24e23182f7092a2752803239da143bd084",
    "split.csv": "48105e865e68cd33f7f0c06fe8350015a8352793a3d45730e8f8fb629701f8cf",
    "reconciliation.csv": "69c3728c64ede8bc106e9ff0def9e879ab57636c836c4212cd3a361c06241781",
}
COHORT_DIGEST_V1 = "854f009fdb30e41304b9959a53bc3fd2e0beede091bd4f47061af931383a5a3c"
SELECTION_ID = "pai-pilot-5da7534bcaa4d30c93f4eae703b3d479"
SELECTION_SHA256 = "bc320a1b502e816ec5d0f04835b7cd97a5071ed9c4b026e00d6868268d464754"
PILOT_CONFIG_SHA256 = "333fe05fa510af95ef7d9774b9047f39a7e012fa7a8c14442a51bc26057c5f3f"
UNI_SHA256 = "56ef09b44a25dc5c7eedc55551b3d47bcd17659a7a33837cf9abc9ec4e2ffb40"
PILOT_PARTITIONS = {"train", "validation"}
C2_PHRASE = "Phase C2 execution authorized"  # required wording of the owner's C2 ledger entry

# The 3 slides already content-verified on this host (inventory pai-wsi-inventory-783f44e2…),
# with their paths under the legacy cache root (read only, never modified).
LOCAL = {
    "SR1482_40X_HE_T368_01": (
        "train/WSI/SR1482_40X_HE_T368_01.czi",
        "9f35f73dfc7e250efa54a343606061ef9aa2826b8ae5ddeada5e8ba08af78d8e",
    ),
    "SR1482_40X_HE_T156_01": (
        "spec0245_corpus/SR1482_40X_HE_T156_01.czi",
        "2363380e036443e6c620b0cae3149abac2c8059a40776e758fe7f4b9df8f4f20",
    ),
    "SR386_40X_HE_T558_01": (
        "validate/WSI/SR386_40X_HE_T558_01.czi",
        "619200db97a683bcba057a2fdf08c34d56209bb0cafbe712615f771c6805bee2",
    ),
}
# The 5 approved downloads (D-9). Their SHA-256 is pinned at first download (wsi/SHA256SUMS).
PENDING = (
    "SR1482_40X_HE_T224_01",
    "SR1482_40X_HE_T130_01",
    "SR386_40X_HE_T440_01",
    "SR386_40X_HE_T489_01",
    "SR386_40X_HE_T044_01",
)
PILOT = frozenset(LOCAL) | frozenset(PENDING)
STAGING_NAME = re.compile(
    r"^(?P<slide>SR(?:386|1482)_40X_HE_T\d{3}_01)\.czi(?:\.part|\.headers|\.transfer\.json)$"
)

Result = tuple[str, str]  # (PASS | FAIL | UNRESOLVED, detail)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def ledger(repo: Path) -> dict[str, dict[str, Any]]:
    doc = yaml.safe_load((repo / COHORT_DIR / "cohort.yaml").read_text())
    return {d["id"]: d for d in doc["decisions"]}


def approved(d: dict[str, Any] | None, gate: str) -> bool:
    return bool(
        d
        and d.get("status") == "approved"
        and gate in d.get("gates", [])
        and all(d.get(k) for k in ("approver", "approved_on", "reference"))
    )


def sums(data: Path) -> dict[str, str]:
    """`wsi/SHA256SUMS` (`<sha256>  <file name>`), written only by the download procedure."""
    path = data / "wsi" / "SHA256SUMS"
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        digest, name = line.split("  ", 1)
        if not re.fullmatch(r"[0-9a-f]{64}", digest) or name in out:
            raise ValueError(f"malformed SHA256SUMS line: {line!r}")
        out[name] = digest
    return out


# --------------------------------------------------------------------------- gates


def g_manifest(a: argparse.Namespace) -> Result:
    bad = [
        n for n, h in MANIFEST_V1.items() if sha256(a.repo / COHORT_DIR / "manifests/v1" / n) != h
    ]
    return ("FAIL", f"changed: {bad}") if bad else ("PASS", "manifest v1 files byte-identical")


def g_selection(a: argparse.Namespace) -> Result:
    path = a.repo / COHORT_DIR / "pilots" / SELECTION_ID / "pilot-selection.json"
    if sha256(path) != SELECTION_SHA256:
        return "FAIL", f"{SELECTION_ID} differs from its recorded SHA-256"
    if sha256(a.repo / COHORT_DIR / "pilot.yaml") != PILOT_CONFIG_SHA256:
        return "FAIL", "pilot.yaml differs from its recorded SHA-256"
    body = json.loads(path.read_text())["body"]
    ids = {s["slide_id"] for s in body["slides"]}
    if body["cohort_digest"] != COHORT_DIGEST_V1 or ids != PILOT:
        return "FAIL", "selection is not the approved 8-slide pilot of surgen-msi@1"
    if body["criteria"]["decision"] != "D-6":
        return "FAIL", "selection is not authorized by D-6"
    return "PASS", f"{SELECTION_ID}: 8 approved slides, cohort {COHORT_DIGEST_V1[:12]}"


def g_no_test(a: argparse.Namespace) -> Result:
    path = a.repo / COHORT_DIR / "pilots" / SELECTION_ID / "pilot-selection.json"
    parts = {s["partition"] for s in json.loads(path.read_text())["body"]["slides"]}
    if not parts <= PILOT_PARTITIONS:
        return "FAIL", f"selection holds partitions {sorted(parts)}"
    wsi = a.data_root / "wsi"
    names = {p.name for p in wsi.iterdir()} if wsi.is_dir() else set()
    extra = names - {f"{s}.czi" for s in PILOT} - {"SHA256SUMS"}
    staging = a.data_root / "staging" / "ebi"
    staged = {p.name for p in staging.iterdir()} if staging.is_dir() else set()
    rogue = sorted(
        n for n in staged if not ((m := STAGING_NAME.match(n)) and m["slide"] in PENDING)
    )
    if extra or rogue:
        return "FAIL", f"unexpected files: wsi={sorted(extra)} staging={rogue}"
    return "PASS", f"partitions {sorted(parts)}; no file outside the 8 pilot slides"


def g_decisions(a: argparse.Namespace) -> Result:
    d = ledger(a.repo)
    d7, d9 = d.get("D-7"), d.get("D-9")
    if not approved(d7, "extraction") or "0.5 um/px" not in str(d7 and d7["summary"]):
        return "FAIL", "D-7 (0.5 um/px +-15%) is not approved"
    terms = ("EBI HTTPS", "MPS", "uni-official@1", "fp32", "batch size 16")
    if not approved(d9, "extraction") or not all(t in str(d9 and d9["summary"]) for t in terms):
        return "FAIL", "D-9 (EBI HTTPS, local Mac MPS, fp32, batch 16) is not approved"
    return "PASS", "D-7 and D-9 approved"


def g_d10(a: argparse.Namespace) -> Result:
    d10 = ledger(a.repo).get("D-10")
    if approved(d10, "pilot") and approved(d10, "extraction"):
        return "PASS", f"D-10 approved on {d10['approved_on']}"  # type: ignore[index]
    return "FAIL", "D-10 (UNI terms, SurGen data terms) is not approved"


def g_c2(a: argparse.Namespace) -> Result:
    if not a.c2_authorization:
        return "FAIL", "no Phase C2 execution authorization given (--c2-authorization)"
    d = ledger(a.repo).get(a.c2_authorization)
    if not (approved(d, "pilot") and C2_PHRASE in str(d and d["summary"])):
        return (
            "FAIL",
            f"{a.c2_authorization} is not an approved {C2_PHRASE!r} decision gating pilot",
        )
    return "PASS", f"{a.c2_authorization} approved on {d['approved_on']}"  # type: ignore[index]


def g_disk(a: argparse.Namespace) -> Result:
    probe = a.data_root if a.data_root.exists() else a.data_root.parent
    while not probe.exists():
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    need = OUTPUT_BUDGET_BYTES
    if a.stage == "download":
        on_disk = 0
        for slide in PENDING:
            for p in (
                a.data_root / "wsi" / f"{slide}.czi",
                a.data_root / "staging/ebi" / f"{slide}.czi.part",
            ):
                on_disk += p.stat().st_size if p.exists() else 0
        need += max(0, DOWNLOAD_BUDGET_BYTES - on_disk)
    detail = f"free {free / GB:.1f} GB; planned writes {need / GB:.1f} GB; "
    detail += f"reserve {RESERVE_BYTES // GB} GB"
    return ("PASS" if free - need >= RESERVE_BYTES else "FAIL"), detail


def g_wsi(a: argparse.Namespace) -> Result:
    if a.stage == "download":  # the 3 local sources must still be the inventoried bytes
        if a.legacy_root is None:
            return "UNRESOLVED", "--legacy-root not given"
        bad = [s for s, (rel, h) in LOCAL.items() if sha256(a.legacy_root / rel) != h]
        return (
            ("FAIL", f"changed: {bad}")
            if bad
            else ("PASS", "3 local pilot WSIs match the inventory")
        )
    pins = sums(a.data_root)
    expected = {f"{s}.czi" for s in PILOT}
    if set(pins) != expected:
        return "FAIL", f"SHA256SUMS lists {len(pins)} files, not the 8 pilot WSIs"
    for slide, (_, h) in LOCAL.items():
        if pins[f"{slide}.czi"] != h:
            return "FAIL", f"{slide}: SHA256SUMS disagrees with the inventory"
    bad = [n for n, h in sorted(pins.items()) if sha256(a.data_root / "wsi" / n) != h]
    return ("FAIL", f"content changed: {bad}") if bad else ("PASS", "8/8 WSIs match their SHA-256")


def g_immutable(a: argparse.Namespace) -> Result:
    if a.stage == "download":
        return "PASS", "no verified downloads yet; checked again before extraction"
    wsi = a.data_root / "wsi"
    writable = [p.name for p in sorted(wsi.iterdir()) if p.stat().st_mode & 0o222]
    if wsi.stat().st_mode & 0o222:
        writable.append("wsi/ (directory)")
    return (
        ("FAIL", f"writable: {writable}")
        if writable
        else ("PASS", "all source WSIs and wsi/ are a-w")
    )


def g_checkpoint(a: argparse.Namespace) -> Result:
    if a.checkpoint is None or not a.checkpoint.is_file():
        return "FAIL", "UNI checkpoint not found"
    got = sha256(a.checkpoint)
    return (
        ("PASS", f"SHA-256 {got[:12]}… = pin")
        if got == UNI_SHA256
        else ("FAIL", f"SHA-256 {got[:12]}… != pin")
    )


def g_mps(a: argparse.Namespace) -> Result:
    # Capability flags only: no tensor is created and no kernel is launched.
    import torch

    ok = torch.backends.mps.is_built() and torch.backends.mps.is_available()
    return (
        ("PASS", f"torch {torch.__version__}: MPS available") if ok else ("FAIL", "MPS unavailable")
    )


def g_power(a: argparse.Namespace) -> Result:
    out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, check=True).stdout
    return ("PASS", "AC power") if "'AC Power'" in out else ("FAIL", "running on battery")


GATES: tuple[tuple[str, str, Callable[[argparse.Namespace], Result]], ...] = (
    ("G1", "manifest v1 immutable", g_manifest),
    ("G2", "approved cohort and pilot selection", g_selection),
    ("G3", "no test-partition slide, no unexpected file", g_no_test),
    ("G4", "D-7 and D-9 approved", g_decisions),
    ("G5", "D-10 licensing approved", g_d10),
    ("G6", "explicit Phase C2 execution authorization", g_c2),
    ("G7", "free disk >= planned writes + 10 GB reserve", g_disk),
    ("G8", "WSI content integrity", g_wsi),
    ("G9", "source WSIs immutable", g_immutable),
    ("G10", "UNI checkpoint integrity", g_checkpoint),
    ("G11", "MPS available", g_mps),
    ("G12", "AC power", g_power),
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--stage", choices=("download", "extract"), required=True)
    p.add_argument("--repo", type=Path, default=Path("."))
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--legacy-root", type=Path)
    p.add_argument("--c2-authorization")
    a = p.parse_args()
    a.repo, a.data_root = a.repo.expanduser().resolve(), a.data_root.expanduser()
    a.checkpoint = a.checkpoint and a.checkpoint.expanduser()
    a.legacy_root = a.legacy_root and a.legacy_root.expanduser()
    failed = 0
    for gid, name, gate in GATES:
        try:
            status, detail = gate(a)
        except Exception as exc:  # fail closed: an unevaluable gate never passes
            status, detail = "UNRESOLVED", f"{type(exc).__name__}: {exc}"
        failed += status != "PASS"
        print(f"{gid:<4} {status:<10} {name}: {detail}")
    print(f"preflight[{a.stage}]: {'GO' if not failed else f'NO-GO ({failed} gate(s) not passed)'}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
```

## 2. Acquisition (EBI HTTPS, 5 files)

### 2.1 Allowlist

The URLs below are the only acquisition targets. Source: legacy `data/metadata/SurGen/MMR_MSI/download_{train,validate}.txt` (VERIFIED, read locally).

| Slide | Partition | Label | URL |
|---|---|---|---|
| SR1482_40X_HE_T224_01 | train | positive | `https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/285/S-BIAD1285/Files/SR1482_WSIs/SR1482_40X_HE_T224_01.czi` |
| SR1482_40X_HE_T130_01 | train | positive | `https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/285/S-BIAD1285/Files/SR1482_WSIs/SR1482_40X_HE_T130_01.czi` |
| SR386_40X_HE_T440_01 | train | negative | `https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/285/S-BIAD1285/Files/SR386_WSIs/SR386_40X_HE_T440_01.czi` |
| SR386_40X_HE_T489_01 | train | positive | `https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/285/S-BIAD1285/Files/SR386_WSIs/SR386_40X_HE_T489_01.czi` |
| SR386_40X_HE_T044_01 | validation | positive | `https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/285/S-BIAD1285/Files/SR386_WSIs/SR386_40X_HE_T044_01.czi` |

### 2.2 `c2_acquire_ebi.sh`

How the script meets each requirement (Task 4):

| Requirement | How |
|---|---|
| Only the 5 approved files | Hard-coded allowlist; the URL basename must equal `<slide>.czi` |
| Unexpected source identity refused | `--max-redirs 0` and `--proto '=https'`: a redirect to another host or path fails the request |
| `.part` temporaries | Writes `staging/ebi/<slide>.czi.part` only |
| Interrupted transfer recovery | `curl --continue-at -` from the `.part` size; up to 5 attempts; a re-run of the script resumes. Refused when the server does not accept byte ranges. |
| Remote change detected | Content-Length, ETag and Last-Modified from the first HEAD are kept in `<slide>.czi.transfer.json`. Any difference on a later attempt **refuses** the resume. The operator must quarantine the `.part` by hand. |
| Size validated | Final size must equal `Content-Length`. A `.part` larger than that is refused. |
| SHA-256 validated | `shasum -a 256` of the complete local file, recomputed after the move. The ETag is stored with `etag_is_digest: false` and never compared with a digest. |
| Never overwrite verified files | An existing `wsi/<slide>.czi` is skipped. `mv -n` never clobbers. A slide already in `SHA256SUMS` is refused. Files and `wsi/` are made `a-w`. |
| Disk reserve | Before each transfer: free − remaining bytes ≥ 10 GB |
| Provenance | Append-only `runs/c2-pilot/acquisition/transfers.jsonl` with: slide id, URL, Content-Length, ETag (metadata), Last-Modified, SHA-256 and its source, size, attempts, first seen and completion time (UTC), curl version, host, D-9, selection id. Plus `.headers` and `.curl.log` (HTTP code, bytes, speed, time) per attempt. |
| Gated | Runs `preflight[download]` first and exits unless it prints GO |

```bash
#!/usr/bin/env bash
# SPEC-008 Phase C2: resumable EBI HTTPS acquisition of the 5 approved pilot WSIs (D-9).
# NOT AUTHORIZED TO RUN until preflight[download] is GO (D-10 + explicit Phase C2 authorization).
#
#   C2_AUTH=<owner decision id> PAI_DATA=~/pathology-ai-data/surgen-msi \
#   REPO=~/ufsc/pathology-ai PREFLIGHT=<path>/c2_preflight.py bash c2_acquire_ebi.sh
#
# Guarantees: only the 5 allowlisted URLs; <name>.part temporaries; resume by HTTP range;
# size = Content-Length; SHA-256 computed locally (an ETag is recorded as metadata, never as
# a digest); a verified file is never overwritten; remote change between attempts -> refuse.
set -euo pipefail
umask 077

: "${C2_AUTH:?owner Phase C2 authorization id required}"
: "${PAI_DATA:?data root required}" "${REPO:?repository required}" "${PREFLIGHT:?preflight required}"
BASE="https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/285/S-BIAD1285/Files"
ALLOW=(
  "SR1482_40X_HE_T224_01|$BASE/SR1482_WSIs/SR1482_40X_HE_T224_01.czi"
  "SR1482_40X_HE_T130_01|$BASE/SR1482_WSIs/SR1482_40X_HE_T130_01.czi"
  "SR386_40X_HE_T440_01|$BASE/SR386_WSIs/SR386_40X_HE_T440_01.czi"
  "SR386_40X_HE_T489_01|$BASE/SR386_WSIs/SR386_40X_HE_T489_01.czi"
  "SR386_40X_HE_T044_01|$BASE/SR386_WSIs/SR386_40X_HE_T044_01.czi"
)
RESERVE=$((10 * 1000 * 1000 * 1000))
MAX_ATTEMPTS=5
STAGE="$PAI_DATA/staging/ebi" WSI="$PAI_DATA/wsi" PROV="$PAI_DATA/runs/c2-pilot/acquisition"
CURL=(curl --silent --show-error --fail --proto '=https' --proto-redir '=https' --max-redirs 0
      --tlsv1.2 --connect-timeout 60)

die() { echo "acquire: REFUSED: $*" >&2; exit 2; }
now() { date -u +%Y-%m-%dT%H:%M:%SZ; }
header() { awk -v k="$(tr '[:upper:]' '[:lower:]' <<<"$1")" -F': ' \
  'tolower($1)==k {sub(/\r$/,"",$2); v=$2} END {print v}' "$2"; }
free_bytes() { df -k "$PAI_DATA" | awk 'NR==2 {print $4 * 1024}'; }

(cd "$REPO" && uv run --offline python "$PREFLIGHT" --stage download --repo . \
   --data-root "$PAI_DATA" --checkpoint "$REPO/../digital-pathology/data/checkpoints/huggingface/uni.pth" \
   --legacy-root "$REPO/../digital-pathology/data/cache/digital_pathology/SurGen" \
   --c2-authorization "$C2_AUTH") || die "preflight[download] is NO-GO"

mkdir -p "$STAGE" "$WSI" "$PROV"
for entry in "${ALLOW[@]}"; do
  slide="${entry%%|*}" url="${entry#*|}" name="${slide}.czi"
  [[ "$(basename "$url")" == "$name" ]] || die "$url does not name $name"
  part="$STAGE/$name.part" meta="$STAGE/$name.transfer.json" hdr="$STAGE/$name.headers"
  if [[ -e "$WSI/$name" ]]; then
    grep -q "  $name\$" "$WSI/SHA256SUMS" 2>/dev/null || die "$name exists in wsi/ without a pin"
    echo "acquire: $name already verified; never re-downloaded or overwritten"; continue
  fi
  for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    # 1. Remote metadata (HEAD). No redirect is followed: the source identity is the exact URL.
    "${CURL[@]}" --head --dump-header "$hdr" --output /dev/null "$url"
    length="$(header Content-Length "$hdr")" etag="$(header ETag "$hdr")"
    modified="$(header Last-Modified "$hdr")" ranges="$(header Accept-Ranges "$hdr")"
    [[ "$length" =~ ^[0-9]+$ && "$length" -gt 0 ]] || die "$name: no Content-Length"
    if [[ -f "$meta" ]]; then  # the remote object must not change under a partial transfer
      python3 - "$meta" "$length" "$etag" "$modified" <<'PY' || die "$name: remote changed since the first attempt; quarantine the .part by hand"
import json, sys
m = json.load(open(sys.argv[1]))
sys.exit(0 if (str(m["content_length"]), m["etag"], m["last_modified"]) == tuple(sys.argv[2:5]) else 1)
PY
    else
      SLIDE="$slide" URL="$url" LEN="$length" ETAG="$etag" MOD="$modified" T="$(now)" \
      python3 -c 'import json,os; e=os.environ; print(json.dumps({"slide_id":e["SLIDE"],"url":e["URL"],
"content_length":int(e["LEN"]),"etag":e["ETAG"],"etag_is_digest":False,"last_modified":e["MOD"],
"first_seen_utc":e["T"],"decision":"D-9","selection":"pai-pilot-5da7534bcaa4d30c93f4eae703b3d479"},indent=1))' >"$meta"
    fi
    have=0; [[ -f "$part" ]] && have="$(stat -f %z "$part")"
    (( have <= length )) || die "$name: .part larger than Content-Length"
    [[ "$have" == 0 || "$ranges" == bytes ]] || die "$name: server does not accept ranges; cannot resume"
    (( $(free_bytes) - (length - have) >= RESERVE )) || die "$name: would breach the 10 GB reserve"
    # 2. Transfer (resumes from the .part size).
    if (( have < length )); then
      "${CURL[@]}" --continue-at - --speed-limit 10240 --speed-time 300 \
        --dump-header "$hdr.get.$attempt" --output "$part" \
        --write-out '%{http_code} %{size_download} %{speed_download} %{time_total}\n' "$url" \
        >>"$STAGE/$name.curl.log" || { echo "acquire: $name attempt $attempt interrupted"; sleep 30; continue; }
    fi
    [[ "$(stat -f %z "$part")" == "$length" ]] || { echo "acquire: $name short; resuming"; continue; }
    # 3. Content identity: SHA-256 of the complete local bytes (trust on first download).
    sha="$(shasum -a 256 "$part" | awk '{print $1}')"
    chmod u+w "$WSI"
    ! grep -q "  $name\$" "$WSI/SHA256SUMS" 2>/dev/null || die "$name already pinned"
    mv -n "$part" "$WSI/$name"
    [[ ! -e "$part" && "$(shasum -a 256 "$WSI/$name" | awk '{print $1}')" == "$sha" ]] \
      || die "$name: move failed or content changed"
    printf '%s  %s\n' "$sha" "$name" >>"$WSI/SHA256SUMS"
    chmod a-w "$WSI/$name" "$WSI/SHA256SUMS" "$WSI"
    # 4. Provenance (append-only).
    python3 - "$meta" "$sha" "$attempt" "$(now)" "$(curl --version | head -1)" "$(hostname -s)" \
      >>"$PROV/transfers.jsonl" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
m.update(sha256=sys.argv[2], sha256_source="local, full file", attempts=int(sys.argv[3]),
         completed_utc=sys.argv[4], client=sys.argv[5], host=sys.argv[6], size_bytes=m["content_length"])
print(json.dumps(m, sort_keys=True))
PY
    echo "acquire: $name verified ($length B, sha256 $sha)"; continue 2
  done
  die "$name: not complete after $MAX_ATTEMPTS attempts (the .part is kept for a later resume)"
done
echo "acquire: 5/5 approved WSIs verified and read-only"
```

## 3. Pilot commands

All commands are run from the repository root, in the shell of §0.

### Step 1. Assemble and verify the 8 WSIs

```bash
# 1a. Gate, then acquire the 5 approved WSIs (§2.2). The script runs preflight[download] itself.
REPO=$PWD bash $PAI_DATA/tools/c2_acquire_ebi.sh 2>&1 | tee -a $LOGS/acquire.log

# 1b. Clone the 3 local slides (APFS clone: no data copied; the legacy files are only read).
chmod u+w $PAI_DATA/wsi
for rel in train/WSI/SR1482_40X_HE_T368_01.czi spec0245_corpus/SR1482_40X_HE_T156_01.czi \
           validate/WSI/SR386_40X_HE_T558_01.czi; do
  name=$(basename $rel)
  [[ -e $PAI_DATA/wsi/$name ]] && { echo "keep $name"; continue; }   # never overwrite
  cp -c -n "$LEGACY/$rel" "$PAI_DATA/wsi/$name"                      # fails if cloning is impossible
  printf '%s  %s\n' "$(shasum -a 256 "$PAI_DATA/wsi/$name" | cut -d' ' -f1)" "$name" \
      >> $PAI_DATA/wsi/SHA256SUMS
  chmod a-w "$PAI_DATA/wsi/$name"
done
chmod a-w $PAI_DATA/wsi/SHA256SUMS $PAI_DATA/wsi
(cd $PAI_DATA/wsi && shasum -a 256 -c SHA256SUMS)                    # 8 x OK

# 1c. Inventory against manifest v1. The new root sits beside the C1 roots, so copies are cross-checked.
uv run pathology-ai-cohort inventory --manifest $V1 \
    --root pilot-wsi=$PAI_DATA/wsi --root legacy-cache=$LEGACY \
    --root legacy-cache-copy=$HOME/projects/digital-pathology/data/cache/digital_pathology/SurGen \
    --root downloads=$HOME/Downloads \
    --pattern '*.czi' --out projects/cohorts/surgen-msi/inventories | tee $RUN/inventory.json
#   Accept: all 8 pilot slides `verified`; no inconsistent copies; no duplicate content;
#   no `unmatched` partial file. The size of each download equals its Content-Length.

# 1d. Re-take the selection with the new inventory. It must name the same 8 slides, status `ready`.
uv run pathology-ai-cohort pilot --manifest $V1 --criteria projects/cohorts/surgen-msi/pilot.yaml \
    --inventory projects/cohorts/surgen-msi/inventories/<new inventory id>/wsi-inventory.json \
    --out projects/cohorts/surgen-msi/pilots | tee $RUN/pilot-selection.json

# 1e. Extraction gate (all 12 gates; G8 re-hashes the 8 files and G9 checks immutability).
uv run python $PREFLIGHT --stage extract --data-root $PAI_DATA --checkpoint $UNI_CKPT \
    --c2-authorization $C2_AUTH | tee $RUN/preflight-extract.txt
```

### Step 2. CZI readability and geometry

```bash
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
UNI_PLAN=(--override plan.width=224 --override plan.height=224
          --override plan.max_patches=100000 --override plan.max_candidates=1000000
          --override extract.max_patches=100000 --override extract.materialize_pixels=false)
for s in "${SLIDES[@]}"; do
  id=$(tr '[:upper:]' '[:lower:]' <<<"$s")
  for k in 1 2; do                                         # twice: A3 plan-digest reproducibility
    uv run pathology-ai-preprocess --file $PAI_DATA/wsi/$s.czi --image-id $id "${UNI_PLAN[@]}" \
        > $RUN/plan-$s-$k.json 2> $LOGS/plan-$s-$k.err
  done
  cmp <(jq -r .plan_digest $RUN/plan-$s-1.json) <(jq -r .plan_digest $RUN/plan-$s-2.json)
done
#   Accept (A3): level 2 (0.44496 um/px, within 15%); counts.not_selected_budget = 0;
#   0 < counts.selected < 100,000; identical plan_digest on both runs. After step 4, the
#   embed summary's plan_digest is compared with these; a difference is reported as a finding.
#   No planning parameter is ever tuned on pilot results.
```

### Step 4. UNI embeddings on MPS

```bash
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
uv run python - $STORES <<'PY' | tee $RUN/store-verification.jsonl
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
V=$PAI_DATA/embeddings/_verification/$RUN_ID
# A5: re-embed one slide into a fresh directory; the content digest must be identical.
uv run pathology-ai-embed --file $PAI_DATA/wsi/SR1482_40X_HE_T156_01.czi \
    --image-id sr1482_40x_he_t156_01 --pipeline wsi-patch-embedding-uni@1.0.0 \
    --output-dir $V/a5 --checkpoint uni=$UNI_CKPT --override embed.device=mps \
    > $RUN/a5.summary.json
cmp <(jq -r .content_digest $RUN/a5.summary.json) \
    <(jq -r .content_digest $RUN/embed-SR1482_40X_HE_T156_01.summary.json)

# A6: interrupt a long slide mid-run, then resume with the identical command.
cmd=(uv run pathology-ai-embed --file $PAI_DATA/wsi/SR386_40X_HE_T558_01.czi
     --image-id sr386_40x_he_t558_01 --pipeline wsi-patch-embedding-uni@1.0.0
     --output-dir $V/a6 --checkpoint uni=$UNI_CKPT --override embed.device=mps)
"${cmd[@]}" > $LOGS/a6-first.out 2>&1 & pid=$!
sleep 180; kill -INT $pid; wait $pid || true                # about 6,000 of 16,171 rows done
"${cmd[@]}" > $RUN/a6.summary.json                          # resumes from verified chunks
cmp <(jq -r .content_digest $RUN/a6.summary.json) \
    <(jq -r .content_digest $RUN/embed-SR386_40X_HE_T558_01.summary.json)

# A11: the sources were never modified.
(cd $PAI_DATA/wsi && shasum -a 256 -c SHA256SUMS)
uv run python $PREFLIGHT --stage extract --data-root $PAI_DATA --checkpoint $UNI_CKPT \
    --c2-authorization $C2_AUTH | tee $RUN/preflight-post.txt
```

### Step 7. Throughput, memory, disk and provenance

```bash
{ uname -a; sw_vers; sysctl -n machdep.cpu.brand_string hw.memsize;
  uv run python -c 'import torch,timm,numpy;print(torch.__version__,timm.__version__,numpy.__version__)';
  git rev-parse HEAD 2>/dev/null; git status --short | head -50; shasum -a 256 projects/cohorts/surgen-msi/cohort.yaml;
} > $RUN/env.txt
# Run ledger (one row per slide; append-only CSV):
#   slide_id,image_id,wsi_sha256,plan_digest,store_id,content_digest,patches,wall_s,
#   peak_rss_bytes,device_peak_bytes,ms_per_patch,store_bytes,status,attempt
# Sources: SHA256SUMS; embed-*.summary.json (plan_digest, embeddings.store_id, content_digest,
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
  - the new selection id.
- It reports no model-quality or label result (ADR-008 D9).
- Then **STOP** for owner review. Phase D needs a separate approval.

## 4. Order and expected duration

| Step | Duration (ESTIMATE) | Network | Writes |
|---|---|---|---|
| 0–1e | ≈ 10 h at 0.6 MB/s (≈ 1.2 h at 5 MB/s) + minutes | EBI only | ≈ 22 GB WSIs, KB of metadata |
| 2–3 | ≈ 1 min | none | JSON |
| 4 | ≈ 50 min MPS | none | ≈ 0.41 GB |
| 5–6 | ≈ 15 min | none | ≤ 0.1 GB |
| 7–8 | report writing | none | text |

## 5. What this plan does not do

- No test-partition slide is opened. G3 and the selection refuse it.
- No full-cohort processing. No manifest v2 build.
- No change to `pipelines/`, the split, or any existing manifest, inventory or selection. New documents only.
- No MIL training or evaluation.
- No modification of the legacy caches, the checkpoint or sibling repositories.
- No tuning of tissue or planning parameters on pilot results.

## 6. Preparation checks (this task)

| Check | Result |
|---|---|
| `c2_preflight.py` | ruff (E, F, W, I, B, UP) clean; `ruff format` clean; `mypy --strict` clean. Read-only run: `download` → NO-GO (G5, G6, G12); `extract` → NO-GO (G5, G6, G8, G9, G12). `--c2-authorization D-9` is refused. |
| `c2_acquire_ebi.sh` | `bash -n` clean (system bash 3.2). Without `C2_AUTH` it exits before doing anything. **Not run**; no request was sent. |
| Commands in §3 | Flags checked against `cli.py` (`pathology-ai-embed`, `pathology-ai-preprocess`, `pathology-ai-imaging-smoke`, `pathology-ai-cohort inventory/pilot`), the two pipeline files and `NumpyEmbeddingStoreProvider`. `jq` is present (`/opt/homebrew/bin/jq`). **Not executed.** |

**STOP.** Execution requires D-10, the Phase C2 authorization (§1.2) and a GO from both preflight stages.
