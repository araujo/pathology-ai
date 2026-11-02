"""SPEC-008 Phase C2 pilot preflight (PROMPT-008D9, revised PROMPT-008D10). Read-only; fails closed.

Every gate must PASS for exit status 0. A gate that cannot be evaluated is UNRESOLVED and
counts as a failure. Nothing is written, downloaded, loaded onto a device or executed.

    uv run --offline python projects/cohorts/surgen-msi/c2-pilot/c2_preflight.py \
        --stage download|extract --data-root ~/pathology-ai-data/surgen-msi \
        --checkpoint ../digital-pathology/data/checkpoints/huggingface/uni.pth \
        [--legacy-root ../digital-pathology/data/cache/digital_pathology/SurGen] \
        [--c2-authorization <owner decision id>] \
        [--retaken-selection <pilots/<id>/pilot-selection.json>] \
        [--retaken-inventory <inventories/<id>/wsi-inventory.json>]

    ... c2_preflight.py --print-binding     # the c2-binding document the owner authorizes

The Phase C2 authorization is bound to exact bytes: the owner's ledger entry names the SHA-256
of `c2-binding.json`, which lists the SHA-256 of every repository file the run depends on
(manifest v1, the approved selection and inventory, the pilot criteria, both pipelines, these
tools and the execution plan) and the external pins. Any later edit to one of them is a FAIL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
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
TOOLS_DIR = COHORT_DIR / "c2-pilot"
BINDING_FILE = TOOLS_DIR / "c2-binding.json"
BINDING_SCHEMA = "pathology-ai.c2-binding/1"
MANIFEST_V1 = {
    "cohort-manifest.json": "f7748332c6ce866e857ad0e724cd9e24e23182f7092a2752803239da143bd084",
    "split.csv": "48105e865e68cd33f7f0c06fe8350015a8352793a3d45730e8f8fb629701f8cf",
    "reconciliation.csv": "69c3728c64ede8bc106e9ff0def9e879ab57636c836c4212cd3a361c06241781",
}
COHORT_DIGEST_V1 = "854f009fdb30e41304b9959a53bc3fd2e0beede091bd4f47061af931383a5a3c"
SELECTION_ID = "pai-pilot-5da7534bcaa4d30c93f4eae703b3d479"
SELECTION_SHA256 = "bc320a1b502e816ec5d0f04835b7cd97a5071ed9c4b026e00d6868268d464754"
INVENTORY_ID = "pai-wsi-inventory-783f44e2ea71bead23ce78c80f069e12"
PILOT_CONFIG_SHA256 = "333fe05fa510af95ef7d9774b9047f39a7e012fa7a8c14442a51bc26057c5f3f"
UNI_SHA256 = "56ef09b44a25dc5c7eedc55551b3d47bcd17659a7a33837cf9abc9ec4e2ffb40"
PILOT_PARTITIONS = {"train", "validation"}
C2_PHRASE = "Phase C2 execution authorized"  # required wording of the owner's C2 ledger entry
BINDING_REF = re.compile(r"c2-binding sha256 ([0-9a-f]{64})")
# The only root of the post-acquisition inventory. With no other root, exactly the 8 pilot
# slides are content-verified, so the verified-first selection must re-take the same 8 slides
# whatever else is on this host (~/Downloads, legacy caches).
RETAKEN_ROOT = "pilot-wsi"

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
SUMS = "SHA256SUMS"
SUMS_NEXT = "SHA256SUMS.next"  # staged replacement of wsi/SHA256SUMS (renamed into place)
# Every file c2_wsi.py may leave in staging/ebi while a slide is in flight. Nothing else is
# accepted there, and at the extract stage staging must be empty (artifacts are archived).
STAGING_ARTIFACT = re.compile(
    r"^(?P<slide>SR(?:386|1482)_40X_HE_T\d{3}_01)\.czi\."
    r"(?:part|clone|transfer\.json|verified\.json|a\d{3}\.(?:head|get)|seam\d{2}\.(?:headers|bytes))"
    r"(?:\.tmp)?$"
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


def parse_sums(text: str) -> dict[str, str]:
    """`<sha256>  <file name>` lines; a malformed or repeated line raises."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        digest, sep, name = line.partition("  ")
        if not sep or not re.fullmatch(r"[0-9a-f]{64}", digest) or not name or name in out:
            raise ValueError(f"malformed {SUMS} line: {line!r}")
        out[name] = digest
    return out


def sums(data: Path) -> dict[str, str]:
    """`wsi/SHA256SUMS`, written only by c2_wsi.py; an absent file is an empty pin set."""
    path = data / "wsi" / SUMS
    return parse_sums(path.read_text()) if path.exists() else {}


def journal_exists(data: Path, name: str) -> bool:
    return (data / "staging" / "ebi" / f"{name}.verified.json").is_file()


def sealed(path: Path) -> bool:
    """No write bit; a WSI also carries the user-immutable flag where the OS has one (macOS).
    SHA256SUMS is only a-w: it is replaced by rename when a pin is added."""
    st = path.lstat()
    if st.st_mode & 0o222:
        return False
    flags = getattr(st, "st_flags", stat.UF_IMMUTABLE)
    return path.suffix != ".czi" or bool(flags & stat.UF_IMMUTABLE)


# --------------------------------------------------------------------------- binding


def bound_files() -> tuple[Path, ...]:
    return (
        *(COHORT_DIR / "manifests/v1" / n for n in sorted(MANIFEST_V1)),
        COHORT_DIR / "pilot.yaml",
        COHORT_DIR / "pilots" / SELECTION_ID / "pilot-selection.json",
        COHORT_DIR / "inventories" / INVENTORY_ID / "wsi-inventory.json",
        Path("pipelines/wsi-patch-embedding-uni.v1.yaml"),
        Path("pipelines/wsi-tissue-patch-extraction.v1.yaml"),
        TOOLS_DIR / "c2_preflight.py",
        TOOLS_DIR / "c2_wsi.py",
        Path(".claude/reports/008-phase-c2-execution-plan.md"),
    )


def binding_document(repo: Path) -> dict[str, Any]:
    """Deterministic function of the bound files and the pins in this module."""
    from c2_wsi import ALLOW  # same directory; imported lazily to keep the gates standalone

    return {
        "schema": BINDING_SCHEMA,
        "selection_id": SELECTION_ID,
        "cohort_digest": COHORT_DIGEST_V1,
        "files": {p.as_posix(): sha256(repo / p) for p in bound_files()},
        "uni_checkpoint_sha256": UNI_SHA256,
        "local_sources": {s: {"legacy_path": rel, "sha256": h} for s, (rel, h) in LOCAL.items()},
        "downloads": dict(sorted(ALLOW.items())),
        "decisions_required": ["D-6", "D-7", "D-9", "D-10"],
    }


def canonical(doc: dict[str, Any]) -> bytes:
    return (json.dumps(doc, indent=1, sort_keys=True, ensure_ascii=True) + "\n").encode()


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
    extra = names - {f"{s}.czi" for s in PILOT} - {SUMS}
    staging = a.data_root / "staging" / "ebi"
    staged = sorted(p.name for p in staging.iterdir()) if staging.is_dir() else []
    if a.stage == "extract":
        rogue = staged  # every staging artifact is archived once its slide is pinned
    else:
        rogue = [
            n
            for n in staged
            if n != SUMS_NEXT and not ((m := STAGING_ARTIFACT.match(n)) and m["slide"] in PILOT)
        ]
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


def check_binding(repo: Path, expected_sha: str) -> list[str]:
    """Problems with c2-binding.json against the authorized digest and the current files."""
    path = repo / BINDING_FILE
    if not path.is_file():
        return [f"{BINDING_FILE} not found"]
    if sha256(path) != expected_sha:
        return [f"{BINDING_FILE} SHA-256 differs from the authorized {expected_sha[:12]}…"]
    recorded = json.loads(path.read_text())
    current = binding_document(repo)
    problems = [
        f"changed since authorization: {k}"
        for k in sorted(set(recorded.get("files", {})) | set(current["files"]))
        if recorded.get("files", {}).get(k) != current["files"].get(k)
    ]
    problems += [
        f"binding field differs: {k}"
        for k in sorted(set(recorded) | set(current))
        if k != "files" and recorded.get(k) != current.get(k)
    ]
    here = Path(__file__).resolve()
    if here != (repo / TOOLS_DIR / "c2_preflight.py").resolve():
        problems.append(f"preflight runs from {here}, not from the bound repository file")
    return problems


def g_c2(a: argparse.Namespace) -> Result:
    if not a.c2_authorization:
        return "FAIL", "no Phase C2 execution authorization given (--c2-authorization)"
    d = ledger(a.repo).get(a.c2_authorization)
    summary = " ".join(str(d and d.get("summary", "")).split())
    if not (approved(d, "pilot") and approved(d, "extraction") and C2_PHRASE in summary):
        return (
            "FAIL",
            f"{a.c2_authorization} is not an approved {C2_PHRASE!r} decision gating pilot and "
            "extraction",
        )
    if SELECTION_ID[:18] not in summary:
        return "FAIL", f"{a.c2_authorization} does not name selection {SELECTION_ID[:18]}"
    ref = BINDING_REF.search(summary)
    if ref is None:
        return "FAIL", f"{a.c2_authorization} does not name 'c2-binding sha256 <64 hex>'"
    problems = check_binding(a.repo, ref[1])
    if problems:
        return "FAIL", "; ".join(problems)
    return (
        "PASS",
        (
            f"{a.c2_authorization} approved on {d['approved_on']}; "  # type: ignore[index]
            f"binding {ref[1][:12]}… matches every bound file"
        ),
    )


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
    pins = sums(a.data_root)
    wsi = a.data_root / "wsi"
    if set(pins) - {f"{s}.czi" for s in PILOT}:
        return "FAIL", f"{SUMS} pins files outside the pilot"
    for slide, (_, h) in LOCAL.items():
        if pins.get(f"{slide}.czi", h) != h:
            return "FAIL", f"{slide}: {SUMS} disagrees with the inventory"
    present = {p.name for p in wsi.glob("*.czi")} if wsi.is_dir() else set()
    if a.stage == "download":  # sources still the inventoried bytes; partial state consistent
        if a.legacy_root is None:
            return "UNRESOLVED", "--legacy-root not given"
        bad = [s for s, (rel, h) in LOCAL.items() if sha256(a.legacy_root / rel) != h]
        if bad:
            return "FAIL", f"legacy sources changed: {bad}"
        missing = sorted(n for n in pins if n not in present)
        orphans = sorted(n for n in present - set(pins) if not journal_exists(a.data_root, n))
        if missing or orphans:
            return "FAIL", f"pinned but absent: {missing}; unpinned without journal: {orphans}"
    elif set(pins) != {f"{s}.czi" for s in PILOT} or present != set(pins):
        return "FAIL", f"{SUMS} lists {len(pins)} files and wsi/ holds {len(present)}, not 8"
    bad = [n for n, h in sorted(pins.items()) if sha256(wsi / n) != h]
    if bad:
        return "FAIL", f"content changed: {bad}"
    if a.stage == "download":
        return "PASS", f"3 legacy sources match the inventory; {len(pins)} pinned file(s) re-hash"
    return "PASS", "8/8 WSIs match their SHA-256"


def g_immutable(a: argparse.Namespace) -> Result:
    wsi = a.data_root / "wsi"
    pins = sums(a.data_root)
    if a.stage == "download" and not pins:
        return "PASS", "nothing pinned yet; checked again before extraction"
    names = sorted(pins) + ([SUMS] if (wsi / SUMS).exists() else [])
    writable = [n for n in names if (wsi / n).exists() and not sealed(wsi / n)]
    if a.stage == "extract" and wsi.stat().st_mode & 0o222:
        writable.append("wsi/ (directory)")
    if writable:
        return "FAIL", f"not sealed (a-w, uchg): {writable}"
    return "PASS", f"{len(pins)} pinned WSI(s) and {SUMS} sealed"


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


def retaken_problems(
    repo: Path, selection_path: Path, inventory_path: Path, pins: dict[str, str]
) -> tuple[str, list[str]]:
    """Check that a selection re-taken after acquisition is the approved pilot.

    The selection id is a digest over the inventory digest and per-slide availability, so it
    always changes after re-inventory. Equivalence is therefore checked field by field: same
    cohort, criteria and slides (identity, group, partition, label, stratum, rank key); every
    slide content-verified with the SHA-256 pinned in wsi/SHA256SUMS; and the document
    reproduces from manifest v1, pilot.yaml and its inventory.
    """
    from domain.pilot import PilotStatus, verify_selection
    from domain.wsi_inventory import SlideAvailability
    from infrastructure.cohort_sources import read_manifest
    from infrastructure.wsi_inventory import load_pilot_criteria, read_inventory, read_pilot

    approved_sel = read_pilot(repo / COHORT_DIR / "pilots" / SELECTION_ID / "pilot-selection.json")
    new = read_pilot(selection_path)
    inventory = read_inventory(inventory_path)
    manifest = read_manifest(repo / COHORT_DIR / "manifests/v1/cohort-manifest.json")
    problems: list[str] = []
    try:
        verify_selection(new, manifest, inventory)
    except Exception as exc:  # any refusal is a failure of this gate
        problems.append(f"does not reproduce: {exc}")
    if new.criteria != load_pilot_criteria(repo / COHORT_DIR / "pilot.yaml"):
        problems.append("criteria differ from pilot.yaml")
    if [r.name for r in inventory.roots] != [RETAKEN_ROOT]:
        problems.append(f"inventory roots {[r.name for r in inventory.roots]} != [{RETAKEN_ROOT}]")
    if (new.cohort_digest, new.criteria) != (approved_sel.cohort_digest, approved_sel.criteria):
        problems.append("cohort or criteria differ from the approved selection")
    identity = ("slide_id", "group_id", "cohort", "partition", "label", "file_name", "stratum")
    keys = (*identity, "rank_key", "provisional")

    def key(s: Any) -> tuple[Any, ...]:
        return tuple(getattr(s, k) for k in keys)

    # Compared as sets: once every pilot slide is verified, verified-first no longer breaks
    # ties, so the order inside a stratum follows the rank key alone and may differ.
    if sorted(map(key, new.slides)) != sorted(map(key, approved_sel.slides)):
        problems.append("slides differ from the approved selection")
    if new.status is not PilotStatus.READY:
        problems.append(f"status {new.status.value}, not ready")
    for s in new.slides:
        if s.availability is not SlideAvailability.VERIFIED:
            problems.append(f"{s.slide_id} not content-verified")
        elif s.sha256 != pins.get(s.file_name):
            problems.append(f"{s.slide_id}: SHA-256 differs from {SUMS}")
    for s in approved_sel.slides:
        if s.sha256 is not None and pins.get(s.file_name) != s.sha256:
            problems.append(f"{s.slide_id}: pin differs from the approved selection")
    return new.selection_id, problems


def g_retaken(a: argparse.Namespace) -> Result:
    if a.stage == "download":
        return "PASS", "taken after acquisition; checked at the extract stage"
    if a.retaken_selection is None or a.retaken_inventory is None:
        return "UNRESOLVED", "--retaken-selection and --retaken-inventory are required"
    new_id, problems = retaken_problems(
        a.repo, a.retaken_selection, a.retaken_inventory, sums(a.data_root)
    )
    if problems:
        return "FAIL", f"{new_id}: " + "; ".join(problems)
    return "PASS", f"{new_id} is the approved pilot {SELECTION_ID[:18]}… (8 slides, ready)"


GATES: tuple[tuple[str, str, Callable[[argparse.Namespace], Result]], ...] = (
    ("G1", "manifest v1 immutable", g_manifest),
    ("G2", "approved cohort and pilot selection", g_selection),
    ("G3", "no test-partition slide, no unexpected file", g_no_test),
    ("G4", "D-7 and D-9 approved", g_decisions),
    ("G5", "D-10 licensing approved", g_d10),
    ("G6", "explicit Phase C2 authorization bound to artifact digests", g_c2),
    ("G7", "free disk >= planned writes + 10 GB reserve", g_disk),
    ("G8", "WSI content integrity", g_wsi),
    ("G9", "source WSIs immutable", g_immutable),
    ("G10", "UNI checkpoint integrity", g_checkpoint),
    ("G11", "MPS available", g_mps),
    ("G12", "AC power", g_power),
    ("G13", "re-taken selection equals the approved pilot", g_retaken),
)


def run_gates(a: argparse.Namespace) -> int:
    failed = 0
    for gid, name, gate in GATES:
        try:
            status, detail = gate(a)
        except Exception as exc:  # fail closed: an unevaluable gate never passes
            status, detail = "UNRESOLVED", f"{type(exc).__name__}: {exc}"
        failed += status != "PASS"
        print(f"{gid:<4} {status:<10} {name}: {detail}", flush=True)
    verdict = "GO" if not failed else f"NO-GO ({failed} gate(s) not passed)"
    print(f"preflight[{a.stage}]: {verdict}", flush=True)
    return 0 if not failed else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument("--stage", choices=("download", "extract"))
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[4])
    p.add_argument("--data-root", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--legacy-root", type=Path)
    p.add_argument("--c2-authorization")
    p.add_argument("--retaken-selection", type=Path)
    p.add_argument("--retaken-inventory", type=Path)
    p.add_argument("--print-binding", action="store_true")
    a = p.parse_args(argv)
    a.repo = a.repo.expanduser().resolve()
    if a.print_binding:
        sys.stdout.buffer.write(canonical(binding_document(a.repo)))
        return 0
    if a.stage is None or a.data_root is None:
        p.error("--stage and --data-root are required")
    a.data_root = a.data_root.expanduser()
    for name in ("checkpoint", "legacy_root", "retaken_selection", "retaken_inventory"):
        value = getattr(a, name)
        setattr(a, name, value and value.expanduser())
    return run_gates(a)


if __name__ == "__main__":
    sys.path.insert(0, os.fspath(Path(__file__).resolve().parent))
    sys.exit(main())
