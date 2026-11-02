"""SPEC-008 Phase C1: `pathology-ai-cohort inventory` and `pilot` end to end (synthetic), plus the
SurGen inventory and pilot regression on the committed documents (no image access needed)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cli import cohort_main
from domain.cohort import build_manifest
from domain.pilot import PilotStatus, verify_selection
from infrastructure.cohort_sources import MANIFEST_FILE, read_manifest, write_cohort_outputs
from infrastructure.wsi_inventory import (
    INVENTORY_FILE,
    PILOT_FILE,
    load_pilot_criteria,
    read_inventory,
    read_pilot,
)
from tests.cohort_support import pilot_definition, pilot_observations

ROOT = Path(__file__).resolve().parents[2]
COHORT = ROOT / "projects" / "cohorts" / "surgen-msi"


def _run(argv: list[str], capsys: pytest.CaptureFixture[str]) -> dict[str, object]:
    assert cohort_main(argv) == 0
    out: dict[str, object] = json.loads(capsys.readouterr().out)
    return out


def test_inventory_then_pilot_through_the_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = build_manifest(pilot_definition(), pilot_observations())
    write_cohort_outputs(tmp_path / "v1", manifest)
    images = tmp_path / "images"
    images.mkdir()
    for name in ("AA_X_T001_01", "AA_X_T006_01", "BB_X_T003_01"):  # one test slide (T006)
        (images / f"{name}.img").write_bytes(name.encode() * 50)
    before = sorted((p.name, p.stat().st_mtime_ns) for p in images.iterdir())
    criteria = tmp_path / "pilot.yaml"
    criteria.write_text(
        "schema: pathology-ai.pilot-config/1\nplan_id: p1\nseed: 3\n"
        "partitions: [train, validation]\nstrata: [cohort, label]\nper_stratum: 1\n"
        "prefer: verified_images\ndecision: D-6\n"
    )
    mpath = str(tmp_path / "v1" / MANIFEST_FILE)
    inv = _run(
        [
            "inventory",
            "--manifest",
            mpath,
            "--root",
            f"disk={images}",
            "--pattern",
            "*.img",
            "--out",
            str(tmp_path / "inv"),
        ],
        capsys,
    )
    assert "never modified" in str(inv["notice"])
    assert inv["availability"] == {
        "verified": 3,
        "inconsistent_copies": 0,
        "unreadable": 0,
        "remote_only": 0,
        "not_found_in_scope": len(manifest.records) - 3,
    }
    assert inv["complete"] is False and inv["duplicate_content_groups"] == []
    gates = inv["cohort_gates"]
    assert isinstance(gates, dict) and gates["provisional"] == len(manifest.eligible())
    assert sorted((p.name, p.stat().st_mtime_ns) for p in images.iterdir()) == before
    inv_path = Path(str(inv["written"]))
    assert (
        inv_path.name == INVENTORY_FILE
        and read_inventory(inv_path).digest == inv["inventory_digest"]
    )

    sel = _run(
        [
            "pilot",
            "--manifest",
            mpath,
            "--criteria",
            str(criteria),
            "--inventory",
            str(inv_path),
            "--out",
            str(tmp_path / "pilots"),
        ],
        capsys,
    )
    slides = sel["slides"]
    assert isinstance(slides, list)
    chosen = {s["slide_id"]: s for s in slides}
    assert "AA_X_T006_01" not in chosen  # verified locally, but a test slide
    assert chosen["AA_X_T001_01"]["availability"] == "verified"
    assert chosen["BB_X_T003_01"]["availability"] == "verified"
    assert sel["status"] == "requires_acquisition" and sel["pending_decisions"] == ["D-10"]
    written = read_pilot(Path(str(sel["written"])))
    verify_selection(written, read_manifest(Path(mpath)), read_inventory(inv_path))
    again = _run(
        ["pilot", "--manifest", mpath, "--criteria", str(criteria), "--inventory", str(inv_path)],
        capsys,
    )
    assert again["selection_digest"] == sel["selection_digest"]


def test_pilot_cli_refuses_a_test_partition(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_cohort_outputs(tmp_path, build_manifest(pilot_definition(), pilot_observations()))
    criteria = tmp_path / "pilot.yaml"
    criteria.write_text(
        "schema: pathology-ai.pilot-config/1\nplan_id: p1\nseed: 3\n"
        "partitions: [train, test]\nstrata: [label]\nper_stratum: 1\nprefer: none\n"
        "decision: D-6\n"
    )
    with pytest.raises(SystemExit) as exc:
        cohort_main(
            ["pilot", "--manifest", str(tmp_path / MANIFEST_FILE), "--criteria", str(criteria)]
        )
    assert exc.value.code == 3
    assert "PILOT_SELECTION" in capsys.readouterr().err


# --------------------------------------------------------------------------- SurGen regression

INVENTORY = COHORT / "inventories" / "pai-wsi-inventory-783f44e2ea71bead23ce78c80f069e12"
PILOT = COHORT / "pilots" / "pai-pilot-5da7534bcaa4d30c93f4eae703b3d479"


@pytest.mark.skipif(
    not (INVENTORY / INVENTORY_FILE).exists() or not (PILOT / PILOT_FILE).exists(),
    reason="SurGen Phase C1 documents are not present",
)
def test_surgen_local_inventory_and_pilot_selection_reproduce() -> None:
    manifest = read_manifest(COHORT / "manifests" / "v1" / MANIFEST_FILE)
    inventory = read_inventory(INVENTORY / INVENTORY_FILE)
    assert inventory.counts()["verified"] == 17 and inventory.duplicates == ()
    assert not inventory.complete and inventory.unmatched == ()
    selection = read_pilot(PILOT / PILOT_FILE)
    assert selection.criteria == load_pilot_criteria(COHORT / "pilot.yaml")
    verify_selection(selection, manifest, inventory)
    assert selection.status is PilotStatus.REQUIRES_ACQUISITION
    assert {s.partition for s in selection.slides} <= {"train", "validation"}
    assert len(selection.slides) == 8 and all(o.shortfall == 0 for o in selection.strata)
    assert {(s.cohort, s.label) for s in selection.to_acquire()} == {
        ("SR1482", "positive"),
        ("SR386", "positive"),
        ("SR386", "negative"),
    }
    assert selection.pending_decisions == ("D-10", "D-7", "D-9")
