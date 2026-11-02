"""SPEC-008 Phase C1: the local WSI object source is read-only and records what it observed.

Synthetic files only. The source must hash every matching regular file exactly, never modify,
create or remove anything under a root, never follow symbolic links, and record a failed read
instead of a digest it never computed.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

import pytest

from domain.cohort import build_manifest
from domain.errors import ExperimentConflictError, WSIInventoryError
from domain.wsi_inventory import (
    Access,
    LocationKind,
    SlideAvailability,
    Verification,
    build_inventory,
)
from infrastructure.config import ConfigurationError
from infrastructure.wsi_inventory import (
    LocalWSIObjectSource,
    ObjectListingSource,
    load_pilot_criteria,
    read_inventory,
    sha256_file,
    write_inventory,
)
from ports import WSIObjectSource
from tests.cohort_support import pilot_definition, pilot_observations


def _snapshot(root: Path) -> dict[str, tuple[int, int, int, str]]:
    out = {}
    for p in sorted(root.rglob("*")):
        st = p.lstat()
        digest = (
            hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() and not p.is_symlink() else ""
        )
        out[p.relative_to(root).as_posix()] = (st.st_size, st.st_mtime_ns, st.st_mode, digest)
    return out


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    (root / "a" / "b").mkdir(parents=True)
    (root / "a" / "AA_X_T001_01.img").write_bytes(b"slide one" * 1000)
    (root / "a" / "b" / "AA_X_T002_01.img").write_bytes(b"slide two")
    (root / "a" / "notes.txt").write_text("not an image")
    (root / "AA_X_T003_01.img.part1").write_bytes(b"partial")
    return root


def test_hashes_match_and_paths_are_relative_to_the_root_name(tree: Path) -> None:
    src = LocalWSIObjectSource({"disk": tree}, ["*.img"], block_bytes=7)
    assert isinstance(src, WSIObjectSource)
    objs = src.observe()
    assert [o.uri for o in objs] == ["disk:a/AA_X_T001_01.img", "disk:a/b/AA_X_T002_01.img"]
    for o in objs:
        data = (tree / o.path).read_bytes()
        assert o.sha256 == hashlib.sha256(data).hexdigest() and o.size_bytes == len(data)
        assert o.kind is LocationKind.LOCAL and o.verification is Verification.CONTENT_VERIFIED
        assert str(tree) not in json.dumps([o.root, o.path, o.note])
    assert sha256_file(tree / "a" / "notes.txt")[1] == len("not an image")


def test_observation_never_modifies_anything(tree: Path) -> None:
    before = _snapshot(tree)
    for p in tree.rglob("*"):
        p.chmod(p.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    try:
        frozen = _snapshot(tree)
        objs = LocalWSIObjectSource({"disk": tree}, ["*.img", "*.txt"]).observe()
        assert len(objs) == 3 and all(o.access is Access.READABLE for o in objs)
        assert _snapshot(tree) == frozen
    finally:
        for p in tree.rglob("*"):
            p.chmod(p.stat().st_mode | stat.S_IWUSR)
    assert {k: v[:2] + v[3:] for k, v in _snapshot(tree).items()} == {
        k: v[:2] + v[3:] for k, v in before.items()
    }


def test_symbolic_links_are_never_followed(tree: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "AA_X_T004_01.img").write_bytes(b"outside")
    (tree / "link.img").symlink_to(tree / "a" / "AA_X_T001_01.img")
    (tree / "linked-dir").symlink_to(outside, target_is_directory=True)
    src = LocalWSIObjectSource({"disk": tree}, ["*.img"])
    uris = [o.uri for o in src.observe()]
    assert "disk:link.img" not in uris and not any("linked-dir" in u for u in uris)
    assert src.skipped_links == 1


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read mode-000 files")
def test_an_unreadable_file_is_recorded_without_a_digest(tree: Path) -> None:
    target = tree / "a" / "AA_X_T001_01.img"
    target.chmod(0)
    try:
        objs = {o.path: o for o in LocalWSIObjectSource({"disk": tree}, ["*.img"]).observe()}
    finally:
        target.chmod(0o644)
    o = objs["a/AA_X_T001_01.img"]
    assert o.access is Access.UNREADABLE and o.verification is Verification.FAILED
    assert o.sha256 is None and "read failed" in o.note


def test_an_empty_file_is_not_verified(tree: Path) -> None:
    (tree / "AA_X_T005_01.img").write_bytes(b"")
    objs = {o.path: o for o in LocalWSIObjectSource({"disk": tree}, ["*.img"]).observe()}
    assert objs["AA_X_T005_01.img"].access is Access.UNREADABLE


def test_a_file_that_changes_while_read_is_not_verified(
    tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import infrastructure.wsi_inventory as mod

    real = mod.sha256_file

    def growing(path: Path, block: int) -> tuple[str, int]:
        result = real(path, block)
        with open(path, "ab") as f:  # simulates a concurrent writer, not the inventory
            f.write(b"more")
        return result

    monkeypatch.setattr(mod, "sha256_file", growing)
    objs = LocalWSIObjectSource({"disk": tree}, ["AA_X_T002_01.img"]).observe()
    assert objs[0].access is Access.UNREADABLE and "changed while it was read" in objs[0].note


def test_roots_must_be_directories(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="not a directory"):
        LocalWSIObjectSource({"disk": tmp_path / "missing"}, ["*.img"])
    with pytest.raises(ConfigurationError):
        LocalWSIObjectSource({}, ["*.img"])
    with pytest.raises(ConfigurationError):
        LocalWSIObjectSource({"disk": tmp_path}, [])
    with pytest.raises(WSIInventoryError):
        LocalWSIObjectSource({"bad name": tmp_path}, ["*.img"])


def test_object_listing_keeps_entity_tags_opaque(tmp_path: Path) -> None:
    listing = tmp_path / "listing.json"
    etag = '"' + "c" * 32 + '-4"'
    listing.write_text(
        json.dumps(
            {
                "Contents": [
                    {"Key": "wsi/AA_X_T001_01.img", "Size": 9000, "ETag": etag},
                    {"Key": "wsi/readme.txt", "Size": 3, "ETag": '"x"'},
                ]
            }
        )
    )
    src = ObjectListingSource("bucket", "remote listing", listing, ["*.img"])
    assert isinstance(src, WSIObjectSource)
    (o,) = src.observe()
    assert o.kind is LocationKind.REMOTE and o.verification is Verification.METADATA_ONLY
    assert o.sha256 is None and o.remote_checksum == "c" * 32 + "-4"
    assert o.remote_checksum_kind == "s3-etag"
    m = build_manifest(pilot_definition(), pilot_observations())
    inv = build_inventory(m, src.roots, src.patterns, (o,))
    assert inv.slide("AA_X_T001_01").availability is SlideAvailability.REMOTE_ONLY


def test_inventory_documents_are_write_once_and_tamper_evident(tree: Path, tmp_path: Path) -> None:
    m = build_manifest(pilot_definition(), pilot_observations())
    src = LocalWSIObjectSource({"disk": tree}, ["*.img"])
    inv = build_inventory(m, src.roots, src.patterns, src.observe())
    out = tmp_path / "out"
    path = write_inventory(out, inv)
    assert path.parent.name == inv.inventory_id
    assert read_inventory(path) == inv
    assert write_inventory(out, inv) == path  # identical content is accepted
    other = build_inventory(m, src.roots, src.patterns, src.observe()[:1])
    with pytest.raises(ExperimentConflictError):
        from infrastructure.mil_artifacts import write_once

        write_once(path, other)
    doc = json.loads(path.read_text())
    doc["body"]["objects"] = doc["body"]["objects"][:1]
    path.write_text(json.dumps(doc))
    with pytest.raises(WSIInventoryError, match="integrity digest"):
        read_inventory(path)


def test_pilot_criteria_configuration_is_strict(tmp_path: Path) -> None:
    good = {
        "schema": "pathology-ai.pilot-config/1",
        "plan_id": "p",
        "seed": 1,
        "partitions": ["train"],
        "strata": ["label"],
        "per_stratum": 1,
        "prefer": "none",
        "decision": "D-6",
    }
    path = tmp_path / "pilot.yaml"
    path.write_text(json.dumps(good))
    assert load_pilot_criteria(path).partitions == ("train",)
    for bad in (
        {**good, "extra": 1},
        {**good, "schema": "other/1"},
        {**good, "prefer": "biggest"},
        {**good, "partitions": "train"},
    ):
        path.write_text(json.dumps(bad))
        with pytest.raises(ConfigurationError):
            load_pilot_criteria(path)
    path.write_text(json.dumps({**good, "partitions": ["test"]}))
    with pytest.raises(Exception, match="never a pilot partition"):
        load_pilot_criteria(path)
