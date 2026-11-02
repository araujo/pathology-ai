"""SPEC-008 Phase C1: WSI inventory contracts (pure domain, synthetic identities)."""

from __future__ import annotations

import dataclasses

import pytest

from domain.cohort import CohortManifest, EligibilityStatus, WSIStatus, build_manifest
from domain.errors import CohortError, WSIInventoryError
from domain.wsi_inventory import (
    Access,
    InventoryRoot,
    LocationKind,
    ObservedObject,
    SlideAvailability,
    Verification,
    build_inventory,
    cohort_wsi_evidence,
    identity_source,
)
from tests.cohort_support import pilot_definition, pilot_observations

A = "a" * 64
B = "b" * 64
ROOTS = (
    InventoryRoot("disk", LocationKind.LOCAL, "local"),
    InventoryRoot("bucket", LocationKind.REMOTE, "listing"),
)


def manifest() -> CohortManifest:
    return build_manifest(pilot_definition(), pilot_observations())


def local(path: str, sha: str = A, size: int = 10, root: str = "disk") -> ObservedObject:
    return ObservedObject(
        root, path, LocationKind.LOCAL, Access.READABLE, Verification.CONTENT_VERIFIED, size, sha
    )


def unreadable(path: str) -> ObservedObject:
    return ObservedObject(
        "disk", path, LocationKind.LOCAL, Access.UNREADABLE, Verification.FAILED, None, None
    )


def remote(path: str, etag: str | None = A, size: int | None = 10) -> ObservedObject:
    return ObservedObject(
        "bucket",
        path,
        LocationKind.REMOTE,
        Access.LISTED,
        Verification.METADATA_ONLY,
        size,
        None,
        remote_checksum=etag,
        remote_checksum_kind=None if etag is None else "s3-etag",
    )


def every_slide(m: CohortManifest) -> list[ObservedObject]:
    return [local(f"x/{r.wsi.file_name}", f"{i:064x}") for i, r in enumerate(m.records)]


def test_copies_that_agree_are_one_verified_slide() -> None:
    m = manifest()
    inv = build_inventory(
        m, ROOTS, ["*.img"], [local("a/AA_X_T001_01.img"), local("b/AA_X_T001_01.img")]
    )
    s = inv.slide("AA_X_T001_01")
    assert s.availability is SlideAvailability.VERIFIED
    assert s.sha256 == A and s.size_bytes == 10
    assert s.copies == ("disk:a/AA_X_T001_01.img", "disk:b/AA_X_T001_01.img")
    assert inv.duplicates == () and s.shares_content_with == ()


def test_copies_that_disagree_are_inconsistent() -> None:
    m = manifest()
    inv = build_inventory(
        m, ROOTS, ["*.img"], [local("a/AA_X_T001_01.img"), local("b/AA_X_T001_01.img", B)]
    )
    s = inv.slide("AA_X_T001_01")
    assert s.availability is SlideAvailability.INCONSISTENT_COPIES and s.sha256 is None


def test_identical_content_under_two_identities_is_a_duplicate() -> None:
    m = manifest()
    inv = build_inventory(
        m,
        ROOTS,
        ["*.img"],
        [local("AA_X_T001_01.img"), local("BB_X_T004_01.img"), local("AA_X_T002_01.img", B)],
    )
    assert [d.slide_ids for d in inv.duplicates] == [("AA_X_T001_01", "BB_X_T004_01")]
    assert inv.slide("AA_X_T001_01").shares_content_with == ("BB_X_T004_01",)
    assert inv.slide("AA_X_T002_01").shares_content_with == ()


def test_excluded_records_are_inventoried_too() -> None:
    """Duplicates can hide among exclusions, so every record is an expected image."""
    m = manifest()
    inv = build_inventory(m, ROOTS, ["*.img"], [local("AA_X_T001_02.img")])
    assert {e.slide_id for e in inv.expected} == {r.slide_id for r in m.records}
    assert inv.slide("AA_X_T001_02").availability is SlideAvailability.VERIFIED


def test_remote_listing_is_metadata_only_and_never_a_content_digest() -> None:
    m = manifest()
    # The entity tag happens to equal a verified local digest: still no duplicate, no promotion.
    inv = build_inventory(
        m, ROOTS, ["*.img"], [remote("AA_X_T002_01.img", etag=A), local("AA_X_T001_01.img", A)]
    )
    s = inv.slide("AA_X_T002_01")
    assert s.availability is SlideAvailability.REMOTE_ONLY and s.sha256 is None
    assert inv.duplicates == ()
    with pytest.raises(WSIInventoryError, match="metadata only"):
        ObservedObject(
            "bucket",
            "k.img",
            LocationKind.REMOTE,
            Access.READABLE,
            Verification.CONTENT_VERIFIED,
            10,
            A,
        )
    with pytest.raises(WSIInventoryError, match="SHA-256"):
        dataclasses.replace(remote("k.img"), sha256=A)


def test_local_objects_carry_no_remote_checksum_and_failed_reads_no_digest() -> None:
    with pytest.raises(WSIInventoryError, match="remote checksum"):
        dataclasses.replace(local("k.img"), remote_checksum="x", remote_checksum_kind="s3-etag")
    with pytest.raises(WSIInventoryError):
        dataclasses.replace(unreadable("k.img"), sha256=A)
    with pytest.raises(WSIInventoryError):
        dataclasses.replace(local("k.img"), access=Access.UNREADABLE)


def test_a_verified_copy_wins_over_an_unreadable_one() -> None:
    m = manifest()
    inv = build_inventory(
        m, ROOTS, ["*.img"], [unreadable("a/AA_X_T001_01.img"), local("b/AA_X_T001_01.img")]
    )
    assert inv.slide("AA_X_T001_01").availability is SlideAvailability.VERIFIED
    inv = build_inventory(m, ROOTS, ["*.img"], [unreadable("a/AA_X_T001_01.img")])
    assert inv.slide("AA_X_T001_01").availability is SlideAvailability.UNREADABLE


@pytest.mark.parametrize("path", ["/abs/x.img", "../x.img", "a/../x.img", "a//x.img", "a\\x.img"])
def test_object_paths_are_relative_to_their_root(path: str) -> None:
    with pytest.raises(WSIInventoryError, match="relative"):
        local(path)


def test_objects_must_belong_to_a_declared_root_of_their_kind() -> None:
    m = manifest()
    with pytest.raises(WSIInventoryError, match="not declared"):
        build_inventory(m, ROOTS, ["*.img"], [local("x.img", root="elsewhere")])
    with pytest.raises(WSIInventoryError, match="kind"):
        build_inventory(m, ROOTS, ["*.img"], [local("x.img", root="bucket")])


def test_unmatched_objects_are_listed_not_guessed() -> None:
    m = manifest()
    inv = build_inventory(m, ROOTS, ["*.img"], [local("renamed-copy.img"), local("AA_X_T1.img")])
    assert inv.unmatched == ("disk:AA_X_T1.img", "disk:renamed-copy.img")
    assert inv.counts()["verified"] == 0


def test_inventory_is_content_addressed_and_order_independent() -> None:
    m = manifest()
    objs = [local("AA_X_T001_01.img"), remote("BB_X_T002_01.img"), local("z.img", B)]
    a = build_inventory(m, ROOTS, ["*.img", "*.img"], objs)
    b = build_inventory(m, tuple(reversed(ROOTS)), ["*.img"], list(reversed(objs)))
    assert a == b and a.digest == b.digest and a.inventory_id.startswith("pai-wsi-inventory-")
    assert a.identity_source == identity_source(m)


def test_an_edited_inventory_is_refused() -> None:
    m = manifest()
    inv = build_inventory(m, ROOTS, ["*.img"], [local("AA_X_T001_01.img")])
    i = [s.slide_id for s in inv.slides].index("AA_X_T002_01")
    forged = dataclasses.replace(inv.slides[i], availability=SlideAvailability.VERIFIED)
    with pytest.raises(WSIInventoryError, match="re-derive"):
        dataclasses.replace(inv, slides=(*inv.slides[:i], forged, *inv.slides[i + 1 :]))
    with pytest.raises(WSIInventoryError, match="re-derive"):
        dataclasses.replace(inv, unmatched=("disk:x.img",))
    with pytest.raises(WSIInventoryError, match="re-derive"):
        dataclasses.replace(inv, objects=())


def test_a_scoped_inventory_never_closes_cohort_gates() -> None:
    m = manifest()
    inv = build_inventory(m, ROOTS, ["*.img"], [local("AA_X_T001_01.img")])
    assert not inv.complete
    assert inv.counts()["not_found_in_scope"] == len(m.records) - 1
    with pytest.raises(WSIInventoryError, match="never closes"):
        cohort_wsi_evidence(inv, m)
    remote_only = build_inventory(
        m,
        ROOTS,
        ["*.img"],
        [remote(r.wsi.file_name, etag=f"{i:064x}") for i, r in enumerate(m.records)],
    )
    with pytest.raises(WSIInventoryError, match="never closes"):
        cohort_wsi_evidence(remote_only, m)


def test_inventory_from_another_manifest_is_refused() -> None:
    m = manifest()
    other = build_manifest(pilot_definition(), pilot_observations("validation"))
    inv = build_inventory(m, ROOTS, ["*.img"], every_slide(m))
    with pytest.raises(WSIInventoryError, match="taken against"):
        cohort_wsi_evidence(inv, other)


def test_a_complete_inventory_closes_gates_in_a_new_manifest_version() -> None:
    m = manifest()
    inv = build_inventory(m, ROOTS, ["*.img"], every_slide(m))
    assert inv.complete
    evidence = cohort_wsi_evidence(inv, m)
    assert {e.status for e in evidence.values()} == {WSIStatus.VERIFIED}
    v2 = build_manifest(pilot_definition(), pilot_observations(), evidence)
    assert v2.eligible() and not any(r.provisional for r in v2.eligible())
    assert [r.slide_id for r in v2.eligible()] == [r.slide_id for r in m.eligible()]


def test_duplicates_and_unreadable_images_become_exclusions_in_the_next_version() -> None:
    m = manifest()
    objs = every_slide(m)
    names = [r.wsi.file_name for r in m.records]
    dup = names.index("BB_X_T004_01.img")
    objs[dup] = local(f"x/{names[dup]}", objs[names.index("AA_X_T001_01.img")].sha256 or A)
    bad = names.index("AA_X_T002_01.img")
    objs[bad] = unreadable(f"x/{names[bad]}")
    inv = build_inventory(m, ROOTS, ["*.img"], objs)
    v2 = build_manifest(pilot_definition(), pilot_observations(), cohort_wsi_evidence(inv, m))
    assert v2.record("AA_X_T001_01").status is EligibilityStatus.EXCLUDED_DUPLICATE
    assert v2.record("BB_X_T004_01").status is EligibilityStatus.EXCLUDED_DUPLICATE
    assert v2.record("AA_X_T002_01").status is EligibilityStatus.EXCLUDED_QUALITY


def test_inventory_errors_are_cohort_errors() -> None:
    assert WSIInventoryError.code == "WSI_INVENTORY_INVALID"
    assert issubclass(WSIInventoryError, CohortError)
