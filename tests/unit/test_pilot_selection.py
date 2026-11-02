"""SPEC-008 Phase C1: deterministic extraction-pilot selection (pure domain, synthetic cohort)."""

from __future__ import annotations

import dataclasses

import pytest

from domain.cohort import CohortManifest, SlidePolicy, SplitPolicy, build_manifest
from domain.errors import PilotSelectionError
from domain.pilot import (
    PilotCriteria,
    PilotStatus,
    Preference,
    rank_key,
    select_pilot,
    verify_selection,
)
from domain.wsi_inventory import (
    Access,
    InventoryRoot,
    LocationKind,
    ObservedObject,
    SlideAvailability,
    Verification,
    WSIInventory,
    build_inventory,
)
from tests.cohort_support import (
    candidate,
    entry,
    observations,
    pilot_definition,
    pilot_observations,
)

ROOTS = (InventoryRoot("disk", LocationKind.LOCAL, "local"),)


def manifest(bb_positive: str = "train", pending: bool = True) -> CohortManifest:
    return build_manifest(pilot_definition(pending), pilot_observations(bb_positive))


def criteria(**overrides: object) -> PilotCriteria:
    base: dict[str, object] = {
        "plan_id": "pilot-test",
        "seed": 7,
        "partitions": ("train", "validation"),
        "strata": ("cohort", "label"),
        "per_stratum": 1,
        "prefer": Preference.NONE,
        "decision": "D-6",
    }
    base.update(overrides)
    return PilotCriteria(**base)  # type: ignore[arg-type]


def inventory(m: CohortManifest, slides: list[str]) -> WSIInventory:
    objs = [
        ObservedObject(
            "disk",
            f"{s}.img",
            LocationKind.LOCAL,
            Access.READABLE,
            Verification.CONTENT_VERIFIED,
            100 + i,
            f"{i + 1:064x}",
        )
        for i, s in enumerate(slides)
    ]
    return build_inventory(m, ROOTS, ["*.img"], objs)


def test_selection_is_deterministic_seeded_and_stratified() -> None:
    m = manifest()
    a = select_pilot(m, criteria(per_stratum=2))
    assert a == select_pilot(m, criteria(per_stratum=2)) and a.digest
    assert {tuple(s.stratum) for s in a.slides} == {
        ("AA", "neg"),
        ("AA", "pos"),
        ("BB", "neg"),
        ("BB", "pos"),
    }
    assert all(o.selected == 2 and o.shortfall == 0 for o in a.strata)
    assert a.status is PilotStatus.REQUIRES_ACQUISITION  # no inventory: nothing is verified
    for o in a.strata:
        chosen = [s for s in a.slides if s.stratum == o.stratum]
        assert [s.rank_key for s in chosen] == sorted(s.rank_key for s in chosen)
    assert a.slides[0].rank_key == rank_key(7, a.slides[0].slide_id)
    seeds = {tuple(s.slide_id for s in select_pilot(m, criteria(seed=k)).slides) for k in range(20)}
    assert len(seeds) > 1


def test_test_slides_are_never_selected() -> None:
    m = manifest()
    sel = select_pilot(m, criteria(per_stratum=10))
    assert {s.partition for s in sel.slides} <= {"train", "validation"}
    test_slides = [r.slide_id for r in m.eligible() if r.partition == "test"]
    assert test_slides and sum(o.outside_pilot_partitions for o in sel.strata) == len(test_slides)
    assert not set(test_slides) & {s.slide_id for s in sel.slides}


def test_the_test_partition_is_refused_outright() -> None:
    with pytest.raises(PilotSelectionError, match="never a pilot partition"):
        criteria(partitions=("train", "test"))


def test_partitions_outside_the_approved_pilot_partitions_are_refused() -> None:
    d = dataclasses.replace(
        pilot_definition(),
        split_policy=SplitPolicy("external@1", "reused_benchmark", ("train",), ("D-5", "D-6")),
    )
    m = build_manifest(d, pilot_observations())
    with pytest.raises(PilotSelectionError, match="not approved pilot partitions"):
        select_pilot(m, criteria())
    assert {s.partition for s in select_pilot(m, criteria(partitions=("train",))).slides} == {
        "train"
    }
    with pytest.raises(PilotSelectionError, match="unknown partition"):
        criteria(partitions=("holdout",))


def test_a_class_absent_from_pilot_partitions_is_a_shortfall_not_a_substitution() -> None:
    m = manifest(bb_positive="test")
    sel = select_pilot(m, criteria(per_stratum=2))
    bb_pos = next(o for o in sel.strata if o.stratum == ("BB", "pos"))
    assert (bb_pos.candidates, bb_pos.selected, bb_pos.shortfall) == (0, 0, 2)
    assert bb_pos.outside_pilot_partitions == 3
    assert sel.status is PilotStatus.INCOMPLETE
    assert not [s for s in sel.slides if s.stratum == ("BB", "pos")]
    assert len(sel.slides) == 6  # the other strata are filled, nothing borrowed


def test_partition_strata_cover_only_pilot_partitions() -> None:
    sel = select_pilot(manifest(), criteria(strata=("partition", "label")))
    assert {o.stratum[0] for o in sel.strata} == {"train", "validation"}


def test_verified_images_come_first_when_preferred() -> None:
    m = manifest()
    plain = select_pilot(m, criteria())
    # Verify the lowest-priority candidate of each stratum: it must now be chosen.
    losers = []
    for o in plain.strata:
        members = [
            r
            for r in m.eligible()
            if (r.cohort, r.label) == o.stratum and r.partition in ("train", "validation")
        ]
        losers.append(max(members, key=lambda r: rank_key(7, r.slide_id)).slide_id)
    inv = inventory(m, losers)
    sel = select_pilot(m, criteria(prefer=Preference.VERIFIED_IMAGES), inv)
    assert sorted(s.slide_id for s in sel.slides) == sorted(losers)
    assert sel.status is PilotStatus.READY and sel.to_acquire() == ()
    assert all(s.availability is SlideAvailability.VERIFIED and s.sha256 for s in sel.slides)
    assert sel.inventory_digest == inv.digest


def test_preference_needs_an_inventory() -> None:
    with pytest.raises(PilotSelectionError, match="needs an inventory"):
        select_pilot(manifest(), criteria(prefer=Preference.VERIFIED_IMAGES))


def test_unverified_slides_are_listed_for_acquisition() -> None:
    m = manifest()
    sel = select_pilot(m, criteria(), inventory(m, ["AA_X_T001_01"]))
    assert sel.status is PilotStatus.REQUIRES_ACQUISITION
    assert {s.availability for s in sel.to_acquire()} == {SlideAvailability.NOT_FOUND_IN_SCOPE}


def test_at_most_one_slide_per_group() -> None:
    d = dataclasses.replace(pilot_definition(), slide_policy=SlidePolicy(("01", "02"), "D-2"))
    obs = observations(
        [entry("AA_X_T001_01", "g:AA:T001"), entry("AA_X_T001_02", "g:AA:T001", index="02")]
        + [entry(f"AA_X_T00{c}_01", f"g:AA:T00{c}") for c in (2, 3)],
        [
            candidate("AA_X_T001_01", "g:AA:T001", "neg"),
            candidate("AA_X_T001_02", "g:AA:T001", "neg", row=3),
            candidate("AA_X_T002_01", "g:AA:T002", "neg", row=4),
            candidate("AA_X_T003_01", "g:AA:T003", "pos", row=5),
        ],
    )
    m = build_manifest(d, obs)
    sel = select_pilot(m, criteria(per_stratum=3))
    groups = [s.group_id for s in sel.slides]
    assert len(groups) == len(set(groups))
    neg = next(o for o in sel.strata if o.stratum == ("AA", "neg"))
    assert (neg.candidates, neg.selected, neg.shortfall) == (3, 2, 1)


def test_the_authorising_decision_must_be_approved_and_gate_the_pilot() -> None:
    m = manifest()
    with pytest.raises(PilotSelectionError, match="approved decision"):
        select_pilot(m, criteria(decision="D-10"))
    with pytest.raises(PilotSelectionError, match="approved decision"):
        select_pilot(m, criteria(decision="D-1"))


def test_pending_decisions_are_carried_into_the_selection() -> None:
    assert select_pilot(manifest(), criteria()).pending_decisions == ("D-10",)
    assert select_pilot(manifest(pending=False), criteria()).pending_decisions == ()


def test_inventory_must_belong_to_the_same_manifest() -> None:
    other = manifest(bb_positive="validation")
    with pytest.raises(Exception, match="taken against"):
        select_pilot(manifest(), criteria(), inventory(other, []))


def test_a_selection_that_does_not_reproduce_is_refused() -> None:
    m = manifest()
    inv = inventory(m, ["AA_X_T001_01"])
    sel = select_pilot(m, criteria(prefer=Preference.VERIFIED_IMAGES), inv)
    verify_selection(sel, m, inv)
    with pytest.raises(PilotSelectionError, match="another inventory"):
        verify_selection(sel, m, None)
    forged = dataclasses.replace(sel, slides=sel.slides[1:])
    with pytest.raises(PilotSelectionError, match="does not reproduce"):
        verify_selection(forged, m, inv)
    with pytest.raises(PilotSelectionError, match="another cohort"):
        verify_selection(sel, manifest(bb_positive="validation"), inv)


@pytest.mark.parametrize(
    "overrides",
    [
        {"seed": -1},
        {"seed": True},
        {"per_stratum": 0},
        {"strata": ()},
        {"strata": ("cohort", "cohort")},
        {"strata": ("site",)},
        {"partitions": ()},
        {"partitions": ("train", "train")},
        {"plan_id": "has space"},
    ],
)
def test_invalid_criteria_are_refused(overrides: dict[str, object]) -> None:
    with pytest.raises(PilotSelectionError):
        criteria(**overrides)
