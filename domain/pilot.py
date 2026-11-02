"""Deterministic extraction-pilot selection over a cohort manifest (SPEC-008 §9, ADR-008 D8/D9).

A pilot is a small, representative set of eligible slides used to check an extraction
configuration (memory, throughput, output integrity, cost) before cohort-wide work. Selection is
generic and fully reproducible from (manifest, criteria, inventory):

* Candidates are eligible records in the requested partitions, which must be a subset of the
  manifest's approved pilot partitions. The test partition is never a pilot partition.
* Strata are the cartesian product of the configured record fields (`cohort`, `label`,
  `partition`) over **all** eligible records, so a class that exists only outside the pilot
  partitions shows up as a shortfall instead of disappearing.
* Within a stratum, candidates are ranked by `SHA-256("<seed>:<slide id>")`, optionally after
  slides whose images an inventory has content-verified. The ranking does not depend on input
  order.
* At most one slide per group is selected.
* An unfilled stratum is reported as a shortfall. Nothing is ever substituted from another
  stratum or partition.

`verify_selection` re-runs the selection and refuses a document that does not reproduce.
"""

from __future__ import annotations

import hashlib
import itertools
import re
from dataclasses import dataclass
from enum import StrEnum

from domain.cohort import CohortManifest, CohortRecord, DecisionStatus
from domain.errors import PilotSelectionError
from domain.hashing import digest, short_digest
from domain.mil import PARTITIONS, Partition
from domain.wsi_inventory import SlideAvailability, WSIInventory, check_identity_source

PILOT_SCHEMA = "pathology-ai.pilot-selection/1"
PILOT_GATE = "pilot"
EXTRACTION_GATE = "extraction"
STRATUM_FIELDS = ("cohort", "label", "partition")
_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class Preference(StrEnum):
    NONE = "none"  # rank by seeded hash only
    VERIFIED_IMAGES = "verified_images"  # content-verified images first, then seeded hash


class PilotStatus(StrEnum):
    READY = "ready"  # every stratum filled and every selected image content-verified
    REQUIRES_ACQUISITION = "requires_acquisition"  # filled; some images must be fetched first
    INCOMPLETE = "incomplete"  # a stratum cannot be filled from the pilot partitions


@dataclass(frozen=True, slots=True)
class PilotCriteria:
    plan_id: str
    seed: int
    partitions: tuple[str, ...]
    strata: tuple[str, ...]
    per_stratum: int
    prefer: Preference
    decision: str

    def __post_init__(self) -> None:
        for what, value in (("plan id", self.plan_id), ("decision", self.decision)):
            if not isinstance(value, str) or not _TOKEN.fullmatch(value):
                raise PilotSelectionError(f"invalid {what} {value!r}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise PilotSelectionError("the seed is a non-negative integer")
        if not self.partitions or len(set(self.partitions)) != len(self.partitions):
            raise PilotSelectionError("pilot partitions must be listed once each")
        for p in self.partitions:
            if p not in PARTITIONS:
                raise PilotSelectionError(f"unknown partition {p!r}")
            if p == Partition.TEST:
                raise PilotSelectionError(
                    "the test partition is never a pilot partition (test isolation)"
                )
        if not self.strata or len(set(self.strata)) != len(self.strata):
            raise PilotSelectionError("strata must be listed once each")
        unknown = [s for s in self.strata if s not in STRATUM_FIELDS]
        if unknown:
            raise PilotSelectionError(f"unknown stratum field(s) {unknown}; use {STRATUM_FIELDS}")
        if isinstance(self.per_stratum, bool) or self.per_stratum < 1:
            raise PilotSelectionError("per_stratum is a positive integer")


@dataclass(frozen=True, slots=True)
class PilotSlide:
    slide_id: str
    group_id: str
    cohort: str
    partition: str
    label: str
    file_name: str
    stratum: tuple[str, ...]
    rank_key: str
    availability: SlideAvailability | None  # None: no inventory was consulted
    sha256: str | None
    size_bytes: int | None
    provisional: bool


@dataclass(frozen=True, slots=True)
class StratumOutcome:
    stratum: tuple[str, ...]
    required: int
    candidates: int  # eligible slides of the stratum in the pilot partitions
    outside_pilot_partitions: int  # eligible slides of the stratum elsewhere (never used)
    selected: int
    verified_selected: int
    shortfall: int


@dataclass(frozen=True, slots=True)
class PilotSelection:
    cohort: str
    cohort_digest: str
    inventory_digest: str | None
    criteria: PilotCriteria
    slides: tuple[PilotSlide, ...]
    strata: tuple[StratumOutcome, ...]
    status: PilotStatus
    pending_decisions: tuple[str, ...]
    schema_version: str = PILOT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != PILOT_SCHEMA:
            raise PilotSelectionError(f"unsupported pilot schema {self.schema_version!r}")

    @property
    def digest(self) -> str:
        return digest(self)

    @property
    def selection_id(self) -> str:
        return short_digest(self, "pilot")

    def to_acquire(self) -> tuple[PilotSlide, ...]:
        return tuple(s for s in self.slides if s.availability is not SlideAvailability.VERIFIED)


def rank_key(seed: int, slide_id: str) -> str:
    return hashlib.sha256(f"{seed}:{slide_id}".encode()).hexdigest()


def _value(record: CohortRecord, field: str) -> str:
    value = getattr(record, field)
    assert isinstance(value, str)
    return value


def _check_policy(manifest: CohortManifest, criteria: PilotCriteria) -> None:
    allowed = set(manifest.split.pilot_partitions)
    outside = [p for p in criteria.partitions if p not in allowed]
    if outside:
        raise PilotSelectionError(
            f"partition(s) {outside} are not approved pilot partitions of {manifest.ref} "
            f"({sorted(allowed)})"
        )
    decisions = {d.decision_id: d for d in manifest.decisions}
    d = decisions.get(criteria.decision)
    if d is None or d.status is not DecisionStatus.APPROVED or PILOT_GATE not in d.gates:
        raise PilotSelectionError(
            f"decision {criteria.decision} must be an approved decision of {manifest.ref} gating "
            f"{PILOT_GATE!r}"
        )


def select_pilot(
    manifest: CohortManifest,
    criteria: PilotCriteria,
    inventory: WSIInventory | None = None,
) -> PilotSelection:
    _check_policy(manifest, criteria)
    if inventory is not None:
        check_identity_source(inventory, manifest)
    elif criteria.prefer is Preference.VERIFIED_IMAGES:
        raise PilotSelectionError("preferring verified images needs an inventory")
    images = {s.slide_id: s for s in inventory.slides} if inventory is not None else {}
    eligible = manifest.eligible()
    if not eligible:
        raise PilotSelectionError(f"{manifest.ref} has no eligible record")
    pilot_parts = set(criteria.partitions)

    def key(r: CohortRecord) -> tuple[str, ...]:
        return tuple(_value(r, f) for f in criteria.strata)

    axes = [
        sorted({_value(r, f) for r in eligible} if f != "partition" else criteria.partitions)
        for f in criteria.strata
    ]
    used_groups: set[str] = set()
    slides: list[PilotSlide] = []
    outcomes: list[StratumOutcome] = []
    for stratum in itertools.product(*axes):
        members = [r for r in eligible if key(r) == stratum]
        inside = [r for r in members if r.partition in pilot_parts]

        def order(r: CohortRecord) -> tuple[int, str]:
            image = images.get(r.slide_id)
            verified = image is not None and image.availability is SlideAvailability.VERIFIED
            first = 0 if criteria.prefer is Preference.NONE or verified else 1
            return first, rank_key(criteria.seed, r.slide_id)

        chosen: list[PilotSlide] = []
        for r in sorted(inside, key=order):
            if len(chosen) == criteria.per_stratum:
                break
            if r.group_id in used_groups:
                continue
            used_groups.add(r.group_id)
            image = images.get(r.slide_id)
            assert r.partition is not None and r.label is not None
            chosen.append(
                PilotSlide(
                    slide_id=r.slide_id,
                    group_id=r.group_id,
                    cohort=r.cohort,
                    partition=r.partition,
                    label=r.label,
                    file_name=r.wsi.file_name,
                    stratum=stratum,
                    rank_key=rank_key(criteria.seed, r.slide_id),
                    availability=None if image is None else image.availability,
                    sha256=None if image is None else image.sha256,
                    size_bytes=None if image is None else image.size_bytes,
                    provisional=r.provisional,
                )
            )
        slides.extend(chosen)
        outcomes.append(
            StratumOutcome(
                stratum=stratum,
                required=criteria.per_stratum,
                candidates=len(inside),
                outside_pilot_partitions=len(members) - len(inside),
                selected=len(chosen),
                verified_selected=sum(s.availability is SlideAvailability.VERIFIED for s in chosen),
                shortfall=criteria.per_stratum - len(chosen),
            )
        )
    if any(o.shortfall for o in outcomes):
        status = PilotStatus.INCOMPLETE
    elif all(s.availability is SlideAvailability.VERIFIED for s in slides):
        status = PilotStatus.READY
    else:
        status = PilotStatus.REQUIRES_ACQUISITION
    definition = manifest.definition
    pending = sorted(
        {d for gate in (PILOT_GATE, EXTRACTION_GATE) for d in definition.pending(gate)}
    )
    return PilotSelection(
        cohort=manifest.ref,
        cohort_digest=manifest.digest,
        inventory_digest=None if inventory is None else inventory.digest,
        criteria=criteria,
        slides=tuple(slides),
        strata=tuple(outcomes),
        status=status,
        pending_decisions=tuple(pending),
    )


def verify_selection(
    selection: PilotSelection,
    manifest: CohortManifest,
    inventory: WSIInventory | None = None,
) -> None:
    """Refuse a selection that does not reproduce from its manifest, criteria and inventory."""
    if selection.cohort_digest != manifest.digest:
        raise PilotSelectionError(f"selection {selection.selection_id} was made on another cohort")
    expected_inventory = None if inventory is None else inventory.digest
    if selection.inventory_digest != expected_inventory:
        raise PilotSelectionError(
            f"selection {selection.selection_id} was made with another inventory"
        )
    if select_pilot(manifest, selection.criteria, inventory) != selection:
        raise PilotSelectionError(f"selection {selection.selection_id} does not reproduce")
