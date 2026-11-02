"""Cohort reconciliation and the versioned cohort manifest (SPEC-008 §6-§8).

Everything here is cohort-, target- and dataset-agnostic. Cohort names, identifier formats,
class names, source targets and policy decisions are configuration; this module only enforces
the rules that make a curated cohort auditable:

* Every label is traceable to a source file (by digest), a row and the raw value.
* Sources that disagree are recorded as a conflict and the slide is excluded. Nothing here ever
  chooses between disagreeing sources (no file precedence, majority or recency). Only an
  explicit, approved `Resolution` can admit such a slide.
* A label given for a case (group) applies to a slide only when the case has exactly one slide
  in the roster. Inheriting a case label across several slides is a decision, never implied.
* A slide is `eligible` only when every reconciliation gate passed and every decision gating
  reconciliation is approved. Integrity checks that need the images themselves are recorded as
  outstanding gates, so an eligible record can be *provisional*.
* A group never spans partitions or sub-cohorts.
* `CohortManifest` (`pathology-ai.cohort/1`) re-derives its split summary and counts from its
  records on construction, so an edited document cannot pass as valid.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from enum import StrEnum

from domain.errors import (
    CohortError,
    CohortReconciliationError,
    GroupLeakageError,
)
from domain.hashing import digest, short_digest
from domain.mil import EXTERNAL_STRATEGY, PARTITIONS

COHORT_SCHEMA = "pathology-ai.cohort/1"
# The gate a decision must be approved for before any slide can become eligible.
RECONCILIATION_GATE = "reconciliation"
# Gates that need the images themselves; they stay outstanding until an inventory verifies them.
WSI_GATES = ("wsi_integrity", "wsi_duplicate_content")
UNASSIGNED = "unassigned"
NO_LABEL = "none"
_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _token(what: str, value: object) -> str:
    if not isinstance(value, str) or not _TOKEN.fullmatch(value):
        raise CohortError(
            f"invalid {what} {value!r}: expected [A-Za-z0-9._:-], starting alphanumeric, "
            "at most 128 characters"
        )
    return value


def _text(what: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CohortError(f"{what} must be a non-empty string")
    return value


def _sha256(what: str, value: object) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise CohortError(f"{what} must be a lowercase hex SHA-256, got {value!r}")
    return value


def _unique(what: str, values: Sequence[str]) -> None:
    dupes = sorted(v for v, n in Counter(values).items() if n > 1)
    if dupes:
        raise CohortError(f"duplicate {what}: {', '.join(dupes)}")


# --------------------------------------------------------------------------- vocabularies


class EligibilityStatus(StrEnum):
    """SPEC-008 §6.3. Every status except `eligible` and `pending_review` is an exclusion."""

    ELIGIBLE = "eligible"
    EXCLUDED_CONFLICT = "excluded_conflict"
    EXCLUDED_MISSING_LABEL = "excluded_missing_label"
    EXCLUDED_MISSING_GROUP = "excluded_missing_group"
    EXCLUDED_DUPLICATE = "excluded_duplicate"
    EXCLUDED_MISSING_WSI = "excluded_missing_wsi"
    EXCLUDED_QUALITY = "excluded_quality"
    PENDING_REVIEW = "pending_review"


class SourceRole(StrEnum):
    ROSTER = "roster"  # lists every slide of the cohort; carries no label
    AUTHORITY = "authority"  # an approved label authority
    CORROBORATING = "corroborating"  # must agree with the authority; never sufficient alone


class ConflictStatus(StrEnum):
    NONE = "none"
    LABEL = "label"
    PARTITION = "partition"
    LABEL_AND_PARTITION = "label_and_partition"


class ResolutionStatus(StrEnum):
    NOT_REQUIRED = "not_required"
    UNRESOLVED = "unresolved"
    APPROVED = "approved"


class DecisionStatus(StrEnum):
    APPROVED = "approved"
    PENDING = "pending"


class WSIStatus(StrEnum):
    NOT_CHECKED = "not_checked"  # no inventory has looked at the image yet
    VERIFIED = "verified"  # present, size and SHA-256 recorded
    MISSING = "missing"  # an inventory looked and did not find it
    UNREADABLE = "unreadable"  # present but cannot be opened consistently


# --------------------------------------------------------------------------- definition


@dataclass(frozen=True, slots=True)
class Decision:
    """An owner decision. Approved decisions name who approved them, when and where."""

    decision_id: str
    summary: str
    status: DecisionStatus
    gates: tuple[str, ...]
    approver: str | None = None
    approved_on: str | None = None
    reference: str | None = None

    def __post_init__(self) -> None:
        _token("decision id", self.decision_id)
        _text(f"decision {self.decision_id} summary", self.summary)
        if not self.gates:
            raise CohortError(f"decision {self.decision_id} gates nothing")
        for gate in self.gates:
            _token(f"decision {self.decision_id} gate", gate)
        approval = (self.approver, self.approved_on, self.reference)
        if self.status is DecisionStatus.APPROVED:
            if not all(isinstance(v, str) and v.strip() for v in approval):
                raise CohortError(
                    f"approved decision {self.decision_id} needs an approver, a date and a "
                    "reference"
                )
            try:
                date.fromisoformat(str(self.approved_on))
            except ValueError:
                raise CohortError(
                    f"decision {self.decision_id}: approved_on must be an ISO date"
                ) from None
        elif self.approver is not None or self.approved_on is not None:
            raise CohortError(f"pending decision {self.decision_id} carries an approval")


@dataclass(frozen=True, slots=True)
class CohortTarget:
    """The curated target. Each source names its own target; only accepted ones may label it."""

    name: str
    classes: tuple[str, ...]
    positive_class: str | None
    accepted_source_targets: tuple[str, ...]
    decision: str

    def __post_init__(self) -> None:
        _token("target name", self.name)
        if len(self.classes) < 2:
            raise CohortError("a target needs at least two classes")
        for c in self.classes:
            _token("class name", c)
        _unique("class name", self.classes)
        if self.positive_class is not None and (
            len(self.classes) != 2 or self.positive_class != self.classes[1]
        ):
            raise CohortError(
                "a binary target lists the negative class first and names the second as "
                "positive_class"
            )
        if not self.accepted_source_targets:
            raise CohortError("a target accepts at least one source target")
        for t in self.accepted_source_targets:
            _token("source target", t)
        _unique("accepted source target", self.accepted_source_targets)


@dataclass(frozen=True, slots=True)
class SlidePolicy:
    """Which slide indices of a case enter the analysis (others are excluded by policy)."""

    included_slide_indices: tuple[str, ...]
    decision: str

    def __post_init__(self) -> None:
        if not self.included_slide_indices:
            raise CohortError("the slide policy includes no slide index")
        for s in self.included_slide_indices:
            _token("slide index", s)


@dataclass(frozen=True, slots=True)
class SplitPolicy:
    """How partitions are obtained. Only an imported (external) split is supported: a cohort
    manifest never generates or repairs a split."""

    strategy: str
    test_role: str
    pilot_partitions: tuple[str, ...]
    decisions: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.strategy != EXTERNAL_STRATEGY:
            raise CohortError(
                f"cohort splits are imported ({EXTERNAL_STRATEGY}), never generated; got "
                f"{self.strategy!r}"
            )
        _token("test role", self.test_role)
        for p in self.pilot_partitions:
            if p not in PARTITIONS:
                raise CohortError(f"unknown pilot partition {p!r}")


@dataclass(frozen=True, slots=True)
class CohortDefinition:
    cohort_id: str
    version: str
    target: CohortTarget
    label_authority_decision: str
    group_decision: str
    slide_policy: SlidePolicy
    split_policy: SplitPolicy
    decisions: tuple[Decision, ...]
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _token("cohort id", self.cohort_id)
        _token("cohort version", self.version)
        ids = [d.decision_id for d in self.decisions]
        _unique("decision id", ids)
        referenced = (
            self.target.decision,
            self.label_authority_decision,
            self.group_decision,
            self.slide_policy.decision,
            *self.split_policy.decisions,
        )
        unknown = sorted(set(referenced) - set(ids))
        if unknown:
            raise CohortError(f"policies reference undeclared decision(s): {', '.join(unknown)}")
        ungated = sorted(
            d.decision_id
            for d in self.decisions
            if d.decision_id in referenced and RECONCILIATION_GATE not in d.gates
        )
        if ungated:
            raise CohortError(
                f"decision(s) {', '.join(ungated)} drive reconciliation but do not gate "
                f"{RECONCILIATION_GATE!r}"
            )

    def pending(self, gate: str) -> tuple[str, ...]:
        return tuple(
            d.decision_id
            for d in self.decisions
            if gate in d.gates and d.status is DecisionStatus.PENDING
        )


# --------------------------------------------------------------------------- observations


@dataclass(frozen=True, slots=True)
class SourceFile:
    """One declared source file, identified by content. Paths never enter a manifest."""

    source_id: str
    role: SourceRole
    file_name: str
    sha256: str
    rows: int
    source_target: str | None
    partition: str | None
    provenance: str

    def __post_init__(self) -> None:
        _token("source id", self.source_id)
        _text("source file name", self.file_name)
        if "/" in self.file_name or "\\" in self.file_name:
            raise CohortError(f"source file name must not be a path: {self.file_name!r}")
        _sha256(f"source {self.source_id} digest", self.sha256)
        if isinstance(self.rows, bool) or self.rows < 0:
            raise CohortError("source row count must be a non-negative integer")
        if (self.role is SourceRole.ROSTER) != (self.source_target is None):
            raise CohortError(
                f"source {self.source_id}: a roster names no target, a label source must"
            )
        if self.source_target is not None:
            _token("source target", self.source_target)
        if self.partition is not None and self.partition not in PARTITIONS:
            raise CohortError(f"source {self.source_id}: unknown partition {self.partition!r}")
        _text("source provenance", self.provenance)


@dataclass(frozen=True, slots=True)
class RosterEntry:
    """One slide the cohort contains, independent of whether any source labels it."""

    slide_id: str
    group_id: str
    cohort: str
    slide_index: str
    file_name: str
    source_id: str
    row: int

    def __post_init__(self) -> None:
        _token("slide id", self.slide_id)
        _token("group id", self.group_id)
        _token("cohort", self.cohort)
        _token("slide index", self.slide_index)
        _text("image file name", self.file_name)
        _token("source id", self.source_id)


@dataclass(frozen=True, slots=True)
class LabelCandidate:
    """One label as a source states it. `slide_id` is `None` when the source labels a case
    (group) rather than a slide."""

    source_id: str
    role: SourceRole
    file_name: str
    row: int
    raw_id: str
    raw_value: str
    value: str
    source_target: str
    group_id: str
    slide_id: str | None
    partition: str | None

    def __post_init__(self) -> None:
        if self.role is SourceRole.ROSTER:
            raise CohortError("a roster source cannot state a label")
        _token("source id", self.source_id)
        _token("group id", self.group_id)
        if self.slide_id is not None:
            _token("slide id", self.slide_id)
        if self.partition is not None and self.partition not in PARTITIONS:
            raise CohortError(f"unknown partition {self.partition!r}")
        if isinstance(self.row, bool) or self.row < 1:
            raise CohortError("source rows are numbered from 1")

    @property
    def location(self) -> str:
        return f"{self.source_id}:{self.file_name}:{self.row}"


@dataclass(frozen=True, slots=True)
class CohortObservations:
    """Everything read from the declared sources, before any reconciliation."""

    sources: tuple[SourceFile, ...]
    roster: tuple[RosterEntry, ...]
    candidates: tuple[LabelCandidate, ...]


@dataclass(frozen=True, slots=True)
class WSIEvidence:
    status: WSIStatus
    file_name: str
    size_bytes: int | None = None
    sha256: str | None = None
    note: str = ""

    def __post_init__(self) -> None:
        _text("image file name", self.file_name)
        if self.status is WSIStatus.VERIFIED:
            if self.size_bytes is None or self.size_bytes < 1 or self.sha256 is None:
                raise CohortError(f"verified image {self.file_name} needs a size and a SHA-256")
            _sha256(f"image {self.file_name} digest", self.sha256)
        elif self.sha256 is not None:
            raise CohortError(f"image {self.file_name}: only a verified image has a digest")


@dataclass(frozen=True, slots=True)
class Resolution:
    """An owner-approved resolution of one conflicting record."""

    slide_id: str
    value: str
    partition: str
    resolver: str
    authority: str
    rationale: str
    approved_on: str

    def __post_init__(self) -> None:
        _token("slide id", self.slide_id)
        if self.partition not in PARTITIONS:
            raise CohortError(f"resolution for {self.slide_id}: unknown partition")
        for what in ("resolver", "authority", "rationale"):
            _text(f"resolution {what}", getattr(self, what))
        try:
            date.fromisoformat(self.approved_on)
        except ValueError:
            raise CohortError("resolution approved_on must be an ISO date") from None


# --------------------------------------------------------------------------- records


@dataclass(frozen=True, slots=True)
class CohortRecord:
    """One roster slide after reconciliation."""

    slide_id: str
    group_id: str
    cohort: str
    slide_index: str
    wsi: WSIEvidence
    status: EligibilityStatus
    reason_code: str | None
    reason: str | None
    label: str | None
    source_targets: tuple[str, ...]
    partition: str | None
    conflict: ConflictStatus
    resolution_status: ResolutionStatus
    resolution: Resolution | None
    candidates: tuple[LabelCandidate, ...]
    outstanding_gates: tuple[str, ...]

    def __post_init__(self) -> None:
        _token("slide id", self.slide_id)
        _token("group id", self.group_id)
        _token("cohort", self.cohort)
        eligible = self.status is EligibilityStatus.ELIGIBLE
        if eligible == bool(self.reason_code):
            raise CohortError(
                f"slide {self.slide_id}: every non-eligible record, and only those, has a reason"
            )
        if bool(self.reason_code) != bool(self.reason):
            raise CohortError(f"slide {self.slide_id}: reason code and reason go together")
        if (self.conflict is ConflictStatus.NONE) != (
            self.resolution_status is ResolutionStatus.NOT_REQUIRED
        ):
            raise CohortError(f"slide {self.slide_id}: only a conflict needs a resolution")
        if (self.resolution_status is ResolutionStatus.APPROVED) != (self.resolution is not None):
            raise CohortError(f"slide {self.slide_id}: approved resolution record mismatch")
        if self.status is EligibilityStatus.EXCLUDED_CONFLICT and (
            self.resolution_status is not ResolutionStatus.UNRESOLVED
        ):
            raise CohortError(f"slide {self.slide_id}: excluded_conflict needs an open conflict")
        if eligible:
            if self.label is None or self.partition is None:
                raise CohortError(f"eligible slide {self.slide_id} needs a label and a partition")
            if self.resolution_status is ResolutionStatus.UNRESOLVED:
                raise CohortError(f"slide {self.slide_id} has an unresolved conflict")
            if not any(c.role is SourceRole.AUTHORITY for c in self.candidates) and (
                self.resolution is None
            ):
                raise CohortError(f"eligible slide {self.slide_id} has no authoritative label")
        elif self.outstanding_gates:
            raise CohortError(f"slide {self.slide_id}: only eligible records have open gates")
        keys = [c.location for c in self.candidates]
        if keys != sorted(keys):
            raise CohortError(f"slide {self.slide_id}: candidates must be sorted by location")

    @property
    def provisional(self) -> bool:
        """Eligible, but integrity gates that need the image are still open."""
        return self.status is EligibilityStatus.ELIGIBLE and bool(self.outstanding_gates)


@dataclass(frozen=True, slots=True)
class PartitionCount:
    partition: str
    slides: int
    groups: int
    class_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class CohortSplit:
    """The imported split over the eligible records (SPEC-007 `external@1`)."""

    strategy: str
    origin: tuple[str, ...]
    test_role: str
    pilot_partitions: tuple[str, ...]
    decisions: tuple[str, ...]
    assignment_digest: str
    partitions: tuple[PartitionCount, ...]


@dataclass(frozen=True, slots=True)
class StatusCount:
    status: EligibilityStatus
    reason_code: str
    partition: str
    label: str
    slides: int


@dataclass(frozen=True, slots=True)
class CohortManifest:
    """A versioned, content-addressed curated cohort (`pathology-ai.cohort/1`)."""

    cohort_id: str
    version: str
    target: CohortTarget
    label_authority_decision: str
    group_decision: str
    slide_policy: SlidePolicy
    decisions: tuple[Decision, ...]
    sources: tuple[SourceFile, ...]
    records: tuple[CohortRecord, ...]
    split: CohortSplit
    counts: tuple[StatusCount, ...]
    limitations: tuple[str, ...] = ()
    schema_version: str = COHORT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != COHORT_SCHEMA:
            raise CohortError(f"unsupported cohort schema {self.schema_version!r}")
        definition = self.definition  # re-validates policy references
        if not self.records:
            raise CohortError("a cohort manifest needs at least one record")
        ids = [r.slide_id for r in self.records]
        if ids != sorted(ids):
            raise CohortError("cohort records must be sorted by slide id")
        _unique("slide id", ids)
        files = {(s.source_id, s.file_name): s for s in self.sources}
        if len(files) != len(self.sources):
            raise CohortError("a source file is listed twice")
        pending = definition.pending(RECONCILIATION_GATE)
        for r in self.records:
            for c in r.candidates:
                if (c.source_id, c.file_name) not in files:
                    raise CohortError(f"slide {r.slide_id} cites undeclared source {c.location}")
            if r.status is EligibilityStatus.ELIGIBLE:
                if pending:
                    raise CohortError(
                        f"slide {r.slide_id} is eligible while decision(s) "
                        f"{', '.join(pending)} are pending"
                    )
                if r.label not in self.target.classes:
                    raise CohortError(f"slide {r.slide_id}: label outside the class vocabulary")
                unaccepted = set(r.source_targets) - set(self.target.accepted_source_targets)
                if unaccepted:
                    raise CohortError(f"slide {r.slide_id}: unaccepted source target(s)")
        _check_groups(self.records)
        split = derive_split(self.records, self.target, definition.split_policy, self.split.origin)
        if split != self.split:
            raise CohortError("the recorded split does not re-derive from the records")
        if derive_counts(self.records) != self.counts:
            raise CohortError("the recorded counts do not re-derive from the records")

    @property
    def definition(self) -> CohortDefinition:
        return CohortDefinition(
            cohort_id=self.cohort_id,
            version=self.version,
            target=self.target,
            label_authority_decision=self.label_authority_decision,
            group_decision=self.group_decision,
            slide_policy=self.slide_policy,
            split_policy=SplitPolicy(
                self.split.strategy,
                self.split.test_role,
                self.split.pilot_partitions,
                self.split.decisions,
            ),
            decisions=self.decisions,
            limitations=self.limitations,
        )

    @property
    def ref(self) -> str:
        return f"{self.cohort_id}@{self.version}"

    @property
    def digest(self) -> str:
        return digest(self)

    @property
    def manifest_id(self) -> str:
        return short_digest(self, "cohort")

    def eligible(self) -> tuple[CohortRecord, ...]:
        return tuple(r for r in self.records if r.status is EligibilityStatus.ELIGIBLE)

    def record(self, slide_id: str) -> CohortRecord:
        for r in self.records:
            if r.slide_id == slide_id:
                return r
        raise CohortError(f"slide {slide_id!r} is not in cohort {self.ref}")


# --------------------------------------------------------------------------- derivations


def _check_groups(records: Iterable[CohortRecord]) -> None:
    """A group never spans sub-cohorts, and its labelled slides never span partitions."""
    cohorts: dict[str, set[str]] = defaultdict(set)
    partitions: dict[str, set[str]] = defaultdict(set)
    for r in records:
        cohorts[r.group_id].add(r.cohort)
        if r.partition is not None and r.status in (
            EligibilityStatus.ELIGIBLE,
            EligibilityStatus.PENDING_REVIEW,
        ):
            partitions[r.group_id].add(r.partition)
    mixed = sorted(g for g, c in cohorts.items() if len(c) > 1)
    if mixed:
        raise CohortReconciliationError(
            f"{len(mixed)} group(s) span several sub-cohorts ({', '.join(mixed[:5])}); the group "
            "key is not qualified by its sub-cohort"
        )
    leaking = sorted(g for g, p in partitions.items() if len(p) > 1)
    if leaking:
        raise GroupLeakageError(
            f"{len(leaking)} group(s) have slides in several partitions: "
            f"{', '.join(leaking[:10])}; the split is not repaired automatically"
        )


def split_assignment(records: Iterable[CohortRecord]) -> tuple[tuple[str, str], ...]:
    """Sorted (slide id, partition) pairs of every eligible record."""
    return tuple(
        sorted(
            (r.slide_id, r.partition)
            for r in records
            if r.status is EligibilityStatus.ELIGIBLE and r.partition is not None
        )
    )


def derive_split(
    records: Sequence[CohortRecord],
    target: CohortTarget,
    policy: SplitPolicy,
    origin: tuple[str, ...],
) -> CohortSplit:
    eligible = [r for r in records if r.status is EligibilityStatus.ELIGIBLE]
    groups: dict[str, set[str]] = {p: set() for p in PARTITIONS}
    counts: dict[str, Counter[str]] = {p: Counter() for p in PARTITIONS}
    slides: Counter[str] = Counter()
    for r in eligible:
        assert r.partition is not None and r.label is not None
        groups[r.partition].add(r.group_id)
        counts[r.partition][r.label] += 1
        slides[r.partition] += 1
    overlap = sorted(
        g
        for i, a in enumerate(PARTITIONS)
        for b in PARTITIONS[i + 1 :]
        for g in groups[a] & groups[b]
    )
    if overlap:
        raise GroupLeakageError(f"groups shared between partitions: {', '.join(overlap[:10])}")
    return CohortSplit(
        strategy=policy.strategy,
        origin=origin,
        test_role=policy.test_role,
        pilot_partitions=policy.pilot_partitions,
        decisions=policy.decisions,
        assignment_digest=digest(split_assignment(records)),
        partitions=tuple(
            PartitionCount(
                p,
                slides[p],
                len(groups[p]),
                tuple((c, counts[p].get(c, 0)) for c in target.classes),
            )
            for p in PARTITIONS
        ),
    )


def derive_counts(records: Iterable[CohortRecord]) -> tuple[StatusCount, ...]:
    """Slides by status, reason, original partition and label: the inclusion/exclusion flow."""
    c: Counter[tuple[EligibilityStatus, str, str, str]] = Counter(
        (r.status, r.reason_code or "", r.partition or UNASSIGNED, r.label or NO_LABEL)
        for r in records
    )
    order = list(EligibilityStatus)
    return tuple(
        StatusCount(s, reason, p, label, n)
        for (s, reason, p, label), n in sorted(
            c.items(), key=lambda kv: (order.index(kv[0][0]), *kv[0][1:])
        )
    )


# --------------------------------------------------------------------------- reconciliation


def _conflict(values: set[str], partitions: set[str]) -> ConflictStatus:
    if len(values) > 1 and len(partitions) > 1:
        return ConflictStatus.LABEL_AND_PARTITION
    if len(values) > 1:
        return ConflictStatus.LABEL
    if len(partitions) > 1:
        return ConflictStatus.PARTITION
    return ConflictStatus.NONE


def _attach(
    definition: CohortDefinition,
    roster: Mapping[str, RosterEntry],
    candidates: Iterable[LabelCandidate],
) -> dict[str, list[LabelCandidate]]:
    by_group: dict[str, list[str]] = defaultdict(list)
    for e in roster.values():
        by_group[e.group_id].append(e.slide_id)
    attached: dict[str, list[LabelCandidate]] = defaultdict(list)
    for c in candidates:
        where = c.location
        if c.source_target not in definition.target.accepted_source_targets:
            raise CohortReconciliationError(
                f"{where}: source target {c.source_target!r} is not accepted for target "
                f"{definition.target.name!r}"
            )
        if c.value not in definition.target.classes:
            raise CohortReconciliationError(f"{where}: label {c.value!r} is not a target class")
        if c.slide_id is not None:
            entry = roster.get(c.slide_id)
            if entry is None:
                raise CohortReconciliationError(
                    f"{where}: slide {c.slide_id} ({c.raw_id!r}) is not in the roster"
                )
            if entry.group_id != c.group_id:
                raise CohortReconciliationError(
                    f"{where}: slide {c.slide_id} belongs to group {entry.group_id}, the source "
                    f"says {c.group_id}"
                )
            attached[c.slide_id].append(c)
            continue
        members = by_group.get(c.group_id, [])
        if len(members) != 1:
            raise CohortReconciliationError(
                f"{where}: {c.raw_id!r} labels case {c.group_id}, which has {len(members)} "
                "roster slide(s); a case label applies to a slide only when the case has exactly "
                "one, and inheriting it across slides is an owner decision"
            )
        attached[members[0]].append(c)
    return attached


def reconcile(
    definition: CohortDefinition,
    observations: CohortObservations,
    wsi: Mapping[str, WSIEvidence] | None = None,
    resolutions: Sequence[Resolution] = (),
) -> tuple[CohortRecord, ...]:
    """One record per roster slide. Deterministic, and never chooses among disagreeing sources."""
    roster: dict[str, RosterEntry] = {}
    for e in observations.roster:
        if e.slide_id in roster:
            raise CohortReconciliationError(
                f"slide {e.slide_id} is listed twice in the roster "
                f"({roster[e.slide_id].file_name}, {e.file_name})"
            )
        roster[e.slide_id] = e
    if not roster:
        raise CohortReconciliationError("the roster lists no slide")
    by_resolution = {r.slide_id: r for r in resolutions}
    if len(by_resolution) != len(resolutions):
        raise CohortReconciliationError("a slide has more than one resolution")
    unknown = sorted(set(by_resolution) - set(roster))
    if unknown:
        raise CohortReconciliationError(f"resolutions for unknown slides: {', '.join(unknown)}")
    for res in by_resolution.values():
        if res.value not in definition.target.classes:
            raise CohortReconciliationError(f"resolution for {res.slide_id}: unknown class")
    attached = _attach(definition, roster, observations.candidates)
    evidence = dict(wsi or {})
    stray = sorted(set(evidence) - set(roster))
    if stray:
        raise CohortReconciliationError(f"image evidence for unknown slides: {', '.join(stray)}")
    digests = Counter(
        e.sha256 for e in evidence.values() if e.status is WSIStatus.VERIFIED and e.sha256
    )
    pending = definition.pending(RECONCILIATION_GATE)
    records = []
    for slide_id in sorted(roster):
        entry = roster[slide_id]
        cands = tuple(sorted(attached.get(slide_id, []), key=lambda c: c.location))
        values = {c.value for c in cands}
        parts = {c.partition for c in cands if c.partition is not None}
        conflict = _conflict(values, parts)
        resolution = by_resolution.get(slide_id)
        if resolution is not None and conflict is ConflictStatus.NONE:
            raise CohortReconciliationError(f"slide {slide_id} has a resolution but no conflict")
        authority = tuple(
            sorted({c.source_target for c in cands if c.role is SourceRole.AUTHORITY})
        )
        if resolution is not None:
            label: str | None = resolution.value
            partition: str | None = resolution.partition
        else:
            label = next(iter(values)) if len(values) == 1 else None
            partition = next(iter(parts)) if len(parts) == 1 else None
        image = evidence.get(slide_id, WSIEvidence(WSIStatus.NOT_CHECKED, entry.file_name))

        status, code, reason = _status(
            definition, entry, cands, conflict, resolution, partition, image, digests, pending
        )
        eligible = status is EligibilityStatus.ELIGIBLE
        records.append(
            CohortRecord(
                slide_id=slide_id,
                group_id=entry.group_id,
                cohort=entry.cohort,
                slide_index=entry.slide_index,
                wsi=image,
                status=status,
                reason_code=code,
                reason=reason,
                label=label,
                source_targets=authority,
                partition=partition,
                conflict=conflict,
                resolution_status=(
                    ResolutionStatus.NOT_REQUIRED
                    if conflict is ConflictStatus.NONE
                    else ResolutionStatus.APPROVED
                    if resolution is not None
                    else ResolutionStatus.UNRESOLVED
                ),
                resolution=resolution,
                candidates=cands,
                outstanding_gates=(
                    WSI_GATES if eligible and image.status is not WSIStatus.VERIFIED else ()
                ),
            )
        )
    _check_groups(records)
    return tuple(records)


def _status(
    definition: CohortDefinition,
    entry: RosterEntry,
    cands: tuple[LabelCandidate, ...],
    conflict: ConflictStatus,
    resolution: Resolution | None,
    partition: str | None,
    image: WSIEvidence,
    digests: Counter[str],
    pending: tuple[str, ...],
) -> tuple[EligibilityStatus, str | None, str | None]:
    """The first failing gate, in a fixed order, decides a record's status."""
    S = EligibilityStatus
    policy = definition.slide_policy
    if entry.slide_index not in policy.included_slide_indices:
        if cands:
            return (
                S.PENDING_REVIEW,
                "labelled_slide_outside_slide_policy",
                f"slide index {entry.slide_index} is excluded by decision {policy.decision}, "
                "but a source labels this slide",
            )
        return (
            S.EXCLUDED_MISSING_LABEL,
            "slide_policy",
            f"slide index {entry.slide_index} is not among {list(policy.included_slide_indices)} "
            f"(decision {policy.decision}); no source labels this slide",
        )
    if conflict is not ConflictStatus.NONE and resolution is None:
        return (
            S.EXCLUDED_CONFLICT,
            "source_conflict",
            f"sources disagree on the {conflict.value.replace('_', ' and ')}: "
            + "; ".join(f"{c.location}={c.raw_value}/{c.partition}" for c in cands),
        )
    if not cands:
        return S.EXCLUDED_MISSING_LABEL, "no_label", "no declared source labels this slide or case"
    if resolution is None and not any(c.role is SourceRole.AUTHORITY for c in cands):
        return (
            S.PENDING_REVIEW,
            "no_authoritative_label",
            f"only corroborating sources label this slide (decision "
            f"{definition.label_authority_decision})",
        )
    if partition is None:
        return S.PENDING_REVIEW, "no_partition", "the sources assign no partition"
    if image.status is WSIStatus.MISSING:
        return S.EXCLUDED_MISSING_WSI, "wsi_missing", image.note or "the image was not found"
    if image.status is WSIStatus.UNREADABLE:
        return S.EXCLUDED_QUALITY, "wsi_unreadable", image.note or "the image cannot be read"
    if image.sha256 is not None and digests[image.sha256] > 1:
        return (
            S.EXCLUDED_DUPLICATE,
            "duplicate_wsi_content",
            f"image content {str(image.sha256)[:16]}… is shared with another slide",
        )
    if pending:
        return (
            S.PENDING_REVIEW,
            "decision_pending",
            f"decision(s) {', '.join(pending)} gating reconciliation are pending",
        )
    return S.ELIGIBLE, None, None


def build_manifest(
    definition: CohortDefinition,
    observations: CohortObservations,
    wsi: Mapping[str, WSIEvidence] | None = None,
    resolutions: Sequence[Resolution] = (),
) -> CohortManifest:
    records = reconcile(definition, observations, wsi, resolutions)
    origin = tuple(
        sorted(
            {
                s.source_id
                for s in observations.sources
                if s.role is not SourceRole.ROSTER and s.partition is not None
            }
            | {c.source_id for c in observations.candidates if c.partition is not None}
        )
    )
    return CohortManifest(
        cohort_id=definition.cohort_id,
        version=definition.version,
        target=definition.target,
        label_authority_decision=definition.label_authority_decision,
        group_decision=definition.group_decision,
        slide_policy=definition.slide_policy,
        decisions=definition.decisions,
        sources=tuple(sorted(observations.sources, key=lambda s: (s.source_id, s.file_name))),
        records=records,
        split=derive_split(records, definition.target, definition.split_policy, origin),
        counts=derive_counts(records),
        limitations=definition.limitations,
    )
