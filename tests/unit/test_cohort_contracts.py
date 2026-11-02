"""SPEC-008 cohort contracts: pure reconciliation rules (synthetic records only)."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from domain.cohort import (
    WSI_GATES,
    CohortManifest,
    CohortRecord,
    ConflictStatus,
    Decision,
    DecisionStatus,
    EligibilityStatus,
    LabelCandidate,
    Resolution,
    ResolutionStatus,
    RosterEntry,
    SourceRole,
    SplitPolicy,
    StatusCount,
    WSIEvidence,
    WSIStatus,
    build_manifest,
    reconcile,
)
from domain.errors import CohortError, CohortReconciliationError, GroupLeakageError
from tests.cohort_support import candidate, definition, entry, observations

S = EligibilityStatus


def _one(
    roster: list[RosterEntry], cands: list[LabelCandidate], **kw: Any
) -> tuple[CohortRecord, ...]:
    return reconcile(definition(), observations(roster, cands), **kw)


def test_consistent_sources_make_a_provisional_eligible_record() -> None:
    (r,) = _one(
        [entry("AA_1_01", "g:AA:1")],
        [
            candidate("AA_1_01", "g:AA:1"),
            candidate("AA_1_01", "g:AA:1", source="other", role=SourceRole.CORROBORATING),
        ],
    )
    assert r.status is S.ELIGIBLE and r.label == "neg" and r.partition == "train"
    assert r.conflict is ConflictStatus.NONE
    assert r.resolution_status is ResolutionStatus.NOT_REQUIRED
    assert r.outstanding_gates == WSI_GATES and r.provisional
    assert [c.source_id for c in r.candidates] == ["labels", "other"]


@pytest.mark.parametrize(
    "second,conflict",
    [
        (dict(value="pos"), ConflictStatus.LABEL),
        (dict(partition="test"), ConflictStatus.PARTITION),
        (dict(value="pos", partition="test"), ConflictStatus.LABEL_AND_PARTITION),
    ],
)
def test_disagreeing_sources_are_excluded_never_chosen(
    second: dict[str, Any], conflict: ConflictStatus
) -> None:
    (r,) = _one(
        [entry("AA_1_01", "g:AA:1")],
        [
            candidate("AA_1_01", "g:AA:1"),
            candidate("AA_1_01", "g:AA:1", source="other", role=SourceRole.CORROBORATING, **second),
        ],
    )
    assert r.status is S.EXCLUDED_CONFLICT and r.conflict is conflict
    assert r.resolution_status is ResolutionStatus.UNRESOLVED
    assert r.reason_code == "source_conflict" and "labels:labels.csv:2" in (r.reason or "")


def test_conflict_within_one_source_across_partitions_is_a_conflict() -> None:
    (r,) = _one(
        [entry("AA_1_01", "g:AA:1")],
        [candidate("AA_1_01", "g:AA:1"), candidate("AA_1_01", "g:AA:1", partition="test", row=3)],
    )
    assert r.status is S.EXCLUDED_CONFLICT and r.conflict is ConflictStatus.PARTITION


def test_only_an_approved_resolution_admits_a_conflicting_record() -> None:
    roster = [entry("AA_1_01", "g:AA:1")]
    cands = [
        candidate("AA_1_01", "g:AA:1"),
        candidate("AA_1_01", "g:AA:1", "pos", source="other", role=SourceRole.CORROBORATING),
    ]
    res = Resolution("AA_1_01", "pos", "train", "owner", "path report 1", "re-review", "2026-11-02")
    (r,) = _one(roster, cands, resolutions=[res])
    assert r.status is S.ELIGIBLE and r.label == "pos"
    assert r.resolution_status is ResolutionStatus.APPROVED and r.resolution == res
    with pytest.raises(CohortReconciliationError, match="no conflict"):
        _one(roster, cands[:1], resolutions=[res])
    with pytest.raises(CohortError, match="rationale"):
        Resolution("AA_1_01", "pos", "train", "owner", "x", " ", "2026-11-02")


def test_missing_and_non_authoritative_labels() -> None:
    records = _one(
        [entry("AA_1_01", "g:AA:1"), entry("AA_2_01", "g:AA:2")],
        [candidate("AA_2_01", "g:AA:2", source="other", role=SourceRole.CORROBORATING)],
    )
    a, b = records
    assert a.status is S.EXCLUDED_MISSING_LABEL and a.reason_code == "no_label"
    assert b.status is S.PENDING_REVIEW and b.reason_code == "no_authoritative_label"


def test_case_level_label_applies_only_to_a_single_slide_case() -> None:
    (r,) = _one([entry("AA_1_01", "g:AA:1")], [candidate(None, "g:AA:1")])
    assert r.status is S.ELIGIBLE and r.slide_id == "AA_1_01"
    with pytest.raises(CohortReconciliationError, match="exactly"):
        _one(
            [entry("AA_1_01", "g:AA:1"), entry("AA_1_02", "g:AA:1", index="02")],
            [candidate(None, "g:AA:1")],
        )


def test_second_slides_are_excluded_by_policy_and_never_inherit() -> None:
    records = _one(
        [entry("BB_2_01", "g:BB:2", "BB"), entry("BB_2_02", "g:BB:2", "BB", "02")],
        [candidate("BB_2_01", "g:BB:2")],
    )
    first, second = records
    assert first.status is S.ELIGIBLE
    assert second.status is S.EXCLUDED_MISSING_LABEL and second.reason_code == "slide_policy"
    assert second.label is None and second.partition is None


def test_included_second_slide_in_another_partition_is_group_leakage() -> None:
    d = definition(
        slide_policy=replace(definition().slide_policy, included_slide_indices=("01", "02"))
    )
    roster = [entry("BB_2_01", "g:BB:2", "BB"), entry("BB_2_02", "g:BB:2", "BB", "02")]
    cands = [
        candidate("BB_2_01", "g:BB:2"),
        candidate("BB_2_02", "g:BB:2", partition="test", row=3),
    ]
    with pytest.raises(GroupLeakageError, match="g:BB:2"):
        reconcile(d, observations(roster, cands))


def test_unqualified_group_key_spanning_sub_cohorts_fails_closed() -> None:
    with pytest.raises(CohortReconciliationError, match="sub-cohort"):
        _one(
            [entry("AA_198_01", "T198", "AA"), entry("BB_198_01", "T198", "BB")],
            [candidate("AA_198_01", "T198"), candidate("BB_198_01", "T198", row=3)],
        )


def test_qualified_keys_keep_colliding_case_numbers_apart() -> None:
    records = _one(
        [entry("AA_198_01", "g:AA:198", "AA"), entry("BB_198_01", "g:BB:198", "BB")],
        [candidate("AA_198_01", "g:AA:198"), candidate("BB_198_01", "g:BB:198", "pos", "test")],
    )
    assert [(r.group_id, r.label, r.partition) for r in records] == [
        ("g:AA:198", "neg", "train"),
        ("g:BB:198", "pos", "test"),
    ]


def test_label_for_a_slide_outside_the_roster_fails_closed() -> None:
    with pytest.raises(CohortReconciliationError, match="not in the roster"):
        _one([entry("AA_1_01", "g:AA:1")], [candidate("AA_9_01", "g:AA:9")])


def test_unaccepted_source_target_is_rejected() -> None:
    with pytest.raises(CohortReconciliationError, match="not accepted"):
        _one([entry("AA_1_01", "g:AA:1")], [candidate("AA_1_01", "g:AA:1", target="OTHER")])


def test_pending_reconciliation_decision_blocks_every_eligibility() -> None:
    d = definition()
    pending = tuple(
        Decision(x.decision_id, x.summary, DecisionStatus.PENDING, x.gates)
        if x.decision_id == "D-1"
        else x
        for x in d.decisions
    )
    (r,) = reconcile(
        replace(d, decisions=pending),
        observations([entry("AA_1_01", "g:AA:1")], [candidate("AA_1_01", "g:AA:1")]),
    )
    assert r.status is S.PENDING_REVIEW and r.reason_code == "decision_pending"


def test_decisions_need_complete_approvals() -> None:
    with pytest.raises(CohortError, match="approver"):
        Decision("D-1", "x", DecisionStatus.APPROVED, ("reconciliation",))
    with pytest.raises(CohortError, match="ISO date"):
        Decision("D-1", "x", DecisionStatus.APPROVED, ("reconciliation",), "o", "yesterday", "r")
    with pytest.raises(CohortError, match="carries an approval"):
        Decision("D-1", "x", DecisionStatus.PENDING, ("reconciliation",), "o")
    with pytest.raises(CohortError, match="undeclared"):
        definition(group_decision="D-404")


def test_split_policy_never_generates() -> None:
    with pytest.raises(CohortError, match="never generated"):
        SplitPolicy("group-stratified@1", "fresh", (), ())


def test_wsi_evidence_drives_missing_duplicate_and_gates() -> None:
    roster = [entry(f"AA_{i}_01", f"g:AA:{i}") for i in (1, 2, 3, 4)]
    cands = [candidate(f"AA_{i}_01", f"g:AA:{i}", row=i + 1) for i in (1, 2, 3, 4)]
    same = "a" * 64
    wsi = {
        "AA_1_01": WSIEvidence(WSIStatus.VERIFIED, "AA_1_01.img", 10, "b" * 64),
        "AA_2_01": WSIEvidence(WSIStatus.MISSING, "AA_2_01.img", note="not in bucket"),
        "AA_3_01": WSIEvidence(WSIStatus.VERIFIED, "AA_3_01.img", 10, same),
        "AA_4_01": WSIEvidence(WSIStatus.VERIFIED, "AA_4_01.img", 10, same),
    }
    a, b, c, d = _one(roster, cands, wsi=wsi)
    assert a.status is S.ELIGIBLE and a.outstanding_gates == () and not a.provisional
    assert b.status is S.EXCLUDED_MISSING_WSI and b.reason == "not in bucket"
    assert c.status is d.status is S.EXCLUDED_DUPLICATE
    with pytest.raises(CohortError, match="digest"):
        WSIEvidence(WSIStatus.MISSING, "x.img", sha256=same)


def _manifest() -> CohortManifest:
    roster = [entry("AA_1_01", "g:AA:1"), entry("AA_2_01", "g:AA:2"), entry("AA_3_01", "g:AA:3")]
    cands = [
        candidate("AA_1_01", "g:AA:1"),
        candidate("AA_2_01", "g:AA:2", "pos", "test", row=3),
    ]
    return build_manifest(definition(), observations(roster, cands))


def test_manifest_counts_and_split_rederive() -> None:
    m = _manifest()
    assert [(c.status, c.reason_code, c.partition, c.label, c.slides) for c in m.counts] == [
        (S.ELIGIBLE, "", "test", "pos", 1),
        (S.ELIGIBLE, "", "train", "neg", 1),
        (S.EXCLUDED_MISSING_LABEL, "no_label", "unassigned", "none", 1),
    ]
    parts = {p.partition: (p.slides, p.groups, p.class_counts) for p in m.split.partitions}
    assert parts["train"] == (1, 1, (("neg", 1), ("pos", 0)))
    assert parts["test"] == (1, 1, (("neg", 0), ("pos", 1)))
    assert m.split.strategy == "external@1" and m.split.test_role == "reused_benchmark"
    assert m.digest == _manifest().digest  # deterministic
    with pytest.raises(CohortError, match="counts"):
        replace(m, counts=(StatusCount(S.ELIGIBLE, "", "train", "neg", 2),))
    with pytest.raises(CohortError, match="split"):
        replace(m, split=replace(m.split, assignment_digest="0" * 64))
    flipped = replace(m.records[0], label="pos")
    with pytest.raises(CohortError):
        replace(m, records=(flipped, *m.records[1:]))


def test_manifest_refuses_eligible_records_under_pending_decisions() -> None:
    m = _manifest()
    pending = tuple(
        Decision(d.decision_id, d.summary, DecisionStatus.PENDING, d.gates) for d in m.decisions
    )
    with pytest.raises(CohortError, match="pending"):
        replace(m, decisions=pending)


def test_non_eligible_records_need_reasons() -> None:
    m = _manifest()
    with pytest.raises(CohortError, match="reason"):
        replace(m.records[2], reason_code=None, reason=None)
    with pytest.raises(CohortError, match="open gates"):
        replace(m.records[2], outstanding_gates=WSI_GATES)
