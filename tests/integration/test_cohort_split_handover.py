"""SPEC-008 → SPEC-007 hand-over: a cohort manifest's labels and imported split drive a real
MIL dataset and `external@1` split over finalized (SYNTHETIC) embedding bags."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from application.cohort_service import import_external_split, slide_labels, split_assignment
from application.mil_dataset import MILDatasetService
from domain.cohort import (
    CohortManifest,
    CohortObservations,
    CohortTarget,
    LabelCandidate,
    RosterEntry,
    SourceFile,
    SourceRole,
    WSIEvidence,
    WSIStatus,
    build_manifest,
)
from domain.errors import CohortError, SplitError
from domain.ids import DatasetId, Version
from domain.mil import EXACT, EXTERNAL_STRATEGY, PARTITIONS, validate_split
from tests.cohort_support import DIGEST, definition
from tests.mil_support import BINARY, Cohort, cohort


def _manifest(c: Cohort, *, verified: bool) -> CohortManifest:
    groups = sorted({s.group_id for s in c.slides})
    partition_of = {g: PARTITIONS[i % 3].value for i, g in enumerate(groups)}
    roster, cands, wsi = [], [], {}
    for n, s in enumerate(c.slides, start=2):
        roster.append(RosterEntry(s.slide_id, s.group_id, "synthetic", "01", s.slide_id, "r", n))
        cands.append(
            LabelCandidate(
                "labels",
                SourceRole.AUTHORITY,
                "labels.csv",
                n,
                s.slide_id,
                s.label,
                s.label,
                "TA",
                s.group_id,
                s.slide_id,
                partition_of[s.group_id],
            )
        )
        if verified:
            wsi[s.slide_id] = WSIEvidence(WSIStatus.VERIFIED, s.slide_id, 1, f"{n:064x}")
    sources = (
        SourceFile("labels", SourceRole.AUTHORITY, "labels.csv", DIGEST, n - 1, "TA", None, "s"),
        SourceFile("r", SourceRole.ROSTER, "roster.csv", DIGEST, n - 1, None, None, "s"),
    )
    d = definition(target=CohortTarget("t", BINARY.classes, BINARY.positive_class, ("TA",), "D-3"))
    return build_manifest(d, CohortObservations(sources, tuple(roster), tuple(cands)), wsi)


def test_cohort_split_is_imported_through_the_spec007_contract(tmp_path: Path) -> None:
    c = cohort(tmp_path)
    m = _manifest(c, verified=True)
    labels = slide_labels(m, BINARY)
    assert all(lbl.source == "cohort:synthetic-cohort@1" for lbl in labels)
    dataset = c.dataset(list(labels))
    split = import_external_split(m, dataset)
    assert split.strategy == EXTERNAL_STRATEGY and split.seed is None
    validate_split(split, dataset)
    imported = {s: p.partition.value for p in split.partitions for s in p.slide_ids}
    assert imported == {s: p.value for s, p in split_assignment(m).items()}
    for part, summary in zip(m.split.partitions, split.partitions, strict=True):
        assert part.partition == summary.partition.value
        assert part.groups == summary.group_count
        assert part.class_counts == summary.class_counts


def test_provisional_records_need_explicit_acceptance(tmp_path: Path) -> None:
    c = cohort(tmp_path)
    m = _manifest(c, verified=False)
    with pytest.raises(CohortError, match="provisional"):
        slide_labels(m, BINARY)
    labels = slide_labels(m, BINARY, allow_provisional=True)
    split = import_external_split(m, c.dataset(list(labels)), allow_provisional=True)
    assert sum(len(p.slide_ids) for p in split.partitions) == len(m.eligible())


def test_split_is_never_narrowed_to_a_partial_dataset(tmp_path: Path) -> None:
    c = cohort(tmp_path)
    m = _manifest(c, verified=True)
    labels = list(slide_labels(m, BINARY))
    dropped = labels[0].slide_id
    partial = MILDatasetService(c.source()).build(
        dataset_id=DatasetId("partial"),
        version=Version("1"),
        task=BINARY,
        policy=EXACT,
        labels=labels[1:],
        stores={k: v for k, v in c.stores.items() if k != dropped},
    )
    with pytest.raises(SplitError, match=f"missing \\({dropped}"):
        import_external_split(m, partial)
    swapped = [replace(labels[0], value=BINARY.classes[1 - BINARY.index(labels[0].value or "")])]
    with pytest.raises(SplitError, match="label or group"):
        import_external_split(m, c.dataset(swapped + labels[1:]))


def test_task_must_match_the_cohort_target(tmp_path: Path) -> None:
    c = cohort(tmp_path)
    m = _manifest(c, verified=True)
    other = replace(BINARY, classes=("x", "y"), positive_class="y")
    with pytest.raises(CohortError, match="differ"):
        slide_labels(m, other)
