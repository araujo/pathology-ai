"""Helpers for SPEC-008 cohort tests: a SYNTHETIC two-sub-cohort world of CSV sources.

Identifiers imitate the shape of real study codes (sub-cohort, case number, slide suffix) but
are invented: no real label is used. The world reproduces the hazards found in Phase A:
case numbers shared between sub-cohorts, unpadded case numbers, a positional `slide_<i>` column
that repeats across files, second slides without labels, and a redundant combined copy.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from domain.cohort import (
    CohortDefinition,
    CohortObservations,
    CohortTarget,
    Decision,
    DecisionStatus,
    LabelCandidate,
    RosterEntry,
    SlidePolicy,
    SourceFile,
    SourceRole,
    SplitPolicy,
)

APPROVAL = {
    "status": "approved",
    "approver": "test owner",
    "approved_on": "2026-11-02",
    "reference": "synthetic test",
}
PATTERN = r"^(?P<cohort>AA|BB)_X_T(?P<case>\d+)(?:_(?P<slide>\d{2}))?(?:\.img)?$"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def csv_bytes(header: str, rows: list[str]) -> bytes:
    return ("\n".join([header, *rows]) + "\n").encode("utf-8")


@dataclass
class World:
    """Synthetic sources on disk plus the configuration document that declares them."""

    root: Path
    files: dict[str, bytes] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)

    def write(self) -> Path:
        for name, data in self.files.items():
            (self.root / name).write_bytes(data)
        path = self.root / "cohort.yaml"
        path.write_text(yaml.safe_dump(self.config, sort_keys=False), encoding="utf-8")
        return path

    def source(self, source_id: str) -> dict[str, Any]:
        for s in self.config["sources"]:
            if s["id"] == source_id:
                return dict(s)
        raise KeyError(source_id)

    def set_file(self, name: str, data: bytes) -> None:
        """Replace a file and re-pin it (a legitimate new version of the source)."""
        self.files[name] = data
        for s in self.config["sources"]:
            for f in s["files"]:
                if f["path"] == name:
                    f["sha256"] = sha(data)


def world(root: Path) -> World:
    """AA: cases 1-4, one slide each. BB: cases 1-4 (same numbers as AA), case 2 also has a
    second slide, case 4 has no label anywhere."""
    files = {
        "roster_aa.csv": csv_bytes(
            "slide_id,status", [f"AA_X_T{c:03d}_01.img,processed" for c in (1, 2, 3, 4)]
        ),
        "roster_bb.csv": csv_bytes(
            "slide_id,status",
            [
                "BB_X_T001_01.img,processed",
                "BB_X_T002_01.img,processed",
                "BB_X_T002_02.img,processed",
                "BB_X_T003_01.img,processed",
                "BB_X_T004_01.img,processed",
            ],
        ),
        # AA labels: unpadded case ids, slide_id = positional index restarting per file.
        "aa_train.csv": csv_bytes(
            "case_id,slide_id,label", ["AA_X_T1,slide_0,0", "AA_X_T2,slide_1,1"]
        ),
        "aa_test.csv": csv_bytes(
            "case_id,slide_id,label", ["AA_X_T3,slide_0,0", "AA_X_T4,slide_1,1"]
        ),
        # BB labels: padded slide ids with suffix.
        "bb_train.csv": csv_bytes(
            "case_id,slide_id,label", ["BB_X_T001_01,slide_0,1", "BB_X_T002_01,slide_1,0"]
        ),
        "bb_test.csv": csv_bytes("case_id,slide_id,label", ["BB_X_T003_01,slide_0,0"]),
        # A combined copy with one partition column.
        "combined.csv": csv_bytes(
            "split,case_id,label,file",
            [
                "train,AA_X_T1,0,AA_X_T001_01.img",
                "train,AA_X_T2,1,AA_X_T002_01.img",
                "test,AA_X_T3,0,AA_X_T003_01.img",
                "test,AA_X_T4,1,AA_X_T004_01.img",
                "train,BB_X_T001_01,1,BB_X_T001_01.img",
                "train,BB_X_T002_01,0,BB_X_T002_01.img",
                "test,BB_X_T003_01,0,BB_X_T003_01.img",
            ],
        ),
    }

    def label_source(sid: str, role: str, target: str, parts: dict[str, str]) -> dict[str, Any]:
        return {
            "id": sid,
            "role": role,
            "root": "data",
            "provenance": "synthetic",
            "source_target": target,
            "id_column": "case_id",
            "label_column": "label",
            "vocabulary": {"0": "neg", "1": "pos"},
            "files": [
                {"path": p, "sha256": sha(files[p]), "partition": part} for p, part in parts.items()
            ],
        }

    decisions = [
        {"id": d, "summary": f"decision {d}", "gates": ["reconciliation"], **APPROVAL}
        for d in ("D-1", "D-2", "D-3", "D-4", "D-5")
    ] + [{"id": "D-9", "summary": "later", "status": "pending", "gates": ["extraction"]}]
    config = {
        "schema": "pathology-ai.cohort-config/1",
        "cohort_id": "synthetic-cohort",
        "version": "1",
        "target": {
            "name": "synthetic_target",
            "classes": ["neg", "pos"],
            "positive_class": "pos",
            "accepted_source_targets": ["TARGET_A", "TARGET_B"],
            "decision": "D-3",
        },
        "identity": {
            "pattern": PATTERN,
            "integer_fields": ["case"],
            "cohort_field": "cohort",
            "slide_field": "slide",
            "group_fields": ["cohort", "case"],
            "slide_id": "{cohort}_X_T{case:03d}_{slide}",
            "group_id": "g:{cohort}:T{case:03d}",
            "decision": "D-4",
        },
        "label_authority": {"decision": "D-1"},
        "slides": {"include_indices": ["01"], "decision": "D-2"},
        "split": {
            "strategy": "external@1",
            "test_role": "reused_benchmark",
            "pilot_partitions": ["train", "validation"],
            "decisions": ["D-5"],
        },
        "partition_names": {"train": "train", "val": "validation", "test": "test"},
        "sources": [
            {
                "id": "roster-aa",
                "role": "roster",
                "root": "data",
                "provenance": "synthetic",
                "id_column": "slide_id",
                "files": [{"path": "roster_aa.csv", "sha256": sha(files["roster_aa.csv"])}],
            },
            {
                "id": "roster-bb",
                "role": "roster",
                "root": "data",
                "provenance": "synthetic",
                "id_column": "slide_id",
                "files": [{"path": "roster_bb.csv", "sha256": sha(files["roster_bb.csv"])}],
            },
            label_source(
                "labels-aa",
                "authority",
                "TARGET_A",
                {"aa_train.csv": "train", "aa_test.csv": "test"},
            ),
            label_source(
                "labels-bb",
                "authority",
                "TARGET_B",
                {"bb_train.csv": "train", "bb_test.csv": "test"},
            ),
            {
                "id": "combined",
                "role": "corroborating",
                "root": "data",
                "provenance": "synthetic",
                "source_target": "TARGET_B",
                "id_column": "case_id",
                "label_column": "label",
                "partition_column": "split",
                "vocabulary": {"0": "neg", "1": "pos"},
                "files": [{"path": "combined.csv", "sha256": sha(files["combined.csv"])}],
            },
        ],
        "decisions": decisions,
    }
    return World(root, files, config)


# --------------------------------------------------------------------------- pure domain inputs

DIGEST = "0" * 64


def definition(**overrides: Any) -> CohortDefinition:
    decisions = tuple(
        Decision(
            d, f"decision {d}", DecisionStatus.APPROVED, ("reconciliation",), "o", "2026-11-02", "r"
        )
        for d in ("D-1", "D-2", "D-3", "D-4", "D-5")
    )
    base: dict[str, Any] = {
        "cohort_id": "synthetic-cohort",
        "version": "1",
        "target": CohortTarget("t", ("neg", "pos"), "pos", ("TA",), "D-3"),
        "label_authority_decision": "D-1",
        "group_decision": "D-4",
        "slide_policy": SlidePolicy(("01",), "D-2"),
        "split_policy": SplitPolicy("external@1", "reused_benchmark", ("train",), ("D-5",)),
        "decisions": decisions,
    }
    base.update(overrides)
    return CohortDefinition(**base)


def entry(slide: str, group: str, cohort: str = "AA", index: str = "01") -> RosterEntry:
    return RosterEntry(slide, group, cohort, index, f"{slide}.img", "roster", 2)


def candidate(
    slide: str | None,
    group: str,
    value: str = "neg",
    partition: str | None = "train",
    *,
    source: str = "labels",
    role: SourceRole = SourceRole.AUTHORITY,
    row: int = 2,
    target: str = "TA",
) -> LabelCandidate:
    return LabelCandidate(
        source, role, "labels.csv", row, slide or group, "0", value, target, group, slide, partition
    )


SOURCES = (
    SourceFile("labels", SourceRole.AUTHORITY, "labels.csv", DIGEST, 3, "TA", None, "synthetic"),
    SourceFile("other", SourceRole.CORROBORATING, "labels.csv", DIGEST, 3, "TA", None, "synthetic"),
    SourceFile("roster", SourceRole.ROSTER, "roster.csv", DIGEST, 3, None, None, "synthetic"),
)


def observations(roster: list[RosterEntry], candidates: list[LabelCandidate]) -> CohortObservations:
    return CohortObservations(SOURCES, tuple(roster), tuple(candidates))


# --------------------------------------------------------------------------- inventory and pilot


def pilot_definition(pending_pilot: bool = True) -> CohortDefinition:
    """Two sub-cohorts; D-6 approves the pilot partitions, D-10 (pending) gates the pilot."""
    approved = tuple(
        Decision(d, f"decision {d}", DecisionStatus.APPROVED, gates, "o", "2026-11-02", "r")
        for d, gates in (
            ("D-1", ("reconciliation",)),
            ("D-2", ("reconciliation",)),
            ("D-3", ("reconciliation",)),
            ("D-4", ("reconciliation",)),
            ("D-5", ("reconciliation",)),
            ("D-6", ("reconciliation", "pilot")),
        )
    )
    pending = (
        (Decision("D-10", "terms", DecisionStatus.PENDING, ("pilot",)),) if pending_pilot else ()
    )
    return definition(
        decisions=approved + pending,
        split_policy=SplitPolicy(
            "external@1", "reused_benchmark", ("train", "validation"), ("D-5", "D-6")
        ),
    )


# (case, label, partition) per sub-cohort; positives of BB live only where `bb_positive` says.
def pilot_cases(bb_positive: str) -> list[tuple[str, int, str, str]]:
    out = []
    for cohort in ("AA", "BB"):
        for case, label, part in (
            (1, "neg", "train"),
            (2, "pos", "train"),
            (3, "neg", "validation"),
            (4, "neg", "train"),
            (5, "pos", "validation"),
            (6, "neg", "test"),
            (7, "pos", "test"),
            (8, "neg", "validation"),
        ):
            if cohort == "BB" and label == "pos":
                part = bb_positive
            out.append((cohort, case, label, part))
    return out


def pilot_observations(bb_positive: str = "train") -> CohortObservations:
    roster, cands = [], []
    for cohort, case, label, part in pilot_cases(bb_positive):
        slide, group = f"{cohort}_X_T{case:03d}_01", f"g:{cohort}:T{case:03d}"
        roster.append(entry(slide, group, cohort))
        cands.append(candidate(slide, group, label, part))
    # A second slide of one case: excluded by the slide policy, still part of the inventory.
    roster.append(entry("AA_X_T001_02", "g:AA:T001", "AA", "02"))
    return observations(roster, cands)
