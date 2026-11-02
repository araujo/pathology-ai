"""SPEC-008 Phase B regression: the SurGen MSI cohort manifest v1.

The written manifest is re-read and re-derived (no external data needed). Rebuilding it from the
pinned sources is opt-in: bind both roots and the rebuilt manifest must equal the written one.

    PATHOLOGY_AI_SURGEN_UPSTREAM=~/projects/SurGen-Dataset \
    PATHOLOGY_AI_SURGEN_LEGACY=../digital-pathology uv run pytest -rs
"""

from __future__ import annotations

import dataclasses
import os
from collections import Counter
from pathlib import Path

import pytest

from application.cohort_service import CohortService, split_assignment
from domain.cohort import CohortDefinition, Decision, DecisionStatus
from infrastructure.cohort_sources import (
    MANIFEST_FILE,
    SPLIT_FILE,
    CsvCohortSourceReader,
    load_cohort_config,
    read_manifest,
    split_table,
)

ROOT = Path(__file__).resolve().parents[2]
COHORT = ROOT / "projects" / "cohorts" / "surgen-msi"
V1 = COHORT / "manifests" / "v1"
DIGEST_V1 = "854f009fdb30e41304b9959a53bc3fd2e0beede091bd4f47061af931383a5a3c"
V1_D7_SUMMARY = "Extraction scale (0.5 um/px +-15% or 1.0 um/px)."
V1_D9_SUMMARY = "Compute, storage and WSI download route for Phases C/D."
# The decisions approved after v1 was built, with the summaries v1 recorded while pending.
V1_PENDING = {"D-7": V1_D7_SUMMARY, "D-9": V1_D9_SUMMARY}
SPLIT_V1 = "0c8c94fc9d73ff7ff87f55c57ef9ce8a8a1fc6dd07e64534daf8f7514d065616"

needs_manifest = pytest.mark.skipif(
    not (V1 / MANIFEST_FILE).exists(), reason="SurGen manifest v1 is not present"
)


@needs_manifest
def test_manifest_v1_rederives_with_the_approved_counts() -> None:
    m = read_manifest(V1 / MANIFEST_FILE)
    assert m.digest == DIGEST_V1 and m.split.assignment_digest == SPLIT_V1
    assert len(m.records) == 1020 and len({r.group_id for r in m.records}) == 843
    status = Counter((r.status.value, r.reason_code) for r in m.records)
    assert status == {
        ("eligible", None): 820,
        ("excluded_missing_label", "no_label"): 23,
        ("excluded_missing_label", "slide_policy"): 177,
    }
    assert all(r.provisional for r in m.eligible())
    parts = {p.partition: (p.slides, p.groups, dict(p.class_counts)) for p in m.split.partitions}
    assert parts == {
        "train": (490, 490, {"negative": 440, "positive": 50}),
        "validation": (165, 165, {"negative": 151, "positive": 14}),
        "test": (165, 165, {"negative": 150, "positive": 15}),
    }
    assert m.split.test_role == "reused_benchmark"
    assert {r.slide_index for r in m.eligible()} == {"01"}
    by_cohort = Counter((r.cohort, r.source_targets) for r in m.eligible())
    assert by_cohort == {("SR386", ("MMR_LOSS",)): 423, ("SR1482", ("MMR_MSI",)): 397}
    # Every eligible slide is corroborated by all three label sources, without conflict.
    assert all(len(r.candidates) == 3 for r in m.eligible())
    assert (V1 / SPLIT_FILE).read_bytes() == split_table(m)
    assert len(split_assignment(m, allow_provisional=True)) == 820


@needs_manifest
@pytest.mark.skipif(
    not (
        os.environ.get("PATHOLOGY_AI_SURGEN_UPSTREAM")
        and os.environ.get("PATHOLOGY_AI_SURGEN_LEGACY")
    ),
    reason="set PATHOLOGY_AI_SURGEN_UPSTREAM and PATHOLOGY_AI_SURGEN_LEGACY to rebuild",
)
def test_manifest_v1_rebuilds_from_the_pinned_sources() -> None:
    cfg = load_cohort_config(COHORT / "cohort.yaml")
    roots = {
        "upstream": Path(os.environ["PATHOLOGY_AI_SURGEN_UPSTREAM"]).expanduser(),
        "legacy": Path(os.environ["PATHOLOGY_AI_SURGEN_LEGACY"]).expanduser(),
    }
    service = CohortService(CsvCohortSourceReader(cfg, roots))
    rebuilt = service.reconcile(_as_of_v1(cfg.definition))
    assert rebuilt.digest == DIGEST_V1
    assert rebuilt == read_manifest(V1 / MANIFEST_FILE)
    # The current definition (D-7, D-9 approved) changes the decision ledger only, not the cohort.
    current = service.reconcile(cfg.definition)
    assert current.version == "2" and current.digest != DIGEST_V1
    assert (current.records, current.split, current.counts, current.sources) == (
        rebuilt.records,
        rebuilt.split,
        rebuilt.counts,
        rebuilt.sources,
    )


def _as_of_v1(definition: CohortDefinition) -> CohortDefinition:
    """The definition manifest v1 was built from: version "1", D-7 and D-9 still pending."""
    decisions = tuple(
        Decision(d.decision_id, V1_PENDING[d.decision_id], DecisionStatus.PENDING, d.gates)
        if d.decision_id in V1_PENDING
        else d
        for d in definition.decisions
    )
    return dataclasses.replace(definition, version="1", decisions=decisions)


@needs_manifest
def test_configuration_differs_from_v1_only_by_the_recorded_approvals() -> None:
    """Manifest v1 is immutable; the configuration has moved on by two recorded decisions."""
    cfg = load_cohort_config(COHORT / "cohort.yaml")
    m = read_manifest(V1 / MANIFEST_FILE)
    assert m.version == "1" and cfg.definition.version == "2"
    assert _as_of_v1(cfg.definition) == m.definition
    d7 = next(d for d in cfg.definition.decisions if d.decision_id == "D-7")
    assert d7.status is DecisionStatus.APPROVED and d7.gates == ("extraction",)
    assert d7.approved_on == "2026-11-02" and "0.5 um/px" in d7.summary
    assert "uni-official@1" in d7.summary
    pending = {
        d.decision_id for d in cfg.definition.decisions if d.status is DecisionStatus.PENDING
    }
    assert pending == {"D-10"}
    d9 = next(d for d in cfg.definition.decisions if d.decision_id == "D-9")
    assert d9.status is DecisionStatus.APPROVED and d9.gates == ("extraction",)
    assert d9.approved_on == "2026-11-02" and "PROMPT-008D9" in str(d9.reference)
    for term in ("EBI HTTPS", "MPS", "uni-official@1", "0.5 um/px", "fp32", "batch size 16"):
        assert term in d9.summary
    # D-9 is planning-only: its own record must say that it authorizes no execution.
    assert "Planning and infrastructure preparation only" in d9.summary
    assert "no WSI download, UNI execution, MIL training or Phase C2 execution" in d9.summary
    for decision_id in V1_PENDING:  # the immutable record keeps its history
        v1 = next(d for d in m.decisions if d.decision_id == decision_id)
        assert v1.status is DecisionStatus.PENDING and v1.summary == V1_PENDING[decision_id]
