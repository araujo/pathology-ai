"""SPEC-008 declarative cohort sources: digest pins, identifier grammar and observed hazards.

Every source here is SYNTHETIC (see `tests/cohort_support.py`); no real label is used.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from application.cohort_service import CohortService
from domain.cohort import EligibilityStatus, SourceRole
from domain.errors import (
    CohortError,
    CohortReconciliationError,
    CohortSourceFormatError,
    CohortSourceIntegrityError,
    ExperimentConflictError,
)
from infrastructure.cohort_sources import (
    MANIFEST_FILE,
    RECONCILIATION_FILE,
    SPLIT_FILE,
    CsvCohortSourceReader,
    load_cohort_config,
    read_manifest,
    write_cohort_outputs,
)
from infrastructure.config import ConfigurationError
from infrastructure.mil_config import load_split_table
from tests.cohort_support import World, csv_bytes, world

S = EligibilityStatus


def _reconcile(w: World):  # type: ignore[no-untyped-def]
    cfg = load_cohort_config(w.write())
    return CohortService(CsvCohortSourceReader(cfg, {"data": w.root})).reconcile(cfg.definition)


def test_synthetic_world_reconciles_with_every_hazard_handled(tmp_path: Path) -> None:
    m = _reconcile(world(tmp_path))
    by = {r.slide_id: r for r in m.records}
    assert len(m.records) == 9 and len({r.group_id for r in m.records}) == 8
    # Unpadded "AA_X_T1" normalises to the padded slide and group.
    aa1 = by["AA_X_T001_01"]
    assert aa1.group_id == "g:AA:T001" and aa1.status is S.ELIGIBLE
    assert [c.raw_id for c in aa1.candidates] == ["AA_X_T1", "AA_X_T1"]
    # Case 1 exists in both sub-cohorts with different labels: kept apart.
    assert (aa1.label, by["BB_X_T001_01"].label) == ("neg", "pos")
    assert by["BB_X_T001_01"].group_id == "g:BB:T001"
    # Second slide excluded by policy; unlabelled case excluded as missing label.
    assert by["BB_X_T002_02"].status is S.EXCLUDED_MISSING_LABEL
    assert by["BB_X_T002_02"].reason_code == "slide_policy"
    assert by["BB_X_T004_01"].reason_code == "no_label"
    # The authority's own target is kept per record (combined copy names another target).
    assert aa1.source_targets == ("TARGET_A",)
    assert {r.slide_id for r in m.eligible()} == {
        "AA_X_T001_01",
        "AA_X_T002_01",
        "AA_X_T003_01",
        "AA_X_T004_01",
        "BB_X_T001_01",
        "BB_X_T002_01",
        "BB_X_T003_01",
    }
    assert all(r.provisional for r in m.eligible())
    roles = {(s.source_id, s.role) for s in m.sources}
    assert ("combined", SourceRole.CORROBORATING) in roles and len(m.sources) == 7


def test_positional_slide_id_column_is_never_an_identifier(tmp_path: Path) -> None:
    w = world(tmp_path)
    w.config["sources"][2]["id_column"] = "slide_id"
    with pytest.raises(CohortSourceFormatError, match="slide_0"):
        _reconcile(w)


def test_unpadded_and_padded_spellings_of_one_case_are_duplicates(tmp_path: Path) -> None:
    w = world(tmp_path)
    w.set_file(
        "aa_train.csv",
        csv_bytes("case_id,slide_id,label", ["AA_X_T1,slide_0,0", "AA_X_T001,slide_1,0"]),
    )
    with pytest.raises(CohortSourceFormatError, match="repeats"):
        _reconcile(w)


def test_edited_source_fails_its_pin_before_parsing(tmp_path: Path) -> None:
    w = world(tmp_path)
    w.files["bb_test.csv"] = csv_bytes("case_id,slide_id,label", ["BB_X_T003_01,slide_0,1"])
    with pytest.raises(CohortSourceIntegrityError, match="pins"):
        _reconcile(w)


def test_cross_target_file_in_place_of_a_label_file_fails_its_pin(tmp_path: Path) -> None:
    """A file of another target (same columns, different values) cannot pass for a label file."""
    w = world(tmp_path)
    other_target = csv_bytes("case_id,slide_id,label", ["BB_X_T003_01,slide_0,1"])
    w.write()
    (tmp_path / "bb_test.csv").write_bytes(other_target)
    cfg = load_cohort_config(tmp_path / "cohort.yaml")
    with pytest.raises(CohortSourceIntegrityError):
        CsvCohortSourceReader(cfg, {"data": tmp_path}).read()


def test_source_declaring_an_unaccepted_target_is_rejected(tmp_path: Path) -> None:
    w = world(tmp_path)
    w.config["sources"][3]["source_target"] = "OTHER_TARGET"
    with pytest.raises(CohortReconciliationError, match="not accepted"):
        _reconcile(w)


def test_conflicting_copy_excludes_the_slide(tmp_path: Path) -> None:
    w = world(tmp_path)
    data = w.files["combined.csv"].replace(b"test,AA_X_T3,0", b"test,AA_X_T3,1")
    w.set_file("combined.csv", data)
    m = _reconcile(w)
    r = m.record("AA_X_T003_01")
    assert r.status is S.EXCLUDED_CONFLICT and r.label is None
    assert len(m.eligible()) == 6


def test_case_label_for_a_multi_slide_case_is_never_inherited(tmp_path: Path) -> None:
    w = world(tmp_path)
    w.set_file(
        "bb_test.csv",
        csv_bytes("case_id,slide_id,label", ["BB_X_T003_01,slide_0,0", "BB_X_T2,x,0"]),
    )
    with pytest.raises(CohortReconciliationError, match="exactly"):
        _reconcile(w)


def test_value_outside_the_vocabulary_names_file_and_line(tmp_path: Path) -> None:
    w = world(tmp_path)
    w.set_file("bb_test.csv", csv_bytes("case_id,slide_id,label", ["BB_X_T003_01,slide_0,FAIL"]))
    with pytest.raises(CohortSourceFormatError, match=r"bb_test.csv\) line 2.*'FAIL'"):
        _reconcile(w)


@pytest.mark.parametrize(
    "path,value,match",
    [
        (("identity", "group_id"), "g:T{case:03d}", "group fields"),
        (("identity", "group_id"), "g:{cohort}:{case}:{slide}", "group fields"),
        (("identity", "slide_id"), "{cohort}_T{case:03d}", "slide field"),
        (("identity", "group_fields"), ["case"], "cohort field"),
        (("split", "strategy"), "group-stratified@1", "never generated"),
        (("target", "colour"), "red", "unknown"),
    ],
)
def test_configuration_rules(
    tmp_path: Path, path: tuple[str, str], value: object, match: str
) -> None:
    w = world(tmp_path)
    w.config[path[0]][path[1]] = value
    with pytest.raises((ConfigurationError, CohortError), match=match):
        load_cohort_config(w.write())


def test_unbound_root_is_refused(tmp_path: Path) -> None:
    cfg = load_cohort_config(world(tmp_path).write())
    with pytest.raises(ConfigurationError, match="not bound"):
        CsvCohortSourceReader(cfg, {})


def test_outputs_are_write_once_and_round_trip(tmp_path: Path) -> None:
    (tmp_path / "w").mkdir()
    m = _reconcile(world(tmp_path / "w"))
    out = tmp_path / "out"
    written = write_cohort_outputs(out, m)
    assert set(written) == {MANIFEST_FILE, SPLIT_FILE, RECONCILIATION_FILE}
    assert write_cohort_outputs(out, m) == written  # identical rewrite is accepted
    back = read_manifest(out / MANIFEST_FILE)
    assert back == m and back.digest == m.digest
    table = load_split_table(out / SPLIT_FILE)  # SPEC-007 split table format
    assert {s: p.value for s, p in table.items()} == {r.slide_id: r.partition for r in m.eligible()}
    doc = json.loads((out / MANIFEST_FILE).read_text())
    doc["body"]["records"][0]["label"] = "pos"
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(doc))
    with pytest.raises(CohortError, match="digest"):
        read_manifest(tampered)
    (out / SPLIT_FILE).write_text("slide_id,partition\n")
    with pytest.raises(ExperimentConflictError):
        write_cohort_outputs(out, m)
