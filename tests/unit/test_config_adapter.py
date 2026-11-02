from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from domain.pipeline import PipelineDefinition
from domain.pipeline_validation import require_valid_structure
from infrastructure.config import (
    ConfigurationError,
    load_pipeline,
    load_project,
    pipeline_from_dict,
    pipeline_to_dict,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINES = sorted((REPO_ROOT / "pipelines").glob("*.yaml"))
PROJECTS = sorted((REPO_ROOT / "projects").rglob("project.yaml"))


def test_yaml_is_only_an_adapter_for_programmatic_pipelines(
    wsi_pipeline: PipelineDefinition,
) -> None:
    loaded = load_pipeline(REPO_ROOT / "pipelines" / "wsi-patch-embed-aggregate.v1.yaml")
    assert loaded == wsi_pipeline


@pytest.mark.parametrize("path", PIPELINES, ids=lambda p: p.name)
def test_pipeline_documents_round_trip_and_validate(path: object) -> None:
    p = load_pipeline(path)  # type: ignore[arg-type]
    require_valid_structure(p)
    assert pipeline_from_dict(yaml.safe_load(yaml.safe_dump(pipeline_to_dict(p)))) == p


def test_unknown_keys_rejected() -> None:
    doc = pipeline_to_dict(load_pipeline(PIPELINES[0]))
    doc["engine"] = "yaml"
    with pytest.raises(ConfigurationError):
        pipeline_from_dict(doc)


def test_three_projects_with_stable_ids_independent_of_paths() -> None:
    projects = [load_project(p) for p in PROJECTS]
    assert sorted(p.namespace for p in projects) == [
        "patologia/cancer/coloretal",
        "patologia/cancer/pulmao",
        "patologia/pele/psoriase",
    ]
    for p in projects:
        assert "/" not in p.project_id.value
        assert p.dataset.dataset_id.value.startswith("synthetic-")
        for _, ref in p.model_bindings:
            assert ref.model_id.value.startswith("synthetic-")


def test_cancer_projects_share_one_pipeline_definition() -> None:
    by_ns = {p.namespace: p for p in (load_project(x) for x in PROJECTS)}
    lung, crc = by_ns["patologia/cancer/pulmao"], by_ns["patologia/cancer/coloretal"]
    assert lung.pipeline == crc.pipeline
    assert lung.dataset != crc.dataset
    assert lung.bindings()["aggregator"] != crc.bindings()["aggregator"]
    assert by_ns["patologia/pele/psoriase"].pipeline != lung.pipeline
