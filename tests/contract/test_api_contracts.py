"""Imaging/Analysis API v1 contract tests, independent of any DPA implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from bootstrap import Platform, build_platform
from domain.geometry import Level0Rect, RectangleGeometry, VoxelBoxGeometry, VoxelIndex
from domain.ids import ArtifactId, EvidenceId, PredictionId, RunId
from domain.prediction import (
    ContinuousScore,
    Detection,
    EvidenceSemantics,
    PredictionResult,
    RegionPrediction,
    SpatialEvidence,
    VolumetricSegmentationReference,
)
from domain.validation import CapabilityStatus, UsageMode
from infrastructure.runtime import DeferredJobExecutor, FixedClock
from plugins.synthetic import catalog
from presentation.api import analysis_v1, imaging_v1
from presentation.api.contracts import load_schema, schema_names

FIXTURES = Path(__file__).parent / "fixtures"


def _validator(api: str, name: str) -> Draft202012Validator:
    return Draft202012Validator(load_schema(api, "v1", name))


def _errors(api: str, name: str, doc: Any) -> list[str]:
    return [e.message for e in _validator(api, name).iter_errors(doc)]


@pytest.mark.parametrize("api", ["imaging", "analysis"])
def test_schemas_are_valid_draft_2020_12(api: str) -> None:
    names = schema_names(api, "v1")
    assert names
    for name in names:
        Draft202012Validator.check_schema(load_schema(api, "v1", name))


def _fixtures(kind: str) -> list[tuple[str, Path]]:
    return [
        (api, p)
        for api in ("imaging", "analysis")
        for p in sorted((FIXTURES / api / kind).glob("*.json"))
    ]


@pytest.mark.parametrize("api,path", _fixtures("valid"), ids=lambda x: getattr(x, "name", x))
def test_valid_fixtures(api: str, path: Path) -> None:
    fx = json.loads(path.read_text())
    assert _errors(api, fx["schema"], fx["document"]) == []


@pytest.mark.parametrize("api,path", _fixtures("invalid"), ids=lambda x: getattr(x, "name", x))
def test_invalid_fixtures_are_rejected(api: str, path: Path) -> None:
    fx = json.loads(path.read_text())
    assert _errors(api, fx["schema"], fx["document"]), fx.get("reason")


def test_serialized_image_descriptors_validate(platform: Platform) -> None:
    lung = platform.project_definitions["patologia/cancer/pulmao"]
    image = platform.datasets.image(platform.datasets.version(lung.dataset).image_ids[0])
    reader = platform.readers.for_image(image)
    assert reader is not None
    doc = imaging_v1.image_descriptor(image, reader.capabilities)
    assert _errors("imaging", "image-descriptor", doc) == []
    assert doc["context"] == {"tissue": None, "disease": None}  # never guessed
    vol = imaging_v1.image_descriptor(catalog.synthetic_volume("syn-ct-0009"))
    assert _errors("imaging", "image-descriptor", vol) == []
    assert vol["access"] == {"regions": False, "thumbnail": False, "volume": False, "slices": False}


@pytest.mark.parametrize(
    "ns", ["patologia/cancer/pulmao", "patologia/cancer/coloretal", "patologia/pele/psoriase"]
)
def test_serialized_results_validate(platform: Platform, ns: str) -> None:
    project = platform.project_definitions[ns]
    img = platform.datasets.version(project.dataset).split("test")[0]
    outcome = platform.projects.run_image(project, img)
    assert outcome.prediction is not None
    doc = analysis_v1.analysis_result(outcome.prediction, outcome.run)
    assert _errors("analysis", "analysis-result", doc) == []
    assert doc["research_use_only"] is True
    assert doc["prediction"]["image_id"] == img.value


def test_all_typed_outputs_and_3d_evidence_serialize(platform: Platform) -> None:
    vol = catalog.synthetic_volume("syn-ct-0002")
    model = catalog.VOLUME_SEGMENTER.ref
    pred = PredictionResult(
        PredictionId("pai-pred-x"),
        vol.image_id,
        catalog.VOLUME_SEG_TASK.ref,
        (model,),
        (
            VolumetricSegmentationReference(
                ArtifactId("pai-art-x"),
                ((0, "background"), (1, "synthetic_structure")),
                vol.spatial_reference,  # type: ignore[arg-type]
            ),
            ContinuousScore("synthetic_volume", 1.0, "ml"),
            RegionPrediction(RectangleGeometry(Level0Rect(0, 0, 1, 1)), "r", None),
            Detection(RectangleGeometry(Level0Rect(0, 0, 1, 1)), "d", 0.5),
        ),
        (
            SpatialEvidence(
                EvidenceId("pai-ev-x"),
                vol.image_id,
                VoxelBoxGeometry(VoxelIndex(0, 0, 0), VoxelIndex(4, 4, 4)),
                EvidenceSemantics.SEGMENTATION_MEMBERSHIP,
                model,
            ),
        ),
        RunId("pai-run-x"),
        CapabilityStatus.SUPPORTED_UNVALIDATED,
        UsageMode.RESEARCH_UNVALIDATED,
    )
    lung = platform.project_definitions["patologia/cancer/pulmao"]
    run = platform.projects.run_image(
        lung, platform.datasets.version(lung.dataset).split("test")[0]
    ).run
    doc = analysis_v1.analysis_result(pred, run)
    assert _errors("analysis", "analysis-result", doc) == []
    kinds = [o["kind"] for o in doc["prediction"]["outputs"]]
    assert kinds == [
        "volumetric_segmentation",
        "continuous_score",
        "region_prediction",
        "detection",
    ]


def test_job_states_and_capabilities_validate() -> None:
    executor = DeferredJobExecutor()
    p = build_platform(clock=FixedClock(), executor=executor)
    lung = p.project_definitions["patologia/cancer/pulmao"]
    img = p.datasets.version(lung.dataset).split("test")[0]
    job = p.analysis.submit(lung, img)
    assert _errors("analysis", "analysis-job", analysis_v1.analysis_job(job)) == []
    executor.drain()
    done = analysis_v1.analysis_job(p.analysis.job(job.job_id))
    assert done["status"] == "COMPLETED"
    assert _errors("analysis", "analysis-job", done) == []
    caps = analysis_v1.capability_list(
        [p.projects.capability(x) for x in p.project_definitions.values()]
    )
    assert _errors("analysis", "capability-list", caps) == []
    assert {c["status"] for c in caps["capabilities"]} == {"SUPPORTED_UNVALIDATED"}
    assert all(c["requires_research_acknowledgement"] for c in caps["capabilities"])
