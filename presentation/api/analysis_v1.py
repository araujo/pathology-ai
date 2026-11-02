"""Analysis API v1 serializers (domain -> transport dict). No server, no DPA wire format."""

from __future__ import annotations

from typing import Any

from application.project_service import ProjectCapability
from domain.geometry import (
    EvidenceGeometry,
    PointGeometry,
    PolygonGeometry,
    RectangleGeometry,
    VoxelBoxGeometry,
)
from domain.job import AnalysisJob
from domain.prediction import (
    ContinuousScore,
    Detection,
    ImageClassification,
    PredictionOutput,
    PredictionResult,
    RegionPrediction,
    SegmentationReference,
    SpatialEvidence,
    VolumetricSegmentationReference,
)
from domain.provenance import PipelineRun
from domain.validation import CapabilityStatus, UsageMode

SCHEMA_VERSION = "pathology-ai.analysis/1.0"


def _ref(identifier: object, version: object) -> dict[str, str]:
    return {"id": str(identifier), "version": str(version)}


def geometry_to_dict(g: EvidenceGeometry) -> dict[str, Any]:
    if isinstance(g, RectangleGeometry):
        r = g.rect
        return {
            "type": "rectangle",
            "frame": g.frame.value,
            "x": r.x,
            "y": r.y,
            "width": r.width,
            "height": r.height,
        }
    if isinstance(g, PolygonGeometry):
        return {"type": "polygon", "frame": g.frame.value, "points": [list(p) for p in g.points]}
    if isinstance(g, PointGeometry):
        return {"type": "point", "frame": g.frame.value, "x": g.x, "y": g.y}
    assert isinstance(g, VoxelBoxGeometry)
    return {
        "type": "voxel_box",
        "frame": g.frame.value,
        "start": [g.start.i, g.start.j, g.start.k],
        "stop": [g.stop.i, g.stop.j, g.stop.k],
    }


def output_to_dict(o: PredictionOutput) -> dict[str, Any]:
    if isinstance(o, ImageClassification):
        return {
            "kind": o.kind.value,
            "label": o.label,
            "scores": dict(o.scores),
            "scores_are_calibrated_probabilities": o.scores_are_calibrated_probabilities,
        }
    if isinstance(o, ContinuousScore):
        return {"kind": o.kind.value, "name": o.name, "value": o.value, "unit": o.unit}
    if isinstance(o, RegionPrediction):
        return {
            "kind": o.kind.value,
            "geometry": geometry_to_dict(o.geometry),
            "label": o.label,
            "score": o.score,
        }
    if isinstance(o, Detection):
        return {
            "kind": o.kind.value,
            "geometry": geometry_to_dict(o.geometry),
            "label": o.label,
            "confidence": o.confidence,
        }
    label_map = [{"value": v, "label": lbl} for v, lbl in o.label_map]
    if isinstance(o, SegmentationReference):
        return {
            "kind": o.kind.value,
            "mask_artifact": o.mask_artifact.value,
            "label_map": label_map,
            "frame": "wsi_level_pixel",
            "level": o.level,
            "downsample": o.downsample,
        }
    assert isinstance(o, VolumetricSegmentationReference)
    return {
        "kind": o.kind.value,
        "mask_artifact": o.mask_artifact.value,
        "label_map": label_map,
        "frame": "volume_voxel_index",
        "shape": list(o.geometry.shape),
        "spacing_mm": list(o.geometry.spacing_mm) if o.geometry.spacing_mm else None,
    }


def evidence_to_dict(e: SpatialEvidence) -> dict[str, Any]:
    return {
        "evidence_id": e.evidence_id.value,
        "image_id": e.image_id.value,
        "semantics": e.semantics.value,
        "geometry": geometry_to_dict(e.geometry),
        "value": e.value,
        "value_is_probability": e.value_is_probability,
        "rank": e.rank,
        "producer": _ref(e.producer.model_id, e.producer.version),
        "localization_validation": (
            e.localization_validation.value if e.localization_validation else None
        ),
    }


def analysis_result(prediction: PredictionResult, run: PipelineRun) -> dict[str, Any]:
    fp = run.fingerprint
    return {
        "schema_version": SCHEMA_VERSION,
        "research_use_only": prediction.usage_mode is UsageMode.RESEARCH_UNVALIDATED,
        "prediction": {
            "prediction_id": prediction.prediction_id.value,
            "image_id": prediction.image_id.value,
            "task": _ref(prediction.task.task_id, prediction.task.version),
            "models": [_ref(m.model_id, m.version) for m in prediction.models],
            "capability_status": prediction.capability_status.value,
            "usage_mode": prediction.usage_mode.value,
            "outputs": [output_to_dict(o) for o in prediction.outputs],
            "evidence": [evidence_to_dict(e) for e in prediction.evidence],
        },
        "provenance": {
            "run_id": run.run_id.value,
            "fingerprint": fp.value,
            "pipeline": _ref(fp.pipeline.pipeline_id, fp.pipeline.version),
            "dataset": _ref(fp.dataset.dataset_id, fp.dataset.version) if fp.dataset else None,
            "models": [
                {
                    "slot": m.slot,
                    "id": m.model.model_id.value,
                    "version": m.model.version.value,
                    "artifact_checksum": m.artifact_checksum,
                }
                for m in fp.models
            ],
            "preprocessing": (
                {
                    "id": fp.preprocessing.contract_id.value,
                    "version": fp.preprocessing.version.value,
                    "digest": fp.preprocessing_digest or "",
                }
                if fp.preprocessing
                else None
            ),
            "components": [
                {"role": c.role, "name": c.name, "version": c.version} for c in fp.components
            ],
            "environment": {
                "python_version": fp.environment.python_version,
                "platform": fp.environment.platform,
                "packages": dict(fp.environment.packages),
            },
            "seed": fp.seed,
            "started_at": run.started_at,
            "finished_at": run.finished_at,
        },
    }


def analysis_job(job: AnalysisJob) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "job_id": job.job_id.value,
        "project_id": job.project_id.value,
        "image_id": job.image_id.value,
        "status": job.status.value,
        "stage": job.stage,
        "prediction_id": job.prediction_id.value if job.prediction_id else None,
        "error": {"code": job.error.code, "message": job.error.message} if job.error else None,
    }


def capability_list(capabilities: list[ProjectCapability]) -> dict[str, Any]:
    items = []
    for cap in capabilities:
        bindings = cap.project.bindings()
        items.append(
            {
                "project_id": cap.project.project_id.value,
                "task": {
                    "task_id": cap.task.task_id.value,
                    "version": cap.task.version.value,
                    "task_type": cap.task.task_type.value,
                    "output_kind": cap.task.output_schema.kind.value,
                    "classes": list(cap.task.output_schema.classes),
                },
                "pipeline": str(cap.project.pipeline),
                "models": [
                    {
                        "slot": slot,
                        "model_id": ref.model_id.value,
                        "version": ref.version.value,
                        "family": cap.models[slot].family,
                        "synthetic": cap.models[slot].synthetic,
                    }
                    for slot, ref in sorted(bindings.items())
                ],
                "modality": cap.modality_code,
                "status": cap.decision.status.value,
                "requires_research_acknowledgement": (
                    cap.decision.status is CapabilityStatus.SUPPORTED_UNVALIDATED
                ),
                "reasons": list(cap.decision.reasons),
            }
        )
    return {"schema_version": SCHEMA_VERSION, "capabilities": items}
