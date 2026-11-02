"""YAML/dict serialization adapters for pipeline and project definitions.

These adapters translate between documents and domain values. Pipelines are
fully constructible without them; the engine never reads YAML.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from domain.dataset import DatasetVersionRef
from domain.errors import DomainError
from domain.ids import (
    DatasetId,
    ModelId,
    PipelineId,
    PreprocessingContractId,
    ProjectId,
    StageId,
    TaskId,
    Version,
)
from domain.image import ClinicalContext
from domain.kinds import DataKind, PortSpec
from domain.modality import SpatialKind
from domain.model import ModelRef
from domain.pipeline import (
    Edge,
    OperationRef,
    PipelineDefinition,
    PipelineRef,
    PortRef,
    ResourceRequirements,
    StageDefinition,
)
from domain.preprocessing import PreprocessingRef
from domain.project import IntendedUse, ProjectDefinition
from domain.task import TaskRef


class ConfigurationError(DomainError, ValueError):
    code = "INVALID_CONFIGURATION"


def _require(doc: Mapping[str, Any], key: str) -> Any:
    if key not in doc:
        raise ConfigurationError(f"missing required key {key!r}")
    return doc[key]


def _port(doc: Mapping[str, Any]) -> PortSpec:
    sk = doc.get("spatial_kind")
    return PortSpec(
        name=_require(doc, "name"),
        kind=DataKind(_require(doc, "kind")),
        spatial_kind=SpatialKind(sk) if sk else None,
        feature_dim=doc.get("feature_dim"),
    )


def _port_doc(p: PortSpec) -> dict[str, Any]:
    out: dict[str, Any] = {"name": p.name, "kind": p.kind.name}
    if p.spatial_kind:
        out["spatial_kind"] = p.spatial_kind.value
    if p.feature_dim:
        out["feature_dim"] = p.feature_dim
    return out


def _ref(text: str) -> PortRef:
    stage, _, port = str(text).partition(".")
    if not port:
        raise ConfigurationError(f"port reference {text!r} must be '<stage>.<port>'")
    return PortRef(None if stage == "$input" else StageId(stage), port)


def pipeline_from_dict(doc: Mapping[str, Any]) -> PipelineDefinition:
    allowed = {
        "schema_version",
        "pipeline_id",
        "version",
        "description",
        "inputs",
        "stages",
        "edges",
        "outputs",
    }
    unknown = set(doc) - allowed
    if unknown:
        raise ConfigurationError(f"unknown pipeline keys: {sorted(unknown)}")
    stages = []
    for s in _require(doc, "stages"):
        op = _require(s, "operation")
        res = s.get("resources", {})
        stages.append(
            StageDefinition(
                stage_id=StageId(_require(s, "id")),
                operation=OperationRef(_require(op, "name"), Version(str(_require(op, "version")))),
                inputs=tuple(_port(p) for p in s.get("inputs", ())),
                outputs=tuple(_port(p) for p in _require(s, "outputs")),
                model_slot=s.get("model_slot"),
                requires_preprocessing=bool(s.get("requires_preprocessing", False)),
                deterministic=bool(s.get("deterministic", True)),
                resources=ResourceRequirements(
                    int(res.get("gpu_count", 0)), int(res.get("memory_mb", 0))
                ),
                config=tuple(sorted((str(k), v) for k, v in s.get("config", {}).items())),
                artifact_dependencies=tuple(s.get("artifact_dependencies", ())),
            )
        )
    return PipelineDefinition(
        pipeline_id=PipelineId(_require(doc, "pipeline_id")),
        version=Version(str(_require(doc, "version"))),
        inputs=tuple(_port(p) for p in _require(doc, "inputs")),
        stages=tuple(stages),
        edges=tuple(Edge(_ref(src), _ref(dst)) for src, dst in _require(doc, "edges")),
        outputs=tuple((str(k), _ref(v)) for k, v in sorted(_require(doc, "outputs").items())),
        schema_version=_require(doc, "schema_version"),
        description=doc.get("description", ""),
    )


def pipeline_to_dict(p: PipelineDefinition) -> dict[str, Any]:
    return {
        "schema_version": p.schema_version,
        "pipeline_id": p.pipeline_id.value,
        "version": p.version.value,
        "description": p.description,
        "inputs": [_port_doc(x) for x in p.inputs],
        "stages": [
            {
                "id": s.stage_id.value,
                "operation": {"name": s.operation.name, "version": s.operation.version.value},
                "inputs": [_port_doc(x) for x in s.inputs],
                "outputs": [_port_doc(x) for x in s.outputs],
                **({"model_slot": s.model_slot} if s.model_slot else {}),
                "requires_preprocessing": s.requires_preprocessing,
                "deterministic": s.deterministic,
                "resources": {
                    "gpu_count": s.resources.gpu_count,
                    "memory_mb": s.resources.memory_mb,
                },
                "config": dict(s.config),
                "artifact_dependencies": list(s.artifact_dependencies),
            }
            for s in p.stages
        ],
        "edges": [[str(e.source), str(e.target)] for e in p.edges],
        "outputs": {name: str(ref) for name, ref in p.outputs},
    }


def load_pipeline(path: Path) -> PipelineDefinition:
    return pipeline_from_dict(yaml.safe_load(path.read_text(encoding="utf-8")))


def project_from_dict(doc: Mapping[str, Any]) -> ProjectDefinition:
    ds = _require(doc, "dataset")
    task = _require(doc, "task")
    pipe = _require(doc, "pipeline")
    pre = doc.get("preprocessing")
    ctx = doc.get("target_context") or {}
    bindings = tuple(
        sorted(
            (
                str(slot),
                ModelRef(ModelId(_require(m, "model_id")), Version(str(_require(m, "version")))),
            )
            for slot, m in _require(doc, "models").items()
        )
    )
    return ProjectDefinition(
        project_id=ProjectId(_require(doc, "project_id")),
        namespace=_require(doc, "namespace"),
        display_name=_require(doc, "display_name"),
        dataset=DatasetVersionRef(
            DatasetId(_require(ds, "dataset_id")), Version(str(_require(ds, "version")))
        ),
        task=TaskRef(TaskId(_require(task, "task_id")), Version(str(_require(task, "version")))),
        pipeline=PipelineRef(
            PipelineId(_require(pipe, "pipeline_id")), Version(str(_require(pipe, "version")))
        ),
        model_bindings=bindings,
        preprocessing=(
            PreprocessingRef(
                PreprocessingContractId(_require(pre, "contract_id")),
                Version(str(_require(pre, "version"))),
            )
            if pre
            else None
        ),
        evaluation_protocol=_require(doc, "evaluation_protocol"),
        intended_use=IntendedUse(_require(doc, "intended_use")),
        research_use_acknowledged=bool(doc.get("research_use_acknowledged", False)),
        target_context=ClinicalContext(ctx.get("tissue"), ctx.get("disease")),
        training_config=doc.get("training_config"),
        seed=int(doc.get("seed", 0)),
    )


def load_project(path: Path) -> ProjectDefinition:
    return project_from_dict(yaml.safe_load(path.read_text(encoding="utf-8")))
