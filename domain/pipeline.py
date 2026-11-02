"""Versioned, typed pipeline definitions (a DAG of stages).

Pipelines are plain domain values constructed programmatically. Serialized
forms (YAML/JSON) are adapters in infrastructure, never the engine itself.
"""

from __future__ import annotations

from dataclasses import dataclass

from domain.ids import PipelineId, StageId, Version
from domain.kinds import PortSpec

PIPELINE_SCHEMA_VERSION = "pathology-ai.pipeline/1"
SUPPORTED_PIPELINE_SCHEMAS = frozenset({PIPELINE_SCHEMA_VERSION})

ConfigValue = str | int | float | bool


@dataclass(frozen=True, slots=True, order=True)
class OperationRef:
    """Names the stage implementation contract (resolved through a registry at run time)."""

    name: str
    version: Version

    def __str__(self) -> str:
        return f"{self.name}@{self.version}"


@dataclass(frozen=True, slots=True)
class ResourceRequirements:
    gpu_count: int = 0
    memory_mb: int = 0


@dataclass(frozen=True, slots=True)
class ResourceBudget:
    """Resources available to one execution; the default models a CPU-only workstation."""

    gpu_count: int = 0
    memory_mb: int = 4096


@dataclass(frozen=True, slots=True)
class StageDefinition:
    stage_id: StageId
    operation: OperationRef
    inputs: tuple[PortSpec, ...]
    outputs: tuple[PortSpec, ...]
    model_slot: str | None = None
    requires_preprocessing: bool = False
    deterministic: bool = True
    resources: ResourceRequirements = ResourceRequirements()
    config: tuple[tuple[str, ConfigValue], ...] = ()
    artifact_dependencies: tuple[str, ...] = ()

    def input(self, name: str) -> PortSpec | None:
        return next((p for p in self.inputs if p.name == name), None)

    def output(self, name: str) -> PortSpec | None:
        return next((p for p in self.outputs if p.name == name), None)

    def config_dict(self) -> dict[str, ConfigValue]:
        return dict(self.config)


@dataclass(frozen=True, slots=True, order=True)
class PortRef:
    """`stage=None` addresses a pipeline-level input."""

    stage: StageId | None
    port: str

    def __str__(self) -> str:
        return f"{self.stage or '$input'}.{self.port}"


@dataclass(frozen=True, slots=True)
class Edge:
    source: PortRef
    target: PortRef


@dataclass(frozen=True, slots=True, order=True)
class PipelineRef:
    pipeline_id: PipelineId
    version: Version

    def __str__(self) -> str:
        return f"{self.pipeline_id}@{self.version}"


@dataclass(frozen=True, slots=True)
class PipelineDefinition:
    pipeline_id: PipelineId
    version: Version
    inputs: tuple[PortSpec, ...]
    stages: tuple[StageDefinition, ...]
    edges: tuple[Edge, ...]
    outputs: tuple[tuple[str, PortRef], ...]
    schema_version: str = PIPELINE_SCHEMA_VERSION
    description: str = ""

    @property
    def ref(self) -> PipelineRef:
        return PipelineRef(self.pipeline_id, self.version)

    def stage(self, stage_id: StageId) -> StageDefinition | None:
        return next((s for s in self.stages if s.stage_id == stage_id), None)

    def input(self, name: str) -> PortSpec | None:
        return next((p for p in self.inputs if p.name == name), None)

    @property
    def model_slots(self) -> tuple[str, ...]:
        return tuple(sorted(s.model_slot for s in self.stages if s.model_slot))
