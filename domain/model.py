"""Model definitions and artifacts.

A model declares what it consumes and produces, which preprocessing it was
built with, and which modalities it is technically compatible with. Model
family and architecture are open strings: no family is privileged.
"""

from __future__ import annotations

from dataclasses import dataclass

from domain.ids import ArtifactId, ModelId, Version
from domain.kinds import PortSpec
from domain.preprocessing import PreprocessingRef
from domain.task import OutputSchema


@dataclass(frozen=True, slots=True, order=True)
class ModelRef:
    model_id: ModelId
    version: Version

    def __str__(self) -> str:
        return f"{self.model_id}@{self.version}"


@dataclass(frozen=True, slots=True)
class ModelDefinition:
    model_id: ModelId
    version: Version
    family: str
    architecture: str
    input: PortSpec
    output: PortSpec
    compatible_modalities: frozenset[str]
    required_preprocessing: PreprocessingRef | None = None
    output_schema: OutputSchema | None = None
    required_upstream_model: ModelRef | None = None
    dependencies: tuple[str, ...] = ()
    synthetic: bool = False
    description: str = ""

    @property
    def ref(self) -> ModelRef:
        return ModelRef(self.model_id, self.version)


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    """Identity of concrete weights. `checksum` is a content digest, never a path."""

    artifact_id: ArtifactId
    model: ModelRef
    checksum: str
    preprocessing: PreprocessingRef | None
    synthetic: bool = False
