"""Project definitions: configuration that binds dataset, task, pipeline and models.

Disease/tissue namespaces are organisational only. A project's identity is its
`project_id`, independent of display names or filesystem paths, and binding a
dataset never implies that a trained model is scientifically valid for it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from domain.dataset import DatasetVersionRef
from domain.ids import ProjectId
from domain.image import UNKNOWN_CONTEXT, ClinicalContext
from domain.model import ModelRef
from domain.pipeline import PipelineRef
from domain.preprocessing import PreprocessingRef
from domain.task import TaskRef


class IntendedUse(StrEnum):
    RESEARCH = "research"
    TECHNICAL_DEMONSTRATION = "technical_demonstration"


@dataclass(frozen=True, slots=True)
class ProjectDefinition:
    project_id: ProjectId
    namespace: str
    display_name: str
    dataset: DatasetVersionRef
    task: TaskRef
    pipeline: PipelineRef
    model_bindings: tuple[tuple[str, ModelRef], ...]
    preprocessing: PreprocessingRef | None
    evaluation_protocol: str
    intended_use: IntendedUse
    research_use_acknowledged: bool = False
    target_context: ClinicalContext = field(default=UNKNOWN_CONTEXT)
    training_config: str | None = None
    seed: int = 0

    def bindings(self) -> dict[str, ModelRef]:
        return dict(self.model_bindings)
