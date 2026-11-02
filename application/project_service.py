"""Project-level orchestration: resolve a project's bindings and run its pipeline.

The same `PipelineDefinition` is shared by any number of projects; a project
only selects dataset, task, models, preprocessing and validation context.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from application.registries import (
    InMemoryDatasetRepository,
    InMemoryPipelineRegistry,
    InMemoryTaskRegistry,
    PreprocessingRegistry,
)
from application.runner import ExecutionOutcome, ExecutionRequest, PipelineRunner
from domain.dataset import Label, require_labels
from domain.errors import NotFoundError, PipelineValidationError
from domain.ids import ImageId
from domain.model import ModelDefinition
from domain.pipeline import ResourceBudget
from domain.project import ProjectDefinition
from domain.task import TaskDefinition
from domain.validation import CapabilityDecision, CapabilityStatus


@dataclass(frozen=True)
class ProjectServices:
    datasets: InMemoryDatasetRepository
    tasks: InMemoryTaskRegistry
    pipelines: InMemoryPipelineRegistry
    preprocessing: PreprocessingRegistry
    runner: PipelineRunner


@dataclass(frozen=True)
class ProjectCapability:
    project: ProjectDefinition
    task: TaskDefinition
    models: Mapping[str, ModelDefinition]
    modality_code: str
    decision: CapabilityDecision


class ProjectRunner:
    def __init__(self, services: ProjectServices, budget: ResourceBudget | None = None) -> None:
        self._s = services
        self._budget = budget or ResourceBudget()

    def request_for(self, project: ProjectDefinition, image_id: ImageId) -> ExecutionRequest:
        version = self._s.datasets.version(project.dataset)
        if image_id not in version.image_ids:
            raise NotFoundError(f"image {image_id} is not part of dataset {project.dataset}")
        return ExecutionRequest(
            pipeline=self._s.pipelines.get(project.pipeline),
            image=self._s.datasets.image(image_id),
            task=self._s.tasks.get(project.task),
            models=project.bindings(),
            preprocessing=(
                self._s.preprocessing.get(project.preprocessing) if project.preprocessing else None
            ),
            dataset=version,
            project_id=project.project_id,
            seed=project.seed,
            research_use_acknowledged=project.research_use_acknowledged,
            target_context=project.target_context,
            budget=self._budget,
        )

    def capability(self, project: ProjectDefinition) -> ProjectCapability:
        """Capability of the project's bound models for its dataset modality and target context."""
        version = self._s.datasets.version(project.dataset)
        req = self.request_for(project, version.image_ids[0])
        models = self._s.runner.definitions(req)
        try:
            self._s.runner.validate(req)
            decision = self._s.runner.capability(req, models)
        except PipelineValidationError as exc:
            decision = CapabilityDecision(
                CapabilityStatus.INCOMPATIBLE,
                tuple(f"[{i.code}] {i.message}" for i in exc.issues),
            )
        return ProjectCapability(project, req.task, models, req.image.modality.code, decision)

    def run_image(self, project: ProjectDefinition, image_id: ImageId) -> ExecutionOutcome:
        return self._s.runner.run(self.request_for(project, image_id))

    def run_split(self, project: ProjectDefinition, split: str) -> tuple[ExecutionOutcome, ...]:
        version = self._s.datasets.version(project.dataset)
        return tuple(self.run_image(project, i) for i in version.split(split))

    def supervised_examples(
        self, project: ProjectDefinition, split: str
    ) -> Mapping[ImageId, Label]:
        """Supervised use requires a label for every image; inference never does."""
        version = self._s.datasets.version(project.dataset)
        ids = version.split(split)
        labels = require_labels(ids, project.task, self._s.datasets.labels(ids, project.task))
        return {lbl.image_id: lbl for lbl in labels}
