"""Real-image preprocessing through the existing `PipelineRunner` (SPEC-004).

The service resolves the reader by content (`ImagingService.open`), runs the requested pipeline
with the opened handle as `ExecutionRequest.source`, and closes the handle when the run ends,
also on failure. It is a thin caller of the runner, not a second engine.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from application.imaging_service import ImagingService
from application.runner import ExecutionOutcome, ExecutionRequest, PipelineRunner
from domain.errors import ExecutionUnsupportedError
from domain.ids import TaskId, Version
from domain.manifest import PatchManifest
from domain.patching import PatchPlan
from domain.pipeline import ConfigValue, PipelineDefinition, ResourceBudget
from domain.task import OutputKind, OutputSchema, TaskDefinition, TaskType
from domain.tissue import TissueMask
from ports import ImageSource, PatchManifestPayload

WSI_PREPROCESSING_TASK = TaskDefinition(
    TaskId("wsi-preprocessing"),
    Version("1"),
    TaskType.PREPROCESSING,
    OutputSchema(OutputKind.PATCH_MANIFEST),
    description="Model-free candidate-tissue detection and patch extraction (no prediction).",
)


@dataclass(frozen=True)
class PreprocessingOutcome:
    execution: ExecutionOutcome
    mask: TissueMask
    plan: PatchPlan
    payload: PatchManifestPayload

    @property
    def manifest(self) -> PatchManifest:
        return self.payload.manifest


class PreprocessingService:
    def __init__(self, *, imaging: ImagingService, runner: PipelineRunner) -> None:
        self._imaging = imaging
        self._runner = runner

    def run(
        self,
        source: ImageSource,
        pipeline: PipelineDefinition,
        *,
        scene: int | None = None,
        backend: str | None = None,
        overrides: Sequence[tuple[str, str, ConfigValue]] = (),
        budget: ResourceBudget | None = None,
    ) -> PreprocessingOutcome:
        """Run `pipeline` on a real image. The pipeline must output `mask`, `plan`, `manifest`.

        No model runs, so the capability decision is SUPPORTED_UNVALIDATED and the run is
        recorded as research use: the candidate-tissue detector is not validated.
        """
        with self._imaging.open(source, scene=scene, backend=backend) as image:
            description = image.describe()
            outcome = self._runner.run(
                ExecutionRequest(
                    pipeline=pipeline,
                    image=description.asset,
                    task=WSI_PREPROCESSING_TASK,
                    models={},
                    preprocessing=None,
                    research_use_acknowledged=True,
                    budget=budget or ResourceBudget(),
                    source=image,
                    config_overrides=tuple(overrides),
                )
            )
        mask = outcome.outputs["mask"].payload
        plan = outcome.outputs["plan"].payload
        payload = outcome.outputs["manifest"].payload
        if not (
            isinstance(mask, TissueMask)
            and isinstance(plan, PatchPlan)
            and isinstance(payload, PatchManifestPayload)
        ):
            raise ExecutionUnsupportedError(
                f"pipeline {pipeline.ref} does not output mask, plan and manifest"
            )
        return PreprocessingOutcome(outcome, mask, plan, payload)
