"""Patch-to-embedding runs through the existing `PipelineRunner` (SPEC-005).

Like `PreprocessingService`, this is a thin caller of the runner, not a second engine: it opens
the image by content, hands the runner the opened handle and a durable store provider for the
requested output location, and closes the handle when the run ends (also on failure). The store
location travels in `ExecutionRequest.extras`, so it never enters a fingerprint or an identity.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from application.imaging_service import ImagingService
from application.runner import ExecutionOutcome, ExecutionRequest, PipelineRunner
from domain.embedding import EmbeddingManifest
from domain.errors import ExecutionUnsupportedError
from domain.ids import TaskId, Version
from domain.manifest import PatchManifest
from domain.patching import PatchPlan
from domain.pipeline import ConfigValue, PipelineDefinition, ResourceBudget
from domain.task import OutputKind, OutputSchema, TaskDefinition, TaskType
from ports import (
    EMBEDDING_STORE_EXTRA,
    EmbeddingBagPayload,
    EmbeddingStoreProvider,
    ImageSource,
    PatchManifestPayload,
)

WSI_EMBEDDING_TASK = TaskDefinition(
    TaskId("wsi-patch-embedding"),
    Version("1"),
    TaskType.FEATURE_EXTRACTION,
    OutputSchema(OutputKind.EMBEDDING_BAG),
    description="Patch-level feature extraction into a durable embedding bag (no prediction).",
)


@dataclass(frozen=True)
class EmbeddingOutcome:
    execution: ExecutionOutcome
    plan: PatchPlan
    patch_manifest: PatchManifest
    payload: EmbeddingBagPayload

    @property
    def manifest(self) -> EmbeddingManifest:
        return self.payload.manifest


class EmbeddingService:
    def __init__(
        self,
        *,
        imaging: ImagingService,
        runner: PipelineRunner,
        stores: Callable[[Path], EmbeddingStoreProvider],
    ) -> None:
        self._imaging = imaging
        self._runner = runner
        self._stores = stores

    def store_provider(self, output_dir: Path) -> EmbeddingStoreProvider:
        return self._stores(output_dir)

    def run(
        self,
        source: ImageSource,
        pipeline: PipelineDefinition,
        output_dir: Path,
        *,
        scene: int | None = None,
        backend: str | None = None,
        overrides: Sequence[tuple[str, str, ConfigValue]] = (),
        budget: ResourceBudget | None = None,
    ) -> EmbeddingOutcome:
        """Run `pipeline` on a real image. It must output `plan`, `manifest` and `embeddings`.

        The encoder is not a validated model, so the run is recorded as research use.
        """
        with self._imaging.open(source, scene=scene, backend=backend) as image:
            outcome = self._runner.run(
                ExecutionRequest(
                    pipeline=pipeline,
                    image=image.describe().asset,
                    task=WSI_EMBEDDING_TASK,
                    models={},
                    preprocessing=None,
                    research_use_acknowledged=True,
                    budget=budget or ResourceBudget(),
                    source=image,
                    config_overrides=tuple(overrides),
                    extras={EMBEDDING_STORE_EXTRA: self._stores(output_dir)},
                )
            )
        plan = outcome.outputs["plan"].payload
        manifest = outcome.outputs["manifest"].payload
        payload = outcome.outputs["embeddings"].payload
        if not (
            isinstance(plan, PatchPlan)
            and isinstance(manifest, PatchManifestPayload)
            and isinstance(payload, EmbeddingBagPayload)
        ):
            raise ExecutionUnsupportedError(
                f"pipeline {pipeline.ref} does not output plan, manifest and embeddings"
            )
        return EmbeddingOutcome(outcome, plan, manifest.manifest, payload)
