"""Adapters and plugins structurally satisfy the port protocols (checked by mypy and at runtime)."""

from __future__ import annotations

from application.model_operations import ModelInvocationOperation
from application.registries import (
    InMemoryArtifactStore,
    InMemoryDatasetRepository,
    InMemoryModelRegistry,
    InMemoryPipelineRegistry,
    InMemoryProvenanceStore,
    InMemoryTaskRegistry,
)
from infrastructure.runtime import (
    DeferredJobExecutor,
    FixedClock,
    InlineJobExecutor,
    LocalEnvironmentProbe,
    SystemClock,
)
from plugins.synthetic import catalog
from plugins.synthetic.imaging import SyntheticSlideReader
from plugins.synthetic.models import (
    SyntheticAttentionAggregator,
    SyntheticDirectSegmenter,
    SyntheticPatchEncoder,
)
from plugins.synthetic.operations import GridPatchOperation, TissueThresholdOperation
from ports import (
    ArtifactStore,
    Clock,
    DatasetRepository,
    Encoder,
    EnvironmentProbe,
    GroundTruthSource,
    ImageReader,
    ImageTileProvider,
    JobExecutor,
    Model,
    ModelRegistry,
    PipelineRegistry,
    ProvenanceStore,
    StageOperation,
    TaskRegistry,
)


def test_static_and_runtime_port_conformance() -> None:
    datasets: DatasetRepository = InMemoryDatasetRepository()
    ground_truth: GroundTruthSource = InMemoryDatasetRepository()
    models: ModelRegistry = InMemoryModelRegistry()
    pipelines: PipelineRegistry = InMemoryPipelineRegistry()
    tasks: TaskRegistry = InMemoryTaskRegistry()
    artifacts: ArtifactStore = InMemoryArtifactStore()
    provenance: ProvenanceStore = InMemoryProvenanceStore()
    executors: tuple[JobExecutor, ...] = (InlineJobExecutor(), DeferredJobExecutor())
    clocks: tuple[Clock, ...] = (SystemClock(), FixedClock())
    env: EnvironmentProbe = LocalEnvironmentProbe()
    reader: ImageReader = SyntheticSlideReader()
    tiles: ImageTileProvider = SyntheticSlideReader()
    encoder: Encoder = SyntheticPatchEncoder(catalog.ENCODER)
    plugins: tuple[Model, ...] = (
        SyntheticAttentionAggregator(catalog.LUNG_AGGREGATOR),
        SyntheticDirectSegmenter(catalog.SKIN_SEGMENTER),
    )
    ops: tuple[StageOperation, ...] = (
        TissueThresholdOperation(),
        GridPatchOperation(),
        ModelInvocationOperation(),
    )
    pairs = [
        (datasets, DatasetRepository),
        (ground_truth, GroundTruthSource),
        (models, ModelRegistry),
        (pipelines, PipelineRegistry),
        (tasks, TaskRegistry),
        (artifacts, ArtifactStore),
        (provenance, ProvenanceStore),
        (env, EnvironmentProbe),
        (reader, ImageReader),
        (tiles, ImageTileProvider),
        (encoder, Encoder),
        *((e, JobExecutor) for e in executors),
        *((c, Clock) for c in clocks),
        *((p, Model) for p in plugins),
        *((o, StageOperation) for o in ops),
    ]
    for obj, proto in pairs:
        assert isinstance(obj, proto), (type(obj).__name__, proto.__name__)
