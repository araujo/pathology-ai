"""Composition root: wires ports to concrete adapters and synthetic plugins.

This is the only module allowed to import every layer.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from application.analysis_service import AnalysisService
from application.embedding_service import EmbeddingService
from application.imaging_service import ImageBackendRegistry, ImagingService
from application.mil_dataset import MILDatasetService
from application.mil_training import MILTrainingService
from application.model_operations import ModelInvocationOperation
from application.preprocessing_service import WSI_PREPROCESSING_TASK, PreprocessingService
from application.project_service import ProjectRunner, ProjectServices
from application.registries import (
    EncoderRegistry,
    InMemoryArtifactStore,
    InMemoryDatasetRepository,
    InMemoryModelRegistry,
    InMemoryPipelineRegistry,
    InMemoryProvenanceStore,
    InMemoryTaskRegistry,
    MILArchitectureRegistry,
    OperationRegistry,
    PreprocessingRegistry,
    ReaderRegistry,
)
from application.runner import PipelineRunner
from domain.image import ChannelOrder
from domain.pipeline import ResourceBudget
from domain.project import ProjectDefinition
from domain.reading import ReadLimits
from infrastructure.config import load_pipeline, load_project
from infrastructure.embedding_store import NumpyEmbeddingStoreProvider
from infrastructure.imaging.integrity import PartialContentFingerprinter
from infrastructure.mil_artifacts import DirectoryCheckpointStore
from infrastructure.mil_bags import NumpyBagSource
from infrastructure.runtime import (
    InlineJobExecutor,
    LocalEnvironmentProbe,
    SystemClock,
)
from plugins.embedding.operations import EmbeddingOperation
from plugins.embedding.reference import ReferenceProjectionFactory
from plugins.imaging.czi import CziBackend
from plugins.imaging.openslide import OpenSlideBackend
from plugins.mil import ATTENTION_MIL, MEAN_POOLING, TorchMILFactory
from plugins.mil.batching import NumpyBatchBuilder
from plugins.models.checkpoints import CheckpointLocator
from plugins.models.fixture import TorchFixtureFactory
from plugins.models.uni import UniFactory
from plugins.preprocessing.detection import SaturationOtsuDetector
from plugins.preprocessing.operations import (
    PatchExtractionOperation,
    PatchPlanOperation,
    TissueDetectionOperation,
)
from plugins.synthetic import catalog
from plugins.synthetic.imaging import SyntheticSlideReader
from plugins.synthetic.models import (
    SyntheticAttentionAggregator,
    SyntheticDirectSegmenter,
    SyntheticPatchEncoder,
)
from plugins.synthetic.operations import GridPatchOperation, TissueThresholdOperation
from ports import Clock, JobExecutor

ROOT_ENV_VAR = "PATHOLOGY_AI_ROOT"
# In a source checkout this module sits next to `pipelines/` and `projects/`; in an
# installed wheel it sits in site-packages, where those definitions are not shipped.
_MODULE_DIR = Path(__file__).resolve().parent


class DefinitionsRootError(FileNotFoundError):
    """No directory holding `pipelines/` and `projects/` could be resolved."""


def _has_definitions(root: Path) -> bool:
    return (root / "pipelines").is_dir() and (root / "projects").is_dir()


def resolve_definitions_root(root: Path | None = None) -> Path:
    """Directory holding `pipelines/` and `projects/`.

    Resolution order: the explicit `root`, then `$PATHOLOGY_AI_ROOT`, then the source
    checkout containing this module. The current working directory is never used, and an
    explicitly configured location that lacks the definitions is an error, not a fallback.
    """
    env = os.environ.get(ROOT_ENV_VAR)
    if root is not None or env:
        source = "root argument" if root is not None else f"${ROOT_ENV_VAR}"
        candidate = Path(root if root is not None else str(env)).expanduser().resolve()
        if not _has_definitions(candidate):
            raise DefinitionsRootError(
                f"{source} {candidate} must contain pipelines/ and projects/ directories"
            )
        return candidate
    if _has_definitions(_MODULE_DIR):
        return _MODULE_DIR
    raise DefinitionsRootError(
        "pipelines/ and projects/ are not packaged with the installed framework; "
        f"pass --root PATH or set ${ROOT_ENV_VAR} to a directory containing them"
    )


@dataclass
class Platform:
    datasets: InMemoryDatasetRepository
    tasks: InMemoryTaskRegistry
    pipelines: InMemoryPipelineRegistry
    preprocessing: PreprocessingRegistry
    models: InMemoryModelRegistry
    operations: OperationRegistry
    readers: ReaderRegistry
    artifacts: InMemoryArtifactStore
    provenance: InMemoryProvenanceStore
    runner: PipelineRunner
    projects: ProjectRunner
    analysis: AnalysisService
    project_definitions: dict[str, ProjectDefinition]
    imaging: ImagingService
    preprocessing_runs: PreprocessingService
    embedding_runs: EmbeddingService
    encoders: EncoderRegistry


def build_platform(
    root: Path | None = None,
    *,
    clock: Clock | None = None,
    executor: JobExecutor | None = None,
    reader_channel_order: ChannelOrder = ChannelOrder.RGB,
    budget: ResourceBudget | None = None,
    read_limits: ReadLimits | None = None,
    checkpoints: Mapping[str, Path] | None = None,
) -> Platform:
    """`checkpoints` binds encoder names to local checkpoint files (operational configuration,
    merged over `$PATHOLOGY_AI_CHECKPOINTS`); paths never enter configs or identities."""
    datasets = InMemoryDatasetRepository()
    tasks = InMemoryTaskRegistry()
    pipelines = InMemoryPipelineRegistry()
    preprocessing = PreprocessingRegistry()
    models = InMemoryModelRegistry()
    operations = OperationRegistry()
    readers = ReaderRegistry()
    artifacts = InMemoryArtifactStore()
    provenance = InMemoryProvenanceStore()

    for contract in catalog.ALL_CONTRACTS:
        preprocessing.add(contract)
    for task in catalog.ALL_TASKS:
        tasks.add(task)
    tasks.add(WSI_PREPROCESSING_TASK)
    for model in catalog.ALL_MODELS:
        models.register_definition(model, catalog.artifact_for(model))
    models.register_plugin(SyntheticPatchEncoder(catalog.ENCODER))
    models.register_plugin(SyntheticAttentionAggregator(catalog.LUNG_AGGREGATOR))
    models.register_plugin(SyntheticAttentionAggregator(catalog.COLORECTAL_AGGREGATOR))
    models.register_plugin(SyntheticDirectSegmenter(catalog.SKIN_SEGMENTER))
    # VOLUME_SEGMENTER intentionally has no plugin: volumetric execution is out of scope.

    for ds in catalog.synthetic_datasets():
        datasets.add_definition(ds.definition)
        for image in ds.images:
            datasets.add_image(image)
        datasets.add_version(ds.version)
        for label in ds.labels:
            datasets.add_label(label)

    operations.add(TissueThresholdOperation())
    operations.add(GridPatchOperation())
    operations.add(ModelInvocationOperation())
    readers.register(SyntheticSlideReader(reader_channel_order))
    # Real-WSI preprocessing (SPEC-004): reader-neutral operations; the detector is replaceable.
    detector = SaturationOtsuDetector()
    operations.add(TissueDetectionOperation({detector.name: detector}))
    operations.add(PatchPlanOperation())
    operations.add(PatchExtractionOperation())
    # Streaming patch embedding (SPEC-005): encoders are selected by name; the NumPy reference
    # encoder is a synthetic architectural fixture. Stores are created per run (EmbeddingService).
    # Foundation-model encoders (SPEC-006) register here too. Their runtime (torch, timm) is
    # imported only when an encoder is created, and their weights come only from locally bound,
    # digest-pinned files.
    locator = CheckpointLocator.from_environment().overridden_by(dict(checkpoints or {}))
    encoders = EncoderRegistry()
    encoders.register(ReferenceProjectionFactory())
    encoders.register(TorchFixtureFactory(locator))
    encoders.register(UniFactory(locator))
    operations.add(EmbeddingOperation(encoders))

    root = resolve_definitions_root(root)
    for path in sorted((root / "pipelines").glob("*.yaml")):
        pipelines.add(load_pipeline(path))
    project_definitions = {
        p.namespace: p
        for p in (load_project(path) for path in sorted((root / "projects").rglob("project.yaml")))
    }

    # Real-image readers (SPEC-003). Optional backends register even when their extra is not
    # installed: resolution then reports READER_BACKEND_UNAVAILABLE instead of misreading.
    image_backends = build_image_backends()
    # The same backend instances also serve the runner, keyed by the format codes they claim.
    for backend in image_backends.backends():
        readers.register(backend)

    runner = PipelineRunner(
        operations=operations,
        readers=readers,
        models=models,
        artifacts=artifacts,
        provenance=provenance,
        clock=clock or SystemClock(),
        environment=LocalEnvironmentProbe(),
    )
    projects = ProjectRunner(
        ProjectServices(datasets, tasks, pipelines, preprocessing, runner), budget
    )
    imaging = build_imaging_service(
        backends=image_backends, artifacts=artifacts, clock=clock, limits=read_limits
    )
    return Platform(
        datasets=datasets,
        tasks=tasks,
        pipelines=pipelines,
        preprocessing=preprocessing,
        models=models,
        operations=operations,
        readers=readers,
        artifacts=artifacts,
        provenance=provenance,
        runner=runner,
        projects=projects,
        analysis=AnalysisService(projects, executor or InlineJobExecutor()),
        project_definitions=project_definitions,
        imaging=imaging,
        preprocessing_runs=PreprocessingService(imaging=imaging, runner=runner),
        embedding_runs=EmbeddingService(
            imaging=imaging, runner=runner, stores=NumpyEmbeddingStoreProvider
        ),
        encoders=encoders,
    )


def build_imaging_service(
    *,
    backends: ImageBackendRegistry | None = None,
    artifacts: InMemoryArtifactStore | None = None,
    clock: Clock | None = None,
    limits: ReadLimits | None = None,
) -> ImagingService:
    """Real-image ingestion only; needs no pipeline or project definitions."""
    return ImagingService(
        backends=backends or build_image_backends(),
        fingerprinter=PartialContentFingerprinter(),
        artifacts=artifacts or InMemoryArtifactStore(),
        clock=clock or SystemClock(),
        environment=LocalEnvironmentProbe(),
        limits=limits,
    )


def build_image_backends() -> ImageBackendRegistry:
    backends = ImageBackendRegistry()
    backends.register(CziBackend())
    backends.register(OpenSlideBackend())
    return backends


# --------------------------------------------------------------------------- MIL (SPEC-007)


@dataclass
class MILServices:
    """Dataset-level MIL workflows over finalized embedding stores (no pipeline runner)."""

    architectures: MILArchitectureRegistry
    bags: NumpyBagSource
    datasets: MILDatasetService
    training: MILTrainingService


def build_mil_architectures() -> MILArchitectureRegistry:
    """MIL architectures by name. Listing them imports no framework."""
    registry = MILArchitectureRegistry()
    registry.register(TorchMILFactory(MEAN_POOLING))
    registry.register(TorchMILFactory(ATTENTION_MIL))
    return registry


def build_mil(store_roots: Sequence[Path], checkpoint_root: Path) -> MILServices:
    """`store_roots` hold finalized embedding stores; `checkpoint_root` receives checkpoints.

    Both are operational locations: they never enter a dataset, split, experiment or
    checkpoint identity."""
    architectures = build_mil_architectures()
    bags = NumpyBagSource(store_roots)
    return MILServices(
        architectures=architectures,
        bags=bags,
        datasets=MILDatasetService(bags),
        training=MILTrainingService(
            bags=bags,
            batches=NumpyBatchBuilder(),
            architectures=architectures,
            checkpoints=DirectoryCheckpointStore(checkpoint_root),
        ),
    )
