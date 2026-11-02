"""Port interfaces (typing.Protocol). Depends only on the domain layer.

Pixel/voxel payloads are typed as `object`: concrete array libraries are a
plugin concern and never leak into ports or the domain.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Protocol, Self, runtime_checkable

from domain.capabilities import PreprocessingRequirements
from domain.cohort import CohortObservations
from domain.dataset import DatasetDefinition, DatasetVersion, DatasetVersionRef, Label
from domain.embedding import (
    ChunkCommit,
    DurableStoreReference,
    EmbeddingManifest,
    EmbeddingMetrics,
    EmbeddingStoreHeader,
    EncoderIdentity,
    ModelInputSpec,
    Precision,
    RecoveryPolicy,
    StoreInspection,
    StoreVerification,
)
from domain.foundation import EncoderDescriptor, EncoderRuntimeReport
from domain.geometry import Level0Rect, LevelRect
from domain.ids import ArtifactId, DatasetId, ImageId, JobId, RunId
from domain.image import ChannelOrder, ImageAsset
from domain.kinds import DataKind
from domain.manifest import PatchManifest
from domain.mil import BagSelection, EmbeddingBagInfo
from domain.mil_training import (
    CheckpointDescriptor,
    LossKind,
    MILArchitectureDescriptor,
    MILModelSpec,
    TrainingConfig,
)
from domain.modality import ImageModality, SpatialKind
from domain.model import ModelArtifact, ModelDefinition, ModelRef
from domain.pipeline import (
    ConfigValue,
    OperationRef,
    PipelineDefinition,
    PipelineRef,
    StageDefinition,
)
from domain.prediction import PredictionOutput, SpatialEvidence
from domain.preprocessing import PreprocessingContract, TransformStep
from domain.provenance import (
    ArtifactIdentity,
    ComponentVersion,
    EnvironmentInfo,
    PipelineRun,
)
from domain.reading import (
    ImageDescription,
    NoDataSemantics,
    PixelRegion,
    ReadLimits,
    RegionReadRequest,
    SourceIntegrity,
)
from domain.task import TaskDefinition, TaskRef
from domain.tissue import MaskGrid
from domain.validation import ValidationRecord
from domain.wsi_inventory import InventoryRoot, ObservedObject

# --------------------------------------------------------------------------- imaging


@dataclass(frozen=True, slots=True)
class ReaderCapabilities:
    formats: frozenset[str]
    spatial_kinds: frozenset[SpatialKind]
    channel_order: ChannelOrder | None
    provides_physical_spacing: bool
    supports_regions: bool = False
    supports_thumbnails: bool = False
    supports_volumes: bool = False
    reports_no_data_regions: bool = False
    # SPEC-003 additions (defaulted, so existing readers are unaffected).
    supports_scenes: bool = False
    detects_content: bool = False


@runtime_checkable
class ImageMetadataProvider(Protocol):
    def describe(self, image_id: ImageId) -> ImageAsset: ...


@runtime_checkable
class ImageTileProvider(Protocol):
    def read_region(self, image: ImageAsset, region: LevelRect) -> object: ...

    def thumbnail(self, image: ImageAsset, max_size: int) -> object: ...


@runtime_checkable
class VolumeProvider(Protocol):
    def read_volume(self, image: ImageAsset) -> object: ...

    def read_slice(self, image: ImageAsset, axis: int, index: int) -> object: ...


@runtime_checkable
class ImageReader(Protocol):
    """A reader plugin. Region/volume access is exposed via the provider protocols."""

    @property
    def component(self) -> ComponentVersion: ...

    @property
    def capabilities(self) -> ReaderCapabilities: ...


# --------------------------------------------------------------------------- real image sources


@dataclass(frozen=True, slots=True)
class ImageSource:
    """A locally readable image file plus its declared modality.

    The path is an infrastructure detail: it never becomes part of an image id, a descriptor
    or an API document. The modality is declared by the caller, never inferred from the format.
    """

    image_id: ImageId
    path: Path
    modality: ImageModality


@dataclass(frozen=True, slots=True)
class SourceProbe:
    """A reader's verdict on a source's content (not its file extension)."""

    recognised: bool
    format_code: str | None
    detail: str


@runtime_checkable
class OpenedImage(Protocol):
    """An open image handle. Must be closed; usable as a context manager."""

    def describe(self) -> ImageDescription: ...

    def read_region(self, request: RegionReadRequest) -> PixelRegion: ...

    def close(self) -> None: ...

    @property
    def closed(self) -> bool: ...

    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None: ...


@runtime_checkable
class ImageBackend(ImageReader, Protocol):
    """A real-image reader adapter selected by content probing and declared capabilities."""

    @property
    def name(self) -> str: ...

    @property
    def backend_component(self) -> ComponentVersion: ...

    def available(self) -> bool: ...

    def probe(self, path: Path) -> SourceProbe: ...

    def open(
        self,
        source: ImageSource,
        *,
        scene: int | None,
        limits: ReadLimits,
        integrity: SourceIntegrity | None,
    ) -> OpenedImage: ...


@runtime_checkable
class SourceFingerprinter(Protocol):
    def fingerprint(self, path: Path) -> SourceIntegrity: ...


# --------------------------------------------------------------------------- processing


@runtime_checkable
class PreprocessingStrategy(Protocol):
    def apply(
        self, pixels: object, contract: PreprocessingContract, conversion: TransformStep | None
    ) -> object: ...


@runtime_checkable
class PatchStrategy(Protocol):
    def plan(
        self, image: ImageAsset, tissue_mask: object, contract: PreprocessingContract
    ) -> tuple[Level0Rect, ...]: ...


@dataclass(frozen=True, slots=True)
class DataValue:
    """A typed value flowing along a pipeline edge, with content-addressed identity."""

    kind: DataKind
    payload: object
    identity: ArtifactIdentity


@dataclass(frozen=True, slots=True)
class PredictionPayload:
    """Final model output: typed outputs plus evidence and auxiliary artifacts."""

    outputs: tuple[PredictionOutput, ...]
    evidence: tuple[SpatialEvidence, ...] = ()
    artifacts: tuple[tuple[ArtifactIdentity, object], ...] = ()


@dataclass(frozen=True, slots=True)
class PatchManifestPayload:
    """A patch manifest plus the pixel artifacts it references (stored by the runner)."""

    manifest: PatchManifest
    artifacts: tuple[tuple[ArtifactIdentity, object], ...] = ()


@dataclass(frozen=True)
class StageContext:
    """Everything a stage may use. `source` is the opened real image, when the run has one."""

    image: ImageAsset
    stage: StageDefinition
    seed: int
    reader: ImageReader | None = None
    preprocessing: PreprocessingContract | None = None
    channel_conversion: TransformStep | None = None
    model: Model | None = None
    model_artifact: ModelArtifact | None = None
    extras: Mapping[str, object] = field(default_factory=dict)
    source: OpenedImage | None = None
    pipeline: PipelineRef | None = None


@runtime_checkable
class Model(Protocol):
    """A model plugin (encoder, aggregator, segmenter, detector, ...)."""

    @property
    def definition(self) -> ModelDefinition: ...

    @property
    def component(self) -> ComponentVersion: ...

    def predict(self, value: DataValue, ctx: StageContext) -> object: ...


@runtime_checkable
class Encoder(Model, Protocol):
    """Marker protocol for feature extractors (a Model whose output is an embedding)."""


@runtime_checkable
class StageOperation(Protocol):
    """Runtime implementation of an `OperationRef`."""

    @property
    def operation(self) -> OperationRef: ...

    @property
    def component(self) -> ComponentVersion: ...

    def run(self, inputs: Mapping[str, DataValue], ctx: StageContext) -> Mapping[str, object]: ...


@runtime_checkable
class CapabilityGated(Protocol):
    """An operation that needs an opened image with declared capabilities.

    The runner evaluates `requirements` against the opened image before any stage runs.
    """

    def requirements(self, config: Mapping[str, ConfigValue]) -> PreprocessingRequirements: ...


@dataclass(frozen=True)
class DetectorInput:
    """Spatially referenced pixels for tissue detection, in the reader's native representation.

    `pixels` is `(grid.height, grid.width, samples)` with one value per mask cell. `acquired` is the
    acquired fraction per cell, or `None` when the reader cannot report validity.
    """

    pixels: object
    channel_order: ChannelOrder
    dtype: str
    grid: MaskGrid
    acquired: object | None
    validity: NoDataSemantics


@dataclass(frozen=True)
class DetectorResult:
    tissue: object  # bool (grid.height, grid.width), False wherever a cell is not acquired
    derived: tuple[tuple[str, ConfigValue], ...]
    channel_view: tuple[str, ...]
    limitations: tuple[str, ...]


@runtime_checkable
class TissueDetector(Protocol):
    """Candidate-tissue detection on a mask grid. Replaceable; never a tumour detector."""

    @property
    def name(self) -> str: ...

    @property
    def component(self) -> ComponentVersion: ...

    def requirements(self, config: Mapping[str, ConfigValue]) -> PreprocessingRequirements: ...

    def detect(self, data: DetectorInput, config: Mapping[str, ConfigValue]) -> DetectorResult: ...


# --------------------------------------------------------------------------- embedding (SPEC-005)

# `StageContext.extras` key under which a run supplies its `EmbeddingStoreProvider`. The store
# location is per-execution infrastructure, so it never enters a stage config or a fingerprint.
EMBEDDING_STORE_EXTRA = "embedding_store"


@dataclass(frozen=True, slots=True)
class EncodeContext:
    """Which rows a batch fills. A row must depend only on its patch; an encoder that cannot
    guarantee batch invariance binds its batch configuration into its identity (ADR-006 D9)."""

    batch_index: int
    first_row: int
    patch_ids: tuple[str, ...]
    precision: Precision
    device: str


@runtime_checkable
class EmbeddingEncoder(Protocol):
    """A batch-oriented, framework-neutral encoder (ADR-005 D5).

    `encode_batch` receives an array shaped `(batch, ...)` exactly as `input_spec` declares (layout,
    dtype, size) and returns `(batch, output_dim)` rows of `output_dtype`. Backends may convert to
    their own tensor types internally; no such type crosses this port.
    """

    @property
    def identity(self) -> EncoderIdentity: ...

    @property
    def input_spec(self) -> ModelInputSpec: ...

    def encode_batch(self, batch: object, context: EncodeContext) -> object: ...

    def working_bytes(self, batch_patches: int) -> int:
        """Upper bound of host memory one `encode_batch` call allocates beyond its input and
        output; it is counted in the run's memory budget. Device memory is reported separately
        (`runtime_report`)."""
        ...

    def runtime_report(self) -> EncoderRuntimeReport:
        """Deterministic limitations (recorded in the manifest) and measured resources
        (recorded in the bounded run metrics only)."""
        ...

    def close(self) -> None: ...


@runtime_checkable
class EmbeddingEncoderFactory(Protocol):
    """Creates a configured encoder for one run; the run closes it (explicit lifecycle)."""

    @property
    def name(self) -> str: ...

    @property
    def descriptor(self) -> EncoderDescriptor:
        """Registry metadata (architecture, checkpoint pin, devices, profiles), without loading."""
        ...

    def input_spec(self, config: Mapping[str, ConfigValue]) -> ModelInputSpec:
        """The input contract `create(config)` would declare, without loading the encoder.

        Raises a typed error when `config` selects an unknown, blocked or incomplete profile."""
        ...

    def create(self, config: Mapping[str, ConfigValue]) -> EmbeddingEncoder: ...


@runtime_checkable
class EmbeddingStore(Protocol):
    """A durable, chunked embedding store (ADR-005 D6/D7).

    Rows are written per chunk and count only once committed. A commit records the chunk's
    integrity after its bytes are durable. `finalize` succeeds only after complete verified
    coverage and is the explicit completion marker.
    """

    @property
    def header(self) -> EmbeddingStoreHeader: ...

    @property
    def reference(self) -> DurableStoreReference: ...

    def inspect(self) -> StoreInspection: ...

    def write_chunk(self, index: int, rows: object) -> None: ...

    def commit_chunk(self, index: int) -> ChunkCommit: ...

    def verify(self) -> StoreVerification: ...

    def finalize(self, manifest: EmbeddingManifest) -> None: ...

    def read_rows(self, start: int, stop: int) -> object: ...

    def close(self) -> None: ...


@runtime_checkable
class EmbeddingStoreProvider(Protocol):
    @property
    def format(self) -> str: ...

    def open(self, header: EmbeddingStoreHeader, *, recovery: RecoveryPolicy) -> EmbeddingStore:
        """Create the store, or reopen and validate an existing one for this header."""
        ...

    def open_existing(self, store_id: str) -> EmbeddingStore:
        """Reopen a store read-only, without a header to compare against."""
        ...


@dataclass(frozen=True, slots=True)
class EmbeddingBagPayload:
    """What `embed.patches` returns: a store reference, its manifest and bounded metrics.

    Embedding rows stay in the durable store; they never travel through the artifact store.
    """

    manifest: EmbeddingManifest
    store: DurableStoreReference
    metrics: EmbeddingMetrics


# --------------------------------------------------------------------------- MIL (SPEC-007)


@runtime_checkable
class EmbeddingBagSource(Protocol):
    """Read-only access to finalized embedding bags by store id (ADR-007 D1).

    `describe` verifies the store (finalized, every chunk re-hashed) before reporting it; a store
    that is incomplete, corrupt or unknown raises a typed error. `read` returns the selected rows
    as a `(len(indices), dim)` array, in the given order; it never reads unselected rows.
    """

    def describe(self, store_id: str) -> EmbeddingBagInfo: ...

    def read(self, store_id: str, indices: tuple[int, ...]) -> object: ...


@dataclass(frozen=True, slots=True)
class MILBatch:
    """Padded bags: `features` `(B, N, D)` float32 and `mask` `(B, N)` bool, True for valid
    instances. Row `j` of bag `b` is instance `selections[b].indices[j]` of that bag."""

    slide_ids: tuple[str, ...]
    features: object
    mask: object
    selections: tuple[BagSelection, ...]
    targets: tuple[int, ...] | None
    padded_bytes: int


@dataclass(frozen=True, slots=True)
class MILOutput:
    """Per-slide logits and, for attention models, per-slide weights over the valid instances
    (in selection order). Plain floats: no framework type crosses the port."""

    logits: tuple[tuple[float, ...], ...]
    attention: tuple[tuple[float, ...], ...] | None


@runtime_checkable
class MILBatchBuilder(Protocol):
    def build(
        self,
        bags: Sequence[tuple[BagSelection, object]],
        targets: tuple[int, ...] | None,
        dim: int,
    ) -> MILBatch: ...


@dataclass(frozen=True, slots=True)
class ModelTensor:
    """One named tensor of a model state: little-endian, C-ordered bytes."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    data: bytes


@runtime_checkable
class MILModel(Protocol):
    """One instantiated MIL model with its optimizer (if trainable). Explicit lifecycle."""

    @property
    def spec(self) -> MILModelSpec: ...

    @property
    def parameter_count(self) -> int: ...

    def train_batch(self, batch: MILBatch) -> float:
        """One optimizer step on a labelled batch; returns the batch's mean loss."""
        ...

    def evaluate_batch(self, batch: MILBatch) -> tuple[float, MILOutput]:
        """Mean loss and outputs without gradients or parameter updates (labelled batch)."""
        ...

    def predict(self, batch: MILBatch) -> MILOutput: ...

    def state(self) -> tuple[ModelTensor, ...]: ...

    def load_state(self, tensors: tuple[ModelTensor, ...]) -> None: ...

    def close(self) -> None: ...


@runtime_checkable
class MILModelFactory(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def descriptor(self) -> MILArchitectureDescriptor:
        """Registry metadata, available without importing any framework."""
        ...

    def runtime(self, device: str) -> tuple[tuple[str, str], ...]:
        """Runtime versions and device kind that results depend on (bound into identities)."""
        ...

    def create(
        self,
        spec: MILModelSpec,
        *,
        loss: LossKind,
        seed: int,
        training: TrainingConfig | None,
        class_weights: tuple[float, ...] | None = None,
        device: str = "cpu",
    ) -> MILModel:
        """A seeded model; trainable (with its optimizer) when `training` is given."""
        ...


@runtime_checkable
class MILCheckpointStore(Protocol):
    """Durable, integrity-checked checkpoints keyed by experiment identity (SPEC-007 §11)."""

    def find(self, experiment_identity: str) -> CheckpointDescriptor | None:
        """The verified descriptor of a complete checkpoint, or None when there is none."""
        ...

    def save(self, descriptor: CheckpointDescriptor, tensors: tuple[ModelTensor, ...]) -> None:
        """Write atomically. An existing checkpoint of the same experiment must be identical;
        a location holding another experiment is never overwritten."""
        ...

    def load(
        self, experiment_identity: str
    ) -> tuple[CheckpointDescriptor, tuple[ModelTensor, ...]]: ...


# --------------------------------------------------------------------------- cohorts (SPEC-008)


@runtime_checkable
class CohortSourceReader(Protocol):
    """Reads a cohort's declared sources, verifying each file against its pinned digest."""

    def read(self) -> CohortObservations: ...


@runtime_checkable
class WSIObjectSource(Protocol):
    """Observes image objects under named roots without modifying them. A local source reads and
    hashes every matching object; a remote source reports listing metadata only."""

    @property
    def roots(self) -> tuple[InventoryRoot, ...]: ...

    @property
    def patterns(self) -> tuple[str, ...]: ...

    def observe(self) -> tuple[ObservedObject, ...]: ...


# --------------------------------------------------------------------------- repositories


@runtime_checkable
class DatasetRepository(Protocol):
    def definition(self, dataset_id: DatasetId) -> DatasetDefinition: ...

    def version(self, ref: DatasetVersionRef) -> DatasetVersion: ...

    def image(self, image_id: ImageId) -> ImageAsset: ...


@runtime_checkable
class GroundTruthSource(Protocol):
    def label_for(self, image_id: ImageId, task: TaskRef) -> Label | None: ...


@runtime_checkable
class ModelRegistry(Protocol):
    def definition(self, ref: ModelRef) -> ModelDefinition: ...

    def artifact(self, ref: ModelRef) -> ModelArtifact: ...

    def plugin(self, ref: ModelRef) -> Model | None: ...

    def validation_records(self, ref: ModelRef) -> tuple[ValidationRecord, ...]: ...

    def all_definitions(self) -> tuple[ModelDefinition, ...]: ...


@runtime_checkable
class PipelineRegistry(Protocol):
    def get(self, ref: PipelineRef) -> PipelineDefinition: ...


@runtime_checkable
class TaskRegistry(Protocol):
    def get(self, ref: TaskRef) -> TaskDefinition: ...


@runtime_checkable
class ArtifactStore(Protocol):
    def put(self, identity: ArtifactIdentity, payload: object) -> None: ...

    def get(self, artifact_id: ArtifactId) -> object: ...

    def exists(self, artifact_id: ArtifactId) -> bool: ...


@runtime_checkable
class ProvenanceStore(Protocol):
    def record(self, run: PipelineRun) -> None: ...

    def get(self, run_id: RunId) -> PipelineRun: ...

    def all(self) -> tuple[PipelineRun, ...]: ...


@runtime_checkable
class JobExecutor(Protocol):
    def submit(self, job_id: JobId, work: Callable[[], None]) -> None: ...


@runtime_checkable
class Clock(Protocol):
    def now_iso(self) -> str: ...


@runtime_checkable
class EnvironmentProbe(Protocol):
    def capture(self) -> EnvironmentInfo: ...


__all__ = [
    "ArtifactStore",
    "CapabilityGated",
    "Clock",
    "CohortSourceReader",
    "DataValue",
    "DatasetRepository",
    "DetectorInput",
    "DetectorResult",
    "EMBEDDING_STORE_EXTRA",
    "EmbeddingBagPayload",
    "EmbeddingBagSource",
    "EmbeddingEncoder",
    "EmbeddingEncoderFactory",
    "EmbeddingStore",
    "EmbeddingStoreProvider",
    "EncodeContext",
    "Encoder",
    "EnvironmentProbe",
    "GroundTruthSource",
    "ImageBackend",
    "ImageMetadataProvider",
    "ImageReader",
    "ImageSource",
    "ImageTileProvider",
    "JobExecutor",
    "MILBatch",
    "MILBatchBuilder",
    "MILCheckpointStore",
    "MILModel",
    "MILModelFactory",
    "MILOutput",
    "Model",
    "ModelRegistry",
    "ModelTensor",
    "OpenedImage",
    "PatchManifestPayload",
    "PatchStrategy",
    "PipelineRegistry",
    "PredictionPayload",
    "PreprocessingStrategy",
    "ProvenanceStore",
    "ReaderCapabilities",
    "SourceFingerprinter",
    "SourceProbe",
    "StageContext",
    "StageOperation",
    "TaskRegistry",
    "TissueDetector",
    "VolumeProvider",
    "WSIObjectSource",
]
