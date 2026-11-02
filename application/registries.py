"""In-process registries and stores implementing the repository ports."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Iterator, Mapping

from domain.dataset import DatasetDefinition, DatasetVersion, DatasetVersionRef, Label
from domain.errors import DuplicateRegistrationError, ExecutionUnsupportedError, NotFoundError
from domain.foundation import EncoderDescriptor
from domain.ids import ArtifactId, DatasetId, ImageId, RunId
from domain.image import ImageAsset
from domain.mil_training import MILArchitectureDescriptor
from domain.model import ModelArtifact, ModelDefinition, ModelRef
from domain.pipeline import OperationRef, PipelineDefinition, PipelineRef
from domain.pipeline_validation import require_valid_structure
from domain.preprocessing import PreprocessingContract, PreprocessingRef
from domain.provenance import ArtifactIdentity, PipelineRun
from domain.task import TaskDefinition, TaskRef
from domain.validation import ValidationRecord
from ports import EmbeddingEncoderFactory, ImageReader, MILModelFactory, Model, StageOperation


class Registry[K: Hashable, V]:
    """Write-once keyed registry; re-registering a key is an error, not an overwrite."""

    def __init__(self, what: str) -> None:
        self._what = what
        self._items: dict[K, V] = {}

    def register(self, key: K, value: V) -> None:
        if key in self._items:
            raise DuplicateRegistrationError(f"{self._what} {key} is already registered")
        self._items[key] = value

    def get(self, key: K) -> V:
        try:
            return self._items[key]
        except KeyError:
            raise NotFoundError(f"{self._what} {key} is not registered") from None

    def find(self, key: K) -> V | None:
        return self._items.get(key)

    def __contains__(self, key: object) -> bool:
        return key in self._items

    def values(self) -> tuple[V, ...]:
        return tuple(self._items[k] for k in sorted(self._items, key=str))


class ReaderRegistry:
    """Readers keyed by format code. Selection never infers modality."""

    def __init__(self) -> None:
        self._by_format: Registry[str, ImageReader] = Registry("reader for format")

    def register(self, reader: ImageReader) -> None:
        for fmt in sorted(reader.capabilities.formats):
            self._by_format.register(fmt, reader)

    def for_image(self, image: ImageAsset) -> ImageReader | None:
        reader = self._by_format.find(image.image_format.code)
        if reader is None:
            return None
        if image.spatial_reference.spatial_kind not in reader.capabilities.spatial_kinds:
            return None
        return reader


class OperationRegistry(Registry[OperationRef, StageOperation]):
    def __init__(self) -> None:
        super().__init__("operation")

    def add(self, op: StageOperation) -> None:
        self.register(op.operation, op)


class EncoderRegistry(Mapping[str, EmbeddingEncoderFactory]):
    """Embedding encoders by stable name (SPEC-006 §6). Resolution fails closed.

    A factory is admitted only with a descriptor that names it. The registry stores factories,
    never loaded models or weights; loading happens per run in `create`, from local files only.
    """

    def __init__(self) -> None:
        self._factories: Registry[str, EmbeddingEncoderFactory] = Registry("embedding encoder")

    def register(self, factory: EmbeddingEncoderFactory) -> None:
        d = factory.descriptor
        if d.name != factory.name:
            raise DuplicateRegistrationError(
                f"encoder factory {factory.name!r} carries the descriptor of {d.name!r}"
            )
        self._factories.register(factory.name, factory)

    def resolve(self, name: str) -> EmbeddingEncoderFactory:
        found = self._factories.find(name)
        if found is None:
            known = ", ".join(self) or "none"
            raise ExecutionUnsupportedError(
                f"embedding encoder {name!r} is not registered (known: {known})",
                (f"encoder:{name}",),
            )
        return found

    def descriptors(self) -> tuple[EncoderDescriptor, ...]:
        return tuple(f.descriptor for f in self._factories.values())

    def __getitem__(self, name: str) -> EmbeddingEncoderFactory:
        found = self._factories.find(name)
        if found is None:
            raise KeyError(name)
        return found

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(f.name for f in self._factories.values()))

    def __len__(self) -> int:
        return len(self._factories.values())


class MILArchitectureRegistry:
    """MIL architectures by stable name (SPEC-007 §7). Resolution fails closed.

    The registry holds factories and their descriptors; no framework is imported to list them.
    """

    def __init__(self) -> None:
        self._factories: Registry[str, MILModelFactory] = Registry("MIL architecture")

    def register(self, factory: MILModelFactory) -> None:
        named = factory.descriptor.name
        if named != factory.name:
            raise DuplicateRegistrationError(
                f"MIL factory {factory.name!r} carries the descriptor of {named!r}"
            )
        self._factories.register(factory.name, factory)

    def resolve(self, name: str) -> MILModelFactory:
        found = self._factories.find(name)
        if found is None:
            known = ", ".join(f.name for f in self._factories.values()) or "none"
            raise ExecutionUnsupportedError(
                f"MIL architecture {name!r} is not registered (known: {known})",
                (f"mil-architecture:{name}",),
            )
        return found

    def descriptors(self) -> tuple[MILArchitectureDescriptor, ...]:
        return tuple(f.descriptor for f in self._factories.values())


class InMemoryModelRegistry:
    def __init__(self) -> None:
        self._defs: Registry[ModelRef, ModelDefinition] = Registry("model")
        self._artifacts: Registry[ModelRef, ModelArtifact] = Registry("model artifact")
        self._plugins: Registry[ModelRef, Model] = Registry("model plugin")
        self._records: dict[ModelRef, list[ValidationRecord]] = {}

    def register_definition(self, d: ModelDefinition, artifact: ModelArtifact) -> None:
        if artifact.model != d.ref:
            raise DuplicateRegistrationError("artifact does not belong to this model definition")
        self._defs.register(d.ref, d)
        self._artifacts.register(d.ref, artifact)

    def register_plugin(self, plugin: Model) -> None:
        ref = plugin.definition.ref
        self._defs.get(ref)  # a plugin needs a registered definition
        self._plugins.register(ref, plugin)

    def add_validation_record(self, record: ValidationRecord) -> None:
        self._defs.get(record.scope.model)
        self._records.setdefault(record.scope.model, []).append(record)

    def definition(self, ref: ModelRef) -> ModelDefinition:
        return self._defs.get(ref)

    def artifact(self, ref: ModelRef) -> ModelArtifact:
        return self._artifacts.get(ref)

    def plugin(self, ref: ModelRef) -> Model | None:
        return self._plugins.find(ref)

    def validation_records(self, ref: ModelRef) -> tuple[ValidationRecord, ...]:
        return tuple(self._records.get(ref, ()))

    def all_definitions(self) -> tuple[ModelDefinition, ...]:
        return self._defs.values()


class InMemoryPipelineRegistry(Registry[PipelineRef, PipelineDefinition]):
    def __init__(self) -> None:
        super().__init__("pipeline")

    def add(self, p: PipelineDefinition) -> None:
        require_valid_structure(p)
        self.register(p.ref, p)


class InMemoryTaskRegistry(Registry[TaskRef, TaskDefinition]):
    def __init__(self) -> None:
        super().__init__("task")

    def add(self, t: TaskDefinition) -> None:
        self.register(t.ref, t)


class PreprocessingRegistry(Registry[PreprocessingRef, PreprocessingContract]):
    def __init__(self) -> None:
        super().__init__("preprocessing contract")

    def add(self, c: PreprocessingContract) -> None:
        self.register(c.ref, c)


class InMemoryDatasetRepository:
    def __init__(self) -> None:
        self._defs: Registry[DatasetId, DatasetDefinition] = Registry("dataset")
        self._versions: Registry[DatasetVersionRef, DatasetVersion] = Registry("dataset version")
        self._images: Registry[ImageId, ImageAsset] = Registry("image")
        self._labels: dict[tuple[ImageId, TaskRef], Label] = {}

    def add_definition(self, d: DatasetDefinition) -> None:
        self._defs.register(d.dataset_id, d)

    def add_version(self, v: DatasetVersion) -> None:
        self._defs.get(v.dataset_id)
        for image_id in v.image_ids:
            self._images.get(image_id)
        self._versions.register(v.ref, v)

    def add_image(self, image: ImageAsset) -> None:
        """Ingestion requires no label and no disease/tissue context."""
        self._images.register(image.image_id, image)

    def add_label(self, label: Label) -> None:
        self._images.get(label.image_id)
        self._labels[(label.image_id, label.task)] = label

    def definition(self, dataset_id: DatasetId) -> DatasetDefinition:
        return self._defs.get(dataset_id)

    def version(self, ref: DatasetVersionRef) -> DatasetVersion:
        return self._versions.get(ref)

    def image(self, image_id: ImageId) -> ImageAsset:
        return self._images.get(image_id)

    def label_for(self, image_id: ImageId, task: TaskRef) -> Label | None:
        return self._labels.get((image_id, task))

    def labels(self, image_ids: Iterable[ImageId], task: TaskRef) -> dict[ImageId, Label]:
        out: dict[ImageId, Label] = {}
        for i in image_ids:
            lbl = self.label_for(i, task)
            if lbl is not None:
                out[i] = lbl
        return out


class InMemoryArtifactStore:
    def __init__(self) -> None:
        self._items: dict[ArtifactId, tuple[ArtifactIdentity, object]] = {}

    def put(self, identity: ArtifactIdentity, payload: object) -> None:
        self._items.setdefault(identity.artifact_id, (identity, payload))

    def get(self, artifact_id: ArtifactId) -> object:
        try:
            return self._items[artifact_id][1]
        except KeyError:
            raise NotFoundError(f"artifact {artifact_id} not found") from None

    def exists(self, artifact_id: ArtifactId) -> bool:
        return artifact_id in self._items


class InMemoryProvenanceStore:
    def __init__(self) -> None:
        self._runs: Registry[RunId, PipelineRun] = Registry("run")

    def record(self, run: PipelineRun) -> None:
        self._runs.register(run.run_id, run)

    def get(self, run_id: RunId) -> PipelineRun:
        return self._runs.get(run_id)

    def all(self) -> tuple[PipelineRun, ...]:
        return self._runs.values()
