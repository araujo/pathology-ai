"""Task definitions and typed output schemas (not limited to binary classification)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from domain.errors import DomainError
from domain.ids import TaskId, Version


class TaskType(StrEnum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    MULTILABEL_CLASSIFICATION = "multilabel_classification"
    REGRESSION = "regression"
    SEGMENTATION = "segmentation"
    DETECTION = "detection"
    QUANTIFICATION = "quantification"
    # Model-free image preprocessing (e.g. tissue candidates and patch extraction). It produces
    # artifacts, never a prediction, so it is not a model capability and never appears in the
    # Analysis API.
    PREPROCESSING = "preprocessing"
    # Patch-level feature extraction (SPEC-005): an embedding bag, never a prediction.
    FEATURE_EXTRACTION = "feature_extraction"


class OutputKind(StrEnum):
    IMAGE_CLASSIFICATION = "image_classification"
    CONTINUOUS_SCORE = "continuous_score"
    REGION_PREDICTION = "region_prediction"
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    VOLUMETRIC_SEGMENTATION = "volumetric_segmentation"
    # Output of a preprocessing task: an artifact manifest, not a prediction.
    PATCH_MANIFEST = "patch_manifest"
    # Output of a feature-extraction task: a durable embedding bag, not a prediction.
    EMBEDDING_BAG = "embedding_bag"


_ALLOWED: dict[TaskType, frozenset[OutputKind]] = {
    TaskType.BINARY_CLASSIFICATION: frozenset({OutputKind.IMAGE_CLASSIFICATION}),
    TaskType.MULTICLASS_CLASSIFICATION: frozenset({OutputKind.IMAGE_CLASSIFICATION}),
    TaskType.MULTILABEL_CLASSIFICATION: frozenset({OutputKind.IMAGE_CLASSIFICATION}),
    TaskType.REGRESSION: frozenset({OutputKind.CONTINUOUS_SCORE}),
    TaskType.QUANTIFICATION: frozenset({OutputKind.CONTINUOUS_SCORE}),
    TaskType.SEGMENTATION: frozenset({OutputKind.SEGMENTATION, OutputKind.VOLUMETRIC_SEGMENTATION}),
    TaskType.DETECTION: frozenset({OutputKind.DETECTION, OutputKind.REGION_PREDICTION}),
    TaskType.PREPROCESSING: frozenset({OutputKind.PATCH_MANIFEST}),
    TaskType.FEATURE_EXTRACTION: frozenset({OutputKind.EMBEDDING_BAG}),
}


class InvalidTaskDefinitionError(DomainError, ValueError):
    code = "INVALID_TASK"


@dataclass(frozen=True, slots=True)
class OutputSchema:
    kind: OutputKind
    classes: tuple[str, ...] = ()
    unit: str | None = None

    def __post_init__(self) -> None:
        if len(set(self.classes)) != len(self.classes):
            raise InvalidTaskDefinitionError("output classes must be unique")


@dataclass(frozen=True, slots=True)
class TaskRef:
    task_id: TaskId
    version: Version


@dataclass(frozen=True, slots=True)
class TaskDefinition:
    task_id: TaskId
    version: Version
    task_type: TaskType
    output_schema: OutputSchema
    target_name: str | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if self.output_schema.kind not in _ALLOWED[self.task_type]:
            raise InvalidTaskDefinitionError(
                f"task type {self.task_type.value} cannot produce {self.output_schema.kind.value}"
            )
        n = len(self.output_schema.classes)
        if self.task_type is TaskType.BINARY_CLASSIFICATION and n != 2:
            raise InvalidTaskDefinitionError("binary classification requires exactly two classes")
        if (
            self.task_type
            in (
                TaskType.MULTICLASS_CLASSIFICATION,
                TaskType.MULTILABEL_CLASSIFICATION,
            )
            and n < 2
        ):
            raise InvalidTaskDefinitionError("multi-class/label tasks require at least two classes")

    @property
    def ref(self) -> TaskRef:
        return TaskRef(self.task_id, self.version)
