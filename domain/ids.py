"""Typed, stable identifiers.

Identifiers are independent of display names and filesystem paths. Each kind
is a distinct type so that, for example, a `DatasetId` cannot be passed where
a `ModelId` is expected.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar, Self

from domain.errors import InvalidIdentifierError

_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_VERSION_PATTERN = re.compile(r"^[0-9A-Za-z][0-9A-Za-z.+_-]{0,63}$")


@dataclass(frozen=True, slots=True, order=True)
class _Identifier:
    value: str
    kind: ClassVar[str] = "identifier"

    def __post_init__(self) -> None:
        if not isinstance(self.value, str) or not _ID_PATTERN.fullmatch(self.value):
            raise InvalidIdentifierError(
                f"invalid {self.kind} {self.value!r}: expected lowercase [a-z0-9._-], "
                "starting alphanumeric, at most 128 characters"
            )
        if "/" in self.value or "\\" in self.value:
            raise InvalidIdentifierError(f"{self.kind} must not be a path: {self.value!r}")

    @classmethod
    def of(cls, value: str) -> Self:
        return cls(value)

    def __str__(self) -> str:
        return self.value


class ProjectId(_Identifier):
    kind = "project_id"


class DatasetId(_Identifier):
    kind = "dataset_id"


class ImageId(_Identifier):
    kind = "image_id"


class TaskId(_Identifier):
    kind = "task_id"


class PipelineId(_Identifier):
    kind = "pipeline_id"


class StageId(_Identifier):
    kind = "stage_id"


class ModelId(_Identifier):
    kind = "model_id"


class ArtifactId(_Identifier):
    kind = "artifact_id"


class PreprocessingContractId(_Identifier):
    kind = "preprocessing_contract_id"


class RunId(_Identifier):
    kind = "run_id"


class JobId(_Identifier):
    kind = "job_id"


class PredictionId(_Identifier):
    kind = "prediction_id"


class EvidenceId(_Identifier):
    kind = "evidence_id"


class ValidationRecordId(_Identifier):
    kind = "validation_record_id"


@dataclass(frozen=True, slots=True, order=True)
class Version:
    """An opaque, explicit version label. Never derived from a filename."""

    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, str) or not _VERSION_PATTERN.fullmatch(self.value):
            raise InvalidIdentifierError(f"invalid version {self.value!r}")

    def __str__(self) -> str:
        return self.value
