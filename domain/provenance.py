"""Artifact identity, execution provenance and reproducibility fingerprints.

`ExecutionFingerprint` is the reproducibility key (what was run, with what);
`PipelineRun.run_id` identifies one concrete execution. Scientific validation
is recorded elsewhere (`validation.ValidationRecord`): a successful run is not
a validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from domain.dataset import DatasetVersionRef
from domain.hashing import digest, short_digest
from domain.ids import ArtifactId, ImageId, ProjectId, RunId
from domain.kinds import DataKind
from domain.model import ModelRef
from domain.pipeline import PipelineRef
from domain.preprocessing import PreprocessingRef
from domain.task import TaskRef
from domain.validation import CapabilityStatus, UsageMode


@dataclass(frozen=True, slots=True, order=True)
class ComponentVersion:
    """A versioned executable component: reader, operation, plugin, package."""

    role: str
    name: str
    version: str


@dataclass(frozen=True, slots=True)
class EnvironmentInfo:
    python_version: str
    platform: str
    packages: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class ModelArtifactRecord:
    slot: str
    model: ModelRef
    artifact_checksum: str


@dataclass(frozen=True, slots=True)
class ExecutionFingerprint:
    pipeline: PipelineRef
    pipeline_digest: str
    dataset: DatasetVersionRef | None
    dataset_digest: str | None
    task: TaskRef
    models: tuple[ModelArtifactRecord, ...]
    preprocessing: PreprocessingRef | None
    preprocessing_digest: str | None
    components: tuple[ComponentVersion, ...]
    environment: EnvironmentInfo
    seed: int
    config_digest: str

    @property
    def value(self) -> str:
        return short_digest(self, "fp")


@dataclass(frozen=True, slots=True)
class ArtifactIdentity:
    """Content-addressed identity; differs whenever any producing input differs."""

    artifact_id: ArtifactId
    kind: DataKind
    lineage: tuple[ArtifactId, ...]

    @staticmethod
    def derive(
        kind: DataKind, producer: object, lineage: tuple[ArtifactId, ...] = ()
    ) -> ArtifactIdentity:
        aid = ArtifactId(
            short_digest({"kind": kind, "producer": producer, "lineage": lineage}, "art")
        )
        return ArtifactIdentity(aid, kind, lineage)


class RunStatus(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class PipelineRun:
    run_id: RunId
    project_id: ProjectId | None
    image_ids: tuple[ImageId, ...]
    fingerprint: ExecutionFingerprint
    capability_status: CapabilityStatus
    usage_mode: UsageMode
    status: RunStatus
    stage_versions: tuple[ComponentVersion, ...]
    artifacts: tuple[ArtifactIdentity, ...]
    started_at: str
    finished_at: str
    error_code: str | None = None

    @property
    def fingerprint_value(self) -> str:
        return self.fingerprint.value

    @property
    def record_digest(self) -> str:
        return digest(self)
