"""Asynchronous analysis job state machine."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from enum import StrEnum

from domain.errors import InvalidJobTransitionError
from domain.ids import ImageId, JobId, PredictionId, ProjectId


class JobStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


TERMINAL_STATUSES = frozenset({JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED})

_TRANSITIONS: dict[JobStatus, frozenset[JobStatus]] = {
    JobStatus.PENDING: frozenset({JobStatus.RUNNING, JobStatus.CANCELLED}),
    JobStatus.RUNNING: frozenset({JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}),
    JobStatus.COMPLETED: frozenset(),
    JobStatus.FAILED: frozenset(),
    JobStatus.CANCELLED: frozenset(),
}


@dataclass(frozen=True, slots=True)
class JobError:
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class AnalysisJob:
    job_id: JobId
    project_id: ProjectId
    image_id: ImageId
    status: JobStatus = JobStatus.PENDING
    stage: str | None = None
    prediction_id: PredictionId | None = None
    error: JobError | None = None

    def transition(self, status: JobStatus, **changes: object) -> AnalysisJob:
        if status not in _TRANSITIONS[self.status]:
            raise InvalidJobTransitionError(f"{self.status.value} -> {status.value} not allowed")
        if status is JobStatus.COMPLETED and changes.get("prediction_id") is None:
            raise InvalidJobTransitionError("a completed job must reference its prediction")
        if status is JobStatus.FAILED and changes.get("error") is None:
            raise InvalidJobTransitionError("a failed job must carry a typed error")
        return dataclasses.replace(self, status=status, **changes)  # type: ignore[arg-type]
