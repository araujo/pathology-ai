"""Asynchronous analysis jobs over project runs (transport-agnostic)."""

from __future__ import annotations

from application.project_service import ProjectRunner
from domain.errors import DomainError, NotFoundError
from domain.hashing import short_digest
from domain.ids import ImageId, JobId
from domain.job import AnalysisJob, JobError, JobStatus
from domain.prediction import PredictionResult
from domain.project import ProjectDefinition
from domain.provenance import PipelineRun
from ports import JobExecutor


class AnalysisService:
    def __init__(self, projects: ProjectRunner, executor: JobExecutor) -> None:
        self._projects = projects
        self._executor = executor
        self._jobs: dict[JobId, AnalysisJob] = {}
        self._results: dict[JobId, tuple[PredictionResult, PipelineRun]] = {}

    def submit(self, project: ProjectDefinition, image_id: ImageId) -> AnalysisJob:
        job_id = JobId(
            short_digest({"p": project.project_id, "i": image_id, "n": len(self._jobs)}, "job")
        )
        job = AnalysisJob(job_id, project.project_id, image_id)
        self._jobs[job_id] = job

        def work() -> None:
            if self._jobs[job_id].status is JobStatus.CANCELLED:
                return
            self._jobs[job_id] = self._jobs[job_id].transition(JobStatus.RUNNING, stage="EXECUTING")
            try:
                outcome = self._projects.run_image(project, image_id)
            except DomainError as exc:
                self._jobs[job_id] = self._jobs[job_id].transition(
                    JobStatus.FAILED, stage=None, error=JobError(exc.code, str(exc))
                )
                return
            if outcome.prediction is None:
                self._jobs[job_id] = self._jobs[job_id].transition(
                    JobStatus.FAILED,
                    stage=None,
                    error=JobError("NO_PREDICTION", "pipeline produced no prediction output"),
                )
                return
            self._results[job_id] = (outcome.prediction, outcome.run)
            self._jobs[job_id] = self._jobs[job_id].transition(
                JobStatus.COMPLETED, stage=None, prediction_id=outcome.prediction.prediction_id
            )

        self._executor.submit(job_id, work)
        return self._jobs[job_id]

    def cancel(self, job_id: JobId) -> AnalysisJob:
        self._jobs[job_id] = self.job(job_id).transition(JobStatus.CANCELLED)
        return self._jobs[job_id]

    def job(self, job_id: JobId) -> AnalysisJob:
        try:
            return self._jobs[job_id]
        except KeyError:
            raise NotFoundError(f"job {job_id} not found") from None

    def result(self, job_id: JobId) -> tuple[PredictionResult, PipelineRun]:
        job = self.job(job_id)
        if job.status is not JobStatus.COMPLETED:
            raise NotFoundError(f"job {job_id} has no result (status {job.status.value})")
        return self._results[job_id]
