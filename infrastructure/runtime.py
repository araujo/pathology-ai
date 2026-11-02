"""Clocks, environment capture and job executors."""

from __future__ import annotations

import platform
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from importlib import metadata

from domain.ids import JobId
from domain.provenance import EnvironmentInfo

_TRACKED_PACKAGES = ("pathology-ai", "numpy", "pyyaml")


class SystemClock:
    def now_iso(self) -> str:
        return datetime.now(UTC).isoformat()


class FixedClock:
    """Deterministic clock for reproducibility tests."""

    def __init__(self, value: str = "2026-01-01T00:00:00+00:00") -> None:
        self.value = value

    def now_iso(self) -> str:
        return self.value


class LocalEnvironmentProbe:
    def capture(self) -> EnvironmentInfo:
        packages: list[tuple[str, str]] = []
        for name in _TRACKED_PACKAGES:
            try:
                packages.append((name, metadata.version(name)))
            except metadata.PackageNotFoundError:
                packages.append((name, "not-installed"))
        return EnvironmentInfo(
            python_version=".".join(map(str, sys.version_info[:3])),
            platform=f"{platform.system()}-{platform.machine()}",
            packages=tuple(packages),
        )


class InlineJobExecutor:
    """Runs work synchronously at submission time (in-process, no workers)."""

    def submit(self, job_id: JobId, work: Callable[[], None]) -> None:
        work()


class DeferredJobExecutor:
    """Queues work until `drain()`; lets tests observe PENDING/RUNNING transitions."""

    def __init__(self) -> None:
        self.queue: list[tuple[JobId, Callable[[], None]]] = []

    def submit(self, job_id: JobId, work: Callable[[], None]) -> None:
        self.queue.append((job_id, work))

    def drain(self) -> None:
        while self.queue:
            _, work = self.queue.pop(0)
            work()
