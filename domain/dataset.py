"""Datasets, dataset versions and labels.

Labels live outside image assets. Supervised use requires labels explicitly;
ingestion and inference never do.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from domain.errors import InvalidIdentifierError, MissingLabelError
from domain.ids import DatasetId, ImageId, Version
from domain.task import TaskRef


@dataclass(frozen=True, slots=True, order=True)
class DatasetVersionRef:
    dataset_id: DatasetId
    version: Version

    def __str__(self) -> str:
        return f"{self.dataset_id}@{self.version}"


@dataclass(frozen=True, slots=True)
class DatasetDefinition:
    dataset_id: DatasetId
    display_name: str
    modality_code: str
    synthetic: bool = False
    description: str = ""


@dataclass(frozen=True, slots=True)
class DatasetVersion:
    dataset_id: DatasetId
    version: Version
    content_digest: str
    image_ids: tuple[ImageId, ...]
    splits: tuple[tuple[str, tuple[ImageId, ...]], ...] = ()

    def __post_init__(self) -> None:
        if len(set(self.image_ids)) != len(self.image_ids):
            raise InvalidIdentifierError("dataset version lists an image twice")
        known = set(self.image_ids)
        seen: set[ImageId] = set()
        for name, members in self.splits:
            for m in members:
                if m not in known:
                    raise InvalidIdentifierError(f"split {name!r} references unknown image {m}")
                if m in seen:
                    raise InvalidIdentifierError(f"image {m} appears in more than one split")
                seen.add(m)

    @property
    def ref(self) -> DatasetVersionRef:
        return DatasetVersionRef(self.dataset_id, self.version)

    def split(self, name: str) -> tuple[ImageId, ...]:
        for n, members in self.splits:
            if n == name:
                return members
        raise InvalidIdentifierError(f"dataset {self.ref} has no split {name!r}")


@dataclass(frozen=True, slots=True)
class Label:
    image_id: ImageId
    task: TaskRef
    value: str | float


def require_labels(
    image_ids: Iterable[ImageId], task: TaskRef, labels: Mapping[ImageId, Label]
) -> tuple[Label, ...]:
    """Return labels for every image or fail listing the unlabelled ones (sorted)."""
    ids = tuple(image_ids)
    missing = sorted(str(i) for i in ids if i not in labels)
    if missing:
        raise MissingLabelError(
            f"{len(missing)} image(s) have no label for task {task.task_id}@{task.version}: "
            + ", ".join(missing)
        )
    out = tuple(labels[i] for i in ids)
    for lbl in out:
        if lbl.task != task:
            raise MissingLabelError(f"label for {lbl.image_id} belongs to another task")
    return out
