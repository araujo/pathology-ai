"""Helpers for SPEC-007 tests: synthetic labelled cohorts written as finalized stores.

Everything here is SYNTHETIC. Labels are the generator's own classes. Metrics computed on these
cohorts show that the pipeline is correct, not any clinical or biomarker performance.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from application.mil_dataset import MILDatasetService
from bootstrap import MILServices, build_mil
from domain.ids import DatasetId, TaskId, Version
from domain.mil import (
    EXACT,
    BagPolicy,
    CompatibilityPolicy,
    LabelStatus,
    MILDatasetManifest,
    MILTask,
    SlideLabel,
    SplitFractions,
    SplitManifest,
    generate_split,
)
from domain.mil_training import (
    EarlyStopping,
    OptimizerConfig,
    OptimizerName,
    TrainingConfig,
)
from domain.task import TaskType
from infrastructure.embedding_store import NumpyEmbeddingStoreProvider
from infrastructure.mil_bags import NumpyBagSource
from plugins.synthetic.mil import SyntheticSlide, synthetic_cohort, write_bag

BINARY = MILTask(
    TaskId("synthetic-target"),
    Version("1"),
    TaskType.BINARY_CLASSIFICATION,
    ("class-a", "class-b"),
    "class-b",
)
MULTICLASS = MILTask(
    TaskId("synthetic-multi"),
    Version("1"),
    TaskType.MULTICLASS_CLASSIFICATION,
    ("class-a", "class-b", "class-c"),
)
DIM = 32
SOURCE = ("synthetic-generator", "1")
BIG = 64 * 1024 * 1024


@dataclass
class Cohort:
    task: MILTask
    slides: tuple[SyntheticSlide, ...]
    stores: dict[str, str]
    root: Path

    def labels(self, **overrides: object) -> list[SlideLabel]:
        return [
            replace(
                SlideLabel(
                    s.slide_id,
                    self.task.ref,
                    s.label,
                    *SOURCE,
                    LabelStatus.ELIGIBLE,
                    group_id=s.group_id,
                ),
                **overrides,  # type: ignore[arg-type]
            )
            for s in self.slides
        ]

    def source(self) -> NumpyBagSource:
        return NumpyBagSource([self.root])

    def dataset(
        self,
        labels: list[SlideLabel] | None = None,
        *,
        policy: CompatibilityPolicy = EXACT,
        source: NumpyBagSource | None = None,
    ) -> MILDatasetManifest:
        return MILDatasetService(source or self.source()).build(
            dataset_id=DatasetId("synthetic-mil"),
            version=Version("1"),
            task=self.task,
            policy=policy,
            labels=labels if labels is not None else self.labels(),
            stores=self.stores,
        )


def cohort(
    root: Path,
    task: MILTask = BINARY,
    *,
    groups_per_class: int = 4,
    seed: int = 0,
    instances: tuple[int, int] = (8, 64),
    separability: float = 2.0,
    variant: str = "a",
    device: str = "cpu",
) -> Cohort:
    slides = synthetic_cohort(
        task.classes,
        binary=task.binary,
        groups_per_class=groups_per_class,
        instances=instances,
        dim=DIM,
        separability=separability,
        seed=seed,
    )
    provider = NumpyEmbeddingStoreProvider(root)
    stores = {
        s.slide_id: write_bag(provider, s.slide_id, s.rows, variant=variant, device=device)
        for s in slides
    }
    return Cohort(task, slides, stores, root)


def split_of(dataset: MILDatasetManifest, seed: int = 7) -> SplitManifest:
    return generate_split(dataset, SplitFractions(0.6, 0.2, 0.2), seed)


def training_config(**overrides: object) -> TrainingConfig:
    base = TrainingConfig(
        seed=13,
        epochs=30,
        batch_slides=4,
        optimizer=OptimizerConfig(OptimizerName.ADAM, 5e-3, 1e-4),
        early_stopping=EarlyStopping(patience=8),
        train_bags=BagPolicy(48, 13, BIG),
        eval_bags=BagPolicy(None, 0, BIG),
        gradient_clip_norm=5.0,
    )
    return replace(base, **overrides)  # type: ignore[arg-type]


def services(stores: Path, checkpoints: Path) -> MILServices:
    return build_mil([stores], checkpoints)


ATTENTION_HP: tuple[tuple[str, str | int | float | bool], ...] = (
    ("attention_dim", 16),
    ("hidden_dim", 32),
)
