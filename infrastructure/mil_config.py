"""YAML experiment configuration and CSV label/split tables for MIL (SPEC-007).

The configuration declares the task (classes are free-form configuration), the dataset and
its label source, the compatibility policy, the split, the model and the training. Unknown
keys are errors, so a misspelt option never falls back silently to a default.

Label table (CSV, header required):
    slide_id,store_id,label,group_id,cohort,site,status,status_reason
`status` is required (`eligible`, `excluded` or `unlabeled`). An empty label is never treated
as a class: it is valid only for `unlabeled` rows. Empty optional cells are absent values.

Split table (CSV): `slide_id,partition` with partitions `train`, `validation`, `test`.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from domain.ids import DatasetId, TaskId, Version
from domain.mil import (
    EXTERNAL_STRATEGY,
    GENERATED_STRATEGY,
    BagPolicy,
    CompatibilityPolicy,
    LabelStatus,
    MILTask,
    Partition,
    SlideLabel,
    SplitFractions,
    compatibility_policy,
)
from domain.mil_training import (
    EarlyStopping,
    OptimizerConfig,
    OptimizerName,
    SelectionMode,
    TrainingConfig,
)
from domain.pipeline import ConfigValue
from domain.task import TaskType
from infrastructure.config import ConfigurationError

CONFIG_SCHEMA = "pathology-ai.mil-experiment-config/1"
LABEL_COLUMNS = (
    "slide_id",
    "store_id",
    "label",
    "group_id",
    "cohort",
    "site",
    "status",
    "status_reason",
)


@dataclass(frozen=True)
class SplitConfig:
    strategy: str
    seed: int | None
    fractions: SplitFractions | None
    file: Path | None


@dataclass(frozen=True)
class MILExperimentConfig:
    name: str
    task: MILTask
    dataset_id: DatasetId
    dataset_version: Version
    labels: Path
    label_source: str
    label_source_version: str
    policy: CompatibilityPolicy
    accepted_key: str | None
    split: SplitConfig
    architecture: str
    hyperparameters: tuple[tuple[str, ConfigValue], ...]
    training: TrainingConfig


def _keys(
    doc: Any, where: str, required: set[str], optional: frozenset[str] | set[str] = frozenset()
) -> dict[str, Any]:
    if not isinstance(doc, Mapping):
        raise ConfigurationError(f"{where} must be a mapping")
    keys = set(doc)
    missing = sorted(required - keys)
    unknown = sorted(keys - required - optional)
    if missing or unknown:
        raise ConfigurationError(
            f"{where}: missing {missing or 'nothing'}, unknown {unknown or 'nothing'}"
        )
    return dict(doc)


def _int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{where} must be an integer")
    return value


def _float(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigurationError(f"{where} must be a number")
    return float(value)


def _str(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigurationError(f"{where} must be a non-empty string (quote versions in YAML)")
    return value


def _bag_policy(doc: Any, where: str) -> BagPolicy:
    d = _keys(doc, where, {"max_instances", "seed", "max_batch_bytes"})
    mi = d["max_instances"]
    return BagPolicy(
        max_instances=None if mi is None else _int(mi, f"{where}.max_instances"),
        seed=_int(d["seed"], f"{where}.seed"),
        max_batch_bytes=_int(d["max_batch_bytes"], f"{where}.max_batch_bytes"),
    )


def _training(doc: Any) -> TrainingConfig:
    d = _keys(
        doc,
        "training",
        {"seed", "epochs", "batch_slides", "optimizer", "early_stopping", "bags"},
        {"gradient_clip_norm", "class_weighting", "device"},
    )
    o = _keys(
        d["optimizer"],
        "training.optimizer",
        {"name", "learning_rate"},
        {"weight_decay", "momentum"},
    )
    e = _keys(
        d["early_stopping"],
        "training.early_stopping",
        {"metric", "mode", "patience"},
        {"min_delta"},
    )
    b = _keys(d["bags"], "training.bags", {"train", "eval"})
    clip = d.get("gradient_clip_norm")
    weighting = d.get("class_weighting", False)
    if not isinstance(weighting, bool):
        raise ConfigurationError("training.class_weighting must be a boolean")
    try:
        optimizer = OptimizerName(_str(o["name"], "training.optimizer.name"))
        mode = SelectionMode(_str(e["mode"], "training.early_stopping.mode"))
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from None
    return TrainingConfig(
        seed=_int(d["seed"], "training.seed"),
        epochs=_int(d["epochs"], "training.epochs"),
        batch_slides=_int(d["batch_slides"], "training.batch_slides"),
        optimizer=OptimizerConfig(
            optimizer,
            _float(o["learning_rate"], "training.optimizer.learning_rate"),
            _float(o.get("weight_decay", 0.0), "training.optimizer.weight_decay"),
            _float(o.get("momentum", 0.0), "training.optimizer.momentum"),
        ),
        early_stopping=EarlyStopping(
            _str(e["metric"], "training.early_stopping.metric"),
            mode,
            _int(e["patience"], "training.early_stopping.patience"),
            _float(e.get("min_delta", 0.0), "training.early_stopping.min_delta"),
        ),
        train_bags=_bag_policy(b["train"], "training.bags.train"),
        eval_bags=_bag_policy(b["eval"], "training.bags.eval"),
        gradient_clip_norm=None if clip is None else _float(clip, "training.gradient_clip_norm"),
        class_weighting=weighting,
        device=_str(d.get("device", "cpu"), "training.device"),
    )


def load_experiment_config(path: Path) -> MILExperimentConfig:
    path = Path(path)
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"{path.name} cannot be read: {exc}") from None
    d = _keys(doc, path.name, {"schema", "name", "task", "dataset", "split", "model", "training"})
    if d["schema"] != CONFIG_SCHEMA:
        raise ConfigurationError(f"{path.name}: schema must be {CONFIG_SCHEMA}")
    t = _keys(d["task"], "task", {"id", "version", "type", "classes"}, {"positive_class"})
    classes = t["classes"]
    if not isinstance(classes, list) or not all(isinstance(c, str) for c in classes):
        raise ConfigurationError("task.classes must be a list of strings")
    try:
        task_type = TaskType(_str(t["type"], "task.type"))
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from None
    task = MILTask(
        TaskId(_str(t["id"], "task.id")),
        Version(_str(t["version"], "task.version")),
        task_type,
        tuple(classes),
        t.get("positive_class"),
    )
    ds = _keys(
        d["dataset"],
        "dataset",
        {"id", "version", "labels", "label_source", "label_source_version", "compatibility_policy"},
        {"accepted_compatibility_key"},
    )
    s = _keys(d["split"], "split", {"strategy"}, {"seed", "fractions", "file"})
    strategy = _str(s["strategy"], "split.strategy")
    if strategy == GENERATED_STRATEGY:
        if "file" in s:
            raise ConfigurationError(f"split.file belongs to {EXTERNAL_STRATEGY}")
        f = _keys(s.get("fractions"), "split.fractions", {"train", "validation", "test"})
        split = SplitConfig(
            strategy,
            _int(s.get("seed"), "split.seed"),
            SplitFractions(
                _float(f["train"], "split.fractions.train"),
                _float(f["validation"], "split.fractions.validation"),
                _float(f["test"], "split.fractions.test"),
            ),
            None,
        )
    elif strategy == EXTERNAL_STRATEGY:
        if "seed" in s or "fractions" in s:
            raise ConfigurationError("an external split has no seed or fractions")
        split = SplitConfig(strategy, None, None, path.parent / _str(s.get("file"), "split.file"))
    else:
        raise ConfigurationError(f"unknown split strategy {strategy!r}")
    m = _keys(d["model"], "model", {"architecture"}, {"hyperparameters"})
    hp = m.get("hyperparameters") or {}
    if not isinstance(hp, Mapping) or not all(
        isinstance(v, str | int | float | bool) for v in hp.values()
    ):
        raise ConfigurationError("model.hyperparameters must map names to scalars")
    accepted = ds.get("accepted_compatibility_key")
    return MILExperimentConfig(
        name=_str(d["name"], "name"),
        task=task,
        dataset_id=DatasetId(_str(ds["id"], "dataset.id")),
        dataset_version=Version(_str(ds["version"], "dataset.version")),
        labels=path.parent / _str(ds["labels"], "dataset.labels"),
        label_source=_str(ds["label_source"], "dataset.label_source"),
        label_source_version=_str(ds["label_source_version"], "dataset.label_source_version"),
        policy=compatibility_policy(
            _str(ds["compatibility_policy"], "dataset.compatibility_policy")
        ),
        accepted_key=None
        if accepted is None
        else _str(accepted, "dataset.accepted_compatibility_key"),
        split=split,
        architecture=_str(m["architecture"], "model.architecture"),
        hyperparameters=tuple(sorted(hp.items())),
        training=_training(d["training"]),
    )


# --------------------------------------------------------------------------- tables


def _rows(path: Path, required: tuple[str, ...]) -> list[dict[str, str]]:
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            header = tuple(reader.fieldnames or ())
            rows = list(reader)
    except OSError as exc:
        raise ConfigurationError(f"{Path(path).name} cannot be read: {exc}") from None
    missing = [c for c in required if c not in header]
    if missing:
        raise ConfigurationError(f"{Path(path).name} lacks column(s) {missing}")
    return rows


def _cell(row: Mapping[str, str], key: str) -> str | None:
    value = (row.get(key) or "").strip()
    return value or None


def load_labels(
    path: Path, task: MILTask, source: str, source_version: str
) -> tuple[tuple[SlideLabel, ...], dict[str, str]]:
    """Label records and the slide -> store mapping from a label table."""
    labels, stores = [], {}
    for n, row in enumerate(_rows(path, LABEL_COLUMNS), start=2):
        status_text = _cell(row, "status")
        try:
            status = LabelStatus(status_text or "")
        except ValueError:
            raise ConfigurationError(
                f"{Path(path).name} line {n}: status must be one of "
                f"{[s.value for s in LabelStatus]}, got {status_text!r}"
            ) from None
        slide_id = _cell(row, "slide_id") or ""
        labels.append(
            SlideLabel(
                slide_id=slide_id,
                task=task.ref,
                value=_cell(row, "label"),
                source=source,
                source_version=source_version,
                status=status,
                group_id=_cell(row, "group_id"),
                cohort=_cell(row, "cohort"),
                site=_cell(row, "site"),
                status_reason=_cell(row, "status_reason"),
            )
        )
        store_id = _cell(row, "store_id")
        if store_id is None:
            raise ConfigurationError(f"{Path(path).name} line {n}: store_id is required")
        stores[slide_id] = store_id
    return tuple(labels), stores


def load_split_table(path: Path) -> dict[str, Partition]:
    out: dict[str, Partition] = {}
    for n, row in enumerate(_rows(path, ("slide_id", "partition")), start=2):
        slide_id, text = _cell(row, "slide_id"), _cell(row, "partition")
        try:
            partition = Partition(text or "")
        except ValueError:
            raise ConfigurationError(
                f"{Path(path).name} line {n}: unknown partition {text!r}"
            ) from None
        if slide_id is None or slide_id in out:
            raise ConfigurationError(f"{Path(path).name} line {n}: missing or repeated slide id")
        out[slide_id] = partition
    return out


def write_labels(path: Path, rows: list[dict[str, str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(LABEL_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in LABEL_COLUMNS})
