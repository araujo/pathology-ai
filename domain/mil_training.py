"""MIL model, training, checkpoint and prediction contracts (SPEC-007 §7, §8, §10, §11).

* `MILModelSpec`: a framework-neutral architecture specification. It names an architecture
  and version, the accepted feature dimension, the number of outputs and sorted
  hyperparameters. Architectures are opaque names resolved by a registry. Nothing here knows a
  framework.
* `TrainingConfig`: the explicit seed, epochs, optimizer, loss, clipping, early stopping and bag
  policies.
* `ExperimentSpec`: binds task, model, training, dataset digest, split digest, compatibility key
  and runtime. Its digest is the experiment identity, so the same declared experiment always
  has the same identity.
* `CheckpointDescriptor` (`pathology-ai.mil-checkpoint/1`): the selected weights, by tensor and
  SHA-256, plus the experiment, the training history and the selected validation metric.
* `PredictionManifest` (`pathology-ai.slide-prediction/1`): one record per slide. It keeps raw
  logits and uncalibrated model scores apart. Calibrated probabilities are absent unless a
  calibration exists, which SPEC-007 does not provide.

Attention weights are model-internal pooling weights. They are recorded as research signals,
never as localisation or clinical explanation (ADR-007 D12).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from domain.errors import (
    CheckpointIncompatibleError,
    EmbeddingCompatibilityError,
    InvalidManifestError,
    ModelInputContractError,
)
from domain.hashing import digest, short_digest
from domain.ids import ImageId
from domain.metrics import DECISION_RULE, SELECTABLE_METRICS
from domain.mil import BagPolicy, BagSelection, MILTask, Partition
from domain.pipeline import ConfigValue
from domain.task import TaskRef

MIL_CHECKPOINT_SCHEMA = "pathology-ai.mil-checkpoint/1"
SLIDE_PREDICTION_SCHEMA = "pathology-ai.slide-prediction/1"
VALIDATION_LOSS = "validation_loss"
PREDICTION_NOTICE = (
    "research use only: slide-level model outputs are not diagnoses; scores are uncalibrated "
    "model outputs; attention weights are pooling weights, not validated localisation or "
    "explanation; models trained on synthetic data have no clinical meaning"
)


def _sorted_pairs(what: str, pairs: tuple[tuple[str, ConfigValue], ...]) -> None:
    keys = [k for k, _ in pairs]
    if keys != sorted(set(keys)):
        raise ModelInputContractError(f"{what} must be sorted by unique key")


# --------------------------------------------------------------------------- model


@dataclass(frozen=True, slots=True)
class MILModelSpec:
    architecture: str
    architecture_version: str
    feature_dim: int
    num_outputs: int
    hyperparameters: tuple[tuple[str, ConfigValue], ...] = ()

    def __post_init__(self) -> None:
        if not self.architecture or not self.architecture_version:
            raise ModelInputContractError("a MIL model names its architecture and version")
        for name in ("feature_dim", "num_outputs"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ModelInputContractError(f"{name} must be a positive integer, got {v!r}")
        _sorted_pairs("hyperparameters", self.hyperparameters)

    @property
    def ref(self) -> str:
        return f"{self.architecture}@{self.architecture_version}"

    def hyperparameter(self, key: str, default: ConfigValue) -> ConfigValue:
        return dict(self.hyperparameters).get(key, default)

    def require_compatible(self, other: MILModelSpec) -> None:
        if self != other:
            fields = [
                f
                for f in (
                    "architecture",
                    "architecture_version",
                    "feature_dim",
                    "num_outputs",
                    "hyperparameters",
                )
                if getattr(self, f) != getattr(other, f)
            ]
            raise CheckpointIncompatibleError(
                f"checkpoint model {other.ref} differs from {self.ref} in: {', '.join(fields)}"
            )


@dataclass(frozen=True, slots=True)
class MILArchitectureDescriptor:
    """Registry metadata of one MIL architecture, available without loading a framework.

    `hyperparameters` lists every accepted key with its default; a spec with any other key is
    rejected, so a typo never silently falls back to a default.
    """

    name: str
    version: str
    description: str
    produces_attention: bool
    hyperparameters: tuple[tuple[str, ConfigValue], ...]
    runtime_extra: str

    def __post_init__(self) -> None:
        _sorted_pairs("architecture hyperparameters", self.hyperparameters)

    def spec(
        self,
        feature_dim: int,
        task: MILTask,
        hyperparameters: tuple[tuple[str, ConfigValue], ...] = (),
    ) -> MILModelSpec:
        """A complete spec: declared values over the architecture's defaults."""
        known = dict(self.hyperparameters)
        unknown = sorted(k for k, _ in hyperparameters if k not in known)
        if unknown:
            raise ModelInputContractError(
                f"{self.name} has no hyperparameter(s) {', '.join(unknown)} "
                f"(known: {', '.join(known) or 'none'})"
            )
        merged = {**known, **dict(hyperparameters)}
        for key, value in merged.items():
            if type(value) is not type(known[key]) and not (
                isinstance(known[key], float) and type(value) is int
            ):
                raise ModelInputContractError(
                    f"{self.name} hyperparameter {key} must be {type(known[key]).__name__}"
                )
        return MILModelSpec(
            self.name,
            self.version,
            feature_dim,
            task.num_outputs,
            tuple(
                sorted(
                    (k, float(v) if isinstance(known[k], float) else v) for k, v in merged.items()
                )
            ),
        )


class LossKind(StrEnum):
    BCE_WITH_LOGITS = "bce-with-logits"
    CROSS_ENTROPY = "cross-entropy"


def loss_for(task: MILTask) -> LossKind:
    return LossKind.BCE_WITH_LOGITS if task.binary else LossKind.CROSS_ENTROPY


# --------------------------------------------------------------------------- training


class OptimizerName(StrEnum):
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    name: OptimizerName
    learning_rate: float
    weight_decay: float = 0.0
    momentum: float = 0.0

    def __post_init__(self) -> None:
        for field in ("learning_rate", "weight_decay", "momentum"):
            v = getattr(self, field)
            if isinstance(v, bool) or not math.isfinite(float(v)) or float(v) < 0:
                raise ModelInputContractError(f"{field} must be a finite non-negative number")
        if self.learning_rate <= 0:
            raise ModelInputContractError("learning_rate must be positive")


class SelectionMode(StrEnum):
    MIN = "min"
    MAX = "max"


@dataclass(frozen=True, slots=True)
class EarlyStopping:
    """Model selection and early stopping, on the validation partition only (SPEC-007 P8)."""

    metric: str = VALIDATION_LOSS
    mode: SelectionMode = SelectionMode.MIN
    patience: int = 10
    min_delta: float = 0.0

    def __post_init__(self) -> None:
        if self.metric != VALIDATION_LOSS and self.metric not in SELECTABLE_METRICS:
            known = ", ".join((VALIDATION_LOSS, *SELECTABLE_METRICS))
            raise ModelInputContractError(f"unknown selection metric {self.metric!r} ({known})")
        if self.metric == VALIDATION_LOSS and self.mode is not SelectionMode.MIN:
            raise ModelInputContractError("validation_loss is minimized")
        if isinstance(self.patience, bool) or self.patience < 1:
            raise ModelInputContractError("patience must be a positive integer")
        if not math.isfinite(self.min_delta) or self.min_delta < 0:
            raise ModelInputContractError("min_delta must be finite and non-negative")

    def improves(self, value: float, best: float | None) -> bool:
        if best is None:
            return True
        if self.mode is SelectionMode.MIN:
            return value < best - self.min_delta
        return value > best + self.min_delta


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    seed: int
    epochs: int
    batch_slides: int
    optimizer: OptimizerConfig
    early_stopping: EarlyStopping
    train_bags: BagPolicy
    eval_bags: BagPolicy
    gradient_clip_norm: float | None = None
    class_weighting: bool = False
    device: str = "cpu"

    def __post_init__(self) -> None:
        for name in ("epochs", "batch_slides"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ModelInputContractError(f"{name} must be a positive integer, got {v!r}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ModelInputContractError("seed must be a non-negative integer")
        if self.gradient_clip_norm is not None and not (
            math.isfinite(self.gradient_clip_norm) and self.gradient_clip_norm > 0
        ):
            raise ModelInputContractError("gradient_clip_norm must be positive when set")
        if not self.device:
            raise ModelInputContractError("device must be named explicitly")


# --------------------------------------------------------------------------- experiment


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    """Everything a training run depends on. Paths and timestamps never enter it."""

    name: str
    task: MILTask
    model: MILModelSpec
    training: TrainingConfig
    dataset_ref: str
    dataset_digest: str
    split_digest: str
    compatibility_policy: str
    compatibility_key: str
    loss: LossKind
    runtime: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if self.model.num_outputs != self.task.num_outputs:
            raise ModelInputContractError(
                f"the model has {self.model.num_outputs} outputs; task "
                f"{self.task.task_id} needs {self.task.num_outputs}"
            )
        if self.loss is not loss_for(self.task):
            raise ModelInputContractError(f"loss {self.loss} does not fit the task type")
        _sorted_pairs("runtime", tuple((k, v) for k, v in self.runtime))

    @property
    def identity(self) -> str:
        return short_digest(self, "milexp")


@dataclass(frozen=True, slots=True)
class EpochRecord:
    epoch: int
    train_loss: float
    validation_loss: float
    selection_value: float | None
    selection_reason: str | None
    improved: bool


@dataclass(frozen=True, slots=True)
class TrainingHistory:
    epochs: tuple[EpochRecord, ...]
    best_epoch: int
    best_value: float
    selection_metric: str
    stopped_early: bool


# --------------------------------------------------------------------------- checkpoint


@dataclass(frozen=True, slots=True)
class TensorEntry:
    """One tensor of a pickle-free checkpoint: little-endian, C-ordered bytes at an offset."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    offset: int
    nbytes: int
    sha256: str


_ITEMSIZE = {"float16": 2, "float32": 4, "float64": 8, "int32": 4, "int64": 8, "bool": 1}


def index_tensors(
    tensors: Sequence[tuple[str, str, tuple[int, ...], bytes]],
) -> tuple[tuple[TensorEntry, ...], str, int]:
    """Entries, whole-payload SHA-256 and size of (name, dtype, shape, bytes) tensors laid out
    by name. Sizes must match dtype and shape exactly."""
    ordered = sorted(tensors, key=lambda t: t[0])
    names = [t[0] for t in ordered]
    if names != sorted(set(names)) or not names:
        raise CheckpointIncompatibleError("a model state needs unique tensor names")
    entries, offset, whole = [], 0, hashlib.sha256()
    for name, dtype, shape, data in ordered:
        count = 1
        for d in shape:
            count *= d
        if dtype not in _ITEMSIZE or len(data) != _ITEMSIZE[dtype] * count:
            raise CheckpointIncompatibleError(f"tensor {name}: {dtype} {shape} size mismatch")
        entries.append(
            TensorEntry(name, dtype, shape, offset, len(data), hashlib.sha256(data).hexdigest())
        )
        whole.update(data)
        offset += len(data)
    return tuple(entries), whole.hexdigest(), offset


@dataclass(frozen=True, slots=True)
class CheckpointDescriptor:
    experiment: ExperimentSpec
    experiment_identity: str
    tensors: tuple[TensorEntry, ...]
    weights_sha256: str
    weights_bytes: int
    history: TrainingHistory
    parameter_count: int
    limitations: tuple[str, ...]
    schema_version: str = MIL_CHECKPOINT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != MIL_CHECKPOINT_SCHEMA:
            raise CheckpointIncompatibleError(
                f"unsupported checkpoint schema {self.schema_version!r}"
            )
        if self.experiment_identity != self.experiment.identity:
            raise CheckpointIncompatibleError("checkpoint experiment identity does not re-derive")
        names = [t.name for t in self.tensors]
        if names != sorted(set(names)) or not names:
            raise CheckpointIncompatibleError("checkpoint tensors must be sorted and unique")
        offset = 0
        for t in self.tensors:
            if t.offset != offset or t.nbytes < 0:
                raise CheckpointIncompatibleError(f"tensor {t.name} is not contiguous in the file")
            offset += t.nbytes
        if offset != self.weights_bytes:
            raise CheckpointIncompatibleError("tensor sizes do not add up to the weights file")

    @property
    def digest(self) -> str:
        return digest(self)

    @property
    def checkpoint_id(self) -> str:
        return short_digest(self, "milckpt")

    def require_accepts(self, compatibility_key: str, feature_dim: int, task: MILTask) -> None:
        """Fail closed unless bags and task match what the model was trained on."""
        exp = self.experiment
        if compatibility_key != exp.compatibility_key:
            raise EmbeddingCompatibilityError(
                f"the model was trained on embeddings {exp.compatibility_key}; these bags are "
                f"{compatibility_key}"
            )
        if feature_dim != exp.model.feature_dim:
            raise CheckpointIncompatibleError(
                f"the model accepts {exp.model.feature_dim} features, the bags have {feature_dim}"
            )
        if task != exp.task:
            raise CheckpointIncompatibleError(
                f"the model predicts {exp.task.task_id}@{exp.task.version} "
                f"{list(exp.task.classes)}, not {task.task_id}@{task.version} {list(task.classes)}"
            )


# --------------------------------------------------------------------------- prediction


@dataclass(frozen=True, slots=True)
class SelectionSummary:
    original_count: int
    selected_count: int
    algorithm: str
    seed: int
    epoch: int | None
    indices_digest: str

    @staticmethod
    def of(selection: BagSelection) -> SelectionSummary:
        return SelectionSummary(
            selection.original_count,
            selection.selected_count,
            selection.algorithm,
            selection.seed,
            selection.epoch,
            selection.digest,
        )


@dataclass(frozen=True, slots=True)
class AttentionSummary:
    """A reference to model attention weights; not a clinical explanation (ADR-007 D12)."""

    semantics: str
    instances: int
    weights_digest: str
    top_patches: tuple[tuple[str, float], ...]
    notice: str = "attention weights are pooling weights, not validated localisation"


class ScoreKind(StrEnum):
    SIGMOID = "sigmoid"  # binary: sigmoid of the one logit, scores = (1 - s, s)
    SOFTMAX = "softmax"


@dataclass(frozen=True, slots=True)
class SlidePrediction:
    slide_id: str
    image_id: ImageId
    predicted_class: str
    logits: tuple[float, ...]
    scores: tuple[tuple[str, float], ...]
    score_kind: ScoreKind
    store_id: str
    embedding_manifest_digest: str
    embedding_identity: str
    selection: SelectionSummary
    attention: AttentionSummary | None = None
    scores_calibrated: bool = False
    calibrated_probabilities: tuple[tuple[str, float], ...] | None = None

    def __post_init__(self) -> None:
        names = [n for n, _ in self.scores]
        if self.predicted_class not in names:
            raise InvalidManifestError("the predicted class must be one of the scored classes")
        if not all(math.isfinite(v) for v in self.logits) or not all(
            math.isfinite(v) for _, v in self.scores
        ):
            raise InvalidManifestError("logits and scores must be finite")
        if self.scores_calibrated or self.calibrated_probabilities is not None:
            raise InvalidManifestError(
                "no calibration exists in SPEC-007; scores are uncalibrated model outputs"
            )

    @property
    def digest(self) -> str:
        return digest(self)


class PredictionPurpose(StrEnum):
    EVALUATION = "evaluation"  # a labelled partition of the experiment's own split
    INFERENCE = "inference"  # any set of compatible bags; labels are not used


@dataclass(frozen=True, slots=True)
class PredictionManifest:
    experiment_identity: str
    checkpoint_id: str
    checkpoint_digest: str
    model: MILModelSpec
    task: TaskRef
    classes: tuple[str, ...]
    positive_class: str | None
    decision_rule: str
    purpose: PredictionPurpose
    partition: Partition | None
    dataset_ref: str | None
    dataset_digest: str | None
    split_digest: str | None
    compatibility_key: str
    bag_policy: BagPolicy
    policy_overridden: bool
    device: str
    predictions: tuple[SlidePrediction, ...]
    limitations: tuple[str, ...] = ()
    schema_version: str = SLIDE_PREDICTION_SCHEMA
    notice: str = PREDICTION_NOTICE

    def __post_init__(self) -> None:
        if self.schema_version != SLIDE_PREDICTION_SCHEMA:
            raise InvalidManifestError(f"unsupported prediction schema {self.schema_version!r}")
        if self.decision_rule != DECISION_RULE:
            raise InvalidManifestError(f"unknown decision rule {self.decision_rule!r}")
        ids = [p.slide_id for p in self.predictions]
        if ids != sorted(set(ids)) or not ids:
            raise InvalidManifestError("predictions must be sorted by unique slide id")
        if (self.purpose is PredictionPurpose.EVALUATION) != (self.partition is not None):
            raise InvalidManifestError("evaluation predictions name their partition")

    @property
    def content_digest(self) -> str:
        return digest(self)
