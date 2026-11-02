"""Supervised MIL training, evaluation and slide-level prediction (SPEC-007 §8–§11, ADR-007 D8).

This is a dataset-level iterative workflow, not a pipeline: it does not run stages and does not
replace `PipelineRunner`. It orchestrates ports only (bag source, batch builder, MIL model
factory, checkpoint store) and never touches arrays or framework types itself.

Test isolation (SPEC-007 P8, ADR-007 D9) is structural:

* `train` reads bags through a guard that admits only the train and validation stores. Any
  other read raises `TEST_ISOLATION`.
* Model selection and early stopping use validation results only.
* The selected checkpoint is persisted before `train` returns. `evaluate` and `predict` load
  it from the checkpoint store, so test results exist only after selection is complete.
* There is no threshold tuning. The decision rule is the argmax of the model scores.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass

from application.mil_dataset import MILDatasetService
from application.registries import MILArchitectureRegistry
from domain.errors import (
    BagBudgetError,
    EmbeddingCompatibilityError,
    InvalidLabelError,
    MILDatasetError,
    ModelSelectionError,
    SplitError,
    TestIsolationError,
)
from domain.hashing import digest
from domain.metrics import DECISION_RULE, ClassificationReport, argmax, classification_report
from domain.mil import (
    BagPolicy,
    EmbeddingBagInfo,
    EmbeddingCompatibility,
    MILDatasetManifest,
    Partition,
    SplitManifest,
    batch_order,
    compatibility_policy,
    validate_split,
)
from domain.mil_training import (
    VALIDATION_LOSS,
    AttentionSummary,
    CheckpointDescriptor,
    EpochRecord,
    ExperimentSpec,
    PredictionManifest,
    PredictionPurpose,
    ScoreKind,
    SelectionSummary,
    SlidePrediction,
    TrainingConfig,
    TrainingHistory,
    index_tensors,
    loss_for,
)
from ports import (
    EmbeddingBagSource,
    MILBatch,
    MILBatchBuilder,
    MILCheckpointStore,
    MILModel,
    MILOutput,
    ModelTensor,
)

TOP_ATTENTION = 5
# Evaluation and inference run on CPU, where the reproducibility contract is exact.
INFERENCE_DEVICE = "cpu"
CHECKPOINT_LIMITATIONS = (
    "trained model artifact for research use; not a validated diagnostic or biomarker model",
    "selected on the validation partition only; the test partition did not influence training",
    "scores are uncalibrated model outputs",
)


@dataclass(frozen=True, slots=True)
class RunMeasurements:
    """Volatile measurements of one run (never part of an identity or a manifest)."""

    seconds: float
    bags_loaded: int
    instances_loaded: int
    max_padded_batch_bytes: int


@dataclass(frozen=True, slots=True)
class TrainingOutcome:
    checkpoint: CheckpointDescriptor
    reused: bool
    measurements: RunMeasurements | None


@dataclass(frozen=True, slots=True)
class EvaluationOutcome:
    report: ClassificationReport
    predictions: PredictionManifest
    loss: float
    measurements: RunMeasurements


class _PartitionGuard:
    """Admits reads of the given stores only (structural test-set isolation)."""

    def __init__(self, inner: EmbeddingBagSource, allowed: frozenset[str], what: str) -> None:
        self._inner, self._allowed, self._what = inner, allowed, what

    def _check(self, store_id: str) -> None:
        if store_id not in self._allowed:
            raise TestIsolationError(
                f"store {store_id} is outside the {self._what} partitions; it cannot be read "
                "before model selection is complete"
            )

    def describe(self, store_id: str) -> EmbeddingBagInfo:
        self._check(store_id)
        return self._inner.describe(store_id)

    def read(self, store_id: str, indices: tuple[int, ...]) -> object:
        self._check(store_id)
        return self._inner.read(store_id, indices)


class _Loader:
    """Selects, reads and pads bags in bounded batches; counts what it loads."""

    def __init__(self, bags: EmbeddingBagSource, builder: MILBatchBuilder, dim: int) -> None:
        self._bags, self._builder, self._dim = bags, builder, dim
        self.bags_loaded = 0
        self.instances_loaded = 0
        self.max_padded = 0

    def batch(
        self,
        items: Sequence[tuple[str, str, int | None]],
        policy: BagPolicy,
        epoch: int | None,
    ) -> MILBatch:
        """`items`: (slide id, store id, target index or None)."""
        loaded = []
        for slide_id, store_id, _ in items:
            info = self._bags.describe(store_id)
            selection = policy.select(slide_id, info.rows, epoch)
            loaded.append((selection, self._bags.read(store_id, selection.indices)))
            self.instances_loaded += selection.selected_count
        targets = tuple(t for _, _, t in items if t is not None)
        batch = self._builder.build(
            loaded, targets if len(targets) == len(items) else None, self._dim
        )
        if batch.padded_bytes > policy.max_batch_bytes:
            raise BagBudgetError(
                f"a padded batch needs {batch.padded_bytes} bytes, above max_batch_bytes="
                f"{policy.max_batch_bytes}"
            )
        self.bags_loaded += len(items)
        self.max_padded = max(self.max_padded, batch.padded_bytes)
        return batch

    def measurements(self, started: float) -> RunMeasurements:
        return RunMeasurements(
            round(time.perf_counter() - started, 6),
            self.bags_loaded,
            self.instances_loaded,
            self.max_padded,
        )


def _chunks[T](items: Sequence[T], size: int) -> list[Sequence[T]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _softmax(xs: Sequence[float]) -> tuple[float, ...]:
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    s = sum(e)
    return tuple(v / s for v in e)


def class_scores(binary: bool, logits: Sequence[float]) -> tuple[float, ...]:
    """Uncalibrated per-class model scores: sigmoid (binary, one logit) or softmax."""
    if binary:
        p = _sigmoid(logits[0])
        return (1.0 - p, p)
    return _softmax(logits)


class MILTrainingService:
    def __init__(
        self,
        *,
        bags: EmbeddingBagSource,
        batches: MILBatchBuilder,
        architectures: MILArchitectureRegistry,
        checkpoints: MILCheckpointStore,
    ) -> None:
        self._bags = bags
        self._batches = batches
        self._architectures = architectures
        self._checkpoints = checkpoints

    @property
    def checkpoints(self) -> MILCheckpointStore:
        return self._checkpoints

    # ------------------------------------------------------------------ experiment

    def experiment(
        self,
        *,
        name: str,
        dataset: MILDatasetManifest,
        split: SplitManifest,
        architecture: str,
        hyperparameters: tuple[tuple[str, str | int | float | bool], ...],
        training: TrainingConfig,
        accepted_key: str | None = None,
    ) -> ExperimentSpec:
        """Bind an experiment to one dataset, one split and one embedding identity."""
        validate_split(split, dataset)
        if accepted_key is not None and accepted_key != dataset.compatibility_key:
            raise EmbeddingCompatibilityError(
                f"the experiment accepts embeddings {accepted_key}; dataset {dataset.ref} holds "
                f"{dataset.compatibility_key}"
            )
        factory = self._architectures.resolve(architecture)
        spec = factory.descriptor.spec(dataset.feature_dim, dataset.task, hyperparameters)
        return ExperimentSpec(
            name=name,
            task=dataset.task,
            model=spec,
            training=training,
            dataset_ref=dataset.ref,
            dataset_digest=dataset.digest,
            split_digest=split.digest,
            compatibility_policy=dataset.compatibility_policy,
            compatibility_key=dataset.compatibility_key,
            loss=loss_for(dataset.task),
            runtime=factory.runtime(training.device),
        )

    def _bind(
        self, experiment: ExperimentSpec, dataset: MILDatasetManifest, split: SplitManifest
    ) -> None:
        validate_split(split, dataset)
        if (experiment.dataset_digest, experiment.split_digest) != (dataset.digest, split.digest):
            raise SplitError(
                "the experiment was declared for another dataset or split; splits are never "
                "regenerated or substituted"
            )

    def _items(
        self, dataset: MILDatasetManifest, slide_ids: Sequence[str]
    ) -> list[tuple[str, str, int | None]]:
        out: list[tuple[str, str, int | None]] = []
        for s in slide_ids:
            r = dataset.record(s)
            if r.label is None:
                raise InvalidLabelError(f"slide {s} has no label")
            out.append((s, r.store_id, dataset.task.index(r.label)))
        return out

    # ------------------------------------------------------------------ training

    def train(
        self, experiment: ExperimentSpec, dataset: MILDatasetManifest, split: SplitManifest
    ) -> TrainingOutcome:
        self._bind(experiment, dataset, split)
        existing = self._checkpoints.find(experiment.identity)
        if existing is not None:
            return TrainingOutcome(existing, reused=True, measurements=None)
        started = time.perf_counter()
        cfg = experiment.training
        train_ids = split.members(Partition.TRAIN)
        val_ids = split.members(Partition.VALIDATION)
        allowed = frozenset(dataset.record(s).store_id for s in (*train_ids, *val_ids))
        guarded = _PartitionGuard(self._bags, allowed, "train/validation")
        MILDatasetService(guarded).verify(dataset, (*train_ids, *val_ids))
        largest = max(dataset.record(s).patch_count for s in (*train_ids, *val_ids))
        cfg.train_bags.check_batch(cfg.batch_slides, largest, dataset.feature_dim)
        cfg.eval_bags.check_batch(cfg.batch_slides, largest, dataset.feature_dim)
        train_items = {
            s: item for s, item in zip(train_ids, self._items(dataset, train_ids), strict=True)
        }
        val_items = self._items(dataset, val_ids)
        weights = self._class_weights(dataset, train_items) if cfg.class_weighting else None
        factory = self._architectures.resolve(experiment.model.architecture)
        model = factory.create(
            experiment.model,
            loss=experiment.loss,
            seed=cfg.seed,
            training=cfg,
            class_weights=weights,
            device=cfg.device,
        )
        parameters = model.parameter_count
        loader = _Loader(guarded, self._batches, dataset.feature_dim)
        stopping = cfg.early_stopping
        records: list[EpochRecord] = []
        best: tuple[int, float, tuple[ModelTensor, ...]] | None = None
        stale = 0
        stopped_early = False
        try:
            for epoch in range(cfg.epochs):
                total = 0.0
                order = batch_order(train_ids, cfg.seed, epoch)
                for chunk in _chunks(order, cfg.batch_slides):
                    batch = loader.batch([train_items[s] for s in chunk], cfg.train_bags, epoch)
                    total += model.train_batch(batch) * len(chunk)
                val_loss, report = self._evaluate_items(model, loader, val_items, dataset, cfg)
                if stopping.metric == VALIDATION_LOSS:
                    value: float | None = val_loss
                    reason = None
                else:
                    m = report.metric(stopping.metric)
                    value, reason = m.value, m.reason
                improved = value is not None and stopping.improves(
                    value, None if best is None else best[1]
                )
                records.append(
                    EpochRecord(epoch, total / len(train_ids), val_loss, value, reason, improved)
                )
                if improved and value is not None:
                    best = (epoch, value, model.state())
                    stale = 0
                else:
                    stale += 1
                    if stale >= stopping.patience:
                        stopped_early = epoch + 1 < cfg.epochs
                        break
        finally:
            model.close()
        if best is None:
            raise ModelSelectionError(
                f"no epoch produced a defined validation {stopping.metric}; "
                f"last reason: {records[-1].selection_reason if records else 'no epoch ran'}"
            )
        epoch, value, state = best
        entries, sha, size = index_tensors([(t.name, t.dtype, t.shape, t.data) for t in state])
        descriptor = CheckpointDescriptor(
            experiment=experiment,
            experiment_identity=experiment.identity,
            tensors=entries,
            weights_sha256=sha,
            weights_bytes=size,
            history=TrainingHistory(tuple(records), epoch, value, stopping.metric, stopped_early),
            parameter_count=parameters,
            limitations=CHECKPOINT_LIMITATIONS,
        )
        self._checkpoints.save(descriptor, state)
        return TrainingOutcome(descriptor, reused=False, measurements=loader.measurements(started))

    def _class_weights(
        self, dataset: MILDatasetManifest, items: dict[str, tuple[str, str, int | None]]
    ) -> tuple[float, ...]:
        k = len(dataset.task.classes)
        counts = [0] * k
        for _, _, t in items.values():
            assert t is not None
            counts[t] += 1
        empty = [dataset.task.classes[c] for c in range(k) if counts[c] == 0]
        if empty:
            raise MILDatasetError(
                f"class weighting needs training slides of every class; none for {empty}"
            )
        n = sum(counts)
        return tuple(n / (k * c) for c in counts)

    def _evaluate_items(
        self,
        model: MILModel,
        loader: _Loader,
        items: Sequence[tuple[str, str, int | None]],
        dataset: MILDatasetManifest,
        cfg: TrainingConfig,
    ) -> tuple[float, ClassificationReport]:
        total = 0.0
        truth: list[int] = []
        scores: list[tuple[float, ...]] = []
        for chunk in _chunks(items, cfg.batch_slides):
            batch = loader.batch(chunk, cfg.eval_bags, None)
            loss, out = model.evaluate_batch(batch)
            total += loss * len(chunk)
            truth.extend(t for _, _, t in chunk if t is not None)
            scores.extend(class_scores(dataset.task.binary, row) for row in out.logits)
        return total / len(items), classification_report(dataset.task, truth, scores)

    # ------------------------------------------------------------------ evaluation / prediction

    def _load(
        self, experiment_identity: str
    ) -> tuple[CheckpointDescriptor, MILModel, tuple[str, ...]]:
        """The verified checkpoint, its model on CPU, and runtime limitations of this use."""
        descriptor, tensors = self._checkpoints.load(experiment_identity)
        exp = descriptor.experiment
        factory = self._architectures.resolve(exp.model.architecture)
        exp.model.require_compatible(
            factory.descriptor.spec(exp.model.feature_dim, exp.task, exp.model.hyperparameters)
        )
        runtime = factory.runtime(INFERENCE_DEVICE)
        notes = descriptor.limitations
        if runtime != exp.runtime:
            notes += (
                f"inference runtime {dict(runtime)} differs from the training runtime "
                f"{dict(exp.runtime)}; outputs are reproducible only within numerical tolerance",
            )
        model = factory.create(
            exp.model, loss=exp.loss, seed=exp.training.seed, training=None, device=INFERENCE_DEVICE
        )
        try:
            model.load_state(tensors)
        except Exception:
            model.close()
            raise
        return descriptor, model, notes

    def evaluate(
        self,
        experiment_identity: str,
        dataset: MILDatasetManifest,
        split: SplitManifest,
        partition: Partition,
    ) -> EvaluationOutcome:
        """Metrics and predictions of a persisted checkpoint on one partition of its own split."""
        started = time.perf_counter()
        descriptor, model, notes = self._load(experiment_identity)
        exp = descriptor.experiment
        try:
            self._bind(exp, dataset, split)
            descriptor.require_accepts(dataset.compatibility_key, dataset.feature_dim, dataset.task)
            ids = split.members(partition)
            if not ids:
                raise SplitError(f"the {partition} partition is empty")
            MILDatasetService(self._bags).verify(dataset, ids)
            items = self._items(dataset, ids)
            largest = max(dataset.record(s).patch_count for s in ids)
            policy = exp.training.eval_bags
            policy.check_batch(exp.training.batch_slides, largest, dataset.feature_dim)
            loader = _Loader(self._bags, self._batches, dataset.feature_dim)
            total = 0.0
            truth: list[int] = []
            scores: list[tuple[float, ...]] = []
            predictions: list[SlidePrediction] = []
            for chunk in _chunks(items, exp.training.batch_slides):
                batch = loader.batch(chunk, policy, None)
                loss, out = model.evaluate_batch(batch)
                total += loss * len(chunk)
                truth.extend(t for _, _, t in chunk if t is not None)
                for b, slide_id in enumerate(batch.slide_ids):
                    record = dataset.record(slide_id)
                    info = self._bags.describe(record.store_id)
                    p = self._prediction(exp, info, slide_id, batch, out, b)
                    scores.append(tuple(v for _, v in p.scores))
                    predictions.append(p)
        finally:
            model.close()
        manifest = PredictionManifest(
            experiment_identity=exp.identity,
            checkpoint_id=descriptor.checkpoint_id,
            checkpoint_digest=descriptor.digest,
            model=exp.model,
            task=exp.task.ref,
            classes=exp.task.classes,
            positive_class=exp.task.positive_class,
            decision_rule=DECISION_RULE,
            purpose=PredictionPurpose.EVALUATION,
            partition=partition,
            dataset_ref=dataset.ref,
            dataset_digest=dataset.digest,
            split_digest=split.digest,
            compatibility_key=dataset.compatibility_key,
            bag_policy=policy,
            policy_overridden=False,
            device=INFERENCE_DEVICE,
            predictions=tuple(sorted(predictions, key=lambda p: p.slide_id)),
            limitations=notes,
        )
        report = classification_report(exp.task, truth, scores)
        return EvaluationOutcome(report, manifest, total / len(items), loader.measurements(started))

    def predict(
        self,
        experiment_identity: str,
        slides: Sequence[tuple[str, str]],
        *,
        policy: BagPolicy | None = None,
    ) -> PredictionManifest:
        """Slide-level inference on any finalized bags (`(slide id, store id)` pairs).

        Every bag must have the compatibility key the model was trained on. The bag policy is the
        checkpoint's evaluation policy unless `policy` overrides it explicitly (recorded).
        """
        descriptor, model, notes = self._load(experiment_identity)
        exp = descriptor.experiment
        ids = [s for s, _ in slides]
        if not slides or len(set(ids)) != len(ids):
            model.close()
            raise MILDatasetError("prediction needs unique slide ids")
        chosen = policy or exp.training.eval_bags
        compat = compatibility_policy(exp.compatibility_policy)
        try:
            infos = {}
            for slide_id, store_id in slides:
                info = self._bags.describe(store_id)
                key = EmbeddingCompatibility.of(info.run, compat).key
                descriptor.require_accepts(key, info.dim, exp.task)
                infos[slide_id] = info
            largest = max(i.rows for i in infos.values())
            chosen.check_batch(exp.training.batch_slides, largest, exp.model.feature_dim)
            loader = _Loader(self._bags, self._batches, exp.model.feature_dim)
            predictions = []
            items: list[tuple[str, str, int | None]] = [(s, st, None) for s, st in sorted(slides)]
            for chunk in _chunks(items, exp.training.batch_slides):
                batch = loader.batch(chunk, chosen, None)
                out = model.predict(batch)
                for b, slide_id in enumerate(batch.slide_ids):
                    predictions.append(
                        self._prediction(exp, infos[slide_id], slide_id, batch, out, b)
                    )
        finally:
            model.close()
        return PredictionManifest(
            experiment_identity=exp.identity,
            checkpoint_id=descriptor.checkpoint_id,
            checkpoint_digest=descriptor.digest,
            model=exp.model,
            task=exp.task.ref,
            classes=exp.task.classes,
            positive_class=exp.task.positive_class,
            decision_rule=DECISION_RULE,
            purpose=PredictionPurpose.INFERENCE,
            partition=None,
            dataset_ref=None,
            dataset_digest=None,
            split_digest=None,
            compatibility_key=exp.compatibility_key,
            bag_policy=chosen,
            policy_overridden=policy is not None and policy != exp.training.eval_bags,
            device=INFERENCE_DEVICE,
            predictions=tuple(predictions),
            limitations=notes,
        )

    def _prediction(
        self,
        exp: ExperimentSpec,
        info: EmbeddingBagInfo,
        slide_id: str,
        batch: MILBatch,
        out: MILOutput,
        b: int,
    ) -> SlidePrediction:
        logits = out.logits[b]
        scores = class_scores(exp.task.binary, logits)
        selection = batch.selections[b]
        attention = None
        if out.attention is not None:
            weights = out.attention[b]
            top = sorted(range(len(weights)), key=lambda j: (-weights[j], j))[:TOP_ATTENTION]
            attention = AttentionSummary(
                semantics="attention_weight",
                instances=len(weights),
                weights_digest=digest(weights),
                top_patches=tuple(
                    (info.records[selection.indices[j]].patch_id, weights[j]) for j in top
                ),
            )
        return SlidePrediction(
            slide_id=slide_id,
            image_id=info.image_id,
            predicted_class=exp.task.classes[argmax(scores)],
            logits=tuple(logits),
            scores=tuple(zip(exp.task.classes, scores, strict=True)),
            score_kind=ScoreKind.SIGMOID if exp.task.binary else ScoreKind.SOFTMAX,
            store_id=info.store_id,
            embedding_manifest_digest=info.manifest_digest,
            embedding_identity=info.embedding_identity,
            selection=SelectionSummary.of(selection),
            attention=attention,
        )
