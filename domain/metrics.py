"""Slide-level classification metrics (SPEC-007 §9). Pure Python; one value per slide.

A metric that is not defined for the given data is a `MetricValue` with `value=None` and an
explicit `reason`. It is never replaced by zero or NaN.

Definitions (`y` true class indices, `p` per-class scores, `ŷ` predicted indices):

* confusion matrix: rows are true classes, columns predicted classes, in vocabulary order;
* accuracy: fraction of slides with `ŷ == y`;
* balanced accuracy: mean recall over the classes that have at least one true slide; undefined
  with fewer than two such classes;
* precision / recall / F1 of a class: undefined when nothing is predicted as (precision), or
  truly is (recall), that class; F1 is undefined when precision or recall is, and 0 when both
  are 0;
* binary precision, recall, F1, AUROC and AUPRC refer to the positive class (second class);
* macro precision / recall / F1 / AUROC: unweighted mean over all classes, undefined if any
  class's value is undefined (the classes are named in the reason);
* AUROC: the Mann-Whitney probability that a random positive outranks a random negative, ties
  counting one half; undefined without both a positive and a negative slide;
* AUPRC: average precision, `sum_n (R_n - R_{n-1}) P_n` over distinct score thresholds in
  decreasing order; undefined without a positive slide.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from domain.errors import InvalidLabelError
from domain.mil import MILTask

DECISION_RULE = "argmax"


@dataclass(frozen=True, slots=True)
class MetricValue:
    value: float | None
    reason: str | None = None

    def __post_init__(self) -> None:
        if (self.value is None) == (self.reason is None):
            raise ValueError("a metric has either a value or a reason it is undefined")
        if self.value is not None and not math.isfinite(self.value):
            raise ValueError("a metric value must be finite")

    @property
    def defined(self) -> bool:
        return self.value is not None


def _ok(value: float) -> MetricValue:
    return MetricValue(float(value))


def _undefined(reason: str) -> MetricValue:
    return MetricValue(None, reason)


@dataclass(frozen=True, slots=True)
class ClassMetrics:
    class_name: str
    support: int
    predicted: int
    precision: MetricValue
    recall: MetricValue
    f1: MetricValue
    auroc: MetricValue


@dataclass(frozen=True, slots=True)
class ClassificationReport:
    """Slide-level metrics of one partition. `summary` names every reported metric."""

    classes: tuple[str, ...]
    binary: bool
    slides: int
    decision_rule: str
    confusion_matrix: tuple[tuple[int, ...], ...]
    accuracy: MetricValue
    balanced_accuracy: MetricValue
    precision: MetricValue
    recall: MetricValue
    f1: MetricValue
    auroc: MetricValue
    auprc: MetricValue
    per_class: tuple[ClassMetrics, ...]

    def metric(self, name: str) -> MetricValue:
        if name not in SELECTABLE_METRICS:
            raise KeyError(f"unknown metric {name!r} (known: {', '.join(SELECTABLE_METRICS)})")
        value: MetricValue = getattr(self, name)
        return value


SELECTABLE_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "auroc",
    "auprc",
)


def auroc(labels: Sequence[bool], scores: Sequence[float]) -> MetricValue:
    pos = [s for s, y in zip(scores, labels, strict=True) if y]
    neg = [s for s, y in zip(scores, labels, strict=True) if not y]
    if not pos or not neg:
        return _undefined(
            f"AUROC needs positive and negative slides ({len(pos)} positive, {len(neg)} negative)"
        )
    # Average ranks with ties, then the Mann-Whitney U statistic.
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    rank_sum = sum(r for r, y in zip(ranks, labels, strict=True) if y)
    u = rank_sum - len(pos) * (len(pos) + 1) / 2.0
    return _ok(u / (len(pos) * len(neg)))


def average_precision(labels: Sequence[bool], scores: Sequence[float]) -> MetricValue:
    positives = sum(labels)
    if positives == 0:
        return _undefined("AUPRC needs at least one positive slide")
    pairs = sorted(zip(scores, labels, strict=True), key=lambda t: -t[0])
    ap, tp, fp, previous_recall = 0.0, 0, 0, 0.0
    i = 0
    while i < len(pairs):
        threshold = pairs[i][0]
        while i < len(pairs) and pairs[i][0] == threshold:
            tp += pairs[i][1]
            fp += not pairs[i][1]
            i += 1
        recall = tp / positives
        ap += (recall - previous_recall) * (tp / (tp + fp))
        previous_recall = recall
    return _ok(ap)


def _prf(tp: int, predicted: int, support: int, name: str) -> tuple[MetricValue, ...]:
    precision = (
        _ok(tp / predicted) if predicted else _undefined(f"no slide was predicted as {name!r}")
    )
    recall = _ok(tp / support) if support else _undefined(f"no slide is truly {name!r}")
    if precision.value is None or recall.value is None:
        f1 = _undefined(f"F1 of {name!r} needs defined precision and recall")
    elif precision.value + recall.value == 0:
        f1 = _ok(0.0)
    else:
        f1 = _ok(2 * precision.value * recall.value / (precision.value + recall.value))
    return precision, recall, f1


def _macro(values: Sequence[MetricValue], classes: Sequence[str], what: str) -> MetricValue:
    undefined = [c for c, v in zip(classes, values, strict=True) if v.value is None]
    if undefined:
        return _undefined(f"macro {what} is undefined for class(es) {', '.join(undefined)}")
    return _ok(sum(v.value for v in values if v.value is not None) / len(values))


def argmax(row: Sequence[float]) -> int:
    """Index of the largest score; the first class wins ties."""
    best = 0
    for i, v in enumerate(row):
        if v > row[best]:
            best = i
    return best


def classification_report(
    task: MILTask,
    truth: Sequence[int],
    scores: Sequence[Sequence[float]],
    predicted: Sequence[int] | None = None,
) -> ClassificationReport:
    """Metrics from true class indices and per-class scores (probabilities, one row per slide).

    The prediction is the argmax of the scores (first class on ties) unless given explicitly.
    """
    k = len(task.classes)
    if len(truth) != len(scores) or not truth:
        raise InvalidLabelError("metrics need one score row per slide and at least one slide")
    for row in scores:
        if len(row) != k or not all(math.isfinite(v) for v in row):
            raise InvalidLabelError(f"every score row needs {k} finite values")
    if any(not 0 <= t < k for t in truth):
        raise InvalidLabelError("true class index outside the vocabulary")
    if predicted is None:
        predicted = [argmax(row) for row in scores]
    confusion = [[0] * k for _ in range(k)]
    for t, p in zip(truth, predicted, strict=True):
        confusion[t][p] += 1
    n = len(truth)
    accuracy = _ok(sum(confusion[c][c] for c in range(k)) / n)
    per_class = []
    for c, name in enumerate(task.classes):
        support = sum(confusion[c])
        npred = sum(confusion[r][c] for r in range(k))
        precision, recall, f1 = _prf(confusion[c][c], npred, support, name)
        labels = [t == c for t in truth]
        per_class.append(
            ClassMetrics(
                name, support, npred, precision, recall, f1, auroc(labels, [r[c] for r in scores])
            )
        )
    present = [m for m in per_class if m.support]
    balanced = (
        _ok(sum(m.recall.value for m in present if m.recall.value is not None) / len(present))
        if len(present) >= 2
        else _undefined(
            f"balanced accuracy needs at least two classes with true slides ({len(present)})"
        )
    )
    if task.binary:
        pos = per_class[1]
        labels = [t == 1 for t in truth]
        precision, recall, f1, roc = pos.precision, pos.recall, pos.f1, pos.auroc
        prc = average_precision(labels, [r[1] for r in scores])
    else:
        names = task.classes
        precision = _macro([m.precision for m in per_class], names, "precision")
        recall = _macro([m.recall for m in per_class], names, "recall")
        f1 = _macro([m.f1 for m in per_class], names, "F1")
        roc = _macro([m.auroc for m in per_class], names, "one-vs-rest AUROC")
        prc = _undefined("AUPRC is reported for binary tasks only (SPEC-007 §9)")
    return ClassificationReport(
        classes=task.classes,
        binary=task.binary,
        slides=n,
        decision_rule=DECISION_RULE,
        confusion_matrix=tuple(tuple(row) for row in confusion),
        accuracy=accuracy,
        balanced_accuracy=balanced,
        precision=precision,
        recall=recall,
        f1=f1,
        auroc=roc,
        auprc=prc,
        per_class=tuple(per_class),
    )
