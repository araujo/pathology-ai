"""Slide-level metric correctness (SPEC-007 §9, AC-015): hand-computed and brute-force checks."""

from __future__ import annotations

import itertools
import random

import pytest

from domain.metrics import (
    MetricValue,
    auroc,
    average_precision,
    classification_report,
)
from tests.mil_support import BINARY, MULTICLASS


def _binary_scores(p: list[float]) -> list[tuple[float, float]]:
    return [(1 - v, v) for v in p]


def test_binary_report_matches_hand_computation() -> None:
    truth = [0, 0, 0, 1, 1, 1, 1]
    p = [0.1, 0.6, 0.3, 0.8, 0.4, 0.9, 0.7]  # predictions (argmax): 0 1 0 1 0 1 1
    r = classification_report(BINARY, truth, _binary_scores(p))
    assert r.confusion_matrix == ((2, 1), (1, 3))
    assert r.accuracy.value == pytest.approx(5 / 7)
    assert r.precision.value == pytest.approx(3 / 4)
    assert r.recall.value == pytest.approx(3 / 4)
    assert r.f1.value == pytest.approx(3 / 4)
    assert r.balanced_accuracy.value == pytest.approx((2 / 3 + 3 / 4) / 2)
    # AUROC: positives {0.8,0.4,0.9,0.7} vs negatives {0.1,0.6,0.3}: 11 of 12 pairs ordered.
    assert r.auroc.value == pytest.approx(11 / 12)
    # AP: ranks .9(+) .8(+) .7(+) .6(-) .4(+) ...: (1 + 1 + 1)/4 + (4/5)/4
    assert r.auprc.value == pytest.approx((1 + 1 + 1 + 4 / 5) / 4)
    assert r.decision_rule == "argmax" and r.slides == 7


def test_multiclass_report_matches_hand_computation() -> None:
    truth = [0, 0, 1, 1, 2, 2]
    scores = [
        (0.7, 0.2, 0.1),
        (0.2, 0.5, 0.3),
        (0.1, 0.8, 0.1),
        (0.3, 0.4, 0.3),
        (0.2, 0.2, 0.6),
        (0.5, 0.1, 0.4),
    ]  # predicted 0 1 1 1 2 0
    r = classification_report(MULTICLASS, truth, scores)
    assert r.confusion_matrix == ((1, 1, 0), (0, 2, 0), (1, 0, 1))
    assert r.accuracy.value == pytest.approx(4 / 6)
    a, b, c = r.per_class
    assert [a.precision.value, a.recall.value] == pytest.approx([0.5, 0.5])
    assert [b.precision.value, b.recall.value] == pytest.approx([2 / 3, 1.0])
    assert [c.precision.value, c.recall.value] == pytest.approx([1.0, 0.5])
    assert r.precision.value == pytest.approx((0.5 + 2 / 3 + 1.0) / 3)
    assert r.recall.value == pytest.approx((0.5 + 1.0 + 0.5) / 3)
    f1s = [0.5, 0.8, 2 / 3]
    assert r.f1.value == pytest.approx(sum(f1s) / 3)
    assert r.balanced_accuracy.value == pytest.approx((0.5 + 1.0 + 0.5) / 3)
    assert r.auprc.value is None and "binary" in (r.auprc.reason or "")
    assert all(m.auroc.defined for m in r.per_class)


def test_undefined_metrics_are_explicit_never_zero() -> None:
    # Only negatives present, nothing predicted positive.
    r = classification_report(BINARY, [0, 0, 0], _binary_scores([0.1, 0.2, 0.3]))
    assert r.accuracy.value == 1.0
    for m in (r.precision, r.recall, r.f1, r.auroc, r.auprc, r.balanced_accuracy):
        assert m.value is None and m.reason
    assert "positive and negative" in (r.auroc.reason or "")
    # Multiclass: a class never predicted makes its precision and the macro value undefined.
    r = classification_report(
        MULTICLASS, [0, 1, 2], [(0.9, 0.1, 0.0), (0.1, 0.9, 0.0), (0.6, 0.4, 0.0)]
    )
    assert r.per_class[2].precision.value is None
    assert r.precision.value is None and "class-c" in (r.precision.reason or "")
    assert r.recall.value == pytest.approx((1 + 1 + 0) / 3)
    with pytest.raises(ValueError):
        MetricValue(None, None)
    with pytest.raises(ValueError):
        MetricValue(float("nan"))


def test_f1_is_zero_when_precision_and_recall_are_zero() -> None:
    r = classification_report(BINARY, [1, 0], _binary_scores([0.2, 0.8]))
    assert r.precision.value == 0.0 and r.recall.value == 0.0 and r.f1.value == 0.0


@pytest.mark.parametrize("seed", range(5))
def test_auroc_equals_the_pairwise_definition(seed: int) -> None:
    rng = random.Random(seed)
    labels = [rng.random() < 0.4 for _ in range(40)]
    labels[0], labels[1] = True, False
    scores = [round(rng.random(), 1) for _ in range(40)]  # many ties
    pos = [s for s, y in zip(scores, labels, strict=True) if y]
    neg = [s for s, y in zip(scores, labels, strict=True) if not y]
    pairs = [(p > n) + 0.5 * (p == n) for p, n in itertools.product(pos, neg)]
    assert auroc(labels, scores).value == pytest.approx(sum(pairs) / len(pairs))


@pytest.mark.parametrize("seed", range(5))
def test_average_precision_equals_the_threshold_definition(seed: int) -> None:
    rng = random.Random(seed)
    labels = [rng.random() < 0.3 for _ in range(30)]
    labels[0] = True
    scores = [round(rng.random(), 1) for _ in range(30)]
    thresholds = sorted(set(scores), reverse=True)
    ap, prev = 0.0, 0.0
    for t in thresholds:
        sel = [y for s, y in zip(scores, labels, strict=True) if s >= t]
        recall = sum(sel) / sum(labels)
        ap += (recall - prev) * (sum(sel) / len(sel))
        prev = recall
    assert average_precision(labels, scores).value == pytest.approx(ap)
    assert average_precision([False, False], [0.1, 0.2]).value is None


def test_inputs_are_validated() -> None:
    with pytest.raises(ValueError):
        classification_report(BINARY, [], [])
    with pytest.raises(ValueError):
        classification_report(BINARY, [0], [(0.5,)])
    with pytest.raises(ValueError):
        classification_report(BINARY, [2], [(0.5, 0.5)])
    with pytest.raises(ValueError):
        classification_report(BINARY, [0], [(float("nan"), 0.5)])
