"""SPEC-007 end to end on SYNTHETIC cohorts: training, evaluation, prediction, isolation.

Accuracy thresholds below check that the pipeline learns a signal the generator injected on
purpose. They are not clinical, biomarker or generalisation claims.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="optional extra pathology-ai[embedding]")

from application.mil_training import MILTrainingService, _PartitionGuard  # noqa: E402
from bootstrap import build_mil_architectures  # noqa: E402
from cli import mil_main  # noqa: E402
from domain.errors import (  # noqa: E402
    BagBudgetError,
    CheckpointIntegrityError,
    CheckpointMissingError,
    EmbeddingCompatibilityError,
    ModelSelectionError,
    SplitError,
    TestIsolationError,
)
from domain.mil import (  # noqa: E402
    BagPolicy,
    EmbeddingBagInfo,
    MILDatasetManifest,
    Partition,
    SplitManifest,
    import_split,
)
from domain.mil_training import EarlyStopping, ExperimentSpec, SelectionMode  # noqa: E402
from infrastructure.mil_artifacts import WEIGHTS, DirectoryCheckpointStore  # noqa: E402
from infrastructure.mil_bags import NumpyBagSource  # noqa: E402
from plugins.mil import ATTENTION_MIL, MEAN_POOLING  # noqa: E402
from plugins.mil.batching import NumpyBatchBuilder  # noqa: E402
from tests.conftest import REPO_ROOT  # noqa: E402
from tests.mil_support import (  # noqa: E402
    ATTENTION_HP,
    BINARY,
    MULTICLASS,
    Cohort,
    cohort,
    split_of,
    training_config,
)

pytestmark = pytest.mark.filterwarnings("error")


class RecordingSource(NumpyBagSource):
    """Records every store described or read (test-isolation evidence)."""

    def __init__(self, roots: list[Path]) -> None:
        super().__init__(roots)
        self.touched: list[str] = []

    def describe(self, store_id: str) -> EmbeddingBagInfo:
        self.touched.append(store_id)
        return super().describe(store_id)


def _service(src: NumpyBagSource, checkpoints: Path) -> MILTrainingService:
    return MILTrainingService(
        bags=src,
        batches=NumpyBatchBuilder(),
        architectures=build_mil_architectures(),
        checkpoints=DirectoryCheckpointStore(checkpoints),
    )


@pytest.fixture(scope="module")
def binary(tmp_path_factory: pytest.TempPathFactory) -> Cohort:
    return cohort(tmp_path_factory.mktemp("bin"), BINARY, groups_per_class=30, seed=1)


@pytest.fixture(scope="module")
def multi(tmp_path_factory: pytest.TempPathFactory) -> Cohort:
    return cohort(tmp_path_factory.mktemp("multi"), MULTICLASS, groups_per_class=15, seed=2)


def _experiment(
    svc: MILTrainingService,
    d: MILDatasetManifest,
    s: SplitManifest,
    arch: str = ATTENTION_MIL,
    **training: object,
) -> ExperimentSpec:
    return svc.experiment(
        name="synthetic",
        dataset=d,
        split=s,
        architecture=arch,
        hyperparameters=ATTENTION_HP if arch == ATTENTION_MIL else (),
        training=training_config(**training),
    )


# --------------------------------------------------------------------------- end to end


def test_binary_attention_mil_end_to_end(binary: Cohort, tmp_path: Path) -> None:
    d, s = binary.dataset(), split_of(binary.dataset())
    svc = _service(binary.source(), tmp_path)
    exp = _experiment(svc, d, s)
    out = svc.train(exp, d, s)
    assert not out.reused and out.measurements is not None
    c = out.checkpoint
    assert svc.checkpoints.find(exp.identity) == c
    h = c.history
    assert h.selection_metric == "validation_loss" and 0 <= h.best_epoch < len(h.epochs)
    assert h.best_value == min(e.validation_loss for e in h.epochs)
    ev = svc.evaluate(exp.identity, d, s, Partition.TEST)
    r = ev.report
    assert r.accuracy.value is not None and r.accuracy.value >= 0.75
    assert r.auroc.value is not None and r.auroc.value >= 0.8
    assert r.auprc.defined and r.balanced_accuracy.defined
    m = ev.predictions
    assert m.schema_version == "pathology-ai.slide-prediction/1"
    assert m.purpose.value == "evaluation" and m.partition is Partition.TEST
    assert (m.experiment_identity, m.checkpoint_id, m.checkpoint_digest) == (
        exp.identity,
        c.checkpoint_id,
        c.digest,
    )
    assert (m.dataset_digest, m.split_digest, m.compatibility_key) == (
        d.digest,
        s.digest,
        d.compatibility_key,
    )
    assert [p.slide_id for p in m.predictions] == list(s.members(Partition.TEST))
    for p in m.predictions:
        rec = d.record(p.slide_id)
        assert p.embedding_manifest_digest == rec.embedding_manifest_digest
        assert p.embedding_identity == rec.embedding_identity and p.store_id == rec.store_id
        assert p.score_kind.value == "sigmoid" and len(p.logits) == 1
        assert sum(v for _, v in p.scores) == pytest.approx(1.0)
        assert not p.scores_calibrated and p.calibrated_probabilities is None
        # The evaluation policy keeps every instance: the selection is complete and recorded.
        assert p.selection.original_count == p.selection.selected_count == rec.patch_count
        assert p.attention is not None and p.attention.semantics == "attention_weight"
        ids = {r.patch_id for r in binary.source().describe(rec.store_id).records}
        assert {pid for pid, _ in p.attention.top_patches} <= ids
    assert "not diagnoses" in m.notice


def test_multiclass_mean_pooling_end_to_end(multi: Cohort, tmp_path: Path) -> None:
    d, s = multi.dataset(), split_of(multi.dataset())
    svc = _service(multi.source(), tmp_path)
    exp = _experiment(svc, d, s, MEAN_POOLING, class_weighting=True)
    svc.train(exp, d, s)
    ev = svc.evaluate(exp.identity, d, s, Partition.TEST)
    assert ev.report.accuracy.value is not None and ev.report.accuracy.value >= 0.8
    assert len(ev.report.per_class) == 3 and ev.report.auroc.defined
    assert ev.report.auprc.value is None  # binary only, explicitly
    p = ev.predictions.predictions[0]
    assert p.score_kind.value == "softmax" and len(p.logits) == 3 and p.attention is None


# --------------------------------------------------------------------------- reproducibility


def _train_and_test(c: Cohort, checkpoints: Path) -> tuple[str, str, str]:
    d, s = c.dataset(), split_of(c.dataset())
    svc = _service(c.source(), checkpoints)
    exp = _experiment(svc, d, s)
    out = svc.train(exp, d, s)
    ev = svc.evaluate(exp.identity, d, s, Partition.TEST)
    return out.checkpoint.weights_sha256, out.checkpoint.digest, ev.predictions.content_digest


def test_training_and_prediction_are_reproducible(binary: Cohort, tmp_path: Path) -> None:
    a = _train_and_test(binary, tmp_path / "a")
    b = _train_and_test(binary, tmp_path / "b")
    assert a == b


_FRESH = """
import json, sys
from pathlib import Path
from tests.integration.test_mil_training import _train_and_test
from tests.mil_support import Cohort, BINARY, cohort
c = cohort(Path(sys.argv[1]), BINARY, groups_per_class=30, seed=1)
print(json.dumps(_train_and_test(c, Path(sys.argv[2]))))
"""


def test_fresh_processes_reproduce_checkpoints_and_predictions(
    binary: Cohort, tmp_path: Path
) -> None:
    here = _train_and_test(binary, tmp_path / "here")
    proc = subprocess.run(
        [sys.executable, "-c", _FRESH, str(binary.root), str(tmp_path / "there")],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert tuple(json.loads(proc.stdout)) == here


def test_rerun_reuses_the_verified_checkpoint_and_reload_reproduces(
    binary: Cohort, tmp_path: Path
) -> None:
    d, s = binary.dataset(), split_of(binary.dataset())
    svc = _service(binary.source(), tmp_path)
    exp = _experiment(svc, d, s)
    first = svc.train(exp, d, s)
    evaluated = svc.evaluate(exp.identity, d, s, Partition.TEST).predictions
    src = RecordingSource([binary.root])
    again = _service(src, tmp_path)
    second = again.train(exp, d, s)
    assert second.reused and second.checkpoint == first.checkpoint and src.rows_read == 0
    test = [(sid, d.record(sid).store_id) for sid in s.members(Partition.TEST)]
    predicted = again.predict(exp.identity, test)
    assert predicted.purpose.value == "inference" and not predicted.policy_overridden
    assert [p.logits for p in predicted.predictions] == [p.logits for p in evaluated.predictions]
    assert predicted.content_digest == again.predict(exp.identity, test).content_digest


# --------------------------------------------------------------------------- isolation


def test_test_partition_is_never_read_during_training(binary: Cohort, tmp_path: Path) -> None:
    d, s = binary.dataset(), split_of(binary.dataset())
    src = RecordingSource([binary.root])
    svc = _service(src, tmp_path)
    exp = _experiment(svc, d, s)
    svc.train(exp, d, s)
    test_stores = {d.record(x).store_id for x in s.members(Partition.TEST)}
    allowed = {
        d.record(x).store_id
        for x in (*s.members(Partition.TRAIN), *s.members(Partition.VALIDATION))
    }
    assert set(src.touched) <= allowed and set(src.reads) <= allowed
    assert not test_stores & (set(src.touched) | set(src.reads))
    guard = _PartitionGuard(src, frozenset(allowed), "train/validation")
    with pytest.raises(TestIsolationError):
        guard.read(next(iter(test_stores)), (0,))


def test_test_labels_do_not_influence_the_model(binary: Cohort, tmp_path: Path) -> None:
    d = binary.dataset()
    s = split_of(d)
    test = set(s.members(Partition.TEST))
    flipped = [
        replace(lbl, value="class-a" if lbl.value == "class-b" else "class-b")
        if lbl.slide_id in test
        else lbl
        for lbl in binary.labels()
    ]
    d2 = binary.dataset(flipped)
    assert d2.digest != d.digest
    s2 = import_split(d2, {x: p.partition for p in s.partitions for x in p.slide_ids})
    svc = _service(binary.source(), tmp_path)
    a = svc.train(_experiment(svc, d, s), d, s).checkpoint
    b = svc.train(_experiment(svc, d2, s2), d2, s2).checkpoint
    assert a.experiment_identity != b.experiment_identity
    assert a.weights_sha256 == b.weights_sha256 and a.history == b.history


def test_evaluation_never_resplits_or_mixes_embeddings(binary: Cohort, tmp_path: Path) -> None:
    d, s = binary.dataset(), split_of(binary.dataset())
    svc = _service(binary.source(), tmp_path)
    exp = _experiment(svc, d, s)
    with pytest.raises(CheckpointMissingError):
        svc.evaluate(exp.identity, d, s, Partition.TEST)  # no selected model yet
    svc.train(exp, d, s)
    with pytest.raises(SplitError):
        svc.evaluate(exp.identity, d, split_of(d, seed=99), Partition.TEST)
    other = cohort(tmp_path / "variant", BINARY, groups_per_class=1, variant="b")
    slides = [(x.slide_id, other.stores[x.slide_id]) for x in other.slides]
    svc_other = _service(NumpyBagSource([binary.root, other.root]), tmp_path)
    with pytest.raises(EmbeddingCompatibilityError):
        svc_other.predict(exp.identity, slides)


def test_undefined_selection_metric_fails_closed(binary: Cohort, tmp_path: Path) -> None:
    d = binary.dataset()
    s = split_of(d)
    # Validation without any positive slide: AUPRC is undefined at every epoch.
    assignment = {x: p.partition for p in s.partitions for x in p.slide_ids}
    for x, part in list(assignment.items()):
        if part is Partition.VALIDATION and d.record(x).label == "class-b":
            assignment[x] = Partition.TRAIN
    s2 = import_split(d, assignment)
    svc = _service(binary.source(), tmp_path)
    exp = _experiment(
        svc, d, s2, epochs=3, early_stopping=EarlyStopping("auprc", SelectionMode.MAX, 2)
    )
    with pytest.raises(ModelSelectionError, match="positive"):
        svc.train(exp, d, s2)
    assert svc.checkpoints.find(exp.identity) is None


# --------------------------------------------------------------------------- resources


def test_budgets_fail_before_any_row_is_read(binary: Cohort, tmp_path: Path) -> None:
    d, s = binary.dataset(), split_of(binary.dataset())
    src = RecordingSource([binary.root])
    svc = _service(src, tmp_path)
    exp = _experiment(svc, d, s, train_bags=BagPolicy(48, 13, 1024))
    with pytest.raises(BagBudgetError):
        svc.train(exp, d, s)
    assert src.rows_read == 0 and svc.checkpoints.find(exp.identity) is None


def test_subsampling_is_recorded_and_policy_overrides_are_explicit(
    binary: Cohort, tmp_path: Path
) -> None:
    d, s = binary.dataset(), split_of(binary.dataset())
    svc = _service(binary.source(), tmp_path)
    exp = _experiment(svc, d, s, eval_bags=BagPolicy(8, 3, 1 << 26))
    svc.train(exp, d, s)
    ev = svc.evaluate(exp.identity, d, s, Partition.TEST).predictions
    big = [p for p in ev.predictions if p.selection.original_count > 8]
    assert big and all(p.selection.selected_count == 8 for p in big)
    assert all(p.selection.algorithm == "sha256-rank@1" and p.selection.seed == 3 for p in big)
    test = [(x, d.record(x).store_id) for x in s.members(Partition.TEST)]
    over = svc.predict(exp.identity, test, policy=BagPolicy(None, 0, 1 << 26))
    assert over.policy_overridden and over.bag_policy.max_instances is None
    assert over.content_digest != svc.predict(exp.identity, test).content_digest


def test_corrupt_checkpoint_blocks_reuse_and_evaluation(binary: Cohort, tmp_path: Path) -> None:
    d, s = binary.dataset(), split_of(binary.dataset())
    svc = _service(binary.source(), tmp_path)
    exp = _experiment(svc, d, s, epochs=2)
    svc.train(exp, d, s)
    weights = tmp_path / exp.identity / WEIGHTS
    raw = bytearray(weights.read_bytes())
    raw[0] ^= 0xFF
    weights.write_bytes(bytes(raw))
    with pytest.raises(CheckpointIntegrityError):
        svc.evaluate(exp.identity, d, s, Partition.TEST)
    with pytest.raises(CheckpointIntegrityError):
        svc.train(exp, d, s)


# --------------------------------------------------------------------------- CLI


def test_cli_run_is_idempotent_and_writes_versioned_artifacts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert mil_main(["synthetic", "--out", str(tmp_path / "syn"), "--groups-per-class", "12"]) == 0
    capsys.readouterr()
    args = [
        "run",
        "--config",
        str(tmp_path / "syn" / "experiment.yaml"),
        "--store-root",
        str(tmp_path / "syn" / "stores"),
        "--out",
        str(tmp_path / "run"),
    ]
    assert mil_main(args) == 0
    first = json.loads(capsys.readouterr().out)
    assert mil_main(args) == 0
    second = json.loads(capsys.readouterr().out)
    assert not first["training"]["reused"] and second["training"]["reused"]
    for key in ("validation", "test"):
        assert first[key]["predictions_digest"] == second[key]["predictions_digest"]
    assert first["split"]["group_overlap"] == []
    assert "not clinical" in first["notice"]
    for name in ("dataset.json", "split.json", "predictions-test.json"):
        assert json.loads((tmp_path / "run" / name).read_text())["digest"]
    exp_id = first["training"]["experiment_identity"]
    test_csv = tmp_path / "slides.csv"
    split = json.loads((tmp_path / "run" / "split.json").read_text())["body"]
    dataset = json.loads((tmp_path / "run" / "dataset.json").read_text())["body"]
    stores = {r["slide_id"]: r["store_id"] for r in dataset["records"]}
    ids = split["partitions"][2]["slide_ids"]
    test_csv.write_text("slide_id,store_id\n" + "".join(f"{i},{stores[i]}\n" for i in ids))
    assert (
        mil_main(
            [
                "predict",
                "--checkpoints",
                str(tmp_path / "run" / "checkpoints"),
                "--experiment",
                exp_id,
                "--slides",
                str(test_csv),
                "--store-root",
                str(tmp_path / "syn" / "stores"),
                "--out",
                str(tmp_path / "pred.json"),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["predictions"] == len(ids)


_NO_TORCH = """
import json, sys
sys.modules["torch"] = None  # the optional runtime is not installed
from pathlib import Path
from cli import mil_main
out = Path(sys.argv[1])
mil_main(["synthetic", "--out", str(out / "syn"), "--groups-per-class", "3"])
cfg, stores = str(out / "syn" / "experiment.yaml"), str(out / "syn" / "stores")
mil_main(["dataset", "--config", cfg, "--store-root", stores, "--out", str(out / "d.json")])
mil_main(["split", "--config", cfg, "--dataset", str(out / "d.json"), "--out", str(out / "s.json")])
try:
    mil_main(["train", "--config", cfg, "--dataset", str(out / "d.json"), "--split",
              str(out / "s.json"), "--store-root", stores, "--checkpoints", str(out / "c")])
except SystemExit as exc:
    print("EXIT", exc.code)
"""


def test_datasets_and_splits_work_without_torch_and_training_names_the_extra(
    tmp_path: Path,
) -> None:
    proc = subprocess.run(
        [sys.executable, "-c", _NO_TORCH, str(tmp_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert "EXIT 3" in proc.stdout, proc.stderr
    assert "MODEL_RUNTIME_UNAVAILABLE" in proc.stderr and "pathology-ai[embedding]" in proc.stderr
    assert (tmp_path / "d.json").exists() and (tmp_path / "s.json").exists()
