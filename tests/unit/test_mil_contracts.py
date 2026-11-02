"""SPEC-007 dataset, label, compatibility, split and bag-selection contracts (no torch needed).

All data is SYNTHETIC.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from application.mil_dataset import MILDatasetService
from domain.errors import (
    BagBudgetError,
    DomainError,
    DuplicateSlideError,
    EmbeddingCompatibilityError,
    EmbeddingStoreCorruptError,
    EmbeddingStoreIncompleteError,
    EmptyBagError,
    ExperimentConflictError,
    GroupLeakageError,
    InvalidLabelError,
    MILDatasetError,
    MissingGroupError,
    SplitError,
)
from domain.ids import TaskId, Version
from domain.mil import (
    EXACT,
    EXACT_ANY_DEVICE,
    BagPolicy,
    EmbeddingCompatibility,
    LabelStatus,
    MILDatasetManifest,
    MILTask,
    Partition,
    PartitionSummary,
    SlideLabel,
    SplitFractions,
    SplitManifest,
    batch_order,
    generate_split,
    import_split,
    validate_split,
)
from domain.task import TaskType
from infrastructure.embedding_store import CHUNKS, DATA, NumpyEmbeddingStoreProvider
from infrastructure.mil_artifacts import read_dataset, read_split, write_once
from infrastructure.mil_bags import NumpyBagSource
from infrastructure.mil_config import load_experiment_config, load_labels, load_split_table
from plugins.synthetic.mil import write_bag
from tests.conftest import REPO_ROOT
from tests.mil_support import BINARY, DIM, MULTICLASS, SOURCE, Cohort, cohort, split_of


@pytest.fixture(scope="module")
def shared(tmp_path_factory: pytest.TempPathFactory) -> Cohort:
    """Read-only cohort shared by tests that do not modify stores (writes are fsync-bound)."""
    return cohort(tmp_path_factory.mktemp("shared"), BINARY, groups_per_class=6)


@pytest.fixture(scope="module")
def multi(tmp_path_factory: pytest.TempPathFactory) -> Cohort:
    return cohort(tmp_path_factory.mktemp("multi"), MULTICLASS, groups_per_class=5, seed=3)


@pytest.fixture
def binary(tmp_path: Path) -> Cohort:
    """A small private cohort for tests that corrupt or replace stores."""
    return cohort(tmp_path / "stores", BINARY, groups_per_class=2)


# --------------------------------------------------------------------------- task and labels


@pytest.mark.parametrize(
    "classes,positive,kind",
    [
        (("a",), None, TaskType.MULTICLASS_CLASSIFICATION),
        (("a", "a"), None, TaskType.MULTICLASS_CLASSIFICATION),
        (("a", "b"), "a", TaskType.BINARY_CLASSIFICATION),  # positive must be second
        (("a", "b"), None, TaskType.BINARY_CLASSIFICATION),
        (("a", "b", "c"), "c", TaskType.BINARY_CLASSIFICATION),
        (("a", "b", "c"), "c", TaskType.MULTICLASS_CLASSIFICATION),
        (("a", " b"), None, TaskType.MULTICLASS_CLASSIFICATION),
        (("a", ""), None, TaskType.MULTICLASS_CLASSIFICATION),
        (("a", "b"), None, TaskType.REGRESSION),
    ],
)
def test_invalid_tasks_are_rejected(
    classes: tuple[str, ...], positive: str | None, kind: TaskType
) -> None:
    with pytest.raises(InvalidLabelError):
        MILTask(TaskId("t"), Version("1"), kind, classes, positive)


def test_class_names_are_free_configuration() -> None:
    task = MILTask(
        TaskId("any-target"),
        Version("2"),
        TaskType.MULTICLASS_CLASSIFICATION,
        ("Grade I", "grade-2", "Mutant / wild type?"),
    )
    assert task.index("Mutant / wild type?") == 2
    assert task.num_outputs == 3
    assert BINARY.num_outputs == 1


@pytest.mark.parametrize(
    "change",
    [
        {"status": LabelStatus.UNLABELED},  # unlabeled with a value
        {"value": None},  # eligible without a label
        {"status": LabelStatus.EXCLUDED},  # excluded without a reason
        {"status_reason": "why"},  # reason without exclusion
        {"slide_id": "../etc/passwd"},
        {"group_id": "a b"},
        {"source": " "},
    ],
)
def test_malformed_label_records_are_rejected(change: dict[str, object]) -> None:
    base = SlideLabel("s1", BINARY.ref, "class-a", *SOURCE, LabelStatus.ELIGIBLE, "g1")
    with pytest.raises(MILDatasetError):
        replace(base, **change)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- dataset manifest


def test_dataset_manifest_is_validated_and_fingerprinted(shared: Cohort, tmp_path: Path) -> None:
    d = shared.dataset()
    assert d.schema_version == "pathology-ai.mil-dataset/1"
    assert len(d.records) == len(shared.slides)
    assert [r.slide_id for r in d.records] == sorted(r.slide_id for r in d.records)
    assert d.feature_dim == DIM and d.counts()["eligible"] == len(d.records)
    r = d.records[0]
    assert r.embedding_manifest_digest and r.embedding_identity.startswith("pai-embid-")
    assert r.compatibility_key == d.compatibility_key and r.patch_count >= 8
    # Deterministic, and independent of where the stores live (paths never enter identities).
    assert shared.dataset().digest == d.digest
    moved = tmp_path / "elsewhere"
    shutil.copytree(shared.root, moved)
    assert shared.dataset(source=NumpyBagSource([moved])).digest == d.digest
    # Any label change is a different dataset.
    labels = shared.labels()
    labels[0] = replace(labels[0], value="class-b" if labels[0].value == "class-a" else "class-a")
    assert shared.dataset(labels).digest != d.digest


def test_invalid_duplicate_missing_and_foreign_labels_are_rejected(shared: Cohort) -> None:
    labels = shared.labels()
    with pytest.raises(InvalidLabelError, match="not in the class vocabulary"):
        shared.dataset([replace(labels[0], value="class-z"), *labels[1:]])
    other = MILTask(TaskId("other"), Version("1"), TaskType.BINARY_CLASSIFICATION, ("x", "y"), "y")
    with pytest.raises(InvalidLabelError, match="belongs to task"):
        shared.dataset([replace(labels[0], task=other.ref), *labels[1:]])
    with pytest.raises(DuplicateSlideError):
        shared.dataset([*labels, labels[0]])
    with pytest.raises(MILDatasetError, match="no embedding store"):
        MILDatasetService(shared.source()).build(
            dataset_id=shared.dataset().dataset_id,
            version=Version("1"),
            task=BINARY,
            policy=EXACT,
            labels=labels,
            stores={k: v for k, v in list(shared.stores.items())[1:]},
        )


def test_one_store_cannot_back_two_slides(shared: Cohort) -> None:
    labels = shared.labels()
    stores = dict(shared.stores)
    stores[labels[1].slide_id] = stores[labels[0].slide_id]
    with pytest.raises(DuplicateSlideError, match="embedding store"):
        MILDatasetService(shared.source()).build(
            dataset_id=shared.dataset().dataset_id,
            version=Version("1"),
            task=BINARY,
            policy=EXACT,
            labels=labels,
            stores=stores,
        )


def test_supervised_slides_need_a_group_but_unlabeled_ones_do_not(shared: Cohort) -> None:
    labels = shared.labels()
    with pytest.raises(MissingGroupError):
        shared.dataset([replace(labels[0], group_id=None), *labels[1:]])
    unlabeled = replace(labels[0], value=None, status=LabelStatus.UNLABELED, group_id=None)
    excluded = replace(labels[1], status=LabelStatus.EXCLUDED, status_reason="QC failure")
    d = shared.dataset([unlabeled, excluded, *labels[2:]])
    assert d.counts() == {"eligible": len(labels) - 2, "excluded": 1, "unlabeled": 1}


def test_mixed_embedding_identities_fail_closed(shared: Cohort, tmp_path: Path) -> None:
    """Same dimension, different encoder variant: never mixed (AC-006)."""
    provider = NumpyEmbeddingStoreProvider(shared.root)
    s = shared.slides[0]
    stores = dict(shared.stores)
    stores[s.slide_id] = write_bag(provider, s.slide_id, s.rows, variant="b")
    with pytest.raises(EmbeddingCompatibilityError, match="encoder parameters"):
        MILDatasetService(shared.source()).build(
            dataset_id=shared.dataset().dataset_id,
            version=Version("1"),
            task=BINARY,
            policy=EXACT,
            labels=shared.labels(),
            stores=stores,
        )


def test_device_is_bound_unless_a_policy_explicitly_relaxes_it(shared: Cohort) -> None:
    provider = NumpyEmbeddingStoreProvider(shared.root)
    s = shared.slides[0]
    stores = dict(shared.stores)
    stores[s.slide_id] = write_bag(provider, s.slide_id, s.rows, device="mps")
    build = MILDatasetService(shared.source()).build
    kw = dict(
        dataset_id=shared.dataset().dataset_id,
        version=Version("1"),
        task=BINARY,
        labels=shared.labels(),
        stores=stores,
    )
    with pytest.raises(EmbeddingCompatibilityError, match="device"):
        build(policy=EXACT, **kw)  # type: ignore[arg-type]
    relaxed = build(policy=EXACT_ANY_DEVICE, **kw)  # type: ignore[arg-type]
    assert relaxed.compatibility_policy == "exact-any-device@1"
    assert relaxed.compatibility_key != shared.dataset().compatibility_key


def test_source_channel_adaptations_do_not_change_the_key(shared: Cohort) -> None:
    """CZI (bgr->rgb) and SVS (rgba->rgb) bags of one encoder share a key; other steps don't."""
    from domain.embedding import DeclaredTransform, TransformChain
    from domain.preprocessing import TransformStep
    from domain.provenance import ComponentVersion

    info = shared.source().describe(shared.stores[shared.slides[0].slide_id])
    impl = ComponentVersion("transform", "x", "1")

    def step(name: str, **p: str) -> DeclaredTransform:
        return DeclaredTransform(TransformStep(name, tuple(sorted(p.items()))), impl)

    bgr = replace(
        info.run,
        transforms=TransformChain((step("channel_reorder", **{"from": "bgr", "to": "rgb"}),)),
    )
    rgba = replace(
        info.run, transforms=TransformChain((step("drop_alpha", **{"from": "rgba", "to": "rgb"}),))
    )
    norm = replace(info.run, transforms=TransformChain((step("normalize", mean="0", std="1"),)))
    keys = {EmbeddingCompatibility.of(r, EXACT).key for r in (info.run, bgr, rgba)}
    assert len(keys) == 1
    assert EmbeddingCompatibility.of(norm, EXACT).key not in keys


def test_accepted_key_is_enforced(shared: Cohort) -> None:
    with pytest.raises(EmbeddingCompatibilityError, match="accepts"):
        MILDatasetService(shared.source()).build(
            dataset_id=shared.dataset().dataset_id,
            version=Version("1"),
            task=BINARY,
            policy=EXACT,
            labels=shared.labels(),
            stores=shared.stores,
            accepted_key="pai-embkey-" + "0" * 32,
        )


def test_non_finalized_and_corrupt_stores_are_rejected(binary: Cohort) -> None:
    first, second = binary.slides[0].slide_id, binary.slides[1].slide_id
    (binary.root / binary.stores[first] / "manifest.json").unlink()
    with pytest.raises(EmbeddingStoreIncompleteError):
        binary.source().describe(binary.stores[first])
    data = binary.root / binary.stores[second] / DATA
    raw = bytearray(data.read_bytes())
    raw[-1] ^= 0xFF
    data.write_bytes(bytes(raw))
    with pytest.raises(EmbeddingStoreCorruptError):
        binary.source().describe(binary.stores[second])
    with pytest.raises(EmbeddingStoreIncompleteError):
        binary.source().describe("pai-emb-" + "0" * 32)
    with pytest.raises(DomainError):
        binary.dataset()


def test_interrupted_store_is_not_a_bag(tmp_path: Path) -> None:
    from tests.mil_support import cohort as make

    c = make(tmp_path, BINARY, groups_per_class=1)
    store = tmp_path / c.stores[c.slides[0].slide_id]
    (store / "manifest.json").unlink()
    for record in sorted((store / CHUNKS).iterdir())[:1]:
        record.unlink()
    with pytest.raises(EmbeddingStoreIncompleteError):
        c.source().describe(store.name)


def test_reverification_detects_a_replaced_store(binary: Cohort) -> None:
    d = binary.dataset()
    MILDatasetService(binary.source()).verify(d)
    s = binary.slides[0]
    shutil.rmtree(binary.root / binary.stores[s.slide_id])
    rows = s.rows.copy()
    rows[0, 0] += 1.0  # same identity, different content
    write_bag(NumpyEmbeddingStoreProvider(binary.root), s.slide_id, rows)
    with pytest.raises(MILDatasetError, match="no longer matches"):
        MILDatasetService(binary.source()).verify(d)


# --------------------------------------------------------------------------- splits


def _groups(dataset: MILDatasetManifest, split: SplitManifest) -> dict[Partition, set[str | None]]:
    return {
        p.partition: {dataset.record(s).group_id for s in p.slide_ids} for p in split.partitions
    }


@pytest.mark.parametrize("seed", range(12))
def test_generated_splits_never_share_a_group(multi: Cohort, seed: int) -> None:
    d = multi.dataset()
    split = generate_split(d, SplitFractions(0.6, 0.2, 0.2), seed)
    groups = _groups(d, split)
    assert not groups[Partition.TRAIN] & groups[Partition.VALIDATION]
    assert not groups[Partition.TRAIN] & groups[Partition.TEST]
    assert not groups[Partition.VALIDATION] & groups[Partition.TEST]
    assigned = sorted(s for p in split.partitions for s in p.slide_ids)
    assert assigned == sorted(r.slide_id for r in d.records)
    validate_split(split, d)


def test_splits_are_deterministic_and_recorded(shared: Cohort) -> None:
    d = shared.dataset()
    a, b = split_of(d, 7), split_of(d, 7)
    assert a == b and a.digest == b.digest and a.split_id.startswith("pai-split-")
    assert split_of(d, 8).digest != a.digest
    assert a.dataset_digest == d.digest and a.seed == 7 and a.strategy == "group-stratified@1"
    total = {c: 0 for c in BINARY.classes}
    for p in a.partitions:
        assert p.slide_ids and p.group_count >= 1
        for cls, n in p.class_counts:
            total[cls] += n
    assert sum(total.values()) == len(d.records)


def test_external_split_is_preserved_and_leakage_is_rejected(shared: Cohort) -> None:
    d = shared.dataset()
    generated = split_of(d)
    assignment = {s: p.partition for p in generated.partitions for s in p.slide_ids}
    imported = import_split(d, assignment)
    assert imported.strategy == "external@1" and imported.seed is None
    assert [p.slide_ids for p in imported.partitions] == [p.slide_ids for p in generated.partitions]
    # Move one slide of a multi-slide group into another partition: the group now leaks.
    by_group: dict[str, list[str]] = {}
    for r in d.records:
        by_group.setdefault(r.group_id or "", []).append(r.slide_id)
    group = next(ids for ids in by_group.values() if len(ids) > 1)
    leaky = dict(assignment)
    leaky[group[0]] = (
        Partition.TEST if assignment[group[1]] is not Partition.TEST else Partition.TRAIN
    )
    with pytest.raises(GroupLeakageError):
        import_split(d, leaky)


def test_split_validation_fails_closed(shared: Cohort) -> None:
    d = shared.dataset()
    split = split_of(d)
    other = shared.dataset([replace(shared.labels()[0], cohort="site-x"), *shared.labels()[1:]])
    with pytest.raises(SplitError, match="never regenerated"):
        validate_split(split, other)
    p = split.partitions[0]
    tampered = replace(
        split,
        partitions=(
            PartitionSummary(p.partition, p.slide_ids, p.group_count + 1, p.class_counts),
            *split.partitions[1:],
        ),
    )
    with pytest.raises(SplitError, match="do not match"):
        validate_split(tampered, d)
    with pytest.raises(GroupLeakageError):
        replace(split, group_overlap=("g1",))
    unlabeled = [
        replace(lbl, value=None, status=LabelStatus.UNLABELED) for lbl in shared.labels()[:1]
    ]
    d2 = shared.dataset([*unlabeled, *shared.labels()[1:]])
    assignment = {s: p.partition for p in split.partitions for s in p.slide_ids}
    with pytest.raises(SplitError, match="unlabeled"):
        import_split(d2, assignment)


def test_split_fractions_are_validated() -> None:
    for bad in ((0.5, 0.5, 0.5), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (1.2, -0.1, -0.1)):
        with pytest.raises(SplitError):
            SplitFractions(*bad)
    SplitFractions(0.8, 0.2, 0.0)


# --------------------------------------------------------------------------- bag selection


def test_subsampling_is_deterministic_bounded_and_recorded() -> None:
    policy = BagPolicy(16, 5, 1 << 20)
    a = policy.select("slide-1", 100, epoch=0)
    assert a == policy.select("slide-1", 100, epoch=0)
    assert a.selected_count == 16 and a.original_count == 100 and not a.complete
    assert list(a.indices) == sorted(set(a.indices)) and 0 <= a.indices[0] and a.indices[-1] < 100
    assert a.algorithm == "sha256-rank@1" and a.seed == 5 and a.epoch == 0
    assert policy.select("slide-1", 100, epoch=1).indices != a.indices
    assert policy.select("slide-2", 100, epoch=0).indices != a.indices
    assert BagPolicy(16, 6, 1 << 20).select("slide-1", 100, 0).indices != a.indices
    full = policy.select("slide-1", 10, epoch=None)
    assert full.complete and full.indices == tuple(range(10))
    assert BagPolicy(None, 0, 1 << 20).select("s", 5000, None).selected_count == 5000
    with pytest.raises(EmptyBagError):
        policy.select("slide-1", 0, None)


def test_selection_is_platform_independent() -> None:
    """Pinned: SHA-256 ranks do not depend on a PRNG implementation or Python version."""
    sel = BagPolicy(4, 1, 1 << 20).select("slide-x", 20, epoch=None)
    assert sel.indices == (7, 10, 13, 18)


def test_batch_budget_is_checked_before_loading() -> None:
    policy = BagPolicy(100, 0, 100 * 32 * 4 * 2 + 200)
    assert policy.check_batch(2, 5000, 32) == 2 * 100 * (32 * 4 + 1)
    with pytest.raises(BagBudgetError, match="max_batch_bytes"):
        policy.check_batch(3, 5000, 32)
    with pytest.raises(BagBudgetError):
        BagPolicy(None, 0, 1000).check_batch(1, 5000, 32)  # unbounded bags must fit
    with pytest.raises(BagBudgetError):
        BagPolicy(0, 0, 10)


def test_batch_order_is_a_seeded_permutation() -> None:
    ids = [f"s{i}" for i in range(20)]
    a = batch_order(ids, 3, 0)
    assert sorted(a) == sorted(ids) and a == batch_order(ids, 3, 0)
    assert a != batch_order(ids, 3, 1)


# --------------------------------------------------------------------------- documents


def test_documents_round_trip_and_are_never_overwritten(shared: Cohort, tmp_path: Path) -> None:
    d = shared.dataset()
    split = split_of(d)
    write_once(tmp_path / "dataset.json", d)
    write_once(tmp_path / "split.json", split)
    write_once(tmp_path / "split.json", split)  # identical: accepted
    assert read_dataset(tmp_path / "dataset.json") == d
    assert read_split(tmp_path / "split.json") == split
    with pytest.raises(ExperimentConflictError):
        write_once(tmp_path / "split.json", split_of(d, 99))
    doc = json.loads((tmp_path / "dataset.json").read_text())
    doc["body"]["records"][0]["label"] = (
        "class-b" if doc["body"]["records"][0]["label"] == "class-a" else "class-a"
    )
    (tmp_path / "dataset.json").write_text(json.dumps(doc))
    with pytest.raises(MILDatasetError, match="digest"):
        read_dataset(tmp_path / "dataset.json")


def test_experiment_configs_and_tables(shared: Cohort, tmp_path: Path) -> None:
    from cli import synthetic_experiment_yaml

    for kind in ("binary", "multiclass"):
        path = REPO_ROOT / "projects" / "experiments" / f"mil-synthetic-{kind}.yaml"
        assert path.read_text() == synthetic_experiment_yaml(kind)  # examples stay in sync
        cfg = load_experiment_config(path)
        assert cfg.task.task_type.value.startswith(kind)
    bad = tmp_path / "bad.yaml"
    bad.write_text(synthetic_experiment_yaml("binary").replace("patience: 8", "patiense: 8"))
    with pytest.raises(DomainError, match="patiense"):
        load_experiment_config(bad)
    table = tmp_path / "labels.csv"
    table.write_text(
        "slide_id,store_id,label,group_id,cohort,site,status,status_reason\n"
        "s1,pai-emb-1,class-a,g1,,,eligible,\n"
        "s2,pai-emb-2,,,,,unlabeled,\n"
        "s3,pai-emb-3,class-b,g3,,,excluded,stain QC\n"
    )
    labels, stores = load_labels(table, BINARY, *SOURCE)
    assert [lbl.status for lbl in labels] == [
        LabelStatus.ELIGIBLE,
        LabelStatus.UNLABELED,
        LabelStatus.EXCLUDED,
    ]
    assert stores == {"s1": "pai-emb-1", "s2": "pai-emb-2", "s3": "pai-emb-3"}
    table.write_text(
        "slide_id,store_id,label,group_id,cohort,site,status,status_reason\ns1,x,class-a,g1,,,,\n"
    )
    with pytest.raises(DomainError, match="status"):
        load_labels(table, BINARY, *SOURCE)  # never inferred
    split_table = tmp_path / "split.csv"
    split_table.write_text("slide_id,partition\ns1,train\ns2,holdout\n")
    with pytest.raises(DomainError, match="partition"):
        load_split_table(split_table)


def test_bag_rows_are_read_by_selection_only(shared: Cohort) -> None:
    src = shared.source()
    s = shared.slides[0]
    store = shared.stores[s.slide_id]
    rows = src.read(store, (0, 2, 3))
    assert rows.dtype == np.float32 and np.array_equal(rows, s.rows[[0, 2, 3]])
    assert src.rows_read == 3
    with pytest.raises(MILDatasetError):
        src.read(store, (3, 2))
    with pytest.raises(EmptyBagError):
        src.read(store, ())
