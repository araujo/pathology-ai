"""SPEC-007 checkpoint persistence (AC-014): pickle-free, verified, never silently overwritten."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from domain.errors import (
    CheckpointIncompatibleError,
    CheckpointIntegrityError,
    CheckpointMissingError,
    EmbeddingCompatibilityError,
    ExperimentConflictError,
)
from domain.mil_training import (
    CheckpointDescriptor,
    EpochRecord,
    ExperimentSpec,
    LossKind,
    MILModelSpec,
    TrainingHistory,
    index_tensors,
)
from infrastructure.mil_artifacts import DESCRIPTOR, WEIGHTS, DirectoryCheckpointStore
from ports import MILCheckpointStore, ModelTensor
from tests.mil_support import BINARY, MULTICLASS, training_config

TENSORS = (
    ModelTensor("classifier.bias", "float32", (1,), b"\x00\x00\x80?"),
    ModelTensor("classifier.weight", "float32", (1, 2), b"\x00\x00\x80?\x00\x00\x00@"),
)


def _experiment(**changes: object) -> ExperimentSpec:
    base = ExperimentSpec(
        name="contract",
        task=BINARY,
        model=MILModelSpec("mean-pooling", "1", 2, 1, (("dropout", 0.0),)),
        training=training_config(),
        dataset_ref="d@1",
        dataset_digest="a" * 64,
        split_digest="b" * 64,
        compatibility_policy="exact@1",
        compatibility_key="pai-embkey-" + "c" * 32,
        loss=LossKind.BCE_WITH_LOGITS,
        runtime=(("device", "cpu"), ("runtime.torch", "x")),
    )
    return replace(base, **changes)  # type: ignore[arg-type]


def _descriptor(
    tensors: tuple[ModelTensor, ...] = TENSORS, exp: ExperimentSpec | None = None
) -> CheckpointDescriptor:
    exp = exp or _experiment()
    entries, sha, size = index_tensors([(t.name, t.dtype, t.shape, t.data) for t in tensors])
    return CheckpointDescriptor(
        experiment=exp,
        experiment_identity=exp.identity,
        tensors=entries,
        weights_sha256=sha,
        weights_bytes=size,
        history=TrainingHistory(
            (EpochRecord(0, 0.7, 0.6, 0.6, None, True),), 0, 0.6, "validation_loss", False
        ),
        parameter_count=3,
        limitations=("contract test",),
    )


def test_save_load_round_trip_and_identity(tmp_path: Path) -> None:
    store = DirectoryCheckpointStore(tmp_path)
    assert isinstance(store, MILCheckpointStore)
    d = _descriptor()
    assert store.find(d.experiment_identity) is None
    store.save(d, TENSORS)
    assert store.find(d.experiment_identity) == d
    loaded, tensors = store.load(d.experiment_identity)
    assert loaded == d and tensors == TENSORS
    assert loaded.checkpoint_id.startswith("pai-milckpt-") and len(loaded.digest) == 64
    assert (tmp_path / d.experiment_identity / WEIGHTS).read_bytes() == b"".join(
        t.data for t in TENSORS
    )
    assert b"pickle" not in (tmp_path / d.experiment_identity / DESCRIPTOR).read_bytes()
    store.save(d, TENSORS)  # identical: idempotent


def test_identity_binds_everything_the_result_depends_on() -> None:
    base = _experiment().identity
    for change in (
        {"dataset_digest": "e" * 64},
        {"split_digest": "e" * 64},
        {"compatibility_key": "pai-embkey-" + "d" * 32},
        {"training": training_config(seed=14)},
        {"runtime": (("device", "cpu"), ("runtime.torch", "y"))},
        {"model": MILModelSpec("mean-pooling", "1", 2, 1, (("dropout", 0.1),))},
    ):
        assert _experiment(**change).identity != base, change


@pytest.mark.parametrize("damage", ["flip", "truncate", "extend"])
def test_corrupt_weights_are_rejected(tmp_path: Path, damage: str) -> None:
    store = DirectoryCheckpointStore(tmp_path)
    d = _descriptor()
    store.save(d, TENSORS)
    path = tmp_path / d.experiment_identity / WEIGHTS
    raw = bytearray(path.read_bytes())
    if damage == "flip":
        raw[3] ^= 0x01
    elif damage == "truncate":
        raw = raw[:-1]
    else:
        raw += b"\x00"
    path.write_bytes(bytes(raw))
    with pytest.raises(CheckpointIntegrityError):
        store.load(d.experiment_identity)
    with pytest.raises(CheckpointIntegrityError):
        store.find(d.experiment_identity)


def test_tampered_or_foreign_descriptors_are_rejected(tmp_path: Path) -> None:
    store = DirectoryCheckpointStore(tmp_path)
    d = _descriptor()
    store.save(d, TENSORS)
    path = tmp_path / d.experiment_identity / DESCRIPTOR
    doc = json.loads(path.read_text())
    doc["body"]["history"]["best_epoch"] = 3
    path.write_text(json.dumps(doc))
    with pytest.raises(CheckpointIntegrityError, match="digest"):
        store.load(d.experiment_identity)
    # A self-consistent document of another experiment, moved into this directory.
    other = _descriptor(exp=_experiment(name="other"))
    other_store = DirectoryCheckpointStore(tmp_path / "o")
    other_store.save(other, TENSORS)
    (tmp_path / d.experiment_identity / DESCRIPTOR).write_bytes(
        (tmp_path / "o" / other.experiment_identity / DESCRIPTOR).read_bytes()
    )
    with pytest.raises(CheckpointIntegrityError, match="another experiment"):
        store.load(d.experiment_identity)


def test_a_different_result_never_overwrites_an_experiment(tmp_path: Path) -> None:
    store = DirectoryCheckpointStore(tmp_path)
    store.save(_descriptor(), TENSORS)
    changed = (TENSORS[0], replace(TENSORS[1], data=b"\x00\x00\x80?\x00\x00@@"))
    with pytest.raises(ExperimentConflictError, match="never overwritten"):
        store.save(_descriptor(changed), changed)
    assert store.load(_descriptor().experiment_identity)[1] == TENSORS


def test_interrupted_writes_are_not_checkpoints(tmp_path: Path) -> None:
    store = DirectoryCheckpointStore(tmp_path)
    d = _descriptor()
    directory = tmp_path / d.experiment_identity
    directory.mkdir()
    (directory / WEIGHTS).write_bytes(b"partial")
    assert store.find(d.experiment_identity) is None
    with pytest.raises(CheckpointMissingError):
        store.load(d.experiment_identity)
    store.save(d, TENSORS)  # the interrupted write is replaced
    assert store.load(d.experiment_identity)[1] == TENSORS
    (tmp_path / "x").mkdir()
    other = _descriptor(exp=_experiment(name="other"))
    (tmp_path / other.experiment_identity).mkdir()
    (tmp_path / other.experiment_identity / "notes.txt").write_text("mine")
    with pytest.raises(ExperimentConflictError, match="unrelated"):
        store.save(other, TENSORS)


def test_descriptor_and_index_validation() -> None:
    with pytest.raises(CheckpointIncompatibleError):
        index_tensors([("w", "float32", (2,), b"\x00" * 4)])  # size mismatch
    with pytest.raises(CheckpointIncompatibleError):
        index_tensors([("w", "complex64", (1,), b"\x00" * 8)])
    with pytest.raises(CheckpointIncompatibleError):
        replace(_descriptor(), experiment_identity="pai-milexp-" + "0" * 32)
    with pytest.raises(CheckpointIntegrityError):
        DirectoryCheckpointStore(Path(".")).save(
            replace(_descriptor(), weights_sha256="0" * 64), TENSORS
        )
    with pytest.raises(CheckpointMissingError):
        DirectoryCheckpointStore(Path(".")).find("../escape")


def test_loads_reject_incompatible_bags_and_tasks() -> None:
    d = _descriptor()
    d.require_accepts(d.experiment.compatibility_key, 2, BINARY)
    with pytest.raises(EmbeddingCompatibilityError):
        d.require_accepts("pai-embkey-" + "f" * 32, 2, BINARY)
    with pytest.raises(CheckpointIncompatibleError, match="features"):
        d.require_accepts(d.experiment.compatibility_key, 3, BINARY)
    with pytest.raises(CheckpointIncompatibleError, match="predicts"):
        d.require_accepts(d.experiment.compatibility_key, 2, MULTICLASS)
