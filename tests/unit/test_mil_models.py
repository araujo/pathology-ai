"""SPEC-007 MIL model contracts: masks, padding invariance, attention, gradients, determinism.

Inputs are random synthetic arrays; no result here has any biological meaning.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="optional extra pathology-ai[embedding]")

from domain.errors import (  # noqa: E402
    CheckpointIncompatibleError,
    EmbeddingOutputError,
    EmptyBagError,
    ExecutionUnsupportedError,
    ModelInputContractError,
)
from domain.mil import BagSelection  # noqa: E402
from domain.mil_training import LossKind, MILModelSpec  # noqa: E402
from plugins.mil import ATTENTION_MIL, DESCRIPTORS, MEAN_POOLING, TorchMILFactory  # noqa: E402
from plugins.mil.batching import NumpyBatchBuilder  # noqa: E402
from plugins.mil.torch_models import TorchMILModel  # noqa: E402
from ports import MILBatch, MILModel, MILModelFactory  # noqa: E402
from tests.mil_support import BINARY, MULTICLASS, training_config  # noqa: E402

DIM = 16
ARCHS = [MEAN_POOLING, ATTENTION_MIL]


def _spec(arch: str, task=BINARY, dim: int = DIM) -> MILModelSpec:  # type: ignore[no-untyped-def]
    hp = (("attention_dim", 8), ("hidden_dim", 12)) if arch == ATTENTION_MIL else ()
    return DESCRIPTORS[arch].spec(dim, task, hp)


def _model(arch: str, task=BINARY, seed: int = 0, train: bool = False) -> TorchMILModel:  # type: ignore[no-untyped-def]
    loss = LossKind.BCE_WITH_LOGITS if task.binary else LossKind.CROSS_ENTROPY
    return TorchMILFactory(arch).create(
        _spec(arch, task), loss=loss, seed=seed, training=training_config() if train else None
    )


def _sel(slide: str, n: int) -> BagSelection:
    return BagSelection(slide, n, tuple(range(n)), "sha256-rank@1", 0, None)


def _batch(bags: list[np.ndarray], targets: tuple[int, ...] | None = None) -> MILBatch:
    return NumpyBatchBuilder().build(
        [(_sel(f"s{i}", len(b)), b) for i, b in enumerate(bags)], targets, DIM
    )


def _bags(*sizes: int, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.standard_normal((n, DIM)).astype(np.float32) for n in sizes]


def test_factories_satisfy_the_ports_without_importing_a_framework_to_list() -> None:
    for arch in ARCHS:
        f = TorchMILFactory(arch)
        assert isinstance(f, MILModelFactory)
        assert f.descriptor.name == arch and f.descriptor.version == "1"
        assert dict(f.runtime("cpu"))["device"] == "cpu"
    assert isinstance(_model(MEAN_POOLING), MILModel)
    assert DESCRIPTORS[ATTENTION_MIL].produces_attention
    assert not DESCRIPTORS[MEAN_POOLING].produces_attention


def test_hyperparameters_are_validated() -> None:
    with pytest.raises(ModelInputContractError, match="hiden_dim"):
        DESCRIPTORS[ATTENTION_MIL].spec(DIM, BINARY, (("hiden_dim", 4),))
    with pytest.raises(ModelInputContractError, match="must be int"):
        DESCRIPTORS[ATTENTION_MIL].spec(DIM, BINARY, (("hidden_dim", "4"),))
    spec = DESCRIPTORS[MEAN_POOLING].spec(DIM, MULTICLASS, (("dropout", 0),))
    assert spec.hyperparameter("dropout", -1.0) == 0.0 and spec.num_outputs == 3


def test_masks_and_padding_are_validated() -> None:
    b = _bags(3)
    with pytest.raises(EmbeddingOutputError):
        NumpyBatchBuilder().build([(_sel("s", 4), b[0])], None, DIM)  # selection/rows mismatch
    with pytest.raises(EmbeddingOutputError):
        NumpyBatchBuilder().build([(_sel("s", 3), b[0].astype(np.float64))], None, DIM)
    bad = b[0].copy()
    bad[0, 0] = np.inf
    with pytest.raises(EmbeddingOutputError):
        NumpyBatchBuilder().build([(_sel("s", 3), bad)], None, DIM)
    batch = _batch(_bags(3, 7, 1))
    mask = np.asarray(batch.mask)
    assert mask.shape == (3, 7) and mask.sum(axis=1).tolist() == [3, 7, 1]
    assert not np.asarray(batch.features)[0, 3:].any()  # padding is zero
    assert batch.padded_bytes == 3 * 7 * DIM * 4 + 3 * 7
    empty = MILBatch(("s",), np.zeros((1, 2, DIM), np.float32), np.zeros((1, 2), bool), (), None, 0)
    with pytest.raises(EmptyBagError):
        _model(MEAN_POOLING).predict(empty)


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("task", [BINARY, MULTICLASS], ids=["binary", "multiclass"])
def test_output_shapes(arch: str, task) -> None:  # type: ignore[no-untyped-def]
    out = _model(arch, task).predict(_batch(_bags(4, 9, 2)))
    assert len(out.logits) == 3 and all(len(r) == task.num_outputs for r in out.logits)
    if arch == ATTENTION_MIL:
        assert out.attention is not None and [len(a) for a in out.attention] == [4, 9, 2]
    else:
        assert out.attention is None


def test_mean_pooling_averages_valid_instances_only() -> None:
    m = _model(MEAN_POOLING)
    bags = _bags(5, 2)
    classifier = m.module.get_submodule("classifier")
    expected = [classifier(torch.from_numpy(b.mean(axis=0))).item() for b in bags]
    out = m.predict(_batch(bags))
    assert [r[0] for r in out.logits] == pytest.approx(expected, abs=1e-6)


def test_attention_is_normalized_over_valid_instances_and_zero_on_padding() -> None:
    m = _model(ATTENTION_MIL)
    batch = _batch(_bags(3, 10))
    x = torch.from_numpy(np.asarray(batch.features))
    mask = torch.from_numpy(np.asarray(batch.mask))
    with torch.no_grad():
        _, weights = m.module(x, mask)
    assert torch.all(weights[0, 3:] == 0)  # exactly zero, not merely small
    assert weights.sum(dim=1).tolist() == pytest.approx([1.0, 1.0], abs=1e-6)
    assert torch.all(weights[mask] > 0)


@pytest.mark.parametrize("arch", ARCHS)
def test_padding_does_not_change_outputs(arch: str) -> None:
    m = _model(arch)
    small, large = _bags(4, 50, seed=3)
    alone = m.predict(_batch([small]))
    padded = m.predict(_batch([small, large]))
    assert padded.logits[0] == pytest.approx(alone.logits[0], abs=1e-6)
    if arch == ATTENTION_MIL:
        assert padded.attention is not None and alone.attention is not None
        assert padded.attention[0] == pytest.approx(alone.attention[0], abs=1e-6)


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("task", [BINARY, MULTICLASS], ids=["binary", "multiclass"])
def test_gradients_reach_every_parameter_and_training_lowers_the_loss(arch: str, task) -> None:  # type: ignore[no-untyped-def]
    m = _model(arch, task, train=True)
    before = {k: v.clone() for k, v in m.module.state_dict().items()}
    batch = _batch(_bags(6, 9, 3, 12), (0, 1, 1, 0) if task.binary else (0, 1, 2, 0))
    first = m.train_batch(batch)
    changed = [k for k, v in m.module.state_dict().items() if not torch.equal(v, before[k])]
    assert sorted(changed) == sorted(before)  # every parameter received a gradient step
    for _ in range(60):
        last = m.train_batch(batch)
    assert last < first


@pytest.mark.parametrize("arch", ARCHS)
def test_initialization_is_seeded_and_isolated_from_the_global_generator(arch: str) -> None:
    def weights(seed: int) -> bytes:
        return b"".join(t.data for t in _model(arch, seed=seed).state())

    torch.manual_seed(123)
    state = torch.get_rng_state()
    a = weights(1)
    assert torch.equal(torch.get_rng_state(), state)  # global generator untouched
    torch.manual_seed(999)
    assert weights(1) == a
    assert weights(2) != a


@pytest.mark.parametrize("arch", ARCHS)
def test_state_round_trips_and_incompatible_states_are_rejected(arch: str) -> None:
    trained = _model(arch, train=True)
    trained.train_batch(_batch(_bags(4, 6), (0, 1)))
    fresh = _model(arch, seed=5)
    fresh.load_state(trained.state())
    batch = _batch(_bags(5, 3, seed=9))
    assert fresh.predict(batch) == trained.predict(batch)
    other = TorchMILFactory(arch).create(
        _spec(arch, MULTICLASS), loss=LossKind.CROSS_ENTROPY, seed=0, training=None
    )
    with pytest.raises(CheckpointIncompatibleError):
        other.load_state(trained.state())
    with pytest.raises(CheckpointIncompatibleError):
        fresh.load_state(trained.state()[1:])
    wrong_dim = TorchMILFactory(arch).create(
        _spec(arch, BINARY, dim=DIM + 1), loss=LossKind.BCE_WITH_LOGITS, seed=0, training=None
    )
    with pytest.raises(CheckpointIncompatibleError):
        wrong_dim.load_state(trained.state())
    with pytest.raises(CheckpointIncompatibleError):
        wrong_dim.predict(batch)  # feature dimension mismatch


def test_factory_refuses_other_architectures_and_inference_models_do_not_train() -> None:
    with pytest.raises(CheckpointIncompatibleError):
        TorchMILFactory(MEAN_POOLING).create(
            _spec(ATTENTION_MIL), loss=LossKind.BCE_WITH_LOGITS, seed=0, training=None
        )
    m = _model(MEAN_POOLING)
    with pytest.raises(Exception, match="inference only"):
        m.train_batch(_batch(_bags(3), (1,)))
    from bootstrap import build_mil_architectures

    with pytest.raises(ExecutionUnsupportedError):
        build_mil_architectures().resolve("transformer-mil")
