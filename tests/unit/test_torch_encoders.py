"""SPEC-006 generic torch runtime and adapters with lightweight local weights (needs torch).

The `torch-fixture` checkpoint is written from a seeded generator into a temporary directory;
its embeddings have no meaning. UNI architecture checks use small invalid checkpoints and a
`meta`-device model, so no real UNI weights are needed here (see the integration tests).
"""

from __future__ import annotations

import importlib.metadata
import socket
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="optional extra pathology-ai[embedding]")

from domain.embedding import Precision  # noqa: E402
from domain.errors import (  # noqa: E402
    CheckpointIncompatibleError,
    CheckpointIntegrityError,
    DeviceUnavailableError,
    ModelInferenceError,
    ModelInputContractError,
    UnsupportedModelInputError,
)
from domain.foundation import CheckpointExpectation  # noqa: E402
from domain.pipeline import ConfigValue  # noqa: E402
from plugins.models.checkpoints import CheckpointLocator, CheckpointStat, sha256_file  # noqa: E402
from plugins.models.fixture import (  # noqa: E402
    DESCRIPTOR,
    TorchFixtureFactory,
    write_fixture_checkpoint,
)
from plugins.models.fixture_net import build_network  # noqa: E402
from plugins.models.torch_runtime import TorchEncoder, select_device, select_dtype  # noqa: E402
from ports import EmbeddingEncoder, EncodeContext  # noqa: E402

pytestmark = pytest.mark.filterwarnings("error")


class NotATensor:
    """Pickled into a checkpoint to prove `weights_only` refuses arbitrary objects."""


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    path = tmp_path_factory.mktemp("fx") / "fixture.pt"
    return path, write_fixture_checkpoint(path)


def _create(path: Path, sha: str, **cfg: Any) -> TorchEncoder:
    factory = TorchFixtureFactory(CheckpointLocator({"torch-fixture": path}))
    return factory.create({"checkpoint_sha256": sha, **cfg})


@pytest.fixture
def encoder(ckpt: tuple[Path, str]) -> Iterator[TorchEncoder]:
    enc = _create(*ckpt)
    yield enc
    enc.close()


def _batch(n: int = 3, px: int = 224, seed: int = 0) -> np.ndarray[Any, Any]:
    return np.random.default_rng(seed).standard_normal((n, 3, px, px)).astype(np.float32)


def _ctx(n: int = 3, device: str = "cpu") -> EncodeContext:
    return EncodeContext(0, 0, tuple(f"p{i}" for i in range(n)), Precision.FP32, device)


def test_fixture_encoder_satisfies_the_port(encoder: TorchEncoder) -> None:
    assert isinstance(encoder, EmbeddingEncoder)
    rows = encoder.encode_batch(_batch(), _ctx())
    assert rows.shape == (3, 32) and rows.dtype == np.float32
    assert np.isfinite(rows).all()


def test_model_is_in_eval_mode_without_gradients(encoder: TorchEncoder) -> None:
    module = encoder.module
    assert not module.training and not any(m.training for m in module.modules())
    assert not any(p.requires_grad for p in module.parameters())
    encoder.encode_batch(_batch(), _ctx())
    assert all(p.grad is None for p in module.parameters())


def test_cpu_inference_is_deterministic_and_row_independent(ckpt: tuple[Path, str]) -> None:
    a, b = _create(*ckpt), _create(*ckpt)
    x = _batch(4)
    ra, rb = a.encode_batch(x, _ctx(4)), b.encode_batch(x, _ctx(4))
    assert np.array_equal(ra, rb)
    single = np.concatenate([a.encode_batch(x[i : i + 1], _ctx(1)) for i in range(4)])
    np.testing.assert_allclose(single, ra, rtol=0, atol=1e-6)  # tolerance, not identity
    assert a.identity == b.identity


def test_identity_and_output_change_with_the_checkpoint(
    ckpt: tuple[Path, str], tmp_path: Path
) -> None:
    other = tmp_path / "other.pt"
    sha = write_fixture_checkpoint(other, seed=7)
    a, b = _create(*ckpt), _create(other, sha)
    assert a.identity.weights_checksum != b.identity.weights_checksum
    assert a.identity != b.identity
    x = _batch()
    assert not np.array_equal(a.encode_batch(x, _ctx()), b.encode_batch(x, _ctx()))


def test_batch_configuration_is_part_of_the_identity(ckpt: tuple[Path, str]) -> None:
    assert _create(*ckpt, batch_size=8).identity != _create(*ckpt, batch_size=16).identity


def test_runtime_versions_are_part_of_the_identity(encoder: TorchEncoder) -> None:
    params = dict(encoder.identity.parameters)
    assert params["runtime.torch"] == importlib.metadata.version("torch")
    assert params["checkpoint_format"] == "torch-state-dict"


@pytest.mark.parametrize(
    "batch,ctx",
    [
        (np.zeros((2, 3, 224, 224), np.float64), _ctx(2)),
        (np.zeros((2, 3, 256, 256), np.float32), _ctx(2)),
        (np.zeros((2, 224, 224, 3), np.float32), _ctx(2)),
        (np.zeros((2, 3, 224, 224), np.float32), _ctx(3)),
        (np.zeros((2, 3, 224, 224), np.float32), _ctx(2, device="mps")),
    ],
)
def test_input_batches_are_validated(encoder: TorchEncoder, batch: Any, ctx: EncodeContext) -> None:
    with pytest.raises(UnsupportedModelInputError):
        encoder.encode_batch(batch, ctx)


def test_output_dimension_mismatch_is_rejected(ckpt: tuple[Path, str]) -> None:
    with pytest.raises(CheckpointIncompatibleError, match="shape"):
        _create(*ckpt, embedding_dim=16)


def _write_state(path: Path, state: dict[str, Any]) -> str:
    torch.save(state, path)
    return sha256_file(path)


def test_architecture_mismatch_is_rejected(tmp_path: Path) -> None:
    state = build_network(32).state_dict()
    missing = dict(state)
    missing.pop("6.bias")
    path = tmp_path / "m.pt"
    with pytest.raises(CheckpointIncompatibleError, match="missing: 6.bias"):
        _create(path, _write_state(path, missing))
    extra = {**state, "head.weight": torch.zeros(2)}
    with pytest.raises(CheckpointIncompatibleError, match="unexpected: head.weight"):
        _create(path, _write_state(path, extra))
    half = {k: v.half() for k, v in state.items()}
    with pytest.raises(CheckpointIncompatibleError, match="dtype"):
        _create(path, _write_state(path, half))


def test_unreadable_or_unsafe_checkpoints_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "garbage.pt"
    path.write_bytes(b"\x00not a checkpoint" * 100)
    with pytest.raises(CheckpointIncompatibleError, match="weights-only"):
        _create(path, sha256_file(path))
    with pytest.raises(CheckpointIncompatibleError, match="weights-only"):
        _create(path, _write_state(path, {"x": NotATensor()}))
    with pytest.raises(CheckpointIncompatibleError, match="flat state dict"):
        _create(path, _write_state(path, {"nested": {"w": torch.zeros(1)}}))


def test_checksum_mismatch_is_rejected_before_deserialization(
    ckpt: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[object] = []
    monkeypatch.setattr(torch, "load", lambda *a, **k: calls.append(a))
    with pytest.raises(CheckpointIntegrityError):
        _create(ckpt[0], "0" * 64)
    assert calls == []


def test_a_checkpoint_replaced_after_hashing_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "w.pt"
    path.write_bytes(b"a")
    stat = CheckpointStat.of(path)
    path.write_bytes(b"bb")
    with pytest.raises(CheckpointIntegrityError, match="changed on disk"):
        stat.unchanged(path)


def test_device_and_precision_selection_fail_closed() -> None:
    assert select_device("cpu", DESCRIPTOR) == torch.device("cpu")
    if not torch.cuda.is_available():
        with pytest.raises(DeviceUnavailableError, match="CUDA is not available"):
            select_device("cuda", DESCRIPTOR)
    with pytest.raises(ModelInputContractError):
        select_device("tpu", DESCRIPTOR)
    assert select_dtype(Precision.FP32, DESCRIPTOR) == torch.float32
    with pytest.raises(DeviceUnavailableError, match="precision fp16"):
        select_dtype(Precision.FP16, DESCRIPTOR)


def test_unknown_precision_is_rejected(ckpt: tuple[Path, str]) -> None:
    with pytest.raises(ModelInputContractError, match="precision"):
        _create(*ckpt, precision="fp8")


def test_runtime_failures_are_typed(encoder: TorchEncoder, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(module: Any, x: Any) -> Any:
        raise RuntimeError("CUDA error: device-side assert triggered")

    monkeypatch.setattr(encoder, "_extract", boom)
    with pytest.raises(ModelInferenceError, match="device-side assert"):
        encoder.encode_batch(_batch(), _ctx())


def test_close_releases_the_model_and_is_idempotent(ckpt: tuple[Path, str]) -> None:
    import weakref

    enc = _create(*ckpt)
    ref = weakref.ref(enc.module)
    enc.close()
    enc.close()
    assert enc.closed and ref() is None
    with pytest.raises(ModelInferenceError, match="closed"):
        enc.encode_batch(_batch(), _ctx())


def test_resources_are_reported_separately_from_the_budget(encoder: TorchEncoder) -> None:
    encoder.encode_batch(_batch(), _ctx())
    report = encoder.runtime_report()
    resources = dict(report.resources)
    assert resources["parameter_bytes"] == sum(p.numel() * 4 for p in encoder.module.parameters())
    assert resources["samples_encoded"] == 3 and resources["device"] == "cpu"
    assert report.limitations  # deterministic statements for the manifest
    # On CPU, activations are host memory and enter the host budget.
    assert encoder.working_bytes(8) > encoder.working_bytes(1) > 0


def test_loading_and_inference_use_no_network(
    ckpt: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_network(*a: Any, **k: Any) -> Any:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    enc = _create(*ckpt)
    enc.encode_batch(_batch(), _ctx())
    enc.close()


def test_uni_architecture_mismatch_is_rejected_without_real_weights(
    ckpt: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("timm")
    from plugins.models.uni import UniFactory

    path, sha = ckpt
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: 1 / 0)
    exp = CheckpointExpectation(sha, "torch-state-dict", path.stat().st_size)
    factory = UniFactory(CheckpointLocator({"uni": path}), checkpoint=exp)
    cfg: dict[str, ConfigValue] = {
        "profile": "uni-official@1",
        "model_spacing_um": 0.5,
        "model_spacing_tolerance": 0.1,
    }
    with pytest.raises(CheckpointIncompatibleError, match="missing: blocks.0.attn"):
        factory.create(cfg)


def test_uni_rejects_wrong_shapes_in_a_vit_l_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    timm = pytest.importorskip("timm")
    from plugins.models.uni import TIMM_ARCHITECTURE, TIMM_KWARGS, UniFactory

    with torch.device("meta"):
        expected = timm.create_model(TIMM_ARCHITECTURE, pretrained=False, **TIMM_KWARGS)
    # A ViT-B-sized embedding in the ViT-L key layout: same keys, wrong shapes (small file).
    state = {k: torch.zeros(1) for k in expected.state_dict()}
    path = tmp_path / "vit-b-like.pt"
    torch.save(state, path)
    exp = CheckpointExpectation(sha256_file(path), "torch-state-dict", path.stat().st_size)
    factory = UniFactory(CheckpointLocator({"uni": path}), checkpoint=exp)
    cfg: dict[str, ConfigValue] = {
        "profile": "uni-official@1",
        "model_spacing_um": 0.5,
        "model_spacing_tolerance": 0.1,
    }
    with pytest.raises(CheckpointIncompatibleError, match="shape: "):
        factory.create(cfg)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available: AC-012 unverified")
def test_cuda_inference_matches_cpu_within_tolerance(ckpt: tuple[Path, str]) -> None:
    cpu, cuda = _create(*ckpt), _create(*ckpt, device="cuda")
    x = _batch()
    np.testing.assert_allclose(
        cuda.encode_batch(x, _ctx(device="cuda")), cpu.encode_batch(x, _ctx()), atol=1e-4
    )
    assert int(dict(cuda.runtime_report().resources)["device_peak_bytes"]) > 0


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available (optional)")
def test_mps_inference_matches_cpu_within_tolerance(ckpt: tuple[Path, str]) -> None:
    cpu, mps = _create(*ckpt), _create(*ckpt, device="mps")
    x = _batch()
    np.testing.assert_allclose(
        mps.encode_batch(x, _ctx(device="mps")), cpu.encode_batch(x, _ctx()), atol=1e-4
    )
    assert int(dict(mps.runtime_report().resources)["device_peak_bytes"]) > 0
