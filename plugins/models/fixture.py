"""`torch-fixture`: a lightweight local PyTorch encoder for tests. Random weights, no meaning.

It exercises the generic runtime end to end (local checkpoint, digest verification, strict
loading, device and precision selection, bounded inference, resource reporting) without any
pretrained model. Its checkpoint is written by `write_fixture_checkpoint` from a seeded
generator; the digest must be declared in the stage config (`checkpoint_sha256`), because the
fixture pins none.

Input: RGB, scaled to [0, 1], ImageNet-normalized, CHW float32, square `input_px` (default 224,
divisible by 32), any or unknown spacing. Output: `embedding_dim` (default 32) float32 rows.
Embeddings carry no biological or clinical meaning.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from domain.embedding import (
    EncoderIdentity,
    ModelInputSpec,
    Normalization,
    Precision,
    TensorLayout,
)
from domain.errors import ModelInputContractError, ProfileUnavailableError
from domain.foundation import (
    DeviceKind,
    EncoderDescriptor,
    PreprocessingProfile,
    ProfileStatus,
)
from domain.image import ChannelOrder
from domain.pipeline import ConfigValue
from domain.provenance import ComponentVersion
from plugins.models import precision_of, runtime_unavailable
from plugins.models.checkpoints import CheckpointLocator, expected_checkpoint, verify_checkpoint
from plugins.preprocessing.detection import cfg_int, cfg_str

if TYPE_CHECKING:
    from plugins.models.torch_runtime import TorchEncoder

TORCH_FIXTURE = "torch-fixture"
FIXTURE_VERSION = "1"
ARCHITECTURE = "conv8s8-relu-conv4s4-relu-avgpool-linear-tanh"
DEFAULT_DIM = 32
DEFAULT_SEED = 20261021
PROFILE = "fixture"
_IMAGENET = Normalization((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

DESCRIPTOR = EncoderDescriptor(
    name=TORCH_FIXTURE,
    adapter=ComponentVersion("encoder", TORCH_FIXTURE, FIXTURE_VERSION),
    architecture=ARCHITECTURE,
    output_dim=DEFAULT_DIM,
    output_dtype="float32",
    devices=frozenset({DeviceKind.CPU, DeviceKind.CUDA, DeviceKind.MPS}),
    precisions=frozenset({Precision.FP32}),
    profiles=(
        PreprocessingProfile(
            profile_id=PROFILE,
            version="1",
            official=False,
            status=ProfileStatus.AVAILABLE,
            description="RGB, [0, 1], ImageNet mean/std, CHW float32, square input_px; any spacing",
            evidence=("test fixture: the contract is its own definition",),
        ),
    ),
    synthetic=True,
    batch_invariant=False,
)


def fixture_input_spec(input_px: int) -> ModelInputSpec:
    if input_px < 32 or input_px % 32:
        raise ModelInputContractError("torch-fixture input_px must be a multiple of 32")
    return ModelInputSpec(
        width=input_px,
        height=input_px,
        channel_order=ChannelOrder.RGB,
        channels=3,
        dtype="float32",
        scaled_range=(0.0, 1.0),
        layout=TensorLayout.CHW,
        normalization=_IMAGENET,
        pixel_spacing_um=None,
        accepts_variable_spacing=True,
        accepts_unknown_spacing=True,
        size_divisor=32,
        precisions=frozenset({Precision.FP32}),
    )


def write_fixture_checkpoint(
    path: Path, *, embedding_dim: int = DEFAULT_DIM, seed: int = DEFAULT_SEED
) -> str:
    """Write a seeded random state dict to `path` and return its SHA-256. Needs torch."""
    try:
        import torch

        from plugins.models.fixture_net import build_network
    except ModuleNotFoundError as exc:
        raise runtime_unavailable(exc) from None
    from plugins.models.checkpoints import sha256_file

    generator = torch.Generator().manual_seed(seed)
    net = build_network(embedding_dim)
    with torch.no_grad():
        for p in net.parameters():
            p.copy_(torch.randn(p.shape, generator=generator) * 0.2)
    torch.save(net.state_dict(), path)
    return sha256_file(path)


class TorchFixtureFactory:
    def __init__(self, locator: CheckpointLocator) -> None:
        self._locator = locator

    @property
    def name(self) -> str:
        return TORCH_FIXTURE

    @property
    def descriptor(self) -> EncoderDescriptor:
        return DESCRIPTOR

    def input_spec(self, config: Mapping[str, ConfigValue]) -> ModelInputSpec:
        profile = cfg_str(config, "profile", PROFILE)
        if profile != PROFILE:
            raise ProfileUnavailableError(f"torch-fixture has no profile {profile!r}")
        return fixture_input_spec(cfg_int(config, "input_px", 224))

    def create(self, config: Mapping[str, ConfigValue]) -> TorchEncoder:
        spec = self.input_spec(config)
        dim = cfg_int(config, "embedding_dim", DEFAULT_DIM)
        try:
            from plugins.models import torch_runtime as rt
            from plugins.models.fixture_net import build_network, forward
        except ModuleNotFoundError as exc:
            raise runtime_unavailable(exc) from None
        expected = expected_checkpoint(DESCRIPTOR, config)
        path = self._locator.path_for(TORCH_FIXTURE)
        verified, stat = verify_checkpoint(path, expected)
        precision = precision_of(config)
        device = rt.select_device(cfg_str(config, "device", "cpu"), DESCRIPTOR)
        dtype = rt.select_dtype(precision, DESCRIPTOR)
        module = rt.load_module(
            lambda: build_network(dim),
            path,
            stat,
            architecture=ARCHITECTURE,
            device=device,
            dtype=dtype,
        )
        rt.probe_output(
            module,
            forward,
            spec,
            device=device,
            dtype=dtype,
            output_dim=dim,
            architecture=ARCHITECTURE,
        )
        versions = rt.package_versions("torch")
        identity = EncoderIdentity(
            component=DESCRIPTOR.adapter,
            model=None,
            weights_checksum=verified.checksum,
            synthetic=True,
            parameters=tuple(
                sorted(
                    (
                        ("architecture", ARCHITECTURE),
                        ("batch_size", cfg_int(config, "batch_size", 32)),
                        ("checkpoint_format", verified.format),
                        ("embedding_dim", dim),
                        ("input_px", spec.width),
                        ("profile", f"{PROFILE}@1"),
                        *((f"runtime.{n}", v) for n, v in versions),
                    )
                )
            ),
            output_dim=dim,
            output_dtype="float32",
        )
        return rt.TorchEncoder(
            module=module,
            extract=forward,
            identity=identity,
            spec=spec,
            device=device,
            dtype=dtype,
            activation_bytes_per_sample=2 * spec.patch_bytes,
            limitations=("torch fixture encoder: random weights; embeddings have no meaning",),
            resources=tuple((f"runtime.{n}", v) for n, v in versions),
        )
