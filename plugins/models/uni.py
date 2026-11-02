"""UNI (MahmoodLab) encoder adapter and its preprocessing profiles (SPEC-006 Phase C).

Everything model-specific about UNI lives here; the generic layers never name it.

Verified against the authors' documentation (discovery report 006 §3):

* architecture: `timm.create_model("vit_large_patch16_224", img_size=224, patch_size=16,
  init_values=1e-5, num_classes=0, dynamic_img_size=True)`, strict state-dict load of
  `pytorch_model.bin` (Hugging Face `MahmoodLab/UNI` @ `b55a5ec`, 1,213,527,781 bytes);
* output: the class token after the final norm, 1024 float32 values;
* preprocessing: `Resize(224)`, `CenterCrop(224)`, `ToTensor()`, ImageNet mean/std, on RGB;
* scale: pretraining data at "20x"; the authors give no micrometres per pixel;
* licence: CC-BY-NC-ND 4.0, gated, non-commercial academic research only; no redistribution.

Profiles (distinct identities, never interchangeable, never substituted for each other):

* `uni-official@1` (AVAILABLE): 224 x 224 patches extracted at a stored level, where `Resize(224)`
  and `CenterCrop(224)` are identities; RGB; [0, 1]; ImageNet normalization; CHW float32. Any
  declared resampling is refused: the official resize (torchvision bilinear, antialiased) is not
  an implemented transform, and area averaging is a different recipe. The physical scale is
  *not* documented as micrometres per pixel, so the profile requires the operator to bind it
  explicitly (`model_spacing_um`, `model_spacing_tolerance`); the binding is part of the identity.
* `uni-legacy-surgen@1` (BLOCKED): the legacy SurGen experiment (BGR, ImageNet constants applied
  in B, G, R positions, 1.0 um/px, level-0 regions of about 2013 px, uint8 `INTER_AREA` to 224).
  Its resampling is not implemented, so it is defined and identified but refused.

Weights are read only from a locally bound file whose SHA-256 must equal the pin below; nothing
is downloaded, and `HF_HUB_OFFLINE=1` is set before timm is imported.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from domain.embedding import (
    EncoderIdentity,
    ModelInputSpec,
    Normalization,
    Precision,
    TensorLayout,
)
from domain.errors import CheckpointIncompatibleError, ProfileUnavailableError
from domain.foundation import (
    CheckpointExpectation,
    DeviceKind,
    EncoderDescriptor,
    PreprocessingProfile,
    ProfileStatus,
)
from domain.ids import ModelId, Version
from domain.image import ChannelOrder
from domain.model import ModelRef
from domain.pipeline import ConfigValue
from domain.provenance import ComponentVersion
from plugins.models import precision_of, runtime_unavailable
from plugins.models.checkpoints import CheckpointLocator, expected_checkpoint, verify_checkpoint
from plugins.preprocessing.detection import cfg_int, cfg_str

if TYPE_CHECKING:
    import torch
    from torch import nn

    from plugins.models.torch_runtime import TorchEncoder

UNI = "uni"
ADAPTER_VERSION = "1"
INPUT_PX = 224
EMBEDDING_DIM = 1024
TIMM_ARCHITECTURE = "vit_large_patch16_224"
TIMM_KWARGS: dict[str, Any] = {
    "img_size": INPUT_PX,
    "patch_size": 16,
    "init_values": 1e-5,
    "num_classes": 0,
    "dynamic_img_size": True,
}
ARCHITECTURE = (
    f"timm/{TIMM_ARCHITECTURE}("
    + ",".join(f"{k}={v}" for k, v in sorted(TIMM_KWARGS.items()))
    + ");output=cls_token"
)
CHECKPOINT = CheckpointExpectation(
    sha256="56ef09b44a25dc5c7eedc55551b3d47bcd17659a7a33837cf9abc9ec4e2ffb40",
    format="torch-state-dict",
    size_bytes=1_213_527_781,
)
MODEL = ModelRef(ModelId("uni"), Version("mass100k.b55a5ec"))
IMAGENET = Normalization((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
LICENSE = "CC-BY-NC-ND-4.0"
ACCESS = (
    "gated (Hugging Face MahmoodLab/UNI): non-commercial academic research only; no "
    "redistribution of weights; derivatives are covered"
)

OFFICIAL = PreprocessingProfile(
    profile_id="uni-official",
    version="1",
    official=True,
    status=ProfileStatus.AVAILABLE,
    description=(
        "authors' recipe on 224x224 stored-level patches: RGB, ToTensor (/255), ImageNet "
        "mean/std, CHW float32; Resize(224)/CenterCrop(224) are identities on 224x224 input; "
        "physical scale bound explicitly by the operator (documented only as 20x)"
    ),
    evidence=(
        "huggingface.co/MahmoodLab/UNI model card (sha b55a5ec): ViT-L/16, 20x, class token, "
        "CC-BY-NC-ND 4.0",
        "github.com/mahmoodlab/UNI@42715ef README and uni/get_encoder/get_encoder.py: "
        "timm kwargs, strict load, Resize(224), CenterCrop(224), ImageNet mean/std",
    ),
)
LEGACY_SURGEN = PreprocessingProfile(
    profile_id="uni-legacy-surgen",
    version="1",
    official=False,
    status=ProfileStatus.BLOCKED,
    description=(
        "legacy SurGen compatibility (not the official recipe): BGR input, ImageNet constants "
        "applied in B,G,R positions, 1.0 um/px, level-0 regions of about 2013 px resized to 224 "
        "with cv2.INTER_AREA on uint8"
    ),
    evidence=(
        "legacy SurGen preprocessing contract surgen-sr1482-uni-224px-1mpp-v1 and its feature "
        "extraction code (read-only inspection; discovery report 006 section 2)",
    ),
    blocked_reason=(
        "the legacy resampling (cv2.INTER_AREA on uint8 with rounding, non-integer factor such "
        "as 2013 -> 224 px) is not an implemented transform; float integer-factor area "
        "averaging would be an approximation and is never substituted"
    ),
)
PROFILES = (OFFICIAL, LEGACY_SURGEN)

DESCRIPTOR = EncoderDescriptor(
    name=UNI,
    adapter=ComponentVersion("encoder", UNI, ADAPTER_VERSION),
    architecture=ARCHITECTURE,
    output_dim=EMBEDDING_DIM,
    output_dtype="float32",
    devices=frozenset({DeviceKind.CPU, DeviceKind.CUDA, DeviceKind.MPS}),
    # Reduced precision is not validated for UNI (SPEC-006 §9): FP32 only.
    precisions=frozenset({Precision.FP32}),
    profiles=PROFILES,
    synthetic=False,
    batch_invariant=False,
    checkpoint=CHECKPOINT,
    license=LICENSE,
    access=ACCESS,
)

# The legacy contract as declared (for identity and documentation); it is never executed.
LEGACY_SURGEN_INPUT_SPEC = ModelInputSpec(
    width=INPUT_PX,
    height=INPUT_PX,
    channel_order=ChannelOrder.BGR,
    channels=3,
    dtype="float32",
    scaled_range=(0.0, 1.0),
    layout=TensorLayout.CHW,
    normalization=IMAGENET,
    pixel_spacing_um=1.0,
    accepts_variable_spacing=False,
    accepts_unknown_spacing=False,
    spacing_tolerance=0.01,
    size_divisor=16,
    precisions=frozenset({Precision.FP32}),
)


def official_input_spec(spacing_um: float, tolerance: float) -> ModelInputSpec:
    return ModelInputSpec(
        width=INPUT_PX,
        height=INPUT_PX,
        channel_order=ChannelOrder.RGB,
        channels=3,
        dtype="float32",
        scaled_range=(0.0, 1.0),
        layout=TensorLayout.CHW,
        normalization=IMAGENET,
        pixel_spacing_um=spacing_um,
        accepts_variable_spacing=False,
        accepts_unknown_spacing=False,
        spacing_tolerance=tolerance,
        size_divisor=16,
        precisions=frozenset({Precision.FP32}),
    )


def _profile(config: Mapping[str, ConfigValue]) -> PreprocessingProfile:
    name = cfg_str(config, "profile", "")
    if not name:
        known = ", ".join(p.ref for p in PROFILES)
        raise ProfileUnavailableError(
            f"encoder uni needs an explicit preprocessing profile (profile: one of {known}); "
            "no profile is chosen by default"
        )
    pid, _, version = name.partition("@")
    for p in PROFILES:
        if p.profile_id == pid and version in ("", p.version):
            if p.status is ProfileStatus.BLOCKED:
                raise ProfileUnavailableError(f"profile {p.ref} is blocked: {p.blocked_reason}")
            return p
    raise ProfileUnavailableError(f"encoder uni has no preprocessing profile {name!r}")


def _number(config: Mapping[str, ConfigValue], key: str) -> float | None:
    value = config.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ProfileUnavailableError(f"{key} must be a number, got {value!r}")
    return float(value)


def _scale_binding(config: Mapping[str, ConfigValue]) -> tuple[float, float]:
    spacing = _number(config, "model_spacing_um")
    tolerance = _number(config, "model_spacing_tolerance")
    if spacing is None or tolerance is None:
        raise ProfileUnavailableError(
            "profile uni-official@1 needs an explicit physical-scale binding: the UNI authors "
            "document pretraining at 20x but state no micrometres per pixel. Declare "
            "model_spacing_um and model_spacing_tolerance (the operator's interpretation, "
            "recorded in the embedding identity); none is assumed"
        )
    return spacing, tolerance


class UniFactory:
    """`uni`: a pretrained ViT-L/16 loaded from a locally bound, digest-pinned checkpoint."""

    def __init__(
        self, locator: CheckpointLocator, *, checkpoint: CheckpointExpectation = CHECKPOINT
    ) -> None:
        # `checkpoint` is replaceable only at composition time (tests use small invalid files to
        # exercise the architecture checks); it is never read from a stage config.
        self._locator = locator
        self._descriptor = (
            DESCRIPTOR if checkpoint == CHECKPOINT else replace(DESCRIPTOR, checkpoint=checkpoint)
        )

    @property
    def name(self) -> str:
        return UNI

    @property
    def descriptor(self) -> EncoderDescriptor:
        return self._descriptor

    def input_spec(self, config: Mapping[str, ConfigValue]) -> ModelInputSpec:
        _profile(config)  # only the official profile is executable
        if cfg_str(config, "resample", "none") != "none":
            raise ProfileUnavailableError(
                "profile uni-official@1 resizes with torchvision Resize(224) (bilinear, "
                "antialiased), which is not an implemented transform; extract 224x224 patches "
                "at a stored level instead of declaring another resampling method"
            )
        return official_input_spec(*_scale_binding(config))

    def create(self, config: Mapping[str, ConfigValue]) -> TorchEncoder:
        spec = self.input_spec(config)
        profile = _profile(config)
        spacing, tolerance = _scale_binding(config)
        try:
            from plugins.models import torch_runtime as rt

            # Never reach the network: timm must not resolve a pretrained configuration.
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            timm = importlib.import_module("timm")
        except ModuleNotFoundError as exc:
            raise runtime_unavailable(exc) from None
        expected = expected_checkpoint(self._descriptor, config)
        path = self._locator.path_for(UNI)
        verified, stat = verify_checkpoint(path, expected)
        device = rt.select_device(cfg_str(config, "device", "cpu"), self._descriptor)
        dtype = rt.select_dtype(precision_of(config), self._descriptor)

        def build() -> nn.Module:
            model: nn.Module = timm.create_model(TIMM_ARCHITECTURE, pretrained=False, **TIMM_KWARGS)
            return model

        module = rt.load_module(
            build, path, stat, architecture=ARCHITECTURE, device=device, dtype=dtype
        )
        if getattr(module, "num_features", None) != EMBEDDING_DIM or (
            getattr(module, "global_pool", None) != "token"
        ):
            raise CheckpointIncompatibleError(
                f"{TIMM_ARCHITECTURE} does not expose a {EMBEDDING_DIM}-d class token"
            )
        rt.probe_output(
            module,
            class_token,
            spec,
            device=device,
            dtype=dtype,
            output_dim=EMBEDDING_DIM,
            architecture=ARCHITECTURE,
        )
        versions = rt.package_versions("torch", "timm")
        identity = EncoderIdentity(
            component=self._descriptor.adapter,
            model=MODEL,
            weights_checksum=verified.checksum,
            synthetic=False,
            parameters=tuple(
                sorted(
                    (
                        ("architecture", ARCHITECTURE),
                        # Torch cannot guarantee batch invariance across hardware (ADR-006 D9).
                        ("batch_size", cfg_int(config, "batch_size", 32)),
                        ("checkpoint_format", verified.format),
                        ("model_spacing_tolerance", tolerance),
                        ("model_spacing_um", spacing),
                        ("output", "cls_token"),
                        ("profile", profile.ref),
                        *((f"runtime.{n}", v) for n, v in versions),
                    )
                )
            ),
            output_dim=EMBEDDING_DIM,
            output_dtype="float32",
        )
        return rt.TorchEncoder(
            module=module,
            extract=class_token,
            identity=identity,
            spec=spec,
            device=device,
            dtype=dtype,
            activation_bytes_per_sample=rt.vit_activation_bytes(
                tokens=(INPUT_PX // 16) ** 2 + 1, dim=EMBEDDING_DIM, heads=16, mlp_ratio=4.0
            ),
            limitations=(
                f"UNI weights are licensed {LICENSE} (non-commercial academic research only); "
                "embeddings are derivatives covered by that licence",
                f"physical scale bound by the operator: {spacing} um/px within "
                f"{tolerance:.2%}, interpreting the documented 20x; the UNI authors state no "
                "um/px",
                "official Resize(224)/CenterCrop(224) are identities on the 224x224 patches "
                "used; ToTensor divides by 255 where the declared scale step multiplies by "
                "float32(1/255) (at most 1 ulp difference)",
            ),
            resources=(
                *((f"runtime.{n}", v) for n, v in versions),
                ("checkpoint_bytes", verified.size_bytes),
            ),
        )


def class_token(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """The class token after the final norm (= the pooled output of `num_classes=0`)."""
    forward_features = getattr(module, "forward_features")  # noqa: B009 (timm API, untyped)
    tokens: torch.Tensor = forward_features(x)
    return tokens[:, 0]
