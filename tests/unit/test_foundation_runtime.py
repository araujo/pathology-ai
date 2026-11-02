"""SPEC-006 generic runtime contracts that need no deep-learning framework.

Checkpoint binding and verification, the encoder registry, descriptors and the UNI profile
contracts are all resolved without importing torch or loading any weights.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from application.registries import EncoderRegistry
from bootstrap import build_platform
from domain.embedding import Precision
from domain.errors import (
    CheckpointIntegrityError,
    CheckpointMissingError,
    DuplicateRegistrationError,
    ExecutionUnsupportedError,
    ModelInputContractError,
    ProfileUnavailableError,
)
from domain.foundation import (
    CheckpointExpectation,
    DeviceKind,
    PreprocessingProfile,
    ProfileStatus,
    parse_device,
)
from domain.image import ChannelOrder
from domain.pipeline import ConfigValue
from plugins.embedding.reference import ReferenceProjectionFactory
from plugins.models.checkpoints import (
    CHECKPOINTS_ENV,
    CheckpointLocator,
    expected_checkpoint,
    parse_bindings,
    verify_checkpoint,
)
from plugins.models.fixture import TorchFixtureFactory
from plugins.models.uni import (
    DESCRIPTOR as UNI_DESCRIPTOR,
)
from plugins.models.uni import (
    LEGACY_SURGEN,
    LEGACY_SURGEN_INPUT_SPEC,
    OFFICIAL,
    UniFactory,
    official_input_spec,
)
from tests.conftest import REPO_ROOT

OFFICIAL_CFG: dict[str, ConfigValue] = {
    "profile": "uni-official@1",
    "model_spacing_um": 0.5,
    "model_spacing_tolerance": 0.15,
}


def _file(tmp_path: Path, data: bytes = b"weights") -> tuple[Path, CheckpointExpectation]:
    path = tmp_path / "ckpt.pt"
    path.write_bytes(data)
    exp = CheckpointExpectation(hashlib.sha256(data).hexdigest(), "torch-state-dict", len(data))
    return path, exp


# --------------------------------------------------------------------------- checkpoints


def test_verified_checkpoint_carries_digest_and_no_path(tmp_path: Path) -> None:
    path, exp = _file(tmp_path)
    verified, _ = verify_checkpoint(path, exp)
    assert verified.checksum == f"sha256:{exp.sha256}"
    assert str(tmp_path) not in repr(verified)


def test_missing_checkpoint_is_rejected(tmp_path: Path) -> None:
    _, exp = _file(tmp_path)
    with pytest.raises(CheckpointMissingError):
        verify_checkpoint(tmp_path / "absent.pt", exp)
    with pytest.raises(CheckpointMissingError, match="regular file"):
        verify_checkpoint(tmp_path, exp)


def test_corrupt_or_truncated_checkpoint_is_rejected(tmp_path: Path) -> None:
    path, exp = _file(tmp_path, b"0123456789")
    path.write_bytes(b"0123456780")  # same size, one byte changed
    with pytest.raises(CheckpointIntegrityError, match="sha256"):
        verify_checkpoint(path, exp)
    path.write_bytes(b"01234")
    with pytest.raises(CheckpointIntegrityError, match="bytes"):
        verify_checkpoint(path, exp)


@pytest.mark.parametrize(
    "ref", ["hf-hub:MahmoodLab/UNI", "https://example.org/w.bin", "s3://bucket/w.pt"]
)
def test_remote_references_are_never_resolved(ref: str) -> None:
    with pytest.raises(CheckpointMissingError, match="never downloaded"):
        parse_bindings([f"uni={ref}"])
    with pytest.raises(CheckpointMissingError):
        CheckpointLocator({"uni": Path(ref)})


def test_bindings_are_explicit_and_unique(tmp_path: Path) -> None:
    assert parse_bindings([f"uni={tmp_path}/a.bin"]) == {"uni": tmp_path / "a.bin"}
    for bad in (["uni"], ["=x"], ["UNI=x"], ["uni=a", "uni=b"]):
        with pytest.raises(ModelInputContractError):
            parse_bindings(bad)


def test_unbound_encoder_fails_closed() -> None:
    with pytest.raises(CheckpointMissingError, match="never downloaded"):
        CheckpointLocator().path_for("uni")


def test_environment_bindings_are_overridden_by_explicit_ones(tmp_path: Path) -> None:
    env = {CHECKPOINTS_ENV: f"uni={tmp_path}/env.bin"}
    loc = CheckpointLocator.from_environment(env)
    assert loc.path_for("uni") == tmp_path / "env.bin"
    assert loc.overridden_by({"uni": tmp_path / "cli.bin"}).path_for("uni") == tmp_path / "cli.bin"


def test_expected_digest_comes_from_the_pin_or_config_never_neither() -> None:
    fixture = TorchFixtureFactory(CheckpointLocator()).descriptor
    with pytest.raises(CheckpointIntegrityError, match="no pinned"):
        expected_checkpoint(fixture, {})
    assert expected_checkpoint(fixture, {"checkpoint_sha256": "a" * 64}).sha256 == "a" * 64
    pin = UNI_DESCRIPTOR.checkpoint
    assert pin is not None
    assert expected_checkpoint(UNI_DESCRIPTOR, {}) == pin
    assert expected_checkpoint(UNI_DESCRIPTOR, {"checkpoint_sha256": f"sha256:{pin.sha256}"})
    with pytest.raises(CheckpointIntegrityError, match="accepts only"):
        expected_checkpoint(UNI_DESCRIPTOR, {"checkpoint_sha256": "b" * 64})


# --------------------------------------------------------------------------- domain contracts


def test_device_parsing_is_explicit() -> None:
    assert parse_device("cpu") == (DeviceKind.CPU, None)
    assert parse_device("cuda:1") == (DeviceKind.CUDA, 1)
    for bad in ("gpu", "cuda:x", "cpu:0", ""):
        with pytest.raises(ModelInputContractError):
            parse_device(bad)


def test_blocked_profiles_need_a_reason() -> None:
    with pytest.raises(ModelInputContractError):
        PreprocessingProfile("p", "1", False, ProfileStatus.BLOCKED, "d", ())
    with pytest.raises(ModelInputContractError):
        PreprocessingProfile("p", "1", False, ProfileStatus.AVAILABLE, "d", (), "why")


def test_checkpoint_expectation_validates_digest() -> None:
    with pytest.raises(ModelInputContractError):
        CheckpointExpectation("XYZ", "torch-state-dict")


# --------------------------------------------------------------------------- registry


def test_registry_resolves_by_stable_name_and_fails_closed() -> None:
    reg = EncoderRegistry()
    reg.register(ReferenceProjectionFactory())
    assert reg.resolve("reference-projection").name == "reference-projection"
    assert list(reg) == ["reference-projection"] and len(reg) == 1
    with pytest.raises(ExecutionUnsupportedError, match="not registered"):
        reg.resolve("uni")
    with pytest.raises(DuplicateRegistrationError):
        reg.register(ReferenceProjectionFactory())


def test_registry_rejects_a_descriptor_for_another_name() -> None:
    class Mislabelled(ReferenceProjectionFactory):
        @property
        def name(self) -> str:
            return "other"

    with pytest.raises(DuplicateRegistrationError, match="descriptor"):
        EncoderRegistry().register(Mislabelled())


def test_platform_registers_encoders_without_importing_torch() -> None:
    platform = build_platform(REPO_ROOT)
    names = [d.name for d in platform.encoders.descriptors()]
    assert names == ["reference-projection", "torch-fixture", "uni"]
    uni = platform.encoders.resolve("uni").descriptor
    assert uni.license == "CC-BY-NC-ND-4.0" and uni.checkpoint is not None
    assert uni.precisions == frozenset({Precision.FP32})
    assert not uni.synthetic and not uni.batch_invariant


# --------------------------------------------------------------------------- UNI profiles


def test_official_and_legacy_profiles_have_distinct_identities() -> None:
    assert OFFICIAL.official and not LEGACY_SURGEN.official
    assert OFFICIAL.ref != LEGACY_SURGEN.ref
    official = official_input_spec(1.0, 0.01)  # even at the legacy scale, the contract differs
    assert official.fingerprint != LEGACY_SURGEN_INPUT_SPEC.fingerprint
    assert official.channel_order is ChannelOrder.RGB
    assert LEGACY_SURGEN_INPUT_SPEC.channel_order is ChannelOrder.BGR


def test_legacy_profile_is_identified_but_blocked() -> None:
    assert LEGACY_SURGEN.status is ProfileStatus.BLOCKED
    factory = UniFactory(CheckpointLocator())
    cfg: dict[str, ConfigValue] = {**OFFICIAL_CFG, "profile": "uni-legacy-surgen@1"}
    with pytest.raises(ProfileUnavailableError, match="INTER_AREA"):
        factory.input_spec(cfg)
    with pytest.raises(ProfileUnavailableError, match="blocked"):
        factory.create(cfg)


def test_no_profile_is_chosen_by_default() -> None:
    factory = UniFactory(CheckpointLocator())
    with pytest.raises(ProfileUnavailableError, match="no profile is chosen"):
        factory.input_spec({})
    with pytest.raises(ProfileUnavailableError, match="no preprocessing profile"):
        factory.input_spec({**OFFICIAL_CFG, "profile": "uni-official@2"})


def test_official_profile_requires_an_explicit_scale_binding() -> None:
    factory = UniFactory(CheckpointLocator())
    with pytest.raises(ProfileUnavailableError, match="20x"):
        factory.input_spec({"profile": "uni-official@1"})
    spec = factory.input_spec(OFFICIAL_CFG)
    assert (spec.width, spec.height, spec.pixel_spacing_um) == (224, 224, 0.5)
    assert spec.spacing_tolerance == 0.15 and not spec.accepts_unknown_spacing
    assert spec.normalization is not None
    assert spec.normalization.mean == (0.485, 0.456, 0.406)
    assert factory.input_spec({**OFFICIAL_CFG, "model_spacing_um": 0.25}) != spec


def test_official_profile_refuses_any_declared_resampling() -> None:
    with pytest.raises(ProfileUnavailableError, match="Resize"):
        UniFactory(CheckpointLocator()).input_spec({**OFFICIAL_CFG, "resample": "area"})


def test_uni_create_fails_closed_without_a_checkpoint() -> None:
    pytest.importorskip("timm")
    with pytest.raises(CheckpointMissingError):
        UniFactory(CheckpointLocator()).create(OFFICIAL_CFG)


def test_uni_rejects_a_checkpoint_that_is_not_the_pinned_one(tmp_path: Path) -> None:
    pytest.importorskip("timm")
    path, _ = _file(tmp_path, b"not uni")
    with pytest.raises(CheckpointIntegrityError):
        UniFactory(CheckpointLocator({"uni": path})).create(OFFICIAL_CFG)


def test_composition_time_pin_is_reported_by_the_descriptor(tmp_path: Path) -> None:
    _, exp = _file(tmp_path)
    factory = UniFactory(CheckpointLocator(), checkpoint=exp)
    assert factory.descriptor.checkpoint == exp
    assert factory.descriptor == replace(UNI_DESCRIPTOR, checkpoint=exp)
