"""Model-independent foundation-model runtime contracts (SPEC-006, ADR-006).

An encoder is described before it is loaded (`EncoderDescriptor`) and explains itself after it
ran (`EncoderRuntimeReport`). Nothing here names a model, a framework or a file format:

* a checkpoint is a controlled artifact identified by its content digest
  (`CheckpointExpectation` before loading, `VerifiedCheckpoint` after verification); its local
  path is operational configuration and never appears in these contracts;
* a preprocessing profile is an independently identified, versioned recipe. A profile whose
  requirements could not be verified, or whose transformations are not implemented, is
  `BLOCKED` with a stated reason. It is never silently replaced by another profile;
* devices and precisions are declared, never inferred.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from domain.embedding import Precision
from domain.errors import ModelInputContractError
from domain.pipeline import ConfigValue
from domain.provenance import ComponentVersion

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


def _require(cond: bool, why: str) -> None:
    if not cond:
        raise ModelInputContractError(why)


class DeviceKind(StrEnum):
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


def parse_device(text: str) -> tuple[DeviceKind, int | None]:
    """`cpu`, `mps`, `cuda` or `cuda:<index>`; anything else is rejected, never guessed."""
    kind, sep, index = text.strip().partition(":")
    try:
        device = DeviceKind(kind)
    except ValueError:
        raise ModelInputContractError(
            f"unknown device {text!r}; use cpu, mps, cuda or cuda:<index>"
        ) from None
    if not sep:
        return device, None
    _require(device is DeviceKind.CUDA and index.isdigit(), f"invalid device {text!r}")
    return device, int(index)


@dataclass(frozen=True, slots=True)
class CheckpointExpectation:
    """What a checkpoint must be: its content digest (identity) and, optionally, its size."""

    sha256: str
    format: str
    size_bytes: int | None = None

    def __post_init__(self) -> None:
        _require(bool(_SHA256.match(self.sha256)), "checkpoint sha256 must be 64 lowercase hex")
        _require(bool(self.format), "checkpoint format must be named")
        _require(
            self.size_bytes is None or (isinstance(self.size_bytes, int) and self.size_bytes > 0),
            "checkpoint size must be a positive integer",
        )


@dataclass(frozen=True, slots=True)
class VerifiedCheckpoint:
    """A checkpoint whose bytes were hashed and matched. Carries no path (ADR-006 D4)."""

    sha256: str
    size_bytes: int
    format: str

    @property
    def checksum(self) -> str:
        return f"sha256:{self.sha256}"


class ProfileStatus(StrEnum):
    AVAILABLE = "available"
    # Defined and identified, but not executable: unverified requirements or missing transforms.
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class PreprocessingProfile:
    """An independently identified, versioned preprocessing recipe of one encoder."""

    profile_id: str
    version: str
    official: bool
    status: ProfileStatus
    description: str
    evidence: tuple[str, ...]
    blocked_reason: str | None = None

    def __post_init__(self) -> None:
        _require(bool(_NAME.match(self.profile_id)), f"invalid profile id {self.profile_id!r}")
        _require(bool(self.version), "profile version must be set")
        _require(
            (self.status is ProfileStatus.BLOCKED) == bool(self.blocked_reason),
            "a blocked profile needs a reason, and only a blocked profile has one",
        )

    @property
    def ref(self) -> str:
        return f"{self.profile_id}@{self.version}"


@dataclass(frozen=True, slots=True)
class EncoderDescriptor:
    """Registry metadata of one encoder adapter, available without loading anything (§6)."""

    name: str
    adapter: ComponentVersion
    architecture: str
    output_dim: int
    output_dtype: str
    devices: frozenset[DeviceKind]
    precisions: frozenset[Precision]
    profiles: tuple[PreprocessingProfile, ...]
    synthetic: bool
    batch_invariant: bool
    checkpoint: CheckpointExpectation | None = None
    license: str | None = None
    access: str | None = None

    def __post_init__(self) -> None:
        _require(bool(_NAME.match(self.name)), f"invalid encoder name {self.name!r}")
        _require(bool(self.architecture), "architecture must be named")
        _require(self.output_dim > 0, "output_dim must be positive")
        _require(self.output_dtype in ("float16", "float32", "float64"), "floating output only")
        _require(bool(self.devices) and bool(self.precisions), "declare devices and precisions")
        _require(bool(self.profiles), "declare at least one preprocessing profile")
        refs = [p.ref for p in self.profiles]
        _require(len(set(p.profile_id for p in self.profiles)) == len(refs), "duplicate profile")

    def profile(self, profile_id: str) -> PreprocessingProfile:
        for p in self.profiles:
            if p.profile_id == profile_id:
                return p
        known = ", ".join(p.profile_id for p in self.profiles)
        raise ModelInputContractError(
            f"encoder {self.name} has no preprocessing profile {profile_id!r} (known: {known})"
        )


@dataclass(frozen=True, slots=True)
class EncoderRuntimeReport:
    """What an encoder reports after use.

    `limitations` are deterministic statements that enter the embedding manifest (and its
    digest). `resources` are measurements (bytes, seconds, versions, thread counts) that enter
    only the bounded run metrics, never an identity.
    """

    limitations: tuple[str, ...] = ()
    resources: tuple[tuple[str, ConfigValue], ...] = ()
