"""Model-bound preprocessing contracts.

A contract records every assumption that, if silently changed, would invalidate
a model: channel order, stain/intensity normalisation, spacing, interpolation,
orientation, patch geometry/sampling and ordered transforms. Contracts are
versioned and fingerprinted; models fail closed when the bound contract is
missing, incomplete or different.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from domain.errors import ChannelOrderMismatchError, MissingPreprocessingError
from domain.hashing import digest
from domain.ids import PreprocessingContractId, Version
from domain.image import ChannelOrder
from domain.modality import SpatialKind


class Interpolation(StrEnum):
    NEAREST = "nearest"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    AREA = "area"
    LANCZOS = "lanczos"


@dataclass(frozen=True, slots=True)
class TransformStep:
    """One explicit, ordered transform, e.g. ("channel_reorder", (("from","bgr"),("to","rgb")))."""

    name: str
    params: tuple[tuple[str, str], ...] = ()

    def param(self, key: str) -> str | None:
        return dict(self.params).get(key)


@dataclass(frozen=True, slots=True)
class PreprocessingRef:
    contract_id: PreprocessingContractId
    version: Version


@dataclass(frozen=True, slots=True)
class PreprocessingContract:
    contract_id: PreprocessingContractId
    version: Version
    spatial_kind: SpatialKind
    model_channel_order: ChannelOrder | None = None
    interpolation: Interpolation | None = None
    # 2D pyramidal
    target_pixel_spacing_um: float | None = None
    stain_normalization: str | None = None  # explicit "none" is allowed; None means undeclared
    patch_size_px: int | None = None
    patch_stride_px: int | None = None
    patch_sampling: str | None = None
    # volumetric
    target_voxel_spacing_mm: tuple[float, float, float] | None = None
    orientation: str | None = None
    intensity_normalization: str | None = None
    transforms: tuple[TransformStep, ...] = ()

    @property
    def ref(self) -> PreprocessingRef:
        return PreprocessingRef(self.contract_id, self.version)

    @property
    def fingerprint(self) -> str:
        return digest(self)

    def missing_fields(self) -> tuple[str, ...]:
        """Fields that must be declared for this spatial kind (sorted, deterministic)."""
        required: dict[str, object]
        if self.spatial_kind is SpatialKind.PYRAMIDAL_2D:
            required = {
                "model_channel_order": self.model_channel_order,
                "interpolation": self.interpolation,
                "target_pixel_spacing_um": self.target_pixel_spacing_um,
                "stain_normalization": self.stain_normalization,
            }
            if self.patch_size_px is not None:
                required["patch_stride_px"] = self.patch_stride_px
                required["patch_sampling"] = self.patch_sampling
        else:
            required = {
                "interpolation": self.interpolation,
                "target_voxel_spacing_mm": self.target_voxel_spacing_mm,
                "orientation": self.orientation,
                "intensity_normalization": self.intensity_normalization,
            }
        return tuple(sorted(k for k, v in required.items() if v is None))

    def require_complete(self) -> None:
        missing = self.missing_fields()
        if missing:
            raise MissingPreprocessingError(
                f"preprocessing contract {self.contract_id}@{self.version} does not declare: "
                + ", ".join(missing)
            )

    def resolve_channel_conversion(self, source: ChannelOrder | None) -> TransformStep | None:
        """Return the explicit reorder transform needed for `source`, or fail closed.

        Never silently swaps BGR/RGB: a mismatch is only accepted when the contract
        itself declares the exact `channel_reorder` transform.
        """
        target = self.model_channel_order
        if target is None:
            raise MissingPreprocessingError("contract does not declare model_channel_order")
        if source is None:
            raise ChannelOrderMismatchError(
                "source channel order is unknown; refusing to assume it matches "
                f"the model's {target.value}"
            )
        if source is target:
            return None
        for step in self.transforms:
            if (
                step.name == "channel_reorder"
                and step.param("from") == source.value
                and step.param("to") == target.value
            ):
                return step
        raise ChannelOrderMismatchError(
            f"source channel order {source.value} differs from model order {target.value} "
            "and the contract declares no explicit channel_reorder transform"
        )
