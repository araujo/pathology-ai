"""Typed data kinds flowing between pipeline stages and models."""

from __future__ import annotations

import re
from dataclasses import dataclass

from domain.errors import InvalidIdentifierError
from domain.modality import SpatialKind

_KIND = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z0-9_]+)*$")


@dataclass(frozen=True, slots=True, order=True)
class DataKind:
    name: str

    def __post_init__(self) -> None:
        if not _KIND.fullmatch(self.name):
            raise InvalidIdentifierError(f"invalid data kind {self.name!r}")

    def __str__(self) -> str:
        return self.name


# Generic vocabulary; plugins may define further kinds without changing the domain.
IMAGE_PYRAMIDAL_2D = DataKind("image.pyramidal_2d")
IMAGE_VOLUME = DataKind("image.volume")
IMAGE_SERIES = DataKind("image.series")
# A bounded, decoded 2D region with explicit spatial and pixel semantics.
IMAGE_REGION_2D = DataKind("image.region_2d")
TISSUE_MASK_2D = DataKind("mask.tissue_2d")
# A deterministic patch plan (every candidate with its decision) and its extraction manifest.
PATCH_PLAN = DataKind("patch_plan")
PATCH_MANIFEST = DataKind("patch_manifest")
PATCH_SET = DataKind("patch_set")
EMBEDDING_BAG = DataKind("embedding_bag")
CLASSIFICATION = DataKind("prediction.classification")
SEGMENTATION_2D = DataKind("prediction.segmentation_2d")
SEGMENTATION_3D = DataKind("prediction.segmentation_3d")
DETECTIONS_2D = DataKind("prediction.detections_2d")


@dataclass(frozen=True, slots=True)
class PortSpec:
    """A named, typed input or output with optional spatial and dimensional constraints."""

    name: str
    kind: DataKind
    spatial_kind: SpatialKind | None = None
    feature_dim: int | None = None

    def __post_init__(self) -> None:
        if not re.fullmatch(r"^[a-z][a-z0-9_]*$", self.name):
            raise InvalidIdentifierError(f"invalid port name {self.name!r}")
        if self.feature_dim is not None and self.feature_dim < 1:
            raise InvalidIdentifierError("feature_dim must be positive")
