"""Spatial references and coordinate frames.

Two disjoint families are modelled:

* 2D pyramidal images: integer pixel grids per pyramid level, with canonical
  level-zero pixel coordinates and optional physical pixel spacing (µm).
* 3D/4D volumes: voxel indices (i, j, k) plus an index-to-physical affine
  (spacing, origin, direction cosines) in patient millimetres.

Values from one family are never accepted where the other is expected, and
missing physical metadata stays missing instead of being fabricated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from domain.errors import (
    CoordinateFrameMismatchError,
    InvalidSpatialReferenceError,
    MissingSpatialMetadataError,
)
from domain.modality import SpatialKind


class CoordinateFrame(StrEnum):
    WSI_LEVEL0_PIXEL = "wsi_level0_pixel"
    WSI_LEVEL_PIXEL = "wsi_level_pixel"
    VOLUME_VOXEL_INDEX = "volume_voxel_index"
    PATIENT_LPS_MM = "patient_lps_mm"


def _finite(*values: float) -> None:
    for v in values:
        if not isinstance(v, int | float) or isinstance(v, bool) or not math.isfinite(v):
            raise InvalidSpatialReferenceError(f"coordinate must be a finite number, got {v!r}")


# --------------------------------------------------------------------------- 2D pyramidal


@dataclass(frozen=True, slots=True)
class PixelSpacing:
    """Physical size of one level-zero pixel, in micrometres."""

    x_um: float
    y_um: float

    def __post_init__(self) -> None:
        _finite(self.x_um, self.y_um)
        if self.x_um <= 0 or self.y_um <= 0:
            raise InvalidSpatialReferenceError("pixel spacing must be positive")


@dataclass(frozen=True, slots=True)
class PyramidLevel:
    index: int
    width: int
    height: int
    downsample: float

    def __post_init__(self) -> None:
        if self.index < 0 or self.width < 1 or self.height < 1:
            raise InvalidSpatialReferenceError(f"invalid pyramid level {self!r}")
        _finite(self.downsample)
        if self.downsample < 1:
            raise InvalidSpatialReferenceError("pyramid downsample must be >= 1")


@dataclass(frozen=True, slots=True)
class Level0Rect:
    """Axis-aligned rectangle in canonical level-zero pixel coordinates."""

    x: float
    y: float
    width: float
    height: float
    frame: CoordinateFrame = CoordinateFrame.WSI_LEVEL0_PIXEL

    def __post_init__(self) -> None:
        _finite(self.x, self.y, self.width, self.height)
        if self.frame is not CoordinateFrame.WSI_LEVEL0_PIXEL:
            raise CoordinateFrameMismatchError(f"Level0Rect cannot use frame {self.frame}")
        if self.width <= 0 or self.height <= 0:
            raise InvalidSpatialReferenceError("rectangle width/height must be positive")


@dataclass(frozen=True, slots=True)
class LevelRect:
    """Rectangle in the pixel grid of a specific pyramid level."""

    level: int
    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        _finite(self.x, self.y, self.width, self.height)
        if self.level < 0 or self.width <= 0 or self.height <= 0:
            raise InvalidSpatialReferenceError(f"invalid level rectangle {self!r}")


@dataclass(frozen=True, slots=True)
class PyramidalGeometry:
    """Geometry of a 2D pyramidal image (e.g. a whole-slide image)."""

    width: int
    height: int
    levels: tuple[PyramidLevel, ...]
    pixel_spacing: PixelSpacing | None = None
    spatial_kind: SpatialKind = SpatialKind.PYRAMIDAL_2D

    def __post_init__(self) -> None:
        if self.spatial_kind is not SpatialKind.PYRAMIDAL_2D:
            raise CoordinateFrameMismatchError("PyramidalGeometry must be PYRAMIDAL_2D")
        if self.width < 1 or self.height < 1:
            raise InvalidSpatialReferenceError("image dimensions must be positive")
        if not self.levels:
            raise InvalidSpatialReferenceError("a pyramidal image needs at least level 0")
        level0 = self.levels[0]
        if (level0.index, level0.width, level0.height, level0.downsample) != (
            0,
            self.width,
            self.height,
            1.0,
        ):
            raise InvalidSpatialReferenceError("level 0 must match native dimensions, downsample 1")
        for i, (prev, cur) in enumerate(zip(self.levels, self.levels[1:], strict=False), start=1):
            if cur.index != i or cur.downsample <= prev.downsample:
                raise InvalidSpatialReferenceError(
                    "pyramid levels must be contiguous with strictly increasing downsample"
                )

    def level(self, index: int) -> PyramidLevel:
        if not 0 <= index < len(self.levels):
            raise InvalidSpatialReferenceError(f"pyramid level {index} does not exist")
        return self.levels[index]

    def to_level0(self, rect: LevelRect) -> Level0Rect:
        """Convert level-local pixel coordinates to canonical level-zero coordinates."""
        ds = self.level(rect.level).downsample
        out = Level0Rect(rect.x * ds, rect.y * ds, rect.width * ds, rect.height * ds)
        self.require_within(out)
        return out

    def from_level0(self, rect: Level0Rect, level: int) -> LevelRect:
        self.require_within(rect)
        ds = self.level(level).downsample
        return LevelRect(level, rect.x / ds, rect.y / ds, rect.width / ds, rect.height / ds)

    def require_within(self, rect: Level0Rect) -> None:
        if (
            rect.x < 0
            or rect.y < 0
            or rect.x + rect.width > self.width
            or (rect.y + rect.height > self.height)
        ):
            raise InvalidSpatialReferenceError(
                f"rectangle {rect!r} lies outside level-0 bounds {self.width}x{self.height}"
            )

    def require_pixel_spacing(self) -> PixelSpacing:
        if self.pixel_spacing is None:
            raise MissingSpatialMetadataError(
                "physical pixel spacing is unavailable for this image; it will not be assumed"
            )
        return self.pixel_spacing


# --------------------------------------------------------------------------- 3D/4D volumes


Vec3 = tuple[float, float, float]
Mat3 = tuple[float, float, float, float, float, float, float, float, float]


@dataclass(frozen=True, slots=True)
class VoxelIndex:
    i: int
    j: int
    k: int
    frame: CoordinateFrame = CoordinateFrame.VOLUME_VOXEL_INDEX

    def __post_init__(self) -> None:
        if self.frame is not CoordinateFrame.VOLUME_VOXEL_INDEX:
            raise CoordinateFrameMismatchError(f"VoxelIndex cannot use frame {self.frame}")
        for v in (self.i, self.j, self.k):
            if not isinstance(v, int) or isinstance(v, bool):
                raise InvalidSpatialReferenceError("voxel indices must be integers")


@dataclass(frozen=True, slots=True)
class PhysicalPoint:
    """Point in patient space (LPS, millimetres)."""

    x_mm: float
    y_mm: float
    z_mm: float
    frame: CoordinateFrame = CoordinateFrame.PATIENT_LPS_MM

    def __post_init__(self) -> None:
        _finite(self.x_mm, self.y_mm, self.z_mm)
        if self.frame is not CoordinateFrame.PATIENT_LPS_MM:
            raise CoordinateFrameMismatchError(f"PhysicalPoint cannot use frame {self.frame}")


@dataclass(frozen=True, slots=True)
class VolumeGeometry:
    """Geometry of a 3D (optionally time-resolved 4D) volume."""

    shape: tuple[int, int, int]
    spacing_mm: Vec3 | None = None
    origin_mm: Vec3 | None = None
    direction: Mat3 | None = None
    time_points: int | None = None
    series_uid: str | None = None

    def __post_init__(self) -> None:
        if len(self.shape) != 3 or any(
            not isinstance(s, int) or isinstance(s, bool) or s < 1 for s in self.shape
        ):
            raise InvalidSpatialReferenceError(
                f"volume shape must be three positive integers, got {self.shape!r}"
            )
        if self.spacing_mm is not None:
            if len(self.spacing_mm) != 3:
                raise InvalidSpatialReferenceError("voxel spacing must have three components")
            _finite(*self.spacing_mm)
            if any(s <= 0 for s in self.spacing_mm):
                raise InvalidSpatialReferenceError("voxel spacing must be positive")
        if self.origin_mm is not None:
            if len(self.origin_mm) != 3:
                raise InvalidSpatialReferenceError("origin must have three components")
            _finite(*self.origin_mm)
        if self.direction is not None:
            if len(self.direction) != 9:
                raise InvalidSpatialReferenceError("direction must be a 3x3 matrix (9 values)")
            _finite(*self.direction)
        if self.time_points is not None and self.time_points < 1:
            raise InvalidSpatialReferenceError("time_points must be positive when present")

    @property
    def spatial_kind(self) -> SpatialKind:
        return SpatialKind.VOLUME_3D if self.time_points is None else SpatialKind.VOLUME_4D

    def require_within(self, index: VoxelIndex) -> None:
        for value, size in zip((index.i, index.j, index.k), self.shape, strict=True):
            if not 0 <= value < size:
                raise InvalidSpatialReferenceError(f"{index!r} outside volume shape {self.shape}")

    def voxel_to_physical(self, index: VoxelIndex) -> PhysicalPoint:
        """origin + direction · (spacing ⊙ index). Requires complete affine metadata."""
        self.require_within(index)
        if self.spacing_mm is None or self.origin_mm is None or self.direction is None:
            raise MissingSpatialMetadataError(
                "voxel->physical transform requires spacing, origin and direction; "
                "missing values are not assumed"
            )
        scaled = (
            index.i * self.spacing_mm[0],
            index.j * self.spacing_mm[1],
            index.k * self.spacing_mm[2],
        )
        d = self.direction
        o = self.origin_mm
        return PhysicalPoint(
            o[0] + d[0] * scaled[0] + d[1] * scaled[1] + d[2] * scaled[2],
            o[1] + d[3] * scaled[0] + d[4] * scaled[1] + d[5] * scaled[2],
            o[2] + d[6] * scaled[0] + d[7] * scaled[1] + d[8] * scaled[2],
        )


SpatialReference = PyramidalGeometry | VolumeGeometry


# --------------------------------------------------------------------------- evidence geometry


@dataclass(frozen=True, slots=True)
class RectangleGeometry:
    rect: Level0Rect

    @property
    def frame(self) -> CoordinateFrame:
        return CoordinateFrame.WSI_LEVEL0_PIXEL


@dataclass(frozen=True, slots=True)
class PolygonGeometry:
    points: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if len(self.points) < 3:
            raise InvalidSpatialReferenceError("a polygon needs at least three points")
        for x, y in self.points:
            _finite(x, y)

    @property
    def frame(self) -> CoordinateFrame:
        return CoordinateFrame.WSI_LEVEL0_PIXEL


@dataclass(frozen=True, slots=True)
class PointGeometry:
    x: float
    y: float

    def __post_init__(self) -> None:
        _finite(self.x, self.y)

    @property
    def frame(self) -> CoordinateFrame:
        return CoordinateFrame.WSI_LEVEL0_PIXEL


@dataclass(frozen=True, slots=True)
class VoxelBoxGeometry:
    """Inclusive-start, exclusive-end voxel box."""

    start: VoxelIndex
    stop: VoxelIndex

    def __post_init__(self) -> None:
        if not (
            self.start.i < self.stop.i and self.start.j < self.stop.j and self.start.k < self.stop.k
        ):
            raise InvalidSpatialReferenceError("voxel box must have positive extent")

    @property
    def frame(self) -> CoordinateFrame:
        return CoordinateFrame.VOLUME_VOXEL_INDEX


Geometry2D = RectangleGeometry | PolygonGeometry | PointGeometry
Geometry3D = VoxelBoxGeometry
EvidenceGeometry = Geometry2D | Geometry3D


def validate_geometry_against(geometry: EvidenceGeometry, reference: SpatialReference) -> None:
    """Fail closed when evidence geometry does not belong to the image's spatial reference."""
    if isinstance(reference, PyramidalGeometry):
        if isinstance(geometry, VoxelBoxGeometry):
            raise CoordinateFrameMismatchError(
                "voxel geometry cannot be attached to a 2D pyramidal image"
            )
        if isinstance(geometry, RectangleGeometry):
            reference.require_within(geometry.rect)
        else:
            pts = (
                geometry.points
                if isinstance(geometry, PolygonGeometry)
                else ((geometry.x, geometry.y),)
            )
            for x, y in pts:
                if not (0 <= x <= reference.width and 0 <= y <= reference.height):
                    raise InvalidSpatialReferenceError(f"point ({x}, {y}) outside level-0 bounds")
        return
    if not isinstance(geometry, VoxelBoxGeometry):
        raise CoordinateFrameMismatchError(
            "level-zero pixel geometry cannot be attached to a volumetric image"
        )
    reference.require_within(geometry.start)
    last = VoxelIndex(geometry.stop.i - 1, geometry.stop.j - 1, geometry.stop.k - 1)
    reference.require_within(last)
