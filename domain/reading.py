"""Real-image reading contracts: requests, limits, valid-data semantics, metadata, provenance.

These contracts are format- and backend-neutral. A reader adapter translates its backend's
native coordinates, pixel types and metadata into them; nothing here names a vendor format.

Pixel payloads (and valid-data masks) are typed `object`: concrete array libraries are a plugin
concern and never leak into the domain.

Coordinate frames:

* requests address one pyramid level's pixel grid (`CoordinateFrame.WSI_LEVEL_PIXEL`);
* every returned region is also expressed in canonical level-zero pixels;
* a backend's native frame (e.g. a scanner's global stage pixels, possibly negative) is related to
  level zero only through an explicit `NativeFrameOffset`.

Valid data versus no data: "acquired" means the source holds acquired pixels for that location.
Acquired pixels include imaged background (e.g. empty glass); telling tissue from background is
not a reader's job. Unacquired locations inside the image bounds are reported only when the
reader can know them; otherwise the report says so. Out-of-bounds requests are rejected, never
padded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum

from domain.errors import (
    CoordinateFrameMismatchError,
    InvalidImageAssetError,
    InvalidSpatialReferenceError,
    RegionLimitExceededError,
    RegionOutOfBoundsError,
)
from domain.geometry import CoordinateFrame, Level0Rect, LevelRect, PyramidalGeometry
from domain.ids import ImageId
from domain.image import ChannelOrder, ImageAsset
from domain.provenance import ArtifactIdentity, ComponentVersion, EnvironmentInfo


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


# --------------------------------------------------------------------------- limits and requests


@dataclass(frozen=True, slots=True)
class ReadLimits:
    """Upper bounds checked before any pixel is decoded.

    Defaults match the Imaging API v1 region-request maximum (4096 x 4096).
    """

    max_width: int = 4096
    max_height: int = 4096
    max_pixels: int = 4096 * 4096

    def __post_init__(self) -> None:
        for v in (self.max_width, self.max_height, self.max_pixels):
            if not _is_int(v) or v < 1:
                raise InvalidSpatialReferenceError("read limits must be positive integers")

    def check(self, width: int, height: int) -> None:
        if width > self.max_width or height > self.max_height:
            raise RegionLimitExceededError(
                f"region {width}x{height} exceeds the {self.max_width}x{self.max_height} limit"
            )
        if width * height > self.max_pixels:
            raise RegionLimitExceededError(
                f"region of {width * height} pixels exceeds the {self.max_pixels}-pixel limit"
            )


@dataclass(frozen=True, slots=True)
class RegionReadRequest:
    """A bounded region in the pixel grid of one pyramid level.

    `scene`, `z` and `t` left as `None` mean "the image has exactly one"; a reader rejects them
    when that is not true. `channels=None` means every native channel in native order; readers
    reject channel subsets they cannot honour instead of guessing.
    """

    level: int
    x: int
    y: int
    width: int
    height: int
    frame: CoordinateFrame = CoordinateFrame.WSI_LEVEL_PIXEL
    scene: int | None = None
    z: int | None = None
    t: int | None = None
    channels: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.frame is not CoordinateFrame.WSI_LEVEL_PIXEL:
            raise CoordinateFrameMismatchError(
                f"region requests use frame {CoordinateFrame.WSI_LEVEL_PIXEL.value}, "
                f"got {self.frame.value}"
            )
        for name in ("level", "x", "y", "width", "height"):
            if not _is_int(getattr(self, name)):
                raise InvalidSpatialReferenceError(f"region {name} must be an integer")
        for name in ("scene", "z", "t"):
            value = getattr(self, name)
            if value is not None and (not _is_int(value) or value < 0):
                raise InvalidSpatialReferenceError(f"region {name} must be a non-negative integer")
        if self.level < 0:
            raise InvalidSpatialReferenceError("pyramid level must be non-negative")
        if self.x < 0 or self.y < 0:
            raise RegionOutOfBoundsError(f"region origin ({self.x}, {self.y}) is negative")
        if self.width < 1 or self.height < 1:
            raise InvalidSpatialReferenceError("region width/height must be positive")
        if self.channels is not None and (
            not self.channels or any(not _is_int(c) or c < 0 for c in self.channels)
        ):
            raise InvalidSpatialReferenceError("channels must be non-negative integer indices")

    def level_rect(self) -> LevelRect:
        return LevelRect(self.level, self.x, self.y, self.width, self.height)


def validate_region(geometry: PyramidalGeometry, request: RegionReadRequest) -> Level0Rect:
    """Fail closed unless the request lies inside its level and inside level-zero bounds.

    Returns the region in canonical level-zero coordinates.
    """
    if not 0 <= request.level < len(geometry.levels):
        raise InvalidSpatialReferenceError(
            f"pyramid level {request.level} does not exist (levels 0..{len(geometry.levels) - 1})"
        )
    level = geometry.levels[request.level]
    if request.x + request.width > level.width or request.y + request.height > level.height:
        raise RegionOutOfBoundsError(
            f"region ({request.x}, {request.y}, {request.width}x{request.height}) leaves level "
            f"{level.index} bounds {level.width}x{level.height}"
        )
    ds = level.downsample
    rect = Level0Rect(request.x * ds, request.y * ds, request.width * ds, request.height * ds)
    # Tolerate float rounding only; anything beyond level zero is out of bounds.
    eps = 1e-9 * max(geometry.width, geometry.height)
    if rect.x + rect.width > geometry.width + eps or rect.y + rect.height > geometry.height + eps:
        raise RegionOutOfBoundsError(
            f"region maps to {rect!r}, outside level-0 bounds {geometry.width}x{geometry.height}"
        )
    return rect


# --------------------------------------------------------------------------- native frame


@dataclass(frozen=True, slots=True)
class NativeFrameOffset:
    """Translation between a backend's native pixel frame and canonical level-zero pixels.

    `native = level0 + origin`. The origin is the native coordinate of level-zero pixel (0, 0)
    and may be negative; it is preserved verbatim, never normalised away.
    """

    origin_x: int
    origin_y: int

    def __post_init__(self) -> None:
        if not (_is_int(self.origin_x) and _is_int(self.origin_y)):
            raise InvalidSpatialReferenceError("native origin must be integer pixels")

    def to_native(self, x: float, y: float) -> tuple[float, float]:
        return (x + self.origin_x, y + self.origin_y)

    def to_level0(self, native_x: float, native_y: float) -> tuple[float, float]:
        return (native_x - self.origin_x, native_y - self.origin_y)

    def rect_to_native(self, rect: Level0Rect) -> tuple[float, float, float, float]:
        nx, ny = self.to_native(rect.x, rect.y)
        return (nx, ny, rect.width, rect.height)


# --------------------------------------------------------------------------- valid data


class NoDataSemantics(StrEnum):
    """What a reader can say about unacquired locations inside the image bounds."""

    # The reader knows which level-zero areas hold acquired data (e.g. scanner tile coverage).
    ACQUISITION_COVERAGE = "acquisition_coverage"
    # The backend marks locations without data as transparent (alpha = 0).
    BACKEND_TRANSPARENCY = "backend_transparency"
    # The reader cannot distinguish acquired from unacquired pixels.
    NOT_REPORTED = "not_reported"


class DataValidity(StrEnum):
    ALL_ACQUIRED = "all_acquired"
    PARTIALLY_ACQUIRED = "partially_acquired"
    NONE_ACQUIRED = "none_acquired"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ValidDataReport:
    """Valid-data information for one returned region.

    `mask`, when present, is a boolean array at the returned resolution: True where acquired.
    """

    semantics: NoDataSemantics
    validity: DataValidity
    acquired_fraction: float | None
    mask: object | None = field(default=None, compare=False, repr=False)
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.semantics is NoDataSemantics.NOT_REPORTED:
            if (
                self.validity is not DataValidity.UNKNOWN
                or self.acquired_fraction is not None
                or self.mask is not None
            ):
                raise InvalidImageAssetError("a reader without no-data semantics reports UNKNOWN")
            if not self.limitations:
                raise InvalidImageAssetError("an UNKNOWN valid-data report must state why")
            return
        f = self.acquired_fraction
        if f is None or not math.isfinite(f) or not 0.0 <= f <= 1.0:
            raise InvalidImageAssetError("acquired_fraction must be within [0, 1]")
        expected = (
            DataValidity.ALL_ACQUIRED
            if f == 1.0
            else DataValidity.NONE_ACQUIRED
            if f == 0.0
            else DataValidity.PARTIALLY_ACQUIRED
        )
        if self.validity is not expected:
            raise InvalidImageAssetError(
                f"validity {self.validity.value} contradicts acquired fraction {f}"
            )

    @staticmethod
    def from_fraction(
        semantics: NoDataSemantics,
        fraction: float,
        mask: object | None,
        limitations: tuple[str, ...] = (),
    ) -> ValidDataReport:
        validity = (
            DataValidity.ALL_ACQUIRED
            if fraction == 1.0
            else DataValidity.NONE_ACQUIRED
            if fraction == 0.0
            else DataValidity.PARTIALLY_ACQUIRED
        )
        return ValidDataReport(semantics, validity, fraction, mask, limitations)


# --------------------------------------------------------------------------- metadata


@dataclass(frozen=True, slots=True)
class SourceIntegrity:
    """Identity of the source bytes, with an explicit statement of what it guarantees."""

    strategy: str
    value: str
    size_bytes: int
    guarantees: str

    @property
    def token(self) -> str:
        return f"{self.strategy}:{self.value}"


@dataclass(frozen=True, slots=True)
class ImageDescription:
    """Everything a reader established about one image (one scene), and what it could not.

    `asset` carries the canonical identity and geometry. The remaining fields describe pixel
    semantics, how the geometry was established, and which metadata is missing. Missing values
    stay missing (`None`) and are listed in `missing_metadata`.
    """

    asset: ImageAsset
    reader: ComponentVersion
    backend: ComponentVersion
    scene: int
    scene_count: int
    native_offset: NativeFrameOffset | None
    pixel_type: str
    dtype: str
    samples_per_pixel: int
    channel_count: int
    z_planes: int
    time_points: int
    spacing_source: str | None
    level_source: str
    no_data: NoDataSemantics
    decoder_transformations: tuple[str, ...]
    acquisition: tuple[tuple[str, str], ...] = ()
    missing_metadata: tuple[str, ...] = ()
    integrity: SourceIntegrity | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.asset.spatial_reference, PyramidalGeometry):
            raise InvalidImageAssetError("image descriptions currently cover 2D pyramidal images")
        if self.asset.channel_order is None:
            raise InvalidImageAssetError("a reader must report the channel order it returns")
        if not _is_int(self.scene) or self.scene < 0 or self.scene_count < 1:
            raise InvalidImageAssetError("invalid scene selection")
        has_spacing = self.asset.spatial_reference.pixel_spacing is not None
        if has_spacing != (self.spacing_source is not None):
            raise InvalidImageAssetError("pixel spacing and its source must be given together")
        if not has_spacing and "physical_spacing" not in self.missing_metadata:
            raise InvalidImageAssetError("absent pixel spacing must be listed as missing")
        for n in (self.samples_per_pixel, self.channel_count, self.z_planes, self.time_points):
            if not _is_int(n) or n < 1:
                raise InvalidImageAssetError("pixel/plane counts must be positive integers")

    @property
    def image_id(self) -> ImageId:
        return self.asset.image_id

    @property
    def geometry(self) -> PyramidalGeometry:
        geometry = self.asset.spatial_reference
        assert isinstance(geometry, PyramidalGeometry)
        return geometry

    @property
    def channel_order(self) -> ChannelOrder:
        assert self.asset.channel_order is not None
        return self.asset.channel_order


# --------------------------------------------------------------------------- results and provenance


@dataclass(frozen=True, slots=True)
class RegionReadProvenance:
    """What was asked, what was returned, and how it was decoded. Canonicalisable."""

    image_id: ImageId
    reader: ComponentVersion
    backend: ComponentVersion
    source_integrity: str | None
    scene: int
    requested: LevelRect
    returned: LevelRect
    level0: Level0Rect
    downsample: float
    native_region: tuple[int, int, int, int] | None
    shape: tuple[int, ...]
    dtype: str
    channel_order: ChannelOrder
    decoder_transformations: tuple[str, ...]
    metadata_limitations: tuple[str, ...]
    validity: DataValidity
    acquired_fraction: float | None


@dataclass(frozen=True, eq=False)
class PixelRegion:
    """A decoded region: native pixels plus explicit spatial, pixel and valid-data semantics."""

    pixels: object
    provenance: RegionReadProvenance
    valid_data: ValidDataReport

    @property
    def level0(self) -> Level0Rect:
        return self.provenance.level0

    @property
    def channel_order(self) -> ChannelOrder:
        return self.provenance.channel_order


INGESTION_USAGE = (
    "image ingestion only: no model inference was performed; decoding an image is not "
    "evidence of scientific or clinical validity"
)


@dataclass(frozen=True, slots=True)
class IngestionRecord:
    """Provenance of a real-image ingestion (read without any model)."""

    record_id: str
    image_id: ImageId
    reader: ComponentVersion
    backend: ComponentVersion
    integrity: SourceIntegrity | None
    description_digest: str
    regions: tuple[RegionReadProvenance, ...]
    artifacts: tuple[ArtifactIdentity, ...]
    environment: EnvironmentInfo
    started_at: str
    finished_at: str
    usage: str = INGESTION_USAGE
