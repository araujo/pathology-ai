"""Explicit translations between canonical level-zero pixels and other 2D pixel frames.

Coordinates produced by other systems (for example a legacy pipeline that anchored its frame at a
padded bounding rectangle) are never taken as level-zero coordinates. They are translated through
the native frame they were derived from:

    other  = native - other_origin
    level0 = native - level0_origin        (`NativeFrameOffset`)
    level0 = other + (other_origin - level0_origin)

The offset depends on both origins, so it is computed per image and never assumed constant.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from domain.errors import InvalidSpatialReferenceError, MissingSpatialMetadataError
from domain.geometry import Level0Rect, PyramidalGeometry
from domain.reading import ImageDescription, NativeFrameOffset

# `ImageDescription.acquisition` key under which a reader may report the native coordinate of the
# top-left of its padded bounding rectangle ("x,y"), when the backend has one.
NATIVE_PADDED_ORIGIN = "native.padded_origin"


class Placement(StrEnum):
    INSIDE = "inside"
    CROSSES_BOUNDS = "crosses_bounds"
    OUTSIDE = "outside"


@dataclass(frozen=True, slots=True)
class FrameTranslation:
    """`level0 = other + (dx, dy)` for one image; `frame` names the other frame."""

    frame: str
    dx: int
    dy: int
    derivation: str

    @staticmethod
    def from_native_origins(
        *, frame: str, other_origin: tuple[int, int], level0: NativeFrameOffset
    ) -> FrameTranslation:
        ox, oy = other_origin
        for v in (ox, oy):
            if not isinstance(v, int) or isinstance(v, bool):
                raise InvalidSpatialReferenceError("frame origins must be integer native pixels")
        return FrameTranslation(
            frame,
            ox - level0.origin_x,
            oy - level0.origin_y,
            f"other origin ({ox}, {oy}) - level-zero origin ({level0.origin_x}, "
            f"{level0.origin_y}) in native pixels",
        )

    def to_level0(self, x: float, y: float) -> tuple[float, float]:
        return (x + self.dx, y + self.dy)

    def from_level0(self, x: float, y: float) -> tuple[float, float]:
        return (x - self.dx, y - self.dy)

    def place(
        self, x: float, y: float, width: float, height: float, geometry: PyramidalGeometry
    ) -> tuple[Placement, Level0Rect | None]:
        """Translate a rectangle and classify it against level-zero bounds (never clipped).

        Returns the level-zero rectangle only when it lies fully inside the image.
        """
        lx, ly = self.to_level0(x, y)
        x1, y1 = lx + width, ly + height
        if x1 <= 0 or y1 <= 0 or lx >= geometry.width or ly >= geometry.height:
            return Placement.OUTSIDE, None
        if lx < 0 or ly < 0 or x1 > geometry.width or y1 > geometry.height:
            return Placement.CROSSES_BOUNDS, None
        return Placement.INSIDE, Level0Rect(lx, ly, width, height)


def padded_origin_translation(description: ImageDescription, frame: str) -> FrameTranslation:
    """Translation from a padded-bounding-rectangle frame reported by the reader."""
    if description.native_offset is None:
        raise MissingSpatialMetadataError("the reader reports no native frame for this image")
    value = dict(description.acquisition).get(NATIVE_PADDED_ORIGIN)
    if value is None:
        raise MissingSpatialMetadataError(
            "the reader reports no padded native origin; a padded-frame translation is not assumed"
        )
    try:
        ox, oy = (int(v) for v in value.split(","))
    except ValueError:
        raise InvalidSpatialReferenceError(f"malformed {NATIVE_PADDED_ORIGIN} {value!r}") from None
    return FrameTranslation.from_native_origins(
        frame=frame, other_origin=(ox, oy), level0=description.native_offset
    )
