"""Candidate-tissue masks: a raster with explicit level-zero geometry and provenance.

A tissue mask answers "does this acquired area look like candidate tissue according to a named
algorithm?". It is not tumour detection, diagnosis or a clinical tissue classification.

Acquisition validity ("was this area acquired at all?") is kept beside it, never inferred from
it: `acquired` may be absent only when the reader cannot report validity, and then the policy
that was applied is recorded.

Mask payloads are typed `object` (array libraries are a plugin concern).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import StrEnum

from domain.capabilities import CapabilityCheck, ValidityPolicy
from domain.errors import InvalidImageAssetError, InvalidSpatialReferenceError
from domain.geometry import CoordinateFrame, Level0Rect, PyramidalGeometry
from domain.ids import ImageId
from domain.pipeline import ConfigValue
from domain.provenance import ComponentVersion
from domain.reading import NoDataSemantics

CANDIDATE_TISSUE_NOTICE = (
    "candidate tissue by a colour heuristic; not tumour detection, diagnosis or clinical tissue "
    "classification"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class MaskSource(StrEnum):
    # One mask cell per pixel of a stored pyramid level.
    STORED_LEVEL = "stored_level"
    # One mask cell per `block_factor` x `block_factor` block of a stored level, read in tiles.
    BLOCK_REDUCED = "block_reduced"


@dataclass(frozen=True, slots=True)
class MaskGrid:
    """Level-zero geometry of a raster mask.

    Cell (row i, column j) covers level-zero `[x_edge(j), x_edge(j+1)) x [y_edge(i), y_edge(i+1))`
    with `x_edge(k) = min(k * cell_x, level0_width)` and the final edge at the image edge. Stored
    levels have `floor`ed dimensions and blocks may be partial, so the last row and column are
    stretched or clamped to end exactly at the level-zero edge; `edge_adjustment` records by how
    much (positive: stretched, negative: clamped).
    """

    width: int
    height: int
    cell_x: float
    cell_y: float
    level0_width: int
    level0_height: int
    source: MaskSource
    source_level: int
    block_factor: int = 1
    frame: CoordinateFrame = CoordinateFrame.WSI_LEVEL0_PIXEL

    def __post_init__(self) -> None:
        for n in (
            self.width,
            self.height,
            self.level0_width,
            self.level0_height,
            self.block_factor,
        ):
            if not isinstance(n, int) or isinstance(n, bool) or n < 1:
                raise InvalidSpatialReferenceError("mask grid sizes must be positive integers")
        if not all(math.isfinite(v) and v >= 1 for v in (self.cell_x, self.cell_y)):
            raise InvalidSpatialReferenceError("mask cells must cover at least one level-0 pixel")
        if self.frame is not CoordinateFrame.WSI_LEVEL0_PIXEL:
            raise InvalidSpatialReferenceError("mask grids are anchored in level-zero pixels")
        if (self.width - 1) * self.cell_x >= self.level0_width or (
            self.height - 1
        ) * self.cell_y >= self.level0_height:
            raise InvalidSpatialReferenceError("mask grid extends beyond the level-zero image")
        if self.source is MaskSource.STORED_LEVEL and self.block_factor != 1:
            raise InvalidSpatialReferenceError("a stored-level mask has block factor 1")

    @staticmethod
    def for_level(geometry: PyramidalGeometry, level: int, block_factor: int = 1) -> MaskGrid:
        lv = geometry.level(level)
        return MaskGrid(
            width=math.ceil(lv.width / block_factor),
            height=math.ceil(lv.height / block_factor),
            cell_x=lv.downsample * block_factor,
            cell_y=lv.downsample * block_factor,
            level0_width=geometry.width,
            level0_height=geometry.height,
            source=MaskSource.STORED_LEVEL if block_factor == 1 else MaskSource.BLOCK_REDUCED,
            source_level=level,
            block_factor=block_factor,
        )

    def x_edge(self, k: int) -> float:
        return (
            float(self.level0_width) if k >= self.width else min(k * self.cell_x, self.level0_width)
        )

    def y_edge(self, k: int) -> float:
        return (
            float(self.level0_height)
            if k >= self.height
            else min(k * self.cell_y, self.level0_height)
        )

    def cell_rect(self, row: int, col: int) -> Level0Rect:
        if not (0 <= row < self.height and 0 <= col < self.width):
            raise InvalidSpatialReferenceError(f"mask cell ({row}, {col}) does not exist")
        x0, x1 = self.x_edge(col), self.x_edge(col + 1)
        y0, y1 = self.y_edge(row), self.y_edge(row + 1)
        return Level0Rect(x0, y0, x1 - x0, y1 - y0)

    @property
    def edge_adjustment(self) -> tuple[float, float]:
        """Level-zero pixels by which the last column/row differs from a regular cell."""
        return (
            self.level0_width - self.width * self.cell_x,
            self.level0_height - self.height * self.cell_y,
        )


@dataclass(frozen=True)
class TissueMask:
    """Candidate-tissue mask plus acquisition validity, both on `grid`.

    `tissue` is a boolean array `(grid.height, grid.width)`. `acquired`, when present, is a float
    array of the same shape with the acquired fraction of each cell (exactly 0/1 for stored-level
    masks). Tissue is only ever true in cells counted as acquired.
    """

    image_id: ImageId
    source_integrity: str | None
    scene: int
    grid: MaskGrid
    reader: ComponentVersion
    detector: ComponentVersion
    parameters: tuple[tuple[str, ConfigValue], ...]
    derived: tuple[tuple[str, ConfigValue], ...]
    input_channel_order: str
    channel_view: tuple[str, ...]
    validity_semantics: NoDataSemantics
    validity_policy: ValidityPolicy
    tissue_fraction: float
    acquired_fraction: float | None
    mask_digest: str
    capability_checks: tuple[CapabilityCheck, ...]
    limitations: tuple[str, ...]
    decoded_pixels: int
    tissue: object = field(compare=False, repr=False)
    acquired: object | None = field(default=None, compare=False, repr=False)
    notice: str = CANDIDATE_TISSUE_NOTICE

    def __post_init__(self) -> None:
        for f in (self.tissue_fraction, self.acquired_fraction):
            if f is not None and not (math.isfinite(f) and 0.0 <= f <= 1.0):
                raise InvalidImageAssetError("mask fractions must lie within [0, 1]")
        reported = self.validity_semantics is not NoDataSemantics.NOT_REPORTED
        if reported != (self.acquired is not None) or reported != (
            self.acquired_fraction is not None
        ):
            raise InvalidImageAssetError(
                "acquired data must be present exactly when the reader reports validity"
            )
        if not reported and self.validity_policy is not ValidityPolicy.ASSUME_ACQUIRED:
            raise InvalidImageAssetError(
                "unreported validity requires the explicit assume_acquired policy"
            )
        if not _SHA256.fullmatch(self.mask_digest):
            raise InvalidImageAssetError("mask digest must be a SHA-256 hex digest")
        if self.decoded_pixels < 1:
            raise InvalidImageAssetError("a mask is derived from at least one decoded pixel")
