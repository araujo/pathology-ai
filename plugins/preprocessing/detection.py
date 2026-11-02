"""Candidate-tissue detection: bounded mask-source reads and the saturation/Otsu baseline.

Mask source. The finest stored level with at most `max_mask_pixels` pixels is read in bounded
tiles, one mask cell per pixel. If even the coarsest stored level is larger, that level is read
in tiles and block-averaged (acquired pixels only) by the smallest integer factor that fits.
Decoding stops at `max_decoded_pixels`; no level is ever decoded in a single unbounded read.

Detector `saturation_otsu`: HSV-style saturation `(max - min) / max` of an explicit RGB view,
Otsu's threshold over acquired cells only (floored at `min_threshold`), optional binary
opening/closing, and finally `tissue &= acquired`. Saturation is a brightfield colour cue: it
flags stained areas as candidate tissue. It does not know the stain, organ or disease, and it is
not a tissue segmentation of validated quality.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from domain.capabilities import PreprocessingRequirements, ValidityPolicy
from domain.errors import (
    CorruptImageError,
    InvalidSpatialReferenceError,
    PreprocessingBudgetError,
)
from domain.geometry import PyramidalGeometry
from domain.image import ChannelOrder
from domain.pipeline import ConfigValue
from domain.provenance import ComponentVersion
from domain.reading import NoDataSemantics, RegionReadRequest
from domain.tissue import MaskGrid
from plugins.preprocessing.arrays import (
    Bool2D,
    Float2D,
    UInt8,
    binary_close,
    binary_open,
    block_reduce,
    otsu_threshold,
)
from ports import DetectorInput, DetectorResult, OpenedImage

SATURATION_OTSU = "saturation_otsu"
SATURATION_OTSU_VERSION = "1"

CHANNEL_VIEWS = {
    ChannelOrder.RGB: "rgb: used as is",
    ChannelOrder.BGR: "bgr->rgb: channel axis reversed explicitly (detector-internal view)",
    ChannelOrder.RGBA: (
        "rgba->rgb: alpha dropped explicitly (detector-internal view); alpha was already "
        "reported by the reader as acquisition validity"
    ),
}
DETECTOR_LIMITATIONS = (
    "saturation is a brightfield colour cue; pale or faint staining, pen marks, bubbles and "
    "coloured debris can be misclassified",
    "candidate tissue only: not tumour detection, diagnosis or clinical tissue classification",
)


# --------------------------------------------------------------------------- config helpers


def cfg_int(cfg: Mapping[str, ConfigValue], key: str, default: int) -> int:
    value = cfg.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidSpatialReferenceError(f"config {key} must be an integer, got {value!r}")
    return value


def cfg_float(cfg: Mapping[str, ConfigValue], key: str, default: float) -> float:
    value = cfg.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise InvalidSpatialReferenceError(f"config {key} must be a number, got {value!r}")
    return float(value)


def cfg_str(cfg: Mapping[str, ConfigValue], key: str, default: str) -> str:
    value = cfg.get(key, default)
    if not isinstance(value, str):
        raise InvalidSpatialReferenceError(f"config {key} must be a string, got {value!r}")
    return value


def cfg_bool(cfg: Mapping[str, ConfigValue], key: str, default: bool) -> bool:
    value = cfg.get(key, default)
    if not isinstance(value, bool):
        raise InvalidSpatialReferenceError(f"config {key} must be a boolean, got {value!r}")
    return value


def validity_policy(cfg: Mapping[str, ConfigValue]) -> ValidityPolicy:
    value = cfg_str(cfg, "validity_policy", ValidityPolicy.REQUIRE_REPORTED.value)
    try:
        return ValidityPolicy(value)
    except ValueError:
        raise InvalidSpatialReferenceError(f"unknown validity_policy {value!r}") from None


# --------------------------------------------------------------------------- mask source


def choose_mask_grid(geometry: PyramidalGeometry, cfg: Mapping[str, ConfigValue]) -> MaskGrid:
    max_cells = cfg_int(cfg, "max_mask_pixels", 1 << 20)
    max_decoded = cfg_int(cfg, "max_decoded_pixels", 1 << 26)
    requested = cfg_int(cfg, "mask_level", -1)
    if max_cells < 1 or max_decoded < 1:
        raise InvalidSpatialReferenceError("mask budgets must be positive")
    if requested >= 0:
        level = geometry.level(requested)
    else:
        fitting = [lv for lv in geometry.levels if lv.width * lv.height <= max_cells]
        level = fitting[0] if fitting else geometry.levels[-1]
    decoded = level.width * level.height
    if decoded > max_decoded:
        raise PreprocessingBudgetError(
            f"tissue detection would decode {decoded} pixels at level {level.index}, above "
            f"max_decoded_pixels={max_decoded}; the image has no stored level small enough"
        )
    factor = max(1, math.ceil(math.sqrt(decoded / max_cells)))
    while math.ceil(level.width / factor) * math.ceil(level.height / factor) > max_cells:
        factor += 1
    return MaskGrid.for_level(geometry, level.index, factor)


def read_detector_input(
    image: OpenedImage, grid: MaskGrid, tile_px: int
) -> tuple[DetectorInput, int]:
    """Read the mask source level in bounded tiles; returns the input and decoded pixel count."""
    d = image.describe()
    level = d.geometry.level(grid.source_level)
    f = grid.block_factor
    tile = max(f, (tile_px // f) * f)
    reported = d.no_data is not NoDataSemantics.NOT_REPORTED
    values = np.zeros((grid.height, grid.width, d.samples_per_pixel), dtype=np.float64)
    acquired = np.zeros((grid.height, grid.width), dtype=np.float64)
    decoded = 0
    for y in range(0, level.height, tile):
        for x in range(0, level.width, tile):
            w, h = min(tile, level.width - x), min(tile, level.height - y)
            region = image.read_region(RegionReadRequest(level.index, x, y, w, h, scene=d.scene))
            pixels = np.asarray(region.pixels)
            if pixels.shape != (h, w, d.samples_per_pixel) or pixels.dtype != np.uint8:
                raise CorruptImageError(f"mask-source read returned {pixels.shape}/{pixels.dtype}")
            mask = region.valid_data.mask
            valid: Bool2D | None = np.asarray(mask, dtype=np.bool_) if mask is not None else None
            if reported and valid is None:
                raise CorruptImageError("the reader reports validity but returned no mask")
            block, frac = block_reduce(pixels, valid if reported else None, f)
            r0, c0 = y // f, x // f
            values[r0 : r0 + block.shape[0], c0 : c0 + block.shape[1]] = block
            acquired[r0 : r0 + frac.shape[0], c0 : c0 + frac.shape[1]] = frac
            decoded += w * h
    pixels8: UInt8 = np.clip(np.rint(values), 0, 255).astype(np.uint8)
    return (
        DetectorInput(
            pixels=pixels8,
            channel_order=d.channel_order,
            dtype=d.dtype,
            grid=grid,
            acquired=acquired if reported else None,
            validity=d.no_data,
        ),
        decoded,
    )


# --------------------------------------------------------------------------- baseline detector


def saturation(rgb: UInt8) -> UInt8:
    mx = rgb.max(axis=-1).astype(np.int32)
    mn = rgb.min(axis=-1).astype(np.int32)
    sat = np.where(mx > 0, ((mx - mn) * 255) // np.maximum(mx, 1), 0)
    out: UInt8 = sat.astype(np.uint8)
    return out


def rgb_view(pixels: UInt8, order: ChannelOrder) -> UInt8:
    if order is ChannelOrder.RGB:
        return pixels
    if order is ChannelOrder.BGR:
        return pixels[..., ::-1]
    if order is ChannelOrder.RGBA:
        return pixels[..., :3]
    raise InvalidSpatialReferenceError(f"no RGB view is defined for {order.value}")


class SaturationOtsuDetector:
    """Deterministic, NumPy-only baseline (SPEC-004 §6.2, ADR-004 R1)."""

    @property
    def name(self) -> str:
        return SATURATION_OTSU

    @property
    def component(self) -> ComponentVersion:
        return ComponentVersion("detector", f"tissue.{SATURATION_OTSU}", SATURATION_OTSU_VERSION)

    def requirements(self, config: Mapping[str, ConfigValue]) -> PreprocessingRequirements:
        return PreprocessingRequirements(
            channel_orders=frozenset(CHANNEL_VIEWS),
            dtypes=frozenset({"uint8"}),
            needs_regions=True,
            validity_policy=validity_policy(config),
        )

    def detect(self, data: DetectorInput, config: Mapping[str, ConfigValue]) -> DetectorResult:
        min_threshold = cfg_int(config, "min_threshold", 8)
        open_radius = cfg_int(config, "open_radius", 0)
        close_radius = cfg_int(config, "close_radius", 0)
        if not 0 <= min_threshold <= 255 or open_radius < 0 or close_radius < 0:
            raise InvalidSpatialReferenceError("invalid saturation_otsu parameters")
        pixels = np.asarray(data.pixels)
        sat = saturation(rgb_view(pixels, data.channel_order))
        acquired: Bool2D = (
            np.asarray(data.acquired) >= 0.5
            if data.acquired is not None
            else np.ones(sat.shape, dtype=np.bool_)
        )
        scored = int(acquired.sum())
        if scored == 0:
            tissue = np.zeros(sat.shape, dtype=np.bool_)
            otsu: ConfigValue = "undefined (no acquired cell)"
            threshold: ConfigValue = "undefined (no acquired cell)"
        else:
            o = otsu_threshold(sat[acquired])
            otsu, threshold = o, max(o, min_threshold)
            tissue = (sat > threshold) & acquired
            tissue = binary_close(binary_open(tissue, open_radius), close_radius) & acquired
        return DetectorResult(
            tissue=tissue,
            derived=(
                ("acquired_cells", scored),
                ("otsu_threshold", otsu),
                ("saturation_threshold", threshold),
                ("tissue_cells", int(tissue.sum())),
            ),
            channel_view=(CHANNEL_VIEWS[data.channel_order],),
            limitations=DETECTOR_LIMITATIONS,
        )


def mask_fractions(
    tissue: Bool2D, acquired: Float2D | None, grid: MaskGrid
) -> tuple[float, float | None]:
    """Area-weighted image fractions (cells at the stretched/clamped edge weigh their real area)."""
    xs = np.diff(np.minimum(np.arange(grid.width + 1) * grid.cell_x, grid.level0_width))
    ys = np.diff(np.minimum(np.arange(grid.height + 1) * grid.cell_y, grid.level0_height))
    xs[-1] += grid.level0_width - min(grid.width * grid.cell_x, grid.level0_width)
    ys[-1] += grid.level0_height - min(grid.height * grid.cell_y, grid.level0_height)
    area = float(xs.sum() * ys.sum())
    t = round(float(ys @ tissue.astype(np.float64) @ xs) / area, 9)
    a = None if acquired is None else round(float(ys @ acquired @ xs) / area, 9)
    return t, a
