"""Synthetic tissue-detection and patch-extraction stage operations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from domain.errors import ExecutionUnsupportedError, MissingPreprocessingError
from domain.geometry import Level0Rect, LevelRect, PyramidalGeometry
from domain.ids import Version
from domain.pipeline import OperationRef
from domain.provenance import ComponentVersion
from plugins.synthetic import PLUGIN_VERSION
from plugins.synthetic.imaging import SyntheticSlideReader
from ports import DataValue, StageContext

TISSUE_THRESHOLD = OperationRef("tissue.threshold", Version("1"))
PATCH_GRID = OperationRef("patch.grid", Version("1"))


@dataclass(frozen=True)
class TissueMask:
    level: int
    downsample: float
    mask: npt.NDArray[np.bool_]


@dataclass(frozen=True)
class PatchSet:
    """Patches in the model's channel order, with canonical level-0 rectangles."""

    level: int
    rects: tuple[Level0Rect, ...]
    pixels: npt.NDArray[np.uint8]  # (N, P, P, 3)


def _reader(ctx: StageContext) -> SyntheticSlideReader:
    if not isinstance(ctx.reader, SyntheticSlideReader):
        raise ExecutionUnsupportedError("synthetic operations require the synthetic reader")
    return ctx.reader


def _geometry(ctx: StageContext) -> PyramidalGeometry:
    g = ctx.image.spatial_reference
    if not isinstance(g, PyramidalGeometry):
        raise ExecutionUnsupportedError("synthetic operations handle 2D pyramidal images only")
    return g


class TissueThresholdOperation:
    @property
    def operation(self) -> OperationRef:
        return TISSUE_THRESHOLD

    @property
    def component(self) -> ComponentVersion:
        return ComponentVersion("operation", str(TISSUE_THRESHOLD), PLUGIN_VERSION)

    def run(self, inputs: Mapping[str, DataValue], ctx: StageContext) -> Mapping[str, object]:
        geom = _geometry(ctx)
        cfg = ctx.stage.config_dict()
        lowest = geom.levels[-1]
        rgb = _reader(ctx).read_region(
            ctx.image, LevelRect(lowest.index, 0, 0, lowest.width, lowest.height)
        )
        threshold = float(cfg.get("max_mean_intensity", 220))
        mask = rgb.mean(axis=-1) < threshold
        return {"mask": TissueMask(lowest.index, lowest.downsample, mask)}


class GridPatchOperation:
    """Regular grid at the contract's spacing; no resampling is performed or faked."""

    @property
    def operation(self) -> OperationRef:
        return PATCH_GRID

    @property
    def component(self) -> ComponentVersion:
        return ComponentVersion("operation", str(PATCH_GRID), PLUGIN_VERSION)

    def run(self, inputs: Mapping[str, DataValue], ctx: StageContext) -> Mapping[str, object]:
        pre = ctx.preprocessing
        if pre is None or pre.patch_size_px is None or pre.patch_stride_px is None:
            raise MissingPreprocessingError("patch extraction requires patch size and stride")
        if pre.target_pixel_spacing_um is None:
            raise MissingPreprocessingError("patch extraction requires a target pixel spacing")
        if pre.stain_normalization != "none":
            raise ExecutionUnsupportedError(
                f"stain normalization {pre.stain_normalization!r} is not implemented"
            )
        geom = _geometry(ctx)
        spacing = geom.require_pixel_spacing()
        level = next(
            (
                lvl
                for lvl in geom.levels
                if abs(spacing.x_um * lvl.downsample - pre.target_pixel_spacing_um)
                <= 0.01 * pre.target_pixel_spacing_um
            ),
            None,
        )
        if level is None:
            raise ExecutionUnsupportedError(
                f"no pyramid level at {pre.target_pixel_spacing_um} µm/px and resampling "
                "is not implemented by the synthetic patcher"
            )
        mask_value = inputs["mask"].payload
        assert isinstance(mask_value, TissueMask)
        cfg = ctx.stage.config_dict()
        min_tissue = float(cfg.get("min_tissue_fraction", 0.5))
        max_patches = int(cfg.get("max_patches", 64))
        size, stride = pre.patch_size_px, pre.patch_stride_px
        rects: list[Level0Rect] = []
        tiles: list[npt.NDArray[np.uint8]] = []
        reader = _reader(ctx)
        scale = level.downsample / mask_value.downsample
        for y in range(0, level.height - size + 1, stride):
            for x in range(0, level.width - size + 1, stride):
                my0, mx0 = int(y * scale), int(x * scale)
                my1 = max(my0 + 1, int((y + size) * scale))
                mx1 = max(mx0 + 1, int((x + size) * scale))
                if mask_value.mask[my0:my1, mx0:mx1].mean() < min_tissue:
                    continue
                region = LevelRect(level.index, x, y, size, size)
                pixels = reader.read_region(ctx.image, region)
                if ctx.channel_conversion is not None:
                    pixels = pixels[..., ::-1].copy()  # explicit, contract-declared reorder
                rects.append(geom.to_level0(region))
                tiles.append(pixels)
                if len(rects) >= max_patches:
                    break
            if len(rects) >= max_patches:
                break
        if not rects:
            raise ExecutionUnsupportedError("no tissue patches found; refusing to fabricate input")
        return {"patches": PatchSet(level.index, tuple(rects), np.stack(tiles))}
