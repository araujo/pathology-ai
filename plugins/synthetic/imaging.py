"""Procedural in-memory pyramidal images and a reader for them (no files, no patient data)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from domain.geometry import LevelRect, PixelSpacing, PyramidalGeometry, PyramidLevel
from domain.hashing import digest
from domain.ids import ImageId
from domain.image import UNKNOWN_CONTEXT, ChannelOrder, ClinicalContext, ImageAsset
from domain.modality import WHOLE_SLIDE_MICROSCOPY, ImageFormat, SpatialKind
from domain.provenance import ComponentVersion
from plugins.synthetic import PLUGIN_VERSION
from ports import ReaderCapabilities

SYNTHETIC_PYRAMID_FORMAT = ImageFormat("synthetic-pyramid-2d")

Pixels = npt.NDArray[np.uint8]


def synthetic_slide(
    image_id: str,
    *,
    width: int = 1024,
    height: int = 768,
    levels: int = 4,
    pixel_spacing_um: float | None = 0.25,
    channel_order: ChannelOrder | None = ChannelOrder.RGB,
    context: ClinicalContext = UNKNOWN_CONTEXT,
) -> ImageAsset:
    pyramid = tuple(
        PyramidLevel(i, max(1, width // 2**i), max(1, height // 2**i), float(2**i))
        for i in range(levels)
    )
    # Level-0 must match exactly; integer division keeps further levels consistent.
    geometry = PyramidalGeometry(
        width,
        height,
        pyramid,
        PixelSpacing(pixel_spacing_um, pixel_spacing_um) if pixel_spacing_um else None,
    )
    return ImageAsset(
        image_id=ImageId(image_id),
        modality=WHOLE_SLIDE_MICROSCOPY,
        image_format=SYNTHETIC_PYRAMID_FORMAT,
        spatial_reference=geometry,
        channel_order=channel_order,
        source_checksum="sha256:" + digest({"synthetic": image_id, "w": width, "h": height}),
        context=context,
    )


def _seed(image: ImageAsset) -> int:
    return int(digest(str(image.image_id))[:8], 16)


class SyntheticSlideReader:
    """Renders RGB content as a pure function of level-0 coordinates and image id."""

    def __init__(self, channel_order: ChannelOrder = ChannelOrder.RGB) -> None:
        self._order = channel_order

    @property
    def component(self) -> ComponentVersion:
        return ComponentVersion(
            "reader", f"synthetic-slide-reader[{self._order.value}]", PLUGIN_VERSION
        )

    @property
    def capabilities(self) -> ReaderCapabilities:
        return ReaderCapabilities(
            formats=frozenset({SYNTHETIC_PYRAMID_FORMAT.code}),
            spatial_kinds=frozenset({SpatialKind.PYRAMIDAL_2D}),
            channel_order=self._order,
            provides_physical_spacing=True,
            supports_regions=True,
            supports_thumbnails=True,
        )

    def read_region(self, image: ImageAsset, region: LevelRect) -> Pixels:
        geom = image.spatial_reference
        assert isinstance(geom, PyramidalGeometry)
        geom.to_level0(region)  # bounds check in canonical coordinates
        ds = geom.level(region.level).downsample
        h, w = int(region.height), int(region.width)
        ys = (region.y + np.arange(h) + 0.5) * ds
        xs = (region.x + np.arange(w) + 0.5) * ds
        yy, xx = np.meshgrid(ys / geom.height, xs / geom.width, indexing="ij")
        rng = np.random.default_rng(_seed(image))
        cx, cy, r = rng.uniform(0.35, 0.65), rng.uniform(0.35, 0.65), rng.uniform(0.2, 0.3)
        inside = (xx - cx) ** 2 + (yy - cy) ** 2 < r**2
        texture = 0.5 + 0.5 * np.sin(xx * 40 + rng.uniform(0, 6)) * np.cos(yy * 40)
        red = np.where(inside, 150 + 60 * texture, 242)
        green = np.where(inside, 60 + 40 * texture, 242)
        blue = np.where(inside, 140 + 50 * (1 - texture), 242)
        rgb = np.stack([red, green, blue], axis=-1).astype(np.uint8)
        return rgb[..., ::-1].copy() if self._order is ChannelOrder.BGR else rgb

    def thumbnail(self, image: ImageAsset, max_size: int) -> Pixels:
        geom = image.spatial_reference
        assert isinstance(geom, PyramidalGeometry)
        level = geom.levels[-1]
        for lvl in geom.levels:
            if max(lvl.width, lvl.height) <= max_size:
                level = lvl
                break
        return self.read_region(image, LevelRect(level.index, 0, 0, level.width, level.height))
