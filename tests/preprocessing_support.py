"""In-memory images for preprocessing tests: an `OpenedImage` over NumPy arrays.

`ArrayImage` implements the reader port exactly like a real adapter (bounds and limits checked
before slicing, valid-data report per declared semantics) and records every request, so tests can
assert what was read. Content is synthetic: no patient data, no file.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from application.preprocessing_service import WSI_PREPROCESSING_TASK
from application.runner import ExecutionOutcome, ExecutionRequest
from bootstrap import Platform
from domain.geometry import LevelRect, PixelSpacing, PyramidalGeometry, PyramidLevel
from domain.ids import ImageId, PipelineId, Version
from domain.image import ChannelOrder, ImageAsset
from domain.modality import WHOLE_SLIDE_MICROSCOPY, ImageFormat, SpatialKind
from domain.pipeline import ConfigValue, PipelineRef
from domain.provenance import ComponentVersion
from domain.reading import (
    DataValidity,
    ImageDescription,
    NoDataSemantics,
    PixelRegion,
    ReadLimits,
    RegionReadProvenance,
    RegionReadRequest,
    ValidDataReport,
    validate_region,
)
from plugins.imaging.common import ClosableHandle
from ports import ReaderCapabilities

PIPELINE = PipelineRef(PipelineId("wsi-tissue-patch-extraction"), Version("1.0.0"))
GLASS = (238, 236, 240)  # bright, nearly achromatic background (acquired)
STAIN = (190, 90, 160)  # saturated "stained" colour (candidate tissue)
READER = ComponentVersion("reader", "test-array", "1")

U8 = npt.NDArray[np.uint8]
B2 = npt.NDArray[np.bool_]


def slide_rgb(
    width: int = 1024,
    height: int = 768,
    tissue: Sequence[tuple[int, int, int, int]] = ((256, 128, 512, 384),),
    unacquired: Sequence[tuple[int, int, int, int]] = ((0, 640, 1024, 128),),
) -> tuple[U8, B2]:
    """Level-zero RGB pixels and acquisition mask. Rectangles are (x, y, w, h)."""
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    rgb[:] = GLASS
    acquired = np.ones((height, width), dtype=np.bool_)
    for x, y, w, h in tissue:
        rgb[y : y + h, x : x + w] = STAIN
    for x, y, w, h in unacquired:
        rgb[y : y + h, x : x + w] = 0
        acquired[y : y + h, x : x + w] = False
    return rgb, acquired


def downsample(a: npt.NDArray[Any], factor: int) -> npt.NDArray[Any]:
    h, w = a.shape[0] // factor, a.shape[1] // factor
    cut = a[: h * factor, : w * factor]
    if a.dtype == np.bool_:
        return cut.reshape(h, factor, w, factor).mean(axis=(1, 3)) >= 0.5
    m = cut.reshape(h, factor, w, factor, *a.shape[2:]).mean(axis=(1, 3))
    return np.rint(m).astype(a.dtype)


class ArrayImage(ClosableHandle):
    """A pyramid of in-memory arrays exposed through the `OpenedImage` port."""

    def __init__(
        self,
        image_id: str,
        level0: U8,
        *,
        acquired: B2 | None = None,
        channel_order: ChannelOrder = ChannelOrder.RGB,
        no_data: NoDataSemantics = NoDataSemantics.ACQUISITION_COVERAGE,
        levels: int = 4,
        spacing_um: float | None = 0.25,
        z_planes: int = 1,
        fmt: str = "test-array",
        limits: ReadLimits | None = None,
    ) -> None:
        self.requests: list[RegionReadRequest] = []
        self._limits = limits or ReadLimits()
        self._pixels = [level0] + [downsample(level0, 2**i) for i in range(1, levels)]
        acq = acquired if acquired is not None else np.ones(level0.shape[:2], dtype=np.bool_)
        self._acquired = [acq] + [downsample(acq, 2**i) for i in range(1, levels)]
        h, w = level0.shape[:2]
        geometry = PyramidalGeometry(
            w,
            h,
            tuple(
                PyramidLevel(i, p.shape[1], p.shape[0], float(2**i))
                for i, p in enumerate(self._pixels)
            ),
            PixelSpacing(spacing_um, spacing_um) if spacing_um else None,
        )
        asset = ImageAsset(
            ImageId(image_id),
            WHOLE_SLIDE_MICROSCOPY,
            ImageFormat(fmt),
            geometry,
            channel_order,
            source_checksum=f"test:{image_id}",
        )
        self._description = ImageDescription(
            asset=asset,
            reader=READER,
            backend=ComponentVersion("backend", "numpy", np.__version__),
            scene=0,
            scene_count=1,
            native_offset=None,
            pixel_type=f"{channel_order.value}8",
            dtype=str(level0.dtype),
            samples_per_pixel=level0.shape[2],
            channel_count=1,
            z_planes=z_planes,
            time_points=1,
            spacing_source="test" if spacing_um else None,
            level_source="test pyramid (2x mean)",
            no_data=no_data,
            decoder_transformations=("none (in-memory)",),
            missing_metadata=() if spacing_um else ("physical_spacing",),
        )

    def _release(self) -> None:
        pass

    def describe(self) -> ImageDescription:
        self._require_open()
        return self._description

    def read_region(self, request: RegionReadRequest) -> PixelRegion:
        self._require_open()
        d = self._description
        level0 = validate_region(d.geometry, request)
        self._limits.check(request.width, request.height)
        self.requests.append(request)
        sl = np.s_[request.y : request.y + request.height, request.x : request.x + request.width]
        pixels = self._pixels[request.level][sl].copy()
        if d.no_data is NoDataSemantics.NOT_REPORTED:
            valid = ValidDataReport(
                NoDataSemantics.NOT_REPORTED, DataValidity.UNKNOWN, None, None, ("test",)
            )
        elif d.no_data is NoDataSemantics.BACKEND_TRANSPARENCY:
            mask = pixels[..., 3] > 0
            valid = ValidDataReport.from_fraction(d.no_data, float(mask.mean()), mask)
        else:
            mask = self._acquired[request.level][sl].copy()
            valid = ValidDataReport.from_fraction(d.no_data, float(mask.mean()), mask)
        prov = RegionReadProvenance(
            image_id=d.image_id,
            reader=d.reader,
            backend=d.backend,
            source_integrity=d.asset.source_checksum,
            scene=0,
            requested=request.level_rect(),
            returned=LevelRect(request.level, request.x, request.y, request.width, request.height),
            level0=level0,
            downsample=d.geometry.levels[request.level].downsample,
            native_region=None,
            shape=tuple(int(s) for s in pixels.shape),
            dtype=d.dtype,
            channel_order=d.channel_order,
            decoder_transformations=d.decoder_transformations,
            metadata_limitations=d.missing_metadata,
            validity=valid.validity,
            acquired_fraction=valid.acquired_fraction,
        )
        return PixelRegion(pixels, prov, valid)


class ArrayReader:
    """`ImageReader` registration for `ArrayImage` formats (runner reader resolution)."""

    def __init__(
        self,
        fmt: str = "test-array",
        channel_order: ChannelOrder = ChannelOrder.RGB,
        *,
        supports_regions: bool = True,
    ) -> None:
        self._caps = ReaderCapabilities(
            formats=frozenset({fmt}),
            spatial_kinds=frozenset({SpatialKind.PYRAMIDAL_2D}),
            channel_order=channel_order,
            provides_physical_spacing=True,
            supports_regions=supports_regions,
            reports_no_data_regions=True,
        )

    @property
    def component(self) -> ComponentVersion:
        return READER

    @property
    def capabilities(self) -> ReaderCapabilities:
        return self._caps


def run_preprocessing(
    platform: Platform,
    image: ArrayImage,
    overrides: Sequence[tuple[str, str, ConfigValue]] = (),
) -> ExecutionOutcome:
    fmt = image.describe().asset.image_format.code
    if platform.readers._by_format.find(fmt) is None:
        platform.readers.register(ArrayReader(fmt, image.describe().channel_order))
    return platform.runner.run(
        ExecutionRequest(
            pipeline=platform.pipelines.get(PIPELINE),
            image=image.describe().asset,
            task=WSI_PREPROCESSING_TASK,
            models={},
            preprocessing=None,
            research_use_acknowledged=True,
            source=image,
            config_overrides=tuple(overrides),
        )
    )


# Small patches so the 1024x768 test slide yields a meaningful grid (level 1 at 0.5 µm/px).
SMALL = (("plan", "width", 64), ("plan", "height", 64))
