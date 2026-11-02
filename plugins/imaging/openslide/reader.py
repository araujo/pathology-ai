"""OpenSlide adapter, limited to the vendor formats tested against a real fixture (Aperio SVS).

Behaviour established on `CMU-1-Small-Region.svs` (see the SPEC-003 report):

* OpenSlide reports only stored pyramid levels (`level_downsamples`); they are used verbatim.
* `read_region` takes a level-zero location and a size in the target level's pixels, and returns
  RGBA. openslide-python converts OpenSlide's premultiplied ARGB to straight (un-premultiplied)
  RGBA; that conversion is recorded. The adapter returns RGBA unchanged and never drops alpha.
* Alpha 0 marks locations where the slide has no data (outside the slide, or holes in sparse
  slides); it is reported as the valid-data mask. Out-of-bounds requests are rejected before
  OpenSlide is called, so transparency only ever describes in-bounds locations.
* Physical spacing comes from `openslide.mpp-x` / `openslide.mpp-y` (µm); absent means unknown.
* OpenSlide's vendor detection also claims formats this adapter has not been tested with (e.g. it
  detects Zeiss CZI as `zeiss` but cannot decode JPEG XR). Only allow-listed vendors are claimed.
"""

from __future__ import annotations

import importlib
import importlib.util
import math
from pathlib import Path
from typing import Any

import numpy as np

from domain.errors import (
    CorruptImageError,
    DomainError,
    ReaderBackendUnavailableError,
    UnsupportedReadOperationError,
)
from domain.geometry import LevelRect, PixelSpacing, PyramidalGeometry, PyramidLevel
from domain.image import ChannelOrder, ImageAsset
from domain.modality import ImageFormat, SpatialKind
from domain.provenance import ComponentVersion
from domain.reading import (
    ImageDescription,
    NoDataSemantics,
    PixelRegion,
    ReadLimits,
    RegionReadProvenance,
    RegionReadRequest,
    SourceIntegrity,
    ValidDataReport,
    validate_region,
)
from plugins.imaging.common import ADAPTER_VERSION, ClosableHandle, distribution_component
from ports import ImageSource, ReaderCapabilities, SourceProbe

APERIO_SVS_FORMAT = ImageFormat("aperio-svs")
BACKEND_NAME = "openslide"
_DISTRIBUTION = "openslide-python"
# OpenSlide vendor -> format code. Extend only with a tested fixture.
_TESTED_VENDORS = {"aperio": APERIO_SVS_FORMAT}

DECODE = (
    "openslide-python: premultiplied ARGB converted to straight RGBA (alpha kept); "
    "otherwise as decoded by OpenSlide"
)
DECODE_LEVEL = "OpenSlide: region read from the stored pyramid level (no resampling requested)"
ALPHA_LIMITATION = (
    "alpha 0 marks locations OpenSlide has no data for; acquired pixels include imaged "
    "background (e.g. glass), which is not distinguished from tissue"
)


def _backend_module() -> Any:
    try:
        return importlib.import_module("openslide")
    except ImportError as exc:
        raise ReaderBackendUnavailableError(
            "the OpenSlide reader needs openslide-python and the OpenSlide library; install "
            "the optional extra `pathology-ai[openslide]`"
        ) from exc


class OpenSlideBackend:
    @property
    def name(self) -> str:
        return BACKEND_NAME

    @property
    def component(self) -> ComponentVersion:
        return ComponentVersion("reader", BACKEND_NAME, ADAPTER_VERSION)

    @property
    def backend_component(self) -> ComponentVersion:
        component = distribution_component(_DISTRIBUTION)
        if self.available():
            try:
                library = str(_backend_module().__library_version__)
            except Exception:  # pragma: no cover - native library failed to load
                library = "unknown"
            return ComponentVersion("backend", _DISTRIBUTION, f"{component.version}+lib{library}")
        return component

    @property
    def capabilities(self) -> ReaderCapabilities:
        return ReaderCapabilities(
            formats=frozenset(f.code for f in _TESTED_VENDORS.values()),
            spatial_kinds=frozenset({SpatialKind.PYRAMIDAL_2D}),
            channel_order=ChannelOrder.RGBA,
            provides_physical_spacing=True,
            supports_regions=True,
            reports_no_data_regions=True,
            detects_content=True,
        )

    def available(self) -> bool:
        if importlib.util.find_spec("openslide") is None:
            return False
        try:
            _backend_module()
        except Exception:
            return False
        return True

    def probe(self, path: Path) -> SourceProbe:
        if not self.available():
            return SourceProbe(False, None, "OpenSlide is not installed; content not probed")
        try:
            vendor = _backend_module().OpenSlide.detect_format(str(path))
        except Exception as exc:
            return SourceProbe(False, None, f"detection failed: {exc}")
        fmt = _TESTED_VENDORS.get(vendor) if vendor else None
        if fmt is None:
            return SourceProbe(False, None, f"vendor {vendor!r} is not a tested OpenSlide format")
        return SourceProbe(True, fmt.code, f"OpenSlide vendor {vendor!r}")

    def open(
        self,
        source: ImageSource,
        *,
        scene: int | None = None,
        limits: ReadLimits | None = None,
        integrity: SourceIntegrity | None = None,
    ) -> OpenSlideImage:
        return OpenSlideImage(
            source,
            scene=scene,
            limits=limits or ReadLimits(),
            integrity=integrity,
            reader=self.component,
            backend=self.backend_component,
        )


def _float_property(props: Any, key: str) -> float | None:
    try:
        value = float(props[key])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None


class OpenSlideImage(ClosableHandle):
    """One open slide (OpenSlide exposes a single image; scene 0)."""

    def __init__(
        self,
        source: ImageSource,
        *,
        scene: int | None,
        limits: ReadLimits,
        integrity: SourceIntegrity | None,
        reader: ComponentVersion,
        backend: ComponentVersion,
    ) -> None:
        if scene not in (None, 0):
            raise UnsupportedReadOperationError(f"scene {scene} does not exist (single image)")
        module = _backend_module()
        vendor = module.OpenSlide.detect_format(str(source.path))
        fmt = _TESTED_VENDORS.get(vendor) if vendor else None
        if fmt is None:
            raise UnsupportedReadOperationError(
                f"OpenSlide vendor {vendor!r} is not a tested format for this adapter"
            )
        self._limits = limits
        try:
            self._slide = module.OpenSlide(str(source.path))
        except Exception as exc:
            raise CorruptImageError(f"cannot open {source.image_id} with OpenSlide: {exc}") from exc
        try:
            self._description = self._describe(source, fmt, integrity, reader, backend)
        except DomainError:
            self.close()
            raise
        except Exception as exc:
            self.close()
            raise CorruptImageError(
                f"cannot read slide structure of {source.image_id}: {exc}"
            ) from exc

    def _release(self) -> None:
        self._slide.close()

    def _describe(
        self,
        source: ImageSource,
        fmt: ImageFormat,
        integrity: SourceIntegrity | None,
        reader: ComponentVersion,
        backend: ComponentVersion,
    ) -> ImageDescription:
        slide = self._slide
        width, height = (int(v) for v in slide.dimensions)
        levels = []
        for i, ((w, h), ds) in enumerate(
            zip(slide.level_dimensions, slide.level_downsamples, strict=True)
        ):
            levels.append(PyramidLevel(i, int(w), int(h), 1.0 if i == 0 else float(ds)))
        props = slide.properties
        mpp_x, mpp_y = (
            _float_property(props, "openslide.mpp-x"),
            _float_property(props, "openslide.mpp-y"),
        )
        spacing = PixelSpacing(mpp_x, mpp_y) if mpp_x and mpp_y else None
        acquisition = tuple(
            (key, str(props[key]))
            for key in ("openslide.vendor", "openslide.objective-power")
            if key in props
        )
        asset = ImageAsset(
            image_id=source.image_id,
            modality=source.modality,
            image_format=fmt,
            spatial_reference=PyramidalGeometry(width, height, tuple(levels), spacing),
            channel_order=ChannelOrder.RGBA,
            source_checksum=integrity.token if integrity else None,
        )
        return ImageDescription(
            asset=asset,
            reader=reader,
            backend=backend,
            scene=0,
            scene_count=1,
            native_offset=None,
            pixel_type="RGBA8 (openslide-python)",
            dtype="uint8",
            samples_per_pixel=4,
            channel_count=4,
            z_planes=1,
            time_points=1,
            spacing_source="file:openslide.mpp-x/openslide.mpp-y (µm)" if spacing else None,
            level_source="OpenSlide stored levels (level_dimensions, level_downsamples)",
            no_data=NoDataSemantics.BACKEND_TRANSPARENCY,
            decoder_transformations=(DECODE,),
            acquisition=acquisition,
            missing_metadata=() if spacing else ("physical_spacing",),
            integrity=integrity,
        )

    def describe(self) -> ImageDescription:
        self._require_open()
        return self._description

    def read_region(self, request: RegionReadRequest) -> PixelRegion:
        self._require_open()
        d = self._description
        if request.scene not in (None, 0):
            raise UnsupportedReadOperationError(f"scene {request.scene} does not exist")
        for dim, value in (("z", request.z), ("t", request.t)):
            if value not in (None, 0):
                raise UnsupportedReadOperationError(f"{dim}={value} does not exist (single plane)")
        if request.channels is not None:
            raise UnsupportedReadOperationError(
                "channel subsets are not supported; regions return native RGBA"
            )
        geometry = d.geometry
        level0 = validate_region(geometry, request)
        self._limits.check(request.width, request.height)
        downsample = geometry.levels[request.level].downsample
        location = (round(level0.x), round(level0.y))
        try:
            image = self._slide.read_region(
                location, request.level, (request.width, request.height)
            )
            pixels = np.asarray(image)
        except Exception as exc:
            raise CorruptImageError(f"OpenSlide decode failed for {d.image_id}: {exc}") from exc
        expected = (request.height, request.width, 4)
        if pixels.shape != expected or pixels.dtype != np.uint8:
            raise CorruptImageError(f"decoder returned {pixels.shape}, expected {expected}")
        mask = pixels[..., 3] > 0
        valid = ValidDataReport.from_fraction(
            NoDataSemantics.BACKEND_TRANSPARENCY, float(mask.mean()), mask, (ALPHA_LIMITATION,)
        )
        transformations = (DECODE,) if request.level == 0 else (DECODE, DECODE_LEVEL)
        provenance = RegionReadProvenance(
            image_id=d.image_id,
            reader=d.reader,
            backend=d.backend,
            source_integrity=d.asset.source_checksum,
            scene=0,
            requested=request.level_rect(),
            returned=LevelRect(request.level, request.x, request.y, request.width, request.height),
            level0=level0,
            downsample=downsample,
            native_region=(location[0], location[1], request.width, request.height),
            shape=tuple(int(s) for s in pixels.shape),
            dtype="uint8",
            channel_order=ChannelOrder.RGBA,
            decoder_transformations=transformations,
            metadata_limitations=d.missing_metadata + valid.limitations,
            validity=valid.validity,
            acquired_fraction=valid.acquired_fraction,
        )
        return PixelRegion(pixels, provenance, valid)
