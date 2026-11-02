"""Zeiss CZI adapter (pylibCZIrw).

Behaviour established on real and generated CZI files (see the SPEC-003 report):

* Native CZI coordinates are global stage pixels and may be negative. Level-zero pixel (0, 0) is
  the top-left of the selected scene's level-zero data rectangle
  (`scenes_bounding_rectangle_no_pyramid`, falling back to `total_bounding_rectangle_no_pyramid`
  for files without scenes). The rectangle that includes pyramid subblocks can be larger, because
  pyramid tiles are padded, so it is never used as the image extent.
* pylibCZIrw decodes `Bgr24` as `uint8 (h, w, 3)` in **BGR** order. The adapter returns it
  unchanged and reports `ChannelOrder.BGR`.
* `read(zoom=...)` accepts any zoom factor, so decoder scaling is not evidence of a stored
  resolution. Pyramid levels are reported only for subsample ratios that stored subblocks
  actually have.
* pylibCZIrw fills areas without level-zero subblocks with `background_pixel` and does not reject
  ROIs outside the image or unknown scene indices. The adapter therefore validates bounds and
  scenes itself, and reports acquisition coverage from level-zero subblock rectangles.
* Physical spacing comes from `Metadata/Scaling/Items/Distance` (metres per pixel). Zero or absent
  values mean "unknown" and are never replaced.
* The origin of the padded `total_bounding_rectangle` (which includes pyramid subblocks) is
  reported as the acquisition fact `native.padded_origin`. It is never used as the image frame;
  it only lets callers translate coordinates produced in that padded frame (`domain.frames`).
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
from domain.frames import NATIVE_PADDED_ORIGIN
from domain.geometry import LevelRect, PixelSpacing, PyramidalGeometry, PyramidLevel
from domain.image import ChannelOrder, ImageAsset
from domain.modality import ImageFormat, SpatialKind
from domain.provenance import ComponentVersion
from domain.reading import (
    ImageDescription,
    NativeFrameOffset,
    NoDataSemantics,
    PixelRegion,
    ReadLimits,
    RegionReadProvenance,
    RegionReadRequest,
    SourceIntegrity,
    ValidDataReport,
    validate_region,
)
from plugins.imaging.common import (
    ADAPTER_VERSION,
    ClosableHandle,
    Tiles,
    coverage_mask,
    distribution_component,
)
from ports import ImageSource, ReaderCapabilities, SourceProbe

CZI_FORMAT = ImageFormat("zeiss-czi")
BACKEND_NAME = "czi-pylibczirw"
_DISTRIBUTION = "pylibczirw"
_MAGIC = b"ZISRAWFILE"
_SUPPORTED_PIXEL_TYPES = {"Bgr24": (ChannelOrder.BGR, "uint8", 3)}
_PLANE = {"C": 0, "Z": 0, "T": 0}
_BACKGROUND = (0, 0, 0)
_RATIO_DECIMALS = 6

DECODE_LEVEL0 = "pylibCZIrw: subblock decompression only (no colour, channel or value change)"
DECODE_PYRAMID = (
    "pylibCZIrw: region composed from stored pyramid subblocks at zoom=1/downsample "
    "(nearest-neighbour placement by the backend)"
)
FILL_LIMITATION = (
    "pixels without level-zero subblocks are filled by the backend: with (0, 0, 0) at level 0, "
    "and with whatever fill the scanner stored in pyramid subblocks at coarser levels; use the "
    "valid-data mask, not pixel values"
)
BACKGROUND_LIMITATION = (
    "acquired pixels include imaged background (e.g. glass); tissue vs background is not "
    "distinguished by the reader"
)


def _backend_module() -> Any:
    try:
        return importlib.import_module("pylibCZIrw.czi")
    except ImportError as exc:
        raise ReaderBackendUnavailableError(
            "the CZI reader needs pylibCZIrw; install the optional extra `pathology-ai[czi]`"
        ) from exc


class CziBackend:
    """Registers the CZI adapter. Only the `Bgr24` pixel type is supported (tested)."""

    @property
    def name(self) -> str:
        return BACKEND_NAME

    @property
    def component(self) -> ComponentVersion:
        return ComponentVersion("reader", BACKEND_NAME, ADAPTER_VERSION)

    @property
    def backend_component(self) -> ComponentVersion:
        return distribution_component(_DISTRIBUTION)

    @property
    def capabilities(self) -> ReaderCapabilities:
        return ReaderCapabilities(
            formats=frozenset({CZI_FORMAT.code}),
            spatial_kinds=frozenset({SpatialKind.PYRAMIDAL_2D}),
            channel_order=ChannelOrder.BGR,
            provides_physical_spacing=True,
            supports_regions=True,
            reports_no_data_regions=True,
            supports_scenes=True,
            detects_content=True,
        )

    def available(self) -> bool:
        return importlib.util.find_spec("pylibCZIrw") is not None

    def probe(self, path: Path) -> SourceProbe:
        """Content sniffing: a CZI file starts with the `ZISRAWFILE` segment id."""
        try:
            with path.open("rb") as handle:
                head = handle.read(len(_MAGIC))
        except OSError as exc:
            return SourceProbe(False, None, f"unreadable: {exc.strerror}")
        if head == _MAGIC:
            missing = "; pylibCZIrw missing, install `pathology-ai[czi]`"
            detail = "ZISRAWFILE header" + ("" if self.available() else missing)
            return SourceProbe(True, CZI_FORMAT.code, detail)
        return SourceProbe(False, None, "no ZISRAWFILE header")

    def open(
        self,
        source: ImageSource,
        *,
        scene: int | None = None,
        limits: ReadLimits | None = None,
        integrity: SourceIntegrity | None = None,
    ) -> CziImage:
        return CziImage(
            source,
            scene=scene,
            limits=limits or ReadLimits(),
            integrity=integrity,
            reader=self.component,
            backend=self.backend_component,
        )


def _spacing_um(metadata: Any) -> PixelSpacing | None:
    """Level-zero spacing in µm from `Scaling/Items/Distance` (metres); None when unknown."""
    try:
        items = metadata["ImageDocument"]["Metadata"]["Scaling"]["Items"]["Distance"]
    except (KeyError, TypeError):
        return None
    if isinstance(items, dict):
        items = [items]
    values: dict[str, float] = {}
    for item in items if isinstance(items, list) else []:
        try:
            values[str(item["@Id"])] = float(item["Value"])
        except (KeyError, TypeError, ValueError):
            continue
    x, y = values.get("X"), values.get("Y")
    if x is None or y is None or not (math.isfinite(x) and math.isfinite(y)) or x <= 0 or y <= 0:
        return None
    return PixelSpacing(x * 1e6, y * 1e6)


def _acquisition(metadata: Any) -> tuple[tuple[str, str], ...]:
    """A small allow-list of non-identifying acquisition facts (no dates, users or names)."""
    out: list[tuple[str, str]] = []
    try:
        info = metadata["ImageDocument"]["Metadata"]["Information"]
    except (KeyError, TypeError):
        return ()
    app = info.get("Application") if isinstance(info, dict) else None
    if isinstance(app, dict):
        for key in ("Name", "Version"):
            if isinstance(app.get(key), str):
                out.append((f"application.{key.lower()}", app[key]))
    image = info.get("Image") if isinstance(info, dict) else None
    if isinstance(image, dict):
        for key in ("OriginalCompressionMethod", "ComponentBitCount"):
            if isinstance(image.get(key), str):
                out.append((f"image.{key}", image[key]))
    return tuple(out)


class CziImage(ClosableHandle):
    """One open CZI scene. Close it (or use it as a context manager)."""

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
        module = _backend_module()
        self._limits = limits
        try:
            self._doc = module.CziReader(str(source.path))
        except DomainError:
            raise
        except Exception as exc:
            raise CorruptImageError(f"cannot open {source.image_id} as CZI: {exc}") from exc
        try:
            self._description, self._tiles, self._scene_arg = self._describe(
                source, scene, integrity, reader, backend
            )
        except DomainError:
            self.close()
            raise
        except Exception as exc:
            self.close()
            raise CorruptImageError(
                f"cannot read CZI structure of {source.image_id}: {exc}"
            ) from exc

    def _release(self) -> None:
        self._doc.close()

    # ------------------------------------------------------------------ metadata

    def _select_scene(self, scene: int | None) -> tuple[int, int | None, Any, int]:
        scenes = dict(self._doc.scenes_bounding_rectangle_no_pyramid)
        if not scenes:
            if scene not in (None, 0):
                raise UnsupportedReadOperationError(f"scene {scene} requested; the file has none")
            return 0, None, self._doc.total_bounding_rectangle_no_pyramid, 1
        if scene is None:
            if len(scenes) > 1:
                raise UnsupportedReadOperationError(
                    f"the file has {len(scenes)} scenes {sorted(scenes)}; select one explicitly"
                )
            scene = next(iter(scenes))
        if scene not in scenes:
            raise UnsupportedReadOperationError(
                f"scene {scene} does not exist (available: {sorted(scenes)})"
            )
        return scene, scene, scenes[scene], len(scenes)

    def _subblocks(self, scene_arg: int | None) -> tuple[Tiles, list[float], list[str]]:
        """Level-zero tiles and stored pyramid ratios of the selected scene."""
        infos: list[Any] = []

        def collect(_index: int, info: Any) -> bool:
            infos.append(info)
            return True

        self._doc.enumerate_subblocks(collect)
        level0: list[tuple[int, int, int, int]] = []
        ratios: set[float] = set()
        notes: list[str] = []
        for info in infos:
            if scene_arg is not None and info.coordinate.to_dict().get("S", 0) != scene_arg:
                continue
            lr, ps = info.logicalRect, info.physicalSize
            if ps.w < 1 or ps.h < 1:
                continue
            rx = round(lr.w / ps.w, _RATIO_DECIMALS)
            ry = round(lr.h / ps.h, _RATIO_DECIMALS)
            if rx != ry:
                notes.append(f"anisotropic subblock ratio {rx}x{ry} ignored")
                continue
            if rx == 1.0:
                level0.append((lr.x, lr.y, lr.x + lr.w, lr.y + lr.h))
            elif rx > 1.0:
                ratios.add(rx)
        tiles = np.asarray(level0, dtype=np.int64).reshape(-1, 4)
        return tiles, sorted(ratios), sorted(set(notes))

    def _describe(
        self,
        source: ImageSource,
        scene: int | None,
        integrity: SourceIntegrity | None,
        reader: ComponentVersion,
        backend: ComponentVersion,
    ) -> tuple[ImageDescription, Tiles, int | None]:
        pixel_types = set(dict(self._doc.pixel_types).values())
        if len(pixel_types) != 1 or next(iter(pixel_types)) not in _SUPPORTED_PIXEL_TYPES:
            raise UnsupportedReadOperationError(
                f"CZI pixel type(s) {sorted(pixel_types)} are not supported "
                f"(supported: {sorted(_SUPPORTED_PIXEL_TYPES)})"
            )
        pixel_type = next(iter(pixel_types))
        order, dtype, samples = _SUPPORTED_PIXEL_TYPES[pixel_type]

        bbox = dict(self._doc.total_bounding_box)
        sizes = {dim: end - start for dim, (start, end) in bbox.items() if dim not in ("X", "Y")}
        multi = {dim: n for dim, n in sizes.items() if n != 1}
        if multi:
            raise UnsupportedReadOperationError(
                f"multi-plane CZI dimensions {multi} are not supported; this adapter reads "
                "single C/Z/T planes only and never picks one implicitly"
            )

        scene_index, scene_arg, rect, scene_count = self._select_scene(scene)
        if rect.w < 1 or rect.h < 1:
            raise CorruptImageError(f"scene {scene_index} has an empty level-zero rectangle")
        offset = NativeFrameOffset(int(rect.x), int(rect.y))

        tiles, ratios, notes = self._subblocks(scene_arg)
        if tiles.shape[0] == 0:
            raise CorruptImageError(f"scene {scene_index} has no level-zero subblocks")
        tiles = tiles - np.array([offset.origin_x, offset.origin_y] * 2, dtype=np.int64)

        width, height = int(rect.w), int(rect.h)
        levels = [PyramidLevel(0, width, height, 1.0)]
        for ratio in ratios:
            w, h = math.floor(width / ratio), math.floor(height / ratio)
            if w >= 1 and h >= 1:
                levels.append(PyramidLevel(len(levels), w, h, ratio))

        spacing = _spacing_um(self._doc.metadata)
        missing = [] if spacing is not None else ["physical_spacing"]
        asset = ImageAsset(
            image_id=source.image_id,
            modality=source.modality,
            image_format=CZI_FORMAT,
            spatial_reference=PyramidalGeometry(width, height, tuple(levels), spacing),
            channel_order=order,
            source_checksum=integrity.token if integrity else None,
        )
        description = ImageDescription(
            asset=asset,
            reader=reader,
            backend=backend,
            scene=scene_index,
            scene_count=scene_count,
            native_offset=offset,
            pixel_type=pixel_type,
            dtype=dtype,
            samples_per_pixel=samples,
            channel_count=sizes.get("C", 1),
            z_planes=sizes.get("Z", 1),
            time_points=sizes.get("T", 1),
            spacing_source=(
                "file:ImageDocument/Metadata/Scaling/Items/Distance (metres per pixel, "
                "converted to micrometres)"
                if spacing
                else None
            ),
            level_source=(
                "stored subblock subsample ratios (logical size / stored size); decoder zoom "
                "factors are not reported as levels" + (f"; {'; '.join(notes)}" if notes else "")
            ),
            no_data=NoDataSemantics.ACQUISITION_COVERAGE,
            decoder_transformations=(DECODE_LEVEL0,),
            acquisition=_acquisition(self._doc.metadata) + self._padded_origin(),
            missing_metadata=tuple(missing),
            integrity=integrity,
        )
        return description, tiles, scene_arg

    def _padded_origin(self) -> tuple[tuple[str, str], ...]:
        padded = self._doc.total_bounding_rectangle
        return ((NATIVE_PADDED_ORIGIN, f"{int(padded.x)},{int(padded.y)}"),)

    def describe(self) -> ImageDescription:
        self._require_open()
        return self._description

    # ------------------------------------------------------------------ regions

    def _check_selection(self, request: RegionReadRequest) -> None:
        d = self._description
        if request.scene is not None and request.scene != d.scene:
            raise UnsupportedReadOperationError(
                f"request for scene {request.scene}, but scene {d.scene} is open"
            )
        for dim, value in (("z", request.z), ("t", request.t)):
            if value not in (None, 0):
                raise UnsupportedReadOperationError(f"{dim}={value} does not exist (single plane)")
        if request.channels is not None:
            raise UnsupportedReadOperationError(
                "channel subsets are not supported; regions return all native samples in "
                f"{d.channel_order.value} order"
            )

    def read_region(self, request: RegionReadRequest) -> PixelRegion:
        self._require_open()
        self._check_selection(request)
        d = self._description
        geometry = d.geometry
        level0 = validate_region(geometry, request)
        self._limits.check(request.width, request.height)

        downsample = geometry.levels[request.level].downsample
        assert d.native_offset is not None
        nx, ny, nw, nh = (round(v) for v in d.native_offset.rect_to_native(level0))
        try:
            pixels = self._doc.read(
                roi=(nx, ny, nw, nh),
                plane=dict(_PLANE),
                scene=self._scene_arg,
                zoom=None if request.level == 0 else 1.0 / downsample,
                background_pixel=_BACKGROUND,
            )
        except Exception as exc:
            raise CorruptImageError(f"CZI decode failed for {d.image_id}: {exc}") from exc
        expected = (request.height, request.width, d.samples_per_pixel)
        if getattr(pixels, "shape", None) != expected or str(pixels.dtype) != d.dtype:
            got = (getattr(pixels, "shape", None), getattr(pixels, "dtype", None))
            raise CorruptImageError(f"decoder returned {got}, expected {expected}/{d.dtype}")
        mask = coverage_mask(self._tiles, level0, downsample, request.width, request.height)
        valid = ValidDataReport.from_fraction(
            NoDataSemantics.ACQUISITION_COVERAGE,
            float(mask.mean()),
            mask,
            (FILL_LIMITATION, BACKGROUND_LIMITATION),
        )
        transformations = (DECODE_LEVEL0,) if request.level == 0 else (DECODE_PYRAMID,)
        provenance = RegionReadProvenance(
            image_id=d.image_id,
            reader=d.reader,
            backend=d.backend,
            source_integrity=d.asset.source_checksum,
            scene=d.scene,
            requested=request.level_rect(),
            returned=LevelRect(request.level, request.x, request.y, request.width, request.height),
            level0=level0,
            downsample=downsample,
            native_region=(nx, ny, nw, nh),
            shape=tuple(int(s) for s in pixels.shape),
            dtype=d.dtype,
            channel_order=d.channel_order,
            decoder_transformations=transformations,
            metadata_limitations=d.missing_metadata + valid.limitations,
            validity=valid.validity,
            acquired_fraction=valid.acquired_fraction,
        )
        return PixelRegion(pixels, provenance, valid)
