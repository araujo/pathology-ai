"""Domain reading contracts: requests, bounds, limits, native frames, valid-data reports."""

from __future__ import annotations

import pytest

from domain.errors import (
    CoordinateFrameMismatchError,
    InvalidImageAssetError,
    InvalidSpatialReferenceError,
    RegionLimitExceededError,
    RegionOutOfBoundsError,
)
from domain.geometry import CoordinateFrame, Level0Rect, PyramidalGeometry, PyramidLevel
from domain.ids import ImageId
from domain.image import ChannelOrder, ImageAsset
from domain.modality import WHOLE_SLIDE_MICROSCOPY, ImageFormat
from domain.provenance import ComponentVersion
from domain.reading import (
    DataValidity,
    ImageDescription,
    NativeFrameOffset,
    NoDataSemantics,
    ReadLimits,
    RegionReadRequest,
    ValidDataReport,
    validate_region,
)

# Non-integer downsamples (e.g. some scanners store 1, 4.0002, 16.0011, ...).
GEOMETRY = PyramidalGeometry(
    1000,
    600,
    (
        PyramidLevel(0, 1000, 600, 1.0),
        PyramidLevel(1, 666, 400, 1.5),
        PyramidLevel(2, 249, 149, 4.0002),
    ),
)


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"frame": CoordinateFrame.WSI_LEVEL0_PIXEL}, CoordinateFrameMismatchError),
        ({"frame": CoordinateFrame.VOLUME_VOXEL_INDEX}, CoordinateFrameMismatchError),
        ({"x": -1}, RegionOutOfBoundsError),
        ({"y": -5}, RegionOutOfBoundsError),
        ({"width": 0}, InvalidSpatialReferenceError),
        ({"x": 1.5}, InvalidSpatialReferenceError),
        ({"level": True}, InvalidSpatialReferenceError),
        ({"level": -1}, InvalidSpatialReferenceError),
        ({"scene": -1}, InvalidSpatialReferenceError),
        ({"channels": ()}, InvalidSpatialReferenceError),
        ({"channels": (0, -1)}, InvalidSpatialReferenceError),
    ],
)
def test_invalid_region_requests_are_rejected(kwargs: dict[str, object], error: type) -> None:
    base: dict[str, object] = {"level": 0, "x": 0, "y": 0, "width": 8, "height": 8}
    with pytest.raises(error):
        RegionReadRequest(**{**base, **kwargs})  # type: ignore[arg-type]


def test_region_maps_to_level0_with_non_integer_downsample() -> None:
    rect = validate_region(GEOMETRY, RegionReadRequest(1, 10, 20, 100, 50))
    assert rect == Level0Rect(15.0, 30.0, 150.0, 75.0)
    rect = validate_region(GEOMETRY, RegionReadRequest(2, 1, 2, 10, 10))
    assert rect.x == pytest.approx(4.0002) and rect.width == pytest.approx(40.002)


@pytest.mark.parametrize(
    "request_",
    [
        RegionReadRequest(0, 999, 0, 2, 1),  # crosses right edge
        RegionReadRequest(0, 0, 599, 1, 2),  # crosses bottom edge
        RegionReadRequest(1, 600, 0, 67, 1),  # level-1 width is 666
        RegionReadRequest(2, 0, 0, 250, 1),
    ],
)
def test_out_of_bounds_regions_are_rejected_not_padded(request_: RegionReadRequest) -> None:
    with pytest.raises(RegionOutOfBoundsError):
        validate_region(GEOMETRY, request_)


def test_missing_level_is_rejected() -> None:
    with pytest.raises(InvalidSpatialReferenceError, match="level 3 does not exist"):
        validate_region(GEOMETRY, RegionReadRequest(3, 0, 0, 1, 1))


def test_full_level_is_within_bounds() -> None:
    for level in GEOMETRY.levels:
        validate_region(GEOMETRY, RegionReadRequest(level.index, 0, 0, level.width, level.height))


def test_read_limits() -> None:
    limits = ReadLimits(max_width=100, max_height=50, max_pixels=1000)
    limits.check(100, 10)
    for w, h in ((101, 1), (1, 51), (40, 40)):
        with pytest.raises(RegionLimitExceededError):
            limits.check(w, h)
    with pytest.raises(InvalidSpatialReferenceError):
        ReadLimits(max_pixels=0)


def test_native_offset_round_trip_with_negative_origin() -> None:
    offset = NativeFrameOffset(-262063, 144720)
    assert offset.to_native(0, 0) == (-262063, 144720)
    assert offset.to_level0(-262063, 144720) == (0, 0)
    assert offset.to_level0(*offset.to_native(123.5, 7)) == (123.5, 7)
    assert offset.rect_to_native(Level0Rect(10, 20, 30, 40)) == (-262053, 144740, 30, 40)
    with pytest.raises(InvalidSpatialReferenceError):
        NativeFrameOffset(1.5, 0)  # type: ignore[arg-type]


def test_valid_data_report_consistency() -> None:
    report = ValidDataReport.from_fraction(NoDataSemantics.ACQUISITION_COVERAGE, 0.25, None)
    assert report.validity is DataValidity.PARTIALLY_ACQUIRED
    assert (
        ValidDataReport.from_fraction(NoDataSemantics.ACQUISITION_COVERAGE, 1.0, None).validity
        is DataValidity.ALL_ACQUIRED
    )
    with pytest.raises(InvalidImageAssetError):
        ValidDataReport(NoDataSemantics.ACQUISITION_COVERAGE, DataValidity.ALL_ACQUIRED, 0.5, None)
    with pytest.raises(InvalidImageAssetError):  # unknown must stay unknown, and say why
        ValidDataReport(NoDataSemantics.NOT_REPORTED, DataValidity.ALL_ACQUIRED, 1.0)
    with pytest.raises(InvalidImageAssetError):
        ValidDataReport(NoDataSemantics.NOT_REPORTED, DataValidity.UNKNOWN, None)
    ValidDataReport(NoDataSemantics.NOT_REPORTED, DataValidity.UNKNOWN, None, None, ("no info",))


def _description(**overrides: object) -> ImageDescription:
    asset = ImageAsset(
        ImageId("img"),
        WHOLE_SLIDE_MICROSCOPY,
        ImageFormat("fake"),
        PyramidalGeometry(10, 10, (PyramidLevel(0, 10, 10, 1.0),)),
        ChannelOrder.BGR,
    )
    fields: dict[str, object] = {
        "asset": asset,
        "reader": ComponentVersion("reader", "fake", "1"),
        "backend": ComponentVersion("backend", "fake", "1"),
        "scene": 0,
        "scene_count": 1,
        "native_offset": None,
        "pixel_type": "x",
        "dtype": "uint8",
        "samples_per_pixel": 3,
        "channel_count": 1,
        "z_planes": 1,
        "time_points": 1,
        "spacing_source": None,
        "level_source": "test",
        "no_data": NoDataSemantics.NOT_REPORTED,
        "decoder_transformations": (),
        "missing_metadata": ("physical_spacing",),
    }
    fields.update(overrides)
    return ImageDescription(**fields)  # type: ignore[arg-type]


def test_description_requires_missing_spacing_to_be_declared() -> None:
    assert _description().geometry.pixel_spacing is None
    with pytest.raises(InvalidImageAssetError, match="listed as missing"):
        _description(missing_metadata=())
    with pytest.raises(InvalidImageAssetError, match="together"):
        _description(spacing_source="file:somewhere")


def test_description_requires_channel_order() -> None:
    asset = ImageAsset(
        ImageId("img"),
        WHOLE_SLIDE_MICROSCOPY,
        ImageFormat("fake"),
        PyramidalGeometry(10, 10, (PyramidLevel(0, 10, 10, 1.0),)),
    )
    with pytest.raises(InvalidImageAssetError, match="channel order"):
        _description(asset=asset)
