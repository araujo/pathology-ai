"""SPEC-003 reader contracts: port conformance, Imaging API v1 compatibility, error taxonomy."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from domain import errors
from domain.geometry import CoordinateFrame
from domain.hashing import canonical_json
from domain.reading import RegionReadRequest
from plugins.imaging.czi import CziBackend
from plugins.imaging.openslide import OpenSlideBackend
from plugins.synthetic.imaging import SyntheticSlideReader
from ports import ImageBackend, ImageReader, OpenedImage
from presentation.api import imaging_v1
from presentation.api.contracts import load_schema
from tests.imaging_support import solid, source, write_czi

FIXTURES = Path(__file__).parent / "fixtures" / "imaging"


def test_backends_conform_to_ports() -> None:
    for backend in (CziBackend(), OpenSlideBackend()):
        typed: ImageBackend = backend
        reader: ImageReader = backend
        assert isinstance(typed, ImageBackend) and isinstance(reader, ImageReader)
        caps = backend.capabilities
        assert caps.supports_regions and caps.detects_content and caps.reports_no_data_regions
        assert not caps.supports_volumes and not caps.supports_thumbnails
    # Existing readers keep working with the extended capability record (defaults only).
    caps = SyntheticSlideReader().capabilities
    assert (caps.supports_scenes, caps.detects_content) == (False, False)


def test_opened_czi_conforms_and_its_descriptor_is_v1_valid(tmp_path: Path) -> None:
    path = write_czi(tmp_path / "s.czi", [(-64, -32, solid(32, 64, (1, 2, 3)))], spacing_um=None)
    with CziBackend().open(source(path, "contract-czi")) as image:
        opened: OpenedImage = image
        assert isinstance(opened, OpenedImage)
        description = image.describe()
    doc = imaging_v1.image_descriptor(description.asset, CziBackend().capabilities)
    validator = Draft202012Validator(load_schema("imaging", "v1", "image-descriptor"))
    assert list(validator.iter_errors(doc)) == []
    assert doc["geometry"]["pixel_spacing_um"] is None  # missing stays missing (null)
    assert doc["access"] == {"regions": True, "thumbnail": False, "volume": False, "slices": False}
    diagnostics = imaging_v1.reader_diagnostics(description)
    assert diagnostics["native_origin"] == [-64, -32]
    assert diagnostics["missing_metadata"] == ["physical_spacing"]
    json.dumps(diagnostics)  # transport-safe


def test_v1_region_request_maps_onto_domain_request() -> None:
    fx = json.loads((FIXTURES / "valid" / "region-request.json").read_text())
    request = imaging_v1.region_request_from_dict(fx["document"])
    assert request == RegionReadRequest(1, 0, 0, 256, 256)
    assert request.frame is CoordinateFrame.WSI_LEVEL_PIXEL
    assert (request.scene, request.z, request.t, request.channels) == (None, None, None, None)
    bad = dict(fx["document"], frame="volume_voxel_index")
    with pytest.raises(errors.CoordinateFrameMismatchError):
        imaging_v1.region_request_from_dict(bad)


def test_region_provenance_is_canonical(tmp_path: Path) -> None:
    path = write_czi(tmp_path / "p.czi", [(0, 0, solid(16, 16, (5, 6, 7)))])
    with CziBackend().open(source(path)) as image:
        region = image.read_region(RegionReadRequest(0, 0, 0, 8, 8))
    summary = imaging_v1.region_summary(region)
    assert summary["channel_order"] == "bgr" and summary["shape"] == [8, 8, 3]
    assert canonical_json(region.provenance)  # digestable for content addressing


@pytest.mark.parametrize(
    "error,code,base",
    [
        (errors.UnsupportedImageFormatError, "UNSUPPORTED_IMAGE_FORMAT", errors.ImageReadError),
        (errors.AmbiguousReaderError, "AMBIGUOUS_READER", errors.ImageReadError),
        (errors.ReaderBackendUnavailableError, "READER_BACKEND_UNAVAILABLE", errors.ImageReadError),
        (errors.CorruptImageError, "CORRUPT_IMAGE", errors.ImageReadError),
        (errors.UnsupportedReadOperationError, "UNSUPPORTED_READ_OPERATION", errors.ImageReadError),
        (errors.RegionLimitExceededError, "REGION_LIMIT_EXCEEDED", errors.ImageReadError),
        (errors.ReaderClosedError, "READER_CLOSED", errors.ImageReadError),
        (
            errors.RegionOutOfBoundsError,
            "REGION_OUT_OF_BOUNDS",
            errors.InvalidSpatialReferenceError,
        ),
    ],
)
def test_reader_error_taxonomy(error: type[errors.DomainError], code: str, base: type) -> None:
    assert error.code == code and issubclass(error, base) and issubclass(error, errors.DomainError)
