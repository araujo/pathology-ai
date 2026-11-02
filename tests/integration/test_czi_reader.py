"""CZI adapter against small CZI files written by pylibCZIrw at test time (no patient data).

These exercise the real decoder and every fail-closed path. They are synthetic-format fixtures,
not real whole-slide images; real-slide evidence is in `test_real_wsi_fixtures.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from bootstrap import build_imaging_service
from domain.errors import (
    CorruptImageError,
    ReaderClosedError,
    RegionLimitExceededError,
    RegionOutOfBoundsError,
    UnsupportedModalityError,
    UnsupportedReadOperationError,
)
from domain.geometry import Level0Rect
from domain.ids import ImageId
from domain.image import ChannelOrder
from domain.kinds import IMAGE_REGION_2D
from domain.modality import CT
from domain.reading import DataValidity, NoDataSemantics, ReadLimits, RegionReadRequest
from infrastructure.runtime import FixedClock
from plugins.imaging.czi import CZI_FORMAT, CziBackend, CziImage
from ports import ImageSource, OpenedImage
from tests.imaging_support import czi_module, open_fd_count, solid, source, write_czi

czi_module()

B, G, R = 10, 120, 230  # deliberately asymmetric so any channel swap is visible


@pytest.fixture
def mosaic(tmp_path: Path) -> Path:
    """Two 64x96 tiles with a 32-pixel gap, at a negative native origin.

    Native layout (x): [-500, -404) tile A, [-404, -372) gap, [-372, -276) tile B.
    """
    a = solid(64, 96, (B, G, R))
    b = solid(64, 96, (40, 50, 60))
    return write_czi(tmp_path / "mosaic.czi", [(-500, 300, a), (-372, 300, b)])


def _open(path: Path, **kwargs: Any) -> CziImage:
    return CziBackend().open(source(path), **kwargs)


def test_opened_image_satisfies_the_port(mosaic: Path) -> None:
    with _open(mosaic) as image:
        opened: OpenedImage = image
        assert isinstance(opened, OpenedImage)


def test_metadata_native_origin_and_single_stored_level(mosaic: Path) -> None:
    with _open(mosaic) as image:
        d = image.describe()
    assert d.asset.image_format == CZI_FORMAT
    assert (d.geometry.width, d.geometry.height) == (224, 64)
    assert d.native_offset is not None
    assert (d.native_offset.origin_x, d.native_offset.origin_y) == (-500, 300)
    # pylibCZIrw happily reads at zoom 0.5, but the file stores no pyramid: one level only.
    assert len(d.geometry.levels) == 1
    assert d.channel_order is ChannelOrder.BGR and d.pixel_type == "Bgr24" and d.dtype == "uint8"
    assert d.geometry.pixel_spacing is not None
    assert d.geometry.pixel_spacing.x_um == pytest.approx(0.5)
    assert d.spacing_source is not None and d.missing_metadata == ()
    assert (d.scene, d.scene_count, d.channel_count, d.z_planes, d.time_points) == (0, 1, 1, 1, 1)
    assert d.no_data is NoDataSemantics.ACQUISITION_COVERAGE


def test_bgr_is_preserved_and_mapped_from_level0(mosaic: Path) -> None:
    with _open(mosaic) as image:
        region = image.read_region(RegionReadRequest(0, 0, 0, 96, 64))
    pixels = region.pixels
    assert isinstance(pixels, np.ndarray)
    assert pixels.shape == (64, 96, 3) and pixels.dtype == np.uint8
    assert tuple(pixels[0, 0]) == (B, G, R)  # exactly what was written: no BGR->RGB swap
    assert region.channel_order is ChannelOrder.BGR
    assert region.provenance.native_region == (-500, 300, 96, 64)
    assert region.level0 == Level0Rect(0, 0, 96, 64)
    assert region.valid_data.validity is DataValidity.ALL_ACQUIRED


def test_two_distinct_regions_map_to_native_coordinates(mosaic: Path) -> None:
    raw = czi_module()
    with _open(mosaic) as image:
        first = image.read_region(RegionReadRequest(0, 10, 5, 20, 30))
        second = image.read_region(RegionReadRequest(0, 140, 20, 50, 40))
    with raw.open_czi(str(mosaic)) as doc:
        ref1 = doc.read(roi=(-490, 305, 20, 30), plane={"C": 0, "Z": 0, "T": 0})
        ref2 = doc.read(roi=(-360, 320, 50, 40), plane={"C": 0, "Z": 0, "T": 0})
    assert np.array_equal(np.asarray(first.pixels), ref1) and np.array_equal(
        np.asarray(second.pixels), ref2
    )
    assert tuple(np.asarray(second.pixels)[0, 0]) == (40, 50, 60)
    assert second.provenance.native_region == (-360, 320, 50, 40)


def test_gap_is_reported_as_unacquired_not_as_black_tissue(mosaic: Path) -> None:
    with _open(mosaic) as image:
        gap = image.read_region(RegionReadRequest(0, 96, 0, 32, 64))
        straddle = image.read_region(RegionReadRequest(0, 80, 0, 64, 64))
    assert gap.valid_data.validity is DataValidity.NONE_ACQUIRED
    assert gap.valid_data.acquired_fraction == 0.0
    assert int(np.asarray(gap.pixels).max()) == 0  # the backend's fill, flagged by the mask
    mask = straddle.valid_data.mask
    assert isinstance(mask, np.ndarray)
    assert mask[:, :16].all() and not mask[:, 16:48].any() and mask[:, 48:].all()
    assert straddle.valid_data.validity is DataValidity.PARTIALLY_ACQUIRED
    assert straddle.provenance.acquired_fraction == pytest.approx(0.5)


def test_out_of_bounds_is_rejected_before_decoding(
    mosaic: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _open(mosaic) as image:

        def must_not_decode(**_: object) -> None:
            raise AssertionError("backend read called for an invalid request")

        monkeypatch.setattr(image._doc, "read", must_not_decode)
        for req in (
            RegionReadRequest(0, 200, 0, 25, 10),
            RegionReadRequest(0, 0, 60, 10, 5),
        ):
            with pytest.raises(RegionOutOfBoundsError):
                image.read_region(req)
        with pytest.raises(RegionOutOfBoundsError):
            RegionReadRequest(0, -1, 0, 10, 10)
        with pytest.raises(Exception, match="level 1 does not exist"):
            image.read_region(RegionReadRequest(1, 0, 0, 10, 10))


def test_oversized_request_is_rejected_before_decoding(
    mosaic: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _open(mosaic, limits=ReadLimits(max_width=64, max_height=64, max_pixels=1024)) as image:
        monkeypatch.setattr(image._doc, "read", lambda **_: pytest.fail("decoded"))
        with pytest.raises(RegionLimitExceededError):
            image.read_region(RegionReadRequest(0, 0, 0, 96, 10))
        with pytest.raises(RegionLimitExceededError):
            image.read_region(RegionReadRequest(0, 0, 0, 40, 40))


def test_payload_size_matches_request(mosaic: Path) -> None:
    with _open(mosaic) as image:
        region = image.read_region(RegionReadRequest(0, 3, 4, 50, 20))
    assert isinstance(region.pixels, np.ndarray)
    assert region.pixels.nbytes == 50 * 20 * 3


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"channels": (0,)}, "channel subsets"),
        ({"z": 1}, "z=1"),
        ({"t": 2}, "t=2"),
        ({"scene": 3}, "scene 3"),
    ],
)
def test_unsupported_selections(mosaic: Path, kwargs: dict[str, Any], match: str) -> None:
    with _open(mosaic) as image, pytest.raises(UnsupportedReadOperationError, match=match):
        image.read_region(RegionReadRequest(0, 0, 0, 4, 4, **kwargs))


def test_scenes_must_be_selected_explicitly(tmp_path: Path) -> None:
    path = write_czi(
        tmp_path / "scenes.czi",
        [(0, 0, solid(32, 32, (1, 2, 3)))],
        extra=[(100, -50, solid(16, 48, (7, 8, 9)), 1)],
    )
    with pytest.raises(UnsupportedReadOperationError, match="select one explicitly"):
        _open(path)
    with pytest.raises(UnsupportedReadOperationError, match="scene 5 does not exist"):
        _open(path, scene=5)  # pylibCZIrw itself would silently read "scene 5"
    with _open(path, scene=1) as image:
        d = image.describe()
        assert (d.scene, d.scene_count, d.geometry.width, d.geometry.height) == (1, 2, 48, 16)
        assert d.native_offset is not None and d.native_offset.origin_y == -50
        region = image.read_region(RegionReadRequest(0, 0, 0, 48, 16, scene=1))
        assert tuple(np.asarray(region.pixels)[0, 0]) == (7, 8, 9)
        assert region.provenance.scene == 1
        with pytest.raises(UnsupportedReadOperationError):
            image.read_region(RegionReadRequest(0, 0, 0, 4, 4, scene=0))


def test_missing_spacing_stays_missing(tmp_path: Path) -> None:
    path = write_czi(tmp_path / "nospacing.czi", [(0, 0, solid(8, 8, (1, 1, 1)))], spacing_um=None)
    with _open(path) as image:
        d = image.describe()
    assert d.geometry.pixel_spacing is None and d.spacing_source is None
    assert d.missing_metadata == ("physical_spacing",)


def test_multi_plane_and_unsupported_pixel_types_are_rejected(tmp_path: Path) -> None:
    z_stack = write_czi(tmp_path / "z.czi", [(0, 0, solid(8, 8, (1, 1, 1)))], z_planes=2)
    with pytest.raises(UnsupportedReadOperationError, match="multi-plane"):
        _open(z_stack)
    gray = write_czi(tmp_path / "gray.czi", [(0, 0, np.full((8, 8), 3, np.uint8))])
    with pytest.raises(UnsupportedReadOperationError, match="Gray8"):
        _open(gray)


def test_corrupted_files_raise_typed_errors(tmp_path: Path, mosaic: Path) -> None:
    truncated = tmp_path / "truncated.czi"
    truncated.write_bytes(mosaic.read_bytes()[:200])
    garbage = tmp_path / "garbage.czi"
    garbage.write_bytes(b"ZISRAWFILE" + bytes(range(256)) * 8)
    for path in (truncated, garbage):
        assert CziBackend().probe(path).recognised  # the header claims CZI...
        with pytest.raises(CorruptImageError):  # ...but it cannot be decoded
            _open(path)


def test_declared_modality_must_fit_2d_pyramids(mosaic: Path) -> None:
    with pytest.raises(UnsupportedModalityError):
        CziBackend().open(ImageSource(ImageId("ct-claim"), mosaic, CT))


def test_handles_are_closed_deterministically(mosaic: Path) -> None:
    before = open_fd_count()
    image = _open(mosaic)
    assert open_fd_count() > before
    image.close()
    assert image.closed and open_fd_count() == before
    image.close()  # idempotent
    with pytest.raises(ReaderClosedError):
        image.read_region(RegionReadRequest(0, 0, 0, 4, 4))
    with pytest.raises(ReaderClosedError):
        image.describe()

    service = build_imaging_service()
    with pytest.raises(RegionOutOfBoundsError), service.open(source(mosaic)) as opened:
        opened.read_region(RegionReadRequest(0, 220, 0, 10, 10))
    assert opened.closed and open_fd_count() == before  # closed on error too
    for _ in range(20):
        service.describe(source(mosaic))
    assert open_fd_count() == before


def test_ingestion_smoke_workflow_produces_artifacts_and_provenance(mosaic: Path) -> None:
    service = build_imaging_service(clock=FixedClock())
    requests = [RegionReadRequest(0, 0, 0, 32, 32), RegionReadRequest(0, 150, 10, 40, 30)]
    outcome = service.ingest(source(mosaic, "ingest-me"), requests)
    record = outcome.record
    assert [v.kind for v in outcome.values] == [IMAGE_REGION_2D, IMAGE_REGION_2D]
    assert len({v.identity.artifact_id for v in outcome.values}) == 2
    assert record.artifacts == tuple(v.identity for v in outcome.values)
    assert record.reader.name == "czi-pylibczirw" and record.backend.name == "pylibczirw"
    assert record.integrity is not None and record.integrity.size_bytes == mosaic.stat().st_size
    assert all(p.source_integrity == record.integrity.token for p in record.regions)
    assert "no model inference" in record.usage
    assert [p.requested for p in record.regions] == [r.level_rect() for r in requests]
    assert service.records() == (record,)
    # Same inputs -> same content-addressed artifacts.
    again = build_imaging_service(clock=FixedClock()).ingest(source(mosaic, "ingest-me"), requests)
    assert [v.identity for v in again.values] == [v.identity for v in outcome.values]
