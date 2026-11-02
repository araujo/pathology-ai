"""Opt-in tests against real whole-slide images (never committed to the repository).

    PATHOLOGY_AI_WSI_FIXTURE_CZI=~/Downloads/digital-pathology-test/SR1482_40X_HE_T198_01.czi
    PATHOLOGY_AI_WSI_FIXTURE_SVS=~/pathology-ai-fixtures/openslide/CMU-1-Small-Region.svs

Without the variables these tests are skipped with an explicit reason. With them, the file's
size and SHA-256 must match the inventory (`.claude/architecture/003-imaging-fixture-inventory.md`).
Expected values were observed on these exact files. Nothing here runs a model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from bootstrap import build_imaging_service
from domain.errors import RegionOutOfBoundsError
from domain.image import ChannelOrder
from domain.reading import DataValidity, NoDataSemantics, RegionReadRequest
from infrastructure.runtime import FixedClock
from plugins.imaging.czi import CziBackend
from plugins.imaging.openslide import OpenSlideBackend
from presentation.api import imaging_v1
from presentation.api.contracts import load_schema
from tests.imaging_support import (
    CZI_FIXTURE_ENV,
    CZI_FIXTURE_SHA256,
    CZI_FIXTURE_SIZE,
    SVS_FIXTURE_ENV,
    SVS_FIXTURE_SHA256,
    SVS_FIXTURE_SIZE,
    fixture_path,
    open_fd_count,
    source,
)

pytestmark = pytest.mark.filterwarnings("error")

PLANE = {"C": 0, "Z": 0, "T": 0}
TISSUE = (36224, 4288)  # level-0 origin of a tissue-rich area of F-CZI-01
OTHER = (20000, 6000)


@pytest.fixture(scope="module")
def czi_path() -> Path:
    pytest.importorskip("pylibCZIrw.czi", reason="optional extra pathology-ai[czi]")
    return fixture_path(CZI_FIXTURE_ENV, CZI_FIXTURE_SIZE, CZI_FIXTURE_SHA256)


@pytest.fixture(scope="module")
def svs_path() -> Path:
    pytest.importorskip("openslide", reason="optional extra pathology-ai[openslide]")
    return fixture_path(SVS_FIXTURE_ENV, SVS_FIXTURE_SIZE, SVS_FIXTURE_SHA256)


def _collect(into: list[Any], item: Any) -> bool:
    into.append(item)
    return True


def _independent_ratios(path: Path) -> tuple[list[float], int]:
    """Stored subsample ratios and level-zero subblock count, straight from pylibCZIrw."""
    from pylibCZIrw import czi

    infos: list[Any] = []
    with czi.open_czi(str(path)) as doc:
        doc.enumerate_subblocks(lambda _i, s: _collect(infos, s))
        layer0: list[Any] = []
        doc.enumerate_subblocks_subset(lambda _i, s: _collect(layer0, s), only_layer0=True)
    ratios = {i.logicalRect.w / i.physicalSize.w for i in infos}
    return sorted(ratios), len(layer0)


# --------------------------------------------------------------------------- CZI (F-CZI-01)


def test_czi_is_selected_by_content_and_described(czi_path: Path) -> None:
    service = build_imaging_service()
    assert service.backends.resolve(source(czi_path)).name == "czi-pylibczirw"
    assert OpenSlideBackend().probe(czi_path).recognised is False  # OpenSlide cannot decode it
    d = service.describe(source(czi_path, "f-czi-01"))
    g = d.geometry
    assert (g.width, g.height) == (40470, 13109)  # = Information/Image SizeX, SizeY
    assert d.native_offset is not None
    assert (d.native_offset.origin_x, d.native_offset.origin_y) == (-262063, 144720)
    assert [lv.downsample for lv in g.levels] == [1, 2, 4, 8, 16, 32, 64]
    assert [(lv.width, lv.height) for lv in g.levels][-1] == (632, 204)
    assert g.pixel_spacing is not None
    assert g.pixel_spacing.x_um == pytest.approx(0.11123987969912646)
    assert g.pixel_spacing.y_um == pytest.approx(0.11123987969912646)
    assert d.channel_order is ChannelOrder.BGR and d.pixel_type == "Bgr24"
    assert (d.scene, d.scene_count, d.channel_count, d.z_planes, d.time_points) == (0, 1, 1, 1, 1)
    assert d.missing_metadata == ()
    assert d.integrity is not None and d.integrity.size_bytes == CZI_FIXTURE_SIZE
    assert ("image.OriginalCompressionMethod", "JpgXr") in d.acquisition
    assert d.backend.version == "6.1.0"


def test_czi_levels_are_stored_resolutions_not_decoder_zoom(czi_path: Path) -> None:
    ratios, layer0 = _independent_ratios(czi_path)
    assert ratios == [1, 2, 4, 8, 16, 32, 64] and layer0 == 216
    d = build_imaging_service().describe(source(czi_path))
    assert [lv.downsample for lv in d.geometry.levels] == ratios
    # The rectangle that includes padded pyramid tiles is larger than the image itself.
    from pylibCZIrw import czi

    with czi.open_czi(str(czi_path)) as doc:
        padded = doc.total_bounding_rectangle
    assert (padded.w, padded.h) == (40512, 13120) != (d.geometry.width, d.geometry.height)


def test_czi_two_regions_match_raw_backend_and_preserve_bgr(czi_path: Path) -> None:
    from pylibCZIrw import czi

    with build_imaging_service().open(source(czi_path)) as image:
        regions = [
            image.read_region(RegionReadRequest(0, x, y, 512, 384)) for x, y in (TISSUE, OTHER)
        ]
    with czi.open_czi(str(czi_path)) as doc:
        for region, (x, y) in zip(regions, (TISSUE, OTHER), strict=True):
            raw = doc.read(roi=(-262063 + x, 144720 + y, 512, 384), plane=dict(PLANE))
            assert np.array_equal(
                np.asarray(region.pixels), raw
            )  # identical bytes: no swap, no normalising
            assert region.provenance.native_region == (-262063 + x, 144720 + y, 512, 384)
            assert region.provenance.shape == (384, 512, 3)
            assert region.channel_order is ChannelOrder.BGR
            assert region.valid_data.validity is DataValidity.ALL_ACQUIRED
    assert not np.array_equal(np.asarray(regions[0].pixels), np.asarray(regions[1].pixels))


def test_czi_level1_maps_to_the_same_level0_area(czi_path: Path) -> None:
    """2x2 means of level-0 pixels agree with the stored level-1 pyramid at the mapped place."""
    x, y = TISSUE
    with build_imaging_service().open(source(czi_path)) as image:
        full = image.read_region(RegionReadRequest(0, x, y, 512, 512))
        half = image.read_region(RegionReadRequest(1, x // 2, y // 2, 256, 256))
    assert half.level0 == full.level0
    assert half.provenance.downsample == 2.0
    pooled = np.asarray(full.pixels, dtype=float).reshape(256, 2, 256, 2, 3).mean(axis=(1, 3))
    aligned = np.abs(pooled - np.asarray(half.pixels)).mean()
    shifted = np.abs(pooled[:, 8:] - np.asarray(half.pixels)[:, :-8]).mean()
    assert aligned < 3.0 < shifted  # observed: 1.58 aligned vs 7.15 shifted by 8 px


def test_czi_unacquired_areas_are_flagged_at_every_level(czi_path: Path) -> None:
    with build_imaging_service().open(source(czi_path)) as image:
        corner = image.read_region(RegionReadRequest(0, 0, 0, 64, 64))
        overview = image.read_region(RegionReadRequest(6, 0, 0, 632, 204))
    # Level 0: the backend fills unscanned area with black ...
    assert corner.valid_data.validity is DataValidity.NONE_ACQUIRED
    assert int(np.asarray(corner.pixels).max()) == 0
    # ... while the stored pyramid has near-white fill there. Only the mask can tell.
    mask = overview.valid_data.mask
    assert isinstance(mask, np.ndarray)
    assert overview.valid_data.validity is DataValidity.PARTIALLY_ACQUIRED
    assert 0.6 < float(mask.mean()) < 0.7  # observed 0.647
    assert np.asarray(overview.pixels)[~mask].mean() > 250
    assert overview.valid_data.semantics is NoDataSemantics.ACQUISITION_COVERAGE


def test_czi_out_of_bounds_and_cleanup(czi_path: Path) -> None:
    before = open_fd_count()
    service = build_imaging_service()
    with service.open(source(czi_path)) as image:
        with pytest.raises(RegionOutOfBoundsError):
            image.read_region(RegionReadRequest(0, 40470 - 10, 0, 20, 20))
        with pytest.raises(RegionOutOfBoundsError):
            image.read_region(RegionReadRequest(6, 600, 200, 64, 8))
    assert open_fd_count() == before


def test_czi_descriptor_is_imaging_api_v1_valid(czi_path: Path) -> None:
    service = build_imaging_service(clock=FixedClock())
    d = service.describe(source(czi_path, "f-czi-01"))
    doc = imaging_v1.image_descriptor(d.asset, CziBackend().capabilities)
    errors = list(
        Draft202012Validator(load_schema("imaging", "v1", "image-descriptor")).iter_errors(doc)
    )
    assert errors == []
    assert doc["channel_order"] == "bgr" and doc["format"] == "zeiss-czi"
    assert czi_path.name not in str(doc) and str(czi_path.parent) not in str(doc)


# --------------------------------------------------------------------------- SVS (F-SVS-01)


def test_svs_metadata(svs_path: Path) -> None:
    service = build_imaging_service()
    assert service.backends.resolve(source(svs_path)).name == "openslide"
    assert CziBackend().probe(svs_path).recognised is False
    d = service.describe(source(svs_path, "f-svs-01"))
    g = d.geometry
    assert (g.width, g.height) == (2220, 2967)
    assert [lv.downsample for lv in g.levels] == [1.0]  # single stored level
    assert g.pixel_spacing is not None and g.pixel_spacing.x_um == pytest.approx(0.499)
    assert d.channel_order is ChannelOrder.RGBA and d.native_offset is None
    assert d.asset.image_format.code == "aperio-svs"
    assert ("openslide.objective-power", "20") in d.acquisition
    assert d.backend.version.endswith("+lib4.0.1")


def test_svs_two_regions_match_raw_backend(svs_path: Path) -> None:
    import openslide

    with build_imaging_service().open(source(svs_path)) as image:
        first = image.read_region(RegionReadRequest(0, 0, 0, 256, 256))
        second = image.read_region(RegionReadRequest(0, 1000, 1500, 300, 200))
        edge = image.read_region(RegionReadRequest(0, 2000, 2900, 220, 67))
        with pytest.raises(RegionOutOfBoundsError):
            image.read_region(RegionReadRequest(0, 2210, 0, 20, 4))
    slide = openslide.OpenSlide(str(svs_path))
    try:
        assert np.array_equal(
            np.asarray(first.pixels), np.asarray(slide.read_region((0, 0), 0, (256, 256)))
        )
        raw = np.asarray(slide.read_region((1000, 1500), 0, (300, 200)))
        assert np.array_equal(np.asarray(second.pixels), raw)
    finally:
        slide.close()
    for region in (first, second, edge):
        assert region.channel_order is ChannelOrder.RGBA
        assert region.valid_data.semantics is NoDataSemantics.BACKEND_TRANSPARENCY
        assert region.valid_data.validity is DataValidity.ALL_ACQUIRED
    assert second.provenance.shape == (200, 300, 4)
    assert not np.array_equal(np.asarray(first.pixels), np.asarray(second.pixels)[:256, :256])


def test_svs_cleanup(svs_path: Path) -> None:
    before = open_fd_count()
    image = OpenSlideBackend().open(source(svs_path))
    image.read_region(RegionReadRequest(0, 0, 0, 16, 16))
    image.close()
    assert image.closed and open_fd_count() == before
