"""Opt-in SPEC-004 tests on the real fixtures F-CZI-01 and F-SVS-01 (see SPEC-003 inventory).

Without the fixture variables these tests skip with a reason. Expected counts were observed on
these exact files with the default `wsi-tissue-patch-extraction@1.0.0` configuration; they pin
behaviour, they are not a scientific validation of the candidate-tissue detector.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from application.preprocessing_service import WSI_PREPROCESSING_TASK, PreprocessingOutcome
from application.runner import ExecutionRequest
from bootstrap import Platform, build_platform
from domain.frames import Placement, padded_origin_translation
from domain.image import ChannelOrder
from domain.reading import ImageDescription, PixelRegion, RegionReadRequest
from infrastructure.runtime import FixedClock
from ports import OpenedImage
from presentation.api import preprocessing_v1
from tests.conftest import REPO_ROOT
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
from tests.preprocessing_support import PIPELINE

pytestmark = pytest.mark.filterwarnings("error")
LEGACY = REPO_ROOT / "tests" / "fixtures" / "legacy" / "f-czi-01-legacy-patch-coords.json"


@pytest.fixture(scope="module")
def czi_path() -> Path:
    pytest.importorskip("pylibCZIrw.czi", reason="optional extra pathology-ai[czi]")
    return fixture_path(CZI_FIXTURE_ENV, CZI_FIXTURE_SIZE, CZI_FIXTURE_SHA256)


@pytest.fixture(scope="module")
def svs_path() -> Path:
    pytest.importorskip("openslide", reason="optional extra pathology-ai[openslide]")
    return fixture_path(SVS_FIXTURE_ENV, SVS_FIXTURE_SIZE, SVS_FIXTURE_SHA256)


def _platform() -> Platform:
    return build_platform(REPO_ROOT, clock=FixedClock())


def _run(path: Path, image_id: str, platform: Platform | None = None) -> PreprocessingOutcome:
    p = platform or _platform()
    return p.preprocessing_runs.run(source(path, image_id), p.pipelines.get(PIPELINE))


class SpyImage:
    """Delegating `OpenedImage` that records every region request."""

    def __init__(self, inner: OpenedImage) -> None:
        self.inner = inner
        self.requests: list[RegionReadRequest] = []

    def describe(self) -> ImageDescription:
        return self.inner.describe()

    def read_region(self, request: RegionReadRequest) -> PixelRegion:
        self.requests.append(request)
        return self.inner.read_region(request)

    def close(self) -> None:
        self.inner.close()

    @property
    def closed(self) -> bool:
        return self.inner.closed

    def __enter__(self) -> SpyImage:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _assert_patches_match_reader(path: Path, outcome: PreprocessingOutcome, n: int = 3) -> None:
    payload = outcome.payload
    picks = [0, len(payload.artifacts) // 2, len(payload.artifacts) - 1][:n]
    with _platform().imaging.open(source(path, outcome.manifest.image_id.value)) as image:
        for i in picks:
            record = payload.manifest.patches[i]
            identity, region = payload.artifacts[i]
            assert isinstance(region, PixelRegion) and identity.artifact_id == record.artifact_id
            lr = record.level_rect
            direct = image.read_region(
                RegionReadRequest(lr.level, int(lr.x), int(lr.y), int(lr.width), int(lr.height))
            )
            assert np.array_equal(np.asarray(region.pixels), np.asarray(direct.pixels))
            assert region.provenance == direct.provenance
            assert region.level0 == record.level0


# --------------------------------------------------------------------------- F-CZI-01


def test_czi_runs_through_the_pipeline_engine(czi_path: Path) -> None:
    outcome = _run(czi_path, "f-czi-01")
    mask, plan, manifest = outcome.mask, outcome.plan, outcome.manifest
    assert outcome.execution.run.status.value == "SUCCEEDED"
    assert manifest.reader.name == "czi-pylibczirw" and manifest.backend.version == "6.1.0"
    # Mask: finest stored level within 1 Mpx -> level 5 (ds 32), one cell per pixel.
    assert (mask.grid.source_level, mask.grid.block_factor) == (5, 1)
    assert (mask.grid.width, mask.grid.height, mask.grid.cell_x) == (1264, 409, 32.0)
    assert mask.grid.edge_adjustment == (22.0, 21.0)
    assert dict(mask.derived)["saturation_threshold"] == 26
    assert mask.tissue_fraction == pytest.approx(0.2604, abs=1e-4)
    assert mask.acquired_fraction == pytest.approx(0.6453, abs=1e-4)
    assert mask.input_channel_order == "bgr" and "bgr->rgb" in mask.channel_view[0]
    # Plan: 0.5 µm/px requested -> stored level 2 (0.44496 µm/px, 11.0% off, within 15%).
    r = plan.resolved
    assert (r.level, r.downsample, r.width_px, r.height_px) == (2, 4.0, 256, 256)
    assert r.spacing_relative_error == pytest.approx(0.11008, abs=1e-5)
    counts = plan.counts()
    assert counts["candidates"] == 39 * 12 == 468
    assert counts["selected"] == 134 and counts["rejected"] == 334
    assert counts["rejected.insufficient_acquired_data"] == 191
    assert counts["rejected.insufficient_tissue"] == 334
    first = manifest.patches[0]
    assert (first.level0.x, first.level0.y, first.level0.width) == (38912.0, 0.0, 1024.0)
    assert first.channel_order is ChannelOrder.BGR and first.samples_per_pixel == 3
    assert all(p.read_acquired_fraction is not None for p in manifest.patches)
    assert len(outcome.payload.artifacts) == 134
    _assert_patches_match_reader(czi_path, outcome)


def test_czi_manifest_is_repeatable_and_serializable(czi_path: Path) -> None:
    a, b = _run(czi_path, "f-czi-01"), _run(czi_path, "f-czi-01")
    assert a.mask.mask_digest == b.mask.mask_digest
    assert a.plan.plan_digest == b.plan.plan_digest
    assert a.manifest == b.manifest
    assert a.manifest.manifest_digest == b.manifest.manifest_digest
    ids_a = [i.artifact_id for i, _ in a.payload.artifacts]
    assert ids_a == [i.artifact_id for i, _ in b.payload.artifacts]
    doc = json.loads(preprocessing_v1.manifest_to_json(a.manifest))
    assert preprocessing_v1.manifest_from_dict(doc) == a.manifest


def test_czi_reads_are_bounded_and_handles_released(czi_path: Path) -> None:
    platform = _platform()
    baseline = open_fd_count()
    with platform.imaging.open(source(czi_path, "f-czi-01")) as opened:
        spy = SpyImage(opened)
        outcome = platform.runner.run(
            ExecutionRequest(
                pipeline=platform.pipelines.get(PIPELINE),
                image=opened.describe().asset,
                task=WSI_PREPROCESSING_TASK,
                models={},
                preprocessing=None,
                research_use_acknowledged=True,
                source=spy,
            )
        )
    assert open_fd_count() == baseline
    assert outcome.run.status.value == "SUCCEEDED"
    levels = {r.level for r in spy.requests}
    assert 0 not in levels  # level zero is never decoded
    assert max(r.width * r.height for r in spy.requests) <= 2048 * 2048
    mask_reads = [r for r in spy.requests if r.level == 5]
    assert sum(r.width * r.height for r in mask_reads) == 1264 * 409
    assert len([r for r in spy.requests if r.level == 2]) == 134


def test_legacy_padded_coordinates_translate_explicitly(czi_path: Path) -> None:
    doc = json.loads(LEGACY.read_text())
    footprint = doc["legacy_patch_footprint_level0_px"]
    from pylibCZIrw import czi

    with _platform().imaging.open(source(czi_path, "f-czi-01")) as image:
        d = image.describe()
        t = padded_origin_translation(d, "legacy-padded")
        assert dict(d.acquisition)["native.padded_origin"] == "-262080,144720"
        assert (t.dx, t.dy) == (-17, 0)  # derived from this slide's origins, not assumed
        placed = [t.place(x, y, footprint, footprint, d.geometry) for x, y in doc["coords"]]
        inside = [(c, rect) for c, (p, rect) in zip(doc["coords"], placed, strict=True) if rect]
        assert len(doc["coords"]) == 66 and len(inside) == 58
        crossing = [
            c
            for c, (p, _) in zip(doc["coords"], placed, strict=True)
            if p is Placement.CROSSES_BOUNDS
        ]
        assert len(crossing) == 8  # legacy x=0 starts 17 px left of the image; y=12078 overruns
        ox, oy = (int(v) for v in dict(d.acquisition)["native.padded_origin"].split(","))
        with czi.open_czi(str(czi_path)) as raw:
            for (cx, cy), rect in inside[:2]:
                legacy = raw.read(
                    roi=(ox + cx, oy + cy, footprint, footprint),
                    plane={"C": 0, "Z": 0, "T": 0},
                    zoom=1.0,
                )
                ours = image.read_region(
                    RegionReadRequest(0, int(rect.x), int(rect.y), footprint, footprint)
                ).pixels
                assert np.array_equal(legacy, np.asarray(ours))
            (cx, cy), _ = inside[0]
            naive = image.read_region(RegionReadRequest(0, cx, cy, footprint, footprint)).pixels
            legacy = raw.read(
                roi=(ox + cx, oy + cy, footprint, footprint),
                plane={"C": 0, "Z": 0, "T": 0},
                zoom=1.0,
            )
            assert not np.array_equal(
                np.asarray(naive), legacy
            )  # untranslated reuse would be wrong


# --------------------------------------------------------------------------- F-SVS-01


def test_svs_runs_through_the_pipeline_engine(svs_path: Path) -> None:
    outcome = _run(svs_path, "f-svs-01")
    mask, plan, manifest = outcome.mask, outcome.plan, outcome.manifest
    assert manifest.reader.name == "openslide"
    # Single stored level: level 0 is read in bounded tiles and block-reduced by 3.
    assert (mask.grid.source_level, mask.grid.block_factor) == (0, 3)
    assert (mask.grid.width, mask.grid.height) == (740, 989)
    assert mask.decoded_pixels == 2220 * 2967
    assert mask.validity_semantics.value == "backend_transparency"
    assert mask.acquired_fraction == 1.0
    assert dict(mask.derived)["saturation_threshold"] == 58
    assert "rgba->rgb" in mask.channel_view[0]
    r = plan.resolved
    assert (r.level, r.width_px) == (0, 256) and r.effective_spacing_um == (0.499, 0.499)
    counts = plan.counts()
    assert (counts["candidates"], counts["selected"]) == (8 * 11, 25)
    assert counts["rejected.insufficient_acquired_data"] == 0
    assert manifest.patches[0].channel_order is ChannelOrder.RGBA
    assert manifest.patches[0].samples_per_pixel == 4
    _assert_patches_match_reader(svs_path, outcome)
    b = _run(svs_path, "f-svs-01")
    assert b.manifest.manifest_digest == manifest.manifest_digest


def test_same_pipeline_definition_serves_both_formats(czi_path: Path, svs_path: Path) -> None:
    platform = _platform()
    czi_out = _run(czi_path, "f-czi-01", platform)
    svs_out = _run(svs_path, "f-svs-01", platform)
    assert czi_out.manifest.pipeline == svs_out.manifest.pipeline == PIPELINE
    assert czi_out.manifest.schema_version == svs_out.manifest.schema_version
    assert czi_out.manifest.plan_spec == svs_out.manifest.plan_spec
    # The stored level differs per format; it is resolved, never assumed.
    assert (czi_out.plan.resolved.level, svs_out.plan.resolved.level) == (2, 0)
