"""Reader conformance for SPEC-004 preprocessing (AC-017, ADR-004 D12-D15).

Every subject runs the same checks through the same port and pipeline. Outcomes:

* PASS: the test passes.
* FAIL: the test fails.
* UNSUPPORTED: skipped with a reason starting with "UNSUPPORTED:" (outside the subject's
  declared capabilities, e.g. content detection for an in-memory image).
* NOT_TESTED: skipped because no fixture is configured ("real fixture not configured").

Subjects: `gen-czi` (a CZI written with pylibCZIrw at test time: lossless synthetic content,
negative native origin, one missing tile), `f-czi-01`, `f-svs-01` (opt-in real fixtures) and
`array-rgba` (in-memory reference implementation of the port; not a file format).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from application.preprocessing_service import WSI_PREPROCESSING_TASK
from application.runner import ExecutionOutcome, ExecutionRequest
from bootstrap import Platform, build_platform
from domain.errors import (
    PreprocessingCapabilityError,
    RegionLimitExceededError,
    RegionOutOfBoundsError,
)
from domain.hashing import digest
from domain.image import ChannelOrder
from domain.kinds import IMAGE_PYRAMIDAL_2D
from domain.patching import PatchPlan
from domain.provenance import ArtifactIdentity
from domain.reading import NoDataSemantics, PixelRegion, ReadLimits, RegionReadRequest
from domain.tissue import TissueMask
from infrastructure.runtime import FixedClock
from ports import OpenedImage, PatchManifestPayload
from tests.conftest import REPO_ROOT
from tests.imaging_support import (
    CZI_FIXTURE_ENV,
    CZI_FIXTURE_SHA256,
    CZI_FIXTURE_SIZE,
    SVS_FIXTURE_ENV,
    SVS_FIXTURE_SHA256,
    SVS_FIXTURE_SIZE,
    czi_module,
    fixture_path,
    open_fd_count,
    source,
    write_czi,
)
from tests.preprocessing_support import GLASS, PIPELINE, STAIN, ArrayImage, ArrayReader

pytestmark = pytest.mark.filterwarnings("error")

SUBJECTS = ("gen-czi", "f-czi-01", "f-svs-01", "array-rgba")
TILE = 256
ORIGIN = (-300, 200)  # native origin of the generated CZI (negative x on purpose)
# 4 x 3 tiles of 256 px; (col, row) -> "stain" | "glass"; tile (3, 2) is missing (unacquired).
LAYOUT = {
    (c, r): ("stain" if 1 <= c <= 2 and r <= 1 else "glass") for c in range(4) for r in range(3)
}
MISSING = (3, 2)


def generated_content() -> tuple[np.ndarray, np.ndarray]:
    """Level-zero RGB content and acquisition mask shared by gen-czi and the array twin."""
    rgb = np.zeros((3 * TILE, 4 * TILE, 3), dtype=np.uint8)
    acquired = np.zeros(rgb.shape[:2], dtype=bool)
    for (c, r), kind in LAYOUT.items():
        if (c, r) == MISSING:
            continue
        sl = np.s_[r * TILE : (r + 1) * TILE, c * TILE : (c + 1) * TILE]
        rgb[sl] = STAIN if kind == "stain" else GLASS
        # A deterministic texture so pixel equality is meaningful.
        rgb[sl] = np.clip(rgb[sl].astype(int) + (np.arange(TILE) % 7)[:, None, None] - 3, 0, 255)
        acquired[sl] = True
    return rgb, acquired


def write_generated_czi(path: Path) -> Path:
    rgb, _ = generated_content()
    tiles = [
        (
            ORIGIN[0] + c * TILE,
            ORIGIN[1] + r * TILE,
            rgb[r * TILE : (r + 1) * TILE, c * TILE : (c + 1) * TILE, ::-1].copy(),
        )
        for (c, r) in sorted(LAYOUT)
        if (c, r) != MISSING
    ]
    return write_czi(path, tiles, spacing_um=0.5)


@dataclass
class Subject:
    name: str
    path: Path | None
    platform: Platform
    tmp: Path

    @contextmanager
    def open(self) -> Iterator[OpenedImage]:
        if self.path is None:
            rgb, acq = generated_content()
            rgba = np.concatenate([rgb, np.where(acq, 255, 0).astype(np.uint8)[..., None]], -1)
            image = ArrayImage(
                "array-rgba",
                rgba,
                channel_order=ChannelOrder.RGBA,
                no_data=NoDataSemantics.BACKEND_TRANSPARENCY,
                levels=1,
                spacing_um=0.5,
                fmt="test-array-rgba",
            )
            if self.platform.readers._by_format.find("test-array-rgba") is None:
                self.platform.readers.register(ArrayReader("test-array-rgba", ChannelOrder.RGBA))
            with image:
                yield image
        else:
            with self.platform.imaging.open(source(self.path, self.name)) as opened:
                yield opened

    def run(
        self, image: OpenedImage, overrides: tuple[tuple[str, str, object], ...] = ()
    ) -> ExecutionOutcome:
        return self.platform.runner.run(
            ExecutionRequest(
                pipeline=self.platform.pipelines.get(PIPELINE),
                image=image.describe().asset,
                task=WSI_PREPROCESSING_TASK,
                models={},
                preprocessing=None,
                research_use_acknowledged=True,
                source=image,
                config_overrides=overrides,  # type: ignore[arg-type]
            )
        )


@pytest.fixture(params=SUBJECTS)
def subject(request: pytest.FixtureRequest, tmp_path: Path) -> Subject:
    platform = build_platform(REPO_ROOT, clock=FixedClock())
    name = request.param
    if name == "gen-czi":
        czi_module()
        return Subject(name, write_generated_czi(tmp_path / "generated.bin"), platform, tmp_path)
    if name == "f-czi-01":
        pytest.importorskip("pylibCZIrw.czi", reason="optional extra pathology-ai[czi]")
        return Subject(
            name,
            fixture_path(CZI_FIXTURE_ENV, CZI_FIXTURE_SIZE, CZI_FIXTURE_SHA256),
            platform,
            tmp_path,
        )
    if name == "f-svs-01":
        pytest.importorskip("openslide", reason="optional extra pathology-ai[openslide]")
        return Subject(
            name,
            fixture_path(SVS_FIXTURE_ENV, SVS_FIXTURE_SIZE, SVS_FIXTURE_SHA256),
            platform,
            tmp_path,
        )
    return Subject(name, None, platform, tmp_path)


def _outputs(outcome: ExecutionOutcome) -> tuple[TissueMask, PatchPlan, PatchManifestPayload]:
    m, p, x = (outcome.outputs[k].payload for k in ("mask", "plan", "manifest"))
    assert isinstance(m, TissueMask) and isinstance(p, PatchPlan)
    assert isinstance(x, PatchManifestPayload)
    return m, p, x


# --------------------------------------------------------------------------- reader checks


def test_content_based_resolution(subject: Subject) -> None:
    if subject.path is None:
        pytest.skip("UNSUPPORTED: in-memory reference image has no file content to detect")
    renamed = subject.tmp / "renamed.unknown-extension"  # a link in tmp: fixtures stay untouched
    renamed.symlink_to(subject.path)
    backend = subject.platform.imaging.backends.resolve(source(renamed))
    expected = {"gen-czi": "czi-pylibczirw", "f-czi-01": "czi-pylibczirw", "f-svs-01": "openslide"}
    assert backend.name == expected[subject.name]


def test_metadata_consistency(subject: Subject) -> None:
    with subject.open() as image:
        d = image.describe()
        reader = subject.platform.readers.for_image(d.asset)
        assert reader is not None and reader.capabilities.supports_regions
        assert reader.capabilities.channel_order is d.channel_order
        g = d.geometry
        assert g.levels[0].width == g.width and g.levels[0].downsample == 1.0
        assert (g.pixel_spacing is None) == ("physical_spacing" in d.missing_metadata)
        assert d.no_data is not NoDataSemantics.NOT_REPORTED
        assert (d.z_planes, d.time_points) == (1, 1)


def test_canonical_coordinate_mapping(subject: Subject) -> None:
    with subject.open() as image:
        g = image.describe().geometry
        lv = g.levels[-1]
        w, h = min(64, lv.width), min(64, lv.height)
        region = image.read_region(RegionReadRequest(lv.index, lv.width - w, lv.height - h, w, h))
        assert region.level0.x == (lv.width - w) * lv.downsample
        assert region.level0.y == (lv.height - h) * lv.downsample
        g.require_within(region.level0)


def test_bounded_reads_and_limits(subject: Subject) -> None:
    with subject.open() as image:
        g = image.describe().geometry
        with pytest.raises(RegionOutOfBoundsError):
            image.read_region(RegionReadRequest(0, g.width - 10, 0, 20, 20))
        limits = ReadLimits()
        if g.width > limits.max_width:
            with pytest.raises(RegionLimitExceededError):
                image.read_region(RegionReadRequest(0, 0, 0, limits.max_width + 1, 1))


def test_valid_data_is_reported_with_a_mask(subject: Subject) -> None:
    with subject.open() as image:
        region = image.read_region(RegionReadRequest(0, 0, 0, 32, 32))
        mask = region.valid_data.mask
        assert mask is not None and np.asarray(mask).shape == (32, 32)
        assert region.valid_data.semantics is image.describe().no_data


# --------------------------------------------------------------------------- preprocessing checks


def test_pipeline_patch_dimensions_channels_and_provenance(subject: Subject) -> None:
    with subject.open() as image:
        d = image.describe()
        mask, plan, payload = _outputs(subject.run(image))
    manifest = payload.manifest
    assert len(manifest.patches) >= 2
    image_identity = ArtifactIdentity.derive(
        IMAGE_PYRAMIDAL_2D, {"image": d.image_id, "checksum": d.asset.source_checksum}
    )
    for record, (identity, region) in zip(manifest.patches, payload.artifacts, strict=True):
        assert isinstance(region, PixelRegion)
        shape = np.asarray(region.pixels).shape
        assert shape == (record.height_px, record.width_px, d.samples_per_pixel)
        assert record.channel_order is d.channel_order is region.channel_order
        assert record.read_provenance_digest == digest(region.provenance)
        assert identity.lineage == (image_identity.artifact_id,)
        assert record.level0 == d.geometry.to_level0(record.level_rect)
    assert mask.input_channel_order == d.channel_order.value
    assert plan.resolved.resampling == "none"


def test_physical_spacing_semantics(subject: Subject) -> None:
    with subject.open() as image:
        d = image.describe()
        spacing = d.geometry.pixel_spacing
        if spacing is None:
            with pytest.raises(PreprocessingCapabilityError):
                subject.run(image)
            return
        plan = _outputs(subject.run(image))[1]
    ds = plan.resolved.downsample
    assert plan.resolved.effective_spacing_um == (spacing.x_um * ds, spacing.y_um * ds)
    err = plan.resolved.spacing_relative_error
    assert err is not None and err <= plan.spec.spacing_tolerance


def test_deterministic_planning(subject: Subject) -> None:
    with subject.open() as image:
        a = _outputs(subject.run(image))
        b = _outputs(subject.run(image))
    assert a[0].mask_digest == b[0].mask_digest
    assert a[1].plan_digest == b[1].plan_digest
    assert a[2].manifest.manifest_digest == b[2].manifest.manifest_digest


def test_handle_cleanup(subject: Subject) -> None:
    if subject.path is None:
        pytest.skip("UNSUPPORTED: in-memory reference image holds no file handle")
    baseline = open_fd_count()
    subject.platform.preprocessing_runs.run(
        source(subject.path, subject.name), subject.platform.pipelines.get(PIPELINE)
    )
    assert open_fd_count() == baseline


# --------------------------------------------------------------------------- cross-backend


def test_same_content_is_equivalent_across_backends(tmp_path: Path) -> None:
    """Lossless identical content through the CZI adapter (BGR, coverage validity) and the
    in-memory reader (RGBA, transparency validity): geometry, decisions and pixels agree exactly
    after the explicit channel view. Exact pixel equality is justified because both paths are
    lossless (ADR-004 D15)."""
    czi_module()
    platform = build_platform(REPO_ROOT, clock=FixedClock())
    gen = Subject("gen-czi", write_generated_czi(tmp_path / "g.czi"), platform, tmp_path)
    twin = Subject("array-rgba", None, platform, tmp_path)
    with gen.open() as a_img, twin.open() as b_img:
        assert a_img.describe().native_offset is not None
        a_mask, a_plan, a_payload = _outputs(gen.run(a_img))
        b_mask, b_plan, b_payload = _outputs(twin.run(b_img))
    assert np.array_equal(np.asarray(a_mask.tissue), np.asarray(b_mask.tissue))
    assert np.array_equal(np.asarray(a_mask.acquired), np.asarray(b_mask.acquired))
    assert a_mask.mask_digest == b_mask.mask_digest
    assert a_mask.derived == b_mask.derived
    assert [
        (d.candidate, d.status, d.reasons, d.tissue_fraction, d.acquired_fraction)
        for d in a_plan.decisions
    ] == [
        (d.candidate, d.status, d.reasons, d.tissue_fraction, d.acquired_fraction)
        for d in b_plan.decisions
    ]
    assert a_plan.counts() == b_plan.counts() and a_plan.counts()["selected"] == 4
    a_regions = [r for _, r in a_payload.artifacts if isinstance(r, PixelRegion)]
    b_regions = [r for _, r in b_payload.artifacts if isinstance(r, PixelRegion)]
    a_px = [np.asarray(r.pixels)[..., ::-1] for r in a_regions]  # explicit bgr -> rgb
    b_px = [np.asarray(r.pixels)[..., :3] for r in b_regions]  # explicit rgba -> rgb
    assert len(a_px) == len(b_px) == 4
    assert all(np.array_equal(x, y) for x, y in zip(a_px, b_px, strict=True))
    # Recorded differences: channel order, validity semantics, reader identity.
    am, bm = a_payload.manifest, b_payload.manifest
    assert (am.patches[0].channel_order, bm.patches[0].channel_order) == (
        ChannelOrder.BGR,
        ChannelOrder.RGBA,
    )
    assert am.patches[0].validity_semantics is not bm.patches[0].validity_semantics
    assert am.schema_version == bm.schema_version and am.plan_spec == bm.plan_spec
