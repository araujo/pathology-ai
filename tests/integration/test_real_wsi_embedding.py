"""Opt-in SPEC-005 tests on the real fixtures F-CZI-01 and F-SVS-01.

Without the fixture variables these tests skip with a reason. Counts were observed on these exact
files with the default `wsi-patch-embedding@1.0.0` configuration. Digests are compared between
runs and processes rather than pinned, because float rounding may differ across platforms. The
reference encoder is synthetic; nothing here validates any scientific property of embeddings.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from application.embedding_service import WSI_EMBEDDING_TASK, EmbeddingOutcome
from application.runner import ExecutionRequest
from bootstrap import Platform, build_platform
from domain.embedding import StoreState
from domain.errors import CorruptImageError
from domain.reading import ImageDescription, PixelRegion, RegionReadRequest
from infrastructure.embedding_store import COORDS, NumpyEmbeddingStoreProvider
from infrastructure.runtime import FixedClock
from plugins.embedding.reference import ReferenceProjectionFactory
from plugins.embedding.transforms import apply_chain
from ports import EMBEDDING_STORE_EXTRA, EncodeContext, OpenedImage
from tests.conftest import REPO_ROOT
from tests.embedding_support import EMBED_PIPELINE
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


def _embed(
    path: Path, image_id: str, out: Path, platform: Platform | None = None
) -> EmbeddingOutcome:
    p = platform or _platform()
    return p.embedding_runs.run(source(path, image_id), p.pipelines.get(EMBED_PIPELINE), out)


@pytest.fixture(scope="module")
def czi_run(
    czi_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> tuple[EmbeddingOutcome, Path]:
    out = tmp_path_factory.mktemp("czi")
    return _embed(czi_path, "f-czi-01", out), out


@pytest.fixture(scope="module")
def svs_run(
    svs_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> tuple[EmbeddingOutcome, Path]:
    out = tmp_path_factory.mktemp("svs")
    return _embed(svs_path, "f-svs-01", out), out


def _assert_rows_match_reader(path: Path, outcome: EmbeddingOutcome, out: Path) -> None:
    """Rows are bit-identical to encoding the reader's patch directly (exact association)."""
    m = outcome.manifest
    store = NumpyEmbeddingStoreProvider(out).open_existing(m.store.store_id)
    rows = store.read_rows(0, m.row_count)
    coords = np.load(store.directory / COORDS)
    encoder = ReferenceProjectionFactory().create({"embedding_dim": m.embedding_dim})
    with _platform().imaging.open(source(path, m.image_id.value)) as image:
        for r in (m.records[0], m.records[m.row_count // 2], m.records[-1]):
            lr = r.level_rect
            region = image.read_region(
                RegionReadRequest(lr.level, int(lr.x), int(lr.y), int(lr.width), int(lr.height))
            )
            assert region.level0 == r.level0
            x = apply_chain(m.transforms, np.asarray(region.pixels))[None]
            ctx = EncodeContext(0, r.row, (r.patch_id,), m.precision, m.device)
            assert np.array_equal(encoder.encode_batch(x, ctx)[0], rows[r.row])
            assert tuple(coords[r.row]) == (r.level0.x, r.level0.y, r.level0.width, r.level0.height)


def _assert_traceable(outcome: EmbeddingOutcome) -> None:
    m, patches = outcome.manifest, outcome.patch_manifest
    assert [r.patch_id for r in m.records] == [p.patch_id for p in patches.patches]
    assert [r.plan_index for r in m.records] == [p.plan_index for p in patches.patches]
    assert [r.level0 for r in m.records] == [p.level0 for p in patches.patches]
    assert m.patch_manifest_digest == patches.manifest_digest
    assert m.plan_digest == outcome.plan.plan_digest == patches.plan_digest


# --------------------------------------------------------------------------- F-CZI-01


def test_czi_produces_a_persistent_embedding_bag(
    czi_path: Path, czi_run: tuple[EmbeddingOutcome, Path]
) -> None:
    outcome, out = czi_run
    m, metrics = outcome.manifest, outcome.payload.metrics
    assert outcome.execution.run.status.value == "SUCCEEDED"
    assert m.reader.name == "czi-pylibczirw"
    assert outcome.patch_manifest.pixel_storage.value == "not_materialized"
    assert (m.row_count, m.embedding_dim, m.embedding_dtype) == (134, 64, "float32")
    assert (m.store.chunk_rows, m.store.chunk_count) == (32, 5)
    assert m.transforms.names == ("channel_reorder", "cast", "scale", "normalize", "layout")
    reorder = m.transforms.steps[0].step
    assert (reorder.param("from"), reorder.param("to")) == ("bgr", "rgb")
    res = m.resolution
    assert res.source_spacing_um == pytest.approx((0.11124, 0.11124), rel=1e-4)
    assert (res.extraction_level, res.extraction_downsample) == (2, 4.0)
    assert res.extraction_spacing_um == pytest.approx((0.44496, 0.44496), rel=1e-4)
    assert res.field_of_view_um == pytest.approx((113.91, 113.91), rel=1e-3)
    assert res.requested_spacing_um == 0.5 and res.resampling == "none"
    assert res.model_input_px == (256, 256) and res.model_required_spacing_um is None
    assert (metrics.region_reads, metrics.max_live_regions, metrics.batches_encoded) == (134, 1, 5)
    first = m.records[0]
    assert (first.level0.x, first.level0.y, first.level0.width) == (38912.0, 0.0, 1024.0)
    _assert_traceable(outcome)
    _assert_rows_match_reader(czi_path, outcome, out)


def test_czi_digests_are_reproducible_and_handles_released(
    czi_path: Path, czi_run: tuple[EmbeddingOutcome, Path], tmp_path: Path
) -> None:
    first = czi_run[0].manifest
    baseline = open_fd_count()
    again = _embed(czi_path, "f-czi-01", tmp_path).manifest
    assert open_fd_count() == baseline
    assert again.content_digest == first.content_digest
    assert (
        again.run_identity == first.run_identity and again.manifest_digest == first.manifest_digest
    )
    reused = _embed(czi_path, "f-czi-01", tmp_path)
    assert reused.payload.metrics.reused_finalized_store
    assert reused.payload.metrics.region_reads == 0


class FailingImage:
    """Delegating `OpenedImage` that fails on the n-th patch-sized read."""

    def __init__(self, inner: OpenedImage, fail_at: int) -> None:
        self.inner, self.fail_at, self.patch_reads = inner, fail_at, 0

    def describe(self) -> ImageDescription:
        return self.inner.describe()

    def read_region(self, request: RegionReadRequest) -> PixelRegion:
        if (request.width, request.height) == (256, 256):
            self.patch_reads += 1
            if self.patch_reads == self.fail_at:
                raise CorruptImageError("injected failure while reading a real patch")
        return self.inner.read_region(request)

    def close(self) -> None:
        self.inner.close()

    @property
    def closed(self) -> bool:
        return self.inner.closed

    def __enter__(self) -> FailingImage:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def test_czi_interrupted_run_resumes_to_identical_content(
    czi_path: Path, czi_run: tuple[EmbeddingOutcome, Path], tmp_path: Path
) -> None:
    platform = _platform()
    provider = NumpyEmbeddingStoreProvider(tmp_path)
    with platform.imaging.open(source(czi_path, "f-czi-01")) as opened:
        failing = FailingImage(opened, fail_at=70)
        with pytest.raises(CorruptImageError):
            platform.runner.run(
                ExecutionRequest(
                    pipeline=platform.pipelines.get(EMBED_PIPELINE),
                    image=opened.describe().asset,
                    task=WSI_EMBEDDING_TASK,
                    models={},
                    preprocessing=None,
                    research_use_acknowledged=True,
                    source=failing,
                    extras={EMBEDDING_STORE_EXTRA: provider},
                )
            )
    store_id = czi_run[0].manifest.store.store_id
    ins = provider.open_existing(store_id).inspect()
    assert ins.state is StoreState.INTERRUPTED and ins.committed_indices == {0, 1}
    resumed = _embed(czi_path, "f-czi-01", tmp_path, platform)
    assert resumed.payload.metrics.patches_resumed == 64
    assert resumed.payload.metrics.region_reads == 134 - 64
    assert resumed.manifest.content_digest == czi_run[0].manifest.content_digest


# --------------------------------------------------------------------------- F-SVS-01


def test_svs_produces_a_persistent_embedding_bag(
    svs_path: Path, svs_run: tuple[EmbeddingOutcome, Path]
) -> None:
    outcome, out = svs_run
    m = outcome.manifest
    assert m.reader.name == "openslide"
    assert (m.row_count, m.embedding_dim, m.store.chunk_count) == (25, 64, 1)
    assert m.transforms.names == ("drop_alpha", "cast", "scale", "normalize", "layout")
    res = m.resolution
    assert (res.extraction_level, res.extraction_spacing_um) == (0, (0.499, 0.499))
    assert res.field_of_view_um == pytest.approx((127.744, 127.744))
    assert any("alpha dropped" in note for note in m.limitations)
    _assert_traceable(outcome)
    _assert_rows_match_reader(svs_path, outcome, out)


def test_one_operation_serves_both_formats(
    czi_run: tuple[EmbeddingOutcome, Path], svs_run: tuple[EmbeddingOutcome, Path]
) -> None:
    czi, svs = czi_run[0].manifest, svs_run[0].manifest
    assert czi.pipeline == svs.pipeline == EMBED_PIPELINE
    assert czi.operation == svs.operation and czi.encoder == svs.encoder
    assert czi.input_spec == svs.input_spec and czi.schema_version == svs.schema_version
    # Format-specific pixel semantics become declared, recorded transforms; never a branch.
    assert czi.transforms.names[0] == "channel_reorder" and svs.transforms.names[0] == "drop_alpha"
    assert czi.resolution.extraction_level != svs.resolution.extraction_level


# --------------------------------------------------------------------------- fresh process (CLI)


def _cli(path: Path, image_id: str, out: Path, manifest: Path) -> dict[str, Any]:
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from cli import embed_main; raise SystemExit(embed_main(sys.argv[1:]))",
            "--file",
            str(path),
            "--image-id",
            image_id,
            "--output-dir",
            str(out),
            "--root",
            str(REPO_ROOT),
            "--manifest-out",
            str(manifest),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    report: dict[str, Any] = json.loads(proc.stdout)
    assert str(path) not in proc.stdout  # never the input path
    return report


def test_cli_in_fresh_processes_reproduces_in_process_digests(
    czi_path: Path,
    svs_path: Path,
    czi_run: tuple[EmbeddingOutcome, Path],
    svs_run: tuple[EmbeddingOutcome, Path],
    tmp_path: Path,
) -> None:
    for path, iid, run in ((czi_path, "f-czi-01", czi_run), (svs_path, "f-svs-01", svs_run)):
        m = run[0].manifest
        report = _cli(path, iid, tmp_path / iid, tmp_path / f"{iid}.json")
        assert report["content_digest"] == m.content_digest
        assert report["manifest_digest"] == m.manifest_digest
        assert report["embedding_identity"] == m.embedding_identity
        written = json.loads((tmp_path / f"{iid}.json").read_text())
        assert written["schema_version"] == "pathology-ai.embedding-manifest/1"
        from domain.hashing import digest

        assert digest(written) == m.manifest_digest
