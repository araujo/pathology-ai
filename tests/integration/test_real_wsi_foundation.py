"""Opt-in SPEC-006 tests on the real fixtures F-CZI-01 and F-SVS-01.

* `torch-fixture` (random local weights) runs on both fixtures whenever the fixture variables
  are set and torch is installed.
* Real UNI runs only when `PATHOLOGY_AI_UNI_CHECKPOINT` points to the authorized local
  checkpoint (its digest must equal the adapter's pin). Without it these tests skip, and a skip
  is never evidence that real UNI works.

Nothing here validates any scientific or clinical property of the embeddings: these tests
establish technical integration only.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="optional extra pathology-ai[embedding]")

from application.embedding_service import EmbeddingOutcome  # noqa: E402
from bootstrap import Platform, build_platform  # noqa: E402
from domain.embedding import StoreState  # noqa: E402
from domain.errors import ModelInferenceError, ProfileUnavailableError  # noqa: E402
from domain.ids import PipelineId, Version  # noqa: E402
from domain.pipeline import ConfigValue, PipelineRef  # noqa: E402
from domain.reading import RegionReadRequest  # noqa: E402
from infrastructure.embedding_store import NumpyEmbeddingStoreProvider  # noqa: E402
from infrastructure.runtime import FixedClock  # noqa: E402
from plugins.embedding.transforms import apply_chain  # noqa: E402
from plugins.models.fixture import write_fixture_checkpoint  # noqa: E402
from ports import EncodeContext  # noqa: E402
from tests.conftest import REPO_ROOT  # noqa: E402
from tests.embedding_support import EMBED_PIPELINE, with_embed_config  # noqa: E402
from tests.imaging_support import (  # noqa: E402
    CZI_FIXTURE_ENV,
    CZI_FIXTURE_SHA256,
    CZI_FIXTURE_SIZE,
    SVS_FIXTURE_ENV,
    SVS_FIXTURE_SHA256,
    SVS_FIXTURE_SIZE,
    fixture_path,
    source,
)

pytestmark = pytest.mark.filterwarnings("error")
UNI_CHECKPOINT_ENV = "PATHOLOGY_AI_UNI_CHECKPOINT"
UNI_PIPELINE = PipelineRef(PipelineId("wsi-patch-embedding-uni"), Version("1.0.0"))


@pytest.fixture(scope="module")
def czi_path() -> Path:
    pytest.importorskip("pylibCZIrw.czi", reason="optional extra pathology-ai[czi]")
    return fixture_path(CZI_FIXTURE_ENV, CZI_FIXTURE_SIZE, CZI_FIXTURE_SHA256)


@pytest.fixture(scope="module")
def svs_path() -> Path:
    pytest.importorskip("openslide", reason="optional extra pathology-ai[openslide]")
    return fixture_path(SVS_FIXTURE_ENV, SVS_FIXTURE_SIZE, SVS_FIXTURE_SHA256)


@pytest.fixture(scope="module")
def uni_checkpoint() -> Path:
    pytest.importorskip("timm", reason="optional extra pathology-ai[embedding]")
    value = os.environ.get(UNI_CHECKPOINT_ENV)
    if not value:
        pytest.skip(f"real UNI weights not configured: set {UNI_CHECKPOINT_ENV} (UNVERIFIED)")
    path = Path(value).expanduser()
    assert path.is_file(), f"{UNI_CHECKPOINT_ENV}={value} is not a file"
    return path


@pytest.fixture(scope="module")
def fixture_ckpt(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    path = tmp_path_factory.mktemp("fx") / "fixture.pt"
    return path, write_fixture_checkpoint(path)


def _store_rows(out: Path, m: Any) -> Any:
    store = NumpyEmbeddingStoreProvider(out).open_existing(m.store.store_id)
    rows = np.asarray(store.read_rows(0, m.row_count))
    assert store.inspect().state is StoreState.FINALIZED
    store.close()
    return rows


def _assert_traceable(outcome: EmbeddingOutcome) -> None:
    m, patches = outcome.manifest, outcome.patch_manifest
    assert [r.patch_id for r in m.records] == [p.patch_id for p in patches.patches]
    assert [r.level0 for r in m.records] == [p.level0 for p in patches.patches]
    assert m.patch_manifest_digest == patches.manifest_digest


def _assert_first_batch_matches_reader(
    platform: Platform, path: Path, outcome: EmbeddingOutcome, out: Path, cfg: dict[str, Any]
) -> None:
    """The first stored chunk equals encoding the reader's patches directly (same batch)."""
    m = outcome.manifest
    rows = _store_rows(out, m)
    n = m.store.chunk_rows if m.row_count >= m.store.chunk_rows else m.row_count
    records = m.records[:n]
    with platform.imaging.open(source(path, m.image_id.value)) as image:
        batch = np.stack(
            [
                apply_chain(
                    m.transforms,
                    np.asarray(
                        image.read_region(
                            RegionReadRequest(
                                r.level_rect.level,
                                int(r.level_rect.x),
                                int(r.level_rect.y),
                                int(r.level_rect.width),
                                int(r.level_rect.height),
                            )
                        ).pixels
                    ),
                )
                for r in records
            ]
        )
    encoder = platform.encoders.resolve(m.encoder.component.name).create(cfg)
    try:
        ctx = EncodeContext(0, 0, tuple(r.patch_id for r in records), m.precision, m.device)
        assert np.array_equal(np.asarray(encoder.encode_batch(batch, ctx)), rows[:n])
    finally:
        encoder.close()


# --------------------------------------------------------------------------- torch fixture


def _fixture_run(path: Path, image_id: str, ckpt: tuple[Path, str], out: Path) -> Any:
    platform = build_platform(REPO_ROOT, clock=FixedClock(), checkpoints={"torch-fixture": ckpt[0]})
    cfg: dict[str, ConfigValue] = {
        "encoder": "torch-fixture",
        "checkpoint_sha256": ckpt[1],
        "input_px": 256,
        "embedding_dim": 32,
    }
    pipe = with_embed_config(platform.pipelines.get(EMBED_PIPELINE), **cfg)
    outcome = platform.embedding_runs.run(source(path, image_id), pipe, out)
    embed = next(s for s in pipe.stages if str(s.stage_id) == "embed").config_dict()
    return platform, outcome, embed


def test_czi_torch_fixture_embedding_bag(
    czi_path: Path, fixture_ckpt: tuple[Path, str], tmp_path: Path
) -> None:
    platform, outcome, cfg = _fixture_run(czi_path, "f-czi-01", fixture_ckpt, tmp_path)
    m = outcome.manifest
    assert (m.row_count, m.embedding_dim) == (134, 32)
    assert m.transforms.names[0] == "channel_reorder" and m.resolution.resampling == "none"
    assert m.resolution.extraction_level == 2
    _assert_traceable(outcome)
    _assert_first_batch_matches_reader(platform, czi_path, outcome, tmp_path, cfg)
    again = _fixture_run(czi_path, "f-czi-01", fixture_ckpt, tmp_path / "again")[1].manifest
    assert again.content_digest == m.content_digest


def test_svs_torch_fixture_embedding_bag(
    svs_path: Path, fixture_ckpt: tuple[Path, str], tmp_path: Path
) -> None:
    platform, outcome, cfg = _fixture_run(svs_path, "f-svs-01", fixture_ckpt, tmp_path)
    m = outcome.manifest
    assert (m.row_count, m.embedding_dim) == (25, 32)
    assert m.transforms.names[0] == "drop_alpha" and m.resolution.extraction_level == 0
    _assert_traceable(outcome)
    _assert_first_batch_matches_reader(platform, svs_path, outcome, tmp_path, cfg)


# --------------------------------------------------------------------------- real UNI


def _uni_platform(checkpoint: Path) -> Platform:
    return build_platform(REPO_ROOT, clock=FixedClock(), checkpoints={"uni": checkpoint})


def _uni_run(platform: Platform, path: Path, image_id: str, out: Path) -> EmbeddingOutcome:
    return platform.embedding_runs.run(
        source(path, image_id), platform.pipelines.get(UNI_PIPELINE), out
    )


def _uni_cfg(platform: Platform) -> dict[str, Any]:
    pipe = platform.pipelines.get(UNI_PIPELINE)
    return next(s for s in pipe.stages if str(s.stage_id) == "embed").config_dict()


def _assert_uni_manifest(m: Any) -> None:
    assert m.encoder.component.name == "uni" and not m.encoder.synthetic
    assert m.encoder.weights_checksum == (
        "sha256:56ef09b44a25dc5c7eedc55551b3d47bcd17659a7a33837cf9abc9ec4e2ffb40"
    )
    params = dict(m.encoder.parameters)
    assert params["profile"] == "uni-official@1" and params["output"] == "cls_token"
    assert (params["model_spacing_um"], params["model_spacing_tolerance"]) == (0.5, 0.15)
    assert (m.embedding_dim, m.embedding_dtype) == (1024, "float32")
    assert m.resolution.model_input_px == (224, 224) and m.resolution.resampling == "none"
    assert m.input_spec.normalization.mean == (0.485, 0.456, 0.406)
    assert any("CC-BY-NC-ND" in t for t in m.limitations)
    assert any("20x" in t for t in m.limitations)


def test_real_uni_on_czi(czi_path: Path, uni_checkpoint: Path, tmp_path: Path) -> None:
    platform = _uni_platform(uni_checkpoint)
    outcome = _uni_run(platform, czi_path, "f-czi-01", tmp_path)
    m = outcome.manifest
    _assert_uni_manifest(m)
    assert m.resolution.extraction_level == 2 and m.transforms.names[0] == "channel_reorder"
    assert m.row_count == len(outcome.patch_manifest.patches) > 0
    rows = _store_rows(tmp_path, m)
    assert np.isfinite(rows).all() and float(np.abs(rows).sum()) > 0
    _assert_traceable(outcome)
    _assert_first_batch_matches_reader(platform, czi_path, outcome, tmp_path, _uni_cfg(platform))


def test_real_uni_on_svs_is_reproducible_and_resumable(
    svs_path: Path, uni_checkpoint: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import plugins.models.uni as uni

    platform = _uni_platform(uni_checkpoint)
    clean = _uni_run(platform, svs_path, "f-svs-01", tmp_path / "clean")
    m = clean.manifest
    _assert_uni_manifest(m)
    assert m.resolution.extraction_level == 0 and m.transforms.names[0] == "drop_alpha"
    _assert_traceable(clean)
    _assert_first_batch_matches_reader(
        platform, svs_path, clean, tmp_path / "clean", _uni_cfg(platform)
    )
    assert _uni_run(platform, svs_path, "f-svs-01", tmp_path / "b").manifest.manifest_digest == (
        m.manifest_digest
    )
    # Interrupt the second batch on a real device call path, then resume from chunk 0.
    original: Callable[[Any, Any], Any] = uni.class_token
    calls = {"n": 0}

    def failing(module: Any, x: Any) -> Any:
        calls["n"] += 1  # 1 = load probe, 2 = batch 0, 3 = batch 1
        if calls["n"] == 3:
            raise RuntimeError("injected device failure")
        return original(module, x)

    with monkeypatch.context() as patch:
        patch.setattr(uni, "class_token", failing)
        with pytest.raises(ModelInferenceError):
            _uni_run(platform, svs_path, "f-svs-01", tmp_path / "resume")
    resumed = _uni_run(platform, svs_path, "f-svs-01", tmp_path / "resume")
    assert resumed.payload.metrics.patches_resumed == m.store.chunk_rows
    assert resumed.manifest.content_digest == m.content_digest


def test_legacy_profile_is_refused_on_czi(
    czi_path: Path, uni_checkpoint: Path, tmp_path: Path
) -> None:
    platform = _uni_platform(uni_checkpoint)
    pipe = with_embed_config(platform.pipelines.get(UNI_PIPELINE), profile="uni-legacy-surgen@1")
    with pytest.raises(ProfileUnavailableError, match="INTER_AREA"):
        platform.embedding_runs.run(source(czi_path, "f-czi-01"), pipe, tmp_path)
    assert not any(tmp_path.iterdir())
