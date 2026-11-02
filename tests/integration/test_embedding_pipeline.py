"""SPEC-005 end-to-end on in-memory slides: `wsi-patch-embedding` through the `PipelineRunner`.

Covers lazy consumption, bounded pixel retention, exact patch-to-row association, identity and
reproducibility, and failure injection at read, encode, write, commit and finalize boundaries
with resume from verified chunks. Slides are synthetic; embeddings carry no medical meaning.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tracemalloc
from pathlib import Path

import numpy as np
import pytest

from application.preprocessing_service import WSI_PREPROCESSING_TASK
from application.runner import ExecutionRequest
from bootstrap import Platform, build_platform
from domain.embedding import EmbeddingStoreHeader, RecoveryPolicy, StoreState
from domain.errors import (
    CorruptImageError,
    EmbeddingBudgetError,
    EmbeddingOutputError,
    EmbeddingStoreCorruptError,
    EmbeddingStoreIncompatibleError,
    ExecutionUnsupportedError,
    PreprocessingCapabilityError,
    ResolutionMismatchError,
)
from domain.image import ChannelOrder
from domain.manifest import PatchManifest
from domain.provenance import RunStatus
from domain.reading import PixelRegion, RegionReadRequest
from infrastructure.embedding_store import (
    CHUNKS,
    COORDS,
    DATA,
    NumpyEmbeddingStore,
    NumpyEmbeddingStoreProvider,
)
from infrastructure.runtime import FixedClock
from plugins.embedding.reference import ReferenceProjectionFactory
from plugins.embedding.transforms import apply_chain
from ports import EMBEDDING_STORE_EXTRA, EncodeContext, PatchManifestPayload
from tests.conftest import REPO_ROOT
from tests.embedding_support import (
    FaultyFactory,
    FaultyProvider,
    TrackingImage,
    bag,
    embedding_operation,
    ensure_reader,
    run_embedding,
    stride,
    tissue_slide,
)
from tests.preprocessing_support import PIPELINE as PREPROCESS_PIPELINE

pytestmark = pytest.mark.filterwarnings("error")
SMALL = stride(256)  # 16 patches
LARGE = stride(64)  # 169 patches
BATCH8 = (("embed", "batch_size", 8),)


def _platform() -> Platform:
    return build_platform(REPO_ROOT, clock=FixedClock())


def _patch_manifest(outcome: object) -> PatchManifest:
    payload = outcome.outputs["manifest"].payload  # type: ignore[attr-defined]
    assert isinstance(payload, PatchManifestPayload)
    return payload.manifest


@pytest.fixture(scope="module")
def clean(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    """Reference digests of an uninterrupted run (169 patches, batches of 8)."""
    out = tmp_path_factory.mktemp("clean")
    m = bag(run_embedding(_platform(), tissue_slide(), out, LARGE + BATCH8)).manifest
    return {"content": m.content_digest, "run": m.run_identity, "store": m.store.store_id}


# --------------------------------------------------------------------------- end to end


def test_patch_plan_to_durable_embedding_bag(tmp_path: Path) -> None:
    platform, image = _platform(), tissue_slide()
    outcome = run_embedding(platform, image, tmp_path, SMALL)
    assert outcome.run.status is RunStatus.SUCCEEDED
    patches = _patch_manifest(outcome)
    payload = bag(outcome)
    m, metrics = payload.manifest, payload.metrics
    assert patches.pixel_storage.value == "not_materialized"  # AC-001: no pixel artifact upstream
    assert m.row_count == len(patches.patches) == 16
    assert [r.patch_id for r in m.records] == [p.patch_id for p in patches.patches]
    assert [r.plan_index for r in m.records] == [p.plan_index for p in patches.patches]
    assert [r.level0 for r in m.records] == [p.level0 for p in patches.patches]
    assert m.plan_digest == patches.plan_digest
    assert m.patch_manifest_digest == patches.manifest_digest
    assert (m.embedding_dim, m.embedding_dtype, m.status) == (64, "float32", StoreState.FINALIZED)
    assert m.transforms.names == ("cast", "scale", "normalize", "layout")  # RGB: no reorder
    assert m.resolution.resampling == "none" and m.resolution.patch_px == (256, 256)
    assert m.resolution.field_of_view_um == (64.0, 64.0)  # 256 px x 0.25 µm/px at level 0
    assert m.encoder.synthetic and "no biological or clinical meaning" in m.limitations[0]
    assert (metrics.patches_encoded, metrics.region_reads, metrics.batches_encoded) == (16, 16, 1)
    assert payload.store.location == str(tmp_path / m.store.store_id)
    assert str(tmp_path) not in json.dumps(m.manifest_digest)  # identities never hold paths
    assert image.closed is False  # the caller owns the handle


def test_no_pixels_or_embedding_rows_pass_through_the_artifact_store(tmp_path: Path) -> None:
    platform = _platform()
    outcome = run_embedding(platform, tissue_slide(), tmp_path, SMALL)
    for identity in outcome.run.artifacts:
        payload = platform.artifacts.get(identity.artifact_id)
        assert not isinstance(payload, np.ndarray | PixelRegion)
        if isinstance(payload, PatchManifestPayload):
            assert payload.artifacts == ()
    assert set(vars(type(bag(outcome))).get("__slots__", ())) <= {"manifest", "store", "metrics"}


def test_rows_map_exactly_to_their_patches(tmp_path: Path) -> None:
    platform, image = _platform(), tissue_slide()
    payload = bag(run_embedding(platform, image, tmp_path, SMALL))
    m = payload.manifest
    store = NumpyEmbeddingStoreProvider(tmp_path).open_existing(m.store.store_id)
    rows = store.read_rows(0, m.row_count)
    coords = np.load(store.directory / COORDS)
    encoder = ReferenceProjectionFactory().create({"embedding_dim": 64})
    for r in (m.records[0], m.records[7], m.records[-1]):
        lr = r.level_rect
        region = image.read_region(
            RegionReadRequest(lr.level, int(lr.x), int(lr.y), int(lr.width), int(lr.height))
        )
        assert region.level0 == r.level0
        x = apply_chain(m.transforms, np.asarray(region.pixels))[None]
        ctx = EncodeContext(0, r.row, (r.patch_id,), m.precision, m.device)
        assert np.array_equal(encoder.encode_batch(x, ctx)[0], rows[r.row])  # bit-exact
        assert tuple(coords[r.row]) == (r.level0.x, r.level0.y, r.level0.width, r.level0.height)
    assert len({rows[i].tobytes() for i in range(m.row_count)}) == m.row_count  # rows differ


# --------------------------------------------------------------------------- identity


def test_digests_are_stable_across_runs_and_batch_sizes(
    tmp_path: Path, clean: dict[str, str]
) -> None:
    for i, batch in enumerate((1, 32)):
        m = bag(
            run_embedding(
                _platform(),
                tissue_slide(),
                tmp_path / str(i),
                LARGE + (("embed", "batch_size", batch),),
            )
        ).manifest
        assert (m.content_digest, m.run_identity) == (clean["content"], clean["run"])
        assert m.store.store_id == clean["store"]  # location-independent identity


def test_identity_changes_with_encoder_configuration(tmp_path: Path) -> None:
    a = bag(run_embedding(_platform(), tissue_slide(), tmp_path, SMALL)).manifest
    b = bag(
        run_embedding(
            _platform(), tissue_slide(), tmp_path, SMALL + (("embed", "embedding_dim", 32),)
        )
    ).manifest
    assert a.store.store_id != b.store.store_id and a.content_digest != b.content_digest
    assert b.embedding_dim == 32


def test_repeat_on_a_finalized_store_reuses_it_without_reading(tmp_path: Path) -> None:
    first = bag(run_embedding(_platform(), tissue_slide(), tmp_path, SMALL))
    image = tissue_slide()
    again = bag(run_embedding(_platform(), image, tmp_path, SMALL))
    assert again.metrics.reused_finalized_store and image.patch_reads == 0
    assert again.manifest == first.manifest


def test_a_different_chunk_layout_at_the_same_location_is_refused(tmp_path: Path) -> None:
    run_embedding(_platform(), tissue_slide(), tmp_path, SMALL + BATCH8)
    with pytest.raises(EmbeddingStoreIncompatibleError):
        run_embedding(_platform(), tissue_slide(), tmp_path, SMALL + (("embed", "batch_size", 4),))


# --------------------------------------------------------------------------- bounded memory


def test_pixel_retention_does_not_grow_with_patch_count(tmp_path: Path) -> None:
    observed = {}
    for name, overrides in (("small", SMALL), ("large", LARGE)):
        image = tissue_slide()
        payload = bag(run_embedding(_platform(), image, tmp_path / name, overrides + BATCH8))
        observed[name] = (image.patch_reads, image.max_live, payload.metrics)
    (n_small, live_small, m_small), (n_large, live_large, m_large) = observed.values()
    assert (n_small, n_large) == (16, 169)  # every patch read exactly once, lazily
    assert live_small == live_large == 1  # never more than one decoded patch alive
    assert m_small.max_live_regions == m_large.max_live_regions == 1
    assert m_small.budget == m_large.budget  # the working set is independent of N
    assert m_large.budget.working_set_bytes <= m_large.budget.memory_budget_bytes


class PeakProvider(NumpyEmbeddingStoreProvider):
    """Starts a fresh traced-memory peak when the embedding stage opens its store."""

    baseline = 0

    def open(
        self, header: EmbeddingStoreHeader, *, recovery: RecoveryPolicy
    ) -> NumpyEmbeddingStore:
        tracemalloc.reset_peak()  # measure the embedding loop, not detection or planning
        self.baseline = tracemalloc.get_traced_memory()[0]
        return super().open(header, recovery=recovery)


def test_traced_peak_memory_does_not_scale_with_patches(tmp_path: Path) -> None:
    increments = {}
    for name, overrides in (("small", SMALL), ("large", LARGE)):
        provider = PeakProvider(tmp_path / name)
        tracemalloc.start()
        try:
            payload = bag(run_embedding(_platform(), tissue_slide(), provider, overrides + BATCH8))
            increments[name] = tracemalloc.get_traced_memory()[1] - provider.baseline
        finally:
            tracemalloc.stop()
    budget = payload.metrics.budget
    # 153 more patches: retaining their pixels would add >= 153 x 192 KiB = 29 MiB.
    assert increments["large"] - increments["small"] < 0.1 * (169 - 16) * budget.decoded_patch_bytes
    # The loop stays within the planned working set (plus store metadata and interpreter noise).
    assert increments["large"] < budget.working_set_bytes + (2 << 20)


# --------------------------------------------------------------------------- checks before reads


def test_resolution_mismatch_fails_before_any_patch_read(tmp_path: Path) -> None:
    image = tissue_slide(patch_px=128)
    overrides = SMALL + (("plan", "width", 128), ("plan", "height", 128))
    with pytest.raises(ResolutionMismatchError):
        run_embedding(_platform(), image, tmp_path, overrides)
    assert image.patch_reads == 0 and not any(tmp_path.iterdir())


def test_budget_violation_fails_before_any_patch_read(tmp_path: Path) -> None:
    image = tissue_slide()
    with pytest.raises(EmbeddingBudgetError):
        run_embedding(
            _platform(), image, tmp_path, SMALL + (("embed", "memory_budget_bytes", 1 << 20),)
        )
    assert image.patch_reads == 0 and not any(tmp_path.iterdir())


def test_undeclared_channel_order_is_rejected_by_the_gate(tmp_path: Path) -> None:
    image = tissue_slide(channel_order=ChannelOrder.BGR, fmt="test-array-bgr")
    with pytest.raises(PreprocessingCapabilityError) as info:
        run_embedding(
            _platform(), image, tmp_path, SMALL + (("embed", "channel_conversions", "none"),)
        )
    assert "channel_representation" in info.value.requirements
    assert image.requests == []  # the gate runs before any stage reads


def test_bgr_slide_uses_the_declared_reorder(tmp_path: Path) -> None:
    rgb = tissue_slide()
    bgr_pixels = rgb._pixels[0][..., ::-1].copy()
    bgr = TrackingImage("emb-slide", bgr_pixels, channel_order=ChannelOrder.BGR, fmt="test-bgr")
    m_rgb = bag(run_embedding(_platform(), rgb, tmp_path / "rgb", SMALL)).manifest
    m_bgr = bag(run_embedding(_platform(), bgr, tmp_path / "bgr", SMALL)).manifest
    assert m_bgr.transforms.names[0] == "channel_reorder"
    rows = [
        NumpyEmbeddingStoreProvider(tmp_path / k).open_existing(m.store.store_id).read_rows(0, 16)
        for k, m in (("rgb", m_rgb), ("bgr", m_bgr))
    ]
    assert np.array_equal(rows[0], rows[1])  # same colours after the declared reorder
    assert m_rgb.run_identity != m_bgr.run_identity  # but distinct, recorded transforms


def test_materialized_manifests_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ExecutionUnsupportedError) as info:
        run_embedding(
            _platform(),
            tissue_slide(),
            tmp_path,
            SMALL + (("extract", "materialize_pixels", True),),
        )
    assert info.value.missing == ("manifest:not_materialized",)


def test_a_durable_store_is_required(tmp_path: Path) -> None:
    platform, image = _platform(), tissue_slide()
    ensure_reader(platform, image)
    from application.embedding_service import WSI_EMBEDDING_TASK
    from tests.embedding_support import EMBED_PIPELINE

    with pytest.raises(ExecutionUnsupportedError) as info:
        platform.runner.run(
            ExecutionRequest(
                pipeline=platform.pipelines.get(EMBED_PIPELINE),
                image=image.describe().asset,
                task=WSI_EMBEDDING_TASK,
                models={},
                preprocessing=None,
                research_use_acknowledged=True,
                source=image,
                config_overrides=SMALL,
            )
        )
    assert info.value.missing == ("store:embedding",)
    assert image.patch_reads == 0


def test_extras_do_not_change_fingerprints(tmp_path: Path) -> None:
    """Additive runner change: per-run services never enter the reproducibility key."""
    fps = []
    for extras in ({}, {EMBEDDING_STORE_EXTRA: NumpyEmbeddingStoreProvider(tmp_path)}):
        platform, image = _platform(), tissue_slide()
        ensure_reader(platform, image)
        run = platform.runner.run(
            ExecutionRequest(
                pipeline=platform.pipelines.get(PREPROCESS_PIPELINE),
                image=image.describe().asset,
                task=WSI_PREPROCESSING_TASK,
                models={},
                preprocessing=None,
                research_use_acknowledged=True,
                source=image,
                config_overrides=(("plan", "width", 64), ("plan", "height", 64)),
                extras=extras,
            )
        ).run
        fps.append(run.fingerprint_value)
    assert fps[0] == fps[1]


# --------------------------------------------------------------------------- failure injection


def _store(root: Path, store_id: str) -> object:
    return NumpyEmbeddingStoreProvider(root).open_existing(store_id)


def _resume(root: Path, clean: dict[str, str], expected_reads: int) -> None:
    image = tissue_slide()
    payload = bag(run_embedding(_platform(), image, root, LARGE + BATCH8))
    assert image.patch_reads == expected_reads
    assert payload.manifest.content_digest == clean["content"]
    assert payload.metrics.patches_resumed == 169 - expected_reads


def test_reader_failure_leaves_an_interrupted_store_that_resumes(
    tmp_path: Path, clean: dict[str, str]
) -> None:
    platform = _platform()
    with pytest.raises(CorruptImageError):
        run_embedding(platform, tissue_slide(fail_on_patch=20), tmp_path, LARGE + BATCH8)
    assert platform.provenance.all()[-1].status is RunStatus.FAILED
    ins = _store(tmp_path, clean["store"]).inspect()  # type: ignore[attr-defined]
    assert ins.state is StoreState.INTERRUPTED and ins.committed_indices == {0, 1}
    _resume(tmp_path, clean, expected_reads=169 - 16)


@pytest.mark.parametrize(
    "mode,error",
    [
        ("raise", RuntimeError),
        ("nan", EmbeddingOutputError),
        ("dim", EmbeddingOutputError),
        ("dtype", EmbeddingOutputError),
    ],
)
def test_encoder_failures_never_commit_the_batch(
    tmp_path: Path, clean: dict[str, str], mode: str, error: type[Exception]
) -> None:
    platform = _platform()
    faulty = FaultyFactory(f"faulty-{mode}", mode, at_batch=2)
    embedding_operation(platform).add_encoder(faulty)
    with pytest.raises(error):
        run_embedding(
            platform,
            tissue_slide(),
            tmp_path,
            LARGE + BATCH8 + (("embed", "encoder", faulty.name),),
        )
    assert all(e.closed for e in faulty.created)  # encoder resources released on failure
    ins = _store(tmp_path, clean["store"]).inspect()  # type: ignore[attr-defined]
    assert ins.committed_indices == {0, 1} and ins.state is StoreState.INTERRUPTED
    _resume(tmp_path, clean, expected_reads=169 - 16)


@pytest.mark.parametrize("boundary", ["write", "commit"])
def test_store_failures_before_commit_lose_only_that_chunk(
    tmp_path: Path, clean: dict[str, str], boundary: str
) -> None:
    with pytest.raises(OSError):
        run_embedding(
            _platform(), tissue_slide(), FaultyProvider(tmp_path, boundary, at=1), LARGE + BATCH8
        )
    ins = _store(tmp_path, clean["store"]).inspect()  # type: ignore[attr-defined]
    assert ins.committed_indices == {0} and 1 in ins.missing
    _resume(tmp_path, clean, expected_reads=169 - 8)


def test_crash_before_finalize_is_not_complete_and_finishes_without_reads(
    tmp_path: Path, clean: dict[str, str]
) -> None:
    with pytest.raises(OSError):
        run_embedding(
            _platform(), tissue_slide(), FaultyProvider(tmp_path, "finalize"), LARGE + BATCH8
        )
    store = _store(tmp_path, clean["store"])
    ins = store.inspect()  # type: ignore[attr-defined]
    assert ins.state is StoreState.INTERRUPTED and not ins.missing  # all rows, yet not complete
    assert ins.finalized_manifest_digest is None
    _resume(tmp_path, clean, expected_reads=0)


def test_corrupt_chunks_fail_closed_unless_recovery_is_explicit(
    tmp_path: Path, clean: dict[str, str]
) -> None:
    run_embedding(_platform(), tissue_slide(), tmp_path, LARGE + BATCH8)
    data = tmp_path / clean["store"] / DATA
    raw = bytearray(data.read_bytes())
    raw[-1] ^= 0x01  # last row -> last chunk (index 21: rows 168..168)
    data.write_bytes(bytes(raw))
    with pytest.raises(EmbeddingStoreCorruptError):
        run_embedding(_platform(), tissue_slide(), tmp_path, LARGE + BATCH8)
    image = tissue_slide()
    recover = (("embed", "recovery", RecoveryPolicy.DISCARD_CORRUPT.value),)
    payload = bag(run_embedding(_platform(), image, tmp_path, LARGE + BATCH8 + recover))
    assert image.patch_reads == 1  # only the corrupt chunk (one row) was recomputed
    assert payload.manifest.content_digest == clean["content"]
    assert len(list((tmp_path / clean["store"] / CHUNKS).glob("*.json"))) == 22


# --------------------------------------------------------------------------- fresh processes

_SCRIPT = """
import json, sys
from pathlib import Path
from bootstrap import build_platform
from infrastructure.runtime import FixedClock
from tests.embedding_support import bag, run_embedding, stride, tissue_slide
p = build_platform(Path(sys.argv[1]), clock=FixedClock())
overrides = stride(64) + (("embed", "batch_size", 8),)
m = bag(run_embedding(p, tissue_slide(), Path(sys.argv[2]), overrides)).manifest
ml = {"torch", "timm", "zarr", "h5py", "cv2"}
mods = sorted(k for k in sys.modules if k.split(".")[0] in ml)
out = {"content": m.content_digest, "run": m.run_identity, "store": m.store.store_id}
print(json.dumps(out | {"ml": mods}))
"""


def test_fresh_processes_reproduce_the_digests(tmp_path: Path, clean: dict[str, str]) -> None:
    results = []
    for i in range(2):
        proc = subprocess.run(
            [sys.executable, "-c", _SCRIPT, str(REPO_ROOT), str(tmp_path / str(i))],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        results.append(json.loads(proc.stdout))
    assert results[0] == results[1]
    assert {k: results[0][k] for k in clean} == clean
    assert results[0]["ml"] == []  # AC-018: no ML framework is imported or needed
