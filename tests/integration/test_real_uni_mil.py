"""Opt-in SPEC-007 AC-018: MIL loading of real embedding bags from F-CZI-01 and F-SVS-01.

* With the fixtures set (and torch), `torch-fixture` bags are embedded from both real slides.
* With `PATHOLOGY_AI_UNI_CHECKPOINT` too, real UNI bags (SPEC-006) are embedded and loaded.

No label is attached to any real slide: no real labelled cohort is available, and labels are
never inferred. These tests establish loading, masking, compatibility and refusal behaviour on
real bags. They make no claim about classification performance.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="optional extra pathology-ai[embedding]")

from application.mil_dataset import MILDatasetService  # noqa: E402
from bootstrap import build_mil, build_platform  # noqa: E402
from domain.errors import EmbeddingCompatibilityError  # noqa: E402
from domain.ids import DatasetId, PipelineId, Version  # noqa: E402
from domain.mil import (  # noqa: E402
    EXACT,
    BagPolicy,
    EmbeddingCompatibility,
    LabelStatus,
    SlideLabel,
)
from domain.mil_training import LossKind  # noqa: E402
from domain.pipeline import PipelineRef  # noqa: E402
from infrastructure.mil_bags import NumpyBagSource  # noqa: E402
from infrastructure.runtime import FixedClock  # noqa: E402
from plugins.mil import ATTENTION_MIL, DESCRIPTORS, TorchMILFactory  # noqa: E402
from plugins.mil.batching import NumpyBatchBuilder  # noqa: E402
from plugins.models.fixture import write_fixture_checkpoint  # noqa: E402
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
from tests.mil_support import BINARY, cohort, split_of, training_config  # noqa: E402

pytestmark = pytest.mark.filterwarnings("error")
UNI_CHECKPOINT_ENV = "PATHOLOGY_AI_UNI_CHECKPOINT"
UNI_PIPELINE = PipelineRef(PipelineId("wsi-patch-embedding-uni"), Version("1.0.0"))


@pytest.fixture(scope="module")
def slides() -> dict[str, Path]:
    pytest.importorskip("pylibCZIrw.czi", reason="optional extra pathology-ai[czi]")
    pytest.importorskip("openslide", reason="optional extra pathology-ai[openslide]")
    return {
        "f-czi-01": fixture_path(CZI_FIXTURE_ENV, CZI_FIXTURE_SIZE, CZI_FIXTURE_SHA256),
        "f-svs-01": fixture_path(SVS_FIXTURE_ENV, SVS_FIXTURE_SIZE, SVS_FIXTURE_SHA256),
    }


@pytest.fixture(scope="module")
def uni_stores(
    slides: dict[str, Path], tmp_path_factory: pytest.TempPathFactory
) -> tuple[Path, dict[str, str]]:
    pytest.importorskip("timm", reason="optional extra pathology-ai[embedding]")
    value = os.environ.get(UNI_CHECKPOINT_ENV)
    if not value:
        pytest.skip(f"real UNI weights not configured: set {UNI_CHECKPOINT_ENV} (UNVERIFIED)")
    root = tmp_path_factory.mktemp("uni-stores")
    platform = build_platform(REPO_ROOT, clock=FixedClock(), checkpoints={"uni": Path(value)})
    stores = {}
    for image_id, path in slides.items():
        out = platform.embedding_runs.run(
            source(path, image_id), platform.pipelines.get(UNI_PIPELINE), root
        )
        stores[image_id] = out.manifest.store.store_id
    return root, stores


@pytest.fixture(scope="module")
def fixture_stores(
    slides: dict[str, Path], tmp_path_factory: pytest.TempPathFactory
) -> tuple[Path, dict[str, str]]:
    root = tmp_path_factory.mktemp("fx-stores")
    ckpt = root.parent / "fixture.pt"
    sha = write_fixture_checkpoint(ckpt)
    platform = build_platform(REPO_ROOT, clock=FixedClock(), checkpoints={"torch-fixture": ckpt})
    pipe = with_embed_config(
        platform.pipelines.get(EMBED_PIPELINE),
        encoder="torch-fixture",
        checkpoint_sha256=sha,
        input_px=256,
        embedding_dim=32,
    )
    stores = {}
    for image_id, path in slides.items():
        out = platform.embedding_runs.run(source(path, image_id), pipe, root)
        stores[image_id] = out.manifest.store.store_id
    return root, stores


def _unlabeled(stores: dict[str, str]) -> list[SlideLabel]:
    return [
        SlideLabel(s, BINARY.ref, None, "none", "0", LabelStatus.UNLABELED) for s in sorted(stores)
    ]


def _check_real_bags(root: Path, stores: dict[str, str], dim: int) -> None:
    src = NumpyBagSource([root])
    infos = {s: src.describe(store) for s, store in stores.items()}
    rows = {s: i.rows for s, i in infos.items()}
    assert len(set(rows.values())) == 2  # the two slides have different patch counts
    for info in infos.values():
        assert info.dim == dim and info.dtype == "float32"
        assert info.embedding_identity.startswith("pai-embid-")
    # One compatibility key across CZI (bgr->rgb) and SVS (rgba->rgb): channel adaptations are
    # source-side; the encoder, weights, input contract and model-side transforms are identical.
    names = {s: i.run.transforms.names[0] for s, i in infos.items()}
    assert names == {"f-czi-01": "channel_reorder", "f-svs-01": "drop_alpha"}
    keys = {EmbeddingCompatibility.of(i.run, EXACT).key for i in infos.values()}
    assert len(keys) == 1
    # An unlabeled dataset: accepted for inference; nothing is inferred or invented.
    d = MILDatasetService(src).build(
        dataset_id=DatasetId("real-fixtures"),
        version=Version("1"),
        task=BINARY,
        policy=EXACT,
        labels=_unlabeled(stores),
        stores=stores,
    )
    assert d.counts() == {"eligible": 0, "excluded": 0, "unlabeled": 2}
    assert all(r.label is None and r.group_id is None for r in d.records)
    assert {r.slide_id: r.patch_count for r in d.records} == rows
    # Variable-length bags, padded with exact masks; every row is read back bit-identically.
    policy = BagPolicy(None, 0, 1 << 30)
    loaded = []
    for s in sorted(stores):
        sel = policy.select(s, infos[s].rows, None)
        data = src.read(stores[s], sel.indices)
        assert np.isfinite(data).all()
        loaded.append((sel, data))
    batch = NumpyBatchBuilder().build(loaded, None, dim)
    mask = np.asarray(batch.mask)
    assert mask.shape == (2, max(rows.values()))
    assert mask.sum(axis=1).tolist() == [rows[s] for s in sorted(stores)]
    for b, s in enumerate(sorted(stores)):
        expected = src.read(stores[s], tuple(range(rows[s])))
        assert np.array_equal(np.asarray(batch.features)[b, : rows[s]], expected)


def test_real_fixture_encoder_bags_load_as_variable_length_bags(
    fixture_stores: tuple[Path, dict[str, str]],
) -> None:
    _check_real_bags(*fixture_stores, dim=32)


def test_real_uni_bags_load_as_variable_length_bags(
    uni_stores: tuple[Path, dict[str, str]],
) -> None:
    _check_real_bags(*uni_stores, dim=1024)


def test_models_trained_on_other_embeddings_refuse_real_uni_bags(
    uni_stores: tuple[Path, dict[str, str]], tmp_path: Path
) -> None:
    """A synthetic-embedding model never silently runs on UNI bags (AC-006, AC-018)."""
    c = cohort(tmp_path / "syn", BINARY, groups_per_class=3)
    d, s = c.dataset(), split_of(c.dataset())
    root, stores = uni_stores
    services = build_mil([c.root, root], tmp_path / "ckpt")
    exp = services.training.experiment(
        name="synthetic",
        dataset=d,
        split=s,
        architecture=ATTENTION_MIL,
        hyperparameters=(),
        training=training_config(epochs=2),
    )
    services.training.train(exp, d, s)
    with pytest.raises(EmbeddingCompatibilityError, match="trained on embeddings"):
        services.training.predict(exp.identity, sorted(stores.items()))


def test_attention_mil_forward_on_real_uni_bags(uni_stores: tuple[Path, dict[str, str]]) -> None:
    """Technical execution only: an untrained, seeded model on real 1024-d bags."""
    root, stores = uni_stores
    src = NumpyBagSource([root])
    policy = BagPolicy(None, 0, 1 << 30)
    loaded = []
    for slide in sorted(stores):
        info = src.describe(stores[slide])
        sel = policy.select(slide, info.rows, None)
        loaded.append((sel, src.read(stores[slide], sel.indices)))
    batch = NumpyBatchBuilder().build(loaded, None, 1024)
    spec = DESCRIPTORS[ATTENTION_MIL].spec(1024, BINARY)
    model = TorchMILFactory(ATTENTION_MIL).create(
        spec, loss=LossKind.BCE_WITH_LOGITS, seed=0, training=None
    )
    start = time.perf_counter()
    out = model.predict(batch)
    seconds = time.perf_counter() - start
    assert all(np.isfinite(r).all() for r in out.logits)
    assert out.attention is not None
    assert [sum(a) for a in out.attention] == pytest.approx([1.0, 1.0], abs=1e-6)
    assert [len(a) for a in out.attention] == [s.selected_count for s, _ in loaded]
    print(  # measurement for the report (pytest -s)
        f"padded_batch_bytes={batch.padded_bytes} forward_seconds={seconds:.4f} "
        f"parameters={model.parameter_count}"
    )
