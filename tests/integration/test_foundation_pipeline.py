"""SPEC-006 end to end on in-memory slides: a real torch encoder through `embed.patches@1`.

The `torch-fixture` encoder (random seeded weights from a local file) runs through the existing
`PipelineRunner`, durable store and recovery machinery. Covers persistence, provenance, failure
injection (inference error, invalid output, unavailable device, missing checkpoint), resume
without mixing identities, reproducibility across fresh processes, and the base install without
torch. Embeddings carry no meaning.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="optional extra pathology-ai[embedding]")

from bootstrap import Platform, build_platform  # noqa: E402
from domain.embedding import StoreState  # noqa: E402
from domain.errors import (  # noqa: E402
    CheckpointMissingError,
    DeviceUnavailableError,
    EmbeddingBudgetError,
    EmbeddingOutputError,
    ModelInferenceError,
)
from domain.pipeline import ConfigValue, PipelineDefinition  # noqa: E402
from domain.provenance import RunStatus  # noqa: E402
from infrastructure.embedding_store import NumpyEmbeddingStoreProvider  # noqa: E402
from infrastructure.runtime import FixedClock  # noqa: E402
from plugins.embedding.transforms import apply_chain  # noqa: E402
from plugins.models import fixture_net  # noqa: E402
from plugins.models.fixture import write_fixture_checkpoint  # noqa: E402
from ports import EncodeContext  # noqa: E402
from tests.conftest import REPO_ROOT  # noqa: E402
from tests.embedding_support import (  # noqa: E402
    EMBED_PIPELINE,
    bag,
    run_embedding,
    stride,
    tissue_slide,
    with_embed_config,
)

pytestmark = pytest.mark.filterwarnings("error")
SMALL = stride(256)  # 16 patches of 256 px
BATCH = 8


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    path = tmp_path_factory.mktemp("fx") / "fixture.pt"
    return path, write_fixture_checkpoint(path)


def _platform(path: Path | None) -> Platform:
    bindings = {} if path is None else {"torch-fixture": path}
    return build_platform(REPO_ROOT, clock=FixedClock(), checkpoints=bindings)


def _pipeline(platform: Platform, sha: str, **extra: ConfigValue) -> PipelineDefinition:
    cfg: dict[str, ConfigValue] = {
        "encoder": "torch-fixture",
        "checkpoint_sha256": sha,
        "input_px": 256,
        "embedding_dim": 32,
        "batch_size": BATCH,
        **extra,
    }
    return with_embed_config(platform.pipelines.get(EMBED_PIPELINE), **cfg)


def _run(path: Path, sha: str, out: Path, image: Any = None, **extra: ConfigValue) -> Any:
    p = _platform(path)
    return run_embedding(p, image or tissue_slide(), out, SMALL, _pipeline(p, sha, **extra))


@pytest.fixture(scope="module")
def clean(ckpt: tuple[Path, str], tmp_path_factory: pytest.TempPathFactory) -> Any:
    return bag(_run(*ckpt, tmp_path_factory.mktemp("clean"))).manifest


def test_torch_encoder_writes_a_durable_embedding_bag(
    ckpt: tuple[Path, str], tmp_path: Path
) -> None:
    outcome = _run(*ckpt, tmp_path)
    payload = bag(outcome)
    m = payload.manifest
    assert outcome.run.status is RunStatus.SUCCEEDED
    assert (m.row_count, m.embedding_dim, m.embedding_dtype) == (16, 32, "float32")
    assert m.status is StoreState.FINALIZED and m.store.chunk_count == 2
    assert m.encoder.weights_checksum == f"sha256:{ckpt[1]}"
    params = dict(m.encoder.parameters)
    assert params["batch_size"] == BATCH and "runtime.torch" in params
    assert any("random weights" in t for t in m.limitations)
    assert str(tmp_path) not in json.dumps(params) and str(ckpt[0]) not in repr(m)
    resources = dict(payload.metrics.encoder_resources)
    assert resources["samples_encoded"] == 16 and int(resources["parameter_bytes"]) > 0
    # Rows are read back from the store only; the encoder's activations entered the budget.
    store = NumpyEmbeddingStoreProvider(tmp_path).open_existing(m.store.store_id)
    rows = np.asarray(store.read_rows(0, 16))
    store.close()
    assert rows.shape == (16, 32) and np.isfinite(rows).all()
    assert payload.metrics.budget.scratch_bytes > BATCH * 2 * 3 * 256 * 256 * 4


def test_rows_match_direct_encoding_of_the_reader_patches(
    ckpt: tuple[Path, str], tmp_path: Path
) -> None:
    platform, image = _platform(ckpt[0]), tissue_slide()
    pipeline = _pipeline(platform, ckpt[1])
    m = bag(run_embedding(platform, image, tmp_path, SMALL, pipeline)).manifest
    store = NumpyEmbeddingStoreProvider(tmp_path).open_existing(m.store.store_id)
    rows = np.asarray(store.read_rows(0, m.row_count))
    store.close()
    encoder = platform.encoders.resolve("torch-fixture").create(
        dict(next(s for s in pipeline.stages if str(s.stage_id) == "embed").config)
    )
    from domain.reading import RegionReadRequest

    first = m.records[:BATCH]
    batch = np.stack(
        [
            apply_chain(
                m.transforms,
                np.asarray(
                    image.read_region(
                        RegionReadRequest(0, int(r.level_rect.x), int(r.level_rect.y), 256, 256)
                    ).pixels
                ),
            )
            for r in first
        ]
    )
    ctx = EncodeContext(0, 0, tuple(r.patch_id for r in first), m.precision, m.device)
    assert np.array_equal(
        np.asarray(encoder.encode_batch(batch, ctx)), rows[:BATCH]
    )  # same batch: exact
    encoder.close()


def test_repeated_runs_reuse_or_reproduce_the_artifact(
    ckpt: tuple[Path, str], clean: Any, tmp_path: Path
) -> None:
    first = bag(_run(*ckpt, tmp_path))
    again = bag(_run(*ckpt, tmp_path))
    assert again.metrics.reused_finalized_store and again.metrics.region_reads == 0
    assert again.manifest.manifest_digest == first.manifest.manifest_digest
    assert first.manifest.content_digest == clean.content_digest
    assert first.manifest.embedding_identity == clean.embedding_identity


def test_different_weights_never_share_a_store(
    ckpt: tuple[Path, str], clean: Any, tmp_path: Path
) -> None:
    other = tmp_path / "other.pt"
    m = bag(_run(other, write_fixture_checkpoint(other, seed=3), tmp_path / "o")).manifest
    assert m.store.store_id != clean.store.store_id
    assert m.run_identity != clean.run_identity and m.content_digest != clean.content_digest


def test_reference_and_torch_encoders_have_distinct_identities(clean: Any, tmp_path: Path) -> None:
    ref = bag(run_embedding(_platform(None), tissue_slide(), tmp_path, SMALL)).manifest
    assert ref.run_identity != clean.run_identity and ref.encoder != clean.encoder


def _failing_forward(on_call: int, mode: str) -> Callable[[Any, Any], Any]:
    calls = {"n": 0}
    original = fixture_net.forward

    def forward(module: Any, x: Any) -> Any:
        calls["n"] += 1  # call 1 is the load-time probe, call 2 is batch 0
        out = original(module, x)
        if calls["n"] != on_call:
            return out
        if mode == "raise":
            raise RuntimeError("injected device failure")
        return out * float("nan")

    return forward


@pytest.mark.parametrize(
    "mode,error", [("raise", ModelInferenceError), ("nan", EmbeddingOutputError)]
)
def test_inference_failure_interrupts_and_resume_completes_identically(
    ckpt: tuple[Path, str],
    clean: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    error: type[Exception],
) -> None:
    with monkeypatch.context() as patch:
        patch.setattr(fixture_net, "forward", _failing_forward(3, mode))  # batch 1
        with pytest.raises(error):
            _run(*ckpt, tmp_path)
    store = NumpyEmbeddingStoreProvider(tmp_path).open_existing(clean.store.store_id)
    inspection = store.inspect()
    store.close()
    assert inspection.state is StoreState.INTERRUPTED
    assert inspection.committed_indices == frozenset({0})  # the failed batch is never committed
    resumed = bag(_run(*ckpt, tmp_path))
    assert resumed.metrics.patches_resumed == BATCH and resumed.metrics.region_reads == 8
    assert resumed.manifest.content_digest == clean.content_digest
    assert resumed.manifest.manifest_digest == clean.manifest_digest


def test_resume_with_other_weights_does_not_touch_the_interrupted_store(
    ckpt: tuple[Path, str], clean: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with monkeypatch.context() as patch:
        patch.setattr(fixture_net, "forward", _failing_forward(3, "raise"))
        with pytest.raises(ModelInferenceError):
            _run(*ckpt, tmp_path)
    other = tmp_path / "other.pt"
    m = bag(_run(other, write_fixture_checkpoint(other, seed=5), tmp_path)).manifest
    assert m.store.store_id != clean.store.store_id
    store = NumpyEmbeddingStoreProvider(tmp_path).open_existing(clean.store.store_id)
    assert store.inspect().state is StoreState.INTERRUPTED  # left as it was, never mixed
    assert store.header.identity.encoder.weights_checksum == f"sha256:{ckpt[1]}"
    store.close()


def test_unavailable_device_fails_before_any_patch_read(
    ckpt: tuple[Path, str], tmp_path: Path
) -> None:
    if torch.cuda.is_available():
        pytest.skip("CUDA is available; the failure path cannot be exercised")
    image = tissue_slide()
    with pytest.raises(DeviceUnavailableError, match="CUDA"):
        _run(*ckpt, tmp_path, image=image, device="cuda")
    assert image.patch_reads == 0 and not any(tmp_path.iterdir())


def test_missing_checkpoint_fails_closed_and_is_recorded(
    ckpt: tuple[Path, str], tmp_path: Path
) -> None:
    platform, image = _platform(None), tissue_slide()
    with pytest.raises(CheckpointMissingError):
        run_embedding(platform, image, tmp_path, SMALL, _pipeline(platform, ckpt[1]))
    assert image.patch_reads == 0
    (run,) = platform.provenance.all()
    assert run.status is RunStatus.FAILED and run.error_code == "CHECKPOINT_MISSING"


def test_encoder_working_memory_is_budgeted_before_reads(
    ckpt: tuple[Path, str], tmp_path: Path
) -> None:
    image = tissue_slide()
    with pytest.raises(EmbeddingBudgetError):
        _run(*ckpt, tmp_path, image=image, memory_budget_bytes=16 << 20)
    assert image.patch_reads == 0


_SCRIPT = """
import json, sys
from pathlib import Path
from bootstrap import build_platform
from infrastructure.runtime import FixedClock
from tests.embedding_support import EMBED_PIPELINE, bag, run_embedding, stride, tissue_slide
from tests.embedding_support import with_embed_config
ckpt, sha, out = Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
p = build_platform(Path(sys.argv[1]), clock=FixedClock(), checkpoints={"torch-fixture": ckpt})
pipe = with_embed_config(p.pipelines.get(EMBED_PIPELINE), encoder="torch-fixture",
    checkpoint_sha256=sha, input_px=256, embedding_dim=32, batch_size=8)
m = bag(run_embedding(p, tissue_slide(), out, stride(256), pipe)).manifest
print(json.dumps({"content": m.content_digest, "manifest": m.manifest_digest,
    "embedding": m.embedding_identity}))
"""


def test_fresh_processes_reproduce_cpu_torch_embeddings(
    ckpt: tuple[Path, str], clean: Any, tmp_path: Path
) -> None:
    results = [
        json.loads(
            subprocess.run(
                [sys.executable, "-c", _SCRIPT, str(REPO_ROOT), str(ckpt[0]), ckpt[1], str(out)],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        )
        for out in (tmp_path / "a", tmp_path / "b")
    ]
    assert results[0] == results[1]
    assert results[0] == {
        "content": clean.content_digest,
        "manifest": clean.manifest_digest,
        "embedding": clean.embedding_identity,
    }


_ISOLATED = """
import json, sys
sys.modules["torch"] = None  # the optional runtime is not installed
sys.modules["timm"] = None
from pathlib import Path
from bootstrap import build_platform
from domain.errors import DomainError
from infrastructure.runtime import FixedClock
from tests.embedding_support import EMBED_PIPELINE, bag, run_embedding, stride, tissue_slide
from tests.embedding_support import with_embed_config
root, ckpt, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
p = build_platform(root, clock=FixedClock(), checkpoints={"torch-fixture": ckpt})
ref = bag(run_embedding(p, tissue_slide(), out / "ref", stride(256))).manifest
pipe = with_embed_config(p.pipelines.get(EMBED_PIPELINE), encoder="torch-fixture",
    checkpoint_sha256="0" * 64, input_px=256, embedding_dim=32)
try:
    run_embedding(p, tissue_slide(), out / "torch", stride(256), pipe)
    code = None
except DomainError as exc:
    code, message = exc.code, str(exc)
print(json.dumps({"ref_rows": ref.row_count, "code": code, "message": message}))
"""


def test_base_runtime_works_without_torch_and_reports_the_missing_extra(
    ckpt: tuple[Path, str], tmp_path: Path
) -> None:
    proc = subprocess.run(
        [sys.executable, "-c", _ISOLATED, str(REPO_ROOT), str(ckpt[0]), str(tmp_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(proc.stdout)
    assert result["ref_rows"] == 16
    assert result["code"] == "MODEL_RUNTIME_UNAVAILABLE"
    assert "pathology-ai[embedding]" in result["message"]
