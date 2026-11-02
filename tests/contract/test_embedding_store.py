"""SPEC-005 store contract: `NumpyEmbeddingStore` against the `EmbeddingStore` port.

Covers incremental writes, reopen, finalization, missing / overlapping / corrupt / interrupted
chunks, recovery policies and incompatible identities. Nothing is accepted because of a shape.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from domain.embedding import (
    EmbeddingManifest,
    EmbeddingRecord,
    EmbeddingRunIdentity,
    EmbeddingStoreHeader,
    PixelState,
    Precision,
    RecoveryPolicy,
    ResolutionRecord,
    StoreDescriptor,
    StoreLayout,
    StoreState,
    build_transform_chain,
    patch_ids_digest,
)
from domain.errors import (
    EmbeddingOutputError,
    EmbeddingStoreConflictError,
    EmbeddingStoreCorruptError,
    EmbeddingStoreIncompatibleError,
    EmbeddingStoreIncompleteError,
)
from domain.geometry import Level0Rect, LevelRect
from domain.ids import ImageId
from domain.image import ChannelOrder
from domain.manifest import PATCH_MANIFEST_SCHEMA, PixelStorage
from domain.patching import ResolvedPatchGeometry
from domain.provenance import ComponentVersion
from infrastructure.embedding_store import (
    CHUNKS,
    DATA,
    FORMAT,
    HEADER,
    MANIFEST,
    NumpyEmbeddingStore,
    NumpyEmbeddingStoreProvider,
    header_from_body,
)
from plugins.embedding.reference import INPUT_SPEC, ReferenceProjectionEncoder
from plugins.embedding.transforms import IMPLEMENTATIONS
from ports import EmbeddingStore, EmbeddingStoreProvider

DIM = 8
COMP = ComponentVersion("test", "store-contract", "1")


def make_header(
    rows: int = 10, chunk_rows: int = 4, dim: int = DIM, seed: str = "a"
) -> EmbeddingStoreHeader:
    records = tuple(
        EmbeddingRecord(
            row=i,
            patch_id=f"pai-patch-{seed}{i:04d}",
            plan_index=3 * i,
            level_rect=LevelRect(1, 256 * i, 0, 256, 256),
            level0=Level0Rect(512.0 * i, 0.0, 512.0, 512.0),
            effective_spacing_um=(0.5, 0.5),
            field_of_view_um=(128.0, 128.0),
        )
        for i in range(rows)
    )
    src = PixelState.of_patch(256, 256, ChannelOrder.RGB, 3, "uint8", (0.5, 0.5))
    chain = build_transform_chain(
        src,
        INPUT_SPEC,
        channel_conversions=frozenset(),
        resample=None,
        implementations=IMPLEMENTATIONS,
    )
    identity = EmbeddingRunIdentity(
        image_id=ImageId("store-contract"),
        source_integrity="test:store",
        scene=0,
        plan_digest="plan",
        patch_manifest_digest=f"manifest-{seed}",
        records_digest=patch_ids_digest(records),
        encoder=ReferenceProjectionEncoder(dim).identity,
        input_spec=INPUT_SPEC,
        transforms=chain,
        precision=Precision.FP32,
        device="cpu",
    )
    return EmbeddingStoreHeader(
        identity.store_id, identity, StoreLayout(rows, dim, "float32", chunk_rows), records
    )


def rows_for(header: EmbeddingStoreHeader, index: int) -> np.ndarray[Any, np.dtype[np.float32]]:
    start, stop = header.layout.chunk_range(index)
    base = np.arange(start, stop, dtype=np.float32)[:, None]
    return (base + np.arange(header.layout.dim, dtype=np.float32) / 100).astype(np.float32)


def fill(store: EmbeddingStore, indices: range | list[int] | None = None) -> None:
    h = store.header
    for i in indices if indices is not None else range(h.layout.chunk_count):
        store.write_chunk(i, rows_for(h, i))
        store.commit_chunk(i)


def manifest_for(store: EmbeddingStore) -> EmbeddingManifest:
    h, v = store.header, store.verify()
    enc = h.identity.encoder
    return EmbeddingManifest(
        image_id=h.identity.image_id,
        source_integrity=h.identity.source_integrity,
        scene=0,
        pipeline=None,
        reader=COMP,
        backend=COMP,
        operation=COMP,
        plan_digest="plan",
        patch_manifest_digest=h.identity.patch_manifest_digest,
        patch_manifest_schema=PATCH_MANIFEST_SCHEMA,
        patch_pixel_storage=PixelStorage.NOT_MATERIALIZED,
        run_identity=h.identity.value,
        encoder=enc,
        input_spec=INPUT_SPEC,
        transforms=h.identity.transforms,
        resolution=ResolutionRecord.of(
            None,
            None,
            ResolvedPatchGeometry(1, 2.0, 256, 256, 256, 256, (0.5, 0.5), None, None),
            PixelState(
                256, 256, ChannelOrder.RGB, 3, "float32", (0.0, 1.0), INPUT_SPEC.layout, True, None
            ),
            INPUT_SPEC,
            h.identity.transforms,
        ),
        precision=Precision.FP32,
        device="cpu",
        embedding_dim=enc.output_dim,
        embedding_dtype=enc.output_dtype,
        row_count=h.layout.rows,
        records=h.records,
        store=StoreDescriptor(
            FORMAT,
            h.schema_version,
            h.store_id,
            h.layout.chunk_rows,
            h.layout.chunk_count,
            v.chunks_digest,
        ),
        content_digest=v.content_digest,
        status=StoreState.FINALIZED,
        limitations=(),
    )


@pytest.fixture
def provider(tmp_path: Path) -> NumpyEmbeddingStoreProvider:
    return NumpyEmbeddingStoreProvider(tmp_path)


def test_adapter_satisfies_the_ports(provider: NumpyEmbeddingStoreProvider) -> None:
    assert isinstance(provider, EmbeddingStoreProvider)
    store = provider.open(make_header(), recovery=RecoveryPolicy.RESUME_VERIFIED)
    assert isinstance(store, EmbeddingStore)
    store.close()


def test_preallocated_shape_is_not_completion(provider: NumpyEmbeddingStoreProvider) -> None:
    h = make_header()
    store = provider.open(h, recovery=RecoveryPolicy.RESUME_VERIFIED)
    data = np.load(store.directory / DATA, mmap_mode="r")
    assert data.shape == (10, DIM)  # the full array exists on disk...
    ins = store.inspect()
    assert ins.state is StoreState.CREATED and ins.missing == (0, 1, 2)  # ...but nothing counts
    with pytest.raises(EmbeddingStoreIncompleteError):
        store.verify()
    with pytest.raises(EmbeddingStoreIncompleteError):
        store.finalize(manifest_for_partial(store))


def manifest_for_partial(store: NumpyEmbeddingStore) -> EmbeddingManifest:
    """A syntactically valid manifest for a store that is not complete (must be refused)."""
    fill(store)
    m = manifest_for(store)
    for path in sorted((store.directory / CHUNKS).iterdir()):
        path.unlink()
    store._committed = None  # forget the session view; the disk is the truth
    return m


def test_writes_are_incremental_and_committed_one_chunk_at_a_time(
    provider: NumpyEmbeddingStoreProvider,
) -> None:
    h = make_header()
    store = provider.open(h, recovery=RecoveryPolicy.RESUME_VERIFIED)
    records = store.directory / CHUNKS
    for i in range(h.layout.chunk_count):
        store.write_chunk(i, rows_for(h, i))
        assert len(list(records.glob("*.json"))) == i  # written, not yet committed
        commit = store.commit_chunk(i)
        assert (commit.index, commit.start, commit.stop) == (i, *h.layout.chunk_range(i))
        assert len(list(records.glob("*.json"))) == i + 1
        assert store.inspect().state is StoreState.IN_PROGRESS
    store.finalize(manifest_for(store))
    assert store.inspect().state is StoreState.FINALIZED
    assert np.array_equal(store.read_rows(4, 8), rows_for(h, 1))


def test_reopen_finalized_store_verifies_and_never_rewrites(
    provider: NumpyEmbeddingStoreProvider,
) -> None:
    h = make_header()
    with provider.open(h, recovery=RecoveryPolicy.RESUME_VERIFIED) as store:
        fill(store)
        m = manifest_for(store)
        store.finalize(m)
    with provider.open(h, recovery=RecoveryPolicy.RESUME_VERIFIED) as again:
        ins = again.inspect()
        assert ins.state is StoreState.FINALIZED
        assert ins.finalized_manifest_digest == m.manifest_digest
        with pytest.raises(EmbeddingStoreConflictError):
            again.write_chunk(0, rows_for(h, 0))
        again.finalize(m)  # idempotent with the identical manifest
        with pytest.raises(EmbeddingStoreConflictError):
            again.finalize(replace(m, limitations=("different",)))
    existing = provider.open_existing(h.store_id)
    assert existing.header == h and existing.inspect().state is StoreState.FINALIZED
    with pytest.raises(EmbeddingStoreConflictError):
        existing.write_chunk(0, rows_for(h, 0))


def test_header_round_trips_exactly(provider: NumpyEmbeddingStoreProvider) -> None:
    h = make_header()
    provider.open(h, recovery=RecoveryPolicy.RESUME_VERIFIED).close()
    body = json.loads((provider.root / h.store_id / HEADER).read_text())["body"]
    assert header_from_body(body) == h


def test_missing_chunks_block_finalization(provider: NumpyEmbeddingStoreProvider) -> None:
    h = make_header()
    store = provider.open(h, recovery=RecoveryPolicy.RESUME_VERIFIED)
    fill(store, [0, 2])
    ins = store.inspect()
    assert ins.missing == (1,) and ins.state is StoreState.IN_PROGRESS
    with pytest.raises(EmbeddingStoreIncompleteError):
        store.verify()


def test_overlapping_repeated_and_unwritten_commits_are_conflicts(
    provider: NumpyEmbeddingStoreProvider,
) -> None:
    h = make_header()
    store = provider.open(h, recovery=RecoveryPolicy.RESUME_VERIFIED)
    with pytest.raises(EmbeddingStoreConflictError):
        store.commit_chunk(0)  # nothing written
    fill(store, [0])
    with pytest.raises(EmbeddingStoreConflictError):
        store.write_chunk(0, rows_for(h, 0))  # overlaps a committed chunk
    with pytest.raises(EmbeddingStoreConflictError):
        store.commit_chunk(0)
    with pytest.raises(EmbeddingStoreConflictError):
        store.write_chunk(1, rows_for(h, 0)[:2])  # wrong row count
    with pytest.raises(EmbeddingStoreConflictError):
        store.write_chunk(1, rows_for(h, 1).astype(np.float64))
    bad = rows_for(h, 1)
    bad[0, 0] = np.inf
    with pytest.raises(EmbeddingOutputError):
        store.write_chunk(1, bad)


def test_interrupted_write_is_not_committed_and_resumes(
    provider: NumpyEmbeddingStoreProvider,
) -> None:
    h = make_header()
    store = provider.open(h, recovery=RecoveryPolicy.RESUME_VERIFIED)
    fill(store, [0])
    store.write_chunk(1, rows_for(h, 1))  # crash before the commit record
    (store.directory / CHUNKS / ".00000002.json.tmp").write_text("{torn")  # torn temporary
    store.close()
    again = provider.open(h, recovery=RecoveryPolicy.RESUME_VERIFIED)
    ins = again.inspect()
    assert ins.state is StoreState.INTERRUPTED
    assert ins.committed_indices == {0} and ins.missing == (1, 2)
    fill(again, [1, 2])
    again.finalize(manifest_for(again))
    assert again.inspect().state is StoreState.FINALIZED


def _flip_byte(path: Path, offset_from_end: int) -> None:
    data = bytearray(path.read_bytes())
    data[-offset_from_end] ^= 0xFF
    path.write_bytes(bytes(data))


def test_corrupt_bytes_are_detected_and_policy_decides(
    provider: NumpyEmbeddingStoreProvider,
) -> None:
    h = make_header()
    with provider.open(h, recovery=RecoveryPolicy.RESUME_VERIFIED) as store:
        fill(store)
        good = manifest_for(store)
        store.finalize(good)
    _flip_byte(provider.root / h.store_id / DATA, 1)  # last row -> chunk 2
    existing = provider.open_existing(h.store_id)
    ins = existing.inspect()
    assert ins.state is StoreState.CORRUPT
    assert [i for i, _ in ins.corrupt] == [2]
    with pytest.raises(EmbeddingStoreCorruptError):
        existing.verify()
    with pytest.raises(EmbeddingStoreCorruptError):
        provider.open(h, recovery=RecoveryPolicy.RESUME_VERIFIED)
    with provider.open(h, recovery=RecoveryPolicy.DISCARD_CORRUPT) as recovered:
        ins = recovered.inspect()
        assert ins.state is StoreState.INTERRUPTED and ins.missing == (2,)
        assert not (recovered.directory / MANIFEST).exists()
        fill(recovered, [2])
        again = manifest_for(recovered)
        recovered.finalize(again)
    assert again.content_digest == good.content_digest


def test_tampered_or_foreign_commit_records_are_corrupt(
    provider: NumpyEmbeddingStoreProvider,
) -> None:
    h = make_header()
    store = provider.open(h, recovery=RecoveryPolicy.RESUME_VERIFIED)
    fill(store, [0, 1])
    record = store.directory / CHUNKS / "00000001.json"
    doc = json.loads(record.read_text())
    doc["body"]["commit"]["stop"] = 9  # edited without re-digesting
    record.write_text(json.dumps(doc))
    assert [i for i, _ in store.inspect().corrupt] == [1]
    record.write_text("not json")
    assert [i for i, _ in store.inspect().corrupt] == [1]


def test_truncated_data_is_corrupt(provider: NumpyEmbeddingStoreProvider) -> None:
    h = make_header()
    with provider.open(h, recovery=RecoveryPolicy.RESUME_VERIFIED) as store:
        fill(store)
    path = provider.root / h.store_id / DATA
    path.write_bytes(path.read_bytes()[:-40])
    with pytest.raises(EmbeddingStoreCorruptError):
        provider.open(h, recovery=RecoveryPolicy.RESUME_VERIFIED)
    with provider.open(h, recovery=RecoveryPolicy.DISCARD_CORRUPT) as recovered:
        assert recovered.inspect().missing == (0, 1, 2)  # nothing on a reallocated array counts


def test_tampered_manifest_is_not_finalized(provider: NumpyEmbeddingStoreProvider) -> None:
    h = make_header()
    with provider.open(h, recovery=RecoveryPolicy.RESUME_VERIFIED) as store:
        fill(store)
        store.finalize(manifest_for(store))
    path = provider.root / h.store_id / MANIFEST
    doc = json.loads(path.read_text())
    doc["body"]["content_digest"] = "0" * 64
    from domain.hashing import digest

    doc["digest"] = digest(doc["body"])  # internally consistent, but wrong for these bytes
    path.write_text(json.dumps(doc))
    ins = provider.open_existing(h.store_id).inspect()
    assert ins.state is StoreState.CORRUPT and ins.finalized_manifest_digest is None


def test_manifest_without_complete_chunks_is_corrupt(provider: NumpyEmbeddingStoreProvider) -> None:
    h = make_header()
    with provider.open(h, recovery=RecoveryPolicy.RESUME_VERIFIED) as store:
        fill(store)
        store.finalize(manifest_for(store))
    (provider.root / h.store_id / CHUNKS / "00000001.json").unlink()
    ins = provider.open_existing(h.store_id).inspect()
    assert ins.state is StoreState.CORRUPT and ins.missing == (1,)


def test_incompatible_layout_is_never_resumed(provider: NumpyEmbeddingStoreProvider) -> None:
    h = make_header()
    with provider.open(h, recovery=RecoveryPolicy.RESUME_VERIFIED) as store:
        fill(store, [0])
    other = make_header(chunk_rows=5)  # same run identity, different chunking
    assert other.store_id == h.store_id
    with pytest.raises(EmbeddingStoreIncompatibleError):
        provider.open(other, recovery=RecoveryPolicy.DISCARD_CORRUPT)
    with provider.open(h, recovery=RecoveryPolicy.RESUME_VERIFIED) as again:
        assert again.inspect().committed_indices == {0}  # the original store is untouched


def test_identity_mismatch_at_a_location_is_incompatible(tmp_path: Path) -> None:
    """A store directory renamed onto another run's id is detected by its header."""
    provider = NumpyEmbeddingStoreProvider(tmp_path)
    a, b = make_header(seed="a"), make_header(seed="b")
    with provider.open(a, recovery=RecoveryPolicy.RESUME_VERIFIED) as store:
        fill(store, [0])
    (tmp_path / a.store_id).rename(tmp_path / b.store_id)
    with pytest.raises(EmbeddingStoreIncompatibleError):
        provider.open(b, recovery=RecoveryPolicy.DISCARD_CORRUPT)


def test_store_ids_cannot_escape_the_root(provider: NumpyEmbeddingStoreProvider) -> None:
    with pytest.raises(EmbeddingStoreConflictError):
        provider.open_existing("../elsewhere")


def test_records_without_header_are_corrupt(provider: NumpyEmbeddingStoreProvider) -> None:
    h = make_header()
    with provider.open(h, recovery=RecoveryPolicy.RESUME_VERIFIED) as store:
        fill(store, [0])
    (provider.root / h.store_id / HEADER).unlink()
    with pytest.raises(EmbeddingStoreCorruptError):
        provider.open(h, recovery=RecoveryPolicy.DISCARD_CORRUPT)


def test_finalize_refuses_a_manifest_for_other_rows(provider: NumpyEmbeddingStoreProvider) -> None:
    h = make_header()
    store = provider.open(h, recovery=RecoveryPolicy.RESUME_VERIFIED)
    fill(store)
    m = manifest_for(store)
    with pytest.raises(EmbeddingStoreConflictError):
        store.finalize(replace(m, content_digest="f" * 64))
    with pytest.raises(EmbeddingStoreConflictError):
        store.finalize(replace(m, run_identity="other"))
    assert store.inspect().state is StoreState.IN_PROGRESS
