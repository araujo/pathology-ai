"""Verified, read-only MIL bag access over SPEC-005 NumPy embedding stores (ADR-007 D1).

A store is described only after `verify()` has re-hashed every committed chunk, the final
manifest re-verifies against the store, and its status is `finalized`. Rows are read from the
read-only memory map by selected index, so only selected rows are ever copied into memory.

Stores are found by id below one or more roots; an id present under two roots is ambiguous and
refused. Roots are operational configuration and never enter an identity.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt

from domain.embedding import StoreState, embedding_identity
from domain.errors import (
    EmbeddingStoreCorruptError,
    EmbeddingStoreIncompleteError,
    EmptyBagError,
    MILDatasetError,
)
from domain.hashing import digest
from domain.mil import EmbeddingBagInfo
from infrastructure.embedding_store import HEADER, NumpyEmbeddingStore, NumpyEmbeddingStoreProvider


class NumpyBagSource:
    def __init__(self, roots: Sequence[Path]) -> None:
        if not roots:
            raise MILDatasetError("at least one embedding store root is required")
        self._providers = [NumpyEmbeddingStoreProvider(Path(r)) for r in roots]
        self._stores: dict[str, NumpyEmbeddingStore] = {}
        self._info: dict[str, EmbeddingBagInfo] = {}
        self.rows_read = 0
        self.reads: list[str] = []

    def _open(self, store_id: str) -> NumpyEmbeddingStore:
        if store_id in self._stores:
            return self._stores[store_id]
        found = [p for p in self._providers if (p.root / store_id / HEADER).exists()]
        if not found:
            raise EmbeddingStoreIncompleteError(
                f"embedding store {store_id} is not under any configured root"
            )
        if len(found) > 1:
            raise MILDatasetError(f"embedding store {store_id} exists under several roots")
        store = found[0].open_existing(store_id)
        self._stores[store_id] = store
        return store

    def describe(self, store_id: str) -> EmbeddingBagInfo:
        if store_id in self._info:
            return self._info[store_id]
        store = self._open(store_id)
        body = store.finalized_manifest()  # verifies every chunk and the manifest
        if body.get("status") != StoreState.FINALIZED.value:
            raise EmbeddingStoreIncompleteError(f"store {store_id} is not finalized")
        header = store.header
        if body.get("run_identity") != header.identity.value:
            raise EmbeddingStoreCorruptError(f"store {store_id}: manifest and header disagree")
        if (body.get("embedding_dim"), body.get("embedding_dtype")) != (
            header.layout.dim,
            header.layout.dtype,
        ):
            raise EmbeddingStoreCorruptError(f"store {store_id}: manifest schema differs")
        content = str(body["content_digest"])
        info = EmbeddingBagInfo(
            store_id=store_id,
            image_id=header.identity.image_id,
            run=header.identity,
            manifest_digest=digest(body),
            embedding_identity=embedding_identity(header.identity.value, content),
            content_digest=content,
            rows=header.layout.rows,
            dim=header.layout.dim,
            dtype=header.layout.dtype,
            records=header.records,
        )
        self._info[store_id] = info
        return info

    def read(self, store_id: str, indices: tuple[int, ...]) -> npt.NDArray[np.float32]:
        if store_id not in self._info:
            self.describe(store_id)  # never read rows of an unverified store
        info = self._info[store_id]
        if not indices:
            raise EmptyBagError(f"no rows selected from store {store_id}")
        if list(indices) != sorted(set(indices)) or indices[0] < 0 or indices[-1] >= info.rows:
            raise MILDatasetError(
                f"store {store_id}: selected rows must be sorted, unique and within "
                f"0..{info.rows - 1}"
            )
        rows = np.asarray(self._stores[store_id].read_selected(indices), dtype=np.float32)
        self.rows_read += len(indices)
        self.reads.append(store_id)
        return rows

    def close(self) -> None:
        for store in self._stores.values():
            store.close()
        self._stores.clear()
