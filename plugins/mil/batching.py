"""Padded MIL batches with valid-instance masks (NumPy, framework-neutral).

Bags of different lengths are zero-padded to the longest selected bag in the batch. The mask is
True exactly for real instances, so padded rows never influence a mask-aware model.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from domain.errors import EmbeddingOutputError, EmptyBagError
from domain.mil import BagSelection
from ports import MILBatch


class NumpyBatchBuilder:
    def build(
        self,
        bags: Sequence[tuple[BagSelection, object]],
        targets: tuple[int, ...] | None,
        dim: int,
    ) -> MILBatch:
        if not bags:
            raise EmptyBagError("a batch needs at least one bag")
        if targets is not None and len(targets) != len(bags):
            raise EmbeddingOutputError("one target per bag is required")
        arrays = []
        for selection, rows in bags:
            a = np.asarray(rows)
            if a.ndim != 2 or a.shape != (selection.selected_count, dim):
                raise EmbeddingOutputError(
                    f"slide {selection.slide_id}: rows {a.shape} do not match "
                    f"({selection.selected_count}, {dim})"
                )
            if a.shape[0] == 0:
                raise EmptyBagError(f"slide {selection.slide_id} has no valid instance")
            if a.dtype != np.float32 or not bool(np.isfinite(a).all()):
                raise EmbeddingOutputError(
                    f"slide {selection.slide_id}: rows must be finite float32"
                )
            arrays.append(a)
        longest = max(a.shape[0] for a in arrays)
        features = np.zeros((len(arrays), longest, dim), dtype=np.float32)
        mask = np.zeros((len(arrays), longest), dtype=np.bool_)
        for b, a in enumerate(arrays):
            features[b, : a.shape[0]] = a
            mask[b, : a.shape[0]] = True
        return MILBatch(
            slide_ids=tuple(s.slide_id for s, _ in bags),
            features=features,
            mask=mask,
            selections=tuple(s for s, _ in bags),
            targets=targets,
            padded_bytes=int(features.nbytes + mask.nbytes),
        )
