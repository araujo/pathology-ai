"""Backend-neutral helpers shared by real-image reader adapters (numpy only)."""

from __future__ import annotations

import math
from importlib import metadata
from types import TracebackType
from typing import Self

import numpy as np
import numpy.typing as npt

from domain.errors import ReaderClosedError
from domain.geometry import Level0Rect
from domain.provenance import ComponentVersion

ADAPTER_VERSION = "0.1.0"

Tiles = npt.NDArray[np.int64]  # (N, 4): x0, y0, x1, y1 half-open, level-zero pixels


def distribution_component(distribution: str) -> ComponentVersion:
    try:
        version = metadata.version(distribution)
    except metadata.PackageNotFoundError:
        version = "not-installed"
    return ComponentVersion("backend", distribution, version)


def coverage_mask(
    tiles: Tiles, rect: Level0Rect, downsample: float, width: int, height: int
) -> npt.NDArray[np.bool_]:
    """Boolean mask at the returned resolution: True where the pixel centre lies in a tile.

    Pixel (row i, column j) of a region read at `downsample` has its centre at level-zero
    `(rect.x + (j + 0.5) * downsample, rect.y + (i + 0.5) * downsample)`.
    """
    mask = np.zeros((height, width), dtype=np.bool_)
    if tiles.size == 0:
        return mask
    x1, y1 = rect.x + rect.width, rect.y + rect.height
    hit = (tiles[:, 0] < x1) & (tiles[:, 2] > rect.x) & (tiles[:, 1] < y1) & (tiles[:, 3] > rect.y)
    for tx0, ty0, tx1, ty1 in tiles[hit].tolist():
        j0 = max(0, math.ceil((tx0 - rect.x) / downsample - 0.5))
        j1 = min(width, math.ceil((tx1 - rect.x) / downsample - 0.5))
        i0 = max(0, math.ceil((ty0 - rect.y) / downsample - 0.5))
        i1 = min(height, math.ceil((ty1 - rect.y) / downsample - 0.5))
        if j0 < j1 and i0 < i1:
            mask[i0:i1, j0:j1] = True
    return mask


class ClosableHandle:
    """Deterministic close-once lifecycle shared by adapters."""

    _closed: bool = False

    @property
    def closed(self) -> bool:
        return self._closed

    def _require_open(self) -> None:
        if self._closed:
            raise ReaderClosedError("image handle is closed")

    def _release(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._release()

    def __enter__(self) -> Self:
        self._require_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
