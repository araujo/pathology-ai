"""NumPy helpers: block reduction, Otsu threshold, binary morphology, area-weighted fractions."""

from __future__ import annotations

import hashlib
import math

import numpy as np
import numpy.typing as npt

from domain.geometry import Level0Rect
from domain.tissue import MaskGrid

Bool2D = npt.NDArray[np.bool_]
Float2D = npt.NDArray[np.float64]
UInt8 = npt.NDArray[np.uint8]

FRACTION_DECIMALS = 9


def block_reduce(
    pixels: UInt8, acquired: Bool2D | None, factor: int
) -> tuple[npt.NDArray[np.float64], Float2D]:
    """Mean of acquired pixels per `factor`x`factor` block (partial edge blocks allowed).

    Returns `(values (h', w', c), acquired_fraction (h', w'))`. Blocks without acquired pixels
    get value 0; when `acquired` is None every pixel counts as acquired. Sums are integer, so
    memory stays at a few bytes per decoded pixel.
    """
    h, w, c = pixels.shape
    acq = np.ones((h, w), dtype=np.bool_) if acquired is None else acquired
    if factor == 1:
        values = np.where(acq[..., None], pixels, 0).astype(np.float64)
        return values, acq.astype(np.float64)
    hb, wb = math.ceil(h / factor), math.ceil(w / factor)
    ph, pw = hb * factor - h, wb * factor - w
    kept = np.pad(np.where(acq[..., None], pixels, np.uint8(0)), ((0, ph), (0, pw), (0, 0)))
    sums = kept.reshape(hb, factor, wb, factor, c).sum(axis=(1, 3), dtype=np.uint32)
    counts = (
        np.pad(acq, ((0, ph), (0, pw)))
        .reshape(hb, factor, wb, factor)
        .sum(axis=(1, 3), dtype=np.uint32)
    )
    inside = np.pad(np.ones((h, w), dtype=np.bool_), ((0, ph), (0, pw)))
    totals = inside.reshape(hb, factor, wb, factor).sum(axis=(1, 3), dtype=np.uint32)
    values = np.where(counts[..., None] > 0, sums / np.maximum(counts, 1)[..., None], 0.0)
    return values, counts / totals


def otsu_threshold(values: UInt8) -> int:
    """Otsu's threshold over uint8 values (pixels > threshold are the upper class).

    Deterministic: the first maximum wins; a single-valued input returns that value.
    """
    hist = np.bincount(values.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total == 0:
        raise ValueError("Otsu threshold of an empty set")
    levels = np.arange(256, dtype=np.float64)
    w0 = np.cumsum(hist)
    m0 = np.cumsum(hist * levels)
    w1 = total - w0
    mt = m0[-1]
    valid = (w0 > 0) & (w1 > 0)
    if not valid.any():
        return int(values.ravel()[0])
    between = np.zeros(256)
    between[valid] = (mt * w0[valid] / total - m0[valid]) ** 2 / (w0[valid] * w1[valid])
    return int(np.argmax(between))


def _shift_reduce(mask: Bool2D, radius: int, *, dilate: bool) -> Bool2D:
    if radius <= 0:
        return mask.copy()
    h, w = mask.shape
    padded = np.pad(mask, radius, mode="edge")
    out = np.zeros_like(mask) if dilate else np.ones_like(mask)
    size = 2 * radius + 1
    for dy in range(size):
        for dx in range(size):
            window = padded[dy : dy + h, dx : dx + w]
            out = (out | window) if dilate else (out & window)
    return out


def binary_open(mask: Bool2D, radius: int) -> Bool2D:
    return _shift_reduce(_shift_reduce(mask, radius, dilate=False), radius, dilate=True)


def binary_close(mask: Bool2D, radius: int) -> Bool2D:
    return _shift_reduce(_shift_reduce(mask, radius, dilate=True), radius, dilate=False)


def mask_digest(tissue: Bool2D, acquired: Float2D | None) -> str:
    h = hashlib.sha256()
    h.update(f"tissue:{tissue.shape[0]}x{tissue.shape[1]};".encode())
    h.update(np.ascontiguousarray(tissue, dtype=np.uint8).tobytes())
    if acquired is not None:
        h.update(b"acquired:float64;")
        h.update(np.ascontiguousarray(np.round(acquired, FRACTION_DECIMALS), dtype="<f8").tobytes())
    return h.hexdigest()


def _edges(n: int, cell: float, extent: int) -> Float2D:
    edges = np.minimum(np.arange(n + 1, dtype=np.float64) * cell, float(extent))
    edges[-1] = float(extent)
    return edges


def _weights(edges: Float2D, lo: float, hi: float) -> tuple[int, Float2D]:
    first = max(int(np.searchsorted(edges, lo, side="right")) - 1, 0)
    last = min(int(np.searchsorted(edges, hi, side="left")), len(edges) - 1)
    starts, stops = edges[first:last], edges[first + 1 : last + 1]
    return first, np.clip(np.minimum(stops, hi) - np.maximum(starts, lo), 0.0, None)


class AreaFractions:
    """Exact area-weighted mean of a cell grid over level-zero rectangles."""

    def __init__(self, grid: MaskGrid) -> None:
        self._xe = _edges(grid.width, grid.cell_x, grid.level0_width)
        self._ye = _edges(grid.height, grid.cell_y, grid.level0_height)

    def mean(self, values: Float2D, rect: Level0Rect) -> float:
        x0, wx = _weights(self._xe, rect.x, rect.x + rect.width)
        y0, wy = _weights(self._ye, rect.y, rect.y + rect.height)
        block = values[y0 : y0 + len(wy), x0 : x0 + len(wx)]
        area = float(wx.sum() * wy.sum())
        if area <= 0:
            return 0.0
        return round(float(wy @ block @ wx) / area, FRACTION_DECIMALS)
