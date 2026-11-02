"""Baseline detector and NumPy helpers (synthetic arrays with known answers)."""

from __future__ import annotations

import numpy as np
import pytest

from domain.capabilities import ValidityPolicy
from domain.errors import InvalidSpatialReferenceError, PreprocessingBudgetError
from domain.geometry import Level0Rect, PyramidalGeometry, PyramidLevel
from domain.image import ChannelOrder
from domain.reading import NoDataSemantics
from domain.tissue import MaskGrid
from plugins.preprocessing.arrays import (
    AreaFractions,
    binary_close,
    binary_open,
    block_reduce,
    mask_digest,
    otsu_threshold,
)
from plugins.preprocessing.detection import (
    SaturationOtsuDetector,
    choose_mask_grid,
    mask_fractions,
    saturation,
)
from ports import DetectorInput, TissueDetector
from tests.preprocessing_support import GLASS, STAIN, slide_rgb


def _input(pixels: np.ndarray, order: ChannelOrder, acquired: np.ndarray | None) -> DetectorInput:
    h, w = pixels.shape[:2]
    g = PyramidalGeometry(w, h, (PyramidLevel(0, w, h, 1.0),))
    return DetectorInput(
        pixels=pixels,
        channel_order=order,
        dtype="uint8",
        grid=MaskGrid.for_level(g, 0),
        acquired=None if acquired is None else acquired.astype(np.float64),
        validity=NoDataSemantics.NOT_REPORTED
        if acquired is None
        else NoDataSemantics.ACQUISITION_COVERAGE,
    )


def test_detector_satisfies_the_port() -> None:
    assert isinstance(SaturationOtsuDetector(), TissueDetector)


def test_saturation_and_otsu_known_values() -> None:
    px = np.array([[GLASS, STAIN, (0, 0, 0), (255, 0, 0)]], dtype=np.uint8)
    assert saturation(px).tolist() == [[4, 134, 0, 255]]
    values = np.array([10] * 50 + [200] * 50, dtype=np.uint8)
    t = otsu_threshold(values)
    assert 10 <= t < 200
    assert otsu_threshold(np.array([7, 7, 7], dtype=np.uint8)) == 7
    with pytest.raises(ValueError):
        otsu_threshold(np.array([], dtype=np.uint8))


def test_detection_finds_the_stained_area_exactly() -> None:
    rgb, acq = slide_rgb(64, 48, tissue=((8, 8, 16, 16),), unacquired=((0, 40, 64, 8),))
    result = SaturationOtsuDetector().detect(_input(rgb, ChannelOrder.RGB, acq), {})
    expected = np.zeros((48, 64), dtype=bool)
    expected[8:24, 8:24] = True
    assert np.array_equal(np.asarray(result.tissue), expected)
    derived = dict(result.derived)
    assert 4 <= int(derived["saturation_threshold"]) < 134
    assert derived["acquired_cells"] == 64 * 40


def test_channel_views_are_explicit_and_order_invariant() -> None:
    rgb, acq = slide_rgb(64, 48, tissue=((8, 8, 16, 16),), unacquired=())
    det = SaturationOtsuDetector()
    rgba = np.concatenate([rgb, np.full((48, 64, 1), 255, np.uint8)], axis=-1)
    masks = {
        order: np.asarray(det.detect(_input(px, order, acq), {}).tissue)
        for order, px in (
            (ChannelOrder.RGB, rgb),
            (ChannelOrder.BGR, rgb[..., ::-1].copy()),
            (ChannelOrder.RGBA, rgba),
        )
    }
    assert all(np.array_equal(m, masks[ChannelOrder.RGB]) for m in masks.values())
    views = {o: det.detect(_input(rgb, o, acq), {}).channel_view for o in (ChannelOrder.BGR,)}
    assert "bgr->rgb" in views[ChannelOrder.BGR][0]
    with pytest.raises(InvalidSpatialReferenceError):
        det.detect(_input(rgb[..., :1], ChannelOrder.GRAYSCALE, acq), {})
    req = det.requirements({})
    assert ChannelOrder.GRAYSCALE not in req.channel_orders
    assert ChannelOrder.MULTICHANNEL not in req.channel_orders


def test_unacquired_pixels_never_shift_the_threshold_or_become_tissue() -> None:
    rgb, acq = slide_rgb(64, 64, tissue=((0, 0, 32, 64),), unacquired=())
    det = SaturationOtsuDetector()
    base = det.detect(_input(rgb, ChannelOrder.RGB, acq), {})
    # Add a large unacquired region filled with saturated garbage: it must be ignored entirely.
    rgb2, acq2 = rgb.copy(), acq.copy()
    rgb2[:, 48:] = (255, 0, 0)
    acq2[:, 48:] = False
    other = det.detect(_input(rgb2, ChannelOrder.RGB, acq2), {})
    assert dict(other.derived)["saturation_threshold"] == dict(base.derived)["saturation_threshold"]
    assert not np.asarray(other.tissue)[:, 48:].any()


def test_no_acquired_cell_yields_an_empty_mask_with_undefined_threshold() -> None:
    rgb, acq = slide_rgb(16, 16, (), ())
    result = SaturationOtsuDetector().detect(_input(rgb, ChannelOrder.RGB, np.zeros_like(acq)), {})
    assert not np.asarray(result.tissue).any()
    assert "undefined" in str(dict(result.derived)["otsu_threshold"])


def test_min_threshold_floors_otsu_on_uniform_glass() -> None:
    rgb, acq = slide_rgb(32, 32, tissue=(), unacquired=())
    rgb[::2, ::2] = (234, 236, 240)  # saturation 6 vs 4: tiny achromatic variation
    result = SaturationOtsuDetector().detect(
        _input(rgb, ChannelOrder.RGB, acq), {"min_threshold": 8}
    )
    assert not np.asarray(result.tissue).any()
    assert dict(result.derived)["saturation_threshold"] == 8


def test_morphology_is_configurable_and_deterministic() -> None:
    m = np.zeros((9, 9), dtype=bool)
    m[2:7, 2:7] = True
    m[0, 8] = True  # isolated speck
    m[4, 4] = False  # pinhole
    opened = binary_open(m, 1)
    assert not opened[0, 8]
    closed = binary_close(m, 1)
    assert closed[4, 4]
    assert np.array_equal(binary_open(m, 0), m)


def test_block_reduce_averages_acquired_pixels_only() -> None:
    px = np.zeros((4, 5, 3), dtype=np.uint8)
    px[:, :2] = 100
    px[:, 2:4] = 50
    px[:, 4] = 250
    acq = np.ones((4, 5), dtype=bool)
    acq[0, 0] = False
    values, frac = block_reduce(px, acq, 2)
    assert values.shape == (2, 3, 3)
    assert values[0, 0, 0] == 100 and frac[0, 0] == 0.75
    assert values[0, 2, 0] == 250 and frac[0, 2] == 1.0  # partial block counts real pixels
    v1, f1 = block_reduce(px, None, 1)
    assert np.array_equal(v1, px) and f1.min() == 1.0


def test_area_fractions_are_exact_area_weights() -> None:
    g = PyramidalGeometry(10, 10, (PyramidLevel(0, 10, 10, 1.0), PyramidLevel(1, 5, 5, 2.0)))
    grid = MaskGrid.for_level(g, 1)
    values = np.zeros((5, 5))
    values[0, 0] = 1.0
    fr = AreaFractions(grid)
    assert fr.mean(values, Level0Rect(0, 0, 2, 2)) == 1.0
    assert fr.mean(values, Level0Rect(1, 1, 2, 2)) == 0.25
    assert fr.mean(values, Level0Rect(0, 0, 4, 4)) == 0.25
    assert fr.mean(values, Level0Rect(1, 0, 3, 2)) == pytest.approx(1 / 3)
    t, a = mask_fractions(values.astype(bool), np.ones((5, 5)), grid)
    assert (t, a) == (0.04, 1.0)


def test_mask_digest_depends_on_content_and_validity() -> None:
    t = np.zeros((3, 3), dtype=bool)
    d1 = mask_digest(t, None)
    t2 = t.copy()
    t2[1, 1] = True
    assert d1 != mask_digest(t2, None) != mask_digest(t, np.ones((3, 3)))
    assert d1 == mask_digest(t.copy(), None)


def test_mask_source_selection_prefers_stored_levels_and_bounds_decoding() -> None:
    levels = tuple(PyramidLevel(i, 4000 // 2**i, 3000 // 2**i, float(2**i)) for i in range(4))
    g = PyramidalGeometry(4000, 3000, levels)
    grid = choose_mask_grid(g, {"max_mask_pixels": 1 << 20})
    assert (grid.source_level, grid.block_factor) == (2, 1)  # 2000x1500 too big; 1000x750 fits
    single = PyramidalGeometry(4000, 3000, (PyramidLevel(0, 4000, 3000, 1.0),))
    reduced = choose_mask_grid(single, {"max_mask_pixels": 1 << 20})
    assert reduced.block_factor == 4 and reduced.width * reduced.height <= 1 << 20
    with pytest.raises(PreprocessingBudgetError, match="max_decoded_pixels"):
        choose_mask_grid(single, {"max_decoded_pixels": 1_000_000})


def test_validity_policy_is_part_of_detector_requirements() -> None:
    det = SaturationOtsuDetector()
    assert det.requirements({}).validity_policy is ValidityPolicy.REQUIRE_REPORTED
    assumed = det.requirements({"validity_policy": "assume_acquired"})
    assert assumed.validity_policy is ValidityPolicy.ASSUME_ACQUIRED
    with pytest.raises(InvalidSpatialReferenceError):
        det.requirements({"validity_policy": "guess"})
