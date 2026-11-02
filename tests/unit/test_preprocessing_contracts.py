"""SPEC-004 domain contracts: mask geometry, patch specs, resolution, grids, decisions, frames."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from domain.capabilities import (
    CheckStatus,
    PreprocessingRequirements,
    ValidityPolicy,
    require_capabilities,
)
from domain.errors import (
    InvalidImageAssetError,
    InvalidSpatialReferenceError,
    MissingSpatialMetadataError,
    PatchBudgetExceededError,
    PreprocessingCapabilityError,
    UnsupportedResolutionError,
)
from domain.frames import FrameTranslation, Placement, padded_origin_translation
from domain.geometry import Level0Rect, PixelSpacing, PyramidalGeometry, PyramidLevel
from domain.image import ChannelOrder
from domain.patching import (
    EdgePolicy,
    PatchDecision,
    PatchSpec,
    PatchStatus,
    PatchUnits,
    RejectionReason,
    SelectionPolicy,
    apply_selection,
    decide,
    grid_size,
    plan_candidates,
    resolve_patch_geometry,
)
from domain.reading import NativeFrameOffset, NoDataSemantics
from domain.tissue import MaskGrid, MaskSource
from tests.preprocessing_support import ArrayImage, slide_rgb


def geometry(spacing: float | None = 0.25, ratio: float = 2.0) -> PyramidalGeometry:
    w, h = 1000, 700
    levels = tuple(
        PyramidLevel(i, int(w // ratio**i), int(h // ratio**i), ratio**i) for i in range(3)
    )
    return PyramidalGeometry(w, h, levels, PixelSpacing(spacing, spacing) if spacing else None)


# --------------------------------------------------------------------------- mask geometry


def test_mask_grid_maps_cells_to_level0_and_reaches_the_image_edge() -> None:
    g = geometry()
    grid = MaskGrid.for_level(g, 2)  # level 2: 250 x 175, ds 4
    assert (grid.width, grid.height, grid.cell_x) == (250, 175, 4.0)
    assert grid.cell_rect(0, 0) == Level0Rect(0, 0, 4, 4)
    assert grid.cell_rect(174, 249) == Level0Rect(996, 696, 4, 4)
    assert grid.edge_adjustment == (0.0, 0.0)
    blocked = MaskGrid.for_level(g, 0, 3)  # partial last block is clamped, never overruns
    assert blocked.source is MaskSource.BLOCK_REDUCED and blocked.width == 334
    last = blocked.cell_rect(0, 333)
    assert last.x + last.width == 1000 and last.width == 1
    assert blocked.edge_adjustment == (-2.0, -2.0)


def test_mask_grid_with_non_integer_downsample_stretches_the_last_cell() -> None:
    g = geometry(ratio=1.5)  # level 1: 666 x 466 at ds 1.5 -> covers 999 x 699
    grid = MaskGrid.for_level(g, 1)
    assert grid.cell_rect(0, 1) == Level0Rect(1.5, 0, 1.5, 1.5)
    last = grid.cell_rect(465, 665)
    assert (last.x + last.width, last.y + last.height) == (1000, 700)
    assert grid.edge_adjustment == (1.0, 1.0)


def test_mask_grid_rejects_cells_beyond_the_image() -> None:
    with pytest.raises(InvalidSpatialReferenceError):
        MaskGrid(300, 10, 4.0, 4.0, 1000, 700, MaskSource.STORED_LEVEL, 2)
    with pytest.raises(InvalidSpatialReferenceError):
        MaskGrid(10, 10, 4.0, 4.0, 1000, 700, MaskSource.STORED_LEVEL, 2, block_factor=2)


# --------------------------------------------------------------------------- specs and resolution


def test_spec_requires_an_explicit_read_resolution() -> None:
    with pytest.raises(InvalidSpatialReferenceError, match="exactly one"):
        PatchSpec(64, 64)
    with pytest.raises(InvalidSpatialReferenceError, match="exactly one"):
        PatchSpec(64, 64, level=0, target_spacing_um=0.5)
    with pytest.raises(InvalidSpatialReferenceError):
        PatchSpec(64, 64, level=0, min_tissue_fraction=1.5)
    with pytest.raises(InvalidSpatialReferenceError):
        PatchSpec(0, 64, level=0)


def test_pixel_patches_at_an_explicit_level() -> None:
    r = resolve_patch_geometry(PatchSpec(64, 32, level=1), geometry())
    assert (r.level, r.downsample, r.width_px, r.height_px) == (1, 2.0, 64, 32)
    assert (r.stride_x_px, r.stride_y_px) == (64, 32)
    assert r.effective_spacing_um == (0.5, 0.5) and r.spacing_relative_error is None


def test_target_spacing_selects_the_closest_stored_level_within_tolerance() -> None:
    r = resolve_patch_geometry(
        PatchSpec(64, 64, target_spacing_um=0.9, spacing_tolerance=0.15), geometry()
    )
    assert r.level == 2 and r.effective_spacing_um == (1.0, 1.0)
    assert r.spacing_relative_error == pytest.approx(1 / 9)
    with pytest.raises(UnsupportedResolutionError, match="resampling is not performed"):
        resolve_patch_geometry(PatchSpec(64, 64, target_spacing_um=0.7), geometry())


def test_physical_units_derive_pixel_sizes_and_need_spacing() -> None:
    spec = PatchSpec(64.0, 32.0, units=PatchUnits.MICROMETERS, level=1, stride_x=48.0)
    r = resolve_patch_geometry(spec, geometry())
    assert (r.width_px, r.height_px, r.stride_x_px, r.stride_y_px) == (128, 64, 96, 64)
    with pytest.raises(MissingSpatialMetadataError):
        resolve_patch_geometry(spec, geometry(spacing=None))
    with pytest.raises(MissingSpatialMetadataError):
        resolve_patch_geometry(PatchSpec(64, 64, target_spacing_um=0.5), geometry(spacing=None))
    # Pixel units at an explicit level work without spacing; the spacing stays unknown.
    r = resolve_patch_geometry(PatchSpec(64, 64, level=0), geometry(spacing=None))
    assert r.effective_spacing_um is None


def test_patch_larger_than_level_is_rejected() -> None:
    with pytest.raises(InvalidSpatialReferenceError, match="does not fit"):
        resolve_patch_geometry(PatchSpec(300, 64, level=2), geometry())


# --------------------------------------------------------------------------- grids


def test_grid_is_row_major_unique_inside_bounds_and_handles_remainders() -> None:
    g = geometry()
    spec = PatchSpec(64, 64, level=1)  # level 1: 500 x 350
    cands = plan_candidates(spec, resolve_patch_geometry(spec, g), g)
    assert len(cands) == 7 * 5 == grid_size(spec, resolve_patch_geometry(spec, g), g)
    assert [c.index for c in cands] == list(range(len(cands)))
    keys = [(c.level_rect.y, c.level_rect.x) for c in cands]
    assert keys == sorted(keys) and len(set(keys)) == len(keys)
    for c in cands:
        g.require_within(c.level0)
        assert c.level0 == g.to_level0(c.level_rect)
    assert max(c.level_rect.x for c in cands) == 384  # 500 - 64 = 436 remainder left uncovered


def test_edge_policy_align_to_edge_covers_the_remainder_without_padding() -> None:
    g = geometry()
    spec = PatchSpec(64, 64, level=1, edge_policy=EdgePolicy.ALIGN_TO_EDGE)
    cands = plan_candidates(spec, resolve_patch_geometry(spec, g), g)
    xs = sorted({c.level_rect.x for c in cands})
    ys = sorted({c.level_rect.y for c in cands})
    assert xs[-1] == 500 - 64 and ys[-1] == 350 - 64
    assert len(cands) == 8 * 6


def test_stride_overlap_and_origin() -> None:
    g = geometry()
    spec = PatchSpec(64, 64, level=1, stride_x=32, stride_y=64, grid_origin_x=10)
    cands = plan_candidates(spec, resolve_patch_geometry(spec, g), g)
    xs = sorted({c.level_rect.x for c in cands})
    assert xs[:3] == [10, 42, 74] and xs[-1] + 64 <= 500


def test_non_integer_downsample_grid_maps_exactly() -> None:
    g = geometry(ratio=1.5)
    spec = PatchSpec(100, 100, level=1)
    cands = plan_candidates(spec, resolve_patch_geometry(spec, g), g)
    assert cands[1].level0 == Level0Rect(150.0, 0.0, 150.0, 150.0)


def test_candidate_budget_is_checked_before_materialising() -> None:
    g = geometry()
    spec = PatchSpec(8, 8, level=0, max_candidates=100)
    with pytest.raises(PatchBudgetExceededError, match="max_candidates"):
        plan_candidates(spec, resolve_patch_geometry(spec, g), g)


# --------------------------------------------------------------------------- decisions and budgets


def _decisions(
    spec: PatchSpec, fractions: Sequence[tuple[float, float | None]]
) -> tuple[PatchDecision, ...]:
    g = geometry()
    cands = plan_candidates(spec, resolve_patch_geometry(spec, g), g)
    return tuple(decide(c, t, a, spec) for c, (t, a) in zip(cands, fractions, strict=False))


def test_tissue_and_acquisition_are_judged_independently() -> None:
    spec = PatchSpec(64, 64, level=1, min_tissue_fraction=0.5, min_acquired_fraction=0.9)
    d = _decisions(spec, [(0.9, 1.0), (0.9, 0.5), (0.1, 1.0), (0.1, 0.2), (0.9, None)])
    assert [x.status for x in d] == [
        PatchStatus.SELECTED,
        PatchStatus.REJECTED,
        PatchStatus.REJECTED,
        PatchStatus.REJECTED,
        PatchStatus.SELECTED,
    ]
    assert d[1].reasons == (RejectionReason.INSUFFICIENT_ACQUIRED_DATA,)
    assert d[2].reasons == (RejectionReason.INSUFFICIENT_TISSUE,)
    assert set(d[3].reasons) == set(RejectionReason)


def test_budget_fails_closed_by_default_and_selection_policies_are_deterministic() -> None:
    fr = [(1.0, 1.0)] * 10
    with pytest.raises(PatchBudgetExceededError, match="explicit selection"):
        apply_selection(
            _decisions(PatchSpec(64, 64, level=1, max_patches=4), fr),
            PatchSpec(64, 64, level=1, max_patches=4),
        )
    for policy, expected in (
        (SelectionPolicy.ROW_MAJOR_FIRST, [0, 1, 2, 3]),
        (SelectionPolicy.UNIFORM_STRIDE, [0, 2, 5, 7]),
    ):
        spec = PatchSpec(64, 64, level=1, max_patches=4, selection=policy)
        out, note = apply_selection(_decisions(spec, fr), spec)
        kept = [d.candidate.index for d in out if d.status is PatchStatus.SELECTED]
        assert kept == expected and policy.value in note
        dropped = [d for d in out if d.status is PatchStatus.NOT_SELECTED_BUDGET]
        assert len(dropped) == 6
        assert apply_selection(_decisions(spec, fr), spec) == (out, note)


# --------------------------------------------------------------------------- capabilities


def test_requirements_report_every_verdict_and_fail_closed() -> None:
    rgb, acq = slide_rgb(64, 64, (), ())
    image = ArrayImage(
        "cap", rgb, acquired=acq, no_data=NoDataSemantics.NOT_REPORTED, spacing_um=None, levels=1
    )
    req = PreprocessingRequirements(
        channel_orders=frozenset({ChannelOrder.BGR}),
        dtypes=frozenset({"uint8"}),
        needs_physical_spacing=True,
        validity_policy=ValidityPolicy.REQUIRE_REPORTED,
    )
    checks = {c.requirement: c for c in req.evaluate(image.describe(), supports_regions=False)}
    assert checks["channel_representation"].status is CheckStatus.UNSUPPORTED
    assert checks["dtype"].status is CheckStatus.SATISFIED
    assert checks["physical_spacing"].status is CheckStatus.UNSUPPORTED
    assert checks["acquisition_validity"].status is CheckStatus.UNSUPPORTED
    assert checks["region_reads"].status is CheckStatus.UNSUPPORTED
    with pytest.raises(PreprocessingCapabilityError) as err:
        require_capabilities(checks.values(), context="test")
    assert set(err.value.requirements) == {
        "channel_representation",
        "physical_spacing",
        "acquisition_validity",
        "region_reads",
    }
    assume = PreprocessingRequirements(validity_policy=ValidityPolicy.ASSUME_ACQUIRED)
    decided = {c.requirement: c for c in assume.evaluate(image.describe(), supports_regions=True)}
    assert decided["acquisition_validity"].status is CheckStatus.DECIDED_BY_POLICY
    require_capabilities(decided.values(), context="test")


def test_multi_plane_images_are_unsupported() -> None:
    rgb, acq = slide_rgb(64, 64, (), ())
    image = ArrayImage("z", rgb, acquired=acq, z_planes=3, levels=1)
    checks = PreprocessingRequirements().evaluate(image.describe(), supports_regions=True)
    assert any(c.requirement == "single_plane" and c.blocking for c in checks)


# --------------------------------------------------------------------------- frames


def test_frame_translation_is_derived_from_native_origins_per_image() -> None:
    t = FrameTranslation.from_native_origins(
        frame="legacy", other_origin=(-262080, 144720), level0=NativeFrameOffset(-262063, 144720)
    )
    assert (t.dx, t.dy) == (-17, 0)
    assert t.to_level0(2013, 8052) == (1996, 8052) and t.from_level0(1996, 8052) == (2013, 8052)
    other = FrameTranslation.from_native_origins(
        frame="legacy", other_origin=(-500, 300), level0=NativeFrameOffset(-480, 290)
    )
    assert (other.dx, other.dy) == (-20, 10)  # never a universal offset
    g = geometry()
    assert t.place(17, 0, 100, 100, g) == (Placement.INSIDE, Level0Rect(0, 0, 100, 100))
    assert t.place(0, 0, 100, 100, g)[0] is Placement.CROSSES_BOUNDS
    assert t.place(5000, 0, 100, 100, g) == (Placement.OUTSIDE, None)


def test_padded_translation_requires_reported_metadata() -> None:
    rgb, acq = slide_rgb(64, 64, (), ())
    with pytest.raises(MissingSpatialMetadataError):
        padded_origin_translation(ArrayImage("p", rgb, acquired=acq, levels=1).describe(), "legacy")


def test_tissue_mask_invariants_are_enforced() -> None:
    from domain.ids import ImageId
    from domain.provenance import ComponentVersion
    from domain.tissue import TissueMask

    grid = MaskGrid.for_level(geometry(), 2)
    c = ComponentVersion("x", "y", "1")
    base = dict(
        image_id=ImageId("m"),
        source_integrity=None,
        scene=0,
        grid=grid,
        reader=c,
        detector=c,
        parameters=(),
        derived=(),
        input_channel_order="rgb",
        channel_view=(),
        validity_semantics=NoDataSemantics.NOT_REPORTED,
        validity_policy=ValidityPolicy.REQUIRE_REPORTED,
        tissue_fraction=0.5,
        acquired_fraction=None,
        mask_digest="0" * 64,
        capability_checks=(),
        limitations=(),
        decoded_pixels=1,
        tissue=object(),
        acquired=None,
    )
    with pytest.raises(InvalidImageAssetError, match="assume_acquired"):
        TissueMask(**base)  # type: ignore[arg-type]
    base["validity_policy"] = ValidityPolicy.ASSUME_ACQUIRED
    TissueMask(**base)  # type: ignore[arg-type]
    base["validity_semantics"] = NoDataSemantics.ACQUISITION_COVERAGE
    with pytest.raises(InvalidImageAssetError, match="exactly when"):
        TissueMask(**base)  # type: ignore[arg-type]
