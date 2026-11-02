from __future__ import annotations

import pytest

from domain.errors import (
    CoordinateFrameMismatchError,
    InvalidIdentifierError,
    InvalidSpatialReferenceError,
    MissingSpatialMetadataError,
)
from domain.geometry import (
    CoordinateFrame,
    Level0Rect,
    LevelRect,
    PhysicalPoint,
    PixelSpacing,
    PointGeometry,
    PyramidalGeometry,
    PyramidLevel,
    RectangleGeometry,
    VolumeGeometry,
    VoxelBoxGeometry,
    VoxelIndex,
    validate_geometry_against,
)
from domain.ids import DatasetId, ModelId

SPACING = PixelSpacing(0.25, 0.25)


def _pyramid(spacing: PixelSpacing | None = SPACING) -> PyramidalGeometry:
    return PyramidalGeometry(
        1000, 800, (PyramidLevel(0, 1000, 800, 1.0), PyramidLevel(1, 500, 400, 2.0)), spacing
    )


def _volume(**kw: object) -> VolumeGeometry:
    args: dict[str, object] = dict(
        shape=(10, 20, 30),
        spacing_mm=(0.5, 0.5, 2.0),
        origin_mm=(10.0, 20.0, 30.0),
        direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    )
    args.update(kw)
    return VolumeGeometry(**args)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", ["", "Upper", "a/b", "../x", " x", "x" * 129])
def test_identifiers_reject_paths_and_display_names(bad: str) -> None:
    with pytest.raises(InvalidIdentifierError):
        DatasetId(bad)


def test_identifier_kinds_are_distinct_types() -> None:
    assert DatasetId("abc") != ModelId("abc")  # type: ignore[comparison-overlap]


def test_level_to_level0_round_trip() -> None:
    g = _pyramid()
    r0 = g.to_level0(LevelRect(1, 10, 20, 30, 40))
    assert (r0.x, r0.y, r0.width, r0.height) == (20, 40, 60, 80)
    back = g.from_level0(r0, 1)
    assert (back.x, back.y, back.width, back.height) == (10, 20, 30, 40)


def test_level_region_outside_bounds_is_rejected() -> None:
    with pytest.raises(InvalidSpatialReferenceError):
        _pyramid().to_level0(LevelRect(1, 490, 0, 20, 10))


def test_unknown_level_rejected() -> None:
    with pytest.raises(InvalidSpatialReferenceError):
        _pyramid().level(5)


def test_level0_must_match_native_dimensions() -> None:
    with pytest.raises(InvalidSpatialReferenceError):
        PyramidalGeometry(1000, 800, (PyramidLevel(0, 999, 800, 1.0),))


def test_missing_pixel_spacing_is_not_fabricated() -> None:
    with pytest.raises(MissingSpatialMetadataError):
        _pyramid(None).require_pixel_spacing()


def test_voxel_to_physical_uses_spacing_origin_direction() -> None:
    p = _volume().voxel_to_physical(VoxelIndex(2, 4, 1))
    assert (p.x_mm, p.y_mm, p.z_mm) == (11.0, 22.0, 32.0)
    assert p.frame is CoordinateFrame.PATIENT_LPS_MM


def test_voxel_to_physical_requires_complete_affine() -> None:
    with pytest.raises(MissingSpatialMetadataError):
        _volume(direction=None).voxel_to_physical(VoxelIndex(0, 0, 0))


@pytest.mark.parametrize("shape", [(10, 20), (10, 0, 5), (1.5, 2, 3)])
def test_volume_wrong_dimensions_rejected(shape: tuple[object, ...]) -> None:
    with pytest.raises(InvalidSpatialReferenceError):
        VolumeGeometry(shape=shape)  # type: ignore[arg-type]


def test_voxel_index_must_be_integers_and_in_bounds() -> None:
    with pytest.raises(InvalidSpatialReferenceError):
        VoxelIndex(1.5, 0, 0)  # type: ignore[arg-type]
    with pytest.raises(InvalidSpatialReferenceError):
        _volume().require_within(VoxelIndex(10, 0, 0))


def test_frames_cannot_be_swapped() -> None:
    with pytest.raises(CoordinateFrameMismatchError):
        Level0Rect(0, 0, 1, 1, frame=CoordinateFrame.VOLUME_VOXEL_INDEX)
    with pytest.raises(CoordinateFrameMismatchError):
        PhysicalPoint(0, 0, 0, frame=CoordinateFrame.WSI_LEVEL0_PIXEL)


def test_wsi_pixel_geometry_cannot_attach_to_volume_and_vice_versa() -> None:
    with pytest.raises(CoordinateFrameMismatchError):
        validate_geometry_against(RectangleGeometry(Level0Rect(0, 0, 5, 5)), _volume())
    box = VoxelBoxGeometry(VoxelIndex(0, 0, 0), VoxelIndex(2, 2, 2))
    with pytest.raises(CoordinateFrameMismatchError):
        validate_geometry_against(box, _pyramid())
    validate_geometry_against(box, _volume())


def test_evidence_geometry_out_of_bounds_rejected() -> None:
    with pytest.raises(InvalidSpatialReferenceError):
        validate_geometry_against(RectangleGeometry(Level0Rect(990, 0, 20, 5)), _pyramid())
    with pytest.raises(InvalidSpatialReferenceError):
        validate_geometry_against(PointGeometry(-1, 5), _pyramid())
    with pytest.raises(InvalidSpatialReferenceError):
        validate_geometry_against(
            VoxelBoxGeometry(VoxelIndex(0, 0, 0), VoxelIndex(11, 2, 2)), _volume()
        )


def test_non_finite_coordinates_rejected() -> None:
    with pytest.raises(InvalidSpatialReferenceError):
        Level0Rect(float("nan"), 0, 1, 1)
