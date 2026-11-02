"""Patch specifications, resolution selection, deterministic grids, decisions and plans.

Planning decides which patches exist; extraction (a separate operation) reads their pixels.

Coordinates:

* Every candidate is defined on the pixel grid of one stored pyramid level (the read level) and
  carries its canonical level-zero rectangle (`x * downsample`, which may be non-integer).
* Grids never pad: a candidate always lies inside the read level, hence inside level zero.
* Native or legacy frames never appear here (see `domain.frames`).

Resolution:

* Pixel-unit patches are read at an explicitly chosen stored level, or at the stored level that
  meets `target_spacing_um` within `spacing_tolerance`.
* Micrometre-unit patches need physical spacing; their pixel size is derived from the effective
  spacing of the chosen level.
* No resampling is performed or implied: a request no stored level meets is rejected.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from domain.capabilities import CapabilityCheck, ValidityPolicy
from domain.errors import (
    InvalidSpatialReferenceError,
    MissingSpatialMetadataError,
    PatchBudgetExceededError,
    UnsupportedResolutionError,
)
from domain.geometry import Level0Rect, LevelRect, PyramidalGeometry
from domain.hashing import digest
from domain.ids import ImageId
from domain.provenance import ComponentVersion


class PatchUnits(StrEnum):
    PIXELS = "pixels"
    MICROMETERS = "micrometers"


class EdgePolicy(StrEnum):
    # Only full patches starting on the regular grid; a remainder strip is left uncovered.
    INSIDE_ONLY = "inside_only"
    # Additionally place one patch flush with the right/bottom edge so the remainder is covered.
    # These edge patches overlap their neighbour; nothing is ever padded.
    ALIGN_TO_EDGE = "align_to_edge"


class SelectionPolicy(StrEnum):
    """What to do when more patches qualify than `max_patches` allows."""

    FAIL_CLOSED = "fail_closed"
    ROW_MAJOR_FIRST = "row_major_first"
    UNIFORM_STRIDE = "uniform_stride"


class PatchStatus(StrEnum):
    SELECTED = "selected"
    REJECTED = "rejected"
    NOT_SELECTED_BUDGET = "not_selected_budget"


class RejectionReason(StrEnum):
    INSUFFICIENT_ACQUIRED_DATA = "insufficient_acquired_data"
    INSUFFICIENT_TISSUE = "insufficient_tissue"


def _positive(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise InvalidSpatialReferenceError(f"{name} must be a finite number")
    if value <= 0:
        raise InvalidSpatialReferenceError(f"{name} must be positive")


def _fraction(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0.0 <= value <= 1.0:
        raise InvalidSpatialReferenceError(f"{name} must lie within [0, 1]")


@dataclass(frozen=True, slots=True)
class PatchSpec:
    """Explicit patch geometry, resolution, filtering and budget."""

    width: float
    height: float
    units: PatchUnits = PatchUnits.PIXELS
    stride_x: float | None = None  # same units as width; None = width (no overlap)
    stride_y: float | None = None
    level: int | None = None
    target_spacing_um: float | None = None
    spacing_tolerance: float = 0.05  # relative
    grid_origin_x: int = 0  # read-level pixels
    grid_origin_y: int = 0
    edge_policy: EdgePolicy = EdgePolicy.INSIDE_ONLY
    min_tissue_fraction: float = 0.5
    min_acquired_fraction: float = 0.9
    validity_policy: ValidityPolicy = ValidityPolicy.REQUIRE_REPORTED
    selection: SelectionPolicy = SelectionPolicy.FAIL_CLOSED
    max_patches: int = 512
    max_candidates: int = 250_000

    def __post_init__(self) -> None:
        _positive("patch width", self.width)
        _positive("patch height", self.height)
        for name in ("stride_x", "stride_y", "target_spacing_um"):
            value = getattr(self, name)
            if value is not None:
                _positive(name, value)
        _positive("spacing_tolerance", self.spacing_tolerance)
        if self.spacing_tolerance >= 1:
            raise InvalidSpatialReferenceError("spacing_tolerance is relative and must be < 1")
        _fraction("min_tissue_fraction", self.min_tissue_fraction)
        _fraction("min_acquired_fraction", self.min_acquired_fraction)
        for name in ("grid_origin_x", "grid_origin_y"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise InvalidSpatialReferenceError(f"{name} must be a non-negative integer")
        for name in ("max_patches", "max_candidates"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise InvalidSpatialReferenceError(f"{name} must be a positive integer")
        if self.level is not None and (not isinstance(self.level, int) or self.level < 0):
            raise InvalidSpatialReferenceError("level must be a non-negative integer")
        if (self.level is None) == (self.target_spacing_um is None):
            raise InvalidSpatialReferenceError(
                "choose the read level explicitly: set exactly one of level or target_spacing_um"
            )

    @property
    def needs_physical_spacing(self) -> bool:
        return self.units is PatchUnits.MICROMETERS or self.target_spacing_um is not None


@dataclass(frozen=True, slots=True)
class ResolvedPatchGeometry:
    """Pixel geometry on one stored level, with the effective spacing that results."""

    level: int
    downsample: float
    width_px: int
    height_px: int
    stride_x_px: int
    stride_y_px: int
    effective_spacing_um: tuple[float, float] | None
    requested_spacing_um: float | None
    spacing_relative_error: float | None
    resampling: str = "none"


def resolve_patch_geometry(spec: PatchSpec, geometry: PyramidalGeometry) -> ResolvedPatchGeometry:
    spacing = geometry.pixel_spacing
    if spec.needs_physical_spacing and spacing is None:
        raise MissingSpatialMetadataError(
            "physical patch units or a target spacing need the image's pixel spacing, which the "
            "reader does not report; use pixel units with an explicit level instead"
        )
    if spec.level is not None:
        level = geometry.level(spec.level)
        error: float | None = None
    else:
        assert spec.target_spacing_um is not None and spacing is not None
        target = spec.target_spacing_um
        scored = sorted(
            (
                max(
                    abs(spacing.x_um * lv.downsample - target),
                    abs(spacing.y_um * lv.downsample - target),
                )
                / target,
                lv.index,
            )
            for lv in geometry.levels
        )
        error, index = scored[0]
        if error > spec.spacing_tolerance:
            available = ", ".join(
                f"level {lv.index}: {spacing.x_um * lv.downsample:.4g} µm/px"
                for lv in geometry.levels
            )
            raise UnsupportedResolutionError(
                f"no stored level within {spec.spacing_tolerance:.0%} of {target} µm/px "
                f"(available: {available}); resampling is not performed"
            )
        level = geometry.level(index)
    effective = (
        (spacing.x_um * level.downsample, spacing.y_um * level.downsample) if spacing else None
    )

    def px(value: float, axis: int) -> int:
        if spec.units is PatchUnits.PIXELS:
            if not float(value).is_integer():
                raise InvalidSpatialReferenceError("pixel-unit patch sizes must be integers")
            return int(value)
        assert effective is not None
        n = round(value / effective[axis])
        if n < 1:
            raise InvalidSpatialReferenceError(f"{value} µm is smaller than one pixel")
        return n

    w, h = px(spec.width, 0), px(spec.height, 1)
    sx = px(spec.stride_x, 0) if spec.stride_x is not None else w
    sy = px(spec.stride_y, 1) if spec.stride_y is not None else h
    if w > level.width or h > level.height:
        raise InvalidSpatialReferenceError(
            f"patch {w}x{h} does not fit level {level.index} ({level.width}x{level.height})"
        )
    return ResolvedPatchGeometry(
        level=level.index,
        downsample=level.downsample,
        width_px=w,
        height_px=h,
        stride_x_px=sx,
        stride_y_px=sy,
        effective_spacing_um=effective,
        requested_spacing_um=spec.target_spacing_um,
        spacing_relative_error=error,
    )


@dataclass(frozen=True, slots=True, order=True)
class PatchCandidate:
    index: int  # row-major position in the grid
    row: int
    col: int
    level_rect: LevelRect
    level0: Level0Rect


def _axis_positions(
    origin: int, size: int, stride: int, extent: int, edge: EdgePolicy
) -> list[int]:
    positions = list(range(origin, extent - size + 1, stride))
    if edge is EdgePolicy.ALIGN_TO_EDGE and extent - size >= 0:
        flush = extent - size
        if not positions or positions[-1] < flush:
            positions.append(flush)
    return positions


def grid_size(spec: PatchSpec, resolved: ResolvedPatchGeometry, geometry: PyramidalGeometry) -> int:
    lv = geometry.level(resolved.level)
    nx = len(
        _axis_positions(
            spec.grid_origin_x, resolved.width_px, resolved.stride_x_px, lv.width, spec.edge_policy
        )
    )
    ny = len(
        _axis_positions(
            spec.grid_origin_y,
            resolved.height_px,
            resolved.stride_y_px,
            lv.height,
            spec.edge_policy,
        )
    )
    return nx * ny


def plan_candidates(
    spec: PatchSpec, resolved: ResolvedPatchGeometry, geometry: PyramidalGeometry
) -> tuple[PatchCandidate, ...]:
    """Row-major (y, then x) candidates; bounded by `spec.max_candidates` before materialising."""
    n = grid_size(spec, resolved, geometry)
    if n > spec.max_candidates:
        raise PatchBudgetExceededError(
            f"the grid has {n} candidates, above max_candidates={spec.max_candidates}"
        )
    lv = geometry.level(resolved.level)
    xs = _axis_positions(
        spec.grid_origin_x, resolved.width_px, resolved.stride_x_px, lv.width, spec.edge_policy
    )
    ys = _axis_positions(
        spec.grid_origin_y, resolved.height_px, resolved.stride_y_px, lv.height, spec.edge_policy
    )
    out: list[PatchCandidate] = []
    for r, y in enumerate(ys):
        for c, x in enumerate(xs):
            rect = LevelRect(lv.index, x, y, resolved.width_px, resolved.height_px)
            out.append(PatchCandidate(len(out), r, c, rect, geometry.to_level0(rect)))
    return tuple(out)


@dataclass(frozen=True, slots=True)
class PatchDecision:
    candidate: PatchCandidate
    tissue_fraction: float
    acquired_fraction: float | None  # None only when validity is unreported (policy recorded)
    status: PatchStatus
    reasons: tuple[RejectionReason, ...] = ()

    def __post_init__(self) -> None:
        _fraction("tissue_fraction", self.tissue_fraction)
        if self.acquired_fraction is not None:
            _fraction("acquired_fraction", self.acquired_fraction)
        if (self.status is PatchStatus.REJECTED) != bool(self.reasons):
            raise InvalidSpatialReferenceError("exactly rejected decisions carry reasons")


def decide(
    candidate: PatchCandidate, tissue: float, acquired: float | None, spec: PatchSpec
) -> PatchDecision:
    """Tissue and acquisition are checked independently; both reasons are recorded."""
    reasons: list[RejectionReason] = []
    if acquired is not None and acquired < spec.min_acquired_fraction:
        reasons.append(RejectionReason.INSUFFICIENT_ACQUIRED_DATA)
    if tissue < spec.min_tissue_fraction:
        reasons.append(RejectionReason.INSUFFICIENT_TISSUE)
    status = PatchStatus.REJECTED if reasons else PatchStatus.SELECTED
    return PatchDecision(candidate, tissue, acquired, status, tuple(reasons))


def apply_selection(
    decisions: tuple[PatchDecision, ...], spec: PatchSpec
) -> tuple[tuple[PatchDecision, ...], str]:
    """Enforce `max_patches` with the declared policy; never truncate silently."""
    eligible = [d.candidate.index for d in decisions if d.status is PatchStatus.SELECTED]
    n, cap = len(eligible), spec.max_patches
    if n <= cap:
        return decisions, f"{n} eligible <= max_patches {cap}; all selected"
    if spec.selection is SelectionPolicy.FAIL_CLOSED:
        raise PatchBudgetExceededError(
            f"{n} patches qualify, above max_patches={cap}; raise the budget or choose an "
            "explicit selection policy (row_major_first, uniform_stride)"
        )
    if spec.selection is SelectionPolicy.ROW_MAJOR_FIRST:
        keep = set(eligible[:cap])
    else:  # UNIFORM_STRIDE: deterministic, evenly spaced over the eligible list
        keep = {eligible[(i * n) // cap] for i in range(cap)}
    out = tuple(
        d
        if d.status is not PatchStatus.SELECTED or d.candidate.index in keep
        else PatchDecision(
            d.candidate, d.tissue_fraction, d.acquired_fraction, PatchStatus.NOT_SELECTED_BUDGET
        )
        for d in decisions
    )
    return out, f"{n} eligible > max_patches {cap}; {spec.selection.value} kept {cap}"


@dataclass(frozen=True, slots=True)
class PatchPlan:
    """Every candidate with its decision. Deterministic for fixed image, mask and spec."""

    image_id: ImageId
    source_integrity: str | None
    scene: int
    planner: ComponentVersion
    spec: PatchSpec
    resolved: ResolvedPatchGeometry
    mask_digest: str
    mask_grid_cell: tuple[float, float]
    decisions: tuple[PatchDecision, ...]
    selection_note: str
    capability_checks: tuple[CapabilityCheck, ...]
    validity_note: str

    @property
    def selected(self) -> tuple[PatchDecision, ...]:
        return tuple(d for d in self.decisions if d.status is PatchStatus.SELECTED)

    def counts(self) -> dict[str, int]:
        out = {
            "candidates": len(self.decisions),
            "selected": 0,
            "rejected": 0,
            "not_selected_budget": 0,
        }
        for reason in RejectionReason:
            out[f"rejected.{reason.value}"] = 0
        for d in self.decisions:
            out[d.status.value] += 1
            for r in d.reasons:
                out[f"rejected.{r.value}"] += 1
        return out

    @property
    def plan_digest(self) -> str:
        return digest(self)
