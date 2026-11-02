"""Capability requirements of preprocessing operations, checked against a described image.

An operation declares what it needs (channel representation, dtype, region reads, physical
spacing, acquisition-validity semantics, a single plane). Before any pixel is processed the
requirements are evaluated against the reader's `ImageDescription`. Every requirement ends in an
explicit verdict; unsupported or unknown capabilities are never assumed (SPEC-004 AC-018).

Requirements name capabilities, never formats or backends.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from domain.errors import PreprocessingCapabilityError
from domain.image import ChannelOrder
from domain.reading import ImageDescription, NoDataSemantics


class ValidityPolicy(StrEnum):
    """How an operation treats images whose reader cannot report acquisition validity."""

    # Fail closed: a reader with `NoDataSemantics.NOT_REPORTED` is unsupported.
    REQUIRE_REPORTED = "require_reported"
    # Explicit, recorded decision: treat every in-bounds pixel as acquired.
    ASSUME_ACQUIRED = "assume_acquired"


class CheckStatus(StrEnum):
    SATISFIED = "satisfied"
    # Not provided by the reader, but resolved by an explicit, recorded policy.
    DECIDED_BY_POLICY = "decided_by_policy"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True, order=True)
class CapabilityCheck:
    requirement: str
    status: CheckStatus
    detail: str

    @property
    def blocking(self) -> bool:
        return self.status in (CheckStatus.UNSUPPORTED, CheckStatus.UNKNOWN)


@dataclass(frozen=True, slots=True)
class PreprocessingRequirements:
    """What an operation needs from the opened image. Empty sets mean "any"."""

    channel_orders: frozenset[ChannelOrder] = frozenset()
    dtypes: frozenset[str] = frozenset()
    needs_regions: bool = True
    needs_physical_spacing: bool = False
    validity_policy: ValidityPolicy | None = None
    single_plane: bool = True

    def evaluate(
        self, description: ImageDescription, *, supports_regions: bool
    ) -> tuple[CapabilityCheck, ...]:
        d = description
        S, U = CheckStatus.SATISFIED, CheckStatus.UNSUPPORTED
        checks = [
            CapabilityCheck(
                "spatial_kind", S, f"{d.geometry.spatial_kind.value}, level-zero pixel frame"
            ),
            CapabilityCheck(
                "scene", S, f"scene {d.scene} of {d.scene_count} selected when the image was opened"
            ),
        ]
        if self.single_plane:
            planes = (d.channel_count, d.z_planes, d.time_points)
            ok = d.z_planes == 1 and d.time_points == 1
            checks.append(
                CapabilityCheck(
                    "single_plane", S if ok else U, f"(c, z, t) = {planes}; one Z/T plane needed"
                )
            )
        if self.channel_orders:
            accepted = ", ".join(sorted(c.value for c in self.channel_orders))
            ok = d.channel_order in self.channel_orders
            checks.append(
                CapabilityCheck(
                    "channel_representation",
                    S if ok else U,
                    f"reader returns {d.channel_order.value}; accepted: {accepted}",
                )
            )
        if self.dtypes:
            ok = d.dtype in self.dtypes
            checks.append(
                CapabilityCheck(
                    "dtype",
                    S if ok else U,
                    f"reader returns {d.dtype}; accepted: {', '.join(sorted(self.dtypes))}",
                )
            )
        if self.needs_regions:
            checks.append(
                CapabilityCheck(
                    "region_reads",
                    S if supports_regions else U,
                    "bounded region reads " + ("supported" if supports_regions else "unavailable"),
                )
            )
        if self.needs_physical_spacing:
            spacing = d.geometry.pixel_spacing
            checks.append(
                CapabilityCheck(
                    "physical_spacing",
                    S if spacing else U,
                    f"{spacing.x_um} x {spacing.y_um} µm/px ({d.spacing_source})"
                    if spacing
                    else "the reader reports no physical spacing; it is never assumed",
                )
            )
        if self.validity_policy is not None:
            checks.append(_validity_check(d.no_data, self.validity_policy))
        return tuple(checks)


def _validity_check(semantics: NoDataSemantics, policy: ValidityPolicy) -> CapabilityCheck:
    if semantics is not NoDataSemantics.NOT_REPORTED:
        return CapabilityCheck(
            "acquisition_validity", CheckStatus.SATISFIED, f"reported as {semantics.value}"
        )
    if policy is ValidityPolicy.ASSUME_ACQUIRED:
        return CapabilityCheck(
            "acquisition_validity",
            CheckStatus.DECIDED_BY_POLICY,
            "the reader cannot report acquisition validity; every in-bounds pixel is treated as "
            "acquired by explicit policy assume_acquired",
        )
    return CapabilityCheck(
        "acquisition_validity",
        CheckStatus.UNSUPPORTED,
        "the reader cannot report acquisition validity and the policy is require_reported; "
        "choose validity_policy=assume_acquired explicitly to proceed",
    )


def require_capabilities(checks: Iterable[CapabilityCheck], *, context: str) -> None:
    """Raise one typed error naming every blocking requirement."""
    blocking = [c for c in checks if c.blocking]
    if blocking:
        raise PreprocessingCapabilityError(
            f"{context}: "
            + "; ".join(f"{c.requirement} {c.status.value}: {c.detail}" for c in blocking),
            tuple(c.requirement for c in blocking),
        )
