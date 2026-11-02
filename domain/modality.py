"""Imaging modality versus file format.

A format (container/encoding) never determines the clinical modality: DICOM
can hold whole-slide microscopy, CT or MRI. Modality must be declared by an
authoritative source (acquisition metadata or curated registration).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from domain.errors import AmbiguousModalityError, InvalidIdentifierError

_CODE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")


class SpatialKind(StrEnum):
    """How pixels/voxels are organised in space."""

    PYRAMIDAL_2D = "pyramidal_2d"
    VOLUME_3D = "volume_3d"
    VOLUME_4D = "volume_4d"


@dataclass(frozen=True, slots=True)
class ImageModality:
    """An imaging modality and the spatial kinds it may legitimately use."""

    code: str
    spatial_kinds: frozenset[SpatialKind]

    def __post_init__(self) -> None:
        if not _CODE.fullmatch(self.code):
            raise InvalidIdentifierError(f"invalid modality code {self.code!r}")
        if not self.spatial_kinds:
            raise InvalidIdentifierError(f"modality {self.code!r} declares no spatial kind")

    def allows(self, kind: SpatialKind) -> bool:
        return kind in self.spatial_kinds

    def __str__(self) -> str:
        return self.code


WHOLE_SLIDE_MICROSCOPY = ImageModality("wsi", frozenset({SpatialKind.PYRAMIDAL_2D}))
CT = ImageModality("ct", frozenset({SpatialKind.VOLUME_3D, SpatialKind.VOLUME_4D}))
MRI = ImageModality("mri", frozenset({SpatialKind.VOLUME_3D, SpatialKind.VOLUME_4D}))

KNOWN_MODALITIES: dict[str, ImageModality] = {m.code: m for m in (WHOLE_SLIDE_MICROSCOPY, CT, MRI)}


@dataclass(frozen=True, slots=True)
class ImageFormat:
    """An open, reader-defined container/encoding code (e.g. a vendor WSI format, NIfTI)."""

    code: str

    def __post_init__(self) -> None:
        if not _CODE.fullmatch(self.code):
            raise InvalidIdentifierError(f"invalid image format code {self.code!r}")

    def __str__(self) -> str:
        return self.code


def require_declared_modality(
    image_format: ImageFormat, declared: ImageModality | None
) -> ImageModality:
    """Return the declared modality; a format alone is never sufficient evidence."""
    if declared is None:
        raise AmbiguousModalityError(
            f"modality must be declared explicitly; format {image_format.code!r} "
            "is not sufficient evidence of imaging modality"
        )
    return declared
