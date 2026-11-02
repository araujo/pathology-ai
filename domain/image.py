"""Image assets: identity, modality, format, geometry and optional clinical context.

Labels are deliberately *not* part of an image asset. Tissue/disease context is
optional, so an image of unknown tissue can be ingested without assigning an
unsupported disease label.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from domain.errors import InvalidImageAssetError, UnsupportedModalityError
from domain.geometry import SpatialReference
from domain.ids import ImageId
from domain.modality import ImageFormat, ImageModality


class ChannelOrder(StrEnum):
    RGB = "rgb"
    BGR = "bgr"
    RGBA = "rgba"
    GRAYSCALE = "grayscale"
    MULTICHANNEL = "multichannel"


@dataclass(frozen=True, slots=True)
class ClinicalContext:
    """Optional, curated context. `None` means unknown, never a guessed default."""

    tissue: str | None = None
    disease: str | None = None

    @property
    def is_unknown(self) -> bool:
        return self.tissue is None and self.disease is None


UNKNOWN_CONTEXT = ClinicalContext()


@dataclass(frozen=True, slots=True)
class ImageAsset:
    image_id: ImageId
    modality: ImageModality
    image_format: ImageFormat
    spatial_reference: SpatialReference
    channel_order: ChannelOrder | None = None
    source_checksum: str | None = None
    context: ClinicalContext = field(default=UNKNOWN_CONTEXT)

    def __post_init__(self) -> None:
        if self.modality is None:
            raise InvalidImageAssetError("image modality must be declared explicitly")
        kind = self.spatial_reference.spatial_kind
        if not self.modality.allows(kind):
            raise UnsupportedModalityError(
                f"modality {self.modality.code!r} does not use spatial kind {kind.value!r}"
            )
