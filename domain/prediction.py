"""Typed prediction outputs and spatial evidence.

Outputs are a discriminated union rather than a single classification schema.
Evidence carries its coordinate frame (via its geometry) and an explicit
semantics field; attention or saliency is never representable as a
localised disease probability unless a validation record backs that
interpretation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from domain.errors import EvidenceSemanticsError, InvalidSpatialReferenceError
from domain.geometry import (
    CoordinateFrame,
    EvidenceGeometry,
    Geometry2D,
    VolumeGeometry,
    VoxelBoxGeometry,
    VoxelIndex,
    validate_geometry_against,
)
from domain.ids import (
    ArtifactId,
    EvidenceId,
    ImageId,
    PredictionId,
    RunId,
    ValidationRecordId,
)
from domain.image import ImageAsset
from domain.model import ModelRef
from domain.task import OutputKind, TaskRef
from domain.validation import CapabilityStatus, UsageMode


class EvidenceSemantics(StrEnum):
    ATTENTION_WEIGHT = "attention_weight"
    SALIENCY = "saliency"
    LOCALIZED_PROBABILITY = "localized_probability"
    DETECTION_CONFIDENCE = "detection_confidence"
    SEGMENTATION_MEMBERSHIP = "segmentation_membership"


_NON_PROBABILISTIC = frozenset({EvidenceSemantics.ATTENTION_WEIGHT, EvidenceSemantics.SALIENCY})


@dataclass(frozen=True, slots=True)
class SpatialEvidence:
    evidence_id: EvidenceId
    image_id: ImageId
    geometry: EvidenceGeometry
    semantics: EvidenceSemantics
    producer: ModelRef
    value: float | None = None
    value_is_probability: bool = False
    rank: int | None = None
    localization_validation: ValidationRecordId | None = None

    def __post_init__(self) -> None:
        if self.value is not None and not math.isfinite(self.value):
            raise EvidenceSemanticsError("evidence value must be finite")
        if self.semantics in _NON_PROBABILISTIC and self.value_is_probability:
            raise EvidenceSemanticsError(
                f"{self.semantics.value} evidence cannot be presented as a probability"
            )
        if self.semantics is EvidenceSemantics.LOCALIZED_PROBABILITY:
            if self.localization_validation is None:
                raise EvidenceSemanticsError(
                    "localized probability evidence requires a localization validation record"
                )
            if self.value is None or not 0.0 <= self.value <= 1.0:
                raise EvidenceSemanticsError("localized probability must lie in [0, 1]")
        if self.value_is_probability and self.value is not None and not 0 <= self.value <= 1:
            raise EvidenceSemanticsError("a probability value must lie in [0, 1]")

    @property
    def frame(self) -> CoordinateFrame:
        return self.geometry.frame


@dataclass(frozen=True, slots=True)
class ImageClassification:
    label: str
    scores: tuple[tuple[str, float], ...]
    scores_are_calibrated_probabilities: bool = False
    kind: OutputKind = OutputKind.IMAGE_CLASSIFICATION

    def __post_init__(self) -> None:
        names = [n for n, _ in self.scores]
        if self.label not in names:
            raise ValueError("predicted label must be one of the scored classes")
        if len(set(names)) != len(names):
            raise ValueError("duplicate class in scores")


@dataclass(frozen=True, slots=True)
class ContinuousScore:
    name: str
    value: float
    unit: str | None = None
    kind: OutputKind = OutputKind.CONTINUOUS_SCORE


@dataclass(frozen=True, slots=True)
class RegionPrediction:
    geometry: Geometry2D
    label: str
    score: float | None = None
    kind: OutputKind = OutputKind.REGION_PREDICTION


@dataclass(frozen=True, slots=True)
class Detection:
    geometry: Geometry2D
    label: str
    confidence: float
    kind: OutputKind = OutputKind.DETECTION


@dataclass(frozen=True, slots=True)
class SegmentationReference:
    """2D mask stored as an artifact; `level` is the pyramid level of the mask grid."""

    mask_artifact: ArtifactId
    label_map: tuple[tuple[int, str], ...]
    level: int
    downsample: float
    kind: OutputKind = OutputKind.SEGMENTATION


@dataclass(frozen=True, slots=True)
class VolumetricSegmentationReference:
    """3D label volume aligned voxel-for-voxel with `geometry`."""

    mask_artifact: ArtifactId
    label_map: tuple[tuple[int, str], ...]
    geometry: VolumeGeometry
    kind: OutputKind = OutputKind.VOLUMETRIC_SEGMENTATION


PredictionOutput = (
    ImageClassification
    | ContinuousScore
    | RegionPrediction
    | Detection
    | SegmentationReference
    | VolumetricSegmentationReference
)


@dataclass(frozen=True, slots=True)
class PredictionResult:
    prediction_id: PredictionId
    image_id: ImageId
    task: TaskRef
    models: tuple[ModelRef, ...]
    outputs: tuple[PredictionOutput, ...]
    evidence: tuple[SpatialEvidence, ...]
    run_id: RunId
    capability_status: CapabilityStatus
    usage_mode: UsageMode

    def __post_init__(self) -> None:
        if not self.outputs:
            raise ValueError("a prediction result needs at least one output")
        if self.capability_status is CapabilityStatus.INCOMPATIBLE:
            raise ValueError("an INCOMPATIBLE combination cannot produce a prediction")
        if (self.capability_status is CapabilityStatus.VALIDATED) != (
            self.usage_mode is UsageMode.VALIDATED_SCOPE
        ):
            raise ValueError("usage mode must agree with capability status")
        for ev in self.evidence:
            if ev.image_id != self.image_id:
                raise InvalidSpatialReferenceError("evidence refers to a different image")

    def validate_against(self, image: ImageAsset) -> None:
        """Fail closed if any evidence/output geometry does not fit the image's reference."""
        if image.image_id != self.image_id:
            raise InvalidSpatialReferenceError("prediction belongs to a different image")
        for ev in self.evidence:
            validate_geometry_against(ev.geometry, image.spatial_reference)
        for out in self.outputs:
            if isinstance(out, RegionPrediction | Detection):
                validate_geometry_against(out.geometry, image.spatial_reference)
            if isinstance(out, VolumetricSegmentationReference):
                box = VoxelBoxGeometry(VoxelIndex(0, 0, 0), VoxelIndex(*out.geometry.shape))
                validate_geometry_against(box, image.spatial_reference)
