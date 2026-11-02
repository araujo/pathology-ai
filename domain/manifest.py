"""Versioned, model-independent patch manifests.

A manifest describes a patch extraction completely without its pixels: image identity and
integrity, pipeline, detector, plan, reader, the coordinate convention, every selected patch with
geometry, quality and read provenance, rejected-candidate counts, and how pixels are stored.

Artifact identity is content-addressed and says nothing about durability: `PixelStorage` states
whether pixel bytes exist anywhere beyond the producing process (SPEC-004 §10).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from domain.capabilities import CapabilityCheck
from domain.errors import InvalidManifestError
from domain.geometry import Level0Rect, LevelRect
from domain.hashing import digest
from domain.ids import ArtifactId, ImageId
from domain.image import ChannelOrder
from domain.patching import PatchSpec, ResolvedPatchGeometry
from domain.pipeline import ConfigValue, PipelineRef
from domain.provenance import ComponentVersion
from domain.reading import DataValidity, NoDataSemantics

PATCH_MANIFEST_SCHEMA = "pathology-ai.patch-manifest/1"
COORDINATE_CONVENTION = (
    "wsi_level0_pixel: origin at the top-left of the opened scene's level-zero data rectangle, "
    "x right, y down, rectangles half-open [x, x + width); read-level rectangles map to level "
    "zero by multiplying with the level's downsample"
)
MANIFEST_NOTICE = (
    "patch extraction only: no model was run; candidate-tissue fractions are not tumour "
    "probabilities or clinical predictions"
)


class PixelStorage(StrEnum):
    # Pixel payloads live in the process-local artifact store; nothing is written to disk.
    IN_MEMORY = "in_memory"
    # Only geometry was planned; no pixel was decoded for these patches.
    NOT_MATERIALIZED = "not_materialized"


@dataclass(frozen=True, slots=True)
class PatchRecord:
    patch_id: str
    plan_index: int
    row: int
    col: int
    scene: int
    level0: Level0Rect
    level_rect: LevelRect
    downsample: float
    width_px: int
    height_px: int
    effective_spacing_um: tuple[float, float] | None
    dtype: str
    channel_order: ChannelOrder
    samples_per_pixel: int
    validity_semantics: NoDataSemantics
    tissue_fraction: float
    planned_acquired_fraction: float | None
    read_validity: DataValidity | None
    read_acquired_fraction: float | None
    read_provenance_digest: str | None
    decoder_transformations: tuple[str, ...]
    artifact_id: ArtifactId | None


@dataclass(frozen=True, slots=True)
class PatchManifest:
    image_id: ImageId
    source_integrity: str | None
    scene: int
    pipeline: PipelineRef | None
    reader: ComponentVersion
    backend: ComponentVersion
    detector: ComponentVersion
    detector_parameters: tuple[tuple[str, ConfigValue], ...]
    detector_derived: tuple[tuple[str, ConfigValue], ...]
    mask_digest: str
    planner: ComponentVersion
    extractor: ComponentVersion
    plan_digest: str
    plan_spec: PatchSpec
    resolved: ResolvedPatchGeometry
    counts: tuple[tuple[str, int], ...]
    selection_note: str
    validity_note: str
    pixel_storage: PixelStorage
    patches: tuple[PatchRecord, ...]
    capability_checks: tuple[CapabilityCheck, ...]
    limitations: tuple[str, ...]
    schema_version: str = PATCH_MANIFEST_SCHEMA
    coordinate_convention: str = COORDINATE_CONVENTION
    notice: str = MANIFEST_NOTICE

    def __post_init__(self) -> None:
        if self.schema_version != PATCH_MANIFEST_SCHEMA:
            raise InvalidManifestError(f"unsupported manifest schema {self.schema_version!r}")
        counts = dict(self.counts)
        if counts.get("selected") != len(self.patches):
            raise InvalidManifestError("selected count does not match the patch records")
        if list(self.counts) != sorted(self.counts):
            raise InvalidManifestError("counts must be sorted by key")
        indices = [p.plan_index for p in self.patches]
        if indices != sorted(set(indices)):
            raise InvalidManifestError("patches must be unique and in plan order")
        if len({p.patch_id for p in self.patches}) != len(self.patches):
            raise InvalidManifestError("patch identities must be unique")
        for p in self.patches:
            for f in (p.tissue_fraction, p.planned_acquired_fraction, p.read_acquired_fraction):
                if f is not None and not (math.isfinite(f) and 0.0 <= f <= 1.0):
                    raise InvalidManifestError(f"patch {p.patch_id} has a fraction outside [0, 1]")
            materialized = p.artifact_id is not None
            if materialized != (self.pixel_storage is PixelStorage.IN_MEMORY):
                raise InvalidManifestError(
                    f"patch {p.patch_id}: artifact reference contradicts pixel storage "
                    f"{self.pixel_storage.value}"
                )
            if p.scene != self.scene:
                raise InvalidManifestError(f"patch {p.patch_id} belongs to another scene")

    @property
    def manifest_digest(self) -> str:
        """Digest of the whole manifest (deterministic; contains no timestamps)."""
        return digest(self)

    @property
    def provenance_digest(self) -> str:
        """Digest of what produced the patches (inputs and components, not the patch list)."""
        return digest(
            {
                "image": self.image_id,
                "integrity": self.source_integrity,
                "scene": self.scene,
                "pipeline": self.pipeline,
                "reader": self.reader,
                "backend": self.backend,
                "detector": self.detector,
                "detector_parameters": self.detector_parameters,
                "mask": self.mask_digest,
                "planner": self.planner,
                "extractor": self.extractor,
                "plan": self.plan_digest,
            }
        )
