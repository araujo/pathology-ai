"""Patch-manifest serialization (`pathology-ai.patch-manifest/1`). Additive: Imaging and
Analysis API v1 are unchanged.

`manifest_to_dict` is canonical: keys are sorted by `canonical_json`, the order of patches is the
plan order, and the document carries its own digests. `manifest_from_dict` rebuilds the domain
value and rejects any document whose digests do not match its content.
"""

from __future__ import annotations

from typing import Any

from domain.capabilities import CapabilityCheck, CheckStatus, ValidityPolicy
from domain.errors import DomainError, InvalidManifestError
from domain.geometry import Level0Rect, LevelRect
from domain.hashing import canonical_json, to_canonical
from domain.ids import ArtifactId, ImageId, PipelineId, Version
from domain.image import ChannelOrder
from domain.manifest import PATCH_MANIFEST_SCHEMA, PatchManifest, PatchRecord, PixelStorage
from domain.patching import (
    EdgePolicy,
    PatchSpec,
    PatchUnits,
    ResolvedPatchGeometry,
    SelectionPolicy,
)
from domain.pipeline import PipelineRef
from domain.provenance import ComponentVersion
from domain.reading import DataValidity, NoDataSemantics


def manifest_to_dict(manifest: PatchManifest) -> dict[str, Any]:
    doc: dict[str, Any] = to_canonical(manifest)
    doc["manifest_digest"] = manifest.manifest_digest
    doc["provenance_digest"] = manifest.provenance_digest
    return doc


def manifest_to_json(manifest: PatchManifest) -> str:
    return canonical_json(manifest_to_dict(manifest))


def _component(d: dict[str, Any]) -> ComponentVersion:
    return ComponentVersion(d["role"], d["name"], d["version"])


def _pairs(items: list[list[Any]]) -> tuple[tuple[str, Any], ...]:
    return tuple((str(k), v) for k, v in items)


def _level0(d: dict[str, Any]) -> Level0Rect:
    return Level0Rect(d["x"], d["y"], d["width"], d["height"])


def _spacing(v: list[float] | None) -> tuple[float, float] | None:
    return None if v is None else (float(v[0]), float(v[1]))


def _record(d: dict[str, Any]) -> PatchRecord:
    lr = d["level_rect"]
    return PatchRecord(
        patch_id=d["patch_id"],
        plan_index=d["plan_index"],
        row=d["row"],
        col=d["col"],
        scene=d["scene"],
        level0=_level0(d["level0"]),
        level_rect=LevelRect(lr["level"], lr["x"], lr["y"], lr["width"], lr["height"]),
        downsample=d["downsample"],
        width_px=d["width_px"],
        height_px=d["height_px"],
        effective_spacing_um=_spacing(d["effective_spacing_um"]),
        dtype=d["dtype"],
        channel_order=ChannelOrder(d["channel_order"]),
        samples_per_pixel=d["samples_per_pixel"],
        validity_semantics=NoDataSemantics(d["validity_semantics"]),
        tissue_fraction=d["tissue_fraction"],
        planned_acquired_fraction=d["planned_acquired_fraction"],
        read_validity=DataValidity(d["read_validity"]) if d["read_validity"] else None,
        read_acquired_fraction=d["read_acquired_fraction"],
        read_provenance_digest=d["read_provenance_digest"],
        decoder_transformations=tuple(d["decoder_transformations"]),
        artifact_id=ArtifactId(d["artifact_id"]["value"]) if d["artifact_id"] else None,
    )


def _spec(d: dict[str, Any]) -> PatchSpec:
    return PatchSpec(
        width=d["width"],
        height=d["height"],
        units=PatchUnits(d["units"]),
        stride_x=d["stride_x"],
        stride_y=d["stride_y"],
        level=d["level"],
        target_spacing_um=d["target_spacing_um"],
        spacing_tolerance=d["spacing_tolerance"],
        grid_origin_x=d["grid_origin_x"],
        grid_origin_y=d["grid_origin_y"],
        edge_policy=EdgePolicy(d["edge_policy"]),
        min_tissue_fraction=d["min_tissue_fraction"],
        min_acquired_fraction=d["min_acquired_fraction"],
        validity_policy=ValidityPolicy(d["validity_policy"]),
        selection=SelectionPolicy(d["selection"]),
        max_patches=d["max_patches"],
        max_candidates=d["max_candidates"],
    )


def _resolved(d: dict[str, Any]) -> ResolvedPatchGeometry:
    return ResolvedPatchGeometry(
        level=d["level"],
        downsample=d["downsample"],
        width_px=d["width_px"],
        height_px=d["height_px"],
        stride_x_px=d["stride_x_px"],
        stride_y_px=d["stride_y_px"],
        effective_spacing_um=_spacing(d["effective_spacing_um"]),
        requested_spacing_um=d["requested_spacing_um"],
        spacing_relative_error=d["spacing_relative_error"],
        resampling=d["resampling"],
    )


def manifest_from_dict(doc: dict[str, Any]) -> PatchManifest:
    """Rebuild and verify a manifest. Any structural or digest problem is INVALID_MANIFEST."""
    try:
        if doc.get("schema_version") != PATCH_MANIFEST_SCHEMA:
            raise InvalidManifestError(f"unsupported schema {doc.get('schema_version')!r}")
        pipe = doc["pipeline"]
        manifest = PatchManifest(
            image_id=ImageId(doc["image_id"]["value"]),
            source_integrity=doc["source_integrity"],
            scene=doc["scene"],
            pipeline=PipelineRef(
                PipelineId(pipe["pipeline_id"]["value"]), Version(pipe["version"]["value"])
            )
            if pipe
            else None,
            reader=_component(doc["reader"]),
            backend=_component(doc["backend"]),
            detector=_component(doc["detector"]),
            detector_parameters=_pairs(doc["detector_parameters"]),
            detector_derived=_pairs(doc["detector_derived"]),
            mask_digest=doc["mask_digest"],
            planner=_component(doc["planner"]),
            extractor=_component(doc["extractor"]),
            plan_digest=doc["plan_digest"],
            plan_spec=_spec(doc["plan_spec"]),
            resolved=_resolved(doc["resolved"]),
            counts=tuple((str(k), int(v)) for k, v in doc["counts"]),
            selection_note=doc["selection_note"],
            validity_note=doc["validity_note"],
            pixel_storage=PixelStorage(doc["pixel_storage"]),
            patches=tuple(_record(p) for p in doc["patches"]),
            capability_checks=tuple(
                CapabilityCheck(c["requirement"], CheckStatus(c["status"]), c["detail"])
                for c in doc["capability_checks"]
            ),
            limitations=tuple(doc["limitations"]),
            schema_version=doc["schema_version"],
            coordinate_convention=doc["coordinate_convention"],
            notice=doc["notice"],
        )
    except InvalidManifestError:
        raise
    except (KeyError, TypeError, ValueError, IndexError, DomainError) as exc:
        raise InvalidManifestError(
            f"malformed patch manifest: {type(exc).__name__}: {exc}"
        ) from exc
    for key, actual in (
        ("manifest_digest", manifest.manifest_digest),
        ("provenance_digest", manifest.provenance_digest),
    ):
        if doc.get(key) != actual:
            raise InvalidManifestError(f"{key} does not match the manifest content")
    return manifest
