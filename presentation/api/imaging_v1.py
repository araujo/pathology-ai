"""Imaging API v1 serializers (domain -> transport dict). No server, no DPA wire format."""

from __future__ import annotations

from typing import Any

from domain.geometry import CoordinateFrame, PyramidalGeometry, VolumeGeometry
from domain.hashing import to_canonical
from domain.image import ImageAsset
from domain.reading import ImageDescription, PixelRegion, RegionReadRequest
from ports import ReaderCapabilities

SCHEMA_VERSION = "pathology-ai.imaging/1.0"


def geometry_to_dict(geometry: PyramidalGeometry | VolumeGeometry) -> dict[str, Any]:
    if isinstance(geometry, PyramidalGeometry):
        spacing = geometry.pixel_spacing
        return {
            "type": "pyramidal_2d",
            "frame": "wsi_level0_pixel",
            "width": geometry.width,
            "height": geometry.height,
            "levels": [
                {
                    "index": lv.index,
                    "width": lv.width,
                    "height": lv.height,
                    "downsample": lv.downsample,
                }
                for lv in geometry.levels
            ],
            "pixel_spacing_um": {"x": spacing.x_um, "y": spacing.y_um} if spacing else None,
        }
    return {
        "type": geometry.spatial_kind.value,
        "frame": "volume_voxel_index",
        "physical_frame": "patient_lps_mm",
        "shape": list(geometry.shape),
        "spacing_mm": list(geometry.spacing_mm) if geometry.spacing_mm else None,
        "origin_mm": list(geometry.origin_mm) if geometry.origin_mm else None,
        "direction": list(geometry.direction) if geometry.direction else None,
        "time_points": geometry.time_points,
        "series_uid": geometry.series_uid,
    }


def image_descriptor(image: ImageAsset, reader: ReaderCapabilities | None = None) -> dict[str, Any]:
    """`reader=None` means no reader is deployed: every access flag is false."""
    return {
        "schema_version": SCHEMA_VERSION,
        "image_id": image.image_id.value,
        "modality": {
            "code": image.modality.code,
            "spatial_kind": image.spatial_reference.spatial_kind.value,
        },
        "format": image.image_format.code,
        "channel_order": image.channel_order.value if image.channel_order else None,
        "source_checksum": image.source_checksum,
        "context": {"tissue": image.context.tissue, "disease": image.context.disease},
        "geometry": geometry_to_dict(image.spatial_reference),
        "access": {
            "regions": bool(reader and reader.supports_regions),
            "thumbnail": bool(reader and reader.supports_thumbnails),
            "volume": bool(reader and reader.supports_volumes),
            "slices": bool(reader and reader.supports_volumes),
        },
    }


def region_request_from_dict(doc: dict[str, Any]) -> RegionReadRequest:
    """Map an Imaging API v1 region request (already schema-valid) onto the domain request.

    v1 has no scene, plane or channel fields, so those stay `None` ("the only one"); a reader
    rejects the request when the image has several. `encoding` is a transport concern.
    """
    return RegionReadRequest(
        level=doc["level"],
        x=doc["x"],
        y=doc["y"],
        width=doc["width"],
        height=doc["height"],
        frame=CoordinateFrame(doc["frame"]),
    )


def reader_diagnostics(description: ImageDescription) -> dict[str, Any]:
    """Reader facts that Imaging API v1 cannot carry. NOT a v1 contract (see SPEC-003 report)."""
    offset = description.native_offset
    return {
        "reader": to_canonical(description.reader),
        "backend": to_canonical(description.backend),
        "scene": description.scene,
        "scene_count": description.scene_count,
        "native_origin": [offset.origin_x, offset.origin_y] if offset else None,
        "pixel_type": description.pixel_type,
        "dtype": description.dtype,
        "samples_per_pixel": description.samples_per_pixel,
        "planes": {
            "c": description.channel_count,
            "z": description.z_planes,
            "t": description.time_points,
        },
        "spacing_source": description.spacing_source,
        "level_source": description.level_source,
        "no_data": description.no_data.value,
        "decoder_transformations": list(description.decoder_transformations),
        "acquisition": dict(description.acquisition),
        "missing_metadata": list(description.missing_metadata),
        "integrity": to_canonical(description.integrity),
    }


def region_summary(region: PixelRegion) -> dict[str, Any]:
    """Provenance of a decoded region (no pixel payload). NOT a v1 contract."""
    out: dict[str, Any] = to_canonical(region.provenance)
    return out
