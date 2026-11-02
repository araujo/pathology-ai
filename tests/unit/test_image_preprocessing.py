from __future__ import annotations

import dataclasses

import pytest

from domain.dataset import Label, require_labels
from domain.errors import (
    AmbiguousModalityError,
    ChannelOrderMismatchError,
    MissingLabelError,
    MissingPreprocessingError,
    UnsupportedModalityError,
)
from domain.geometry import VolumeGeometry
from domain.ids import ImageId
from domain.image import UNKNOWN_CONTEXT, ChannelOrder, ImageAsset
from domain.modality import (
    CT,
    MRI,
    WHOLE_SLIDE_MICROSCOPY,
    ImageFormat,
    require_declared_modality,
)
from domain.preprocessing import TransformStep
from plugins.synthetic import catalog
from plugins.synthetic.imaging import synthetic_slide


def test_dicom_format_never_implies_modality() -> None:
    dicom = ImageFormat("dicom")
    with pytest.raises(AmbiguousModalityError):
        require_declared_modality(dicom, None)
    for modality in (WHOLE_SLIDE_MICROSCOPY, CT, MRI):
        assert require_declared_modality(dicom, modality) is modality


def test_wsi_modality_rejects_volume_geometry() -> None:
    with pytest.raises(UnsupportedModalityError):
        ImageAsset(
            ImageId("x-1"),
            WHOLE_SLIDE_MICROSCOPY,
            ImageFormat("dicom"),
            VolumeGeometry(shape=(4, 4, 4)),
        )


def test_ct_modality_rejects_pyramidal_geometry() -> None:
    slide = synthetic_slide("x-2")
    with pytest.raises(UnsupportedModalityError):
        dataclasses.replace(slide, modality=CT)


def test_unknown_tissue_image_is_ingestible_without_labels() -> None:
    img = synthetic_slide("unknown-tissue-0001")
    assert img.context is UNKNOWN_CONTEXT and img.context.is_unknown
    assert not hasattr(img, "label")


def test_supervised_use_requires_labels() -> None:
    task = catalog.LUNG_TASK.ref
    ids = (ImageId("a-1"), ImageId("a-2"))
    labels = {ids[0]: Label(ids[0], task, "synthetic_negative")}
    with pytest.raises(MissingLabelError, match="a-2"):
        require_labels(ids, task, labels)


def test_complete_contracts_have_no_missing_fields() -> None:
    for c in catalog.ALL_CONTRACTS:
        c.require_complete()


@pytest.mark.parametrize(
    "field",
    [
        "model_channel_order",
        "interpolation",
        "target_pixel_spacing_um",
        "stain_normalization",
        "patch_stride_px",
    ],
)
def test_missing_2d_transform_fields_fail_closed(field: str) -> None:
    broken = dataclasses.replace(catalog.PATCH_CONTRACT, **{field: None})  # type: ignore[arg-type]
    with pytest.raises(MissingPreprocessingError, match=field):
        broken.require_complete()


@pytest.mark.parametrize(
    "field", ["target_voxel_spacing_mm", "orientation", "intensity_normalization"]
)
def test_missing_volume_transform_fields_fail_closed(field: str) -> None:
    broken = dataclasses.replace(catalog.VOLUME_CONTRACT, **{field: None})  # type: ignore[arg-type]
    with pytest.raises(MissingPreprocessingError, match=field):
        broken.require_complete()


def test_bgr_rgb_never_silently_swapped() -> None:
    c = catalog.PATCH_CONTRACT
    assert c.resolve_channel_conversion(ChannelOrder.RGB) is None
    with pytest.raises(ChannelOrderMismatchError):
        c.resolve_channel_conversion(ChannelOrder.BGR)
    with pytest.raises(ChannelOrderMismatchError):
        c.resolve_channel_conversion(None)
    step = TransformStep("channel_reorder", (("from", "bgr"), ("to", "rgb")))
    explicit = dataclasses.replace(c, transforms=(step,))
    assert explicit.resolve_channel_conversion(ChannelOrder.BGR) == step


def test_contract_fingerprint_changes_with_any_assumption() -> None:
    base = catalog.PATCH_CONTRACT
    variants = [
        dataclasses.replace(base, target_pixel_spacing_um=1.0),
        dataclasses.replace(base, stain_normalization="macenko"),
        dataclasses.replace(base, patch_size_px=64),
        dataclasses.replace(base, model_channel_order=ChannelOrder.BGR),
    ]
    prints = {base.fingerprint, *(v.fingerprint for v in variants)}
    assert len(prints) == 5
