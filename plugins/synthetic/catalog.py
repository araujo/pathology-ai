"""Synthetic catalog: contracts, tasks, models, datasets and images.

All identifiers are prefixed `synthetic-`/`syn-`. Nothing here is a validated
medical asset, and no validation records are registered.
"""

from __future__ import annotations

from dataclasses import dataclass

from domain.dataset import DatasetDefinition, DatasetVersion, Label
from domain.geometry import VolumeGeometry
from domain.hashing import digest
from domain.ids import (
    ArtifactId,
    DatasetId,
    ImageId,
    ModelId,
    PreprocessingContractId,
    TaskId,
    Version,
)
from domain.image import ChannelOrder, ImageAsset
from domain.kinds import (
    CLASSIFICATION,
    EMBEDDING_BAG,
    IMAGE_PYRAMIDAL_2D,
    IMAGE_VOLUME,
    PATCH_SET,
    SEGMENTATION_2D,
    SEGMENTATION_3D,
    PortSpec,
)
from domain.modality import CT, ImageFormat, ImageModality, SpatialKind
from domain.model import ModelArtifact, ModelDefinition
from domain.preprocessing import Interpolation, PreprocessingContract
from domain.task import OutputKind, OutputSchema, TaskDefinition, TaskType
from plugins.synthetic.imaging import synthetic_slide

V1 = Version("1.0.0")
EMBED_DIM = 16

PATCH_CONTRACT = PreprocessingContract(
    contract_id=PreprocessingContractId("synthetic-wsi-patch32-mpp0.5"),
    version=V1,
    spatial_kind=SpatialKind.PYRAMIDAL_2D,
    model_channel_order=ChannelOrder.RGB,
    interpolation=Interpolation.AREA,
    target_pixel_spacing_um=0.5,
    stain_normalization="none",
    patch_size_px=32,
    patch_stride_px=32,
    patch_sampling="grid_tissue_fraction",
)

DIRECT_CONTRACT = PreprocessingContract(
    contract_id=PreprocessingContractId("synthetic-wsi-direct-mpp2"),
    version=V1,
    spatial_kind=SpatialKind.PYRAMIDAL_2D,
    model_channel_order=ChannelOrder.RGB,
    interpolation=Interpolation.AREA,
    target_pixel_spacing_um=2.0,
    stain_normalization="none",
)

VOLUME_CONTRACT = PreprocessingContract(
    contract_id=PreprocessingContractId("synthetic-volume-iso1mm"),
    version=V1,
    spatial_kind=SpatialKind.VOLUME_3D,
    interpolation=Interpolation.BILINEAR,
    target_voxel_spacing_mm=(1.0, 1.0, 1.0),
    orientation="LPS",
    intensity_normalization="window[-1000,1000]->[0,1]",
)

BINARY_CLASSES = ("synthetic_negative", "synthetic_positive")


def _binary_task(task_id: str) -> TaskDefinition:
    return TaskDefinition(
        TaskId(task_id),
        V1,
        TaskType.BINARY_CLASSIFICATION,
        OutputSchema(OutputKind.IMAGE_CLASSIFICATION, BINARY_CLASSES),
        target_name="synthetic_endpoint",
        description="Synthetic binary slide-level endpoint (no clinical meaning).",
    )


LUNG_TASK = _binary_task("synthetic-lung-binary-endpoint")
COLORECTAL_TASK = _binary_task("synthetic-colorectal-binary-endpoint")
SKIN_SEG_TASK = TaskDefinition(
    TaskId("synthetic-skin-region-segmentation"),
    V1,
    TaskType.SEGMENTATION,
    OutputSchema(OutputKind.SEGMENTATION, ("background", "synthetic_region")),
    description="Synthetic 2D region segmentation (no clinical meaning).",
)
VOLUME_SEG_TASK = TaskDefinition(
    TaskId("synthetic-volume-structure-segmentation"),
    V1,
    TaskType.SEGMENTATION,
    OutputSchema(OutputKind.VOLUMETRIC_SEGMENTATION, ("background", "synthetic_structure")),
    description="Schema-only volumetric segmentation task.",
)

ENCODER = ModelDefinition(
    model_id=ModelId("synthetic-patch-encoder"),
    version=V1,
    family="synthetic_feature_extractor",
    architecture="colour-statistics random projection",
    input=PortSpec("patches", PATCH_SET),
    output=PortSpec("embeddings", EMBEDDING_BAG, feature_dim=EMBED_DIM),
    compatible_modalities=frozenset({"wsi"}),
    required_preprocessing=PATCH_CONTRACT.ref,
    dependencies=("numpy",),
    synthetic=True,
)


def _aggregator(model_id: str, task: TaskDefinition) -> ModelDefinition:
    return ModelDefinition(
        model_id=ModelId(model_id),
        version=V1,
        family="synthetic_bag_aggregator",
        architecture="attention-pooled linear head (untrained)",
        input=PortSpec("embeddings", EMBEDDING_BAG, feature_dim=EMBED_DIM),
        output=PortSpec("prediction", CLASSIFICATION),
        compatible_modalities=frozenset({"wsi"}),
        required_preprocessing=PATCH_CONTRACT.ref,
        output_schema=task.output_schema,
        required_upstream_model=ENCODER.ref,
        dependencies=("numpy",),
        synthetic=True,
    )


LUNG_AGGREGATOR = _aggregator("synthetic-aggregator-lung", LUNG_TASK)
COLORECTAL_AGGREGATOR = _aggregator("synthetic-aggregator-colorectal", COLORECTAL_TASK)

SKIN_SEGMENTER = ModelDefinition(
    model_id=ModelId("synthetic-direct-segmenter-skin"),
    version=V1,
    family="synthetic_direct_segmenter",
    architecture="intensity threshold (untrained)",
    input=PortSpec("image", IMAGE_PYRAMIDAL_2D, SpatialKind.PYRAMIDAL_2D),
    output=PortSpec("segmentation", SEGMENTATION_2D),
    compatible_modalities=frozenset({"wsi"}),
    required_preprocessing=DIRECT_CONTRACT.ref,
    output_schema=SKIN_SEG_TASK.output_schema,
    dependencies=("numpy",),
    synthetic=True,
)

VOLUME_SEGMENTER = ModelDefinition(
    model_id=ModelId("synthetic-volume-segmenter"),
    version=V1,
    family="synthetic_volumetric_segmenter",
    architecture="definition only (no implementation)",
    input=PortSpec("volume", IMAGE_VOLUME, SpatialKind.VOLUME_3D),
    output=PortSpec("segmentation", SEGMENTATION_3D),
    compatible_modalities=frozenset({"ct", "mri"}),
    required_preprocessing=VOLUME_CONTRACT.ref,
    output_schema=VOLUME_SEG_TASK.output_schema,
    synthetic=True,
)

ALL_CONTRACTS = (PATCH_CONTRACT, DIRECT_CONTRACT, VOLUME_CONTRACT)
ALL_TASKS = (LUNG_TASK, COLORECTAL_TASK, SKIN_SEG_TASK, VOLUME_SEG_TASK)
ALL_MODELS = (ENCODER, LUNG_AGGREGATOR, COLORECTAL_AGGREGATOR, SKIN_SEGMENTER, VOLUME_SEGMENTER)


def artifact_for(model: ModelDefinition) -> ModelArtifact:
    return ModelArtifact(
        artifact_id=ArtifactId(f"synthetic-weights-{model.model_id}"),
        model=model.ref,
        checksum="synthetic:" + digest({"model": model.ref})[:32],
        preprocessing=model.required_preprocessing,
        synthetic=True,
    )


@dataclass(frozen=True)
class SyntheticDataset:
    definition: DatasetDefinition
    version: DatasetVersion
    images: tuple[ImageAsset, ...]
    labels: tuple[Label, ...]


def _dataset(
    dataset_id: str, prefix: str, n: int, task: TaskDefinition, labelled: int, seed: int
) -> SyntheticDataset:
    images = tuple(
        synthetic_slide(f"{prefix}-{i:04d}", width=1024 + 64 * (i % 3), height=768)
        for i in range(1, n + 1)
    )
    ids = tuple(img.image_id for img in images)
    train, test = ids[: n - 2], ids[n - 2 :]
    labels = tuple(
        Label(image_id, task.ref, task.output_schema.classes[(k + seed) % 2])
        if task.output_schema.classes
        else Label(image_id, task.ref, 0.0)
        for k, image_id in enumerate(train[:labelled])
    )
    version = DatasetVersion(
        DatasetId(dataset_id),
        Version("0.1.0"),
        content_digest="sha256:" + digest([img.source_checksum for img in images]),
        image_ids=ids,
        splits=(("train", train), ("test", test)),
    )
    return SyntheticDataset(
        DatasetDefinition(DatasetId(dataset_id), f"{dataset_id} (synthetic)", "wsi", True),
        version,
        images,
        labels,
    )


def synthetic_datasets() -> tuple[SyntheticDataset, ...]:
    return (
        # one training image is intentionally unlabelled
        _dataset("synthetic-lung-slides", "syn-lung", 6, LUNG_TASK, labelled=3, seed=0),
        _dataset("synthetic-colorectal-slides", "syn-crc", 6, COLORECTAL_TASK, labelled=4, seed=1),
        _dataset("synthetic-skin-slides", "syn-skin", 4, SKIN_SEG_TASK, labelled=0, seed=2),
    )


SYNTHETIC_VOLUME_FORMAT = ImageFormat("synthetic-volume")


def synthetic_volume(image_id: str, modality: ImageModality = CT) -> ImageAsset:
    """Metadata-only volume (no voxels, no reader): exercises 3D contracts only."""
    return ImageAsset(
        image_id=ImageId(image_id),
        modality=modality,
        image_format=SYNTHETIC_VOLUME_FORMAT,
        spatial_reference=VolumeGeometry(
            shape=(128, 128, 64),
            spacing_mm=(0.8, 0.8, 2.5),
            origin_mm=(-51.2, -51.2, 0.0),
            direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            series_uid=None,
        ),
        channel_order=ChannelOrder.GRAYSCALE,
        source_checksum=None,
    )
