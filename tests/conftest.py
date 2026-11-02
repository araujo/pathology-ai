from __future__ import annotations

from pathlib import Path

import pytest

from bootstrap import Platform, build_platform
from domain.ids import PipelineId, StageId, Version
from domain.kinds import (
    CLASSIFICATION,
    EMBEDDING_BAG,
    IMAGE_PYRAMIDAL_2D,
    PATCH_SET,
    TISSUE_MASK_2D,
    PortSpec,
)
from domain.modality import SpatialKind
from domain.pipeline import (
    Edge,
    OperationRef,
    PipelineDefinition,
    PortRef,
    ResourceRequirements,
    StageDefinition,
)
from infrastructure.runtime import FixedClock

REPO_ROOT = Path(__file__).resolve().parents[1]

IMG = PortSpec("image", IMAGE_PYRAMIDAL_2D, SpatialKind.PYRAMIDAL_2D)
INVOKE = OperationRef("model.invoke", Version("1"))


def build_wsi_pipeline() -> PipelineDefinition:
    """Programmatic construction of the WSI template (no YAML involved)."""
    s = StageId
    return PipelineDefinition(
        pipeline_id=PipelineId("wsi-patch-embed-aggregate"),
        version=Version("1.0.0"),
        description="Patch encoder followed by a bag-level (MIL-style) aggregator.",
        inputs=(IMG,),
        stages=(
            StageDefinition(
                s("tissue"),
                OperationRef("tissue.threshold", Version("1")),
                (IMG,),
                (PortSpec("mask", TISSUE_MASK_2D),),
                resources=ResourceRequirements(memory_mb=256),
                config=(("max_mean_intensity", 220),),
            ),
            StageDefinition(
                s("patches"),
                OperationRef("patch.grid", Version("1")),
                (IMG, PortSpec("mask", TISSUE_MASK_2D)),
                (PortSpec("patches", PATCH_SET),),
                requires_preprocessing=True,
                resources=ResourceRequirements(memory_mb=512),
                config=(("max_patches", 48), ("min_tissue_fraction", 0.5)),
            ),
            StageDefinition(
                s("encode"),
                INVOKE,
                (PortSpec("patches", PATCH_SET),),
                (PortSpec("embeddings", EMBEDDING_BAG),),
                model_slot="encoder",
                resources=ResourceRequirements(memory_mb=512),
            ),
            StageDefinition(
                s("aggregate"),
                INVOKE,
                (PortSpec("embeddings", EMBEDDING_BAG),),
                (PortSpec("prediction", CLASSIFICATION),),
                model_slot="aggregator",
                resources=ResourceRequirements(memory_mb=256),
                config=(("evidence_top_k", 5),),
            ),
        ),
        edges=(
            Edge(PortRef(None, "image"), PortRef(s("tissue"), "image")),
            Edge(PortRef(None, "image"), PortRef(s("patches"), "image")),
            Edge(PortRef(s("tissue"), "mask"), PortRef(s("patches"), "mask")),
            Edge(PortRef(s("patches"), "patches"), PortRef(s("encode"), "patches")),
            Edge(PortRef(s("encode"), "embeddings"), PortRef(s("aggregate"), "embeddings")),
        ),
        outputs=(("prediction", PortRef(s("aggregate"), "prediction")),),
    )


@pytest.fixture
def platform() -> Platform:
    return build_platform(REPO_ROOT, clock=FixedClock())


@pytest.fixture
def wsi_pipeline() -> PipelineDefinition:
    return build_wsi_pipeline()
