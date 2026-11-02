from __future__ import annotations

import dataclasses

import pytest

from domain.errors import PipelineValidationError
from domain.ids import StageId, Version
from domain.image import ChannelOrder
from domain.kinds import EMBEDDING_BAG, SEGMENTATION_2D, PortSpec
from domain.modality import CT, WHOLE_SLIDE_MICROSCOPY
from domain.pipeline import (
    Edge,
    PipelineDefinition,
    PortRef,
    ResourceBudget,
    ResourceRequirements,
)
from domain.pipeline_validation import (
    BindingContext,
    require_valid_structure,
    topological_order,
    validate_binding,
    validate_structure,
)
from plugins.synthetic import catalog
from plugins.synthetic.imaging import synthetic_slide


def _ctx(**kw: object) -> BindingContext:
    base: dict[str, object] = dict(
        models={"encoder": catalog.ENCODER, "aggregator": catalog.LUNG_AGGREGATOR},
        preprocessing=catalog.PATCH_CONTRACT,
        modality=WHOLE_SLIDE_MICROSCOPY,
        source_channel_order=ChannelOrder.RGB,
        source_geometry=synthetic_slide("t-1").spatial_reference,
    )
    base.update(kw)
    return BindingContext(**base)  # type: ignore[arg-type]


def codes(issues: tuple[object, ...]) -> set[str]:
    return {i.code for i in issues}  # type: ignore[attr-defined]


def test_programmatic_pipeline_is_valid_and_ordered(wsi_pipeline: PipelineDefinition) -> None:
    require_valid_structure(wsi_pipeline)
    assert [str(s) for s in topological_order(wsi_pipeline)] == [
        "tissue",
        "patches",
        "encode",
        "aggregate",
    ]
    assert validate_binding(wsi_pipeline, _ctx()) == ()


def test_cycle_detected(wsi_pipeline: PipelineDefinition) -> None:
    looped = dataclasses.replace(
        wsi_pipeline,
        edges=wsi_pipeline.edges
        + (Edge(PortRef(StageId("aggregate"), "prediction"), PortRef(StageId("tissue"), "image")),),
    )
    issues = validate_structure(looped)
    assert "CYCLE" in codes(issues)
    with pytest.raises(PipelineValidationError) as exc:
        topological_order(looped)
    assert exc.value.codes == ("CYCLE",)


def test_kind_mismatch_detected(wsi_pipeline: PipelineDefinition) -> None:
    edges = tuple(
        Edge(PortRef(None, "image"), e.target) if str(e.target) == "encode.patches" else e
        for e in wsi_pipeline.edges
    )
    issues = validate_structure(dataclasses.replace(wsi_pipeline, edges=edges))
    assert "KIND_MISMATCH" in codes(issues)


def test_unknown_port_and_unconnected_input(wsi_pipeline: PipelineDefinition) -> None:
    edges = wsi_pipeline.edges[:-1] + (
        Edge(PortRef(StageId("encode"), "nope"), PortRef(StageId("aggregate"), "embeddings")),
    )
    c = codes(validate_structure(dataclasses.replace(wsi_pipeline, edges=edges)))
    assert {"UNKNOWN_PORT", "UNCONNECTED_INPUT"} <= c


def test_unsupported_schema_version(wsi_pipeline: PipelineDefinition) -> None:
    p = dataclasses.replace(wsi_pipeline, schema_version="pathology-ai.pipeline/99")
    assert "UNSUPPORTED_SCHEMA" in codes(validate_structure(p))


def test_validation_is_deterministic(wsi_pipeline: PipelineDefinition) -> None:
    broken = dataclasses.replace(wsi_pipeline, edges=())
    first = validate_structure(broken)
    assert first == validate_structure(broken)
    assert list(first) == sorted(first, key=lambda i: i.sort_key())


def test_wrong_feature_dimension(wsi_pipeline: PipelineDefinition) -> None:
    wrong = dataclasses.replace(
        catalog.LUNG_AGGREGATOR, input=PortSpec("embeddings", EMBEDDING_BAG, feature_dim=1024)
    )
    issues = validate_binding(
        wsi_pipeline, _ctx(models={"encoder": catalog.ENCODER, "aggregator": wrong})
    )
    assert "FEATURE_DIM_MISMATCH" in codes(issues)


def test_wrong_image_dimensions(wsi_pipeline: PipelineDefinition) -> None:
    from plugins.synthetic.catalog import synthetic_volume

    issues = validate_binding(
        wsi_pipeline, _ctx(source_geometry=synthetic_volume("v-1").spatial_reference)
    )
    assert "WRONG_DIMENSIONS" in codes(issues)


def test_unsupported_modality(wsi_pipeline: PipelineDefinition) -> None:
    assert "UNSUPPORTED_MODALITY" in codes(
        validate_binding(wsi_pipeline, _ctx(modality=CT, source_geometry=None))
    )


def test_unsupported_model_input(wsi_pipeline: PipelineDefinition) -> None:
    models = {"encoder": catalog.ENCODER, "aggregator": catalog.SKIN_SEGMENTER}
    c = codes(validate_binding(wsi_pipeline, _ctx(models=models)))
    assert {"UNSUPPORTED_MODEL_INPUT", "MODEL_OUTPUT_MISMATCH"} <= c


def test_missing_and_unused_model_bindings(wsi_pipeline: PipelineDefinition) -> None:
    c = codes(
        validate_binding(
            wsi_pipeline, _ctx(models={"encoder": catalog.ENCODER, "extra": catalog.ENCODER})
        )
    )
    assert {"MISSING_MODEL_BINDING", "UNUSED_MODEL_BINDING"} <= c


def test_missing_preprocessing(wsi_pipeline: PipelineDefinition) -> None:
    assert "MISSING_PREPROCESSING" in codes(
        validate_binding(wsi_pipeline, _ctx(preprocessing=None))
    )


def test_incomplete_preprocessing_contract(wsi_pipeline: PipelineDefinition) -> None:
    incomplete = dataclasses.replace(catalog.PATCH_CONTRACT, stain_normalization=None)
    issues = validate_binding(wsi_pipeline, _ctx(preprocessing=incomplete))
    assert any(i.code == "MISSING_PREPROCESSING" and "stain" in i.message for i in issues)


def test_preprocessing_bound_to_model(wsi_pipeline: PipelineDefinition) -> None:
    other = dataclasses.replace(catalog.PATCH_CONTRACT, version=Version("2.0.0"))
    assert "PREPROCESSING_MISMATCH" in codes(
        validate_binding(wsi_pipeline, _ctx(preprocessing=other))
    )


def test_channel_order_mismatch(wsi_pipeline: PipelineDefinition) -> None:
    assert "CHANNEL_ORDER_MISMATCH" in codes(
        validate_binding(wsi_pipeline, _ctx(source_channel_order=ChannelOrder.BGR))
    )


def test_missing_pixel_spacing(wsi_pipeline: PipelineDefinition) -> None:
    geom = synthetic_slide("t-2", pixel_spacing_um=None).spatial_reference
    assert "MISSING_SPATIAL_METADATA" in codes(
        validate_binding(wsi_pipeline, _ctx(source_geometry=geom))
    )


def test_upstream_encoder_must_match(wsi_pipeline: PipelineDefinition) -> None:
    other_encoder = dataclasses.replace(catalog.ENCODER, version=Version("9.0.0"))
    issues = validate_binding(
        wsi_pipeline, _ctx(models={"encoder": other_encoder, "aggregator": catalog.LUNG_AGGREGATOR})
    )
    assert "UPSTREAM_MODEL_MISMATCH" in codes(issues)


def test_resource_requirements(wsi_pipeline: PipelineDefinition) -> None:
    stages = tuple(
        dataclasses.replace(s, resources=ResourceRequirements(gpu_count=2))
        if s.model_slot == "encoder"
        else s
        for s in wsi_pipeline.stages
    )
    p = dataclasses.replace(wsi_pipeline, stages=stages)
    assert "RESOURCE_UNAVAILABLE" in codes(validate_binding(p, _ctx()))
    assert validate_binding(p, _ctx(budget=ResourceBudget(gpu_count=2))) == ()


def test_model_stage_arity(wsi_pipeline: PipelineDefinition) -> None:
    stages = tuple(
        dataclasses.replace(s, outputs=s.outputs + (PortSpec("extra", SEGMENTATION_2D),))
        if s.model_slot == "aggregator"
        else s
        for s in wsi_pipeline.stages
    )
    assert "MODEL_STAGE_ARITY" in codes(
        validate_structure(dataclasses.replace(wsi_pipeline, stages=stages))
    )
