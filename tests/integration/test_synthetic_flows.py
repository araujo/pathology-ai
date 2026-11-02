"""Synthetic end-to-end flows. Contract tests only: no medical validity is implied."""

from __future__ import annotations

import dataclasses

import pytest

from application.registries import InMemoryModelRegistry
from application.runner import ExecutionRequest
from bootstrap import Platform, build_platform
from domain.errors import (
    ExecutionUnsupportedError,
    MissingLabelError,
    PipelineValidationError,
    UnvalidatedUseError,
)
from domain.ids import ModelId, PipelineId, ValidationRecordId, Version
from domain.image import ChannelOrder
from domain.kinds import CLASSIFICATION, EMBEDDING_BAG, PortSpec
from domain.modality import MRI
from domain.pipeline import PipelineRef, ResourceBudget
from domain.prediction import (
    EvidenceSemantics,
    ImageClassification,
    SegmentationReference,
)
from domain.preprocessing import TransformStep
from domain.provenance import RunStatus
from domain.validation import (
    CapabilityStatus,
    UsageMode,
    ValidationRecord,
    ValidationScope,
)
from infrastructure.runtime import DeferredJobExecutor, FixedClock
from plugins.synthetic import catalog
from plugins.synthetic.models import EmbeddingBag
from ports import DataValue, PredictionPayload, StageContext

LUNG = "patologia/cancer/pulmao"
CRC = "patologia/cancer/coloretal"
SKIN = "patologia/pele/psoriase"


def _first_test_image(platform: Platform, ns: str):  # type: ignore[no-untyped-def]
    project = platform.project_definitions[ns]
    return project, platform.datasets.version(project.dataset).split("test")[0]


# ---------------------------------------------------------------- AC-002 dataset reuse


def test_two_projects_execute_the_same_pipeline_definition(platform: Platform) -> None:
    lung, lung_img = _first_test_image(platform, LUNG)
    crc, crc_img = _first_test_image(platform, CRC)
    a = platform.projects.run_image(lung, lung_img)
    b = platform.projects.run_image(crc, crc_img)
    assert a.run.fingerprint.pipeline == b.run.fingerprint.pipeline
    assert a.run.fingerprint.pipeline_digest == b.run.fingerprint.pipeline_digest
    assert a.run.fingerprint.dataset != b.run.fingerprint.dataset
    assert a.run.fingerprint.value != b.run.fingerprint.value
    for outcome in (a, b):
        assert outcome.run.status is RunStatus.SUCCEEDED
        assert outcome.prediction is not None
        assert isinstance(outcome.prediction.outputs[0], ImageClassification)


def test_all_split_images_run(platform: Platform) -> None:
    lung = platform.project_definitions[LUNG]
    outcomes = platform.projects.run_split(lung, "test")
    assert len(outcomes) == 2 and all(o.prediction for o in outcomes)


# ---------------------------------------------------------------- AC-004 non-MIL & volumetric


def test_direct_segmentation_pipeline_runs_without_patches_or_mil(platform: Platform) -> None:
    skin, img = _first_test_image(platform, SKIN)
    outcome = platform.projects.run_image(skin, img)
    assert outcome.prediction is not None
    (seg,) = outcome.prediction.outputs
    assert isinstance(seg, SegmentationReference)
    assert platform.artifacts.exists(seg.mask_artifact)
    ops = {c.name for c in outcome.run.stage_versions if c.role == "operation"}
    assert ops == {"model.invoke@1"}


def _volume_request(platform: Platform, **kw: object) -> ExecutionRequest:
    base: dict[str, object] = dict(
        pipeline=platform.pipelines.get(
            PipelineRef(PipelineId("volume-segmentation-3d"), Version("1.0.0"))
        ),
        image=catalog.synthetic_volume("syn-ct-0001"),
        task=catalog.VOLUME_SEG_TASK,
        models={"segmenter_3d": catalog.VOLUME_SEGMENTER.ref},
        preprocessing=catalog.VOLUME_CONTRACT,
        research_use_acknowledged=True,
        budget=ResourceBudget(gpu_count=1, memory_mb=16384),
    )
    base.update(kw)
    return ExecutionRequest(**base)  # type: ignore[arg-type]


def test_volumetric_pipeline_validates_but_reports_unsupported_execution(
    platform: Platform,
) -> None:
    req = _volume_request(platform)
    platform.runner.validate(req)  # representation is valid
    with pytest.raises(ExecutionUnsupportedError) as exc:
        platform.runner.run(req)
    assert "reader:synthetic-volume/volume_3d" in exc.value.missing
    assert "operation:volume.reconstruct_series@1" in exc.value.missing


def test_volumetric_pipeline_accepts_mri_too(platform: Platform) -> None:
    req = _volume_request(platform, image=catalog.synthetic_volume("syn-mr-0001", MRI))
    platform.runner.validate(req)


def test_volumetric_pipeline_requires_gpu_budget(platform: Platform) -> None:
    with pytest.raises(PipelineValidationError) as exc:
        platform.runner.validate(_volume_request(platform, budget=ResourceBudget()))
    assert "RESOURCE_UNAVAILABLE" in exc.value.codes


def test_wsi_image_rejected_by_volumetric_pipeline(platform: Platform) -> None:
    _, img_id = _first_test_image(platform, LUNG)
    with pytest.raises(PipelineValidationError) as exc:
        platform.runner.validate(_volume_request(platform, image=platform.datasets.image(img_id)))
    assert {"UNSUPPORTED_MODALITY", "WRONG_DIMENSIONS"} <= set(exc.value.codes)


# ---------------------------------------------------------------- AC-003 new model plugin


class _MaxPoolAggregator:
    """A brand-new synthetic model defined only in this test module."""

    def __init__(self) -> None:
        self.definition = dataclasses.replace(
            catalog.LUNG_AGGREGATOR,
            model_id=ModelId("synthetic-maxpool-aggregator"),
            family="synthetic_maxpool",
            architecture="max pooling + fixed threshold",
            input=PortSpec("embeddings", EMBEDDING_BAG, feature_dim=catalog.EMBED_DIM),
            output=PortSpec("prediction", CLASSIFICATION),
        )

    @property
    def component(self):  # type: ignore[no-untyped-def]
        from domain.provenance import ComponentVersion

        return ComponentVersion("model_plugin", str(self.definition.ref), "test")

    def predict(self, value: DataValue, ctx: StageContext) -> object:
        bag = value.payload
        assert isinstance(bag, EmbeddingBag)
        score = float((bag.features.max(axis=0).mean() + 1) / 2)
        classes = catalog.BINARY_CLASSES
        return PredictionPayload(
            outputs=(
                ImageClassification(
                    classes[int(score > 0.5)], ((classes[0], 1 - score), (classes[1], score))
                ),
            )
        )


def test_new_model_plugin_registers_and_runs_without_core_changes(platform: Platform) -> None:
    plugin = _MaxPoolAggregator()
    models = platform.models
    assert isinstance(models, InMemoryModelRegistry)
    models.register_definition(plugin.definition, catalog.artifact_for(plugin.definition))
    models.register_plugin(plugin)
    lung, img = _first_test_image(platform, LUNG)
    project = dataclasses.replace(
        lung,
        model_bindings=(("aggregator", plugin.definition.ref), ("encoder", catalog.ENCODER.ref)),
    )
    outcome = platform.projects.run_image(project, img)
    assert outcome.prediction is not None
    assert plugin.definition.ref in outcome.prediction.models
    assert outcome.prediction.evidence == ()


# ---------------------------------------------------------------- AC-007 validation awareness


def test_unvalidated_use_requires_research_opt_in(platform: Platform) -> None:
    lung, img = _first_test_image(platform, LUNG)
    with pytest.raises(UnvalidatedUseError):
        platform.projects.run_image(dataclasses.replace(lung, research_use_acknowledged=False), img)


def test_scoped_validation_record_yields_validated_only_for_its_scope(platform: Platform) -> None:
    lung = platform.project_definitions[LUNG]
    crc = platform.project_definitions[CRC]
    record = ValidationRecord(
        ValidationRecordId("synthetic-validation-fixture-1"),
        ValidationScope(
            dataset=lung.dataset,
            modality_code="wsi",
            tissue="lung",
            disease="cancer",
            task=lung.task,
            model=catalog.LUNG_AGGREGATOR.ref,
            preprocessing=catalog.PATCH_CONTRACT.ref,
            evaluation_protocol="synthetic fixture, not a real evaluation",
        ),
        evidence_reference="synthetic test fixture",
        recorded_on="2026-10-15",
        synthetic=True,
    )
    models = platform.models
    assert isinstance(models, InMemoryModelRegistry)
    models.add_validation_record(record)
    assert platform.projects.capability(lung).decision.status is CapabilityStatus.VALIDATED
    assert (
        platform.projects.capability(crc).decision.status is CapabilityStatus.SUPPORTED_UNVALIDATED
    )
    _, img = _first_test_image(platform, LUNG)
    outcome = platform.projects.run_image(
        dataclasses.replace(lung, research_use_acknowledged=False), img
    )
    assert outcome.prediction is not None
    assert outcome.prediction.usage_mode is UsageMode.VALIDATED_SCOPE
    # Same model, different disease context: never inherits the validation.
    other = dataclasses.replace(
        lung, target_context=dataclasses.replace(lung.target_context, disease="other")
    )
    assert (
        platform.projects.capability(other).decision.status
        is CapabilityStatus.SUPPORTED_UNVALIDATED
    )


def test_capability_reports_incompatible_binding(platform: Platform) -> None:
    lung = platform.project_definitions[LUNG]
    broken = dataclasses.replace(
        lung,
        model_bindings=(
            ("aggregator", catalog.SKIN_SEGMENTER.ref),
            ("encoder", catalog.ENCODER.ref),
        ),
    )
    assert platform.projects.capability(broken).decision.status is CapabilityStatus.INCOMPATIBLE


# ---------------------------------------------------------------- AC-008 reproducibility


def test_execution_records_full_provenance(platform: Platform) -> None:
    lung, img = _first_test_image(platform, LUNG)
    outcome = platform.projects.run_image(lung, img)
    fp = outcome.run.fingerprint
    assert fp.dataset == lung.dataset and fp.dataset_digest
    assert {m.slot for m in fp.models} == {"encoder", "aggregator"}
    assert all(m.artifact_checksum.startswith("synthetic:") for m in fp.models)
    assert fp.preprocessing == lung.preprocessing and fp.preprocessing_digest
    roles = {c.role for c in fp.components}
    assert {"reader", "operation", "model_plugin", "package"} <= roles
    assert dict(fp.environment.packages)["numpy"] != "not-installed"
    assert fp.seed == lung.seed
    assert outcome.run.project_id == lung.project_id
    assert platform.provenance.get(outcome.run.run_id) == outcome.run
    assert outcome.run.run_id.value.startswith("pai-run-")
    assert outcome.prediction is not None and outcome.prediction.run_id == outcome.run.run_id
    assert all(platform.artifacts.exists(a.artifact_id) for a in outcome.run.artifacts)


def test_reruns_are_deterministic_across_platform_instances() -> None:
    def run_once() -> tuple[object, ...]:
        p = build_platform(clock=FixedClock())
        lung, img = _first_test_image(p, LUNG)
        o = p.projects.run_image(lung, img)
        assert o.prediction is not None
        return (
            o.run.fingerprint.value,
            o.prediction.outputs,
            tuple((e.evidence_id, e.value) for e in o.prediction.evidence),
            tuple(a.artifact_id for a in o.run.artifacts),
        )

    assert run_once() == run_once()


def test_changing_preprocessing_changes_artifact_identities(platform: Platform) -> None:
    lung, img = _first_test_image(platform, LUNG)
    req = platform.projects.request_for(lung, img)
    a = platform.runner.run(req)
    b = platform.runner.run(dataclasses.replace(req, seed=req.seed + 1))
    assert a.run.fingerprint.value != b.run.fingerprint.value
    assert {x.artifact_id for x in a.run.artifacts}.isdisjoint(
        {x.artifact_id for x in b.run.artifacts}
    )


def test_failed_execution_is_recorded(platform: Platform) -> None:
    lung, img = _first_test_image(platform, LUNG)
    req = platform.projects.request_for(lung, img)
    no_tissue = dataclasses.replace(
        req.pipeline,
        stages=tuple(
            dataclasses.replace(s, config=(("max_mean_intensity", 0),))
            if str(s.stage_id) == "tissue"
            else s
            for s in req.pipeline.stages
        ),
    )
    with pytest.raises(ExecutionUnsupportedError):
        platform.runner.run(dataclasses.replace(req, pipeline=no_tissue))
    (run,) = platform.provenance.all()
    assert run.status is RunStatus.FAILED and run.error_code == "EXECUTION_UNSUPPORTED"


# ---------------------------------------------------------------- AC-006 / AC-011 at runtime


def test_attention_evidence_semantics(platform: Platform) -> None:
    lung, img = _first_test_image(platform, LUNG)
    outcome = platform.projects.run_image(lung, img)
    assert outcome.prediction is not None
    ev = outcome.prediction.evidence
    assert ev and all(e.semantics is EvidenceSemantics.ATTENTION_WEIGHT for e in ev)
    assert not any(e.value_is_probability for e in ev)
    assert [e.rank for e in ev] == list(range(1, len(ev) + 1))
    (cls,) = outcome.prediction.outputs
    assert isinstance(cls, ImageClassification) and not cls.scores_are_calibrated_probabilities


def test_bgr_reader_fails_closed_then_runs_with_explicit_transform() -> None:
    p = build_platform(clock=FixedClock(), reader_channel_order=ChannelOrder.BGR)
    lung, img = _first_test_image(p, LUNG)
    req = p.projects.request_for(lung, img)
    req = dataclasses.replace(req, image=dataclasses.replace(req.image, channel_order=None))
    with pytest.raises(PipelineValidationError) as exc:
        p.runner.run(req)
    assert "CHANNEL_ORDER_MISMATCH" in exc.value.codes

    rgb = build_platform(clock=FixedClock())
    expected = rgb.projects.run_image(lung, img).prediction

    step = TransformStep("channel_reorder", (("from", "bgr"), ("to", "rgb")))
    assert req.preprocessing is not None
    contract = dataclasses.replace(req.preprocessing, transforms=(step,))
    outcome = p.runner.run(dataclasses.replace(req, preprocessing=contract))
    assert outcome.prediction is not None and expected is not None
    # Same pixels reach the model after the explicit reorder.
    assert outcome.prediction.outputs == expected.outputs


def test_missing_pixel_spacing_fails_closed(platform: Platform) -> None:
    from plugins.synthetic.imaging import synthetic_slide

    lung, img = _first_test_image(platform, LUNG)
    req = platform.projects.request_for(lung, img)
    no_spacing = synthetic_slide("syn-nospacing-0001", pixel_spacing_um=None)
    with pytest.raises(PipelineValidationError) as exc:
        platform.runner.run(dataclasses.replace(req, image=no_spacing, dataset=None))
    assert "MISSING_SPATIAL_METADATA" in exc.value.codes


def test_unlabelled_images_run_inference_but_block_supervised_use(platform: Platform) -> None:
    lung = platform.project_definitions[LUNG]
    with pytest.raises(MissingLabelError):
        platform.projects.supervised_examples(lung, "train")
    train = platform.datasets.version(lung.dataset).split("train")
    unlabelled = [i for i in train if platform.datasets.label_for(i, lung.task) is None]
    assert unlabelled
    assert platform.projects.run_image(lung, unlabelled[0]).prediction is not None
    crc = platform.project_definitions[CRC]
    assert len(platform.projects.supervised_examples(crc, "train")) == 4


# ---------------------------------------------------------------- async jobs


def test_async_job_lifecycle() -> None:
    executor = DeferredJobExecutor()
    p = build_platform(clock=FixedClock(), executor=executor)
    lung, img = _first_test_image(p, LUNG)
    job = p.analysis.submit(lung, img)
    assert job.status.value == "PENDING"
    executor.drain()
    done = p.analysis.job(job.job_id)
    assert done.status.value == "COMPLETED" and done.prediction_id is not None
    prediction, run = p.analysis.result(job.job_id)
    assert prediction.prediction_id == done.prediction_id and run.run_id == prediction.run_id

    failing = p.analysis.submit(dataclasses.replace(lung, research_use_acknowledged=False), img)
    executor.drain()
    failed = p.analysis.job(failing.job_id)
    assert failed.status.value == "FAILED"
    assert failed.error is not None and failed.error.code == "UNVALIDATED_USE_NOT_ACKNOWLEDGED"

    cancelled = p.analysis.cancel(p.analysis.submit(lung, img).job_id)
    executor.drain()  # queued work for a cancelled job is skipped
    assert p.analysis.job(cancelled.job_id).status.value == "CANCELLED"
