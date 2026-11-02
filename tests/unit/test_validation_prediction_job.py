from __future__ import annotations

import dataclasses

import pytest

from domain.dataset import DatasetVersionRef
from domain.errors import (
    EvidenceSemanticsError,
    IncompatibleModelError,
    InvalidJobTransitionError,
    InvalidSpatialReferenceError,
    UnvalidatedUseError,
)
from domain.geometry import Level0Rect, RectangleGeometry
from domain.ids import (
    DatasetId,
    EvidenceId,
    ImageId,
    JobId,
    PredictionId,
    ProjectId,
    RunId,
    TaskId,
    ValidationRecordId,
    Version,
)
from domain.job import AnalysisJob, JobError, JobStatus
from domain.prediction import (
    ContinuousScore,
    Detection,
    EvidenceSemantics,
    ImageClassification,
    PredictionResult,
    SpatialEvidence,
)
from domain.task import (
    InvalidTaskDefinitionError,
    OutputKind,
    OutputSchema,
    TaskDefinition,
    TaskType,
)
from domain.validation import (
    CapabilityQuery,
    CapabilityStatus,
    UsageMode,
    ValidationRecord,
    ValidationScope,
    authorize_use,
    resolve_capability,
)
from plugins.synthetic import catalog

AGG = catalog.LUNG_AGGREGATOR
DS = DatasetVersionRef(DatasetId("synthetic-lung-slides"), Version("0.1.0"))


def _scope(**kw: object) -> ValidationScope:
    base: dict[str, object] = dict(
        dataset=DS,
        modality_code="wsi",
        tissue="lung",
        disease="cancer",
        task=catalog.LUNG_TASK.ref,
        model=AGG.ref,
        preprocessing=catalog.PATCH_CONTRACT.ref,
        evaluation_protocol="synthetic-protocol",
    )
    base.update(kw)
    return ValidationScope(**base)  # type: ignore[arg-type]


RECORD = ValidationRecord(
    ValidationRecordId("synthetic-validation-record-1"),
    _scope(),
    "synthetic fixture",
    "2026-10-15",
    synthetic=True,
)


def _query(**kw: object) -> CapabilityQuery:
    base: dict[str, object] = dict(
        model=AGG.ref,
        modality_code="wsi",
        task=catalog.LUNG_TASK.ref,
        preprocessing=catalog.PATCH_CONTRACT.ref,
        tissue="lung",
        disease="cancer",
    )
    base.update(kw)
    return CapabilityQuery(**base)  # type: ignore[arg-type]


def test_validated_requires_exact_scope_match() -> None:
    d = resolve_capability(AGG, _query(), [RECORD], catalog.LUNG_TASK)
    assert d.status is CapabilityStatus.VALIDATED
    assert d.validation_record == RECORD.record_id


@pytest.mark.parametrize(
    "change",
    [
        {"disease": "other-disease"},
        {"tissue": "colorectum"},
        {"tissue": None},
        {"disease": None},
        {"dataset": DatasetVersionRef(DatasetId("other-ds"), Version("1"))},
    ],
)
def test_validation_does_not_transfer_across_scope(change: dict[str, object]) -> None:
    d = resolve_capability(AGG, _query(**change), [RECORD], catalog.LUNG_TASK)
    assert d.status is CapabilityStatus.SUPPORTED_UNVALIDATED


def test_validation_scope_bound_to_model_and_preprocessing_version() -> None:
    other_version = dataclasses.replace(AGG, version=Version("2.0.0"))
    d = resolve_capability(other_version, _query(model=other_version.ref), [RECORD])
    assert d.status is CapabilityStatus.SUPPORTED_UNVALIDATED


def test_incompatible_modality_and_preprocessing() -> None:
    d = resolve_capability(AGG, _query(modality_code="ct"), [RECORD])
    assert d.status is CapabilityStatus.INCOMPATIBLE
    d = resolve_capability(AGG, _query(preprocessing=catalog.DIRECT_CONTRACT.ref), [RECORD])
    assert d.status is CapabilityStatus.INCOMPATIBLE
    d = resolve_capability(AGG, _query(), [RECORD], catalog.SKIN_SEG_TASK)
    assert d.status is CapabilityStatus.INCOMPATIBLE


def test_authorize_use_requires_research_opt_in() -> None:
    unvalidated = resolve_capability(AGG, _query(tissue=None), [RECORD])
    with pytest.raises(UnvalidatedUseError):
        authorize_use(unvalidated, research_use_acknowledged=False)
    assert authorize_use(unvalidated, True) is UsageMode.RESEARCH_UNVALIDATED
    incompatible = resolve_capability(AGG, _query(modality_code="mri"), [])
    with pytest.raises(IncompatibleModelError):
        authorize_use(incompatible, True)


def test_validation_scope_requires_every_dimension() -> None:
    with pytest.raises(ValueError):
        _scope(disease="")


def _evidence(**kw: object) -> SpatialEvidence:
    base: dict[str, object] = dict(
        evidence_id=EvidenceId("ev-1"),
        image_id=ImageId("img-1"),
        geometry=RectangleGeometry(Level0Rect(0, 0, 10, 10)),
        semantics=EvidenceSemantics.ATTENTION_WEIGHT,
        producer=AGG.ref,
        value=0.2,
    )
    base.update(kw)
    return SpatialEvidence(**base)  # type: ignore[arg-type]


def test_attention_cannot_be_presented_as_probability() -> None:
    with pytest.raises(EvidenceSemanticsError):
        _evidence(value_is_probability=True)
    with pytest.raises(EvidenceSemanticsError):
        _evidence(semantics=EvidenceSemantics.SALIENCY, value_is_probability=True)


def test_localized_probability_requires_localization_validation() -> None:
    with pytest.raises(EvidenceSemanticsError):
        _evidence(semantics=EvidenceSemantics.LOCALIZED_PROBABILITY, value=0.7)
    ok = _evidence(
        semantics=EvidenceSemantics.LOCALIZED_PROBABILITY,
        value=0.7,
        value_is_probability=True,
        localization_validation=ValidationRecordId("synthetic-loc-validation"),
    )
    assert ok.value == 0.7
    with pytest.raises(EvidenceSemanticsError):
        _evidence(
            semantics=EvidenceSemantics.LOCALIZED_PROBABILITY,
            value=1.5,
            localization_validation=ValidationRecordId("synthetic-loc-validation"),
        )


def test_typed_outputs_are_not_forced_into_classification() -> None:
    outs = (
        ContinuousScore("synthetic_fraction", 0.3, "ratio"),
        Detection(RectangleGeometry(Level0Rect(1, 1, 2, 2)), "synthetic_object", 0.9),
    )
    assert {o.kind for o in outs} == {OutputKind.CONTINUOUS_SCORE, OutputKind.DETECTION}


def test_prediction_rejects_evidence_for_other_image() -> None:
    with pytest.raises(InvalidSpatialReferenceError):
        PredictionResult(
            PredictionId("p-1"),
            ImageId("img-2"),
            catalog.LUNG_TASK.ref,
            (AGG.ref,),
            (
                ImageClassification(
                    "synthetic_positive", (("synthetic_negative", 0.1), ("synthetic_positive", 0.9))
                ),
            ),
            (_evidence(),),
            RunId("r-1"),
            CapabilityStatus.SUPPORTED_UNVALIDATED,
            UsageMode.RESEARCH_UNVALIDATED,
        )


def test_prediction_usage_mode_must_match_capability() -> None:
    with pytest.raises(ValueError):
        PredictionResult(
            PredictionId("p-1"),
            ImageId("img-1"),
            catalog.LUNG_TASK.ref,
            (AGG.ref,),
            (ContinuousScore("x", 1.0),),
            (),
            RunId("r-1"),
            CapabilityStatus.SUPPORTED_UNVALIDATED,
            UsageMode.VALIDATED_SCOPE,
        )


def test_task_type_and_output_schema_must_agree() -> None:
    with pytest.raises(InvalidTaskDefinitionError):
        TaskDefinition(
            TaskId("t"), Version("1"), TaskType.REGRESSION, OutputSchema(OutputKind.SEGMENTATION)
        )
    with pytest.raises(InvalidTaskDefinitionError):
        TaskDefinition(
            TaskId("t"),
            Version("1"),
            TaskType.BINARY_CLASSIFICATION,
            OutputSchema(OutputKind.IMAGE_CLASSIFICATION, ("a", "b", "c")),
        )
    TaskDefinition(
        TaskId("t"),
        Version("1"),
        TaskType.MULTILABEL_CLASSIFICATION,
        OutputSchema(OutputKind.IMAGE_CLASSIFICATION, ("a", "b", "c")),
    )


def test_job_state_machine() -> None:
    job = AnalysisJob(JobId("j-1"), ProjectId("p"), ImageId("i"))
    running = job.transition(JobStatus.RUNNING)
    with pytest.raises(InvalidJobTransitionError):
        running.transition(JobStatus.COMPLETED)  # completed needs a prediction id
    with pytest.raises(InvalidJobTransitionError):
        running.transition(JobStatus.FAILED)  # failed needs a typed error
    done = running.transition(JobStatus.COMPLETED, prediction_id=PredictionId("p-1"))
    with pytest.raises(InvalidJobTransitionError):
        done.transition(JobStatus.RUNNING)
    with pytest.raises(InvalidJobTransitionError):
        job.transition(JobStatus.COMPLETED, prediction_id=PredictionId("p-1"))
    failed = running.transition(JobStatus.FAILED, error=JobError("X", "y"))
    assert failed.status is JobStatus.FAILED
