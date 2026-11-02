"""Scoped scientific validation and capability resolution.

Technical compatibility ("this model can execute on this input") and
scientific validation ("this model was evaluated for exactly this scope") are
separate questions. A `VALIDATED` decision requires an explicit
`ValidationRecord` whose scope matches the request on every scoped dimension;
validation for one disease, tissue, modality, task, model version or
preprocessing version never implies validation for another.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from domain.dataset import DatasetVersionRef
from domain.errors import IncompatibleModelError, UnvalidatedUseError
from domain.ids import ValidationRecordId
from domain.model import ModelDefinition, ModelRef
from domain.preprocessing import PreprocessingRef
from domain.task import TaskDefinition, TaskRef


class CapabilityStatus(StrEnum):
    VALIDATED = "VALIDATED"
    SUPPORTED_UNVALIDATED = "SUPPORTED_UNVALIDATED"
    INCOMPATIBLE = "INCOMPATIBLE"


class UsageMode(StrEnum):
    VALIDATED_SCOPE = "validated_scope"
    RESEARCH_UNVALIDATED = "research_unvalidated"


@dataclass(frozen=True, slots=True)
class ValidationScope:
    """Every dimension is mandatory: a validation claim without full scope is not accepted."""

    dataset: DatasetVersionRef
    modality_code: str
    tissue: str
    disease: str
    task: TaskRef
    model: ModelRef
    preprocessing: PreprocessingRef
    evaluation_protocol: str

    def __post_init__(self) -> None:
        for name in ("modality_code", "tissue", "disease", "evaluation_protocol"):
            if not getattr(self, name):
                raise ValueError(f"validation scope requires a non-empty {name}")


@dataclass(frozen=True, slots=True)
class ValidationRecord:
    """A scientific validation record, separate from any execution record."""

    record_id: ValidationRecordId
    scope: ValidationScope
    evidence_reference: str
    recorded_on: str
    synthetic: bool = False


@dataclass(frozen=True, slots=True)
class CapabilityQuery:
    model: ModelRef
    modality_code: str
    task: TaskRef
    preprocessing: PreprocessingRef | None
    tissue: str | None = None
    disease: str | None = None
    dataset: DatasetVersionRef | None = None


@dataclass(frozen=True, slots=True)
class CapabilityDecision:
    status: CapabilityStatus
    reasons: tuple[str, ...]
    validation_record: ValidationRecordId | None = None

    @property
    def technically_compatible(self) -> bool:
        return self.status is not CapabilityStatus.INCOMPATIBLE


def technical_issues(
    model: ModelDefinition, query: CapabilityQuery, task: TaskDefinition | None
) -> tuple[str, ...]:
    issues: list[str] = []
    if model.ref != query.model:
        issues.append(f"query targets {query.model}, definition is {model.ref}")
    if query.modality_code not in model.compatible_modalities:
        issues.append(f"modality {query.modality_code!r} not supported by {model.ref}")
    if model.required_preprocessing is not None and (
        query.preprocessing != model.required_preprocessing
    ):
        issues.append(
            f"preprocessing {query.preprocessing} differs from required "
            f"{model.required_preprocessing}"
        )
    if task is not None and model.output_schema is not None:
        if task.output_schema.kind is not model.output_schema.kind:
            issues.append(
                f"task output {task.output_schema.kind.value} != "
                f"model output {model.output_schema.kind.value}"
            )
        elif task.output_schema.classes != model.output_schema.classes:
            issues.append("task classes differ from model output classes")
    return tuple(sorted(issues))


def _scope_matches(scope: ValidationScope, query: CapabilityQuery) -> bool:
    if query.tissue is None or query.disease is None or query.preprocessing is None:
        return False  # unknown context can never be presumed inside a validated scope
    return (
        scope.model == query.model
        and scope.modality_code == query.modality_code
        and scope.task == query.task
        and scope.preprocessing == query.preprocessing
        and scope.tissue == query.tissue
        and scope.disease == query.disease
        and (query.dataset is None or scope.dataset == query.dataset)
    )


def resolve_capability(
    model: ModelDefinition,
    query: CapabilityQuery,
    records: Iterable[ValidationRecord],
    task: TaskDefinition | None = None,
) -> CapabilityDecision:
    issues = technical_issues(model, query, task)
    if issues:
        return CapabilityDecision(CapabilityStatus.INCOMPATIBLE, issues)
    matching = sorted(
        (r for r in records if _scope_matches(r.scope, query)), key=lambda r: r.record_id
    )
    if matching:
        return CapabilityDecision(
            CapabilityStatus.VALIDATED,
            (f"scope matched by validation record {matching[0].record_id}",),
            matching[0].record_id,
        )
    return CapabilityDecision(
        CapabilityStatus.SUPPORTED_UNVALIDATED,
        ("technically compatible; no validation record covers this exact scope",),
    )


def authorize_use(decision: CapabilityDecision, research_use_acknowledged: bool) -> UsageMode:
    """Fail closed: incompatible never runs; unvalidated runs only with explicit opt-in."""
    if decision.status is CapabilityStatus.INCOMPATIBLE:
        raise IncompatibleModelError("; ".join(decision.reasons))
    if decision.status is CapabilityStatus.VALIDATED:
        return UsageMode.VALIDATED_SCOPE
    if not research_use_acknowledged:
        raise UnvalidatedUseError(
            "combination is SUPPORTED_UNVALIDATED; explicit research-use acknowledgement required"
        )
    return UsageMode.RESEARCH_UNVALIDATED
