"""Deterministic pipeline validation.

`validate_structure` checks the graph alone. `validate_binding` checks a
pipeline against concrete model bindings, the bound preprocessing contract,
the input modality/source and a resource budget. Both return sorted issues;
the `require_*` variants raise `PipelineValidationError`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from domain.errors import (
    ChannelOrderMismatchError,
    MissingPreprocessingError,
    PipelineValidationError,
    ValidationIssue,
)
from domain.geometry import PyramidalGeometry, SpatialReference
from domain.ids import StageId
from domain.image import ChannelOrder
from domain.kinds import PortSpec
from domain.modality import ImageModality
from domain.model import ModelDefinition
from domain.pipeline import (
    SUPPORTED_PIPELINE_SCHEMAS,
    PipelineDefinition,
    PortRef,
    ResourceBudget,
    StageDefinition,
)
from domain.preprocessing import PreprocessingContract


def _sorted(issues: list[ValidationIssue]) -> tuple[ValidationIssue, ...]:
    return tuple(sorted(set(issues), key=ValidationIssue.sort_key))


def _resolve_port(p: PipelineDefinition, ref: PortRef, *, as_source: bool) -> PortSpec | None:
    if ref.stage is None:
        return p.input(ref.port) if as_source else None
    stage = p.stage(ref.stage)
    if stage is None:
        return None
    return stage.output(ref.port) if as_source else stage.input(ref.port)


def topological_order(p: PipelineDefinition) -> tuple[StageId, ...]:
    """Kahn's algorithm with lexical tie-breaking; raises on cycles."""
    order, cyclic = _toposort(p)
    if cyclic:
        raise PipelineValidationError(
            (ValidationIssue("CYCLE", "stages form a cycle: " + ", ".join(cyclic)),)
        )
    return order


def _toposort(p: PipelineDefinition) -> tuple[tuple[StageId, ...], tuple[str, ...]]:
    ids = {s.stage_id for s in p.stages}
    deps: dict[StageId, set[StageId]] = {sid: set() for sid in ids}
    for e in p.edges:
        if e.source.stage in ids and e.target.stage in ids:
            assert e.source.stage is not None and e.target.stage is not None
            deps[e.target.stage].add(e.source.stage)
    order: list[StageId] = []
    ready = sorted(sid for sid, d in deps.items() if not d)
    remaining = {sid: set(d) for sid, d in deps.items()}
    while ready:
        sid = ready.pop(0)
        order.append(sid)
        del remaining[sid]
        for other, d in remaining.items():
            if sid in d:
                d.discard(sid)
                if not d and other not in ready:
                    ready.append(other)
        ready.sort()
    return tuple(order), tuple(sorted(str(s) for s in remaining))


def validate_structure(p: PipelineDefinition) -> tuple[ValidationIssue, ...]:
    issues: list[ValidationIssue] = []
    if p.schema_version not in SUPPORTED_PIPELINE_SCHEMAS:
        issues.append(
            ValidationIssue("UNSUPPORTED_SCHEMA", f"schema {p.schema_version!r}", str(p.ref))
        )
    seen: set[StageId] = set()
    for s in p.stages:
        if s.stage_id in seen:
            issues.append(ValidationIssue("DUPLICATE_STAGE", "duplicate stage id", str(s.stage_id)))
        seen.add(s.stage_id)
        for group, ports in (("input", s.inputs), ("output", s.outputs)):
            names = [x.name for x in ports]
            if len(set(names)) != len(names):
                issues.append(
                    ValidationIssue("DUPLICATE_PORT", f"duplicate {group} port", str(s.stage_id))
                )
        if s.model_slot is not None and (len(s.inputs) != 1 or len(s.outputs) != 1):
            issues.append(
                ValidationIssue(
                    "MODEL_STAGE_ARITY",
                    "a model-slot stage must have exactly one input and one output",
                    str(s.stage_id),
                )
            )
    slots = [s.model_slot for s in p.stages if s.model_slot]
    if len(set(slots)) != len(slots):
        issues.append(ValidationIssue("DUPLICATE_MODEL_SLOT", "model slots must be unique"))

    incoming: dict[PortRef, int] = {}
    for e in p.edges:
        loc = f"{e.source}->{e.target}"
        src = _resolve_port(p, e.source, as_source=True)
        dst = _resolve_port(p, e.target, as_source=False)
        if src is None:
            issues.append(ValidationIssue("UNKNOWN_PORT", f"unknown source {e.source}", loc))
        if dst is None:
            issues.append(ValidationIssue("UNKNOWN_PORT", f"unknown target {e.target}", loc))
        if src is None or dst is None:
            continue
        incoming[e.target] = incoming.get(e.target, 0) + 1
        if src.kind != dst.kind:
            issues.append(
                ValidationIssue("KIND_MISMATCH", f"{src.kind} cannot feed {dst.kind}", loc)
            )
        if src.spatial_kind and dst.spatial_kind and src.spatial_kind != dst.spatial_kind:
            issues.append(
                ValidationIssue(
                    "SPATIAL_KIND_MISMATCH",
                    f"{src.spatial_kind.value} cannot feed {dst.spatial_kind.value}",
                    loc,
                )
            )
        if src.feature_dim and dst.feature_dim and src.feature_dim != dst.feature_dim:
            issues.append(
                ValidationIssue(
                    "FEATURE_DIM_MISMATCH", f"dim {src.feature_dim} != {dst.feature_dim}", loc
                )
            )
    for s in p.stages:
        for port in s.inputs:
            n = incoming.get(PortRef(s.stage_id, port.name), 0)
            if n == 0:
                issues.append(
                    ValidationIssue("UNCONNECTED_INPUT", f"input {port.name!r}", str(s.stage_id))
                )
            elif n > 1:
                issues.append(
                    ValidationIssue("MULTIPLE_SOURCES", f"input {port.name!r}", str(s.stage_id))
                )
    for name, ref in p.outputs:
        if _resolve_port(p, ref, as_source=True) is None or ref.stage is None:
            issues.append(ValidationIssue("UNKNOWN_OUTPUT", f"output {name!r} -> {ref}"))
    _, cyclic = _toposort(p)
    if cyclic:
        issues.append(ValidationIssue("CYCLE", "stages form a cycle: " + ", ".join(cyclic)))
    return _sorted(issues)


def require_valid_structure(p: PipelineDefinition) -> None:
    issues = validate_structure(p)
    if issues:
        raise PipelineValidationError(issues)


@dataclass(frozen=True)
class BindingContext:
    """What a concrete execution would bind to a pipeline."""

    models: Mapping[str, ModelDefinition]
    preprocessing: PreprocessingContract | None
    modality: ImageModality
    source_channel_order: ChannelOrder | None = None
    source_geometry: SpatialReference | None = None
    budget: ResourceBudget = field(default_factory=ResourceBudget)


def _effective_out(stage: StageDefinition, ctx: BindingContext) -> PortSpec | None:
    if stage.model_slot and stage.model_slot in ctx.models:
        return ctx.models[stage.model_slot].output
    return stage.outputs[0] if len(stage.outputs) == 1 else None


def validate_binding(p: PipelineDefinition, ctx: BindingContext) -> tuple[ValidationIssue, ...]:
    issues = list(validate_structure(p))
    if issues:
        return _sorted(issues)

    for port in p.inputs:
        if port.spatial_kind and not ctx.modality.allows(port.spatial_kind):
            issues.append(
                ValidationIssue(
                    "UNSUPPORTED_MODALITY",
                    f"pipeline input {port.name!r} expects {port.spatial_kind.value}; "
                    f"modality {ctx.modality.code!r} does not provide it",
                )
            )
    if ctx.source_geometry is not None:
        for port in p.inputs:
            if port.spatial_kind and port.spatial_kind != ctx.source_geometry.spatial_kind:
                issues.append(
                    ValidationIssue(
                        "WRONG_DIMENSIONS",
                        f"input {port.name!r} expects {port.spatial_kind.value}, image is "
                        f"{ctx.source_geometry.spatial_kind.value}",
                    )
                )

    slots = set(p.model_slots)
    for unused in sorted(set(ctx.models) - slots):
        issues.append(ValidationIssue("UNUSED_MODEL_BINDING", f"slot {unused!r} not in pipeline"))

    needs_preprocessing = False
    for s in p.stages:
        loc = str(s.stage_id)
        needs_preprocessing |= s.requires_preprocessing
        if s.model_slot is None:
            continue
        model = ctx.models.get(s.model_slot)
        if model is None:
            issues.append(ValidationIssue("MISSING_MODEL_BINDING", f"slot {s.model_slot!r}", loc))
            continue
        if model.required_preprocessing is not None:
            needs_preprocessing = True
        if model.input.kind != s.inputs[0].kind:
            issues.append(
                ValidationIssue(
                    "UNSUPPORTED_MODEL_INPUT",
                    f"{model.ref} consumes {model.input.kind}, stage provides {s.inputs[0].kind}",
                    loc,
                )
            )
        if model.output.kind != s.outputs[0].kind:
            issues.append(
                ValidationIssue(
                    "MODEL_OUTPUT_MISMATCH",
                    f"{model.ref} produces {model.output.kind}, stage declares {s.outputs[0].kind}",
                    loc,
                )
            )
        if ctx.modality.code not in model.compatible_modalities:
            issues.append(
                ValidationIssue(
                    "UNSUPPORTED_MODALITY",
                    f"{model.ref} is not compatible with modality {ctx.modality.code!r}",
                    loc,
                )
            )
        if (
            model.required_preprocessing is not None
            and ctx.preprocessing is not None
            and (model.required_preprocessing != ctx.preprocessing.ref)
        ):
            issues.append(
                ValidationIssue(
                    "PREPROCESSING_MISMATCH",
                    f"{model.ref} requires {model.required_preprocessing.contract_id}@"
                    f"{model.required_preprocessing.version}, bound "
                    f"{ctx.preprocessing.contract_id}@{ctx.preprocessing.version}",
                    loc,
                )
            )
        if model.required_upstream_model is not None:
            bound = {m.ref for m in ctx.models.values()}
            if model.required_upstream_model not in bound:
                issues.append(
                    ValidationIssue(
                        "UPSTREAM_MODEL_MISMATCH",
                        f"{model.ref} requires upstream {model.required_upstream_model}",
                        loc,
                    )
                )

    # Feature dimensions along edges, using the bound models' effective ports.
    for e in p.edges:
        if e.source.stage is None or e.target.stage is None:
            continue
        src_stage = p.stage(e.source.stage)
        dst_stage = p.stage(e.target.stage)
        assert src_stage is not None and dst_stage is not None
        src = _effective_out(src_stage, ctx)
        dst: PortSpec | None = (
            ctx.models[dst_stage.model_slot].input
            if dst_stage.model_slot and dst_stage.model_slot in ctx.models
            else dst_stage.input(e.target.port)
        )
        if (
            src
            and dst
            and src.feature_dim
            and dst.feature_dim
            and (src.feature_dim != dst.feature_dim)
        ):
            issues.append(
                ValidationIssue(
                    "FEATURE_DIM_MISMATCH",
                    f"{e.source} produces dim {src.feature_dim}, {e.target} expects "
                    f"{dst.feature_dim}",
                    f"{e.source}->{e.target}",
                )
            )

    if needs_preprocessing:
        pre = ctx.preprocessing
        if pre is None:
            issues.append(
                ValidationIssue("MISSING_PREPROCESSING", "no preprocessing contract is bound")
            )
        else:
            if pre.spatial_kind not in {port.spatial_kind for port in p.inputs}:
                issues.append(
                    ValidationIssue(
                        "PREPROCESSING_MISMATCH",
                        f"contract is for {pre.spatial_kind.value}, pipeline input differs",
                    )
                )
            for missing in pre.missing_fields():
                issues.append(
                    ValidationIssue("MISSING_PREPROCESSING", f"contract does not declare {missing}")
                )
            if pre.model_channel_order is not None and (
                ctx.source_channel_order is not None or ctx.source_geometry is not None
            ):
                try:
                    pre.resolve_channel_conversion(ctx.source_channel_order)
                except (ChannelOrderMismatchError, MissingPreprocessingError) as exc:
                    issues.append(ValidationIssue(exc.code, str(exc)))
            if (
                pre.target_pixel_spacing_um is not None
                and isinstance(ctx.source_geometry, PyramidalGeometry)
                and ctx.source_geometry.pixel_spacing is None
            ):
                issues.append(
                    ValidationIssue(
                        "MISSING_SPATIAL_METADATA",
                        "contract resamples to a physical spacing but the image has none",
                    )
                )

    gpu = max((s.resources.gpu_count for s in p.stages), default=0)
    mem = max((s.resources.memory_mb for s in p.stages), default=0)
    if gpu > ctx.budget.gpu_count:
        issues.append(
            ValidationIssue(
                "RESOURCE_UNAVAILABLE", f"needs {gpu} GPU(s), budget allows {ctx.budget.gpu_count}"
            )
        )
    if mem > ctx.budget.memory_mb:
        issues.append(
            ValidationIssue(
                "RESOURCE_UNAVAILABLE", f"needs {mem} MB, budget allows {ctx.budget.memory_mb}"
            )
        )
    return _sorted(issues)


def require_valid_binding(p: PipelineDefinition, ctx: BindingContext) -> None:
    issues = validate_binding(p, ctx)
    if issues:
        raise PipelineValidationError(issues)
