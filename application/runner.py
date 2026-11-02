"""Pipeline execution service: validate, authorise, execute, record provenance.

Order of checks (fail closed at each step):
1. structural + binding validation (kinds, dims, modality, preprocessing, resources);
2. capability resolution and use authorisation (validated vs research opt-in);
3. executability (registered operations, model plugins and a reader);
4. capability gating of operations that need an opened image (SPEC-004 AC-018);
5. execution in deterministic topological order;
6. provenance record (also written for failed executions).

A run on a real image carries the opened image handle (`ExecutionRequest.source`, a port type).
The caller owns its lifecycle; the runner never imports, opens or closes a reader backend.
Likewise `ExecutionRequest.extras` carries caller-owned runtime services (e.g. a durable store);
they reach stages through `StageContext.extras` and never influence fingerprints.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace

from application.registries import (
    InMemoryArtifactStore,
    InMemoryProvenanceStore,
    OperationRegistry,
    ReaderRegistry,
)
from application.version import __version__
from domain.capabilities import require_capabilities
from domain.dataset import DatasetVersion
from domain.errors import (
    DomainError,
    ExecutionUnsupportedError,
    PipelineValidationError,
    ValidationIssue,
)
from domain.hashing import digest, short_digest
from domain.ids import PredictionId, ProjectId, RunId, StageId
from domain.image import UNKNOWN_CONTEXT, ClinicalContext, ImageAsset
from domain.kinds import DataKind
from domain.model import ModelDefinition, ModelRef
from domain.pipeline import (
    ConfigValue,
    PipelineDefinition,
    PortRef,
    ResourceBudget,
    StageDefinition,
)
from domain.pipeline_validation import (
    BindingContext,
    require_valid_binding,
    topological_order,
)
from domain.prediction import PredictionResult
from domain.preprocessing import PreprocessingContract, TransformStep
from domain.provenance import (
    ArtifactIdentity,
    ComponentVersion,
    ExecutionFingerprint,
    ModelArtifactRecord,
    PipelineRun,
    RunStatus,
)
from domain.task import TaskDefinition
from domain.validation import (
    CapabilityDecision,
    CapabilityQuery,
    CapabilityStatus,
    UsageMode,
    authorize_use,
    resolve_capability,
)
from ports import (
    CapabilityGated,
    Clock,
    DataValue,
    EnvironmentProbe,
    ImageReader,
    ModelRegistry,
    OpenedImage,
    PatchManifestPayload,
    PredictionPayload,
    StageContext,
)

PACKAGE_COMPONENT = ComponentVersion("package", "pathology-ai", __version__)


@dataclass(frozen=True)
class ExecutionRequest:
    pipeline: PipelineDefinition
    image: ImageAsset
    task: TaskDefinition
    models: Mapping[str, ModelRef]
    preprocessing: PreprocessingContract | None
    dataset: DatasetVersion | None = None
    project_id: ProjectId | None = None
    seed: int = 0
    research_use_acknowledged: bool = False
    target_context: ClinicalContext = UNKNOWN_CONTEXT
    budget: ResourceBudget = field(default_factory=ResourceBudget)
    # The opened real image (SPEC-004); `None` for in-memory/synthetic readers.
    source: OpenedImage | None = None
    # Per-execution stage configuration: (stage id, existing config key, value).
    config_overrides: tuple[tuple[str, str, ConfigValue], ...] = ()
    # Runtime-local services handed to every stage as `StageContext.extras` (e.g. the durable
    # embedding store of SPEC-005). Never part of the fingerprint or of any artifact identity.
    extras: Mapping[str, object] = field(default_factory=dict)


def _same_type(old: ConfigValue, new: ConfigValue) -> bool:
    if isinstance(old, bool) or isinstance(new, bool):
        return isinstance(old, bool) and isinstance(new, bool)
    if isinstance(old, float):
        return isinstance(new, int | float)
    return type(old) is type(new)


def effective_stages(req: ExecutionRequest) -> dict[str, StageDefinition]:
    """Stages with per-execution overrides applied. Only existing keys of the same type may be
    overridden, so a typo or a type change can never silently add configuration."""
    stages = {str(s.stage_id): s for s in req.pipeline.stages}
    merged: dict[str, dict[str, ConfigValue]] = {k: s.config_dict() for k, s in stages.items()}
    issues: list[ValidationIssue] = []
    seen: set[tuple[str, str]] = set()
    for stage_id, key, value in req.config_overrides:
        loc = f"{stage_id}.{key}"
        if (stage_id, key) in seen:
            issues.append(ValidationIssue("INVALID_CONFIG_OVERRIDE", "duplicate override", loc))
        seen.add((stage_id, key))
        if stage_id not in stages:
            issues.append(ValidationIssue("INVALID_CONFIG_OVERRIDE", "no such stage", loc))
        elif key not in merged[stage_id]:
            issues.append(ValidationIssue("INVALID_CONFIG_OVERRIDE", "no such config key", loc))
        elif not _same_type(merged[stage_id][key], value):
            issues.append(
                ValidationIssue(
                    "INVALID_CONFIG_OVERRIDE",
                    f"{type(value).__name__} does not match {type(merged[stage_id][key]).__name__}",
                    loc,
                )
            )
        else:
            merged[stage_id][key] = value
    if issues:
        raise PipelineValidationError(tuple(sorted(issues, key=ValidationIssue.sort_key)))
    return {
        k: replace(s, config=tuple(sorted(merged[k].items()))) if req.config_overrides else s
        for k, s in stages.items()
    }


@dataclass(frozen=True)
class ExecutionOutcome:
    run: PipelineRun
    outputs: Mapping[str, DataValue]
    prediction: PredictionResult | None
    decision: CapabilityDecision


def combine_decisions(decisions: Mapping[str, CapabilityDecision]) -> CapabilityDecision:
    """INCOMPATIBLE dominates; VALIDATED only when every predictive model is validated."""
    if not decisions:
        return CapabilityDecision(
            CapabilityStatus.SUPPORTED_UNVALIDATED, ("no predictive model in pipeline",)
        )
    reasons = tuple(f"{slot}: {r}" for slot, d in sorted(decisions.items()) for r in d.reasons)
    statuses = {d.status for d in decisions.values()}
    if CapabilityStatus.INCOMPATIBLE in statuses:
        return CapabilityDecision(CapabilityStatus.INCOMPATIBLE, reasons)
    if statuses == {CapabilityStatus.VALIDATED}:
        records = sorted(str(d.validation_record) for d in decisions.values())
        return CapabilityDecision(CapabilityStatus.VALIDATED, reasons + tuple(records))
    return CapabilityDecision(CapabilityStatus.SUPPORTED_UNVALIDATED, reasons)


class PipelineRunner:
    def __init__(
        self,
        *,
        operations: OperationRegistry,
        readers: ReaderRegistry,
        models: ModelRegistry,
        artifacts: InMemoryArtifactStore,
        provenance: InMemoryProvenanceStore,
        clock: Clock,
        environment: EnvironmentProbe,
    ) -> None:
        self._operations = operations
        self._readers = readers
        self._models = models
        self._artifacts = artifacts
        self._provenance = provenance
        self._clock = clock
        self._environment = environment
        self._counter = 0

    # ------------------------------------------------------------------ checks

    def definitions(self, req: ExecutionRequest) -> dict[str, ModelDefinition]:
        return {slot: self._models.definition(ref) for slot, ref in sorted(req.models.items())}

    def validate(self, req: ExecutionRequest) -> dict[str, ModelDefinition]:
        defs = self.definitions(req)
        effective_stages(req)
        reader = self._readers.for_image(req.image)
        source_order = req.image.channel_order or (
            reader.capabilities.channel_order if reader else None
        )
        require_valid_binding(
            req.pipeline,
            BindingContext(
                models=defs,
                preprocessing=req.preprocessing,
                modality=req.image.modality,
                source_channel_order=source_order,
                source_geometry=req.image.spatial_reference,
                budget=req.budget,
            ),
        )
        return defs

    def capability(
        self, req: ExecutionRequest, defs: Mapping[str, ModelDefinition]
    ) -> CapabilityDecision:
        decisions: dict[str, CapabilityDecision] = {}
        for slot, d in defs.items():
            if d.output_schema is None:
                continue  # feature extractors are checked technically by binding validation
            query = CapabilityQuery(
                model=d.ref,
                modality_code=req.image.modality.code,
                task=req.task.ref,
                preprocessing=req.preprocessing.ref if req.preprocessing else None,
                tissue=req.target_context.tissue,
                disease=req.target_context.disease,
                dataset=req.dataset.ref if req.dataset else None,
            )
            decisions[slot] = resolve_capability(
                d, query, self._models.validation_records(d.ref), req.task
            )
        return combine_decisions(decisions)

    def missing_for_execution(self, req: ExecutionRequest) -> tuple[str, ...]:
        missing: list[str] = []
        for stage in req.pipeline.stages:
            if stage.operation not in self._operations:
                missing.append(f"operation:{stage.operation}")
            if stage.model_slot:
                ref = req.models.get(stage.model_slot)
                if ref is not None and self._models.plugin(ref) is None:
                    missing.append(f"model_plugin:{ref}")
        if self._readers.for_image(req.image) is None:
            missing.append(
                f"reader:{req.image.image_format.code}/"
                f"{req.image.spatial_reference.spatial_kind.value}"
            )
        if req.source is None and any(
            isinstance(self._operations.find(s.operation), CapabilityGated)
            for s in req.pipeline.stages
        ):
            missing.append("source:opened_image")
        return tuple(sorted(set(missing)))

    def gate(self, req: ExecutionRequest, reader: ImageReader) -> None:
        """Check the opened image against every gated operation before any stage runs."""
        if req.source is None:
            return
        description = req.source.describe()
        if (
            description.image_id != req.image.image_id
            or description.asset.spatial_reference != req.image.spatial_reference
            or description.asset.source_checksum != req.image.source_checksum
        ):
            raise ExecutionUnsupportedError(
                f"the opened source ({description.image_id}) does not match the requested image "
                f"{req.image.image_id}",
                ("source:matching_image",),
            )
        stages = effective_stages(req)
        for sid in topological_order(req.pipeline):
            op = self._operations.get(stages[str(sid)].operation)
            if isinstance(op, CapabilityGated):
                checks = op.requirements(stages[str(sid)].config_dict()).evaluate(
                    description, supports_regions=reader.capabilities.supports_regions
                )
                require_capabilities(checks, context=f"stage {sid} ({op.operation})")

    # ------------------------------------------------------------------ execution

    def run(self, req: ExecutionRequest) -> ExecutionOutcome:
        defs = self.validate(req)
        decision = self.capability(req, defs)
        usage = authorize_use(decision, req.research_use_acknowledged)
        missing = self.missing_for_execution(req)
        if missing:
            raise ExecutionUnsupportedError(
                f"pipeline {req.pipeline.ref} is valid but not executable here: "
                + ", ".join(missing),
                missing,
            )

        reader = self._readers.for_image(req.image)
        assert reader is not None
        self.gate(req, reader)
        conversion = (
            req.preprocessing.resolve_channel_conversion(
                req.image.channel_order or reader.capabilities.channel_order
            )
            if req.preprocessing is not None and req.preprocessing.model_channel_order
            else None
        )
        started = self._clock.now_iso()
        components: set[ComponentVersion] = {PACKAGE_COMPONENT, reader.component}
        produced: list[ArtifactIdentity] = []
        values: dict[PortRef, DataValue] = {}
        image_identity = ArtifactIdentity.derive(
            req.pipeline.inputs[0].kind,
            {"image": req.image.image_id, "checksum": req.image.source_checksum},
        )
        for port in req.pipeline.inputs:
            values[PortRef(None, port.name)] = DataValue(port.kind, req.image, image_identity)

        error_code: str | None = None
        try:
            for sid in topological_order(req.pipeline):
                components |= self._run_stage(req, sid, defs, reader, conversion, values, produced)
        except DomainError as exc:
            error_code = exc.code
            raise
        except Exception:
            error_code = "INTERNAL_ERROR"
            raise
        finally:
            fingerprint = self._fingerprint(req, defs, components, reader.component)
            self._counter += 1
            run = PipelineRun(
                run_id=RunId(
                    short_digest(
                        {
                            "fp": fingerprint.value,
                            "image": req.image.image_id,
                            "t": started,
                            "n": self._counter,
                        },
                        "run",
                    )
                ),
                project_id=req.project_id,
                image_ids=(req.image.image_id,),
                fingerprint=fingerprint,
                capability_status=decision.status,
                usage_mode=usage,
                status=RunStatus.FAILED if error_code else RunStatus.SUCCEEDED,
                stage_versions=tuple(sorted(c for c in components if c.role != "package")),
                artifacts=tuple(produced),
                started_at=started,
                finished_at=self._clock.now_iso(),
                error_code=error_code,
            )
            self._provenance.record(run)

        outputs = {name: values[ref] for name, ref in req.pipeline.outputs}
        prediction = self._prediction(req, run, outputs, decision, usage)
        return ExecutionOutcome(run, outputs, prediction, decision)

    def _run_stage(
        self,
        req: ExecutionRequest,
        sid: StageId,
        defs: Mapping[str, ModelDefinition],
        reader: ImageReader,
        conversion: TransformStep | None,
        values: dict[PortRef, DataValue],
        produced: list[ArtifactIdentity],
    ) -> set[ComponentVersion]:
        stage = effective_stages(req)[str(sid)]
        op = self._operations.get(stage.operation)
        model = None
        artifact = None
        used = {op.component}
        if stage.model_slot:
            ref = req.models[stage.model_slot]
            model = self._models.plugin(ref)
            artifact = self._models.artifact(ref)
            assert model is not None
            used.add(model.component)
        inputs: dict[str, DataValue] = {}
        for e in req.pipeline.edges:
            if e.target.stage == sid:
                inputs[e.target.port] = values[e.source]
        ctx = StageContext(
            image=req.image,
            stage=stage,
            seed=req.seed,
            reader=reader,
            preprocessing=req.preprocessing,
            channel_conversion=conversion,
            model=model,
            model_artifact=artifact,
            extras=req.extras,
            source=req.source,
            pipeline=req.pipeline.ref,
        )
        raw = op.run(inputs, ctx)
        expected = (
            {stage.outputs[0].name: defs[stage.model_slot].output.kind}
            if stage.model_slot
            else {p.name: p.kind for p in stage.outputs}
        )
        if set(raw) != set(expected):
            raise ExecutionUnsupportedError(
                f"stage {sid} returned ports {sorted(raw)}, declared {sorted(expected)}"
            )
        lineage = tuple(sorted(v.identity.artifact_id for v in inputs.values()))
        producer = {
            "stage": sid,
            "operation": stage.operation,
            "config": stage.config,
            "model_checksum": artifact.checksum if artifact else None,
            "preprocessing": req.preprocessing.fingerprint if req.preprocessing else None,
            "seed": req.seed,
        }
        for name, payload in sorted(raw.items()):
            kind: DataKind = expected[name]
            identity = ArtifactIdentity.derive(kind, {**producer, "port": name}, lineage)
            values[PortRef(sid, name)] = DataValue(kind, payload, identity)
            self._artifacts.put(identity, payload)
            produced.append(identity)
            if isinstance(payload, PredictionPayload | PatchManifestPayload):
                for aux_identity, aux in payload.artifacts:
                    self._artifacts.put(aux_identity, aux)
                    produced.append(aux_identity)
        return used

    def _fingerprint(
        self,
        req: ExecutionRequest,
        defs: Mapping[str, ModelDefinition],
        components: set[ComponentVersion],
        reader_component: ComponentVersion,
    ) -> ExecutionFingerprint:
        models = tuple(
            ModelArtifactRecord(slot, d.ref, self._models.artifact(d.ref).checksum)
            for slot, d in sorted(defs.items())
        )
        return ExecutionFingerprint(
            pipeline=req.pipeline.ref,
            pipeline_digest=digest(req.pipeline),
            dataset=req.dataset.ref if req.dataset else None,
            dataset_digest=req.dataset.content_digest if req.dataset else None,
            task=req.task.ref,
            models=models,
            preprocessing=req.preprocessing.ref if req.preprocessing else None,
            preprocessing_digest=req.preprocessing.fingerprint if req.preprocessing else None,
            components=tuple(sorted(components | {reader_component})),
            environment=self._environment.capture(),
            seed=req.seed,
            config_digest=digest(
                {"budget": req.budget, "context": req.target_context}
                | ({"overrides": sorted(req.config_overrides)} if req.config_overrides else {})
            ),
        )

    def _prediction(
        self,
        req: ExecutionRequest,
        run: PipelineRun,
        outputs: Mapping[str, DataValue],
        decision: CapabilityDecision,
        usage: UsageMode,
    ) -> PredictionResult | None:
        payloads = [v.payload for v in outputs.values() if isinstance(v.payload, PredictionPayload)]
        if not payloads:
            return None
        outs = tuple(o for p in payloads for o in p.outputs)
        evidence = tuple(e for p in payloads for e in p.evidence)
        result = PredictionResult(
            prediction_id=PredictionId(short_digest({"run": run.run_id}, "pred")),
            image_id=req.image.image_id,
            task=req.task.ref,
            models=tuple(ref for _, ref in sorted(req.models.items())),
            outputs=outs,
            evidence=evidence,
            run_id=run.run_id,
            capability_status=decision.status,
            usage_mode=usage,
        )
        result.validate_against(req.image)
        for o in outs:
            if o.kind is not req.task.output_schema.kind:
                raise ExecutionUnsupportedError(
                    f"model output {o.kind.value} does not match task output "
                    f"{req.task.output_schema.kind.value}"
                )
        return result
