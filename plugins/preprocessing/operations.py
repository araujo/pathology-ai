"""Typed pipeline operations: `tissue.detect@1`, `patch.plan@1`, `patch.extract@1`.

All three read the image only through `StageContext.source` (the `OpenedImage` port) and declare
their capability requirements, which the runner checks before any stage runs.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from domain.capabilities import (
    CapabilityCheck,
    PreprocessingRequirements,
    ValidityPolicy,
    require_capabilities,
)
from domain.errors import (
    CorruptImageError,
    ExecutionUnsupportedError,
    InvalidSpatialReferenceError,
    NoAcquiredDataError,
    NoPatchesSelectedError,
    NoTissueDetectedError,
    PatchBudgetExceededError,
    PreprocessingBudgetError,
)
from domain.hashing import digest, short_digest
from domain.ids import Version
from domain.kinds import IMAGE_REGION_2D
from domain.manifest import PatchManifest, PatchRecord, PixelStorage
from domain.patching import (
    EdgePolicy,
    PatchPlan,
    PatchSpec,
    PatchUnits,
    SelectionPolicy,
    apply_selection,
    decide,
    plan_candidates,
    resolve_patch_geometry,
)
from domain.pipeline import ConfigValue, OperationRef
from domain.provenance import ArtifactIdentity, ComponentVersion
from domain.reading import ImageDescription, NoDataSemantics, RegionReadRequest
from domain.tissue import TissueMask
from plugins.preprocessing import PLUGIN_VERSION
from plugins.preprocessing.arrays import AreaFractions, mask_digest
from plugins.preprocessing.detection import (
    cfg_bool,
    cfg_float,
    cfg_int,
    cfg_str,
    choose_mask_grid,
    mask_fractions,
    read_detector_input,
    validity_policy,
)
from ports import DataValue, OpenedImage, PatchManifestPayload, StageContext, TissueDetector

TISSUE_DETECT = OperationRef("tissue.detect", Version("1"))
PATCH_PLAN_OP = OperationRef("patch.plan", Version("1"))
PATCH_EXTRACT = OperationRef("patch.extract", Version("1"))

STORAGE_LIMITATION = (
    "pixel artifacts are held in the process-local in-memory artifact store; content-addressed "
    "artifact ids are not evidence that pixel bytes were persisted"
)


def _component(ref: OperationRef) -> ComponentVersion:
    return ComponentVersion("operation", str(ref), PLUGIN_VERSION)


def _source(ctx: StageContext) -> tuple[OpenedImage, ImageDescription]:
    if ctx.source is None:
        raise ExecutionUnsupportedError(
            f"{ctx.stage.operation} needs an opened real image (ExecutionRequest.source)",
            ("source:opened_image",),
        )
    return ctx.source, ctx.source.describe()


def _checks(
    requirements: PreprocessingRequirements, description: ImageDescription, ctx: StageContext
) -> tuple[CapabilityCheck, ...]:
    supports = bool(ctx.reader and ctx.reader.capabilities.supports_regions)
    checks = requirements.evaluate(description, supports_regions=supports)
    require_capabilities(checks, context=str(ctx.stage.operation))
    return checks


# --------------------------------------------------------------------------- detection


class TissueDetectionOperation:
    """Builds the bounded mask source and delegates to a named `TissueDetector`."""

    def __init__(self, detectors: Mapping[str, TissueDetector]) -> None:
        self._detectors = dict(detectors)

    @property
    def operation(self) -> OperationRef:
        return TISSUE_DETECT

    @property
    def component(self) -> ComponentVersion:
        return _component(TISSUE_DETECT)

    def _detector(self, config: Mapping[str, ConfigValue]) -> TissueDetector:
        name = cfg_str(config, "detector", "")
        try:
            return self._detectors[name]
        except KeyError:
            known = ", ".join(sorted(self._detectors)) or "none"
            raise ExecutionUnsupportedError(
                f"tissue detector {name!r} is not registered (known: {known})",
                (f"detector:{name}",),
            ) from None

    def requirements(self, config: Mapping[str, ConfigValue]) -> PreprocessingRequirements:
        return self._detector(config).requirements(config)

    def run(self, inputs: Mapping[str, DataValue], ctx: StageContext) -> Mapping[str, object]:
        cfg = ctx.stage.config_dict()
        detector = self._detector(cfg)
        image, d = _source(ctx)
        checks = _checks(detector.requirements(cfg), d, ctx)
        grid = choose_mask_grid(d.geometry, cfg)
        data, decoded = read_detector_input(image, grid, cfg_int(cfg, "tile_px", 2048))
        result = detector.detect(data, cfg)
        tissue = np.asarray(result.tissue, dtype=np.bool_)
        acquired = None if data.acquired is None else np.asarray(data.acquired, dtype=np.float64)
        if tissue.shape != (grid.height, grid.width):
            raise CorruptImageError(f"detector returned mask {tissue.shape}, grid is {grid}")
        if acquired is not None and bool((tissue & (acquired < 0.5)).any()):
            raise CorruptImageError("detector marked unacquired cells as tissue")
        t_frac, a_frac = mask_fractions(tissue, acquired, grid)
        policy = validity_policy(cfg)
        limitations = list(result.limitations)
        dx, dy = grid.edge_adjustment
        if dx or dy:
            limitations.append(
                f"mask edge cells adjusted by ({dx:g}, {dy:g}) level-0 px to end at the image edge"
            )
        if grid.block_factor > 1:
            limitations.append(
                f"mask cells average {grid.block_factor}x{grid.block_factor} pixels of level "
                f"{grid.source_level}; a cell counts as acquired when >= 50% of it is acquired"
            )
        if d.no_data is NoDataSemantics.NOT_REPORTED:
            limitations.append("acquisition validity not reported; assumed acquired by policy")
        mask = TissueMask(
            image_id=d.image_id,
            source_integrity=d.asset.source_checksum,
            scene=d.scene,
            grid=grid,
            reader=d.reader,
            detector=detector.component,
            parameters=tuple(sorted(cfg.items())),
            derived=result.derived,
            input_channel_order=d.channel_order.value,
            channel_view=result.channel_view,
            validity_semantics=d.no_data,
            validity_policy=policy,
            tissue_fraction=t_frac,
            acquired_fraction=a_frac,
            mask_digest=mask_digest(tissue, acquired),
            capability_checks=checks,
            limitations=tuple(limitations),
            decoded_pixels=decoded,
            tissue=tissue,
            acquired=acquired,
        )
        return {"mask": mask}


# --------------------------------------------------------------------------- planning


def patch_spec(cfg: Mapping[str, ConfigValue]) -> PatchSpec:
    """`level: -1` and `target_spacing_um: 0` mean "unset"; strides of 0 mean "= size"."""
    level = cfg_int(cfg, "level", -1)
    target = cfg_float(cfg, "target_spacing_um", 0.0)
    sx, sy = cfg_float(cfg, "stride_x", 0.0), cfg_float(cfg, "stride_y", 0.0)
    try:
        return PatchSpec(
            width=cfg_float(cfg, "width", 256),
            height=cfg_float(cfg, "height", 256),
            units=PatchUnits(cfg_str(cfg, "units", "pixels")),
            stride_x=sx or None,
            stride_y=sy or None,
            level=level if level >= 0 else None,
            target_spacing_um=target or None,
            spacing_tolerance=cfg_float(cfg, "spacing_tolerance", 0.05),
            grid_origin_x=cfg_int(cfg, "grid_origin_x", 0),
            grid_origin_y=cfg_int(cfg, "grid_origin_y", 0),
            edge_policy=EdgePolicy(cfg_str(cfg, "edge_policy", "inside_only")),
            min_tissue_fraction=cfg_float(cfg, "min_tissue_fraction", 0.5),
            min_acquired_fraction=cfg_float(cfg, "min_acquired_fraction", 0.9),
            validity_policy=validity_policy(cfg),
            selection=SelectionPolicy(cfg_str(cfg, "selection", "fail_closed")),
            max_patches=cfg_int(cfg, "max_patches", 512),
            max_candidates=cfg_int(cfg, "max_candidates", 250_000),
        )
    except ValueError as exc:
        if isinstance(exc, InvalidSpatialReferenceError):
            raise
        raise InvalidSpatialReferenceError(f"invalid patch configuration: {exc}") from None


class PatchPlanOperation:
    @property
    def operation(self) -> OperationRef:
        return PATCH_PLAN_OP

    @property
    def component(self) -> ComponentVersion:
        return _component(PATCH_PLAN_OP)

    def requirements(self, config: Mapping[str, ConfigValue]) -> PreprocessingRequirements:
        spec = patch_spec(config)
        return PreprocessingRequirements(
            needs_regions=False,
            needs_physical_spacing=spec.needs_physical_spacing,
            validity_policy=spec.validity_policy,
        )

    def run(self, inputs: Mapping[str, DataValue], ctx: StageContext) -> Mapping[str, object]:
        spec = patch_spec(ctx.stage.config_dict())
        _, d = _source(ctx)
        checks = _checks(self.requirements(ctx.stage.config_dict()), d, ctx)
        mask = inputs["mask"].payload
        if not isinstance(mask, TissueMask):
            raise ExecutionUnsupportedError("patch.plan needs a TissueMask input")
        if mask.image_id != d.image_id or mask.scene != d.scene:
            raise ExecutionUnsupportedError("the tissue mask belongs to another image or scene")
        geometry = d.geometry
        resolved = resolve_patch_geometry(spec, geometry)
        candidates = plan_candidates(spec, resolved, geometry)
        fractions = AreaFractions(mask.grid)
        tissue = np.asarray(mask.tissue, dtype=np.float64)
        acquired = None if mask.acquired is None else np.asarray(mask.acquired, dtype=np.float64)
        decisions = tuple(
            decide(
                c,
                fractions.mean(tissue, c.level0),
                None if acquired is None else fractions.mean(acquired, c.level0),
                spec,
            )
            for c in candidates
        )
        decisions, note = apply_selection(decisions, spec)
        validity_note = (
            f"acquired fractions from {mask.validity_semantics.value} at mask resolution"
            if acquired is not None
            else "acquisition validity not reported; acquired-data filter not applied "
            f"(policy {ValidityPolicy.ASSUME_ACQUIRED.value})"
        )
        plan = PatchPlan(
            image_id=d.image_id,
            source_integrity=d.asset.source_checksum,
            scene=d.scene,
            planner=self.component,
            spec=spec,
            resolved=resolved,
            mask_digest=mask.mask_digest,
            mask_grid_cell=(mask.grid.cell_x, mask.grid.cell_y),
            decisions=decisions,
            selection_note=note,
            capability_checks=checks,
            validity_note=validity_note,
        )
        if not plan.selected:
            counts = ", ".join(f"{k}={v}" for k, v in sorted(plan.counts().items()) if v)
            if mask.acquired_fraction == 0.0:
                raise NoAcquiredDataError(f"the image has no acquired data ({counts})")
            if mask.tissue_fraction == 0.0:
                raise NoTissueDetectedError(f"no candidate tissue was detected ({counts})")
            raise NoPatchesSelectedError(f"every candidate was rejected ({counts})")
        return {"plan": plan}


# --------------------------------------------------------------------------- extraction


class PatchExtractionOperation:
    @property
    def operation(self) -> OperationRef:
        return PATCH_EXTRACT

    @property
    def component(self) -> ComponentVersion:
        return _component(PATCH_EXTRACT)

    def requirements(self, config: Mapping[str, ConfigValue]) -> PreprocessingRequirements:
        return PreprocessingRequirements(needs_regions=cfg_bool(config, "materialize_pixels", True))

    def run(self, inputs: Mapping[str, DataValue], ctx: StageContext) -> Mapping[str, object]:
        cfg = ctx.stage.config_dict()
        materialize = cfg_bool(cfg, "materialize_pixels", True)
        batch_size = cfg_int(cfg, "batch_size", 32)
        max_patches = cfg_int(cfg, "max_patches", 512)
        max_batch_bytes = cfg_int(cfg, "max_batch_bytes", 64 << 20)
        max_output_bytes = cfg_int(cfg, "max_output_bytes", 256 << 20)
        if min(batch_size, max_patches, max_batch_bytes, max_output_bytes) < 1:
            raise InvalidSpatialReferenceError("extraction budgets must be positive")
        image, d = _source(ctx)
        checks = _checks(self.requirements(cfg), d, ctx)
        mask, plan = inputs["mask"].payload, inputs["plan"].payload
        if not isinstance(mask, TissueMask) or not isinstance(plan, PatchPlan):
            raise ExecutionUnsupportedError("patch.extract needs a TissueMask and a PatchPlan")
        if (plan.image_id, plan.scene, plan.mask_digest) != (d.image_id, d.scene, mask.mask_digest):
            raise ExecutionUnsupportedError("the plan does not belong to this image, scene or mask")

        selected = plan.selected
        r = plan.resolved
        patch_bytes = r.width_px * r.height_px * d.samples_per_pixel * np.dtype(d.dtype).itemsize
        if len(selected) > max_patches:
            raise PatchBudgetExceededError(
                f"{len(selected)} selected patches exceed extraction max_patches={max_patches}"
            )
        if materialize and patch_bytes * min(batch_size, len(selected)) > max_batch_bytes:
            raise PreprocessingBudgetError(
                f"a batch of {batch_size} patches needs {patch_bytes * batch_size} bytes, above "
                f"max_batch_bytes={max_batch_bytes}"
            )
        if materialize and patch_bytes * len(selected) > max_output_bytes:
            raise PreprocessingBudgetError(
                f"{len(selected)} patches need {patch_bytes * len(selected)} bytes, above "
                f"max_output_bytes={max_output_bytes}"
            )

        image_artifact = inputs["image"].identity.artifact_id
        spacing = r.effective_spacing_um
        records: list[PatchRecord] = []
        artifacts: list[tuple[ArtifactIdentity, object]] = []
        for start in range(0, len(selected), batch_size):
            for decision in selected[start : start + batch_size]:
                c = decision.candidate
                lr = c.level_rect
                patch_id = short_digest(
                    {
                        "image": d.image_id,
                        "integrity": d.asset.source_checksum,
                        "scene": d.scene,
                        "level_rect": lr,
                        "level0": c.level0,
                    },
                    "patch",
                )
                read_validity = read_fraction = read_digest = artifact_id = None
                transformations: tuple[str, ...] = ()
                if materialize:
                    region = image.read_region(
                        RegionReadRequest(
                            lr.level,
                            int(lr.x),
                            int(lr.y),
                            int(lr.width),
                            int(lr.height),
                            scene=d.scene,
                        )
                    )
                    shape = getattr(region.pixels, "shape", None)
                    if shape != (r.height_px, r.width_px, d.samples_per_pixel):
                        raise CorruptImageError(f"patch {patch_id} decoded with shape {shape}")
                    if region.level0 != c.level0:
                        raise CorruptImageError(f"patch {patch_id} was read at {region.level0}")
                    identity = ArtifactIdentity.derive(
                        IMAGE_REGION_2D, {"region": region.provenance}, (image_artifact,)
                    )
                    artifacts.append((identity, region))
                    read_validity = region.valid_data.validity
                    read_fraction = region.valid_data.acquired_fraction
                    read_digest = digest(region.provenance)
                    transformations = region.provenance.decoder_transformations
                    artifact_id = identity.artifact_id
                records.append(
                    PatchRecord(
                        patch_id=patch_id,
                        plan_index=c.index,
                        row=c.row,
                        col=c.col,
                        scene=d.scene,
                        level0=c.level0,
                        level_rect=lr,
                        downsample=r.downsample,
                        width_px=r.width_px,
                        height_px=r.height_px,
                        effective_spacing_um=spacing,
                        dtype=d.dtype,
                        channel_order=d.channel_order,
                        samples_per_pixel=d.samples_per_pixel,
                        validity_semantics=d.no_data,
                        tissue_fraction=decision.tissue_fraction,
                        planned_acquired_fraction=decision.acquired_fraction,
                        read_validity=read_validity,
                        read_acquired_fraction=read_fraction,
                        read_provenance_digest=read_digest,
                        decoder_transformations=transformations,
                        artifact_id=artifact_id,
                    )
                )

        limitations = list(mask.limitations)
        if spacing is None:
            limitations.append("physical spacing unknown: patches are defined in pixels only")
        if r.spacing_relative_error:
            limitations.append(
                f"effective spacing differs from the requested {r.requested_spacing_um} µm/px by "
                f"{r.spacing_relative_error:.2%} (no resampling performed)"
            )
        limitations.append(
            STORAGE_LIMITATION if materialize else "pixels not materialized (manifest only)"
        )
        manifest = PatchManifest(
            image_id=d.image_id,
            source_integrity=d.asset.source_checksum,
            scene=d.scene,
            pipeline=ctx.pipeline,
            reader=d.reader,
            backend=d.backend,
            detector=mask.detector,
            detector_parameters=mask.parameters,
            detector_derived=mask.derived,
            mask_digest=mask.mask_digest,
            planner=plan.planner,
            extractor=self.component,
            plan_digest=plan.plan_digest,
            plan_spec=plan.spec,
            resolved=r,
            counts=tuple(sorted(plan.counts().items())),
            selection_note=plan.selection_note,
            validity_note=plan.validity_note,
            pixel_storage=PixelStorage.IN_MEMORY if materialize else PixelStorage.NOT_MATERIALIZED,
            patches=tuple(records),
            capability_checks=tuple(
                sorted(set(mask.capability_checks + plan.capability_checks + checks))
            ),
            limitations=tuple(limitations),
        )
        return {"manifest": PatchManifestPayload(manifest, tuple(artifacts))}
