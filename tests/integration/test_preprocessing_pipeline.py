"""SPEC-004 pipeline through `PipelineRunner` on in-memory images with known content.

Covers: end-to-end execution, typed outputs, determinism and artifact identity, pixel correctness
against the reader, capability gating (AC-018), negative paths, budgets, per-run overrides,
manifest serialization, and that the synthetic pipelines are untouched.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from application.preprocessing_service import WSI_PREPROCESSING_TASK
from application.runner import ExecutionRequest
from bootstrap import Platform, build_platform
from domain.errors import (
    ExecutionUnsupportedError,
    InvalidManifestError,
    NoAcquiredDataError,
    NoPatchesSelectedError,
    NoTissueDetectedError,
    PatchBudgetExceededError,
    PipelineValidationError,
    PreprocessingBudgetError,
    PreprocessingCapabilityError,
    ReaderClosedError,
    UnsupportedResolutionError,
)
from domain.image import ChannelOrder
from domain.kinds import IMAGE_REGION_2D, PATCH_MANIFEST, PATCH_PLAN, TISSUE_MASK_2D
from domain.manifest import PixelStorage
from domain.patching import PatchPlan, PatchStatus, RejectionReason
from domain.provenance import RunStatus
from domain.reading import NoDataSemantics, PixelRegion, ReadLimits
from domain.tissue import TissueMask
from domain.validation import CapabilityStatus, UsageMode
from infrastructure.runtime import FixedClock
from plugins.preprocessing.operations import PATCH_EXTRACT, PATCH_PLAN_OP, TISSUE_DETECT
from ports import PatchManifestPayload
from presentation.api import preprocessing_v1
from presentation.api.contracts import load_schema
from tests.conftest import REPO_ROOT
from tests.preprocessing_support import (
    PIPELINE,
    SMALL,
    ArrayImage,
    ArrayReader,
    run_preprocessing,
    slide_rgb,
)

pytestmark = pytest.mark.filterwarnings("error")
TISSUE_CELLS = {(r, c) for r in (1, 2, 3) for c in (2, 3, 4, 5)}


@pytest.fixture
def platform() -> Platform:
    return build_platform(REPO_ROOT, clock=FixedClock())


def image(name: str = "syn-1", **kwargs: object) -> ArrayImage:
    rgb, acq = slide_rgb()
    return ArrayImage(name, rgb, acquired=acq, **kwargs)  # type: ignore[arg-type]


def payloads(outcome) -> tuple[TissueMask, PatchPlan, PatchManifestPayload]:  # type: ignore[no-untyped-def]
    mask = outcome.outputs["mask"].payload
    plan = outcome.outputs["plan"].payload
    manifest = outcome.outputs["manifest"].payload
    assert isinstance(mask, TissueMask) and isinstance(plan, PatchPlan)
    assert isinstance(manifest, PatchManifestPayload)
    return mask, plan, manifest


# --------------------------------------------------------------------------- end to end


def test_real_preprocessing_pipeline_runs_through_the_runner(platform: Platform) -> None:
    outcome = run_preprocessing(platform, image(), SMALL)
    mask, plan, payload = payloads(outcome)
    run = outcome.run
    assert run.status is RunStatus.SUCCEEDED and outcome.prediction is None
    assert outcome.decision.status is CapabilityStatus.SUPPORTED_UNVALIDATED
    assert run.usage_mode is UsageMode.RESEARCH_UNVALIDATED
    assert {c.name for c in run.stage_versions if c.role == "operation"} == {
        str(TISSUE_DETECT),
        str(PATCH_PLAN_OP),
        str(PATCH_EXTRACT),
    }
    kinds = [a.kind for a in run.artifacts]
    assert (
        kinds.count(TISSUE_MASK_2D) == kinds.count(PATCH_PLAN) == kinds.count(PATCH_MANIFEST) == 1
    )
    assert kinds.count(IMAGE_REGION_2D) == 12
    for identity in run.artifacts:
        assert platform.artifacts.exists(identity.artifact_id)

    assert mask.grid.source_level == 0 and mask.validity_semantics.value == "acquisition_coverage"
    assert plan.resolved.level == 1 and plan.resolved.effective_spacing_um == (0.5, 0.5)
    counts = plan.counts()
    assert counts["candidates"] == 48 and counts["selected"] == 12
    assert {(d.candidate.row, d.candidate.col) for d in plan.selected} == TISSUE_CELLS
    bottom = [d for d in plan.decisions if d.candidate.row == 5]
    assert all(RejectionReason.INSUFFICIENT_ACQUIRED_DATA in d.reasons for d in bottom)
    assert all(d.acquired_fraction == 0.0 for d in bottom)

    manifest = payload.manifest
    assert manifest.pipeline == PIPELINE and manifest.pixel_storage is PixelStorage.IN_MEMORY
    assert [p.plan_index for p in manifest.patches] == [d.candidate.index for d in plan.selected]
    assert manifest.mask_digest == mask.mask_digest and manifest.plan_digest == plan.plan_digest
    assert dict(manifest.counts)["rejected.insufficient_tissue"] == 36
    assert "not tumour probabilities" in manifest.notice


def test_extracted_pixels_match_direct_reader_reads(platform: Platform) -> None:
    img = image()
    outcome = run_preprocessing(platform, img, SMALL)
    _, _, payload = payloads(outcome)
    assert len(payload.artifacts) == 12
    for record, (identity, region) in zip(payload.manifest.patches, payload.artifacts, strict=True):
        assert isinstance(region, PixelRegion) and identity.artifact_id == record.artifact_id
        lr = record.level_rect
        assert lr.level == 1 and (record.width_px, record.height_px) == (64, 64)
        assert record.level0.width == record.level0.height == 128
        direct = img._pixels[1][int(lr.y) : int(lr.y) + 64, int(lr.x) : int(lr.x) + 64]
        assert np.array_equal(np.asarray(region.pixels), direct)
        assert record.channel_order is ChannelOrder.RGB and record.dtype == "uint8"
        assert record.read_acquired_fraction == 1.0 and record.tissue_fraction == 1.0


def test_runs_are_deterministic_and_identities_stable(platform: Platform) -> None:
    a = run_preprocessing(platform, image(), SMALL)
    b = run_preprocessing(build_platform(REPO_ROOT, clock=FixedClock()), image(), SMALL)
    ma, pa, xa = payloads(a)
    mb, pb, xb = payloads(b)
    assert ma.mask_digest == mb.mask_digest
    assert pa == pb and pa.plan_digest == pb.plan_digest
    assert xa.manifest == xb.manifest
    assert xa.manifest.manifest_digest == xb.manifest.manifest_digest
    assert [i.artifact_id for i in a.run.artifacts] == [i.artifact_id for i in b.run.artifacts]
    assert a.run.fingerprint.value == b.run.fingerprint.value
    # A different image identity changes every derived identity.
    c = run_preprocessing(platform, image("syn-2"), SMALL)
    assert payloads(c)[2].manifest.manifest_digest != xa.manifest.manifest_digest


def test_config_overrides_change_identity_and_are_validated(platform: Platform) -> None:
    base = run_preprocessing(platform, image(), SMALL)
    other = run_preprocessing(platform, image(), (*SMALL, ("plan", "min_tissue_fraction", 0.25)))
    assert base.run.fingerprint.value != other.run.fingerprint.value
    assert payloads(base)[2].manifest.manifest_digest != payloads(other)[2].manifest.manifest_digest
    for bad in (
        ("nope", "width", 64),
        ("plan", "widht", 64),
        ("plan", "width", "64"),
        ("plan", "selection", 3),
    ):
        with pytest.raises(PipelineValidationError) as err:
            run_preprocessing(platform, image(), (bad,))
        assert err.value.codes == ("INVALID_CONFIG_OVERRIDE",)


def test_manifest_only_mode_reads_no_patch_pixels(platform: Platform) -> None:
    img = image()
    outcome = run_preprocessing(platform, img, (*SMALL, ("extract", "materialize_pixels", False)))
    manifest = payloads(outcome)[2].manifest
    assert manifest.pixel_storage is PixelStorage.NOT_MATERIALIZED
    assert all(p.artifact_id is None and p.read_validity is None for p in manifest.patches)
    assert all(r.level == 0 for r in img.requests)  # only mask-source reads happened
    assert IMAGE_REGION_2D not in {a.kind for a in outcome.run.artifacts}


def test_edge_policy_and_overlap_through_the_runner(platform: Platform) -> None:
    plan = payloads(
        run_preprocessing(
            platform,
            image(),
            (
                ("plan", "width", 96),
                ("plan", "height", 96),
                ("plan", "stride_x", 48.0),
                ("plan", "edge_policy", "align_to_edge"),
                ("plan", "min_tissue_fraction", 0.0),
                ("plan", "min_acquired_fraction", 0.0),
                ("plan", "selection", "row_major_first"),
            ),
        )
    )[1]
    xs = sorted({d.candidate.level_rect.x for d in plan.decisions})
    assert xs[:3] == [0, 48, 96] and xs[-1] == 512 - 96
    assert max(d.candidate.level_rect.y for d in plan.decisions) == 384 - 96


# --------------------------------------------------------------------------- AC-018 gating


def test_unsupported_channel_representation_is_rejected_before_any_read(platform: Platform) -> None:
    gray = np.full((256, 256, 1), 200, dtype=np.uint8)
    img = ArrayImage("gray", gray, channel_order=ChannelOrder.GRAYSCALE, fmt="test-gray", levels=2)
    with pytest.raises(PreprocessingCapabilityError) as err:
        run_preprocessing(platform, img, SMALL)
    assert "channel_representation" in err.value.requirements
    assert img.requests == []


def test_unknown_validity_requires_an_explicit_policy(platform: Platform) -> None:
    img = image(no_data=NoDataSemantics.NOT_REPORTED, fmt="test-unknown")
    with pytest.raises(PreprocessingCapabilityError) as err:
        run_preprocessing(platform, img, SMALL)
    assert err.value.requirements == ("acquisition_validity",)
    assert img.requests == []
    assumed = (
        *SMALL,
        ("tissue", "validity_policy", "assume_acquired"),
        ("plan", "validity_policy", "assume_acquired"),
    )
    mask, plan, payload = payloads(
        run_preprocessing(
            platform, image(no_data=NoDataSemantics.NOT_REPORTED, fmt="test-unknown"), assumed
        )
    )
    assert mask.acquired is None and mask.acquired_fraction is None
    assert all(d.acquired_fraction is None for d in plan.decisions)
    assert "assume_acquired" in plan.validity_note
    checks = {c.requirement: c.status.value for c in payload.manifest.capability_checks}
    assert checks["acquisition_validity"] == "decided_by_policy"


def test_missing_spacing_fails_for_physical_requests_but_not_for_explicit_levels(
    platform: Platform,
) -> None:
    with pytest.raises(PreprocessingCapabilityError) as err:
        run_preprocessing(platform, image(spacing_um=None, fmt="test-nospacing"), SMALL)
    assert err.value.requirements == ("physical_spacing",)
    explicit = (*SMALL, ("plan", "level", 1), ("plan", "target_spacing_um", 0.0))
    _, plan, payload = payloads(
        run_preprocessing(platform, image(spacing_um=None, fmt="test-nospacing"), explicit)
    )
    assert plan.resolved.effective_spacing_um is None
    assert any("physical spacing unknown" in x for x in payload.manifest.limitations)


def test_multi_plane_and_missing_region_support_are_gated(platform: Platform) -> None:
    with pytest.raises(PreprocessingCapabilityError) as err:
        run_preprocessing(platform, image(z_planes=2, fmt="test-z"), SMALL)
    assert "single_plane" in err.value.requirements
    platform.readers.register(ArrayReader("test-noregions", supports_regions=False))
    with pytest.raises(PreprocessingCapabilityError) as err:
        run_preprocessing(platform, image(fmt="test-noregions"), SMALL)
    assert "region_reads" in err.value.requirements


def test_unsupported_resolution_is_rejected(platform: Platform) -> None:
    with pytest.raises(UnsupportedResolutionError):
        run_preprocessing(platform, image(), (*SMALL, ("plan", "target_spacing_um", 0.35)))


def test_gated_pipeline_without_source_is_not_executable(platform: Platform) -> None:
    img = image()
    platform.readers.register(ArrayReader())
    req = ExecutionRequest(
        pipeline=platform.pipelines.get(PIPELINE),
        image=img.describe().asset,
        task=WSI_PREPROCESSING_TASK,
        models={},
        preprocessing=None,
        research_use_acknowledged=True,
    )
    with pytest.raises(ExecutionUnsupportedError) as err:
        platform.runner.run(req)
    assert "source:opened_image" in err.value.missing


def test_closed_or_mismatched_source_is_rejected(platform: Platform) -> None:
    img = image()
    img.close()
    with pytest.raises(ReaderClosedError):
        run_preprocessing(platform, img, SMALL)
    a, b = image("a"), image("b")
    platform.readers.register(ArrayReader())
    req = ExecutionRequest(
        pipeline=platform.pipelines.get(PIPELINE),
        image=a.describe().asset,
        task=WSI_PREPROCESSING_TASK,
        models={},
        preprocessing=None,
        research_use_acknowledged=True,
        source=b,
    )
    with pytest.raises(ExecutionUnsupportedError, match="does not match"):
        platform.runner.run(req)


# --------------------------------------------------------------------------- empty results


def test_empty_results_say_why(platform: Platform) -> None:
    rgb, acq = slide_rgb(tissue=())
    with pytest.raises(NoTissueDetectedError):
        run_preprocessing(platform, ArrayImage("glass", rgb, acquired=acq), SMALL)
    rgb, acq = slide_rgb(unacquired=((0, 0, 1024, 768),))
    with pytest.raises(NoAcquiredDataError):
        run_preprocessing(platform, ArrayImage("void", rgb, acquired=acq), SMALL)
    rgb, acq = slide_rgb(tissue=((256, 128, 40, 40),))  # tissue smaller than any patch
    with pytest.raises(NoPatchesSelectedError) as err:
        run_preprocessing(platform, ArrayImage("speck", rgb, acquired=acq), SMALL)
    assert type(err.value) is NoPatchesSelectedError and "rejected" in str(err.value)
    failed = [r for r in platform.provenance.all() if r.status is RunStatus.FAILED]
    assert {r.error_code for r in failed} == {
        "NO_TISSUE_DETECTED",
        "NO_ACQUIRED_DATA",
        "NO_PATCHES_SELECTED",
    }


# --------------------------------------------------------------------------- budgets


def test_patch_count_budget_fails_closed_or_selects_explicitly(platform: Platform) -> None:
    with pytest.raises(PatchBudgetExceededError):
        run_preprocessing(platform, image(), (*SMALL, ("plan", "max_patches", 5)))
    plan = payloads(
        run_preprocessing(
            platform,
            image(),
            (*SMALL, ("plan", "max_patches", 5), ("plan", "selection", "uniform_stride")),
        )
    )[1]
    assert plan.counts()["selected"] == 5 and plan.counts()["not_selected_budget"] == 7
    assert "uniform_stride" in plan.selection_note
    with pytest.raises(PatchBudgetExceededError):
        run_preprocessing(platform, image(), (*SMALL, ("extract", "max_patches", 3)))
    with pytest.raises(PatchBudgetExceededError, match="max_candidates"):
        run_preprocessing(platform, image(), (*SMALL, ("plan", "max_candidates", 10)))


def test_memory_budgets_are_enforced_before_reading(platform: Platform) -> None:
    img = image()
    with pytest.raises(PreprocessingBudgetError, match="max_output_bytes"):
        run_preprocessing(platform, img, (*SMALL, ("extract", "max_output_bytes", 100_000)))
    assert all(r.level == 0 for r in img.requests)  # no patch was read
    with pytest.raises(PreprocessingBudgetError, match="max_batch_bytes"):
        run_preprocessing(platform, image(), (*SMALL, ("extract", "max_batch_bytes", 10_000)))
    with pytest.raises(PreprocessingBudgetError, match="max_decoded_pixels"):
        run_preprocessing(platform, image(), (*SMALL, ("tissue", "max_decoded_pixels", 1000)))


def test_reads_are_bounded_tiles_and_never_a_full_level_zero(platform: Platform) -> None:
    rgb, acq = slide_rgb()
    img = ArrayImage("tiles", rgb, acquired=acq, levels=1, limits=ReadLimits(300, 300, 90_000))
    outcome = run_preprocessing(
        platform,
        img,
        (
            ("plan", "width", 64),
            ("plan", "height", 64),
            ("plan", "level", 0),
            ("plan", "target_spacing_um", 0.0),
            ("tissue", "tile_px", 256),
            ("tissue", "max_mask_pixels", 50_000),
        ),
    )
    mask = payloads(outcome)[0]
    assert mask.grid.block_factor == 4 and mask.decoded_pixels == 1024 * 768
    assert max(r.width * r.height for r in img.requests) <= 256 * 256
    assert not any(r.width == 1024 and r.height == 768 for r in img.requests)


# --------------------------------------------------------------------------- manifest


def test_manifest_round_trips_validates_and_rejects_tampering(platform: Platform) -> None:
    manifest = payloads(run_preprocessing(platform, image(), SMALL))[2].manifest
    text = preprocessing_v1.manifest_to_json(manifest)
    assert text == preprocessing_v1.manifest_to_json(manifest)
    doc = json.loads(text)
    validator = Draft202012Validator(load_schema("preprocessing", "v1", "patch-manifest"))
    assert [e.message for e in validator.iter_errors(doc)] == []
    assert preprocessing_v1.manifest_from_dict(doc) == manifest
    tampered = json.loads(text)
    tampered["patches"][0]["level0"]["x"] += 1
    with pytest.raises(InvalidManifestError, match="digest"):
        preprocessing_v1.manifest_from_dict(tampered)
    broken = json.loads(text)
    del broken["patches"][0]["patch_id"]
    with pytest.raises(InvalidManifestError):
        preprocessing_v1.manifest_from_dict(broken)
    assert list(validator.iter_errors(broken))
    with pytest.raises(InvalidManifestError):
        preprocessing_v1.manifest_from_dict({**json.loads(text), "schema_version": "x/2"})
    with pytest.raises(InvalidManifestError, match="selected count"):
        replace(manifest, patches=manifest.patches[:-1])


def test_synthetic_pipelines_are_unchanged(platform: Platform) -> None:
    """The synthetic operations still run on the synthetic reader without a source."""
    names = {str(o.operation) for o in platform.operations.values()}
    assert {"tissue.threshold@1", "patch.grid@1", "model.invoke@1"} <= names
    for ns, project in sorted(platform.project_definitions.items()):
        image_id = platform.datasets.version(project.dataset).split("test")[0]
        outcome = platform.projects.run_image(project, image_id)
        assert outcome.run.status is RunStatus.SUCCEEDED, ns
        assert all(
            c.name not in {str(TISSUE_DETECT), str(PATCH_PLAN_OP), str(PATCH_EXTRACT)}
            for c in outcome.run.stage_versions
        )


def test_patch_statuses_are_complete(platform: Platform) -> None:
    plan = payloads(run_preprocessing(platform, image(), SMALL))[1]
    assert {d.status for d in plan.decisions} == {PatchStatus.SELECTED, PatchStatus.REJECTED}
