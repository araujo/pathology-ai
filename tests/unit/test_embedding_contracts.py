"""SPEC-005 domain contracts: model input, transforms, resolution, batching, identity, manifests."""

from __future__ import annotations

from dataclasses import replace

import pytest

from domain.embedding import (
    BatchPolicy,
    DeclaredTransform,
    EmbeddingRecord,
    ModelInputSpec,
    Normalization,
    PixelState,
    Precision,
    ResampleMethod,
    StoreLayout,
    TensorLayout,
    TransformChain,
    build_transform_chain,
    patch_ids_digest,
    validate_model_input,
    validate_records,
)
from domain.errors import (
    ChannelOrderMismatchError,
    EmbeddingBudgetError,
    InvalidManifestError,
    MissingSpatialMetadataError,
    ModelInputContractError,
    ResolutionMismatchError,
)
from domain.geometry import Level0Rect, LevelRect
from domain.image import ChannelOrder
from domain.preprocessing import TransformStep
from plugins.embedding.reference import INPUT_SPEC
from plugins.embedding.transforms import IMPLEMENTATIONS

RGB, BGR, RGBA = ChannelOrder.RGB, ChannelOrder.BGR, ChannelOrder.RGBA
BOTH = frozenset({(BGR, RGB), (RGBA, RGB)})


def source(order: ChannelOrder = RGB, size: int = 256, spacing: float | None = 0.5) -> PixelState:
    samples = 4 if order is RGBA else 3
    return PixelState.of_patch(
        size, size, order, samples, "uint8", None if spacing is None else (spacing, spacing)
    )


def chain(
    src: PixelState,
    spec: ModelInputSpec = INPUT_SPEC,
    conversions: frozenset[tuple[ChannelOrder, ChannelOrder]] = BOTH,
    resample: ResampleMethod | None = None,
) -> TransformChain:
    return build_transform_chain(
        src,
        spec,
        channel_conversions=conversions,
        resample=resample,
        implementations=IMPLEMENTATIONS,
    )


def spaced_spec(spacing: float = 0.5, tolerance: float = 0.05) -> ModelInputSpec:
    return replace(
        INPUT_SPEC,
        pixel_spacing_um=spacing,
        spacing_tolerance=tolerance,
        accepts_variable_spacing=False,
        accepts_unknown_spacing=False,
    )


# --------------------------------------------------------------------------- ModelInputSpec


def test_reference_spec_is_explicit_about_everything() -> None:
    s = INPUT_SPEC
    assert (s.width, s.height, s.channel_order, s.channels) == (256, 256, RGB, 3)
    assert (s.dtype, s.scaled_range, s.layout) == ("float32", (0.0, 1.0), TensorLayout.CHW)
    assert s.normalization == Normalization((0.5,) * 3, (0.5,) * 3)
    assert s.pixel_spacing_um is None and s.accepts_variable_spacing
    assert s.precisions == frozenset({Precision.FP32})
    assert s.patch_bytes == 256 * 256 * 3 * 4


@pytest.mark.parametrize(
    "change",
    [
        {"channels": 4},
        {"dtype": "uint8"},
        {"width": 250},  # not divisible by the declared size_divisor (8)
        {"scaled_range": (-1.0, 1.0)},
        {"accepts_variable_spacing": False},  # neither a spacing nor variable spacing declared
        {"pixel_spacing_um": 0.5},  # a required spacing contradicts variable spacing
        {"normalization": Normalization((0.5,), (0.5,))},
        {"spacing_tolerance": 1.5},
        {"precisions": frozenset()},
    ],
)
def test_invalid_model_input_specs_are_rejected(change: dict[str, object]) -> None:
    with pytest.raises(ModelInputContractError):
        replace(INPUT_SPEC, **change)  # type: ignore[arg-type]


def test_normalization_rejects_nonpositive_std() -> None:
    with pytest.raises(ModelInputContractError):
        Normalization((0.5,), (0.0,))


def test_source_state_requires_known_numeric_range_and_matching_samples() -> None:
    with pytest.raises(ModelInputContractError):
        PixelState.of_patch(256, 256, RGB, 3, "float32", None)
    with pytest.raises(ModelInputContractError):
        PixelState.of_patch(256, 256, RGBA, 3, "uint8", None)


# --------------------------------------------------------------------------- transform chain


def test_chain_for_rgb_has_no_channel_or_spatial_step() -> None:
    c = chain(source(RGB))
    assert c.names == ("cast", "scale", "normalize", "layout")
    out = validate_model_input(source(RGB), INPUT_SPEC, c, Precision.FP32)
    assert (out.width, out.height, out.spacing_um) == (256, 256, (0.5, 0.5))


def test_bgr_needs_a_declared_reorder_and_records_it() -> None:
    c = chain(source(BGR))
    assert c.names[0] == "channel_reorder"
    step = c.steps[0]
    assert step.step.param("from") == "bgr" and step.step.param("to") == "rgb"
    assert step.implementation == IMPLEMENTATIONS["channel_reorder"]
    with pytest.raises(ChannelOrderMismatchError):
        chain(source(BGR), conversions=frozenset({(RGBA, RGB)}))


def test_rgba_needs_declared_alpha_drop() -> None:
    assert chain(source(RGBA)).names[0] == "drop_alpha"
    with pytest.raises(ChannelOrderMismatchError):
        chain(source(RGBA), conversions=frozenset())


def test_size_mismatch_without_declared_resampling_is_a_resolution_mismatch() -> None:
    with pytest.raises(ResolutionMismatchError):
        chain(source(RGB, size=512))


def test_declared_area_resampling_is_recorded_with_its_scale_effect() -> None:
    src = source(RGB, size=512, spacing=0.25)
    c = chain(src, resample=ResampleMethod.AREA)
    assert "resample" in c.names
    out = validate_model_input(src, INPUT_SPEC, c, Precision.FP32)
    assert out.spacing_um == (0.5, 0.5)  # 2x area reduction doubles the spacing


def test_area_resampling_by_a_non_integer_factor_is_rejected() -> None:
    with pytest.raises(ModelInputContractError):
        chain(source(RGB, size=384), resample=ResampleMethod.AREA)


def test_required_spacing_rejects_unknown_spacing() -> None:
    spec = spaced_spec()
    src = source(RGB, spacing=None)
    with pytest.raises(MissingSpatialMetadataError):
        validate_model_input(src, spec, chain(src, spec), Precision.FP32)


def test_required_spacing_outside_tolerance_is_a_resolution_mismatch() -> None:
    spec = spaced_spec(1.0, 0.05)
    src = source(RGB, spacing=0.5)
    with pytest.raises(ResolutionMismatchError):
        validate_model_input(src, spec, chain(src, spec), Precision.FP32)
    # Within tolerance the same spec is accepted; no universal spacing is imposed anywhere else.
    ok = source(RGB, spacing=0.98)
    assert validate_model_input(ok, spec, chain(ok, spec), Precision.FP32).spacing_um


def test_unknown_spacing_needs_explicit_acceptance() -> None:
    spec = replace(INPUT_SPEC, accepts_unknown_spacing=False)
    src = source(RGB, spacing=None)
    with pytest.raises(MissingSpatialMetadataError):
        validate_model_input(src, spec, chain(src, spec), Precision.FP32)
    assert validate_model_input(src, INPUT_SPEC, chain(src), Precision.FP32).spacing_um is None


def test_unsupported_precision_is_rejected_before_reads() -> None:
    with pytest.raises(ModelInputContractError):
        validate_model_input(source(), INPUT_SPEC, chain(source()), Precision.FP16)


def test_a_chain_that_misses_the_declared_input_is_rejected() -> None:
    full = chain(source(RGB))
    without_normalize = TransformChain(tuple(t for t in full.steps if t.step.name != "normalize"))
    with pytest.raises(ModelInputContractError):
        validate_model_input(source(RGB), INPUT_SPEC, without_normalize, Precision.FP32)


def test_misordered_steps_are_rejected() -> None:
    full = chain(source(BGR))
    assert full.names[-2:] == ("normalize", "layout")
    swapped = TransformChain((*full.steps[:-2], full.steps[-1], full.steps[-2]))
    with pytest.raises(ModelInputContractError):
        validate_model_input(source(BGR), INPUT_SPEC, swapped, Precision.FP32)


def test_declared_transforms_validate_names_and_parameters() -> None:
    impl = IMPLEMENTATIONS["cast"]
    with pytest.raises(ModelInputContractError):
        DeclaredTransform(TransformStep("sharpen"), impl)
    with pytest.raises(ModelInputContractError):
        DeclaredTransform(TransformStep("cast", (("to", "float32"),)), impl)


def test_chain_identity_changes_with_any_step() -> None:
    a = chain(source(RGB))
    b = chain(source(BGR))
    c = chain(source(RGB), spec=replace(INPUT_SPEC, normalization=None))
    assert len({a.fingerprint, b.fingerprint, c.fingerprint}) == 3
    assert a.fingerprint == chain(source(RGB)).fingerprint


# --------------------------------------------------------------------------- batch policy


def policy(**kw: int) -> BatchPolicy:
    base = {
        "max_batch_patches": 32,
        "max_input_batch_bytes": 32 << 20,
        "max_output_batch_bytes": 1 << 20,
        "memory_budget_bytes": 128 << 20,
    }
    return BatchPolicy(**(base | kw))  # type: ignore[arg-type]


def plan(p: BatchPolicy, patches: int = 1000) -> object:
    return p.plan(
        patches=patches,
        decoded_patch_bytes=196_608,
        model_input_patch_bytes=INPUT_SPEC.patch_bytes,
        output_row_bytes=256,
        scratch_bytes=1 << 20,
    )


def test_budget_is_bounded_by_batch_not_patch_count() -> None:
    small, large = (
        policy().plan(
            patches=40,
            decoded_patch_bytes=196_608,
            model_input_patch_bytes=INPUT_SPEC.patch_bytes,
            output_row_bytes=256,
            scratch_bytes=1 << 20,
        ),
        policy().plan(
            patches=4_000_000,
            decoded_patch_bytes=196_608,
            model_input_patch_bytes=INPUT_SPEC.patch_bytes,
            output_row_bytes=256,
            scratch_bytes=1 << 20,
        ),
    )
    assert small == large
    assert small.batch_patches == 32
    assert small.working_set_bytes == 32 * INPUT_SPEC.patch_bytes + 32 * 256 + (1 << 20)


def test_small_runs_use_smaller_batches() -> None:
    b = policy().plan(
        patches=5,
        decoded_patch_bytes=1,
        model_input_patch_bytes=INPUT_SPEC.patch_bytes,
        output_row_bytes=256,
        scratch_bytes=0,
    )
    assert b.batch_patches == 5


@pytest.mark.parametrize(
    "kw",
    [
        {"max_input_batch_bytes": 1 << 20},
        {"max_output_batch_bytes": 1024},
        {"memory_budget_bytes": 8 << 20},
    ],
)
def test_each_budget_fails_closed(kw: dict[str, int]) -> None:
    with pytest.raises(EmbeddingBudgetError):
        plan(policy(**kw))


def test_policy_rejects_unbounded_or_unsupported_settings() -> None:
    with pytest.raises(EmbeddingBudgetError):
        policy(max_batch_patches=0)
    with pytest.raises(EmbeddingBudgetError):
        policy(max_in_flight_batches=2)  # prefetch none means exactly one batch in flight
    with pytest.raises(EmbeddingBudgetError):
        plan(policy(), patches=0)


# --------------------------------------------------------------------------- records and layout


def record(row: int, pid: str, index: int) -> EmbeddingRecord:
    return EmbeddingRecord(
        row, pid, index, LevelRect(0, 0, 0, 256, 256), Level0Rect(0, 0, 256, 256), None, None
    )


def test_records_must_be_ordered_unique_and_row_aligned() -> None:
    validate_records((record(0, "a", 3), record(1, "b", 7)))
    for bad in (
        (record(0, "a", 3), record(2, "b", 7)),
        (record(0, "a", 7), record(1, "b", 3)),
        (record(0, "a", 3), record(1, "a", 7)),
    ):
        with pytest.raises(InvalidManifestError):
            validate_records(bad)


def test_patch_id_digest_binds_rows_to_patches() -> None:
    a = (record(0, "a", 3), record(1, "b", 7))
    b = (record(0, "b", 3), record(1, "a", 7))
    assert patch_ids_digest(a) != patch_ids_digest(b)


def test_layout_chunks_cover_rows_exactly_once() -> None:
    lay = StoreLayout(rows=70, dim=8, dtype="float32", chunk_rows=32)
    ranges = [lay.chunk_range(i) for i in range(lay.chunk_count)]
    assert ranges == [(0, 32), (32, 64), (64, 70)]
    with pytest.raises(InvalidManifestError):
        lay.chunk_range(3)
