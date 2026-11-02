"""Model-independent embedding contracts (SPEC-005).

Five concepts are kept apart (SPEC-005 §10, ADR-005 D3/D4):

1. source resolution: what the reader reports (`ImageDescription`);
2. extraction resolution: the stored level the patch plan selected (`ResolvedPatchGeometry`);
3. patch field of view: extraction pixels times effective spacing, when spacing is known;
4. model-required input: `ModelInputSpec`, declared by the encoder, never by the image;
5. transformations: an explicit, ordered `TransformChain`, recorded with every embedding.

The chain is composed from declared steps only and simulated on a `PixelState` before any pixel is
read. A geometry or scale the declared steps cannot reach fails with `RESOLUTION_MISMATCH`; a
channel order no declared conversion reaches fails with `CHANNEL_ORDER_MISMATCH`. Nothing here
resamples, swaps channels or chooses a spacing by default.

Durable embedding stores are described by a run identity (what will be computed), a layout (how
rows are chunked), chunk commits (what has been verified on disk) and, once finalized, an
`EmbeddingManifest` that binds the run identity to the stored content digest. Runtime-local paths
and timestamps never enter any identity or digest here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from domain.errors import (
    ChannelOrderMismatchError,
    EmbeddingBudgetError,
    InvalidManifestError,
    MissingSpatialMetadataError,
    ModelInputContractError,
    ResolutionMismatchError,
)
from domain.geometry import Level0Rect, LevelRect, PixelSpacing
from domain.hashing import digest, short_digest
from domain.ids import ImageId
from domain.image import ChannelOrder
from domain.manifest import PatchManifest, PixelStorage
from domain.model import ModelRef
from domain.patching import ResolvedPatchGeometry
from domain.pipeline import ConfigValue, PipelineRef
from domain.preprocessing import TransformStep
from domain.provenance import ComponentVersion

EMBEDDING_MANIFEST_SCHEMA = "pathology-ai.embedding-manifest/1"
EMBEDDING_STORE_SCHEMA = "pathology-ai.embedding-store/1"
EMBEDDING_NOTICE = (
    "embedding extraction only: no prediction was made; embeddings are not diagnoses, and "
    "embeddings from a synthetic reference encoder carry no biological or clinical meaning"
)

# Numeric interpretation of the dtypes the contracts understand. Integer sources are read as
# [0, max]; floating values carry the range they were scaled to.
_INTEGER_RANGE: dict[str, tuple[float, float]] = {"uint8": (0.0, 255.0), "uint16": (0.0, 65535.0)}
_ITEMSIZE = {"uint8": 1, "uint16": 2, "float16": 2, "float32": 4, "float64": 8}
_FLOAT_DTYPES = frozenset({"float16", "float32", "float64"})
_CHANNELS = {
    ChannelOrder.RGB: 3,
    ChannelOrder.BGR: 3,
    ChannelOrder.RGBA: 4,
    ChannelOrder.GRAYSCALE: 1,
}


def itemsize(dtype: str) -> int:
    try:
        return _ITEMSIZE[dtype]
    except KeyError:
        raise ModelInputContractError(f"unsupported dtype {dtype!r}") from None


def _pos_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ModelInputContractError(f"{name} must be a positive integer, got {value!r}")


def _finite(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ModelInputContractError(f"{name} must be a finite number, got {value!r}")


def _num(value: float) -> str:
    return repr(float(value))


def _nums(values: tuple[float, ...]) -> str:
    return ",".join(_num(v) for v in values)


# --------------------------------------------------------------------------- model input


class TensorLayout(StrEnum):
    HWC = "hwc"
    CHW = "chw"


class Precision(StrEnum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"


class ResampleMethod(StrEnum):
    # Exact block averaging by an integer factor; any other ratio is rejected, never approximated.
    AREA = "area"


@dataclass(frozen=True, slots=True)
class Normalization:
    """Per-channel `(x - mean) / std`, applied after scaling to `ModelInputSpec.scaled_range`."""

    mean: tuple[float, ...]
    std: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.mean or len(self.mean) != len(self.std):
            raise ModelInputContractError("normalization needs one mean and one std per channel")
        for m in self.mean:
            _finite("normalization mean", m)
        for s in self.std:
            _finite("normalization std", s)
            if s <= 0:
                raise ModelInputContractError("normalization std must be positive")


@dataclass(frozen=True, slots=True)
class ModelInputSpec:
    """What an encoder requires of one input patch. Declared by the encoder, never inferred.

    Spacing is either required (`pixel_spacing_um` within `spacing_tolerance`, relative) or
    explicitly declared variable (`accepts_variable_spacing`). Unknown spacing is accepted only
    when `accepts_unknown_spacing` says so; it is never interpreted as a default value.
    """

    width: int
    height: int
    channel_order: ChannelOrder
    channels: int
    dtype: str
    scaled_range: tuple[float, float]
    layout: TensorLayout
    normalization: Normalization | None
    pixel_spacing_um: float | None
    accepts_variable_spacing: bool
    accepts_unknown_spacing: bool
    spacing_tolerance: float = 0.0
    size_divisor: int = 1
    precisions: frozenset[Precision] = frozenset({Precision.FP32})

    def __post_init__(self) -> None:
        _pos_int("input width", self.width)
        _pos_int("input height", self.height)
        _pos_int("size_divisor", self.size_divisor)
        if self.width % self.size_divisor or self.height % self.size_divisor:
            raise ModelInputContractError(
                f"input {self.width}x{self.height} is not divisible by {self.size_divisor}"
            )
        expected = _CHANNELS.get(self.channel_order)
        if expected is None or expected != self.channels:
            raise ModelInputContractError(
                f"{self.channels} channels do not match channel order {self.channel_order.value}"
            )
        if self.dtype not in _FLOAT_DTYPES:
            raise ModelInputContractError(f"model input dtype {self.dtype!r} must be floating")
        lo, hi = self.scaled_range
        _finite("scaled_range low", lo)
        _finite("scaled_range high", hi)
        if lo != 0.0 or hi <= lo:
            raise ModelInputContractError("scaled_range must be (0, high) with high > 0")
        if self.normalization is not None and len(self.normalization.mean) != self.channels:
            raise ModelInputContractError("normalization must have one value per channel")
        if self.pixel_spacing_um is None:
            if not self.accepts_variable_spacing:
                raise ModelInputContractError(
                    "declare a required pixel spacing or accept variable spacing explicitly"
                )
        else:
            _finite("pixel_spacing_um", self.pixel_spacing_um)
            if self.pixel_spacing_um <= 0:
                raise ModelInputContractError("pixel_spacing_um must be positive")
            if self.accepts_variable_spacing or self.accepts_unknown_spacing:
                raise ModelInputContractError(
                    "a model that requires a spacing cannot also accept variable or unknown spacing"
                )
        _finite("spacing_tolerance", self.spacing_tolerance)
        if not 0.0 <= self.spacing_tolerance < 1.0:
            raise ModelInputContractError("spacing_tolerance is relative and must lie in [0, 1)")
        if not self.precisions:
            raise ModelInputContractError("declare at least one supported precision")

    @property
    def patch_bytes(self) -> int:
        return self.width * self.height * self.channels * itemsize(self.dtype)

    @property
    def fingerprint(self) -> str:
        return digest(self)


# --------------------------------------------------------------------------- transforms


@dataclass(frozen=True, slots=True)
class PixelState:
    """The geometric and numeric interpretation of one patch at a point in the chain."""

    width: int
    height: int
    channel_order: ChannelOrder
    channels: int
    dtype: str
    value_range: tuple[float, float]
    layout: TensorLayout
    normalized: bool
    spacing_um: tuple[float, float] | None

    @property
    def nbytes(self) -> int:
        return self.width * self.height * self.channels * itemsize(self.dtype)

    @staticmethod
    def of_patch(
        width: int,
        height: int,
        channel_order: ChannelOrder,
        samples: int,
        dtype: str,
        spacing_um: tuple[float, float] | None,
    ) -> PixelState:
        """A decoded patch exactly as the reader returns it (native HWC, native dtype)."""
        if dtype not in _INTEGER_RANGE:
            raise ModelInputContractError(
                f"source dtype {dtype!r} has no declared numeric range; only "
                f"{', '.join(sorted(_INTEGER_RANGE))} sources are supported"
            )
        if _CHANNELS.get(channel_order) != samples:
            raise ModelInputContractError(
                f"{samples} samples per pixel do not match channel order {channel_order.value}"
            )
        return PixelState(
            width,
            height,
            channel_order,
            samples,
            dtype,
            _INTEGER_RANGE[dtype],
            TensorLayout.HWC,
            False,
            spacing_um,
        )


# Each step name maps to its required parameters. Implementations are versioned separately.
TRANSFORM_PARAMETERS: dict[str, frozenset[str]] = {
    "channel_reorder": frozenset({"from", "to"}),
    "drop_alpha": frozenset({"from", "to"}),
    "cast": frozenset({"from", "to"}),
    "scale": frozenset({"factor", "from_range", "to_range"}),
    "resample": frozenset({"method", "from", "to"}),
    "normalize": frozenset({"mean", "std"}),
    "layout": frozenset({"from", "to"}),
}
_REORDERS = {
    (ChannelOrder.BGR, ChannelOrder.RGB),
    (ChannelOrder.RGB, ChannelOrder.BGR),
}


@dataclass(frozen=True, slots=True)
class DeclaredTransform:
    """A `TransformStep` bound to the versioned component that implements it."""

    step: TransformStep
    implementation: ComponentVersion

    def __post_init__(self) -> None:
        required = TRANSFORM_PARAMETERS.get(self.step.name)
        if required is None:
            raise ModelInputContractError(f"unknown transform {self.step.name!r}")
        keys = [k for k, _ in self.step.params]
        if set(keys) != required or len(keys) != len(required) or keys != sorted(keys):
            raise ModelInputContractError(
                f"transform {self.step.name} needs exactly the sorted parameters {sorted(required)}"
            )


@dataclass(frozen=True, slots=True)
class TransformChain:
    """An ordered, validated transformation sequence. Empty means "pixels used as decoded"."""

    steps: tuple[DeclaredTransform, ...] = ()

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(t.step.name for t in self.steps)

    @property
    def fingerprint(self) -> str:
        return digest(self)

    def apply_to(self, state: PixelState) -> PixelState:
        for t in self.steps:
            state = _advance(state, t.step)
        return state


def _size(text: str) -> tuple[int, int]:
    w, _, h = text.partition("x")
    return int(w), int(h)


def _advance(s: PixelState, step: TransformStep) -> PixelState:
    """The state after `step`, or a typed error when the step does not apply to `s`."""
    p = dict(step.params)

    def require(cond: bool, why: str) -> None:
        if not cond:
            raise ModelInputContractError(f"transform {step.name}: {why}")

    if step.name in ("channel_reorder", "drop_alpha"):
        src, dst = ChannelOrder(p["from"]), ChannelOrder(p["to"])
        require(s.channel_order is src, f"input is {s.channel_order.value}, step expects {src}")
        require(s.layout is TensorLayout.HWC, "channel steps operate on HWC patches")
        if step.name == "channel_reorder":
            require((src, dst) in _REORDERS, f"{src}->{dst} is not a pure reorder")
        else:
            require((src, dst) == (ChannelOrder.RGBA, ChannelOrder.RGB), "only rgba->rgb")
        return PixelState(
            s.width,
            s.height,
            dst,
            _CHANNELS[dst],
            s.dtype,
            s.value_range,
            s.layout,
            s.normalized,
            s.spacing_um,
        )
    if step.name == "cast":
        require(p["from"] == s.dtype, f"input dtype is {s.dtype}, step expects {p['from']}")
        require(p["to"] in _FLOAT_DTYPES, "casts produce floating values")
        return PixelState(
            s.width,
            s.height,
            s.channel_order,
            s.channels,
            p["to"],
            s.value_range,
            s.layout,
            s.normalized,
            s.spacing_um,
        )
    if step.name == "scale":
        require(s.dtype in _FLOAT_DTYPES, "scale needs floating input (cast first)")
        require(not s.normalized, "scale after normalization is not allowed")
        lo, hi = (float(v) for v in p["from_range"].split(","))
        tlo, thi = (float(v) for v in p["to_range"].split(","))
        require((lo, hi) == s.value_range, f"input range is {s.value_range}")
        require(lo == 0.0 and tlo == 0.0, "only ranges starting at 0 are scaled")
        require(math.isclose(float(p["factor"]) * hi, thi, rel_tol=1e-12), "factor mismatch")
        return PixelState(
            s.width,
            s.height,
            s.channel_order,
            s.channels,
            s.dtype,
            (tlo, thi),
            s.layout,
            s.normalized,
            s.spacing_um,
        )
    if step.name == "resample":
        require(ResampleMethod(p["method"]) is ResampleMethod.AREA, "unknown method")
        fw, fh = _size(p["from"])
        tw, th = _size(p["to"])
        require((fw, fh) == (s.width, s.height), f"input is {s.width}x{s.height}")
        require(s.dtype in _FLOAT_DTYPES, "resample operates on floating values")
        require(s.layout is TensorLayout.HWC, "resample operates on HWC patches")
        require(
            fw % tw == 0 and fh % th == 0 and fw // tw == fh // th,
            f"area resampling needs one integer factor ({fw}x{fh} -> {tw}x{th})",
        )
        factor = fw // tw
        spacing = None if s.spacing_um is None else tuple(v * factor for v in s.spacing_um)
        return PixelState(
            tw,
            th,
            s.channel_order,
            s.channels,
            s.dtype,
            s.value_range,
            s.layout,
            s.normalized,
            None if spacing is None else (spacing[0], spacing[1]),
        )
    if step.name == "normalize":
        require(s.dtype in _FLOAT_DTYPES, "normalize needs floating input")
        require(not s.normalized, "already normalized")
        require(s.layout is TensorLayout.HWC, "normalize operates per channel on HWC patches")
        require(len(p["mean"].split(",")) == s.channels, "one mean per channel")
        require(len(p["std"].split(",")) == s.channels, "one std per channel")
        return PixelState(
            s.width,
            s.height,
            s.channel_order,
            s.channels,
            s.dtype,
            s.value_range,
            s.layout,
            True,
            s.spacing_um,
        )
    # layout
    require(TensorLayout(p["from"]) is s.layout, f"input layout is {s.layout.value}")
    return PixelState(
        s.width,
        s.height,
        s.channel_order,
        s.channels,
        s.dtype,
        s.value_range,
        TensorLayout(p["to"]),
        s.normalized,
        s.spacing_um,
    )


def build_transform_chain(
    source: PixelState,
    spec: ModelInputSpec,
    *,
    channel_conversions: frozenset[tuple[ChannelOrder, ChannelOrder]],
    resample: ResampleMethod | None,
    implementations: Mapping[str, ComponentVersion],
) -> TransformChain:
    """Compose the chain from declared conversions and the encoder's own contract.

    Channel conversions and resampling are added only when explicitly declared. Numeric steps
    (cast, scale, normalize, layout) follow from the declared `ModelInputSpec`. The resulting
    chain is validated by `validate_model_input`.
    """

    def declared(name: str, **params: str) -> DeclaredTransform:
        impl = implementations.get(name)
        if impl is None:
            raise ModelInputContractError(f"no implementation is registered for {name!r}")
        return DeclaredTransform(TransformStep(name, tuple(sorted(params.items()))), impl)

    steps: list[DeclaredTransform] = []
    src, dst = source.channel_order, spec.channel_order
    if src is not dst:
        if (src, dst) not in channel_conversions:
            allowed = ", ".join(sorted(f"{a.value}->{b.value}" for a, b in channel_conversions))
            raise ChannelOrderMismatchError(
                f"patches are {src.value}, the encoder expects {dst.value}, and no matching "
                f"conversion is declared (declared: {allowed or 'none'})"
            )
        if (src, dst) == (ChannelOrder.RGBA, ChannelOrder.RGB):
            steps.append(declared("drop_alpha", **{"from": src.value, "to": dst.value}))
        elif (src, dst) in _REORDERS:
            steps.append(declared("channel_reorder", **{"from": src.value, "to": dst.value}))
        else:
            raise ChannelOrderMismatchError(f"no transform converts {src.value} to {dst.value}")
    if source.dtype != spec.dtype:
        steps.append(declared("cast", **{"from": source.dtype, "to": spec.dtype}))
    lo, hi = source.value_range
    if (lo, hi) != spec.scaled_range:
        steps.append(
            declared(
                "scale",
                factor=_num(spec.scaled_range[1] / hi),
                from_range=_nums((lo, hi)),
                to_range=_nums(spec.scaled_range),
            )
        )
    if (source.width, source.height) != (spec.width, spec.height):
        if resample is None:
            raise ResolutionMismatchError(
                f"patches are {source.width}x{source.height} px, the encoder requires "
                f"{spec.width}x{spec.height} px, and no resampling is declared; resampling is "
                "never implicit"
            )
        steps.append(
            declared(
                "resample",
                method=resample.value,
                **{
                    "from": f"{source.width}x{source.height}",
                    "to": f"{spec.width}x{spec.height}",
                },
            )
        )
    if spec.normalization is not None:
        steps.append(
            declared(
                "normalize", mean=_nums(spec.normalization.mean), std=_nums(spec.normalization.std)
            )
        )
    if spec.layout is not TensorLayout.HWC:
        steps.append(
            declared("layout", **{"from": TensorLayout.HWC.value, "to": spec.layout.value})
        )
    chain = TransformChain(tuple(steps))
    chain.apply_to(source)  # every step must apply to the state the previous one produced
    return chain


def validate_model_input(
    source: PixelState, spec: ModelInputSpec, chain: TransformChain, precision: Precision
) -> PixelState:
    """Simulate `chain` on `source` and check the result against `spec`. Reads no pixel."""
    if precision not in spec.precisions:
        supported = ", ".join(sorted(p.value for p in spec.precisions))
        raise ModelInputContractError(
            f"precision {precision.value} is not supported by the encoder ({supported})"
        )
    out = chain.apply_to(source)
    if (out.width, out.height) != (spec.width, spec.height):
        raise ResolutionMismatchError(
            f"the chain yields {out.width}x{out.height} px; the encoder requires "
            f"{spec.width}x{spec.height} px"
        )
    mismatches = [
        name
        for name, ok in (
            ("channel_order", out.channel_order is spec.channel_order),
            ("channels", out.channels == spec.channels),
            ("dtype", out.dtype == spec.dtype),
            ("scaled_range", out.value_range == spec.scaled_range),
            ("layout", out.layout is spec.layout),
            ("normalization", out.normalized == (spec.normalization is not None)),
        )
        if not ok
    ]
    if mismatches:
        raise ModelInputContractError(
            "the transform chain does not produce the declared model input: "
            + ", ".join(mismatches)
        )
    if spec.pixel_spacing_um is not None:
        if out.spacing_um is None:
            raise MissingSpatialMetadataError(
                f"the encoder requires {spec.pixel_spacing_um} µm/px but the image reports no "
                "physical spacing; unknown spacing is never assumed"
            )
        target = spec.pixel_spacing_um
        error = max(abs(v - target) / target for v in out.spacing_um)
        if error > spec.spacing_tolerance:
            raise ResolutionMismatchError(
                f"model input spacing {out.spacing_um[0]:.6g} x {out.spacing_um[1]:.6g} µm/px is "
                f"{error:.2%} from the required {target} µm/px (tolerance "
                f"{spec.spacing_tolerance:.2%}); choose another stored level or declare a "
                "resampling step"
            )
    elif out.spacing_um is None and not spec.accepts_unknown_spacing:
        raise MissingSpatialMetadataError(
            "the image reports no physical spacing and the encoder does not accept unknown spacing"
        )
    return out


@dataclass(frozen=True, slots=True)
class ResolutionRecord:
    """Source, extraction, field-of-view and model-input resolution, kept separate (AC-004)."""

    source_spacing_um: tuple[float, float] | None
    source_spacing_source: str | None
    extraction_level: int
    extraction_downsample: float
    extraction_spacing_um: tuple[float, float] | None
    requested_spacing_um: float | None
    spacing_relative_error: float | None
    patch_px: tuple[int, int]
    field_of_view_um: tuple[float, float] | None
    model_input_px: tuple[int, int]
    model_input_spacing_um: tuple[float, float] | None
    model_required_spacing_um: float | None
    model_accepts_variable_spacing: bool
    resampling: str

    @staticmethod
    def of(
        source_spacing: PixelSpacing | None,
        spacing_source: str | None,
        resolved: ResolvedPatchGeometry,
        model_input: PixelState,
        spec: ModelInputSpec,
        chain: TransformChain,
    ) -> ResolutionRecord:
        eff = resolved.effective_spacing_um
        resample = next((t.step for t in chain.steps if t.step.name == "resample"), None)
        return ResolutionRecord(
            source_spacing_um=(
                None if source_spacing is None else (source_spacing.x_um, source_spacing.y_um)
            ),
            source_spacing_source=spacing_source,
            extraction_level=resolved.level,
            extraction_downsample=resolved.downsample,
            extraction_spacing_um=eff,
            requested_spacing_um=resolved.requested_spacing_um,
            spacing_relative_error=resolved.spacing_relative_error,
            patch_px=(resolved.width_px, resolved.height_px),
            field_of_view_um=(
                None if eff is None else (resolved.width_px * eff[0], resolved.height_px * eff[1])
            ),
            model_input_px=(model_input.width, model_input.height),
            model_input_spacing_um=model_input.spacing_um,
            model_required_spacing_um=spec.pixel_spacing_um,
            model_accepts_variable_spacing=spec.accepts_variable_spacing,
            resampling=(
                "none"
                if resample is None
                else " ".join(
                    (str(resample.param("method")), str(resample.param("from")), "->")
                    + (str(resample.param("to")),)
                )
            ),
        )


# --------------------------------------------------------------------------- batching


class PrefetchPolicy(StrEnum):
    # Read, transform, encode and write strictly in sequence on the calling thread. The only
    # policy offered: native reader handles are not assumed thread- or fork-safe (ADR-005 D10).
    NONE = "none"


@dataclass(frozen=True, slots=True)
class BatchBudget:
    """The bounded working set of one embedding run, derived before any pixel is read."""

    batch_patches: int
    decoded_patch_bytes: int
    model_input_batch_bytes: int
    output_batch_bytes: int
    scratch_bytes: int
    in_flight_batches: int
    working_set_bytes: int
    memory_budget_bytes: int


@dataclass(frozen=True, slots=True)
class BatchPolicy:
    """Explicit batch, in-flight and memory budgets (SPEC-005 §6.3).

    The working set counts, per in-flight batch, the model-input batch buffer; plus one encoder
    output batch; plus per-patch scratch (the decoded region and transform temporaries). A run
    whose working set exceeds `memory_budget_bytes`, or a batch above its byte limits, fails
    before any pixel is read.
    """

    max_batch_patches: int
    max_input_batch_bytes: int
    max_output_batch_bytes: int
    memory_budget_bytes: int
    max_in_flight_batches: int = 1
    prefetch: PrefetchPolicy = PrefetchPolicy.NONE
    device: str = "cpu"
    precision: Precision = Precision.FP32

    def __post_init__(self) -> None:
        for name in (
            "max_batch_patches",
            "max_input_batch_bytes",
            "max_output_batch_bytes",
            "memory_budget_bytes",
            "max_in_flight_batches",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise EmbeddingBudgetError(f"{name} must be a positive integer, got {value!r}")
        if self.prefetch is PrefetchPolicy.NONE and self.max_in_flight_batches != 1:
            raise EmbeddingBudgetError(
                "prefetch none processes exactly one batch at a time; set max_in_flight_batches 1"
            )
        if not self.device:
            raise EmbeddingBudgetError("device must be named explicitly")

    def plan(
        self,
        *,
        patches: int,
        decoded_patch_bytes: int,
        model_input_patch_bytes: int,
        output_row_bytes: int,
        scratch_bytes: int,
    ) -> BatchBudget:
        if patches < 1:
            raise EmbeddingBudgetError("an embedding run needs at least one patch")
        b = min(self.max_batch_patches, patches)
        model_input = b * model_input_patch_bytes
        output = b * output_row_bytes
        if model_input > self.max_input_batch_bytes:
            raise EmbeddingBudgetError(
                f"a batch of {b} model inputs needs {model_input} bytes, above "
                f"max_input_batch_bytes={self.max_input_batch_bytes}"
            )
        if output > self.max_output_batch_bytes:
            raise EmbeddingBudgetError(
                f"a batch of {b} embeddings needs {output} bytes, above "
                f"max_output_batch_bytes={self.max_output_batch_bytes}"
            )
        working = self.max_in_flight_batches * model_input + output + scratch_bytes
        if working > self.memory_budget_bytes:
            raise EmbeddingBudgetError(
                f"the working set needs {working} bytes ({self.max_in_flight_batches} x "
                f"{model_input} model input + {output} output + {scratch_bytes} scratch), above "
                f"memory_budget_bytes={self.memory_budget_bytes}"
            )
        return BatchBudget(
            batch_patches=b,
            decoded_patch_bytes=decoded_patch_bytes,
            model_input_batch_bytes=model_input,
            output_batch_bytes=output,
            scratch_bytes=scratch_bytes,
            in_flight_batches=self.max_in_flight_batches,
            working_set_bytes=working,
            memory_budget_bytes=self.memory_budget_bytes,
        )


# --------------------------------------------------------------------------- identity and records


@dataclass(frozen=True, slots=True)
class EncoderIdentity:
    """Implementation, model and weights identity of one configured encoder."""

    component: ComponentVersion
    model: ModelRef | None
    weights_checksum: str | None
    synthetic: bool
    parameters: tuple[tuple[str, ConfigValue], ...]
    output_dim: int
    output_dtype: str

    def __post_init__(self) -> None:
        _pos_int("output_dim", self.output_dim)
        if self.output_dtype not in _FLOAT_DTYPES:
            raise ModelInputContractError("embeddings must be floating values")
        if list(self.parameters) != sorted(self.parameters):
            raise ModelInputContractError("encoder parameters must be sorted by key")


@dataclass(frozen=True, slots=True)
class EmbeddingRecord:
    """One embedding row and the patch it was computed from. Association is by patch id."""

    row: int
    patch_id: str
    plan_index: int
    level_rect: LevelRect
    level0: Level0Rect
    effective_spacing_um: tuple[float, float] | None
    field_of_view_um: tuple[float, float] | None


def records_from_manifest(manifest: PatchManifest) -> tuple[EmbeddingRecord, ...]:
    """Rows follow the patch manifest's plan order; each row carries its patch identity."""
    return tuple(
        EmbeddingRecord(
            row=row,
            patch_id=p.patch_id,
            plan_index=p.plan_index,
            level_rect=p.level_rect,
            level0=p.level0,
            effective_spacing_um=p.effective_spacing_um,
            field_of_view_um=(
                None
                if p.effective_spacing_um is None
                else (
                    p.width_px * p.effective_spacing_um[0],
                    p.height_px * p.effective_spacing_um[1],
                )
            ),
        )
        for row, p in enumerate(manifest.patches)
    )


def validate_records(records: tuple[EmbeddingRecord, ...]) -> None:
    if [r.row for r in records] != list(range(len(records))):
        raise InvalidManifestError("embedding rows must be 0..n-1 in order")
    indices = [r.plan_index for r in records]
    if indices != sorted(set(indices)):
        raise InvalidManifestError("embedding records must follow plan order without repeats")
    if len({r.patch_id for r in records}) != len(records):
        raise InvalidManifestError("patch identities must be unique")


def patch_ids_digest(records: tuple[EmbeddingRecord, ...]) -> str:
    return digest([(r.row, r.patch_id, r.plan_index) for r in records])


@dataclass(frozen=True, slots=True)
class EmbeddingRunIdentity:
    """What a run will compute: binds patches, encoder, input contract and transforms (D8).

    Batch sizes, memory budgets, paths and timestamps are not part of it.
    """

    image_id: ImageId
    source_integrity: str | None
    scene: int
    plan_digest: str
    patch_manifest_digest: str
    records_digest: str
    encoder: EncoderIdentity
    input_spec: ModelInputSpec
    transforms: TransformChain
    precision: Precision
    device: str
    schema_version: str = EMBEDDING_STORE_SCHEMA

    @property
    def value(self) -> str:
        return digest(self)

    @property
    def store_id(self) -> str:
        return short_digest(self, "emb")


# --------------------------------------------------------------------------- durable store


class StoreState(StrEnum):
    CREATED = "created"  # header written, no committed chunk
    IN_PROGRESS = "in_progress"  # a live writer holds the store and has committed chunks
    INTERRUPTED = "interrupted"  # reopened with verified chunks but incomplete or unfinalized
    FINALIZED = "finalized"  # complete coverage verified and the final manifest committed
    CORRUPT = "corrupt"
    INCOMPATIBLE = "incompatible"


class RecoveryPolicy(StrEnum):
    # Resume from verified committed chunks; any corrupt chunk or record fails the run.
    RESUME_VERIFIED = "resume_verified"
    # Explicit: drop commit records that fail verification and recompute those chunks.
    DISCARD_CORRUPT = "discard_corrupt"


@dataclass(frozen=True, slots=True)
class StoreLayout:
    rows: int
    dim: int
    dtype: str
    chunk_rows: int

    def __post_init__(self) -> None:
        _pos_int("rows", self.rows)
        _pos_int("dim", self.dim)
        _pos_int("chunk_rows", self.chunk_rows)
        itemsize(self.dtype)

    @property
    def chunk_count(self) -> int:
        return -(-self.rows // self.chunk_rows)

    def chunk_range(self, index: int) -> tuple[int, int]:
        if not 0 <= index < self.chunk_count:
            raise InvalidManifestError(f"chunk {index} outside 0..{self.chunk_count - 1}")
        start = index * self.chunk_rows
        return start, min(self.rows, start + self.chunk_rows)

    @property
    def row_bytes(self) -> int:
        return self.dim * itemsize(self.dtype)


@dataclass(frozen=True, slots=True)
class EmbeddingStoreHeader:
    """Everything a store needs to be created or reopened and checked for compatibility."""

    store_id: str
    identity: EmbeddingRunIdentity
    layout: StoreLayout
    records: tuple[EmbeddingRecord, ...]
    schema_version: str = EMBEDDING_STORE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != EMBEDDING_STORE_SCHEMA:
            raise InvalidManifestError(f"unsupported store schema {self.schema_version!r}")
        if self.store_id != self.identity.store_id:
            raise InvalidManifestError("store id does not derive from the run identity")
        if len(self.records) != self.layout.rows:
            raise InvalidManifestError("one record per stored row is required")
        validate_records(self.records)
        if self.layout.dim != self.identity.encoder.output_dim:
            raise InvalidManifestError("layout dimension differs from the encoder output")
        if self.layout.dtype != self.identity.encoder.output_dtype:
            raise InvalidManifestError("layout dtype differs from the encoder output")

    @property
    def digest(self) -> str:
        return digest(self)

    def chunk_records(self, index: int) -> tuple[EmbeddingRecord, ...]:
        start, stop = self.layout.chunk_range(index)
        return self.records[start:stop]


@dataclass(frozen=True, slots=True)
class ChunkCommit:
    """A verified chunk: its row range, the digest of its bytes and of its patch identities."""

    index: int
    start: int
    stop: int
    rows_sha256: str
    patch_ids_digest: str


@dataclass(frozen=True, slots=True)
class StoreInspection:
    state: StoreState
    committed: tuple[ChunkCommit, ...]
    missing: tuple[int, ...]
    corrupt: tuple[tuple[int, str], ...]
    finalized_manifest_digest: str | None

    @property
    def committed_indices(self) -> frozenset[int]:
        return frozenset(c.index for c in self.committed)


@dataclass(frozen=True, slots=True)
class StoreVerification:
    """Complete, verified coverage: every chunk committed and its bytes re-hashed."""

    rows: int
    chunk_count: int
    chunks_digest: str
    content_digest: str


@dataclass(frozen=True, slots=True)
class StoreDescriptor:
    """How and where (by id, never by path) the embeddings are stored."""

    format: str
    schema_version: str
    store_id: str
    chunk_rows: int
    chunk_count: int
    chunks_digest: str


@dataclass(frozen=True, slots=True)
class DurableStoreReference:
    """A handle to a finalized store. `location` is runtime-local and never part of an identity."""

    format: str
    schema_version: str
    store_id: str
    location: str


@dataclass(frozen=True, slots=True)
class EmbeddingMetrics:
    """Bounded execution metrics (counts and bytes; no per-patch data)."""

    patches: int
    patches_encoded: int
    patches_resumed: int
    batches_encoded: int
    chunks_committed: int
    chunks_resumed: int
    region_reads: int
    partially_acquired_reads: int
    max_live_regions: int
    budget: BatchBudget
    reused_finalized_store: bool
    # Encoder-reported measurements (SPEC-006): parameter bytes, device peak memory, timings,
    # runtime versions. Volatile by nature, so never part of a manifest or an identity.
    encoder_resources: tuple[tuple[str, ConfigValue], ...] = ()


# --------------------------------------------------------------------------- manifest


@dataclass(frozen=True, slots=True)
class EmbeddingManifest:
    """Versioned description of a finalized embedding bag (`pathology-ai.embedding-manifest/1`)."""

    image_id: ImageId
    source_integrity: str | None
    scene: int
    pipeline: PipelineRef | None
    reader: ComponentVersion
    backend: ComponentVersion
    operation: ComponentVersion
    plan_digest: str
    patch_manifest_digest: str
    patch_manifest_schema: str
    patch_pixel_storage: PixelStorage
    run_identity: str
    encoder: EncoderIdentity
    input_spec: ModelInputSpec
    transforms: TransformChain
    resolution: ResolutionRecord
    precision: Precision
    device: str
    embedding_dim: int
    embedding_dtype: str
    row_count: int
    records: tuple[EmbeddingRecord, ...]
    store: StoreDescriptor
    content_digest: str
    status: StoreState
    limitations: tuple[str, ...]
    schema_version: str = EMBEDDING_MANIFEST_SCHEMA
    notice: str = EMBEDDING_NOTICE

    def __post_init__(self) -> None:
        if self.schema_version != EMBEDDING_MANIFEST_SCHEMA:
            raise InvalidManifestError(f"unsupported manifest schema {self.schema_version!r}")
        if self.status is not StoreState.FINALIZED:
            raise InvalidManifestError("an embedding manifest describes a finalized store only")
        if self.row_count != len(self.records) or self.row_count < 1:
            raise InvalidManifestError("row count does not match the embedding records")
        validate_records(self.records)
        if (self.embedding_dim, self.embedding_dtype) != (
            self.encoder.output_dim,
            self.encoder.output_dtype,
        ):
            raise InvalidManifestError("embedding schema differs from the encoder output")
        if not self.content_digest:
            raise InvalidManifestError("a finalized manifest needs a content digest")

    @property
    def manifest_digest(self) -> str:
        return digest(self)

    @property
    def embedding_identity(self) -> str:
        """Run identity bound to the stored content: equal only for equal embeddings."""
        return embedding_identity(self.run_identity, self.content_digest)


def embedding_identity(run_identity: str, content_digest: str) -> str:
    """The identity of a finalized bag: its run identity bound to its stored content digest."""
    return short_digest({"run": run_identity, "content": content_digest}, "embid")
