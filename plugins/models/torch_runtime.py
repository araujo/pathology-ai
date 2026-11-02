"""Generic PyTorch runtime for embedding encoders (SPEC-006 Phase B). Knows no model by name.

Import this module lazily (inside `create`): it imports torch at module level, and the base
install has no torch. Adapters catch `ModuleNotFoundError` and raise `MODEL_RUNTIME_UNAVAILABLE`.

Loading (`load_module`) is fail-closed and download-free:

1. the caller has already verified the local file against its pinned SHA-256;
2. the file is deserialized with `torch.load(weights_only=True)`: tensors and plain containers
   only, never arbitrary pickled objects;
3. the architecture is built on the `meta` device (no memory, no random initialization), and the
   checkpoint's keys, shapes and dtypes are compared with it before anything is assigned;
4. tensors are assigned with `load_state_dict(strict=True, assign=True)`; any tensor left on
   `meta` (state the checkpoint does not provide) is rejected;
5. the file's stat is rechecked, the module is put in evaluation mode with gradients disabled,
   and moved to the requested device and precision.

`TorchEncoder` runs each bounded batch under `torch.inference_mode()`, converts the output to a
host float32 array, and reports parameter memory, device peak memory and timings separately
from the host-side budget (`working_bytes`).
"""

from __future__ import annotations

import gc
import importlib.metadata
import pickle
import platform
import time
import zipfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from domain.embedding import EncoderIdentity, ModelInputSpec, Precision, TensorLayout
from domain.errors import (
    CheckpointIncompatibleError,
    DeviceUnavailableError,
    DomainError,
    ModelInferenceError,
    UnsupportedModelInputError,
)
from domain.foundation import DeviceKind, EncoderDescriptor, EncoderRuntimeReport, parse_device
from domain.pipeline import ConfigValue
from plugins.models.checkpoints import CheckpointStat
from ports import EncodeContext

Extractor = Callable[[nn.Module, torch.Tensor], torch.Tensor]
_DTYPES = {
    Precision.FP32: torch.float32,
    Precision.FP16: torch.float16,
    Precision.BF16: torch.bfloat16,
}
_MAX_LISTED = 5


# --------------------------------------------------------------------------- environment


def package_versions(*names: str) -> tuple[tuple[str, str], ...]:
    """Installed versions of the runtime packages an adapter depends on (for identity)."""
    out = []
    for name in names:
        try:
            out.append((name, importlib.metadata.version(name)))
        except importlib.metadata.PackageNotFoundError:
            out.append((name, "unknown"))
    return tuple(out)


def select_device(text: str, descriptor: EncoderDescriptor) -> torch.device:
    """The requested device, if the encoder supports it and it is available. Never falls back."""
    kind, index = parse_device(text)
    if kind not in descriptor.devices:
        supported = ", ".join(sorted(d.value for d in descriptor.devices))
        raise DeviceUnavailableError(
            f"encoder {descriptor.name} does not support device {text!r} (supported: {supported})"
        )
    if kind is DeviceKind.CUDA:
        if not torch.cuda.is_available():
            raise DeviceUnavailableError(
                f"device {text!r} was requested but CUDA is not available in this runtime"
            )
        if index is not None and index >= torch.cuda.device_count():
            raise DeviceUnavailableError(
                f"device {text!r} was requested but only {torch.cuda.device_count()} CUDA "
                "device(s) exist"
            )
    if kind is DeviceKind.MPS and not torch.backends.mps.is_available():
        raise DeviceUnavailableError(f"device {text!r} was requested but MPS is not available")
    return torch.device(text)


def select_dtype(precision: Precision, descriptor: EncoderDescriptor) -> torch.dtype:
    """Explicit precision only; there is no automatic mixed precision (SPEC-006 §9)."""
    if precision not in descriptor.precisions:
        supported = ", ".join(sorted(p.value for p in descriptor.precisions))
        raise DeviceUnavailableError(
            f"encoder {descriptor.name} does not support precision {precision.value} "
            f"(validated: {supported})"
        )
    return _DTYPES[precision]


# --------------------------------------------------------------------------- loading


def read_state_dict(path: Path) -> dict[str, torch.Tensor]:
    """Deserialize tensors only (`weights_only=True`); anything else is incompatible."""
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, RuntimeError, EOFError, ValueError, zipfile.BadZipFile) as exc:
        raise CheckpointIncompatibleError(
            f"checkpoint cannot be read as a weights-only state dict: {_first_line(exc)}"
        ) from None
    if not isinstance(obj, Mapping) or not all(
        isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in obj.items()
    ):
        raise CheckpointIncompatibleError("checkpoint is not a flat state dict of named tensors")
    return dict(obj)


def _first_line(exc: BaseException) -> str:
    text = str(exc).strip()
    return f"{type(exc).__name__}: {text.splitlines()[0] if text else ''}"


def _listed(items: list[str]) -> str:
    more = f" (+{len(items) - _MAX_LISTED} more)" if len(items) > _MAX_LISTED else ""
    return ", ".join(items[:_MAX_LISTED]) + more


def check_state_dict(
    expected: Mapping[str, torch.Tensor], state: Mapping[str, torch.Tensor], architecture: str
) -> None:
    """Keys, shapes and dtypes must match the architecture exactly (strict, typed errors)."""
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    shapes = sorted(
        f"{k} {tuple(state[k].shape)}!={tuple(expected[k].shape)}"
        for k in set(expected) & set(state)
        if tuple(state[k].shape) != tuple(expected[k].shape)
    )
    dtypes = sorted(
        f"{k} {state[k].dtype}!={expected[k].dtype}"
        for k in set(expected) & set(state)
        if state[k].dtype != expected[k].dtype
    )
    problems = [
        f"{label}: {_listed(items)}"
        for label, items in (
            ("missing", missing),
            ("unexpected", unexpected),
            ("shape", shapes),
            ("dtype", dtypes),
        )
        if items
    ]
    if problems:
        raise CheckpointIncompatibleError(
            f"checkpoint does not match architecture {architecture}: " + "; ".join(problems)
        )


def load_module(
    build: Callable[[], nn.Module],
    path: Path,
    stat: CheckpointStat,
    *,
    architecture: str,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Module:
    """Construct `build()` on `meta`, check and assign the verified checkpoint, eval, move."""
    with torch.device("meta"):
        module = build()
    expected = module.state_dict()
    state = read_state_dict(path)
    check_state_dict(expected, state, architecture)
    try:
        module.load_state_dict(state, strict=True, assign=True)
    except RuntimeError as exc:
        raise CheckpointIncompatibleError(
            f"checkpoint does not load into {architecture}: {str(exc).splitlines()[0]}"
        ) from None
    del state, expected
    left = [n for n, t in [*module.named_parameters(), *module.named_buffers()] if t.is_meta]
    if left:
        raise CheckpointIncompatibleError(
            f"architecture {architecture} has state the checkpoint does not provide: "
            f"{_listed(left)}"
        )
    stat.unchanged(path)
    module.eval()
    module.requires_grad_(False)
    return module.to(device=device, dtype=dtype)


def parameter_bytes(module: nn.Module) -> tuple[int, int]:
    params = list(module.parameters())
    return sum(p.numel() for p in params), sum(p.numel() * p.element_size() for p in params)


def vit_activation_bytes(
    *, tokens: int, dim: int, heads: int, mlp_ratio: float, itemsize: int = 4
) -> int:
    """Conservative per-sample host activation bound of one transformer block in inference.

    Live tensors within a block: residual, norm, qkv (3x), attention output and projection
    (about 7 x tokens x dim), two MLP hidden tensors (2 x mlp_ratio x tokens x dim) and up to two
    attention score maps (2 x heads x tokens^2). A 25 % margin covers allocator slack. Measured
    on ViT-L/16 (CPU, torch 2.14): about 14.4 MiB per sample; this bound gives about 20.4 MiB.
    """
    elems = tokens * dim * (7 + 2 * mlp_ratio) + 2 * heads * tokens * tokens
    return int(elems * itemsize * 1.25)


# --------------------------------------------------------------------------- encoder


class TorchEncoder:
    """A loaded torch module behind the framework-neutral `EmbeddingEncoder` port."""

    def __init__(
        self,
        *,
        module: nn.Module,
        extract: Extractor,
        identity: EncoderIdentity,
        spec: ModelInputSpec,
        device: torch.device,
        dtype: torch.dtype,
        activation_bytes_per_sample: int,
        limitations: tuple[str, ...],
        resources: tuple[tuple[str, ConfigValue], ...],
    ) -> None:
        self._module: nn.Module | None = module
        self._extract = extract
        self._identity = identity
        self._spec = spec
        self._device = device
        self._dtype = dtype
        self._activation = activation_bytes_per_sample
        self._limitations = limitations
        count, nbytes = parameter_bytes(module)
        self._static: tuple[tuple[str, ConfigValue], ...] = (
            *resources,
            ("device", str(device)),
            ("parameter_count", count),
            ("parameter_bytes", nbytes),
            ("torch_threads", torch.get_num_threads()),
            ("deterministic_algorithms", torch.are_deterministic_algorithms_enabled()),
            ("platform", platform.platform()),
        )
        self._batches = 0
        self._samples = 0
        self._seconds = 0.0
        self._device_peak = 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

    @property
    def identity(self) -> EncoderIdentity:
        return self._identity

    @property
    def input_spec(self) -> ModelInputSpec:
        return self._spec

    @property
    def closed(self) -> bool:
        return self._module is None

    @property
    def module(self) -> nn.Module:
        if self._module is None:
            raise ModelInferenceError("the encoder is closed; its model was released")
        return self._module

    def _input_shape(self) -> tuple[int, int, int]:
        s = self._spec
        if s.layout is TensorLayout.CHW:
            return (s.channels, s.height, s.width)
        return (s.height, s.width, s.channels)

    def encode_batch(self, batch: object, context: EncodeContext) -> npt.NDArray[np.float32]:
        module = self.module
        x: npt.NDArray[Any] = np.asarray(batch)
        if x.ndim != 4 or x.shape[1:] != self._input_shape():
            raise UnsupportedModelInputError(
                f"expected (b, {', '.join(map(str, self._input_shape()))}), got {x.shape}"
            )
        if str(x.dtype) != self._spec.dtype:
            raise UnsupportedModelInputError(f"expected {self._spec.dtype} input, got {x.dtype}")
        if len(context.patch_ids) != x.shape[0] or context.device != str(self._device):
            raise UnsupportedModelInputError("encode context does not match the batch or device")
        if _DTYPES[context.precision] != self._dtype:
            raise UnsupportedModelInputError("encode context precision differs from the model")
        start = time.perf_counter()
        try:
            with torch.inference_mode():
                inputs = torch.from_numpy(np.ascontiguousarray(x)).to(
                    device=self._device, dtype=self._dtype
                )
                out = self._extract(module, inputs)
                if out.requires_grad:
                    raise ModelInferenceError("inference produced a tensor that tracks gradients")
                rows: npt.NDArray[np.float32] = (
                    out.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy().copy()
                )
            if self._device.type == "cuda":
                torch.cuda.synchronize(self._device)
                self._device_peak = max(
                    self._device_peak, int(torch.cuda.max_memory_allocated(self._device))
                )
            elif self._device.type == "mps":
                self._device_peak = max(self._device_peak, int(torch.mps.driver_allocated_memory()))
        except DomainError:
            raise
        except Exception as exc:  # device errors, out-of-memory, kernel failures
            raise ModelInferenceError(
                f"{self._identity.component.name} failed on batch {context.batch_index} "
                f"({context.device}): {_first_line(exc)}"
            ) from exc
        self._seconds += time.perf_counter() - start
        self._batches += 1
        self._samples += int(x.shape[0])
        return rows

    def working_bytes(self, batch_patches: int) -> int:
        """Host memory of one call beyond its input and output buffers.

        On CPU the activations live in host memory and are counted. On an accelerator they live
        in device memory (reported separately); the host keeps the output copy only.
        """
        out = batch_patches * self._identity.output_dim * 4
        if self._device.type == "cpu":
            return batch_patches * self._activation + 2 * out
        return 2 * out

    def runtime_report(self) -> EncoderRuntimeReport:
        measured: tuple[tuple[str, ConfigValue], ...] = (
            ("batches_encoded", self._batches),
            ("samples_encoded", self._samples),
            ("encode_seconds", round(self._seconds, 6)),
        )
        if self._device.type in ("cuda", "mps"):
            measured += (("device_peak_bytes", self._device_peak),)
        return EncoderRuntimeReport(self._limitations, self._static + measured)

    def close(self) -> None:
        """Release the model (idempotent). Later calls to `encode_batch` fail."""
        if self._module is None:
            return
        self._module = None
        gc.collect()
        if self._device.type == "cuda":
            torch.cuda.empty_cache()
        elif self._device.type == "mps":
            torch.mps.empty_cache()


def probe_output(
    module: nn.Module,
    extract: Extractor,
    spec: ModelInputSpec,
    *,
    device: torch.device,
    dtype: torch.dtype,
    output_dim: int,
    architecture: str,
) -> None:
    """One zero-input forward pass: the output contract is checked when the model is loaded."""
    s = spec
    shape = (
        (1, s.channels, s.height, s.width)
        if s.layout is TensorLayout.CHW
        else (
            1,
            s.height,
            s.width,
            s.channels,
        )
    )
    try:
        with torch.inference_mode():
            out = extract(module, torch.zeros(shape, device=device, dtype=dtype))
    except Exception as exc:
        raise CheckpointIncompatibleError(
            f"{architecture} cannot run the declared input {shape}: {type(exc).__name__}: {exc}"
        ) from None
    if tuple(out.shape) != (1, output_dim):
        raise CheckpointIncompatibleError(
            f"{architecture} produces {tuple(out.shape)}, the encoder declares (1, {output_dim})"
        )
    if not bool(torch.isfinite(out).all()):
        raise CheckpointIncompatibleError(f"{architecture} produces non-finite output")
