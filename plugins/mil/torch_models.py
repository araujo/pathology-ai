"""PyTorch MIL models and the `MILModel` handle. Imports torch: import lazily (`plugins.mil`).

Determinism contract (CPU):
* Initialization runs under `torch.random.fork_rng` seeded with the experiment seed, so it does
  not depend on, or disturb, the process's global generator.
* Training keeps its own generator state (dropout) and restores the global one after each step.
* Every forward and backward pass runs with `torch.use_deterministic_algorithms(True)`. The
  previous setting is restored afterwards.
* Identical seed, data, configuration, torch version, device type and thread count give
  bit-identical weights. Across devices or thread counts, results agree only within numerical
  tolerance; device and torch version are bound into the experiment identity.

Masks: invalid (padded) instances are excluded exactly. Mean pooling multiplies them by zero
and divides by the valid count. Attention sets their scores to `-inf` before the softmax, so
their weights are exactly zero.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

import numpy as np
import torch
from torch import nn

from domain.errors import (
    CheckpointIncompatibleError,
    DeviceUnavailableError,
    EmptyBagError,
    ModelInferenceError,
)
from domain.foundation import DeviceKind, parse_device
from domain.mil_training import LossKind, MILModelSpec, OptimizerName, TrainingConfig
from ports import MILBatch, MILOutput, ModelTensor

_NUMPY_DTYPES = {
    torch.float32: "float32",
    torch.float64: "float64",
    torch.float16: "float16",
    torch.int64: "int64",
    torch.int32: "int32",
    torch.bool: "bool",
}


def torch_version() -> str:
    return str(torch.__version__)


def _device(text: str) -> torch.device:
    kind, index = parse_device(text)
    if kind is DeviceKind.CUDA and (
        not torch.cuda.is_available() or (index is not None and index >= torch.cuda.device_count())
    ):
        raise DeviceUnavailableError(f"device {text!r} was requested but is not available")
    if kind is DeviceKind.MPS and not torch.backends.mps.is_available():
        raise DeviceUnavailableError(f"device {text!r} was requested but MPS is not available")
    return torch.device(text)


@contextlib.contextmanager
def _deterministic() -> Iterator[None]:
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous)


# --------------------------------------------------------------------------- modules


class MeanPoolingMIL(nn.Module):
    def __init__(self, feature_dim: int, outputs: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(feature_dim, outputs)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, None]:
        m = mask.unsqueeze(-1).to(x.dtype)
        pooled = (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        return self.classifier(self.dropout(pooled)), None


class AttentionMIL(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        outputs: int,
        *,
        hidden_dim: int,
        attention_dim: int,
        gated: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        self.project = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ReLU())
        self.attention_v = nn.Linear(hidden_dim, attention_dim)
        self.attention_u = nn.Linear(hidden_dim, attention_dim) if gated else None
        self.attention_w = nn.Linear(attention_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, outputs)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.project(x)  # (B, N, H)
        a = torch.tanh(self.attention_v(h))
        if self.attention_u is not None:
            a = a * torch.sigmoid(self.attention_u(h))
        scores = self.attention_w(a).squeeze(-1)  # (B, N)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1)  # exactly 0 on padding
        pooled = torch.bmm(weights.unsqueeze(1), h).squeeze(1)  # (B, H)
        return self.classifier(self.dropout(pooled)), weights


def build_module(spec: MILModelSpec) -> nn.Module:
    hp = dict(spec.hyperparameters)
    if spec.architecture == "mean-pooling":
        return MeanPoolingMIL(spec.feature_dim, spec.num_outputs, float(hp["dropout"]))
    if spec.architecture == "attention-mil":
        return AttentionMIL(
            spec.feature_dim,
            spec.num_outputs,
            hidden_dim=int(hp["hidden_dim"]),
            attention_dim=int(hp["attention_dim"]),
            gated=bool(hp["gated"]),
            dropout=float(hp["dropout"]),
        )
    raise CheckpointIncompatibleError(f"unknown MIL architecture {spec.ref}")


# --------------------------------------------------------------------------- handle


class TorchMILModel:
    def __init__(
        self,
        spec: MILModelSpec,
        *,
        loss: LossKind,
        seed: int,
        training: TrainingConfig | None,
        class_weights: tuple[float, ...] | None,
        device: str,
    ) -> None:
        self._spec = spec
        self._loss_kind = loss
        self._device = _device(device)
        self._training = training
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            module = build_module(spec)
            self._rng = torch.get_rng_state()
        self._module: nn.Module | None = module.to(self._device, torch.float32)
        weight: torch.Tensor | None = None
        if class_weights is not None:
            w = torch.tensor(class_weights, dtype=torch.float32, device=self._device)
            if loss is LossKind.BCE_WITH_LOGITS:
                if len(class_weights) != 2:
                    raise CheckpointIncompatibleError("binary class weights need two values")
                weight = (w[1] / w[0]).reshape(1)
            else:
                weight = w
        self._criterion: nn.Module = (
            nn.BCEWithLogitsLoss(pos_weight=weight)
            if loss is LossKind.BCE_WITH_LOGITS
            else nn.CrossEntropyLoss(weight=weight)
        )
        self._optimizer: torch.optim.Optimizer | None = None
        if training is not None:
            o = training.optimizer
            params = list(module.parameters())
            if o.name is OptimizerName.ADAM:
                self._optimizer = torch.optim.Adam(
                    params, lr=o.learning_rate, weight_decay=o.weight_decay
                )
            elif o.name is OptimizerName.ADAMW:
                self._optimizer = torch.optim.AdamW(
                    params, lr=o.learning_rate, weight_decay=o.weight_decay
                )
            else:
                self._optimizer = torch.optim.SGD(
                    params, lr=o.learning_rate, momentum=o.momentum, weight_decay=o.weight_decay
                )

    @property
    def spec(self) -> MILModelSpec:
        return self._spec

    @property
    def module(self) -> nn.Module:
        if self._module is None:
            raise ModelInferenceError("the MIL model is closed")
        return self._module

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.module.parameters())

    def _tensors(self, batch: MILBatch) -> tuple[torch.Tensor, torch.Tensor]:
        x = np.asarray(batch.features)
        m = np.asarray(batch.mask)
        if x.ndim != 3 or x.shape[2] != self._spec.feature_dim or x.dtype != np.float32:
            raise CheckpointIncompatibleError(
                f"batch features {x.shape} {x.dtype} do not fit feature_dim "
                f"{self._spec.feature_dim} float32"
            )
        if m.shape != x.shape[:2] or m.dtype != np.bool_:
            raise EmptyBagError("the mask must be a (bags, instances) boolean array")
        if not bool(m.any(axis=1).all()):
            raise EmptyBagError("every bag needs at least one valid instance")
        return (
            torch.from_numpy(np.ascontiguousarray(x)).to(self._device),
            torch.from_numpy(np.ascontiguousarray(m)).to(self._device),
        )

    def _targets(self, batch: MILBatch) -> torch.Tensor:
        if batch.targets is None:
            raise CheckpointIncompatibleError("a labelled batch is required")
        if self._loss_kind is LossKind.BCE_WITH_LOGITS:
            return torch.tensor(batch.targets, dtype=torch.float32, device=self._device).reshape(
                -1, 1
            )
        return torch.tensor(batch.targets, dtype=torch.int64, device=self._device)

    def _output(
        self, logits: torch.Tensor, weights: torch.Tensor | None, batch: MILBatch
    ) -> MILOutput:
        rows = logits.detach().to("cpu", torch.float64).numpy()
        attention = None
        if weights is not None:
            w = weights.detach().to("cpu", torch.float64).numpy()
            attention = tuple(
                tuple(float(v) for v in w[b, : s.selected_count])
                for b, s in enumerate(batch.selections)
            )
        return MILOutput(tuple(tuple(float(v) for v in r) for r in rows), attention)

    def train_batch(self, batch: MILBatch) -> float:
        if self._optimizer is None or self._training is None:
            raise ModelInferenceError("this MIL model was created for inference only")
        module = self.module
        x, m = self._tensors(batch)
        target = self._targets(batch)
        with torch.random.fork_rng(devices=[]), _deterministic():
            torch.set_rng_state(self._rng)
            module.train()
            logits, _ = module(x, m)
            loss = self._criterion(logits, target)
            if not bool(torch.isfinite(loss)):
                raise ModelInferenceError("training produced a non-finite loss")
            self._optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self._training.gradient_clip_norm is not None:
                nn.utils.clip_grad_norm_(module.parameters(), self._training.gradient_clip_norm)
            self._optimizer.step()
            self._rng = torch.get_rng_state()
        return float(loss.detach().item())

    def evaluate_batch(self, batch: MILBatch) -> tuple[float, MILOutput]:
        module = self.module
        x, m = self._tensors(batch)
        target = self._targets(batch)
        with torch.inference_mode(), _deterministic():
            module.eval()
            logits, weights = module(x, m)
            loss = float(self._criterion(logits, target).item())
        return loss, self._output(logits, weights, batch)

    def predict(self, batch: MILBatch) -> MILOutput:
        module = self.module
        x, m = self._tensors(batch)
        with torch.inference_mode(), _deterministic():
            module.eval()
            logits, weights = module(x, m)
        return self._output(logits, weights, batch)

    def state(self) -> tuple[ModelTensor, ...]:
        out = []
        for name, t in sorted(self.module.state_dict().items()):
            a = t.detach().to("cpu").contiguous().numpy()
            dtype = _NUMPY_DTYPES.get(t.dtype)
            if dtype is None:
                raise CheckpointIncompatibleError(f"tensor {name} has unsupported dtype {t.dtype}")
            le = a.astype(a.dtype.newbyteorder("<"), copy=False)
            out.append(ModelTensor(name, dtype, tuple(int(d) for d in a.shape), le.tobytes()))
        return tuple(out)

    def load_state(self, tensors: tuple[ModelTensor, ...]) -> None:
        module = self.module
        expected = module.state_dict()
        given = {t.name: t for t in tensors}
        problems = sorted(set(expected) ^ set(given))
        for name, ref in expected.items():
            t = given.get(name)
            if t is None:
                continue
            if t.shape != tuple(ref.shape) or t.dtype != _NUMPY_DTYPES.get(ref.dtype):
                problems.append(f"{name} {t.dtype}{list(t.shape)}!={ref.dtype}{list(ref.shape)}")
        if problems:
            raise CheckpointIncompatibleError(
                f"checkpoint does not fit {self._spec.ref}: {', '.join(problems[:5])}"
            )
        state: dict[str, Any] = {}
        for name, t in given.items():
            dt = np.dtype(t.dtype).newbyteorder("<")
            array = np.frombuffer(t.data, dtype=dt).reshape(t.shape).astype(dt.newbyteorder("="))
            state[name] = torch.from_numpy(array.copy())
        module.load_state_dict(state, strict=True)

    def close(self) -> None:
        self._module = None
        self._optimizer = None
        if self._device.type == "cuda":
            torch.cuda.empty_cache()
        elif self._device.type == "mps":
            torch.mps.empty_cache()
