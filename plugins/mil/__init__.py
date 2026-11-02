"""MIL architectures behind the `MILModelFactory` port (SPEC-007 §7, ADR-007 D7).

* `mean-pooling@1`: the mean of the valid instances, then a linear classifier. A baseline.
* `attention-mil@1`: attention-based MIL pooling (Ilse, Tomczak & Welling, 2018). Instances are
  projected to a hidden representation; an optionally gated attention network scores each
  instance; invalid instances are masked; scores are normalized over the valid instances;
  the weighted sum of instance representations is classified.

Both accept any declared feature dimension and produce one logit (binary, BCE) or one logit per
class (multiclass, cross-entropy). Neither knows an encoder, a tissue or a target.

This module imports no framework. `torch_models` imports torch at module level and is
imported only when a model is created; without torch, creation raises
`MODEL_RUNTIME_UNAVAILABLE`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from domain.errors import CheckpointIncompatibleError
from domain.foundation import parse_device
from domain.mil_training import (
    LossKind,
    MILArchitectureDescriptor,
    MILModelSpec,
    TrainingConfig,
)
from plugins.models import RUNTIME_EXTRA, runtime_unavailable

if TYPE_CHECKING:
    from plugins.mil.torch_models import TorchMILModel

PLUGIN_VERSION = "0.1.0"
MEAN_POOLING = "mean-pooling"
ATTENTION_MIL = "attention-mil"

DESCRIPTORS = {
    MEAN_POOLING: MILArchitectureDescriptor(
        name=MEAN_POOLING,
        version="1",
        description="masked mean of valid instances, dropout, linear classifier",
        produces_attention=False,
        hyperparameters=(("dropout", 0.0),),
        runtime_extra=RUNTIME_EXTRA,
    ),
    ATTENTION_MIL: MILArchitectureDescriptor(
        name=ATTENTION_MIL,
        version="1",
        description=(
            "linear+ReLU instance projection, (gated) tanh attention, masked softmax over valid "
            "instances, attention-weighted sum, dropout, linear classifier"
        ),
        produces_attention=True,
        hyperparameters=(
            ("attention_dim", 128),
            ("dropout", 0.0),
            ("gated", True),
            ("hidden_dim", 128),
        ),
        runtime_extra=RUNTIME_EXTRA,
    ),
}


class TorchMILFactory:
    """Creates seeded torch MIL models of one architecture (lazy torch import)."""

    def __init__(self, architecture: str) -> None:
        self._descriptor = DESCRIPTORS[architecture]

    @property
    def name(self) -> str:
        return self._descriptor.name

    @property
    def descriptor(self) -> MILArchitectureDescriptor:
        return self._descriptor

    def runtime(self, device: str) -> tuple[tuple[str, str], ...]:
        kind, _ = parse_device(device)
        try:
            from plugins.mil import torch_models
        except ModuleNotFoundError as exc:
            raise runtime_unavailable(exc) from None
        return (("device", kind.value), ("runtime.torch", torch_models.torch_version()))

    def create(
        self,
        spec: MILModelSpec,
        *,
        loss: LossKind,
        seed: int,
        training: TrainingConfig | None,
        class_weights: tuple[float, ...] | None = None,
        device: str = "cpu",
    ) -> TorchMILModel:
        if (spec.architecture, spec.architecture_version) != (
            self._descriptor.name,
            self._descriptor.version,
        ):
            d = self._descriptor
            raise CheckpointIncompatibleError(
                f"factory {d.name}@{d.version} cannot build {spec.ref}"
            )
        parse_device(device)
        try:
            from plugins.mil.torch_models import TorchMILModel
        except ModuleNotFoundError as exc:
            raise runtime_unavailable(exc) from None
        return TorchMILModel(
            spec,
            loss=loss,
            seed=seed,
            training=training,
            class_weights=class_weights,
            device=device,
        )
