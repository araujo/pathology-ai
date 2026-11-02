"""Foundation-model encoder adapters behind the `EmbeddingEncoder` port (SPEC-006, ADR-006).

* `checkpoints`: local checkpoint binding and SHA-256 verification (standard library only);
* `torch_runtime`: the generic PyTorch runtime (device, precision, strict state-dict loading,
  bounded inference, resource reporting). It imports torch at module level and is therefore
  imported lazily, only when an encoder is created, so the base install never needs torch;
* `fixture`: a lightweight local torch encoder for tests (random weights, no meaning);
* one module per pretrained model adapter, holding everything specific to that model
  (architecture, checkpoint pin, output extraction and preprocessing profiles).

The generic modules know no model by name. Weights are never downloaded: every checkpoint is a
local file bound by the operator and verified against a pinned digest before it is read.
"""

from __future__ import annotations

from collections.abc import Mapping

from domain.embedding import Precision
from domain.errors import ModelInputContractError, ModelRuntimeUnavailableError
from domain.pipeline import ConfigValue

PLUGIN_VERSION = "0.1.0"
RUNTIME_EXTRA = "pathology-ai[embedding]"


OPTIONAL_RUNTIME_MODULES = frozenset({"torch", "timm", "torchvision"})


def runtime_unavailable(exc: ModuleNotFoundError) -> Exception:
    """The typed error for a missing optional runtime module (torch, timm). Any other missing
    module is a defect, not an optional dependency, and is returned unchanged."""
    if (exc.name or "").split(".")[0] not in OPTIONAL_RUNTIME_MODULES:
        return exc
    return ModelRuntimeUnavailableError(
        f"the model runtime needs {exc.name or 'an optional module'}, which is not installed; "
        f"install {RUNTIME_EXTRA!r}"
    )


def precision_of(config: Mapping[str, ConfigValue]) -> Precision:
    """The explicitly configured precision (`fp32` unless stated); never automatic."""
    text = config.get("precision", "fp32")
    try:
        return Precision(str(text))
    except ValueError:
        raise ModelInputContractError(f"unknown precision {text!r}") from None
