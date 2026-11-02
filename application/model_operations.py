"""Generic stage operations that delegate to the bound `Model` port."""

from __future__ import annotations

from collections.abc import Mapping

from application.version import __version__
from domain.errors import ExecutionUnsupportedError
from domain.ids import Version
from domain.pipeline import OperationRef
from domain.provenance import ComponentVersion
from ports import DataValue, StageContext

MODEL_INVOKE = OperationRef("model.invoke", Version("1"))


class ModelInvocationOperation:
    """Runs whatever model is bound to the stage's slot on its single input."""

    @property
    def operation(self) -> OperationRef:
        return MODEL_INVOKE

    @property
    def component(self) -> ComponentVersion:
        return ComponentVersion("operation", str(MODEL_INVOKE), __version__)

    def run(self, inputs: Mapping[str, DataValue], ctx: StageContext) -> Mapping[str, object]:
        if ctx.model is None:
            raise ExecutionUnsupportedError(f"stage {ctx.stage.stage_id} has no model plugin")
        (value,) = inputs.values()
        return {ctx.stage.outputs[0].name: ctx.model.predict(value, ctx)}
