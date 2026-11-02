"""Rebuild frozen domain dataclasses from their canonical JSON form (`domain.hashing`).

`decode(cls, data)` follows each dataclass's resolved type hints: dataclasses, `StrEnum`s,
homogeneous `tuple[X, ...]`, fixed `tuple[A, B]`, `X | None` and JSON scalars (kept exactly as
parsed). Domain validation runs in each constructor. Callers then compare `digest(result)` with
the digest of the source body, so any decoding that does not reproduce the document exactly
fails closed.
"""

from __future__ import annotations

import dataclasses
import types
import typing
from enum import Enum
from functools import cache
from typing import Any

_SCALARS = (str, int, float, bool)


class DecodeError(ValueError):
    pass


@cache
def _hints(cls: type) -> dict[str, Any]:
    return typing.get_type_hints(cls)


def decode[T](cls: type[T], data: Any) -> T:
    result: T = _decode(cls, data, cls.__name__)
    return result


def _decode(tp: Any, data: Any, where: str) -> Any:
    origin = typing.get_origin(tp)
    if origin in (types.UnionType, typing.Union):
        args = typing.get_args(tp)
        if data is None:
            if type(None) in args:
                return None
            raise DecodeError(f"{where}: null is not allowed")
        options = [a for a in args if a is not type(None)]
        if all(a in _SCALARS for a in options):
            if isinstance(data, _SCALARS):
                return data
            raise DecodeError(f"{where}: expected a scalar")
        if len(options) == 1:
            return _decode(options[0], data, where)
        raise DecodeError(f"{where}: ambiguous union {tp}")
    if origin is tuple:
        args = typing.get_args(tp)
        if not isinstance(data, list):
            raise DecodeError(f"{where}: expected a list")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_decode(args[0], v, f"{where}[{i}]") for i, v in enumerate(data))
        if len(args) != len(data):
            raise DecodeError(f"{where}: expected {len(args)} items")
        return tuple(
            _decode(a, v, f"{where}[{i}]") for i, (a, v) in enumerate(zip(args, data, strict=True))
        )
    if isinstance(tp, type) and issubclass(tp, Enum):
        try:
            return tp(data)
        except ValueError:
            raise DecodeError(f"{where}: invalid {tp.__name__} {data!r}") from None
    if isinstance(tp, type) and dataclasses.is_dataclass(tp):
        if not isinstance(data, dict):
            raise DecodeError(f"{where}: expected an object")
        hints = _hints(tp)
        names = {f.name for f in dataclasses.fields(tp)}
        if set(data) != names:
            raise DecodeError(f"{where}: fields {sorted(set(data) ^ names)} differ")
        kwargs = {n: _decode(hints[n], data[n], f"{where}.{n}") for n in names}
        return tp(**kwargs)
    if tp in _SCALARS:
        if tp is float and isinstance(data, int) and not isinstance(data, bool):
            return data
        if not isinstance(data, tp) or (tp is int and isinstance(data, bool)):
            raise DecodeError(f"{where}: expected {tp.__name__}")
        return data
    raise DecodeError(f"{where}: unsupported type {tp}")
