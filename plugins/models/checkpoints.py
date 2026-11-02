"""Local checkpoint binding and verification (standard library only).

A checkpoint path is operational configuration: it is bound per deployment (composition root,
CLI `--checkpoint NAME=PATH`, or `$PATHOLOGY_AI_CHECKPOINTS`) and never enters a stage config,
a fingerprint or an identity. The checkpoint's SHA-256 is its scientific identity: it is pinned
(by the adapter or the stage config) and the file is hashed before any byte is deserialized.

Nothing here downloads, authenticates or resolves remote references: hub identifiers and URLs
are rejected. Verification assumes a local filesystem with a single writer; the file's size,
inode and modification time are compared again after loading (`CheckpointStat.unchanged`) so a
file replaced between hashing and loading is rejected rather than trusted.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from domain.errors import (
    CheckpointIntegrityError,
    CheckpointMissingError,
    ModelInputContractError,
)
from domain.foundation import CheckpointExpectation, EncoderDescriptor, VerifiedCheckpoint
from domain.pipeline import ConfigValue

CHECKPOINTS_ENV = "PATHOLOGY_AI_CHECKPOINTS"
_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_REMOTE = re.compile(r"^[a-z][a-z0-9+.-]*:(//)?", re.IGNORECASE)
_CHUNK = 1 << 20


def _reject_remote(text: str) -> None:
    # A Windows drive letter is a single character before the colon; anything longer that looks
    # like "scheme:" (https://, s3://, hf-hub:...) is a remote reference.
    m = _REMOTE.match(text)
    if m and len(m.group(0).rstrip(":/")) > 1:
        raise CheckpointMissingError(
            f"remote checkpoint reference {text!r} is not supported; weights are never "
            "downloaded or authenticated. Bind a local file."
        )


def parse_bindings(items: Iterable[str]) -> dict[str, Path]:
    """`NAME=PATH` items (CLI flags or `$PATHOLOGY_AI_CHECKPOINTS` entries)."""
    out: dict[str, Path] = {}
    for item in items:
        name, sep, path = item.strip().partition("=")
        name = name.strip()
        if not sep or not _NAME.match(name) or not path.strip():
            raise ModelInputContractError(f"invalid checkpoint binding {item!r}; use NAME=PATH")
        if name in out:
            raise ModelInputContractError(f"checkpoint for {name!r} is bound twice")
        _reject_remote(path.strip())
        out[name] = Path(path.strip()).expanduser()
    return out


class CheckpointLocator:
    """Encoder name -> local checkpoint file. Operational; never part of any identity."""

    def __init__(self, bindings: Mapping[str, Path] | None = None) -> None:
        self._paths: dict[str, Path] = {}
        for name, path in (bindings or {}).items():
            if not _NAME.match(name):
                raise ModelInputContractError(f"invalid encoder name {name!r}")
            _reject_remote(str(path))
            self._paths[name] = Path(path).expanduser()

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> CheckpointLocator:
        """`$PATHOLOGY_AI_CHECKPOINTS`: `NAME=PATH` entries separated by the path separator."""
        text = (os.environ if environ is None else environ).get(CHECKPOINTS_ENV, "")
        items = [i for i in text.split(os.pathsep) if i.strip()]
        return cls(parse_bindings(items))

    def overridden_by(self, bindings: Mapping[str, Path]) -> CheckpointLocator:
        return CheckpointLocator({**self._paths, **bindings})

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._paths))

    def path_for(self, name: str) -> Path:
        path = self._paths.get(name)
        if path is None:
            raise CheckpointMissingError(
                f"no local checkpoint is bound for encoder {name!r}; bind one with "
                f"--checkpoint {name}=PATH or ${CHECKPOINTS_ENV} (weights are never downloaded)"
            )
        return path


def expected_checkpoint(
    descriptor: EncoderDescriptor, config: Mapping[str, ConfigValue]
) -> CheckpointExpectation:
    """The digest a run must verify: the adapter's pin, or `checkpoint_sha256` from the stage
    config. Both may be given only if they agree. Unpinned weights are never loaded."""
    configured = config.get("checkpoint_sha256")
    if configured is not None and not isinstance(configured, str):
        raise ModelInputContractError("checkpoint_sha256 must be a hex string")
    pinned = descriptor.checkpoint
    if pinned is not None:
        if configured is not None and configured.removeprefix("sha256:") != pinned.sha256:
            raise CheckpointIntegrityError(
                f"stage config pins checkpoint {configured}, but encoder {descriptor.name} "
                f"accepts only sha256:{pinned.sha256}"
            )
        return pinned
    if configured is None:
        raise CheckpointIntegrityError(
            f"encoder {descriptor.name} has no pinned checkpoint digest; declare "
            "checkpoint_sha256 in the stage config (unverified weights are never loaded)"
        )
    return CheckpointExpectation(
        sha256=configured.removeprefix("sha256:"), format="torch-state-dict"
    )


@dataclass(frozen=True, slots=True)
class CheckpointStat:
    size: int
    inode: int
    mtime_ns: int

    @staticmethod
    def of(path: Path) -> CheckpointStat:
        st = path.stat()
        return CheckpointStat(st.st_size, st.st_ino, st.st_mtime_ns)

    def unchanged(self, path: Path) -> None:
        try:
            now = CheckpointStat.of(path)
        except OSError as exc:
            raise CheckpointIntegrityError(f"checkpoint disappeared while loading: {exc}") from None
        if now != self:
            raise CheckpointIntegrityError(
                "checkpoint changed on disk between verification and loading; refusing it"
            )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def verify_checkpoint(
    path: Path, expected: CheckpointExpectation
) -> tuple[VerifiedCheckpoint, CheckpointStat]:
    """Hash the local file and compare it with the pinned digest (and size, when pinned)."""
    _reject_remote(str(path))
    if not path.exists():
        raise CheckpointMissingError("the bound checkpoint file does not exist")
    if not path.is_file():
        raise CheckpointMissingError("the bound checkpoint is not a regular file")
    stat = CheckpointStat.of(path)
    if expected.size_bytes is not None and stat.size != expected.size_bytes:
        raise CheckpointIntegrityError(
            f"checkpoint has {stat.size} bytes, expected {expected.size_bytes}; it is truncated, "
            "corrupt or a different checkpoint"
        )
    try:
        actual = sha256_file(path)
    except OSError as exc:
        raise CheckpointMissingError(f"the bound checkpoint cannot be read: {exc}") from None
    if actual != expected.sha256:
        raise CheckpointIntegrityError(
            f"checkpoint sha256 is {actual}, expected {expected.sha256}; it is corrupt or a "
            "different checkpoint"
        )
    stat.unchanged(path)
    return VerifiedCheckpoint(actual, stat.size, expected.format), stat
