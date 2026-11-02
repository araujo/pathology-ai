"""Durable MIL artifacts (SPEC-007 §6, §10, §11): JSON documents and pickle-free checkpoints.

Documents (dataset, split and prediction manifests) are written once as
`{"body": canonical, "digest": sha256(body)}`. Reading re-derives the digest and rebuilds the
domain object, which must re-digest to the same value. Rewriting a path with different content
raises `EXPERIMENT_CONFLICT`: scientific artifacts are never silently replaced.

A checkpoint is a directory `<root>/<experiment_identity>/`:

    weights.bin      tensors as raw little-endian C-ordered bytes, concatenated by name
    checkpoint.json  the `CheckpointDescriptor` (tensor index with per-tensor SHA-256, the
                     whole-file SHA-256, experiment, history); written last, so it marks completion

No pickle is ever written or read. Loading re-hashes the weights before any tensor is returned.
A directory without `checkpoint.json` is an interrupted write and is overwritten. A complete
checkpoint is reused only if identical, and otherwise refused.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from domain.errors import (
    CheckpointIncompatibleError,
    CheckpointIntegrityError,
    CheckpointMissingError,
    DomainError,
    ExperimentConflictError,
    MILDatasetError,
)
from domain.hashing import digest, to_canonical
from domain.mil import MILDatasetManifest, SplitManifest
from domain.mil_training import (
    CheckpointDescriptor,
    PredictionManifest,
    TensorEntry,
    index_tensors,
)
from infrastructure.canonical import DecodeError, decode
from infrastructure.embedding_store import _fsync_path, _full_fsync
from ports import ModelTensor

WEIGHTS, DESCRIPTOR = "weights.bin", "checkpoint.json"
_EXPERIMENT_ID = re.compile(r"^pai-milexp-[0-9a-f]{32}$")


def _document(body: object) -> bytes:
    canonical = to_canonical(body)
    doc = {"body": canonical, "digest": digest(canonical)}
    return (json.dumps(doc, sort_keys=True, indent=1, ensure_ascii=False) + "\n").encode("utf-8")


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        _full_fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_path(path.parent)


def read_document[T](cls: type[T], path: Path, error: type[DomainError]) -> T:
    """Rebuild a domain object; any unreadable, edited or non-round-tripping document fails."""
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        body = doc["body"]
        ok = doc["digest"] == digest(body)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise error(f"{Path(path).name} is unreadable: {exc}") from None
    if not ok:
        raise error(f"{Path(path).name} fails its integrity digest")
    try:
        value = decode(cls, body)
    except DomainError:
        raise
    except (DecodeError, TypeError, ValueError) as exc:
        raise error(f"{Path(path).name} does not describe a {cls.__name__}: {exc}") from None
    if digest(value) != digest(body):
        raise error(f"{Path(path).name} does not round-trip to its digest")
    return value


def write_once(path: Path, value: object) -> None:
    """Write `value`, or accept an identical existing document; never replace another one."""
    path = Path(path)
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8")).get("body")
        except (OSError, ValueError, AttributeError):
            existing = None
        if existing is not None and digest(existing) == digest(value):
            return
        raise ExperimentConflictError(
            f"{path.name} already holds another artifact; it is never overwritten"
        )
    _atomic_write(path, _document(value))


def read_dataset(path: Path) -> MILDatasetManifest:
    return read_document(MILDatasetManifest, path, MILDatasetError)


def read_split(path: Path) -> SplitManifest:
    from domain.errors import SplitError

    return read_document(SplitManifest, path, SplitError)


def read_predictions(path: Path) -> PredictionManifest:
    from domain.errors import InvalidManifestError

    return read_document(PredictionManifest, path, InvalidManifestError)


# --------------------------------------------------------------------------- checkpoints


def tensor_index(tensors: tuple[ModelTensor, ...]) -> tuple[tuple[TensorEntry, ...], str, int]:
    return index_tensors([(t.name, t.dtype, t.shape, t.data) for t in tensors])


class DirectoryCheckpointStore:
    """Checkpoints below one root directory, one subdirectory per experiment identity."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    def _dir(self, experiment_identity: str) -> Path:
        if not _EXPERIMENT_ID.fullmatch(experiment_identity):
            raise CheckpointMissingError(f"invalid experiment identity {experiment_identity!r}")
        return self._root / experiment_identity

    def _descriptor(self, directory: Path) -> CheckpointDescriptor:
        try:
            return read_document(
                CheckpointDescriptor, directory / DESCRIPTOR, CheckpointIntegrityError
            )
        except (CheckpointIncompatibleError, CheckpointIntegrityError):
            raise
        except DomainError as exc:  # a constructor rejected the content
            raise CheckpointIntegrityError(f"{DESCRIPTOR}: {exc}") from None

    def _verified(
        self, directory: Path, descriptor: CheckpointDescriptor
    ) -> tuple[ModelTensor, ...]:
        if descriptor.experiment_identity != directory.name:
            raise CheckpointIntegrityError("the checkpoint belongs to another experiment directory")
        try:
            data = (directory / WEIGHTS).read_bytes()
        except OSError as exc:
            raise CheckpointIntegrityError(f"{WEIGHTS} is unreadable: {exc}") from None
        if len(data) != descriptor.weights_bytes:
            raise CheckpointIntegrityError(
                f"{WEIGHTS} holds {len(data)} bytes, the descriptor declares "
                f"{descriptor.weights_bytes}"
            )
        if hashlib.sha256(data).hexdigest() != descriptor.weights_sha256:
            raise CheckpointIntegrityError(f"{WEIGHTS} fails its SHA-256")
        out = []
        for t in descriptor.tensors:
            chunk = data[t.offset : t.offset + t.nbytes]
            if hashlib.sha256(chunk).hexdigest() != t.sha256:
                raise CheckpointIntegrityError(f"tensor {t.name} fails its SHA-256")
            out.append(ModelTensor(t.name, t.dtype, t.shape, chunk))
        return tuple(out)

    def find(self, experiment_identity: str) -> CheckpointDescriptor | None:
        directory = self._dir(experiment_identity)
        if not (directory / DESCRIPTOR).exists():
            return None
        descriptor = self._descriptor(directory)
        self._verified(directory, descriptor)
        return descriptor

    def load(
        self, experiment_identity: str
    ) -> tuple[CheckpointDescriptor, tuple[ModelTensor, ...]]:
        directory = self._dir(experiment_identity)
        if not (directory / DESCRIPTOR).exists():
            raise CheckpointMissingError(f"no complete checkpoint for {experiment_identity}")
        descriptor = self._descriptor(directory)
        return descriptor, self._verified(directory, descriptor)

    def load_path(self, directory: Path) -> tuple[CheckpointDescriptor, tuple[ModelTensor, ...]]:
        """Load a checkpoint directory given explicitly (e.g. on the command line)."""
        directory = Path(directory)
        if not (directory / DESCRIPTOR).exists():
            raise CheckpointMissingError(f"{directory.name} holds no complete checkpoint")
        descriptor = self._descriptor(directory)
        return descriptor, self._verified(directory, descriptor)

    def save(self, descriptor: CheckpointDescriptor, tensors: tuple[ModelTensor, ...]) -> None:
        entries, sha, size = tensor_index(tensors)
        if (entries, sha, size) != (
            descriptor.tensors,
            descriptor.weights_sha256,
            descriptor.weights_bytes,
        ):
            raise CheckpointIntegrityError("the descriptor does not index these tensors")
        directory = self._dir(descriptor.experiment_identity)
        if (directory / DESCRIPTOR).exists():
            existing = self._descriptor(directory)
            if existing.digest == descriptor.digest:
                self._verified(directory, existing)
                return  # already saved, identically
            raise ExperimentConflictError(
                f"{directory.name} already holds a checkpoint of this experiment with other "
                "results (checkpoint "
                f"{existing.checkpoint_id} vs {descriptor.checkpoint_id}); it is never overwritten"
            )
        if directory.exists():
            unrelated = sorted(
                p.name
                for p in directory.iterdir()
                if p.name not in (WEIGHTS, f".{WEIGHTS}.tmp", f".{DESCRIPTOR}.tmp")
            )
            if unrelated:
                raise ExperimentConflictError(f"{directory.name} holds unrelated files {unrelated}")
        payload = b"".join(t.data for t in sorted(tensors, key=lambda t: t.name))
        _atomic_write(directory / WEIGHTS, payload)
        _atomic_write(directory / DESCRIPTOR, _document(descriptor))
        _fsync_path(self._root)
