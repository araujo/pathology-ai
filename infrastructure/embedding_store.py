"""NumPy `.npy` embedding store with JSON commit records (SPEC-005 §8, ADR-005 D6/D7).

One store is one directory `<root>/<store_id>/`:

    header.json          run identity, layout and the row -> patch mapping (written once)
    embeddings.npy       preallocated (rows, dim) array, written in place via `open_memmap`
    level0.npy           (rows, 4) float64 canonical level-zero rectangles (x, y, w, h), row-aligned
    chunks/<index>.json  one commit record per chunk: row range, SHA-256 of its bytes, patch ids
    manifest.json        the final embedding manifest; its presence is the completion marker

Every JSON document is `{"body": ..., "digest": sha256(canonical body)}`, so a torn or edited
document is detected. The preallocated array says nothing about completeness: rows count only
when a commit record exists *and* the bytes still hash to it, and the store is complete only
when `manifest.json` exists and the full coverage re-verifies.

Write order (crash consistency):

1. creation: the arrays are allocated and synced, then `header.json` is written last;
2. chunk: rows are written into the map, `flush()`ed (msync) and the file is synced; only then is
   the commit record written to a temporary file, synced, renamed into place and the directory
   synced;
3. finalize: every chunk is re-hashed; then `manifest.json` is written the same way.

A crash before step 2's rename leaves the chunk uncommitted; it is recomputed on resume.

Filesystem assumptions (not transactional guarantees):

* a local POSIX filesystem where `rename` within one directory is atomic and `fsync` on a file
  and on its directory makes data and the directory entry durable. On macOS `F_FULLFSYNC` is
  used when available, because plain `fsync` there does not flush the drive cache;
* exactly one writer per store. There is no lock: concurrent writers to one store are
  unsupported. Network filesystems and object stores are out of scope;
* integrity is SHA-256 over the bytes; it detects corruption, it does not authenticate anyone.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from domain.embedding import (
    EMBEDDING_STORE_SCHEMA,
    ChunkCommit,
    DeclaredTransform,
    DurableStoreReference,
    EmbeddingManifest,
    EmbeddingRecord,
    EmbeddingRunIdentity,
    EmbeddingStoreHeader,
    EncoderIdentity,
    ModelInputSpec,
    Normalization,
    Precision,
    RecoveryPolicy,
    StoreInspection,
    StoreLayout,
    StoreState,
    StoreVerification,
    TensorLayout,
    TransformChain,
    patch_ids_digest,
)
from domain.errors import (
    EmbeddingOutputError,
    EmbeddingStoreConflictError,
    EmbeddingStoreCorruptError,
    EmbeddingStoreError,
    EmbeddingStoreIncompatibleError,
    EmbeddingStoreIncompleteError,
)
from domain.geometry import CoordinateFrame, Level0Rect, LevelRect
from domain.hashing import digest, to_canonical
from domain.ids import ImageId, ModelId, Version
from domain.image import ChannelOrder
from domain.model import ModelRef
from domain.preprocessing import TransformStep
from domain.provenance import ComponentVersion

FORMAT = "numpy-npy-directory/1"
CHUNK_SCHEMA = "pathology-ai.embedding-chunk/1"
CONTENT_SCHEMA = b"pathology-ai.embedding-content/1\n"
HEADER, DATA, COORDS, CHUNKS, MANIFEST = (
    "header.json",
    "embeddings.npy",
    "level0.npy",
    "chunks",
    "manifest.json",
)
_STORE_ID = re.compile(r"^pai-emb-[0-9a-f]{32}$")
_CHUNK_FILE = re.compile(r"^(\d{8})\.json$")
# Pseudo chunk indices used to report corruption outside the chunk records.
MANIFEST_ENTRY, COORDS_ENTRY, DATA_ENTRY = -1, -2, -3

Rows = npt.NDArray[Any]


# --------------------------------------------------------------------------- durable files


_F_FULLFSYNC: int | None = getattr(fcntl, "F_FULLFSYNC", None)


def _full_fsync(fd: int) -> None:
    """`fsync`, upgraded to `F_FULLFSYNC` where the platform offers it (macOS)."""
    if _F_FULLFSYNC is not None:
        try:
            fcntl.fcntl(fd, _F_FULLFSYNC)
            return
        except OSError:
            pass
    os.fsync(fd)


def _fsync_path(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        _full_fsync(fd)
    finally:
        os.close(fd)


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        _full_fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_path(path.parent)


def _document(body: object) -> bytes:
    canonical = to_canonical(body)
    doc = {"body": canonical, "digest": digest(canonical)}
    return (json.dumps(doc, sort_keys=True, indent=1, ensure_ascii=False) + "\n").encode("utf-8")


def _read_document(path: Path) -> dict[str, Any]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        body = doc["body"]
        ok = isinstance(body, dict) and doc["digest"] == digest(body)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise EmbeddingStoreCorruptError(f"{path.name} is unreadable: {exc}") from None
    if not ok:
        raise EmbeddingStoreCorruptError(f"{path.name} fails its integrity digest")
    result: dict[str, Any] = body
    return result


# --------------------------------------------------------------------------- header parsing


def _cv(d: dict[str, Any]) -> ComponentVersion:
    return ComponentVersion(d["role"], d["name"], d["version"])


def _pair(v: list[float] | None) -> tuple[float, float] | None:
    return None if v is None else (float(v[0]), float(v[1]))


def _spec(d: dict[str, Any]) -> ModelInputSpec:
    n = d["normalization"]
    return ModelInputSpec(
        width=d["width"],
        height=d["height"],
        channel_order=ChannelOrder(d["channel_order"]),
        channels=d["channels"],
        dtype=d["dtype"],
        scaled_range=(float(d["scaled_range"][0]), float(d["scaled_range"][1])),
        layout=TensorLayout(d["layout"]),
        normalization=None if n is None else Normalization(tuple(n["mean"]), tuple(n["std"])),
        pixel_spacing_um=d["pixel_spacing_um"],
        accepts_variable_spacing=d["accepts_variable_spacing"],
        accepts_unknown_spacing=d["accepts_unknown_spacing"],
        spacing_tolerance=d["spacing_tolerance"],
        size_divisor=d["size_divisor"],
        precisions=frozenset(Precision(p) for p in d["precisions"]),
    )


def _encoder(d: dict[str, Any]) -> EncoderIdentity:
    m = d["model"]
    return EncoderIdentity(
        component=_cv(d["component"]),
        model=None
        if m is None
        else ModelRef(ModelId(m["model_id"]["value"]), Version(m["version"]["value"])),
        weights_checksum=d["weights_checksum"],
        synthetic=d["synthetic"],
        parameters=tuple((k, v) for k, v in d["parameters"]),
        output_dim=d["output_dim"],
        output_dtype=d["output_dtype"],
    )


def _chain(d: dict[str, Any]) -> TransformChain:
    return TransformChain(
        tuple(
            DeclaredTransform(
                TransformStep(t["step"]["name"], tuple((k, v) for k, v in t["step"]["params"])),
                _cv(t["implementation"]),
            )
            for t in d["steps"]
        )
    )


def _record(d: dict[str, Any]) -> EmbeddingRecord:
    lr, l0 = d["level_rect"], d["level0"]
    return EmbeddingRecord(
        row=d["row"],
        patch_id=d["patch_id"],
        plan_index=d["plan_index"],
        level_rect=LevelRect(lr["level"], lr["x"], lr["y"], lr["width"], lr["height"]),
        level0=Level0Rect(
            l0["x"], l0["y"], l0["width"], l0["height"], CoordinateFrame(l0["frame"])
        ),
        effective_spacing_um=_pair(d["effective_spacing_um"]),
        field_of_view_um=_pair(d["field_of_view_um"]),
    )


def header_from_body(body: dict[str, Any]) -> EmbeddingStoreHeader:
    """Rebuild a header from its canonical JSON body; the result must re-digest identically."""
    try:
        i = body["identity"]
        identity = EmbeddingRunIdentity(
            image_id=ImageId(i["image_id"]["value"]),
            source_integrity=i["source_integrity"],
            scene=i["scene"],
            plan_digest=i["plan_digest"],
            patch_manifest_digest=i["patch_manifest_digest"],
            records_digest=i["records_digest"],
            encoder=_encoder(i["encoder"]),
            input_spec=_spec(i["input_spec"]),
            transforms=_chain(i["transforms"]),
            precision=Precision(i["precision"]),
            device=i["device"],
            schema_version=i["schema_version"],
        )
        lay = body["layout"]
        header = EmbeddingStoreHeader(
            store_id=body["store_id"],
            identity=identity,
            layout=StoreLayout(lay["rows"], lay["dim"], lay["dtype"], lay["chunk_rows"]),
            records=tuple(_record(r) for r in body["records"]),
            schema_version=body["schema_version"],
        )
    except EmbeddingStoreError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise EmbeddingStoreCorruptError(f"store header cannot be parsed: {exc}") from None
    if header.digest != digest(body):
        raise EmbeddingStoreCorruptError("store header does not round-trip to its digest")
    return header


# --------------------------------------------------------------------------- store


class NumpyEmbeddingStore:
    """One store directory. Writable stores belong to exactly one writer at a time."""

    def __init__(self, directory: Path, header: EmbeddingStoreHeader, *, writable: bool) -> None:
        self._dir = directory
        self._header = header
        self._writable = writable
        self._array: np.memmap[Any, np.dtype[Any]] | None = None
        self._written: dict[int, str] = {}
        self._session_commits: set[int] = set()
        self._committed: set[int] | None = None
        self._closed = False

    # -------------------------------------------------------------- properties

    @property
    def header(self) -> EmbeddingStoreHeader:
        return self._header

    @property
    def directory(self) -> Path:
        return self._dir

    @property
    def reference(self) -> DurableStoreReference:
        return DurableStoreReference(
            FORMAT, EMBEDDING_STORE_SCHEMA, self._header.store_id, str(self._dir)
        )

    @property
    def _layout(self) -> StoreLayout:
        return self._header.layout

    def __enter__(self) -> NumpyEmbeddingStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._array is not None:
            if self._writable:
                self._array.flush()
            self._array = None
        self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise EmbeddingStoreError("the embedding store is closed")

    def _require_writable(self) -> None:
        self._require_open()
        if not self._writable:
            raise EmbeddingStoreConflictError("the store was opened read-only")

    # -------------------------------------------------------------- data access

    def _data(self) -> np.memmap[Any, np.dtype[Any]]:
        if self._array is None:
            _check_npy(self._dir / DATA, self._layout)
            mode = "r+" if self._writable else "r"
            try:
                self._array = np.lib.format.open_memmap(self._dir / DATA, mode=mode)
            except (OSError, ValueError) as exc:
                raise EmbeddingStoreCorruptError(f"{DATA} cannot be mapped: {exc}") from None
        return self._array

    def _chunk_sha(self, index: int) -> str:
        start, stop = self._layout.chunk_range(index)
        return hashlib.sha256(np.ascontiguousarray(self._data()[start:stop]).tobytes()).hexdigest()

    def _commit(self, index: int) -> ChunkCommit:
        start, stop = self._layout.chunk_range(index)
        return ChunkCommit(
            index=index,
            start=start,
            stop=stop,
            rows_sha256=self._chunk_sha(index),
            patch_ids_digest=patch_ids_digest(self._header.chunk_records(index)),
        )

    def read_rows(self, start: int, stop: int) -> Rows:
        self._require_open()
        if not 0 <= start < stop <= self._layout.rows:
            raise EmbeddingStoreConflictError(f"rows [{start}, {stop}) outside the store")
        return np.array(self._data()[start:stop])

    def read_selected(self, indices: tuple[int, ...]) -> Rows:
        """Copies of the given rows only (sorted, unique, in range), in index order."""
        self._require_open()
        if not indices or list(indices) != sorted(set(indices)):
            raise EmbeddingStoreConflictError(
                "selected rows must be a sorted, unique, non-empty set"
            )
        if indices[0] < 0 or indices[-1] >= self._layout.rows:
            raise EmbeddingStoreConflictError(f"rows outside 0..{self._layout.rows - 1}")
        return np.asarray(self._data()[np.asarray(indices, dtype=np.int64)])

    # -------------------------------------------------------------- inspection

    def _chunk_path(self, index: int) -> Path:
        return self._dir / CHUNKS / f"{index:08d}.json"

    def _record_files(self) -> Iterator[tuple[int, Path]]:
        folder = self._dir / CHUNKS
        for path in sorted(folder.iterdir()) if folder.is_dir() else ():
            match = _CHUNK_FILE.fullmatch(path.name)
            if match:  # temporary files from interrupted commits are ignored
                yield int(match.group(1)), path

    def _check_record(self, index: int, path: Path) -> ChunkCommit:
        body = _read_document(path)
        expected = self._commit_body(self._expected_commit(index))
        if body.get("schema") != CHUNK_SCHEMA:
            raise EmbeddingStoreCorruptError(f"chunk {index}: unknown record schema")
        if (
            body.get("store_id") != self._header.store_id
            or body.get("run_identity") != self._header.identity.value
        ):
            raise EmbeddingStoreCorruptError(f"chunk {index}: record belongs to another store")
        recorded = body.get("commit", {})
        for key in ("index", "start", "stop", "patch_ids_digest"):
            if recorded.get(key) != expected["commit"][key]:
                raise EmbeddingStoreCorruptError(f"chunk {index}: record {key} does not match")
        actual = self._chunk_sha(index)
        if recorded.get("rows_sha256") != actual:
            raise EmbeddingStoreCorruptError(f"chunk {index}: stored rows fail their SHA-256")
        return ChunkCommit(
            index, recorded["start"], recorded["stop"], actual, recorded["patch_ids_digest"]
        )

    def _expected_commit(self, index: int) -> ChunkCommit:
        start, stop = self._layout.chunk_range(index)
        return ChunkCommit(
            index, start, stop, "", patch_ids_digest(self._header.chunk_records(index))
        )

    def _commit_body(self, commit: ChunkCommit) -> dict[str, Any]:
        return {
            "schema": CHUNK_SCHEMA,
            "store_id": self._header.store_id,
            "run_identity": self._header.identity.value,
            "commit": to_canonical(commit),
        }

    def _check_coords(self) -> None:
        try:
            coords = np.load(self._dir / COORDS, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise EmbeddingStoreCorruptError(f"{COORDS} is unreadable: {exc}") from None
        if not np.array_equal(coords, _coords(self._header)):
            raise EmbeddingStoreCorruptError(f"{COORDS} differs from the header's records")

    def inspect(self) -> StoreInspection:
        """Re-verify every commit record against the stored bytes. Nothing is trusted by shape."""
        self._require_open()
        committed: list[ChunkCommit] = []
        corrupt: list[tuple[int, str]] = []
        try:
            self._data()
        except EmbeddingStoreCorruptError as exc:
            corrupt.append((DATA_ENTRY, str(exc)))
        try:
            self._check_coords()
        except EmbeddingStoreCorruptError as exc:
            corrupt.append((COORDS_ENTRY, str(exc)))
        if not corrupt:
            for index, path in self._record_files():
                try:
                    if index >= self._layout.chunk_count:
                        raise EmbeddingStoreCorruptError(f"chunk {index} is outside the layout")
                    committed.append(self._check_record(index, path))
                except EmbeddingStoreCorruptError as exc:
                    corrupt.append((index, str(exc)))
        done = {c.index for c in committed}
        missing = tuple(i for i in range(self._layout.chunk_count) if i not in done)
        manifest_digest: str | None = None
        manifest_path = self._dir / MANIFEST
        if manifest_path.exists():
            try:
                body = _read_document(manifest_path)
                manifest_digest = digest(body)
                if not corrupt and not missing:
                    self._check_manifest_body(body, self._content(tuple(committed)))
                elif not corrupt:
                    raise EmbeddingStoreCorruptError(
                        f"{MANIFEST} exists but chunks {list(missing)} are not committed"
                    )
            except EmbeddingStoreCorruptError as exc:
                corrupt.append((MANIFEST_ENTRY, str(exc)))
                manifest_digest = None
        self._committed = done
        if corrupt:
            state = StoreState.CORRUPT
        elif manifest_digest is not None:
            state = StoreState.FINALIZED
        elif not committed:
            state = StoreState.CREATED
        elif self._session_commits:
            state = StoreState.IN_PROGRESS
        else:
            state = StoreState.INTERRUPTED
        return StoreInspection(
            state=state,
            committed=tuple(committed),
            missing=missing,
            corrupt=tuple(sorted(corrupt)),
            finalized_manifest_digest=manifest_digest,
        )

    def _content(self, commits: tuple[ChunkCommit, ...]) -> StoreVerification:
        lay = self._layout
        h = hashlib.sha256(CONTENT_SCHEMA)
        h.update(f"{lay.dtype}:{lay.rows}x{lay.dim}\n".encode())
        for c in sorted(commits, key=lambda c: c.index):
            h.update(np.ascontiguousarray(self._data()[c.start : c.stop]).tobytes())
        return StoreVerification(
            rows=lay.rows,
            chunk_count=lay.chunk_count,
            chunks_digest=digest(sorted(commits, key=lambda c: c.index)),
            content_digest=h.hexdigest(),
        )

    def _check_manifest_body(self, body: dict[str, Any], v: StoreVerification) -> None:
        store = body.get("store", {})
        checks = (
            (store.get("store_id"), self._header.store_id, "store id"),
            (store.get("format"), FORMAT, "format"),
            (body.get("run_identity"), self._header.identity.value, "run identity"),
            (body.get("content_digest"), v.content_digest, "content digest"),
            (store.get("chunks_digest"), v.chunks_digest, "chunk digest"),
            (body.get("records"), to_canonical(self._header.records), "records"),
        )
        for found, expected, what in checks:
            if found != expected:
                raise EmbeddingStoreCorruptError(f"{MANIFEST}: {what} does not match the store")

    def verify(self) -> StoreVerification:
        """Complete, verified coverage or a typed error. Re-hashes every committed chunk."""
        ins = self.inspect()
        if ins.corrupt and any(i != MANIFEST_ENTRY for i, _ in ins.corrupt):
            raise EmbeddingStoreCorruptError("; ".join(why for _, why in ins.corrupt))
        if ins.missing:
            raise EmbeddingStoreIncompleteError(
                f"{len(ins.missing)} of {self._layout.chunk_count} chunks are not committed "
                f"(first missing: {ins.missing[:5]})"
            )
        if ins.corrupt:
            raise EmbeddingStoreCorruptError("; ".join(why for _, why in ins.corrupt))
        return self._content(ins.committed)

    def finalized_manifest(self) -> dict[str, Any]:
        """The verified manifest body of a finalized store (raises when not finalized)."""
        self.verify()
        path = self._dir / MANIFEST
        if not path.exists():
            raise EmbeddingStoreIncompleteError("the store has not been finalized")
        return _read_document(path)

    # -------------------------------------------------------------- writes

    def _known_committed(self) -> set[int]:
        if self._committed is None:
            self.inspect()
        assert self._committed is not None
        return self._committed

    def write_chunk(self, index: int, rows: object) -> None:
        self._require_writable()
        lay = self._layout
        start, stop = lay.chunk_range(index)
        if index in self._known_committed():
            raise EmbeddingStoreConflictError(
                f"chunk {index} is already committed; committed rows are never rewritten"
            )
        array = np.asarray(rows)
        if array.shape != (stop - start, lay.dim) or array.dtype != np.dtype(lay.dtype):
            raise EmbeddingStoreConflictError(
                f"chunk {index} must be ({stop - start}, {lay.dim}) {lay.dtype}, got "
                f"{array.shape} {array.dtype}"
            )
        if not bool(np.isfinite(array).all()):
            raise EmbeddingOutputError(f"chunk {index} contains non-finite values")
        data = self._data()
        data[start:stop] = array
        data.flush()
        with open(self._dir / DATA, "r+b") as f:
            _full_fsync(f.fileno())
        self._written[index] = self._chunk_sha(index)

    def commit_chunk(self, index: int) -> ChunkCommit:
        self._require_writable()
        if index in self._known_committed():
            raise EmbeddingStoreConflictError(f"chunk {index} is already committed")
        written = self._written.get(index)
        if written is None:
            raise EmbeddingStoreConflictError(f"chunk {index} was not written in this session")
        commit = self._commit(index)
        if commit.rows_sha256 != written:
            raise EmbeddingStoreCorruptError(f"chunk {index} changed between write and commit")
        _atomic_write(self._chunk_path(index), _document(self._commit_body(commit)))
        del self._written[index]
        self._known_committed().add(index)
        self._session_commits.add(index)
        return commit

    def finalize(self, manifest: EmbeddingManifest) -> None:
        self._require_writable()
        v = self.verify()
        h = self._header
        problems = [
            what
            for what, ok in (
                ("store id", manifest.store.store_id == h.store_id),
                ("format", manifest.store.format == FORMAT),
                ("run identity", manifest.run_identity == h.identity.value),
                ("content digest", manifest.content_digest == v.content_digest),
                ("chunk digest", manifest.store.chunks_digest == v.chunks_digest),
                ("records", manifest.records == h.records),
                ("chunk layout", manifest.store.chunk_rows == h.layout.chunk_rows),
            )
            if not ok
        ]
        if problems:
            raise EmbeddingStoreConflictError(
                "the manifest does not describe this store: " + ", ".join(problems)
            )
        path = self._dir / MANIFEST
        if path.exists():
            if digest(_read_document(path)) == manifest.manifest_digest:
                return  # already finalized with this exact manifest
            raise EmbeddingStoreConflictError(
                "the store is already finalized with another manifest"
            )
        _atomic_write(path, _document(manifest))

    # -------------------------------------------------------------- recovery

    def discard_unverified(self, inspection: StoreInspection) -> None:
        """Explicit recovery: drop what failed verification so it is recomputed."""
        self._require_writable()
        entries = {i for i, _ in inspection.corrupt}
        if DATA_ENTRY in entries:
            self._array = None
            _allocate(self._dir, self._header)
            for _, path in self._record_files():
                path.unlink()
        if COORDS_ENTRY in entries:
            _write_coords(self._dir, self._header)
        for index, path in self._record_files():
            if index in entries:
                path.unlink()
        if entries:
            with contextlib.suppress(FileNotFoundError):
                (self._dir / MANIFEST).unlink()
            _fsync_path(self._dir / CHUNKS)
            _fsync_path(self._dir)
        self._committed = None


# --------------------------------------------------------------------------- creation


def _check_npy(path: Path, layout: StoreLayout) -> None:
    """Header and exact size are checked before mapping: a writable `open_memmap` would
    silently zero-extend a truncated file instead of failing."""
    try:
        with open(path, "rb") as f:
            major, _ = np.lib.format.read_magic(f)
            read = (
                np.lib.format.read_array_header_1_0
                if major == 1
                else np.lib.format.read_array_header_2_0
            )
            shape, fortran, dtype = read(f)
            offset = f.tell()
        size = path.stat().st_size
    except (OSError, ValueError) as exc:
        raise EmbeddingStoreCorruptError(f"{DATA} is unreadable: {exc}") from None
    expected = (layout.rows, layout.dim)
    if shape != expected or fortran or dtype != np.dtype(layout.dtype):
        raise EmbeddingStoreCorruptError(
            f"{DATA} declares {shape} {dtype}, expected {expected} {layout.dtype}"
        )
    if size != offset + layout.rows * layout.row_bytes:
        raise EmbeddingStoreCorruptError(
            f"{DATA} holds {size} bytes, expected {offset + layout.rows * layout.row_bytes}"
        )


def _coords(header: EmbeddingStoreHeader) -> npt.NDArray[np.float64]:
    return np.array(
        [(r.level0.x, r.level0.y, r.level0.width, r.level0.height) for r in header.records],
        dtype="<f8",
    ).reshape(len(header.records), 4)


def _write_coords(directory: Path, header: EmbeddingStoreHeader) -> None:
    tmp = directory / f".{COORDS}.tmp"
    with open(tmp, "wb") as f:
        np.save(f, _coords(header), allow_pickle=False)
        f.flush()
        _full_fsync(f.fileno())
    os.replace(tmp, directory / COORDS)


def _allocate(directory: Path, header: EmbeddingStoreHeader) -> None:
    lay = header.layout
    array = np.lib.format.open_memmap(
        directory / DATA, mode="w+", dtype=np.dtype(lay.dtype), shape=(lay.rows, lay.dim)
    )
    array.flush()
    del array
    with open(directory / DATA, "r+b") as f:
        _full_fsync(f.fileno())


class NumpyEmbeddingStoreProvider:
    """Creates and reopens `NumpyEmbeddingStore`s below one root directory."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    @property
    def format(self) -> str:
        return FORMAT

    @property
    def root(self) -> Path:
        return self._root

    def _directory(self, store_id: str) -> Path:
        if not _STORE_ID.fullmatch(store_id):
            raise EmbeddingStoreConflictError(f"invalid store id {store_id!r}")
        return self._root / store_id

    def open(
        self, header: EmbeddingStoreHeader, *, recovery: RecoveryPolicy
    ) -> NumpyEmbeddingStore:
        directory = self._directory(header.store_id)
        if (directory / HEADER).exists():
            body = _read_document(directory / HEADER)
            if body.get("schema_version") != EMBEDDING_STORE_SCHEMA:
                raise EmbeddingStoreIncompatibleError(
                    f"store schema {body.get('schema_version')!r} is not {EMBEDDING_STORE_SCHEMA}"
                )
            expected = to_canonical(header)
            for key, what in (("identity", "run identity"), ("layout", "layout")):
                if body.get(key) != expected[key]:
                    raise EmbeddingStoreIncompatibleError(
                        f"the existing store at this location has another {what}; it is never "
                        "resumed or overwritten by an incompatible run"
                    )
            if digest(body) != header.digest:
                raise EmbeddingStoreIncompatibleError("the existing store maps other patches")
            store = NumpyEmbeddingStore(directory, header, writable=True)
            ins = store.inspect()
            if ins.corrupt:
                if recovery is RecoveryPolicy.RESUME_VERIFIED:
                    detail = "; ".join(why for _, why in ins.corrupt)
                    raise EmbeddingStoreCorruptError(
                        f"{detail} (recovery policy {recovery.value}; choose "
                        f"{RecoveryPolicy.DISCARD_CORRUPT.value} to recompute)"
                    )
                store.discard_unverified(ins)
            return store
        if directory.exists():
            leftovers = [p.name for p in directory.iterdir() if not p.name.startswith(".")]
            if (directory / CHUNKS).is_dir() and any((directory / CHUNKS).iterdir()):
                raise EmbeddingStoreCorruptError("commit records exist without a store header")
            if leftovers and set(leftovers) - {DATA, COORDS, CHUNKS}:
                raise EmbeddingStoreConflictError(
                    f"{directory.name} holds unrelated files: {sorted(leftovers)}"
                )
        directory.mkdir(parents=True, exist_ok=True)
        (directory / CHUNKS).mkdir(exist_ok=True)
        _allocate(directory, header)
        _write_coords(directory, header)
        _atomic_write(directory / HEADER, _document(header))
        _fsync_path(self._root)
        return NumpyEmbeddingStore(directory, header, writable=True)

    def open_existing(self, store_id: str) -> NumpyEmbeddingStore:
        directory = self._directory(store_id)
        if not (directory / HEADER).exists():
            raise EmbeddingStoreIncompleteError(f"no store {store_id} under this root")
        header = header_from_body(_read_document(directory / HEADER))
        if header.store_id != store_id:
            raise EmbeddingStoreCorruptError("the header names another store")
        return NumpyEmbeddingStore(directory, header, writable=False)
