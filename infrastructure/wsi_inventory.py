"""Image object sources for the WSI inventory, plus inventory and pilot documents (SPEC-008 §5, §9).

`LocalWSIObjectSource` walks named local roots and hashes every regular file whose name matches
one of the declared patterns. It is strictly read-only:

* files are opened read-only and only read; nothing is created, renamed, touched or deleted
  under a root;
* symbolic links are never followed (they are counted and skipped), so an inventory cannot
  escape its roots or count one file twice through a link;
* the size and modification time are compared before and after hashing, and an object that
  changed while it was read is recorded as unreadable rather than with a digest it never had.

Paths are recorded relative to the root name; the directory a name is bound to never enters an
inventory, in the same way cohort source roots are bound at run time.

`ObjectListingSource` turns a saved object-store listing (`list-objects-v2` JSON: `Key`, `Size`,
`ETag`) into metadata-only observations. The entity tag is kept as an opaque remote checksum; it
is never treated as a SHA-256 of the content.

Inventories and pilot selections are write-once JSON documents (`{"body", "digest"}`).
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from domain.errors import PilotSelectionError, WSIInventoryError
from domain.pilot import PilotCriteria, PilotSelection, Preference
from domain.wsi_inventory import (
    Access,
    InventoryRoot,
    LocationKind,
    ObservedObject,
    Verification,
    WSIInventory,
)
from infrastructure.config import ConfigurationError
from infrastructure.mil_artifacts import read_document, write_once

INVENTORY_FILE = "wsi-inventory.json"
PILOT_FILE = "pilot-selection.json"
PILOT_CONFIG_SCHEMA = "pathology-ai.pilot-config/1"
LOCAL_ROOT = "local directory bound at run time"
_BLOCK = 8 << 20


def _matches(name: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, p) for p in patterns)


def sha256_file(path: Path, block_bytes: int = _BLOCK) -> tuple[str, int]:
    """SHA-256 and byte count of a file, read sequentially in read-only mode."""
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as f:
        while block := f.read(block_bytes):
            h.update(block)
            n += len(block)
    return h.hexdigest(), n


class LocalWSIObjectSource:
    """Hashes matching regular files under named local directories. Never writes."""

    def __init__(
        self,
        roots: Mapping[str, Path],
        patterns: Sequence[str],
        *,
        block_bytes: int = _BLOCK,
        progress: Callable[[str, int], None] | None = None,
    ) -> None:
        if not roots:
            raise ConfigurationError("bind at least one inventory root (NAME=PATH)")
        if not patterns:
            raise ConfigurationError("declare at least one object name pattern")
        self._roots = {name: Path(path) for name, path in roots.items()}
        for name, path in self._roots.items():
            InventoryRoot(name, LocationKind.LOCAL, LOCAL_ROOT)  # validates the name
            if not path.is_dir():
                raise ConfigurationError(f"inventory root {name!r} is not a directory")
        self._patterns = tuple(sorted(set(patterns)))
        self._block = block_bytes
        self._progress = progress
        self.skipped_links = 0

    @property
    def roots(self) -> tuple[InventoryRoot, ...]:
        return tuple(
            InventoryRoot(name, LocationKind.LOCAL, LOCAL_ROOT) for name in sorted(self._roots)
        )

    @property
    def patterns(self) -> tuple[str, ...]:
        return self._patterns

    def observe(self) -> tuple[ObservedObject, ...]:
        self.skipped_links = 0
        out: list[ObservedObject] = []
        for name in sorted(self._roots):
            out.extend(self._walk(name, self._roots[name]))
        return tuple(sorted(out, key=lambda o: o.uri))

    def _walk(self, root_name: str, root: Path) -> Iterable[ObservedObject]:
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            base = Path(dirpath)
            dirnames[:] = sorted(d for d in dirnames if not (base / d).is_symlink())
            for fname in sorted(filenames):
                if not _matches(fname, self._patterns):
                    continue
                path = base / fname
                if path.is_symlink():
                    self.skipped_links += 1
                    continue
                if not path.is_file():
                    continue
                rel = path.relative_to(root).as_posix()
                yield self._observe(root_name, rel, path)

    def _observe(self, root_name: str, rel: str, path: Path) -> ObservedObject:
        def failed(note: str, size: int | None = None) -> ObservedObject:
            return ObservedObject(
                root_name,
                rel,
                LocationKind.LOCAL,
                Access.UNREADABLE,
                Verification.FAILED,
                size,
                None,
                note=note,
            )

        try:
            before = path.stat()
            if self._progress is not None:
                self._progress(f"{root_name}:{rel}", before.st_size)
            sha, n = sha256_file(path, self._block)
            after = path.stat()
        except OSError as exc:
            return failed(f"read failed: {exc.strerror or type(exc).__name__}")
        stamp = (before.st_size, before.st_mtime_ns, before.st_ino)
        if stamp != (after.st_size, after.st_mtime_ns, after.st_ino):
            return failed("the file changed while it was read", after.st_size)
        if n != before.st_size:
            return failed(f"read {n} bytes of {before.st_size}", before.st_size)
        if n == 0:
            return failed("empty file", 0)
        return ObservedObject(
            root_name,
            rel,
            LocationKind.LOCAL,
            Access.READABLE,
            Verification.CONTENT_VERIFIED,
            n,
            sha,
        )


class ObjectListingSource:
    """Metadata-only observations from a saved object-store listing (`list-objects-v2` JSON)."""

    def __init__(self, name: str, description: str, listing: Path, patterns: Sequence[str]) -> None:
        self._root = InventoryRoot(name, LocationKind.REMOTE, description)
        self._listing = Path(listing)
        self._patterns = tuple(sorted(set(patterns)))

    @property
    def roots(self) -> tuple[InventoryRoot, ...]:
        return (self._root,)

    @property
    def patterns(self) -> tuple[str, ...]:
        return self._patterns

    def observe(self) -> tuple[ObservedObject, ...]:
        try:
            doc = json.loads(self._listing.read_text(encoding="utf-8"))
            contents = doc.get("Contents", [])
        except (OSError, ValueError, AttributeError) as exc:
            raise WSIInventoryError(f"listing {self._listing.name} is unreadable: {exc}") from None
        out = []
        for item in contents:
            key = str(item["Key"])
            if not _matches(key.rsplit("/", 1)[-1], self._patterns):
                continue
            etag = item.get("ETag")
            out.append(
                ObservedObject(
                    self._root.name,
                    key,
                    LocationKind.REMOTE,
                    Access.LISTED,
                    Verification.METADATA_ONLY,
                    int(item["Size"]) if "Size" in item else None,
                    None,
                    remote_checksum=None if etag is None else str(etag).strip('"'),
                    remote_checksum_kind=None if etag is None else "s3-etag",
                )
            )
        return tuple(sorted(out, key=lambda o: o.uri))


# --------------------------------------------------------------------------- documents


def write_inventory(directory: Path, inventory: WSIInventory) -> Path:
    """Write once under `<directory>/<inventory id>/`; identical content is accepted."""
    path = Path(directory) / inventory.inventory_id / INVENTORY_FILE
    write_once(path, inventory)
    return path


def read_inventory(path: Path) -> WSIInventory:
    return read_document(WSIInventory, path, WSIInventoryError)


def write_pilot(directory: Path, selection: PilotSelection) -> Path:
    path = Path(directory) / selection.selection_id / PILOT_FILE
    write_once(path, selection)
    return path


def read_pilot(path: Path) -> PilotSelection:
    return read_document(PilotSelection, path, PilotSelectionError)


def load_pilot_criteria(path: Path) -> PilotCriteria:
    """`pathology-ai.pilot-config/1`: plan id, seed, partitions, strata, quota, preference."""
    path = Path(path)
    try:
        doc: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"{path.name} cannot be read: {exc}") from None
    keys = {
        "schema",
        "plan_id",
        "seed",
        "partitions",
        "strata",
        "per_stratum",
        "prefer",
        "decision",
    }
    if not isinstance(doc, Mapping) or set(doc) != keys:
        found = sorted(doc) if isinstance(doc, Mapping) else []
        raise ConfigurationError(
            f"{path.name} must have exactly the keys {sorted(keys)}, has {found}"
        )
    if doc["schema"] != PILOT_CONFIG_SCHEMA:
        raise ConfigurationError(f"{path.name}: schema must be {PILOT_CONFIG_SCHEMA}")
    for k in ("partitions", "strata"):
        if not isinstance(doc[k], list) or not all(isinstance(v, str) for v in doc[k]):
            raise ConfigurationError(f"{path.name}: {k} must be a list of strings")
    try:
        prefer = Preference(str(doc["prefer"]))
    except ValueError:
        raise ConfigurationError(
            f"{path.name}: prefer must be one of {[p.value for p in Preference]}"
        ) from None
    return PilotCriteria(
        plan_id=str(doc["plan_id"]),
        seed=doc["seed"],
        partitions=tuple(doc["partitions"]),
        strata=tuple(doc["strata"]),
        per_stratum=doc["per_stratum"],
        prefer=prefer,
        decision=str(doc["decision"]),
    )
