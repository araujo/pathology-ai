"""Source integrity for large image files without hashing every byte on every read.

`PartialContentFingerprinter` hashes the file size plus its first and last `chunk_bytes`. It
detects truncation, extension, replacement by a different file and changes to headers and trailing
directories (where many WSI containers keep their index). It does **not** detect modification of
bytes strictly between the two chunks; set `full=True` for a complete SHA-256 when that guarantee
is needed and reading the whole file is acceptable.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from domain.reading import SourceIntegrity

PARTIAL_STRATEGY = "size+sha256-head-tail/v1"
FULL_STRATEGY = "sha256/v1"
_PARTIAL_GUARANTEES = (
    "detects size changes and changes within the first and last {n} bytes; does not detect "
    "changes strictly between them"
)
_FULL_GUARANTEES = "detects any change to the file content"


class PartialContentFingerprinter:
    def __init__(self, chunk_bytes: int = 1 << 20, *, full: bool = False) -> None:
        if chunk_bytes < 1:
            raise ValueError("chunk_bytes must be positive")
        self._chunk = chunk_bytes
        self._full = full

    def fingerprint(self, path: Path) -> SourceIntegrity:
        size = path.stat().st_size
        h = hashlib.sha256()
        with path.open("rb") as handle:
            if self._full:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    h.update(block)
                return SourceIntegrity(FULL_STRATEGY, h.hexdigest(), size, _FULL_GUARANTEES)
            h.update(str(size).encode("ascii"))
            h.update(handle.read(self._chunk))
            if size > self._chunk:
                handle.seek(max(self._chunk, size - self._chunk))
                h.update(handle.read(self._chunk))
        return SourceIntegrity(
            PARTIAL_STRATEGY,
            h.hexdigest(),
            size,
            _PARTIAL_GUARANTEES.format(n=self._chunk),
        )
