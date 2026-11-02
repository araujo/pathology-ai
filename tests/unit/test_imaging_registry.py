"""Reader registration, content-based resolution, integrity and optional-backend isolation.

Uses fake backends: none of this needs an imaging library.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from application.imaging_service import ImageBackendRegistry
from bootstrap import build_image_backends, build_imaging_service
from domain.errors import (
    AmbiguousReaderError,
    DuplicateRegistrationError,
    NotFoundError,
    ReaderBackendUnavailableError,
    UnsupportedImageFormatError,
    UnsupportedReadOperationError,
)
from domain.geometry import Level0Rect
from domain.image import ChannelOrder
from domain.modality import SpatialKind
from domain.provenance import ComponentVersion
from domain.reading import ReadLimits, SourceIntegrity
from infrastructure.imaging.integrity import (
    FULL_STRATEGY,
    PARTIAL_STRATEGY,
    PartialContentFingerprinter,
)
from plugins.imaging.common import coverage_mask
from plugins.imaging.czi import CziBackend
from ports import ImageBackend, ImageSource, OpenedImage, ReaderCapabilities, SourceProbe
from tests.imaging_support import source


class FakeBackend:
    def __init__(self, name: str, magic: bytes, *, available: bool = True) -> None:
        self._name, self._magic, self._available = name, magic, available
        self.opened = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def component(self) -> ComponentVersion:
        return ComponentVersion("reader", self._name, "1")

    @property
    def backend_component(self) -> ComponentVersion:
        return ComponentVersion("backend", self._name, "1")

    @property
    def capabilities(self) -> ReaderCapabilities:
        return ReaderCapabilities(
            formats=frozenset({self._name}),
            spatial_kinds=frozenset({SpatialKind.PYRAMIDAL_2D}),
            channel_order=ChannelOrder.RGB,
            provides_physical_spacing=False,
            detects_content=True,
        )

    def available(self) -> bool:
        return self._available

    def probe(self, path: Path) -> SourceProbe:
        ok = path.read_bytes().startswith(self._magic)
        return SourceProbe(ok, self._name if ok else None, "magic")

    def open(
        self,
        source: ImageSource,
        *,
        scene: int | None,
        limits: ReadLimits,
        integrity: SourceIntegrity | None,
    ) -> OpenedImage:
        raise AssertionError("not opened in these tests")


def _file(tmp_path: Path, name: str, content: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(content)
    return path


def _registry(*backends: FakeBackend) -> ImageBackendRegistry:
    registry = ImageBackendRegistry()
    for b in backends:
        registry.register(b)
    return registry


def test_fake_backend_satisfies_the_port() -> None:
    backend: ImageBackend = FakeBackend("aaa", b"AAA")
    assert isinstance(backend, ImageBackend)


def test_resolution_uses_content_not_extension(tmp_path: Path) -> None:
    registry = _registry(FakeBackend("aaa", b"AAA"), FakeBackend("bbb", b"BBB"))
    misnamed = _file(tmp_path, "slide.aaa", b"BBB...")
    assert registry.resolve(source(misnamed)).name == "bbb"
    unknown = _file(tmp_path, "slide.bbb", b"plain text")
    with pytest.raises(UnsupportedImageFormatError, match="tried: aaa, bbb"):
        registry.resolve(source(unknown))


def test_ambiguity_is_an_error_not_a_choice(tmp_path: Path) -> None:
    registry = _registry(FakeBackend("one", b"X"), FakeBackend("two", b"X"))
    path = _file(tmp_path, "f", b"X")
    with pytest.raises(AmbiguousReaderError, match="one, two"):
        registry.resolve(source(path))
    assert registry.resolve(source(path), backend="two").name == "two"


def test_recognised_but_unavailable_backend(tmp_path: Path) -> None:
    registry = _registry(FakeBackend("aaa", b"AAA", available=False))
    with pytest.raises(ReaderBackendUnavailableError):
        registry.resolve(source(_file(tmp_path, "f", b"AAA")))


def test_missing_file_and_duplicates(tmp_path: Path) -> None:
    registry = _registry(FakeBackend("aaa", b"AAA"))
    with pytest.raises(NotFoundError):
        registry.resolve(source(tmp_path / "missing"))
    with pytest.raises(DuplicateRegistrationError):
        registry.register(FakeBackend("aaa", b"ZZZ"))
    with pytest.raises(NotFoundError):
        registry.resolve(source(_file(tmp_path, "f", b"AAA")), backend="nope")


def test_ingest_requires_regions(tmp_path: Path) -> None:
    service = build_imaging_service(backends=_registry(FakeBackend("aaa", b"AAA")))
    with pytest.raises(UnsupportedReadOperationError):
        service.ingest(source(_file(tmp_path, "f", b"AAA")), [])


def test_default_backends_are_registered_by_name() -> None:
    names = [b.name for b in build_image_backends().backends()]
    assert names == ["czi-pylibczirw", "openslide"]
    for b in build_image_backends().backends():
        assert isinstance(b, ImageBackend)


def test_missing_czi_extra_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without pylibCZIrw a CZI is still recognised, and reported as needing the extra."""
    monkeypatch.setitem(sys.modules, "pylibCZIrw", None)
    monkeypatch.setitem(sys.modules, "pylibCZIrw.czi", None)
    monkeypatch.setattr("importlib.util.find_spec", lambda name, *a: None)
    backend = CziBackend()
    path = _file(tmp_path, "x.bin", b"ZISRAWFILE" + b"\0" * 64)
    assert not backend.available()
    assert backend.probe(path).recognised
    with pytest.raises(ReaderBackendUnavailableError, match=r"pathology-ai\[czi\]"):
        _registry_with(backend).resolve(source(path))
    with pytest.raises(ReaderBackendUnavailableError, match=r"pathology-ai\[czi\]"):
        backend.open(source(path))


def _registry_with(backend: CziBackend) -> ImageBackendRegistry:
    registry = ImageBackendRegistry()
    registry.register(backend)
    return registry


def test_partial_fingerprint_guarantees(tmp_path: Path) -> None:
    data = bytearray(b"a" * 10_000)
    path = _file(tmp_path, "f", bytes(data))
    partial = PartialContentFingerprinter(chunk_bytes=1000)
    full = PartialContentFingerprinter(full=True)
    p0, f0 = partial.fingerprint(path), full.fingerprint(path)
    assert p0.strategy == PARTIAL_STRATEGY and f0.strategy == FULL_STRATEGY
    assert "does not detect" in p0.guarantees
    data[5000] = ord("b")  # middle: outside both chunks
    path.write_bytes(bytes(data))
    assert partial.fingerprint(path) == p0
    assert full.fingerprint(path) != f0
    data[-1] = ord("c")  # tail chunk
    path.write_bytes(bytes(data))
    assert partial.fingerprint(path) != p0
    path.write_bytes(bytes(data[:-1]))  # truncation
    assert partial.fingerprint(path).size_bytes == 9_999


def test_coverage_mask_uses_pixel_centres() -> None:
    tiles = np.array([[0, 0, 100, 100], [150, 0, 250, 100]], dtype=np.int64)
    mask = coverage_mask(tiles, Level0Rect(50, 0, 200, 100), 1.0, 200, 100)
    assert mask[:, :50].all() and not mask[:, 50:100].any() and mask[:, 100:].all()
    # Downsample 4: pixel j covers level-0 [50 + 4j, 54 + 4j); centre 52 + 4j.
    mask4 = coverage_mask(tiles, Level0Rect(50, 0, 200, 100), 4.0, 50, 25)
    assert mask4[0, :12].all() and not mask4[0, 12:25].any() and mask4[0, 25:].all()
    assert coverage_mask(tiles[:0], Level0Rect(0, 0, 10, 10), 1.0, 10, 10).sum() == 0
