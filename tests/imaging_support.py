"""Helpers for reader tests: tiny CZI files written with pylibCZIrw, and fixture discovery.

Generated CZI files contain no patient data. They are synthetic-format fixtures: they exercise
the real decoder but do not count as real whole-slide images (SPEC-003 AC-002).
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from domain.ids import ImageId
from domain.modality import WHOLE_SLIDE_MICROSCOPY
from ports import ImageSource

CZI_FIXTURE_ENV = "PATHOLOGY_AI_WSI_FIXTURE_CZI"
SVS_FIXTURE_ENV = "PATHOLOGY_AI_WSI_FIXTURE_SVS"

# F-CZI-01: SurGen (EMBL-EBI BioImage Archive S-BIAD1285, CC BY 4.0),
# SR1482_WSIs/SR1482_40X_HE_T198_01.czi. Archive lists the same byte size.
CZI_FIXTURE_SIZE = 77_963_616
CZI_FIXTURE_SHA256 = "8634d0551cb7c137ef9dc87f50a69f71a261c3e7817fc6a56aa8e329a13e8be4"
# F-SVS-01: OpenSlide test data, Aperio/CMU-1-Small-Region.svs (CC0-1.0); sha256 from index.json.
SVS_FIXTURE_SIZE = 1_938_955
SVS_FIXTURE_SHA256 = "ed92d5a9f2e86df67640d6f92ce3e231419ce127131697fbbce42ad5e002c8a7"

PLANE = {"C": 0, "Z": 0, "T": 0}


def czi_module() -> Any:
    return pytest.importorskip("pylibCZIrw.czi", reason="optional extra pathology-ai[czi]")


def write_czi(
    path: Path,
    tiles: Sequence[tuple[int, int, Any]],
    *,
    spacing_um: float | None = 0.5,
    scene: int = 0,
    extra: Sequence[tuple[int, int, Any, int]] = (),
    z_planes: int = 1,
) -> Path:
    """Write `(x, y, array)` tiles at native coordinates (may be negative).

    `extra` adds `(x, y, array, scene)` tiles in other scenes. Arrays are written verbatim: a
    `(h, w, 3) uint8` array becomes Bgr24, so its last axis is B, G, R.
    """
    czi = czi_module()
    with czi.create_czi(str(path), exist_ok=True) as writer:
        for z in range(z_planes):
            for x, y, data in tiles:
                writer.write(
                    data=data, plane={"C": 0, "T": 0, "Z": z}, location=(x, y), scene=scene
                )
        for x, y, data, s in extra:
            writer.write(data=data, plane=dict(PLANE), location=(x, y), scene=s)
        if spacing_um is not None:
            writer.write_metadata(scale_x=spacing_um * 1e-6, scale_y=spacing_um * 1e-6)
    return path


def solid(h: int, w: int, bgr: tuple[int, int, int]) -> Any:
    return np.broadcast_to(np.array(bgr, dtype=np.uint8), (h, w, 3)).copy()


def source(path: Path, image_id: str = "test-image") -> ImageSource:
    return ImageSource(ImageId(image_id), path, WHOLE_SLIDE_MICROSCOPY)


def fixture_path(env: str, size: int, sha256: str) -> Path:
    """Opt-in real fixture: skip with a reason when unset, fail when it is the wrong file."""
    value = os.environ.get(env)
    if not value:
        pytest.skip(f"real fixture not configured: set {env} (see 003-imaging-fixture-inventory)")
    path = Path(value).expanduser()
    assert path.is_file(), f"{env}={value} is not a file"
    assert path.stat().st_size == size, f"{env} has unexpected size {path.stat().st_size}"
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            h.update(block)
    assert h.hexdigest() == sha256, f"{env} has unexpected sha256 {h.hexdigest()}"
    return path


def open_fd_count() -> int:
    """Open file descriptors of this process (macOS/Linux)."""
    for d in ("/dev/fd", "/proc/self/fd"):
        if os.path.isdir(d):
            return len(os.listdir(d))
    pytest.skip("cannot count open file descriptors on this platform")
