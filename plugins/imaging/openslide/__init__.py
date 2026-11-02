"""OpenSlide reader adapter (optional extra `pathology-ai[openslide]`).

OpenSlide and openslide-python are LGPL-2.1. Only vendor formats tested against a real fixture
are claimed (see `.claude/architecture/003-reader-capability-matrix.md`).
"""

from plugins.imaging.openslide.reader import APERIO_SVS_FORMAT, OpenSlideBackend, OpenSlideImage

__all__ = ["APERIO_SVS_FORMAT", "OpenSlideBackend", "OpenSlideImage"]
