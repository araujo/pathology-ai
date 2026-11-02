"""Zeiss CZI reader adapter backed by pylibCZIrw (optional extra `pathology-ai[czi]`).

pylibCZIrw is LGPL-3.0. Its use is approved for research and prototyping; commercial
redistribution requires a separate compliance review.
"""

from plugins.imaging.czi.reader import CZI_FORMAT, CziBackend, CziImage

__all__ = ["CZI_FORMAT", "CziBackend", "CziImage"]
