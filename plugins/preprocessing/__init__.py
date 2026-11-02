"""Reader-neutral real-WSI preprocessing: candidate tissue, patch planning, patch extraction.

Operations read pixels only through the `OpenedImage` port; nothing here knows a file format or a
reader backend (SPEC-004 / ADR-004 D2, D12-D13).
"""

PLUGIN_VERSION = "0.1.0"
