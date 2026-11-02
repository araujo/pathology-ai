"""Real-image reader adapters. Each backend lives in its own subpackage and optional extra.

Backend libraries are imported lazily inside their adapter, so this package (and the base
install) imports cleanly without any optional extra.
"""
