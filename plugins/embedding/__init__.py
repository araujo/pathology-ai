"""Reader-neutral patch embedding (SPEC-005): lazy patch reads, explicit transforms, bounded
batches, a deterministic NumPy reference encoder and the `embed.patches@1` operation.

Pixels are read only through the `OpenedImage` port and embeddings are written only through the
`EmbeddingStore` port; nothing here knows a file format, a reader backend or a storage layout.
"""

PLUGIN_VERSION = "0.1.0"
