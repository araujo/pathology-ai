# PROMPT-006 — Foundation Model Integration & Real Embedding Extraction

**Save as:** `.claude/prompts/006-foundation-model-integration-and-real-embeddings.md`

## Role

You are the implementation engineer responsible for SPEC-006 in the `pathology-ai` repository.

Implement a reusable foundation-model inference architecture, starting with UNI, while preserving the existing model-independent streaming pipeline.

## Authoritative Documents

Read these files before making changes:

1. `.claude/specs/006-foundation-model-integration-and-real-embeddings.md`
2. `.claude/adr/006-foundation-model-adapter-architecture.md`
3. `.claude/specs/005-streaming-patch-to-embedding-pipeline.md`
4. `.claude/adr/005-streaming-embedding-architecture.md`
5. `.claude/reports/005-streaming-patch-to-embedding-pipeline.md`
6. Relevant SPEC/ADR-001 through SPEC/ADR-004.

If an authoritative document is missing, stop and request it.

Do not invent missing requirements.

## Phase A — Discovery and Architecture Verification

Before modifying implementation code:

1. Inspect the existing `EmbeddingEncoder`, `ModelInputSpec`, `TransformChain`, `BatchPolicy`, and `EmbeddingStore` contracts.
2. Inspect `embed.patches@1`, the encoder factory, and `PipelineRunner` integration.
3. Inspect the legacy UNI implementation in `../digital-pathology` strictly read-only.
4. Verify authoritative UNI model documentation, including:
   - Architecture.
   - Expected input dimensions.
   - Preprocessing.
   - Physical-scale assumptions.
   - Output embedding dimension.
   - Checkpoint format.
   - License and access requirements.
5. Compare official UNI preprocessing against the legacy SurGen implementation.
6. Identify any transformations not yet supported by SPEC-005.
7. Verify optional PyTorch/timm compatibility with Python 3.12.
8. Inspect whether authorized UNI checkpoints are already available locally.

Write:

`.claude/reports/006-foundation-model-discovery.md`

Do not assume the legacy preprocessing is scientifically equivalent to the official model recipe.

If official requirements cannot be verified, mark the official profile blocked rather than guessing.

Proceed with generic runtime work and fixture-backed tests where possible.

## Phase B — Generic Model Runtime

Implement or extend:

- Model registry.
- Model adapter factory.
- Local checkpoint resolver.
- SHA-256 checkpoint verification.
- Optional PyTorch/timm dependencies.
- Device selection.
- Precision configuration.
- Model/runtime provenance.
- Typed loading and inference errors.

Reuse existing contracts.

Do not introduce UNI-specific behavior into generic domain or application layers.

Do not introduce a second pipeline runner.

## Phase C — UNI Adapter

Implement the UNI encoder as a plugin.

Requirements:

- Verified model architecture.
- Strict checkpoint compatibility validation.
- Correct embedding output extraction.
- Explicit official UNI preprocessing profile.
- Separate legacy SurGen compatibility profile.
- Distinct scientific identities for the profiles.
- Evaluation mode and inference without gradients.
- Bounded batch inference.
- Validated output shape, dtype, and finite values.
- Controlled model resource lifecycle.

Do not silently substitute official preprocessing with legacy preprocessing.

Do not download checkpoints during execution.

If exact legacy preprocessing cannot be reproduced with existing declared transformations, fail clearly and document the limitation.

## Phase D — Pipeline Integration

Integrate the real encoder through:

`embed.patches@1`

Preserve:

- SPEC-004 patch identity and canonical coordinates.
- Explicit stored-level selection.
- Lazy pixel reads.
- Bounded host memory.
- Model-specific preprocessing.
- Incremental embedding writes.
- Chunk integrity.
- Safe resume.
- Stable provenance.

Run the existing CZI and SVS fixtures through a lightweight local torch encoder.

If an authorized real UNI checkpoint is available, run UNI end to end on both fixtures as well.

If weights are unavailable, do not claim real UNI inference was validated.

## Phase E — Validation

Execute:

- Ruff checks and formatting.
- Mypy strict.
- Full pytest suite.
- SPEC-001 through SPEC-005 regressions.
- CPU torch inference tests.
- CUDA tests when available.
- Missing and corrupt checkpoint tests.
- Architecture and output-dimension mismatch tests.
- Model preprocessing validation tests.
- CZI and SVS integration tests.
- Interrupted inference and resume tests.
- Reproducibility tests across fresh processes.
- Installed-wheel smoke tests.
- Optional-dependency isolation tests.

For every SPEC-006 acceptance criterion, report:

- PASS, FAIL, BLOCKED, or NOT VERIFIED.
- Test name and evidence.
- Whether execution used synthetic fixture weights or real UNI weights.

Do not count a skipped real UNI test as proof that real UNI works.

## Mandatory Constraints

- No universal µm/px.
- No implicit resampling.
- No implicit channel reordering.
- No UNI-specific branches in generic layers.
- No second orchestration engine.
- No model downloads during inference.
- No copying legacy code.
- No modifications to sibling repositories.
- No Git commits.
- No unsupported clinical claims.
- No marking unverified requirements as PASS.

## Deliverables

Create or update:

- Generic model runtime and registry.
- UNI plugin.
- Explicit preprocessing profiles.
- Local checkpoint verification.
- CPU/CUDA validation tests.
- Real WSI integration tests.
- `.claude/architecture/006-foundation-model-integration.md`
- `.claude/reports/006-foundation-model-discovery.md`
- `.claude/reports/006-foundation-model-integration.md`

The final report must include:

1. Implementation summary.
2. Verified model documentation and licensing evidence.
3. Architectural decisions and deviations.
4. Model registry and adapter structure.
5. Official and legacy preprocessing differences.
6. Checkpoint integrity evidence.
7. CZI/SVS execution results.
8. Real UNI execution status.
9. CPU/CUDA memory and timing measurements.
10. Reproducibility results.
11. Packaging and regression results.
12. AC-001 through AC-020 evidence matrix.
13. Remaining limitations and blockers.

Keep SPEC-006 and ADR-006 as Proposed until owner approval.

**Begin with Phase A. Continue through all non-blocked implementation phases. Stop for approval if a substantive architectural conflict, unapproved dependency, or licensing issue requires an owner decision.**