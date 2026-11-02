# pathology-ai

Independent foundation for a modality-aware, model-agnostic medical imaging AI platform.

**Research use only.** The bundled models are synthetic. They demonstrate architecture contracts
and are not medically valid models, and they are not evidence of parity with any other system.
Real images can be read (SPEC-003). Pretrained foundation-model inference (UNI, SPEC-006) is
available as an optional runtime with operator-supplied, licensed weights. Successful inference
establishes technical integration only, not clinical validity.

```bash
uv sync
uv run pytest
uv run pathology-ai-demo
```

## Layout

The framework's packages live directly at the repository root. The distribution is named
`pathology-ai`, but there is no `pathology_ai` import package.

```text
domain/            pure contracts (standard library only)
ports/             Protocol interfaces
application/       registries, runner, project and analysis services
infrastructure/    YAML adapters, clocks, environment probe, job executors
plugins/synthetic/ synthetic reader, operations and models (numpy)
plugins/embedding/ lazy patch embedding, explicit transforms, NumPy reference encoder
plugins/imaging/   real-image reader adapters (optional extras: czi, openslide)
plugins/models/    foundation-model encoders: generic torch runtime, UNI (optional extra: embedding)
plugins/mil/       MIL models: mean pooling, attention MIL (torch, optional extra: embedding)
presentation/api/  Imaging and Analysis API v1 serializers and JSON Schemas
bootstrap.py       composition root
cli.py             pathology-ai-demo
pipelines/         pipeline definitions (YAML, not packaged)
projects/          project definitions (YAML, not packaged)
tests/
```

```python
from domain.pipeline import PipelineDefinition
from application.runner import PipelineRunner
```

## Real images (SPEC-003)

Reader backends are optional extras. The base install reads no real files.

```bash
uv pip install 'pathology-ai[czi]'        # Zeiss CZI via pylibczirw (LGPL-3.0)
uv pip install 'pathology-ai[openslide]'  # Aperio SVS via OpenSlide (LGPL-2.1)
pathology-ai-imaging-smoke --file slide.czi --image-id my-slide --modality wsi \
    --region 0,1000,2000,512,512 --region 1,500,1000,256,256
```

`pathology-ai-imaging-smoke` prints metadata and read provenance as JSON. It runs no model. Decoding a
slide is not evidence that any model is valid for it. Formats are claimed only where a real fixture
was tested; see `.claude/architecture/003-reader-capability-matrix.md`. The pylibczirw licence
(LGPL-3.0) is approved for research and prototyping; commercial redistribution requires a separate
compliance review.

Opt-in real-fixture tests:

```bash
PATHOLOGY_AI_WSI_FIXTURE_CZI=/path/SR1482_40X_HE_T198_01.czi \
PATHOLOGY_AI_WSI_FIXTURE_SVS=/path/CMU-1-Small-Region.svs uv run pytest -rs
```

## Patch embedding (SPEC-005)

`wsi-patch-embedding` plans patches without decoding their pixels, then reads them lazily in
bounded batches. Each batch is encoded and committed to a durable NumPy store (`.npy` plus JSON
commit records). An interrupted run resumes from verified chunks when repeated with the same
arguments. The bundled encoder is a deterministic NumPy **reference fixture**, and its
embeddings have no biological or clinical meaning.

```bash
pathology-ai-embed --file slide.czi --image-id my-slide --output-dir embeddings/ \
    --manifest-out my-slide.embedding-manifest.json
```

The store lives at `embeddings/<store_id>/`. `embeddings.npy` holds the rows, `level0.npy` holds
the row-aligned level-zero rectangles, and `header.json` maps each row to its patch.
`manifest.json` is written last: until it exists, the store is not complete. See
`.claude/architecture/005-streaming-embedding-pipeline.md`.

## Foundation-model embeddings (SPEC-006)

The torch runtime is an optional extra. The base install never imports torch or timm, and asking
for a torch encoder without it fails with `MODEL_RUNTIME_UNAVAILABLE`.

```bash
uv pip install 'pathology-ai[embedding]'   # torch (BSD-3-Clause), timm (Apache-2.0)
pathology-ai-embed --file slide.czi --image-id my-slide --output-dir embeddings/ \
    --pipeline wsi-patch-embedding-uni@1.0.0 --checkpoint uni=/path/to/pytorch_model.bin
```

- **Weights.** Weights are never downloaded. Bind a local file with `--checkpoint NAME=PATH`
  or `PATHOLOGY_AI_CHECKPOINTS=NAME=PATH[:NAME=PATH]`. The file's SHA-256 must match the
  adapter's pin. The path is operational and never enters an identity. The digest does.
- **UNI licence.** UNI weights are CC-BY-NC-ND 4.0, gated, for non-commercial academic
  research only. Obtain them yourself. They are not distributed here.
- **Preprocessing profiles.** `uni-official@1` follows the authors' recipe on 224 px patches
  extracted at a stored level. Its physical scale (documented only as "20x") must be bound
  explicitly (`model_spacing_um`, `model_spacing_tolerance`). `uni-legacy-surgen@1` is
  identified but blocked, because its uint8 `INTER_AREA` resampling is not implemented.
- **Devices.** `device` is `cpu` (default), `cuda` or `mps`. A requested device that is
  unavailable fails, with no fallback. Precision is fp32 only.

Opt-in real-UNI tests: `PATHOLOGY_AI_UNI_CHECKPOINT=/path/pytorch_model.bin` together with the
fixture variables. See `.claude/architecture/006-foundation-model-integration.md`.

## MIL slide-level classification (SPEC-007)

`pathology-ai-mil` trains and evaluates slide-level classifiers on finalized embedding stores.
Task, classes and labels are configuration. The framework knows no disease, biomarker or
encoder.

```bash
pathology-ai-mil synthetic --out demo/            # SYNTHETIC cohort: stores, labels.csv, experiment.yaml
pathology-ai-mil run --config demo/experiment.yaml --store-root demo/stores --out demo/run
```

`run` performs these steps:

1. It builds a dataset manifest (`pathology-ai.mil-dataset/1`). Every store is verified, and
   the embedding identities must all be compatible.
2. It makes a group-disjoint split (`pathology-ai.mil-split/1`).
3. It trains with early stopping on the validation partition only, and persists a
   pickle-free checkpoint.
4. It evaluates the validation and test partitions, writing `pathology-ai.slide-prediction/1`
   manifests.

The steps are also available separately: `dataset`, `split`, `train`, `evaluate` and
`predict`. Datasets and splits work on the base install. Training needs the `embedding`
extra.

Constraints:

- Supervised slides need a `group_id`. Splits never share a group, and missing groups fail
  closed.
- Bags from different encoders, weights, input contracts or devices are never mixed.
- Re-running an experiment reuses its verified checkpoint. Any artifact with different
  content is refused, never overwritten.

Metrics on synthetic cohorts demonstrate the pipeline only. No real labelled cohort has been
evaluated, and nothing here is clinical or biomarker performance. See
`.claude/architecture/007-mil-training-and-classification.md`.

Opt-in real-bag tests embed F-CZI-01 and F-SVS-01. They use UNI when
`PATHOLOGY_AI_UNI_CHECKPOINT` is set:

```bash
PATHOLOGY_AI_WSI_FIXTURE_CZI=… PATHOLOGY_AI_WSI_FIXTURE_SVS=… PATHOLOGY_AI_UNI_CHECKPOINT=… \
  uv run pytest tests/integration/test_real_uni_mil.py -rs
```

## Cohort reconciliation (SPEC-008)

`pathology-ai-cohort` reconciles declared label sources into a versioned, write-once cohort
manifest (`pathology-ai.cohort/1`). A YAML file under `projects/cohorts/<cohort>/` declares
the cohort: target, identifier grammar, owner decisions, and every source file with a pinned
SHA-256. The reconciliation code itself contains no cohort-specific logic.

```bash
pathology-ai-cohort verify --manifest projects/cohorts/surgen-msi/manifests/v1/cohort-manifest.json
# A new version is only ever written to a new directory (never over v1):
pathology-ai-cohort reconcile --config projects/cohorts/surgen-msi/cohort.yaml \
    --source-root upstream=~/projects/SurGen-Dataset --source-root legacy=../digital-pathology \
    --out projects/cohorts/surgen-msi/manifests/v2
```

The configuration now declares version "2" because D-7 and D-9 were approved after v1 was built. Manifest v1 stays immutable. It is reproduced by the opt-in test `tests/integration/test_surgen_cohort_manifest.py`, which rebuilds it from the configuration with those two decisions reverted.

The manifest is the only input: no image is read and no model is run.

- If sources disagree, the slide is excluded, and only an approved resolution can admit it.
- A case-level label never spreads to a case's other slides by itself.
- The split is imported (`external@1`). It is never generated or repaired.

`split.csv` uses the SPEC-007 split-table format. A slide stays *provisional* until its image
has been verified, and SPEC-007 hand-over refuses provisional records unless they are
accepted explicitly.

See `.claude/reports/008-label-reconciliation.md`.

### WSI inventory and pilot selection (SPEC-008 Phase C1)

```bash
pathology-ai-cohort inventory --manifest projects/cohorts/surgen-msi/manifests/v1/cohort-manifest.json \
    --root downloads=~/Downloads --pattern '*.czi' --out projects/cohorts/surgen-msi/inventories
pathology-ai-cohort pilot --manifest projects/cohorts/surgen-msi/manifests/v1/cohort-manifest.json \
    --criteria projects/cohorts/surgen-msi/pilot.yaml \
    --inventory projects/cohorts/surgen-msi/inventories/<inventory id>/wsi-inventory.json \
    --out projects/cohorts/surgen-msi/pilots
```

`inventory` reads and hashes every matching file under the named roots. It never writes to the
roots, follows symbolic links or decodes an image. Paths are recorded relative to the root name.
- A saved object-store listing (`--listing NAME=FILE`) is recorded as metadata only. Its ETag
  is never treated as a SHA-256.
- A slide that was not found is `not_found_in_scope`, not missing. A scoped inventory never
  closes the cohort's image integrity gates.

`pilot` makes a seeded, stratified selection from the manifest's approved pilot partitions. It
refuses the test partition and reports unfilled strata as shortfalls instead of substituting
other slides. See `.claude/reports/008-phase-c1-readiness.md`.

## Installed wheel

`pipelines/` and `projects/` are not shipped in the wheel. An installed CLI needs to be told where
they are. It never uses the current working directory.

```bash
pathology-ai-demo --root /path/to/dir-with-pipelines-and-projects
# or
PATHOLOGY_AI_ROOT=/path/to/dir-with-pipelines-and-projects pathology-ai-demo
```

Governance, specifications, ADRs and reports live under `.claude/`.
