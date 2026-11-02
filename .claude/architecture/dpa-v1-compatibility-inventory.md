# DPA v1 Compatibility Inventory (read-only)

**Status:** Inventory complete for the sources listed below. **Compatibility is not implemented and parity is not claimed.** No adapter exists, and nothing has been run against DPA.
**Date:** 2026-10-15
**Method:** I read the source files below without modifying them. No legacy code was executed or imported.

## 1. Sources inspected

| Side | File | Revision |
|---|---|---|
| DP (server consumed by DPA) | `digital-pathology/scripts/analysis_api/schemas.py` | `main@c85df8a`, read from the working tree |
| DP | `digital-pathology/scripts/analysis_api/job.py` (`STATUSES`, `STAGES`) | same |
| DP | `digital-pathology/scripts/analysis_api/routes.py` (paths, status codes) | same |
| DP | `digital-pathology/scripts/analysis_api/capability_catalog.py` (`tumor_type=task.disease_context`) | same |
| DPA backend (client of DP) | `hal-viewer/backend/src/hal_viewer/infrastructure/ai_analysis/http/digital_pathology_analysis_client.py` | `main@e248202`, read from the working tree |
| DPA backend | `hal-viewer/backend/src/hal_viewer/application/ai_analysis/result_adapter.py`, `.../dto.py` | same |
| DPA frontend contract | `hal-viewer/contracts/schemas/slide/v1/slide.schema.json` | same |

Both repositories had uncommitted changes before this work started: DP had 4 entries in `git status --porcelain` and hal-viewer had 9. All of them are untracked `.claude/` documents or DPA sqlite data files. None of the inspected source files appear in either status list, so the files I read match their committed revisions.

**Not inspected:** the DPA frontend's use of the result (`HttpAIAnalysisApi.ts` was only located), DP's `orchestrator.py` and `artifact_resolution.py`, and runtime traffic. Every field below was checked against source code, not against observed traffic.

## 2. DP `/api/v1` surface consumed by DPA (verified)

| Operation | DP route | DPA client usage |
|---|---|---|
| Capabilities | `GET /api/v1/analysis-capabilities` → `CapabilitiesResponse` | Requires a non-empty `task_id`, `task_type`, `display_name` and `tumor_type`. `task_version` is converted to a string. Each model needs `model_id`, `model_family` and `display_name`. |
| Submit | `POST /api/v1/analyses` → 202 `SubmitAnalysisResponse{analysis_job_id,status}` | Sends schema `"2.0"` with `canonical_slide_id`, `source{uri, format, checksum{algorithm,value}?, size_bytes?}`, `task_id`, `task_version` (an int ≥ 1) and `model_id`. The legacy `"1.0"` schema uses `slide_name` instead. Both use `extra="forbid"`. |
| Job | `GET /api/v1/analyses/{id}` → `AnalysisJobResponse` | Reads `analysis_job_id`, `status` (must be one of PENDING, RUNNING, COMPLETED, FAILED, CANCELLED), and the optional `stage`, `result_id` and `error{code,message}`. |
| Result | `GET /api/v1/analyses/{id}/result` → `AnalysisResultResponse`. Returns 409 `ANALYSIS_NOT_COMPLETED` if the job isn't done. | Details below. |
| Errors | `{"detail": {"code", "message"}}` with 404, 409, 422 or 500 | DPA maps 404, 409 and other 4xx responses to typed errors. |

**Result fields (`schema_version` "1.2", or "1.3" when `canonical_slide_id` is present):**

- `prediction`: `prediction_id`, `model_id`, `task_id`, `task_version` (a string that DPA converts with **`int()`**), `slide_name`, `predicted_value` (str), `scores` (dict[str, float]), `task{task_type, target_name?}`, `model{model_family, model_version?}`, `source{dataset_id?, case_id?, split?}`, `canonical_slide_id?`.
- `spatial_evidence[]`: `evidence_id` (DPA requires it when the version is ≥ 1.2 and fails closed on a malformed version), `patch_index?`, `coordinate_system` (must be `"level_zero"`), `x`, `y`, `width`, `height`, `score?`, `rank?`, `evidence_type` (str), `geometry` (rectangle, polygon or point, all `level_zero`).
- **DPA domain constraints:** DPA rejects any item where `patch_index`, `score` or `rank` is null. It rejects polygon and point geometry with `UnsupportedSpatialEvidenceGeometryError`. Only rectangles reach the viewer.

## 3. Mapping to Pathology AI native v1

Status key: **✓** maps directly. **~** maps with an explicit adapter rule. **✗** is a gap, where the adapter would have to refuse or the data is absent.

| DP/DPA v1 field | Pathology AI v1 | Status / gap |
|---|---|---|
| Job `status` values | `AnalysisJob.status` | ✓ Identical set of five states, with the same terminal states. |
| `analysis_job_id` | `job_id` (`pai-job-…`) | ~ Passed through as an opaque string. Not interchangeable with DP IDs. |
| Job `stage` (DP's 9-value `STAGES` enum) | `stage` is free text (`EXECUTING` or null) | ~ DPA treats it as an optional string. DP's stage vocabulary isn't reproduced. |
| Job `progress`, `created_at`, `started_at`, `completed_at`, `slide_name`, `task_*`, `model_id` | not on the job | ✗ The job schema lacks timestamps and the task/model echo. Needed only if the DPA frontend reads them (not verified). |
| `result_id` | `prediction_id` | ~ |
| Submit `canonical_slide_id` + `source{uri,format,checksum}` | `image_id` (from the Imaging API) | ✗ Pathology AI has no source-locator registration. An adapter would need an image registry that maps a DPA locator to an `ImageAsset`, with a declared modality, and a real reader for the source format. |
| Submit `task_id`/`task_version` (int)/`model_id` | `project_id` (which binds task, models and preprocessing) | ✗ A (task, model) → project mapping is needed. Pathology AI versions are strings like `1.0.0`, but **DPA requires an integer `task_version`**, so a versioning rule is needed. |
| Capabilities `tumor_type` (required, non-empty) | `target_context.disease` (nullable) | ✗ It may not be fabricated. Projects without a disease can't be listed through v1 without a policy decision. |
| Capabilities: validation status | `status` with three states + `requires_research_acknowledgement` | ✗ DP v1 has no field for this, so **research-only status would be invisible in DPA.** |
| `predicted_value`, `scores` | `ImageClassification.label`, `scores` | ~ Only for one image-classification output. Pathology AI also says whether scores are calibrated, and v1 can't carry that. |
| `task.task_type`, `target_name` | `TaskDefinition.task_type`, `target_name` | ~ Enumeration values differ (e.g. `binary_classification`). The DP value set was not verified. |
| `model.model_family`, `model_version` | `ModelDefinition.family`, `version` | ~ Pathology AI binds several models per prediction (encoder and aggregator), but v1 carries one `model_id`. A rule is needed for which slot is reported. |
| `source.dataset_id`, `split`, `case_id` | `fingerprint.dataset`; split comes from dataset membership; there is no case concept | ~ / ✗ `case_id` should be null, which v1 allows. |
| `spatial_evidence[].evidence_id` | `SpatialEvidence.evidence_id` (`pai-ev-…`) | ~ Opaque. **Not DP's `compute_evidence_id`**, so IDs are not comparable across systems. |
| `coordinate_system: level_zero`, x/y/w/h, rectangle geometry | `RectangleGeometry(Level0Rect)`, `frame=wsi_level0_pixel` | ✓ Same canonical frame. Equivalent pixel values were not verified against a real slide. |
| `patch_index` (required by DPA) | none | ✗ The synthetic aggregator emits rank but not patch index. An adapter must not invent one; the patch order would have to be carried through. |
| `score` | `value` | ~ Only when `value` is not null. |
| `evidence_type` (str) | `semantics` (`attention_weight`, …) | ~ DP's `evidence_type` value set was not verified. The **attention-vs-probability distinction is lost** in v1. |
| Polygon/point geometry | `PolygonGeometry`, `PointGeometry` | ~ Valid in DP v1, but DPA rejects them. |
| Segmentation, detection, region, continuous, volumetric outputs | typed outputs | ✗ No v1 representation. The adapter must refuse them; it must never squeeze them into a classification. |
| Voxel/volumetric evidence | `VoxelBoxGeometry` | ✗ No v1 representation (`level_zero` only). |
| Provenance (fingerprint, preprocessing, environment) | `provenance` block | ✗ Not carried by v1. |
| `schema_version` "1.2"/"1.3" | `pathology-ai.analysis/1.0` | ~ Different namespaces. A v1 adapter would emit "1.3" for canonical submissions. |

## 4. Imaging side

DPA doesn't consume an imaging API from DP today. It reads slides from a shared mount and serves DZI tiles itself (hal-viewer `slide/v1`: `slide_id`, `tile_source{type:"dzi", url}`, `metadata{width, height, tile_size, format, levels, source_format?, description?}`).

The Pathology AI Imaging API v1 (`image-descriptor`, `region-request`, `slice-request`, `thumbnail-request`) therefore has **no existing DPA consumer**. Its level-0 `width`/`height` correspond to the DPA slide metadata, but its pyramid levels are native levels rather than DZI levels. A future DPA integration would need its own SPEC.

## 5. Conclusions and preconditions for a future adapter

1. Pathology AI is **not DPA compatible today**. There is no adapter, no test against the DPA client, and no shadow run.
2. A v1 compatibility profile is feasible only for one combination: a single image-classification output on 2D pyramidal images, with rectangle evidence that includes a real patch index.
3. Decisions needed before building one:
   - an integer `task_version` mapping;
   - the source of `tumor_type`;
   - which model slot is reported as `model_id`;
   - how research-only status is surfaced, given that v1 has no field for it;
   - whether evidence IDs need to be comparable with DP's.
4. Required verification: contract tests that replay these responses through DPA's `digital_pathology_analysis_client.py` parsing rules, followed by shadow comparison against DP for the same slides.
