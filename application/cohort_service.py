"""Cohort reconciliation and its hand-over to SPEC-007 (SPEC-008 §6-§8).

The service owns no cohort knowledge: the definition and the source reader carry it. It turns
reconciled cohort records into the SPEC-007 contracts:

* `slide_labels`: `SlideLabel` records of the eligible slides, for `MILDatasetService.build`.
* `split_assignment` / `import_external_split`: the imported partition membership, validated by
  SPEC-007 `import_split` (`external@1`, `GROUP_LEAKAGE`) once a dataset with bags exists.

Provisional records (eligible, with image integrity gates still open) are refused unless the
caller explicitly accepts them, so supervised use waits for a manifest version whose images
have been verified.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from domain.cohort import (
    CohortDefinition,
    CohortManifest,
    CohortRecord,
    Resolution,
    WSIEvidence,
    build_manifest,
)
from domain.errors import CohortError, SplitError
from domain.mil import (
    LabelStatus,
    MILDatasetManifest,
    MILTask,
    Partition,
    SlideLabel,
    SplitManifest,
    import_split,
)
from ports import CohortSourceReader


class CohortService:
    def __init__(self, reader: CohortSourceReader) -> None:
        self._reader = reader

    def reconcile(
        self,
        definition: CohortDefinition,
        wsi: Mapping[str, WSIEvidence] | None = None,
        resolutions: Sequence[Resolution] = (),
    ) -> CohortManifest:
        return build_manifest(definition, self._reader.read(), wsi, resolutions)


def _eligible(manifest: CohortManifest, allow_provisional: bool) -> tuple[CohortRecord, ...]:
    records = manifest.eligible()
    provisional = [r.slide_id for r in records if r.provisional]
    if provisional and not allow_provisional:
        raise CohortError(
            f"{len(provisional)} eligible slide(s) of {manifest.ref} are provisional (open gates "
            f"{sorted({g for r in records for g in r.outstanding_gates})}); verify the images and "
            "build a new manifest version, or accept provisional records explicitly"
        )
    return records


def split_assignment(
    manifest: CohortManifest, *, allow_provisional: bool = False
) -> dict[str, Partition]:
    out: dict[str, Partition] = {}
    for r in _eligible(manifest, allow_provisional):
        assert r.partition is not None
        out[r.slide_id] = Partition(r.partition)
    return out


def slide_labels(
    manifest: CohortManifest, task: MILTask, *, allow_provisional: bool = False
) -> tuple[SlideLabel, ...]:
    """SPEC-007 label records of the eligible slides. The source is the manifest itself."""
    if task.classes != manifest.target.classes or task.positive_class != (
        manifest.target.positive_class
    ):
        raise CohortError(
            f"task {task.task_id}@{task.version} classes {list(task.classes)} differ from the "
            f"cohort target {list(manifest.target.classes)}"
        )
    return tuple(
        SlideLabel(
            slide_id=r.slide_id,
            task=task.ref,
            value=r.label,
            source=f"cohort:{manifest.ref}",
            source_version=manifest.digest,
            status=LabelStatus.ELIGIBLE,
            group_id=r.group_id,
            cohort=r.cohort,
        )
        for r in _eligible(manifest, allow_provisional)
    )


def import_external_split(
    manifest: CohortManifest,
    dataset: MILDatasetManifest,
    *,
    allow_provisional: bool = False,
) -> SplitManifest:
    """The cohort's imported split as a SPEC-007 `SplitManifest` over `dataset`.

    The dataset's supervised slides must be exactly the cohort's eligible slides, with the same
    labels and groups. A slide lost on the way (e.g. a failed extraction) needs a new cohort
    manifest version; the split is never silently narrowed.
    """
    assignment = split_assignment(manifest, allow_provisional=allow_provisional)
    supervised = {r.slide_id: r for r in dataset.supervised_records()}
    if set(supervised) != set(assignment):
        missing = sorted(set(assignment) - set(supervised))
        extra = sorted(set(supervised) - set(assignment))
        raise SplitError(
            f"dataset {dataset.ref} does not cover cohort {manifest.ref}: "
            f"{len(missing)} eligible slide(s) missing ({', '.join(missing[:5])}), "
            f"{len(extra)} supervised slide(s) not eligible ({', '.join(extra[:5])})"
        )
    for slide_id, d in supervised.items():
        c = manifest.record(slide_id)
        if (d.label, d.group_id) != (c.label, c.group_id):
            raise SplitError(f"slide {slide_id}: dataset label or group differs from the cohort")
    return import_split(dataset, assignment)
