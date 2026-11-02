"""WSI inventory against a cohort manifest, and its scoped summary (SPEC-008 §5, AC-008).

The service owns no cohort or storage knowledge: object sources observe, the manifest supplies
the canonical identities. The summary joins what was observed with each record's eligibility,
partition and label, and states plainly that a scoped inventory leaves the cohort's image
integrity gates open.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from domain.cohort import UNASSIGNED, CohortManifest, EligibilityStatus
from domain.errors import WSIInventoryError
from domain.wsi_inventory import (
    SlideAvailability,
    WSIInventory,
    build_inventory,
    check_identity_source,
)
from ports import WSIObjectSource


class WSIInventoryService:
    def __init__(self, sources: Sequence[WSIObjectSource]) -> None:
        if not sources:
            raise WSIInventoryError("an inventory needs at least one object source")
        self._sources = tuple(sources)

    def inventory(self, manifest: CohortManifest) -> WSIInventory:
        roots = [r for s in self._sources for r in s.roots]
        names = [r.name for r in roots]
        if len(set(names)) != len(names):
            raise WSIInventoryError(f"inventory root names repeat: {sorted(names)}")
        patterns = {p for s in self._sources for p in s.patterns}
        objects = [o for s in self._sources for o in s.observe()]
        return build_inventory(manifest, roots, patterns, objects)


def scoped_summary(inventory: WSIInventory, manifest: CohortManifest) -> dict[str, object]:
    """Observed slides joined with their cohort records; cohort-wide gates are reported as-is."""
    check_identity_source(inventory, manifest)
    found = [
        s for s in inventory.slides if s.availability is not (SlideAvailability.NOT_FOUND_IN_SCOPE)
    ]
    rows = []
    by_status: Counter[str] = Counter()
    eligible_by: Counter[tuple[str, str, str]] = Counter()
    for s in found:
        r = manifest.record(s.slide_id)
        by_status[f"{r.status.value}:{r.reason_code or '-'}"] += 1
        if r.status is EligibilityStatus.ELIGIBLE:
            eligible_by[(r.partition or UNASSIGNED, r.label or "-", s.availability.value)] += 1
        rows.append(
            {
                "slide_id": s.slide_id,
                "cohort": r.cohort,
                "availability": s.availability.value,
                "status": r.status.value,
                "reason_code": r.reason_code,
                "partition": r.partition,
                "label": r.label,
                "size_bytes": s.size_bytes,
                "sha256": s.sha256,
                "copies": len(s.copies),
                "shares_content_with": list(s.shares_content_with),
            }
        )
    eligible = manifest.eligible()
    return {
        "inventory_id": inventory.inventory_id,
        "inventory_digest": inventory.digest,
        "cohort": manifest.ref,
        "roots": [r.name for r in inventory.roots],
        "patterns": list(inventory.patterns),
        "objects": len(inventory.objects),
        "unmatched_objects": len(inventory.unmatched),
        "availability": inventory.counts(),
        "complete": inventory.complete,
        "duplicate_content_groups": [list(d.slide_ids) for d in inventory.duplicates],
        "found_by_status": dict(sorted(by_status.items())),
        "found_eligible_by_partition_label_availability": [
            {"partition": p, "label": lab, "availability": a, "slides": n}
            for (p, lab, a), n in sorted(eligible_by.items())
        ],
        "cohort_gates": {
            "eligible": len(eligible),
            "provisional": sum(r.provisional for r in eligible),
            "note": (
                "unchanged: a scoped inventory never closes cohort-wide image integrity gates"
                if not inventory.complete
                else "inventory is complete; gates close only in a new manifest version"
            ),
        },
        "found": rows,
    }
