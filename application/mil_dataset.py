"""Labelled MIL datasets and group-aware splits over finalized embedding bags (SPEC-007 §6).

A dataset is built from label records and the store each slide's bag lives in. Every store is
verified (finalized, every chunk re-hashed) before its identities enter the manifest. A
dataset that has been written can be re-verified later against the stores, so a replaced or
re-embedded store is detected rather than silently mixed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from domain.errors import EmbeddingCompatibilityError, MILDatasetError
from domain.ids import DatasetId, Version
from domain.mil import (
    CompatibilityPolicy,
    EmbeddingBagInfo,
    EmbeddingCompatibility,
    MILDatasetManifest,
    MILTask,
    Partition,
    SlideLabel,
    SplitFractions,
    SplitManifest,
    build_dataset_manifest,
    compatibility_policy,
    generate_split,
    import_split,
)
from ports import EmbeddingBagSource


class MILDatasetService:
    def __init__(self, bags: EmbeddingBagSource) -> None:
        self._bags = bags

    def build(
        self,
        *,
        dataset_id: DatasetId,
        version: Version,
        task: MILTask,
        policy: CompatibilityPolicy,
        labels: Sequence[SlideLabel],
        stores: Mapping[str, str],
        accepted_key: str | None = None,
        limitations: tuple[str, ...] = (),
    ) -> MILDatasetManifest:
        """`stores` maps each slide id to the id of the finalized store holding its bag."""
        unknown = sorted(set(stores) - {lbl.slide_id for lbl in labels})
        if unknown:
            raise MILDatasetError(f"stores given for slides without labels: {', '.join(unknown)}")
        bags: dict[str, EmbeddingBagInfo] = {}
        for lbl in labels:
            store_id = stores.get(lbl.slide_id)
            if store_id is None:
                raise MILDatasetError(f"slide {lbl.slide_id} has no embedding store")
            bags[lbl.slide_id] = self._bags.describe(store_id)
        return build_dataset_manifest(
            dataset_id=dataset_id,
            version=version,
            task=task,
            policy=policy,
            labels=labels,
            bags=bags,
            accepted_key=accepted_key,
            limitations=limitations,
        )

    def verify(self, dataset: MILDatasetManifest, slide_ids: Sequence[str] | None = None) -> None:
        """Re-verify the stores of `slide_ids` (all slides by default) against the manifest."""
        policy = compatibility_policy(dataset.compatibility_policy)
        for slide_id in (
            slide_ids if slide_ids is not None else [r.slide_id for r in dataset.records]
        ):
            r = dataset.record(slide_id)
            info = self._bags.describe(r.store_id)
            key = EmbeddingCompatibility.of(info.run, policy).key
            mismatches = [
                what
                for what, ok in (
                    ("embedding manifest", info.manifest_digest == r.embedding_manifest_digest),
                    ("embedding identity", info.embedding_identity == r.embedding_identity),
                    ("patch count", info.rows == r.patch_count),
                    ("feature dimension", info.dim == r.feature_dim),
                    ("image", info.image_id == r.image_id),
                )
                if not ok
            ]
            if key != r.compatibility_key:
                raise EmbeddingCompatibilityError(
                    f"slide {slide_id}: store {r.store_id} now has compatibility key {key}, the "
                    f"dataset recorded {r.compatibility_key}"
                )
            if mismatches:
                raise MILDatasetError(
                    f"slide {slide_id}: store {r.store_id} no longer matches the dataset "
                    f"({', '.join(mismatches)})"
                )

    def split(
        self, dataset: MILDatasetManifest, fractions: SplitFractions, seed: int
    ) -> SplitManifest:
        return generate_split(dataset, fractions, seed)

    def import_split(
        self, dataset: MILDatasetManifest, assignment: Mapping[str, Partition]
    ) -> SplitManifest:
        return import_split(dataset, assignment)
