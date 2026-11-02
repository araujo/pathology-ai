"""Labelled slide datasets, group-aware splits and bag selection for MIL (SPEC-007).

Everything here is task-, disease- and encoder-agnostic. Class names are configuration; the
framework never interprets them. Embeddings are referenced by store identity and digests,
never by path, so dataset and split digests do not depend on where stores live.

* `MILTask`: the class vocabulary of one experiment. Binary tasks list the negative class first
  and declare the positive class explicitly.
* `SlideLabel`: one slide's label record, with source, version, group and eligibility.
* `EmbeddingCompatibility`: the encoder, weights, input contract, model-side transforms and
  precision (and, under the default policy, device) a bag was produced with. Equal feature
  dimensions never imply compatibility.
* `MILDatasetManifest` (`pathology-ai.mil-dataset/1`): slides, verified embedding identities,
  labels and groups, fingerprinted.
* `SplitManifest` (`pathology-ai.mil-split/1`): group-disjoint train/validation/test membership,
  generated once (or imported) and fingerprinted.
* `BagPolicy` / `BagSelection`: explicit, deterministic and recorded patch subsampling.

Determinism does not depend on a pseudo-random generator: ranks are SHA-256 digests of the seed
and stable identifiers, so splits and selections are identical on every platform and Python
version.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from domain.embedding import (
    DeclaredTransform,
    EmbeddingRecord,
    EmbeddingRunIdentity,
    EncoderIdentity,
    Precision,
)
from domain.errors import (
    BagBudgetError,
    DuplicateSlideError,
    EmbeddingCompatibilityError,
    EmptyBagError,
    GroupLeakageError,
    InvalidLabelError,
    MILDatasetError,
    MissingGroupError,
    SplitError,
)
from domain.hashing import digest, short_digest
from domain.ids import DatasetId, ImageId, TaskId, Version
from domain.provenance import ComponentVersion
from domain.task import TaskRef, TaskType

MIL_DATASET_SCHEMA = "pathology-ai.mil-dataset/1"
MIL_SPLIT_SCHEMA = "pathology-ai.mil-split/1"
GROUPING_FIELD = "group_id"
# Slide, group, cohort and site identifiers come from external label sources, so they may use
# upper case; they must still be single tokens, never paths.
_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_MAX_CLASS_NAME = 128


def _token(what: str, value: object) -> str:
    if not isinstance(value, str) or not _TOKEN.fullmatch(value):
        raise MILDatasetError(
            f"invalid {what} {value!r}: expected [A-Za-z0-9._:-], starting alphanumeric, "
            "at most 128 characters"
        )
    return value


def _optional_token(what: str, value: str | None) -> None:
    if value is not None:
        _token(what, value)


def rank(seed: int, *parts: object) -> str:
    """A platform-independent pseudo-random rank: SHA-256 of the seed and stable identifiers."""
    return hashlib.sha256(repr((seed, *parts)).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- task


@dataclass(frozen=True, slots=True)
class MILTask:
    """A slide-level classification target. Class names are opaque configuration.

    Binary tasks have exactly two classes, the negative first, and name `positive_class`
    (which must be the second). Multiclass tasks have at least two classes and no positive class.
    """

    task_id: TaskId
    version: Version
    task_type: TaskType
    classes: tuple[str, ...]
    positive_class: str | None = None

    def __post_init__(self) -> None:
        if self.task_type not in (
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTICLASS_CLASSIFICATION,
        ):
            raise InvalidLabelError(
                f"MIL supports binary and multiclass classification, not {self.task_type.value}"
            )
        for name in self.classes:
            if not isinstance(name, str) or not name.strip() or len(name) > _MAX_CLASS_NAME:
                raise InvalidLabelError(f"invalid class name {name!r}")
            if name != name.strip():
                raise InvalidLabelError(f"class name {name!r} has surrounding whitespace")
        if len(set(self.classes)) != len(self.classes):
            raise InvalidLabelError("class names must be unique")
        if self.task_type is TaskType.BINARY_CLASSIFICATION:
            if len(self.classes) != 2:
                raise InvalidLabelError("binary classification needs exactly two classes")
            if self.positive_class != self.classes[1]:
                raise InvalidLabelError(
                    "binary tasks list the negative class first and name the second as "
                    f"positive_class (classes {list(self.classes)}, positive "
                    f"{self.positive_class!r})"
                )
        else:
            if len(self.classes) < 2:
                raise InvalidLabelError("multiclass classification needs at least two classes")
            if self.positive_class is not None:
                raise InvalidLabelError("multiclass tasks have no positive class")

    @property
    def ref(self) -> TaskRef:
        return TaskRef(self.task_id, self.version)

    @property
    def binary(self) -> bool:
        return self.task_type is TaskType.BINARY_CLASSIFICATION

    @property
    def num_outputs(self) -> int:
        """One logit for binary tasks (BCE), one per class otherwise (cross-entropy)."""
        return 1 if self.binary else len(self.classes)

    def index(self, label: str) -> int:
        try:
            return self.classes.index(label)
        except ValueError:
            raise InvalidLabelError(
                f"label {label!r} is not in the class vocabulary {list(self.classes)}"
            ) from None


# --------------------------------------------------------------------------- labels


class LabelStatus(StrEnum):
    ELIGIBLE = "eligible"  # labelled and usable for supervised training and evaluation
    EXCLUDED = "excluded"  # deliberately withheld from supervised use (reason required)
    UNLABELED = "unlabeled"  # no label: inference only


@dataclass(frozen=True, slots=True)
class SlideLabel:
    """One slide's label record. Group, cohort and site are optional metadata."""

    slide_id: str
    task: TaskRef
    value: str | None
    source: str
    source_version: str
    status: LabelStatus
    group_id: str | None = None
    cohort: str | None = None
    site: str | None = None
    status_reason: str | None = None

    def __post_init__(self) -> None:
        _token("slide_id", self.slide_id)
        _optional_token("group_id", self.group_id)
        _optional_token("cohort", self.cohort)
        _optional_token("site", self.site)
        if not self.source.strip() or not self.source_version.strip():
            raise InvalidLabelError(f"slide {self.slide_id}: label source and version are required")
        if self.status is LabelStatus.UNLABELED:
            if self.value is not None:
                raise InvalidLabelError(f"slide {self.slide_id}: an unlabeled slide has no value")
        elif self.value is None or not self.value.strip():
            raise InvalidLabelError(f"slide {self.slide_id}: status {self.status} needs a label")
        if (self.status is LabelStatus.EXCLUDED) != bool(self.status_reason):
            raise InvalidLabelError(
                f"slide {self.slide_id}: an excluded slide needs a reason, and only an excluded "
                "slide has one"
            )

    def validate_for(self, task: MILTask) -> None:
        if self.task != task.ref:
            raise InvalidLabelError(
                f"slide {self.slide_id}: label belongs to task {self.task.task_id}@"
                f"{self.task.version}, not {task.task_id}@{task.version}"
            )
        if self.value is not None:
            task.index(self.value)


# --------------------------------------------------------------------------- compatibility


@dataclass(frozen=True, slots=True)
class CompatibilityPolicy:
    """Which parts of an embedding run identity must match for bags to be mixed."""

    policy_id: str
    version: str
    binds_device: bool

    @property
    def ref(self) -> str:
        return f"{self.policy_id}@{self.version}"


# Default: everything that determines the numbers, including the device.
EXACT = CompatibilityPolicy("exact", "1", binds_device=True)
# Explicit relaxation: identical encoder, weights, input and precision on different devices.
# Report 006 measured CPU vs MPS agreement within a tolerance only; choosing this policy is a
# recorded experiment decision, never a default.
EXACT_ANY_DEVICE = CompatibilityPolicy("exact-any-device", "1", binds_device=False)
COMPATIBILITY_POLICIES = {p.ref: p for p in (EXACT, EXACT_ANY_DEVICE)}
# Declared source adaptations produce the same declared model input from different native channel
# orders (e.g. BGR or RGBA sources). They are not part of the compatibility key; every other
# transform is.
SOURCE_ADAPTATIONS = frozenset({"channel_reorder", "drop_alpha"})


def compatibility_policy(ref: str) -> CompatibilityPolicy:
    try:
        return COMPATIBILITY_POLICIES[ref]
    except KeyError:
        known = ", ".join(sorted(COMPATIBILITY_POLICIES))
        raise EmbeddingCompatibilityError(
            f"unknown compatibility policy {ref!r} (known: {known})"
        ) from None


@dataclass(frozen=True, slots=True)
class EmbeddingCompatibility:
    """The fields of an embedding run that a compatibility policy binds."""

    policy: str
    encoder: EncoderIdentity
    input_spec_fingerprint: str
    model_transforms: tuple[DeclaredTransform, ...]
    precision: Precision
    device: str | None

    @staticmethod
    def of(identity: EmbeddingRunIdentity, policy: CompatibilityPolicy) -> EmbeddingCompatibility:
        return EmbeddingCompatibility(
            policy=policy.ref,
            encoder=identity.encoder,
            input_spec_fingerprint=identity.input_spec.fingerprint,
            model_transforms=tuple(
                t for t in identity.transforms.steps if t.step.name not in SOURCE_ADAPTATIONS
            ),
            precision=identity.precision,
            device=identity.device if policy.binds_device else None,
        )

    @property
    def key(self) -> str:
        return short_digest(self, "embkey")

    @property
    def feature_dim(self) -> int:
        return self.encoder.output_dim


def describe_difference(a: EmbeddingCompatibility, b: EmbeddingCompatibility) -> str:
    """Names the bound fields that differ (for error messages)."""
    fields = (
        ("encoder component", a.encoder.component, b.encoder.component),
        ("encoder model", a.encoder.model, b.encoder.model),
        ("weights", a.encoder.weights_checksum, b.encoder.weights_checksum),
        ("encoder parameters", a.encoder.parameters, b.encoder.parameters),
        ("feature dimension", a.encoder.output_dim, b.encoder.output_dim),
        ("feature dtype", a.encoder.output_dtype, b.encoder.output_dtype),
        ("input contract", a.input_spec_fingerprint, b.input_spec_fingerprint),
        ("model transforms", a.model_transforms, b.model_transforms),
        ("precision", a.precision, b.precision),
        ("device", a.device, b.device),
        ("policy", a.policy, b.policy),
    )
    differing = [name for name, x, y in fields if x != y]
    return ", ".join(differing) or "none"


@dataclass(frozen=True, slots=True)
class EmbeddingBagInfo:
    """A verified, finalized embedding bag as read from its store (not yet labelled)."""

    store_id: str
    image_id: ImageId
    run: EmbeddingRunIdentity
    manifest_digest: str
    embedding_identity: str
    content_digest: str
    rows: int
    dim: int
    dtype: str
    records: tuple[EmbeddingRecord, ...]

    def __post_init__(self) -> None:
        if self.rows < 1 or len(self.records) != self.rows:
            raise EmptyBagError(f"store {self.store_id} holds no rows")
        if self.dim != self.run.encoder.output_dim:
            raise MILDatasetError(f"store {self.store_id}: dimension differs from its encoder")


# --------------------------------------------------------------------------- dataset


@dataclass(frozen=True, slots=True)
class MILDatasetRecord:
    """One slide: its verified embedding identities, label and grouping metadata."""

    slide_id: str
    image_id: ImageId
    store_id: str
    embedding_manifest_digest: str
    embedding_identity: str
    compatibility_key: str
    encoder: ComponentVersion
    weights_checksum: str | None
    input_spec_fingerprint: str
    feature_dim: int
    feature_dtype: str
    patch_count: int
    label: str | None
    label_status: LabelStatus
    label_source: str
    label_source_version: str
    group_id: str | None
    cohort: str | None
    site: str | None
    status_reason: str | None = None

    @property
    def supervised(self) -> bool:
        return self.label_status is LabelStatus.ELIGIBLE


@dataclass(frozen=True, slots=True)
class MILDatasetManifest:
    """A versioned, fingerprinted labelled slide dataset (`pathology-ai.mil-dataset/1`).

    All records share one compatibility key under the declared policy, and records are sorted by
    slide id. Every supervised record has a group identifier: group-aware splitting needs it, and
    slide-level splitting is never implied.
    """

    dataset_id: DatasetId
    version: Version
    task: MILTask
    compatibility_policy: str
    compatibility_key: str
    feature_dim: int
    feature_dtype: str
    grouping_field: str
    records: tuple[MILDatasetRecord, ...]
    limitations: tuple[str, ...] = ()
    schema_version: str = MIL_DATASET_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != MIL_DATASET_SCHEMA:
            raise MILDatasetError(f"unsupported dataset schema {self.schema_version!r}")
        compatibility_policy(self.compatibility_policy)
        if self.grouping_field != GROUPING_FIELD:
            raise MILDatasetError(f"grouping field must be {GROUPING_FIELD!r}")
        if not self.records:
            raise MILDatasetError("a dataset needs at least one slide")
        ids = [r.slide_id for r in self.records]
        if ids != sorted(ids):
            raise MILDatasetError("dataset records must be sorted by slide id")
        _no_duplicates("slide id", ids)
        _no_duplicates("embedding store", [r.store_id for r in self.records])
        for r in self.records:
            if r.compatibility_key != self.compatibility_key:
                raise EmbeddingCompatibilityError(
                    f"slide {r.slide_id} has compatibility key {r.compatibility_key}, the dataset "
                    f"declares {self.compatibility_key}"
                )
            if (r.feature_dim, r.feature_dtype) != (self.feature_dim, self.feature_dtype):
                raise EmbeddingCompatibilityError(f"slide {r.slide_id}: feature schema differs")
            if r.patch_count < 1:
                raise EmptyBagError(f"slide {r.slide_id} has an empty bag")
            label = SlideLabel(
                r.slide_id,
                self.task.ref,
                r.label,
                r.label_source,
                r.label_source_version,
                r.label_status,
                r.group_id,
                r.cohort,
                r.site,
                r.status_reason,
            )
            label.validate_for(self.task)
            if r.supervised and r.group_id is None:
                raise MissingGroupError(
                    f"slide {r.slide_id} is eligible for supervised use but has no "
                    f"{self.grouping_field}; slide-level splitting is never implied"
                )

    @property
    def ref(self) -> str:
        return f"{self.dataset_id}@{self.version}"

    @property
    def digest(self) -> str:
        return digest(self)

    def record(self, slide_id: str) -> MILDatasetRecord:
        for r in self.records:
            if r.slide_id == slide_id:
                return r
        raise MILDatasetError(f"slide {slide_id!r} is not in dataset {self.ref}")

    def supervised_records(self) -> tuple[MILDatasetRecord, ...]:
        return tuple(r for r in self.records if r.supervised)

    def counts(self) -> dict[str, int]:
        c = Counter(r.label_status.value for r in self.records)
        return {s.value: c.get(s.value, 0) for s in LabelStatus}


def _no_duplicates(what: str, values: Iterable[str]) -> None:
    dupes = sorted(v for v, n in Counter(values).items() if n > 1)
    if dupes:
        raise DuplicateSlideError(f"duplicate {what}: {', '.join(dupes)}")


def build_dataset_manifest(
    *,
    dataset_id: DatasetId,
    version: Version,
    task: MILTask,
    policy: CompatibilityPolicy,
    labels: Sequence[SlideLabel],
    bags: Mapping[str, EmbeddingBagInfo],
    accepted_key: str | None = None,
    limitations: tuple[str, ...] = (),
) -> MILDatasetManifest:
    """Associate labels with verified bags (`bags` maps slide id to its bag).

    Fails closed on duplicate slides or stores, labels of another task or outside the vocabulary,
    supervised slides without a group, missing bags, and mixed or unaccepted embedding identities.
    """
    _no_duplicates("slide id", [lbl.slide_id for lbl in labels])
    for lbl in labels:
        lbl.validate_for(task)
    missing = sorted(lbl.slide_id for lbl in labels if lbl.slide_id not in bags)
    if missing:
        raise MILDatasetError(f"no embedding bag for slide(s): {', '.join(missing)}")
    extra = sorted(set(bags) - {lbl.slide_id for lbl in labels})
    if extra:
        raise MILDatasetError(f"embedding bags without a label record: {', '.join(extra)}")
    if not labels:
        raise MILDatasetError("a dataset needs at least one slide")
    by_key: dict[str, EmbeddingCompatibility] = {}
    records = []
    for lbl in sorted(labels, key=lambda x: x.slide_id):
        info = bags[lbl.slide_id]
        compat = EmbeddingCompatibility.of(info.run, policy)
        by_key.setdefault(compat.key, compat)
        records.append(
            MILDatasetRecord(
                slide_id=lbl.slide_id,
                image_id=info.image_id,
                store_id=info.store_id,
                embedding_manifest_digest=info.manifest_digest,
                embedding_identity=info.embedding_identity,
                compatibility_key=compat.key,
                encoder=info.run.encoder.component,
                weights_checksum=info.run.encoder.weights_checksum,
                input_spec_fingerprint=info.run.input_spec.fingerprint,
                feature_dim=info.dim,
                feature_dtype=info.dtype,
                patch_count=info.rows,
                label=lbl.value,
                label_status=lbl.status,
                label_source=lbl.source,
                label_source_version=lbl.source_version,
                group_id=lbl.group_id,
                cohort=lbl.cohort,
                site=lbl.site,
                status_reason=lbl.status_reason,
            )
        )
    if len(by_key) > 1:
        first, *others = sorted(by_key)
        detail = "; ".join(
            f"{other} differs from {first} in {describe_difference(by_key[first], by_key[other])}"
            for other in others
        )
        raise EmbeddingCompatibilityError(
            f"the dataset mixes {len(by_key)} embedding identities under policy {policy.ref}: "
            f"{detail}"
        )
    (key,) = by_key
    if accepted_key is not None and key != accepted_key:
        raise EmbeddingCompatibilityError(
            f"the bags have compatibility key {key}; the experiment accepts {accepted_key}"
        )
    compat = by_key[key]
    return MILDatasetManifest(
        dataset_id=dataset_id,
        version=version,
        task=task,
        compatibility_policy=policy.ref,
        compatibility_key=key,
        feature_dim=compat.feature_dim,
        feature_dtype=compat.encoder.output_dtype,
        grouping_field=GROUPING_FIELD,
        records=tuple(records),
        limitations=limitations,
    )


# --------------------------------------------------------------------------- splits


class Partition(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


PARTITIONS = (Partition.TRAIN, Partition.VALIDATION, Partition.TEST)
GENERATED_STRATEGY = "group-stratified@1"
EXTERNAL_STRATEGY = "external@1"


@dataclass(frozen=True, slots=True)
class SplitFractions:
    train: float
    validation: float
    test: float

    def __post_init__(self) -> None:
        values = (self.train, self.validation, self.test)
        if any(isinstance(v, bool) or not 0.0 <= float(v) <= 1.0 for v in values):
            raise SplitError("split fractions must lie in [0, 1]")
        if abs(sum(values) - 1.0) > 1e-9:
            raise SplitError(f"split fractions must sum to 1, got {sum(values)}")
        if self.train <= 0 or self.validation <= 0:
            raise SplitError("train and validation fractions must be positive")

    def of(self, p: Partition) -> float:
        return {
            Partition.TRAIN: self.train,
            Partition.VALIDATION: self.validation,
            Partition.TEST: self.test,
        }[p]


@dataclass(frozen=True, slots=True)
class PartitionSummary:
    partition: Partition
    slide_ids: tuple[str, ...]
    group_count: int
    class_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class SplitManifest:
    """Immutable, group-disjoint partition membership (`pathology-ai.mil-split/1`)."""

    dataset_ref: str
    dataset_digest: str
    strategy: str
    grouping_field: str
    seed: int | None
    fractions: SplitFractions | None
    partitions: tuple[PartitionSummary, ...]
    group_overlap: tuple[str, ...] = ()
    schema_version: str = MIL_SPLIT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != MIL_SPLIT_SCHEMA:
            raise SplitError(f"unsupported split schema {self.schema_version!r}")
        if tuple(p.partition for p in self.partitions) != PARTITIONS:
            raise SplitError("a split lists train, validation and test, in that order")
        if self.group_overlap:
            raise GroupLeakageError(
                f"groups shared between partitions: {', '.join(self.group_overlap)}"
            )
        if self.strategy == GENERATED_STRATEGY:
            if self.seed is None or self.fractions is None:
                raise SplitError("a generated split records its seed and fractions")
        elif self.strategy == EXTERNAL_STRATEGY:
            if self.seed is not None or self.fractions is not None:
                raise SplitError("an imported split has no seed or fractions")
        else:
            raise SplitError(f"unknown split strategy {self.strategy!r}")
        for p in self.partitions:
            if list(p.slide_ids) != sorted(set(p.slide_ids)):
                raise SplitError(f"{p.partition} slide ids must be sorted and unique")
        _no_duplicates(
            "slide id across partitions", [s for p in self.partitions for s in p.slide_ids]
        )
        if not self.members(Partition.TRAIN) or not self.members(Partition.VALIDATION):
            raise SplitError("train and validation partitions must not be empty")

    def members(self, partition: Partition) -> tuple[str, ...]:
        for p in self.partitions:
            if p.partition is partition:
                return p.slide_ids
        raise SplitError(f"no partition {partition}")  # unreachable after validation

    @property
    def digest(self) -> str:
        return digest(self)

    @property
    def split_id(self) -> str:
        return short_digest(self, "split")


def _summaries(
    dataset: MILDatasetManifest, assignment: Mapping[str, Partition]
) -> tuple[tuple[PartitionSummary, ...], tuple[str, ...]]:
    groups: dict[Partition, set[str]] = {p: set() for p in PARTITIONS}
    out = []
    for p in PARTITIONS:
        ids = sorted(s for s, q in assignment.items() if q is p)
        counts: Counter[str] = Counter()
        for s in ids:
            r = dataset.record(s)
            assert r.label is not None and r.group_id is not None
            counts[r.label] += 1
            groups[p].add(r.group_id)
        out.append(
            PartitionSummary(
                p,
                tuple(ids),
                len(groups[p]),
                tuple((c, counts.get(c, 0)) for c in dataset.task.classes),
            )
        )
    overlap: set[str] = set()
    for i, a in enumerate(PARTITIONS):
        for b in PARTITIONS[i + 1 :]:
            overlap |= groups[a] & groups[b]
    return tuple(out), tuple(sorted(overlap))


def _check_assignment(dataset: MILDatasetManifest, assignment: Mapping[str, Partition]) -> None:
    for slide_id in sorted(assignment):
        r = dataset.record(slide_id)
        if not r.supervised:
            raise SplitError(
                f"slide {slide_id} is {r.label_status.value}; only eligible labelled slides can "
                "be assigned to a supervised partition"
            )
        if r.group_id is None:
            raise MissingGroupError(f"slide {slide_id} has no {dataset.grouping_field}")


def generate_split(
    dataset: MILDatasetManifest, fractions: SplitFractions, seed: int
) -> SplitManifest:
    """Group-stratified split (`group-stratified@1`) of every eligible slide.

    Groups are the unit of assignment, so a group never spans partitions. Groups are stratified
    by the sorted set of labels they contain. Within a stratum, groups are ordered by a seeded
    SHA-256 rank and assigned one by one to the partition furthest below its target slide count
    (ties: train, validation, test).
    """
    records = dataset.supervised_records()
    if not records:
        raise SplitError("the dataset has no eligible labelled slide")
    by_group: dict[str, list[MILDatasetRecord]] = {}
    for r in records:
        if r.group_id is None:
            raise MissingGroupError(f"slide {r.slide_id} has no {dataset.grouping_field}")
        by_group.setdefault(r.group_id, []).append(r)
    strata: dict[tuple[str, ...], list[str]] = {}
    for g, members in by_group.items():
        stratum = tuple(sorted({m.label or "" for m in members}))
        strata.setdefault(stratum, []).append(g)
    assignment: dict[str, Partition] = {}
    for stratum in sorted(strata):
        ordered = sorted(strata[stratum], key=lambda g: (rank(seed, "group", g), g))
        total = sum(len(by_group[g]) for g in ordered)
        filled = dict.fromkeys(PARTITIONS, 0)
        for g in ordered:
            target = max(
                PARTITIONS,
                key=lambda p: (
                    fractions.of(p) * total - filled[p] if fractions.of(p) > 0 else float("-inf"),
                    -PARTITIONS.index(p),
                ),
            )
            filled[target] += len(by_group[g])
            for m in by_group[g]:
                assignment[m.slide_id] = target
    partitions, overlap = _summaries(dataset, assignment)
    if fractions.test > 0 and not partitions[2].slide_ids:
        raise SplitError("the test partition is empty; the dataset has too few groups")
    return SplitManifest(
        dataset_ref=dataset.ref,
        dataset_digest=dataset.digest,
        strategy=GENERATED_STRATEGY,
        grouping_field=dataset.grouping_field,
        seed=seed,
        fractions=fractions,
        partitions=partitions,
        group_overlap=overlap,
    )


def import_split(dataset: MILDatasetManifest, assignment: Mapping[str, Partition]) -> SplitManifest:
    """Preserve an externally supplied split exactly; validate it, never regenerate it."""
    _check_assignment(dataset, assignment)
    partitions, overlap = _summaries(dataset, assignment)
    if overlap:
        raise GroupLeakageError(
            f"the supplied split shares {len(overlap)} group(s) between partitions: "
            f"{', '.join(overlap)}"
        )
    return SplitManifest(
        dataset_ref=dataset.ref,
        dataset_digest=dataset.digest,
        strategy=EXTERNAL_STRATEGY,
        grouping_field=dataset.grouping_field,
        seed=None,
        fractions=None,
        partitions=partitions,
    )


def validate_split(split: SplitManifest, dataset: MILDatasetManifest) -> None:
    """The split belongs to this exact dataset and its recorded summaries re-derive exactly."""
    if split.dataset_digest != dataset.digest:
        raise SplitError(
            f"split {split.split_id} was made for dataset digest {split.dataset_digest[:16]}…, "
            f"not {dataset.ref} ({dataset.digest[:16]}…); splits are never regenerated silently"
        )
    if split.grouping_field != dataset.grouping_field:
        raise SplitError("split and dataset use different grouping fields")
    assignment = {s: p.partition for p in split.partitions for s in p.slide_ids}
    _check_assignment(dataset, assignment)
    partitions, overlap = _summaries(dataset, assignment)
    if overlap:
        raise GroupLeakageError(f"groups shared between partitions: {', '.join(overlap)}")
    if partitions != split.partitions:
        raise SplitError("the split's recorded class counts or groups do not match the dataset")


# --------------------------------------------------------------------------- bag selection

SELECTION_ALGORITHM = "sha256-rank@1"


@dataclass(frozen=True, slots=True)
class BagPolicy:
    """Explicit bag-size and batch budgets.

    `max_instances` bounds the instances of one bag (`None`: every instance). Larger bags are
    subsampled deterministically by `SELECTION_ALGORITHM`: the `max_instances` indices with the
    smallest SHA-256 rank of (seed, slide id, epoch, index), returned in plan order. Evaluation
    and inference use `epoch=None`, so the same bag always yields the same selection.
    `max_batch_bytes` bounds one padded float32 batch (features plus mask).
    """

    max_instances: int | None
    seed: int
    max_batch_bytes: int
    algorithm: str = SELECTION_ALGORITHM

    def __post_init__(self) -> None:
        if self.max_instances is not None and (
            isinstance(self.max_instances, bool) or self.max_instances < 1
        ):
            raise BagBudgetError("max_instances must be a positive integer or unbounded")
        if isinstance(self.max_batch_bytes, bool) or self.max_batch_bytes < 1:
            raise BagBudgetError("max_batch_bytes must be a positive integer")
        if self.algorithm != SELECTION_ALGORITHM:
            raise BagBudgetError(f"unknown selection algorithm {self.algorithm!r}")

    def select(self, slide_id: str, count: int, epoch: int | None) -> BagSelection:
        if count < 1:
            raise EmptyBagError(f"slide {slide_id} has no instance to select")
        if self.max_instances is None or count <= self.max_instances:
            indices = tuple(range(count))
        else:
            ranked = sorted(range(count), key=lambda i: rank(self.seed, slide_id, epoch, i))
            indices = tuple(sorted(ranked[: self.max_instances]))
        return BagSelection(slide_id, count, indices, self.algorithm, self.seed, epoch)

    def padded_bytes(self, batch_slides: int, instances: int, dim: int) -> int:
        return batch_slides * instances * (dim * 4 + 1)

    def check_batch(self, batch_slides: int, largest_bag: int, dim: int) -> int:
        """The worst-case padded batch size in bytes, or `BagBudgetError` before any loading."""
        instances = (
            largest_bag if self.max_instances is None else min(largest_bag, self.max_instances)
        )
        need = self.padded_bytes(batch_slides, instances, dim)
        if need > self.max_batch_bytes:
            raise BagBudgetError(
                f"a padded batch of {batch_slides} bag(s) x {instances} instances x {dim} features "
                f"needs {need} bytes, above max_batch_bytes={self.max_batch_bytes}; lower the "
                "batch size or set max_instances"
            )
        return need


@dataclass(frozen=True, slots=True)
class BagSelection:
    """Which rows of a bag were used: reproducible from (algorithm, seed, slide, epoch)."""

    slide_id: str
    original_count: int
    indices: tuple[int, ...]
    algorithm: str
    seed: int
    epoch: int | None

    @property
    def selected_count(self) -> int:
        return len(self.indices)

    @property
    def complete(self) -> bool:
        return self.selected_count == self.original_count

    @property
    def digest(self) -> str:
        return digest(self.indices)


def batch_order(slide_ids: Sequence[str], seed: int, epoch: int) -> tuple[str, ...]:
    """Deterministic per-epoch training order."""
    return tuple(sorted(slide_ids, key=lambda s: (rank(seed, "order", epoch, s), s)))
