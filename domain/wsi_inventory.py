"""Whole-slide image inventory (SPEC-008 §5, AC-008): what was observed, where, and how well.

An inventory records every image object found under a set of named roots and maps it to the
canonical slide identities of one cohort manifest (by the image file name each record declares).
It is cohort- and format-agnostic: the identities, file names and roots are inputs.

Evidence levels are kept apart and never promoted:

* `content_verified`: a local object whose bytes were read end to end in this inventory and
  hashed with SHA-256. Only these carry a `sha256`.
* `metadata_only`: a remote object known from a listing (size, and possibly an opaque checksum
  the store reports, such as an object-store entity tag). An entity tag is not a SHA-256 of the
  content (multipart uploads, server-side encryption and store-specific algorithms all change
  it), so it is recorded as `remote_checksum` and never compared with a content digest.
* `failed`: a local object that could not be read consistently.

Duplicate content is detected only between content-verified objects of *different* slide
identities. Several verified copies of one slide must agree; otherwise the slide is
`inconsistent_copies`.

An inventory is *scoped*: a slide not found under the inventoried roots is `not_found_in_scope`,
which is not evidence that the image is missing. `cohort_wsi_evidence` therefore refuses to turn
an inventory into cohort integrity evidence unless every slide of the cohort has been verified
(or positively failed) by it: closing the duplicate-content gate for one slide needs the content
of every other slide.

`WSIInventory` (`pathology-ai.wsi-inventory/1`) re-derives its per-slide records, duplicate groups
and unmatched objects from its observations on construction, so an edited document is refused.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from domain.cohort import CohortManifest, WSIEvidence, WSIStatus
from domain.errors import WSIInventoryError
from domain.hashing import digest, short_digest

INVENTORY_SCHEMA = "pathology-ai.wsi-inventory/1"
CONTENT_HASH = "sha256/full-file/v1"
_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _token(what: str, value: object) -> str:
    if not isinstance(value, str) or not _TOKEN.fullmatch(value):
        raise WSIInventoryError(f"invalid {what} {value!r}")
    return value


def _unique(what: str, values: Sequence[str]) -> None:
    dupes = sorted(v for v, n in Counter(values).items() if n > 1)
    if dupes:
        raise WSIInventoryError(f"duplicate {what}: {', '.join(dupes[:10])}")


class LocationKind(StrEnum):
    LOCAL = "local"  # a filesystem the inventory can read
    REMOTE = "remote"  # an object store or server known only from a listing


class Access(StrEnum):
    READABLE = "readable"  # every byte was read in this inventory
    UNREADABLE = "unreadable"  # present, but reading failed or the object changed while read
    LISTED = "listed"  # named by a remote listing; no byte was read


class Verification(StrEnum):
    CONTENT_VERIFIED = "content_verified"
    METADATA_ONLY = "metadata_only"
    FAILED = "failed"


class SlideAvailability(StrEnum):
    VERIFIED = "verified"  # at least one content-verified copy; all verified copies agree
    INCONSISTENT_COPIES = "inconsistent_copies"  # verified copies disagree on content
    UNREADABLE = "unreadable"  # local copies exist; none could be read
    REMOTE_ONLY = "remote_only"  # only listed remotely; content not verified
    NOT_FOUND_IN_SCOPE = "not_found_in_scope"  # no copy under the inventoried roots


# Availabilities that settle a slide's image integrity one way or the other.
SETTLED = (
    SlideAvailability.VERIFIED,
    SlideAvailability.INCONSISTENT_COPIES,
    SlideAvailability.UNREADABLE,
)


@dataclass(frozen=True, slots=True)
class InventoryRoot:
    """A named place that was inventoried. Local roots are bound to directories at run time;
    their absolute paths never enter an inventory."""

    name: str
    kind: LocationKind
    description: str

    def __post_init__(self) -> None:
        _token("root name", self.name)
        if not isinstance(self.description, str) or not self.description.strip():
            raise WSIInventoryError(f"root {self.name} needs a description")


@dataclass(frozen=True, slots=True)
class ExpectedImage:
    """A canonical slide identity and the image file name its cohort record declares."""

    slide_id: str
    file_name: str

    def __post_init__(self) -> None:
        _token("slide id", self.slide_id)
        if not self.file_name or "/" in self.file_name or "\\" in self.file_name:
            raise WSIInventoryError(f"slide {self.slide_id}: invalid image file name")


@dataclass(frozen=True, slots=True)
class ObservedObject:
    """One object found under a root, with exactly the evidence that was obtained for it."""

    root: str
    path: str
    kind: LocationKind
    access: Access
    verification: Verification
    size_bytes: int | None
    sha256: str | None
    remote_checksum: str | None = None
    remote_checksum_kind: str | None = None
    note: str = ""

    def __post_init__(self) -> None:
        _token("root name", self.root)
        parts = self.path.split("/")
        if (
            not self.path
            or self.path.startswith("/")
            or "\\" in self.path
            or any(p in ("", ".", "..") for p in parts)
        ):
            raise WSIInventoryError(
                f"object path {self.path!r} must be relative to its root, without '.' or '..'"
            )
        if self.size_bytes is not None and (
            isinstance(self.size_bytes, bool) or self.size_bytes < 0
        ):
            raise WSIInventoryError(f"{self.uri}: size must be a non-negative integer")
        if self.sha256 is not None and not _SHA256.fullmatch(self.sha256):
            raise WSIInventoryError(f"{self.uri}: sha256 must be lowercase hex")
        verified = self.verification is Verification.CONTENT_VERIFIED
        if verified != (self.sha256 is not None):
            raise WSIInventoryError(
                f"{self.uri}: a SHA-256 is recorded exactly for content-verified objects"
            )
        if (self.remote_checksum is None) != (self.remote_checksum_kind is None):
            raise WSIInventoryError(f"{self.uri}: a remote checksum names its kind")
        if self.kind is LocationKind.LOCAL:
            expected = {
                Access.READABLE: Verification.CONTENT_VERIFIED,
                Access.UNREADABLE: Verification.FAILED,
            }.get(self.access)
            if expected is not self.verification:
                raise WSIInventoryError(
                    f"{self.uri}: a local object is readable and content-verified, or unreadable "
                    "and failed"
                )
            if self.remote_checksum is not None:
                raise WSIInventoryError(f"{self.uri}: a local object has no remote checksum")
            if verified and (self.size_bytes is None or self.size_bytes < 1):
                raise WSIInventoryError(f"{self.uri}: a verified object has a positive size")
        else:
            if self.access is not Access.LISTED or self.verification is not (
                Verification.METADATA_ONLY
            ):
                raise WSIInventoryError(
                    f"{self.uri}: a remote object is listed metadata only; its content is "
                    "verified only after it has been fetched and hashed locally"
                )
            if self.remote_checksum_kind is not None:
                _token("remote checksum kind", self.remote_checksum_kind)

    @property
    def uri(self) -> str:
        return f"{self.root}:{self.path}"

    @property
    def file_name(self) -> str:
        return self.path.rsplit("/", 1)[-1]


@dataclass(frozen=True, slots=True)
class SlideInventoryRecord:
    slide_id: str
    file_name: str
    availability: SlideAvailability
    sha256: str | None
    size_bytes: int | None
    copies: tuple[str, ...]
    shares_content_with: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DuplicateContent:
    """Identical verified content under different slide identities."""

    sha256: str
    slide_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class WSIInventory:
    """A content-addressed observation of image objects against canonical slide identities."""

    identity_source: str
    roots: tuple[InventoryRoot, ...]
    patterns: tuple[str, ...]
    expected: tuple[ExpectedImage, ...]
    objects: tuple[ObservedObject, ...]
    slides: tuple[SlideInventoryRecord, ...]
    duplicates: tuple[DuplicateContent, ...]
    unmatched: tuple[str, ...]
    content_hash: str = CONTENT_HASH
    schema_version: str = INVENTORY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != INVENTORY_SCHEMA:
            raise WSIInventoryError(f"unsupported inventory schema {self.schema_version!r}")
        if self.content_hash != CONTENT_HASH:
            raise WSIInventoryError(f"unsupported content hash {self.content_hash!r}")
        if not self.identity_source.strip():
            raise WSIInventoryError("an inventory names the source of its identities")
        if not self.roots:
            raise WSIInventoryError("an inventory covers at least one root")
        if not self.patterns:
            raise WSIInventoryError("an inventory declares the object name patterns it matched")
        roots = {r.name: r for r in self.roots}
        _unique("root", [r.name for r in self.roots])
        if [r.name for r in self.roots] != sorted(roots):
            raise WSIInventoryError("roots must be sorted by name")
        ids = [e.slide_id for e in self.expected]
        if ids != sorted(ids):
            raise WSIInventoryError("expected images must be sorted by slide id")
        _unique("slide id", ids)
        _unique("expected image file name", [e.file_name for e in self.expected])
        uris = [o.uri for o in self.objects]
        if uris != sorted(uris):
            raise WSIInventoryError("objects must be sorted by uri")
        _unique("object uri", uris)
        for o in self.objects:
            root = roots.get(o.root)
            if root is None:
                raise WSIInventoryError(f"{o.uri}: root {o.root!r} is not declared")
            if root.kind is not o.kind:
                raise WSIInventoryError(f"{o.uri}: object kind differs from its root's")
        derived = derive_slides(self.expected, self.objects)
        if derived != (self.slides, self.duplicates, self.unmatched):
            raise WSIInventoryError("the recorded slide summary does not re-derive from objects")

    @property
    def digest(self) -> str:
        return digest(self)

    @property
    def inventory_id(self) -> str:
        return short_digest(self, "wsi-inventory")

    @property
    def complete(self) -> bool:
        """Every expected slide is settled (verified, inconsistent or unreadable)."""
        return all(s.availability in SETTLED for s in self.slides)

    def counts(self) -> dict[str, int]:
        c = Counter(s.availability.value for s in self.slides)
        return {a.value: c.get(a.value, 0) for a in SlideAvailability}

    def slide(self, slide_id: str) -> SlideInventoryRecord:
        for s in self.slides:
            if s.slide_id == slide_id:
                return s
        raise WSIInventoryError(f"slide {slide_id!r} is not in this inventory")


def derive_slides(
    expected: Iterable[ExpectedImage], objects: Iterable[ObservedObject]
) -> tuple[tuple[SlideInventoryRecord, ...], tuple[DuplicateContent, ...], tuple[str, ...]]:
    """Per-slide availability, cross-identity duplicate content and unmatched object uris."""
    by_name = {e.file_name: e.slide_id for e in expected}
    copies: dict[str, list[ObservedObject]] = defaultdict(list)
    unmatched: list[str] = []
    for o in objects:
        slide_id = by_name.get(o.file_name)
        if slide_id is None:
            unmatched.append(o.uri)
        else:
            copies[slide_id].append(o)
    content: dict[str, str | None] = {}
    partial: list[tuple[str, str, SlideAvailability, int | None, tuple[str, ...]]] = []
    for e in sorted(expected, key=lambda x: x.slide_id):
        found = sorted(copies.get(e.slide_id, []), key=lambda o: o.uri)
        verified = [o for o in found if o.verification is Verification.CONTENT_VERIFIED]
        digests = {o.sha256 for o in verified}
        sizes = {o.size_bytes for o in verified}
        size: int | None = None
        if len(digests) > 1 or len(sizes) > 1:
            availability = SlideAvailability.INCONSISTENT_COPIES
            content[e.slide_id] = None
        elif verified:
            availability = SlideAvailability.VERIFIED
            content[e.slide_id] = verified[0].sha256
            size = verified[0].size_bytes
        elif any(o.kind is LocationKind.LOCAL for o in found):
            availability = SlideAvailability.UNREADABLE
        elif found:
            availability = SlideAvailability.REMOTE_ONLY
        else:
            availability = SlideAvailability.NOT_FOUND_IN_SCOPE
        partial.append((e.slide_id, e.file_name, availability, size, tuple(o.uri for o in found)))
    holders: dict[str, list[str]] = defaultdict(list)
    for slide_id, sha in content.items():
        if sha is not None:
            holders[sha].append(slide_id)
    duplicates = tuple(
        DuplicateContent(sha, tuple(sorted(ids)))
        for sha, ids in sorted(holders.items())
        if len(ids) > 1
    )
    shared = {s: d.slide_ids for d in duplicates for s in d.slide_ids}
    slides = tuple(
        SlideInventoryRecord(
            slide_id=slide_id,
            file_name=file_name,
            availability=availability,
            sha256=content.get(slide_id),
            size_bytes=size,
            copies=uris,
            shares_content_with=tuple(s for s in shared.get(slide_id, ()) if s != slide_id),
        )
        for slide_id, file_name, availability, size, uris in partial
    )
    return slides, duplicates, tuple(sorted(unmatched))


def identity_source(manifest: CohortManifest) -> str:
    return f"cohort:{manifest.ref}#{manifest.digest}"


def expected_images(manifest: CohortManifest) -> tuple[ExpectedImage, ...]:
    """Every record of the manifest (eligible or not): duplicates can hide among exclusions."""
    return tuple(ExpectedImage(r.slide_id, r.wsi.file_name) for r in manifest.records)


def build_inventory(
    manifest: CohortManifest,
    roots: Iterable[InventoryRoot],
    patterns: Iterable[str],
    objects: Iterable[ObservedObject],
) -> WSIInventory:
    expected = expected_images(manifest)
    observed = tuple(sorted(objects, key=lambda o: o.uri))
    slides, duplicates, unmatched = derive_slides(expected, observed)
    return WSIInventory(
        identity_source=identity_source(manifest),
        roots=tuple(sorted(roots, key=lambda r: r.name)),
        patterns=tuple(sorted(set(patterns))),
        expected=expected,
        objects=observed,
        slides=slides,
        duplicates=duplicates,
        unmatched=unmatched,
    )


def check_identity_source(inventory: WSIInventory, manifest: CohortManifest) -> None:
    if inventory.identity_source != identity_source(manifest):
        raise WSIInventoryError(
            f"inventory {inventory.inventory_id} was taken against {inventory.identity_source}, "
            f"not {identity_source(manifest)}"
        )


def cohort_wsi_evidence(
    inventory: WSIInventory, manifest: CohortManifest
) -> dict[str, WSIEvidence]:
    """Image evidence for a new cohort manifest version, from a *complete* inventory only.

    A scoped inventory cannot close any cohort gate: a slide that was not found is not known to
    be missing, and the absence of duplicate content can only be established over every slide.
    """
    check_identity_source(inventory, manifest)
    if not inventory.complete:
        open_ = {k: v for k, v in inventory.counts().items() if v and k not in SETTLED}
        raise WSIInventoryError(
            f"inventory {inventory.inventory_id} does not settle every slide of {manifest.ref} "
            f"({open_}); a scoped inventory never closes cohort integrity gates"
        )
    out: dict[str, WSIEvidence] = {}
    for s in inventory.slides:
        if s.availability is SlideAvailability.VERIFIED:
            out[s.slide_id] = WSIEvidence(WSIStatus.VERIFIED, s.file_name, s.size_bytes, s.sha256)
        else:
            out[s.slide_id] = WSIEvidence(
                WSIStatus.UNREADABLE,
                s.file_name,
                note=f"{s.availability.value} in inventory {inventory.inventory_id}",
            )
    return out
