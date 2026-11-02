"""Real-image reader registration, capability resolution and the ingestion smoke workflow.

Reader backends are registered by name and selected by probing the source content (never the
file extension). Exactly one available backend must recognise a source; zero or several is a
typed error. The pipeline engine (`application.runner`) is not involved: ingesting an image needs
no model and is not evidence that any model is valid for it.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

from application.registries import Registry
from domain.errors import (
    AmbiguousReaderError,
    NotFoundError,
    ReaderBackendUnavailableError,
    UnsupportedImageFormatError,
    UnsupportedReadOperationError,
)
from domain.hashing import digest, short_digest
from domain.kinds import IMAGE_PYRAMIDAL_2D, IMAGE_REGION_2D
from domain.provenance import ArtifactIdentity
from domain.reading import (
    ImageDescription,
    IngestionRecord,
    PixelRegion,
    ReadLimits,
    RegionReadRequest,
)
from ports import (
    ArtifactStore,
    Clock,
    DataValue,
    EnvironmentProbe,
    ImageBackend,
    ImageSource,
    OpenedImage,
    SourceFingerprinter,
    SourceProbe,
)


class ImageBackendRegistry:
    """Named reader backends; resolution is deterministic and content-based."""

    def __init__(self) -> None:
        self._backends: Registry[str, ImageBackend] = Registry("image backend")

    def register(self, backend: ImageBackend) -> None:
        self._backends.register(backend.name, backend)

    def get(self, name: str) -> ImageBackend:
        return self._backends.get(name)

    def backends(self) -> tuple[ImageBackend, ...]:
        return self._backends.values()

    def resolve(self, source: ImageSource, backend: str | None = None) -> ImageBackend:
        """The single backend that recognises `source`'s content and can run here."""
        if not source.path.is_file():
            raise NotFoundError(f"image source for {source.image_id} is not a readable file")
        candidates = (self.get(backend),) if backend is not None else self.backends()
        recognised: list[tuple[ImageBackend, SourceProbe]] = []
        for b in candidates:
            probe = b.probe(source.path)
            if probe.recognised:
                recognised.append((b, probe))
        usable = [(b, p) for b, p in recognised if b.available()]
        if len(usable) == 1:
            return usable[0][0]
        if len(usable) > 1:
            names = ", ".join(sorted(b.name for b, _ in usable))
            raise AmbiguousReaderError(
                f"several readers recognise {source.image_id} ({names}); select one explicitly"
            )
        if recognised:
            details = "; ".join(f"{b.name}: {p.detail}" for b, p in recognised)
            raise ReaderBackendUnavailableError(
                f"{source.image_id} is recognised but no backend is installed ({details})"
            )
        tried = ", ".join(b.name for b in candidates) or "none registered"
        raise UnsupportedImageFormatError(
            f"no reader recognises the content of {source.image_id} (tried: {tried})"
        )


@dataclass(frozen=True)
class IngestionOutcome:
    description: ImageDescription
    regions: tuple[PixelRegion, ...]
    values: tuple[DataValue, ...]
    record: IngestionRecord


class ImagingService:
    """Opens real images through the registry and records read provenance."""

    def __init__(
        self,
        *,
        backends: ImageBackendRegistry,
        fingerprinter: SourceFingerprinter,
        artifacts: ArtifactStore,
        clock: Clock,
        environment: EnvironmentProbe,
        limits: ReadLimits | None = None,
    ) -> None:
        self.backends = backends
        self._fingerprinter = fingerprinter
        self._artifacts = artifacts
        self._clock = clock
        self._environment = environment
        self.limits = limits or ReadLimits()
        self._records: list[IngestionRecord] = []

    @contextmanager
    def open(
        self, source: ImageSource, *, scene: int | None = None, backend: str | None = None
    ) -> Iterator[OpenedImage]:
        """Open `source`; the handle is always closed when the block exits."""
        reader = self.backends.resolve(source, backend)
        integrity = self._fingerprinter.fingerprint(source.path)
        opened = reader.open(source, scene=scene, limits=self.limits, integrity=integrity)
        try:
            yield opened
        finally:
            opened.close()

    def describe(
        self, source: ImageSource, *, scene: int | None = None, backend: str | None = None
    ) -> ImageDescription:
        with self.open(source, scene=scene, backend=backend) as image:
            return image.describe()

    def ingest(
        self,
        source: ImageSource,
        regions: Sequence[RegionReadRequest],
        *,
        scene: int | None = None,
        backend: str | None = None,
    ) -> IngestionOutcome:
        """Real image -> registry -> metadata -> bounded regions -> artifacts -> provenance."""
        if not regions:
            raise UnsupportedReadOperationError("ingestion needs at least one bounded region")
        started = self._clock.now_iso()
        with self.open(source, scene=scene, backend=backend) as image:
            description = image.describe()
            read = tuple(image.read_region(r) for r in regions)

        image_identity = ArtifactIdentity.derive(
            IMAGE_PYRAMIDAL_2D,
            {"image": description.image_id, "checksum": description.asset.source_checksum},
        )
        values: list[DataValue] = []
        for region in read:
            identity = ArtifactIdentity.derive(
                IMAGE_REGION_2D, {"region": region.provenance}, (image_identity.artifact_id,)
            )
            self._artifacts.put(identity, region)
            values.append(DataValue(IMAGE_REGION_2D, region, identity))

        description_digest = digest(description)
        record = IngestionRecord(
            record_id=short_digest(
                {
                    "description": description_digest,
                    "regions": [r.provenance for r in read],
                    "n": len(self._records),
                },
                "ingest",
            ),
            image_id=description.image_id,
            reader=description.reader,
            backend=description.backend,
            integrity=description.integrity,
            description_digest=description_digest,
            regions=tuple(r.provenance for r in read),
            artifacts=tuple(v.identity for v in values),
            environment=self._environment.capture(),
            started_at=started,
            finished_at=self._clock.now_iso(),
        )
        self._records.append(record)
        return IngestionOutcome(description, read, tuple(values), record)

    def records(self) -> tuple[IngestionRecord, ...]:
        return tuple(self._records)
