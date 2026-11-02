"""Typed domain errors. Every fail-closed check raises a subclass of `DomainError`."""

from __future__ import annotations


class DomainError(Exception):
    """Base class for every domain-level failure."""

    code: str = "DOMAIN_ERROR"


class InvalidIdentifierError(DomainError, ValueError):
    code = "INVALID_IDENTIFIER"


class InvalidSpatialReferenceError(DomainError, ValueError):
    code = "INVALID_SPATIAL_REFERENCE"


class CoordinateFrameMismatchError(InvalidSpatialReferenceError):
    code = "COORDINATE_FRAME_MISMATCH"


class MissingSpatialMetadataError(DomainError):
    """Required physical metadata (spacing, orientation) is absent; never fabricated."""

    code = "MISSING_SPATIAL_METADATA"


class AmbiguousModalityError(DomainError):
    """A format alone cannot determine the imaging modality (e.g. DICOM)."""

    code = "AMBIGUOUS_MODALITY"


class UnsupportedModalityError(DomainError):
    code = "UNSUPPORTED_MODALITY"


class InvalidImageAssetError(DomainError, ValueError):
    code = "INVALID_IMAGE_ASSET"


class MissingLabelError(DomainError):
    """A labelled operation (e.g. supervised training) received an unlabelled asset."""

    code = "MISSING_LABEL"


class PreprocessingContractError(DomainError):
    code = "PREPROCESSING_CONTRACT"


class MissingPreprocessingError(PreprocessingContractError):
    code = "MISSING_PREPROCESSING"


class PreprocessingMismatchError(PreprocessingContractError):
    code = "PREPROCESSING_MISMATCH"


class ChannelOrderMismatchError(PreprocessingContractError):
    """Source channel order differs from the model's expectation without an explicit transform."""

    code = "CHANNEL_ORDER_MISMATCH"


class PipelineValidationError(DomainError):
    """Aggregates every issue found by a deterministic pipeline validation."""

    code = "PIPELINE_INVALID"

    def __init__(self, issues: tuple[ValidationIssue, ...]) -> None:
        self.issues = issues
        joined = "; ".join(f"[{i.code}] {i.message}" for i in issues)
        super().__init__(joined or "pipeline invalid")

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(i.code for i in self.issues)


class ValidationIssue:
    """A single, deterministic, ordered validation finding."""

    __slots__ = ("code", "message", "location")

    def __init__(self, code: str, message: str, location: str = "") -> None:
        self.code = code
        self.message = message
        self.location = location

    def sort_key(self) -> tuple[str, str, str]:
        return (self.location, self.code, self.message)

    def __repr__(self) -> str:
        return f"ValidationIssue({self.code!r}, {self.message!r}, {self.location!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ValidationIssue) and self.sort_key() == other.sort_key()

    def __hash__(self) -> int:
        return hash(self.sort_key())


class UnsupportedModelInputError(DomainError):
    code = "UNSUPPORTED_MODEL_INPUT"


class CapabilityError(DomainError):
    code = "CAPABILITY"


class IncompatibleModelError(CapabilityError):
    code = "INCOMPATIBLE_MODEL"


class UnvalidatedUseError(CapabilityError):
    """A SUPPORTED_UNVALIDATED combination was used without explicit research opt-in."""

    code = "UNVALIDATED_USE_NOT_ACKNOWLEDGED"


class EvidenceSemanticsError(DomainError, ValueError):
    """Evidence claims an interpretation it is not validated for."""

    code = "EVIDENCE_SEMANTICS"


class ExecutionUnsupportedError(DomainError):
    """The pipeline is structurally valid but this deployment cannot execute it."""

    code = "EXECUTION_UNSUPPORTED"

    def __init__(self, message: str, missing: tuple[str, ...] = ()) -> None:
        self.missing = missing
        super().__init__(message)


class ResourceUnavailableError(DomainError):
    code = "RESOURCE_UNAVAILABLE"


class NotFoundError(DomainError, LookupError):
    code = "NOT_FOUND"


class DuplicateRegistrationError(DomainError):
    code = "DUPLICATE_REGISTRATION"


class InvalidJobTransitionError(DomainError):
    code = "INVALID_JOB_TRANSITION"


# --------------------------------------------------------------------------- image reading


class ImageReadError(DomainError):
    """Base class for failures while opening or reading a real image source."""

    code = "IMAGE_READ"


class UnsupportedImageFormatError(ImageReadError):
    """No registered reader recognises the source content (extension is never evidence)."""

    code = "UNSUPPORTED_IMAGE_FORMAT"


class AmbiguousReaderError(ImageReadError):
    """More than one reader claims the source; selection is never arbitrary."""

    code = "AMBIGUOUS_READER"


class ReaderBackendUnavailableError(ImageReadError):
    """The source is recognised, but the reader's optional backend is not installed."""

    code = "READER_BACKEND_UNAVAILABLE"


class CorruptImageError(ImageReadError):
    """The source claims a supported format but cannot be decoded consistently."""

    code = "CORRUPT_IMAGE"


class UnsupportedReadOperationError(ImageReadError):
    """Scene, plane, channel, pixel-type or other selection the reader cannot honour."""

    code = "UNSUPPORTED_READ_OPERATION"


class RegionLimitExceededError(ImageReadError):
    """A region request exceeds the configured resource limits; nothing was decoded."""

    code = "REGION_LIMIT_EXCEEDED"


class RegionOutOfBoundsError(InvalidSpatialReferenceError):
    """A region request leaves the image bounds; it is rejected, never padded."""

    code = "REGION_OUT_OF_BOUNDS"


class ReaderClosedError(ImageReadError):
    code = "READER_CLOSED"


# --------------------------------------------------------------------------- WSI preprocessing


class PreprocessingCapabilityError(DomainError):
    """The opened image or its reader lacks a capability the configured preprocessing needs.

    `requirements` names every unmet requirement, so the caller can act on all of them at once.
    """

    code = "PREPROCESSING_CAPABILITY"

    def __init__(self, message: str, requirements: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.requirements = requirements


class UnsupportedResolutionError(DomainError):
    """No stored pyramid level meets the requested resolution within the declared tolerance."""

    code = "UNSUPPORTED_RESOLUTION"


class PatchBudgetExceededError(DomainError):
    """A patch-count budget would be exceeded and the selection policy does not allow subsetting."""

    code = "PATCH_BUDGET_EXCEEDED"


class PreprocessingBudgetError(DomainError):
    """A memory, decoded-pixel or output-byte budget would be exceeded."""

    code = "PREPROCESSING_BUDGET_EXCEEDED"


class NoPatchesSelectedError(DomainError):
    """A patch plan selected nothing. Subclasses say why; the counts are in the message."""

    code = "NO_PATCHES_SELECTED"


class NoAcquiredDataError(NoPatchesSelectedError):
    code = "NO_ACQUIRED_DATA"


class NoTissueDetectedError(NoPatchesSelectedError):
    code = "NO_TISSUE_DETECTED"


class InvalidManifestError(DomainError, ValueError):
    code = "INVALID_MANIFEST"


# --------------------------------------------------------------------------- embedding (SPEC-005)


class ModelInputContractError(DomainError, ValueError):
    """A model input specification or transform chain is malformed or cannot be composed."""

    code = "MODEL_INPUT_CONTRACT"


class ResolutionMismatchError(DomainError):
    """The selected extraction geometry cannot meet the encoder's declared input requirements
    with the declared transforms; nothing is resampled implicitly."""

    code = "RESOLUTION_MISMATCH"


class EmbeddingBudgetError(DomainError):
    """A batch, in-flight or memory budget of an embedding run would be exceeded."""

    code = "EMBEDDING_BUDGET_EXCEEDED"


class EmbeddingOutputError(DomainError):
    """An encoder returned rows of the wrong shape or dtype, or non-finite values."""

    code = "EMBEDDING_OUTPUT_INVALID"


class EmbeddingStoreError(DomainError):
    code = "EMBEDDING_STORE"


class EmbeddingStoreIncompleteError(EmbeddingStoreError):
    """Not every expected row is covered by a verified, committed chunk."""

    code = "EMBEDDING_STORE_INCOMPLETE"


class EmbeddingStoreCorruptError(EmbeddingStoreError):
    """Persisted bytes or commit records fail integrity verification."""

    code = "EMBEDDING_STORE_CORRUPT"


class EmbeddingStoreIncompatibleError(EmbeddingStoreError):
    """An existing store belongs to another run identity or layout; it is never mixed."""

    code = "EMBEDDING_STORE_INCOMPATIBLE"


class EmbeddingStoreConflictError(EmbeddingStoreError):
    """An overlapping, repeated, out-of-range or uncommitted write or commit."""

    code = "EMBEDDING_STORE_CONFLICT"


# --------------------------------------------------------------------------- foundation models


class ModelRuntimeUnavailableError(CapabilityError):
    """The optional model runtime (e.g. a deep-learning framework) is not installed."""

    code = "MODEL_RUNTIME_UNAVAILABLE"


class CheckpointError(DomainError):
    code = "CHECKPOINT"


class CheckpointMissingError(CheckpointError):
    """No local checkpoint is configured, or the configured file does not exist."""

    code = "CHECKPOINT_MISSING"


class CheckpointIntegrityError(CheckpointError):
    """The checkpoint bytes do not match the pinned digest or size."""

    code = "CHECKPOINT_INTEGRITY"


class CheckpointIncompatibleError(CheckpointError):
    """The checkpoint cannot be read, or does not fit the declared architecture or output."""

    code = "CHECKPOINT_INCOMPATIBLE"


class DeviceUnavailableError(CapabilityError):
    """A requested device or precision is not available or not supported by the encoder."""

    code = "DEVICE_UNAVAILABLE"


class ProfileUnavailableError(CapabilityError):
    """A preprocessing profile is unknown, blocked or not configured completely."""

    code = "PREPROCESSING_PROFILE_UNAVAILABLE"


class ModelInferenceError(DomainError):
    """The model runtime failed while encoding a batch (e.g. a device error)."""

    code = "MODEL_INFERENCE"


# --------------------------------------------------------------------------- MIL (SPEC-007)


class MILDatasetError(DomainError, ValueError):
    """A labelled dataset, label record or dataset manifest is malformed or inconsistent."""

    code = "MIL_DATASET_INVALID"


class InvalidLabelError(MILDatasetError):
    """A label is outside the task's class vocabulary, belongs to another task, or is malformed."""

    code = "INVALID_LABEL"


class DuplicateSlideError(MILDatasetError):
    """A slide identifier, or the embedding store behind it, appears more than once."""

    code = "DUPLICATE_SLIDE"


class MissingGroupError(MILDatasetError):
    """Supervised use needs a patient/group identifier; slide-level splitting is never implied."""

    code = "MISSING_GROUP"


class SplitError(DomainError, ValueError):
    """A split manifest is malformed, does not belong to the dataset, or cannot be generated."""

    code = "INVALID_SPLIT"


class GroupLeakageError(SplitError):
    """Two partitions share a patient/group identifier."""

    code = "GROUP_LEAKAGE"


class EmbeddingCompatibilityError(DomainError):
    """Embedding bags of different encoder, weights, preprocessing or scale identities would mix,
    or a bag does not match what a model was trained on."""

    code = "EMBEDDING_INCOMPATIBLE"


class EmptyBagError(DomainError):
    """A bag has no valid instance."""

    code = "EMPTY_BAG"


class BagBudgetError(DomainError):
    """A bag, batch or memory budget of MIL loading would be exceeded."""

    code = "MIL_BUDGET_EXCEEDED"


class ExperimentConflictError(DomainError):
    """An artifact location already holds another experiment, or the same experiment with other
    results; it is never overwritten."""

    code = "EXPERIMENT_CONFLICT"


class ModelSelectionError(DomainError):
    """No epoch produced a defined validation selection metric."""

    code = "MODEL_SELECTION"


class TestIsolationError(DomainError):
    """Test data was requested before model selection was complete."""

    __test__ = False  # not a pytest test class
    code = "TEST_ISOLATION"


# --------------------------------------------------------------------------- cohorts (SPEC-008)


class CohortError(DomainError, ValueError):
    """A cohort definition, record or manifest is malformed or internally inconsistent."""

    code = "COHORT_INVALID"


class CohortSourceIntegrityError(CohortError):
    """A declared cohort source file is missing, or its bytes differ from the pinned digest."""

    code = "COHORT_SOURCE_INTEGRITY"


class CohortSourceFormatError(CohortError):
    """A source row cannot be read as declared: unknown column, unparsable identifier, value
    outside the declared vocabulary, or a repeated identifier."""

    code = "COHORT_SOURCE_FORMAT"


class CohortReconciliationError(CohortError):
    """Sources cannot be reconciled without a decision nobody has made (e.g. a case-level label
    for a case with several slides, or a label for a slide outside the roster)."""

    code = "COHORT_RECONCILIATION"


class WSIInventoryError(CohortError):
    """An image inventory is malformed, inconsistent, or used beyond what it observed (e.g. a
    scoped inventory offered as evidence for slides it never looked at)."""

    code = "WSI_INVENTORY_INVALID"


class PilotSelectionError(CohortError):
    """A pilot selection request violates the cohort's pilot policy (e.g. a partition outside the
    approved pilot partitions, or the test partition)."""

    code = "PILOT_SELECTION"
