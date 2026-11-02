"""`embed.patches@1`: a SPEC-004 patch plan and manifest in, a durable embedding bag out.

The whole batching loop lives inside this one stage (ADR-005 D2). Before any pixel is read the
operation:

1. checks that the manifest belongs to the plan and the opened image, and that it holds no
   pixels (`materialize_pixels: false`);
2. composes the transform chain from the declared channel conversions and resampling plus the
   encoder's `ModelInputSpec`, and validates it (`RESOLUTION_MISMATCH`,
   `CHANNEL_ORDER_MISMATCH`, `MISSING_SPATIAL_METADATA`, `MODEL_INPUT_CONTRACT`);
3. plans the batch budget (`EMBEDDING_BUDGET_EXCEEDED`);
4. derives the run identity and opens the durable store for it, resuming only from verified
   chunks and refusing incompatible stores.

It returns only an `EmbeddingBagPayload`: the finalized manifest, a store reference and bounded
metrics. Embedding rows never pass through the in-memory artifact store.
"""

from __future__ import annotations

from collections.abc import Mapping

from domain.capabilities import PreprocessingRequirements, require_capabilities
from domain.embedding import (
    BatchPolicy,
    EmbeddingManifest,
    EmbeddingMetrics,
    EmbeddingRunIdentity,
    EmbeddingStoreHeader,
    PixelState,
    Precision,
    PrefetchPolicy,
    RecoveryPolicy,
    ResampleMethod,
    ResolutionRecord,
    StoreDescriptor,
    StoreLayout,
    StoreState,
    build_transform_chain,
    itemsize,
    patch_ids_digest,
    records_from_manifest,
    validate_model_input,
)
from domain.errors import (
    EmbeddingStoreIncompatibleError,
    ExecutionUnsupportedError,
    ModelInputContractError,
    PatchBudgetExceededError,
)
from domain.ids import Version
from domain.image import ChannelOrder
from domain.manifest import PatchManifest, PixelStorage
from domain.patching import PatchPlan
from domain.pipeline import ConfigValue, OperationRef
from domain.provenance import ComponentVersion
from domain.reading import ImageDescription
from plugins.embedding import PLUGIN_VERSION
from plugins.embedding.streaming import LazyPatchSource, encode_into_store
from plugins.embedding.transforms import IMPLEMENTATIONS
from plugins.preprocessing.detection import cfg_int, cfg_str
from ports import (
    EMBEDDING_STORE_EXTRA,
    DataValue,
    EmbeddingBagPayload,
    EmbeddingEncoder,
    EmbeddingEncoderFactory,
    EmbeddingStoreProvider,
    OpenedImage,
    PatchManifestPayload,
    StageContext,
)

EMBED_PATCHES = OperationRef("embed.patches", Version("1"))
# Transform temporaries per patch, counted in the working set: the floating copy of the decoded
# patch plus up to two further patch-sized intermediates (e.g. normalize, layout). Encoder-internal
# memory is declared by the encoder (`EmbeddingEncoder.working_bytes`) and counted as well.
SCRATCH_FLOAT_COPIES = 3


def _component() -> ComponentVersion:
    return ComponentVersion("operation", str(EMBED_PATCHES), PLUGIN_VERSION)


def channel_conversions(
    cfg: Mapping[str, ConfigValue],
) -> frozenset[tuple[ChannelOrder, ChannelOrder]]:
    """`"bgr->rgb,rgba->rgb"`; `"none"` declares no conversion."""
    text = cfg_str(cfg, "channel_conversions", "none").strip()
    if text in ("", "none"):
        return frozenset()
    out: set[tuple[ChannelOrder, ChannelOrder]] = set()
    for item in text.split(","):
        src, sep, dst = item.strip().partition("->")
        try:
            if not sep:
                raise ValueError(item)
            out.add((ChannelOrder(src.strip()), ChannelOrder(dst.strip())))
        except ValueError:
            raise ModelInputContractError(
                f"invalid channel conversion {item!r}; use e.g. bgr->rgb"
            ) from None
    return frozenset(out)


def batch_policy(cfg: Mapping[str, ConfigValue]) -> BatchPolicy:
    try:
        prefetch = PrefetchPolicy(cfg_str(cfg, "prefetch", "none"))
        precision = Precision(cfg_str(cfg, "precision", "fp32"))
    except ValueError as exc:
        raise ModelInputContractError(f"invalid embedding configuration: {exc}") from None
    return BatchPolicy(
        max_batch_patches=cfg_int(cfg, "batch_size", 32),
        max_input_batch_bytes=cfg_int(cfg, "max_input_batch_bytes", 32 << 20),
        max_output_batch_bytes=cfg_int(cfg, "max_output_batch_bytes", 1 << 20),
        memory_budget_bytes=cfg_int(cfg, "memory_budget_bytes", 128 << 20),
        max_in_flight_batches=cfg_int(cfg, "max_in_flight_batches", 1),
        prefetch=prefetch,
        device=cfg_str(cfg, "device", "cpu"),
        precision=precision,
    )


def _resample(cfg: Mapping[str, ConfigValue]) -> ResampleMethod | None:
    text = cfg_str(cfg, "resample", "none")
    if text == "none":
        return None
    try:
        return ResampleMethod(text)
    except ValueError:
        raise ModelInputContractError(f"unknown resampling method {text!r}") from None


def _recovery(cfg: Mapping[str, ConfigValue]) -> RecoveryPolicy:
    try:
        return RecoveryPolicy(cfg_str(cfg, "recovery", "resume_verified"))
    except ValueError as exc:
        raise ModelInputContractError(f"invalid recovery policy: {exc}") from None


class EmbeddingOperation:
    """Encoders are selected by name (stage config `encoder`), like tissue detectors."""

    def __init__(self, encoders: Mapping[str, EmbeddingEncoderFactory]) -> None:
        self._encoders = dict(encoders)

    def add_encoder(self, factory: EmbeddingEncoderFactory) -> None:
        if factory.name in self._encoders:
            raise ExecutionUnsupportedError(f"encoder {factory.name!r} is already registered")
        self._encoders[factory.name] = factory

    @property
    def operation(self) -> OperationRef:
        return EMBED_PATCHES

    @property
    def component(self) -> ComponentVersion:
        return _component()

    def _factory(self, cfg: Mapping[str, ConfigValue]) -> EmbeddingEncoderFactory:
        name = cfg_str(cfg, "encoder", "")
        try:
            return self._encoders[name]
        except KeyError:
            known = ", ".join(sorted(self._encoders)) or "none"
            raise ExecutionUnsupportedError(
                f"embedding encoder {name!r} is not registered (known: {known})",
                (f"encoder:{name}",),
            ) from None

    def requirements(self, config: Mapping[str, ConfigValue]) -> PreprocessingRequirements:
        """Region reads, integer pixels, and a channel order the declared conversions reach."""
        spec = self._factory(config).input_spec(config)
        reachable = {spec.channel_order} | {
            src for src, dst in channel_conversions(config) if dst is spec.channel_order
        }
        return PreprocessingRequirements(
            channel_orders=frozenset(reachable),
            dtypes=frozenset({"uint8", "uint16"}),
            needs_regions=True,
            needs_physical_spacing=spec.pixel_spacing_um is not None,
        )

    def run(self, inputs: Mapping[str, DataValue], ctx: StageContext) -> Mapping[str, object]:
        cfg = ctx.stage.config_dict()
        factory = self._factory(cfg)
        conversions = channel_conversions(cfg)
        resample = _resample(cfg)
        policy = batch_policy(cfg)
        recovery = _recovery(cfg)
        max_patches = cfg_int(cfg, "max_patches", 100_000)
        image, d = self._source(ctx)
        supports = bool(ctx.reader and ctx.reader.capabilities.supports_regions)
        require_capabilities(
            self.requirements(cfg).evaluate(d, supports_regions=supports),
            context=str(EMBED_PATCHES),
        )
        plan, manifest = self._inputs(inputs, d)
        if not 1 <= len(manifest.patches) <= max_patches:
            raise PatchBudgetExceededError(
                f"{len(manifest.patches)} patches; embedding accepts 1..{max_patches}"
            )
        provider = ctx.extras.get(EMBEDDING_STORE_EXTRA)
        if not isinstance(provider, EmbeddingStoreProvider):
            raise ExecutionUnsupportedError(
                f"{EMBED_PATCHES} needs a durable embedding store (ExecutionRequest.extras"
                f"[{EMBEDDING_STORE_EXTRA!r}])",
                ("store:embedding",),
            )
        encoder = factory.create(cfg)
        try:
            return {
                "embeddings": self._embed(
                    ctx,
                    image,
                    d,
                    plan,
                    manifest,
                    encoder,
                    provider,
                    policy,
                    recovery,
                    conversions,
                    resample,
                )
            }
        finally:
            encoder.close()

    @staticmethod
    def _source(ctx: StageContext) -> tuple[OpenedImage, ImageDescription]:
        if ctx.source is None:
            raise ExecutionUnsupportedError(
                f"{EMBED_PATCHES} needs an opened real image (ExecutionRequest.source)",
                ("source:opened_image",),
            )
        return ctx.source, ctx.source.describe()

    @staticmethod
    def _inputs(
        inputs: Mapping[str, DataValue], d: ImageDescription
    ) -> tuple[PatchPlan, PatchManifest]:
        plan, payload = inputs["plan"].payload, inputs["manifest"].payload
        if not isinstance(plan, PatchPlan) or not isinstance(payload, PatchManifestPayload):
            raise ExecutionUnsupportedError(f"{EMBED_PATCHES} needs a PatchPlan and a manifest")
        m = payload.manifest
        if m.plan_digest != plan.plan_digest:
            raise ExecutionUnsupportedError("the patch manifest was not produced from this plan")
        if (m.image_id, m.scene, m.source_integrity) != (
            d.image_id,
            d.scene,
            d.asset.source_checksum,
        ):
            raise ExecutionUnsupportedError("the patch manifest belongs to another image or scene")
        if m.pixel_storage is not PixelStorage.NOT_MATERIALIZED or payload.artifacts:
            raise ExecutionUnsupportedError(
                f"{EMBED_PATCHES} reads pixels lazily; produce the manifest with "
                "materialize_pixels: false so no stage retains every patch's pixels",
                ("manifest:not_materialized",),
            )
        return plan, m

    def _embed(
        self,
        ctx: StageContext,
        image: OpenedImage,
        d: ImageDescription,
        plan: PatchPlan,
        manifest: PatchManifest,
        encoder: EmbeddingEncoder,
        provider: EmbeddingStoreProvider,
        policy: BatchPolicy,
        recovery: RecoveryPolicy,
        conversions: frozenset[tuple[ChannelOrder, ChannelOrder]],
        resample: ResampleMethod | None,
    ) -> EmbeddingBagPayload:
        spec, identity = encoder.input_spec, encoder.identity
        r = manifest.resolved
        source_state = PixelState.of_patch(
            r.width_px,
            r.height_px,
            d.channel_order,
            d.samples_per_pixel,
            d.dtype,
            r.effective_spacing_um,
        )
        # Everything below up to the store is validated without reading a pixel.
        chain = build_transform_chain(
            source_state,
            spec,
            channel_conversions=conversions,
            resample=resample,
            implementations=IMPLEMENTATIONS,
        )
        model_input = validate_model_input(source_state, spec, chain, policy.precision)
        resolution = ResolutionRecord.of(
            d.geometry.pixel_spacing, d.spacing_source, r, model_input, spec, chain
        )
        float_patch = r.width_px * r.height_px * d.samples_per_pixel * itemsize(spec.dtype)
        budget = policy.plan(
            patches=len(manifest.patches),
            decoded_patch_bytes=source_state.nbytes,
            model_input_patch_bytes=spec.patch_bytes,
            output_row_bytes=identity.output_dim * itemsize(identity.output_dtype),
            scratch_bytes=source_state.nbytes
            + SCRATCH_FLOAT_COPIES * float_patch
            + encoder.working_bytes(min(policy.max_batch_patches, len(manifest.patches))),
        )
        records = records_from_manifest(manifest)
        run = EmbeddingRunIdentity(
            image_id=d.image_id,
            source_integrity=d.asset.source_checksum,
            scene=d.scene,
            plan_digest=plan.plan_digest,
            patch_manifest_digest=manifest.manifest_digest,
            records_digest=patch_ids_digest(records),
            encoder=identity,
            input_spec=spec,
            transforms=chain,
            precision=policy.precision,
            device=policy.device,
        )
        header = EmbeddingStoreHeader(
            store_id=run.store_id,
            identity=run,
            layout=StoreLayout(
                len(records), identity.output_dim, identity.output_dtype, budget.batch_patches
            ),
            records=records,
        )
        source = LazyPatchSource(image, d, r.width_px, r.height_px)
        store = provider.open(header, recovery=recovery)
        try:
            inspection = store.inspect()
            reused = inspection.state is StoreState.FINALIZED
            counters = None
            if not reused:
                counters = encode_into_store(
                    source=source,
                    chain=chain,
                    encoder=encoder,
                    store=store,
                    header=header,
                    budget=budget,
                    policy=policy,
                    skip=inspection.committed_indices,
                )
            verified = store.verify()
            # Limitations are deterministic statements of the configured encoder; they must not
            # depend on whether this run encoded or reused the store.
            report = encoder.runtime_report()
            out = EmbeddingManifest(
                image_id=d.image_id,
                source_integrity=d.asset.source_checksum,
                scene=d.scene,
                pipeline=ctx.pipeline,
                reader=d.reader,
                backend=d.backend,
                operation=self.component,
                plan_digest=plan.plan_digest,
                patch_manifest_digest=manifest.manifest_digest,
                patch_manifest_schema=manifest.schema_version,
                patch_pixel_storage=manifest.pixel_storage,
                run_identity=run.value,
                encoder=identity,
                input_spec=spec,
                transforms=chain,
                resolution=resolution,
                precision=policy.precision,
                device=policy.device,
                embedding_dim=identity.output_dim,
                embedding_dtype=identity.output_dtype,
                row_count=len(records),
                records=records,
                store=StoreDescriptor(
                    format=provider.format,
                    schema_version=header.schema_version,
                    store_id=header.store_id,
                    chunk_rows=header.layout.chunk_rows,
                    chunk_count=header.layout.chunk_count,
                    chunks_digest=verified.chunks_digest,
                ),
                content_digest=verified.content_digest,
                status=StoreState.FINALIZED,
                limitations=_limitations(identity.synthetic, resolution, chain.names)
                + report.limitations,
            )
            if reused:
                if inspection.finalized_manifest_digest != out.manifest_digest:
                    raise EmbeddingStoreIncompatibleError(
                        "the store was finalized with different provenance; it is not overwritten"
                    )
            else:
                store.finalize(out)
            reference = store.reference
        finally:
            store.close()
        metrics = EmbeddingMetrics(
            patches=len(records),
            patches_encoded=counters.patches_encoded if counters else 0,
            patches_resumed=counters.patches_resumed if counters else len(records),
            batches_encoded=counters.batches_encoded if counters else 0,
            chunks_committed=counters.chunks_committed if counters else 0,
            chunks_resumed=counters.chunks_resumed if counters else header.layout.chunk_count,
            region_reads=source.reads,
            partially_acquired_reads=source.partially_acquired,
            max_live_regions=source.max_live,
            budget=budget,
            reused_finalized_store=reused,
            encoder_resources=report.resources,
        )
        return EmbeddingBagPayload(out, reference, metrics)


def _limitations(
    synthetic: bool, resolution: ResolutionRecord, transforms: tuple[str, ...]
) -> tuple[str, ...]:
    out: list[str] = []
    if synthetic:
        out.append(
            "synthetic reference encoder: embeddings are an architectural test fixture with no "
            "biological or clinical meaning"
        )
    else:
        out.append(
            "pretrained encoder: successful inference establishes technical integration only, "
            "not clinical validity, biomarker accuracy or diagnostic use"
        )
    if resolution.extraction_spacing_um is None:
        out.append("physical spacing unknown: field of view is defined in pixels only")
    if resolution.spacing_relative_error:
        out.append(
            f"extraction spacing differs from the requested {resolution.requested_spacing_um} "
            f"µm/px by {resolution.spacing_relative_error:.2%} (no resampling performed)"
        )
    if "drop_alpha" in transforms:
        out.append("alpha dropped by declared transform; validity was taken from the patch plan")
    return tuple(out)
