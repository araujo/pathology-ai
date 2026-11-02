"""`pathology-ai-demo`: run the synthetic projects and print a JSON summary.

Composition entry point (may import every layer). Output is synthetic and
has no clinical meaning.

Pipeline and project definitions are not packaged in the wheel. From a source
checkout they are found next to this module; an installed CLI needs
`--root PATH` or `$PATHOLOGY_AI_ROOT` (see `bootstrap.resolve_definitions_root`).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from application.mil_training import EvaluationOutcome, TrainingOutcome
from application.runner import ExecutionRequest
from bootstrap import (
    ROOT_ENV_VAR,
    DefinitionsRootError,
    MILServices,
    build_imaging_service,
    build_mil,
    build_platform,
)
from domain.cohort import CohortManifest
from domain.errors import DomainError, ExecutionUnsupportedError, MILDatasetError
from domain.hashing import canonical_json, to_canonical
from domain.ids import ImageId, PipelineId, Version
from domain.metrics import ClassificationReport
from domain.mil import MILDatasetManifest, Partition, SplitManifest
from domain.mil_training import ExperimentSpec, PredictionManifest
from domain.modality import KNOWN_MODALITIES
from domain.pipeline import ConfigValue, PipelineRef, ResourceBudget
from domain.reading import RegionReadRequest
from infrastructure.mil_config import MILExperimentConfig
from plugins.models.checkpoints import CHECKPOINTS_ENV, parse_bindings
from plugins.synthetic import catalog
from plugins.synthetic.mil import PLUGIN_VERSION as PLUGIN_VERSION_MIL
from ports import ImageSource
from presentation.api import analysis_v1, imaging_v1, preprocessing_v1

PREPROCESS_PIPELINE = PipelineRef(PipelineId("wsi-tissue-patch-extraction"), Version("1.0.0"))
EMBED_PIPELINE = PipelineRef(PipelineId("wsi-patch-embedding"), Version("1.0.0"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pathology-ai-demo", description="Run the synthetic projects (not medical)."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help=f"directory containing pipelines/ and projects/ (default: ${ROOT_ENV_VAR}, "
        "then the source checkout)",
    )
    args = parser.parse_args(argv)
    try:
        platform = build_platform(args.root)
    except DefinitionsRootError as exc:
        parser.exit(2, f"pathology-ai-demo: error: {exc}\n")
    summary: dict[str, object] = {"notice": "SYNTHETIC architecture demonstration; not medical"}
    projects = []
    for namespace, project in sorted(platform.project_definitions.items()):
        image_id = platform.datasets.version(project.dataset).split("test")[0]
        outcome = platform.projects.run_image(project, image_id)
        assert outcome.prediction is not None
        result = analysis_v1.analysis_result(outcome.prediction, outcome.run)
        projects.append(
            {
                "namespace": namespace,
                "project_id": project.project_id.value,
                "pipeline": str(project.pipeline),
                "image_id": image_id.value,
                "capability_status": result["prediction"]["capability_status"],
                "outputs": result["prediction"]["outputs"],
                "evidence_count": len(result["prediction"]["evidence"]),
                "fingerprint": result["provenance"]["fingerprint"],
            }
        )
    summary["projects"] = projects

    volume_request = ExecutionRequest(
        pipeline=platform.pipelines.get(
            PipelineRef(PipelineId("volume-segmentation-3d"), Version("1.0.0"))
        ),
        image=catalog.synthetic_volume("syn-ct-0001"),
        task=catalog.VOLUME_SEG_TASK,
        models={"segmenter_3d": catalog.VOLUME_SEGMENTER.ref},
        preprocessing=catalog.VOLUME_CONTRACT,
        research_use_acknowledged=True,
        budget=ResourceBudget(gpu_count=1, memory_mb=16384),
    )
    try:
        platform.runner.run(volume_request)
        summary["volumetric"] = "unexpectedly executed"
    except ExecutionUnsupportedError as exc:
        summary["volumetric"] = {"status": "EXECUTION_UNSUPPORTED", "missing": list(exc.missing)}
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def _region(text: str) -> RegionReadRequest:
    try:
        level, x, y, w, h = (int(v) for v in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("expected LEVEL,X,Y,WIDTH,HEIGHT") from None
    return RegionReadRequest(level, x, y, w, h)


def imaging_smoke_main(argv: Sequence[str] | None = None) -> int:
    """`pathology-ai-imaging-smoke`: ingest bounded regions of a real image (no model).

    Prints metadata and read provenance as JSON; never pixels and never the file path.
    """
    parser = argparse.ArgumentParser(
        prog="pathology-ai-imaging-smoke",
        description="Read metadata and bounded regions from a real image. No inference.",
    )
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument(
        "--modality",
        required=True,
        choices=sorted(KNOWN_MODALITIES),
        help="declared imaging modality (never inferred from the file format)",
    )
    parser.add_argument("--scene", type=int, default=None)
    parser.add_argument("--backend", default=None, help="force a registered backend by name")
    parser.add_argument(
        "--region", type=_region, action="append", required=True, help="LEVEL,X,Y,WIDTH,HEIGHT"
    )
    args = parser.parse_args(argv)
    service = build_imaging_service()
    source = ImageSource(ImageId(args.image_id), args.file, KNOWN_MODALITIES[args.modality])
    try:
        outcome = service.ingest(source, args.region, scene=args.scene, backend=args.backend)
    except DomainError as exc:
        parser.exit(3, f"pathology-ai-imaging-smoke: {exc.code}: {exc}\n")
    reader = service.backends.get(outcome.record.reader.name).capabilities
    report = {
        "notice": outcome.record.usage,
        "image_descriptor": imaging_v1.image_descriptor(outcome.description.asset, reader),
        "reader_diagnostics": imaging_v1.reader_diagnostics(outcome.description),
        "regions": [
            {
                **imaging_v1.region_summary(region),
                "artifact_id": value.identity.artifact_id.value,
            }
            for region, value in zip(outcome.regions, outcome.values, strict=True)
        ],
        "ingestion_record_id": outcome.record.record_id,
    }
    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def _override(text: str) -> tuple[str, str, ConfigValue]:
    target, sep, raw = text.partition("=")
    stage, dot, key = target.partition(".")
    if not (sep and dot and stage and key):
        raise argparse.ArgumentTypeError("expected STAGE.KEY=VALUE")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    if not isinstance(value, str | int | float | bool):
        raise argparse.ArgumentTypeError("override values must be scalars")
    return stage, key, value


def preprocess_main(argv: Sequence[str] | None = None) -> int:
    """`pathology-ai-preprocess`: candidate tissue + patch extraction on a real WSI (no model).

    Runs `wsi-tissue-patch-extraction` through the pipeline engine and prints a JSON summary
    (never pixels, never the file path). `--manifest-out` writes the canonical manifest.
    """
    parser = argparse.ArgumentParser(
        prog="pathology-ai-preprocess",
        description="Candidate-tissue detection and patch extraction on a real WSI. No inference.",
    )
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--modality", default="wsi", choices=sorted(KNOWN_MODALITIES))
    parser.add_argument("--scene", type=int, default=None)
    parser.add_argument("--backend", default=None, help="force a registered backend by name")
    parser.add_argument("--root", type=Path, default=None, help="directory with pipelines/")
    parser.add_argument(
        "--override", type=_override, action="append", default=[], help="STAGE.KEY=VALUE"
    )
    parser.add_argument("--manifest-out", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        platform = build_platform(args.root)
    except DefinitionsRootError as exc:
        parser.exit(2, f"pathology-ai-preprocess: error: {exc}\n")
    source = ImageSource(ImageId(args.image_id), args.file, KNOWN_MODALITIES[args.modality])
    try:
        outcome = platform.preprocessing_runs.run(
            source,
            platform.pipelines.get(PREPROCESS_PIPELINE),
            scene=args.scene,
            backend=args.backend,
            overrides=args.override,
        )
    except DomainError as exc:
        parser.exit(3, f"pathology-ai-preprocess: {exc.code}: {exc}\n")
    manifest = outcome.manifest
    if args.manifest_out is not None:
        args.manifest_out.write_text(preprocessing_v1.manifest_to_json(manifest) + "\n")
    run = outcome.execution.run
    report = {
        "notice": manifest.notice,
        "pipeline": str(PREPROCESS_PIPELINE),
        "run": {"status": run.status.value, "usage_mode": run.usage_mode.value},
        "image_id": manifest.image_id.value,
        "reader": f"{manifest.reader.name} {manifest.reader.version}",
        "mask": {
            "grid": [outcome.mask.grid.width, outcome.mask.grid.height],
            "cell_level0_px": [outcome.mask.grid.cell_x, outcome.mask.grid.cell_y],
            "tissue_fraction": outcome.mask.tissue_fraction,
            "acquired_fraction": outcome.mask.acquired_fraction,
            "derived": dict(outcome.mask.derived),
        },
        "resolved": to_canonical(manifest.resolved),
        "counts": dict(manifest.counts),
        "manifest_digest": manifest.manifest_digest,
        "provenance_digest": manifest.provenance_digest,
        "plan_digest": manifest.plan_digest,
        "first_patches": [
            {"patch_id": r.patch_id, "level0": to_canonical(r.level0)} for r in manifest.patches[:3]
        ],
    }
    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def embed_main(argv: Sequence[str] | None = None) -> int:
    """`pathology-ai-embed`: streaming patch embedding of a real WSI into a durable store.

    Runs `wsi-patch-embedding` through the pipeline engine and prints a JSON summary (never pixels,
    embeddings or the input path). The store is written below `--output-dir`; an interrupted run
    resumes from verified chunks when repeated with the same arguments. `--manifest-out` writes
    the canonical embedding manifest. The reference encoder is synthetic: no medical meaning.
    """
    parser = argparse.ArgumentParser(
        prog="pathology-ai-embed",
        description="Streaming patch embedding of a real WSI into a durable store. No prediction.",
    )
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--modality", default="wsi", choices=sorted(KNOWN_MODALITIES))
    parser.add_argument("--scene", type=int, default=None)
    parser.add_argument("--backend", default=None, help="force a registered backend by name")
    parser.add_argument("--root", type=Path, default=None, help="directory with pipelines/")
    parser.add_argument(
        "--override", type=_override, action="append", default=[], help="STAGE.KEY=VALUE"
    )
    parser.add_argument("--manifest-out", type=Path, default=None)
    parser.add_argument(
        "--pipeline",
        default=f"{EMBED_PIPELINE.pipeline_id}@{EMBED_PIPELINE.version}",
        help="pipeline ID@VERSION (e.g. wsi-patch-embedding-uni@1.0.0)",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help=f"bind an encoder to a local checkpoint file (also ${CHECKPOINTS_ENV}); "
        "weights are never downloaded",
    )
    args = parser.parse_args(argv)
    pid, _, version = str(args.pipeline).partition("@")
    try:
        pipeline_ref = PipelineRef(PipelineId(pid), Version(version or "1.0.0"))
        platform = build_platform(args.root, checkpoints=parse_bindings(args.checkpoint))
    except DefinitionsRootError as exc:
        parser.exit(2, f"pathology-ai-embed: error: {exc}\n")
    except DomainError as exc:
        parser.exit(2, f"pathology-ai-embed: error: {exc.code}: {exc}\n")
    source = ImageSource(ImageId(args.image_id), args.file, KNOWN_MODALITIES[args.modality])
    try:
        outcome = platform.embedding_runs.run(
            source,
            platform.pipelines.get(pipeline_ref),
            args.output_dir,
            scene=args.scene,
            backend=args.backend,
            overrides=args.override,
        )
    except DomainError as exc:
        parser.exit(3, f"pathology-ai-embed: {exc.code}: {exc}\n")
    m, metrics = outcome.manifest, outcome.payload.metrics
    if args.manifest_out is not None:
        args.manifest_out.write_text(canonical_json(m) + "\n")
    run = outcome.execution.run
    report = {
        "notice": m.notice,
        "pipeline": str(pipeline_ref),
        "run": {"status": run.status.value, "usage_mode": run.usage_mode.value},
        "image_id": m.image_id.value,
        "reader": f"{m.reader.name} {m.reader.version}",
        "patch_counts": dict(outcome.patch_manifest.counts),
        "plan_digest": m.plan_digest,
        "patch_manifest_digest": m.patch_manifest_digest,
        "encoder": to_canonical(m.encoder),
        "transforms": list(m.transforms.names),
        "resolution": to_canonical(m.resolution),
        "embeddings": {
            "rows": m.row_count,
            "dim": m.embedding_dim,
            "dtype": m.embedding_dtype,
            "store_format": m.store.format,
            "store_id": m.store.store_id,
            "chunks": m.store.chunk_count,
            "chunk_rows": m.store.chunk_rows,
        },
        "run_identity": m.run_identity,
        "content_digest": m.content_digest,
        "embedding_identity": m.embedding_identity,
        "manifest_digest": m.manifest_digest,
        "limitations": list(m.limitations),
        "metrics": to_canonical(metrics),
        "first_rows": [
            {"row": r.row, "patch_id": r.patch_id, "level0": to_canonical(r.level0)}
            for r in m.records[:3]
        ],
    }
    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


# --------------------------------------------------------------------------- MIL (SPEC-007)

MIL_NOTICE = (
    "research use only: slide-level MIL results are technical evidence; metrics on synthetic "
    "data are not clinical or biomarker performance"
)


def synthetic_experiment_yaml(kind: str, labels: str = "labels.csv") -> str:
    """The example experiment configuration of a synthetic cohort (kept in sync with the
    checked-in `projects/experiments/mil-synthetic-*.yaml`)."""
    if kind == "binary":
        task = (
            "  id: synthetic-binary-target\n"
            '  version: "1"\n'
            "  type: binary_classification\n"
            "  classes: [class-a, class-b]\n"
            "  positive_class: class-b\n"
        )
        model = (
            "  architecture: attention-mil\n"
            "  hyperparameters: {attention_dim: 16, dropout: 0.0, gated: true, hidden_dim: 32}\n"
        )
    else:
        task = (
            "  id: synthetic-multiclass-target\n"
            '  version: "1"\n'
            "  type: multiclass_classification\n"
            "  classes: [class-a, class-b, class-c]\n"
        )
        model = "  architecture: mean-pooling\n  hyperparameters: {dropout: 0.0}\n"
    return (
        "# SYNTHETIC example: generated classes with an injected signal; no clinical meaning.\n"
        "schema: pathology-ai.mil-experiment-config/1\n"
        f"name: synthetic-{kind}\n"
        f"task:\n{task}"
        "dataset:\n"
        f"  id: synthetic-{kind}\n"
        '  version: "1"\n'
        f"  labels: {labels}\n"
        "  label_source: synthetic-generator\n"
        f'  label_source_version: "{PLUGIN_VERSION_MIL}"\n'
        "  compatibility_policy: exact@1\n"
        "split:\n"
        "  strategy: group-stratified@1\n"
        "  seed: 7\n"
        "  fractions: {train: 0.6, validation: 0.2, test: 0.2}\n"
        f"model:\n{model}"
        "training:\n"
        "  seed: 13\n"
        "  epochs: 40\n"
        "  batch_slides: 4\n"
        "  optimizer: {name: adam, learning_rate: 0.005, weight_decay: 0.0001}\n"
        "  gradient_clip_norm: 5.0\n"
        "  class_weighting: false\n"
        "  early_stopping: {metric: validation_loss, mode: min, patience: 8, min_delta: 0.0}\n"
        "  device: cpu\n"
        "  bags:\n"
        "    train: {max_instances: 48, seed: 13, max_batch_bytes: 67108864}\n"
        "    eval: {max_instances: null, seed: 0, max_batch_bytes: 67108864}\n"
    )


def _mil_synthetic(args: argparse.Namespace) -> dict[str, object]:
    from infrastructure.embedding_store import NumpyEmbeddingStoreProvider
    from infrastructure.mil_config import write_labels
    from plugins.synthetic.mil import synthetic_cohort, write_bag

    out = Path(args.out)
    binary = args.task == "binary"
    classes = ("class-a", "class-b") if binary else ("class-a", "class-b", "class-c")
    cohort = synthetic_cohort(
        classes, binary=binary, groups_per_class=args.groups_per_class, seed=args.seed
    )
    provider = NumpyEmbeddingStoreProvider(out / "stores")
    rows = []
    for slide in cohort:
        store_id = write_bag(provider, slide.slide_id, slide.rows)
        rows.append(
            {
                "slide_id": slide.slide_id,
                "store_id": store_id,
                "label": slide.label,
                "group_id": slide.group_id,
                "status": "eligible",
            }
        )
    write_labels(out / "labels.csv", rows)
    (out / "experiment.yaml").write_text(synthetic_experiment_yaml(args.task), encoding="utf-8")
    return {
        "notice": "SYNTHETIC cohort: seeded noise with an injected class signal",
        "slides": len(cohort),
        "groups": len({s.group_id for s in cohort}),
        "stores": str(out / "stores"),
        "labels": str(out / "labels.csv"),
        "config": str(out / "experiment.yaml"),
    }


def _mil_dataset(services: MILServices, cfg: MILExperimentConfig) -> MILDatasetManifest:
    from infrastructure.mil_config import load_labels

    labels, stores = load_labels(cfg.labels, cfg.task, cfg.label_source, cfg.label_source_version)
    return services.datasets.build(
        dataset_id=cfg.dataset_id,
        version=cfg.dataset_version,
        task=cfg.task,
        policy=cfg.policy,
        labels=labels,
        stores=stores,
        accepted_key=cfg.accepted_key,
    )


def _mil_split(
    services: MILServices, cfg: MILExperimentConfig, dataset: MILDatasetManifest
) -> SplitManifest:
    from infrastructure.mil_config import load_split_table

    s = cfg.split
    if s.file is not None:
        return services.datasets.import_split(dataset, load_split_table(s.file))
    assert s.fractions is not None and s.seed is not None
    return services.datasets.split(dataset, s.fractions, s.seed)


def _report_json(report: ClassificationReport) -> object:
    return to_canonical(report)


def _dataset_summary(d: MILDatasetManifest) -> dict[str, object]:
    return {
        "ref": d.ref,
        "digest": d.digest,
        "slides": len(d.records),
        "status_counts": d.counts(),
        "compatibility_policy": d.compatibility_policy,
        "compatibility_key": d.compatibility_key,
        "feature_dim": d.feature_dim,
    }


def _split_summary(s: SplitManifest) -> dict[str, object]:
    return {
        "split_id": s.split_id,
        "digest": s.digest,
        "strategy": s.strategy,
        "partitions": {
            p.partition.value: {
                "slides": len(p.slide_ids),
                "groups": p.group_count,
                "classes": dict(p.class_counts),
            }
            for p in s.partitions
        },
        "group_overlap": list(s.group_overlap),
    }


def _training_summary(outcome: TrainingOutcome) -> dict[str, object]:
    c = outcome.checkpoint
    h = c.history
    return {
        "experiment_identity": c.experiment_identity,
        "checkpoint_id": c.checkpoint_id,
        "checkpoint_digest": c.digest,
        "weights_sha256": c.weights_sha256,
        "reused": outcome.reused,
        "model": c.experiment.model.ref,
        "parameters": c.parameter_count,
        "runtime": dict(c.experiment.runtime),
        "epochs_run": len(h.epochs),
        "best_epoch": h.best_epoch,
        "selection_metric": h.selection_metric,
        "best_value": h.best_value,
        "stopped_early": h.stopped_early,
        "measurements": None
        if outcome.measurements is None
        else to_canonical(outcome.measurements),
    }


def mil_main(argv: Sequence[str] | None = None) -> int:
    """`pathology-ai-mil`: labelled datasets, group-aware splits, MIL training, evaluation and
    slide-level prediction over finalized embedding stores. Prints JSON summaries."""
    parser = argparse.ArgumentParser(
        prog="pathology-ai-mil",
        description="MIL slide-level classification over finalized embedding bags. Research use.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def stores(p: argparse.ArgumentParser) -> None:
        p.add_argument("--store-root", type=Path, action="append", required=True)

    syn = sub.add_parser("synthetic", help="write a SYNTHETIC labelled cohort of stores")
    syn.add_argument("--out", type=Path, required=True)
    syn.add_argument("--task", choices=("binary", "multiclass"), default="binary")
    syn.add_argument("--groups-per-class", type=int, default=30)
    syn.add_argument("--seed", type=int, default=0)

    ds = sub.add_parser("dataset", help="build and write a dataset manifest")
    ds.add_argument("--config", type=Path, required=True)
    ds.add_argument("--out", type=Path, required=True)
    stores(ds)

    sp = sub.add_parser("split", help="generate or import, then write, a split manifest")
    sp.add_argument("--config", type=Path, required=True)
    sp.add_argument("--dataset", type=Path, required=True)
    sp.add_argument("--out", type=Path, required=True)

    tr = sub.add_parser("train", help="train and persist the validation-selected checkpoint")
    tr.add_argument("--config", type=Path, required=True)
    tr.add_argument("--dataset", type=Path, required=True)
    tr.add_argument("--split", type=Path, required=True)
    tr.add_argument("--checkpoints", type=Path, required=True)
    stores(tr)

    ev = sub.add_parser("evaluate", help="metrics and predictions on one partition")
    ev.add_argument("--checkpoints", type=Path, required=True)
    ev.add_argument("--experiment", required=True)
    ev.add_argument("--dataset", type=Path, required=True)
    ev.add_argument("--split", type=Path, required=True)
    ev.add_argument("--partition", choices=[p.value for p in Partition], required=True)
    ev.add_argument("--out", type=Path, required=True)
    stores(ev)

    pr = sub.add_parser("predict", help="slide-level inference on compatible bags")
    pr.add_argument("--checkpoints", type=Path, required=True)
    pr.add_argument("--experiment", required=True)
    pr.add_argument("--slides", type=Path, required=True, help="CSV with slide_id,store_id")
    pr.add_argument("--out", type=Path, required=True)
    stores(pr)

    run = sub.add_parser("run", help="dataset, split, train, then evaluate validation and test")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    stores(run)

    args = parser.parse_args(argv)
    try:
        report = _mil_command(args)
    except DomainError as exc:
        parser.exit(3, f"pathology-ai-mil: {exc.code}: {exc}\n")
    json.dump({"notice": MIL_NOTICE, **report}, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def _mil_command(args: argparse.Namespace) -> dict[str, object]:
    from infrastructure.mil_artifacts import read_dataset, read_split, write_once
    from infrastructure.mil_config import load_experiment_config

    if args.command == "synthetic":
        return _mil_synthetic(args)
    checkpoints = getattr(args, "checkpoints", None) or Path(getattr(args, "out", "."))
    services = build_mil(getattr(args, "store_root", None) or [Path(".")], checkpoints)
    if args.command == "dataset":
        cfg = load_experiment_config(args.config)
        dataset = _mil_dataset(services, cfg)
        write_once(args.out, dataset)
        return {"dataset": _dataset_summary(dataset)}
    if args.command == "split":
        cfg = load_experiment_config(args.config)
        dataset = read_dataset(args.dataset)
        split = _mil_split(services, cfg, dataset)
        write_once(args.out, split)
        return {"split": _split_summary(split)}
    if args.command == "train":
        cfg = load_experiment_config(args.config)
        dataset, split = read_dataset(args.dataset), read_split(args.split)
        exp = _mil_experiment(services, cfg, dataset, split)
        return {"training": _training_summary(services.training.train(exp, dataset, split))}
    if args.command == "evaluate":
        dataset, split = read_dataset(args.dataset), read_split(args.split)
        outcome = services.training.evaluate(
            args.experiment, dataset, split, Partition(args.partition)
        )
        write_once(args.out, outcome.predictions)
        return _evaluation_summary(outcome)
    if args.command == "predict":
        slides = _slide_table(args.slides)
        manifest = services.training.predict(args.experiment, slides)
        write_once(args.out, manifest)
        return {
            "predictions": len(manifest.predictions),
            "content_digest": manifest.content_digest,
            "classes": list(manifest.classes),
            "counts": _predicted_counts(manifest),
        }
    # run
    cfg = load_experiment_config(args.config)
    out = Path(args.out)
    services = build_mil(args.store_root, out / "checkpoints")
    dataset = _mil_dataset(services, cfg)
    write_once(out / "dataset.json", dataset)
    split = _mil_split(services, cfg, dataset)
    write_once(out / "split.json", split)
    exp = _mil_experiment(services, cfg, dataset, split)
    training = services.training.train(exp, dataset, split)
    result: dict[str, object] = {
        "dataset": _dataset_summary(dataset),
        "split": _split_summary(split),
        "training": _training_summary(training),
    }
    for partition in (Partition.VALIDATION, Partition.TEST):
        if not split.members(partition):
            continue
        outcome = services.training.evaluate(exp.identity, dataset, split, partition)
        write_once(out / f"predictions-{partition.value}.json", outcome.predictions)
        result[partition.value] = _evaluation_summary(outcome)
    return result


def _mil_experiment(
    services: MILServices,
    cfg: MILExperimentConfig,
    dataset: MILDatasetManifest,
    split: SplitManifest,
) -> ExperimentSpec:
    return services.training.experiment(
        name=cfg.name,
        dataset=dataset,
        split=split,
        architecture=cfg.architecture,
        hyperparameters=cfg.hyperparameters,
        training=cfg.training,
        accepted_key=cfg.accepted_key,
    )


def _predicted_counts(manifest: PredictionManifest) -> dict[str, int]:
    counts = dict.fromkeys(manifest.classes, 0)
    for p in manifest.predictions:
        counts[p.predicted_class] += 1
    return counts


def _evaluation_summary(outcome: EvaluationOutcome) -> dict[str, object]:
    m = outcome.predictions
    return {
        "partition": None if m.partition is None else m.partition.value,
        "slides": len(m.predictions),
        "loss": outcome.loss,
        "metrics": _report_json(outcome.report),
        "predictions_digest": m.content_digest,
        "checkpoint_id": m.checkpoint_id,
        "measurements": to_canonical(outcome.measurements),
    }


def _slide_table(path: Path) -> list[tuple[str, str]]:
    import csv

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not {"slide_id", "store_id"} <= set(reader.fieldnames or ()):
            raise MILDatasetError(f"{path.name} needs slide_id and store_id columns")
        return [(row["slide_id"].strip(), row["store_id"].strip()) for row in reader]


# --------------------------------------------------------------------------- cohorts (SPEC-008)

COHORT_NOTICE = (
    "Cohort reconciliation only: no image was read, no model was run. Eligibility is "
    "provisional while image integrity gates are open. No clinical claim is made."
)
INVENTORY_NOTICE = (
    "Image inventory: files were read and hashed, never modified, decoded or uploaded. A scoped "
    "inventory never closes cohort-wide integrity gates. No model was run."
)
PILOT_NOTICE = (
    "Pilot selection only: no image was read or downloaded, no model was run. Pending decisions "
    "gate any pilot execution."
)


def _root_binding(text: str) -> tuple[str, Path]:
    name, sep, path = text.partition("=")
    if not sep or not name or not path:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    return name, Path(path).expanduser()


def _cohort_summary(manifest: CohortManifest) -> dict[str, object]:
    from collections import Counter

    from application.cohort_service import split_assignment

    records = manifest.records
    status = Counter(r.status.value for r in records)
    reasons = Counter(f"{r.status.value}:{r.reason_code}" for r in records if r.reason_code)
    provisional = sum(r.provisional for r in records)
    assignment = split_assignment(manifest, allow_provisional=True)
    return {
        "cohort": manifest.ref,
        "manifest_id": manifest.manifest_id,
        "manifest_digest": manifest.digest,
        "records": len(records),
        "groups": len({r.group_id for r in records}),
        "status": dict(sorted(status.items())),
        "reasons": dict(sorted(reasons.items())),
        "provisional": provisional,
        "split": {
            "strategy": manifest.split.strategy,
            "test_role": manifest.split.test_role,
            "assignment_digest": manifest.split.assignment_digest,
            "slides": len(assignment),
            "partitions": to_canonical(manifest.split.partitions),
        },
        "sources": [
            {"source": s.source_id, "file": s.file_name, "sha256": s.sha256, "rows": s.rows}
            for s in manifest.sources
        ],
        "pending_decisions": [
            d.decision_id for d in manifest.decisions if d.status.value == "pending"
        ],
    }


def cohort_main(argv: Sequence[str] | None = None) -> int:
    """`pathology-ai-cohort`: reconcile declared label sources into a versioned, write-once
    cohort manifest, or verify an existing one. Prints JSON summaries."""
    parser = argparse.ArgumentParser(
        prog="pathology-ai-cohort",
        description="Cohort label reconciliation and manifests (SPEC-008). Research use.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("reconcile", help="read sources, reconcile, write the manifest once")
    rec.add_argument("--config", type=Path, required=True)
    rec.add_argument(
        "--source-root", type=_root_binding, action="append", default=[], metavar="NAME=PATH"
    )
    rec.add_argument("--resolutions", type=Path, help="owner-approved conflict resolutions")
    rec.add_argument("--out", type=Path, help="directory for manifest, split and reconciliation")
    ver = sub.add_parser("verify", help="re-read a manifest and re-derive every summary")
    ver.add_argument("--manifest", type=Path, required=True)
    inv = sub.add_parser(
        "inventory", help="hash image files under named roots (read-only) and map them to slides"
    )
    inv.add_argument("--manifest", type=Path, required=True)
    inv.add_argument("--root", type=_root_binding, action="append", default=[], metavar="NAME=PATH")
    inv.add_argument(
        "--listing",
        type=_root_binding,
        action="append",
        default=[],
        metavar="NAME=FILE",
        help="saved object-store listing (list-objects-v2 JSON); metadata only, never hashed",
    )
    inv.add_argument("--pattern", action="append", required=True, help="file name glob")
    inv.add_argument("--out", type=Path, help="directory; writes <inventory id>/ once")
    pil = sub.add_parser("pilot", help="deterministic pilot selection from approved partitions")
    pil.add_argument("--manifest", type=Path, required=True)
    pil.add_argument("--criteria", type=Path, required=True)
    pil.add_argument("--inventory", type=Path, help="inventory taken against the same manifest")
    pil.add_argument("--out", type=Path, help="directory; writes <selection id>/ once")
    args = parser.parse_args(argv)
    try:
        report = _cohort_command(args)
    except DomainError as exc:
        parser.exit(3, f"pathology-ai-cohort: {exc.code}: {exc}\n")
    notice = {"inventory": INVENTORY_NOTICE, "pilot": PILOT_NOTICE}.get(args.command, COHORT_NOTICE)
    json.dump({"notice": notice, **report}, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def _cohort_command(args: argparse.Namespace) -> dict[str, object]:
    from application.cohort_service import CohortService
    from infrastructure.cohort_sources import (
        CsvCohortSourceReader,
        load_cohort_config,
        load_resolutions,
        read_manifest,
        write_cohort_outputs,
    )

    if args.command == "verify":
        return _cohort_summary(read_manifest(args.manifest))
    if args.command == "inventory":
        return _inventory_command(args, read_manifest(args.manifest))
    if args.command == "pilot":
        return _pilot_command(args, read_manifest(args.manifest))
    config = load_cohort_config(args.config)
    reader = CsvCohortSourceReader(config, dict(args.source_root))
    resolutions = load_resolutions(args.resolutions) if args.resolutions else ()
    manifest = CohortService(reader).reconcile(config.definition, resolutions=resolutions)
    summary = _cohort_summary(manifest)
    if args.out is not None:
        summary["written"] = write_cohort_outputs(args.out, manifest)
    return summary


def _inventory_command(args: argparse.Namespace, manifest: CohortManifest) -> dict[str, object]:
    from application.wsi_inventory_service import WSIInventoryService, scoped_summary
    from infrastructure.wsi_inventory import (
        LocalWSIObjectSource,
        ObjectListingSource,
        write_inventory,
    )
    from ports import WSIObjectSource

    def progress(uri: str, size: int) -> None:
        sys.stderr.write(f"hashing {uri} ({size / 1e9:.2f} GB)\n")

    sources: list[WSIObjectSource] = []
    if args.root:
        sources.append(LocalWSIObjectSource(dict(args.root), args.pattern, progress=progress))
    for name, listing in args.listing:
        sources.append(
            ObjectListingSource(
                name, "remote object listing (metadata only)", listing, args.pattern
            )
        )
    inventory = WSIInventoryService(sources).inventory(manifest)
    summary = scoped_summary(inventory, manifest)
    summary["skipped_symlinks"] = sum(getattr(s, "skipped_links", 0) for s in sources)
    if args.out is not None:
        summary["written"] = str(write_inventory(args.out, inventory))
    return summary


def _pilot_command(args: argparse.Namespace, manifest: CohortManifest) -> dict[str, object]:
    from domain.pilot import select_pilot
    from infrastructure.wsi_inventory import load_pilot_criteria, read_inventory, write_pilot

    inventory = read_inventory(args.inventory) if args.inventory else None
    selection = select_pilot(manifest, load_pilot_criteria(args.criteria), inventory)
    summary: dict[str, object] = {
        "selection_id": selection.selection_id,
        "selection_digest": selection.digest,
        **to_canonical(selection),
    }
    if args.out is not None:
        summary["written"] = str(write_pilot(args.out, selection))
    return summary


if __name__ == "__main__":
    raise SystemExit(main())
