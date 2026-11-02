"""Declarative cohort configuration and CSV source reading (SPEC-008 §6-§7).

A cohort is declared in YAML (`pathology-ai.cohort-config/1`): its target, the identifier
grammar, the owner decisions, and every source file with its pinned SHA-256. Unknown keys are
errors, so a misspelt option never falls back silently to a default.

Identifiers are parsed by one regular expression with named fields, then re-rendered through
templates, so differently formatted spellings of one identifier (e.g. unpadded numbers or a file
extension) normalise to the same slide and group ids. Rules enforced at load time:

* the slide template uses every group field plus the slide field;
* the group template uses exactly the declared group fields (so a group key is always qualified
  by the sub-cohort when one is declared) and never the slide field;
* a label source reads its identifier from one declared column; any other column (for example
  a positional index that merely looks like an identifier) is never read.

Source files are bound to named roots at run time (`roots`), in the same way checkpoints are
bound: paths never enter a manifest, digests do. A file whose bytes differ from its pin fails
with `COHORT_SOURCE_INTEGRITY` before any row is parsed.

Outputs: the manifest is a write-once JSON document (`{"body", "digest"}`); the split table uses
the SPEC-007 `slide_id,partition` format; the reconciliation table lists one row per slide.
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
import string
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from domain.cohort import (
    CohortDefinition,
    CohortManifest,
    CohortObservations,
    CohortTarget,
    Decision,
    DecisionStatus,
    LabelCandidate,
    Resolution,
    RosterEntry,
    SlidePolicy,
    SourceFile,
    SourceRole,
    SplitPolicy,
    split_assignment,
)
from domain.errors import (
    CohortError,
    CohortSourceFormatError,
    CohortSourceIntegrityError,
    ExperimentConflictError,
)
from domain.mil import PARTITIONS
from infrastructure.config import ConfigurationError
from infrastructure.mil_artifacts import _atomic_write, read_document, write_once

CONFIG_SCHEMA = "pathology-ai.cohort-config/1"
MANIFEST_FILE = "cohort-manifest.json"
SPLIT_FILE = "split.csv"
RECONCILIATION_FILE = "reconciliation.csv"


def _keys(
    doc: Any, where: str, required: set[str], optional: set[str] | None = None
) -> dict[str, Any]:
    if not isinstance(doc, Mapping):
        raise ConfigurationError(f"{where} must be a mapping")
    keys = set(doc)
    missing = sorted(required - keys)
    unknown = sorted(keys - required - (optional or set()))
    if missing or unknown:
        raise ConfigurationError(
            f"{where}: missing {missing or 'nothing'}, unknown {unknown or 'nothing'}"
        )
    return dict(doc)


def _str(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigurationError(f"{where} must be a non-empty string (quote versions in YAML)")
    return value


def _strs(value: Any, where: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
        raise ConfigurationError(f"{where} must be a list of strings")
    return tuple(value)


def _fields(template: str) -> set[str]:
    out = set()
    for _, name, _, _ in string.Formatter().parse(template):
        if name is not None:
            if not name.isidentifier():
                raise ConfigurationError(f"template field {name!r} must be a plain name")
            out.add(name)
    return out


# --------------------------------------------------------------------------- identity


@dataclass(frozen=True)
class IdentityGrammar:
    pattern: re.Pattern[str]
    integer_fields: tuple[str, ...]
    cohort_field: str
    slide_field: str
    group_fields: tuple[str, ...]
    slide_template: str
    group_template: str

    def __post_init__(self) -> None:
        names = set(self.pattern.groupindex)
        declared = {self.cohort_field, self.slide_field, *self.group_fields, *self.integer_fields}
        if declared - names:
            raise ConfigurationError(
                f"identity fields {sorted(declared - names)} are not named groups of the pattern"
            )
        if self.cohort_field not in self.group_fields:
            raise ConfigurationError("identity.group_fields must include the cohort field")
        if self.slide_field in self.group_fields:
            raise ConfigurationError("the slide field cannot be part of the group key")
        if _fields(self.group_template) != set(self.group_fields):
            raise ConfigurationError(
                f"identity.group_id must use exactly the group fields {list(self.group_fields)}"
            )
        if _fields(self.slide_template) != {*self.group_fields, self.slide_field}:
            raise ConfigurationError(
                "identity.slide_id must use every group field and the slide field"
            )

    def parse(self, raw: str, where: str) -> tuple[str, str, str | None, str | None]:
        """(cohort, group id, slide id or None, slide index or None) of one raw identifier."""
        m = self.pattern.fullmatch(raw)
        if m is None:
            raise CohortSourceFormatError(f"{where}: identifier {raw!r} does not match the grammar")
        values: dict[str, Any] = dict(m.groupdict())
        for name in self.integer_fields:
            if values.get(name) is not None:
                values[name] = int(values[name])
        missing = [f for f in self.group_fields if values.get(f) is None]
        if missing:
            raise CohortSourceFormatError(f"{where}: identifier {raw!r} lacks {missing}")
        group_id = self.group_template.format(**values)
        index = values.get(self.slide_field)
        slide_id = None if index is None else self.slide_template.format(**values)
        return str(values[self.cohort_field]), group_id, slide_id, index


# --------------------------------------------------------------------------- configuration


@dataclass(frozen=True)
class SourceFileSpec:
    path: str
    sha256: str
    partition: str | None


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    role: SourceRole
    root: str
    provenance: str
    id_column: str
    files: tuple[SourceFileSpec, ...]
    source_target: str | None = None
    label_column: str | None = None
    partition_column: str | None = None
    vocabulary: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class CohortConfig:
    definition: CohortDefinition
    identity: IdentityGrammar
    partition_names: tuple[tuple[str, str], ...]
    sources: tuple[SourceSpec, ...]

    @property
    def roots(self) -> tuple[str, ...]:
        return tuple(sorted({s.root for s in self.sources}))


def _decision(doc: Any, i: int) -> Decision:
    where = f"decisions[{i}]"
    d = _keys(
        doc,
        where,
        {"id", "summary", "status", "gates"},
        {"approver", "approved_on", "reference"},
    )
    try:
        status = DecisionStatus(_str(d["status"], f"{where}.status"))
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from None
    approved_on = d.get("approved_on")
    return Decision(
        decision_id=_str(d["id"], f"{where}.id"),
        summary=_str(d["summary"], f"{where}.summary"),
        status=status,
        gates=_strs(d["gates"], f"{where}.gates"),
        approver=d.get("approver"),
        approved_on=None if approved_on is None else str(approved_on),
        reference=d.get("reference"),
    )


def _source(doc: Any, i: int, partition_names: Mapping[str, str]) -> SourceSpec:
    where = f"sources[{i}]"
    d = _keys(
        doc,
        where,
        {"id", "role", "root", "provenance", "id_column", "files"},
        {"source_target", "label_column", "partition_column", "vocabulary"},
    )
    try:
        role = SourceRole(_str(d["role"], f"{where}.role"))
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from None
    files = []
    if not isinstance(d["files"], list) or not d["files"]:
        raise ConfigurationError(f"{where}.files must be a non-empty list")
    for j, f in enumerate(d["files"]):
        fd = _keys(f, f"{where}.files[{j}]", {"path", "sha256"}, {"partition"})
        part = fd.get("partition")
        if part is not None and part not in partition_names:
            raise ConfigurationError(f"{where}.files[{j}]: unknown partition {part!r}")
        files.append(
            SourceFileSpec(
                _str(fd["path"], f"{where}.files[{j}].path"),
                _str(fd["sha256"], f"{where}.files[{j}].sha256").lower(),
                None if part is None else partition_names[part],
            )
        )
    label_keys = ("source_target", "label_column", "vocabulary")
    if role is SourceRole.ROSTER:
        extra = [k for k in (*label_keys, "partition_column") if k in d]
        if extra or any(f.partition for f in files):
            raise ConfigurationError(f"{where}: a roster has no label, target or partition")
        vocabulary: tuple[tuple[str, str], ...] = ()
    else:
        missing = [k for k in label_keys if k not in d]
        if missing:
            raise ConfigurationError(f"{where}: a label source needs {missing}")
        vocab = d["vocabulary"]
        if not isinstance(vocab, Mapping) or not vocab:
            raise ConfigurationError(f"{where}.vocabulary must map raw values to classes")
        vocabulary = tuple(
            sorted((str(k), _str(v, f"{where}.vocabulary")) for k, v in vocab.items())
        )
        per_file = all(f.partition is not None for f in files)
        if per_file == ("partition_column" in d) or (
            not per_file and any(f.partition is not None for f in files)
        ):
            raise ConfigurationError(
                f"{where}: partitions come either from every file or from partition_column"
            )
    return SourceSpec(
        source_id=_str(d["id"], f"{where}.id"),
        role=role,
        root=_str(d["root"], f"{where}.root"),
        provenance=_str(d["provenance"], f"{where}.provenance"),
        id_column=_str(d["id_column"], f"{where}.id_column"),
        files=tuple(files),
        source_target=d.get("source_target"),
        label_column=d.get("label_column"),
        partition_column=d.get("partition_column"),
        vocabulary=vocabulary,
    )


def load_cohort_config(path: Path) -> CohortConfig:
    path = Path(path)
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"{path.name} cannot be read: {exc}") from None
    d = _keys(
        doc,
        path.name,
        {
            "schema",
            "cohort_id",
            "version",
            "target",
            "identity",
            "label_authority",
            "slides",
            "split",
            "partition_names",
            "sources",
            "decisions",
        },
        {"limitations"},
    )
    if d["schema"] != CONFIG_SCHEMA:
        raise ConfigurationError(f"{path.name}: schema must be {CONFIG_SCHEMA}")
    t = _keys(
        d["target"],
        "target",
        {"name", "classes", "accepted_source_targets", "decision"},
        {"positive_class"},
    )
    ident = _keys(
        d["identity"],
        "identity",
        {
            "pattern",
            "integer_fields",
            "cohort_field",
            "slide_field",
            "group_fields",
            "slide_id",
            "group_id",
            "decision",
        },
    )
    try:
        pattern = re.compile(_str(ident["pattern"], "identity.pattern"))
    except re.error as exc:
        raise ConfigurationError(f"identity.pattern: {exc}") from None
    identity = IdentityGrammar(
        pattern=pattern,
        integer_fields=_strs(ident["integer_fields"], "identity.integer_fields"),
        cohort_field=_str(ident["cohort_field"], "identity.cohort_field"),
        slide_field=_str(ident["slide_field"], "identity.slide_field"),
        group_fields=_strs(ident["group_fields"], "identity.group_fields"),
        slide_template=_str(ident["slide_id"], "identity.slide_id"),
        group_template=_str(ident["group_id"], "identity.group_id"),
    )
    pn = d["partition_names"]
    if not isinstance(pn, Mapping) or not all(v in PARTITIONS for v in pn.values()):
        raise ConfigurationError(
            f"partition_names must map source names to {[str(p) for p in PARTITIONS]}"
        )
    partition_names = {str(k): str(v) for k, v in pn.items()}
    sl = _keys(d["slides"], "slides", {"include_indices", "decision"})
    sp = _keys(d["split"], "split", {"strategy", "test_role", "pilot_partitions", "decisions"})
    la = _keys(d["label_authority"], "label_authority", {"decision"})
    if not isinstance(d["decisions"], list) or not isinstance(d["sources"], list):
        raise ConfigurationError("decisions and sources must be lists")
    definition = CohortDefinition(
        cohort_id=_str(d["cohort_id"], "cohort_id"),
        version=_str(d["version"], "version"),
        target=CohortTarget(
            name=_str(t["name"], "target.name"),
            classes=_strs(t["classes"], "target.classes"),
            positive_class=t.get("positive_class"),
            accepted_source_targets=_strs(
                t["accepted_source_targets"], "target.accepted_source_targets"
            ),
            decision=_str(t["decision"], "target.decision"),
        ),
        label_authority_decision=_str(la["decision"], "label_authority.decision"),
        group_decision=_str(ident["decision"], "identity.decision"),
        slide_policy=SlidePolicy(
            _strs(sl["include_indices"], "slides.include_indices"),
            _str(sl["decision"], "slides.decision"),
        ),
        split_policy=SplitPolicy(
            _str(sp["strategy"], "split.strategy"),
            _str(sp["test_role"], "split.test_role"),
            _strs(sp["pilot_partitions"], "split.pilot_partitions"),
            _strs(sp["decisions"], "split.decisions"),
        ),
        decisions=tuple(_decision(x, i) for i, x in enumerate(d["decisions"])),
        limitations=_strs(d.get("limitations") or [], "limitations"),
    )
    sources = tuple(_source(x, i, partition_names) for i, x in enumerate(d["sources"]))
    ids = [s.source_id for s in sources]
    if len(set(ids)) != len(ids):
        raise ConfigurationError("source ids must be unique")
    if not any(s.role is SourceRole.ROSTER for s in sources):
        raise ConfigurationError("a cohort declares at least one roster source")
    if not any(s.role is SourceRole.AUTHORITY for s in sources):
        raise ConfigurationError("a cohort declares at least one authoritative label source")
    for s in sources:
        for value in dict(s.vocabulary).values():
            if value not in definition.target.classes:
                raise ConfigurationError(f"source {s.source_id}: {value!r} is not a target class")
    return CohortConfig(definition, identity, tuple(sorted(partition_names.items())), sources)


def load_resolutions(path: Path) -> tuple[Resolution, ...]:
    """Owner-approved conflict resolutions (YAML list). Absent file: none."""
    try:
        doc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"{Path(path).name} cannot be read: {exc}") from None
    if not isinstance(doc, list):
        raise ConfigurationError("resolutions must be a list")
    keys = {"slide_id", "value", "partition", "resolver", "authority", "rationale", "approved_on"}
    out = []
    for i, r in enumerate(doc):
        d = _keys(r, f"resolutions[{i}]", keys)
        out.append(Resolution(**{k: str(d[k]) for k in keys}))
    return tuple(out)


# --------------------------------------------------------------------------- reading


def _verified_bytes(path: Path, pin: str, where: str) -> bytes:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise CohortSourceIntegrityError(f"{where}: {path.name} cannot be read: {exc}") from None
    actual = hashlib.sha256(data).hexdigest()
    if actual != pin:
        raise CohortSourceIntegrityError(
            f"{where}: {path.name} has SHA-256 {actual}, the configuration pins {pin}"
        )
    return data


def _rows(data: bytes, columns: tuple[str, ...], where: str) -> list[dict[str, str]]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CohortSourceFormatError(f"{where}: not UTF-8: {exc}") from None
    reader = csv.DictReader(io.StringIO(text, newline=""))
    header = tuple(reader.fieldnames or ())
    missing = [c for c in columns if c not in header]
    if missing:
        raise CohortSourceFormatError(f"{where} lacks column(s) {missing} (has {list(header)})")
    rows = list(reader)
    for n, row in enumerate(rows, start=2):
        if None in row or any(v is None for v in row.values()):
            raise CohortSourceFormatError(f"{where} line {n}: wrong number of cells")
    return rows


class CsvCohortSourceReader:
    """Reads the declared CSV sources of one cohort configuration."""

    def __init__(self, config: CohortConfig, roots: Mapping[str, Path]) -> None:
        missing = sorted(set(config.roots) - set(roots))
        if missing:
            raise ConfigurationError(
                f"source root(s) {missing} are not bound; pass NAME=PATH for each"
            )
        self._config = config
        self._roots = {k: Path(v) for k, v in roots.items()}

    def read(self) -> CohortObservations:
        sources: list[SourceFile] = []
        roster: list[RosterEntry] = []
        candidates: list[LabelCandidate] = []
        for spec in self._config.sources:
            for f in spec.files:
                path = self._roots[spec.root] / f.path
                where = f"source {spec.source_id} ({path.name})"
                data = _verified_bytes(path, f.sha256, where)
                columns = tuple(
                    c
                    for c in (spec.id_column, spec.label_column, spec.partition_column)
                    if c is not None
                )
                rows = _rows(data, columns, where)
                sources.append(
                    SourceFile(
                        source_id=spec.source_id,
                        role=spec.role,
                        file_name=path.name,
                        sha256=f.sha256,
                        rows=len(rows),
                        source_target=spec.source_target,
                        partition=f.partition,
                        provenance=spec.provenance,
                    )
                )
                if spec.role is SourceRole.ROSTER:
                    roster.extend(self._roster(spec, path.name, rows, where))
                else:
                    candidates.extend(self._labels(spec, f, path.name, rows, where))
        return CohortObservations(tuple(sources), tuple(roster), tuple(candidates))

    def _roster(
        self, spec: SourceSpec, name: str, rows: list[dict[str, str]], where: str
    ) -> list[RosterEntry]:
        out = []
        for n, row in enumerate(rows, start=2):
            raw = row[spec.id_column].strip()
            cohort, group_id, slide_id, index = self._config.identity.parse(
                raw, f"{where} line {n}"
            )
            if slide_id is None or index is None:
                raise CohortSourceFormatError(
                    f"{where} line {n}: roster identifier {raw!r} names no slide"
                )
            out.append(RosterEntry(slide_id, group_id, cohort, index, raw, spec.source_id, n))
        return out

    def _labels(
        self,
        spec: SourceSpec,
        f: SourceFileSpec,
        name: str,
        rows: list[dict[str, str]],
        where: str,
    ) -> list[LabelCandidate]:
        assert spec.label_column is not None and spec.source_target is not None
        vocabulary = dict(spec.vocabulary)
        partitions = dict(self._config.partition_names)
        seen: dict[str, int] = {}
        out = []
        for n, row in enumerate(rows, start=2):
            at = f"{where} line {n}"
            raw_id = row[spec.id_column].strip()
            _, group_id, slide_id, _ = self._config.identity.parse(raw_id, at)
            key = slide_id or group_id
            if key in seen:
                raise CohortSourceFormatError(
                    f"{at}: {raw_id!r} repeats the identifier of line {seen[key]}"
                )
            seen[key] = n
            raw_value = row[spec.label_column].strip()
            if raw_value not in vocabulary:
                raise CohortSourceFormatError(
                    f"{at}: label {raw_value!r} is outside the declared vocabulary "
                    f"{sorted(vocabulary)}"
                )
            if spec.partition_column is not None:
                raw_part = row[spec.partition_column].strip()
                if raw_part not in partitions:
                    raise CohortSourceFormatError(f"{at}: unknown partition {raw_part!r}")
                partition: str | None = partitions[raw_part]
            else:
                partition = f.partition
            out.append(
                LabelCandidate(
                    source_id=spec.source_id,
                    role=spec.role,
                    file_name=name,
                    row=n,
                    raw_id=raw_id,
                    raw_value=raw_value,
                    value=vocabulary[raw_value],
                    source_target=spec.source_target,
                    group_id=group_id,
                    slide_id=slide_id,
                    partition=partition,
                )
            )
        return out


# --------------------------------------------------------------------------- outputs


def read_manifest(path: Path) -> CohortManifest:
    return read_document(CohortManifest, path, CohortError)


def _write_bytes_once(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() == data:
            return
        raise ExperimentConflictError(f"{path.name} already holds other content; never replaced")
    _atomic_write(path, data)


def split_table(manifest: CohortManifest) -> bytes:
    """The eligible slides' imported partitions, in the SPEC-007 `slide_id,partition` format."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["slide_id", "partition"])
    w.writerows(split_assignment(manifest.records))
    return buf.getvalue().encode("utf-8")


RECONCILIATION_COLUMNS = (
    "slide_id",
    "group_id",
    "cohort",
    "slide_index",
    "status",
    "reason_code",
    "label",
    "partition",
    "source_targets",
    "conflict",
    "resolution_status",
    "candidates",
    "outstanding_gates",
    "wsi_status",
    "wsi_file",
)


def reconciliation_table(manifest: CohortManifest) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(RECONCILIATION_COLUMNS)
    for r in manifest.records:
        w.writerow(
            [
                r.slide_id,
                r.group_id,
                r.cohort,
                r.slide_index,
                r.status.value,
                r.reason_code or "",
                r.label or "",
                r.partition or "",
                ";".join(r.source_targets),
                r.conflict.value,
                r.resolution_status.value,
                ";".join(f"{c.location}={c.raw_value}" for c in r.candidates),
                ";".join(r.outstanding_gates),
                r.wsi.status.value,
                r.wsi.file_name,
            ]
        )
    return buf.getvalue().encode("utf-8")


def write_cohort_outputs(directory: Path, manifest: CohortManifest) -> dict[str, str]:
    """Write the manifest, split and reconciliation tables once; returns file -> SHA-256."""
    directory = Path(directory)
    write_once(directory / MANIFEST_FILE, manifest)
    _write_bytes_once(directory / SPLIT_FILE, split_table(manifest))
    _write_bytes_once(directory / RECONCILIATION_FILE, reconciliation_table(manifest))
    return {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in (MANIFEST_FILE, SPLIT_FILE, RECONCILIATION_FILE)
    }
