"""Architecture rules enforced by static import analysis (AST), not by convention."""

from __future__ import annotations

import ast
import re
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# First-party layers are root-level packages (ADR-002).
# layer -> allowed internal layers (besides itself)
ALLOWED: dict[str, set[str]] = {
    "domain": set(),
    "ports": {"domain"},
    "application": {"domain", "ports"},
    "infrastructure": {"domain", "ports", "application"},
    "plugins": {"domain", "ports"},
    "presentation": {"domain", "ports", "application"},
}
THIRD_PARTY_ALLOWED: dict[str, set[str]] = {
    "domain": set(),
    "ports": set(),
    "application": set(),
    # numpy is a base dependency; infrastructure uses it only for the `.npy` embedding store.
    "infrastructure": {"yaml", "numpy"},
    "plugins": {"numpy"},
    "presentation": set(),
}
# Optional reader backends (SPEC-003): allowed only inside their own adapter subpackage.
SCOPED_THIRD_PARTY: dict[str, tuple[str, ...]] = {
    "pylibCZIrw": ("plugins/imaging/czi/",),
    "openslide": ("plugins/imaging/openslide/",),
    # Optional model runtime (SPEC-006 / ADR-006 D3): only inside the model adapter package and,
    # for trainable MIL models (SPEC-007), the MIL plugin package.
    "torch": ("plugins/models/", "plugins/mil/"),
    "timm": ("plugins/models/",),
}
# Composition root: may import every layer, but no layer may import it.
COMPOSITION_ROOT = {"bootstrap", "cli"}
FIRST_PARTY = set(ALLOWED) | COMPOSITION_ROOT
OBSOLETE_NAMESPACE = "pathology_ai"
LEGACY_MODULES = {"digital_pathology", "hal_viewer", "scripts", "pipeline", "analysis_api"}


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            out.append(node.module)
        elif (
            isinstance(node, ast.Call)
            and getattr(node.func, "attr", getattr(node.func, "id", None)) == "import_module"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            out.append(node.args[0].value)  # lazy imports count too
    return out


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _modules(layer: str) -> list[Path]:
    return sorted((ROOT / layer).rglob("*.py"))


def _framework_modules() -> list[Path]:
    paths = [p for layer in sorted(ALLOWED) for p in _modules(layer)]
    return paths + [ROOT / f"{name}.py" for name in sorted(COMPOSITION_ROOT)]


def _violations(layer: str, paths: list[Path]) -> list[str]:
    violations = []
    stdlib = sys.stdlib_module_names
    for path in paths:
        for name in _imports(path):
            top = name.split(".")[0]
            if top == OBSOLETE_NAMESPACE:
                violations.append(f"{path.name} imports obsolete namespace {name}")
            elif top in COMPOSITION_ROOT:
                violations.append(f"{path.name} imports composition root {name}")
            elif top in ALLOWED:
                if top != layer and top not in ALLOWED[layer]:
                    violations.append(f"{path.name} imports {name}")
            elif top == "__future__" or top in stdlib:
                continue
            elif top in SCOPED_THIRD_PARTY:
                if not _relative(path).startswith(SCOPED_THIRD_PARTY[top]):
                    violations.append(f"{path.name} imports {name} outside its adapter")
            elif top not in THIRD_PARTY_ALLOWED[layer]:
                violations.append(f"{path.name} imports third-party {name}")
    return violations


@pytest.mark.parametrize("layer", sorted(ALLOWED))
def test_layer_dependency_rules(layer: str) -> None:
    assert _modules(layer), f"layer {layer} has no modules"
    assert _violations(layer, _modules(layer)) == []


def test_first_party_names_do_not_shadow_stdlib() -> None:
    assert FIRST_PARTY.isdisjoint(sys.stdlib_module_names)


def test_domain_is_framework_free() -> None:
    banned = {
        "numpy",
        "torch",
        "timm",
        "torchvision",
        "pydantic",
        "fastapi",
        "yaml",
        "PIL",
        "cv2",
        "pandas",
    }
    for path in _modules("domain") + _modules("ports") + _modules("application"):
        for name in _imports(path):
            assert name.split(".")[0] not in banned, f"{path} imports {name}"


def test_no_legacy_imports_anywhere() -> None:
    for path in _framework_modules() + sorted((ROOT / "tests").rglob("*.py")):
        for name in _imports(path):
            assert name.split(".")[0] not in LEGACY_MODULES, f"{path} imports {name}"


def test_no_runtime_reference_to_legacy_repositories() -> None:
    for path in _framework_modules():
        text = path.read_text(encoding="utf-8")
        assert "digital-pathology" not in text and "hal-viewer" not in text, path


def test_obsolete_package_namespace_is_gone() -> None:
    """ADR-002 D1/D7: no `pathology_ai/` directory, wrapper or import remains."""
    assert not (ROOT / OBSOLETE_NAMESPACE).exists()
    for path in _framework_modules() + sorted((ROOT / "tests").rglob("*.py")):
        for name in _imports(path):
            assert name.split(".")[0] != OBSOLETE_NAMESPACE, f"{path} imports {name}"


def test_wheel_includes_exactly_the_first_party_modules() -> None:
    """Explicit discovery (ADR-002 D3): every root package ships; tests/configs never do."""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    included = set(pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["only-include"])
    expected = set(ALLOWED) | {f"{name}.py" for name in COMPOSITION_ROOT}
    assert included == expected
    root_packages = {p.parent.name for p in ROOT.glob("*/__init__.py")}
    assert root_packages - {"tests"} == set(ALLOWED)
    entry = pyproject["project"]["scripts"]["pathology-ai-demo"]
    assert entry.split(":")[0] in COMPOSITION_ROOT


CORE_FORBIDDEN = [
    r"\bczi\b",
    r"\bkras\b",
    r"\buni\b",
    r"colorectal",
    r"coloretal",
    r"\bh&e\b",
    r"\bhematoxylin",
    r"\babmil\b",
    r"surgen",
    r"\bzeiss\b",
    r"aperio",
    r"openslide",
    r"pylibczirw",
]


@pytest.mark.parametrize("layer", ["domain", "ports", "application"])
def test_core_has_no_hardcoded_dataset_vendor_model_or_target(layer: str) -> None:
    for path in _modules(layer):
        text = path.read_text(encoding="utf-8").lower()
        for pattern in CORE_FORBIDDEN:
            assert not re.search(pattern, text), f"{path} matches {pattern}"


def test_no_docs_directory() -> None:
    assert not (ROOT / "docs").exists()


@pytest.mark.parametrize(
    "source,layer",
    [
        ("from plugins.synthetic import catalog\n", "domain"),
        ("import numpy\n", "domain"),
        ("from infrastructure.config import load_pipeline\n", "application"),
        ("from presentation.api import analysis_v1\n", "domain"),
        ("import yaml\n", "ports"),
        ("from plugins.synthetic import catalog\n", "presentation"),
        ("from application.runner import PipelineRunner\n", "plugins"),
        ("import infrastructure.runtime\n", "ports"),
        ("from bootstrap import build_platform\n", "application"),
        ("from cli import main\n", "presentation"),
        ("from pathology_ai.domain.ids import ImageId\n", "domain"),
        ("from pylibCZIrw import czi\n", "plugins"),
        ("import openslide\n", "plugins"),
        ("importlib.import_module('pylibCZIrw.czi')\n", "plugins"),
        ("import pylibCZIrw\n", "infrastructure"),
        ("import pathology_ai\n", "application"),
        ("import torch\n", "application"),
        ("import timm\n", "infrastructure"),
        ("importlib.import_module('timm')\n", "plugins"),
        ("import torch\n", "domain"),
        ("from plugins.mil import TorchMILFactory\n", "application"),
    ],
)
def test_rule_detects_planted_violation(tmp_path: Path, source: str, layer: str) -> None:
    """Guard against a vacuous rule: a planted bad import must be reported."""
    bad = tmp_path / "bad.py"
    bad.write_text(source)
    assert _violations(layer, [bad])


def test_reader_backends_are_imported_only_by_their_adapters() -> None:
    """Each optional backend appears (eagerly or lazily) only in its own adapter subpackage."""
    seen: dict[str, set[str]] = {lib: set() for lib in SCOPED_THIRD_PARTY}
    for path in _framework_modules():
        for name in _imports(path):
            top = name.split(".")[0]
            if top in seen:
                seen[top].add(_relative(path))
    for lib, files in seen.items():
        assert files, f"{lib} is not used by any adapter"
        assert all(f.startswith(SCOPED_THIRD_PARTY[lib]) for f in files), (lib, files)


# SPEC-006 P1 / ADR-006 D1: the generic model runtime knows no model by name; model specifics
# live in their own adapter module.
GENERIC_MODEL_RUNTIME = [
    "plugins/models/__init__.py",
    "plugins/models/checkpoints.py",
    "plugins/models/torch_runtime.py",
    "plugins/models/fixture.py",
    "plugins/models/fixture_net.py",
    "plugins/embedding/operations.py",
    "plugins/embedding/streaming.py",
    "plugins/embedding/transforms.py",
]
MODEL_NAMES = [r"\buni\b", r"uni2", r"virchow", r"\bplip\b", r"mahmood", r"surgen", r"vit_large"]


@pytest.mark.parametrize("relative", GENERIC_MODEL_RUNTIME)
def test_generic_model_runtime_names_no_model(relative: str) -> None:
    text = (ROOT / relative).read_text(encoding="utf-8").lower()
    for pattern in MODEL_NAMES:
        assert not re.search(pattern, text), f"{relative} matches {pattern}"


def test_model_runtime_is_imported_lazily() -> None:
    """Only adapter modules may import the torch runtime, and only inside functions."""
    for path in _framework_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:  # module level only
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
            for name in names:
                eager = name.split(".")[0] in {"torch", "timm"} or name.startswith(
                    (
                        "plugins.models.torch_runtime",
                        "plugins.models.fixture_net",
                        "plugins.mil.torch_models",
                    )
                )
                allowed = _relative(path) in {
                    "plugins/models/torch_runtime.py",
                    "plugins/models/fixture_net.py",
                    "plugins/mil/torch_models.py",
                }
                assert not eager or allowed, f"{_relative(path)} imports {name} eagerly"


# SPEC-007 AC-001: generic MIL code knows no disease, tissue, biomarker, dataset or encoder. Class
# names, targets and encoders are configuration.
GENERIC_MIL_MODULES = [
    "domain/mil.py",
    "domain/mil_training.py",
    "domain/metrics.py",
    "application/mil_dataset.py",
    "application/mil_training.py",
    "infrastructure/mil_artifacts.py",
    "infrastructure/mil_bags.py",
    "infrastructure/mil_config.py",
    "infrastructure/canonical.py",
    "plugins/mil/__init__.py",
    "plugins/mil/batching.py",
    "plugins/mil/torch_models.py",
]
MIL_FORBIDDEN = (
    CORE_FORBIDDEN
    + MODEL_NAMES
    + [
        r"\bbraf\b",
        r"\bnras\b",
        r"\bmsi\b",
        r"tumou?r",
        r"carcinoma",
        r"cancer",
        r"\blung\b",
        r"\bskin\b",
        r"psoria",
        r"\bmutant\b",
        r"\bwild[ -]?type\b",
    ]
)


@pytest.mark.parametrize("relative", GENERIC_MIL_MODULES)
def test_generic_mil_code_is_task_disease_and_encoder_agnostic(relative: str) -> None:
    text = (ROOT / relative).read_text(encoding="utf-8").lower()
    for pattern in MIL_FORBIDDEN:
        assert not re.search(pattern, text), f"{relative} matches {pattern}"


def test_mil_framework_code_is_isolated_and_lazy() -> None:
    """Only `plugins/mil/torch_models.py` imports torch among MIL modules; the rest of MIL
    (datasets, splits, metrics, artifacts, application services) runs without it."""
    for relative in GENERIC_MIL_MODULES:
        names = {n.split(".")[0] for n in _imports(ROOT / relative)}
        if relative == "plugins/mil/torch_models.py":
            assert "torch" in names
        else:
            assert "torch" not in names, relative


# SPEC-008 / ADR-008 D1: cohort reconciliation is generic. Cohort names, identifier formats,
# targets and decisions are configuration under projects/.
GENERIC_COHORT_MODULES = [
    "domain/cohort.py",
    "domain/wsi_inventory.py",
    "domain/pilot.py",
    "application/cohort_service.py",
    "application/wsi_inventory_service.py",
    "infrastructure/cohort_sources.py",
    "infrastructure/wsi_inventory.py",
]
COHORT_FORBIDDEN = MIL_FORBIDDEN + [r"\bsr386\b", r"\bsr1482\b", r"\bmmr", r"dmmr", r"\bmss\b"]


@pytest.mark.parametrize("relative", GENERIC_COHORT_MODULES)
def test_generic_cohort_code_names_no_cohort_or_target(relative: str) -> None:
    text = (ROOT / relative).read_text(encoding="utf-8").lower()
    for pattern in COHORT_FORBIDDEN:
        assert not re.search(pattern, text), f"{relative} matches {pattern}"


def test_wsi_inventory_source_opens_files_read_only() -> None:
    """The inventory adapter never opens anything for writing (SPEC-008 C1: never modify WSIs).
    Its only writes are the write-once documents, delegated to `mil_artifacts`."""
    tree = ast.parse((ROOT / "infrastructure/wsi_inventory.py").read_text(encoding="utf-8"))
    modes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "open":
            mode = node.args[1] if len(node.args) > 1 else None
            modes.append(mode.value if isinstance(mode, ast.Constant) else None)
        attr = getattr(node.func, "attr", None) if isinstance(node, ast.Call) else None
        assert attr not in {"write_bytes", "write_text", "unlink", "rename", "replace", "touch"}, (
            attr
        )
    assert modes == ["rb"]
