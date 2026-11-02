"""Resolution of the pipelines/ and projects/ root (SPEC-002 §5.6): explicit, never the cwd."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import bootstrap
import cli
from bootstrap import ROOT_ENV_VAR, DefinitionsRootError, resolve_definitions_root

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _no_env_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ROOT_ENV_VAR, raising=False)


@pytest.fixture
def installed_layout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Simulate a wheel install: the module directory holds no definitions."""
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()
    monkeypatch.setattr(bootstrap, "_MODULE_DIR", site_packages)
    return site_packages


def test_source_checkout_is_the_default_root() -> None:
    assert resolve_definitions_root() == REPO_ROOT


def test_explicit_root_wins_over_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(ROOT_ENV_VAR, str(tmp_path))
    assert resolve_definitions_root(REPO_ROOT) == REPO_ROOT


def test_environment_root_is_used(monkeypatch: pytest.MonkeyPatch, installed_layout: Path) -> None:
    monkeypatch.setenv(ROOT_ENV_VAR, str(REPO_ROOT))
    assert resolve_definitions_root() == REPO_ROOT


@pytest.mark.parametrize("via_env", [False, True])
def test_configured_root_without_definitions_fails_instead_of_falling_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, via_env: bool
) -> None:
    if via_env:
        monkeypatch.setenv(ROOT_ENV_VAR, str(tmp_path))
    with pytest.raises(DefinitionsRootError, match="must contain pipelines/ and projects/"):
        resolve_definitions_root(None if via_env else tmp_path)


def test_installed_layout_never_uses_the_working_directory(
    monkeypatch: pytest.MonkeyPatch, installed_layout: Path
) -> None:
    monkeypatch.chdir(REPO_ROOT)  # the cwd has definitions, but must not be consulted
    with pytest.raises(DefinitionsRootError, match="--root PATH"):
        resolve_definitions_root()


def test_cli_reports_missing_root_without_traceback(
    installed_layout: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main([])
    assert exc.value.code == 2
    assert ROOT_ENV_VAR in capsys.readouterr().err


def test_cli_runs_with_explicit_root(
    installed_layout: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["--root", str(REPO_ROOT)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert [p["namespace"] for p in summary["projects"]] == [
        "patologia/cancer/coloretal",
        "patologia/cancer/pulmao",
        "patologia/pele/psoriase",
    ]
    assert summary["volumetric"]["status"] == "EXECUTION_UNSUPPORTED"
