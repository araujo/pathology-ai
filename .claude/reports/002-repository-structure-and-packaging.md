# Report 002 — Repository Structure & Packaging Refactor

**Date:** 2026-10-16
**SPEC:** `../specs/002-repository-structure-and-packaging.md` (Accepted 2026-10-16)
**ADR:** `../adr/002-root-level-package-layout.md` (Accepted 2026-10-16)
**Prompt:** `../prompts/002-repository-structure-and-packaging.md`
**Repository:** `pathology-ai`, branch `main`. **No commits have been made** (none were requested). All files are still untracked, so there is no commit SHA and no `git diff`. Before/after evidence comes from hashes and captured outputs.

## 1. Executive outcome

The framework's packages now live at the repository root, and `pathology_ai/` has been removed completely, with no wrapper, alias or duplicate.

- **Tests:** `ruff`, `ruff format`, `mypy --strict` and `pytest -W error` are all clean. **166 tests pass, 0 fail**: the 149 original tests plus 17 new architecture and resource-resolution tests.
- **Wheel:** contains exactly the six layer packages, `bootstrap.py`, `cli.py` and the ten API contract JSON files.
- **External install:** after a clean, non-editable install, imports and the `pathology-ai-demo` CLI work from a directory outside the checkout.
- **Behavior:** across all three synthetic projects, these are **byte-identical to the pre-refactor baseline**:
  - the demo output;
  - fingerprints, run IDs, artifact IDs and record digests;
  - pipeline digests;
  - serialized analysis results.
- **Contracts and definitions:** the API schemas, fixtures and YAML definitions are byte-identical.

**A pre-existing defect was found and fixed.** Before this work, the installed CLI failed outside the checkout with `NotFoundError: pipeline volume-segmentation-3d@1.0.0 is not registered`. The cause was that `bootstrap.py` looked for `pipelines/` and `projects/` relative to its own file, which in an installed wheel is `site-packages`. The owner chose this resolution (2026-10-16):

- The definitions stay outside the wheel.
- The CLI finds them through `--root PATH`, then `$PATHOLOGY_AI_ROOT`, then the source checkout.
- The current working directory is never used.

This refactor does **not** add real imaging support, scientific validation or DPA compatibility. All executable models are still synthetic.

## 2. Acceptance criteria

| AC | Status | Evidence |
|---|---|---|
| AC-001 Required root layout | **PASS** | `domain/`, `ports/`, `application/`, `infrastructure/`, `plugins/synthetic/`, `presentation/api/`, `bootstrap.py` and `cli.py` are at the root. `ls pathology_ai` gives "No such file or directory". Guard test: `tests/architecture/test_dependency_rules.py::test_obsolete_package_namespace_is_gone`. |
| AC-002 Import migration | **PASS** | `grep -rn pathology_ai` over `*.py` (excluding `.venv`) matches only `tests/architecture/test_dependency_rules.py`, which uses the name to *detect* the obsolete namespace. In the clean install, `import pathology_ai` raises `ModuleNotFoundError`. |
| AC-003 Packaging integrity | **PASS** | `uv build` produces `pathology_ai-0.1.0-py3-none-any.whl` with 57 entries (§8). Hatch `only-include` is explicit. Guard test: `::test_wheel_includes_exactly_the_first_party_modules`. |
| AC-004 External working directory | **PASS** | §8: clean venv, non-editable wheel install, cwd outside the checkout. Every module resolves from `site-packages` and the checkout is not on `sys.path`. The CLI works with `--root`, with `$PATHOLOGY_AI_ROOT`, and with a copy that holds only `pipelines/` and `projects/`. |
| AC-005 Existing test suite | **PASS** | All 149 baseline tests still pass. 145 test IDs are unchanged. 4 IDs changed only because their parametrized import strings changed from `pathology_ai.X` to `X` (the same 4 planted violations). No behavioral test was removed. |
| AC-006 Synthetic pipeline parity | **PASS** | Both cancer projects run `wsi-patch-embed-aggregate@1.0.0`, and psoriasis runs `wsi-direct-segmentation@1.0.0`. Outputs, `SUPPORTED_UNVALIDATED` status and evidence counts are byte-identical (§10). |
| AC-007 Volumetric contract preservation | **PASS** | `volume-segmentation-3d@1.0.0` still validates and still reports `EXECUTION_UNSUPPORTED` with the same four missing items (demo output identical). Integration tests `::test_volumetric_pipeline_*` pass. |
| AC-008 API contract preservation | **PASS** | SHA-256 of all 10 contract files and all 20 fixtures is identical before and after (§9). All 29 contract tests pass. |
| AC-009 Architecture rules | **PASS** | The 6 layer rules now detect root-level imports. New rules cover the composition root, the obsolete namespace, wheel contents and stdlib shadowing. There are 12 planted violations (previously 6), each detected (§11). |
| AC-010 Provenance integrity | **PASS** | Fingerprints, run IDs, artifact IDs, record digests and pipeline digests are byte-identical. No fingerprint algorithm was changed (§10). |
| AC-011 Repository independence | **PASS** | An rsync copy of the repository with no sibling directories, after `uv sync`, gives 166 passed and a byte-identical demo. |
| AC-012 Legacy repository protection | **PASS** | `digital-pathology` is at `c85df8a966e8…` and `hal-viewer` at `e248202983bc…`. Their `git status --porcelain` output (4 and 9 entries) is identical before and after (`diff` is empty). Only read-only git commands were run in them. |
| AC-013 Documentation accuracy | **PASS** | `README.md` and `../architecture/001-platform-boundaries.md` are updated. `dpa-v1-compatibility-inventory.md` needed no change: its file paths all refer to the sibling repositories. `001-platform-foundations.md` (report) is unchanged as a historical record. |
| AC-014 No functional scope expansion | **PASS** | No readers, models, inference, training or API servers were added. The only runtime change is how the definitions directory is located (§6), which fixes the defect in §1. |

## 3. Before/after structure

```text
Before                                   After
pathology-ai/                            pathology-ai/
├── pathology_ai/                        ├── domain/
│   ├── __init__.py  (__version__)       ├── ports/
│   ├── domain/                          ├── application/   (+ version.py)
│   ├── ports/                           ├── infrastructure/
│   ├── application/                     ├── plugins/synthetic/
│   ├── infrastructure/                  ├── presentation/api/contracts/
│   ├── plugins/synthetic/               ├── bootstrap.py
│   ├── presentation/api/contracts/      ├── cli.py
│   ├── bootstrap.py                     ├── pipelines/
│   └── cli.py                           ├── projects/
├── pipelines/                           ├── tests/
├── projects/                            ├── .claude/
├── tests/                               ├── pyproject.toml
├── .claude/                             ├── uv.lock
├── pyproject.toml                       └── README.md
├── uv.lock
└── README.md
```

## 4. File migration inventory

All moves were whole-directory `mv` operations. File contents are unchanged apart from import lines and the edits listed in the next table.

| Old path | New path | Content change |
|---|---|---|
| `pathology_ai/domain/` (19 modules) | `domain/` | imports only |
| `pathology_ai/ports/__init__.py` | `ports/__init__.py` | imports only |
| `pathology_ai/application/` (6 modules) | `application/` | imports only. `runner.py` and `model_operations.py` now import `__version__` from `application.version`. |
| `pathology_ai/infrastructure/` (3 modules) | `infrastructure/` | imports only |
| `pathology_ai/plugins/` (6 modules) | `plugins/` | imports only |
| `pathology_ai/presentation/` (5 modules + 10 JSON) | `presentation/` | imports only; JSON byte-identical |
| `pathology_ai/bootstrap.py` | `bootstrap.py` | imports, plus root resolution (§6) |
| `pathology_ai/cli.py` | `cli.py` | imports, plus `--root` option (§6) |
| `pathology_ai/__init__.py` (`__version__ = "0.1.0"`) | **removed**; value moved to `application/version.py` | same value |

Other changes:

| File | Change |
|---|---|
| `pyproject.toml` | Packaging changes (§5) |
| `tests/**/*.py`, 10 files | Mechanical rewrite of `pathology_ai.X` to `X` |
| `tests/architecture/test_dependency_rules.py` | Root-level layer detection plus new rules (§11) |
| `tests/unit/test_bootstrap_resources.py` | **New:** 8 tests for root resolution and CLI errors |
| `README.md` | Layout and installed-CLI usage |
| `.claude/architecture/001-platform-boundaries.md` | Layout note, layer table, rule list, new §6 on packaging and resources |
| `.claude/specs/002-…md`, `.claude/adr/002-…md` | Status changed to Accepted (2026-10-16) |
| `uv.lock` | Rewritten by `uv sync`. The editable `pathology-ai` entry no longer pins a static version because the version is now dynamic. Dependencies are unchanged (23 packages resolved). |

The import rewrite was a single `sed` pass with three rules:

- `from pathology_ai import __version__` becomes `from application.version import __version__`.
- `from pathology_ai.` and `import pathology_ai.` become `from ` and `import `.
- Planted test strings are rewritten the same way.

`ruff check --fix` then reordered one import block in `application/runner.py`.

## 5. Packaging and entry-point changes

```toml
[project]
dynamic = ["version"]                       # was: version = "0.1.0"

[project.scripts]
pathology-ai-demo = "cli:main"              # was: "pathology_ai.cli:main"

[tool.hatch.version]
path = "application/version.py"            # single source for the version

[tool.hatch.build.targets.wheel]
only-include = ["domain", "ports", "application", "infrastructure",
                "plugins", "presentation", "bootstrap.py", "cli.py"]
                                            # was: packages = ["pathology_ai"]

[tool.mypy]
files = [<the six packages>, "bootstrap.py", "cli.py", "tests"]
                                            # was: ["pathology_ai", "tests"]
```

- The distribution name `pathology-ai` and version `0.1.0` are unchanged.
- Python is still ≥3.12, built with uv and hatchling. Dependencies are unchanged.
- No `sys.path` manipulation was added.

## 6. Resource-loading strategy

| Resource | Packaged? | Resolution |
|---|---|---|
| Imaging and Analysis v1 JSON Schemas, `operations.json` | Yes (`presentation/api/contracts/`) | `importlib.resources.files(__name__)`, unchanged |
| `pipelines/*.yaml`, `projects/**/project.yaml` | No | `bootstrap.resolve_definitions_root(root)` (below) |
| Plugin registration | Code | Static wiring in `bootstrap.build_platform`, unchanged |

`resolve_definitions_root` looks in this order:

1. The explicit `root` argument (CLI `--root PATH`).
2. `$PATHOLOGY_AI_ROOT`.
3. The directory containing `bootstrap.py`, but only if it has both `pipelines/` and `projects/`. This is true in a source checkout, so `uv run pathology-ai-demo` behaves exactly as before.

The current working directory is never consulted. An explicitly configured location without both directories raises `DefinitionsRootError` rather than falling back. The CLI turns that error into exit status 2 with a one-line message.

`build_platform(root=...)` keeps its signature. The default changed from the module-level `REPO_ROOT` to `None`, which means "resolve".

## 7. Test results and exact commands

Environment: macOS arm64, Python 3.12.13, uv 0.11.17, numpy 2.5.4, pyyaml 6.0.3, ruff 0.17.1, mypy 2.4.0, pytest 9.1.1.

| Command | Baseline (before) | After |
|---|---|---|
| `uv sync` | Resolved 23 packages | Resolved 23 packages |
| `uv run ruff check .` | All checks passed | All checks passed |
| `uv run ruff format --check .` | 68 files already formatted (SPEC-001 report said 62; no file was unformatted, and I did not investigate the counting difference) | 69 files already formatted |
| `uv run mypy` | no issues in 58 source files | no issues in 59 source files |
| `uv run pytest -q -W error` | **149 passed** | **166 passed**, 0 failed, 0 skipped, 0 warnings |
| By suite | unit 81 · contract 29 · architecture 19 · integration 20 | unit 89 · contract 29 · architecture 28 · integration 20 |
| `uv run pathology-ai-demo` | exit 0 | exit 0, output byte-identical (`cmp`) |
| `uv build` | wheel built, but its CLI fails outside the checkout | sdist and wheel built |

The baseline matches the SPEC-001 report: 149 tests and the same three fingerprints.

## 8. Wheel contents and external-installation evidence

**Wheel** (`pathology_ai-0.1.0-py3-none-any.whl`, 57 entries):

| Top level | Entries |
|---|---|
| `application/` | 7 (6 modules + `version.py`) |
| `domain/` | 19 |
| `infrastructure/` | 3 |
| `plugins/`, `plugins/synthetic/` | 1 + 5 |
| `ports/` | 1 |
| `presentation/`, `presentation/api/`, `presentation/api/contracts/` | 1 + 3 + 1 |
| `presentation/api/contracts/analysis/v1/` | 5 JSON |
| `presentation/api/contracts/imaging/v1/` | 5 JSON |
| `bootstrap.py`, `cli.py` | 2 |
| `pathology_ai-0.1.0.dist-info/` | 4 |

- **Absent:** `pathology_ai/`, `tests/`, `pipelines/`, `projects/` and `.claude/`.
- **Entry point:** `pathology-ai-demo = cli:main`.
- The `-0.1.0.dist-info` name comes from normalizing the distribution name. It is not an import package.
- The **sdist** includes `tests/`, `pipelines/`, `projects/`, `.claude/` and `uv.lock` (hatch's default sdist content, as before). This is source distribution only; none of it is installed.

**Clean install outside the checkout:**

```text
uv venv -p 3.12 <scratch>/venv
uv pip install <repo>/dist/pathology_ai-0.1.0-py3-none-any.whl numpy==2.5.4 pyyaml==6.0.3
cd <scratch>/outside
```

| Check | Result |
|---|---|
| Checkout on `sys.path` | no (asserted) |
| `domain`, `domain.pipeline`, `ports`, `application.runner`, `application.version`, `infrastructure.config`, `plugins.synthetic.catalog`, `presentation.api.analysis_v1`, `presentation.api.contracts`, `bootstrap`, `cli` | all import from `…/venv/lib/python3.12/site-packages/…` |
| `import pathology_ai` | `ModuleNotFoundError` |
| `schema_names("analysis","v1")` / `("imaging","v1")` | 4 + 4 schemas loaded from the installed package |
| `importlib.metadata.version("pathology-ai")` | `0.1.0` |
| `pathology-ai-demo` (no root) | exit 2: `pipelines/ and projects/ are not packaged with the installed framework; pass --root PATH or set $PATHOLOGY_AI_ROOT …` |
| `pathology-ai-demo --root <repo>` | exit 0, byte-identical to the baseline demo |
| `PATHOLOGY_AI_ROOT=<repo> pathology-ai-demo` | exit 0, byte-identical |
| `pathology-ai-demo --root <copy holding only pipelines/ and projects/>` | byte-identical |
| Provenance probe on the installed code | byte-identical to the baseline probe |

numpy and pyyaml were pinned to the development versions on purpose. The fingerprint includes package versions by design (SPEC-001 report, risk 3), so different versions would correctly produce different fingerprints.

## 9. API contract hash comparison (SHA-256, before = after)

| File (`presentation/api/contracts/`) | SHA-256 |
|---|---|
| `analysis/v1/analysis-job.schema.json` | `4a71f85e0f1b28cc6fc98f907ed6f1e6c82edd99c5e29f0a911dbb398d33f2ed` |
| `analysis/v1/analysis-result.schema.json` | `80c9fc9522eef62b843e20263ca818a6892a2715116a415c7519c38e3c7b6dd6` |
| `analysis/v1/capability-list.schema.json` | `895f8ef4d49b6a9157a557bafdf2c746e964bcebe863319578ea3366b29ecae7` |
| `analysis/v1/operations.json` | `8cb2d7a58dbbbe6a1cc2c9a19d0b373bb8eb4f4025f9a9a5d24313c1ba81046e` |
| `analysis/v1/submit-analysis-request.schema.json` | `d36f1915d4ffd94fbe9a2a7403dfb04c56516b4fcf12f28fa1d6b4051ba70e74` |
| `imaging/v1/image-descriptor.schema.json` | `aef439f70df97bfb708cde16f53d25b3f93084fc82a4996ef81ae83dda9e9ac4` |
| `imaging/v1/operations.json` | `d2c8a45ae722d2a8316dc2bdd75cc8d01caed45161161ed8a24855dc0d6a962c` |
| `imaging/v1/region-request.schema.json` | `a89b4e899fbca8e636f5a1a0a126d5cf661be94b02a3252de63fa6fc1fe15007` |
| `imaging/v1/slice-request.schema.json` | `45b9afd17840ed52b54c525b47d76257fe99fd53f583a4210c1d5266ac78abb4` |
| `imaging/v1/thumbnail-request.schema.json` | `7fae5289e3de379676686ebaf82300a4d1486b7b542323bf432bf135f6c01692` |

- The 20 files under `tests/contract/fixtures/` and the 6 pipeline and project YAML files have identical hash lists before and after (`cmp`).
- The only change is the resource *location*, from `pathology_ai/presentation/api/contracts/` to `presentation/api/contracts/`. Contract IDs (`pathology-ai.imaging/1.0`, `pathology-ai.analysis/1.0`) and `$id` URLs are unchanged.

## 10. Synthetic prediction and provenance comparison

The probe uses `build_platform(root, clock=FixedClock())`, runs the first `test` image of each project, and dumps the following:

- the fingerprint;
- `run_id` and `record_digest`;
- capability status, usage mode and run status;
- model, component and stage-version records;
- the environment;
- artifact IDs;
- the digest of the serialized analysis result;
- the pipeline digest.

The before and after outputs are **byte-identical**: the post-refactor in-checkout run, the installed-wheel run and the demo all match.

| Project | Pipeline (digest prefix) | Fingerprint | Run ID | Artifacts |
|---|---|---|---|---|
| `patologia/cancer/pulmao` | `wsi-patch-embed-aggregate@1.0.0` (`8afdd7d0…`) | `pai-fp-05e0480b394e11fb874c3099a94f6cc5` | `pai-run-5059a0f3…` | `pai-art-e38ebbef…`, `f5c588ae…`, `03be3ef8…`, `70d2c80d…` |
| `patologia/cancer/coloretal` | `wsi-patch-embed-aggregate@1.0.0` (`8afdd7d0…`) | `pai-fp-7c764da5c48906780bc236e8936742d4` | `pai-run-876391d2…` | `pai-art-5a98bb67…`, `9937dcb0…`, `456eefef…`, `0359bcdc…` |
| `patologia/pele/psoriase` | `wsi-direct-segmentation@1.0.0` (`2ed3d07c…`) | `pai-fp-f565b8ddc13220b378dee739d14baeac` | `pai-run-ce7db95f…` | `pai-art-e3af4af3…`, `775b9818…` |

**Predictions:**

- **pulmao:** `synthetic_positive`, scores `{negative 0.004985…, positive 0.995014…}`, 5 attention evidence items.
- **coloretal:** `synthetic_positive`, scores `{0.004468…, 0.995531…}`, 5 evidence items.
- **psoriase:** segmentation mask `pai-art-775b9818…`, level 3.

All three are `SUPPORTED_UNVALIDATED`. Scores carry `scores_are_calibrated_probabilities: false`.

**Why identities did not change:**

- `domain/hashing.py` canonicalizes dataclasses by field name and value, never by module path.
- The only code identity in provenance is `ComponentVersion("package", "pathology-ai", "0.1.0")` plus installed distribution versions, and all of these are unchanged.

No fingerprint logic was modified.

## 11. Architecture dependency validation

`tests/architecture/test_dependency_rules.py` now identifies layers by their root-level top-level name.

**Unchanged rules:**

- the 6 layer rules (`application` no longer needs the `pathology_ai.__init__` exception);
- framework-free core;
- no legacy imports;
- no references to legacy repositories (now also covering `bootstrap.py` and `cli.py`);
- no hard-coded dataset, vendor or target terms in the core;
- no `docs/` directory.

**New rules:**

| Test | Guards |
|---|---|
| Any layer importing `bootstrap` or `cli` | Composition root can't be imported inward |
| Any `pathology_ai` import, or the `pathology_ai/` directory existing | `test_obsolete_package_namespace_is_gone` |
| `only-include` differs from the six packages + `bootstrap.py` + `cli.py`, or a root package exists outside that list (except `tests`) | `test_wheel_includes_exactly_the_first_party_modules`; the entry point must target the composition root |
| A first-party top-level name equals a stdlib module | `test_first_party_names_do_not_shadow_stdlib` |

**Planted violations (12, all detected):**

- the original 6, with rewritten paths;
- `application` imported from `plugins`;
- `infrastructure` imported from `ports`;
- `bootstrap` imported from `application`;
- `cli` imported from `presentation`;
- `pathology_ai.domain` imported from `domain`;
- `pathology_ai` imported from `application`.

The planted `pathology_ai` imports show the obsolete-namespace rule is not vacuous.

No cross-repository imports exist (`test_no_legacy_imports_anywhere`).

## 12. Legacy repository status comparison

| Repository | HEAD before | HEAD after | `status --porcelain` before → after |
|---|---|---|---|
| `../digital-pathology` | `main@c85df8a966e8cc063a61c509d038f1b1cae89c3a` | same | 4 → 4 entries, `diff` empty |
| `../hal-viewer` | `main@e248202983bc6cdb497ed00ef0168e6b47338e09` | same | 9 → 9 entries, `diff` empty |

Both are unchanged since SPEC-001. A read-only `grep` found no reference to `pathology_ai` or `pathology-ai` in either repository, so there are no external consumers of the removed namespace (ADR-002 D7).

## 13. Known limitations and unresolved risks

1. **Generic top-level names.** Installing `domain`, `ports`, `application`, `infrastructure`, `plugins`, `presentation`, `bootstrap` and `cli` into `site-packages` would conflict with any other distribution that ships the same names. `cli` and `bootstrap` are especially generic. There is no collision today (checked against the dev environment and the stdlib). This is accepted in ADR-002, and the architecture test only covers stdlib collisions.
2. **The installed CLI needs a definitions root.** This is intended and documented, but a bare `pathology-ai-demo` after `pip install` exits with status 2.
3. **Editable installs put the repository root on `sys.path`** (hatch's `.pth`). In development, other root directories such as `tests` become importable too. The wheel does not do this, and the clean-install check (§8) asserts the checkout is absent from `sys.path`.
4. **Fingerprints depend on the environment** (SPEC-001 risk 3, unchanged). The installed-wheel comparison pinned numpy and pyyaml to the development versions.
5. **SPEC examples vs. code.** The SPEC uses `from ports.model import Model`, but `ports` is a single module, so the actual import is `from ports import Model`. I did not split `ports`; that would be a structural change outside this SPEC.
6. All SPEC-001 limitations still apply: synthetic models only, in-memory stores, no DPA adapter, no CT/MRI execution.

## 14. Deviations from SPEC-002

- **Added: `--root` / `$PATHOLOGY_AI_ROOT` resolution in `bootstrap.py` and `cli.py`.** SPEC §5.6 and prompt §4.4 require it: the pre-existing installed CLI could not locate its definitions. The owner chose the strategy before implementation. It changes no pipeline, model or contract behavior.
- **Version moved to `application/version.py`, read by hatch (`dynamic = ["version"]`).** This was needed because `pathology_ai/__init__.py` was removed. The value `0.1.0` and its role in provenance are unchanged.
- No other deviations. No `docs/` directory was created, no compatibility package was added, and no tests were deleted.
