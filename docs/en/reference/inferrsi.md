# InferRSI source snapshot

[`inferrsi/`](../../../inferrsi/README.md) contains the complete tracked source tree of
[InferMatrixLab/InferRSI](https://github.com/InferMatrixLab/InferRSI), an inference optimization workbench.

## Provenance and layout

- Source revision: [`e0ca06cb99e6087292279fab13c56f602db209ec`][source-revision] (`main`, imported on 2026-10-20).
- All 600 tracked files retain their original contents and paths beneath `inferrsi/`, including dotfiles,
  source code, tests, documentation, recipes, benchmarks, evidence, packaged skills and license notices.
- The `inferrsi` distribution uses [`inferrsi/pyproject.toml`](../../../inferrsi/pyproject.toml).
  Install it separately from the root `agentinfer` distribution.
- The copied `inferrsi/.github/` retains upstream workflow and issue templates. GitHub discovers active workflows
  from the repository-root `.github/workflows/`; the nested workflows remain source reference material.
- Root pre-commit hooks exclude this snapshot to preserve upstream formatting. Use InferRSI's own checks below.

## Installation and local checks

From the AgentInfer repository root, using Python 3.10 or later:

```bash
cd inferrsi
python -m pip install -e ".[dev]"
python -m inferrsi --help
python -m pytest tests
python -m ruff check src tests benchmarks/harness
```

The CPU tests validate workbench behavior; accelerator experiments and performance claims require their own
target environments and evidence. Imported historical results retain their upstream verification scope.

## Documentation

Start with the [InferRSI README](../../../inferrsi/README.md),
[quickstart](../../../inferrsi/docs/quickstart.md),
[operation guide](../../../inferrsi/docs/operations.md), and
[contributor guide](../../../inferrsi/docs/contributing.md).

[source-revision]: https://github.com/InferMatrixLab/InferRSI/tree/e0ca06cb99e6087292279fab13c56f602db209ec
