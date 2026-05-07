# Repository Guidelines

## Project Structure & Module Organization

This repository currently contains the Flusso design document and license only:

- `doc.md`: primary product and architecture design notes for the single-node multi-GPU scheduler.
- `LICENSE`: project license.
- `AGENTS.md`: contributor guidance for future changes.

When implementation begins, keep source code in `flusso/`, tests in `tests/`, and examples or sample configs in `examples/` or `configs/`. Do not commit generated logs, local experiment output, or runtime state.

## Environment, Build, Test, and Development Commands

No build system or test runner is defined yet. Do not add command references unless the tooling exists.

Prefer `uv` for Python environment and dependency management when implementation begins. If a `pyproject.toml` is added, use commands such as:

```bash
uv sync
uv run pytest
uv run ruff check .
uv run python -m flusso
```

Keep dependency changes in `pyproject.toml` and commit `uv.lock` when present.

Current checks:

```bash
git status --short
```

Shows local changes.

```bash
sed -n '1,120p' doc.md
```

Reviews the design context.

Once code is added, document the exact `uv run ...` workflow here.

## Coding Style & Naming Conventions

For the planned CLI scheduler, Python is a natural default. Keep package code under `flusso/` and prefer clear module names such as `scheduler`, `gpu_probe`, `task_store`, and `cli`. Keep command examples explicit about GPU behavior, especially `CUDA_VISIBLE_DEVICES` handling.

Use concise Markdown in documentation. Preserve the existing Chinese design context in `doc.md` unless intentionally translating or restructuring it.

## Testing Guidelines

There are no tests yet. Future scheduler code should cover dependency handling, GPU selection, task state transitions, retries, and cancellation. Mock `nvidia-smi` output rather than requiring real GPUs in CI. Run tests through `uv`, for example `uv run pytest`.

Name tests by behavior, for example:

```text
tests/test_scheduler_dependencies.py
tests/test_gpu_probe.py
```

## Commit & Pull Request Guidelines

The current Git history only contains `Initial commit`, so no detailed convention has been established. Use short, imperative commit messages such as:

```text
Add scheduler design notes
Implement GPU probing
```

Pull requests should include a summary, rationale, verification steps, and user-facing command or behavior changes. For scheduler behavior, include example commands and note whether real GPU access is required.

## Security & Configuration Tips

Do not commit local machine paths, credentials, API keys, experiment logs, or scheduler state databases. Treat submitted shell commands as user-controlled input, and document any assumptions about process isolation before implementing execution features.
