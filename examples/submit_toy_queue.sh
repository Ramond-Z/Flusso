#!/usr/bin/env bash
set -euo pipefail

export FLUSSO_HOME="${FLUSSO_HOME:-/tmp/flusso-toy}"

uv run flusso init
uv run flusso submit --gpus 0 --name toy-short -- uv run python examples/toy_job.py --steps 2 --label short
uv run flusso submit --gpus 0 --name toy-medium -- uv run python examples/toy_job.py --steps 4 --label medium
uv run flusso submit --gpus 0 --name toy-long -- uv run python examples/toy_job.py --steps 6 --label long
uv run flusso ls
