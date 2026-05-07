# Flusso Toy Examples

These examples are intentionally small and safe to run. They do not train a real model; they only show how jobs are submitted, scheduled, and logged.

Use an isolated state directory while trying them:

```bash
export FLUSSO_HOME=/tmp/flusso-toy
uv run flusso init
```

## CPU-only smoke job

This runs without `nvidia-smi` because it requests zero GPUs:

```bash
uv run flusso submit --gpus 0 --name toy-cpu -- uv run python examples/toy_job.py --steps 3 --label cpu
uv run flusso daemon run
```

In another terminal:

```bash
FLUSSO_HOME=/tmp/flusso-toy uv run flusso ls
FLUSSO_HOME=/tmp/flusso-toy uv run flusso logs 1
```

Stop the foreground daemon with `Ctrl-C` after the job finishes.

## Fake GPU training job

Run this on a machine with idle NVIDIA GPUs visible to `nvidia-smi`:

```bash
uv run flusso submit --gpus 1 --name toy-train -- uv run python examples/fake_train.py --epochs 3
uv run flusso daemon run
```

The log should show the `CUDA_VISIBLE_DEVICES` value injected by Flusso.

## Submit several toy jobs

This script initializes an isolated state directory and submits a small queue:

```bash
examples/submit_toy_queue.sh
```

Then run:

```bash
FLUSSO_HOME=/tmp/flusso-toy uv run flusso daemon run
```
