from __future__ import annotations

import argparse
import os
import time


def main() -> None:
    parser = argparse.ArgumentParser(description="Fake GPU training script for Flusso.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=0.5)
    args = parser.parse_args()

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    print(f"visible GPUs: {visible or '<none>'}", flush=True)
    for epoch in range(1, args.epochs + 1):
        print(f"epoch {epoch}/{args.epochs}: loss={1 / epoch:.4f}", flush=True)
        time.sleep(args.sleep)
    print("training complete", flush=True)


if __name__ == "__main__":
    main()
