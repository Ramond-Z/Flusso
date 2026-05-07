from __future__ import annotations

import argparse
import os
import time


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny Flusso demo job.")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--label", default="toy")
    parser.add_argument("--sleep", type=float, default=0.2)
    args = parser.parse_args()

    print(f"[{args.label}] cwd={os.getcwd()}", flush=True)
    print(f"[{args.label}] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}", flush=True)
    for step in range(1, args.steps + 1):
        print(f"[{args.label}] step {step}/{args.steps}", flush=True)
        time.sleep(args.sleep)
    print(f"[{args.label}] done", flush=True)


if __name__ == "__main__":
    main()
