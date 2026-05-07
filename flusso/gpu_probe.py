from __future__ import annotations

import subprocess
from dataclasses import dataclass

from .config import GPUIdleConfig


@dataclass(frozen=True)
class GPUInfo:
    index: int
    uuid: str
    memory_used_mb: int
    utilization_percent: int
    has_compute_app: bool = False


@dataclass(frozen=True)
class GPUProbeResult:
    gpus: list[GPUInfo]
    available: bool = True
    error: str | None = None


def _run_nvidia_smi(args: list[str]) -> str:
    return subprocess.check_output(
        ["nvidia-smi", *args],
        stderr=subprocess.STDOUT,
        text=True,
    )


def _parse_gpu_rows(output: str) -> list[tuple[int, str, int, int]]:
    rows: list[tuple[int, str, int, int]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        index, uuid, memory, utilization = [part.strip() for part in line.split(",")]
        rows.append((int(index), uuid, int(memory), int(utilization)))
    return rows


def _parse_compute_uuids(output: str) -> set[str]:
    uuids: set[str] = set()
    for line in output.splitlines():
        if not line.strip():
            continue
        uuid = line.split(",", 1)[0].strip()
        if uuid:
            uuids.add(uuid)
    return uuids


def query_gpus() -> GPUProbeResult:
    try:
        gpu_output = _run_nvidia_smi(
            [
                "--query-gpu=index,uuid,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ]
        )
        compute_output = _run_nvidia_smi(
            [
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ]
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        return GPUProbeResult(gpus=[], available=False, error=str(exc))

    compute_uuids = _parse_compute_uuids(compute_output)
    gpus = [
        GPUInfo(
            index=index,
            uuid=uuid,
            memory_used_mb=memory,
            utilization_percent=utilization,
            has_compute_app=uuid in compute_uuids,
        )
        for index, uuid, memory, utilization in _parse_gpu_rows(gpu_output)
    ]
    return GPUProbeResult(gpus=gpus)


def idle_gpu_indices(
    result: GPUProbeResult,
    *,
    config: GPUIdleConfig,
    internally_occupied: set[int] | None = None,
) -> list[int]:
    if not result.available:
        return []
    occupied = internally_occupied or set()
    idle = []
    for gpu in result.gpus:
        if gpu.index in occupied:
            continue
        if gpu.has_compute_app:
            continue
        if gpu.memory_used_mb >= config.memory_threshold_mb:
            continue
        if gpu.utilization_percent >= config.utilization_threshold_percent:
            continue
        idle.append(gpu.index)
    return sorted(idle)
