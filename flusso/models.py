from __future__ import annotations

from dataclasses import dataclass


PENDING = "PENDING"
RUNNING = "RUNNING"
SUCCEEDED = "SUCCEEDED"
FAILED = "FAILED"
CANCELLED = "CANCELLED"
HELD = "HELD"
SKIPPED = "SKIPPED"


@dataclass(frozen=True)
class Job:
    id: int
    name: str | None
    command: str
    gpu_required: int
    status: str
    working_directory: str
    group_id: int | None = None
    assigned_gpus: str | None = None
    pid: int | None = None
    process_group_id: int | None = None
    exit_code: int | None = None
    log_path: str | None = None
    created_at: str | None = None
    started_at: str | None = None
    ended_at: str | None = None


def format_gpu_list(gpus: list[int]) -> str:
    return ",".join(str(gpu) for gpu in gpus)


def parse_gpu_list(value: str | None) -> list[int]:
    if not value:
        return []
    return [int(part) for part in value.split(",") if part != ""]
