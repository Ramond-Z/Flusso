from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .models import Job, format_gpu_list


@dataclass(frozen=True)
class StartedProcess:
    pid: int
    process_group_id: int
    log_path: Path
    popen: subprocess.Popen


def log_path_for_job(logs_dir: Path, job: Job) -> Path:
    return logs_dir / f"job-{job.id}.log"


def start_job(job: Job, assigned_gpus: list[int], logs_dir: Path) -> StartedProcess:
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_path_for_job(logs_dir, job)
    log_file = log_path.open("ab")
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": format_gpu_list(assigned_gpus),
    }
    try:
        process = subprocess.Popen(
            job.command,
            shell=True,
            cwd=job.working_directory,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
    finally:
        log_file.close()
    return StartedProcess(
        pid=process.pid,
        process_group_id=os.getpgid(process.pid),
        log_path=log_path,
        popen=process,
    )
