from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path

from .config import FlussoConfig
from .gpu_probe import idle_gpu_indices, query_gpus
from .models import FAILED, SUCCEEDED, parse_gpu_list
from . import runner, store


class Scheduler:
    def __init__(self, conn: sqlite3.Connection, config: FlussoConfig):
        self.conn = conn
        self.config = config
        self.processes = {}

    def refresh_running_jobs(self) -> None:
        for job_id, process in list(self.processes.items()):
            exit_code = process.poll()
            if exit_code is None:
                continue
            status = SUCCEEDED if exit_code == 0 else FAILED
            store.mark_finished(self.conn, job_id, status=status, exit_code=exit_code)
            del self.processes[job_id]

    def internally_occupied_gpus(self) -> set[int]:
        occupied: set[int] = set()
        for job in store.running_jobs(self.conn):
            occupied.update(parse_gpu_list(job.assigned_gpus))
        return occupied

    def run_once(self) -> list[int]:
        self.refresh_running_jobs()
        probe = query_gpus()
        free_gpus = idle_gpu_indices(
            probe,
            config=self.config.gpu_idle,
            internally_occupied=self.internally_occupied_gpus(),
        )
        launched: list[int] = []

        for job in store.pending_jobs_fifo(self.conn):
            if len(free_gpus) < job.gpu_required:
                continue
            assigned = free_gpus[: job.gpu_required]
            started = runner.start_job(job, assigned, self.config.logs_dir)
            store.mark_running(
                self.conn,
                job.id,
                assigned_gpus=assigned,
                pid=started.pid,
                process_group_id=started.process_group_id,
                log_path=str(started.log_path),
            )
            self.processes[job.id] = started.popen
            free_gpus = free_gpus[job.gpu_required :]
            launched.append(job.id)

        return launched

    def run_forever(self) -> None:
        while True:
            self.run_once()
            time.sleep(self.config.scheduler.interval_seconds)


def is_process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True
