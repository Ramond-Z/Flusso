from __future__ import annotations

import sqlite3
from pathlib import Path

from .models import HELD, Job, PENDING, RUNNING, format_gpu_list


SCHEMA = """
CREATE TABLE IF NOT EXISTS groups (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    command TEXT NOT NULL,
    gpu_required INTEGER NOT NULL,
    status TEXT NOT NULL,
    working_directory TEXT NOT NULL,
    group_id INTEGER REFERENCES groups(id),
    assigned_gpus TEXT,
    pid INTEGER,
    process_group_id INTEGER,
    exit_code INTEGER,
    log_path TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    started_at TEXT,
    ended_at TEXT
);

CREATE TABLE IF NOT EXISTS dependencies (
    job_id INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    depends_on_job_id INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    PRIMARY KEY (job_id, depends_on_job_id)
);
"""


def connect(db_path: Path | str) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: Path | str) -> None:
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)


def _row_to_job(row: sqlite3.Row) -> Job:
    return Job(
        id=row["id"],
        name=row["name"],
        command=row["command"],
        gpu_required=row["gpu_required"],
        status=row["status"],
        working_directory=row["working_directory"],
        group_id=row["group_id"],
        assigned_gpus=row["assigned_gpus"],
        pid=row["pid"],
        process_group_id=row["process_group_id"],
        exit_code=row["exit_code"],
        log_path=row["log_path"],
        created_at=row["created_at"],
        started_at=row["started_at"],
        ended_at=row["ended_at"],
    )


def create_job(
    conn: sqlite3.Connection,
    *,
    command: str,
    gpu_required: int,
    working_directory: str,
    name: str | None = None,
    status: str = PENDING,
) -> Job:
    cursor = conn.execute(
        """
        INSERT INTO jobs (name, command, gpu_required, status, working_directory)
        VALUES (?, ?, ?, ?, ?)
        """,
        (name, command, gpu_required, status, working_directory),
    )
    conn.commit()
    return get_job(conn, cursor.lastrowid)


def get_job(conn: sqlite3.Connection, job_id: int) -> Job:
    row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if row is None:
        raise KeyError(f"job {job_id} does not exist")
    return _row_to_job(row)


def list_jobs(conn: sqlite3.Connection) -> list[Job]:
    rows = conn.execute("SELECT * FROM jobs ORDER BY id ASC").fetchall()
    return [_row_to_job(row) for row in rows]


def pending_jobs_fifo(conn: sqlite3.Connection) -> list[Job]:
    rows = conn.execute(
        """
        SELECT *
        FROM jobs
        WHERE status = ?
        ORDER BY created_at ASC, id ASC
        """,
        (PENDING,),
    ).fetchall()
    return [_row_to_job(row) for row in rows]


def running_jobs(conn: sqlite3.Connection) -> list[Job]:
    rows = conn.execute(
        "SELECT * FROM jobs WHERE status = ? ORDER BY id ASC",
        (RUNNING,),
    ).fetchall()
    return [_row_to_job(row) for row in rows]


def delete_unscheduled_job(conn: sqlite3.Connection, job_id: int) -> Job:
    job = get_job(conn, job_id)
    if job.status not in {PENDING, HELD}:
        raise ValueError(f"job {job_id} has status {job.status} and cannot be deleted")
    conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
    conn.commit()
    return job


def mark_running(
    conn: sqlite3.Connection,
    job_id: int,
    *,
    assigned_gpus: list[int],
    pid: int,
    process_group_id: int,
    log_path: str,
) -> Job:
    conn.execute(
        """
        UPDATE jobs
        SET status = ?,
            assigned_gpus = ?,
            pid = ?,
            process_group_id = ?,
            log_path = ?,
            started_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (RUNNING, format_gpu_list(assigned_gpus), pid, process_group_id, log_path, job_id),
    )
    conn.commit()
    return get_job(conn, job_id)


def mark_finished(conn: sqlite3.Connection, job_id: int, *, status: str, exit_code: int) -> Job:
    conn.execute(
        """
        UPDATE jobs
        SET status = ?,
            exit_code = ?,
            ended_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (status, exit_code, job_id),
    )
    conn.commit()
    return get_job(conn, job_id)
