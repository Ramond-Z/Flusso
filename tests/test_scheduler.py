from dataclasses import dataclass

from flusso import store
from flusso.config import load_config
from flusso.gpu_probe import GPUInfo, GPUProbeResult
from flusso.models import FAILED, RUNNING, SKIPPED, SUCCEEDED
from flusso.scheduler import Scheduler


@dataclass
class FakeStarted:
    pid: int
    process_group_id: int
    log_path: object
    popen: object


class FakeProcess:
    def poll(self):
        return None


def make_scheduler(tmp_path):
    config = load_config(tmp_path / "state")
    config.logs_dir.mkdir(parents=True)
    store.init_db(config.db_path)
    conn = store.connect(config.db_path)
    return Scheduler(conn, config), conn


def test_scheduler_does_not_launch_when_resources_are_insufficient(monkeypatch, tmp_path):
    scheduler, conn = make_scheduler(tmp_path)
    store.create_job(conn, command="train", gpu_required=2, working_directory=str(tmp_path))
    monkeypatch.setattr(
        "flusso.scheduler.query_gpus",
        lambda: GPUProbeResult([GPUInfo(0, "GPU-0", 0, 0)]),
    )

    assert scheduler.run_once() == []
    assert store.get_job(conn, 1).status != RUNNING


def test_scheduler_launches_fifo_with_lowest_idle_gpus(monkeypatch, tmp_path):
    scheduler, conn = make_scheduler(tmp_path)
    store.create_job(conn, command="first", gpu_required=2, working_directory=str(tmp_path))
    store.create_job(conn, command="second", gpu_required=1, working_directory=str(tmp_path))
    launched = []

    monkeypatch.setattr(
        "flusso.scheduler.query_gpus",
        lambda: GPUProbeResult(
            [
                GPUInfo(2, "GPU-2", 0, 0),
                GPUInfo(0, "GPU-0", 0, 0),
                GPUInfo(3, "GPU-3", 0, 0),
            ]
        ),
    )

    def fake_start_job(job, assigned, logs_dir):
        launched.append((job.id, assigned))
        return FakeStarted(
            pid=1000 + job.id,
            process_group_id=1000 + job.id,
            log_path=logs_dir / f"job-{job.id}.log",
            popen=FakeProcess(),
        )

    monkeypatch.setattr("flusso.runner.start_job", fake_start_job)

    assert scheduler.run_once() == [1, 2]
    assert launched == [(1, [0, 2]), (2, [3])]
    assert store.get_job(conn, 1).status == RUNNING
    assert store.get_job(conn, 1).assigned_gpus == "0,2"


def test_scheduler_waits_for_dependencies_to_succeed(monkeypatch, tmp_path):
    scheduler, conn = make_scheduler(tmp_path)
    parent = store.create_job(conn, command="parent", gpu_required=1, working_directory=str(tmp_path))
    child = store.create_job(
        conn,
        command="child",
        gpu_required=1,
        working_directory=str(tmp_path),
        depends_on=[parent.id],
    )
    launched = []

    monkeypatch.setattr(
        "flusso.scheduler.query_gpus",
        lambda: GPUProbeResult([GPUInfo(0, "GPU-0", 0, 0), GPUInfo(1, "GPU-1", 0, 0)]),
    )

    def fake_start_job(job, assigned, logs_dir):
        launched.append(job.id)
        return FakeStarted(
            pid=1000 + job.id,
            process_group_id=1000 + job.id,
            log_path=logs_dir / f"job-{job.id}.log",
            popen=FakeProcess(),
        )

    monkeypatch.setattr("flusso.runner.start_job", fake_start_job)

    assert scheduler.run_once() == [parent.id]
    assert launched == [parent.id]
    assert store.get_job(conn, child.id).status != RUNNING

    scheduler.processes.clear()
    store.mark_finished(conn, parent.id, status=SUCCEEDED, exit_code=0)

    assert scheduler.run_once() == [child.id]


def test_scheduler_skips_jobs_when_dependency_failed(monkeypatch, tmp_path):
    scheduler, conn = make_scheduler(tmp_path)
    parent = store.create_job(conn, command="parent", gpu_required=1, working_directory=str(tmp_path))
    child = store.create_job(
        conn,
        command="child",
        gpu_required=1,
        working_directory=str(tmp_path),
        depends_on=[parent.id],
    )
    store.mark_finished(conn, parent.id, status=FAILED, exit_code=1)

    monkeypatch.setattr(
        "flusso.scheduler.query_gpus",
        lambda: GPUProbeResult([GPUInfo(0, "GPU-0", 0, 0)]),
    )

    assert scheduler.run_once() == []
    assert store.get_job(conn, child.id).status == SKIPPED
