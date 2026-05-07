from flusso import store
import pytest

from flusso.models import HELD, PENDING, RUNNING


def test_init_db_and_create_job(tmp_path):
    db_path = tmp_path / "flusso.db"
    store.init_db(db_path)

    with store.connect(db_path) as conn:
        job = store.create_job(
            conn,
            command="python train.py",
            gpu_required=2,
            working_directory=str(tmp_path),
            name="train",
        )

        assert job.id == 1
        assert job.status == PENDING
        assert job.name == "train"
        assert store.get_job(conn, 1).command == "python train.py"


def test_fifo_pending_jobs_and_running_transition(tmp_path):
    db_path = tmp_path / "flusso.db"
    store.init_db(db_path)

    with store.connect(db_path) as conn:
        first = store.create_job(conn, command="a", gpu_required=1, working_directory=str(tmp_path))
        second = store.create_job(conn, command="b", gpu_required=1, working_directory=str(tmp_path))

        pending_ids = [job.id for job in store.pending_jobs_fifo(conn)]
        assert pending_ids == [first.id, second.id]

        running = store.mark_running(
            conn,
            first.id,
            assigned_gpus=[0, 2],
            pid=123,
            process_group_id=123,
            log_path=str(tmp_path / "job-1.log"),
        )

        assert running.status == RUNNING
        assert running.assigned_gpus == "0,2"
        assert [job.id for job in store.pending_jobs_fifo(conn)] == [second.id]


def test_delete_unscheduled_job_allows_pending_and_held(tmp_path):
    db_path = tmp_path / "flusso.db"
    store.init_db(db_path)

    with store.connect(db_path) as conn:
        pending = store.create_job(conn, command="a", gpu_required=1, working_directory=str(tmp_path))
        held = store.create_job(
            conn,
            command="b",
            gpu_required=1,
            working_directory=str(tmp_path),
            status=HELD,
        )

        assert store.delete_unscheduled_job(conn, pending.id).id == pending.id
        assert store.delete_unscheduled_job(conn, held.id).id == held.id
        assert store.list_jobs(conn) == []


def test_delete_unscheduled_job_rejects_running_job(tmp_path):
    db_path = tmp_path / "flusso.db"
    store.init_db(db_path)

    with store.connect(db_path) as conn:
        job = store.create_job(conn, command="a", gpu_required=1, working_directory=str(tmp_path))
        store.mark_running(
            conn,
            job.id,
            assigned_gpus=[0],
            pid=123,
            process_group_id=123,
            log_path=str(tmp_path / "job-1.log"),
        )

        with pytest.raises(ValueError, match="RUNNING"):
            store.delete_unscheduled_job(conn, job.id)

        assert store.get_job(conn, job.id).status == RUNNING
