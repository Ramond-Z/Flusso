from flusso import store
import pytest

from flusso.models import CANCELLED, FAILED, HELD, PENDING, RUNNING, SKIPPED, SUCCEEDED


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


def test_create_job_rejects_missing_dependency(tmp_path):
    db_path = tmp_path / "flusso.db"
    store.init_db(db_path)

    with store.connect(db_path) as conn:
        with pytest.raises(ValueError, match="dependency job 999 does not exist"):
            store.create_job(
                conn,
                command="child",
                gpu_required=1,
                working_directory=str(tmp_path),
                depends_on=[999],
            )

        assert store.list_jobs(conn) == []


def test_pending_jobs_fifo_only_returns_jobs_with_succeeded_dependencies(tmp_path):
    db_path = tmp_path / "flusso.db"
    store.init_db(db_path)

    with store.connect(db_path) as conn:
        parent = store.create_job(conn, command="parent", gpu_required=1, working_directory=str(tmp_path))
        child = store.create_job(
            conn,
            command="child",
            gpu_required=1,
            working_directory=str(tmp_path),
            depends_on=[parent.id],
        )

        assert [job.id for job in store.pending_jobs_fifo(conn)] == [parent.id]
        assert store.job_dependencies(conn, child.id) == [parent.id]

        store.mark_finished(conn, parent.id, status=SUCCEEDED, exit_code=0)

        assert [job.id for job in store.pending_jobs_fifo(conn)] == [child.id]


def test_skip_jobs_with_failed_dependencies_marks_blocked_pending_jobs(tmp_path):
    db_path = tmp_path / "flusso.db"
    store.init_db(db_path)

    with store.connect(db_path) as conn:
        parent = store.create_job(conn, command="parent", gpu_required=1, working_directory=str(tmp_path))
        child = store.create_job(
            conn,
            command="child",
            gpu_required=1,
            working_directory=str(tmp_path),
            depends_on=[parent.id],
        )
        store.mark_finished(conn, parent.id, status=FAILED, exit_code=1)

        skipped = store.skip_jobs_with_failed_dependencies(conn)

        assert [job.id for job in skipped] == [child.id]
        assert store.get_job(conn, child.id).status == SKIPPED
        assert store.pending_jobs_fifo(conn) == []


def test_skip_jobs_with_failed_dependencies_waits_for_nonterminal_dependencies(tmp_path):
    db_path = tmp_path / "flusso.db"
    store.init_db(db_path)

    with store.connect(db_path) as conn:
        parent = store.create_job(conn, command="parent", gpu_required=1, working_directory=str(tmp_path))
        child = store.create_job(
            conn,
            command="child",
            gpu_required=1,
            working_directory=str(tmp_path),
            depends_on=[parent.id],
        )

        assert store.skip_jobs_with_failed_dependencies(conn) == []
        assert store.get_job(conn, child.id).status == PENDING


def test_cancel_unscheduled_job_allows_pending_and_held(tmp_path):
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

        cancelled_pending = store.cancel_unscheduled_job(conn, pending.id)
        cancelled_held = store.cancel_unscheduled_job(conn, held.id)

        assert cancelled_pending.status == CANCELLED
        assert cancelled_held.status == CANCELLED
        assert [job.status for job in store.list_jobs(conn)] == [CANCELLED, CANCELLED]
        assert store.pending_jobs_fifo(conn) == []


def test_cancel_unscheduled_job_rejects_running_job(tmp_path):
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
            store.cancel_unscheduled_job(conn, job.id)

        assert store.get_job(conn, job.id).status == RUNNING
