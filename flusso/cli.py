from __future__ import annotations

from datetime import datetime, timezone, tzinfo
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from . import store
from .config import ensure_state, load_config
from .scheduler import Scheduler


app = typer.Typer(help="Single-node multi-GPU scheduler.")
daemon_app = typer.Typer(help="Run the foreground scheduler daemon.")
app.add_typer(daemon_app, name="daemon")
console = Console()


def _config():
    config = load_config()
    ensure_state(config)
    store.init_db(config.db_path)
    return config


def format_local_timestamp(value: str | None, local_tz: tzinfo | None = None) -> str:
    if not value:
        return "-"
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return value
    target_tz = local_tz or datetime.now().astimezone().tzinfo
    return parsed.astimezone(target_tz).strftime("%Y-%m-%d %H:%M:%S %Z")


@app.command()
def init() -> None:
    """Create Flusso state files."""
    config = _config()
    console.print(f"Initialized Flusso state at {config.state_dir}")


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def submit(
    ctx: typer.Context,
    gpus: int = typer.Option(..., "--gpus", min=0, help="Number of GPUs required."),
    name: Optional[str] = typer.Option(None, "--name", help="Job name."),
    cwd: Path = typer.Option(Path.cwd(), "--cwd", help="Working directory."),
    after: list[int] = typer.Option(
        [],
        "--after",
        help="Job ID that must succeed before this job can run. Repeat for multiple dependencies.",
    ),
) -> None:
    """Submit a shell command as a pending job."""
    command = " ".join(ctx.args)
    if not command:
        raise typer.BadParameter("COMMAND is required after --")
    config = _config()
    with store.connect(config.db_path) as conn:
        try:
            job = store.create_job(
                conn,
                command=command,
                gpu_required=gpus,
                working_directory=str(cwd.expanduser().resolve()),
                name=name,
                depends_on=after,
            )
        except ValueError as exc:
            console.print(str(exc))
            raise typer.Exit(1) from None
    console.print(f"Submitted job {job.id}")


@app.command("ls")
def list_queue() -> None:
    """Show the job queue."""
    config = _config()
    with store.connect(config.db_path) as conn:
        jobs = store.list_jobs(conn)
        dependencies = {job.id: store.job_dependencies(conn, job.id) for job in jobs}

    table = Table()
    table.add_column("ID", justify="right")
    table.add_column("NAME")
    table.add_column("STATUS")
    table.add_column("GPUS", justify="right")
    table.add_column("ASSIGNED")
    table.add_column("DEPENDS")
    table.add_column("CREATED (LOCAL)")
    for job in jobs:
        table.add_row(
            str(job.id),
            job.name or "-",
            job.status,
            str(job.gpu_required),
            job.assigned_gpus or "-",
            ",".join(str(job_id) for job_id in dependencies[job.id]) or "-",
            format_local_timestamp(job.created_at),
        )
    console.print(table)


@app.command()
def cancel(job_id: int) -> None:
    """Cancel a job that has not been scheduled yet."""
    config = _config()
    with store.connect(config.db_path) as conn:
        try:
            job = store.cancel_unscheduled_job(conn, job_id)
        except KeyError:
            console.print(f"Job {job_id} does not exist")
            raise typer.Exit(1) from None
        except ValueError as exc:
            console.print(str(exc))
            raise typer.Exit(1) from None
    console.print(f"Cancelled job {job.id}")


@app.command()
def logs(job_id: int) -> None:
    """Print a job log."""
    config = _config()
    with store.connect(config.db_path) as conn:
        job = store.get_job(conn, job_id)
    if not job.log_path:
        console.print(f"Job {job_id} has no log yet")
        raise typer.Exit(1)
    path = Path(job.log_path)
    if not path.exists():
        console.print(f"Log file does not exist: {path}")
        raise typer.Exit(1)
    console.print(path.read_text(encoding="utf-8", errors="replace"), end="")


@daemon_app.command("run")
def daemon_run() -> None:
    """Run the scheduler loop in the foreground."""
    config = _config()
    with store.connect(config.db_path) as conn:
        Scheduler(conn, config).run_forever()


if __name__ == "__main__":
    app()
