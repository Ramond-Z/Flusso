from zoneinfo import ZoneInfo

from typer.testing import CliRunner

from flusso.cli import app, format_local_timestamp


runner = CliRunner()


def test_cancel_pending_job_from_cli_keeps_job_in_ls(monkeypatch, tmp_path):
    monkeypatch.setenv("FLUSSO_HOME", str(tmp_path / "state"))

    submit = runner.invoke(app, ["submit", "--gpus", "0", "--name", "cancel-me", "--", "echo", "ok"])
    assert submit.exit_code == 0

    cancelled = runner.invoke(app, ["cancel", "1"])
    assert cancelled.exit_code == 0
    assert "Cancelled job 1" in cancelled.output

    listed = runner.invoke(app, ["ls"])
    assert listed.exit_code == 0
    assert "cancel-me" in listed.output
    assert "CANCELLED" in listed.output


def test_format_local_timestamp_converts_sqlite_utc_timestamp():
    assert (
        format_local_timestamp("2026-05-07 08:00:00", ZoneInfo("Asia/Shanghai"))
        == "2026-05-07 16:00:00 CST"
    )


def test_format_local_timestamp_preserves_unknown_format():
    assert format_local_timestamp("not-a-timestamp") == "not-a-timestamp"
