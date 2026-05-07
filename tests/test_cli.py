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


def test_submit_rejects_missing_after_dependency(monkeypatch, tmp_path):
    monkeypatch.setenv("FLUSSO_HOME", str(tmp_path / "state"))

    submitted = runner.invoke(app, ["submit", "--gpus", "0", "--after", "999", "--", "echo", "ok"])

    assert submitted.exit_code == 1
    assert "dependency job 999 does not exist" in submitted.output


def test_ls_shows_dependencies(monkeypatch, tmp_path):
    monkeypatch.setenv("FLUSSO_HOME", str(tmp_path / "state"))

    parent = runner.invoke(app, ["submit", "--gpus", "0", "--name", "parent", "--", "echo", "parent"])
    assert parent.exit_code == 0
    child = runner.invoke(
        app,
        ["submit", "--gpus", "0", "--name", "child", "--after", "1", "--", "echo", "child"],
    )
    assert child.exit_code == 0

    listed = runner.invoke(app, ["ls"])

    assert listed.exit_code == 0
    assert "DEPENDS" in listed.output
    assert "child" in listed.output
    assert "1" in listed.output


def test_format_local_timestamp_converts_sqlite_utc_timestamp():
    assert (
        format_local_timestamp("2026-05-07 08:00:00", ZoneInfo("Asia/Shanghai"))
        == "2026-05-07 16:00:00 CST"
    )


def test_format_local_timestamp_preserves_unknown_format():
    assert format_local_timestamp("not-a-timestamp") == "not-a-timestamp"
