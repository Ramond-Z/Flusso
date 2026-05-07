from typer.testing import CliRunner

from flusso.cli import app


runner = CliRunner()


def test_delete_pending_job_from_cli(monkeypatch, tmp_path):
    monkeypatch.setenv("FLUSSO_HOME", str(tmp_path / "state"))

    submit = runner.invoke(app, ["submit", "--gpus", "0", "--name", "delete-me", "--", "echo", "ok"])
    assert submit.exit_code == 0

    deleted = runner.invoke(app, ["delete", "1"])
    assert deleted.exit_code == 0
    assert "Deleted job 1" in deleted.output

    listed = runner.invoke(app, ["ls"])
    assert listed.exit_code == 0
    assert "delete-me" not in listed.output
