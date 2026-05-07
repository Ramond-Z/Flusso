import os
import subprocess

from flusso.models import Job, PENDING
from flusso.runner import start_job


class FakePopen:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.pid = 4321


def test_start_job_injects_cuda_and_logs(monkeypatch, tmp_path):
    created = {}

    def fake_popen(*args, **kwargs):
        process = FakePopen(*args, **kwargs)
        created["process"] = process
        return process

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(os, "getpgid", lambda pid: pid)

    job = Job(
        id=7,
        name="train",
        command="python train.py",
        gpu_required=2,
        status=PENDING,
        working_directory=str(tmp_path),
    )

    started = start_job(job, [1, 3], tmp_path / "logs")

    process = created["process"]
    assert process.args[0] == "python train.py"
    assert process.kwargs["shell"] is True
    assert process.kwargs["cwd"] == str(tmp_path)
    assert process.kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "1,3"
    assert process.kwargs["stderr"] == subprocess.STDOUT
    assert process.kwargs["preexec_fn"] == os.setsid
    assert started.log_path == tmp_path / "logs" / "job-7.log"
