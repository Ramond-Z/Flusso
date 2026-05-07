import subprocess

from flusso.config import GPUIdleConfig
from flusso.gpu_probe import idle_gpu_indices, query_gpus


def test_query_gpus_marks_compute_apps(monkeypatch):
    outputs = [
        "0, GPU-0, 100, 0\n1, GPU-1, 200, 5\n",
        "GPU-1, 991\n",
    ]

    def fake_check_output(*args, **kwargs):
        return outputs.pop(0)

    monkeypatch.setattr(subprocess, "check_output", fake_check_output)

    result = query_gpus()

    assert result.available is True
    assert [gpu.index for gpu in result.gpus if gpu.has_compute_app] == [1]


def test_idle_gpu_indices_apply_thresholds_and_internal_occupancy(monkeypatch):
    outputs = [
        "0, GPU-0, 100, 0\n1, GPU-1, 1200, 0\n2, GPU-2, 100, 20\n3, GPU-3, 100, 0\n",
        "",
    ]
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: outputs.pop(0))

    result = query_gpus()
    idle = idle_gpu_indices(
        result,
        config=GPUIdleConfig(memory_threshold_mb=1000, utilization_threshold_percent=10),
        internally_occupied={3},
    )

    assert idle == [0]


def test_query_gpus_unavailable_when_nvidia_smi_fails(monkeypatch):
    def fail(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(subprocess, "check_output", fail)

    result = query_gpus()

    assert result.available is False
    assert idle_gpu_indices(result, config=GPUIdleConfig()) == []
