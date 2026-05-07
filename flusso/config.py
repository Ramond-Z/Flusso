from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_CONFIG = """gpu_idle:
  memory_threshold_mb: 1000
  utilization_threshold_percent: 10
  required_consecutive_idle_checks: 3
  check_interval_seconds: 5
scheduler:
  interval_seconds: 5
"""


@dataclass(frozen=True)
class GPUIdleConfig:
    memory_threshold_mb: int = 1000
    utilization_threshold_percent: int = 10
    required_consecutive_idle_checks: int = 3
    check_interval_seconds: int = 5


@dataclass(frozen=True)
class SchedulerConfig:
    interval_seconds: float = 5.0


@dataclass(frozen=True)
class FlussoConfig:
    state_dir: Path
    db_path: Path
    logs_dir: Path
    config_path: Path
    gpu_idle: GPUIdleConfig = GPUIdleConfig()
    scheduler: SchedulerConfig = SchedulerConfig()


def default_state_dir() -> Path:
    configured = os.environ.get("FLUSSO_HOME")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".flusso"


def load_config(state_dir: Path | None = None) -> FlussoConfig:
    root = (state_dir or default_state_dir()).expanduser()
    return FlussoConfig(
        state_dir=root,
        db_path=root / "flusso.db",
        logs_dir=root / "logs",
        config_path=root / "config.yaml",
    )


def ensure_state(config: FlussoConfig) -> None:
    config.state_dir.mkdir(parents=True, exist_ok=True)
    config.logs_dir.mkdir(parents=True, exist_ok=True)
    if not config.config_path.exists():
        config.config_path.write_text(DEFAULT_CONFIG, encoding="utf-8")
