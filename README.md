# Flusso

Flusso 是一个轻量级的单机多 GPU 任务调度器。它允许用户提交 shell command，并声明任务需要多少张 GPU；前台 daemon 会观察 GPU 空闲状态，在资源满足时启动任务，并通过 `CUDA_VISIBLE_DEVICES` 注入分配结果。

当前版本是 MVP，重点覆盖单任务提交、FIFO 调度、日志记录和队列查看。更完整的设计背景见 [doc.md](doc.md)。

## 功能状态

已实现：

- `flusso init`：初始化本地状态目录、SQLite 数据库、日志目录和默认配置。
- `flusso submit`：提交一个 `PENDING` 任务。
- `flusso daemon run`：以前台模式运行调度循环。
- `flusso ls`：查看任务队列。
- `flusso cancel JOB_ID`：取消尚未被调度的任务，并在队列中保留记录。
- `flusso logs JOB_ID`：查看任务日志。
- GPU 空闲检测：通过 `nvidia-smi` 检查显存、利用率和 compute process。
- 任务启动：独立 process group，写 stdout/stderr 到日志文件，并设置 `CUDA_VISIBLE_DEVICES`。

暂未实现：

- 后台 daemon 的 `start/stop/status` 管理。
- 任务依赖、任务组、运行中任务取消、重试、暂停/恢复。
- GPU 状态表格和 `logs -f`。
- 多机调度或强制资源隔离。

## 安装与开发环境

本项目使用 `uv` 管理依赖和命令运行。

```bash
uv sync
```

查看 CLI：

```bash
uv run flusso --help
```

运行测试：

```bash
uv run pytest
```

## 快速开始

默认状态目录是 `~/.flusso`：

```text
~/.flusso/
  flusso.db
  logs/
  config.yaml
```

初始化：

```bash
uv run flusso init
```

提交一个需要 1 张 GPU 的任务：

```bash
uv run flusso submit --gpus 1 --name train-demo -- python train.py --config configs/demo.yaml
```

启动前台调度器：

```bash
uv run flusso daemon run
```

在另一个终端查看队列：

```bash
uv run flusso ls
```

队列中的时间以本地时区显示；内部 SQLite 时间戳使用 UTC 保存。

取消尚未被调度的任务：

```bash
uv run flusso cancel 1
```

查看日志：

```bash
uv run flusso logs 1
```

前台 daemon 可以用 `Ctrl-C` 停止。

## Toy Examples

推荐先使用临时状态目录，避免污染真实的 `~/.flusso`：

```bash
export FLUSSO_HOME=/tmp/flusso-toy
uv run flusso init
```

运行一个不需要 GPU 的 toy job：

```bash
uv run flusso submit --gpus 0 --name toy-cpu -- uv run python examples/toy_job.py --steps 3 --label cpu
uv run flusso daemon run
```

在另一个终端查看状态和日志：

```bash
FLUSSO_HOME=/tmp/flusso-toy uv run flusso ls
FLUSSO_HOME=/tmp/flusso-toy uv run flusso logs 1
```

也可以一次性提交多个 toy job：

```bash
examples/submit_toy_queue.sh
FLUSSO_HOME=/tmp/flusso-toy uv run flusso daemon run
```

更多示例见 [examples/README.md](examples/README.md)。

## 调度行为

当前调度策略是简单 FIFO：

1. 刷新本 daemon 启动的运行中任务。
2. 调用 `nvidia-smi` 采样 GPU 状态。
3. 找出所有 `PENDING` job。
4. 对每个 job，若空闲 GPU 数量足够，则选择编号最小的 N 张 GPU。
5. 启动任务并写入 `RUNNING`、`assigned_gpus`、`pid`、`process_group_id` 和日志路径。

`flusso cancel JOB_ID` 只允许取消尚未被调度的任务，也就是状态为 `PENDING` 或 `HELD` 的任务。取消后任务状态会变为 `CANCELLED`，并继续保留在 `flusso ls` 中。已经进入 `RUNNING`、`SUCCEEDED` 或 `FAILED` 的任务不会被这个命令修改。

GPU 被视为空闲需要满足：

- 没有被本调度器的 `RUNNING` job 占用。
- 没有外部 compute process。
- 显存占用低于 1000 MB。
- GPU utilization 低于 10%。

如果 `nvidia-smi` 不可用，调度器会保守地认为没有可调度 GPU。请求 `--gpus 0` 的任务仍可运行。

## 项目结构

```text
flusso/
  cli.py         # Typer CLI
  config.py      # 状态目录和默认配置
  gpu_probe.py   # nvidia-smi 解析和 GPU 空闲判断
  models.py      # Job model 和状态常量
  runner.py      # 进程启动、日志重定向、CUDA_VISIBLE_DEVICES
  scheduler.py   # FIFO 调度循环
  store.py       # SQLite schema 和 job CRUD
tests/           # 单元测试
examples/        # toy examples
doc.md           # 产品和架构设计文档
```

## 配置

默认配置文件位于 `~/.flusso/config.yaml`，初始化时自动生成：

```yaml
gpu_idle:
  memory_threshold_mb: 1000
  utilization_threshold_percent: 10
  required_consecutive_idle_checks: 3
  check_interval_seconds: 5
scheduler:
  interval_seconds: 5
```

当前 MVP 已使用显存和利用率阈值；连续空闲检查字段已保留，后续版本会严格实现连续采样。

## 安全说明

Flusso 会按 shell command 启动用户提交的命令。请只提交可信命令，不要把未审查的外部输入直接传给 `flusso submit`。当前版本不提供容器、cgroups 或权限隔离。
