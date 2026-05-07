# Flusso：单机多 GPU 任务调度器设计文档

## 1. 项目背景

在共享 GPU 服务器上，用户经常需要运行多个训练、评估、采样或数据处理任务。每个任务可能需要不同数量的 GPU，并且任务之间常常存在先后依赖关系，例如：

```text
preprocess -> train -> evaluate -> visualize
```

传统做法通常依赖手动指定 `CUDA_VISIBLE_DEVICES`、`tmux`、`nohup` 或 shell 脚本。这种方式存在几个问题：

* 不容易排队；
* 不容易表达任务依赖；
* GPU 空闲时不能自动启动任务；
* 手动选择 GPU 容易和其他用户冲突；
* 训练失败后，下游 evaluate 任务可能被错误执行；
* 日志、状态、重试、取消等操作缺乏统一管理。

本项目命名为 **Flusso**。Flusso 旨在实现一个**单机多 GPU 的轻量级个人任务调度器**，用于管理用户自己的训练任务队列。它不试图强制管理整个服务器，也不要求其他用户必须使用本工具，而是通过观察真实 GPU 状态，尽量避免与外部手动运行的任务冲突。

“Flusso” 来自意大利语，意为“流动”。这个名字强调本工具希望管理的是实验任务之间自然、有序的流动：任务在资源不足时等待，在 GPU 空闲时启动，并按照依赖关系从训练流向评估、采样和可视化。

---

## 2. 项目目标

本工具的核心目标是：

> 在一台共享的多 GPU Linux 服务器上，允许用户提交命令行任务，并指定所需 GPU 数量；调度器根据真实 GPU 空闲状态自动启动任务，同时支持任务依赖关系，避免训练、评估等步骤顺序错误。

具体目标包括：

1. **命令行任务提交**

   * 任务以 shell command 形式给出；
   * 调度器不关心任务内部是 PyTorch、JAX、TensorFlow、CUDA 程序还是普通 shell 脚本。

2. **GPU 数量申请**

   * 每个任务可以声明需要多少张 GPU；
   * 调度器在启动任务时自动选择空闲 GPU；
   * 通过 `CUDA_VISIBLE_DEVICES` 将 GPU 分配给任务。

3. **外部 GPU 占用感知**

   * 不要求其他用户通过本工具提交任务；
   * 调度器需要检测 `nvidia-smi` 中已有的外部 GPU 进程；
   * 已被其他用户或外部进程占用的 GPU 不应被分配。

4. **任务依赖 / 任务组**

   * 支持将多个任务组织为一个 pipeline；
   * 支持 `train -> evaluate` 这类顺序依赖；
   * 只有上游任务成功后，下游任务才允许启动。

5. **日志与状态管理**

   * 记录任务状态；
   * 保存 stdout / stderr；
   * 支持查看任务日志；
   * 支持查看队列状态和 GPU 状态。

6. **任务控制**

   * 支持取消任务；
   * 支持重试失败任务；
   * 支持暂停 / 恢复 pending 任务。

---

## 3. 非目标

为了保持项目简单，第一阶段不考虑以下能力：

1. **多机调度**

   * 本项目只考虑单台多 GPU 机器。

2. **强制资源隔离**

   * 本工具不会阻止其他用户手动运行任务；
   * 也不会通过 cgroups、容器或权限系统强制限制其他用户。

3. **完整公平调度策略**

   * 不实现 Slurm 级别的多用户公平队列；
   * 第一版主要面向个人或小团队自用。

4. **自动抢占 / 迁移任务**

   * 已经运行的任务不会被迁移到其他 GPU；
   * 不支持 checkpoint-aware preemption。

5. **复杂 Web UI**

   * 第一版只实现 CLI；
   * Web UI 可作为未来扩展。

6. **自动理解训练脚本**

   * 调度器不解析 Python 代码；
   * 不自动识别 batch size、显存需求、模型大小等。

---

## 4. 设计原则

### 4.1 非侵入式

工具不要求改动用户训练代码。所有任务都以 command 形式提交，例如：

```bash
flusso submit --gpus 2 -- python train.py --config configs/exp001.yaml
```

调度器只负责启动进程，并注入环境变量：

```bash
CUDA_VISIBLE_DEVICES=2,3 python train.py --config configs/exp001.yaml
```

### 4.2 保守调度

在共享服务器上，误把被别人使用的 GPU 当作空闲 GPU 会造成严重冲突。因此调度器应采用保守策略：

* 如果 GPU 上存在外部 compute process，则视为不可用；
* 如果 GPU 显存占用超过阈值，则视为不可用；
* 如果 GPU utilization 持续较高，则视为不可用；
* 如果状态不确定，默认不调度到该 GPU。

### 4.3 依赖优先于资源

一个任务能否启动，需要同时满足两个条件：

```text
所有依赖任务成功完成 + GPU 资源足够空闲
```

即使 GPU 空闲，如果依赖任务尚未完成，也不能启动。

### 4.4 任务与实现解耦

调度器只理解以下抽象：

* command；
* working directory；
* environment variables；
* required GPU count；
* dependencies；
* status；
* logs。

调度器不关心具体训练框架。

---

## 5. 核心概念

## 5.1 Job

Job 是调度器管理的最小执行单元。

一个 Job 包含：

```text
id
name
command
gpu_required
status
dependencies
assigned_gpus
working_directory
environment_variables
created_at
started_at
ended_at
exit_code
log_path
pid
process_group_id
```

### Job 状态

```text
PENDING     等待调度
RUNNING     正在运行
SUCCEEDED   成功结束，exit code = 0
FAILED      失败结束，exit code != 0
CANCELLED   被用户取消
HELD        被暂停，不参与调度
SKIPPED     因上游失败而跳过
```

状态转换示意：

```text
PENDING -> RUNNING -> SUCCEEDED
PENDING -> RUNNING -> FAILED
PENDING -> CANCELLED
PENDING -> HELD -> PENDING
PENDING -> SKIPPED
RUNNING -> CANCELLED
FAILED  -> PENDING    # retry
SKIPPED -> PENDING    # retry / retry cascade
```

---

## 5.2 Job Group

Job Group 表示一组具有依赖关系的任务，例如一个实验 pipeline。

示例：

```yaml
name: exp001

jobs:
  train:
    gpus: 4
    cmd: python train.py --config configs/exp001.yaml

  eval:
    gpus: 1
    needs: [train]
    cmd: python eval.py --ckpt outputs/exp001/latest.pt

  sample:
    gpus: 1
    needs: [train]
    cmd: python sample.py --ckpt outputs/exp001/latest.pt
```

这里 `eval` 和 `sample` 都依赖 `train`。如果 `train` 失败，默认情况下 `eval` 和 `sample` 不会运行。

---

## 5.3 GPU State

调度器维护两类 GPU 状态：

### 内部状态

由本调度器启动的任务占用了哪些 GPU。

例如：

```text
GPU 0 -> job 12
GPU 1 -> job 12
GPU 2 -> free
GPU 3 -> job 15
```

### 外部观测状态

通过 `nvidia-smi` 或 NVML 观察到的真实 GPU 使用情况。

例如：

```text
GPU 0 -> my job 12
GPU 1 -> my job 12
GPU 2 -> external process by user alice
GPU 3 -> my job 15
```

最终不可用 GPU 是二者的并集：

```text
unavailable_gpus = internally_assigned_gpus ∪ externally_occupied_gpus
```

---

## 6. GPU 空闲判定策略

一张 GPU 被认为可用，需要满足以下条件：

```text
1. 没有被本调度器已运行任务占用；
2. 没有外部 compute process；
3. 显存占用低于阈值；
4. GPU utilization 低于阈值；
5. 连续多次采样均满足空闲条件。
```

推荐默认参数：

```yaml
gpu_idle:
  memory_threshold_mb: 1000
  utilization_threshold_percent: 10
  required_consecutive_idle_checks: 3
  check_interval_seconds: 5
```

这意味着一张 GPU 需要连续约 15 秒处于空闲状态，才会被认为可用于新任务。

---

## 7. 调度策略

### 7.1 调度循环

调度器 daemon 周期性执行：

```python
while True:
    refresh_running_jobs()
    gpu_state = query_gpu_state()
    runnable_jobs = find_runnable_pending_jobs()

    for job in order_jobs(runnable_jobs):
        free_gpus = get_current_free_gpus(gpu_state)
        if len(free_gpus) >= job.gpu_required:
            assigned = select_gpus(free_gpus, job.gpu_required)
            launch_job(job, assigned)

    sleep(schedule_interval)
```

### 7.2 Runnable Job 条件

一个 pending job 可以进入调度候选队列，当且仅当：

```text
- status == PENDING
- 所有 dependencies 的状态都是 SUCCEEDED
- job 没有被 hold
```

如果任一依赖失败，默认将下游任务标记为 `SKIPPED`。

### 7.3 排序策略

第一版可以采用简单的 FIFO：

```text
created_at 越早，优先级越高
```

未来可以扩展：

```text
priority
shortest job first
smallest GPU count first
fair-share by user
deadline-aware scheduling
```

### 7.4 GPU 选择策略

第一版可以采用简单策略：

```text
选择编号最小的 N 张空闲 GPU
```

例如：

```text
free_gpus = [1, 3, 5, 6]
job.gpu_required = 2
assigned = [1, 3]
```

未来可以扩展：

* 尽量选择连续 GPU；
* 按 NVLink topology 选择；
* 避免温度过高的 GPU；
* 避免显存碎片较多的 GPU。

---

## 8. 任务启动方式

调度器启动任务时，应使用独立 process group，便于取消整个任务树。

示例：

```python
subprocess.Popen(
    command,
    shell=True,
    cwd=working_directory,
    env={
        **os.environ,
        **job_env,
        "CUDA_VISIBLE_DEVICES": "2,3",
    },
    stdout=log_file,
    stderr=subprocess.STDOUT,
    preexec_fn=os.setsid,
)
```

取消任务时：

```python
os.killpg(process_group_id, signal.SIGTERM)
```

如果一段时间后仍未退出，再发送：

```python
os.killpg(process_group_id, signal.SIGKILL)
```

---

## 9. CLI 设计

命令名确定为：`flusso`

### 9.1 初始化

```bash
flusso init
```

创建本地状态目录，例如：

```text
~/.flusso/
  flusso.db
  logs/
  config.yaml
```

### 9.2 启动 daemon

```bash
flusso daemon start
flusso daemon stop
flusso daemon status
```

也可以支持前台运行，方便调试：

```bash
flusso daemon run
```

### 9.3 提交单个任务

```bash
flusso submit --gpus 2 --name exp001-train -- python train.py --config configs/exp001.yaml
```

可选参数：

```bash
--cwd PATH
--env KEY=VALUE
--priority N
--hold
--pool 0,1,2,3
```

### 9.4 提交任务组

```bash
flusso submit-group exp001.yaml
```

配置文件示例：

```yaml
name: exp001

defaults:
  cwd: /home/user/projects/my_project
  env:
    WANDB_PROJECT: pointcloud-generation

jobs:
  train:
    gpus: 4
    cmd: python train.py --config configs/exp001.yaml

  eval:
    gpus: 1
    needs: [train]
    cmd: python eval.py --ckpt outputs/exp001/latest.pt

  sample:
    gpus: 1
    needs: [train]
    cmd: python sample.py --ckpt outputs/exp001/latest.pt
```

### 9.5 查看队列

```bash
flusso ls
```

输出示例：

```text
ID   NAME          STATUS    GPUS  ASSIGNED  DEPENDS  CREATED
12   exp001-train  RUNNING   4     0,1,2,3   -        10:20
13   exp001-eval   PENDING   1     -         12       10:21
14   exp002-train  PENDING   2     -         -        10:25
```

### 9.6 查看 GPU 状态

```bash
flusso status
```

输出示例：

```text
GPU  STATUS     OWNER     PID     MEMORY     UTIL   COMMAND
0    MY_JOB     me        12345   22000MiB   96%    exp001-train
1    MY_JOB     me        12345   21900MiB   97%    exp001-train
2    EXTERNAL   alice     23210   18000MiB   91%    python train_other.py
3    FREE       -         -       80MiB      0%     -
```

### 9.7 查看任务详情

```bash
flusso show 12
```

显示：

```text
id
name
status
command
gpus
assigned_gpus
dependencies
cwd
env
created_at
started_at
ended_at
exit_code
log_path
```

### 9.8 查看日志

```bash
flusso logs 12
flusso logs 12 -f
```

### 9.9 取消任务

```bash
flusso cancel 12
```

如果取消一个 pending 任务，则直接标记为 `CANCELLED`。

如果取消一个 running 任务，则向其 process group 发送终止信号。

### 9.10 重试任务

```bash
flusso retry 12
```

将失败或取消的任务重新置为 `PENDING`。

也可以支持级联重试：

```bash
flusso retry 12 --cascade
```

用于重试某个失败的上游任务以及被跳过的下游任务。

---

## 10. 配置文件

全局配置文件：

```yaml
# ~/.flusso/config.yaml

gpu_pool: [0, 1, 2, 3, 4, 5, 6, 7]

scheduler:
  interval_seconds: 5
  policy: fifo

gpu_idle:
  memory_threshold_mb: 1000
  utilization_threshold_percent: 10
  required_consecutive_idle_checks: 3
  check_interval_seconds: 5

logging:
  log_dir: ~/.flusso/logs

cancel:
  sigterm_timeout_seconds: 20
```

---

## 11. 存储设计

第一版建议使用 SQLite。

优点：

* 无需额外服务；
* 适合单机；
* 易于查询；
* 易于备份；
* Python / Rust / Go 都有成熟支持。

### 11.1 jobs 表

```sql
CREATE TABLE jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    group_id INTEGER,
    name TEXT,
    command TEXT NOT NULL,
    gpu_required INTEGER NOT NULL,
    status TEXT NOT NULL,
    assigned_gpus TEXT,
    cwd TEXT,
    env_json TEXT,
    pid INTEGER,
    process_group_id INTEGER,
    exit_code INTEGER,
    log_path TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    ended_at TEXT
);
```

### 11.2 dependencies 表

```sql
CREATE TABLE dependencies (
    job_id INTEGER NOT NULL,
    depends_on_job_id INTEGER NOT NULL,
    PRIMARY KEY (job_id, depends_on_job_id)
);
```

### 11.3 groups 表

```sql
CREATE TABLE groups (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL
);
```

---

## 12. MVP 开发计划

### Phase 1: 最小可运行版本

目标：能提交任务、排队、分配空闲 GPU、运行 command。

功能：

* `flusso init`
* `flusso daemon run`
* `flusso submit --gpus N -- command`
* `flusso ls`
* `flusso logs`
* `flusso cancel` pending / held 任务
* SQLite 存储
* 基于 `nvidia-smi` 的 GPU 空闲检测
* `CUDA_VISIBLE_DEVICES` 注入

### Phase 2: 任务依赖和任务组

目标：支持实验 pipeline。

功能：

* `flusso submit-group pipeline.yaml`
* `needs` 依赖；
* 上游失败时下游 `SKIPPED`；
* `flusso retry --cascade`。

### Phase 3: 任务控制与鲁棒性

目标：提升可用性。

功能：

* `flusso cancel` running 任务
* 进程组管理；
* daemon crash 后恢复 running jobs 状态；
* 更清晰的错误状态；
* 外部进程展示。

### Phase 4: 使用体验优化

目标：让工具更适合日常实验。

功能：

* `flusso status`
* `flusso show`
* `flusso logs -f`
* 任务优先级；
* GPU pool；
* 配置文件；
* shell completion。

---

## 13. 关键风险与边界情况

### 13.1 外部用户抢占 GPU

本工具无法阻止其他用户手动使用同一张 GPU。因此只能保证：

```text
本工具不会主动调度到已经被检测为占用的 GPU。
```

但无法保证：

```text
别人不会在本工具启动任务之后，再手动占用同一张 GPU。
```

### 13.2 `nvidia-smi` 状态不完全可靠

某些进程可能短时间内占用 GPU，但采样时未被检测到。因此需要连续多次采样，并采用保守阈值。

### 13.3 子进程泄漏

训练脚本可能启动多个子进程。取消任务时必须终止整个 process group，而不是只 kill 父进程。

### 13.4 Daemon 崩溃恢复

如果 daemon 崩溃，已经运行的任务可能仍在执行。重启 daemon 后，需要根据 pid 和 process group 检查任务是否仍然存活。

---

## 14. 技术栈决策

第一版确定使用 **Python** 实现。

选择 Python 的原因：

* 开发速度快，适合快速迭代 MVP；
* 方便实现 CLI、YAML 配置、SQLite 存储和日志管理；
* 方便调用 `nvidia-smi` 或后续接入 NVML；
* 更适合作为个人 / 小团队实验基础设施；
* 目标用户通常熟悉 Python，便于后续修改、扩展和贡献。

### 14.1 MVP 推荐依赖

MVP 阶段优先使用：

```text
Typer        构建 CLI
Rich         输出表格、状态和日志片段
PyYAML       解析 pipeline 配置
sqlite3      使用 Python 标准库管理状态数据库
subprocess   启动任务
psutil       管理进程和进程组
nvidia-smi   第一版 GPU 状态检测
```

第一版可以先通过 `nvidia-smi --query-gpu` 和 `nvidia-smi --query-compute-apps` 获取 GPU 状态，避免过早引入 NVML 绑定。后续如果需要更稳定、更低开销的 GPU 查询，再接入 `pynvml`。

### 14.2 暂不使用 Rust

Rust 适合后续重写 daemon 或发布单 binary，但第一版不采用 Rust。当前优先目标是尽快获得一个可用、可验证、容易修改的调度器原型。

---

## 15. 项目一句话介绍

> Flusso is a lightweight, non-invasive GPU job scheduler for shared single-node servers. It watches available GPUs, queues command-based jobs, and runs experiment flows in dependency order.

中文版本：

> Flusso 是一个用于共享单机多 GPU 服务器的轻量级任务调度器，支持命令行任务提交、GPU 空闲检测、自动排队、任务依赖和实验 pipeline 管理。
