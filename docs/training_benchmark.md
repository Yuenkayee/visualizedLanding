# 服务器训练压力测试与参数选择

此工具针对本项目的真实数据加载、原图光度增强、512 输入缩放、分割/关键点网络、损失、反向传播、梯度裁剪、AdamW 更新和训练指标统计进行测试。训练器和测试器共用 `process_batch`，验证阶段也单独计时。支持 RTX 5090、RTX PRO 6000 Blackwell；沿用当前 Python 3.12 / PyTorch 2.8.0+cu128 环境，无新增 Python 依赖。

工具选择实际吞吐量较高且通过重复测试的参数。显存使用率不需要达到 100%；GPU 利用率高也不自动代表完整训练更快。结果只代表本次数据、候选范围、设备负载及测试时长，不保证是全局最优参数或长时间训练的稳定性结论。

## 在服务器运行

先结束同卡上的训练/渲染，再在仓库根目录启动。其他任务争用 GPU、CPU 或磁盘会影响结果；脚本不停止其他任务。

```bash
bash shell/benchmark_training.sh -- \
  --checkpoint visual_training/checkpoints/multi_camera/best.pt \
  --output outputs/benchmarks/server_tuning

tmux attach -t landing_benchmark
```

默认后台会话为 `landing_benchmark`，按 `Ctrl+b` 后按 `d` 离开。启动前自动检查/安装缺少的项目依赖并进行 GPU 检查，使用当前 Python，不创建 `.venv`。可在 `--` 前传入 `--python /root/miniconda3/bin/python`、`--session NAME`、`--foreground`、`--log PATH`。主日志是 `outputs/logs/benchmark_*.log`，相邻 `.exit_code` 保存返回码。

默认以 `configs/multi_camera_finetune_rtx5090.yaml` 为基础。若评估初次训练参数，追加 `--config configs/multi_camera_training_rtx5090.yaml`；外置数据盘可追加 `--data-root /实际路径/multi_camera`。测试只读取 train/val，test split 保留作最终模型评价。`--checkpoint` 可省略，此时每组从同一随机初始化开始测速度；使用微调配置正式训练时仍须提供初始权重。

每组模型和优化器仅在测试子进程中更新，不写入训练 checkpoint。已有权重、原始配置、训练图片和 split 文件不会被修改。结果目录必须是新目录；重跑请使用另一个 `--output`。

## 默认搜索与复测

- 先测基础配置，然后扫描 batch size **8、16、24、32、48、64** × num_workers **4、8、12、16**，默认共 24 组（基础参数已包含在其中）。CPU threads 沿用基础配置的 8，prefetch_factor 默认 2。
- 每组在独立进程中运行，训练和验证分别先预热 10 batch；筛选阶段计时训练 40 batch、验证 10 batch。
- 将筛选中最快的 3 组、原始参数，以及保持原 batch size 时最快的组合纳入复测，去重后最多 5 组。每组再独立运行 **2 次**，每次计时训练 120 batch、验证 30 batch。
- 样本在各自 split 中使用固定种子的随机顺序，覆盖不同航次而非只测试首个航次。样本不足时重复取样，并在 `dataset_restarts` 标明；小样本结果不能代表完整数据的 IO 性能。预取严格止于测试窗口，避免训练队列残留影响验证计时。
- 显存 OOM、非有限损失/参数、加载失败、超时的候选不参与推荐。默认每组超时 900 秒，终止该组及其数据加载进程后继续。主报告每组完成后更新，Ctrl+C 中止时保留已完成的记录。
- 默认要求 PyTorch 峰值 reserved 显存不超过测试前可用显存的 **85%**。设备总显存、测试前可用显存、allocated/reserved 峰值均保留在报告中。
- 复测两次的预计 epoch 时间相对极差须不超过 15%。按其中位数选择；相差不超过 3% 时优先较小 batch、较少 worker/预取/线程。没有稳定候选时返回非零，不输出推荐 YAML。

默认搜索最多约 34 次独立测量，耗时取决于实测 batch 速度、进程启动和候选失败情况。主日志每组打印开始和结果，长时间未结束时每 30 秒打印存活状态及该组日志位置。

可先缩小范围检查流程：

```bash
bash shell/benchmark_training.sh --foreground -- \
  --checkpoint visual_training/checkpoints/multi_camera/best.pt \
  --output outputs/benchmarks/quick_check \
  --batch-sizes 16 32 --workers 8 12 \
  --warmup 5 --steps 20 --val-steps 5 --confirm-steps 60 --top-k 2
```

在 RTX PRO 6000 96 GB 或筛选结果仍随 batch 增大而提速时，扩展候选；列表做笛卡尔积，组合越多耗时越长：

```bash
bash shell/benchmark_training.sh -- \
  --checkpoint visual_training/checkpoints/multi_camera/best.pt \
  --output outputs/benchmarks/extended_tuning \
  --batch-sizes 32 48 64 96 128 --workers 8 12 16 \
  --threads 4 8 --prefetch-factors 1 2
```

原始参数始终额外纳入比较。使用 `--repeats 3 --confirm-steps 300` 可增加复测长度；遇到较慢存储可以提高 `--trial-timeout`。不要为了消除 OOM 报错而自动修改图像分辨率或精度，这两项保持基础配置设置。

## 结果与正式训练

结果目录包含：

| 文件 | 用途 |
| --- | --- |
| `report.json` | 全部候选、失败原因、计时、遥测原始采样和推荐参数 |
| `recommended.yaml` | 已测试且稳定候选中的较快配置，可能改变 batch size |
| `recommended_same_batch.yaml` | 保持原 batch size 的推荐，适合当前正在排查收敛的微调任务；无合格候选时不生成 |
| `screen_*/`、`confirm_*/` | 每组 `spec.json`、`result.json`、`trial.log` |

推荐配置保留网络结构、图像尺寸、精度、学习率、weight decay、scheduler、早停、epoch 上限及数据 split。若指定了 `--checkpoint`，自动填入 `init_checkpoint`。正式权重写入报告目录下新的 `training_checkpoints/` 或 `training_checkpoints_same_batch/`。

当前任务建议先用保持原 batch size 的配置：

```bash
bash shell/train_rtx5090.sh --session landing_finetune_tuned -- \
  --config outputs/benchmarks/server_tuning/recommended_same_batch.yaml
```

要采用允许改变 batch size 的推荐，则将配置改为 `recommended.yaml`。改变 batch size 会改变每轮优化器更新次数、梯度噪声和训练轨迹，工具不自动线性放大学习率；需观察验证损失、关键点误差及 IoU，再决定是否采用。压力测试过程中的模型更新仅用于测性能，不是已完成的正式训练。

## 指标的含义

| 字段 | 解释 |
| --- | --- |
| `train/val.images_per_second` | 数据加载、传输、计算、指标累计在内的完整阶段吞吐量 |
| `estimated_epoch_s` | train 图片数 / 训练吞吐量 + val 图片数 / 验证吞吐量；不含 epoch 启动、周期打印和 checkpoint 写盘，属于估计值 |
| `loader_wait_mean_ms/p95_ms` | 主进程取 batch 的阻塞时间，含图片读取/解码/增强跟不上的等待 |
| `loader_wait_wall_fraction` | 上述等待 / 测量墙钟时间；异步 CUDA 可能同时工作，因此不能当作 GPU 空闲比例 |
| `cuda_step_mean_ms` | CUDA events 覆盖传输、训练/验证步骤和指标计算的平均跨度，可能含 CPU 提交间隙；不等于纯 kernel 耗时 |
| `cuda_peak_allocated/reserved_gib` | 覆盖预热、训练、验证的 PyTorch 显存峰值 |
| `telemetry.summary.gpu_util_percent` | nvidia-smi 定时采样的 GPU 利用率；还记录功耗、温度、设备总已用显存 |
| `allocated_cpu_percent` | 此测试进程组消耗的 CPU 核数 / 可用 CPU 配额，含数据加载子进程 |
| `process_group_rss_gib` | 进程组 RSS 之和，共享页可能重复计数，不能视作精确物理内存占用 |
| `available_memory_gib` | Linux MemAvailable，并在可获取时结合 cgroup v2 内存限制 |

计时仅在预热结束和测量结束时同步 CUDA，不逐 batch 强制同步。遥测覆盖预热、训练和验证，约每秒采样，不能用单次 0% 判定 GPU 没有工作。nvidia-smi 缺失或遥测失败会记录错误，PyTorch 计时仍继续；GPU 通过 UUID 匹配，避免 `CUDA_VISIBLE_DEVICES` 重映射后监控错卡。非 Linux CPU 回归测试可以没有 CPU/内存遥测。

等待时间较长且增加 worker 后吞吐提高，说明数据准备值得优化；增加 worker 无收益时应检查存储、CPU 配额及 `/dev/shm`。容器出现 DataLoader bus error / shared memory 错误，应查看该组日志并增大容器共享内存或降低 worker、prefetch、batch；工具不会修改系统/容器限制。大部分显存空闲但增加 batch 不提速时，继续占用显存没有性能收益。

复测顺序随机化并重新测试基准，以降低文件缓存和温度造成的顺序偏差。结果主要反映预热后的训练性能；存储缓存、其他租户负载、GPU 频率变化仍可能影响测量。

## 本地验证

```bash
python -m pytest -q tests/test_training_benchmark.py tests/test_training_shell.py \
  tests/test_training_refinement.py tests/test_gpu_training.py
```

包含真实小数据 CPU/BF16 子进程测试、吞吐/标注读取、原 checkpoint 不被修改、OOM 记录、超时终止、稳定候选选择、双配置导出、tmux 参数传递和训练步骤回归。服务器性能参数必须由目标机器实测，本地 CPU 测试不产生 RTX 5090 的性能结论。

本次在 macOS / Python 3.12.14 / PyTorch 2.8.0 下连同 `test_training_pipeline.py` 验证上述范围，59 项通过、1 项 NVIDIA 硬件测试跳过；静态检查和 shell 语法检查通过。
