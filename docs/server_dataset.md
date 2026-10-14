# Ubuntu 22.04 / RTX 4090 训练集生成

适用于 16 vCPU、120 GB RAM、RTX 4090 24 GB，镜像 Python 3.12 / PyTorch 2.8.0 / CUDA 12.8。在仓库根目录运行；先将本次修改同步到服务器。

## 直接运行

```bash
# python 应指向镜像的 Python 3.12；也可以填绝对路径。
export PYTHON_BIN=python
"$PYTHON_BIN" --version

# 先行测试：自动检查/安装依赖，然后实际 GPU 渲染、校验数据、训练前后向检查。
bash shell/generate_landing_dataset_server.sh test

# 正式生成：同样自动检查/安装依赖。
bash shell/generate_landing_dataset_server.sh full
```

| 模式 | 架次/天气 | 每阶段帧数 | 分辨率 | Cycles samples | 总帧数 | train / val / test |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| test | 3 | 2 | 640×480 | 4 | 90 | 30 / 30 / 30 |
| full | 12 | 60 | 1280×960 | 24 | 10,800 | 7,200 / 1,800 / 1,800 |

以上数量针对默认五种天气、三个阶段。test 保留完整架次划分、20 Hz 轨迹、云台动力学、安装位置、独立编码器、雷达与视觉退化；仅减少架次、输出帧数、分辨率和采样数。测试数据用于检查流程和视觉质量，正式训练使用 full。

默认输出分别为 `data/landing_server_test/` 和 `data/landing/`。正式目录与 `configs/landing_training.yaml` 的 `data_root` 一致。为了避免混入旧样本，脚本要求输出目录不存在或为空；重新测试/生成时使用新的 `--output`，脚本不自动删除已有数据，也不做断点续跑。若正式输出改了目录，训练时相应修改 `data_root`。

```bash
# 自定义存储路径、规模；12 架次/天气改为 60 时总量为 54,000 帧。
bash shell/generate_landing_dataset_server.sh full \
  --output /data/landing_v2 --sorties-per-weather 60 --frames-per-stage 60

# 只检查，不安装；仍会执行 CUDA 张量运算和 32×32 的 Cycles 探测渲染。
bash shell/check_server_dependencies.sh --check

# 单独预装环境；生成命令内部也会完成此步骤。
bash shell/check_server_dependencies.sh
```

正式任务可在 `tmux` 中执行；或使用以下命令保存总日志并在断开 SSH 后继续：

```bash
mkdir -p outputs/logs
nohup env PYTHON_BIN=python bash shell/generate_landing_dataset_server.sh full \
  > outputs/logs/landing_server_full.log 2>&1 &
tail -f outputs/logs/landing_server_full.log
```

不要同时运行两次正式生成命令并指向同一目录。输出目录中的 `SUCCESS.json` 表示渲染、索引合并和数据检查全部成功；失败时命令返回非零。`server_run.json` 记录运行参数、设备、耗时及结果；各工作进程的完整 Blender 日志在 `logs/worker_00.log` 等文件，实时帧数在 `workers/*_progress.json`。

## 环境与依赖

- 协调器、轨迹规划、训练接口检查使用镜像 Python 3.12，依赖为 `requirements-server.txt`。符合版本的 PyTorch 2.8.0/cu128 直接复用；缺失、损坏或 CUDA 构建不符时从官方 cu128 wheel 源安装。PyTorch CUDA 运行时检查还会执行实际 GPU 张量运算。
- Blender 4.5.3 LTS 的官方 `bpy` wheel 只有 Python 3.11，没有 cp312 wheel。脚本使用锁定的 uv 自动下载独立 Python 3.11，在 `.venv-render-server/` 安装 `requirements-render-server.txt`。其中没有 PyTorch，也不会将 cp311 包注入 Python 3.12。共享数值依赖见 `requirements-data.txt`。
- 新配置锁定直接依赖，并检查它们的传递依赖约束；不是所有传递依赖的完整锁文件。不会使用旧的 `requirements.txt` / `requirements-blender.txt`，也无需运行旧版 `check_dependencies.sh`、激活旧 `.venv` 或执行 `pip install -e .`。
- 先检查已安装包版本和所需依赖关系；仅在缺失/不符时调用 pip。之后实际导入模块、检查 GPU，并执行小型 Cycles 渲染。首次运行需要联网下载 Python、bpy 及缺少的包；后续已满足依赖时不重复安装。
- 检查 Blender 所需 Linux 动态库。缺失时，具有 root 或免密 sudo 权限的环境自动用 apt 安装；否则打印需要管理员执行的具体 apt 命令。无显示器运行，不需要 Xvfb 或桌面 Blender。
- 使用的 `PYTHON_BIN` 环境应允许 pip 安装依赖，镜像 NVIDIA 驱动必须可用。脚本不会安装或替换宿主机驱动。容器需要 GPU 透传；OptiX 还需要宿主提供 `libnvoptix.so.1` 等图形驱动库。必要时在启动容器时配置 `NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics`；单纯在已启动容器里设置变量不会挂载缺失库。

## 并行与性能参数

默认最多 16 个 CPU 进程生成轨迹，随后使用 2 个独立渲染进程共享一个 GPU，每个分配 8 个 Cycles CPU 线程。线程数根据容器 CPU 配额/可用核数调整。BLAS/OpenMP/OpenCV 小算子使用单线程，避免各进程内部重复开启大量线程。每个工作进程独占一组完整架次，使用独立解释器启动，统一由协调器合并 split/report；随机种子与架次划分不受调度顺序影响。

显式选择 OptiX，禁用混合 CPU 渲染；GPU 不可用即报错，不会静默转为 CPU。渲染开启场景缓存，每架次清理旧场景孤立资源，海面波形批量更新。RGB 保留去噪，OptiX 模式使用 GPU 去噪，再施加原有相机噪声模型；分割标签通道关闭去噪并在输出时二值化。

```bash
# 服务器默认配置，也可以显式指定：
bash shell/generate_landing_dataset_server.sh full \
  --plan-workers 16 --workers 2 --threads 8

# 显存不足或实测单进程更快时：
bash shell/generate_landing_dataset_server.sh test \
  --output /data/landing_test_one_worker --workers 1 --threads 16

# 宿主缺少 OptiX、但 CUDA Cycles 可用时显式选择 CUDA：
bash shell/generate_landing_dataset_server.sh test \
  --output /data/landing_test_cuda --device CUDA

# 另一个终端观察利用率与显存：
watch -n 1 nvidia-smi
```

2 个进程是针对单张 24 GB GPU 的初始配置，并非已在该服务器测出的最优值。CPU 场景更新/雷达射线与 GPU 渲染可以在不同进程交错执行；更多渲染进程会增加显存、上下文切换和 CPU 竞争。可比较测试任务 `server_run.json` 中各进程的 `elapsed_s` 决定使用 1 或 2 个；首次运行还包含内核编译与缓存预热，不应直接当作稳定吞吐率。

## 产物检查

完整数据格式沿用原闭环接口：`rendered/`、`annotations/`（JSON、二值 mask、LiDAR NPZ）、`trajectories/`、`calibration/`、`splits/` 和原有报告。新增 `validation_report.json`、`server_run.json`、`SUCCESS.json` 与五种天气×三个阶段的 `preview.png`。

协调器核对计划中的每一帧和 split，拒绝漏帧、重复帧与遗漏架次。校验器检查所有标注、文件存在性、动态外参、固定安装点、云台时间戳、阶段/天气/划分覆盖。test 解码全部 RGB/mask/LiDAR，full 对每个 split×天气×阶段解码至少一组。最后用实际 `HDataset` 和分割/关键点网络在 CUDA 上执行三个 split 的小型前向、损失与反向计算；这用于接口检查，不是训练精度评估。

本次开发在 macOS 上完成 CPU 并行/回归测试和 Blender 4.5.3 LTS 的实际 CPU 渲染，并对 Linux x86_64 / Python 3.12、3.11 两组依赖进行解析。远程 RTX 4090 的驱动、OptiX 与实际吞吐率需以服务器 test 的结果为准。

正式数据生成完成后，可运行 `bash shell/train_landing_server.sh pretrain` 检查训练，再运行 `bash shell/train_landing_server.sh full` 自动启动 tmux 后台训练。两种模式均保存日志，均不关机，详见 [服务器训练指南](server_training.md)。
