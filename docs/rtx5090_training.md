# Ubuntu 22.04 / RTX 5090 视觉训练

使用 `configs/multi_camera_training_rtx5090.yaml` 训练已有四相机共享网络。输入仍为单张图像，四路相机的样本共享权重；batch=16 表示 16 张图像。三阶段轨迹、天气、遮挡、关键点顺序和完整航次 split 沿用现有多相机数据设计。

## 环境与安装

目标环境为 Ubuntu 22.04、x86_64、Python 3.12 和一张 RTX 5090（32 GB）。`nvidia-smi` 应能识别显卡，使用支持 RTX 5090 和 CUDA 12.8 的 NVIDIA 驱动，例如支持该显卡的 R570 或更新驱动。安装脚本不修改系统驱动。PyTorch wheel 自带所需 CUDA 运行库，本项目训练无需另外安装完整 CUDA Toolkit 或 `nvcc`。

原默认锁中的 PyTorch 2.5.1 / CUDA 12.4 不适合 RTX 5090 的 Blackwell SM 120。本配置与服务器现有版本对齐，单独锁定 PyTorch **2.8.0+cu128** 及其 CUDA 12.8 传递依赖（`nvidia-cuda-runtime-cu12==12.8.90`），其他直接依赖沿用原版本。`requirements.in` 是包元数据约束，实际安装使用完整精确锁 `requirements-rtx5090.txt`。

| 项目 | RTX 5090 profile |
| --- | --- |
| Python | 3.12.x（不固定补丁版本） |
| PyTorch | 2.8.0+cu128 |
| PyTorch CUDA 运行时 | 12.8（运行库包 12.8.90） |

这里检查的是 `torch.version.cuda`，不以系统 `nvcc` 或 `nvidia-smi` 显示的 CUDA 版本代替。服务器已安装 Python 3.12 时，uv 可使用该解释器；安装脚本仍把项目依赖安装到 `.venv`，不会自动继承系统或 Conda 环境里的 PyTorch 包。已存在的 `.venv` 如果满足精确依赖锁则直接复用，否则补齐或更新依赖。

若旧项目 `.venv` 是 Python 3.11，脚本会给出版本错误并停止。确认没有任务在使用该环境后，可先保留备份再重新安装：

```bash
# 仅在已有 .venv 确认为 Python 3.11 时执行
mv .venv ".venv-py311-$(date +%Y%m%d_%H%M%S)"
bash shell/check_dependencies.sh --rtx5090
```

默认 Blender/MATLAB profile 继续使用 Python 3.11 与原依赖锁；RTX 5090 环境后续检查必须带 `--rtx5090`。

在服务器的仓库根目录执行：

```bash
nvidia-smi
bash shell/check_dependencies.sh --rtx5090
bash shell/check_dependencies.sh --rtx5090 --check
```

`--rtx5090` 选择训练环境，自动跳过 Blender 和 MATLAB；脚本创建 `.venv`，安装或纠正缺失/版本不符的依赖。数据已生成时服务器只需此环境。GPU 预检查核对 Python 3.12、PyTorch 2.8.0、PyTorch CUDA 运行时 12.8、SM 120、BF16，并以 64×64 图像实际执行网络前向、损失、反向及 fused AdamW 更新。缺少 GPU、驱动不可用或装错 PyTorch 会报错。之后检查和补装也应使用 `--rtx5090`，避免默认依赖锁重新安装 PyTorch 2.5.1。需要 Blender/MATLAB 时使用独立环境。

首次安装优先复用可用的 `uv`；若没有，则通过已有 Python 的 pip 从 PyPI 安装到项目 `.tools/uv-bootstrap`，不修改系统 Python，也不需要把 uv 加入 PATH。pip 不可用或安装失败时，回退到带连接超时、TLS 中断重试的 GitHub release 下载。Ubuntu 若尚未安装 pip，可先执行：

```bash
sudo apt update
sudo apt install -y python3-pip ca-certificates
```

原先的 `curl: (56) ... unexpected eof` 表示网络连接被中断；脚本现在增加了另一条下载路径和重试，不能保证绕过服务器的网络限制。它继承 `HTTPS_PROXY`、`PIP_INDEX_URL` 等设置，不关闭 TLS 校验。PyTorch CUDA wheel 仍从 requirements 中的官方源安装。

如果系统没有 Python 3.12，uv 还需从 GitHub 下载 Python；若该步骤受阻，配置可用代理，或通过 `LANDING_PYTHON` 指定已经安装的 3.12 解释器：

```bash
LANDING_PYTHON=/path/to/python3.12 bash shell/check_dependencies.sh --rtx5090
```

可以单独检查 GPU，或选择其他显卡编号：

```bash
.venv/bin/python -m visual_training.check_gpu
.venv/bin/python -m visual_training.check_gpu --device cuda:1
```

## tmux 后台安装与训练

服务器已安装 tmux 时，在仓库根目录运行以下入口即可。依赖安装、GPU 检查、训练按顺序执行；任何前置步骤失败都不会启动训练：

```bash
bash shell/train_rtx5090.sh
```

默认创建后台会话 `landing_train`，SSH 断开后任务继续。脚本输出本次日志路径，安装及训练输出均写入 `outputs/logs/rtx5090_*.log`，结束后相邻的 `.log.exit_code` 文件保存退出码（0 为成功）。日志每次自动使用新文件；训练权重仍写入训练配置的输出目录。重复启动同名会话会被拒绝，不会并行启动第二份训练。

```bash
tmux attach -t landing_train
```

查看时按 `Ctrl+b`，松开后按 `d`，即可离开会话并保持任务运行。任务完成或失败后，会话自动结束，此时查看启动脚本输出的日志路径。tmux 保持 SSH 断开后的进程运行，不提供服务器重启后的自动恢复；现有训练入口也没有断点续训参数。

可以把数据路径、批量、轮数等训练参数放在 `--` 后：

```bash
bash shell/train_rtx5090.sh -- \
  --data-root /data/visualizedLanding/multi_camera --batch-size 16
```

需要在当前终端排查时，使用 `bash shell/train_rtx5090.sh --foreground`，仍会保存日志。`--session NAME` 可指定会话名称，`--log PATH` 可指定日志文件（追加写入）。后台入口将当前 SSH 会话的代理、包源、Python 路径和 `CUDA_VISIBLE_DEVICES` 等设置传入任务，避免已有 tmux 服务保留过期环境。

## 数据迁移

将完整 `data/multi_camera/` 复制到服务器，包括 `rendered/`、`annotations/`、`splits/` 及标定等索引。不要只复制图片，也不要重新随机划分相邻帧。训练/评估加载器按指定数据根目录重新定位 `annotations/` 内的标注，兼容现有 split 中保存的 Mac 绝对路径，也支持相对路径；原航次归属和 split 文件内容保持不变。

默认路径是仓库内 `data/multi_camera`。若数据位于独立磁盘，通过 `--data-root` 指定绝对路径：

```bash
.venv/bin/python -m visual_training.train \
  --config configs/multi_camera_training_rtx5090.yaml \
  --data-root /data/visualizedLanding/multi_camera
```

## 训练参数

| 参数 | 初始值 | 用途 |
| --- | --- | --- |
| 图像尺寸 | 512×512 | 保持现有训练尺度 |
| batch size | 16 | 单卡起始批量，需要在服务器实测显存 |
| epoch / learning rate | 40 / 0.001 | 沿用现有优化设置 |
| device / precision | cuda:0 / BF16 | 卷积混合精度；模型参数、关键点坐标和损失使用 FP32 |
| CPU threads / loader workers | 8 / 8 | 并行读取、缩放和光度增强，worker 随机流独立 |
| multiprocessing context | spawn | 避免在 CUDA 初始化后 fork 子进程 |
| pin memory / persistent workers / prefetch | true / true / 2 | 异步向 GPU 传输并减少等待 |
| channels last / cuDNN benchmark / TF32 | true / true / true | 固定输入尺寸下启用 GPU 优化 |
| optimizer | fused AdamW | 减少优化器启动开销 |

BF16 不使用 FP16 的梯度缩放；若通过 CLI 改为 FP16，训练器会启用 GradScaler。空间 softargmax、分割 Dice 归约和关键点监督使用 FP32，避免低精度坐标网格及大图像的 FP16 求和溢出。此设置保留网络与 checkpoint 接口，不改变导航的 PnP/LiDAR/ESKF 定义。

训练日志包含设备、PyTorch/CUDA 版本、每轮耗时和 CUDA 峰值已分配显存；`history.json` 保存在权重目录。默认输出 `visual_training/checkpoints/multi_camera/{best,last}.pt`，与多相机导航的权重位置一致。随机种子固定数据顺序和增强，但 cuDNN benchmark/TF32 不承诺逐位可复现。

batch=16 是起点，尚未在实际 RTX 5090 上测量吞吐或完整训练显存。若显存不足，先改为 batch=8；若 CPU/存储跟不上、容器共享内存不足或加载进程异常，可降低 worker 数。示例：

```bash
.venv/bin/python -m visual_training.train \
  --config configs/multi_camera_training_rtx5090.yaml \
  --batch-size 8 --num-workers 4
```

服务器 CPU 至少应能提供配置所需的线程和内存；CPU 较少时同时调整 YAML 中的 `threads`。显卡有余量时可实测 batch=24/32，暂不自动增大学习率。512 缩放后中距 H 仍可能很小，GPU 加速不替代小目标定位精度评估。

## 评估与导出

```bash
.venv/bin/python -m visual_training.evaluate \
  --root data/multi_camera \
  --checkpoint visual_training/checkpoints/multi_camera/best.pt \
  --image-size 512 --device cuda:0 \
  --output outputs/metrics/multi_camera_vision.json

.venv/bin/python -m visual_training.export_model \
  --checkpoint visual_training/checkpoints/multi_camera/best.pt \
  --output visual_training/checkpoints/multi_camera/h_detector.onnx \
  --image-size 512
```

数据迁移到独立磁盘时，评估的 `--root` 与训练的 `--data-root` 应一致。评估使用 FP32，按阶段/天气/相机输出分组指标。ONNX 导出在 CPU 执行；此环境锁定的是 CPU `onnxruntime`，GPU 训练通过 PyTorch CUDA 完成。训练权重保存为 FP32，可供原有 CPU/MPS/PyTorch 导航加载，也可导出为原有三输出 ONNX。

ONNX 导出显式选择 `dynamo=False`，保持现有 opset 17、动态 batch 和三个输出的接口。服务器上可以运行针对训练和导出的回归测试（会生成临时小数据，不覆盖正式权重）：

```bash
.venv/bin/python -m pytest -q tests/test_gpu_training.py tests/test_training_shell.py
```

测试包含版本拒绝、BF16 损失/梯度、spawn 数据加载、1 epoch 训练、权重重载评估、PyTorch/ONNX 数值比对，以及有 RTX 5090 时的真实 CUDA 前向/反向更新。完整分辨率显存和吞吐仍由正式训练日志确认。

## 依据与验证边界

- [PyTorch 历史版本安装说明](https://pytorch.org/get-started/previous-versions/)：2.8.0 的 cu128 官方源。
- [CUDA 12.8 发布说明](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html)：SM 120 支持及驱动要求。

本机为 macOS，无 NVIDIA CUDA 设备。已在独立 Python 3.12.14 / PyTorch 2.8.0 环境完成完整回归：66 项通过、1 项 NVIDIA 硬件测试跳过，包含 CPU BF16 训练、并行加载、权重重载评估与 ONNX 数值比对。Linux / Python 3.12 的 49 个锁定包通过 binary-only 安装 dry-run。RTX 5090 的内核、驱动、完整分辨率显存和吞吐需在目标服务器执行上述检查与训练。详细结果见 [验证记录](validation.md)。
