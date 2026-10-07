# Ubuntu 22.04 / RTX 5090 视觉训练

使用 `configs/multi_camera_training_rtx5090.yaml` 训练已有四相机共享网络。输入仍为单张图像，四路相机的样本共享权重；batch=16 表示 16 张图像。三阶段轨迹、天气、遮挡、关键点顺序和完整航次 split 沿用现有多相机数据设计。

## 环境与安装

目标环境为 Ubuntu 22.04、x86_64、Python 3.11 和一张 RTX 5090（32 GB）。`nvidia-smi` 应能识别显卡，使用支持 RTX 5090 和 CUDA 12.8 的 NVIDIA 驱动，例如支持该显卡的 R570 或更新驱动。安装脚本不修改系统驱动。PyTorch wheel 自带所需 CUDA 运行库，本项目训练无需另外安装完整 CUDA Toolkit 或 `nvcc`。

原默认锁中的 PyTorch 2.5.1 / CUDA 12.4 不适合 RTX 5090 的 Blackwell SM 120。本配置单独锁定 PyTorch **2.7.1+cu128** 及其 CUDA 12.8 传递依赖，其他直接依赖沿用原版本。`requirements.in` 是包元数据约束，实际安装使用完整精确锁 `requirements-rtx5090.txt`。

在服务器的仓库根目录执行：

```bash
nvidia-smi
bash shell/check_dependencies.sh --rtx5090
bash shell/check_dependencies.sh --rtx5090 --check
```

`--rtx5090` 选择训练环境，自动跳过 Blender 和 MATLAB；脚本创建 `.venv`，安装或纠正缺失/版本不符的依赖。数据已生成时服务器只需此环境。GPU 预检查核对 CUDA、SM 120、BF16，并以 64×64 图像实际执行网络前向、损失、反向及 fused AdamW 更新。缺少 GPU、驱动不可用或装错 PyTorch 会报错。之后检查和补装也应使用 `--rtx5090`，避免默认依赖锁重新安装 PyTorch 2.5.1。需要 Blender/MATLAB 时使用独立环境。

可以单独检查 GPU，或选择其他显卡编号：

```bash
.venv/bin/python -m visual_training.check_gpu
.venv/bin/python -m visual_training.check_gpu --device cuda:1
```

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

## 依据与验证边界

- [PyTorch 2.7 发布说明](https://pytorch.org/blog/pytorch-2-7/)：官方 Blackwell 支持及 CUDA 12.8 wheel。
- [PyTorch 历史版本安装说明](https://pytorch.org/get-started/previous-versions/)：2.7.1 的 cu128 官方源。
- [CUDA 12.8 发布说明](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html)：SM 120 支持及驱动要求。

本机为 macOS，无 NVIDIA CUDA 设备。CPU BF16、并行加载、数据迁移、权重评估和 ONNX 导出可在本机验证；RTX 5090 的内核、驱动、完整分辨率显存和吞吐需在目标服务器执行上述检查与训练。详细结果见 [验证记录](validation.md)。
