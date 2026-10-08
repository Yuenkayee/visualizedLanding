# Ubuntu 22.04 / RTX 5090、RTX PRO 6000 Blackwell 视觉训练

使用 `configs/multi_camera_training_rtx5090.yaml` 训练已有四相机共享网络。输入仍为单张图像，四路相机的样本共享权重；batch=16 表示 16 张图像。三阶段轨迹、天气、遮挡、关键点顺序和完整航次 split 沿用现有多相机数据设计。

同一配置适用于用户服务器的 RTX PRO 6000 Blackwell Server Edition（96 GB），沿用 Python 3.12 / PyTorch 2.8.0+cu128 / CUDA 12.8。该服务器的训练 GPU 冒烟检查已通过；下面的数据渲染还需要单独检查 Blender/Cycles。

## 环境与安装

目标环境为 Ubuntu 22.04、x86_64、Python 3.12 和一张 RTX 5090（32 GB）。`nvidia-smi` 应能识别显卡，使用支持 RTX 5090 和 CUDA 12.8 的 NVIDIA 驱动，例如支持该显卡的 R570 或更新驱动。安装脚本不修改系统驱动。PyTorch wheel 自带所需 CUDA 运行库，本项目训练无需另外安装完整 CUDA Toolkit 或 `nvcc`。

原默认锁中的 PyTorch 2.5.1 / CUDA 12.4 不适合 RTX 5090 的 Blackwell SM 120。本配置与服务器现有版本对齐，单独锁定 PyTorch **2.8.0+cu128** 及其 CUDA 12.8 传递依赖（`nvidia-cuda-runtime-cu12==12.8.90`），其他直接依赖沿用原版本。`requirements.in` 是包元数据约束，实际安装使用完整精确锁 `requirements-rtx5090.txt`。

| 项目 | RTX 5090 profile |
| --- | --- |
| Python | 3.12.x（不固定补丁版本） |
| PyTorch | 2.8.0+cu128 |
| PyTorch CUDA 运行时 | 12.8（运行库包 12.8.90） |

这里检查的是 `torch.version.cuda`，不以系统 `nvcc` 或 `nvidia-smi` 显示的 CUDA 版本代替。安装和训练默认使用服务器当前的 Python 环境，不创建、不自动选择项目 `.venv`，也不下载 uv 或 Python。

解释器选择顺序：`--python PATH` → `LANDING_PYTHON` → 当前 PATH 中的 `python` → `python3`。先激活服务器已有的 Python 3.12 / PyTorch 环境，再在仓库根目录运行：

```bash
python --version
bash shell/check_dependencies.sh --rtx5090
bash shell/check_dependencies.sh --rtx5090 --check
```

脚本会打印实际解释器的绝对路径。`--rtx5090` 自动跳过 Blender/MATLAB，核对完整依赖锁，通过所选解释器的 `python -m pip` 补齐缺少的包，并纠正与锁文件不符的版本；已满足锁定版本的包直接复用。安装后重新核对版本、模块导入和项目包的传递依赖，最后执行实际 GPU 前向、反向和 fused AdamW 更新。安装失败、导入失败或 GPU 检查失败都会返回非零退出码。

这些安装发生在当前环境中，可能更新该环境的已有包。精确锁仍是 `torch==2.8.0+cu128`；如果预装包仅标记 `2.8.0`，pip 会按锁调整为官方 cu128 build。脚本检查项目自身的依赖声明，不要求同一环境中无关工具的依赖也符合本项目。

`--check` 只检查、不安装，也不引导安装 pip。正常安装时若所选解释器没有 pip，先尝试其 `ensurepip`；若解释器不提供 ensurepip、安装目录不可写或系统环境策略禁止 pip 写入，脚本会保留原始错误并停止，不自动提权或绕过系统策略。包下载保留重试/超时，继承当前 pip 配置及代理。

可以显式选择现有 Python/Conda 环境，不必修改 PATH：

```bash
bash shell/check_dependencies.sh --rtx5090 --python /opt/conda/bin/python
# 或使用环境变量（路径按服务器实际位置填写）
LANDING_PYTHON=/opt/conda/bin/python bash shell/check_dependencies.sh --rtx5090
```

旧项目 `.venv` 可以保留，脚本不自动读取它。如果当前终端激活的就是该环境，需要先退出该环境或用 `--python` 指向服务器已有的 Python 3.12。选中的 Python 版本错误时会立即停止，不自动切换环境。默认 Blender/MATLAB profile 仍要求 Python 3.11；服务器训练始终使用 `--rtx5090`。

也可用同一 Python 单独检查 GPU：

```bash
python -m visual_training.check_gpu
python -m visual_training.check_gpu --device cuda:1
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

显式指定已有环境并后台启动：

```bash
bash shell/train_rtx5090.sh --python /opt/conda/bin/python
```

后台启动前会将解释器固定为绝对路径，并将相同路径传入依赖安装、GPU 检查和训练。即使 tmux 服务保留了以前 SSH 会话的 PATH，也不会误用另一个 Python。

需要在当前终端排查时，使用 `bash shell/train_rtx5090.sh --foreground`，仍会保存日志。`--session NAME` 可指定会话名称，`--log PATH` 可指定日志文件（追加写入）。后台入口将当前 SSH 会话的代理、包源、Python 路径和 `CUDA_VISIBLE_DEVICES` 等设置传入任务，避免已有 tmux 服务保留过期环境。

## 在服务器直接生成训练数据

`FileNotFoundError: data/multi_camera/splits/train.json` 表示训练入口未找到训练集索引。先完整生成数据，再启动训练；只检查 PyTorch 依赖、只生成轨迹或预览，都不会生成训练所需的三个 split。

使用更新后的入口，不需要 `.venv`，也不需要把 `bpy` 安装进 Python 3.12：

```bash
# 安装缺少的渲染依赖，并实际渲染一张小图验证 GPU；此步不生成数据集
bash shell/generate_multi_camera_dataset.sh \
  --python /root/miniconda3/bin/python --setup-only

# 可选：60 张四视角预览，查看 data/multi_camera_preview/multi_preview.png
bash shell/generate_multi_camera_dataset.sh \
  --python /root/miniconda3/bin/python \
  --preview --sorties-per-weather 3 --width 640 --samples 8 \
  --output data/multi_camera_preview

# 正式生成；自动再次检查依赖和 GPU，可不单独执行 --setup-only
bash shell/generate_multi_camera_dataset.sh --python /root/miniconda3/bin/python
```

解释器路径按服务器实际位置调整；省略 `--python` 时采用当前环境，选择顺序同训练脚本。Linux x86_64 首次运行会依次尝试 Blender 官方源、南京大学和 Berkeley OCF 的 HTTPS 镜像，下载 **Blender 4.5.3 LTS** 便携版，校验固定 SHA256，并解压至项目 `.tools/`。403/404 等永久错误会立即切换来源；网络中断、超时、408/429 或服务端故障每个来源最多尝试三次。压缩包校验不符也会换源，绝不解压未通过校验的文件。成功下载的压缩包会复用。无需 sudo，也不会替换系统 Blender 或 NVIDIA 驱动。已有 Blender 4.5 LTS 时可用 `--blender /opt/blender-4.5.3/blender` 跳过下载；其他版本会明确报错。

出现 `HTTP Error 403: Forbidden` 表示下载请求被站点或中间代理拒绝，尚未开始检查 GPU。仅凭这条日志无法区分出口 IP、代理规则或站点策略。更新脚本后重新运行原命令即可自动换源；也可以只使用服务器可访问的指定镜像：

```bash
bash shell/generate_multi_camera_dataset.sh \
  --python /root/miniconda3/bin/python --setup-only \
  --blender-download-url https://mirrors.nju.edu.cn/blender/release/Blender4.5/blender-4.5.3-linux-x64.tar.xz
```

`--blender-download-url` 替换默认下载源列表。所有来源仍必须通过项目固定的官方 SHA256，不关闭 HTTPS 证书验证。如果服务器访问所有下载源都失败，在另一台可联网的机器下载同一 Linux x64 压缩包，复制到服务器后执行：

```bash
bash shell/generate_multi_camera_dataset.sh \
  --python /root/miniconda3/bin/python --setup-only \
  --blender-archive /root/downloads/blender-4.5.3-linux-x64.tar.xz
```

本地包同样验证 SHA256，错误或缺失时停止，不回退联网下载 Blender。路径可替换为实际位置；相对路径以仓库根目录为基准。该选项只省去 Blender 程序包下载，首次安装缺少的 NumPy 等渲染依赖仍需要访问 pip 源。安装成功后无需重复提供来源参数，后续直接复用 `.tools/` 中的 Blender。

Blender 4.5 自带 Python 3.11，渲染用 NumPy/SciPy/OpenCV/PyYAML 按 `requirements-render.txt` 安装到 `.tools/blender-4.5-py311-site/`，使用其内置 Python 选择二进制包。Python 3.12 只负责启动和安装调度；渲染不加载训练环境的 site-packages，不修改已有 PyTorch/CUDA。`.tools/` 是便携程序及渲染包目录，不是 `.venv`。

默认 `--cycles-device OPTIX --gpu-index 0`，在 Cycles 中仅启用指定 GPU。日志中必须出现 `cycles_smoke_ok` 和目标显卡名称，之后才开始规划和渲染。PyTorch 的 CUDA 检查成功不等于 Cycles 检查成功；OptiX 不可用时可显式尝试 `--cycles-device CUDA`，仍需通过实际渲染。失败不会静默改用 CPU。`--gpu-index` 是该 Cycles 后端枚举的索引，按日志确认设备名称，不假设与 PyTorch 的 `cuda:N` 一致。驱动缺失、容器未暴露 GPU 或缺少系统动态库时，脚本会保留原始错误并停止。

`--renderer blender --cycles-device CPU` 是 Blender 的真实场景 CPU 渲染；`--renderer cpu` 是用于流程检查的简化投影图，二者不同。`--renderer cpu` 和 `--plan-only` 直接使用所选 Python，检查/补齐四个生成依赖，不下载 Blender、不探测 GPU。

正式默认配置生成 **60 航次 × 3 阶段 × 每阶段 60 帧 × 4 相机 = 43,200 张 RGB 图**，同时输出 mask、关键点/位姿标注、标定及同步 LiDAR。包含晴朗、雾、眩光、夜间、夜雾和既定姿态扰动。train/val/test 按完整航次划分，默认分别为 **28,800 / 7,200 / 7,200 张图**。分辨率 1280×960、24 samples；CPU 仍参与轨迹、建场景、标注和文件处理。

生成完成后才会发布 `data/multi_camera/splits/{train,val,test}.json`；`multi_dataset_report.json` 记录渲染设备和实际数量。不能根据某张图片出现就提前训练。若输出目录已有 `splits/`，脚本会拒绝覆盖，请另选 `--output`，训练时相应设置 `--data-root`。中途失败的目录没有最终 split，可修复后重跑；目前会重新渲染，不提供逐帧断点续跑。

较长任务可以在已有 tmux 中运行并保存日志。在仓库根目录先创建会话：

```bash
tmux new -s landing_data
```

然后在新会话中执行（路径按实际仓库位置调整）：

```bash
cd /root/visualizedLanding
mkdir -p outputs/logs
set -o pipefail
bash shell/generate_multi_camera_dataset.sh --python /root/miniconda3/bin/python \
  2>&1 | tee outputs/logs/multi_camera_generate.log
```

按 `Ctrl+b` 后按 `d` 可脱离；`tmux attach -t landing_data` 返回。命令结束后会保留交互 shell，便于查看错误；同名日志会被下次 `tee` 覆盖，需要保留时更换日志名。渲染耗时和磁盘占用取决于机器及天气，尚未测量全量吞吐。生成成功后，在仓库根目录启动训练：

```bash
bash shell/train_rtx5090.sh --python /root/miniconda3/bin/python
```

## 数据迁移

将完整 `data/multi_camera/` 复制到服务器，包括 `rendered/`、`annotations/`、`splits/` 及标定等索引。不要只复制图片，也不要重新随机划分相邻帧。训练/评估加载器按指定数据根目录重新定位 `annotations/` 内的标注，兼容现有 split 中保存的 Mac 绝对路径，也支持相对路径；原航次归属和 split 文件内容保持不变。

默认路径是仓库内 `data/multi_camera`。若数据位于独立磁盘，通过 `--data-root` 指定绝对路径：

```bash
python -m visual_training.train \
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
python -m visual_training.train \
  --config configs/multi_camera_training_rtx5090.yaml \
  --batch-size 8 --num-workers 4
```

服务器 CPU 至少应能提供配置所需的线程和内存；CPU 较少时同时调整 YAML 中的 `threads`。显卡有余量时可实测 batch=24/32，暂不自动增大学习率。512 缩放后中距 H 仍可能很小，GPU 加速不替代小目标定位精度评估。

## 评估与导出

```bash
python -m visual_training.evaluate \
  --root data/multi_camera \
  --checkpoint visual_training/checkpoints/multi_camera/best.pt \
  --image-size 512 --device cuda:0 \
  --output outputs/metrics/multi_camera_vision.json

python -m visual_training.export_model \
  --checkpoint visual_training/checkpoints/multi_camera/best.pt \
  --output visual_training/checkpoints/multi_camera/h_detector.onnx \
  --image-size 512
```

数据迁移到独立磁盘时，评估的 `--root` 与训练的 `--data-root` 应一致。评估使用 FP32，按阶段/天气/相机输出分组指标。ONNX 导出在 CPU 执行；此环境锁定的是 CPU `onnxruntime`，GPU 训练通过 PyTorch CUDA 完成。训练权重保存为 FP32，可供原有 CPU/MPS/PyTorch 导航加载，也可导出为原有三输出 ONNX。

ONNX 导出显式选择 `dynamo=False`，保持现有 opset 17、动态 batch 和三个输出的接口。服务器上可以运行针对训练和导出的回归测试（会生成临时小数据，不覆盖正式权重）：

```bash
python -m pytest -q tests/test_gpu_training.py tests/test_training_shell.py
```

测试包含版本拒绝、BF16 损失/梯度、spawn 数据加载、1 epoch 训练、权重重载评估、PyTorch/ONNX 数值比对，以及有 RTX 5090 时的真实 CUDA 前向/反向更新。完整分辨率显存和吞吐仍由正式训练日志确认。

## 依据与验证边界

- [PyTorch 历史版本安装说明](https://pytorch.org/get-started/previous-versions/)：2.8.0 的 cu128 官方源。
- [CUDA 12.8 发布说明](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html)：SM 120 支持及驱动要求。
- [Blender 4.4 Cycles 发布说明](https://developer.blender.org/docs/release_notes/4.4/cycles/)：新增 Blackwell 支持；本入口固定使用后续的 4.5.3 LTS。
- [Blender 4.5.3 官方校验和](https://download.blender.org/release/Blender4.5/blender-4.5.3.sha256)：Linux 便携包的固定 SHA256 来源。

本机为 macOS，无 NVIDIA CUDA 设备。已在独立 Python 3.12.14 / PyTorch 2.8.0 环境完成完整回归：66 项通过、1 项 NVIDIA 硬件测试跳过，包含 CPU BF16 训练、并行加载、权重重载评估与 ONNX 数值比对。Linux / Python 3.12 的 49 个锁定包通过 binary-only 安装 dry-run。RTX 5090 的内核、驱动、完整分辨率显存和吞吐需在目标服务器执行上述检查与训练。详细结果见 [验证记录](validation.md)。
