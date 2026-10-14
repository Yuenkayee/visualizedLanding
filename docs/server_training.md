# RTX 4090 预训练检查与正式训练

在服务器仓库根目录运行 `shell/train_landing_server.sh`，使用镜像 Python 3.12 / PyTorch 2.8.0 / CUDA 12.8。脚本每次先检查 `requirements-training-server.txt` 及所需传递依赖，缺失或版本不符时自动安装，再执行 CUDA 运算检查。复用符合版本的 PyTorch；训练无需安装 Blender。正式模式还检查 tmux，缺失时自动用 apt 安装；非 root 用户需要免密 sudo，否则脚本输出管理员需执行的安装命令。

## 运行命令

```bash
cd /root/autodl-tmp/visualizedLanding
export PYTHON_BIN=python

# 预训练检查：前台运行，仅保存日志，使用正式数据的 train/val 划分。
bash shell/train_landing_server.sh pretrain

# 正式训练：自动创建 detached tmux 会话 landing_train。
bash shell/train_landing_server.sh full
```

默认数据目录为 `data/landing/`，需要先完成正式训练集生成。若暂时只有此前生成的 90 帧测试集，可先验证训练能否运行：

```bash
bash shell/train_landing_server.sh pretrain --data-root data/landing_server_test
```

小测试集只有 30 张训练图像，不能充分反映正式数据的加载吞吐率；建议完整数据生成后再跑默认预训练。正式模式会拒绝生成器标记为 `mode: test` 的数据和未完成生成的数据。启动前检查 train/val/test 标注与图像是否存在、架次是否跨划分泄漏；训练与参数选择只使用 train/val，不用 test 计算损失或选择权重。

| 行为 | pretrain | full |
| --- | --- | --- |
| 运行位置 | 当前终端前台 | 自动创建 tmux 后台会话 |
| 默认时长 | 2 epoch，每轮最多 50 个训练 batch、10 个验证 batch | 40 epoch，完整 train/val |
| GPU 参数探测 | 自动选择 batch size | 启动时重新探测，可显式指定 |
| 日志 | 保存依赖、训练、指标、配置和状态日志 | 同左 |
| 模型权重/优化器/训练结果 | 不写入磁盘 | 保存 best.pt、last.pt、history.json |
| 结束后行为 | 返回终端 | 训练命令结束，tmux 会话通常随之退出 |
| 自动关机 | 不执行 | 不执行 |

这里的“预训练”是短程训练检查，不是预训练权重阶段。正式训练从固定随机种子初始化，不继承短程检查的参数。GPU batch 探测中也会执行真实优化器更新，但这些参数随后被丢弃。

## 查看后台任务与日志

```bash
tmux attach -t landing_train
```

在 tmux 内按 **Ctrl+B，然后 D** 可离开会话，任务继续执行；之后可以断开 SSH。不要用 Ctrl+C 代替 detach，Ctrl+C 会中断训练。再次启动同名会话会被拒绝，避免误开第二个任务；需要另一个名字时指定 `--session landing_train_v2`。

启动时打印本次日志和权重绝对路径。每次调用使用独立时间戳目录：

- 日志：`outputs/logs/training/pretrain_<时间戳>_<标识>/` 或 `full_<时间戳>_<标识>/`。
- `setup.log`：Python 依赖检查/安装、CUDA 与 tmux 检查。
- `train.log`：完整训练标准输出与错误，包括异常堆栈。
- `metrics.jsonl`：batch 探测、训练进度、epoch 损失、验证 IoU/归一化关键点误差、吞吐率、峰值显存。
- `resolved_config.json`：最终选定的 batch size、加载进程数和其余参数。
- `status.json`：`starting` / `running` / `complete` / `failed`。正式模式启动成功仅表示 tmux 创建成功，训练完成以此文件的 `status: complete` 为准。
- `launcher.log`：启动阶段错误（如果有）。

正式权重默认位于 `visual_training/checkpoints/landing/full_<时间戳>_<标识>/`，`best.pt` 按最低验证损失选择，`last.pt` 保存最后一个完成的 epoch；文件写入使用临时文件替换，避免中途写坏旧 checkpoint。预训练目录中的配置、状态和指标属于日志，不保存 `.pt`、优化器或 `history.json`。

```bash
# 将路径替换为启动时打印的具体日志目录。
tail -f outputs/logs/training/full_<时间戳>_<标识>/train.log
cat outputs/logs/training/full_<时间戳>_<标识>/status.json
```

## 性能与训练参数

服务器配置文件为 `configs/landing_training_server.yaml`：512×512 输入、原 channels=16 / 4 关键点网络、AdamW、学习率 0.001、40 epoch。保留完整分辨率和网络结构，启用 BF16 混合精度、channels-last、TF32、cuDNN 调优、锁页内存、异步拷贝与数据预取。关键点 softmax 和损失累积使用 FP32；非有限损失或梯度会使任务明确失败。

默认比较 batch size 16、8、4 的真实前向、反向和优化器更新速度，排除首次预热；遇到显存不足会尝试更小 batch。选择可用显存 85% 预算内测得最快的候选，最大值默认限制为 16，避免仅为占满显存无限扩大 batch、改变优化行为。实际训练阶段还单独记录包含数据加载的吞吐率；GPU 探测速度不是端到端训练速度。小样本测试不能证明收敛或最终精度，需根据正式训练的验证指标和独立测试集评估确认效果。

默认最多 8 个并行数据加载进程、4 个 PyTorch CPU 线程；依据容器可用核数调整。每个数据进程限制 OpenCV/BLAS 线程，并使用独立的数据增强随机种子。训练/验证分别使用持久 worker 池，只有当前阶段的池持续加载。若容器 `/dev/shm` 太小，会自动减少 worker（必要时降到 0）并记录 `loader_shared_memory_limit`，避免容器虽然有 120 GB RAM 却因共享内存不足崩溃。可在创建容器时配置例如 `--shm-size=4g`，也可按平台方式调整共享内存。

```bash
# 若要固定使用预训练日志中选中的 batch（这里以 8 为例）：
bash shell/train_landing_server.sh full --batch-size 8

# 自定义正式数据和权重目录：
bash shell/train_landing_server.sh full \
  --data-root /data/landing_v2 --output /data/landing_weights_v2 \
  --session landing_train_v2 --epochs 40 --num-workers 8

# 更长的前台检查，仍不保存权重：
bash shell/train_landing_server.sh pretrain --epochs 3 --train-batches 100

# 使用 FP32 做数值对照：
bash shell/train_landing_server.sh pretrain --precision fp32

# 仅检查训练环境（不安装）：
bash shell/check_server_dependencies.sh --training --check

# 观察 GPU：
watch -n 1 nvidia-smi
```

建议数据生成结束后再开始训练，避免 Blender 与训练竞争 GPU 显存。`--output` 要求新目录或空目录，脚本不会覆盖旧模型，也不提供自动断点续训。既有评估、ONNX 导出与导航推理接口仍读取相同 checkpoint 格式，使用本次实际 `best.pt` 路径即可。
