# 四场景训练与离线导航回放（2026-10-16）

所有命令在仓库根目录运行。服务器使用 Python 3.12 / PyTorch 2.8.0+cu128，不安装本地 Python 3.11 的 requirements.txt。训练、准备、回放脚本都检查依赖，缺失时安装，并保存日志；没有关机操作。

## 已有训练集的处理

保留 clear、fog、night、night_fog，排除独立的 glare 天气类别。夜间灯光的正常光晕参数仍属于夜间场景。原始图片/标签无需重新渲染，也不要删除旧数据来迁就新配置。

```bash
export PYTHON_BIN=python
# 先验证路径、三个split、航次隔离及36组抽样图像/雷达，不产生新训练集
bash shell/prepare_landing_four_weather.sh test \
  --source data/landing --output data/landing_four_weather
# 正式建立四场景副本；输出目录必须不存在
bash shell/prepare_landing_four_weather.sh full \
  --source data/landing --output data/landing_four_weather
```

对本次旧10800帧数据，得到48航次、8640帧，train/val/test=5760/1440/1440。保留原有划分，不把验证/测试困难帧混入训练。使用相对manifest；可识别旧服务器绝对路径。图片、JSON、点云、轨迹优先硬链接，跨文件系统自动复制，不用软链接；整个新目录可以独立传输。硬链接共享文件内容，后续不要就地编辑任一副本的图像/标签；修订标签应写新文件并原子替换。脚本不修改原文件，已有输出会被拒绝。`preparation.json`、`SUCCESS.json` 表示完整完成，缺少完成标记不能用于训练。准备日志在 `outputs/logs/dataset_prepare/`。

数据增强在线进行，仅作用于训练集：缩放/平移/小角度变换同步更新关键点、掩码、可见性；局部遮挡更新监督；4%目标缺失背景样本；着舰阶段1.5倍抽样。增强不是新的云台物理仿真；其图像不会用于传感器回放。读图先缩放到网络分辨率，再用float32增强，并缓存小型JSON元数据，减少原分辨率浮点运算和反复读JSON。

## 预训练和正式训练

```bash
bash shell/train_landing_server.sh pretrain --data-root data/landing_four_weather
bash shell/train_landing_server.sh full --data-root data/landing_four_weather \
  --session landing_four_weather --output visual_training/checkpoints/four_weather_v1
# 查看后台训练，Ctrl-B后按D脱离，训练继续
 tmux attach -t landing_four_weather
```

预训练前台运行2轮，每轮最多50训练batch/10验证batch；只保存日志，不保存权重/优化器/history/曲线。正式训练自动tmux后台，最多40轮，验证损失连续10轮不改善早停。两者均不关机。输出模型目录必须新建或为空，不能覆盖旧实验。日志路径会在启动时打印，位于 `outputs/logs/training/`。

服务器默认512输入、16基础通道，BF16/channels-last/TF32、最多8数据worker和4计算线程，根据显存余量及实测前后向吞吐量选择batch（上限16）。不承诺固定GPU利用率；以真实训练images/s、batch探测和显存日志为准。容器共享内存不足会减少worker，可在容器配置中扩大 `/dev/shm`。保留显式DataLoader清理，训练完成事件发生在worker退出之后。

新模型增加4×4全局上下文、四点可见性输出；继续使用soft-argmax，不盲目切换argmax。连续中心高斯软热图sigma=1输入像素，坐标SmoothL1 beta=0.01（归一化坐标），可见性BCE权重0.2。Plateau学习率factor=0.5、patience=3、min_lr=1e-5。新结构从头训练；旧权重仍按其保存的model_config加载回放，不会被当作新结构继续训练。

正式目录自动生成：

- `best.pt` / `best_loss.pt`：最低验证损失；`best_keypoint.pt`：优先四点≤5px成功率，再比较P99和均值；`last.pt`：最后完成轮。
- `history.json`，`metrics/epochs.json`：每轮损失分项、裁剪前总梯度范数、学习率、性能、按阶段/天气的尾部指标。
- `metrics/analysis.md` 和 `metrics/training_curves.png`：自动总结和曲线；不再依赖历史outputs目录中的临时分析脚本。

四点成功率仅以四点全部可见帧为分母。原始CE和新软标签CE的绝对值不可横向直接比较。`best_keypoint`不是`best_pose`；选择导航候选须在同一验证集回放并比较RMSE、P95、最长拒绝间隔。未自动运行昂贵的每轮导航来生成所谓best_pose。

为控制变量，提供两个独立配置（同一数据、种子、分辨率、最多40轮）：

```bash
# 原结构/原关键点损失，只加学习率调度和早停（增强实现共同优化）
bash shell/train_landing_server.sh full --config configs/landing_training_schedule.yaml \
  --session landing_schedule --output visual_training/checkpoints/four_schedule
# 相比上项仅改变连续高斯热图监督
bash shell/train_landing_server.sh full --config configs/landing_training_soft.yaml \
  --session landing_soft --output visual_training/checkpoints/four_soft
```

请顺序执行实验，避免一张4090同时承担多次训练。默认完整改进配置为 `configs/landing_training_server.yaml`；上述对照用来判断新模块效果，不是要求运行三个训练后才能使用回放。

## 离线导航回放

```bash
# 快速模式：每种天气一个完整航次，仍包含接近/悬停/下降，不截断末段
bash shell/replay_landing_offline.sh \
  --data-root data/landing_four_weather \
  --checkpoint visual_training/checkpoints/four_weather_v1/best.pt \
  --mode test --split val --device cuda --output outputs/replay/four_weather_v1_test
# 正式验证：该split全部四场景航次，输出目录必须不存在
bash shell/replay_landing_offline.sh \
  --data-root data/landing_four_weather \
  --checkpoint visual_training/checkpoints/four_weather_v1/best.pt \
  --mode full --split val --device cuda --compare-baseline \
  --output outputs/replay/four_weather_v1_val
```

替换checkpoint即可回放后续训练。对比 `best_keypoint.pt` 时换新的 `--output`。固定验证方案后再用 `--split test` 作保留数据评估；当前已分析过的旧test只能作为回归，最终验收建议另留未分析的新航次。`--sequences <航次1> <航次2>` 可指定完整航次；`--weathers clear fog` 可选天气。

本机已有旧权重直接运行：

```bash
PYTHON_BIN=.venv/bin/python bash shell/replay_landing_offline.sh \
  --data-root data/landing_server/landing \
  --checkpoint visual_training/checkpoints/full_20261014_184359_0f5f3d/best.pt \
  --mode test --split val --device auto --compare-baseline \
  --output outputs/replay/old_best_four_weather
```

回放无内环、无控制反馈、无轨迹平滑；按实际抽样时间顺序读取RGB、云台编码器和LiDAR，输出起落架中心相对着舰中心的位置。甲板坐标原点=着舰中心，+X舰艏、+Y左舷、+Z法向向上，单位米。相机固定安装位置与云台旋转外参保持原接口。默认仅首帧使用真值位置/相机姿态作明确的初始化先验，初速度为0，随后真值仅评分，不参与更新。可用 `--initial-position-error 2 -1 2 --initial-rotation-error 2 -1 3` 检验初始化偏差；`--initialization config` 改用navigation.yaml的先验（此时initial_position应填起落架的甲板坐标）。

输出 `analysis.md`、`summary.json`、每航次全程/着舰放大PNG/PDF、每帧JSON/CSV。逐帧包含三个方向真值/估计/误差、健康状态、预测协方差、视觉/LiDAR接受、视觉拒绝原因、距上次视觉时间、更新校正量。更新校正量是滤波预测前后之差，不是含真实运动的相邻位置差。日志在 `outputs/logs/replay/`。未显式指定输出目录时使用时间戳目录。

`--compare-baseline`共享同一次网络推理，baseline禁用分割几何恢复、可见性门限、自适应像素噪声和PnP LM细化，并恢复原加速度噪声0.5；current使用当前配置。旧权重下可以直接复现原导航；新权重下这仅是导航消融，并不是旧网络的性能。默认回放FP32并记录checkpoint SHA256，避免混用其他权重的缓存。

## 末段误差/跳变的修复机制

1. 外角点与内部H拐角混淆导致PnP离群。现在使用完整分割的外凸包直线拟合恢复四外角，并检查完整三条笔画的投影IoU、像素尺度、画面边界和航向先验；只有几何证据完整且与网络点明显不一致才替换点。截断/太小的H不强行补角；可见性分支拒绝不可观测点。分割均值置信度没有被当作角点可信度。
2. 共面PnP双解仍用因果先验消歧，各分支分别LM细化；保留正深度、重投影和ESKF创新门限。恢复点使用更保守像素噪声，残差还会增加观测噪声；图像小目标的Jacobian协方差自然增大，不能仅凭低重投影误差宣称精确定位。
3. 原0.5的加速度过程噪声在稀疏末段数据中使预测过度自信，拒绝真实机动后的视觉/雷达更新；默认配置改为1.5，仍保留创新门限，而不是输出限幅或利用真值拉回。该白噪声谱幅值需要在真实传感器频率与机动范围下重新标定。
4. 视觉长时间丢失时，雷达平面不能补齐水平位置/偏航可观测性。健康状态和拒绝间隔会保留，不通过平滑图像或删除失败帧隐藏。现有数据仍无法证明完全遮挡/真实飞行情况下不会漂移。

## 可选：增加真实动力学覆盖的新数据

四场景过滤和在线增强足以运行新训练；扩大云台动力学覆盖必须重新渲染，不能只修改旧图片标签。提供独立种子配置，20航次/天气（80独立航次、14400帧），每航次固定一组随机带宽、阻尼、速率、加速度和编码器噪声，含少量关闭速率前馈/从零角度捕获目标。轨迹、渲染、标签与编码器用同一组参数；相机BODY安装平移不变。范围是待标定的工程假设，不包括完整通信延迟模型。

```bash
bash shell/generate_landing_dataset_server.sh test \
  --config configs/landing_dataset_robust.yaml --output data/landing_robust_test
bash shell/generate_landing_dataset_server.sh full \
  --config configs/landing_dataset_robust.yaml --output data/landing_robust
bash shell/train_landing_server.sh full --data-root data/landing_robust \
  --session landing_robust --output visual_training/checkpoints/four_weather_robust
```

此配置本身只有四种天气，无需再次过滤。不要把相同航次的相邻帧拆到不同split；保留新测试航次直到模型选择结束。生成脚本仍按原方式运行；远程断开保护可用 `tmux new -s landing_data` 后执行生成指令，完成后Ctrl-B、D脱离。正式训练则由训练脚本自动启动tmux。
