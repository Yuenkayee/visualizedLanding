# Visualized Landing / 舰载 H 甲板视觉导航

此仓库实现一个可运行的研究基线：参数化护卫舰/H 甲板建模、RGB 与激光数据生成、分割/关键点训练、平面 PnP、误差状态滤波以及离线传感器闭环。Python 3.11，米/秒/弧度，RGB 图像。`external/` 保持原样，MATLAB 适配层只读取其中的舰船运动模型。

## 安装和检查

```bash
bash shell/check_dependencies.sh
source .venv/bin/activate
bash shell/check_dependencies.sh --check
pytest -q
```

脚本创建项目 `.venv`，自动安装缺失/版本不符的锁定依赖。没有 Python 3.11 时使用 uv 下载；没有 Blender 应用时安装锁定的 `bpy` 4.2 运行时。发现 MATLAB R2024b 后安装其 Engine；MATLAB 应用及许可证需自行提供。可用 `--without-blender --without-matlab` 只安装 CPU 数据/训练/导航基线，`--with-matlab` 强制检查 MATLAB。

`requirements.txt` 锁定核心直接和传递依赖；`requirements-blender.txt` 是包含核心依赖的 Blender 扩展锁；`requirements-matlab.txt` 锁定需要本机 MATLAB 的可选 Engine。CUDA 包使用平台条件标记，macOS 无需安装它们。不要把 `bpy` 装进不同版本的 Blender 内置 Python。

## 生成数据、训练、导出

```bash
python -m visual_training.data.build_dataset
python -m visual_training.train
python -m visual_training.evaluate
python -m visual_training.export_model
```

默认数据生成器是快速 CPU 针孔投影，提供训练和端到端检查用的简化图像。按**完整序列**划分 train/val/test，至少需要三个独立序列。20 个 epoch 是可调参数，训练完成不代表已达到真实图像精度；应结合独立 Blender/真实数据评估。

三维护卫舰渲染：

```bash
bash shell/render_sequence.sh --frames 30 --sequence-id sea_001
bash shell/render_sequence.sh --frames 30 --sequence-id sea_002
bash shell/render_sequence.sh --frames 30 --sequence-id sea_003
python -m visual_training.data.build_dataset --from-rendered
```

`configs/scene.yaml` 控制舰船、H 尺寸、海面、天气和光照。输出 RGB、二值 H mask、可见关键点、位姿真值和 LiDAR `.npz`；场景保存至 `visual_modeling/assets/scenes/frigate_landing.blend`。可用 `--trajectory path.json` 读取真实轨迹（数组，每行含 timestamp、T_deck_camera 和可选 T_world_deck）。外部 GLB/OBJ/Blend 模型可配置 `ship.asset_path`；必须已获授权并预先对齐 +X 船首方向。来源与限制见 [开源资产说明](docs/model_sources.md)。

安装了 `bpy` 时，`python -m visual_modeling.blender.render_overview` 生成整艘护卫舰预览；`bash shell/fetch_reference_assets.sh` 可选下载固定提交的 VRX 参考船模及许可证。

## 闭环与回放

```bash
python -m simulation.offline.run_closed_loop
python -m simulation.evaluation.plot_results outputs/logs/closed_loop.jsonl
python -m simulation.offline.run_replay --sequence synthetic_42_000
python -m simulation.experiments.run_batch --seeds 1 2 3
python -m simulation.experiments.run_ablations
```

默认闭环使用 CPU 图像与甲板激光射线，输入导航的是模拟传感器，真值只用于渲染和评估。选择神经网络：在 `configs/navigation.yaml` 设置训练得到的 `.pt` 或 `.onnx` checkpoint。无 checkpoint 时使用白色 H 的经典轮廓检测；初始姿态是明确的航向先验。

```bash
python -m simulation.offline.run_closed_loop --backend matlab --duration 2
```

MATLAB 后端调用现有 `ship_motion_model`，飞机侧默认仍为受限加速度点质量控制器。`external/inner_loop/simu/closeloop_landing.slx` 没有经过本项目信号映射校验，因此不会被冒充成已经接通的内环。通过 MATLAB callbacks 接入经过验证的模型，详见 [离线仿真](docs/offline_simulation.md)。

## 能力边界

H 图案对称，不能独立确定绝对航向；激光平面不观测水平位置/yaw；目前没有 IMU 输入。滤波使用常速度/角速度模型，测量过期会刹车/悬停。接近甲板时完整 H 超出视场，经典检测会失效，系统安全停止下降；此基线不保证触舰，也不代表真实飞行安全验证。终端阶段需要更宽视场、部分标记模型、额外不对称标记或其他传感器。

所有运行命令在仓库根目录执行。数据、权重、日志和 `.blend` 是运行产物，不纳入 Git；`.gitkeep` 保留目录结构。
