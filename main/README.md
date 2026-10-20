# 导航接入内环的离线闭环宿主

主入口为 `main/run.py`，结果分析为 `main/analyze.py`，MATLAB 适配器在 `main/matlab/`。
复用现有 `NavigationPipeline`、训练场景、LiDAR 和云台动力学；不新增直升机动力学、制导或飞行控制律。

**当前仓库模型的实际状态（2026-10-20）**：`external/innerloop/simu/innerLoop.slx` 是坐标转换模型。用户更新的 `innerLoop_landing.slx` 已由内部 `innerLoop1` 子系统生成飞机/舰船状态，制导和控制在该子系统内部闭环；导航输入仅接到根层转换模块。使用 `main/navigation_observer.yaml` 可按当前实际航迹旁路评估导航，原 SLX 不被修改。`main/closed_loop.yaml` 仍是供将来导航反馈控制使用的通用端口合同，不能直接用于当前根接口。`smoke` 仍只使用积分器验收夹具。

## 当前模型的导航旁路评估

此处评估的是导航位置估计；不自动评价制导的目标预测误差。使用下列入口时，导航估计只进入转换/记录支路，不作用于飞行控制。启动会检查自主子系统无外部输入、根导航输入只连接转换模块、转换输出只连接记录端，避免误接入导航控制。

```bash
# 端口、0.02 s 时钟和导航旁路检查
bash shell/run_landing_closed_loop.sh inspect --config main/navigation_observer.yaml

# 本地：真实模型、现有 four_weather_v1/best.pt、Blender
bash shell/run_landing_closed_loop.sh run --config main/navigation_observer.yaml --duration 60

# Ubuntu + MATLAB/工具箱/许可证已就绪
bash shell/run_landing_closed_loop.sh run --config main/navigation_observer.yaml \
  --checkpoint /absolute/path/to/best.pt --duration 60 --device cuda --render-device OPTIX
```

初始化脚本 `main/matlab/landing_setup_observer.m` 加载仓库自带 `landingParam.mat`。原状态输出在 `innerLoop1/helicopter_state` / `innerLoop1/ship_state`，宿主在工作副本中加观察分支，无需增加源模型根输出。舰船状态的速度实际为 BODY 坐标，配置 `ship_velocity_frame: body` 在进入导航桥接层前旋转为 NED；原 SLX 转换模块自身的速度根输出尚不做该修正，请以宿主规范化后的传感器状态为准。状态位置不会自动平移去迎合制导目标。

每步在同一时间戳保存 `gear_relative_estimate_ned` 和 `gear_relative_truth_ned`，均为**轮底中心减 H 中心**的 NED 向量，单位 m。真值从 native 状态的同一转换内核取得，估计由本轮导航转换得到。输出 `gear_relative_ned_comparison.png/pdf` 为 North/East/Down 三行真值与估计曲线，`gear_relative_ned.csv` 为数据；无效估计以红点显示。旧的甲板系 `position_comparison.png/pdf` 保留。`controller_feedback.png` 在旁路运行中只是转换后的估计，不能解释为实际控制输入。

STOP 触发后读取保留的末帧输出，退出 Python 循环，写 `termination_reason: model_stop`、MATLAB `stop_reason` 和 `terminal_state.json`，照常生成图和误差统计。边界时刻可能出现最后一次 `advanced_s=0`：Simulink 在暂停时刻的下一次执行才求值 STOP，这是正常终止，不是假装推进一个 dt。运行错误仍报错；不会吞掉求解器错误。`duration_reached` 表示达到用户指定的运行上限。

**当前场景初值（2026-10-20 复核）**：制导 `Ship_trace` 和导出的 `Ship_motion` 质心初值均为 `[50;0;0]`，直升机质心初值为 `[0;0;-50]`。训练 H 偏移仍为 `[-45;0;-5.5]`，所以水平名义轮底中心在甲板系的先验为 `[-5;0;41.6]`，已同步到旁路配置。这个先验由既定场景初值计算，不读取运行时真值。源模型的自主子系统现名为 `innerLoop1`，配置的 `autonomous_subsystem` 与两个 `outputs` 路径已同步；以后重命名时也需更新这些路径。

按用户要求，STOP 保留 `abs(-CG_D-2.9)<0.01`，当前只验证导航对相对位置的估计和接口连通，不据此判定成功触舰。制导的目标点与 H 点仍是不同参考点，但不妨碍将同一时刻的导航估计与轮底减 H 的真值比较。

磁盘文件复核时，`innerLoop1/HELI_CONT/UWV_CONT/Inner system/MATLAB System3` 的 `B_10` 模块参数仍为 `B_5`。这不阻断接口连通；若 MATLAB 中已改为 `B_10`，需将修改保存到实际运行的 `external/innerloop/simu/innerLoop_landing.slx`。旁路配置不覆盖该控制器参数。

该模型用到的 `bodyToNedVel`、`nedVelToBody` 原先依赖本机仓库外 `HELI_CONT/src/trans` 的 MATLAB 路径；现已将同一实现纳入 `external/inner_loop/model/`，避免拷贝仓库到 Ubuntu 后丢失这两个转换类。它们只旋转速度，不改变速度单位。

## 训练几何参数

唯一来源为 `configs/landing_dataset.yaml` 的 `camera_geometry(config, rng=None)`，使用名义安装值，不把每航次随机标定误差写成固定几何：

```matlab
H_ship_body = [-45; 0; -5.5];
gear_body = [0; 0; 2.9];
T_body_camera = [0 -0.9945218953682733 -0.1045284632676536 2;
                1  0                   0                  0;
                0 -0.1045284632676536  0.9945218953682733 1.11;
                0  0                   0                  1];
```

BODY 为前/右/下，米；T 把 OpenCV 相机坐标变换至机体系。光轴下倾 96°。
云台只改变旋转，光心位置始终 `[2,0,1.11]`。轮底中心偏移与舰上 H 中心偏移保持独立。
这些是当前训练模型尺寸；不是实机标定结论。每次运行都会写 `run.json` 记录解析值。

`configure` 原位更新两个 SLX 的三个 Model Workspace 变量及同名模块参数表达式，同时对齐 0.02 s 固定步长并检查控制器采样周期，保存原文件备份；不会重建模型或删除已有制导/控制连线。直接打开模型即可看到这些参数。后续改变训练几何后请重新执行：

```bash
bash shell/run_landing_closed_loop.sh configure
```

## 完整内环模型的端口合同

可以在现有完整模型外包一层根模型；具体端口名由 `main/closed_loop.yaml` 的 `inputs` / `outputs` 映射。所有根输入必须显式映射，不默认为 0。飞机与舰船状态必须由这个模型内部推进，并来自同一个 NED 原点。

| 默认根输入 | 尺寸 | 含义 |
|---|---|---|
| nav_position_ned | 3 | 起落架中心减 H 中心，NED，m；甲板上方 Down 分量为负 |
| nav_velocity_ned | 3 | 上述相对向量在 NED 中的时间导数，m/s；已补偿偏移点与甲板转动 |
| nav_valid | 1 | 0/1。控制器必须据此执行其既有导航失效处理 |
| nav_timestamp | 1 | 此次反馈对应的仿真时刻，s；下一次更新前保持该时间戳 |

可选映射字段：`covariance_ned`（3×3），`T_deck_body`（4×4），`distance_m`（标量）。将额外根端口加入 `inputs` 即可。固定任务指令、航迹和控制参数通过现有初始化脚本或模型工作区提供。

| 默认根输出 | 尺寸 | 原生状态顺序 |
|---|---|---|
| helicopter_state | 12 | `[CG_NED_m(3); velocity_BODY_knot(3); roll,pitch,yaw_rad(3); p,q,r_BODY_rad_s(3)]` |
| ship_state | 12 | `[CG_NED_m(3); velocity_NED_knot(3); roll,pitch,heading_rad(3); p,q,r_BODY_rad_s(3)]` |

姿态 `R_NED_body=Rz(yaw)Ry(pitch)Rx(roll)`，节转 m/s 使用 `1852/3600`。
速度原点为质心，p/q/r 不是欧拉角导数。若原模型内部用 m/s，在根输出适配层转换为 knot。
状态输出不能直接依赖本轮尚未生成的导航反馈；初始状态必须由模型自身的初值确定。
已有 `uvw_Gen` 接收 **目标减飞机**，而这里输出 **飞机起落架减目标**；若复用该模块，需要在其入口取负，并对齐它使用的目标高度和参考点。不要把距离模长当三维误差。
舰姿/角速度继续采用当前接口的同步无延迟通信假设。

在配置副本中设置 `model`、`setup_script`、`navigation.initial_position` / `initial_velocity`（起落架中心在甲板系的先验）及实际端口名。航向先验由云台参考姿态给定，可显式覆盖 `initial_rotation`。宿主不会从真值自动初始化导航。

## 时序与状态保持

每个 `dt` 严格执行：读取当前模型状态 → 从当前状态渲染相机/射线生成 LiDAR → 运行现有导航 → 转换并写入反馈 → 推进内环 → 读取下一状态。`events.jsonl` 保存六个阶段。

`dt=0.02 s`，内环固定步长 `plant_step=0.02 s`，导航预测、编码器读取与控制反馈为 50 Hz；相机 10 Hz（每 5 步）、LiDAR 5 Hz（每 10 步）。没有新图像/点云的 tick 仍推进导航预测，读取同刻编码器并写入反馈。频率必须整除宿主频率，时长必须整除 dt。0.05 s 不能作为这个内环的反馈周期；旧配置会被启动校验拒绝。

两个接口 SLX 的 FixedStep、模型构建脚本、验收夹具和 CETI 默认离散周期均为 0.02 s；H_inf、LPV_H_inf、NDO、LPV_NDO 也使用 0.02 s。启动时会检查源模型的固定步长及这些模块的采样参数，防止工作副本覆盖步长后掩盖不一致。已有点质量仿真入口的默认反馈也为 50 Hz。云台内部 `integration_step_s=0.002` 是用于伺服数值稳定性的积分细分，不是内环时钟；相机曝光/读出时间和训练数据采样率各自保留其物理含义。

MATLAB `simulation` 保持一次连续仿真，每段通过 `step(PauseTime=...)` 推进；暂停时修改工作副本中的反馈 Constant。控制器内部积分器、滤波器及 MATLAB System 私有状态跨 tick 保留。原 SLX 不被运行过程修改。快照调用与转换模块相同的 `innerLoop_conversion` 内核；测试通过实际转换 SLX 检查 15 个输出一致。运行宿主不要求 `innerLoop.slx` 永久保持纯转换模型，因此可以继续在该文件中接入完整控制器和动力学。

初始化时先对模型做一次 `StopTime=0` 的零时长采样，以获得真实初值；`initialize()` 刚完成时运行时端口尚未计算，不能直接当作初值读取。随后启动一次持续仿真，第一段积分已使用 t=0 导航反馈。零时长采样不推进时间；随机动力学/初始化回调应使用自身固定种子的随机流，且初始化脚本应可重复执行。

云台在每个内环子步读取当前机体/舰船姿态，保留有限带宽、角速度/加速度/角度限制，编码器与渲染真值分离。相机曝光/滚动快门只利用当前平移速度和历史/当前旋转作短时外推，不提前推进内环获取未来姿态。Blender 持久子进程复用训练场景、海面、遮挡和噪声；本地支持 CPU Cycles，4090 使用 OPTIX。CPU 简化图像模式适用于接口测试，不用于判断训练场景中的导航精度。

## 环境和命令

在仓库根目录执行。脚本自动检查/补装 Python 依赖，记录控制台日志，不关机；`check --check` 只检查不安装。默认本地选 `.venv/bin/python`，Linux 选当前 `python`，可用 `PYTHON_BIN` 覆盖。

本地需要 MATLAB R2024b + Simulink + 与实际控制器匹配的工具箱和许可证。默认还检查 Control System Toolbox（`Control_Toolbox`）和 Robust Control Toolbox（`Robust_Toolbox`），因为仓库的 H_inf/LPV_H_inf 使用 `ss`、`c2d` 和 `hinfsyn`。`required_toolboxes` 填 MATLAB 许可证 feature 名称。已在本机 Apple Silicon MATLAB Engine 上实际运行验收。旧的 `simulation.offline.run_closed_loop --backend matlab` 默认仍是点质量代理，不是此入口。

```bash
# 检查/补装环境；--check 可改为只检查
bash shell/run_landing_closed_loop.sh check --renderer blender

# 检查现有模型的根端口/缺失 MATLAB System 类
bash shell/run_landing_closed_loop.sh inspect --model external/innerloop/simu/innerLoop_landing.slx

# 无需完整飞机模型的接口验收，真实 Simulink + 当前导航 + 简化传感器
bash shell/run_landing_closed_loop.sh smoke --duration 2

# 真实训练权重和 Blender 的传感器链路验收，仍然使用测试夹具
bash shell/run_landing_closed_loop.sh smoke --duration 0.2 \
  --renderer blender --render-device CPU \
  --checkpoint visual_training/checkpoints/full_20261014_184359_0f5f3d/best.pt

# 接好上述端口并设置先验/初始化脚本后，运行实际模型
bash shell/run_landing_closed_loop.sh run --config main/closed_loop.yaml \
  --model /absolute/path/to/complete_inner_loop.slx \
  --checkpoint /absolute/path/to/latest/best.pt --duration 60

# 从日志重新生成分析，不需要 MATLAB、训练集或权重
bash shell/run_landing_closed_loop.sh analyze outputs/closed_loop/<运行目录>
```

Ubuntu 22.04 x86_64 是 MATLAB R2024b 支持的平台；Python 3.12 可运行该版 Engine。服务器硬件足够，RTX4090 给 PyTorch CUDA 和 Blender OPTIX 使用；Simulink 执行不要求 MATLAB 的 GPU 支持。**服务器镜像里的 PyTorch/CUDA 不包含 MATLAB/Simulink**，必须另外安装正版 MATLAB R2024b、Simulink 和控制器需要的工具箱，并配置许可证；MATLAB Runtime 无法替代。检查脚本不会尝试通过 pip 安装或绕过这些产品/许可证。

```bash
export MATLAB_ROOT=/usr/local/MATLAB/R2024b
export PATH="$MATLAB_ROOT/bin:$PATH"
export PYTHON_BIN=python
# 复用此前数据生成的 cp311 bpy 环境；缺失时检查脚本会创建
export RENDER_PYTHON="$PWD/.venv-render-server/bin/python"
bash shell/run_landing_closed_loop.sh check --renderer blender
bash shell/run_landing_closed_loop.sh smoke --duration 0.2 \
  --renderer blender --render-device OPTIX --device cuda \
  --checkpoint visual_training/checkpoints/full_20261014_184359_0f5f3d/best.pt
# 完整模型/先验配置好后：
bash shell/run_landing_closed_loop.sh run --config main/closed_loop.yaml \
  --model /absolute/path/to/complete_inner_loop.slx \
  --checkpoint /absolute/path/to/latest/best.pt --device cuda --render-device OPTIX
```

Python 依赖保留服务器 Torch 2.8.0+cu128，不套用本机 Torch 2.5 的全量锁文件；bpy 在单独 Python 3.11 环境中运行。若现有模型使用只支持 Windows 的 DLL/S-function，或缺失本地编译库，仍需补齐相应平台版本；端口检查不是模型可移植性的替代证明。

## 产物与复核

每次生成独立 `outputs/closed_loop/<运行目录>/`，已存在的指定输出目录会被拒绝，避免覆盖。

- `run.json`：解析配置、几何、模型路径、权重 SHA256、环境、真实运行类型、完成/失败状态。
- `steps.jsonl`：每步当前真值、导航估计、实际控制反馈、下一时刻真值、耗时。
- `events.jsonl`：六阶段因果顺序；故障时保留已写日志。
- `position_comparison.png/pdf`、`controller_feedback.png`、`comparison.csv`、`metrics.json`、`analysis.md`：位置对比、控制输入、误差和有效反馈比例。
- `renderer.log` 与 `sensors/`：渲染日志与当前观测。`--save-sensors` 保存所有观测 NPZ；默认只保留当前观测，控制磁盘用量。
- `simulink/`：临时工作副本/验收夹具；`configure` 的 `backups/` 保存更新前模型。
- `outputs/logs/closed_loop/<时间_PID>/console.log`：含依赖检查和启动日志。

“完成”仅表示指定仿真时间执行完毕。真正触舰条件、控制品质、导航失效保护与飞机动力学正确性需要由实际模型和任务指标确认，接口夹具的误差曲线不能作为飞行性能结果。
