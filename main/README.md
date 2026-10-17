# 导航接入内环的离线闭环宿主

主入口为 `main/run.py`，结果分析为 `main/analyze.py`，MATLAB 适配器在 `main/matlab/`。
复用现有 `NavigationPipeline`、训练场景、LiDAR 和云台动力学；不新增直升机动力学、制导或飞行控制律。

**当前仓库模型的实际状态**：`external/innerloop/simu/innerLoop.slx` 是 15 输入/15 输出的坐标转换模型，尚无飞机动力学。`innerLoop_landing.slx` 含制导和控制子系统，但目前仍以外部飞机/舰船状态为根输入，缺少完整原生状态根输出。因此不能直接宣称这两个文件已经完成直升机着舰。`inspect` 会列出缺失接口、类及其实际来源路径；补齐现有模型或提供一个符合下述端口合同的完整模型，即可用相同的 `run` 入口闭环。`smoke` 自动生成的 Simulink 积分器夹具只用于接口验收。

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

`configure` 原位更新两个 SLX 的三个 Model Workspace 变量及同名模块参数表达式，保存原文件备份；不会重建模型或删除已有制导/控制连线。直接打开模型即可看到这些参数。后续改变训练几何后请重新执行：

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

`dt=0.05 s`，内环固定步长 `plant_step=0.01 s`；相机 10 Hz、LiDAR 5 Hz。没有新图像/点云的 tick 仍推进导航预测，读取同刻编码器并写入反馈。频率必须整除宿主频率，时长必须整除 dt。不要简单提高传感器配置频率以掩盖时序错误。

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
