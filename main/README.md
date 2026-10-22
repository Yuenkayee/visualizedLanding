# 导航接入内环的离线闭环宿主

主入口为 `main/run.py`，结果分析为 `main/analyze.py`，MATLAB 适配器在 `main/matlab/`。
复用现有 `NavigationPipeline`、训练场景、LiDAR 和云台动力学；不新增直升机动力学、制导或飞行控制律。

**当前仓库模型的实际状态（2026-10-22）**：`external/innerloop/simu/innerLoop.slx` 是坐标转换模型。用户更新的 `innerLoop_landing.slx` 已由内部 `innerLoop` 子系统生成飞机/舰船状态，制导和控制在该子系统内部闭环；导航输入仅接到根层转换模块。使用 `main/navigation_observer.yaml` 可按当前实际航迹旁路评估导航，原 SLX 不被修改。`main/closed_loop.yaml` 仍是供将来导航反馈控制使用的通用端口合同，不能直接用于当前根接口。`smoke` 仍只使用积分器验收夹具。

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

初始化脚本 `main/matlab/landing_setup_observer.m` 加载仓库自带 `landingParam.mat`。原状态输出在 `innerLoop/helicopter_state` / `innerLoop/ship_state`，宿主在工作副本中加观察分支，无需增加源模型根输出。舰船状态的速度实际为 BODY 坐标，配置 `ship_velocity_frame: body` 在进入导航桥接层前旋转为 NED；原 SLX 转换模块自身的速度根输出尚不做该修正，请以宿主规范化后的传感器状态为准。状态位置不会自动平移去迎合制导目标。

2026-10-22 更新的模型将自主子系统从 `innerLoop1` 改名为 `innerLoop`。修改模块名称时，必须同时更新配置的 `autonomous_subsystem`、`outputs.helicopter_state`、`outputs.ship_state`。若报 `Autonomous subsystem is missing`，先检查报错列出的实际子系统和配置路径，再运行 `inspect`。共享 MATLAB 会话中若已打开旧模型，先保存需要保留的编辑并关闭该模型，再从磁盘重载。

每步在同一时间戳保存 `gear_relative_estimate_ned` 和 `gear_relative_truth_ned`，均为**轮底中心减 H 中心**的 NED 向量，单位 m。真值从 native 状态的同一转换内核取得，估计由本轮导航转换得到。输出 `gear_relative_ned_comparison.png/pdf` 为 North/East/Down 三行真值与估计曲线，`gear_relative_ned.csv` 为数据；无效估计以红点显示。旧的甲板系 `position_comparison.png/pdf` 保留。`controller_feedback.png` 在旁路运行中只是转换后的估计，不能解释为实际控制输入。

STOP 触发后读取保留的末帧输出，退出 Python 循环，写 `termination_reason: model_stop`、MATLAB `stop_reason` 和 `terminal_state.json`，照常生成图和误差统计。边界时刻可能出现最后一次 `advanced_s=0`：Simulink 在暂停时刻的下一次执行才求值 STOP，这是正常终止，不是假装推进一个 dt。运行错误仍报错；不会吞掉求解器错误。`duration_reached` 表示达到用户指定的运行上限。

每个内环主步另写 `simulink/plant_steps.jsonl`，记录起始时间、目标时间、实际时间、前后状态及 `step()` 的 `final_step` 标志。若仍出现时间校验错误，控制台给出这些数值，并在该文件保留出错步。终止后 `Simulation.Time` 可能为 NaN（JSON 中为 null），此时以保留输出的 `terminal_time` 为准。旧版 `Plant stopped or missed requested time` 未包含这些信息，单凭该消息不能判断 STOP、暂停或超步；控制台进度每 1 s 才打印一次，最后看到 `iteration=0` 不表示必然在第二步报错。

**场景初值（2026-10-20 复核）**：制导 `Ship_trace` 和导出的 `Ship_motion` 质心初值均为 `[50;0;0]`，直升机质心初值为 `[0;0;-50]`。训练 H 偏移仍为 `[-45;0;-5.5]`，所以水平名义轮底中心在甲板系的先验为 `[-5;0;41.6]`，已同步到旁路配置。这个先验由既定场景初值计算，不读取运行时真值。后续修改初始状态时也需重新核对该先验。

2026-10-20 的模型按用户要求保留 `abs(-CG_D-2.9)<0.01`，制导目标与 H 点使用不同参考点；该历史配置不能据此判定成功触舰。用户于 2026-10-22 更新了制导模型以考虑舰船几何，当前目标与 STOP 以 SLX 中保存的实现为准；本次路径修复不覆盖它们。

控制器参数修改需保存到实际运行的 `external/innerloop/simu/innerLoop_landing.slx`；旁路配置不覆盖控制器增益。

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

MATLAB `simulation` 保持一次连续仿真，每个内环周期通过 `step(NumberOfSteps=1)` 推进一个主步，再校验实际时间与整数 tick 计算的目标时间；暂停时修改工作副本中的反馈 Constant。同时检查 `step()` 返回的 `finalstep` 和 `Status`：普通模式的最后一步可能暂停而非立即成为 inactive，此时先结束执行并保留末帧，再按模型停止处理。不得通过放宽时间误差、额外补步或重写时间戳来掩盖时钟错位。控制器内部积分器、滤波器及 MATLAB System 私有状态跨 tick 保留。原 SLX 不被运行过程修改。快照调用与转换模块相同的 `innerLoop_conversion` 内核；测试通过实际转换 SLX 检查 15 个输出一致。运行宿主不要求 `innerLoop.slx` 永久保持纯转换模型，因此可以继续在该文件中接入完整控制器和动力学。

初始化时先对模型做一次 `StopTime=0` 的零时长采样，以获得真实初值；`initialize()` 刚完成时运行时端口尚未计算，不能直接当作初值读取。随后启动一次持续仿真，第一段积分已使用 t=0 导航反馈。零时长采样不推进时间；随机动力学/初始化回调应使用自身固定种子的随机流，且初始化脚本应可重复执行。

云台在每个内环子步读取当前机体/舰船姿态，保留有限带宽、角速度/加速度/角度限制，编码器与渲染真值分离。相机曝光/滚动快门只利用当前平移速度和历史/当前旋转作短时外推，不提前推进内环获取未来姿态。Blender 持久子进程复用训练场景、海面、遮挡和噪声；本地支持 CPU Cycles，4090 使用 OPTIX。CPU 简化图像模式适用于接口测试，不用于判断训练场景中的导航精度。

## Ubuntu 22.04 一次部署

当前部署入口和默认权重位于 `single_cam` 分支。新服务器克隆时使用 `git clone --branch single_cam git@github.com:Yuenkayee/visualizedLanding.git`，已有仓库同步该分支后再运行下列命令。

所有命令在克隆后的仓库根目录执行。统一入口为 `shell/run_landing_closed_loop.sh`，可以由 Ubuntu 自带 Python 3.10 启动部署，无需先手动安装 Python 3.12。`check` 自动安装缺失依赖；`check --check` 只检查并生成报告，不安装、不登录。执行日志从系统安装阶段开始保存在 `outputs/logs/closed_loop/<时间_PID>/console.log`。

Linux 自动部署支持 Ubuntu 22.04 x86_64，安装系统包需要 root 或 sudo 权限。GPU 仿真使用云平台已经配置好的 NVIDIA 驱动和 GPU/graphics 设备访问；脚本检查 PyTorch CUDA 运算与真实 Cycles OPTIX 小图渲染，不修改云宿主机内核驱动，也不安装另一套 CUDA Toolkit。容器里驱动/设备不可见时，需要在云平台选择 GPU 镜像并开放设备。

首次部署和账号授权：

```bash
bash shell/run_landing_closed_loop.sh check --login \
  --config main/navigation_observer.yaml --device cuda --render-device OPTIX
```

MATLAB 首次授权需要本人按终端提示输入账号/OTP；成功进入 `>>` 后输入 `exit`，部署会继续。此交互直接连接 `/dev/tty`，不写入 `console.log`，必须在交互式 SSH 或 tmux 内执行。已有许可证文件、网络许可证或已经登录时，省略 `--login`。账号授权设置 `MATLAB_LICENSE_MODE=online`；如许可证使用 MathWorks New Licensing Technology，设置 `MATLAB_LICENSE_MODE=onlinelicensing`。环境检查与实际仿真都向 Engine 传入同一个 `-licmode`，Linux 同时传入 `-nodisplay`。

`--login` 无既有配置时默认选择 `online`；成功退出交互 MATLAB 后，仅将模式和本地安装/用户标识保存在 `.tools/closed-loop/matlab-startup.json`，不存账号、密码或令牌。同一用户、同一安装目录的后续 `check/run` 自动复用该选择，环境变量可覆盖。无配置时非交互启动保持 `auto`（不传 `-licmode`，保留 MATLAB 默认许可证搜索）；许可证文件/网络许可证用户也可显式设置 `MATLAB_LICENSE_MODE=file` 并配置自己的 `MLM_LICENSE_FILE`，无需 `--login`。

若日志已经出现 Blender/OPTIX 渲染成功，但随后报 `Unable to launch MVM server: License Error: Licensing shutdown`，当前阻塞是 MATLAB 授权启动。尤其注意旧版本仅登录时传 `-licmode online`、Engine 未传的配置差异。更新代码后，用同一 Linux 用户执行：

```bash
export MATLAB_LICENSE_MODE=online
bash shell/run_landing_closed_loop.sh check --login \
  --config main/navigation_observer.yaml --device cuda --render-device OPTIX
```

本次日志目录的 `matlab-startup.json` 记录实际模式、启动选项和结果。该错误没有具体授权编号时，不能仅凭它判定许可证过期；若直接执行 `"$MATLAB_ROOT/bin/matlab" -licmode online -nodisplay` 也失败，应根据交互终端的具体错误检查账号许可、MathWorks Service Host 和官方授权服务连接。无需因此重装 bpy/CUDA。官方说明：[终端账号授权](https://www.mathworks.com/matlabcentral/answers/547302-is-login-named-user-available-on-machines-without-a-display)、[Login Named User 启动模式](https://www.mathworks.com/matlabcentral/answers/168441-what-is-login-named-user-and-how-can-i-use-it)。

如果已经确认交互 MATLAB 能进入 `>>`，但退出后 Engine 即使使用 `-licmode online` 仍启动失败，可以连接一个保持运行的已授权 MATLAB 会话。此方式使用 MathWorks 官方的 [共享 Engine 接口](https://www.mathworks.com/help/matlab/matlab_external/connect-python-to-running-matlab-session.html)，仍然依赖有效 MATLAB/工具箱许可证；它避免了 Python 创建另一个 MATLAB 进程，但不证明原错误的底层原因已修复。

在服务器创建专用 tmux 会话：

```bash
tmux new -s landing_matlab
```

在该 tmux 内启动 MATLAB（账号输入留在交互终端）。此命令只移除独立 MATLAB 子进程的动态库覆盖；Python 仍经原有 shell 入口使用 C++ 兼容设置：

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH \
  /root/MATLAB/R2024b/bin/matlab -licmode online -nodisplay -nosplash
```

进入 `>>` 后执行：

```matlab
matlab.engine.shareEngine('LandingMATLAB')
disp(matlab.engine.engineName)
```

确认打印 `LandingMATLAB`，保持 MATLAB 运行，按 `Ctrl+B` 后按 `D` 脱离 tmux（此流程不要输入 `exit`）。在同一服务器/容器、同一 Linux 用户的 shell 中执行：

```bash
cd /root/autodl-tmp/visualizedLanding
export MATLAB_ROOT=/root/MATLAB/R2024b
export PYTHON_BIN=/root/miniconda3/bin/python3
export MATLAB_SHARED_SESSION=LandingMATLAB
bash shell/run_landing_closed_loop.sh check \
  --config main/navigation_observer.yaml --device cuda --render-device OPTIX
bash shell/run_landing_closed_loop.sh run \
  --config main/navigation_observer.yaml --device cuda --render-device OPTIX
```

此流程不加 `--login`。环境检查和实际仿真都会按名称连接，检查返回 `ready` 后即可运行；名称不存在或连接失败时直接报错，不自动启动另一会话。报告区分 `connection=shared`、`status=connected`、`license_mode=existing_session`，并验证实际安装为指定目录的 R2024b。环境检查/仿真结束时 `engine.quit()` 对连接来的共享会话仅断开 Python 连接，MATLAB 留在 tmux；仿真仍执行自身模型清理。该会话供一个仿真作业使用，运行期间不要在 MATLAB 终端手动修改工作区。全部完成后可 `tmux attach -t landing_matlab` 并在 MATLAB 输入 `exit`。取消共享连接模式使用 `unset MATLAB_SHARED_SESSION`。

自动检查/安装的内容：

- MATLAB R2024b、Simulink、Control System Toolbox、Robust Control Toolbox，通过官方 MPM 安装；新装采用 R2024b 最新更新，补装工具箱保留已有更新级别。默认目录 `~/MATLAB/R2024b`，root 用户即 `/root/MATLAB/R2024b`。已有标准路径安装会复用，也可用 `MATLAB_ROOT` 显式指定。
- Linux 启动入口在运行 Python 前，将所选 MATLAB 的 `bin/glnxa64` 加入 `LD_LIBRARY_PATH`，使 MATLAB Engine 的隔离构建和运行进程识别自定义安装目录；保留现有库路径，不修改 `.bashrc`。仅设置 `MATLAB_ROOT` 和 `PATH` 不足以让 PyPI 的 Engine 构建器找到 `/root/MATLAB/R2024b`。
- 若 Python/Conda 的 `libstdc++.so.6` 缺少 MATLAB R2024b 所需的 `GLIBCXX_3.4.30`，部署会检查 Ubuntu 系统 C++ 库（缺少符号时通过 apt 更新），在独立 Python 进程验证已安装的 Torch/Engine 导入后，仅为本次检查和仿真设置 `LD_PRELOAD`。不删除或替换 Conda 文件；已有预加载项保留。选择结果记录在同次日志目录的 `cpp-runtime.json`。`--check` 不安装/更新系统包。直接使用 `python -m` 不经过此启动修复，服务器请使用 shell 入口。
- MathWorks 官方 Ubuntu 22.04 系统依赖和 Blender 运行库、编译工具、pip/venv、tmux。依赖清单随仓库保存，不再从 GitHub 临时下载清单。
- 导航使用 Python 3.12：优先复用项目 `.venv` 或云镜像的兼容 Python；否则自动创建 `.venv`。显式 `PYTHON_BIN` 必须可运行且为 Python 3.11/3.12，不会悄悄改用其他解释器。自动部署推荐 Linux 3.12。
- Blender 使用独立 `.venv-render-server` 的 Python 3.11 和 bpy 4.5.3。支持 `RENDER_PYTHON` 或 `--render-python` 指向已有的 Python 3.11 环境。
- Blender 子进程使用独立的动态库环境：从继承的 `LD_LIBRARY_PATH` / `LD_PRELOAD` 中移除 MATLAB 安装目录，保留 CUDA/NVIDIA 路径及系统 C++ 兼容库。此规则覆盖导入检查、Cycles 设备测试、闭环传感器进程和训练集渲染进程。Ubuntu 依赖包含 `libvulkan1`；导入 Blender 不要求桌面/X server。闭环检查在同次日志目录写入 `renderer-environment.json`，记录渲染解释器、动态库路径和导入结果；失败时附加 USD 库的 `ldd` 信息，供定位 `undefined symbol: vkGetPhysicalDeviceXlibPresentationSupportKHR` 等冲突。MATLAB 主进程的环境保持原有配置。
- PyTorch 2.8.0+cu128、MATLAB Engine 24.2.2、数值/图像/绘图依赖，实际 CUDA 运算、OPTIX 渲染和工具箱许可证 checkout。
- 加载默认权重，核对其文件存在和模型格式，计算 SHA256 并记录到 `environment.json`。

Python 下载会重试；uv 获取运行时失败时，会自动下载固定版本源码并编译（3.12.12 / 3.11.14），可能需要几分钟。默认使用 GitHub 二进制和 `www.python.org` 源码，镜像模式下二者均切换。已完成的步骤在重跑时复用。源码、构建运行时和 MPM 缓存在 `.tools/closed-loop/`；失败环境保留为 `.venv.backup.*`，均被 Git 忽略。可通过 `LANDING_PYTHON_312` / `LANDING_PYTHON_311` 指定已有解释器。

国内服务器下载较慢时，先用 Ctrl+C 停止当前安装，再启用镜像配置重跑：

```bash
export LANDING_DOWNLOAD_PROFILE=china
bash shell/run_landing_closed_loop.sh check \
  --config main/navigation_observer.yaml --device cuda --render-device OPTIX
```

该配置只作用于启动进程及其子进程；不写全局 pip/uv 配置，不改 `/etc/apt/sources.list`。apt 在需要安装或更新时使用项目 `.tools/closed-loop/apt/sources.list`，仍验证 Ubuntu 官方签名；`--check` 不创建此列表、不执行 apt 安装。Python 包索引与 CPython 解释器下载是两种不同服务，不能都填写同一个 PyPI 地址。

| 下载内容 | `china` 配置使用的来源 |
| --- | --- |
| pip/uv 的常规包（含 uv、bpy、MATLAB Engine） | 清华 PyPI：`https://pypi.tuna.tsinghua.edu.cn/simple` |
| uv 下载的 CPython 二进制 | 南大：`https://mirror.nju.edu.cn/github-release/astral-sh/python-build-standalone` |
| 二进制下载失败后的 Python 源码 | 清华：`https://mirrors.tuna.tsinghua.edu.cn/python` |
| Ubuntu 22.04 系统包和源码编译依赖 | 清华：`https://mirrors.tuna.tsinghua.edu.cn/ubuntu` |
| PyTorch 2.8.0+cu128 | 保留 `https://download.pytorch.org/whl/cu128`，已有正确版本不重装 |
| MATLAB 主程序、工具箱、授权 | 保留 MathWorks 官方服务 |

显式选择 `china` 会覆盖当前终端遗留的 `PIP_INDEX_URL`、`UV_DEFAULT_INDEX`、`UV_PYTHON_INSTALL_MIRROR`。需要自定义时，在启动前设置 `LANDING_PYPI_INDEX`、`LANDING_PYTHON_BINARY_MIRROR`、`LANDING_PYTHON_SOURCE_MIRROR` 或 `LANDING_APT_MIRROR`。PyTorch 专用源可用 `LANDING_TORCH_INDEX` 指定，但必须包含 `torch==2.8.0+cu128` 及依赖。使用 `LANDING_DOWNLOAD_PROFILE=default` 则保留调用终端原有的源配置。相关格式见 [清华 PyPI 帮助](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)、[清华 Python 源码镜像](https://mirrors.tuna.tsinghua.edu.cn/help/python/) 和 [uv 运行时镜像配置](https://docs.astral.sh/uv/reference/environment/#uv_python_install_mirror)。网络可达性仍取决于服务器出口，不关闭 TLS 校验。

不要从 macOS 复制虚拟环境到 Linux。若之前留下了指向不存在文件的 `PYTHON_BIN` / `RENDER_PYTHON`，先 `unset PYTHON_BIN RENDER_PYTHON` 再部署。自定义 MATLAB 目录示例：

```bash
export MATLAB_ROOT=/root/MATLAB/R2024b
bash shell/run_landing_closed_loop.sh check --device cuda --render-device OPTIX
# 以后仅核对，不安装、不登录
bash shell/run_landing_closed_loop.sh check --check --device cuda --render-device OPTIX
```

`check/run/inspect` 的 shell 默认配置为 `main/navigation_observer.yaml`；`smoke/configure` 默认保留 `main/closed_loop.yaml`。直接调用 Python 模块不会执行系统/MATLAB 安装，完整部署请使用 shell 入口。

## 默认权重与仿真命令

`visual_training/checkpoints/four_weather_v1/best.pt` 已作为普通 Git 文件纳入仓库，克隆即有，不需要 Git LFS 或另行 SCP。SHA256 为 `1605b4a258414d931f08c630dc1bd83c3946f1a91bfa595614ee0e54df186ab3`。其他训练权重、训练集、历史结果仍不随 Git 同步。真实航迹仿真现场生成图像/点云，不需要训练集和预生成 `.blend`。

先做模型端口检查及 2 秒实际航迹试运行：

```bash
bash shell/run_landing_closed_loop.sh inspect --config main/navigation_observer.yaml
bash shell/run_landing_closed_loop.sh run --config main/navigation_observer.yaml \
  --duration 2 --device cuda --render-device OPTIX
```

正式运行（60 秒为仿真时间上限，源模型 STOP 可以提前结束）：

```bash
tmux new-session -d -s landing -c "$PWD" \
  'bash shell/run_landing_closed_loop.sh run --config main/navigation_observer.yaml --duration 60 --device cuda --render-device OPTIX'
tmux attach -t landing
```

`Ctrl+B` 后按 `D` 离开会话并保持运行；`Ctrl+C` 请求终止。结束不会关机。`check --login` 应先在交互终端完成，再启动后台运行。当前配置仍然是**自主控制闭环、导航旁路估计**，导航不反馈控制器；本次部署改动没有改变这个评估模式。

可选接口夹具 smoke（不代表实际飞机着舰）：

```bash
bash shell/run_landing_closed_loop.sh smoke --duration 0.2 \
  --renderer blender --device cuda --render-device OPTIX \
  --checkpoint visual_training/checkpoints/four_weather_v1/best.pt
```

终端第一步、约每推进 1 秒仿真时间和最后一步打印 `event=step`、`iteration`、仿真 `time`、`feedback_valid` 和本步实际耗时 `wall_s`。结束打印 `elapsed_s`；STOP 打印 `model_stop`。渲染耗时时两条进度的实际间隔可能较长，不是每秒墙钟心跳。Blender 详细信息写到结果目录 `renderer.log`。

本地 macOS 保留已有 MATLAB R2024b 和 Python 环境，不执行 apt/MPM 自动安装。可以用以下命令检查和运行：

```bash
bash shell/run_landing_closed_loop.sh check --check --renderer cpu --device cpu
bash shell/run_landing_closed_loop.sh run --config main/navigation_observer.yaml --duration 2
# 仅分析已有结果，不启动 MATLAB
bash shell/run_landing_closed_loop.sh analyze outputs/closed_loop/<运行目录>
```

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
