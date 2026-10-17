# innerLoop：着舰接口与转换模型（无动力学/控制器）

`innerLoop.slx` 是 MATLAB/Simulink R2024b 实际模型，包含 15 个根输入、一个 MATLAB System 转换模块、15 个根输出。`build_innerLoop.m` 可重建模型；`convert.m` 是 MATLAB System 类，`innerLoop_conversion.m` 是其调用的几何转换函数。构建函数会重建整个模型；已有手工改动时应更新嵌入转换函数与相关端口，避免覆盖其他模块。

云台模型和完整算法说明见 [稳定云台与闭环接口](../../../docs/gimbal_navigation.md)。`innerLoop_landing.slx` 同步更新转换模块和云台端口，保留已有制导/控制子系统。`upgrade_innerLoop_gimbal.m` 可在原模型中迁移，避免重建已有控制器。

## 输入

全部为 double。向量为列向量，时间为非负仿真秒。

| 端口 | 尺寸 | 定义 |
|---|---|---|
| timestamp | 1 | 当前已推进完成的状态时刻 |
| helicopter_state | 12 | `[CG_NED_m(3); velocity_BODY_knot(3); roll_pitch_yaw_rad(3); omega_BODY_rad_s(3)]` |
| ship_state | 12 | `[CG_NED_m(3); velocity_NED_knot(3); roll_pitch_heading_rad(3); omega_BODY_rad_s(3)]` |
| nav_T_deck_camera | 4×4 | 已有 NavigationEstimate.T_deck_camera |
| nav_velocity_deck | 3 | 已有 NavigationEstimate.velocity，甲板系相机相对速度，m/s |
| nav_covariance | 6×6 | 已有估计协方差：[甲板平移误差；相机局部右乘旋转误差] |
| nav_healthy | 1 | 0/1，导航健康状态 |
| nav_timestamp | 1 | 已有 NavigationEstimate.timestamp；控制接口要求匹配当前状态时刻 |
| gimbal_angles_rad | 3 | XYZ 实际轴角，仅用于渲染真值，Rbc=Rz Ry Rx Rbc_neutral |
| nav_R_body_camera | 3×3 | 当前编码器相机到机体旋转，用于导航反馈 |
| nav_gimbal_omega_camera | 3 | 相机相对机体角速度，以相机系表示，rad/s |
| nav_gimbal_covariance | 3×3 | 编码器相机局部右乘角度误差协方差 |
| nav_gimbal_valid | 1 | 编码器有效性 0/1 |
| nav_gimbal_timestamp | 1 | 编码器测量时刻，须等于当前状态时刻 |
| nav_angular_velocity_camera | 3 | NavigationEstimate 相机相对甲板角速度，相机系 rad/s |


姿态定义 `R_NED_body=Rz(yaw)Ry(pitch)Rx(roll)`。角速度是机体系角速度 p/q/r，不是 Euler 角导数；新增角速度输入是为了正确计算偏移点速度与旋转甲板系速度，不能互相替代。飞机和舰船必须使用同一个 NED 原点。

当前离线仿真沿用训练模型的名义尺寸：`gear_body=[0;0;2.9]`、`H_ship_body=[-45;0;-5.5]`，相机固定安装位置 `[2;0;1.11]`，光轴下倾 96°。解析矩阵和运行入口见 [`main/README.md`](../../../main/README.md)。这些参数用于对齐训练与闭环，不代表实机尺寸已标定；三者不是同一点。

## MATLAB System 参数

双击 `Frame_Unit_ReferencePoint_Conversion` 模块，在参数窗口填入以下三个非可调参数（Nontunable），不再使用信号输入端口：

| 参数 | 尺寸 | 定义 |
|---|---|---|
| T_body_camera | 4×4 | 相机中立位置 OpenCV 系到飞机机体系的标定变换；平移始终不变，旋转由云台更新 |
| H_ship_body | 3×1 | 舰船质心到 H 中心的偏移，舰体 forward/right/down 系，米 |
| gear_body | 3×1 | UH-60 质心到轮底平面中心的偏移，机体 forward/right/down 系，米 |

模型参数表达式引用同名 Model Workspace 变量，已通过 `main` 的 `configure` 命令填入当前训练几何。训练几何调整后运行 `bash shell/run_landing_closed_loop.sh configure` 重新同步；命令保留原 SLX 备份，不重建控制子系统。`convert` 类本身仍保留 NaN 作为缺少配置时的错误检查。模型的 PreLoadFcn 自动将模型所在目录加入 MATLAB 路径，供加载 convert 类和转换函数。

Python/MATLAB 适配层沿用 `config.geometry` 三个字段，通过 SimulationInput.setVariable 设置模型工作区参数，不再生成几何信号。通过适配层运行时应保留模块中默认的同名参数表达式；若手工将其改为字面常量，模块将使用该常量而非 config.geometry。中立标定在仿真前确定；动态旋转通过信号端口输入，无云台尺寸或平移自由度。缺少导航编码器时反馈无效，不能用实际轴角替代导航测量。

## 输出

| 端口 | 尺寸 | 用途 |
|---|---|---|
| T_world_deck | 4×4 | H 中心/甲板系在局部 ENU world 中的真值位姿，供现有渲染合同 |
| T_deck_camera | 4×4 | 相机在 deck 中的真值位姿，供传感器生成与评估 |
| velocity_deck | 3 | 相机相对 H 中心在旋转 deck 系中的位置导数，m/s |
| helicopter_velocity_ned_mps | 3 | 飞机 CG 体轴 knot 速度旋转为 NED m/s |
| ship_velocity_ned_mps | 3 | 舰船 CG NED knot 速度转换为 m/s |
| gear_relative_truth_ned | 3 | 轮底中心减 H 中心的 NED 真值，仅评估 |
| gear_relative_estimate_ned | 3 | 轮底中心减 H 中心的 NED 导航反馈，m |
| relative_distance_estimate_m | 1 | 上一向量的欧氏模长；不替代带方向的控制误差 |
| relative_covariance_ned | 3×3 | 导航位姿协方差传播到 NED 相对位置 |
| feedback_valid | 1 | healthy、同步时间、有限数据、合法姿态及半正定协方差同时满足 |
| navigation_velocity_camera_ned_mps | 3 | nav_velocity_deck 向量旋转到 NED；不是轮底相对速度，也不是惯性绝对速度 |
| state_timestamp | 1 | 当前状态时间 |
| T_body_camera_actual | 4×4 | 实际动态外参，仅用于渲染和评估 |
| nav_T_deck_body | 4×4 | 用导航相机位姿和编码器得到的机体位姿 |
| navigation_velocity_gear_ned_mps | 3 | 轮底相对 H 的 NED 位置导数，含云台速率和旋转甲板补偿 |


轮底在甲板上方时，NED 下向误差为负。deck 为船首/左舷/向上，H 原点，`R_NED_deck=R_NED_ship diag(1,-1,-1)`；world 是现有代码的 ENU，用 `A=[0 1 0;1 0 0;0 0 -1]` 转换。

当前假设所有舰船状态通信无延迟。转换函数内部直接使用 `ship_state(7:9)` 的横摇、纵摇和艏向计算 `nav_R_ned_deck=Rz(heading)Ry(pitch)Rx(roll)diag(1,-1,-1)`，无需外部矩阵输入。反馈位置仍由导航位姿与安装参数计算：

`r_NED = nav_R_ned_deck * (t_deck_camera + R_deck_camera * nav_R_body_camera' * (gear_body-camera_body))`

舰船姿态按理想同步通信值使用。输出协方差包含导航位姿与编码器角度误差；安装尺寸和舰船姿态不确定性尚未包含。后续考虑通信延迟或姿态噪声时，需要重新设计时间对齐和有效性判断。

## 因果顺序及动力学接入

推荐使用 [`main` 的持续 Simulink 仿真适配器](../../../main/README.md)，其 `landing_live_initialize/step` 已实现下文 plant callback，保持控制器内部状态，并在内环子步更新云台。`innerLoop` 本身仍是转换边界，不是待推进的飞机模型；完整模型须提供原生飞机/舰船状态根输出与导航反馈根输入。

SLX 是快照转换边界，`sim(..., StopTime='0')` 不推进动力学、不包含积分器。宿主的 MATLAB callbacks 管理状态与时间：

1. `initialize_external_model` 选择 `initialize_callback='innerloop_initialize'`，校验输入和 geometry。
2. `read_sensor_truth` 返回与 Python SensorTruth 一致的 JSON，并登记当前时刻已读取。
3. Python 用真值生成相机/LiDAR，通过 `read_gimbal_telemetry` 读取独立编码器测量，再运行导航。
4. `write_navigation_estimate` 写入原有 NavigationEstimate JSON；转换成轮底 NED 反馈并标记待推进。
5. `advance_external_model(dt)` 调用用户 `plant_step_callback(s,dt)`；它消费 `s.feedback`、推进飞机/舰船与控制器状态、更新舰船通信状态、令时间准确增加 dt。随后 `innerloop_update_gimbal` 推进持续的云台状态；回调可在其积分子步提前调用该函数，提高机体姿态输入采样率。
6. 下一次 `read_sensor_truth` 读取新状态。

首次推进前必须读取当前状态并提供导航反馈；同一控制时刻的重复反馈、旧时间反馈均拒绝。动力学不存在时明确报 `landing:MissingDynamics`。`allow_state_hold=true` 仅用于接口测试：状态固定，时间推进，不能称为着舰动力学闭环。传感器异步时刻可保持最近一次控制反馈，控制 tick 到期后写入新反馈。云台积分采用连续状态和细分步长，快照 `sim(StopTime=0)` 本身不积分。

控制器只应消费估计输出与 feedback_valid；真值输出不接控制反馈。有效性为 false 时的悬停/退出逻辑需由后续控制器实现，此 SLX 不假定这种能力。

Python 的 `MatlabExternalModel` 可以直接选择以上 callback 配置，沿用相机位姿主字段，并增加动态外参、编码器和角速度字段；新增 `read_navigation_feedback()` 可检查实际送到内环的轮底误差。启用该模型前应同步配置 camera.yaml、建模/导航 H 尺寸和初始化相机相对位姿，勿用原默认相机先验替代已知安装几何。

## 重建与验证

MATLAB 根目录运行：

```matlab
addpath('external/innerloop/simu'); upgrade_innerLoop_gimbal; % 原位迁移两个模型
% build_innerLoop; % 仅在需要重建空白接口模型时使用
addpath('simulation/matlab'); test_innerloop_interfaces; test_gimbal_interfaces;
```

测试运行实际 SLX，覆盖 knot 转换、NED/ENU/甲板旋转、不同偏移点、旋转参考系速度有限差分、协方差、失效/过期反馈及因果顺序。测试尺寸为人工非对称夹具，不是 UH-60 实际参数。
