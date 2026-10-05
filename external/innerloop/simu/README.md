# innerLoop：着舰接口与转换模型（无动力学/控制器）

`innerLoop.slx` 是 MATLAB/Simulink R2024b 实际模型，包含 12 个根输入、一个 MATLAB Function 转换模块、12 个根输出。`build_innerLoop.m` 可重建模型；`innerLoop_conversion.m` 是嵌入模块的可审阅源码。修改源码后需重新执行构建函数。

路径按本次要求使用 `external/innerloop/simu`，原有 `external/inner_loop` 及其 SLX 不作改动。

## 输入

全部为 double。向量为列向量，时间为非负仿真秒。

| 端口 | 尺寸 | 定义 |
|---|---|---|
| timestamp | 1 | 当前已推进完成的状态时刻 |
| helicopter_state | 12 | `[CG_NED_m(3); velocity_BODY_knot(3); roll_pitch_yaw_rad(3); omega_BODY_rad_s(3)]` |
| ship_state | 12 | `[CG_NED_m(3); velocity_NED_knot(3); roll_pitch_heading_rad(3); omega_BODY_rad_s(3)]` |
| T_body_camera | 4×4 | 相机 OpenCV 系到飞机机体系的刚体变换，包括安装位置 |
| H_ship_body | 3 | 舰船质心到 H 中心的偏移，舰体 forward/right/down 系，米 |
| gear_body | 3 | UH-60 质心到起落架轮底平面中心的偏移，机体 forward/right/down 系，米 |
| nav_T_deck_camera | 4×4 | 已有 NavigationEstimate.T_deck_camera |
| nav_velocity_deck | 3 | 已有 NavigationEstimate.velocity，甲板系相机相对速度，m/s |
| nav_covariance | 6×6 | 已有估计协方差：[甲板平移误差；相机局部右乘旋转误差] |
| nav_healthy | 1 | 0/1，导航健康状态 |
| nav_timestamp | 1 | 已有 NavigationEstimate.timestamp；控制接口要求匹配当前状态时刻 |
| nav_R_ned_deck | 3×3 | 导航可获得的甲板系到 NED 旋转，不能从模型真值自动取用 |

姿态定义 `R_NED_body=Rz(yaw)Ry(pitch)Rx(roll)`。角速度是机体系角速度 p/q/r，不是 Euler 角导数；新增角速度输入是为了正确计算偏移点速度与旋转甲板系速度，不能互相替代。飞机和舰船必须使用同一个 NED 原点。

UH-60 几何尚未指定具体型号、质心/起落架构型及标定结果，因此模型不内置未经核实的尺寸。`gear_body` 要由所选构型的轮胎接触点确定中心、明确轮胎载荷/压缩状态；着陆后是否使用固定值或时变值由后续模型决定。H 中心相对舰船质心和相机安装外参同样必须显式输入。三者默认不能视为同一点。

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

轮底在甲板上方时，NED 下向误差为负。deck 为船首/左舷/向上，H 原点，`R_NED_deck=R_NED_ship diag(1,-1,-1)`；world 是现有代码的 ENU，用 `A=[0 1 0;1 0 0;0 0 -1]` 转换。

反馈计算仅使用导航位姿、安装参数与 **独立导航姿态 nav_R_ned_deck**：

`r_NED = nav_R_ned_deck * (t_deck_camera + R_deck_camera * R_body_camera' * (gear_body-camera_body))`

导航 NED 姿态和安装尺寸的不确定性尚未包含在输出协方差中；该协方差是这些参数给定时的条件协方差。可用飞机惯导姿态加相机安装姿态和视觉相对姿态求 nav_R_ned_deck，或由舰船姿态链路获得。具体来源待后续导航接口补充，绝不从 truth 通道静默替代。

## 因果顺序及动力学接入

SLX 是快照转换边界，`sim(..., StopTime='0')` 不推进动力学、不包含积分器。宿主的 MATLAB callbacks 管理状态与时间：

1. `initialize_external_model` 选择 `initialize_callback='innerloop_initialize'`，校验输入和 geometry。
2. `read_sensor_truth` 返回与 Python SensorTruth 一致的 JSON，并登记当前时刻已读取。
3. Python 用真值生成相机/LiDAR，再运行导航。
4. `write_navigation_estimate` 写入原有 NavigationEstimate JSON；转换成轮底 NED 反馈并标记待推进。
5. `advance_external_model(dt)` 调用用户 `plant_step_callback(s,dt)`；它消费 `s.feedback`、推进飞机/舰船与控制器状态、更新导航姿态来源、令时间准确增加 dt。
6. 下一次 `read_sensor_truth` 读取新状态。

调用顺序错误、重复反馈、旧时间反馈均拒绝。动力学不存在时明确报 `landing:MissingDynamics`。`allow_state_hold=true` 仅用于接口测试：状态固定，时间推进，不能称为着舰动力学闭环。测试时相机/雷达/控制可同频，或保证每次状态推进前已有该时刻的控制反馈；任意异步调度策略需要显式设计保持和控制采样。

控制器只应消费估计输出与 feedback_valid；真值输出不接控制反馈。有效性为 false 时的悬停/退出逻辑需由后续控制器实现，此 SLX 不假定这种能力。

Python 的 `MatlabExternalModel` 可以直接选择以上 callback 配置，不需要更换 SensorTruth/NavigationEstimate 合同；新增 `read_navigation_feedback()` 可检查实际送到内环的轮底误差。启用该模型前应同步配置 camera.yaml、建模/导航 H 尺寸和初始化相机相对位姿，勿用原默认相机先验替代已知安装几何。

## 重建与验证

MATLAB 根目录运行：

```matlab
addpath('external/innerloop/simu'); build_innerLoop;
addpath('simulation/matlab'); test_innerloop_interfaces;
```

测试运行实际 SLX，覆盖 knot 转换、NED/ENU/甲板旋转、不同偏移点、旋转参考系速度有限差分、协方差、失效/过期反馈及因果顺序。测试尺寸为人工非对称夹具，不是 UH-60 实际参数。
