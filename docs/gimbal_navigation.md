# 稳定云台与闭环接口

相机光心仍固定在原来的机体安装位置，只有朝向随云台转动。控制目标是保持固定的 **相对甲板姿态**，不是把相机位置锁在空中，也不是逐帧理想 look-at H；平移、透视尺度、机身遮挡、触及云台限位时的出画仍然存在。

## 动力学与输入假设

`navigation/common/gimbal.py` 与 `external/innerloop/simu/gimbal_step.m` 使用相同的三轴 XYZ 模型：

```
R_body_camera(t) = Rz(qz) Ry(qy) Rx(qx) R_body_camera_neutral
t_body_camera(t) = t_body_camera_neutral
qdd = wn² (q_command - q) + 2 ζ wn (q_command_dot - qdot)
```

这是 `J qdd + B qdot = torque` 的闭环二阶伺服近似；忽略轴间惯性耦合、摩擦、云台尺寸及轴心间距。加速度限幅代表等效力矩限制，并另有轴速率与转角限幅；到达机械限位时取消向外速度。`rate_feedforward=false` 可关闭速度前馈。默认固有频率 10 Hz、阻尼比 0.9、轴速率 3 rad/s、轴加速度 40 rad/s²、转角范围约 ±80°/±80°/±120°，都是待辨识的设计参数。积分自动细分到不大于 2 ms 和 `0.1/wn`，不依赖图像采样率。

相对甲板目标旋转可由 `R_deck_camera_reference` 显式给出；缺省采用水平飞机原安装角对应的视角。云台控制器使用机载姿态与舰船姿态遥测，仿真暂按无延迟、无噪声提供这两路控制输入。导航滤波器不接收飞机真值姿态，只接收图像、雷达及云台编码器。未来若要模拟 AHRS 或舰姿通信误差，应在云台控制输入处加入误差，不能把理想相机姿态作为视觉观测。

`initial_angles_rad=null` 表示仿真开始前已经捕获目标姿态、初始轴速度为零；指定角度可测试捕获过程。每个航次独立初始化，航次中云台连续积分。相同时间戳的重复读取不推进动力学。`enabled=false` 用于刚性安装消融。Python 参数默认取自 `configs/gimbal.yaml`；MATLAB 单独运行默认值在 `gimbal_defaults.m`，Python Engine 会发送完整解析配置。

## 导航参考点与雷达

网络输出的 H 分割、四角顺序和 PnP 输入保持原合同。PnP 仍输出相机位姿 `T_deck_camera`。当设置 `navigation.filter_point_body`（默认 CG `[0,0,0]`）时，ESKF 的平移/速度改为该固定机体点，旋转/角速度仍为相机相对甲板的量：

```
lever_camera = R_body_camera' (camera_body - filter_point_body)
p_filter = p_camera - R_deck_camera lever_camera
v_camera = v_filter + R_deck_camera cross(omega_deck_camera - omega_body_camera, lever_camera)
```

这样大幅机体摇摆产生的相机杠杆臂运动不会进入常速度平移模型。初始位置先验现在对应 `filter_point_body`；姿态先验仍对应相机。`filter_point_body=null` 可保留原相机平移状态。视觉位姿协方差通过雅可比转换，考虑编码器右乘角度误差；输出再次转换回原相机合同。这里采用独立误差近似，未估计固定安装尺寸/舰姿误差，也未维护编码器与滤波状态的跨时刻相关性。

默认雷达继续固定在机体，不随相机云台转动。`lidar_mount=body` 必须提供固定 `T_body_lidar`，测量时刻使用：

```
T_camera_lidar(t) = inverse(T_body_camera_encoder(t)) @ T_body_lidar
```

雷达点云转换到滤波参考点后再做甲板关联、平面拟合与 ESKF 更新；编码器角度协方差也进入平面法向观测噪声。若实际雷达与相机共同安装于云台，可显式配置 `lidar_mount=camera`，此时使用固定 `T_camera_lidar`。不允许机体固定雷达静默使用旧的相机固定外参。

`GimbalSample` 含测量时刻、动态 `T_body_camera`、相机相对机体角速度（相机系）、3×3 相机局部右乘角度协方差、有效性和限幅标记。CameraFrame/LidarFrame 在各自测量时刻携带该样本；NavigationEstimate 另含相机相对甲板角速度及同刻云台样本。缺失、无效或时间不匹配的编码器不能生成有效机体反馈；只有触及限幅并不自动使几何变换无效，编码器仍可测量实际姿态。

机体位姿为 `T_deck_body=T_deck_camera @ inverse(T_body_camera_encoder)`。轮底位置使用当前编码器将机体偏移转换到相机系；速度同时扣除云台相对角速度，再补上旋转甲板的输运项。位置协方差包括 PnP/ESKF 及编码器的贡献。实现分别位于 `navigation/common/reference_points.py` 和 `innerLoop_conversion.m`。

## 训练、渲染与回放

三阶段生成器保留原来的 0.6/0.05/0.3 rad 机体姿态包络，使用上述有限带宽云台计算每个时刻的真实相机姿态。机身、轮子与支柱随飞机转动；RGB、mask、关键点、遮挡射线和雷达各自使用同刻变换。曝光模糊和滚动读出基于云台作用后的实际相机运动。

航次顶层 `T_body_camera` 是中立安装标定；每行 `T_body_camera` 才是实际动态外参。标注 schema_version=2、计划 schema_version=3，额外保存编码器样本、轴角/轴速率、跟踪误差、云台配置、雷达安装方式及外参，并生成同刻雷达 NPZ。回放使用保存的编码器，不用标注真值反推编码器。原有 HDataset/网络损失接口兼容，旧刚性数据和旧权重应重新生成/训练以覆盖新成像分布。

默认 `configs/offline_simulation.yaml` 通过 `landing_config` 复用 `landing_dataset.yaml` 的相机内参、原安装点、起落架点与云台配置。快速 CPU 数据生成器和无 landing_config 的闭环仍使用 `configs/camera.yaml` 的简化夹具。实际 MATLAB 飞行模型需使用已标定的 camera/geometry/雷达配置，不能把训练代理尺寸当成标定数据。

默认 mock 的 `position_reference=gear` 把轮底中心作为点质量运动参考点，和训练轨迹一样独立指定该点运动与机体姿态，因此该演示在轮底点滤波平移。`position_reference=body` 则在 CG 滤波。二者都不是六自由度飞行动力学；后者在强制持续大摇摆时可能超过有界加速度控制器的轮底补偿能力。实际 innerloop 原生状态始终是 CG，不改变其物理含义。

## Simulink 与时间推进

`innerLoop.slx` 仍是无飞行动力学的快照转换边界。原 8 个输入保留，新增 7 个输入和 3 个输出，详见同目录 README。`innerLoop_landing.slx` 同步迁移接口，保留其现有制导/控制子系统。`upgrade_innerLoop_gimbal` 在原模型中更换转换模块并保存周围连接；`build_innerLoop` 则用于从零重建接口模型。

宿主 `innerloop_initialize/innerloop_advance` 管理持续存在的 `s.gimbal_state`。`plant_step_callback(s,dt)` 消费 `s.feedback`，更新飞机、舰船和时间，随后云台积分到同一时刻。该回调应按足够高的内环频率推进；事件间仅有端点姿态时，云台命令采用线性插值。外部回调也可在自己的积分子步调用 `innerloop_update_gimbal(s)`，末端同刻更新不会重复推进。

首次必须读取当前传感器真值、写入导航反馈才能推进；相机/雷达之间允许保持最近一次控制反馈，控制到期时写入新的同刻反馈。缺少真实动力学回调仍报 `landing:MissingDynamics`，`allow_state_hold=true` 只验证接口。`read_gimbal_telemetry` 是独立测量通道；转换模型的真实轴角仅用于渲染真值，导航端口缺少编码器时绝不回填真值。

## 验证

```
.venv/bin/python -m pytest -q
.venv/bin/python -m simulation.offline.run_closed_loop
```

MATLAB 根目录执行：

```matlab
addpath('simulation/matlab'); setup_paths(pwd);
test_innerloop_interfaces;
test_gimbal_interfaces;
```

单元测试包含 0.6 rad 三轴摇摆下的稳定误差、角度/速率/加速度限幅、固定光心、杠杆臂补偿、位置协方差数值雅可比、转动云台下的雷达融合、过期编码器、训练样本及实际 SLX 接口。云台测试中的最大姿态残差约 0.02573 rad（1.47°，含零初速度启动），不代表实物性能。完整飞机/舰船控制器及多轮接触动力学不由这组接口测试验证。
