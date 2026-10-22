# 稳定云台与闭环接口

相机光心仍固定在原来的机体安装位置，只有朝向随云台转动。内层快速稳定相对甲板姿态；外层根据导航估计缓慢调整参考，将 H 保持在视场中央区域。平移、透视尺度、机身遮挡、触及云台限位时的出画仍然存在。

## 动力学与输入假设

`navigation/common/gimbal.py` 与 `external/innerloop/simu/gimbal_step.m` 使用相同的三轴 XYZ 模型：

```
R_body_camera(t) = Rz(qz) Ry(qy) Rx(qx) R_body_camera_neutral
t_body_camera(t) = t_body_camera_neutral
qdd = wn² (q_command - q) + 2 ζ wn (q_command_dot - qdot)
```

这是 `J qdd + B qdot = torque` 的闭环二阶伺服近似；忽略轴间惯性耦合、摩擦、云台尺寸及轴心间距。加速度限幅代表等效力矩限制，并另有轴速率与转角限幅；到达机械限位时取消向外速度。`rate_feedforward=false` 可关闭速度前馈。默认固有频率 10 Hz、阻尼比 0.9、轴速率 3 rad/s、轴加速度 40 rad/s²、转角范围约 ±80°/±80°/±120°，都是待辨识的设计参数。积分自动细分到不大于 2 ms 和 `0.1/wn`，不依赖图像采样率。

相对甲板初始参考旋转可由 `R_deck_camera_reference` 显式给出；缺省采用水平飞机原安装角对应的视角。稳定内层使用机载姿态与舰船姿态遥测，仿真暂按无延迟、无噪声提供这两路控制输入。导航滤波器不接收飞机真值姿态。未来若要模拟 AHRS 或舰姿通信误差，应在云台控制输入处加入误差，不能把理想相机姿态作为视觉观测。

`navigation/common/gimbal_tracking.py` 实现慢速外层，输入为有效且编码器同步的导航估计，目标为已知 H 四角的角度中心。默认时间常数 0.6 s、参考速率上限 0.35 rad/s、加速度上限 0.7 rad/s²，40° 居中死区在当前广角下约为中央六成画幅。死区避免远距离无必要转动而缩小 H 的成像；近处采用四角范围避免只对准中心而裁掉近端。失效时平滑减速保持，不用真实位置兜底。`tracking.enabled=false` 恢复固定参考，`enabled=false` 则关闭整个执行机构。

Python 宿主统一计算慢参考，`GimbalReference` 独立携带时间、`R_deck_camera` 与 `angular_velocity_deck`；MATLAB 在反馈阶段锁存，随后通过原 `gimbal_step` 执行。命令不替代编码器，也不瞬间旋转相机。参考角速度变化作为 ESKF 下一步预测的已知输入，保留滤波器估计的残余角速度，不刷新观测时间、不重置相机姿态。这样参考减速时预测不会继续按旧角速度旋转。纯 MATLAB 伺服入口未提供外层命令时仍采用固定参考。

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

三阶段生成器保留原来的 0.6/0.05/0.3 rad 机体姿态包络，使用上述有限带宽云台计算每个时刻的真实相机姿态。外层以带 0.15 m 标准差噪声的合成位置代理驱动，明确标注 `synthetic_noisy_pose_proxy`；当前观测只影响后续状态。这是训练数据合成假设，不是实际导航性能。机身、轮子与支柱随飞机转动；RGB、mask、关键点、遮挡射线和雷达各自使用同刻变换。曝光模糊和滚动读出基于云台作用后的实际相机运动。

航次顶层 `T_body_camera` 是中立安装标定；每行 `T_body_camera` 才是实际动态外参。顶层 `R_deck_camera_reference` 为初始参考，每行另外记录实际使用的参考、下一步命令及跟踪状态。标注 schema_version=2、计划 schema_version=3，兼容旧编码器、轴状态及雷达字段。回放使用保存的编码器，带命令的新数据额外提供已知预测输入，均不由真值反推测量。旧数据视角保持不变；如需扩充新视角，须重新渲染图像和全部标签，不能只改旧图的外参。现有网络/权重格式兼容。

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
