# 离线仿真

新增接口模型为 `external/innerloop/simu/innerLoop.slx`，与已有 `external/inner_loop` 路径区分。它将原生 NED/knot 状态转换为当前 SensorTruth，并将导航相机位姿转换为 UH-60 轮底平面中心相对 H 中心的 NED 反馈。它不含动力学或控制器；安装偏移以 MATLAB System 模块的三个固定参数 T_body_camera、H_ship_body、gear_body 填入，必须由用户标定/指定；舰船姿态按无延迟通信状态使用，在转换函数内部计算甲板到 NED 的旋转。完整端口、callback 顺序和重建说明见该目录 README。`configs/innerloop_interface.yaml` 是待补参数模板，不能直接作为已完成的动力学配置运行。

调度器以每个传感器的整数 tick/rate 计算时刻，避免浮点 modulo 漂移，并将同一时刻事件合并为一次状态事务。读取当前状态、生成到期的传感器观测，按相机 → 激光更新，预测至当前时刻，写回一次导航反馈后才允许下一次推进。即使传感器事件不与控制采样重合，也完成导航反馈事务；控制日志仍按控制采样记录。

mock 模型：有界加速度点质量飞机 + 正弦海面甲板运动。控制误差来自 NavigationEstimate，healthy=false 时刹车；不存在将真实飞机位姿直接替代导航估计的捷径。CPU 图像和解析甲板 LiDAR 提供快速闭环输入，不能代表完整 Blender 光照与真实传感器。默认目标高度 0.8 m，但完整 H 在近距离会出视场，健康机制将停止盲降，因此默认运行未必触达目标；reached_target 为真值指标，不能靠日志完成等价推断。

MATLAB 模型：`setup_paths` 只 addpath 已有 external 源码；`initialize_external_model` 实例化 ship_motion_model System object；`read_sensor_truth` 计算世界/甲板相对位姿和包含角速度项的相对速度；反馈和推进使用 JSON 合同。

现有 SLX 未声明适配层可用的逐步接口，不能自动假定其信号名、采样周期和反馈单位。要接入真实内环，在 plant 配置声明 callback 函数名，这些函数放在用户自有 MATLAB path 或 `simulation/matlab`，保持 `external/` 不变：

- initialize_callback(s) → s：初始化用户模型状态；
- truth_callback(s) → truth：返回 data_contracts 指定的字段；
- feedback_callback(s, estimate) → s：写入导航反馈，保留 timestamp；
- advance_callback(s, dt) → s：推进模型，必须令 timestamp 增长 dt；
- finalize_callback(s)：释放模型资源。

默认 MATLAB 飞机控制器仍为点质量 surrogate，不使用 existing SLX/LPV 控制器。真实控制器必须经单位、坐标、因果次序和反馈路径验证后才能报告为已接通。

回放读取某一个完整序列，相机与 LiDAR 同步输入导航。批次试验对随机种子循环；消融包括融合、仅视觉、仅激光、纯预测。仅平面激光无法建立完整健康位姿，模型会悬停，这属于可观测性限制。

训练/导航/闭环产物写入 data、visual_training/checkpoints、outputs 对应目录。测试中的数据、日志在 pytest 临时目录生成。MATLAB 与 Blender 的本机图形/许可证/进程访问可能需要离开限制沙箱运行。

多相机闭环使用 configs/multi_camera_offline.yaml 的 camera_rig_config，参考相机固定 C0。四路传感器从同一当前状态渲染，每个相机事件仅调用一次 process_camera_bundle；LiDAR 外参也是 C0 到 LiDAR。写回内环的 NavigationEstimate 保持 C0 合同；初始化验证 innerLoop 的 T_body_camera 与 C0 一致。独立 read_navigation_context 提供无延迟舰船姿态，用于记录轮底 NED 反馈，真实内环的 convert 继续按自身舰船状态计算同一旋转。多相机数据回放入口为 simulation.offline.run_multi_camera_replay，读取逐航次实际标定、同步四路及共享 LiDAR。运行方法和先验设置见 multi_camera_navigation.md。

configs/innerloop_navigation.yaml 是不含动力学的实际 SLX 导航接口测试入口，.venv/bin/python -m simulation.offline.verify_innerloop_navigation 运行合同检查和完整传感器导航往返。宿主读取实际 SLX 的轮底反馈，逐次与 Python 转换结果比对，并输出所有状态时刻的因果审计日志。显式 allow_state_hold 仅推进时间；下一状态也可由测试/真实回调提供。模型路径和完整信号对应见 [内环导航对齐](innerloop_navigation_alignment.md)。
