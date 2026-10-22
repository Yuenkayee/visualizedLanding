# 数据合同 v1

CameraFrame：timestamp 为非负有限仿真秒，image 为 H×W×3 uint8 RGB。CameraCalibration 含 K、width、height、OpenCV distortion 和 T_camera_lidar。LidarFrame 的 points 为 N×3 浮点有限数组，坐标为 LiDAR 系；允许空云，不允许 NaN/Inf。

PoseMeasurement：timestamp、T_deck_camera、6×6 covariance、非负 reprojection_error 和 source。协方差顺序是甲板系位置误差与相机局部右乘旋转误差，必须对称半正定。NavigationEstimate 另含 deck velocity、healthy 和 diagnostics；to_dict 输出 JSON 安全格式。

渲染标注 JSON：schema_version=1、timestamp、sequence_id、frame_id、image/mask（相对数据根目录）、keypoints（像素 u,v）、visibility、points_deck、T_deck_camera、K、distortion、width、height。visibility 同时检查视锥及 Blender 甲板遮挡射线；不可见关键点不参与训练损失。mask PNG 是带真实几何遮挡的 H 二值标注；边缘抗锯齿按 >127 二值化，不用 RGB 阈值冒充真值。

对应 LiDAR 文件 `{frame}_lidar.npz` 存 timestamp 和 points。训练 split JSON 是 annotation 绝对路径数组；数据根目录改变后应重新生成 split。单帧不跨序列划分，不以相邻帧随机划分测试集。

闭环 JSONL 每行 timestamp、truth、estimate。truth 含 T_world_deck、T_deck_camera、velocity_deck；estimate 含 T_deck_camera、velocity、covariance、healthy、diagnostics。truth 只能用于传感器生成与离线评估。回放没有真实速度标注时填零，不应将回放日志的速度指标解释为真实动力学结果。

所有传感器按测量时刻排序。相同时间戳允许不同传感器，旧时刻包不可应用到已传播的新状态。Matlab Engine 边界使用 JSON，避免 Python/MATLAB 列向量和矩阵布局歧义。

多相机：CameraRig 保存 reference_camera_id、各路 K/尺寸/distortion/T_body_camera，以及参考相机到 LiDAR 的 T_reference_lidar。同步组输入为 camera_id → CameraFrame 映射，至少一台已标定相机、曝光时间相同，不允许重复组。NavigationEstimate 的 T_deck_camera/velocity/covariance 始终属于固定参考 C0，selected_camera_id 仅表示测量来源。协方差转换含杆臂耦合。

多相机标注增加 camera_id 和 bundle_id（sequence/frame），每航次四路共享 split。bundles/sequence.json 列出 timestamp、stage/weather、理想通信的 ship_euler_rad、各路 annotation 路径及共享 LiDAR 文件路径；calibration/sequence.json 是实际安装误差后的标定。visibility head 的监督沿用 keypoint_train_visibility（几何可见且光度启发式可观测）；旧双输出 checkpoint 仍支持，可见性视为未提供。

多相机回放及闭环日志增加 feedback，含 gear_relative_estimate_ned（轮底中心减 H 中心，北/东/地，m）、relative_distance_estimate_m、relative_covariance_ned、feedback_valid。舰船通信上下文单独返回 timestamp、R_ned_deck，不向导航提供直升机真值。

innerLoop 导航日志的 feedback.source=innerLoop.slx 表示实际模型转换输出，已经与 Python 同条件计算核对。events 列出该状态时刻到期的传感器/控制事件；audit JSONL 还含 operations，记录完整读状态/传感器生成/导航/反馈/推进顺序。控制 JSONL 按 control_hz 采样，audit 则包括传感器单独到期的时刻；同一时刻只读一次并写一次反馈。推进后上一时刻的反馈不再允许作为当前反馈读取。
