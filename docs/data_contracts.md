# 数据合同 v1

CameraFrame：timestamp 为非负有限仿真秒，image 为 H×W×3 uint8 RGB。CameraCalibration 含 K、width、height、OpenCV distortion 和 T_camera_lidar。LidarFrame 的 points 为 N×3 浮点有限数组，坐标为 LiDAR 系；允许空云，不允许 NaN/Inf。

PoseMeasurement：timestamp、T_deck_camera、6×6 covariance、非负 reprojection_error 和 source。协方差顺序是甲板系位置误差与相机局部右乘旋转误差，必须对称半正定。NavigationEstimate 另含 deck velocity、healthy 和 diagnostics；to_dict 输出 JSON 安全格式。

渲染标注 JSON：schema_version=1、timestamp、sequence_id、frame_id、image/mask（相对数据根目录）、keypoints（像素 u,v）、visibility、points_deck、T_deck_camera、K、distortion、width、height。visibility 同时检查视锥及 Blender 甲板遮挡射线；不可见关键点不参与训练损失。mask PNG 是带真实几何遮挡的 H 二值标注；边缘抗锯齿按 >127 二值化，不用 RGB 阈值冒充真值。

对应 LiDAR 文件 `{frame}_lidar.npz` 存 timestamp 和 points。训练 split JSON 是 annotation 绝对路径数组；数据根目录改变后应重新生成 split。单帧不跨序列划分，不以相邻帧随机划分测试集。

闭环 JSONL 每行 timestamp、truth、estimate。truth 含 T_world_deck、T_deck_camera、velocity_deck；estimate 含 T_deck_camera、velocity、covariance、healthy、diagnostics。truth 只能用于传感器生成与离线评估。回放没有真实速度标注时填零，不应将回放日志的速度指标解释为真实动力学结果。

所有传感器按测量时刻排序。相同时间戳允许不同传感器，旧时刻包不可应用到已传播的新状态。Matlab Engine 边界使用 JSON，避免 Python/MATLAB 列向量和矩阵布局歧义。
