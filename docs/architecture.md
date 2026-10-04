# 架构

`visual_modeling` 负责三维护卫舰、海天环境、天气与相机，以及同一帧的 RGB/可见 H mask/标注/LiDAR。`visual_training` 负责独立序列划分、数据读取、双头神经网络训练、测试与 ONNX 导出。`navigation` 负责输入合同、视觉关联、PnP 及模糊解、激光平面与误差状态滤波。

`simulation.bridge` 对 mock/MATLAB 提供统一 read_sensor_truth/write_navigation_estimate/advance 接口；`simulation.offline` 对多个独立频率进行时间调度、缓冲和 JSONL 记录；`simulation.evaluation` 把估计和真值比较并绘图；`simulation.experiments` 用相同随机种子运行批次及传感器消融。

数据流：plant truth → renderer/sensor noise → camera/LiDAR contracts → navigation estimate → controller feedback → plant.advance。真值不得直接用于观测或初始化，初始位姿配置属于明确的先验。相同时间点按 camera、LiDAR、control 顺序处理，状态只向前传播；延迟数据需要重新回放，当前不支持状态回滚。

误差状态 12 维：[位置、速度、局部旋转误差、角速度]。预测使用 Van Loan 离散化；观测经卡方 Mahalanobis 门限筛选，更新使用 Joseph 协方差形式和旋转误差重置。视觉提供 6DoF 位姿，LiDAR 提供法向和高度约束。健康度取决于最近完整位姿年龄及位置协方差；仅激光更新不会延长完整姿态健康期。

配置均为 YAML。CLI 入口用 `python -m ...`；Blender 支持独立应用和 `bpy` 运行时。Python 包根目录均有 `__init__.py`，没有对 `external/` 源文件的写入。
