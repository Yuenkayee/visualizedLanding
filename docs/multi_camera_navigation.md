# 四相机着舰导航与训练

本版本实现四个机体固定视角、共享 H 分割/四角关键点/可见性网络、逐相机 PnP 与门限检查、带滞回的最佳相机选择、单个固定参考 ESKF，以及 LiDAR 平面融合。每个同步图像组最多进行一次视觉更新；四台相机共用一份网络权重。跨相机部分关键点联合 PnP 和多观测同时融合尚未实现，不能将四路图像中的零散角点直接当成完整单相机 PnP 输入。

## 安装与标定

体轴为前/右/下，位置相对于直升机质心，单位米。默认配置继承 `landing_dataset.yaml`，由 `multi_camera_dataset.yaml` 覆盖以下设计值：

| 相机 | 安装位置 | 名义光轴 | 用途 |
| --- | --- | --- | --- |
| C0 | [2, 0, 1.11] | 向下 96°，相对垂直向后 6° | 机腹主视角、固定导航参考 |
| C1 | [4, 0, 0.85] | 前下方，向下 55° | 中距进近、俯仰扰动补充 |
| C2 | [0, -0.9, 0.9] | 向下，绕体轴 X 转 -12° | 左侧下视补充 |
| C3 | [0, 0.9, 0.9] | 向下，绕体轴 X 转 +12° | 右侧下视补充 |

四台均使用 1280×960、矫正后水平/垂直 140°/145°针孔模型。安装点必须通过当前椭球机体/轮胎/支柱代理的外部安装检查；这些位置和起落架 [0,0,2.9] m 是设计参数，必须用实际机体 CAD/安装测量替换。镜头畸变需要先矫正或提供各相机真实 OpenCV distortion。0.5°固定安装误差独立采样，每航次导出实际外参；回放读取实际标定，闭环示例使用标称设计。相机固定于机体，不能根据目标位置逐帧转动光轴。

`calibration/rig_design.json` 保存标称 K、尺寸、畸变、`T_body_camera` 和 `T_reference_lidar`；`calibration/{sequence}.json` 保存该航次实际值。LiDAR 默认为与 C0 共址、同向的占位外参，需用真实标定替换。回放拒绝与数据标定不一致的 LiDAR 外参。

## 固定参考与反馈

`T_ab` 将 b 系坐标转换至 a 系。相机 i 的 PnP 输出先转换为 C0：

```text
T_deck_C0 = T_deck_Ci · inv(T_body_Ci) · T_body_C0
```

同时传播 6×6 协方差，包括相机间杆臂引入的位置/姿态耦合；误差约定为甲板系位置误差、相机局部右乘旋转误差。相机切换只改变观测来源，滤波状态、速度、LiDAR 外参和输出参考始终属于 C0。

理想同步通信提供舰船横摇、纵摇、艏向后，甲板到 NED 的旋转为 `R_NED_ship · diag(1,-1,-1)`。轮底平面中心减 H 中心的 NED 向量为：

```text
r_NED = R_NED_deck · [t_deck_C0 + R_deck_C0 · R_body_C0ᵀ · (gear_body - p_body_C0)]
```

`gear_feedback_ned` 输出北/东/地三轴有符号距离、欧氏距离、NED 协方差和 `feedback_valid`。上述定义在甲板上方通常有负的 NED 地向分量。甲板法向高度为零与 NED 地向差为零并不等价，倾斜甲板且着舰位置偏心时尤其如此。

`MultiCameraPipeline.process_camera_bundle` 接收 `{camera_id: CameraFrame}`。可缺少部分相机，但输入集合必须非空，曝光时间差须不超过 1e-8 s，图像尺寸必须匹配标定，不接受重复/旧图像组。异步硬件必须先完成时间对齐，此版本不实现异步回滚。

各相机依次检查网络可见性、四角几何、平面对应、重投影误差及 ESKF 创新门限。质量分数综合网络置信度、目标像素尺度、清晰度、重投影误差和转换后的协方差；默认新视角分数至少高出当前可用视角 20% 才切换，当前视角不可用时立即选其他可用视角。图像清晰度采用全图 Laplacian 方差，是启发式指标。日志包含每路拒绝原因、分数、选中相机和切换标记。

所有视角失效时依靠预测及可用 LiDAR；LiDAR 平面只约束高度/倾斜，不能恢复水平位置和绝对 yaw，也不会延长完整位姿的健康时限。H 对称性仍需要合理航向先验。

## 训练集

共享一条物理航次后再生成四路观测：同一时刻的机体/甲板位姿、海浪时间、舰尾流运动残差和天气参数一致，各相机固定安装误差、传感器噪声独立。保留三阶段速度过渡、第一阶段 0.6→0.05 rad、第三阶段 0.3 rad 和约 3 s 周期的姿态波动，以及舰船横纵摇和升沉。

末端偏心接触按实际甲板平面增加平滑 XY 偏移，默认 25%航次对中、其他航次在 ±0.75 m 内均匀抽样。末端轮底中心的甲板法向距离为零；机身质心和体轴速度随偏移重新计算。此运动学轨迹保留计算得到的侧滑量，不能宣称是经过动力学控制验证的零侧滑飞行。

五类天气为晴朗、雾、太阳眩光、夜间、夜雾。Blender 图像包含海面波浪、船尾水面尾迹、机体/起落架、真实场景光照和遮挡 mask。CPU 渲染仅用于快速执行检查。空气尾流是相关状态扰动代理，水面尾迹是视觉模型，两者均不是 CFD。

默认 60 航次，每航次每阶段 60 个时刻、每时刻 4 张图，共 10,800 图像组、43,200 张图像；按天气分层、完整航次划分为 train/val/test = 40/10/10 航次。四路与所有相邻时刻跟随该航次进入同一集合，避免视角/时序泄漏。保留全遮挡、出画、过暗和眩光样本。不可用关键点不参与坐标/热图损失，可见性 head 则对所有四角学习可用/不可用；mask 仍保持几何真值。

数据根目录结构：

```text
calibration/{sequence}.json
trajectories/C0/{sequence}.json     # C1/C2/C3 同样结构
rendered/{sequence}/C0/{frame}.png
annotations/{sequence}/C0/{frame}.json
annotations/{sequence}/C0/{frame}_mask.png
lidar/{sequence}/{frame}.npz       # 同步图像组共享一份 LiDAR
bundles/{sequence}.json           # 时间、阶段、天气、舰船姿态、四路标注和 LiDAR 路径
splits/train.json                 # val.json/test.json 同样结构
multi_plan.json                   # 四路与联合几何覆盖统计
planning_report.json             # C0 单路统计，已包括偏心轨迹
multi_dataset_report.json         # 渲染结果及图像组索引
```

当前 LiDAR 使用有噪声/丢点的有限甲板解析射线，不包含整船杂波、机体自遮挡或天气引起的激光退化。几何覆盖报告包括“至少一路四角齐全”和“多路四角并集齐全”；后者只是布设诊断，不表示已实现联合解算。

## 运行

所有命令在仓库根目录执行，使用现有锁定依赖即可。

```bash
# 全量轨迹/覆盖规划，无需渲染
bash shell/generate_multi_camera_dataset.sh --plan-only

# 查看四视角预览（五类天气 × 三阶段，共 60 图）
bash shell/generate_multi_camera_dataset.sh --preview --width 640 --samples 8 --output data/multi_camera_preview

# 正式 Blender 数据（较耗时），随后训练共享网络
bash shell/generate_multi_camera_dataset.sh
.venv/bin/python -m visual_training.train --config configs/multi_camera_training.yaml
.venv/bin/python -m visual_training.evaluate --root data/multi_camera --checkpoint visual_training/checkpoints/multi_camera/best.pt --image-size 512 --output outputs/metrics/multi_camera_vision.json
.venv/bin/python -m visual_training.export_model --checkpoint visual_training/checkpoints/multi_camera/best.pt --output visual_training/checkpoints/multi_camera/h_detector.onnx --image-size 512

# CPU 小型数据/训练执行检查
bash shell/generate_multi_camera_dataset.sh --renderer cpu --sorties-per-weather 3 --frames-per-stage 3 --width 320 --output data/multi_camera_smoke
.venv/bin/python -m visual_training.train --config configs/multi_camera_training.yaml --data-root data/multi_camera_smoke --output visual_training/checkpoints/multi_camera_smoke --image-size 64 --epochs 1
```

评估同时提供相机、阶段、天气分组，关键点可见性 accuracy/precision/recall，以及图像组“至少一路预测四角可用”的误报/漏报数量。图像组可见性指标不等价于定位正确率；必须进一步评估 NED 距离误差、健康比例、失效时长及切换时的误差变化。中距 H 在缩放图像中很小，应根据实测目标像素尺度选择训练分辨率或后续加入 ROI 放大。

数据回放：先在 `multi_camera_navigation.yaml` 填入 C0 的独立启动先验及训练权重路径。默认 [0,0,30] 是运行模板，不能直接当成所有航次后方 50 m / 高 50 m 的实际 C0 初值。禁止从标注位姿自动初始化。

```bash
.venv/bin/python -m simulation.offline.run_multi_camera_replay --sequence landing_20261006_clear_000
.venv/bin/python -m simulation.offline.run_closed_loop --config configs/multi_camera_offline.yaml
```

回放读取实际四相机标定，以真值仅作误差统计；输出 JSONL 带 NED 轮底反馈。`max_switch_step_m` 包含相邻时刻实际运动，不是纯相机切换跳变。闭环入口保持“当前状态 → 同步传感器 → 导航更新 → 写反馈 → 下一次推进”顺序；CPU/点质量示例用于接口检查，不代表完整三阶段动力学着舰。

接入 `innerLoop.slx` 时，`convert.T_body_camera` 必须填写固定 **C0** 的 `T_body_camera`，另填实际 `H_ship_body`、`gear_body`。Python 初始化检查 C0 外参一致性；C1/C2/C3 参数留在 Python rig 中，切换时不修改 SLX 参数。`read_navigation_context.m` 单独提供无延迟舰船姿态。以后补充动力学控制和运动预测窗口逻辑时，继续使用现有 callback 因果协议。

实际 SLX 的无动力学导航事务已经通过 `simulation.offline.verify_innerloop_navigation` 验证；`configs/innerloop_navigation.yaml` 可直接运行静态接口测试。模型端口、异频调度和实际 NED 反馈核对见 [内环导航对齐](innerloop_navigation_alignment.md)。

## 本次验证与覆盖限制

60 航次 20 Hz 稠密轨迹的几何可见性结果：

| 阶段 | 至少一路 H 中心可见（航次均值） | 至少一路四角齐全（航次均值 / 最差航次） |
| --- | --- | --- |
| 进近 | 100% | 100% / 100% |
| 等待窗口 | 100% | 100% / 100% |
| 末端下降 | 100% | 89.98% / 83.03% |

末端四角同时不可用的最长区间约 2.29 s，超过默认 1 s 健康期限，健康反馈可能失效。扩大相机数量改善了覆盖，但当前四角 PnP 不能保证连续触舰导航。结果基于配置中的随机相位与幅值采样、代理机体和矫正后镜头，不是所有极限姿态组合、实际 UH-60 或恶劣天气网络性能保证。

本次实际生成 60 张 Blender 图像和 540 张 CPU 图像，跑通一轮共享可见性网络训练、分组评估、ONNX 三输出导出及同步视觉/LiDAR 回放。完整 43,200 张 Blender 训练集和正式 40 epoch 训练尚未执行。一轮测试模型存在可见性误报和较大 NED 定位误差，不能作为可用导航权重。详细验证见 `validation.md`；预览和完整覆盖摘要位于 `output/figure/multi_camera_training_preview.png`、`output/figure/multi_camera_optics_report.json`。
