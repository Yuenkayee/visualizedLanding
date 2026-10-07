# 验证记录（2026-10-04）

本机环境为 macOS arm64、Python 3.11.16、Blender bpy 4.2.0 和 MATLAB R2024b。`external/` Git diff 为空。

- pytest：19 项通过，覆盖坐标变换、合同与时间、调度/缓冲、PnP 模糊关联、误差状态滤波、离群观测拒绝、激光平面、真实估计反馈路径和数据/训练梯度。
- Python 语法编译、Ruff F/E9 静态检查、Git whitespace 检查及 shell 语法检查通过。
- CPU 数据生成：3 个独立序列，每序列 3 帧，随后跑通 1 epoch 训练、测试评估及 ONNX 导出。此项仅为执行验证；1 epoch 模型的 mask IoU 为 0，不能作为已训练可用模型交付。
- ONNX 与 PyTorch 随机输入比较：mask logits 最大误差 2.66×10⁻⁶，关键点最大误差 1.79×10⁻⁷。
- Blender：实际渲染 RGB、独立可见 H mask、关键点 JSON 和 LiDAR；检查四个可见关键点均对齐 mask 边界。生成了整舰预览和 .blend 场景。
- MATLAB：0.2 秒传感器闭环实际调用外部 ship_motion_model，产生 5 个控制记录，反馈接口正常；飞机侧是本项目 surrogate，未调用 SLX 内环。
- mock：5 秒简化场景位置 RMSE 0.269 m、101 样本；完整 20 秒位置 RMSE 2.497 m、401 样本，健康比例 0.372，末端高度 3.90 m，未到达 0.8 m 目标。近甲板 H 出视场后停止盲降，健康丢失后的常速度预测产生水平漂移。
- 回放、四种消融和结果绘图实际运行完成。

Blender 4.2 的 macOS cp311 wheel 内部 WHEEL 标记错误地写成 cp39；pip/uv 的通用平台检查因此误报。安装脚本单独检查全部已声明依赖的版本约束，并通过原生模块导入、运行时版本及本记录中的实际渲染核实兼容性，不修改第三方包元数据。

没有 Blender/真实数据域的模型精度评估，没有真实内环控制器或触舰验证。上述结果不能推广为真实飞行性能。

## innerLoop 接口模型（2026-10-05）

新增 `external/innerloop/simu/innerLoop.slx` 在 MATLAB/Simulink R2024b 中生成并编译。`test_innerloop_interfaces` 实际执行 SLX，验证节速转换、NED/ENU/甲板/相机变换、轮底与 H 偏移、协方差传播、旋转坐标系速度有限差分、读/反馈/推进顺序及缺失动力学显式错误。人工几何夹具不是 UH-60 实际尺寸。原有 Python 19 项测试通过。

现有 Python `MatlabExternalModel` 到新 SLX 的端到端往返通过：读取 SensorTruth、写入 NavigationEstimate、读取轮底 NED 反馈、推进显式静态测试时钟并读取下一状态。Simulink 缓存输出至临时目录。此验证不包含飞机/舰船动力学、UH-60 真实起落架尺寸或控制器性能。

### 甲板 NED 姿态内部计算

按舰船状态通信无延迟的假设，convert 内部使用舰船横摇、纵摇、艏向计算 R_NED_deck，删除外部 nav_R_ned_deck 端口及配置/适配要求。对现有 SLX 原位更新，未用重建函数覆盖已有模型布局。实际模型编译和 MATLAB 接口测试通过，新增倾斜舰船下位置、速度向量、协方差旋转及有偏导航位置保持测试；SLX 根输入确认由 12 个降为 11 个。Python 19 项测试通过。

### MATLAB System 几何参数

转换模块改为 convert < matlab.System，三个 Nontunable 参数为 T_body_camera、H_ship_body、gear_body，删除其根信号输入，剩余 8 个信号输入。Model Workspace 参数初始为 NaN，setup 阶段拒绝未填写或非法几何。适配层通过 SimulationInput 设置参数。实际 SLX 测试覆盖安装参数更改后参考点变化、未填写参数拒绝、原有坐标/单位/反馈顺序；独立临时目录中重建脚本成功生成同样的 MATLAB System 结构。Python 19 项测试通过。

## 三阶段着舰训练数据初版（2026-10-06，后续调整见下文）

新增固定安装单相机、轮底中心三阶段轨迹、天气分层航次划分与 Blender/CPU 生成入口。轨迹使用不随甲板摇摆倾斜的水平航向系，悬停目标在 NED 中为 [0,0,-5] m；真实甲板系单独用于成像及 PnP。默认 60 航次已生成全轨迹和光学计划：40/10/10 航次分配到 train/val/test，正式渲染目标为 10,800 图像，尚未执行全量 Blender 渲染。

默认 60 航次的全轨迹视锥检查：三个阶段 H 中心入画率均为 100%；approach/window_hold 四角入画率为 100%；touchdown 最差航次四角入画率约 97.0%。中距最小 H 投影尺度约 20.2 px（1280×960），仅表示几何覆盖，未包含真实 UH-60 遮挡、实际镜头畸变或网络误差。设计几何和 130°/126°视场均需替换为实际标定值。

五种天气 × 三阶段共 15 张 Blender 样例完成并人工检查，所有样例四角几何可见；生成 `output/figure/landing_three_stage_preview.png`。日间雾、海面反射/眩光和夜间甲板照明可见。正式样本渲染建议提高默认 24 samples；本次预览仅 4 samples，仍有路径追踪噪声。

小型 CPU 数据集：15 完整航次、135 图像，分层 split 各 5 航次；在更新的关键点光度监督规则下跑通一个 epoch 和分组评估。仅执行验证，不能据此报告网络在 Blender/真实场景的定位性能。新增阶段/相机参考点、NED 垂向悬停与倾斜甲板的区别、速度过渡、窗口事件、视锥、完整航次划分、训练合同与滚动读出标签一致性测试；总共 24 项 pytest 通过。静态 F/E9 检查、Python 编译和 shell 语法检查通过。

### 机腹表面安装与自遮挡更新（2026-10-06）

默认光心更新为机体 [2,0,1.11] m，光轴向下 96°（相对垂直向下向机尾偏 6°），矫正后水平/垂直视场更新为 140°/145°。简化机腹安装点表面约 z=1.027 m，光心突出约 8.3 cm；该外形是可替换设计代理，不是 UH-60 CAD/实测尺寸。无倾斜接触几何下相机距甲板 1.79 m，60 航次中最小甲板法向高度约 1.59 m。

加入统一的机身、主轮、尾轮、支柱椭球代理，同时用于安装点检查、射线遮挡报告、Blender RGB/mask 和 CPU 流程测试遮挡投影。旧光心 [0,0,0.1] m 会被拒绝；相机偏置依然通过固定 T_body_camera 生效。calibration/design.json 导出标称 K、T_body_camera，各航次保存加入固定安装误差后的实际矩阵。

默认 60 航次全轨迹中 H 中心同时入画且无遮挡的比例均为 100%；approach/window_hold 四角可见比例为 100%；touchdown 最差航次四角可见比例约 99.31%。中距 H 最小尺度约 22.24 px。这些结果只验证代理几何和采样扰动，不代表镜头实装可行性或网络检测性能。

含机身和起落架的五类天气 × 三阶段共 15 张 Blender 样例重新渲染，8 samples，全部样例四个外角几何可见；可见部分 H 被起落架遮挡及飞机阴影。更新 output/figure/landing_three_stage_preview.png、landing_optics_report.json，并生成 landing_camera_mount.png 侧视示意图。全量 10,800 张仍未渲染。

26 项 pytest 通过，新增覆盖机身内安装拒绝、主轮/机身射线遮挡、目标前/后遮挡物区分和 CPU 遮挡投影；静态 F/E9 检查通过。

### 按仿真结果扩大姿态晃动（2026-10-06）

以用户所述幅值为单边 Euler 偏移峰值：第一阶段 phi/theta/psi 初始 0.6 rad，前 15% 保持、15%–80% 连续衰减至 0.05 rad，第二阶段暂用 0.05 rad；第三阶段前 12% 从 0.05 升至 0.3 rad，后续保持。航次/轴独立幅值系数 0.9–1.0，周期 2.7–3.3 s，全航次相位连续。稠密轨迹改为 20 Hz，保留小幅高频机体振动。舰船晃动、相机位置/角度/FOV 和天气参数保持原设计。

主姿态波动替代原 1.5° Gaussian 残差，取消主姿态的距离/高度衰减及从 CG 速度反复修正 yaw。后者会擦除用户指定的 psi 波动；直线轮底轨迹与大偏航同时给定的运动学代理无法保证零侧滑，逐帧保存实际计算值。轨迹/图像标注新增姿态包络、Euler 偏移、周期、SO(3) 有限差分体轴角速度；plan.json（schema_version=2）按航次保存姿态统计，新增 planning_report.json 汇总视场和最长连续不可见时间。

默认 60 航次的 H 中心可见率：approach 航次平均 92.67%、最差 86.69%，window_hold/touchdown 均 100%；四角同时可见率：approach 平均 90.09%、最差 84.11%，window_hold 100%，touchdown 平均 83.99%、最差 74.04%。中心最长连续不可见约 1.25 s，末端最长四角不齐约 2.39 s。最小中距投影尺度约 10.51 px，最小甲板法向相机高度约 0.923 m。以上均为采样代理几何结果，旧小扰动可见率不适用于本版。

15 张 Blender 样例按每阶段最大姿态模长重新选择、实际渲染（640×480，8 samples），检查了目标出画和起落架遮挡。12 张几何上四角齐全，3 张不满足四角/像素面积条件；保留全部样本，不按 PnP 可用性筛选。更新 landing_three_stage_preview.png、landing_optics_report.json，新增 landing_attitude_profile.png 和可复现绘图入口。正式 10,800 张图像与完整训练尚未执行。

29 项 pytest 通过：新增包络阶段峰值与边界一阶连续、三秒周期、psi 不被覆盖、采样频率拒绝，以及强扰动遮挡样本在 HDataset/关键点可见性屏蔽下的有限损失和梯度。静态 F/E9、Python 编译、shell 语法和 Git whitespace 检查通过。

## 四相机实现（2026-10-07）

新增 C0 机腹、C1 前下方、C2/C3 左右侧下视固定安装，以及共享分割/关键点/可见性网络。逐路 PnP、协方差转换和创新筛选后带 20%分数滞回选择一份视觉观测，使用固定 C0 ESKF 和共享 LiDAR。含偏心着舰的四路同步训练生成、完整航次划分、同步回放及已有闭环入口均已接入。平面 PnP 新增逐 IPPE 解精化和迭代候选，处理近正视、旋转角接近 pi 时 OpenCV 的数值误差。

38 项 pytest 全部通过。新增验证包括非平凡安装姿态/杆臂下的转换 Jacobian 有限差分、NED 轮底反馈、不同源切换保持参考、逐路创新拒绝和替代视角、分数滞回、时间/尺寸检查、视觉过期、切换后共享 LiDAR 外参、同步世界状态/天气种子、完整航次 split、不对中/对中甲板接触、全不可见样本梯度和旧 checkpoint 兼容。PnP 测试使用人工精确投影关键点；真实回放使用模型图像推理，不使用这些人工关键点或真值初始化。

完整默认 60 航次、20 Hz 轨迹规划通过，正式目标 43,200 张图、40/10/10 航次 split。进近/窗口阶段至少一路中心及四角可见的比例为 100%；末端至少一路中心可见比例为 100%，至少一路四角齐全的航次均值约 89.976%、最差航次约 83.026%，最长四角不齐区间约 2.295 s。统计包含代理机体遮挡和偏心接触，不包含网络误差或真实 UH-60 CAD。相机数量增加并未消除末端视觉失效。

实际 Blender 渲染五种天气 × 三阶段 × 四路，共 60 张 RGB/mask/标注（640×480，4 samples）；按同一物理时刻对照检查，预览中的 15 个图像组有 13 个至少一路满足四角/像素面积几何条件。检查了机体/起落架遮挡、眩光、夜间甲板照明与雾。`output/figure/multi_camera_training_preview.png` 为四列同步对照，`multi_camera_optics_report.json` 保存完整规划摘要。4 samples 图像含渲染噪声，正式配置使用 24 samples。

小型 CPU 数据实际生成 15 航次、540 图、135 图像组，包含共享 LiDAR 和实际安装标定。共享可见性网络完成 64×64 的 1 epoch 训练，跑通相机/阶段/天气分组评估及三输出 ONNX 导出。PyTorch/ONNX 在 batch=2 随机输入上的最大绝对误差分别约 3.50×10⁻⁶（mask）、5.96×10⁻⁷（关键点）、3.58×10⁻⁷（可见性）。旧双输出权重兼容测试通过。

小型测试集 IoU 约 0.465，但可见性 head 将全部关键点预测为可见，存在 146 个不可见关键点误报、5 个不可用图像组误报。一个完整小型航次的 9 组图像与 LiDAR 回放执行完成，8 次视觉更新、4 次切换，健康样本 NED 三轴 RMSE 约 [10.75,3.01,13.52] m；默认粗略先验和仅 1 epoch 网络均不适合精确导航，此结果是执行检查，不能宣称模型已经训练可用。相邻切换样本的位移包含实际运动，不能当作切换瞬间跳变。完整 43,200 张 Blender 图像、正式训练和真实飞行导航精度尚未验证。

新增 `read_navigation_context.m` 在本机 MATLAB 实际执行，确认给定舰船三轴姿态与 Python 的 R_NED_deck 一致。Python 多相机 mock 短闭环通过；SLX 使用固定 C0 参数和既有位姿反馈接口。此次未补充或验证真实直升机/舰船动力学、控制器和窗口预测系统。

## 导航与实际 innerLoop.slx 对齐（2026-10-07）

本机实际执行 `external/innerloop/simu/innerLoop.slx`，确认 8 个根输入、12 个根输出及三个 convert 工作区参数与固定 C0 NavigationEstimate 合同相符。模型启动新增实际文件/端口/参数表达式检查，输出按已校验的 Outport 编号读取；Simulink 输出 Dataset 信号标签可能为空。验证以 SHA-256 核实 SLX 执行未改写模型，不涉及 innerLoop_landing.slx。

修正非整比传感器/控制采样下可能在没有新反馈时推进的问题：调度器合并同刻事件，每个状态时刻完成一次读状态、生成到期观测、导航解算和反馈事务，再推进并读取下一状态。日志现在读取实际 SLX NED 距离/模长/协方差/有效标志，逐次与 Python 独立计算比对，避免仅记录 Python 推算而未检查内环接收值。推进后失效旧反馈/读标志，错误参考相机反馈被拒绝。

`simulation.offline.verify_innerloop_navigation` 实际运行通过，配置相机 10 Hz、LiDAR 7 Hz、控制日志 20 Hz、0.3 s，采用显式静态接口模式。完成 9 次状态读取、9 次反馈写入、8 次推进和 9 次 SLX 反馈核对；4 组四相机（16 张 CPU 图像）完成 4 次图像检测/PnP/ESKF 更新，3 帧 LiDAR 全部接受；7 个控制记录。SLX 与 Python 转换最大绝对差约 3.55×10⁻¹⁵。静态简化场景相机位置 RMSE 约 0.130 m、健康比例 1.0，不代表动态着舰精度。

MATLAB 合同测试新增错误 C0 参考拒绝、推进后陈旧反馈拒绝和测试状态回调：回调读取当前已转换反馈、替换下一时刻的原生状态，验证后续真值位姿变化正确。缺少动力学且没有显式状态保持依然报错。此测试回调不包含动力学或控制律。

Python 全部 41 项测试通过，新增共同安装几何/冲突拒绝、实际反馈错误和旧时间拒绝、严格因果 fixture 在非控制采样雷达时刻执行等检查。可运行配置为 `configs/innerloop_navigation.yaml`，可审阅验证记录为 `output/verification/innerloop_navigation.json`，完整说明见 `docs/innerloop_navigation_alignment.md`。

## RTX 5090 训练配置（2026-10-07）

新增 Ubuntu 22.04 x86_64 / Python 3.11 训练 profile，锁定 PyTorch 2.7.1+cu128 及 CUDA 12.8 传递依赖。配置采用 512×512、batch 16、40 epoch、BF16、8 worker / spawn、锁页内存、channels last 和 fused AdamW；关键点 softargmax 与损失使用 FP32。GPU 安装检查实际执行 BF16 前向/反向/优化器更新，不将版本匹配当作 GPU 已可用。

本机为 macOS arm64 / PyTorch 2.5.1，无 NVIDIA CUDA。49 项 pytest 通过，1 项 RTX 5090 硬件测试按条件跳过。新增实际验证包括 CPU BF16 有/无可见关键点的有限损失/梯度、512×512 FP16 logits 的 FP32 损失归约、两个 worker 的独立可重复光度增强、带 spawn/persistent workers/channels last 的 1 epoch BF16 训练、FP32 checkpoint 保存及分组评估、三输出 ONNX 导出并在 batch=2 下与 PyTorch 比对（rtol 1e-4 / atol 1e-5）。数据复制后按新根目录加载旧绝对/相对 split，且不改写划分文件；源数据仍存在和移走后均通过。

沙箱内 DataLoader 子进程因 OpenMP 共享内存权限失败，移到沙箱外完成上述回归。ONNX 导出保留 GroupNorm 的形状追踪提示，PyTorch/ONNX 数值比对通过。Ruff F/E9、格式、Python 编译、shell 语法和 Git whitespace 检查通过；原默认依赖锁的版本检查通过。

尚未在 Ubuntu / RTX 5090 上执行安装或完整训练，未报告 CUDA 吞吐、512×512 训练显存峰值或模型精度。服务器须执行 `shell/check_dependencies.sh --rtx5090`（包含实际 GPU 预检查）再开始正式训练；batch 16 是待实测的起始值。具体命令见 [RTX 5090 训练说明](rtx5090_training.md)。

### 安装下载恢复与 tmux 启动

安装入口优先复用 uv，其次通过 pip 从 PyPI 安装项目本地的 uv 0.9.5，再回退到带 TLS 中断重试的 GitHub 下载。Python 环境创建失败单独报告；支持 LANDING_PYTHON 指定已有 3.11。新增 tmux 后台入口按安装/检查成功后才训练的顺序执行，保存独立日志及退出码，并传递当前代理/包源/GPU 环境。

7 项脚本集成测试通过：隔离 PATH，以受控替身模拟包源失败、curl EOF、Python 下载失败、已有 tmux 环境以及训练程序，验证回退、安装失败停止、参数中的空格、GPU 检查先于训练、重复会话拒绝、日志唯一性和完成状态保留。这些测试未连接真实 tmux 服务或 NVIDIA GPU。另从 PyPI 实际下载 macOS arm64 的 uv 0.9.5 wheel，使用 pip --target 安装到临时目录，确认 bin/uv --version 可运行；原项目依赖版本检查、shell 语法、Ruff F/E9 和 Git whitespace 检查通过。未修改训练算法，本次未重复完整训练测试。
