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

## 三阶段着舰训练数据初版（2026-10-06，机腹安装更新见下文）

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
