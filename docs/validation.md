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
