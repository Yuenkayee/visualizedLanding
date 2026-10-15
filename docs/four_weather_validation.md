# 四场景导航回归结果

本次使用原第19轮 best.pt，四场景验证集8航次/1440帧，着舰段480帧；首帧真值初始化，其后只融合图像、编码器和LiDAR。

|着舰指标|原导航|修改后|
|---|---:|---:|
|三维RMSE m|0.2094|0.0581|
|三维P95 m|0.3929|0.0565|
|最大误差 m|1.4749|0.5387|
|最大预测到更新校正 m|1.6909|0.9564|
|最长观测视觉间隔 s|1.3990|0.8498|
|视觉接受帧/480|372|455|

八个航次着舰RMSE均下降。35帧着舰更新通过完整H分割恢复外角点；加速度过程噪声修订使正常机动后的观测不再被过度拒绝。仍有个别约0.54m误差，不能声称跳变完全消失。

接近阶段小目标仍有限制：当前RMSE 0.945m、最大4.505m；其最大单次校正由3.733m增加到4.601m，故本次改进并非所有时刻的校正幅度都下降。夜雾002的末帧误差由0.051m增至0.071m，虽该航次整体着舰RMSE下降，末帧也并非逐一更优。

本次参数在旧验证集上检查，独立测试集未用于本轮调参。新网络尚未在4090完成正式训练；上述导航改善不能视为新训练效果或真实着舰验收。

详细实施原因、服务器操作与控制变量配置见仓库 docs/four_weather_training_replay.md。


运行命令：

```bash
PYTHON_BIN=.venv/bin/python bash shell/replay_landing_offline.sh \
  --data-root data/landing_four_weather \
  --checkpoint visual_training/checkpoints/full_20261014_184359_0f5f3d/best.pt \
  --mode full --split val --compare-baseline --device mps \
  --output outputs/replay/four_weather_validation_new
```

本次实际图表/CSV/JSON：`outputs/replay/four_weather_validation_final/`。

使用本地Torch2.5.1 / MPS FP32。四场景训练集准备及准备/回放脚本的依赖检查均已在本地执行；尚未在远程CUDA12.8环境执行本次新模型。512输入、2个spawn worker的实际图片训练/验证冒烟完成，无保存权重且正常退出。
