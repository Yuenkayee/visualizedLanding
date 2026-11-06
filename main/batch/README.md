# 论文 5.2.3 闭环批量试验

本入口仅实现导航参与控制的闭环试验。保留已验证的
`external/innerloop/simu/innerLoop_landing.slx`、控制器、导航参数、四天气权重与名义初值。
不修改 `docs/论文.docx`，不包含 N0—N3 的固定输入导航评测。

正式矩阵为 4 天气 × 2 海况 × 10 种子 × 4 方法 = **320 架次**。
默认最长仿真时间 60 s，模型原有自动 STOP 提前结束。

| 方法 | 控制端位置 | 有效性 |
|---|---|---|
| C0 | 真实直升机质心减舰船质心的 NED 位置 | 真值参照；导航仅旁路运行 |
| C1 | 学习前端、纯视觉 ESKF、原有信号整形 | 仅跳过雷达融合更新；其他健康策略保持 |
| C2 | 完整融合的原始 50 Hz 位置 | 同时运行原整形器，只采用其有效位 |
| C3 | 完整融合与原有有界因果整形 | 原有有效位 |

两种海况分别使用 Hs=3.25/5.00 m、CETI 平均风速12/19 m/s。
两处 `tauAndFDI` 的全部属性必须一致；仅设置 Hs 和 RandomSeed，核验现有
Tp=11.2 s、浪向150°、64分量、0.2–2 rad/s；CETI 仅设置 Uinf 与白噪声 seed。
修改只发生在每架次的工作副本，原始 SLX 文件不保存修改。
因此海况差值代表**风浪共同变化**，不能解释成只改变 Hs 的因果效应。

## Ubuntu 22.04 依赖与安装

在仓库根目录执行；以下沿用已验证的 Python 和 MATLAB 安装，不创建一个不同的导航环境。

```bash
cd /root/autodl-tmp/visualizedLanding
export PYTHON_BIN=/root/miniconda3/bin/python3
export MATLAB_ROOT=/root/MATLAB/R2024b
export RENDER_PYTHON="$PWD/.venv-render-server/bin/python"
export MATLAB_LICENSE_MODE=online
export LANDING_DOWNLOAD_PROFILE=china

# 检查并安装缺失依赖；同原单次闭环部署脚本。
bash shell/run_landing_batch.sh check --device cuda --render-device OPTIX

# 可选：只检查，不安装。
bash shell/run_landing_batch.sh check --check --device cuda --render-device OPTIX
```

检查范围：Ubuntu动态库、NumPy/SciPy真实导入、Torch/CUDA、检查点、独立
Python3.11的bpy渲染环境、MATLAB R2024b Engine、Simulink/Control/Robust工具箱和授权。
批量运行前额外在新进程执行 `from scipy.stats import chi2`，防止先前混装NumPy问题复现。
不在并发worker中运行pip，不在批量过程中重新安装依赖。

## 并发压力测试

每个worker有独立Python、MATLAB、Blender进程和Simulink缓存。自动启动的MATLAB
在该worker内复用；各架次重新初始化动态状态和随机流。
MATLAB/BLAS/OpenCV/Torch默认各1线程，Cycles2线程；不改变图像分辨率或采样质量。
推荐在22vCPU、110GB内存、96GB显存服务器上先试4个worker，再依据实测决定是否增大。

```bash
# 8架次，每架次2秒，仅验证接口、并发授权与资源占用。
# 默认不会继承单次运行的 MATLAB_SHARED_SESSION 到所有worker。
bash shell/run_landing_batch.sh run --pilot \
  --weather clear --seeds-per-weather 1 --duration 2 --workers 4 \
  --output outputs/batch/pressure_2s

# 相同小矩阵的8架次完整闭环；核验终止数据与全部图表。
bash shell/run_landing_batch.sh run --pilot \
  --weather clear --seeds-per-weather 1 --workers 4 \
  --output outputs/batch/pilot_full
```

查看 `analysis/status.json` 的 `complete`、`pairing_passed`、
`peak_batch_process_tree_rss_gib`、`peak_total_gpu_memory_gib`；显存为整卡占用，
若还有其他GPU进程应一并考虑。`resources.jsonl` 每30秒记录进程树RSS和GPU利用率。
短测的超时是预期结果，不构成触舰成功率证据。每架次初次编译和原有控制器计算
也可能耗时数分钟，2秒是物理仿真时长，不是壁钟运行时间。

若服务器只能连接手动登录后的MATLAB，先在四个独立tmux会话中分别启动：

```bash
tmux new-session -s landing_batch1
/root/MATLAB/R2024b/bin/matlab -licmode online -nodisplay -nosplash \
  -r "matlab.engine.shareEngine('LandingBatch1')"
```

用 `Ctrl+B D` 分离，重复创建 `landing_batch2/3/4` 和 `LandingBatch2/3/4`。
必须是同一系统用户且授权允许同时运行的四个会话。在运行命令后附加：

```bash
--matlab-sessions LandingBatch1,LandingBatch2,LandingBatch3,LandingBatch4
```

同一共享会话不能提供给两个worker或同时用于单次仿真。脚本拒绝重复会话名；
一次批次内独占会话，各批次的共享会话还需由使用者隔离。

## 正式批量与恢复

```bash
bash shell/run_landing_batch.sh run --workers 4 \
  --output outputs/batch/formal_523

# Ctrl+C、进程错误或机器中断后：已完成架次不重复运行。
bash shell/run_landing_batch.sh run --resume --workers 4 \
  --output outputs/batch/formal_523

# 仅重算统计/重绘，不启动MATLAB或渲染器。
bash shell/run_landing_batch.sh analyze --output outputs/batch/formal_523
```

新批次要求输出目录不存在；`--resume`读取原manifest的场景、设备和策略，
只允许调整worker数量/专用会话安排。源码、模型、权重、配置或论文内容的哈希
改变时拒绝续跑，避免混入两个实验定义。修复实现后应使用新批次目录。
`plan --output ...`可先冻结矩阵，随后用 `run --resume --output ...`执行。
正式模式不接受命令行缩短时长、减少天气/种子或CPU近似渲染；这些覆盖只用于`--pilot`。

飞离规则在 `main/batch_closed_loop.yaml` 预先冻结：至少运行10s；连续10s的
三维起落架/H距离均大于200m且大于初始距离+150m；净向外运动至少30m；
90%以上相邻步向外；窗口内从峰值回落不超过2m。
这是试验异常筛选规则，不是制导算法参数。任一方法触发时，将同天气/种子编号的
两海况、四方法共8架次整体排除，用下一个确定性种子重做8架次，最多替换5次。
临时向外运动、导航无效、普通超时均不自动换种子。

接口、授权、依赖、进程异常会停止批次，不以新种子掩盖错误。
`--resume`归档中断架次并使用相同种子补跑。所有尝试、排除原因、
触发时刻和距离证据都保留；额外提供全部尝试的描述性统计，显示筛选的选择偏差。

## 图表、数据与统计口径

所有结果在批次目录下，论文文件不被写入。

| 论文项 | `analysis/`输出 |
|---|---|
| 表5-15 | `table_5_15.csv/json`：满足数/总数、Wilson区间、条件落点/速度、Jctrl、任务时长 |
| 表5-16 | `table_5_16.csv/json`、`table_5_16_case_evidence.csv`、`attempt_ledger.csv/json` |
| 表5-17 | `table_5_17.csv/json`：阶段完成时长、进入/完成/删失、等待、回退、任务时长 |
| 图5-11 | `figures/figure_5_11_*`、`figure_5_11_motion_series.csv`、`figure_5_11_motion_metrics.csv` |
| 图5-12 | `figures/figure_5_12_*`，原始50Hz相对航迹/速度和阶段见对应series |
| 图5-13 | `figures/figure_5_13_*`、`figure_5_13_touchdowns.csv`，包括每点允许区、名义点和误差 |
| 图5-14 | `figures/figure_5_14_*`、`figure_5_14_phases.csv`与代表架次阶段条带 |
| 图5-15 | `figures/figure_5_15_*`，每天气/海况/方法的全程、末段、首次恢复及最大观测校正放大图 |
| 图5-16 | `figures/figure_5_16_*`、`figure_5_16_paired.csv/json`，每项有效配对数及缺失原因 |
| 核验与分析 | `pairing_audit.json`、`representative_cases.csv`、`sorties.csv/json`、`report.md`、`status.json` |

`groups/<weather_slot>/attempt_XX/seaN_CM/series.jsonl.gz`保留每个控制步的
物理时间、当前/下一状态、真实起落架位置速度、甲板/机体旋转、原生12维状态、
CG真值/原始/影子整形、控制实际消费量、制导误差、速度指令/舰船前馈、
阶段、健康/整形/消费有效位及传感器采样和拒绝诊断。
`simulink/batch_model.json`保存实际块路径、写后读取参数、完整波浪分量及原控制幅值。
保存float64，不为节省空间丢弃末段或无效帧。

当前触舰政策完全采用用户指定的原模型STOP代理，候选点和合格子集相同。
不另设未知的速度、倾斜、接触高度或保持时间门限，不宣称稳定着舰。
“接触但不合格”“导航主动中止”“STOP与几何不一致”等额外分类标为未启用，
接口异常在尝试台账独立列示，不静默进入成功分母。

甲板坐标原点为H，x向舰艏、y向左舷、z向上；名义CG目标按真实姿态和杆臂
转换为名义起落架点。绘图甲板24×14m来自现有程序化几何；允许区基于仓库
暂定轮胎椭球投影侵蚀，仅作几何参考，不是已认证着舰包络或新STOP门限。

不将失败末帧当作触舰点，不将未进入阶段置零，不把未完成阶段的停留时间混入
完成时长。按架次统计；按天气分层和种子配对bootstrap（10000次）。
等权汇总遇到某天气无条件样本时留空并列原因；不偷偷改变天气权重。
成功率取当前STOP代理，径向误差与成功任务时长的差值要求双方均有候选点。
同时输出全部架次任务时长，避免早停失败看起来更快。

原制导为atan渐近限幅；图中的95%边界是“接近制导渐近幅值”，并非执行机构硬饱和。
Jctrl采用原三轴渐近幅值归一化、W=I。舰船前馈和相同海况下的舰船状态必须
在共同时间段一致，否则分析报错并留下核验记录。
波浪RMS/峰值以C0实际任务时段计算、注明时长，重复方法的相同输入不当成独立样本。
当前视觉海面仅有一致海况标签，仍为现有程序化海面，不能复现动力学的同一随机波场。

## 磁盘空间

保留压缩50Hz序列、参数、标量指标和PNG，不保留逐帧RGB、掩膜、点云或重复模型。
每架次结束清理私有Simulink缓存与模型副本，console/worker日志压缩。
正式320架次预计结果约 **2–5GB**，建议为已有依赖的服务器预留 **20GB**
输出空间以覆盖活动缓存、排除替换架次和绘图；高替换率会增加用量。
短测的`status.json`提供按实际压缩字节数外推的60s×320序列空间。
MATLAB、Torch/CUDA、bpy及安装下载缓存不含在20GB中；全新安装建议另外预留50–80GB。

## 开发验证

```bash
python -m pytest -q tests/test_closed_loop_batch.py tests/test_main_closed_loop.py \
  tests/test_cg_signal.py tests/test_navigation_connection.py
```

单元测试与合成图表只验证实现和统计口径，不能替代服务器完整闭环预试验或正式结果。
