# 论文5.2.3：新版闭环批量试验（15张图）

入口为 `shell/run_landing_batch.sh` / `python -m main.batch`。新计划采用
`analysis_profile: thesis_523_15`。旧 manifest 仍走旧报告流程，不能通过改写
旧 manifest 将缺少实际EI状态量的旧结果冒充新试验。

保留导航算法、控制器、训练权重、名义初值及源模型；仅在每架次独立的模型副本中
设置本次风浪与种子，并分支记录观测信号。没有改动 `docs/论文.docx`。
仓库实际模型为已验证的 `external/innerloop/simu/innerLoop_landing.slx`；
用户提及的 `external/innerloop/simu/closeLoop_landing.slx` 不存在。
不使用另一目录下名称相近的模型。

| 试验海况标签 | CETI平均风速 | 两处tauAndFDI有效波高 |
|---|---:|---:|
| 5 | 10 m/s | 2 m |
| 6 | 12 m/s | 2.5 m |

这是用户指定的本次试验参数，不是对海况分级的通用定义，也不沿用旧批次的
12/19 m/s、3.25/5 m。两海况同时改变风浪，差异解释为联合效应。
两处海浪块的所有参数写后核验一致，Hs和RandomSeed同步设置；Tp=11.2 s、浪向150°、
64分量、频率0.2–2 rad/s保持原值。CETI仅改Uinf和原噪声模块seed，噪声功率不改。

正式矩阵：四天气×十种子×两海况×C0—C3 = **320架次**，40个配对组，每组8架次。
`--c3-only`：同样四天气、两海况、十种子，共 **80架次**，40组，每组2架次。
均最长 **80 s**；种子由原master_seed=5232026、天气、slot、replacement生成，
同组跨方法、跨海况一致。新批次从attempt 0开始；若旧批次有替换组，比较时须按实际种子核对，
不能仅凭相同slot声称配对。代表曲线预先固定slot 0及其整组替换，不挑成功案例。

| 方法 | 控制端位置 | 有效性 |
|---|---|---|
| C0 | 真值CG相对位置 | 真值参照；导航旁路运行 |
| C1 | 纯视觉ESKF与原整形器 | 仅跳过雷达融合，其余策略保留 |
| C2 | 完整融合的原始50 Hz位置 | 同时运行原整形器，采用其有效位 |
| C3 | 完整融合与原有界因果整形 | 原有有效位 |

## Ubuntu 22.04准备及依赖检查

先停止使用这些共享MATLAB会话的旧任务，在仓库根目录执行。

```bash
cd /root/autodl-tmp/visualizedLanding
export PYTHON_BIN=/root/miniconda3/bin/python3
export MATLAB_ROOT=/root/MATLAB/R2024b
export RENDER_PYTHON="$PWD/.venv-render-server/bin/python"
export MATLAB_LICENSE_MODE=online
export LANDING_DOWNLOAD_PROFILE=china
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

对于此前已经准备好的六个**专用、空闲**共享会话，先获取实际名称，避免把
`MATLAB_1226`误写为`LandingBatch1`。下面只有恰好六个会话时才自动选择：

```bash
LANDING_SESSIONS="$("$PYTHON_BIN" - <<'PY'
import matlab.engine
names=sorted(matlab.engine.find_matlab())
if len(names)!=6:
    raise SystemExit(f'需要6个专用空闲会话；当前发现：{names}')
print(','.join(names))
PY
)"
export LANDING_SESSIONS
```

若还有其他共享会话，手动把六个目标会话的实际名称填入
`export LANDING_SESSIONS=名称1,名称2,名称3,名称4,名称5,名称6`。
不自动判断已有会话是否正被其他工作占用。不存在会话时，可按已有授权方式启动：

```bash
mkdir -p outputs/logs/matlab
for i in 1 2 3 4 5 6; do
  tmux new-session -d -s "landing_batch_$i" \
    "$MATLAB_ROOT/bin/matlab -licmode online -nodisplay -nosplash -r \"matlab.engine.shareEngine('LandingBatch$i')\"" \
    >"outputs/logs/matlab/start_$i.log" 2>&1
done
```

若授权需要交互，进入对应tmux完成登录；账户必须允许实际使用的并发数。
启动后重新执行发现名称的命令，不以固定等待秒数代替成功检查。
脚本不会自动打开网页或提交凭据。自动拥有的Engine会话授权正常时，可省略所有
`--matlab-sessions`，worker自行启动独立MATLAB。

```bash
# 安装缺少的依赖，然后检查。不要在仿真正在运行时重装。
bash shell/run_landing_batch.sh check \
  --device cuda --render-device OPTIX --matlab-sessions "$LANDING_SESSIONS"

# 只检查，不安装。
bash shell/run_landing_batch.sh check --check \
  --device cuda --render-device OPTIX --matlab-sessions "$LANDING_SESSIONS"
```

检查Ubuntu动态库、Python3.12、NumPy/SciPy真实导入与dtype一致性、Torch/CUDA、四天气权重、
独立Python3.11/bpy与OPTIX、MATLAB R2024b/Engine、Simulink/Control/Robust授权、中文字体。
新增对两个海况模型工作副本的**零时长编译检查**，核验实际舰船状态/导数和原STOP观测点，
源SLX哈希必须不变。原模型控制器初始化可能耗时数分钟；报告保存在
`outputs/logs/closed_loop/<时间>/environment.json`。六会话逐个启动与并发资源能力由压力测试验证。
若NumPy仍报告core/_core类型不一致，先按 `main/README.md` 的环境恢复段处理；普通force-reinstall不能保证清除未登记残留。

## 六并发压力测试

```bash
# 8架次、每架次2秒：接口、编译、6个独立worker与输出链路。
bash shell/run_landing_batch.sh run --pilot \
  --weather clear --seeds-per-weather 1 --duration 2 \
  --workers 6 --matlab-sessions "$LANDING_SESSIONS" \
  --output outputs/batch/pressure_523_15_2s_01

# 32架次、每架次10秒：覆盖四天气，观察持续GPU/内存压力。
bash shell/run_landing_batch.sh run --pilot \
  --seeds-per-weather 1 --duration 10 \
  --workers 6 --matlab-sessions "$LANDING_SESSIONS" \
  --output outputs/batch/pressure_523_15_10s_01
```

输出目录必须为新目录；重跑换后缀，恢复已有计划使用`--resume`。压力测试中的超时是预期，
不作为80 s正式闭环的成功率证据。12 vCPU、62 GB RAM、32 GB显存先沿用六worker；
每worker MATLAB/BLAS/Torch/OpenCV各1线程，Cycles2线程，保持渲染质量。
`resources.jsonl`每30 s记录内存与GPU；`analysis/status.json`给出峰值，RSS包括独立共享MATLAB，
但共享内存页可能重复计入，显存为整卡总量。检查没有OOM及进程异常后再执行完整时长。

## 两海况C3前置测试：80架次

```bash
mkdir -p outputs/logs/batch
nohup bash shell/run_landing_batch.sh run --c3-only \
  --workers 6 --matlab-sessions "$LANDING_SESSIONS" \
  --output outputs/batch/c3_523_15_t80_01 \
  > outputs/logs/batch/c3_523_15_t80_01.log 2>&1 < /dev/null &
```

不需要旧`formal_523_w6`结果作输入，不使用旧`main.batch.c3`的基线commit约束。
C3单方法也生成相同15张图及全部表格；C3-C1/C2差值标为不适用，不能解释成零差异。
压力测试、C3检查和正式批次应顺序运行，不能共用同一批MATLAB会话同时运行。

## 完整正式批次：320架次

```bash
nohup bash shell/run_landing_batch.sh run \
  --workers 6 --matlab-sessions "$LANDING_SESSIONS" \
  --output outputs/batch/formal_523_15_t80_01 \
  > outputs/logs/batch/formal_523_15_t80_01.log 2>&1 < /dev/null &
```

`--pilot`不得用于正式结果。可先用`plan`冻结计划，随后以`run --resume`启动。
记录git版本、源码/模型/权重/配置/论文哈希及landing几何快照；恢复时不允许源内容漂移。
批量过程中不要git pull或修改源文件。

```bash
# 查看完成数、当前任务和分析状态（不依赖MATLAB）。
bash shell/run_landing_batch.sh status --output outputs/batch/formal_523_15_t80_01
# 实时日志；Ctrl-C只退出查看。
tail -f outputs/logs/batch/formal_523_15_t80_01.log

# 中断后恢复同一份计划、种子与输出目录；已经完成的架次不会重跑。
nohup bash shell/run_landing_batch.sh run --resume \
  --workers 6 --matlab-sessions "$LANDING_SESSIONS" \
  --output outputs/batch/formal_523_15_t80_01 \
  >> outputs/logs/batch/formal_523_15_t80_01.log 2>&1 < /dev/null &

# 仿真已完成但绘图失败时，只重新分析，不重跑仿真。
bash shell/run_landing_batch.sh analyze --output outputs/batch/formal_523_15_t80_01
```

停止运行时，先用`ps -p "$(cat 输出目录/coordinator.pid)" -o pid,args`核对仍是该批次
`main.batch run`进程，再对这个PID发送`kill -TERM PID`。协调器清理worker及自己拥有的子进程，
不关闭独立共享MATLAB。`coordinator.pid`启动后才创建，不要向未知PID或全部python进程发信号。
`finalized_groups`是整组配对的提交数；例如320架次=40组×8，不是320×40。

## 终止、统计与输出

原`innerLoop/LAND_CMD/SCOPE_/Stop Simulation`输入触发并停止即为触舰，不新增接触门槛。
另按论文计算固定H中心的实际EI，EI<1是安全触舰，EI≥1是非安全，缺测为未知。
直接记录实际`Ship_motion`动力学的eta/nu/eta_dot/nu_dot（SI），用刚体运动学换算H的速度、
加速度；角速度项取Euler角导数，不以机体p/q替代。权重按论文状态顺序
[25,.25,400,25,100,.25,277.78,44.44]。
**模型内部制导EI的实现顺序和权重与论文评价式不同，保留原实现，离线评价单列，二者不能混用。**

安全率分母包含全部纳入架次；另报触舰发生率与EI完整的已触舰架次中的条件安全比例。
实际触舰点才参与落点分布，不使用超时末帧补点。未完成阶段为删失，未进入为缺失。
条件统计按天气等权；任一所需天气层无样本时不悄悄改权重；配对bootstrap保留缺失原因与数量。
95% Wilson端点处理浮点舍入，避免旧版负yerr报错。Jctrl只使用真实推进的时间区间。

持续飞离规则保留原设置：至少10 s，10 s窗口内距离始终超过max(200 m,初始距离+150 m)，
净远离≥30 m、向外步占比≥0.9、最大回返≤2 m；触发后整个配对组换种子，最多5次替换。
普通超时、导航失效、非安全触舰不换种子。剔除证据和全部尝试保留，主结果必须同时报告选择偏差。
依赖、接口或数值异常暂停批次，恢复同种子，不把基础设施错误转换成“坏随机种子”。

完成后在`输出目录/analysis/`读取：

- `report.md`、`status.json`：结果分析、完整性、样本计数、资源/空间估计。
- `table_5_15/16/17.csv/json`：表现统计、终止分类、阶段完成/删失等。
- `sorties.csv/json`、`table_5_16_case_evidence.csv`：逐架次指标、异常时序、EI各状态项及贡献。
- `attempt_ledger.*`、`all_attempts_sensitivity.csv`：替换/中断及全部尝试的敏感性描述。
- `pairing_audit.json`、`physical_conditions_audit.json`：种子、海浪相位、实际舰船与前馈一致性。
- `figures/`：恰好15张中文PNG；`figure_index.json`映射到原始数值来源。
- `plot_sources/*.npz`：代表曲线全采样精度源数据；全部原始50Hz记录仍在`groups/.../series.jsonl.gz`。

| 图号 | 文件数 | 内容 |
|---|---:|---|
| 5-11 | 1 | 两海况风浪、横纵摇、H运动和实际EI |
| 5-12 | 2 | 每海况晴天/夜雾的轨迹、法向高度/速度 |
| 5-13 | 2 | 每海况四天气的全部触舰点与EI分类 |
| 5-14 | 2 | 每海况阶段完成时长及已触舰总时长 |
| 5-15 | 2 | 每海况夜雾的导航/消费误差、制导、指令与有效性 |
| 5-16 | 2 | 安全率及成对比较 |
| 5-17 | 4 | 每天气两海况的触舰EI箱线/散点与ECDF |

不存图像帧、点云或视频，不重复保存全量CSV时间序列，不降低原始日志采样率。
每架次结束删除工作SLX和缓存、压缩日志，保留模型参数审计及失败证据。
现有本地80 s结果折算320×80 s压缩序列约0.9 GiB；新记录增加实际EI所需24个数值，
预计正式320架次结果约**2–4 GiB**，C3的80架次约**0.5–1 GiB**。
建议数据盘正式运行前保留**10 GiB以上**，C3保留**5 GiB以上**，供六worker活动缓存和替换组使用。
大量替换时空间随尝试次数增加；极端所有组均替换5次为1920架次，建议至少30 GiB余量。
安装MATLAB/渲染环境的空间另算。压力测试结束后的status外推值更适合该服务器。
