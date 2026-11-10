# C3 四天气完整闭环试验：风速 10 m/s、有效波高 2 m

入口是 `shell/run_landing_c3.sh`（Python 模块 `main.batch.c3`）。默认晴天、雾天、夜间、夜雾
各 10 架次，共 **40 架次**，单架次最多 60 s。原晴天 10 架次的启动器及结果保留。
本次重新运行四种天气的 40 架次，不把原晴天结果直接混入新批次。

- CETI Mean wind speed = **10 m/s**；Ship_motion 和 PredictModel 两处 tauAndFDI 的
  SignificantWaveHeight 均为 **2 m**。两处波浪块的 RandomSeed 和其他参数由原配置函数核验一致。
- 使用已验证的 `external/innerloop/simu/innerLoop_landing.slx`，仅配置各测例的临时副本。
  CETI 噪声功率、其他波浪参数、算法、权重、初值及 STOP 判据不变。
- 从 `outputs/batch/formal_523_w6` 对应天气/slot 的最终纳入 sea5/C3 测例读取 wind/wave/sensor
  种子及 attempt，并与原 worker 的生成函数逐个核对。每个新测例记录原测例路径。
  该基线目录必须包含 manifest、各组 result.json 以及对应 C3 的 requested、run、metrics、batch_model。
- 本组合标记为自定义风浪工况，**不称为原 5 级或 6 级海况**。
- 为保证比较，超时、持续飞离均保留在分母中，不替换随机种子。基础设施异常停止，续跑重试同一测例；
  中断记录另存。共享 MATLAB 会话由用户管理，worker 结束不会关闭它们。
- 对基线算法和模型作文件内容校验，允许只增加启动器或文档的 Git 提交；计划记录实际 HEAD。
  计划生成后冻结源码和配置，期间不要更新代码；单独重建分析不要求运行旧算法环境。

## 1. 依赖及共享 MATLAB 会话

先把本次代码同步到服务器。在已经完成单次验证的 Ubuntu 22.04 / MATLAB R2024b 环境使用：

```bash
cd /root/autodl-tmp/visualizedLanding
export PYTHON_BIN=/root/miniconda3/bin/python3
export MATLAB_ROOT=/root/MATLAB/R2024b
export RENDER_PYTHON="$PWD/.venv-render-server/bin/python"
export MATLAB_LICENSE_MODE=online
export LANDING_DOWNLOAD_PROFILE=china
export LD_LIBRARY_PATH="$MATLAB_ROOT/bin/glnxa64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# 要求这六个会话已经登录、共享且空闲。不会新启动 MATLAB，也不要求固定会话名称。
LANDING_C3_SESSIONS="$("$PYTHON_BIN" - <<'PY'
import matlab.engine
names = sorted(matlab.engine.find_matlab())
if len(names) != 6:
    raise SystemExit(f"需要选择六个空闲共享会话，当前发现：{names}")
print(",".join(names))
PY
)"
export LANDING_C3_SESSIONS
printf '本批次使用：%s\n' "$LANDING_C3_SESSIONS"
export MATLAB_SHARED_SESSION="${LANDING_C3_SESSIONS%%,*}"

# 只读检查 Python 数值库、导航导入、MATLAB、权重、CUDA 与 OPTIX 渲染。
test -n "$LANDING_C3_SESSIONS" && \
  bash shell/run_landing_c3.sh check --check --device cuda --render-device OPTIX
```

发现会话数量不是六个时，按照打印的真实名称手动设置
`export LANDING_C3_SESSIONS='名称1,名称2,名称3,名称4,名称5,名称6'`，再设置 MATLAB_SHARED_SESSION。
`find_matlab()` 只列出可连接的会话，不能据此判断会话是否正被其他任务占用。
不要把此前日志里的 `LandingBatch1` 或 `MATLAB_1226` 当作当前必然存在的名称。

如检查提示缺依赖，停止使用该 Python 环境的仿真作业后，运行以下安装/检查命令，再做只读检查：

```bash
bash shell/run_landing_c3.sh check --device cuda --render-device OPTIX
```

NumPy 核心残留、账户授权失败属于环境修复问题；普通安装不能保证修复它们。
对应诊断会保留在 `outputs/logs/closed_loop/`。批量 run 只检查依赖，不并发安装软件。

## 2. 六并发短试运行

四种天气各两架次，共 8 架次，每架次 2 s；输出目录须尚不存在。可检查六 worker 的启动、
渲染、数值记录和完整分析，但不能用于判断着舰成功率，2 s 超时正常。

```bash
bash shell/run_landing_c3.sh run \
  --pilot --seeds-per-weather 2 --duration 2 \
  --workers 6 --matlab-sessions "$LANDING_C3_SESSIONS" \
  --output outputs/batch/c3_four_weather_w10_h2_smoke_01
```

若需检查长期负载，可在另一新目录把 `--duration 2` 改为 `--duration 60`；仍保留 `--pilot`。
短试运行完全结束后再使用同一组会话进行正式测试。

## 3. 后台正式运行

```bash
mkdir -p outputs/logs/c3_batch
nohup bash shell/run_landing_c3.sh run \
  --workers 6 --matlab-sessions "$LANDING_C3_SESSIONS" \
  --output outputs/batch/c3_four_weather_w10_h2_01 \
  > outputs/logs/c3_batch/formal_01.log 2>&1 < /dev/null &
echo $! > outputs/logs/c3_batch/formal_01.pid
```

不要加 `--pilot`。默认 40 架次，不是原 C0–C3 × 两海况的 320 架次。
若先单独用 `plan --output ...` 创建计划，首次运行也要加 `--resume`。

```bash
# 当前完成架次、各天气进度及 worker 当前测例
bash shell/run_landing_c3.sh status --output outputs/batch/c3_four_weather_w10_h2_01

# Ctrl+C 只退出日志查看，不终止后台仿真。
tail -f outputs/logs/c3_batch/formal_01.log
```

原 supervisor 的 `Completed current-round runs X/40` 表示已完成架次；`finalized groups`
在本入口结束后统一确认，运行中为 0 正常。后台完成后 `analysis/status.json` 应显示
`expected=40`、`included=40`、`complete=true`、`analysis_complete=true`；这不等于 40 次均达到 STOP。

中断后确认旧协调器和 worker 已退出，再续跑（可换用新的空闲共享会话名称）：

```bash
nohup bash shell/run_landing_c3.sh run --resume \
  --workers 6 --matlab-sessions "$LANDING_C3_SESSIONS" \
  --output outputs/batch/c3_four_weather_w10_h2_01 \
  >> outputs/logs/c3_batch/formal_01.log 2>&1 < /dev/null &
echo $! > outputs/logs/c3_batch/formal_01.pid
```

## 4. 分析、图表及存储

完成后自动分析，全部输出在 `outputs/batch/c3_four_weather_w10_h2_01/analysis/`：

| 文件 | 内容 |
|---|---|
| `report.md`、`status.json` | 中文结果摘要、完整性、STOP/超时/飞离、三阶段完成数、参数和种子核验 |
| `sorties.csv/json` | 每架次误差、相对速度、耗时、控制代价、无效等待等 |
| `table_5_15/16/17.csv/json` | 触舰统计、失败分类、阶段完成时长和删失统计，另含四天气等权汇总 |
| `figures/figure_5_11_*` | 海浪、舰体运动、甲板运动和分布（使用 C3 自身任务时段） |
| `figures/figure_5_12_*`、`figure_5_13_*` | 代表轨迹、触舰点及误差分布 |
| `figures/figure_5_14_*` | 阶段时长、完成状态、代表时间条；`*_all_sorties.png` 展示该天气全部架次 |
| `figure_5_14_all_sorties.csv` | 全部测例阶段起止时间、有效控制标记和终止原因 |
| `figures/figure_5_15_*` | 代表测例全程、末端及存在时的恢复/校正局部控制响应 |
| `figures/figure_5_16_rates.png` | 各天气及等权汇总 STOP 比例与 Wilson 95% 区间 |
| `figure_5_16_applicability.json` | 单一 C3/单一风浪工况不做 C0–C2 或两海况配对差值 |
| `attempt_ledger.*`、`all_attempts_sensitivity.csv` | 完成及中断记录；本次不换种子，完成测例全保留 |
| `analysis_provenance.json`、`command_time_audit.*` | 分析输入指纹及控制差分时间审计 |

图表统计复用原正式分析模块，未触舰帧不代替落点，未完成阶段不混入完成时长。
代表样本固定为每种天气 slot 0，不依据是否成功选取。原始数值序列在
`groups/<weather>_sXX/attempt_XX/C3/series.jsonl.gz`；实际模型参数在其 `simulink/batch_model.json`。

离线分析修正了一个旧指标问题：STOP 末帧可能出现重复时间戳，以极小时间差相除会使
Jctrl/指令变化率虚高。本入口仅在分析时使用时间真实推进的差分；原 `metrics.json` 不变，
旧数值另外保存在 sorties 的 `*_recorded_legacy*` 字段。旧、新控制代价须统一该处理口径再比较。
其他正式入口仍保持其原指标口径。

只重建图表无需 MATLAB 或 GPU，也不需要重新运行仿真：

```bash
bash shell/run_landing_c3.sh analyze --output outputs/batch/c3_four_weather_w10_h2_01

# 给原晴天10架次补全正式图表，另存目录，保留原 analysis。
bash shell/run_landing_c3.sh analyze \
  --output outputs/batch/c3_clear_w10_h2 \
  --analysis-output outputs/batch/c3_clear_w10_h2_full_analysis
```

不要用原 `run_landing_batch.sh analyze` 分析此自定义矩阵。
默认输出不保存 RGB/点云，完成后清理临时模型/缓存并压缩日志。现有晴天10架次约 32.5 MiB，
其中数值序列约 19.9 MiB；新生成完整分析约 2.7 MiB。预计40架次最终占用 **0.2–0.5 GB**，
六并发临时模型/渲染缓存建议预留 **5–10 GB**；频繁中断留下的证据会额外占用空间。
该估计不含已有 MATLAB、Python、模型权重和历史结果。离线验证已完成；实际六并发需在服务器试跑确认。
