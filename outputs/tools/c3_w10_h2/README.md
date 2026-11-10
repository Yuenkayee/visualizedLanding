# 晴天 C3：平均风速 10 m/s、有效波高 2.0 m

这是独立实验启动器，沿用 `ff91248` 的算法方案，按基线记录的 SHA256 核验算法、
权重、模型、MATLAB 配置与 worker 等运行文件，而不要求当前 HEAD 恰好为 `ff91248`。
提交启动器或修改文档后也可以运行；manifest 记录实际 HEAD 和算法参考提交。
不修改算法、权重、源 SLX 或仓库原有批量入口。原入口固定 C0–C3 × 两海况；
本启动器直接复用其 supervisor、worker、记录、指标与清理逻辑，仅生成自定义任务清单。
不要用原 `shell/run_landing_batch.sh run/analyze` 处理本目录的自定义批次。

- 晴天、C3、原初值、原 STOP、每架次最长 60 s，共 10 架次。
- 从 `outputs/batch/formal_523_w6` 最终纳入的晴天/sea5/C3 读取组号、attempt 和种子，
  并核对原 worker 的种子生成结果完全一致；不使用退回前的恢复策略。
- CETI Uinf=10 m/s，两处 tauAndFDI SignificantWaveHeight=2.0 m；
  两处 RandomSeed 相同，其余波浪参数、CETI 噪声功率保持原模型值。
- 标记为自定义风浪条件，不称为原论文 5 级或 6 级海况。
- 超时/原远离判据均保留，不换种子；基础设施异常停止，resume 重试同一测例。
- 不保存图像/点云，保留压缩 50 Hz 序列、配置、模型参数审计、阶段与终止指标。

本文件夹已纳入 Git，可提交、推送后在服务器通过 `git pull --ff-only` 同步。
也可将同版本压缩包解压到服务器仓库根目录；不要用旧压缩包覆盖已经更新的脚本。

旧启动器的 `This launcher requires the rolled-back ff91248 commit` 来自对 HEAD 的
精确匹配：提交启动器本身就会使 HEAD 改变。更新后不需要再次 reset，也不需要放宽
算法或模型文件检查。该报错发生在创建批次 manifest 和启动 worker 之前，可重新执行
原试运行命令（不加 `--resume`）。已有其他批次仍按其冻结的脚本和源码执行，不修改旧 manifest。

```bash
cd /root/autodl-tmp/visualizedLanding
export PYTHON_BIN=/root/miniconda3/bin/python3
export MATLAB_ROOT=/root/MATLAB/R2024b
export RENDER_PYTHON="$PWD/.venv-render-server/bin/python"
export MATLAB_LICENSE_MODE=online

# 查看共享会话；为后面的变量填写六个当前空闲的真实名称。
"$PYTHON_BIN" -c 'import matlab.engine; print(",".join(matlab.engine.find_matlab()))'
export LANDING_C3_SESSIONS='LandingBatch1,LandingBatch2,LandingBatch3,LandingBatch4,LandingBatch5,LandingBatch6'

# 可选：6 架次 × 2 秒并发试运行（使用不同输出目录）。
bash outputs/tools/c3_w10_h2/run.sh run \
  --limit 6 --duration 2 --workers 6 --matlab-sessions "$LANDING_C3_SESSIONS" \
  --output outputs/batch/c3_clear_w10_h2_smoke

# 正式 10 架次；输出目录须尚不存在。启动器先只读检查依赖，不执行安装。
mkdir -p outputs/logs/batch
nohup bash outputs/tools/c3_w10_h2/run.sh run \
  --workers 6 --matlab-sessions "$LANDING_C3_SESSIONS" \
  --output outputs/batch/c3_clear_w10_h2 \
  > outputs/logs/batch/c3_clear_w10_h2.log 2>&1 < /dev/null &
echo $! > outputs/logs/batch/c3_clear_w10_h2.pid
tail -f outputs/logs/batch/c3_clear_w10_h2.log
```

`LandingBatch1,...` 只是示例名称。若实际是 `MATLAB_1226,...`，使用实际名称。
试运行结束后再开始正式批次，不同时占用同一组会话。Ctrl+C 退出 tail 不会停止后台任务。

```bash
# 中断后先确认原协调器和 worker 已退出，再使用同一目录续跑。
nohup bash outputs/tools/c3_w10_h2/run.sh run --resume \
  --workers 6 --matlab-sessions "$LANDING_C3_SESSIONS" \
  --output outputs/batch/c3_clear_w10_h2 \
  >> outputs/logs/batch/c3_clear_w10_h2.log 2>&1 < /dev/null &
echo $! > outputs/logs/batch/c3_clear_w10_h2.pid

# 只分析已有结果。
bash outputs/tools/c3_w10_h2/run.sh analyze --output outputs/batch/c3_clear_w10_h2
```

结果在 `outputs/batch/c3_clear_w10_h2/analysis/`：

- `status.json`：完成数、STOP/超时/远离数、三阶段完成数、物理参数与种子核验。
- `report.md`：结果摘要与解释。
- `paired_cases.csv/json`：原 12 m/s、3.25 m 的 before 与本次 10 m/s、2.0 m 的 after。
- `sorties.json`：现有指标函数生成的逐架次完整指标。

每例原始数值记录位于 `groups/clear_sXX/attempt_XX/C3/series.jsonl.gz`，
实际写入 MATLAB 的参数位于同目录 `simulink/batch_model.json`。
正式批次完成应为 `completed=10`、`planned=10`、`complete=true`；完成不等于达到 STOP。
原 supervisor 的 finalized groups 在结束生成汇总时统一确认，运行中为 0 是正常现象。

只生成计划可用 `run.sh plan --output <新目录>`；计划生成后必须通过 `run --resume` 启动。
`--limit` 或 `--duration` 缩小的测试会明确标为 pilot，不与 10 架次完整结果混用。
两个风浪参数一起改变，仅能评价联合变化。各自全程的无效时长不能当作共同时间段比较。

启动器已做离线计划、原种子/配置核验、现有 worker 调度接口与汇总校验；
未在本地执行这 10 次 MATLAB/GPU 闭环仿真。预计最终数值结果为数十至数百 MB，
六并发的临时模型/缓存建议预留约 5 GB。
