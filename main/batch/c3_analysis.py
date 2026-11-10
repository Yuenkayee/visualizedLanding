"""Formal-report metrics and figures for the single-condition C3 experiment."""
import copy
import json
from pathlib import Path
import numpy as np
from .common import read_json, write_json, sha
from .analysis import csv_file, tables
from .command_analysis import increments
from .metrics import read_series


def resource_peaks(out):
    rss = gpu = 0.
    path = out/'resources.jsonl'
    if not path.exists():
        return rss, gpu
    for line in path.read_text().splitlines():
        sample = json.loads(line); processes = {}
        for row in sample.get('process_rss_kib', '').splitlines()[1:]:
            fields = row.split(None, 3)
            if len(fields) == 4:
                try: processes[int(fields[0])] = (int(fields[1]), int(fields[2]))
                except ValueError: pass
        members = set(sample.get('workers', []))
        for _ in range(12):
            members.update(pid for pid, (parent, _) in processes.items() if parent in members)
        rss = max(rss, sum(value for pid, (_, value) in processes.items() if pid in members)/1024**2)
        for row in sample.get('gpu_memory_mib_utilization_percent', '').splitlines():
            try: gpu = max(gpu, float(row.split(',')[0])/1024)
            except ValueError: pass
    return rss, gpu


def analyze(out, destination=None):
    # Import locally to avoid starting workers or checking installed MATLAB.
    from .c3 import CONDITION, read_completed, require
    out = Path(out); manifest = read_json(out/'manifest.json')
    require(manifest.get('experiment') in ('c3_four_weather_w10_h2', 'c3_clear_w10_h2'),
            'Expected a C3 wind10/Hs2 experiment directory')
    policy = copy.deepcopy(manifest['policy'])
    require(policy['methods'] == ['C3'] and policy['seas'] == {CONDITION: dict(wind_mps=10., wave_height_m=2.)},
            'Report policy does not match C3 / wind10 / Hs2')
    policy['analysis_advancing_command_intervals'] = True
    tasks = read_json(out/'tasks.json')
    expected_tasks = [dict(group=g, attempt=g['attempt'], sea=CONDITION, method='C3',
                          case=str(Path('groups')/g['id']/f"attempt_{g['attempt']:02d}"/'C3'))
                      for g in manifest['groups']]
    require(tasks == expected_tasks, 'Frozen task list does not match manifest')
    rows, missing = read_completed(out, manifest)
    data = Path(destination) if destination else out/'analysis'
    if destination:
        require(data != out and out not in data.parents and data not in out.parents,
                'Separate analysis destination must be outside the input batch')
    data.mkdir(parents=True, exist_ok=True)
    # Make incomplete reruns visibly incomplete even if plotting subsequently fails.
    write_json(data/'status.json', dict(complete=False, analysis_complete=False,
        expected=len(tasks), included=len(rows), missing=missing))
    ledger = []; audits = []; evidence_hashes = {'manifest.json': sha(out/'manifest.json'), 'tasks.json': sha(out/'tasks.json')}
    for task in tasks:
        folder = out/task['case']
        if (folder/'requested.json').exists():
            ledger.append(dict(case=task['case'], group=task['group']['id'], attempt=task['attempt'],
                status='included' if (folder/'metrics.json').exists() else 'incomplete',
                error=read_json(folder/'worker_error.json') if (folder/'worker_error.json').exists() else None))
        for archive in folder.parent.glob('C3_interrupted_*'):
            ledger.append(dict(case=str(archive.relative_to(out)), group=task['group']['id'],
                attempt=task['attempt'], status='interrupted',
                error=read_json(archive/'worker_error.json') if (archive/'worker_error.json').exists() else None))
    for row in rows:
        folder = out/row['path']; series = read_series(folder); model = read_json(folder/'simulink/batch_model.json')
        _, _, command = increments(series, model, policy['command_cost_weights'])
        row['Jctrl_recorded_legacy'] = row['Jctrl']
        row['command_rate_peak_recorded_legacy_mps2'] = row['command_rate_peak_mps2']
        row.update(Jctrl=command['Jctrl'], command_rate_peak_mps2=command['command_rate_peak_mps2'], inclusion='included')
        audits.append(dict(case=row['path'], recorded_Jctrl=row['Jctrl_recorded_legacy'], **command))
        for name in ('requested.json', 'run.json', 'metrics.json', 'series.jsonl.gz', 'simulink/batch_model.json'):
            evidence_hashes[str(Path(row['path'])/name)] = sha(folder/name)
    csv_file(data/'attempt_ledger.csv', ledger); write_json(data/'attempt_ledger.json', ledger)
    csv_file(data/'all_attempts_sensitivity.csv', [dict(weather=w, sea=CONDITION, method='C3',
        attempted=len(group), candidates=sum(r['candidate'] for r in group),
        runaways=sum(r['termination']=='runaway' for r in group),
        note='No seed replacement; all completed outcomes included. Interrupted infrastructure runs are in attempt_ledger.')
        for w in ['equal_weather', *policy['weather']]
        for group in [[r for r in rows if w=='equal_weather' or r['weather']==w]]])
    csv_file(data/'sorties.csv', rows); write_json(data/'sorties.json', rows)
    csv_file(data/'command_time_audit.csv', audits); write_json(data/'command_time_audit.json', audits)
    write_json(data/'analysis_provenance.json', dict(input_batch=str(out.resolve()), input_sha256=evidence_hashes,
        analysis_source_sha256={p.name: sha(p) for p in Path(__file__).parent.glob('*.py')},
        condition=manifest['condition'], no_simulation_files_rewritten=True))
    csv_file(data/'table_5_16_case_evidence.csv', [dict(case=r['path'], weather=r['weather'], termination=r['termination'],
        first_anomaly=r['first_anomaly'], last_valid_s=r['last_valid_s'], terminal_stage=r['terminal_stage'],
        terminal_truth=r['terminal_truth'], accompanied_events=r['accompanied_events'],
        runaway_evidence=r['runaway_evidence'], cause='unverified; temporal order is not causation') for r in rows])
    applicability = dict(applicable=False, reason='Only C3 and one wind/wave condition were run; no C0-C2 or two-sea paired contrasts.')
    write_json(data/'pairing_audit.json', applicability)
    write_json(data/'figure_5_16_paired.json', [])
    csv_file(data/'figure_5_16_paired.csv', [], ['weather', 'left', 'right', 'metric', 'pairs', 'mean', 'low', 'high'])
    write_json(data/'figure_5_16_applicability.json', applicability)
    t15, t16, t17 = tables(rows, policy, data)
    if rows:
        from .figures import figures
        figures(out, rows, policy, data, t15, t17, [])
        all_sortie_phases(out, rows, policy, data)
    peak, gpu = resource_peaks(out)
    size = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    planned_duration = policy['duration_s']
    per_sortie = [((out/r['path']/'series.jsonl.gz').stat().st_size /
                  max(r['duration_s'], .02)*planned_duration) for r in rows]
    status = dict(complete=not missing, analysis_complete=True, pilot=manifest['pilot'], expected=len(tasks), included=len(rows),
        candidate_count=sum(r['candidate'] for r in rows), timeout=sum(r['termination']=='timeout' for r in rows),
        runaway=sum(r['termination']=='runaway' for r in rows), missing=missing, seed_replacements=0,
        physical_conditions_and_seeds_passed=not missing, seed_checks_completed=len(rows), pairing_applicable=False,
        phases_completed=[sum(r['phases'][j]['completed'] for r in rows) for j in range(3)],
        peak_worker_process_tree_rss_gib=peak, peak_total_gpu_memory_gib=gpu,
        memory_note='Externally shared MATLAB processes are not worker descendants; RSS excludes those MATLAB sessions.',
        stored_bytes=size, planned_duration_s=planned_duration,
        estimated_planned_series_gib=float(np.median(per_sortie)*len(tasks)/1024**3) if per_sortie else None,
        command_omitted_nonadvancing_intervals=sum(a['omitted_nonadvancing_intervals'] for a in audits))
    write_json(data/'status.json', status)
    lines = ['# C3 闭环结果：平均风速 10 m/s，有效波高 2 m', '',
        f"完成 {len(rows)}/{len(tasks)} 架次；STOP 代理 {status['candidate_count']}，超时 {status['timeout']}，持续飞离 {status['runaway']}。试运行：{manifest['pilot']}。",
        f"单架次仿真时间上限 {planned_duration:g} s；达到原 STOP 或持续飞离判据时仍提前终止。",
        '所有已完成架次均进入分母，超时和飞离不换种子；基础设施错误单列。代表轨迹预先固定为每种天气的 slot 0，不按成功与否挑选。',
        '统计、表5-15～5-17和图5-11～5-16复用正式分析流程。图5-11使用C3实际任务时段的载荷；图5-16仅给出C3各天气及等权汇总的STOP率和Wilson区间，方法差值、两海况差值不适用。',
        'STOP沿用原模型判据，不代表经过额外速度、位置或稳定接触门限检验。没有STOP的终端帧不作为落点。阶段未完成时的停留不混入完成时长，未进入不记为零。',
        '工况是自定义10 m/s与2 m组合，不标为原5/6级海况。两处波浪块由原MATLAB配置代码写后读取并核验全部参数一致。',
        '导航、控制、权重和原始记录均未改变。Jctrl及变化率仅使用真实推进的控制时间间隔；重复STOP时间戳不除以极小数。旧值保存在sorties的legacy字段，剔除间隔见command_time_audit。',
        'Jctrl按原三轴atan渐近幅值归一化，W=I；95%近边界不等于执行机构硬饱和。与旧报告对比前应采用同一离线处理口径。',
        '天气等权汇总要求各层有条件样本；某天气无候选点时，条件分布汇总留空。没有按天气强行配对不同种子。',
        '图中的甲板允许区仅作几何参考；视觉海面不是动力学同一随机波面。导航拒绝和控制异常的先后关系不证明因果。',
        f"当前批次目录 {size/1024**3:.3f} GiB；{len(tasks)}架次各{planned_duration:g}s的压缩序列估计 {status['estimated_planned_series_gib']} GiB。共享MATLAB内存未计入worker进程树RSS。", '', '## 天气统计', '']
    for r in t15:
        lines.append(f"- {r['weather']}：STOP {r['candidates']}/{r['total']}；Wilson95% [{r['wilson_low']}, {r['wilson_high']}]；径向误差中位数 {r['all_candidates_radial_error_m']['median']} m。")
    lines += ['', '数据：sorties.csv/json、表5-15～5-17、figure_5_14_all_sorties.csv、各图同名数据及groups下series.jsonl.gz。完整性以status.json为准，报告不改变试验原始记录。']
    (data/'report.md').write_text('\n\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)


def all_sortie_phases(out, rows, policy, data):
    from .figures import plt, save, PHASES
    # A compact all-case timeline complements the preregistered representative strips.
    phase_data = []
    for weather in policy['weather']:
        group = sorted([r for r in rows if r['weather']==weather], key=lambda r:r['slot'])
        if not group: continue
        fig, ax = plt.subplots(figsize=(12, max(3, .45*len(group))))
        colors = ['#625294', '#4e9b7c', '#d3ad3d']
        for i, m in enumerate(group):
            series = read_series(out/m['path']); intervals=[]
            for r in series:
                dt = r['t1']-r['t']
                if dt<=1e-9: continue
                key = (r['stage'], bool(r['u']['control_valid']))
                if intervals and intervals[-1][2:]==key and abs(intervals[-1][1]-r['t'])<1e-8:
                    intervals[-1]=(intervals[-1][0],r['t1'],*key)
                else: intervals.append((r['t'],r['t1'],*key))
            for start,end,stage,valid in intervals:
                ax.broken_barh([(start,end-start)],(i-.3,.6),facecolors=colors[stage])
                if not valid:ax.broken_barh([(start,end-start)],(i-.3,.12),facecolors='red')
                phase_data.append(dict(case=m['path'], weather=weather, slot=m['slot'], start_s=start,end_s=end,
                                       stage=stage, control_valid=valid, termination=m['termination']))
            ax.plot(m['duration_s'],i,'o' if m['candidate'] else 'x',color='black')
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=c,label=s) for c,s in zip(colors,PHASES)]+[Patch(color='red',label='Invalid control')],
                  fontsize=8, loc='lower left', bbox_to_anchor=(0,1.01), ncol=4)
        ax.set(yticks=range(len(group)),yticklabels=[f"s{r['slot']:02d} {r['termination']}" for r in group],
               xlabel='Time / s',xlim=(0,policy['duration_s']*1.01))
        save(fig,data/'figures'/f'figure_5_14_{weather}_all_sorties.png',f'C3 {weather}: all sorties; circle=STOP, cross=other termination')
    csv_file(data/'figure_5_14_all_sorties.csv',phase_data)
