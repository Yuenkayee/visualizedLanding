"""Revised thesis tables 5-15..17 and exactly 15 PNGs, with offline source data."""
from pathlib import Path
from math import comb
import json
import numpy as np
from .common import read_json, write_json, seeds, sha
from .metrics import summarize, read_series, arrays, allowed_region
from .analysis import csv_file, selection
from .statistics import wilson
from .report_policy import conditions
from .energy import validate_policy
from .thesis_metrics import trace

CATEGORIES=('safe_touchdown','unsafe_touchdown','touchdown_ei_unknown','navigation_abort',
            'timeout','interface_numerical','runaway')


def description(rows, key, weathers):
    groups=[[float(r[key]) for r in rows if r['weather']==w and r.get(key) is not None
             and np.isfinite(r[key])] for w in weathers]
    result=dict(n=sum(map(len,groups)),n_by_weather=dict(zip(weathers,map(len,groups))))
    if not all(groups):
        return dict(result, **{k:None for k in ('mean','median','q25','q75','p95','max','std')})
    v=np.concatenate(groups);w=np.concatenate([np.full(len(g),1/len(g)/len(groups)) for g in groups])
    ix=np.argsort(v);x=v[ix];cdf=np.cumsum(w[ix]);mean=float(w@v)
    result.update(mean=mean,std=float(np.sqrt(w@((v-mean)**2))),max=float(v.max()))
    for k,q in [('q25',.25),('median',.5),('q75',.75),('p95',.95)]:
        result[k]=float(x[min(np.searchsorted(cdf,q-1e-14),len(x)-1)])
    return result


def tables(rows, policy, data):
    t15,t16,t17=[],[],[]
    for weather in ['equal_weather',*policy['weather']]:
        strata=policy['weather'] if weather=='equal_weather' else [weather]
        for sea in conditions(policy):
            for method in policy['methods']:
                group=selection(rows,weather,sea,method);n=len(group)
                base=dict(weather=weather,sea=sea,method=method,total=n)
                counts=[sum(r['weather']==w for r in group) for w in strata]
                balanced=bool(all(counts) and len(set(counts))==1)
                touched=[r for r in group if r['candidate']];safe=[r for r in group if r['safe']]
                known=[r for r in touched if r['ei_complete']]
                row=dict(base,touchdowns=len(touched),safe=len(safe),unsafe=sum(r['termination']=='unsafe_touchdown' for r in group),
                         ei_unknown=len(touched)-len(known),no_touchdown=n-len(touched),ei_complete=len(known))
                for name,field in [('safe','safe'),('touchdown','candidate')]:
                    k=sum(r[field] for r in group)
                    row[name+'_rate']=float(np.mean([np.mean([r[field] for r in group if r['weather']==w]) for w in strata])) if all(counts) else None
                    low,high=wilson(k,n) if balanced else (None,None)
                    row[name+'_wilson_low']=low;row[name+'_wilson_high']=high
                # Conditional proportion explicitly reports its complete-EI denominator.
                row['conditional_safe_numerator']=len(safe);row['conditional_safe_denominator']=len(known)
                row['conditional_safe_proportion']=len(safe)/len(known) if known else None
                row['conditional_safe_equal_weather']=description(known,'safe',strata)['mean']
                row['wilson_note']='balanced sorties' if balanced else 'unbalanced or missing weather; interval withheld'
                for subset,name in [(touched,'all_touchdowns'),(safe,'safe_touchdowns')]:
                    for key in ('ei','error_x_m','error_y_m','radial_error_m','horizontal_speed_mps','normal_speed_mps','duration_s'):
                        row[name+'_'+key]=description(subset,key,strata)
                for key in ('duration_s','Jctrl','invalid_wait_s','nav_invalid_s','longest_nav_invalid_s'):
                    row[key]=description(group,key,strata)
                for subset,name in [(group,'all_tasks'),(safe,'safe_tasks')]:
                    feedback={}
                    for source in ('raw_source','consumed_source','consumed_evaluation'):
                        feedback[source]={}
                        for validity in ('full','valid'):
                            records=[dict(weather=r['weather'],**r['feedback_errors'][source][validity]) for r in subset]
                            feedback[source][validity]={key:description(records,key,strata)
                                for key in ('rmse_norm_m','p95_norm_m','peak_norm_m')}
                            feedback[source][validity]['rmse_axis_m']=[description(
                                [dict(weather=r['weather'],value=r['rmse_axis_m'][j] if r['rmse_axis_m'] else None) for r in records],
                                'value',strata) for j in range(3)]
                    row[name+'_feedback_errors']=feedback
                t15.append(row)
                for category in CATEGORIES:
                    cases=[r for r in group if r['termination']==category]
                    t16.append(dict(base,category=category,count=len(cases),proportion=len(cases)/n if n else None,
                        case_ids=[r['path'] for r in cases],
                        note='No inferred causal attribution; infrastructure interruptions separately listed in attempt_ledger'))
                for stage in range(3):
                    entered=[r for r in group if r['phases'][stage]['entered']]
                    completed=[r for r in group if r['phases'][stage]['completed']]
                    def phase_desc(cases,key):
                        return description([dict(weather=r['weather'],value=r['phases'][stage][key]) for r in cases],'value',strata)
                    t17.append(dict(base,stage=stage,entered=len(entered),completed=len(completed),
                        unentered=n-len(entered),unfinished=len(entered)-len(completed),
                        completed_duration_s=phase_desc(completed,'completed_duration_s'),
                        censored_dwell_s=phase_desc([r for r in entered if not r['phases'][stage]['completed']],'dwell_s'),
                        waiting_s=phase_desc(entered,'invalid_wait_s'),valid_dwell_s=phase_desc(entered,'valid_dwell_s'),
                        regressions=sum(r['phases'][stage]['regressions'] for r in group),
                        unfinished_cases=[dict(case=r['path'],termination=r['termination'],dwell_s=r['phases'][stage]['dwell_s'])
                            for r in entered if not r['phases'][stage]['completed']],
                        all_task_duration_s=description(group,'duration_s',strata),
                        touchdown_task_duration_s=description(touched,'duration_s',strata),
                        safe_task_duration_s=description(safe,'duration_s',strata)))
    for name,value in [('table_5_15',t15),('table_5_16',t16),('table_5_17',t17)]:
        write_json(data/(name+'.json'),value);csv_file(data/(name+'.csv'),value)
    return t15,t16,t17


def paired(rows,left,right,metric,weather,policy):
    strata=policy['weather'] if weather=='equal_weather' else [weather]
    index={(r['weather'],r['slot'],r['sea'],r['method']):r for r in rows}
    diffs={w:[] for w in strata};missing={};pairs=[];discordant=[0,0]
    for w in strata:
        for slot in range(policy['seeds_per_weather']):
            a,b=index.get((w,slot,*left)),index.get((w,slot,*right));reason=None
            if a is None or b is None:reason='missing_run'
            elif any(a[k]!=b[k] for k in ('replacement','wave_seed','wind_seed','sensor_seed')):reason='unpaired_seed'
            elif metric in ('ei','radial_error_m','touchdown_duration_s') and not (a['candidate'] and b['candidate']):reason='both_touchdowns_required'
            elif metric=='safe_duration_s' and not(a['safe'] and b['safe']):reason='both_safe_required'
            key='duration_s' if metric.endswith('_duration_s') else metric
            if not reason and (a.get(key) is None or b.get(key) is None):reason='unknown_metric'
            if reason:missing[reason]=missing.get(reason,0)+1;continue
            diff=float(a[key])-float(b[key]);diffs[w].append(diff)
            pairs.append(dict(weather=w,slot=slot,replacement=a['replacement'],left=a[key],right=b[key],difference=diff))
            if metric=='safe':
                discordant[0]+=int(a[key] and not b[key]);discordant[1]+=int(b[key] and not a[key])
    counts={w:len(g) for w,g in diffs.items()};mean=low=high=None
    if all(counts.values()):
        mean=float(np.mean([np.mean(g) for g in diffs.values()]))
        rng=np.random.default_rng(policy['bootstrap_seed']);boot=[]
        for values in diffs.values():
            v=np.array(values);boot.append(v[rng.integers(len(v),size=(policy['bootstrap_samples'],len(v)))].mean(axis=1))
        low,high=map(float,np.quantile(np.mean(boot,axis=0),[.025,.975]))
    discord=sum(discordant);pvalue=None
    if metric=='safe' and pairs:
        pvalue=min(1.,2*sum(comb(discord,k) for k in range(min(discordant)+1))/2**discord) if discord else 1.
    return dict(weather=weather,left=f'sea{left[0]}_{left[1]}',right=f'sea{right[0]}_{right[1]}',metric=metric,
        mean=mean,low=low,high=high,pairs=len(pairs),counts_by_weather=counts,missing=missing,raw_pairs=pairs,
        discordant_left_only=discordant[0],discordant_right_only=discordant[1],mcnemar_exact_p=pvalue,
        interval_note='paired seed bootstrap within weather; one-pair intervals can degenerate; no multiplicity correction')


def audit_group(out, group_rows, policy):
    """Compare actual common physical times; terminal duplicate rows are excluded."""
    audits=[]
    for sea in conditions(policy):
        subset=[r for r in group_rows if r['sea']==sea];reference=None;delta_ship=delta_ff=0.;ncommon=[]
        for m in subset:
            rows=read_series(out/m['path']);samples={round(r['t1'],8):r for r in rows if r['t1']-r['t']>1e-9}
            if reference is None:reference=samples;continue
            common=sorted(set(reference)&set(samples));ncommon.append(len(common))
            for key in common:
                delta_ship=max(delta_ship,float(np.max(abs(np.array(samples[key]['s1']['ship'])-reference[key]['s1']['ship']))))
                delta_ff=max(delta_ff,float(np.max(abs(np.array(samples[key]['u']['feedforward'])-reference[key]['u']['feedforward'])))*1852/3600)
        same=len({(r['wave_seed'],r['wind_seed'],r['sensor_seed']) for r in subset})==1
        audits.append(dict(group=group_rows[0]['group'],sea=sea,method_comparison_applicable=len(subset)>1,
            common_samples=min(ncommon) if ncommon else None,ship_difference=delta_ship,feedforward_difference_mps=delta_ff,
            passed=len(subset)==len(policy['methods']) and same and delta_ship<1e-7 and delta_ff<1e-7 and all(ncommon)))
    a=next((r for r in group_rows if r['sea']==5),None);b=next((r for r in group_rows if r['sea']==6 and a and r['method']==a['method']),None)
    if a and b:
        ma=read_json(out/a['path']/'simulink/batch_model.json');mb=read_json(out/b['path']/'simulink/batch_model.json')
        same=all(a[k]==b[k] for k in ('wave_seed','wind_seed','sensor_seed','replacement'))
        phases=np.array_equal(ma['wave_component']['phase'],mb['wave_component']['phase'])
        audits.append(dict(group=a['group'],sea='5_vs_6',passed=same and phases,same_seeds=same,same_phases=phases))
    return audits


def physical_audit(m,model,policy):
    expected=seeds(policy['master_seed'],m['weather'],m['slot'],m['replacement'])
    condition=policy['seas'].get(str(m['sea']),policy['seas'].get(m['sea']))
    checks=dict(seeds=all(m[k]==expected[k] for k in expected),
        wind=model.get('wind_mps')==condition['wind_mps'],wind_seed=model.get('wind_seed')==m['wind_seed'],
        waves=model.get('wave_parameters',{}).get('SignificantWaveHeight')==condition['wave_height_m'],
        wave_seed=model.get('wave_parameters',{}).get('RandomSeed')==m['wave_seed'],
        actual_taps=model.get('evaluation',{}).get('profile')=='thesis_523_15')
    if 'prediction_window_s' in condition:
        checks['prediction_window']=model.get('prediction_window',{}).get('lead_interval_s')==condition['prediction_window_s']
    return dict(case=m['path'],checks=checks,passed=all(checks.values()))


def analyze(out):
    out=Path(out);manifest=read_json(out/'manifest.json');policy=manifest['policy'];validate_policy(policy)
    data=out/'analysis';data.mkdir(exist_ok=True);(data/'plot_sources').mkdir(exist_ok=True)
    write_json(data/'status.json',dict(complete=False,analysis_complete=False))
    rows,attempted,ledger,audit,physical=[],[],[],[],[];evidence={'manifest.json':sha(out/'manifest.json')}
    landing=manifest['landing_snapshot'];geometry=dict(gear_body=landing['geometry']['gear_body_m'],H_ship_body=landing['geometry']['H_ship_body_m'])
    for group in manifest['groups']:
        directory=out/'groups'/group['id'];result=read_json(directory/'result.json') if (directory/'result.json').exists() else None
        selected=[]
        for folder in sorted(directory.glob('attempt_*')):
            attempt=int(folder.name.split('_')[-1]);attempt_status=read_json(folder/'attempt.json') if (folder/'attempt.json').exists() else {}
            status='included' if result and attempt==result['attempt'] else attempt_status.get('status','incomplete')
            for case in sorted(folder.glob('sea*')):
                if not case.is_dir():continue
                inclusion='interrupted' if '_interrupted_' in case.name else status
                rel=str(case.relative_to(out));error=read_json(case/'worker_error.json') if (case/'worker_error.json').exists() else None
                ledger.append(dict(case=rel,group=group['id'],attempt=attempt,status=inclusion,error=error))
                if not (case/'metrics.json').exists():continue
                # Recompute offline to keep the report reproducible without mutating simulation evidence.
                m=summarize(case,policy);m.update(path=rel,inclusion=inclusion);attempted.append(m)
                if inclusion!='included':continue
                if rel not in result['cases']:raise ValueError('Included case is not in frozen group result')
                if (m['group']!=group['id'] or m['weather']!=group['weather'] or m['slot']!=group['slot']
                        or m['replacement']!=attempt or case.name!=f"sea{m['sea']}_{m['method']}"
                        or m['method'] not in policy['methods'] or m['sea'] not in conditions(policy)):
                    raise ValueError(f'Run identity differs from frozen paired group: {rel}')
                if read_json(case/'run.json')['config']!=read_json(case/'requested.json'):
                    raise ValueError(f'Runtime configuration differs from request: {rel}')
                rows.append(m);selected.append(m);model=read_json(case/'simulink/batch_model.json')
                physical.append(physical_audit(m,model,policy))
                for name in ('run.json','requested.json','series.jsonl.gz','simulink/batch_model.json'):
                    evidence[rel+'/'+name]=sha(case/name)
                if m['slot']==policy['representative_slot']:
                    source=f"{group['id']}_sea{m['sea']}_{m['method']}.npz"
                    np.savez_compressed(data/'plot_sources'/source,**trace(read_series(case),model,geometry))
                    m['plot_source']='plot_sources/'+source
        if selected:audit.extend(audit_group(out,selected,policy))
    csv_file(data/'attempt_ledger.csv',ledger);write_json(data/'attempt_ledger.json',ledger)
    csv_file(data/'sorties.csv',rows);write_json(data/'sorties.json',rows)
    csv_file(data/'all_attempts.csv',attempted)
    csv_file(data/'table_5_16_case_evidence.csv',[dict(case=r['path'],weather=r['weather'],sea=r['sea'],method=r['method'],
        inclusion=r['inclusion'],termination=r['termination'],stop_input=r['stop_input'],ei=r['ei'],ei_terminal=r['ei_terminal'],
        first_anomaly=r['first_anomaly'],last_valid_s=r['last_valid_s'],stage=r['terminal_stage'],terminal_truth=r['terminal_truth'],
        events=r['accompanied_events'],runaway=r['runaway_evidence'],cause='unverified') for r in attempted])
    sensitivity=[]
    for weather in ['equal_weather',*policy['weather']]:
        for sea in conditions(policy):
            for method in policy['methods']:
                group=selection(attempted,weather,sea,method)
                sensitivity.append(dict(weather=weather,sea=sea,method=method,attempted=len(group),
                    touchdowns=sum(r['candidate'] for r in group),safe=sum(r['safe'] for r in group),
                    runaways=sum(r['termination']=='runaway' for r in group),
                    note='descriptive only: all completed attempts including excluded groups; not IID'))
    csv_file(data/'all_attempts_sensitivity.csv',sensitivity)
    t15,t16,t17=tables(rows,policy,data)
    contrasts=[((s,'C3'),(s,m)) for s in conditions(policy) for m in ('C1','C2') if m in policy['methods'] and 'C3' in policy['methods']]
    contrasts += [((6,m),(5,m)) for m in policy['methods'] if {5,6} <= set(conditions(policy))]
    comparisons=[paired(rows,l,r,metric,w,policy) for w in ['equal_weather',*policy['weather']] for l,r in contrasts
        for metric in ('safe','candidate','ei','radial_error_m','duration_s','touchdown_duration_s','safe_duration_s','Jctrl')]
    csv_file(data/'figure_5_16_paired.csv',comparisons);write_json(data/'figure_5_16_paired.json',comparisons)
    csv_file(data/'figure_5_14_phases.csv',[dict(case=r['path'],weather=r['weather'],sea=r['sea'],method=r['method'],termination=r['termination'],**p) for r in rows for p in r['phases']])
    touches=[]
    for m in rows:
        if not m['candidate']:continue
        region=allowed_region(m['terminal_truth']['db'],landing,geometry)
        touches.append(dict(case=m['path'],weather=m['weather'],sea=m['sea'],method=m['method'],slot=m['slot'],
            x=m['touchdown_position'][0],y=m['touchdown_position'][1],z=m['touchdown_position'][2],
            nominal=m['nominal_gear_point'],ei=m['ei'],outcome=m['touchdown_outcome'],
            error_x=m['error_x_m'],error_y=m['error_y_m'],radial_error=m['radial_error_m'],allowed_region=region))
    csv_file(data/'figure_5_13_touchdowns.csv',touches);write_json(data/'figure_5_13_touchdowns.json',touches)
    csv_file(data/'figure_5_17_energy.csv',[dict(case=r['path'],weather=r['weather'],sea=r['sea'],method=r['method'],
        touchdown=r['candidate'],ei=r['ei'],outcome=r['touchdown_outcome'],state=r['ei_terminal']) for r in rows])
    csv_file(data/'figure_5_11_motion_metrics.csv',[dict(case=r['path'],weather=r['weather'],sea=r['sea'],method=r['method'],
        duration_s=r['duration_s'],rms=r['motion_rms'],peak=r['motion_peak'],ei=r['motion_ei']) for r in rows])
    forcing_method='C0' if 'C0' in policy['methods'] else policy['methods'][0]
    forcing_summary=[]
    for sea in conditions(policy):
        group=[r for r in rows if r['sea']==sea and r['method']==forcing_method]
        representative=next((r for r in group if r['weather']==policy['weather'][0] and r['slot']==policy['representative_slot']),None)
        for j,quantity in enumerate(('wave_m','roll_deg','pitch_deg','H_up_displacement_m','H_normal_speed_mps','actual_EI')):
            for statistic in ('rms','peak'):
                def value(r):return r['motion_'+statistic][j] if j<5 else r['motion_ei'][statistic]
                values=[value(r) for r in group if value(r) is not None]
                rep=value(representative) if representative else None
                forcing_summary.append(dict(sea=sea,method=forcing_method,quantity=quantity,statistic=statistic,n=len(values),
                    minimum=min(values) if values else None,maximum=max(values) if values else None,
                    median=float(np.median(values)) if values else None,representative_value=rep,
                    representative_percentile=float(np.mean(np.array(values)<=rep)) if values and rep is not None else None,
                    duration_note='actual task duration per realization; differing stop times are retained in per-case data'))
    csv_file(data/'figure_5_11_coverage.csv',forcing_summary);write_json(data/'figure_5_11_coverage.json',forcing_summary)
    csv_file(data/'command_time_audit.csv',[dict(case=r['path'],**r['command_time_audit']) for r in rows])
    write_json(data/'pairing_audit.json',audit);write_json(data/'physical_conditions_audit.json',physical)
    from .thesis_figures import figures
    images=figures(rows,policy,data,t15,comparisons,landing,geometry)
    expected=len(manifest['groups'])*len(conditions(policy))*len(policy['methods']);complete=len(rows)==expected
    size=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    normalized=[(out/r['path']/'series.jsonl.gz').stat().st_size/max(r['duration_s'],.02)*policy['duration_s'] for r in rows]
    peak_rss=peak_gpu=0.
    if (out/'resources.jsonl').exists():
        for line in (out/'resources.jsonl').read_text().splitlines():
            sample=json.loads(line);processes={}
            for line in sample.get('process_rss_kib','').splitlines()[1:]:
                f=line.split(None,3)
                if len(f)==4:
                    try:processes[int(f[0])]=(int(f[1]),int(f[2]))
                    except ValueError:pass
            members=set(sample.get('workers',[])+sample.get('matlab_pids',[]))
            for _ in range(12):members.update(pid for pid,(ppid,_) in processes.items() if ppid in members)
            peak_rss=max(peak_rss,sum(rss for pid,(_,rss) in processes.items() if pid in members)/1024**2)
            for line in sample.get('gpu_memory_mib_utilization_percent','').splitlines():
                try:peak_gpu=max(peak_gpu,float(line.split(',')[0])/1024)
                except ValueError:pass
    report=dict(complete=complete,analysis_complete=True,pilot=manifest['pilot'],experiment=manifest.get('experiment'),
        included=len(rows),expected=expected,finalized_groups=sum((out/'groups'/g['id']/'result.json').exists() for g in manifest['groups']),
        touchdowns=sum(r['candidate'] for r in rows),safe=sum(r['safe'] for r in rows),
        unsafe=sum(r['termination']=='unsafe_touchdown' for r in rows),ei_unknown=sum(r['termination']=='touchdown_ei_unknown' for r in rows),
        excluded_attempts=sum(x['status']=='excluded_runaway' for x in ledger),interrupted_attempts=sum(x['status']=='interrupted' or x['error'] is not None for x in ledger),
        pairing_passed=bool(audit) and all(x['passed'] for x in audit),physical_conditions_passed=bool(physical) and all(x['passed'] for x in physical),
        image_count=len(images),stored_gib=size/1024**3,
        peak_batch_process_tree_rss_gib=peak_rss,peak_total_gpu_memory_gib=peak_gpu,
        rss_note='Sum of process RSS including dedicated shared MATLAB; shared pages can be counted repeatedly; sampled every 30 s',
        estimated_planned_series_gib=float(np.median(normalized)*expected/1024**3) if normalized else None,
        estimated_320x80_series_gib=float(np.median(normalized)*320*80/policy['duration_s']/1024**3) if normalized else None)
    write_json(data/'status.json',report)
    write_json(data/'analysis_provenance.json',dict(input_sha256=evidence,policy=policy,
        analysis_sources={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
        source_verification=read_json(out/'source_verification.json') if (out/'source_verification.json').exists() else None,
        simulation_records_unchanged=True,images=images))
    lines=['# 闭环批量分析',
        f"纳入 {len(rows)}/{expected} 架次，触舰 {report['touchdowns']}，安全触舰 {report['safe']}，非安全触舰 {report['unsafe']}，EI 未知 {report['ei_unknown']}。完整批次：{complete}；试运行：{manifest['pilot']}。",
        '；'.join(f"条件{k}：平均风速{v['wind_mps']:g} m/s、有效波高{v['wave_height_m']:g} m" for k,v in policy['seas'].items())
        +f"。每架次上限{policy['duration_s']:g} s。"+('本批次只有一种工况和一种方法，无配对差值。' if len(conditions(policy))==len(policy['methods'])==1 else ''),
        '原着舰 STOP 输入触发且仿真终止即为触舰。安全分类另按论文公式，以实际舰船动力学的SI状态及导数刚体换算至固定H，EI<1为安全，EI≥1为非安全；缺测为未知，未触舰无落点、无触舰EI。',
        '表5-15安全触舰率以全部纳入架次为分母；触舰发生率、EI完整触舰样本中的安全比例单列。天气等权条件统计遇缺失天气留空。Wilson仅对平衡样本给出，配对区间按天气分层重采样随机种子；小样本退化区间不代表精确。',
        '阶段完成时长与未完成停留分开；未进入不记零。图5-14的总时长仅包含已触舰架次。全任务、已触舰、安全子集总时长均保存在表5-17。',
        '满足预声明的持续飞离判据时，整组海况×方法共同更换种子，避免破坏配对。超时、非安全触舰和导航失效不换种子。排除带来选择偏差，须结合all_attempts_sensitivity和attempt_ledger阅读主结果。基础设施失败暂停批次，恢复时重跑相同种子。',
        f'图5-11至5-17输出{len(images)}张PNG；代表种子固定slot 0，不按结果择优。绘图源NPZ保留代表曲线完整50Hz精度；全部架次的series.jsonl.gz保留原始诊断。图5-11载荷仅覆盖实际任务时段，不外推停止后的波动。',
        '图5-15原始导航误差在t时刻对齐；控制实际消费量与t1真值对齐，并另存其相对t真值的误差。重复终止时间戳不参与指令差分。95%边界指原atan渐近幅值，不能解释为执行机构硬饱和。甲板允许区仅作示意，不改变STOP判据。',
        '轨迹法向高度是起落架相对H的甲板坐标；原STOP监视原模型自身信号，二者不是同一判据。视觉海面仍为既有程序化渲染，不等于动力学随机波面。',
        f"配对审计：{report['pairing_passed']}；工况审计：{report['physical_conditions_passed']}；当前目录 {report['stored_gib']:.3f} GiB。",
        f"按已完成任务的压缩字节/仿真秒外推，本次计划序列约 {report['estimated_planned_series_gib']} GiB；320×80 s约 {report['estimated_320x80_series_gib']} GiB，另需活动缓存和替换组。",'## 天气等权结果']
    for r in t15:
        if r['weather']=='equal_weather':lines.append(f"- 条件{r['sea']} {r['method']}：触舰 {r['touchdowns']}/{r['total']}；安全 {r['safe']}/{r['total']}，Wilson95% [{r['safe_wilson_low']}, {r['safe_wilson_high']}]；EI未知 {r['ei_unknown']}；触舰径向误差中位数 {r['all_touchdowns_radial_error_m']['median']} m。")
    lines+=['## 激励覆盖核查','以下按每组一个参考方法取样，不重复计入四种方法；各实际任务时段不同，不能直接把RMS差异解释为稳态谱强度差异。']
    for r in forcing_summary:
        if r['statistic']=='peak':lines.append(f"- 条件{r['sea']}，{r['quantity']}峰值范围 [{r['minimum']}, {r['maximum']}]，代表值 {r['representative_value']}，经验分位 {r['representative_percentile']}，独立组数 {r['n']}。")
    lines+=['', '时间先后不等于因果，异常证据见table_5_16_case_evidence。方法比较仅限本批次实际安排的配对。全部尝试、缺失配对数和不可判定样本均保留，未将它们绘制成零值。']
    (data/'report.md').write_text('\n\n'.join(lines)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    if rows and not(report['pairing_passed'] and report['physical_conditions_passed']):
        raise RuntimeError('Physical condition or paired-forcing audit failed; inspect analysis/*audit.json')
    return report
