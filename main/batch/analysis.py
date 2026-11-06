"""Tables 5-15..17, figures 5-11..16 and auditable source data."""
import csv
import json
from pathlib import Path
import numpy as np
import yaml
from .common import read_json, write_json, METHODS
from .metrics import TERMINALS, read_series, arrays, motion, allowed_region
from .statistics import describe, wilson, paired


def csv_file(path, rows, fields=None):
    rows=list(rows)
    fields=fields or list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for r in rows:
            writer.writerow({k: json.dumps(v,ensure_ascii=False,allow_nan=False) if isinstance(v,(list,dict)) else v for k,v in r.items()})


def selection(rows,weather,sea,method):
    return [r for r in rows if r["sea"]==sea and r["method"]==method and (weather=="equal_weather" or r["weather"]==weather)]


def balanced_description(rows,key,weathers):
    groups=[[r[key] for r in rows if r["weather"]==w and r.get(key) is not None] for w in weathers]
    out=describe([x for group in groups for x in group])
    out["n_by_weather"]={w:len(g) for w,g in zip(weathers,groups)}
    if not all(groups):
        for k in ("mean","median","q25","q75","std"): out[k]=None
        return out
    v=np.concatenate(groups); weights=np.concatenate([np.full(len(g),1/len(g)/len(groups)) for g in groups])
    idx=np.argsort(v); x=v[idx]; cumulative=np.cumsum(weights[idx])
    out.update(mean=float(weights@v), std=float(np.sqrt(weights@((v-weights@v)**2))))
    for name,q in (("q25",.25),("median",.5),("q75",.75)):
        out[name]=float(x[min(np.searchsorted(cumulative,q),len(x)-1)])
    return out


def tables(rows,policy,data):
    t15,t16,t17=[],[],[]
    for weather in ["equal_weather",*policy["weather"]]:
        for sea in (5,6):
            for method in METHODS:
                group=selection(rows,weather,sea,method); n=len(group); k=sum(r["candidate"] for r in group)
                strata=policy["weather"] if weather=="equal_weather" else [weather]
                counts=[sum(r["weather"]==w for r in group) for w in strata]
                balanced=len(set(counts))==1 and all(counts)
                rate=float(np.mean([np.mean([r["candidate"] for r in group if r["weather"]==w]) for w in strata])) if all(counts) else None
                low,high=wilson(k,n) if balanced else [None,None]
                base=dict(weather=weather,sea=sea,method=method,total=n,candidates=k,qualified=k,
                    no_candidate=n-k,rate=rate,wilson_low=low,wilson_high=high,
                    rate_interval_policy="Wilson on balanced independent sorties" if balanced else "incomplete/unbalanced; interval withheld")
                row=dict(base)
                for subset in ("all_candidates","qualified"):
                    for metric in ("error_x_m","error_y_m","radial_error_m","horizontal_speed_mps","normal_speed_mps"):
                        row[f"{subset}_{metric}"]=balanced_description([r for r in group if r["candidate"]],metric,strata)
                for metric in ("Jctrl","duration_s","invalid_wait_s"):
                    row[metric]=balanced_description(group,metric,strata)
                row["candidate_duration_s"]=balanced_description([r for r in group if r["candidate"]],"duration_s",strata)
                t15.append(row)
                for category in TERMINALS:
                    cases=[r for r in group if r["termination"]==category]
                    t16.append(dict(weather=weather,sea=sea,method=method,category=category,count=len(cases),total=n,
                        proportion=len(cases)/n if n else None,case_ids=[r["path"] for r in cases],
                        applicable=category in ("candidate_stop_proxy","timeout","interface_numerical","runaway"),
                        note="user STOP proxy: no separate unqualified/geometry/nav-abort thresholds"))
                for stage in range(3):
                    durations=[dict(weather=r["weather"],value=r["phases"][stage]["completed_duration_s"]) for r in group]
                    entered=sum(r["phases"][stage]["entered"] for r in group)
                    completed=sum(r["phases"][stage]["completed"] for r in group)
                    t17.append(dict(weather=weather,sea=sea,method=method,stage=stage,total=n,
                        entered=entered,completed=completed,unentered=n-entered,unfinished=entered-completed,
                        completed_duration_s=balanced_description(durations,"value",strata),
                        censored_dwell_s=describe([r["phases"][stage]["dwell_s"] for r in group if not r["phases"][stage]["completed"]]),
                        waiting_s=describe([r["phases"][stage]["invalid_wait_s"] for r in group]),
                        regressions=sum(r["regressions"] for r in group),
                        unfinished_cases=[dict(case=r["path"],termination=r["termination"],dwell_s=r["phases"][stage]["dwell_s"]) for r in group if r["phases"][stage]["entered"] and not r["phases"][stage]["completed"]],
                        all_task_duration_s=balanced_description(group,"duration_s",strata),
                        candidate_task_duration_s=balanced_description([r for r in group if r["candidate"]],"duration_s",strata)))
    for name,table in (("table_5_15",t15),("table_5_16",t16),("table_5_17",t17)):
        csv_file(data/f"{name}.csv",table); write_json(data/f"{name}.json",table)
    return t15,t16,t17


def pairing_audit(out,rows):
    audit=[]
    for group in sorted(set(r["group"] for r in rows)):
        cases=[r for r in rows if r["group"]==group]
        for sea in (5,6):
            subset=[r for r in cases if r["sea"]==sea]
            if len(subset)!=4: continue
            loaded=[read_series(out/r["path"]) for r in subset]
            n=min(map(len,loaded))
            # Ship motion / FF must agree at common time samples across methods.
            ff=[arrays(x)["feedforward"][:n] for x in loaded]
            ships=[np.array([v["s"]["ship"] for v in x[:n]]) for x in loaded]
            value=dict(group=group,sea=sea,common_samples=n,
                feedforward_max_difference_mps=max(float(np.max(abs(x-ff[0]))) for x in ff),
                ship_state_max_difference=max(float(np.max(abs(x-ships[0]))) for x in ships),
                same_seeds=len({(r["wave_seed"],r["wind_seed"],r["sensor_seed"]) for r in subset})==1)
            value["passed"]=value["same_seeds"] and value["feedforward_max_difference_mps"]<1e-7 and value["ship_state_max_difference"]<1e-7
            audit.append(value)
        sea5=[r for r in cases if r["sea"]==5 and r["method"]=="C0"]
        sea6=[r for r in cases if r["sea"]==6 and r["method"]=="C0"]
        if sea5 and sea6:
            a=read_json(out/sea5[0]["path"]/"simulink/batch_model.json")
            b=read_json(out/sea6[0]["path"]/"simulink/batch_model.json")
            same_seeds=all(sea5[0][k]==sea6[0][k] for k in ("wave_seed","wind_seed","sensor_seed"))
            audit.append(dict(group=group,sea="5_vs_6",passed=same_seeds and bool(np.allclose(a["wave_component"]["phase"],b["wave_component"]["phase"],rtol=0,atol=0)),
                              same_seeds=same_seeds))
    return audit


def analyze(out):
    out=Path(out); manifest=read_json(out/"manifest.json"); policy=manifest["policy"]
    data=out/"analysis"; data.mkdir(exist_ok=True)
    rows,attempted,ledger=[],[],[]
    for group in manifest["groups"]:
        directory=out/"groups"/group["id"]
        chosen=read_json(directory/"result.json")["attempt"] if (directory/"result.json").exists() else None
        for attempt_path in sorted(directory.glob("attempt_*")):
            if not attempt_path.is_dir(): continue
            attempt=int(attempt_path.name.split("_")[-1])
            status="included" if attempt==chosen else "excluded_runaway" if (attempt_path/"attempt.json").exists() else "incomplete"
            for case in sorted(attempt_path.glob("sea*")):
                if not case.is_dir(): continue
                interrupted="_interrupted_" in case.name
                if (case/"metrics.json").exists():
                    r=read_json(case/"metrics.json"); r["path"]=str(case.relative_to(out))
                    r["inclusion"]=status if not interrupted else "interrupted"
                    attempted.append(r)
                    if r["inclusion"]=="included": rows.append(r)
                if (case/"requested.json").exists():
                    ledger.append(dict(case=str(case.relative_to(out)),group=group["id"],attempt=attempt,
                        status="interrupted" if interrupted else status,
                        error=read_json(case/"worker_error.json") if (case/"worker_error.json").exists() else None,
                        result=read_json(case/"run.json").get("termination_reason") if (case/"run.json").exists() else None))
    csv_file(data/"attempt_ledger.csv",ledger);write_json(data/"attempt_ledger.json",ledger)
    csv_file(data/"sorties.csv",rows);write_json(data/"sorties.json",rows)
    csv_file(data/"table_5_16_case_evidence.csv",[dict(case=r["path"],weather=r["weather"],sea=r["sea"],
        method=r["method"],inclusion=r["inclusion"],termination=r["termination"],
        first_anomaly=r["first_anomaly"],last_valid_s=r["last_valid_s"],stage=r["terminal_stage"],
        terminal_truth=r["terminal_truth"],accompanied_events=r["accompanied_events"],
        runaway_evidence=r["runaway_evidence"],cause="unverified; use matched-seed single-factor reruns") for r in attempted])
    # Descriptive sensitivity only: replacements are selected, not iid trials.
    sensitivity=[]
    for weather in ["equal_weather",*policy["weather"]]:
        for sea in (5,6):
            for method in METHODS:
                a=selection(attempted,weather,sea,method)
                sensitivity.append(dict(weather=weather,sea=sea,method=method,attempted=len(a),
                    candidates=sum(r["candidate"] for r in a),runaways=sum(r["termination"]=="runaway" for r in a),
                    note="all completed attempts, including superseded groups; descriptive, selection-dependent"))
    csv_file(data/"all_attempts_sensitivity.csv",sensitivity)
    if not rows:
        write_json(data/"status.json",dict(complete=False,included=0,attempts=len(ledger)))
        (data/"report.md").write_text("没有完整纳入的配对组；请检查 attempt_ledger，不报告成功率。\n")
        return
    t15,t16,t17=tables(rows,policy,data)
    comparisons=[]
    for weather in ["equal_weather",*policy["weather"]]:
        pairs=[((s,"C3"),(s,m)) for s in (5,6) for m in ("C1","C2")]+[((6,m),(5,m)) for m in METHODS]
        for left,right in pairs:
            for metric in ("candidate","radial_error_m","duration_s","successful_duration_s","Jctrl"):
                comparisons.append(paired(rows,left,right,metric,weather,policy))
    csv_file(data/"figure_5_16_paired.csv",comparisons);write_json(data/"figure_5_16_paired.json",comparisons)
    audit=pairing_audit(out,rows);write_json(data/"pairing_audit.json",audit)
    from .figures import figures
    figures(out,rows,policy,data,t15,t17,comparisons)
    expected=len(manifest["groups"])*8; complete=len(rows)==expected
    peak=0;gpu_peak=0
    if (out/"resources.jsonl").exists():
        for line in (out/"resources.jsonl").read_text().splitlines():
            sample=json.loads(line)
            # Compute batch process trees only; total-machine utilization is
            # retained separately and must not be mistaken for batch memory.
            proc={}
            for row in sample.get("process_rss_kib","").splitlines()[1:]:
                v=row.split(None,3)
                if len(v)==4:
                    try: proc[int(v[0])]=(int(v[1]),int(v[2]))
                    except ValueError: pass
            members=set(sample["workers"])
            for _ in range(10): members.update(pid for pid,(parent,_) in proc.items() if parent in members)
            peak=max(peak,sum(rss for pid,(_,rss) in proc.items() if pid in members)/1024**2)
            for line in sample.get("gpu_memory_mib_utilization_percent","").splitlines():
                try: gpu_peak=max(gpu_peak,float(line.split(",")[0])/1024)
                except ValueError: pass
    storage=sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    series_bytes=sum((out/r["path"]/"series.jsonl.gz").stat().st_size for r in rows)
    # Normalize short pilot records by physical duration; full runs are usually
    # <=60s. Keep this separate from active MATLAB caches and environment space.
    per60=float(np.median([(out/r["path"]/"series.jsonl.gz").stat().st_size/max(r["duration_s"],.02)*60 for r in rows]))
    report=dict(complete=complete,pilot=manifest["pilot"],included=len(rows),expected=expected,
        source_verification=read_json(out/"source_verification.json") if (out/"source_verification.json").exists() else None,
        candidate_count=sum(r["candidate"] for r in rows),attempts=len(ledger),
        excluded_attempts=sum(x["status"]=="excluded_runaway" for x in ledger),
        interrupted_attempts=sum(x["status"]=="interrupted" or x["error"] is not None for x in ledger),
        pairing_passed=all(a["passed"] for a in audit),peak_batch_process_tree_rss_gib=peak,
        peak_total_gpu_memory_gib=gpu_peak,stored_bytes=storage,series_bytes=series_bytes,
        estimated_320x60s_series_gib=per60*320/1024**3)
    write_json(data/"status.json",report)
    lines=[f"# 闭环批量结果（{'预试验，不纳入正式统计' if manifest['pilot'] else '正式批次'}）", "",
        f"完整配对组纳入 {len(rows)}/{expected} 架次，STOP 触舰候选 {report['candidate_count']} 架次。完整性：{complete}。",
        f"排除并替换 {report['excluded_attempts']} 架次；中断/接口异常 {report['interrupted_attempts']} 次。参见 attempt_ledger.csv 和 all_attempts_sensitivity.csv。",
        "采用用户指定的原模型自动 STOP 代理判据，全部候选点即合格子集；没有新增接触位置、速度、姿态门限，不声称稳定着舰。未 STOP 的末帧不是触舰点。",
        "两级海况同时改变有效波高与平均风速（3.25m/12m·s⁻¹，5.00m/19m·s⁻¹），因此差异是风浪联合变化效应。初值保留已验证模型名义状态，未杜撰初始扰动分布。",
        "波面按实际 tauAndFDI 频率、相位、相遇速度重建；视觉海面仍是现有程序化海面，海况标签一致但不是相同随机波场。",
        "甲板绘制为现有渲染几何24×14m；允许区由暂定轮胎椭球投影侵蚀获得，仅作几何参考，不参与当前STOP判定。名义起落架点按每架次终端机姿换算。",
        "阶段停留只按实际状态0/1/2累计；未进入为缺失，已进入未完成单独删失；导航无效等待与悬停有效停留分别记录，后者不直接归因为海况窗口等待。",
        "指令变化代价以原atan制导渐近幅值归一化，W=I；近边界为95%渐近值，不能解释为执行机构硬饱和。",
        "表中条件分布采用天气等权经验分位数；若某天气无候选点，等权条件汇总为空。满足率Wilson区间只对天气样本数平衡的汇总给出。",
        "成对自助法按天气分层、种子配对；落点误差和成功总时长仅使用双方均有候选点的配对。另保留全部架次总时长，不能把失败早停解释为更快。",
        f"种子、舰船运动和前馈一致性检查：{report['pairing_passed']}（pairing_audit.json）。",
        f"实测批量进程树峰值RSS {peak:.2f} GiB，整卡峰值显存 {gpu_peak:.2f} GiB；当前目录 {storage/1024**3:.3f} GiB。",
        f"按本批每60s压缩序列外推320架次约 {report['estimated_320x60s_series_gib']:.2f} GiB，另需活动MATLAB缓存、图表和替换架次空间。", "",
        "## 结果与限制", ""]
    for r in t15:
        if r["weather"]=="equal_weather":
            lines.append(f"- 海况{r['sea']} {r['method']}：候选 {r['candidates']}/{r['total']}；Wilson95% [{r['wilson_low']}, {r['wilson_high']}]；径向误差中位数 {r['all_candidates_radial_error_m']['median']} m。")
    lines += ["", "配对差值及区间见 figure_5_16_paired.csv。区间跨零或样本太少时不宣称优势；单对自助区间退化不代表精确。排除飞离会造成选择偏差，主结果仅适用于通过该筛选的载荷集合，必须同时检查全部尝试的描述性敏感性统计。",
        "失败案例参见 sorties.json 的 first_anomaly、last_valid_s、terminal_truth 与原始 diagnostics。视觉拒绝、导航失效和动力学异常的时间先后仅是证据链；建议固定相同种子分别重跑C0、C1、C2、C3核查，不能据相关性声称因果。",
        "", "所有PNG的原始数据为对应CSV及每架次series.jsonl.gz；可独立执行analyze重新生成。图5-15消费量记录其实际求值时刻t1，同时保留导航来源时刻t，避免混淆一个控制步的时间关系。"]
    (data/"report.md").write_text("\n\n".join(lines)+"\n")
    print(json.dumps(report,indent=2))
    if not report["pairing_passed"]: raise RuntimeError("Pairing/forcing/FF audit failed; results need investigation")
