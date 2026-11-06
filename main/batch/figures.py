"""Publication-source plots for thesis 5.2.3(10); no thesis document edits."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import yaml
from .common import METHODS, read_json, ROOT
from .metrics import read_series, arrays, motion, nominal_point, allowed_region
from .analysis import csv_file, selection

COLORS=dict(zip(METHODS,["#444444","#db7b23","#467ac0","#15946a"]))
PHASES=["Approach", "Hover", "Final descent"]


def save(fig, path, title):
    fig.suptitle(title,fontsize=12)
    fig.tight_layout(rect=[0,0,1,.96])
    fig.savefig(path,dpi=160)
    plt.close(fig)


def boxpoints(ax, groups, labels=METHODS, ylabel=None):
    for i,(values,label) in enumerate(zip(groups,labels)):
        v=np.array([x for x in values if x is not None],dtype=float)
        if len(v):
            ax.boxplot([v],positions=[i],widths=.45,showfliers=False)
            jitter=np.linspace(-.13,.13,len(v)) if len(v)>1 else np.zeros(1)
            ax.scatter(i+jitter,v,s=14,color=COLORS.get(label,"#467ac0"),alpha=.7)
        ax.text(i,.98,f"n={len(v)}",transform=ax.get_xaxis_transform(),ha="center",va="top",fontsize=7)
    ax.set_xticks(range(len(labels)),labels);ax.set_xlim(-.6,len(labels)-.4)
    if ylabel:ax.set_ylabel(ylabel)
    ax.grid(alpha=.2,axis="y")


def deck(ax,region):
    ax.add_patch(Rectangle((-12,-7),24,14,fill=False,color="black",lw=1,label="Deck 24x14 m"))
    low,high=region
    ax.add_patch(Rectangle(low,np.subtract(high,low)[0],np.subtract(high,low)[1],fill=False,color="gray",ls="--",label="Provisional level-wheel region"))
    ax.plot(0,0,"k+",ms=9,label="H");ax.set_aspect("equal",adjustable="box")
    ax.set_xlabel("Bow x / m");ax.set_ylabel("Port y / m")


def events(ax,rows,color):
    for r in rows:
        if r["stage"]!=int(r["u"]["stage"]):ax.axvline(r["t1"],color=color,ls=":",alpha=.3,lw=.7)
    ax.axvline(rows[-1]["t1"],color=color,ls="--",alpha=.45,lw=.7)


def figures(out,metrics,policy,data,t15,t17,comparisons):
    plots=data/"figures";plots.mkdir(exist_ok=True)
    landing=yaml.safe_load((ROOT/read_json(out/"manifest.json")["base_config"]["landing_config"]).read_text())
    geometry=dict(gear_body=landing["geometry"]["gear_body_m"],H_ship_body=landing["geometry"]["H_ship_body_m"])
    level_region=allowed_region(np.diag([1,-1,-1]),landing,geometry)
    representatives=[];motion_points=[];touchdowns=[];phase_rows=[]
    # Every plotted trace retains full 50 Hz source in one compressed series.
    # Representative slot is preregistered. If excluded, use its whole-group
    # replacement, never search for the prettiest/successful trajectory.
    for m in metrics:
        case=out/m["path"]
        if m["slot"]==policy["representative_slot"]: representatives.append(m)
        for phase in m["phases"]:phase_rows.append(dict(case=m["path"],weather=m["weather"],sea=m["sea"],method=m["method"],**phase))
        if m["method"]=="C0":
            for i,name in enumerate(("wave_m","roll_deg","pitch_deg","H_up_displacement_m","H_normal_speed_mps")):
                motion_points.append(dict(case=m["path"],weather=m["weather"],sea=m["sea"],slot=m["slot"],quantity=name,rms=m["motion_rms"][i],peak=m["motion_peak"][i],duration_s=m["duration_s"]))
        if m["candidate"]:
            region=allowed_region(np.asarray(m["terminal_truth"]["db"]),landing,geometry)
            p=np.asarray(m["touchdown_position"])
            touchdowns.append(dict(case=m["path"],weather=m["weather"],sea=m["sea"],method=m["method"],
                slot=m["slot"],x=p[0],y=p[1],z=p[2],nominal=m["nominal_gear_point"],qualified=m["qualified"],
                error_x=m["error_x_m"],error_y=m["error_y_m"],radial_error=m["radial_error_m"],
                allowed_lower=region[0],allowed_upper=region[1],inside_provisional_region=bool(np.all(p[:2]>=region[0]) and np.all(p[:2]<=region[1]))))
    csv_file(data/"figure_5_11_motion_metrics.csv",motion_points)
    csv_file(data/"figure_5_13_touchdowns.csv",touchdowns)
    csv_file(data/"figure_5_14_phases.csv",phase_rows)
    csv_file(data/"representative_cases.csv",[dict(case=m["path"],weather=m["weather"],sea=m["sea"],method=m["method"],slot=m["slot"],replacement=m["replacement"]) for m in representatives])
    motion_source=[]
    for weather in policy["weather"]:
        print(f"Rendering figures 5-11..5-15: {weather}",flush=True)
        selected=[m for m in representatives if m["weather"]==weather]
        if not selected:continue
        loaded={(m["sea"],m["method"]):(m,read_series(out/m["path"]),read_json(out/m["path"]/"simulink/batch_model.json")) for m in selected}
        fig,ax=plt.subplots(1,5,figsize=(18,3.6))
        for sea in (5,6):
            item=loaded.get((sea,"C0"))
            if item is None:continue
            m,rows,model=item;v=motion(rows,model);t=np.array([r["t"] for r in rows]);speed=np.linalg.norm(m["wave_speed_mps"][:2])
            for j,label in enumerate(("Wave / m","Roll / deg","Pitch / deg","H up displacement / m","H normal speed / m/s")):
                ax[j].plot(t,v[:,j],"-" if sea==5 else "--",label=f"Sea {sea}");ax[j].set(xlabel="Time / s",ylabel=label);ax[j].grid(alpha=.2)
            for i,at in enumerate(t):motion_source.append(dict(weather=weather,sea=sea,time_s=float(at),wave_m=v[i,0],roll_deg=v[i,1],pitch_deg=v[i,2],H_up_m=v[i,3],H_normal_mps=v[i,4],wave_encounter_speed_mps=float(speed)))
        ax[0].legend(fontsize=8)
        save(fig,plots/f"figure_5_11_{weather}.png",f"5-11 {weather}: Hs=3.25/5.00m, Uwind=12/19m/s; Tp=11.2s, beta=150deg; encounter speed={speed:.3g}m/s")
        fig,ax=plt.subplots(2,5,figsize=(15,6))
        for j,name in enumerate(("wave_m","roll_deg","pitch_deg","H_up_displacement_m","H_normal_speed_mps")):
            for i,metric in enumerate(("rms","peak")):
                values=[[r[metric] for r in motion_points if r["weather"]==weather and r["sea"]==sea and r["quantity"]==name] for sea in (5,6)]
                boxpoints(ax[i,j],values,["Sea5","Sea6"],f"{name} {metric}")
        save(fig,plots/f"figure_5_11_{weather}_distribution.png",f"5-11 {weather}: one C0 input realization per sea/seed; actual-duration RMS / absolute peak")
        trajectory(plots,weather,loaded,level_region,geometry)
        touchdown_plot(plots,weather,metrics,level_region)
        phases_plot(plots,weather,metrics,loaded)
        for (sea,method),(m,rows,model) in loaded.items():
            control_plot(plots,weather,sea,method,rows,model)
    csv_file(data/"figure_5_11_motion_series.csv",motion_source)
    comparison_plot(plots,policy,t15,comparisons)


def trajectory(plots,weather,loaded,region,geometry):
    fig=plt.figure(figsize=(20,8));axes=[]
    for row,sea in enumerate((5,6)):
        aa=[fig.add_subplot(2,5,row*5+1,projection="3d")]+[fig.add_subplot(2,5,row*5+j+1) for j in range(1,5)];axes.append(aa)
        deck(aa[1],region)
        for lower,upper,style in (([-12,-7],[12,7],"-"),(region[0],region[1],"--")):
            x=[lower[0],upper[0],upper[0],lower[0],lower[0]]
            y=[lower[1],lower[1],upper[1],upper[1],lower[1]]
            aa[0].plot(x,y,[0]*5,color="gray",ls=style,lw=.7)
        for method in METHODS:
            item=loaded.get((sea,method))
            if item is None:continue
            m,rows,model=item;a=arrays(rows);t=np.r_[a["t"],rows[-1]["t1"]]
            p=np.vstack((a["p"],rows[-1]["s1"]["p"]));v=np.vstack((a["v"],rows[-1]["s1"]["v"]));c=COLORS[method]
            aa[0].plot(*p.T,color=c,label=method);aa[1].plot(p[:,0],p[:,1],color=c,label=method)
            target=nominal_point(rows[-1]["s1"],model,geometry)
            aa[1].plot(target[0],target[1],"+",color=c)
            aa[0].scatter(*target,marker="+",color=c,s=15)
            marker="o" if m["candidate"] else "x"
            aa[0].scatter(*p[-1],marker=marker,color=c,s=22);aa[1].plot(*p[-1,:2],marker,color=c)
            for j,y in enumerate((p[:,2],np.linalg.norm(v[:,:2],axis=1),v[:,2]),2):
                aa[j].plot(t,y,color=c,label=method);aa[j].plot(t[-1],y[-1],marker,color=c);events(aa[j],rows,c)
            boundaries=[i+1 for i,r in enumerate(rows) if r["stage"]!=int(r["u"]["stage"])]
            if boundaries:
                aa[0].scatter(*p[boundaries].T,marker="s",s=12,color=c);aa[1].scatter(*p[boundaries,:2].T,marker="s",s=12,color=c)
        aa[0].set(xlabel="Bow / m",ylabel="Port / m",zlabel="Deck normal / m",title=f"Sea {sea}")
        aa[0].scatter(0,0,0,marker="+",color="black");aa[0].legend(fontsize=7)
        for j,ylabel in enumerate(("Height / m","Horizontal rel speed / m/s","Normal rel speed / m/s"),2):aa[j].set(xlabel="Time / s",ylabel=ylabel);aa[j].grid(alpha=.2)
    for j in (1,2,3,4):
        limits=[ax[j].get_xlim() for ax in axes];ylimits=[ax[j].get_ylim() for ax in axes]
        for aa in axes:aa[j].set_xlim(min(x[0] for x in limits),max(x[1] for x in limits));aa[j].set_ylim(min(x[0] for x in ylimits),max(x[1] for x in ylimits))
    # Keep 3D comparison scales identical across the two seas as well.
    for getter,setter in (("get_xlim","set_xlim"),("get_ylim","set_ylim"),("get_zlim","set_zlim")):
        bounds=[getattr(aa[0],getter)() for aa in axes]
        for aa in axes:getattr(aa[0],setter)(min(x[0] for x in bounds),max(x[1] for x in bounds))
    save(fig,plots/f"figure_5_12_{weather}.png",f"5-12 {weather}: preregistered paired slot; squares=phase, circles=STOP candidates, crosses=other termination")


def touchdown_plot(plots,weather,metrics,region):
    fig,ax=plt.subplots(2,4,figsize=(15,8))
    positions=[r["touchdown_position"] for r in metrics if r["weather"]==weather and r["candidate"]]
    extent=np.max(abs(np.array(positions)[:,:2]),axis=0) if positions else [12,7]
    extent=np.maximum(extent,[12,7])+1
    for i,sea in enumerate((5,6)):
        for j,method in enumerate(METHODS):
            group=selection(metrics,weather,sea,method);candidates=[r for r in group if r["candidate"]];a=ax[i,j];deck(a,region)
            if candidates:
                points=np.array([r["touchdown_position"] for r in candidates]);target=np.array([r["nominal_gear_point"] for r in candidates])
                a.scatter(points[:,0],points[:,1],c=COLORS[method],s=20,label="STOP proxy")
                a.scatter(target[:,0],target[:,1],marker="+",s=15,c="gray",label="Attitude-converted nominal")
                a.plot(*points[:,:2].mean(axis=0),"D",color="red",ms=5,label="Candidate mean")
            a.set(xlim=(-extent[0],extent[0]),ylim=(-extent[1],extent[1]),title=f"Sea {sea} {method}: N={len(group)}, candidate={len(candidates)}\nqualified={len(candidates)}, no point={len(group)-len(candidates)}")
    ax[0,0].legend(fontsize=6)
    save(fig,plots/f"figure_5_13_{weather}.png",f"5-13 {weather}: H-centred tilted deck frame; all STOP candidates qualify under current proxy")
    fig,ax=plt.subplots(4,3,figsize=(13,12))
    for row,(sea,subset) in enumerate((s,k) for s in (5,6) for k in ("all_candidates","qualified")):
        for j,metric in enumerate(("error_x_m","error_y_m","radial_error_m")):
            boxpoints(ax[row,j],[[r[metric] for r in selection(metrics,weather,sea,m) if r["candidate"]] for m in METHODS],ylabel=f"Sea {sea} {subset}\n{metric}")
    save(fig,plots/f"figure_5_13_{weather}_errors.png",f"5-13 {weather}: error relative to attitude-converted nominal gear target; no terminal-frame substitution")


def phases_plot(plots,weather,metrics,loaded):
    fig,ax=plt.subplots(2,3,figsize=(13,7),sharey="col")
    for i,sea in enumerate((5,6)):
        for j in range(3):
            boxpoints(ax[i,j],[[r["phases"][j]["completed_duration_s"] for r in selection(metrics,weather,sea,m)] for m in METHODS],ylabel=f"Sea {sea} {PHASES[j]} / s")
    save(fig,plots/f"figure_5_14_{weather}.png",f"5-14 {weather}: completed phases only; unentered and censored dwell excluded")
    fig,ax=plt.subplots(2,4,figsize=(16,7))
    for i,sea in enumerate((5,6)):
        for j in range(3):
            base=np.zeros(4)
            for status,color in (("unentered","#aaaaaa"),("unfinished","#d59965"),("completed","#73a78f")):
                y=[]
                for method in METHODS:
                    group=selection(metrics,weather,sea,method)
                    count=sum((not r["phases"][j]["entered"]) if status=="unentered" else (r["phases"][j]["entered"] and not r["phases"][j]["completed"]) if status=="unfinished" else r["phases"][j]["completed"] for r in group)
                    y.append(count/len(group) if group else 0)
                ax[i,j].bar(METHODS,y,bottom=base,color=color,label=status);base+=y
            ax[i,j].set(title=f"Sea {sea} {PHASES[j]}",ylim=(0,1),ylabel="Fraction")
        boxpoints(ax[i,3],[[r["duration_s"] for r in selection(metrics,weather,sea,m) if r["candidate"]] for m in METHODS],ylabel=f"Sea {sea} candidate duration / s")
    ax[0,0].legend(fontsize=7)
    save(fig,plots/f"figure_5_14_{weather}_outcomes.png",f"5-14 {weather}: phase outcomes and candidate-only total durations")
    fig,ax=plt.subplots(2,1,figsize=(13,6),sharex=True)
    cmap=plt.get_cmap("viridis",3)
    for i,sea in enumerate((5,6)):
        for j,method in enumerate(METHODS):
            item=loaded.get((sea,method))
            if item is None:continue
            m,rows,_=item
            for r in rows:
                width=r["t1"]-r["t"]
                ax[i].broken_barh([(r["t"],width)],(j-.28,.56),facecolors=cmap(r["stage"]))
                if not bool(r["u"]["control_valid"]):ax[i].broken_barh([(r["t"],width)],(j-.28,.16),facecolors="red")
                if int(r["u"]["stage"])<r["stage"]:ax[i].plot(r["t1"],j,"v",color="red")
            ax[i].plot(rows[-1]["t1"],j,"o" if m["candidate"] else "x",color="black")
        ax[i].set(yticks=range(4),yticklabels=METHODS,ylabel=f"Sea {sea}",xlabel="Time / s")
    save(fig,plots/f"figure_5_14_{weather}_strips.png",f"5-14 {weather}: purple/green/yellow=approach/hover/descent; red=invalid control wait; triangles=regressions")


def control_plot(plots,weather,sea,method,rows,model):
    a=arrays(rows);t=a["t"];t1=a["t1"]
    true=np.array([r["cg_true"] for r in rows]);raw=np.array([r["cg_raw"] for r in rows]);used=np.array([r["u"]["cg_used"] for r in rows]).reshape(-1,3)
    error=np.array([r["u"]["error"] for r in rows]).reshape(-1,3)
    velocity=np.array([r["s1"]["velocity_ned_mps"] for r in rows])
    derivative=np.diff(a["command"],axis=0)/np.maximum(np.diff(t1),1e-12)[:,None]
    limits=np.asarray(model["command_limit_mps"]).ravel()
    windows=[("full",None),("final",(max(0,t1[-1]-4),t1[-1]))]
    recovering=np.flatnonzero(np.diff(a["nav_valid"].astype(int))==1)
    if len(recovering):
        center=t[recovering[0]+1];windows.append(("recovery",(max(0,center-1),min(t1[-1],center+2))))
    corrections=np.array([r["diagnostics"].get("vision_position_correction_m",0) or 0 for r in rows])
    if np.max(corrections)>0:
        center=t[np.argmax(corrections)];windows.append(("correction",(max(0,center-1),min(t1[-1],center+2))))
    for label,window in windows:
        fig,ax=plt.subplots(6,3,figsize=(15,13),sharex=True)
        for j,direction in enumerate("NED"):
            ax[0,j].plot(t,true[:,j],color="black",label="Truth at source t")
            ax[0,j].plot(t,raw[:,j],color="#d58a36",alpha=.8,label="Raw at source t")
            ax[0,j].plot(t1,used[:,j],color=COLORS[method],label="Consumed at evaluation t1")
            ax[0,j].set_ylabel(f"CG {direction} / m")
            ax[1,j].plot(t1,error[:,j]);ax[1,j].set_ylabel(f"Target-CG {direction} / m")
            ax[2,j].plot(t1,a["command"][:,j],label="Command");ax[2,j].plot(t1,velocity[:,j],label="Actual")
            ax[2,j].plot(t1,a["feedforward"][:,j],"--",label="Ship FF");ax[2,j].set_ylabel(f"Velocity {direction} / m/s")
            ax[3,j].plot(t1[1:],derivative[:,j]);ax[3,j].set_ylabel(f"d command {direction} / dt")
            ax[4,j].plot(t1,abs(a["relative"][:,j])/limits[j]);ax[4,j].axhline(.95,color="red",ls="--");ax[4,j].set_ylabel("Relative / atan bound")
            ax[5,j].step(t,a["nav_valid"],where="post",label="Nav valid")
            ax[5,j].step(t,a["shaper_valid"].astype(int)+1.2,where="post",label="Shadow shaper +1.2")
            ax[5,j].step(t1,a["control_valid"].astype(int)+2.4,where="post",label="Consumed gate +2.4")
            ax[5,j].step(t,a["stage"]+3.8,where="post",label="Phase +3.8");ax[5,j].set(xlabel="Time / s",ylabel="Flags / stage")
            for k in range(6):
                events(ax[k,j],rows,COLORS[method]);ax[k,j].grid(alpha=.15)
                ax[k,j].fill_between(t,0,1,where=~a["nav_valid"],transform=ax[k,j].get_xaxis_transform(),color="red",alpha=.07,step="post")
                ax[k,j].fill_between(t,0,1,where=~a["shaper_valid"],transform=ax[k,j].get_xaxis_transform(),color="blue",alpha=.07,step="post")
                if window and window[1]>window[0]:ax[k,j].set_xlim(*window)
        for k in (0,2,5):ax[k,0].legend(fontsize=6,loc="best")
        save(fig,plots/f"figure_5_15_{weather}_sea{sea}_{method}_{label}.png",f"5-15 {weather} sea{sea} {method}{' ORACLE truth feedback' if method=='C0' else ''}: {label}; red=nav invalid, blue=shaper invalid; 95% atan bound is not actuator saturation")


def comparison_plot(plots,policy,table,comparisons):
    weathers=[*policy["weather"],"equal_weather"]
    fig,axes=plt.subplots(1,len(weathers),figsize=(4*len(weathers),4),squeeze=False,sharey=True)
    for ax,w in zip(axes[0],weathers):
        for sea,offset in ((5,-.12),(6,.12)):
            for j,method in enumerate(METHODS):
                r=next(x for x in table if x["weather"]==w and x["sea"]==sea and x["method"]==method)
                if r["rate"] is None:continue
                lo,hi=r["wilson_low"],r["wilson_high"]
                err=None if lo is None else np.array([[r["rate"]-lo],[hi-r["rate"]]])
                ax.errorbar(j+offset,r["rate"],yerr=err,fmt="o" if sea==5 else "s",capsize=3,color=COLORS[method])
                ax.annotate(f"{r['candidates']}/{r['total']}",(j+offset,r["rate"]),xytext=(0,7 if sea==5 else -14),textcoords="offset points",fontsize=6,ha="center")
        ax.set(xticks=range(4),xticklabels=METHODS,ylim=(-.08,1.15),title=w,ylabel="STOP candidate rate");ax.grid(alpha=.2)
    save(fig,plots/"figure_5_16_rates.png","5-16 STOP proxy candidate rates and Wilson 95% intervals; circles=Sea5, squares=Sea6")
    for w in weathers:
        fig,ax=plt.subplots(1,5,figsize=(22,7),sharey=True)
        for j,metric in enumerate(("candidate","radial_error_m","duration_s","successful_duration_s","Jctrl")):
            rows=[r for r in comparisons if r["weather"]==w and r["metric"]==metric]
            for i,r in enumerate(rows):
                if r["mean"] is not None:
                    ax[j].errorbar(r["mean"],i,xerr=np.maximum(0,np.array([[r["mean"]-r["low"]],[r["high"]-r["mean"]]])),fmt="o",capsize=3)
                ax[j].text(.98,i,f"n={r['pairs']}"+ ("; missing strata" if r["mean"] is None else ""),transform=ax[j].get_yaxis_transform(),ha="right",fontsize=7)
            ax[j].axvline(0,color="gray",ls="--");ax[j].set(yticks=range(len(rows)),yticklabels=[r["left"]+" - "+r["right"] for r in rows],xlabel=metric,title=metric);ax[j].grid(alpha=.15)
        save(fig,plots/f"figure_5_16_{w}_paired.png",f"5-16 {w}: paired seed bootstrap 95%; radial error / successful duration require candidates in BOTH runs")
