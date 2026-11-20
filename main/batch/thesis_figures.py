"""Exactly 15 Chinese publication figures; all data retained in sibling files."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager, ft2font
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from .report_policy import conditions
from .common import write_json
from .metrics import allowed_region

WEATHER=dict(clear='晴天',fog='雾天',night='夜间',night_fog='夜雾')
COLORS=dict(C0='#373737',C1='#d16b14',C2='#426db2',C3='#087f68')
WC=['#2785ad','#dd7c2b','#7d5ba6','#35874b']
PHASES=['接近','悬停','最终下降']
OUTCOME=dict(safe_touchdown='安全触舰',unsafe_touchdown='非安全触舰',touchdown_ei_unknown='触舰EI未知',not_touched='未触舰')


def configure_font():
    # Ubuntu's fonts-noto-cjk TTC can expose the JP face first to FreeType;
    # it also contains the required Chinese glyphs. Prefer SC when available.
    preferred=('Noto Sans CJK SC','Source Han Sans SC','Songti SC','PingFang SC','Microsoft YaHei',
               'Noto Sans CJK JP','Noto Serif CJK SC','Noto Serif CJK JP','SimHei','Arial Unicode MS')
    for attempt in range(2):
        candidates=sorted(font_manager.fontManager.ttflist,key=lambda x:preferred.index(x.name) if x.name in preferred else 100)
        for entry in candidates:
            if entry.name not in preferred:continue
            try:
                charmap=ft2font.FT2Font(entry.fname).get_charmap()
                if all(ord(c) in charmap for c in '着舰触浪晴雾况'):
                    plt.rcParams.update({'font.family':entry.name,'axes.unicode_minus':False,'font.size':9,
                        'axes.labelsize':9,'axes.titlesize':10,'legend.fontsize':8,'savefig.dpi':180})
                    return dict(name=entry.name,path=entry.fname)
            except (RuntimeError,OSError):continue
        if attempt==0:
            # Refresh after apt installed fonts, without deleting user's cache.
            for filename in font_manager.findSystemFonts():
                try:font_manager.fontManager.addfont(filename)
                except (RuntimeError,OSError):pass
    raise RuntimeError('Chinese plotting font missing: Ubuntu: sudo apt-get install fonts-noto-cjk; then rerun check/analyze')


def short_label(sea):
    return f"{sea}级设定" if isinstance(sea,int) else "自定义风浪工况"


def label(sea,policy):
    c=policy['seas'].get(str(sea),policy['seas'].get(sea))
    return f"{short_label(sea)}：风速{c['wind_mps']:g} m/s，有效波高{c['wave_height_m']:g} m"


def save(fig,plots,name,title,footer=''):
    fig.suptitle(title,fontsize=13)
    if footer:fig.text(.5,.012,footer,ha='center',va='bottom',fontsize=8)
    fig.tight_layout(rect=(0,.055,1,.94),h_pad=2.2,w_pad=2.)
    fig.savefig(plots/name)
    plt.close(fig)


def empty(ax,text='无可用样本'):
    ax.text(.5,.5,text,ha='center',va='center',transform=ax.transAxes,color='#777777')


def deck(ax,region):
    ax.add_patch(Rectangle((-12,-7),24,14,fill=False,color='#555555',lw=1))
    lo,hi=np.array(region)
    ax.add_patch(Rectangle(lo,*(hi-lo),fill=False,ls='--',color='#888888',lw=.8))
    ax.plot(0,0,'k+',ms=9);ax.set_aspect('equal',adjustable='box')
    ax.set(xlabel='舰艏方向 / m',ylabel='左舷方向 / m');ax.grid(alpha=.15)


def marker(m):
    return 'o' if m['safe'] else 'x' if m['candidate'] and m['ei_complete'] else 'o' if m['candidate'] else 'v'


def endpoint(ax,x,y,m):
    color=COLORS[m['method']]
    ax.plot(x,y,marker=marker(m),mfc=color if m['safe'] else 'none',mec=color,ls='none',ms=6,alpha=.65)


def padded(values, fallback=(-1,1)):
    v=np.asarray(values).ravel()
    if not len(v):return fallback
    lo,hi=float(v.min()),float(v.max());pad=max((hi-lo)*.06,.02)
    return lo-pad,hi+pad


def events(ax,z,color,index=0):
    changes=np.flatnonzero(z['stage']!=z['next_stage'])
    for i in changes:
        ax.axvline(z['t1'][i],color=color,ls=':' if z['next_stage'][i]>z['stage'][i] else '-.',alpha=.45,lw=.8)
    # Invalid intervals shown as a narrow method-specific band on the x axis.
    bad=~z['valid'];start=None
    for i,flag in enumerate(np.r_[bad,False]):
        if flag and start is None:start=i
        if not flag and start is not None:
            ax.axvspan(z['t'][start],z['t1'][i-1],ymin=.015+index*.018,ymax=.03+index*.018,color=color,alpha=.65)
            start=None
    ax.axvline(z['t1'][-1],color=color,ls='--',alpha=.4,lw=.7)


def method_legend(ax,methods,extras=None):
    handles=[Line2D([],[],color=COLORS[m],label=m) for m in methods]
    if extras:handles+=extras
    ax.legend(handles=handles,loc='best',ncol=min(4,len(handles)))


def figures(rows,policy,data,t15,comparisons,landing,geometry):
    font=configure_font();write_json(data/'plot_font.json',font)
    plots=data/'figures';plots.mkdir(exist_ok=True)
    # A new-profile directory must not silently mix with legacy dozens-of-files plots.
    expected={'figure_5_11.png','figure_5_16_rates.png','figure_5_16_paired.png'}
    expected|={f'figure_5_{n}_sea{s}.png' for n in (12,13,14,15) for s in conditions(policy)}
    expected|={f'figure_5_17_{w}.png' for w in policy['weather']}
    extra={p.name for p in plots.glob('*.png')}-expected
    if extra:raise ValueError(f'Unexpected legacy PNGs in new-profile output: {sorted(extra)}; use a new output directory')
    region=allowed_region(np.diag([1,-1,-1]),landing,geometry)
    representatives=[m for m in rows if 'plot_source' in m]
    traces={}
    for m in representatives:
        with np.load(data/m['plot_source'],allow_pickle=False) as z:
            traces[(m['weather'],m['sea'],m['method'])]=(m,{k:z[k] for k in z.files})
    write_json(data/'representative_cases.json',[dict(case=m['path'],weather=m['weather'],sea=m['sea'],method=m['method'],
        slot=m['slot'],replacement=m['replacement'],source=m['plot_source'],outcome=m['touchdown_outcome']) for m in representatives])
    forcing(plots,traces,policy)
    for sea in conditions(policy):
        trajectories(plots,traces,policy,sea,region)
        touchdowns(plots,rows,policy,sea,region)
        phases(plots,rows,policy,sea)
        control(plots,traces,policy,sea)
    rates(plots,t15,policy)
    paired_plot(plots,comparisons,policy)
    for weather in policy['weather']:energy_distribution(plots,rows,policy,weather)
    found={p.name for p in plots.glob('*.png')}
    if found!=expected:raise AssertionError(f'Figure set mismatch: {found ^ expected}')
    index=dict(count=len(found),files=sorted(found),
        sources={'5-11':'plot_sources/*.npz (motion, ei); figure_5_11_motion_metrics.csv',
                 '5-12':'plot_sources/*.npz (p, v, stages); representative_cases.json',
                 '5-13':'figure_5_13_touchdowns.csv/json; table_5_15.json',
                 '5-14':'figure_5_14_phases.csv; sorties.json; table_5_17.json',
                 '5-15':'plot_sources/*.npz (raw/used errors, control, validity); sorties.json',
                 '5-16':'table_5_15.json; figure_5_16_paired.json',
                 '5-17':'figure_5_17_energy.csv; sorties.json'},
        units_note='p/v use deck bow-port-up; feedback/guidance/commands use NED; state_time and energy_time include STOP endpoint',
        event_note='dotted: forward phase transition; dash-dot: regression; dashed: termination; lower colored bands: invalid control')
    write_json(data/'figure_index.json',index)
    (data/'plot_sources/README.md').write_text('NPZ为完整采样精度的代表架次绘图源，使用numpy.load(..., allow_pickle=False)读取。\n'
        't为导航来源时刻；t1为控制实际求值时刻；state_time对应p/v；energy_time对应实际H中心EI。\n'
        'raw_error按t真值对齐，used_error按t1真值对齐，used_source_error按t真值对齐。\n'
        'motion列为重建波面、横摇角、纵摇角、H竖向位移、H法向速度，单位m、deg、deg、m、m/s。\n'
        'rate_time/rate排除了不推进时间的终止帧。EI缺测在NPZ中用NaN，其余表格用null。\n'
        '全部架次的原始50Hz日志仍保存在groups下series.jsonl.gz。\n')
    return sorted(found)


def forcing(plots,traces,policy):
    fig,axes=plt.subplots(2,3,figsize=(12.6,7.3));axes=axes.ravel()
    method='C0' if 'C0' in policy['methods'] else policy['methods'][0]
    weather=policy['weather'][0]
    names=['波面位移 / m','横摇角 / °','纵摇角 / °','H竖向位移 / m','H法向速度 / (m/s)','实际H中心 EI']
    ends=[];details=[]
    for sea,c in zip(conditions(policy), ['#236da4','#d46b28']):
        item=traces.get((weather,sea,method))
        if not item:continue
        m,z=item;ends.append(z['t1'][-1]);params=m['wave_parameters']
        details.append(f"{short_label(sea)}：Hs={m['actual_conditions']['wave_height_m']:g} m，Tp={params.get('PeakPeriod',11.2):g} s，浪向{params.get('RelativeWaveDirectionDeg',150):g}°，航速{np.linalg.norm(m['wave_speed_mps'][:2]):.2f} m/s")
        for j in range(5):axes[j].plot(z['t'],z['motion'][:,j],color=c,lw=1,label=short_label(sea))
        axes[5].plot(z['energy_time'],z['ei'],color=c,lw=1,label=short_label(sea))
    axes[5].axhline(1,color='black',ls='--',lw=.8)
    for ax,name in zip(axes,names):
        ax.set(xlabel='时间 / s',ylabel=name);ax.grid(alpha=.2)
        if ends:ax.set_xlim(0,max(min(ends),.02))
    axes[0].legend()
    save(fig,plots,'figure_5_11.png','图5-11 风浪输入与实际甲板运动',
         '；'.join(details)+'\n'+f'固定槽位{policy["representative_slot"]}，{WEATHER[weather]}，{method}共同物理时段；位移/法向为上正，姿态为NED约定；虚线EI=1。')


def trajectories(plots,traces,policy,sea,region):
    weathers=list(dict.fromkeys([policy['weather'][0],policy['weather'][-1]]))
    fig,axes=plt.subplots(len(weathers),3,squeeze=False,figsize=(13.8,4*len(weathers)))
    allz=[z for (w,s,m),(_,z) in traces.items() if w in weathers]
    limits=[padded(np.r_[[-12,12],*[z['p'][:,0] for z in allz]]),
            padded(np.r_[[-7,7],*[z['p'][:,1] for z in allz]])]
    for row,w in enumerate(weathers):
        deck(axes[row,0],region);found=False;summary=[]
        for i,method in enumerate(policy['methods']):
            item=traces.get((w,sea,method))
            if item is None:continue
            found=True;m,z=item;c=COLORS[method];p=z['p'];v=z['v'];t=z['state_time']
            axes[row,0].plot(p[:,0],p[:,1],color=c,lw=1)
            axes[row,0].plot(*m['nominal_gear_point'][:2],'+',color=c,ms=7)
            endpoint(axes[row,0],*p[-1,:2],m)
            for ax,y in zip(axes[row,1:],(p[:,2],v[:,2])):
                ax.plot(t,y,color=c,lw=1);endpoint(ax,t[-1],y[-1],m);events(ax,z,c,i)
                ax.set_xlabel('时间 / s');ax.grid(alpha=.2)
            ei=f"，EI={m['ei']:.3g}" if m['ei'] is not None else ''
            summary.append(f"{method}：{OUTCOME[m['touchdown_outcome']]}{ei}")
        for ax in axes[row]:ax.set_title(WEATHER[w])
        axes[row,1].set_ylabel('起落架相对H法向高度 / m');axes[row,2].set_ylabel('起落架相对H法向速度 / (m/s)')
        axes[row,0].set(xlim=limits[0],ylim=limits[1])
        if allz:
            axes[row,1].set_ylim(*padded(np.concatenate([z['p'][:,2] for z in allz])))
            axes[row,2].set_ylim(*padded(np.concatenate([z['v'][:,2] for z in allz])))
        for ax in axes[row,1:]:ax.set_xlim(0,policy['duration_s'])
        if not found:
            for ax in axes[row]:empty(ax,'未安排该天气或测例未完成')
        else:
            method_legend(axes[row,0],policy['methods'])
            axes[row,1].text(.02,.98,'\n'.join(summary),va='top',transform=axes[row,1].transAxes,fontsize=7)
    save(fig,plots,f'figure_5_12_sea{sea}.png',f'图5-12 代表着舰轨迹 · {label(sea,policy)}',
         '黑十字为H，彩色十字为姿态换算名义起落架点；虚框为几何参考区。点线：阶段推进；点划线：回退；下沿色带：无效等待；虚线：终止。')


def touchdowns(plots,rows,policy,sea,region):
    fig,axes=plt.subplots(2,2,figsize=(12.2,9.2))
    points=[r['touchdown_position'] for r in rows if r['candidate']]+[r['nominal_gear_point'] for r in rows if r['candidate']]
    xlim=padded([-12,12]+[p[0] for p in points]);ylim=padded([-7,7]+[p[1] for p in points])
    for ax in axes.ravel()[len(policy['weather']):]:ax.set_visible(False)
    for ax,w in zip(axes.ravel(),policy['weather']):
        deck(ax,region);ax.set_title(WEATHER[w]);lines=[]
        ax.set(xlim=xlim,ylim=ylim)
        for method in policy['methods']:
            g=[r for r in rows if r['sea']==sea and r['weather']==w and r['method']==method]
            c=COLORS[method]
            for m in g:
                if not m['candidate']:continue
                endpoint(ax,*m['touchdown_position'][:2],m)
                ax.plot(*m['nominal_gear_point'][:2],'+',color=c,ms=4,alpha=.55)
            n=len(g);t=sum(m['candidate'] for m in g);s=sum(m['safe'] for m in g);u=sum(m['candidate'] and not m['ei_complete'] for m in g)
            lines.append(f'{method}：{n}/{t}/{s}/{u}/{n-t}')
        # Counts use a separate title line, never cover points on the deck.
        ax.set_title(WEATHER[w]+'\n'+'；'.join(lines[:2])+'\n'+'；'.join(lines[2:]),fontsize=9)
        if not any(r['sea']==sea and r['weather']==w and r['candidate'] for r in rows):empty(ax,'无已触舰点')
    method_legend(axes[0,0],policy['methods'])
    save(fig,plots,f'figure_5_13_sea{sea}.png',f'图5-13 触舰点分布 · {label(sea,policy)}',
         '计数依次为：总数/触舰/安全/未知/未触舰。实心圆：EI<1；叉：EI≥1；空心圆：EI未知；彩色十字：名义点。不对落点添加抖动。')


def phases(plots,rows,policy,sea):
    fig,axes=plt.subplots(2,2,figsize=(12,8));methods=policy['methods']
    def values(r,j):return r['phases'][j]['completed_duration_s'] if j<3 else r['duration_s'] if r['candidate'] else None
    for j,ax in enumerate(axes.ravel()):
        for i,method in enumerate(methods):
            for wi,w in enumerate(WEATHER):
                v=[values(r,j) for r in rows if r['sea']==sea and r['method']==method and r['weather']==w]
                v=np.array([x for x in v if x is not None]);x=i+(wi-1.5)*.18
                if len(v):
                    ax.boxplot([v],positions=[x],widths=.14,showfliers=False,patch_artist=True,
                        boxprops=dict(facecolor=WC[wi],alpha=.18),medianprops=dict(color=WC[wi]))
                    ax.scatter(np.full(len(v),x),v,color=WC[wi],s=12,alpha=.65,zorder=3)
        full=[values(r,j) for r in rows];full=[v for v in full if v is not None]
        ax.set_ylim(0,max(full)*1.12 if full and max(full)>0 else policy['duration_s'])
        ax.set(xticks=range(len(methods)),xticklabels=methods,xlim=(-.6,len(methods)-.4),ylabel='时间 / s',
               title=PHASES[j]+'完成时长' if j<3 else '全部已触舰架次总时长');ax.grid(alpha=.2,axis='y')
    axes[0,0].legend(handles=[Line2D([],[],marker='o',color=c,label=WEATHER[w],ls='none') for w,c in zip(WEATHER,WC)],ncol=4)
    save(fig,plots,f'figure_5_14_sea{sea}.png',f'图5-14 分阶段时间分布 · {label(sea,policy)}',
         '箱线与逐架次散点；未完成不混入完成时长，未进入不记零。完成/进入数、删失停留、等待、回退及安全子集见表5-17。')


def control(plots,traces,policy,sea):
    fig,axes=plt.subplots(3,2,figsize=(13.2,10.4));axes=axes.ravel();recoveries=[]
    weather='night_fog' if 'night_fog' in policy['weather'] else policy['weather'][0]
    outcomes=[]
    for i,method in enumerate(policy['methods']):
        item=traces.get((weather,sea,method))
        if not item:continue
        m,z=item;c=COLORS[method];raw=z['raw_error'];used=z['used_error'];t=z['t'];t1=z['t1']
        outcomes.append(f"{method}：EI={m['ei']:.3g}" if m['ei'] is not None else f"{method}：{OUTCOME[m['touchdown_outcome']]}")
        for ax,j in [(axes[0],None),(axes[1],2)]:
            ax.plot(t,np.linalg.norm(raw[:,:2],axis=1) if j is None else raw[:,j],ls='--',color=c,lw=.85,alpha=.65)
            ax.plot(t1,np.linalg.norm(used[:,:2],axis=1) if j is None else used[:,j],color=c,lw=1)
        axes[2].plot(t1,z['guidance_error'][:,2],color=c,lw=1)
        axes[3].plot(t1,z['command'][:,2],color=c,lw=1)
        axes[3].plot(t1,z['actual_velocity'][:,2],color=c,ls='--',lw=.85)
        axes[4].plot(z['rate_time'],np.linalg.norm(z['rate'],axis=1),color=c,lw=1)
        # Each method gets a readable lane, 0/1 within lane; no hidden overlap.
        base=i*1.5
        axes[5].step(t,base+.28*z['healthy'],where='post',color=c,ls=':',lw=1)
        axes[5].step(t,base+.42+.28*z['shaper_valid'],where='post',color=c,ls='--',lw=1)
        axes[5].step(t1,base+.84+.28*z['valid'],where='post',color=c,lw=1)
        for ax in axes[:5]:events(ax,z,c,i)
        near=np.flatnonzero(z['near_bound'])
        if len(near):axes[3].scatter(t1[near],z['command'][near,2],color=c,marker='|',s=8)
        changes=np.flatnonzero((~z['valid'][:-1]) & z['valid'][1:])+1
        if len(changes):recoveries.append(float(t1[changes[0]]))
    names=['水平位置反馈误差范数 / m','下向位置反馈有符号误差 / m','下向制导误差 / m',
           '下向速度指令与实际速度 / (m/s)','指令变化率范数 / (m/s²)','导航与控制有效性']
    for ax,name in zip(axes,names):ax.set(xlabel='时间 / s',ylabel=name,xlim=(0,policy['duration_s']));ax.grid(alpha=.2)
    axes[5].set(yticks=[i*1.5+.55 for i in range(len(policy['methods']))],yticklabels=policy['methods'],ylim=(-.1,len(policy['methods'])*1.5))
    method_legend(axes[0],policy['methods'])
    axes[0].set_title('实线：实际消费量对t1真值；虚线：原始导航对t真值',fontsize=9)
    axes[3].set_title('实线：指令；虚线：实际速度；短竖线：近边界',fontsize=9)
    axes[5].set_title('每行自下而上：健康性、整形有效、实际控制有效（高为1）',fontsize=9)
    if recoveries:
        # Zoom remains inside the same PNG; deterministic first recovery.
        center=min(recoveries);inset=axes[2].inset_axes([.57,.45,.40,.38]);lo=max(0,center-1);hi=center+2
        for method in policy['methods']:
            item=traces.get((weather,sea,method))
            if not item:continue
            z=item[1];mask=(z['t1']>=lo)&(z['t1']<=hi)
            inset.plot(z['t1'][mask],z['guidance_error'][mask,2],color=COLORS[method],lw=.8)
        inset.set_title('首次有效性恢复附近',fontsize=7);inset.tick_params(labelsize=6)
    elif not any(k[0]==weather and k[1]==sea for k in traces):
        for ax in axes:empty(ax,'该天气代表测例未完成')
    rate_max=max([float(np.max(np.linalg.norm(z['rate'],axis=1))) for (w,s,_),(m,z) in traces.items()
                  if w==weather and s==sea and len(z['rate'])]+[.001])
    axes[4].set_ylim(0,rate_max*1.1)
    save(fig,plots,f'figure_5_15_sea{sea}.png',f'图5-15 {WEATHER[weather]}导航与控制响应 · {label(sea,policy)}',
         '；'.join(outcomes)+'\n同图5-12固定种子；C0消费真值参考。点线：阶段推进；点划线：回退；下沿色带：无效等待；虚线竖线：终止。完整三轴数据保留。')


def rates(plots,t15,policy):
    fig,axes=plt.subplots(2,2,figsize=(11.8,8.3));methods=policy['methods']
    for ax in axes.ravel()[len(policy['weather']):]:ax.set_visible(False)
    for ax,w in zip(axes.ravel(),policy['weather']):
        for sea,offset,symbol in zip(conditions(policy), [0] if len(conditions(policy))==1 else [-.12,.12], ['o','s']):
            for i,method in enumerate(methods):
                r=next((r for r in t15 if (r['weather'],r['sea'],r['method'])==(w,sea,method)),None)
                if not r or r['safe_rate'] is None:continue
                rate=r['safe_rate'];lo=r['safe_wilson_low'];hi=r['safe_wilson_high'];err=None
                if lo is not None:err=np.maximum(0,np.array([[rate-lo],[hi-rate]]))
                ax.errorbar(i+offset,rate,yerr=err,fmt=symbol,color=COLORS[method],capsize=3,ms=5)
                ax.annotate(f"{r['safe']}/{r['total']}\n触{r['touchdowns']} 未知{r['ei_unknown']}",(i+offset,rate),
                    xytext=(-3,12) if sea==5 else (3,-28),textcoords='offset points',ha='right' if sea==5 else 'left',fontsize=7)
        ax.set(title=WEATHER[w],xticks=range(len(methods)),xticklabels=methods,xlim=(-.6,len(methods)-.4),
               ylim=(-.25,1.25),yticks=[0,.25,.5,.75,1],ylabel='安全触舰率')
        ax.grid(alpha=.2,axis='y')
    axes[0,0].legend(handles=[Line2D([],[],marker=s,color='black',ls='none',label=short_label(sea)) for sea,s in zip(conditions(policy), ['o','s'])])
    save(fig,plots,'figure_5_16_rates.png','图5-16 安全触舰率与95% Wilson区间',
         '分母为全部纳入架次；未触舰和EI未知不计入安全数。标注为安全数/总数；触舰发生率与条件安全比例见表5-15。')


def paired_plot(plots,comparisons,policy):
    fig,axes=plt.subplots(3,2,figsize=(12.3,9.2))
    for j,metric in enumerate(('safe','ei','radial_error_m')):
        for col in range(2):
            ax=axes[j,col]
            selected=[r for r in comparisons if r['weather']=='equal_weather' and r['metric']==metric and
                ((r['left'].split('_')[0]==r['right'].split('_')[0])==(col==0))]
            labs=[]
            for i,r in enumerate(selected):
                left=r['left'].replace('sea','条件');right=r['right'].replace('sea','条件')
                counts='/'.join(str(r['counts_by_weather'].get(w,0)) for w in WEATHER)
                labs.append(f"{left} 减 {right}\n配对数{r['pairs']}：{counts}")
                if r['mean'] is None:ax.text(i,.5,'不可估计',transform=ax.get_xaxis_transform(),ha='center',fontsize=8);continue
                ax.errorbar(i,r['mean'],yerr=np.maximum(0,[[r['mean']-r['low']],[r['high']-r['mean']]]),fmt='o',color='#236da4',capsize=4)
            ax.axhline(0,color='gray',lw=.8,ls='--');ax.set_xticks(range(len(labs)),labs,fontsize=8)
            ax.set_ylabel(['安全触舰率差','触舰EI差','触舰径向误差差 / m'][j]);ax.grid(alpha=.2,axis='y')
            if not selected:empty(ax,'未安排相应配对方法')
            if j==0:ax.set_title('同海况下的方法配对差值' if col==0 else '同方法下的6级减5级差值')
    save(fig,plots,'figure_5_16_paired.png','图5-16 天气等权配对差值及95%自助区间',
         '配对数后依次为晴/雾/夜/夜雾；EI、误差仅使用双方均触舰且数据完整的配对。任何天气层无样本时总体不可估计。')


def energy_distribution(plots,rows,policy,weather):
    fig,axes=plt.subplots(len(conditions(policy)),2,squeeze=False,figsize=(12.4,4.7*len(conditions(policy))));methods=policy['methods']
    vmax=max([r['ei'] for r in rows if r['weather']==weather and r['ei'] is not None]+[1.2])
    for row,sea in enumerate(conditions(policy)):
        ax,ecdf=axes[row];counts=[];any_data=False
        for i,method in enumerate(methods):
            group=[r for r in rows if r['weather']==weather and r['sea']==sea and r['method']==method]
            v=np.sort([r['ei'] for r in group if r['ei'] is not None]);c=COLORS[method]
            touched=sum(r['candidate'] for r in group);unknown=touched-len(v);safe=int((v<1).sum())
            counts.append(f'{method}：{len(group)}/{touched}/{len(v)}/{unknown}/{safe}/{len(group)-touched}')
            if not len(v):continue
            any_data=True
            ax.boxplot([v],positions=[i],widths=.45,showfliers=False,patch_artist=True,boxprops=dict(facecolor=c,alpha=.15))
            ax.scatter(np.full(int((v<1).sum()),i),v[v<1],color=c,s=16,zorder=3)
            ax.scatter(np.full(int((v>=1).sum()),i),v[v>=1],color=c,marker='x',s=20,zorder=3)
            ecdf.step(np.r_[0,v,vmax*1.04],np.r_[0,np.arange(1,len(v)+1)/len(v),1],where='post',color=c,label=method)
            ecdf.plot(1,safe/len(v),'o',color=c,mfc='white',ms=5)
        ax.axhline(1,color='black',ls='--',lw=.8);ecdf.axvline(1,color='black',ls='--',lw=.8)
        ax.set(xticks=range(len(methods)),xticklabels=methods,ylabel='触舰时刻 EI',ylim=(0,vmax*1.08))
        ecdf.set(xlabel='触舰时刻 EI',ylabel='经验累积分布',ylim=(-.03,1.06),xlim=(0,vmax*1.04))
        if vmax>10:
            ax.set_yscale('symlog',linthresh=1);ecdf.set_xscale('symlog',linthresh=1)
            ecdf.set_xlabel('触舰时刻 EI（1以内线性，外侧对数）')
        text=f'{short_label(sea)}\n'+'；'.join(counts[:2])+'\n'+'；'.join(counts[2:])
        ax.set_title(text,fontsize=9);ecdf.set_title(f'{short_label(sea)}：空心点为严格EI<1的比例',fontsize=9)
        for a in (ax,ecdf):a.grid(alpha=.2)
        if any_data:ecdf.legend()
        else:empty(ax,'无EI完整的触舰样本');empty(ecdf,'无可估计分布')
        if vmax>4 and any_data:
            inset=ecdf.inset_axes([.57,.12,.39,.35])
            for method in methods:
                v=np.sort([r['ei'] for r in rows if r['weather']==weather and r['sea']==sea and r['method']==method and r['ei'] is not None])
                if len(v):inset.step(np.r_[0,v],np.r_[0,np.arange(1,len(v)+1)/len(v)],where='post',color=COLORS[method],lw=.8)
            inset.set(xlim=(0,2),ylim=(0,1));inset.axvline(1,ls='--',color='gray',lw=.6);inset.set_title('阈值附近',fontsize=7);inset.tick_params(labelsize=6)
    save(fig,plots,f'figure_5_17_{weather}.png',f'图5-17 {WEATHER[weather]}触舰时刻实际舰船EI分布',
         '计数：总数/触舰/EI完整/未知/安全/未触舰。圆点：安全；叉：非安全；虚线EI=1，等值不属安全。分布分母为EI完整触舰数。')
