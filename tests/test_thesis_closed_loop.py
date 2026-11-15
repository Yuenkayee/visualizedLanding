"""Paper EI, STOP semantics, censored statistics, paired data and 15-file reports."""
import copy
import gzip
import json
from pathlib import Path
import numpy as np
import pytest
import yaml
from main.batch.common import ROOT, read_json, write_json, sha
from main.batch.energy import evaluate, rotation, classify, WEIGHTS
from main.batch.metrics import summarize
from main.batch.thesis_analysis import analyze, tables, paired
from main.batch.__main__ import parser, create, round_tasks, supervise
from test_closed_loop_batch import fake_case


@pytest.fixture
def policy():
    p=yaml.safe_load((ROOT/'main/batch_closed_loop.yaml').read_text())
    p.update(master_seed=17,seeds_per_weather=2,bootstrap_samples=100)
    return p


def modern_case(path,policy,weather='clear',sea=5,method='C3',slot=0,outcome='safe'):
    _,rows,model=fake_case(path,policy,weather,sea,method,slot,candidate=outcome!='timeout')
    for i,r in enumerate(rows):
        # Identical actual ship dynamics across methods. A fixed east velocity
        # makes the terminal EI analytic and independent of the nominal point.
        ve=.1 if outcome=='safe' else .2 if outcome=='boundary' else .3
        telemetry=dict(ship_eta=[0,0,0,0,0,0],ship_nu=[0,ve,0,0,0,0],
            ship_eta_dot=[0,ve,0,0,0,0],ship_nu_dot=[0]*6,touchdown_stop=float(i==5 and outcome!='timeout'))
        if outcome=='unknown':telemetry.pop('ship_nu_dot')
        r['u'].update(telemetry);r['u0']=dict(r['u'],touchdown_stop=0)
        r['nav_healthy']=r['nav_valid']
    if outcome=='other_stop':rows[-1]['u']['touchdown_stop']=0
    # Include a repeated terminal timestamp: it must not generate an impulse.
    last=copy.deepcopy(rows[-1]);last.update(t=last['t1']);last['u']['command'][0]+=3
    rows.append(last)
    with gzip.open(path/'series.jsonl.gz','wt') as f:
        for r in rows:f.write(json.dumps(r)+'\n')
    meta=read_json(path/'run.json');seed=meta['config']['batch'];condition=policy['seas'][sea]
    model.update(wind_mps=condition['wind_mps'],wind_seed=seed['wind_seed'],
        wave_parameters=dict(SignificantWaveHeight=condition['wave_height_m'],RandomSeed=seed['wave_seed']),
        evaluation=dict(profile='thesis_523_15'))
    write_json(path/'simulink/batch_model.json',model)
    m=summarize(path,policy);write_json(path/'metrics.json',m)
    return m,rows,model


def test_energy_has_exact_fixed_h_transport_and_euler_rates():
    eta=[0,0,0,.2,-.3,.4];nu=[2,3,4,.1,.2,-.3];dnu=[.5,.3,.1,.02,-.03,.04]
    deta=[0,0,0,.07,.09,.11];h=np.array([-45.,0,-5.5])
    u=dict(zip(('ship_eta','ship_nu','ship_eta_dot','ship_nu_dot'),(eta,nu,deta,dnu)))
    value=evaluate(u,h);r=rotation(*eta[3:]);w=np.array(nu[3:]);v=np.array(nu[:3]);alpha=np.array(dnu[3:])
    assert value['H_velocity_ned_mps']==pytest.approx(r@(v+np.cross(w,h)))
    assert value['H_acceleration_ned_mps2']==pytest.approx(r@(np.array(dnu[:3])+np.cross(w,v)+np.cross(alpha,h)+np.cross(w,np.cross(w,h))))
    assert value['state']['roll_rate_radps']==.07
    assert value['state']['pitch_rate_radps']==.09
    assert value['value']==pytest.approx(sum(WEIGHTS*np.array(list(value['state'].values()))**2))
    # Pure steady yaw: fixed H experiences centripetal acceleration.
    yaw=dict(ship_eta=[0]*6,ship_nu=[0,0,0,0,0,2],ship_eta_dot=[0,0,0,0,0,2],ship_nu_dot=[0]*6)
    assert evaluate(yaw,[1,0,0])['H_acceleration_ned_mps2']==pytest.approx([-4,0,0])


def test_energy_rigid_transport_agrees_with_independent_finite_differences():
    # Constant world rotation about y; translating origin with known acceleration.
    t=.7;h=np.array([-45.,0,-5.5]);rate=.13;v=np.array([2,.3,-.2]);acc=np.array([.1,.2,-.3])
    r=rotation(0,rate*t,0);omega=np.array([0,rate,0]);vb=r.T@(v+acc*t)
    u=dict(ship_eta=[*(v*t+.5*acc*t*t),0,rate*t,0],ship_nu=[*vb,*omega],
        ship_eta_dot=[*(v+acc*t),0,rate,0],ship_nu_dot=[*(r.T@acc-np.cross(omega,vb)),0,0,0])
    def p(s):return v*s+.5*acc*s*s+rotation(0,rate*s,0)@h
    dt=1e-3;answer=evaluate(u,h)
    assert answer['H_velocity_ned_mps']==pytest.approx((p(t+dt)-p(t-dt))/(2*dt),abs=1e-7)
    assert answer['H_acceleration_ned_mps2']==pytest.approx((p(t+dt)-2*p(t)+p(t-dt))/(dt*dt),abs=1e-7)


@pytest.mark.parametrize('outcome,terminal,touched,safe',[
    ('safe','safe_touchdown',True,True),('boundary','unsafe_touchdown',True,False),
    ('unsafe','unsafe_touchdown',True,False),('unknown','touchdown_ei_unknown',True,False),
    ('timeout','timeout',False,False),('other_stop','interface_numerical',False,False)])
def test_stop_ei_and_unknown_are_separate(tmp_path,policy,outcome,terminal,touched,safe):
    m,rows,model=modern_case(tmp_path/'case',policy,outcome=outcome)
    assert m['termination']==terminal and m['candidate']==touched and m['safe']==safe
    assert m['command_time_audit']['omitted_nonadvancing_intervals']==1
    assert m['Jctrl']<1
    if not touched:assert m['ei'] is None and m['touchdown_position'] is None
    if outcome=='boundary':assert m['ei']==pytest.approx(1.)
    if outcome=='unknown':assert m['ei_terminal']['value'] is None


def test_ei_missing_not_filled_and_strict_threshold():
    assert evaluate({},[-45,0,-5.5])['value'] is None
    assert classify(True,1)=='unsafe_touchdown'
    assert classify(True,None)=='touchdown_ei_unknown'
    assert classify(False,.1)=='not_touched'


def test_new_plans_and_c3_paired_replacement(tmp_path):
    full=create(parser().parse_args(['plan','--output','outputs/batch/unused']))
    c3=create(parser().parse_args(['plan','--c3-only','--output','outputs/batch/unused']))
    assert len(full['groups'])*len(full['policy']['methods'])*2==320
    assert c3['base_config']['duration']==80 and len(c3['groups'])*2==80
    assert c3['policy']['seas']=={5:dict(wind_mps=10.,wave_height_m=2.),6:dict(wind_mps=12.,wave_height_m=2.5)}
    c3['groups']=c3['groups'][:1];tasks=round_tasks(tmp_path,c3)
    assert len(tasks)==2 and {t['method'] for t in tasks}=={'C3'}
    for i,t in enumerate(tasks):write_json(tmp_path/t['case']/'metrics.json',dict(termination='runaway' if i==0 else 'timeout'))
    replacement=round_tasks(tmp_path,c3)
    assert len(replacement)==2 and {t['attempt'] for t in replacement}=={1}


@pytest.mark.parametrize('workers',[1,6])
def test_empty_explicit_sessions_never_fall_back_to_starting_matlab(tmp_path,workers):
    args=parser().parse_args(['run','--output','outputs/batch/unused','--workers',str(workers),'--matlab-sessions',''])
    with pytest.raises(ValueError,match='distinct dedicated'):
        supervise(tmp_path,{},args)


def test_denominators_and_conditional_missing_pairs(tmp_path,policy):
    rows=[]
    for w in policy['weather']:
        for slot,outcome in enumerate(('safe','timeout')):
            for method in ('C1','C3'):
                m,_,_=modern_case(tmp_path/f'{w}_{slot}_{method}',policy,w,method=method,slot=slot,outcome=outcome)
                m['path']=m['case_id'];rows.append(m)
    t15,_,t17=tables(rows,policy,tmp_path)
    r=next(x for x in t15 if (x['weather'],x['sea'],x['method'])==('equal_weather',5,'C3'))
    assert r['total']==8 and r['touchdowns']==4 and r['safe_rate']==.5
    assert r['conditional_safe_proportion']==1
    p=paired(rows,(5,'C3'),(5,'C1'),'ei','equal_weather',policy)
    assert p['pairs']==4 and p['mean']==0 and p['missing']['both_touchdowns_required']==4
    for m in rows:
        if m['weather']=='fog':m['ei']=None
    p=paired(rows,(5,'C3'),(5,'C1'),'ei','equal_weather',policy)
    assert p['mean'] is None and p['counts_by_weather']['fog']==0
    p=paired(rows,(5,'C3'),(5,'C1'),'safe','equal_weather',policy)
    assert p['pairs']==8 and p['mean']==0


@pytest.mark.parametrize('methods',[['C0','C1','C2','C3'],['C3']])
def test_complete_report_has_exactly_15_images_and_preserves_evidence(tmp_path,policy,methods):
    policy['methods']=methods;policy['seeds_per_weather']=1;policy['duration_s']=.12
    groups=[];out=tmp_path/'batch';out.mkdir();hashes={}
    for w in policy['weather']:
        g=dict(id=w+'_s00',weather=w,slot=0);groups.append(g);cases=[]
        for sea in (5,6):
            for method in methods:
                relative=f'groups/{g["id"]}/attempt_00/sea{sea}_{method}';case=out/relative
                modern_case(case,policy,w,sea,method,outcome='safe' if sea==5 else 'boundary');cases.append(relative)
                hashes[relative]=sha(case/'series.jsonl.gz')
        result=dict(status='included',attempt=0,cases=cases)
        write_json(out/'groups'/g['id']/'result.json',result)
        write_json(out/'groups'/g['id']/'attempt_00/attempt.json',result)
    landing=yaml.safe_load((ROOT/'configs/landing_dataset.yaml').read_text())
    manifest=dict(policy=policy,groups=groups,pilot=True,landing_snapshot=landing,experiment='synthetic')
    write_json(out/'manifest.json',manifest)
    status=analyze(out)
    assert status['analysis_complete'] and status['complete']
    assert status['pairing_passed'] and status['physical_conditions_passed']
    assert status['image_count']==15 and len(list((out/'analysis/figures').glob('*.png')))==15
    assert all(sha(out/p/'series.jsonl.gz')==h for p,h in hashes.items())
    assert status['safe']==4*len(methods) and status['unsafe']==4*len(methods)
