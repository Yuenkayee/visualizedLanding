"""Evaluation-only metrics for the revised §5.2.3 experiment."""
import numpy as np
from .energy import evaluate, classify, validate_policy
from .command_analysis import increments
from .recording import KNOT


def vector(rows, key):
    return np.array([r['u'][key] for r in rows], dtype=float).reshape(-1,3)


def error_summary(values, valid):
    def stats(x):
        if not len(x): return dict(n=0, rmse_axis_m=None, rmse_norm_m=None, p95_norm_m=None, peak_norm_m=None)
        norm=np.linalg.norm(x,axis=1)
        return dict(n=len(x), rmse_axis_m=np.sqrt(np.mean(x*x,axis=0)).tolist(),
                    rmse_norm_m=float(np.sqrt(np.mean(norm*norm))),
                    p95_norm_m=float(np.quantile(norm,.95)),peak_norm_m=float(norm.max()))
    return dict(full=stats(values), valid=stats(values[valid]))


def augment(m, rows, model, metadata, policy):
    validate_policy(policy)
    h=metadata['geometry']['H_ship_body']
    energy=[evaluate(r['u'],h) for r in rows]
    terminal=energy[-1]
    ei=terminal['value'] if m['candidate'] else None
    outcome=classify(m['candidate'],ei)
    m.update(touchdown=m['candidate'], safe=outcome=='safe_touchdown', qualified=outcome=='safe_touchdown',
        ei=ei, ei_complete=m['candidate'] and ei is not None,
        touchdown_outcome=outcome, ei_terminal=terminal if m['candidate'] else None,
        qualification_policy='Original landing STOP = touchdown; actual fixed-H paper EI < 1 = safe touchdown')
    if m['candidate']: m['termination']=outcome
    elif metadata.get('termination_reason')=='model_stop': m['termination']='interface_numerical'
    m['stop_input']=bool(np.asarray(rows[-1]['u'].get('touchdown_stop',0)).item())
    m['stop_reason']=rows[-1].get('stop_reason')
    _,_,audit=increments(rows,model,policy['command_cost_weights'])
    m.update(Jctrl=audit['Jctrl'], command_rate_peak_mps2=audit['command_rate_peak_mps2'], command_time_audit=audit)
    actual=np.array([r['cg_true'] for r in rows]).reshape(-1,3)
    # The latch evaluated at t1 holds the estimate sourced at t. Compare at
    # BOTH times explicitly; hold error is distinct from the source error.
    current=np.array([np.asarray(r['s1']['heli'])[:3]-np.asarray(r['s1']['ship'])[:3] for r in rows])
    consumed=vector(rows,'cg_used'); raw=np.array([r['cg_raw'] for r in rows]).reshape(-1,3)
    valid=np.array([bool(r['u']['control_valid']) for r in rows])
    m['feedback_errors']=dict(raw_source=error_summary(raw-actual,np.array([r['nav_valid'] for r in rows])),
        consumed_source=error_summary(consumed-actual,valid), consumed_evaluation=error_summary(consumed-current,valid))
    guidance=vector(rows,'target')-current
    stages=np.array([r['stage'] for r in rows])
    for phase in m['phase_tracking']:
        ix=stages==phase['stage']
        phase['tracking_error']=error_summary(guidance[ix],valid[ix])
        phase['cg_tracking_rmse_m']=phase['tracking_error']['full']['rmse_norm_m']
    for phase in m['phases']:
        phase['regressions']=sum(int(r['u']['stage'])<r['stage'] and r['stage']==phase['stage'] for r in rows)
    finite=[x['value'] for x in energy if x['value'] is not None]
    m['motion_ei']=dict(n=len(finite), unknown=len(rows)-len(finite),
        rms=float(np.sqrt(np.mean(np.square(finite)))) if finite else None,
        peak=max(finite) if finite else None, below_one_fraction=float(np.mean(np.array(finite)<1)) if finite else None)
    m['actual_conditions']=dict(wind_mps=model.get('wind_mps'),
        wave_height_m=model.get('wave_parameters',{}).get('SignificantWaveHeight'))
    m['wave_parameters']=model.get('wave_parameters',{})


def trace(rows, model, geometry):
    """Compact, full-rate figure source. Terminal point is explicitly appended."""
    from .metrics import motion
    t=np.array([r['t'] for r in rows]);t1=np.array([r['t1'] for r in rows])
    states=[r['s'] for r in rows]+[rows[-1]['s1']]
    ts=np.r_[t,t1[-1]]
    # Actual dynamics telemetry at t0 then each endpoint. Duplicate terminal
    # timestamps stay in the provenance, and are omitted from derivatives.
    u=[rows[0].get('u0',{})]+[r['u'] for r in rows]
    et=np.r_[t[0],t1]
    energy=[evaluate(v,geometry['H_ship_body']) for v in u]
    ei=np.array([np.nan if v['value'] is None else v['value'] for v in energy])
    actual=np.array([r['cg_true'] for r in rows]).reshape(-1,3)
    current=np.array([np.asarray(r['s1']['heli'])[:3]-np.asarray(r['s1']['ship'])[:3] for r in rows])
    raw=np.array([r['cg_raw'] for r in rows]).reshape(-1,3)-actual
    used=vector(rows,'cg_used')-current
    rate_t,rate,_=increments(rows,model)
    return dict(t=t,t1=t1,state_time=ts,energy_time=et,ei=ei,
        p=np.array([s['p'] for s in states]),v=np.array([s['v'] for s in states]),
        motion=motion(rows,model),raw_error=raw,used_error=used,
        used_source_error=vector(rows,'cg_used')-actual,
        guidance_error=vector(rows,'error'),tracking_error=vector(rows,'target')-current,
        command=vector(rows,'command')*KNOT,relative_command=vector(rows,'relative_command')*KNOT,
        actual_velocity=np.array([r['s1']['velocity_ned_mps'] for r in rows]),
        rate_time=rate_t,rate=rate,stage=np.array([r['stage'] for r in rows]),
        next_stage=np.array([int(r['u']['stage']) for r in rows]),
        valid=np.array([bool(r['u']['control_valid']) for r in rows]),
        nav_valid=np.array([r['nav_valid'] for r in rows]),
        healthy=np.array([r.get('nav_healthy',r['nav_valid']) for r in rows]),
        shaper_valid=np.array([r['shaper_valid'] for r in rows]),
        near_bound=np.any(abs(vector(rows,'relative_command')*KNOT)>=.95*np.asarray(model['command_limit_mps']).ravel(),axis=1))
