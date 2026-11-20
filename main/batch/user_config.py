"""Validate, freeze and name a configurable closed-loop batch before dispatch."""
import json
import math
import sys
import tempfile
from datetime import datetime
from pathlib import Path

SEAS = (5, 6)
WEATHER = ('clear', 'fog', 'night', 'night_fog')
METHODS = ('C0', 'C1', 'C2', 'C3')


def selected(raw, key, allowed):
    values = raw.get(key, [])
    if not isinstance(values, list) or any(type(x) is not type(allowed[0]) or x not in allowed for x in values):
        raise ValueError(f'{key}: expected a list drawn from {allowed}')
    if len(set(values)) != len(values):
        raise ValueError(f'{key}: duplicate selection')
    return [x for x in allowed if not values or x in values]


def condition(c):
    if not isinstance(c, dict) or set(c) != {'wind_mps','wave_height_m','prediction_window_s'}:
        raise ValueError('Each condition requires wind_mps, wave_height_m and prediction_window_s')
    def number(x):
        return type(x) in (float,int) and math.isfinite(x)
    for key in ('wind_mps','wave_height_m'):
        if not number(c[key]) or c[key] <= 0:
            raise ValueError(f'{key}: expected a positive finite number')
    w = c['prediction_window_s']
    if not isinstance(w,list) or len(w)!=2 or not all(number(x) and 0<=x<=80 and abs(x*50-round(x*50))<1e-8 for x in w) or w[0]>=w[1]:
        raise ValueError('prediction_window_s: use [lower, upper] within 0..80 s on the 0.02 s grid, lower < upper')
    return c


def resolve(raw):
    if not isinstance(raw,dict) or set(raw)-{'conditions','sea_levels','weather','methods','duration_s'} or set(raw.get('conditions',{}))!={'5','6'}:
        raise ValueError('Expected conditions for 5 and 6, and optional sea_levels, weather, methods, duration_s')
    duration=raw.get('duration_s',80)
    if type(duration) not in (int,float) or not math.isfinite(duration) or duration < 0.02 or abs(duration*50-round(duration*50))>1e-8:
        raise ValueError('duration_s must be a positive multiple of 0.02 s')
    cs={s:condition(raw['conditions'][str(s)]) for s in SEAS}
    return dict(duration_s=duration, seas={s:cs[s] for s in selected(raw,'sea_levels',SEAS)}, weather=selected(raw,'weather',WEATHER), methods=selected(raw,'methods',METHODS),condition_preset='user_json')


def prepare(source, parent):
    text=Path(source).read_text(); p=resolve(json.loads(text)); parent=Path(parent); parent.mkdir(parents=True,exist_ok=True)
    label='_'.join([datetime.now().strftime('%Y%m%d_%H%M%S'),'-'.join(f"s{s}w{c['wind_mps']:g}h{c['wave_height_m']:g}ei{c['prediction_window_s'][0]:g}to{c['prediction_window_s'][1]:g}" for s,c in p['seas'].items()),'-'.join(p['weather']),'-'.join(p['methods']),f"t{p['duration_s']:g}s"])
    out=Path(tempfile.mkdtemp(prefix='batch_'+label+'_',dir=parent)); (out/'batch_config.json').write_text(text)
    return out


if __name__=='__main__':
    print(prepare(sys.argv[1],sys.argv[2]))
