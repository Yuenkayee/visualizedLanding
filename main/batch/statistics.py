"""Wilson binomial intervals and weather-stratified paired seed bootstrap."""
import numpy as np


def wilson(k, n):
    if not n: return [None, None]
    z = 1.959963984540054; p = k/n; d = 1+z*z/n
    mid = (p+z*z/(2*n))/d
    half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    # These endpoints are exact mathematically. Subtracting/adding the rounded
    # center and half-width can otherwise exclude p by a few ulps at 0 or 1.
    low = 0.0 if k == 0 else max(0.0, float(mid-half))
    high = 1.0 if k == n else min(1.0, float(mid+half))
    return [low, high]


def describe(values):
    v = np.array([x for x in values if x is not None],dtype=float)
    if not len(v): return dict(n=0, mean=None, median=None, q25=None, q75=None, std=None)
    return dict(n=len(v), mean=float(v.mean()), median=float(np.median(v)),
                q25=float(np.quantile(v,.25)), q75=float(np.quantile(v,.75)),
                std=float(v.std(ddof=1)) if len(v)>1 else None)


def paired(rows, left, right, metric, weather, policy):
    weathers = policy["weather"] if weather == "equal_weather" else [weather]
    index = {(r["weather"],r["slot"],r["sea"],r["method"]):r for r in rows}
    diffs, missing, point_pairs = {}, {}, []
    for w in weathers:
        diffs[w] = []
        for slot in range(policy["seeds_per_weather"]):
            a,b = index.get((w,slot,*left)),index.get((w,slot,*right))
            reason = None
            if a is None or b is None: reason="missing_run"
            elif a["replacement"] != b["replacement"]: reason="unpaired_replacement"
            elif metric in ("radial_error_m","successful_duration_s") and not (a["candidate"] and b["candidate"]): reason="both_candidates_required"
            if reason:
                missing[reason]=missing.get(reason,0)+1; continue
            key = "duration_s" if metric == "successful_duration_s" else metric
            diff=float(a[key])-float(b[key]); diffs[w].append(diff)
            point_pairs.append(dict(weather=w,slot=slot,difference=diff))
    # Equal weather means every requested stratum contributes equally. Do not
    # quietly change weights if a stratum has no conditional candidate pairs.
    counts={w:len(v) for w,v in diffs.items()}; complete=all(counts.values())
    mean = low = high = None
    if complete:
        mean=float(np.mean([np.mean(v) for v in diffs.values()]))
        rng=np.random.default_rng(policy["bootstrap_seed"])
        draws=[]
        for values in diffs.values():
            v=np.asarray(values); draws.append(v[rng.integers(0,len(v),size=(policy["bootstrap_samples"],len(v)))].mean(axis=1))
        low,high=map(float,np.quantile(np.mean(draws,axis=0),[.025,.975]))
    return dict(weather=weather, left=f"sea{left[0]}_{left[1]}",right=f"sea{right[0]}_{right[1]}",
        metric=metric, mean=mean, low=low, high=high, pairs=sum(counts.values()),
        counts_by_weather=counts, missing=missing, raw_pairs=point_pairs,
        interval_note="one pair gives a degenerate bootstrap, not evidence of precision" if any(n==1 for n in counts.values()) else "paired seed bootstrap")
