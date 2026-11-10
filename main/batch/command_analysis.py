"""Offline command differences; duplicate terminal timestamps are not time steps."""
import numpy as np
from .metrics import arrays


def increments(rows, model, weights=(1., 1., 1.)):
    a = arrays(rows)
    dt = np.diff(a['t1'])
    if np.any(~np.isfinite(dt)) or np.any(dt < -1e-9):
        raise ValueError('Control evaluation timestamps must be finite and nondecreasing')
    valid = (dt > 1e-9) & (a['dt'][1:] > 1e-9)
    delta = np.diff(a['command'], axis=0)[valid]
    span = dt[valid]
    limits = np.asarray(model['command_limit_mps']).ravel()
    if limits.shape != (3,) or np.any(limits <= 0):
        raise ValueError('Expected three positive guidance bounds')
    derivative = delta / span[:, None]
    cost = float(np.sum((delta / limits)**2 * np.asarray(weights) / span[:, None]))
    audit = dict(Jctrl=cost, command_rate_peak_mps2=(np.max(abs(derivative), axis=0).tolist()
                 if len(derivative) else [None]*3),
                 used_intervals=int(valid.sum()), omitted_nonadvancing_intervals=int((~valid).sum()),
                 omitted_indices=np.flatnonzero(~valid).tolist(),
                 definition='sum W_i*(delta command_i / existing atan bound_i)^2 / dt; dt>1e-9 and advancing plant step')
    return a['t1'][1:][valid], derivative, audit
