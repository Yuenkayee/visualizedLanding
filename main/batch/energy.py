"""Paper EI at fixed H, from actual ship dynamics; no control-path changes."""
import numpy as np

WEIGHTS = np.square(1. / np.array([.2, .1, 2., 2., .06, .05, .15, .2]))
TERMS = ('H_east_velocity_mps', 'H_east_acceleration_mps2',
         'H_down_velocity_mps', 'H_down_acceleration_mps2',
         'roll_rad', 'roll_rate_radps', 'pitch_rad', 'pitch_rate_radps')
KINEMATICS = ('ship_eta', 'ship_nu', 'ship_eta_dot', 'ship_nu_dot')


def validate_policy(policy):
    if not np.array_equal(policy.get('ei_weights'), WEIGHTS):
        raise ValueError('Evaluation EI weights must match thesis equation (5-72) in its stated order')
    if policy['duration_s'] <= 0 or policy['bootstrap_samples'] < 1:
        raise ValueError('Positive duration and bootstrap sample count required')
    if policy.get("condition_preset") == "clear_c3_w10_h2":
        if (policy["weather"] != ["clear"] or policy["methods"] != ["C3"] or
                policy["seas"] != {"custom_w10_h2": dict(wind_mps=10., wave_height_m=2.)}):
            raise ValueError("Invalid custom clear C3 condition")
        return
    if policy.get('condition_preset') == 'user_json':
        from .user_config import condition
        for v in policy['seas'].values(): condition(v)
        return
    for sea, wind, wave in [(5, 12., 2.), (6, 15., 3.)]:
        condition=policy['seas'].get(sea, policy['seas'].get(str(sea)))
        if condition != dict(wind_mps=wind, wave_height_m=wave):
            raise ValueError(f'New experiment sea {sea} must use wind={wind}, Hs={wave}')


def rotation(phi, theta, psi):
    cr,sr,cp,sp,cy,sy=np.cos(phi),np.sin(phi),np.cos(theta),np.sin(theta),np.cos(psi),np.sin(psi)
    return np.array([[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],
                     [sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],[-sp,cp*sr,cp*cr]])


def evaluate(telemetry, h_body):
    """R maps ship FRD to NED. nu_dot is the derivative of BODY components.

    a_H = R [v_dot + omega x v + alpha x r + omega x (omega x r)].
    Euler rates come from eta_dot, never substitute body p/q for phi/theta-dot.
    Missing/nonfinite inputs are unknown, never zero-filled or predicted.
    """
    try:
        eta,nu,deta,dnu=[np.asarray(telemetry[k], dtype=float).reshape(6) for k in KINEMATICS]
        h=np.asarray(h_body, dtype=float).reshape(3)
        if not np.isfinite(np.r_[eta,nu,deta,dnu,h]).all():
            raise ValueError('nonfinite actual ship state')
        r=rotation(*eta[3:6]); omega=nu[3:]; alpha=dnu[3:]
        velocity=r @ (nu[:3]+np.cross(omega,h))
        acceleration=r @ (dnu[:3]+np.cross(omega,nu[:3])+np.cross(alpha,h)+np.cross(omega,np.cross(omega,h)))
        x=np.array([velocity[1],acceleration[1],velocity[2],acceleration[2],eta[3],deta[3],eta[4],deta[4]])
        contributions=WEIGHTS*x*x
        ei=float(contributions.sum())
        if not np.isfinite(ei): raise ValueError('nonfinite EI')
        return dict(value=ei, state=dict(zip(TERMS,x.tolist())), contributions=contributions.tolist(),
                    H_velocity_ned_mps=velocity.tolist(), H_acceleration_ned_mps2=acceleration.tolist(),
                    H_position_ned_m=(eta[:3]+r@h).tolist(), unknown_reason=None)
    except (KeyError, ValueError, TypeError) as exc:
        return dict(value=None, state=None, contributions=None, unknown_reason=str(exc))


def classify(touched, ei):
    if not touched: return 'not_touched'
    if ei is None: return 'touchdown_ei_unknown'
    return 'safe_touchdown' if ei < 1. else 'unsafe_touchdown'
