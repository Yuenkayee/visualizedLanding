"""Report dimensions come from the frozen experiment, not assumed sea labels."""
from .common import METHODS


def conditions(policy):
    return tuple(int(s) if str(s).isdigit() else s for s in policy.get('seas', {5: {}, 6: {}}))


def methods(policy):
    return tuple(policy.get('methods', METHODS))


def condition_label(sea, policy=None):
    if sea in (5, 6):
        return f'Sea {sea}'
    values = (policy or {}).get('seas', {}).get(str(sea), {})
    if values:
        return f"Wind {values['wind_mps']:g} m/s, Hs {values['wave_height_m']:g} m"
    if sea == 'custom_w10_h2':
        return 'Wind 10 m/s, Hs 2 m'
    return str(sea)
