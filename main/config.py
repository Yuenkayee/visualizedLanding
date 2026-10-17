"""Resolve one geometry source for rendering, navigation and Simulink."""

from copy import deepcopy
from pathlib import Path
import numpy as np
import yaml
from visual_modeling.trajectories.landing import camera_geometry

ROOT = Path(__file__).resolve().parents[1]


def path(value):
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def load(filename="main/closed_loop.yaml"):
    c = yaml.safe_load(path(filename).read_text())
    landing = yaml.safe_load(path(c["landing_config"]).read_text())
    landing = deepcopy(landing)
    if c.get("render_width"):
        scale = c["render_width"] / landing["camera"]["width"]
        landing["camera"]["width"] = c["render_width"]
        landing["camera"]["height"] = round(landing["camera"]["height"] * scale)
    if c.get("samples"):
        landing["render"]["samples"] = c["samples"]
    cal, mount = camera_geometry(landing)  # nominal installation, no random calibration
    geometry = dict(
        T_body_camera=mount.tolist(),
        H_ship_body=landing["geometry"]["H_ship_body_m"],
        gear_body=landing["geometry"]["gear_body_m"],
    )
    return c, landing, cal, geometry


def validate(c):
    dt, h, duration = (float(c[k]) for k in ("dt", "plant_step", "duration"))
    if not all(np.isfinite([dt, h, duration])) or min(dt, h, duration) <= 0:
        raise ValueError("dt, plant_step and duration must be finite and positive")
    for ratio in (dt / h, duration / dt):
        if abs(ratio - round(ratio)) > 1e-7:
            raise ValueError(
                "dt must be a multiple of plant_step; duration a multiple of dt"
            )
    for name in ("camera_hz", "lidar_hz"):
        rate = float(c[name])
        if (
            rate <= 0
            or not np.isfinite(rate)
            or abs(1 / (dt * rate) - round(1 / (dt * rate))) > 1e-7
        ):
            raise ValueError(f"{name} must divide the host rate 1/dt")
    if c["weather"] not in ("clear", "fog", "night", "night_fog"):
        raise ValueError("Choose one of the four supported weather scenarios")
    if c["renderer"] not in ("cpu", "blender"):
        raise ValueError("renderer must be cpu or blender")
