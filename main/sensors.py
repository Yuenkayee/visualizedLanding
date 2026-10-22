"""Causal sensor generation from a single current plant snapshot.

The Blender worker runs in a separate interpreter (server bpy uses Python 3.11,
navigation may use 3.12). Only RGB/point clouds cross back to the navigator.
"""

import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
from shell.render_environment import renderer_environment
import sys
import threading
import numpy as np
from scipy.spatial.transform import Rotation
from main.config import ROOT
from navigation.common.frames import invert
from navigation.common.contracts import GimbalSample
from visual_modeling.trajectories.landing import camera_geometry
from visual_training.data.generate_landing_dataset import _cpu_image, _blender_scene
from visual_modeling.sensors.airframe_visibility import airframe_primitives
from visual_modeling.sensors.lidar_raycast import (
    beam_directions,
    raycast_deck,
    raycast_scene,
)
from visual_modeling.sensors.lidar_degradation import degrade_lidar
from visual_modeling.sensors.landing_camera import apply_landing_effects
from visual_modeling.blender.build_h_marker import marker_points
from visual_modeling.blender.export_annotations import make_annotation

PREFIX = "LANDING_SENSOR_REPLY "


def python_executable(value):
    # Do not resolve the venv's python symlink to its base interpreter: doing
    # so discards the venv and its numpy/bpy installations.
    value = os.path.expanduser(str(value))
    if os.sep not in value:
        found = shutil.which(value)
        if found is None:
            raise FileNotFoundError(value)
        return found
    return os.path.abspath(ROOT / value)


class SensorGenerator:
    def __init__(self, config, landing, lidar, directory):
        self.c, self.landing, self.lidar = config, landing, lidar
        self.directory = Path(directory)
        self.cal, _ = camera_geometry(landing)
        self.weather = landing["weather"][config["weather"]]
        self.primitives = airframe_primitives(landing["geometry"])
        self.points = marker_points(
            landing["marker"]["width"], landing["marker"]["length"]
        )
        self.rng_camera = np.random.default_rng(config["seed"] + 17)
        self.rng_lidar = np.random.default_rng(config["seed"] + 29)
        self.directions = beam_directions(
            **{
                k: lidar[k]
                for k in ("horizontal_fov", "vertical_fov", "columns", "rows")
            }
        )
        self.previous = None
        self.scene = None
        self.metadata = dict(
            renderer=config["renderer"],
            motion_model="causal previous/current rotation; current translational velocity",
        )
        if config["renderer"] == "blender":
            from visual_modeling.blender.cycles_device import configure_cycles

            self.metadata.update(
                configure_cycles(config["render_device"], config["render_threads"])
            )
            self.scene = _blender_scene(landing, self.weather)

    def observe(self, truth, camera, lidar, index):
        t = truth.timestamp
        pose = truth.T_deck_camera
        mount = truth.T_body_camera
        output = dict(timestamp=np.array(t))
        if self.scene is not None:
            import bpy
            from mathutils import Matrix
            from visual_modeling.blender.apply_poses import apply_poses
            from visual_modeling.blender.landing_environment import update_environment

            ship, cam, ocean, helicopter = self.scene
            apply_poses(ship, cam, truth.T_world_deck, pose)
            helicopter.matrix_world = Matrix(truth.T_world_deck @ pose @ invert(mount))
            sea = {
                **self.landing["render"],
                "sea_level_m": -self.landing["geometry"]["H_ship_body_m"][2] - 6,
            }
            update_environment(ocean, ship, cam, t, sea)
            bpy.context.view_layer.update()
        if camera:
            if self.scene is None:
                image, mask = _cpu_image(
                    self.cal,
                    pose,
                    self.rng_camera,
                    self.landing["marker"],
                    self.weather,
                    mount,
                    self.primitives,
                )
            else:
                import cv2
                from visual_modeling.blender.render_frame import render_frame

                image_file, mask_file = (
                    self.directory / "current_rgb.png",
                    self.directory / "current_mask.png",
                )
                render_frame(image_file, mask_file, self.landing["render"]["samples"])
                image = cv2.cvtColor(cv2.imread(str(image_file)), cv2.COLOR_BGR2RGB)
                mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            # Extrapolate only the exposure/readout interval. Never request a
            # future plant state before writing current navigation feedback.
            exposure = max(
                self.landing["camera"].get("exposure_s", 0),
                self.landing["camera"].get("readout_s", 0),
                0.001,
            )
            following = pose.copy()
            following[:3, 3] += truth.velocity_deck * exposure
            if self.previous is not None and t > self.previous[0]:
                angular = Rotation.from_matrix(
                    self.previous[1][:3, :3].T @ pose[:3, :3]
                ).as_rotvec() / (t - self.previous[0])
                following[:3, :3] = (
                    pose[:3, :3] @ Rotation.from_rotvec(angular * exposure).as_matrix()
                )
            annotation = make_annotation(
                t, "live", index, "", "", self.points, pose, self.cal
            )
            image, _ = apply_landing_effects(
                image,
                mask,
                annotation,
                self.weather,
                self.landing["camera"],
                self.rng_camera,
                pose,
                following,
                exposure,
                self.cal,
            )
            output["image"] = image
        if lidar:
            T = pose @ self.cal.camera_lidar_at(GimbalSample(t, mount), t)
            cloud = (
                raycast_scene(
                    truth.T_world_deck @ T,
                    self.directions,
                    max_range=self.lidar["max_range"],
                )
                if self.scene is not None
                else raycast_deck(T, self.directions, max_range=self.lidar["max_range"])
            )
            output["points"] = degrade_lidar(
                cloud,
                rng=self.rng_lidar,
                **{
                    k: self.lidar[k]
                    for k in ("range_std", "dropout", "min_range", "max_range")
                },
            )
        self.previous = (t, pose.copy())
        return output


class Sensors:
    def __init__(self, config, landing, lidar, output):
        self.c, self.landing, self.lidar = config, landing, lidar
        self.directory = Path(output) / "sensors"
        self.directory.mkdir()
        self.worker = None
        self.log = None

    def __enter__(self):
        if self.c["renderer"] == "cpu":
            self.generator = SensorGenerator(
                self.c, self.landing, self.lidar, self.directory
            )
            self.metadata = self.generator.metadata
        else:
            python = self.c.get("render_python") or os.environ.get("RENDER_PYTHON")
            if not python:
                server = ROOT / ".venv-render-server/bin/python"
                python = (
                    str(server)
                    if sys.platform == "linux" and server.exists()
                    else sys.executable
                )
            self.log = (self.directory.parent / "renderer.log").open("w", buffering=1)
            self.messages = queue.Queue()
            self.worker = subprocess.Popen(
                [python_executable(python), "-u", "-m", "main.sensor_worker"],
                cwd=ROOT,
                env=renderer_environment(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            self.reader = threading.Thread(target=self._read, daemon=True)
            self.reader.start()
            try:
                self.metadata = self._request(
                    dict(
                        config=self.c,
                        landing=self.landing,
                        lidar=self.lidar,
                        directory=str(self.directory),
                    )
                )
            except BaseException:
                self.__exit__()
                raise
        return self

    def _read(self):
        try:
            for line in self.worker.stdout:
                if line.startswith(PREFIX):
                    self.messages.put(json.loads(line[len(PREFIX) :]))
                else:
                    self.log.write(line)
        finally:
            self.messages.put(dict(error="Renderer exited; inspect renderer.log"))

    def _request(self, message):
        self.worker.stdin.write(json.dumps(message, allow_nan=False) + "\n")
        self.worker.stdin.flush()
        try:
            response = self.messages.get(timeout=self.c.get("sensor_timeout_s", 300))
        except queue.Empty as exc:
            raise TimeoutError("Renderer timed out; inspect renderer.log") from exc
        if "error" in response:
            raise RuntimeError(response["error"])
        return response

    def observe(self, truth, camera, lidar, index):
        filename = self.directory / (
            f"{index:06d}.npz" if self.c.get("save_sensors") else "current.npz"
        )
        if self.worker is None:
            result = self.generator.observe(truth, camera, lidar, index)
            if self.c.get("save_sensors"):
                np.savez_compressed(filename, **result)
        else:
            self._request(
                dict(
                    truth=truth.to_dict(),
                    camera=camera,
                    lidar=lidar,
                    index=index,
                    output=str(filename),
                )
            )
            with np.load(filename, allow_pickle=False) as data:
                result = {k: data[k].copy() for k in data.files}
        if abs(float(result["timestamp"]) - truth.timestamp) > 1e-8:
            raise ValueError("Stale sensor response")
        return result

    def __exit__(self, *exc):
        if self.worker is not None:
            if self.worker.poll() is None:
                try:
                    self.worker.stdin.write('{"stop":true}\n')
                    self.worker.stdin.flush()
                    self.worker.wait(timeout=10)
                except (BrokenPipeError, subprocess.TimeoutExpired):
                    self.worker.terminate()
                    try:
                        self.worker.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.worker.kill()
                        self.worker.wait()
            self.reader.join(timeout=5)
            self.worker.stdin.close()
            self.worker.stdout.close()
            self.worker = None
        if self.log is not None:
            self.log.close()
            self.log = None
