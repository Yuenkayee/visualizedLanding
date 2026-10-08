"""Plan and render stage-balanced, weather-stratified complete landing sorties.
python -m visual_training.data.generate_landing_dataset --plan-only
python -m visual_training.data.generate_landing_dataset --renderer blender
"""

import argparse
import copy
import json
import sys
from pathlib import Path
import cv2
import numpy as np
import yaml
from navigation.common.contracts import CameraCalibration
from visual_modeling.trajectories.landing import (
    make_sortie,
    selected_frames,
    optical_report,
    camera_geometry,
    preview_frames,
    attitude_report,
    STAGES,
)
from visual_modeling.blender.build_h_marker import marker_points
from visual_modeling.blender.export_annotations import (
    make_annotation,
    export_annotations,
)
from visual_modeling.sensors.landing_camera import apply_landing_effects
from visual_training.data.build_dataset import synthetic_frame
from navigation.common.frames import invert, transform_points
from visual_modeling.sensors.airframe_visibility import (
    airframe_primitives,
    airframe_occlusion,
    projected_airframe_mask,
)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def summarize_plan(config, plan):
    report = dict(
        trajectory_reference_frame="ship_heading_forward_port_up_gravity_level",
        visibility_model="in-frame plus provisional fuselage/wheel/strut ray occlusion; not real UH-60 CAD",
        design_camera=config["camera"],
        design_geometry=config["geometry"],
        attitude_design=config["motion"]["attitude"],
        trajectory_hz=config["trajectory_hz"],
        sorties=len(plan),
        stages={},
        split_counts={
            name: sum(s["split"] == name for s in plan)
            for name in ("train", "val", "test")
        },
    )
    for stage in STAGES:
        optics = [entry["optics"][stage] for entry in plan]
        summary = {}
        for key in optics[0]:
            if key == "frames":
                continue
            if key.startswith("max_"):
                summary[key] = max(o[key] for o in optics)
            elif key.startswith("min_"):
                summary[key] = min(o[key] for o in optics)
            else:
                summary["min_" + key] = min(o[key] for o in optics)
        for key in ("centre_visible_fraction", "four_corners_visible_fraction"):
            summary["mean_" + key] = float(np.mean([o[key] for o in optics]))
        report["stages"][stage] = summary
    return report


def plan_dataset(config, output):
    root = Path(output).resolve()
    count = config["sorties_per_weather"]
    ratios = np.array(config["split_ratios"], float)
    if (
        count < 3
        or ratios.shape != (3,)
        or (ratios <= 0).any()
        or not np.isclose(ratios.sum(), 1)
    ):
        raise ValueError(
            "need >=3 sorties per weather and positive train/val/test ratios summing to one"
        )
    events = {}
    event_path = config["mission"].get("window_events_file")
    if event_path:
        events = json.loads(Path(event_path).read_text())
    rng = np.random.default_rng(config["seed"])
    plan = []
    for weather in config["weather"]:
        indices = np.arange(count)
        rng.shuffle(indices)
        val = max(1, int(round(count * ratios[1])))
        test = max(1, int(round(count * ratios[2])))
        if val + test >= count:
            raise ValueError("split leaves no training sorties")
        assignment = {int(i): "val" for i in indices[:val]}
        assignment.update({int(i): "test" for i in indices[val : val + test]})
        for i in range(count):
            sequence = f"landing_{config['seed']}_{weather}_{i:03d}"
            seed = int(rng.integers(0, 2**32 - 1))
            wait = None
            if event_path:
                if sequence not in events:
                    raise ValueError(f"missing window wait for {sequence}")
                wait = events[sequence]
            sortie = make_sortie(config, seed, sequence, weather, wait)
            report = optical_report(sortie, config["marker"])
            path = root / "trajectories" / f"{sequence}.json"
            write_json(path, sortie)
            plan.append(
                dict(
                    sequence_id=sequence,
                    weather=weather,
                    split=assignment.get(i, "train"),
                    trajectory=str(path),
                    optics=report,
                    attitude=attitude_report(sortie),
                )
            )
    calibration, mount = camera_geometry(config)
    write_json(
        root / "calibration" / "design.json",
        dict(
            camera=config["camera"],
            geometry=config["geometry"],
            K=calibration.K.tolist(),
            T_body_camera=mount.tolist(),
            note="Provisional design geometry; replace with calibrated installation",
        ),
    )
    write_json(root / "plan.json", dict(schema_version=2, config=config, sorties=plan))
    write_json(root / "planning_report.json", summarize_plan(config, plan))
    return plan


def _blender_scene(config, weather):
    import bpy
    from visual_modeling.blender.build_ship import build_ship
    from visual_modeling.blender.build_sea_sky import build_sea_sky
    from visual_modeling.blender.configure_camera import configure_camera
    from visual_modeling.blender.landing_environment import (
        prepare_weather,
        make_visual_wake,
    )
    from visual_modeling.blender.build_airframe_proxy import build_airframe_proxy

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    # Full rigs build hundreds of sortie/view scenes; release unused meshes,
    # volumes and materials instead of accumulating them across render batches.
    bpy.data.orphans_purge(do_recursive=True)
    ship = build_ship(dict(marker=config["marker"]))
    r = config["render"]
    ocean = build_sea_sky(
        dict(
            size=r["sea_size_m"],
            grid=r["sea_grid"],
            wave_amplitude=r["wave_amplitude_m"],
            sun_elevation=weather["sun_elevation_deg"],
        )
    )
    cal, _ = camera_geometry(config)
    camera = configure_camera(
        dict(
            width=cal.width,
            height=cal.height,
            fx=cal.K[0, 0],
            fy=cal.K[1, 1],
            cx=cal.K[0, 2],
            cy=cal.K[1, 2],
        )
    )
    prepare_weather(weather, ship, camera)
    make_visual_wake(ship)
    helicopter = build_airframe_proxy(config["geometry"])
    return ship, camera, ocean, helicopter


def _cpu_image(cal, pose, rng, marker, weather, mount, primitives):
    image, mask = synthetic_frame(cal, pose, rng, marker)
    blocked = projected_airframe_mask(cal, pose, mount, primitives)
    image[blocked] = [35, 40, 45]
    mask[blocked] = 0
    if weather["deck_lights"]:
        # CPU smoke fallback only; production uses physical Blender lights.
        image = np.clip(image.astype(float) * 0.12, 0, 255).astype("uint8")
        illuminated = cv2.GaussianBlur(mask.astype(float) / 255, (0, 0), 25)
        image = np.clip(image + illuminated[:, :, None] * 70, 0, 255).astype("uint8")
    if weather["fog_density"]:
        f = 1 - np.exp(-weather["fog_density"] * np.linalg.norm(pose[:3, 3]))
        image = np.clip(image * (1 - f) + np.array([120, 135, 145]) * f, 0, 255).astype(
            "uint8"
        )
    return image, mask


def render_dataset(
    config, plan, output, renderer="blender", preview=False, write_splits=True
):
    root = Path(output).resolve()
    paths = {name: [] for name in ["train", "val", "test"]}
    summary = []
    if renderer not in ("blender", "cpu"):
        raise ValueError("unknown renderer")
    points = marker_points(config["marker"]["width"], config["marker"]["length"])
    scene = None
    preview_weather = set()
    for entry in plan:
        if preview:
            if entry["weather"] in preview_weather:
                continue
            preview_weather.add(entry["weather"])
        sortie = json.loads(Path(entry["trajectory"]).read_text())
        weather = copy.deepcopy(config["weather"][entry["weather"]])
        # Fixed within a sortie, varied independently across sorties.
        weather_rng = np.random.default_rng(
            sortie.get("environment_seed", sortie["seed"]) + 17
        )
        rng = np.random.default_rng(sortie.get("sensor_seed", sortie["seed"]) + 37)
        weather["sun_energy"] *= float(weather_rng.uniform(0.85, 1.15))
        weather["sensor_noise"] *= float(weather_rng.uniform(0.85, 1.15))
        weather["fog_density"] *= float(weather_rng.uniform(0.6, 1.5))
        if renderer == "blender":
            scene = _blender_scene(config, weather)
        cal = CameraCalibration(
            sortie["camera"]["K"], sortie["camera"]["width"], sortie["camera"]["height"]
        )
        mount = np.asarray(sortie["T_body_camera"])
        primitives = airframe_primitives(sortie["geometry"])
        rows = selected_frames(sortie, config["frames_per_stage"])
        if preview:
            rows = preview_frames(
                sortie, config.get("preview", {}).get("pose_selection", "max_attitude")
            )
        for row in rows:
            i = row["frame_id"]
            sequence = sortie["sequence_id"]
            camera_id = sortie.get("camera_id")
            stem = f"{sequence}/" + (f"{camera_id}/" if camera_id else "") + f"{i:06d}"
            image_path = f"rendered/{stem}.png"
            mask_path = f"annotations/{stem}_mask.png"
            label_path = root / f"annotations/{stem}.json"
            pose = np.asarray(row["T_deck_camera"])
            visibility = None
            self_occluded = airframe_occlusion(
                mount[:3, 3], transform_points(mount @ invert(pose), points), primitives
            )
            if renderer == "blender":
                from visual_modeling.blender.apply_poses import apply_poses
                from visual_modeling.blender.render_frame import render_frame
                from visual_modeling.blender.landing_environment import (
                    update_environment,
                )
                from visual_modeling.blender.export_annotations import (
                    keypoint_visibility,
                )
                import bpy

                from mathutils import Matrix

                ship, camera, ocean, helicopter = scene
                apply_poses(ship, camera, np.array(row["T_world_deck"]), pose)
                helicopter.matrix_world = Matrix(row["T_world_body"])
                sea = copy.deepcopy(config["render"])
                sea["sea_level_m"] = -config["geometry"]["H_ship_body_m"][2] - 6
                update_environment(ocean, ship, camera, row["timestamp"], sea)
                bpy.context.view_layer.update()
                render_frame(
                    root / image_path, root / mask_path, config["render"]["samples"]
                )
                image = cv2.cvtColor(
                    cv2.imread(str(root / image_path)), cv2.COLOR_BGR2RGB
                )
                mask = cv2.imread(str(root / mask_path), cv2.IMREAD_GRAYSCALE)
                visibility = keypoint_visibility(points, ship, camera)
            else:
                image, mask = _cpu_image(
                    cal, pose, rng, config["marker"], weather, mount, primitives
                )
                visibility = ~self_occluded
            annotation = make_annotation(
                row["timestamp"],
                sequence,
                i,
                image_path,
                mask_path,
                points,
                pose,
                cal,
                visibility,
            )
            next_index = min(i + 1, len(sortie["rows"]) - 1)
            if next_index == i:
                following = None
                dt = 1 / config["trajectory_hz"]
            else:
                following = np.array(sortie["rows"][next_index]["T_deck_camera"])
                dt = sortie["rows"][next_index]["timestamp"] - row["timestamp"]
            image, mask = apply_landing_effects(
                image,
                mask,
                annotation,
                weather,
                config["camera"],
                rng,
                pose,
                following,
                dt,
                cal,
            )
            (root / image_path).parent.mkdir(parents=True, exist_ok=True)
            (root / mask_path).parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(
                str(root / image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            ) or not cv2.imwrite(str(root / mask_path), mask):
                raise OSError("image/mask write failed")
            annotation.update(
                stage=row["stage"],
                weather=entry["weather"],
                split=entry["split"],
                renderer=renderer,
                gear_relative_heading_m=row["gear_relative_heading_m"],
                gear_relative_ned_m=row["gear_relative_ned_m"],
                gear_relative_deck_m=row["gear_relative_deck_m"],
                T_world_deck=row["T_world_deck"],
                T_world_body=row["T_world_body"],
                T_body_camera=sortie["T_body_camera"],
                geometry=sortie["geometry"],
                sideslip_rad=row["sideslip_rad"],
                touchdown=row["touchdown"],
                window_source=sortie["window_source"],
                self_occluded_keypoints=self_occluded.tolist(),
                helicopter_euler_rad=row["helicopter_euler_rad"],
                helicopter_angular_velocity_body_rad_s=row[
                    "helicopter_angular_velocity_body_rad_s"
                ],
                attitude_oscillation_rad=row["attitude_oscillation_rad"],
                attitude_amplitude_rad=row["attitude_amplitude_rad"],
                attitude_period_s=sortie["attitude_model"]["period_s"],
                stage_progress=row["stage_progress"],
                camera_id=camera_id or "single",
                bundle_id=f"{sequence}/{i:06d}",
            )
            # In-frame geometry does not mean recoverable in dark/fog/glare.
            pixels = int((mask > 127).sum())
            v = sum(annotation["visibility"])
            annotation["geometric_pnp_ready"] = bool(v == 4 and pixels >= 25)
            annotation["marker_pixels"] = pixels
            foreground = mask > 127
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(float)
            ring = (
                cv2.dilate(foreground.astype("uint8"), np.ones((9, 9), "uint8")) > 0
            ) & ~foreground
            contrast = (
                float(abs(np.mean(gray[foreground]) - np.mean(gray[ring])))
                if foreground.any() and ring.any()
                else 0.0
            )
            observable = contrast >= max(6.0, 2 * weather["sensor_noise"])
            annotation["image_quality"] = dict(
                marker_contrast=contrast,
                photometrically_observable=bool(observable),
                criterion="heuristic global H/background contrast; not a detection guarantee",
            )
            annotation["keypoint_train_visibility"] = (
                np.asarray(annotation["visibility"]) & observable
            ).tolist()
            export_annotations(label_path, annotation)
            paths[entry["split"]].append(str(label_path))
            summary.append(
                dict(
                    sequence_id=sequence,
                    stage=row["stage"],
                    weather=entry["weather"],
                    split=entry["split"],
                    visibility=v,
                    marker_pixels=pixels,
                    geometric_pnp_ready=annotation["geometric_pnp_ready"],
                    photometrically_observable=bool(observable),
                    image=str(root / image_path),
                    stage_progress=row["stage_progress"],
                    timestamp=row["timestamp"],
                    attitude_oscillation_rad=row["attitude_oscillation_rad"],
                    camera_id=camera_id or "single",
                    bundle_id=f"{sequence}/{i:06d}",
                )
            )
            print(
                json.dumps(
                    dict(
                        sequence=sequence,
                        stage=row["stage"],
                        frame=i,
                        renderer=renderer,
                    )
                ),
                flush=True,
            )
    if not preview and write_splits:
        for split, labels in paths.items():
            if not labels:
                raise ValueError(f"empty {split} split")
            write_json(root / "splits" / f"{split}.json", labels)
    report = dict(
        renderer=renderer,
        preview_only=preview,
        frames=len(summary),
        by_stage_weather={},
        frames_index=summary,
    )
    for stage in STAGES:
        for weather in config["weather"]:
            subset = [
                r for r in summary if r["stage"] == stage and r["weather"] == weather
            ]
            report["by_stage_weather"][f"{stage}/{weather}"] = dict(
                frames=len(subset),
                geometric_pnp_ready_fraction=float(
                    np.mean([r["geometric_pnp_ready"] for r in subset])
                )
                if subset
                else None,
                photometrically_observable_fraction=float(
                    np.mean([r["photometrically_observable"] for r in subset])
                )
                if subset
                else None,
            )
    write_json(root / "dataset_report.json", report)
    if preview:
        tiles = []
        for item in summary:
            image = cv2.imread(item["image"])
            image = cv2.resize(image, (384, 288))
            cv2.rectangle(image, (0, 0), (384, 25), (10, 10, 10), -1)
            cv2.putText(
                image,
                f"{item['weather']} / {item['stage']}",
                (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            tiles.append(image)
        sheet = np.vstack(
            [np.hstack(tiles[i : i + 3]) for i in range(0, len(tiles), 3)]
        )
        cv2.imwrite(str(root / "preview.png"), sheet)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/landing_dataset.yaml")
    p.add_argument("--output")
    p.add_argument("--renderer", choices=["blender", "cpu"], default="blender")
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--sorties-per-weather", type=int)
    p.add_argument("--frames-per-stage", type=int)
    p.add_argument("--width", type=int)
    p.add_argument("--samples", type=int)
    p.add_argument("--preview", action="store_true")
    args = p.parse_args(argv)
    config = yaml.safe_load(Path(args.config).read_text())
    for name in ["sorties_per_weather", "frames_per_stage"]:
        if getattr(args, name) is not None:
            config[name] = getattr(args, name)
    if args.width:
        ratio = args.width / config["camera"]["width"]
        config["camera"]["height"] = round(config["camera"]["height"] * ratio)
        config["camera"]["width"] = args.width
    if args.samples is not None:
        config["render"]["samples"] = args.samples
    output = args.output or config["output"]
    plan = plan_dataset(config, output)
    if args.plan_only:
        print(
            json.dumps(
                dict(
                    sorties=len(plan),
                    output=str(Path(output).resolve()),
                    optics=plan[0]["optics"],
                    attitude=plan[0]["attitude"],
                ),
                indent=2,
            )
        )
        return
    print(
        json.dumps(
            {
                k: v
                for k, v in render_dataset(
                    config, plan, output, args.renderer, args.preview
                ).items()
                if k != "frames_index"
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:])
