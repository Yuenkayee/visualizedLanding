"""Four synchronized rigid views of each landing sortie; complete-sortie splits."""

import argparse
import copy
import json
import sys
from pathlib import Path
import cv2
import numpy as np
import yaml
from navigation.common.camera_rig import CameraRig, RigCamera
from navigation.common.frames import invert, ned_to_enu
from visual_modeling.trajectories.landing import (
    camera_geometry,
    optical_report,
    STAGES,
    _max_failure_duration,
    cruise_progress,
)
from visual_training.data.generate_landing_dataset import (
    plan_dataset,
    render_dataset,
    summarize_plan,
    write_json,
)


def load_config(path):
    override = yaml.safe_load(Path(path).read_text())
    config = yaml.safe_load(Path(override.pop("base_config")).read_text())
    config.update(override)
    return config


def offset_landing(sortie, settings):
    """Smooth touchdown offset in the actual deck plane, with rigid gear lever."""
    rng = np.random.default_rng(sortie["seed"] + 101)
    extent = np.asarray(settings["half_extent_deck_m"], float)
    fraction = settings["centred_fraction"]
    if extent.shape != (2,) or (extent < 0).any() or not 0 <= fraction <= 1:
        raise ValueError("invalid touchdown offset distribution")
    offset = (
        np.r_[rng.uniform(-extent, extent), 0]
        if rng.random() >= fraction
        else np.zeros(3)
    )
    A = ned_to_enu()[:3, :3]
    heading = sortie["rows"][0]["ship_euler_rad"][2]
    H = np.array(
        [
            [np.cos(heading), np.sin(heading), 0],
            [np.sin(heading), -np.cos(heading), 0],
            [0, 0, -1],
        ]
    )
    for row in sortie["rows"]:
        if row["stage"] != "touchdown":
            continue
        weight = cruise_progress(row["stage_progress"])
        delta_deck = weight * offset
        delta_world = np.asarray(row["T_world_deck"])[:3, :3] @ delta_deck
        body = np.asarray(row["T_world_body"])
        body[:3, 3] += delta_world
        row["T_world_body"] = body.tolist()
        row["helicopter_cg_ned_m"] = (A @ body[:3, 3]).tolist()
        row["gear_relative_deck_m"] = (
            np.asarray(row["gear_relative_deck_m"]) + delta_deck
        ).tolist()
        ned = np.asarray(row["gear_relative_ned_m"]) + A @ delta_world
        row["gear_relative_ned_m"] = ned.tolist()
        row["gear_relative_heading_m"] = (H.T @ ned).tolist()
        row["gear_relative_nominal_heading_m"] = (
            np.asarray(row["gear_relative_nominal_heading_m"]) + H.T @ A @ delta_world
        ).tolist()
        row["T_deck_camera"] = (
            invert(row["T_world_deck"]) @ body @ np.asarray(sortie["T_body_camera"])
        ).tolist()
    # Update CG/body velocity to retain consistency after adding the offset.
    rows = sortie["rows"]
    t = np.array([r["timestamp"] for r in rows])
    v = np.gradient(np.array([r["helicopter_cg_ned_m"] for r in rows]), t, axis=0)
    for row, velocity in zip(rows, v):
        R = A @ np.asarray(row["T_world_body"])[:3, :3]
        vb = R.T @ velocity
        row["helicopter_velocity_body_knots"] = (vb * 3600 / 1852).tolist()
        row["sideslip_rad"] = float(np.arctan2(vb[1], np.hypot(vb[0], vb[2])))
    sortie["touchdown_offset_deck_m"] = offset.tolist()
    return sortie


def camera_sortie(base, config, key, index):
    sortie = copy.deepcopy(base)
    camera_config = copy.deepcopy(config)
    camera_config["camera"].update(config["cameras"][key])
    cal, mount = camera_geometry(
        camera_config, np.random.default_rng(base["seed"] + 200 + index)
    )
    sortie.update(
        camera_id=key,
        environment_seed=base["seed"],
        sensor_seed=base["seed"] + 300 + index,
        T_body_camera=mount.tolist(),
        camera=dict(
            K=cal.K.tolist(),
            width=cal.width,
            height=cal.height,
            distortion=cal.distortion.tolist(),
        ),
    )
    for row in sortie["rows"]:
        row["T_deck_camera"] = (
            invert(row["T_world_deck"]) @ np.asarray(row["T_world_body"]) @ mount
        ).tolist()
    return sortie, camera_config


def nominal_rig(config):
    cameras = []
    for key, settings in config["cameras"].items():
        cc = copy.deepcopy(config)
        cc["camera"].update(settings)
        cal, mount = camera_geometry(cc)
        cameras.append(RigCamera(key, cal, mount))
    rig = CameraRig(cameras, config["reference_camera_id"])
    return CameraRig.from_dict(
        rig.to_dict(), config.get("T_reference_lidar", np.eye(4))
    )


def union_coverage(reports):
    result = {}
    for stage in STAGES:
        frames = [r[stage]["frames"] for r in reports.values()]
        centres = np.array([[f["centre_visible"] for f in stream] for stream in frames])
        corners = np.array(
            [[f["keypoints_visible"] for f in stream] for stream in frames]
        )
        centre = centres.any(axis=0)
        single = corners.all(axis=2).any(axis=0)
        joint = corners.any(axis=0).all(axis=1)
        t = [f["timestamp"] for f in frames[0]]
        result[stage] = dict(
            any_centre_visible_fraction=float(centre.mean()),
            any_single_camera_four_corners_fraction=float(single.mean()),
            joint_keypoints_covered_fraction=float(joint.mean()),
            max_all_centres_invisible_s=_max_failure_duration(t, centre),
            max_no_single_camera_four_corners_s=_max_failure_duration(t, single),
        )
    return result


def plan_multi_dataset(config, output):
    root = Path(output).resolve()
    rig = nominal_rig(config)
    if config["reference_camera_id"] != next(iter(config["cameras"])):
        raise ValueError("put the reference camera first in cameras")
    write_json(root / "calibration" / "rig_design.json", rig.to_dict())
    # Reuse the same weather-stratified sortie seeds and mission generator.
    reference_config = copy.deepcopy(config)
    reference_config["camera"].update(config["cameras"][config["reference_camera_id"]])
    original = plan_dataset(reference_config, root)
    entries, views = [], {key: [] for key in config["cameras"]}
    configs = {}
    for entry in original:
        base = offset_landing(
            json.loads(Path(entry["trajectory"]).read_text()),
            config["touchdown_offset"],
        )
        write_json(Path(entry["trajectory"]), base)
        reports, rig_cameras = {}, []
        for index, key in enumerate(config["cameras"]):
            sortie, cc = camera_sortie(base, config, key, index)
            configs[key] = cc
            path = root / "trajectories" / key / f"{base['sequence_id']}.json"
            write_json(path, sortie)
            reports[key] = optical_report(sortie, config["marker"], include_frames=True)
            camera = sortie["camera"]
            from navigation.common.contracts import CameraCalibration

            rig_cameras.append(
                RigCamera(
                    key,
                    CameraCalibration(
                        camera["K"],
                        camera["width"],
                        camera["height"],
                        camera["distortion"],
                    ),
                    sortie["T_body_camera"],
                )
            )
            views[key].append(dict(entry, trajectory=str(path)))
        rig = CameraRig(rig_cameras, config["reference_camera_id"])
        rig = CameraRig.from_dict(
            rig.to_dict(), config.get("T_reference_lidar", np.eye(4))
        )
        write_json(root / "calibration" / f"{base['sequence_id']}.json", rig.to_dict())
        union = union_coverage(reports)
        compact = {
            key: {
                s: {k: v for k, v in values.items() if k != "frames"}
                for s, values in r.items()
            }
            for key, r in reports.items()
        }
        entry["optics"] = compact[config["reference_camera_id"]]
        # The reusable single-view plan also refers to the adjusted C0 trajectory.
        write_json(
            Path(entry["trajectory"]),
            json.loads(
                Path(views[config["reference_camera_id"]][-1]["trajectory"]).read_text()
            ),
        )
        entries.append(
            dict(
                sequence_id=base["sequence_id"],
                weather=entry["weather"],
                split=entry["split"],
                coverage=union,
                cameras=compact,
                touchdown_offset_deck_m=base["touchdown_offset_deck_m"],
            )
        )
    summary = dict(
        sorties=len(entries),
        cameras=len(views),
        planned_images=len(entries) * config["frames_per_stage"] * 3 * len(views),
        reference_camera_id=config["reference_camera_id"],
        coverage={},
        split_counts=original
        and {
            s: sum(e["split"] == s for e in entries) for s in ("train", "val", "test")
        },
    )
    for stage in STAGES:
        summary["coverage"][stage] = {
            ("max_" if k.startswith("max_") else "min_") + k: (
                max if k.startswith("max_") else min
            )(e["coverage"][stage][k] for e in entries)
            for k in entries[0]["coverage"][stage]
        }
        summary["coverage"][stage].update(
            {
                "mean_" + k: float(np.mean([e["coverage"][stage][k] for e in entries]))
                for k in entries[0]["coverage"][stage]
                if k.endswith("fraction")
            }
        )
    write_json(
        root / "multi_plan.json", dict(config=config, sorties=entries, summary=summary)
    )
    write_json(
        root / "plan.json",
        dict(schema_version=2, config=reference_config, sorties=original),
    )
    write_json(
        root / "planning_report.json", summarize_plan(reference_config, original)
    )
    return views, configs, summary


def generate_lidar_and_bundles(config, root, indexes, views):
    from visual_modeling.sensors.lidar_raycast import beam_directions, raycast_deck
    from visual_modeling.sensors.lidar_degradation import degrade_lidar

    lidar = yaml.safe_load(
        Path(config.get("lidar_config", "configs/lidar.yaml")).read_text()
    )
    directions = beam_directions(
        **{k: lidar[k] for k in ("horizontal_fov", "vertical_fov", "columns", "rows")}
    )
    extrinsic = np.asarray(config.get("T_reference_lidar", np.eye(4)))
    rig = CameraRig.from_dict(
        json.loads((root / "calibration" / "rig_design.json").read_text()), extrinsic
    )
    by_sequence = {}
    for item in indexes:
        by_sequence.setdefault(item["sequence_id"], {}).setdefault(
            item["bundle_id"], []
        ).append(item)
    reference = config["reference_camera_id"]
    for sequence, groups in by_sequence.items():
        entry = next(e for e in views[reference] if e["sequence_id"] == sequence)
        sortie = json.loads(Path(entry["trajectory"]).read_text())
        rng = np.random.default_rng(sortie["seed"] + 501)
        bundles = []
        for bundle_id, items in sorted(groups.items()):
            frame_id = int(bundle_id.split("/")[-1])
            row = sortie["rows"][frame_id]
            cloud = degrade_lidar(
                raycast_deck(
                    np.asarray(row["T_deck_camera"])
                    @ rig.reference.calibration.T_camera_lidar,
                    directions,
                    max_range=lidar["max_range"],
                ),
                rng=rng,
                **{
                    k: lidar[k]
                    for k in ("range_std", "dropout", "min_range", "max_range")
                },
            )
            path = root / "lidar" / sequence / f"{frame_id:06d}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, timestamp=row["timestamp"], points=cloud)
            bundles.append(
                dict(
                    bundle_id=bundle_id,
                    timestamp=row["timestamp"],
                    stage=row["stage"],
                    weather=sortie["weather"],
                    ship_euler_rad=row["ship_euler_rad"],
                    lidar=str(path.relative_to(root)),
                    annotations={
                        i[
                            "camera_id"
                        ]: f"annotations/{sequence}/{i['camera_id']}/{frame_id:06d}.json"
                        for i in items
                    },
                )
            )
        write_json(root / "bundles" / f"{sequence}.json", bundles)


def generate(
    config,
    output,
    renderer="blender",
    plan_only=False,
    preview=False,
    render_runtime=None,
):
    root = Path(output).resolve()
    # Do not overwrite a completed dataset with a preview or a partially rendered rig.
    if (root / "splits").exists():
        raise FileExistsError(
            f"{root}/splits already exists. Choose a new --output directory; "
            "existing training data will not be overwritten."
        )
    print(json.dumps(dict(event="planning_started", output=str(root))), flush=True)
    views, configs, summary = plan_multi_dataset(config, root)
    print(json.dumps(dict(event="planning_complete", **summary)), flush=True)
    if plan_only:
        return summary
    indexes = []
    for key, entries in views.items():
        print(json.dumps(dict(event="camera_started", camera=key)), flush=True)
        report = render_dataset(
            configs[key], entries, root, renderer, preview, write_splits=False
        )
        indexes.extend(report["frames_index"])
    groups = {}
    for item in indexes:
        groups.setdefault(item["bundle_id"], []).append(item)
    generate_lidar_and_bundles(config, root, indexes, views)
    split_paths = {}
    if not preview:
        for split in ("train", "val", "test"):
            paths = [
                str(
                    root
                    / "annotations"
                    / x["sequence_id"]
                    / x["camera_id"]
                    / (Path(x["image"]).stem + ".json")
                )
                for x in indexes
                if x["split"] == split
            ]
            if not paths:
                raise ValueError(f"empty {split} split")
            split_paths[split] = paths
    report = dict(
        renderer=renderer,
        preview_only=preview,
        frames=len(indexes),
        bundles=len(groups),
        any_geometric_pnp_ready_fraction=float(
            np.mean(
                [any(x["geometric_pnp_ready"] for x in xs) for xs in groups.values()]
            )
        ),
        frames_index=indexes,
        planning=summary,
        render_runtime=render_runtime,
        split_image_counts={name: len(paths) for name, paths in split_paths.items()},
    )
    write_json(root / "multi_dataset_report.json", report)
    write_json(root / "dataset_report.json", report)
    if split_paths:
        # Publish all three indexes only after all views and LiDAR bundles succeed.
        staging = root / ".splits_pending"
        for name, paths in split_paths.items():
            write_json(staging / f"{name}.json", paths)
        staging.rename(root / "splits")
    if preview:
        tiles = []
        for bundle, items in sorted(groups.items()):
            for item in sorted(items, key=lambda x: x["camera_id"]):
                img = cv2.resize(cv2.imread(item["image"]), (320, 240))
                cv2.rectangle(img, (0, 0), (320, 24), (10, 10, 10), -1)
                cv2.putText(
                    img,
                    f"{item['weather']} {item['stage']} {item['camera_id']}",
                    (5, 17),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                )
                tiles.append(img)
        cv2.imwrite(
            str(root / "multi_preview.png"),
            np.vstack(
                [
                    np.hstack(tiles[i : i + len(views)])
                    for i in range(0, len(tiles), len(views))
                ]
            ),
        )
    return {k: v for k, v in report.items() if k != "frames_index"}


def main(argv=None, render_runtime=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/multi_camera_dataset.yaml")
    p.add_argument("--output")
    p.add_argument("--renderer", choices=["cpu", "blender"], default="blender")
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--preview", action="store_true")
    for flag in ("sorties-per-weather", "frames-per-stage", "width", "samples"):
        p.add_argument("--" + flag, type=int)
    a = p.parse_args(argv)
    c = load_config(a.config)
    for key in ("sorties_per_weather", "frames_per_stage"):
        if getattr(a, key) is not None:
            c[key] = getattr(a, key)
    if a.width:
        c["camera"]["height"] = round(
            c["camera"]["height"] * a.width / c["camera"]["width"]
        )
        c["camera"]["width"] = a.width
    if a.samples is not None:
        c["render"]["samples"] = a.samples
    print(
        json.dumps(
            generate(
                c,
                a.output or c["output"],
                a.renderer,
                a.plan_only,
                a.preview,
                render_runtime=render_runtime,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:])
