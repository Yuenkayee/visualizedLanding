"""Causal fixed-record navigation replay with separate scoring/fault channels."""
from copy import deepcopy
import json
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

from main.navigation_test_data import (NED_FROM_WORLD, clean, digest, read_jsonl, state_from_config, write_json)
from navigation.common.cg_signal import CgSignal, relative_cg
from navigation.common.contracts import CameraCalibration, CameraFrame, GimbalReference, GimbalSample, LidarFrame
from navigation.common.frames import invert
from navigation.common.reference_points import body_point_estimate
from navigation.pose.solve_pnp import project_points
from visual_modeling.blender.build_h_marker import marker_points


def fault_active(fault, row, rows):
    if not fault or "stage" not in fault:
        return True
    if "_interval" in fault:
        start,end=fault["_interval"]
        return start <= row["timestamp"] < end-1e-9
    stage = [r for r in rows if r["stage"] == fault["stage"]]
    if not stage:
        return False
    start = stage[0]["timestamp"] + fault.get("start_fraction",0.) * (
        stage[-1]["timestamp"]+stage[-1]["dt"]-stage[0]["timestamp"])
    return start <= row["timestamp"] < start + fault["duration_s"] - 1e-9


def observation(row, data, case, rows):
    """Change only the indicated measurement channel, never its scoring truth."""
    encoder = GimbalSample(**row["gimbal"])
    camera, lidar = row["sensors"]["camera"], row["sensors"]["lidar"]
    image, points = data.get("image"), data.get("points")
    fault = case.get("fault", {})
    kind = fault.get("kind") if fault_active(fault,row,rows) else None
    rng = lambda channel: np.random.default_rng(np.random.SeedSequence([case["seed"],row["iteration"],channel]))
    events = []
    if camera and (kind in ("camera_gap","joint_gap") or
                   (kind=="camera_dropout" and rng(81).random()<fault["probability"])):
        camera=False;events.append("camera_missing")
    if lidar and (kind=="joint_gap" or (kind=="lidar_dropout" and rng(82).random()<fault["probability"])):
        lidar=False;events.append("lidar_missing")
    if lidar and kind=="lidar_point_loss":
        points=points[rng(83).random(len(points)) >= fault["probability"]]
        events.append("lidar_point_loss")
    if kind=="mount_bias":
        encoder.T_body_camera[:3,3] += np.asarray(fault["offset_m"])
        encoder.T_body_camera[:3,:3] = encoder.T_body_camera[:3,:3] @ Rotation.from_euler("xyz",fault["rotation_deg"],degrees=True).as_matrix()
        events.append("mount_bias")
    if kind=="stale_encoder":
        encoder.timestamp=max(0.,encoder.timestamp-fault["offset_s"])
        events.append("stale_encoder")
    return encoder,camera,lidar,image,points,events


def image_quality(data, detection, cal, pose, points):
    """Labels are consumed AFTER navigation, solely for front-end scoring."""
    pixels = np.asarray(data.get("label_keypoints",project_points(points,pose,cal)))
    size = float(np.min(np.ptp(pixels,axis=0)))
    margin = float(np.min(np.c_[pixels[:,0],pixels[:,1],cal.width-pixels[:,0],cal.height-pixels[:,1]]))
    hull=cv2.convexHull(pixels.astype(np.float32))
    area=float(cv2.contourArea(hull))
    _,poly=cv2.intersectConvexConvex(hull, np.array([[0,0],[cal.width,0],[cal.width,cal.height],[0,cal.height]],np.float32))
    inside=0. if poly is None else float(cv2.contourArea(poly))
    q=dict(target_scale_px=size,edge_margin_px=margin,truncation_fraction=float(np.clip(1-inside/max(area,1e-9),0,1)),
           mask_iou=None,keypoint_error_px=None,keypoint_error_normalized=None,
           labels_available="label_mask" in data)
    if "label_mask" in data:
        truth=data["label_mask"]>0
        predicted=np.zeros_like(truth) if detection is None else detection.mask>0
        union=np.count_nonzero(truth | predicted)
        q["mask_iou"] = float(np.count_nonzero(truth & predicted)/union) if union else None
    if detection is not None and "label_visibility" in data:
        vis=data["label_visibility"].astype(bool)
        kp=detection.keypoints
        if not detection.ordered:
            a,b=linear_sum_assignment(np.linalg.norm(pixels[:,None,:]-kp[None,:,:],axis=2))
            kp=kp[b[np.argsort(a)]]
        if vis.any():
            err=float(np.mean(np.linalg.norm(kp[vis]-pixels[vis],axis=1)))
            q.update(keypoint_error_px=err,keypoint_error_normalized=err/size if size>1e-6 else None,
                     visible_keypoints=int(vis.sum()))
    return q


def replay_record(record, target, case, config, checkpoint, detector=None):
    from main.navigation_test_methods import make_method
    record, target=Path(record),Path(target)
    meta=json.loads((record/"run.json").read_text())
    rows=list(read_jsonl(record/"steps.jsonl"))
    cal=CameraCalibration(**meta["camera"])
    geometry=meta["geometry"]
    points=marker_points(meta["marker"]["width"],meta["marker"]["length"])
    nav={**meta["navigation"],"checkpoint":str(checkpoint),"device":config["device"]}
    case=deepcopy(case)
    fault=case.get("fault",{})
    if "stage" in fault:
        stage=[r for r in rows if r["stage"]==fault["stage"]]
        start=(stage[0]["timestamp"]+fault.get("start_fraction",0.)*(stage[-1]["timestamp"]+stage[-1]["dt"]-stage[0]["timestamp"])) if stage else float("inf")
        fault["_interval"]=[start,start+fault["duration_s"]]
    if fault.get("kind")=="initial_position_bias":
        nav["initial_position"]=(np.asarray(nav["initial_position"])+fault["offset_m"]).tolist()
    if fault.get("kind")=="initial_attitude_bias":
        nav["initial_rotation"]=(np.asarray(nav["initial_rotation"]) @ Rotation.from_euler("xyz",fault["rotation_deg"],degrees=True).as_matrix()).tolist()
    method=case["method"]
    pipe=make_method(method,cal,points,nav,state_from_config(nav,rows[0]["timestamp"]),detector=detector)
    signal=CgSignal(meta.get("cg_signal"))
    target.mkdir(parents=True,exist_ok=True)
    run=dict(status="running",method=method,case=case,record=str(record),
        input_record_sha256=digest(record/"record_hashes.json") if (record/"record_hashes.json").exists() else None,
        attempt_id=config.get("attempt_id"),runtime=config.get("runtime"),recording_runtime=meta.get("runtime"),
        recording_attempt_id=meta.get("attempt_id"),
        planned_duration_s=meta["planned_duration_s"],completed_duration_s=0.,
        start_timestamp=rows[0]["timestamp"],
        device=config["device"],torch_threads=config["torch_threads"],render_threads=config["render_threads"],
        execution=config.get("execution",{}),
        resolution=[cal.width,cal.height],precision="float32",gpu_synchronization="before and after inference blocks",
        fault_active_steps=0,
        excluded_from_formal=case.get("excluded_from_formal",False),
        navigation_applied_to_control=False, navigation=pipe.config,
        covariance_note="NED using communicated ship attitude assumed exact; shaped output has no invented covariance",
        output_direction="helicopter minus ship (CG); helicopter gear minus H centre (gear)",
        N1_policy="camera times only, no velocity or held/shaped output", trajectory_source=meta.get("trajectory_source"))
    write_json(target/"run.json",run)
    previous_R=None
    pending=[]
    started=time.perf_counter()
    try:
        with (target/"steps.jsonl").open("w",buffering=1) as stream:
            for row in rows:
                wall=time.perf_counter();t=row["timestamp"];dt=row["dt"]
                pipe.reset_telemetry()
                if fault and fault_active(fault,row,rows):
                    run["fault_active_steps"]+=1
                data={}
                if row["sensors"]["camera"] or row["sensors"]["lidar"]:
                    filename=Path(row["sensor_file"])
                    if not filename.is_absolute():filename=record/filename
                    with np.load(filename,allow_pickle=False) as arrays:
                        data={k:arrays[k].copy() for k in arrays.files}
                    if abs(float(data["timestamp"])-t)>1e-8:raise ValueError(f"Stale recorded sensor: {filename}")
                encoder,camera,lidar,image,cloud,events=observation(row,data,case,rows)
                # Delayed packets are delivered at their true arrival time, then
                # rejected by the existing monotonic-time protection. No retiming.
                kind=fault.get("kind")
                if kind=="delayed_camera" and camera:
                    pending.append((t+fault["delay_s"],"camera",CameraFrame(t,image,encoder)));camera=False
                if kind=="delayed_lidar" and lidar:
                    pending.append((t+fault["delay_s"],"lidar",LidarFrame(t,cloud,encoder)));lidar=False
                pipe.process_gimbal(encoder,t)
                delayed_rejected=0
                while pending and pending[0][0] <= t+1e-8:
                    _,channel,packet=pending.pop(0)
                    if abs(packet.timestamp-t)>1e-8:
                        delayed_rejected+=1;events.append(channel+"_delayed_rejected")
                    else:
                        raise AssertionError("Delay study requires a nonzero arrival delay")
                if camera:pipe.process_camera(CameraFrame(t,image,encoder))
                if lidar:pipe.process_lidar(LidarFrame(t,cloud,encoder))
                estimate=pipe.estimate()
                diag=deepcopy(pipe.diagnostics if estimate is None else estimate.diagnostics)
                diag.update(delayed_packets_rejected=delayed_rejected, fault_events=events)
                diag["detector_kind"]="classical" if method=="N0" else "learned"
                if camera:
                    diag["pnp_solved"]=bool(pipe.raw_pose is not None or any(
                        a.get("reprojection_error") is not None for a in diag.get("vision_attempts",[])))
                    if method=="N0" and diag.get("vision_source")=="learned":
                        diag["vision_source"]="classical"
                if not camera:
                    for key in list(diag):
                        if key.startswith("vision_") and key not in ("vision_recovery_pending","vision_recovery_count","vision_recovery_required"):
                            diag.pop(key)
                    diag["vision_accepted"]=False
                if not lidar:
                    for key in ("lidar_mahalanobis","plane_rms","lidar_rejection"):
                        diag.pop(key,None)
                    diag["lidar_accepted"]=False
                truth=row["truth"];T=np.asarray(truth["T_deck_camera"])
                truth_R=NED_FROM_WORLD @ np.asarray(truth["T_world_deck"])[:3,:3]
                R=np.asarray(row.get("ship_rotation_ned_deck",truth_R))
                if kind=="ship_attitude_bias":
                    R=R @ Rotation.from_euler("xyz",fault["rotation_deg"],degrees=True).as_matrix()
                omega=row.get("ship_angular_velocity_deck")
                if omega is None:
                    omega=np.zeros(3) if previous_R is None or dt<=0 else Rotation.from_matrix(previous_R.T @ R).as_rotvec()/dt
                previous_R=R
                body=T @ invert(truth["T_body_camera"])
                cg_truth=truth_R @ (body[:3,3] + np.diag([1.,-1.,-1.]) @ geometry["H_ship_body"])
                values=dict(timestamp=t,dt=dt,stage=row["stage"],iteration=row["iteration"],
                    camera_sample=bool(row["sensors"]["camera"]),camera_received=camera,lidar_sample=bool(row["sensors"]["lidar"]),lidar_received=lidar,
                    gear_truth_ned=truth_R @ truth["gear_position_deck"],cg_truth_ned=cg_truth,
                    gear_estimate_ned=None,gear_covariance_ned=None,cg_estimate_ned=None,cg_covariance_ned=None,
                    cg_shaped_ned=None,attitude_error_rad=None,navigation_valid=False,cg_signal_valid=False,estimate_available=False)
                initialized = (estimate is not None and (method=="N1" or pipe.health.last_measurement is not None))
                synchronized = encoder.valid and abs(encoder.timestamp-t)<1e-8
                if initialized and synchronized:
                    p,_,cov=body_point_estimate(estimate,geometry["gear_body"])
                    if method!="N1" and pipe.filter_point_body is not None and np.allclose(pipe.filter_point_body,geometry["gear_body"]):
                        cov=pipe.state.P[:3,:3] # same physical filter reference, no double round-trip lever variance
                    cg=relative_cg(estimate,geometry,R,omega)
                    values.update(estimate_available=True,navigation_valid=bool(estimate.healthy),
                        gear_estimate_ned=R @ p,gear_covariance_ned=R @ cov @ R.T,
                        cg_estimate_ned=cg["cg_relative_estimate_ned"],cg_covariance_ned=cg["cg_relative_covariance_ned"],
                        attitude_error_rad=float(Rotation.from_matrix(T[:3,:3].T @ estimate.T_deck_camera[:3,:3]).magnitude()))
                    if method!="N1":
                        old=signal.position.copy();old_feed=signal.feed_velocity.copy();was_initialized=signal.initialized
                        shaped=signal.step(t,cg["cg_relative_estimate_ned"],cg["navigation_velocity_cg_ned_mps"],estimate.healthy,t)
                        if signal.initialized:
                            values.update(cg_shaped_ned=shaped["cg_relative_estimate_50hz_ned"],cg_signal_valid=shaped["cg_signal_valid"])
                        diag["cg_tracking_error_m"]=signal.tracking_error if signal.initialized else None
                        # For fresh valid inputs the predictor uses trapezoidal velocity.
                        correction=signal.position-old-.5*(old_feed+signal.feed_velocity)*dt
                        diag["cg_correction_limited"]=bool(was_initialized and estimate.healthy and
                            np.linalg.norm(correction)>=signal.parameters[1]*dt-1e-8)
                elif method!="N1":
                    shaped=signal.step(t,np.zeros(3),np.zeros(3),False,t)
                    if signal.initialized:
                        values["cg_shaped_ned"]=shaped["cg_relative_estimate_50hz_ned"]
                    diag["cg_tracking_error_m"]=None
                reference=row.get("gimbal_reference")
                if reference:
                    pipe.process_gimbal_reference(GimbalReference(**reference))
                navigation_s=time.perf_counter()-wall
                quality=dict(gimbal_saturated=encoder.saturated,gimbal_reference_error_rad=row.get("gimbal_reference_error_rad"),
                    lidar_points=int(len(cloud)) if lidar else None,
                    plane_inlier_ratio=diag.get("plane_inlier_ratio"),plane_rms=diag.get("plane_rms"))
                if row["sensors"]["camera"]:
                    quality.update(image_quality(data,pipe.last_detection if camera else None,cal,T,points))
                timing={**row.get("timing",{}),**pipe.timing,"navigation_and_io_s":navigation_s,
                        "scoring_s":time.perf_counter()-wall-navigation_s,"total_s":time.perf_counter()-wall}
                values.update(diagnostics=diag,quality=quality,timing=timing)
                if pipe.raw_pose is not None:
                    values["raw_pnp_pose"] = dict(T_deck_camera=pipe.raw_pose.T_deck_camera,
                        covariance=pipe.raw_pose.covariance,reprojection_error=pipe.raw_pose.reprojection_error)
                stream.write(json.dumps(clean(values),allow_nan=False)+"\n")
                run["completed_duration_s"] += dt
                if row["iteration"]%500==0:
                    print(json.dumps(dict(event="replay",case=case["id"],method=method,time=t,
                        wall_s=time.perf_counter()-started)),flush=True)
        run.update(status="complete",pending_delayed_packets_at_end=len(pending))
    except BaseException as exc:
        run.update(status="failed",error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        run["elapsed_s"]=time.perf_counter()-started
        write_json(target/"run.json",run)
        if run["status"]=="complete":
            write_json(target/"result_hashes.json",{name:digest(target/name) for name in ("run.json","steps.jsonl")})
    return run
