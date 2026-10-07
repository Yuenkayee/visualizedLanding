"""Execute real SLX ports and the rendered multi-camera navigation transaction loop."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import yaml
from simulation.bridge.matlab_session import MatlabSession
from simulation.offline.logger import read_log
from simulation.offline.run_closed_loop import run_closed_loop


def verify(config):
    if (
        config.get("backend") != "matlab"
        or config.get("plant", {}).get("initialize_callback") != "innerloop_initialize"
    ):
        raise ValueError("verification requires the innerLoop MATLAB interface backend")
    root = Path(__file__).resolve().parents[2]
    file = root / "external/innerloop/simu/innerLoop.slx"
    before = hashlib.sha256(file.read_bytes()).hexdigest()
    with MatlabSession() as session:
        contract_tests = session.call("test_innerloop_interfaces")
    result = run_closed_loop(config)
    audit = read_log(result["audit_log"])
    if not audit:
        raise AssertionError("no interface transactions")
    for index, row in enumerate(audit):
        operations = row["operations"]
        expected = ([] if index == 0 else ["advance_innerloop"]) + [
            "read_current_state"
        ]
        if "camera" in row["events"]:
            expected.append("generate_camera_observations")
        if "lidar" in row["events"]:
            expected.append("generate_lidar_observations")
        expected += [
            "navigation_solution",
            "write_navigation_feedback",
            "read_and_verify_slx_feedback",
        ]
        if operations != expected:
            raise AssertionError(f"invalid causal sequence at t={row['timestamp']}")
        if (
            row["timestamp"] != row["truth"]["timestamp"]
            or row["timestamp"] != row["estimate"]["timestamp"]
            or row["timestamp"] != row["feedback"]["timestamp"]
        ):
            raise AssertionError("timestamp mismatch")
        if (
            row["feedback"]["source"] != "innerLoop.slx"
            or row["estimate"]["diagnostics"]["reference_camera_id"] != "C0"
        ):
            raise AssertionError("wrong model/reference")
        if index and row["timestamp"] <= audit[index - 1]["timestamp"]:
            raise AssertionError("state time did not advance")
    counts = result["transactions"]
    if (
        counts["state_reads"] != len(audit)
        or counts["feedback_writes"] != len(audit)
        or counts["advances"] != len(audit) - 1
        or counts["checked_slx_feedbacks"] != len(audit)
    ):
        raise AssertionError("a read, write, advancement or SLX check is missing")
    if not counts["visual_updates"] or not counts["lidar_updates"]:
        raise AssertionError("sensor navigation updates were not exercised")
    held = config["plant"].get("allow_state_hold") and not config["plant"].get(
        "plant_step_callback"
    )
    if held and any(
        not np.allclose(
            row["truth"]["T_deck_camera"], audit[0]["truth"]["T_deck_camera"]
        )
        for row in audit
    ):
        raise AssertionError("state-hold test unexpectedly changed physical state")
    if hashlib.sha256(file.read_bytes()).hexdigest() != before:
        raise AssertionError("verification unexpectedly modified the SLX")
    output = dict(
        passed=True,
        model=str(file),
        model_sha256=before,
        matlab_contract_tests=contract_tests,
        transactions=counts,
        control_samples=result["navigation"]["samples"],
        max_slx_python_feedback_difference=result["max_slx_python_feedback_difference"],
        navigation=result["navigation"],
        mode="interface_state_hold" if held else "interface_callback",
        dynamics_tested=False,
        audit_log=result["audit_log"],
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/innerloop_navigation.yaml")
    parser.add_argument(
        "--output", default="outputs/metrics/innerloop_navigation_verification.json"
    )
    args = parser.parse_args()
    result = verify(yaml.safe_load(Path(args.config).read_text()))
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
