import copy
import json
import numpy as np
import pytest
from main.config import load, validate
from main.analyze import control_analysis


def test_navigation_feedback_config_is_explicit():
    c, _, _, _ = load("main/navigation_feedback.yaml")
    validate(c)
    assert c["evaluation_mode"] == "feedback"
    assert c["input_contract"] == "conversion"
    assert c["dt"] == c["plant_step"] == .02
    assert c["control_interface"] == "cg_position"
    for key, value in [("evaluation_mode", "observer"), ("control_invalid_policy", "truth"),
                       ("input_contract", "feedback"), ("control_interface", "unknown")]:
        bad = copy.deepcopy(c); bad[key] = value
        with pytest.raises(ValueError):
            validate(bad)


def test_control_analysis_checks_consumed_position_sign_and_invalid_gate(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    rows=[]
    for i in range(3):
        p=np.array([-95.+i, 1., -50.]); target=np.array([-45.,0.,-13.4])
        rows.append(dict(timestamp=i*.02,cg_relative_estimate_50hz_ned=p.tolist(),
            control_used=dict(source_timestamp=i*.02,cg_relative_ned_m=p.tolist(),valid=i!=1,
                target_relative_ned_m=target.tolist(),error_ned_m=(target-p).tolist(),
                velocity_command_knots=[30.,0.,0.],relative_velocity_command_knots=[0.,0.,0.])))
    result=control_analysis(rows,tmp_path)
    assert result["max_position_latch_error_m"]==0
    assert result["max_reference_sign_error_m"]==0
    assert result["source_timestamp_checked"] and result["invalid_relative_command_zero"]
    assert (tmp_path/"navigation_control.png").is_file()
    assert (tmp_path/"navigation_control.csv").is_file()
    rows[1]["control_used"]["relative_velocity_command_knots"]=[1.,0.,0.]
    rows[2]["control_used"]["source_timestamp"]=0
    rows[0]["control_used"]["cg_relative_ned_m"]=[0.,0.,0.]
    result=control_analysis(rows,tmp_path)
    assert not result["invalid_relative_command_zero"] and not result["source_timestamp_checked"]
    assert result["max_position_latch_error_m"]==95
    json.dumps(result,allow_nan=False)
