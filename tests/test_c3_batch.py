"""Frozen four-weather C3 plans and offline reports, without MATLAB/GPU."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from main.batch import c3, common, c3_analysis
from main.batch.command_analysis import increments
from main.batch.recording import KNOT
from test_custom_c3_launcher import experiment
from test_closed_loop_batch import fake_case

REPOSITORY = Path(__file__).resolve().parents[1]


@pytest.fixture
def full_baseline(experiment, monkeypatch):
    _, old_args, head, root = experiment
    monkeypatch.setattr(c3, "ROOT", root)
    for name in ("main/batch/c3.py", "shell/run_landing_c3.sh"):
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((REPOSITORY / name).read_bytes())
    baseline = Path(old_args.baseline)
    manifest = common.read_json(baseline / "manifest.json")
    config = common.read_json(baseline / "groups/clear_s00/attempt_00/sea5_C3/requested.json")
    landing = config["landing_config"]
    (root / landing).write_bytes((REPOSITORY / landing).read_bytes())
    manifest["source_sha256"][landing] = common.sha(root / landing)
    manifest["groups"] = []
    for weather in common.WEATHERS:
        for slot in range(10):
            # Include a baseline replacement: the new batch must preserve it.
            attempt = 2 if weather == "fog" and slot == 0 else 0
            group = dict(id=f"{weather}_s{slot:02d}", weather=weather, slot=slot)
            seed = common.seeds(manifest["policy"]["master_seed"], weather, slot, attempt)
            relative = Path("groups") / group["id"] / f"attempt_{attempt:02d}/sea5_C3"
            case = baseline / relative
            (case / "simulink").mkdir(parents=True, exist_ok=True)
            request = copy.deepcopy(config)
            request.update(weather=weather, seed=seed["sensor_seed"], batch=dict(method="C3", sea=5, **seed))
            common.write_json(case / "requested.json", request)
            common.write_json(case / "metrics.json", seed)
            common.write_json(case / "run.json", dict(status="complete", config=request))
            common.write_json(case / "simulink/batch_model.json", {})
            common.write_json(baseline / "groups" / group["id"] / "result.json",
                              dict(status="included", attempt=attempt, seeds=seed, cases=[str(relative)]))
            manifest["groups"].append(group)
    common.write_json(baseline / "manifest.json", manifest)
    args = SimpleNamespace(baseline=str(baseline), output=str(root / "outputs/batch/full"),
                           seeds_per_weather=10, duration=60., pilot=False)
    return args, head, root


def test_full_plan_keeps_weather_seeds_attempts_and_actual_head(full_baseline):
    args, head, root = full_baseline
    manifest = c3.make_plan(args)
    tasks = common.read_json(Path(args.output) / "tasks.json")
    assert manifest["git_commit"] == head
    assert not manifest["pilot"] and len(tasks) == 40
    assert [t["group"]["weather"] for t in tasks[:4]] == list(common.WEATHERS)
    assert all(sum(t["group"]["weather"] == w for t in tasks) == 10 for w in common.WEATHERS)
    assert {t["method"] for t in tasks} == {"C3"}
    assert manifest["policy"]["seas"] == {c3.CONDITION: dict(wind_mps=10., wave_height_m=2.)}
    fog = tasks[1]
    assert fog["attempt"] == 2
    assert fog["group"]["seeds"] == common.seeds(manifest["policy"]["master_seed"], "fog", 0, 2)
    c3.verify(Path(args.output), manifest)
    (root / "main/run.py").write_text("runtime changed\n")
    with pytest.raises(RuntimeError, match="Source/model/config changed"):
        c3.verify(Path(args.output), manifest)


def test_short_plans_require_explicit_pilot(full_baseline):
    args, _, _ = full_baseline
    args.seeds_per_weather, args.duration = 2, 2.
    with pytest.raises(RuntimeError, match="require --pilot"):
        c3.make_plan(args)
    assert not Path(args.output).exists()
    args.pilot = True
    manifest = c3.make_plan(args)
    assert manifest["pilot"] and len(manifest["groups"]) == 8


def test_changed_algorithm_is_rejected_before_planning(full_baseline):
    args, _, root = full_baseline
    (root / "main/run.py").write_text("changed algorithm\n")
    with pytest.raises(RuntimeError, match="Baseline algorithm/model/weights/sensors changed"):
        c3.make_plan(args)
    assert not Path(args.output).exists()


@pytest.fixture
def report_batch(tmp_path):
    out = tmp_path / "batch"
    policy = yaml.safe_load((REPOSITORY / "main/batch_closed_loop.yaml").read_text())
    policy.update(master_seed=17, methods=["C3"], seeds_per_weather=2, duration_s=.12,
                  seas={c3.CONDITION: dict(wind_mps=10., wave_height_m=2.)})
    groups, tasks = [], []
    for weather in common.WEATHERS:
        for slot in range(2):
            group = dict(id=f"{weather}_s{slot:02d}", weather=weather, slot=slot, attempt=0,
                         seeds=common.seeds(17, weather, slot, 0))
            path = f"groups/{group['id']}/attempt_00/C3"
            case = out / path
            metrics, _, model = fake_case(case, policy, weather=weather, slot=slot, candidate=slot == 0)
            metrics["sea"] = c3.CONDITION
            metadata = common.read_json(case / "run.json")
            metadata["status"] = "complete"
            metadata["config"]["duration"] = .12
            metadata["config"]["batch"].update(sea=c3.CONDITION, wind_mps=10., wave_height_m=2.)
            model.update(wind_mps=10., wind_seed=group["seeds"]["wind_seed"],
                         wave_parameters=dict(SignificantWaveHeight=2., RandomSeed=group["seeds"]["wave_seed"]),
                         wave_blocks=["Ship_motion/tauAndFDI", "PredictModel/tauAndFDI"])
            for name, value in (("run.json", metadata), ("requested.json", metadata["config"]),
                                ("metrics.json", metrics), ("simulink/batch_model.json", model)):
                common.write_json(case / name, value)
            tasks.append(dict(group=group, attempt=0, sea=c3.CONDITION, method="C3", case=path))
            groups.append(group)
    config = yaml.safe_load((REPOSITORY / "main/navigation_feedback.yaml").read_text())
    manifest = dict(experiment="c3_four_weather_w10_h2", pilot=True, policy=policy,
                    groups=groups, condition=dict(wind_mps=10., wave_height_m=2.), base_config=config,
                    landing_snapshot=yaml.safe_load((REPOSITORY / config["landing_config"]).read_text()))
    common.write_json(out / "manifest.json", manifest)
    common.write_json(out / "tasks.json", tasks)
    return out, manifest


def test_command_cost_excludes_duplicate_terminal_sample(report_batch):
    from main.batch.metrics import read_series
    out, _ = report_batch
    case = out / "groups/clear_s00/attempt_00/C3"
    rows = read_series(case)
    model = common.read_json(case / "simulink/batch_model.json")
    _, _, normal = increments(rows, model)
    assert normal["Jctrl"] == pytest.approx(5 * (.01*KNOT)**2 / .02)
    duplicate = copy.deepcopy(rows[-1])
    duplicate["t"] = duplicate["t1"]
    duplicate["u"]["command"][0] += 100
    rows.append(duplicate)
    times, rate, corrected = increments(rows, model)
    assert corrected["Jctrl"] == normal["Jctrl"]
    assert corrected["omitted_nonadvancing_intervals"] == 1 and len(times) == 5
    assert rate[:, 0] == pytest.approx(np.full(5, .5*KNOT))
    rows[-1]["t1"] = .01
    with pytest.raises(ValueError, match="nondecreasing"):
        increments(rows, model)


def test_reports_preserve_raw_data_and_censored_denominators(report_batch, tmp_path):
    out, manifest = report_batch
    before = {str(p): common.sha(p) for p in out.rglob("*") if p.is_file()}
    dest = tmp_path / "offline-report"
    c3_analysis.analyze(out, dest)
    assert before == {str(p): common.sha(p) for p in out.rglob("*") if p.is_file()}
    status = common.read_json(dest / "status.json")
    assert status["complete"] and status["included"] == 8 and status["candidate_count"] == 4
    assert status["physical_conditions_and_seeds_passed"]
    t15 = common.read_json(dest / "table_5_15.json")
    assert len(t15) == 5 and t15[0]["total"] == 8 and t15[0]["rate"] == .5
    final = [r for r in common.read_json(dest / "table_5_17.json") if r["stage"] == 2]
    assert final[0]["completed"] == 4 and final[0]["unfinished"] == 4
    assert not common.read_json(dest / "figure_5_16_applicability.json")["applicable"]
    assert (dest / "figure_5_14_all_sorties.csv").stat().st_size > 0
    for weather in common.WEATHERS:
        for suffix in (f"figure_5_11_{weather}.png", f"figure_5_12_{weather}.png",
                       f"figure_5_13_{weather}.png", f"figure_5_14_{weather}_all_sorties.png"):
            assert (dest / "figures" / suffix).stat().st_size > 0
    assert not list((dest / "figures").glob("*_paired.png"))
    assert not c3.finalize(out, manifest)
    assert common.read_json(out / "groups/clear_s00/result.json")["status"] == "included"


def test_recorded_seed_drift_is_not_reported_as_comparable(report_batch):
    out, manifest = report_batch
    path = out / "groups/clear_s00/attempt_00/C3/metrics.json"
    value = common.read_json(path)
    value["sensor_seed"] += 1
    common.write_json(path, value)
    with pytest.raises(RuntimeError, match="Recorded seed mismatch"):
        c3.read_completed(out, manifest)


def test_single_condition_plot_handles_no_stop_and_all_stop(tmp_path):
    from main.batch.figures import comparison_plot, touchdown_plot
    from main.batch.statistics import wilson
    policy = dict(weather=["clear"], seas={c3.CONDITION: dict(wind_mps=10., wave_height_m=2.)}, methods=["C3"])
    for k in (0, 10):
        low, high = wilson(k, 10)
        table = [dict(weather=w, sea=c3.CONDITION, method="C3", total=10, candidates=k,
                      rate=k/10, wilson_low=low, wilson_high=high) for w in ("clear", "equal_weather")]
        comparison_plot(tmp_path, policy, table, [])
    touchdown_plot(tmp_path, "clear", [], ([-10,-5], [10,5]), policy)
    assert (tmp_path / "figure_5_13_clear.png").exists()
