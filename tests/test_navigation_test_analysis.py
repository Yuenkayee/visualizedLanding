"""Scientific-contract tests; fixtures are synthetic, never thesis evidence."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from main import navigation_test_analysis as analysis


def row(t=0., dt=1., error=1., valid=True, camera=True, **changes):
    result = dict(timestamp=t, dt=dt, stage="approach", camera_sample=camera,
                  lidar_sample=True, estimate_available=True, navigation_valid=valid,
                  cg_signal_valid=valid, gear_truth_ned=[0., 0., 0.],
                  gear_estimate_ned=[error, 0., 0.], gear_covariance_ned=np.eye(3).tolist(),
                  cg_truth_ned=[0., 0., 0.], cg_estimate_ned=[error, 0., 0.],
                  cg_shaped_ned=[error, 0., 0.], cg_covariance_ned=(4*np.eye(3)).tolist(),
                  attitude_error_rad=.1, diagnostics=dict(vision_accepted=valid, lidar_accepted=valid),
                  timing={}, quality={})
    result.update(changes)
    return result


def run(tmp_path, method="N3", case_id="c", planned=1., completed=None, status="complete", **case_fields):
    case = dict(id=case_id, pair_id=case_id, weather="clear", motion="low", seed=1,
                suite="main", condition="baseline", gimbal_mode="tracking", **case_fields)
    info = dict(status=status, planned_duration_s=planned,
                completed_duration_s=planned if completed is None else completed)
    return analysis.Run(case, method, tmp_path, info, "complete" if status == "complete" else "incomplete", [])


def save_run(root, case, method, rows, **info):
    target = root/"results"/case["id"]/method
    target.mkdir(parents=True)
    metadata = dict(status="complete", planned_duration_s=1., completed_duration_s=1.,
                    source="shared descriptive trajectory name")
    metadata.update(info)
    (target/"run.json").write_text(json.dumps(metadata))
    (target/"steps.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))


def manifest(root, cases, **changes):
    data = dict(cases=cases, evaluation=dict(false_valid_error_m=1., max_outage_s=.5,
                                           bootstrap_samples=20, bootstrap_seed=17))
    data.update(changes)
    (root/"manifest.json").write_text(json.dumps(data))


def metric(summary, method="N3", reference="gear", sample_set="finite", stage="all", case_id="c"):
    return next(r for r in summary["flight_metrics"] if
                (r["case_id"], r["method"], r["reference"], r["sample_set"], r["stage"]) ==
                (case_id, method, reference, sample_set, stage))


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def test_uneven_dt_weights_availability_not_frame_count(tmp_path):
    r = run(tmp_path)
    rows = [row(dt=.1, valid=True), row(.1, .9, valid=False)]
    stats, events = analysis._availability(rows, r, "gear", "all", {})
    assert stats["availability"] == pytest.approx(.1)
    assert stats["covered_duration_s"] == pytest.approx(1.)
    assert stats["longest_outage_s"] == pytest.approx(.9)
    assert stats["false_valid_events"] is None
    assert events[0]["start_s"] == pytest.approx(.1)


def test_failed_run_tail_and_unknown_stage_are_not_hidden(tmp_path):
    r = run(tmp_path, planned=10., completed=2., status="failed")
    rows = [row(dt=1.), row(1., 1.)]
    stats, _ = analysis._availability(rows, r, "gear", "all", dict(max_outage_s=.5))
    assert stats["availability"] == pytest.approx(.2)
    assert stats["uncovered_duration_s"] == pytest.approx(8.)
    assert stats["longest_outage_s"] == 8.
    assert stats["outage_limit_exceedances"] == 1
    assert analysis._availability(rows, r, "gear", "descent", {})[0]["availability"] is None
    assert analysis._availability(rows, r, "gear", "approach", {})[0]["duration_denominator_s"] == 2.


def test_overlaps_and_gaps_use_no_invented_duration(tmp_path):
    r = run(tmp_path, planned=4.)
    rows = [row(dt=3.), row(1., .5), row(3., 1.)]
    stats, _ = analysis._availability(rows, r, "gear", "all", {})
    assert stats["covered_duration_s"] == 2.5
    assert stats["longest_outage_s"] == 1.5


def test_missing_and_nonfinite_are_not_zero_error_and_covariance_is_matched():
    rows = [row(error=2.), row(1., error=float("nan")),
            row(2., gear_estimate_ned=None, cg_shaped_ned=None),
            row(3., gear_truth_ned=None)]
    values = analysis._accuracy(rows, "N3", "gear", "finite", set())
    assert values["samples"] == 1 and values["rmse_3d_m"] == 2.
    assert values["nees_mean"] == 4.
    cg = analysis._accuracy(rows[:1], "N3", "cg", "finite", set())
    shaped = analysis._accuracy(rows[:1], "N3", "cg_shaped", "finite", set())
    assert cg["nees_mean"] == 1.
    assert shaped["nees_mean"] is None and shaped["chi2_95_coverage"] is None
    assert analysis._accuracy(rows[1:], "N3", "gear", "finite", set())["rmse_3d_m"] is None
    json.dumps(analysis._clean(values), allow_nan=False)


@pytest.mark.parametrize("cov", [None, [[1, 0], [0, 1]], np.zeros((3, 3)).tolist(),
                                 [[1, 2, 0], [0, 1, 0], [0, 0, 1]],
                                 np.diag([-1., 1., 1.]).tolist(), np.full((3, 3), np.nan).tolist()])
def test_bad_covariance_does_not_remove_finite_errors(cov):
    values = analysis._accuracy([row(gear_covariance_ned=cov)], "N3", "gear", "finite", set())
    assert values["samples"] == 1
    assert values["nees_samples"] == 0 and values["nees_mean"] is None


def test_chi_square_ellipsoid_coverage_and_valid_scope():
    rows = [row(error=1.), row(1., error=4., valid=False)]
    finite = analysis._accuracy(rows, "N3", "gear", "finite", set())
    valid = analysis._accuracy(rows, "N3", "gear", "declared_valid", set())
    assert finite["chi2_95_coverage"] == .5
    assert valid["samples"] == 1 and valid["chi2_95_coverage"] == 1.


def test_shaped_finite_output_is_independent_of_raw_availability():
    r = row(estimate_available=False, gear_estimate_ned=None, cg_estimate_ned=None,
            cg_shaped_ned=[3., 0., 0.], cg_signal_valid=False)
    assert analysis._values(r, "N3", "gear")[1] is None
    assert analysis._accuracy([r], "N3", "cg_shaped", "finite", set())["rmse_3d_m"] == 3.
    assert analysis._accuracy([r], "N3", "cg_shaped", "declared_valid", set())["samples"] == 0


def test_N1_has_no_phantom_times_or_shaped_output(tmp_path):
    rows = [row(0., .1, error=1.), row(.1, .1, error=100., camera=False),
            row(.2, .1, error=3.), row(.3, .1, gear_estimate_ned=None)]
    r = run(tmp_path, method="N1", planned=.4)
    values = analysis._accuracy(rows, "N1", "gear", "finite", set())
    assert values["samples"] == 2 and values["rmse_3d_m"] == pytest.approx(np.sqrt(5.))
    assert values["J_m_mean"] == 2.
    assert analysis._accuracy(rows, "N1", "cg_shaped", "finite", set())["samples"] == 0
    stats, _ = analysis._availability(rows, r, "gear", "all", {})
    assert stats["availability"] == pytest.approx(.5)
    assert stats["camera_coverage"] == pytest.approx(2/3)
    samples = list(analysis._sample_records(rows, r))
    assert len(samples) == 4 and samples[1]["gear_estimate_n"] is None


def test_common_main_does_not_depend_on_failed_ablation_and_is_not_valid_intersection(tmp_path):
    case = dict(id="c", methods=["N0", "N1", "N2", "N3", "A1"])
    manifest(tmp_path, [case])
    for method in case["methods"]:
        rows = [row(valid=method != "N0")] if method != "A1" else [row(estimate_available=False, cg_shaped_ned=None)]
        save_run(tmp_path, case, method, rows)
    runs, _ = analysis._discover(tmp_path, json.loads((tmp_path/"manifest.json").read_text()))
    common = analysis._common_times(runs)
    assert common["common_camera"][1]["gear"] == {0.}
    assert common["common_camera_all_methods"][1]["gear"] == set()
    assert common["common_camera_pair_A1"][1]["gear"] == set()
    assert common["common_camera"][1]["cg_shaped"] == set()  # N1 never shapes.


def test_J_uses_error_not_real_motion_and_breaks_on_missing():
    rows = [row(gear_truth_ned=[0, 0, 0], gear_estimate_ned=[1, 0, 0]),
            row(1., gear_truth_ned=[10, 0, 0], gear_estimate_ned=[11, 0, 0]),
            row(2., gear_estimate_ned=None), row(3., gear_estimate_ned=[100, 0, 0])]
    result = analysis._accuracy(rows, "N3", "gear", "finite", set())
    assert result["J_m_n"] == 1 and result["J_m_max"] == 0.


def test_recovery_starts_at_accepted_confirmation_not_held_diagnostic(tmp_path):
    rows = [row(0., .1), row(.1, .1, valid=False, camera=False),
            row(.2, .1, valid=False, camera=False, diagnostics=dict(vision_accepted=True)),
            row(.3, .1, valid=False, diagnostics=dict(vision_accepted=True, vision_recovery_pending=True)),
            row(.4, .1, valid=False, diagnostics=dict(vision_accepted=False)), row(.5, .1)]
    events = analysis._recoveries(rows, run(tmp_path))
    assert len(events) == 1
    assert events[0]["first_qualified_vision_s"] == .3
    assert events[0]["recovery_time_s"] == pytest.approx(.2)
    assert events[0]["kind"] == "recovery"
    censored = analysis._recoveries(rows[:-1], run(tmp_path))[0]
    assert censored["censored"] and censored["recovery_time_s"] is None


def test_false_valid_episodes_duration_and_strict_thresholds(tmp_path):
    r = run(tmp_path)
    rows = [row(0., .1, error=1.), row(.1, .2, error=2.), row(.3, .3, error=2.), row(.6, .4, valid=False)]
    stats, _ = analysis._availability(rows, r, "gear", "all", dict(false_valid_error_m=1., max_outage_s=.4))
    assert stats["false_valid_events"] == 1
    assert stats["false_valid_duration_s"] == pytest.approx(.5)
    assert stats["outage_limit_exceedances"] == 0
    assert analysis._availability(rows, r, "gear", "all", dict(false_valid_error_m=2.))[0]["false_valid_events"] == 0


def test_timing_first_per_component_and_missing_not_zero():
    rows = [row(0., .1, timing=dict(inference_s=.9, matlab_s=None)),
            row(.1, .1, camera=False, timing=dict(inference_s=0., matlab_s=None)),
            row(.2, .1, timing=dict(inference_s=.1)), row(.3, .1, timing=dict(inference_s=.3))]
    timing = analysis._timing_metrics(rows)
    cold = next(r for r in timing if r["reference"] == "timing:inference_s" and r["stage"] == "all" and r["sample_set"] == "first_recorded_call")
    warm = next(r for r in timing if r["reference"] == "timing:inference_s" and r["stage"] == "all" and r["sample_set"] == "subsequent_calls")
    assert cold["seconds_mean"] == .9
    assert warm["seconds_n"] == 2 and warm["seconds_mean"] == pytest.approx(.2)
    assert all(r["seconds_mean"] is None for r in timing if r["reference"] == "timing:matlab_s")


def test_lidar_requires_event_flag_and_held_vision_is_counted_once(tmp_path):
    rows = [row(0., .5), row(.5, .5, camera=False)]
    for r in rows:
        r.pop("lidar_sample")
    result = analysis._diagnostics(rows, run(tmp_path), [])
    assert result["camera_events"] == 1
    assert result["vision_acceptance_observations"] == 1
    assert result["lidar_events"] is None and result["lidar_acceptance_fraction"] is None


def test_solver_and_rejection_rates_have_actual_denominators(tmp_path):
    rows = [row(diagnostics=dict(pnp_solved=True, vision_accepted=True, vision_mahalanobis=1.,
                                 lidar_accepted=True, lidar_mahalanobis=2., lidar_associated_points=80,
                                 lidar_association_rejection_fraction=.2), quality=dict(lidar_points=100)),
            row(1., diagnostics=dict(pnp_solved=True, vision_accepted=False, vision_mahalanobis=20.,
                                     vision_rejection="innovation_gate", lidar_accepted=False,
                                     lidar_rejection="no_deck_plane", lidar_associated_points=0,
                                     lidar_association_rejection_fraction=1.), quality=dict(lidar_points=20)),
            row(2., diagnostics=dict(pnp_solved=False, vision_accepted=False, vision_rejection="pnp_failed"))]
    stats = analysis._diagnostics(rows, run(tmp_path, planned=3.), [])
    assert stats["pnp_solve_observations"] == 3
    assert stats["pnp_solve_fraction"] == pytest.approx(2/3)
    assert stats["vision_acceptance_fraction"] == pytest.approx(1/3)
    assert stats["vision_innovation_rejection_observations"] == 2
    assert stats["vision_innovation_rejection_fraction"] == .5
    assert stats["lidar_association_input_points"] == 120
    assert stats["lidar_association_rejected_points"] == 40
    assert stats["lidar_association_rejection_fraction_pooled"] == pytest.approx(1/3)


def test_lag_is_a_diagnostic_and_never_shifts_accuracy():
    velocity = np.random.default_rng(13).normal(size=(60, 3))
    position = np.cumsum(velocity, axis=0)
    shaped = np.vstack([np.zeros((2, 3)), position[:-2]])
    rows = [row(float(i), cg_truth_ned=p.tolist(), cg_estimate_ned=p.tolist(), cg_shaped_ned=s.tolist())
            for i, (p, s) in enumerate(zip(position, shaped))]
    assert analysis._lag(rows, "N3")["lag_s"] == 2.
    assert analysis._accuracy(rows, "N3", "cg_shaped", "finite", set())["rmse_3d_m"] > 0
    assert analysis._lag([row(float(i)) for i in range(10)], "N3")["lag_s"] is None


def bootstrap_record(value, unit, weather="clear", weight=1., cohort="complete"):
    return dict(value=value, independent_unit=unit, weather=weather, motion="low",
                gimbal_mode="tracking", weight=weight, cohort=cohort)


def test_bootstrap_resamples_flights_stratifies_and_deduplicates_sources():
    rows = [bootstrap_record(1., "a"), bootstrap_record(3., "b"), bootstrap_record(100., "c", "fog")]
    first = analysis._bootstrap(rows, 1000, 42)
    assert first == analysis._bootstrap(rows, 1000, 42)
    result = {r["estimand"]: r for r in first}
    assert result["flight_equal"]["estimate"] == pytest.approx(104/3)
    assert result["scenario_equal"]["estimate"] == 51.
    assert result["flight_equal"]["ci_low"] >= 102/3
    assert result["flight_equal"]["ci_high"] <= 106/3
    assert result["flight_equal"]["n_units"] == 3
    assert result["flight_equal"]["singleton_strata"] == 1
    duplicated = analysis._bootstrap(rows + [rows[0]]*100, 1000, 42)
    assert duplicated[0]["estimate"] == first[0]["estimate"]
    assert duplicated[0]["ci_low"] == first[0]["ci_low"]
    assert analysis._bootstrap([rows[0]]*100, 1000, 42)[0]["ci_low"] is None


def test_paired_bootstrap_preserves_covariance_and_pairs_special_suites(tmp_path):
    records, runs = [], []
    for seed, base in ((1, 1.), (2, 100.)):
        for suite, cid, mode, error, method in (("main", f"main{seed}", "tracking", base, "N3"),
                                               ("main", f"main{seed}", "tracking", base+2., "N2"),
                                               ("stress", f"stress{seed}", "tracking", base+3., "N3"),
                                               ("gimbal", f"fixed{seed}", "fixed", base+4., "N3")):
            r = run(tmp_path, method=method, case_id=cid)
            r.case.update(suite=suite, pair_id=f"pair{seed}", seed=seed, gimbal_mode=mode,
                          condition="dropout" if suite == "stress" else "baseline")
            runs.append(r)
            records.append(dict(r.meta, reference="gear", stage="all", sample_set="finite", common_methods="",
                                planned_duration_s=1., rmse_3d_m=error))
    warnings = set()
    pairs = analysis._paired(records, runs, warnings)
    assert not warnings
    assert {p["value"] for p in pairs if p["comparison"] == "method"} == {2.}
    assert {p["value"] for p in pairs if p["comparison"] == "stress"} == {3.}
    assert {p["value"] for p in pairs if p["comparison"] == "gimbal"} == {4.}
    result = analysis._bootstrap([p for p in pairs if p["comparison"] == "method"], 200, 42)[0]
    assert result["estimate"] == result["ci_low"] == result["ci_high"] == 2.


def test_source_description_is_not_an_independence_key(tmp_path):
    r = run(tmp_path)
    r.info["source"] = "same trajectory description for 240 independent runs"
    assert r.meta["independent_unit"] == "pair:c"
    r.case["source_record"] = "actual-record"
    assert r.meta["independent_unit"] == "source:actual-record"


def test_schema_failures_exclusions_and_missing_generation(tmp_path):
    cases = [dict(id="c", methods=["N3"]), dict(id="generation_failed", methods=["N3"]),
             dict(id="smoke", methods=["N3"], excluded_from_formal=True)]
    manifest(tmp_path, cases, failures=[dict(case="generation_failed", phase="record", error="renderer failed")])
    save_run(tmp_path, cases[0], "N3", [row()], planned_duration_s=10., completed_duration_s=1.)
    save_run(tmp_path, cases[2], "N3", [row()])
    runs, failures = analysis._discover(tmp_path, json.loads((tmp_path/"manifest.json").read_text()))
    assert [r.cohort for r in runs] == ["incomplete", "incomplete", "excluded_complete"]
    assert any(f.get("error") == "renderer failed" for f in failures)
    assert any(f.get("case_id") == "generation_failed" and f["kind"] == "generation_or_missing_results" for f in failures)


def test_smoke_fallback_and_compact_tables_never_create_formal_CIs(tmp_path):
    case = dict(id="c", methods=list(analysis.METHODS), excluded_from_formal=True,
                suite="main", pair_id="p", source_record="one-record", weather="clear", motion="low", gimbal_mode="tracking")
    manifest(tmp_path, [case])
    for method in case["methods"]:
        save_run(tmp_path, case, method, [row(dt=.5), row(.5, .5, error=2.)])
    result = analysis.analyze(tmp_path, bootstrap_samples=5)
    assert result["representative_case"] == "c"
    assert result["representative_cohort"] == "excluded_complete"
    assert len(read_csv(tmp_path/"figure_5_8.csv")) == 14
    assert (tmp_path/"table_5_13.csv").stat().st_size < 150_000
    assert (tmp_path/"table_5_14.csv").stat().st_size < 700_000
    tables = read_csv(tmp_path/"table_5_13.csv") + read_csv(tmp_path/"table_5_14.csv")
    assert all(not v for r in tables for k, v in r.items() if k.endswith(("_ci_low", "_ci_high")))
    assert all(r["stage"] not in ("hover", "descent") for r in tables)
    assert "本批没有可纳入正式比较的完整架次" in (tmp_path/"analysis.md").read_text()


def test_numeric_report_describes_measured_values_and_paired_interval(tmp_path):
    records, runs = [], []
    for i in range(2):
        for method, error in (("N3", 1.+i), ("N2", 3.+i)):
            r = run(tmp_path, case_id=f"c{i}", method=method)
            runs.append(r)
            for sample_set in ("finite", "declared_valid"):
                records.append(dict(r.meta, reference="gear", stage="all", sample_set=sample_set, common_methods="",
                                    planned_duration_s=1., samples=20, rmse_3d_m=error))
            records.append(dict(r.meta, reference="gear", stage="all", sample_set="timeline", common_methods="",
                                planned_duration_s=1., duration_denominator_s=1., availability=.75))
    pairs = analysis._paired(records, runs, set())
    analysis._wide_tables(tmp_path, records, pairs, 50, 42)
    prose = "\n".join(analysis._numeric_findings(tmp_path, dict(runs=[])))
    assert "1.5 / 1.5 m" in prose
    assert "75.0%" in prose
    assert "N2 − N3" in prose and "[2, 2]" in prose
    assert "不构成因果归因" in prose


def test_noobservations_end_to_end_writes_null_metrics_and_real_artifacts(tmp_path):
    case = dict(id="c", pair_id="p", weather="fog", motion="high", seed=42, suite="main",
                gimbal_mode="tracking", condition="baseline", methods=["N1", "N3"])
    manifest(tmp_path, [case])
    rows = [row(0., .1, estimate_available=False, gear_estimate_ned=None, cg_estimate_ned=None,
                cg_shaped_ned=None, valid=False),
            row(.1, .9, estimate_available=False, gear_estimate_ned=None, cg_estimate_ned=None,
                cg_shaped_ned=None, valid=False, camera=False)]
    for method in case["methods"]:
        save_run(tmp_path, case, method, rows)
    result = analysis.analyze(tmp_path)
    assert result["evaluation"]["bootstrap_samples"] == 20
    assert metric(result)["rmse_3d_m"] is None
    assert metric(result, sample_set="timeline")["availability"] == 0.
    assert metric(result, sample_set="timeline")["uncovered_duration_s"] == 1.
    for name in result["artifacts"]:
        assert (tmp_path/name).is_file(), name
    assert (tmp_path/"figure_5_8.png").stat().st_size > 1000
    json.dumps(result, allow_nan=False)
    assert "raw_steps" not in result
    sensitivity = [r for r in read_csv(tmp_path/"sensitivity.csv") if r["method"] == "N3" and r["reference"] == "gear" and r["stage"] == "all"]
    assert {float(r["threshold"]) for r in sensitivity if r["threshold_kind"] == "max_outage_s"} == {.2, .5, 1.}
    assert all(r["false_valid_events"] == "0" for r in sensitivity)


def test_plot_bands_zoom_and_actual_innovation_gates(tmp_path):
    r = run(tmp_path)
    rows = [row(0., .25, diagnostics=dict(vision_accepted=True, vision_position_correction_m=1., vision_mahalanobis=2., vision_mahalanobis_threshold=9.)),
            row(.25, .25, diagnostics=dict(vision_accepted=False, vision_position_correction_m=3., vision_mahalanobis=12., vision_mahalanobis_threshold=9.)),
            row(.5, .25, diagnostics=dict(vision_accepted=True, vision_position_correction_m=3.)), row(.75, .25)]
    samples = list(analysis._sample_records(rows, r))
    zoom = analysis._zoom_window(samples)
    assert zoom["event_s"] == .25
    assert samples[0]["gear_band95_n"] == pytest.approx(1.96)
    assert samples[0]["cg_band95_n"] == pytest.approx(3.92)
    assert samples[0]["cg_shaped_band95_n"] is None
    analysis._plots(tmp_path, samples, [], zoom)
    innovations = read_csv(tmp_path/"figure_5_10_innovations.csv")
    assert [r["threshold"] for r in innovations if r["sensor"] == "vision"][:2] == ["9.0", "9.0"]


def test_empty_manifest_graceful(tmp_path):
    manifest(tmp_path, [])
    result = analysis.analyze(tmp_path, bootstrap_samples=5)
    assert result["case_count"] == 0 and result["representative_case"] is None
    assert result["flight_metrics"] == []
    assert (tmp_path/"analysis.md").exists()


@pytest.mark.parametrize("field,value", [("false_valid_error_m", -1), ("max_outage_s", float("nan"))])
def test_rejects_invalid_evaluation_thresholds(field, value):
    with pytest.raises(ValueError):
        analysis._evaluation(dict(evaluation={field: value}), 10)
