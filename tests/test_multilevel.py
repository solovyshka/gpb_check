import json

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from diagnose import gini, load_models_and_scores, prepare_frame
from multilevel import (calibration_gap_decomposition, calibration_mapping,
                        fit_quality_regression, regression_quality_panel, run_levels)
from ranking import WeightedGini, pair_transitions, transition_statistics


def test_exact_pair_transitions_against_exhaustive_with_ties():
    rng = np.random.default_rng(12)
    for n in (0, 1, 2, 20, 121):
        y = rng.integers(0, 2, size=n)
        a, b = rng.integers(0, 5, size=(2, n))
        actual = pair_transitions(y, a, b)
        bad, good = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        expected = np.zeros((len(bad), 3, 3), dtype=int)
        for i, row in enumerate(bad):
            for j in good:
                before = int(np.sign(a[row] - a[j])) + 1
                after = int(np.sign(b[row] - b[j])) + 1
                expected[i, before, after] += 1
        assert np.array_equal(actual, expected)
        stats = transition_statistics(actual)
        assert stats["pairs_bad_good_total"] == len(bad) * len(good)
        if len(bad) and len(good):
            assert stats["gini_delta_from_pairs"] == pytest.approx(gini(y, b) - gini(y, a))


def test_pair_gain_and_loss_are_not_only_net_count():
    y, old, new = [1, 0, 0], [0.5, 0.1, 0.9], [0.5, 0.9, 0.1]
    stats = transition_statistics(pair_transitions(y, old, new))
    assert stats["pairs_wrong_to_correct"] == 1
    assert stats["pairs_correct_to_wrong"] == 1
    assert stats["pair_net_units"] == 0


def test_fast_gini_matches_weighted_auc_with_zero_weights_and_ties():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 300)
    scores = rng.integers(0, 10, 300)
    evaluate = WeightedGini(y, scores)
    for _ in range(10):
        weight = rng.integers(0, 5, 300)
        assert evaluate(weight) == pytest.approx(gini(y, scores, weight))


def level_config():
    return {"enabled": True, "final_models": {
        "dub_22": {"raw_score_col": "score_dub_22", "pd_col": "pd_dub_22", "raw_kind": "margin", "pd_target": "y"},
        "dub_23": {"raw_score_col": "score_dub_23", "pd_col": "pd_dub_23", "raw_kind": "margin", "pd_target": "y"}},
        "submodels": {}, "regression": {"enabled": False}}


def sample_frame(n=400):
    rng = np.random.default_rng(43)
    a = rng.normal(size=n)
    b = a + rng.normal(0, .4, n)
    y = rng.binomial(1, expit(a - 1.5))
    return pd.DataFrame({"App_id": [f"a{i}" for i in range(n)],
                         "issue_dt": ["2026-02-01"] * (n // 2) + ["2026-04-01"] * (n - n // 2),
                         "y": y, "_period": ["2026-02"] * (n // 2) + ["2026-04"] * (n - n // 2),
                         "_window": ["reference"] * (n // 2) + ["recent"] * (n - n // 2),
                         "score_dub_22": a, "score_dub_23": b,
                         "pd_dub_22": expit(a - 1.5), "pd_dub_23": expit(b - 1.5)})


def options():
    return dict(target="y", id_col="App_id", date_col="issue_dt", bootstrap_iterations=10, seed=42,
                analysis_levels=level_config(), min_defaults=2, min_goods=2)


def test_strict_calibration_invariant_and_rounded_pd_changes_ties():
    frame, cfg = sample_frame(), options()
    tables, _ = run_levels(frame, cfg)
    cal = tables["calibration_metrics"]
    assert cal.delta_gini.abs().max() < 1e-12
    assert cal.pairs_improved.eq(0).all() and cal.pairs_worsened.eq(0).all()
    decomposition = tables["calibration_gap_decomposition"]
    assert decomposition.gap_change_due_to_calibration.abs().max() < 1e-12
    frame.pd_dub_23 = frame.pd_dub_23.round(1)
    rounded, _ = run_levels(frame, cfg)
    cal = rounded["calibration_metrics"].query("model == 'dub_23'")
    assert cal.single_nondecreasing_mapping.all()
    assert not cal.single_strictly_increasing_mapping.any()
    assert (cal.pairs_correct_to_tie + cal.pairs_wrong_to_tie).sum() > 0


def test_segment_calibration_can_change_global_ranking():
    frame, cfg = sample_frame(), options()
    frame["segment"] = np.where(np.arange(len(frame)) % 2, "A", "B")
    frame.pd_dub_23 = expit(frame.score_dub_23 + np.where(frame.segment == "A", -3, 3))
    cfg["analysis_levels"]["final_models"]["dub_23"]["calibration_group_cols"] = ["segment"]
    tables, _ = run_levels(frame, cfg)
    cal = tables["calibration_metrics"]
    global_ = cal.query("model == 'dub_23' and scope == 'all'").loc[lambda d: d.calibration_group.isna()]
    assert not global_.single_nondecreasing_mapping.iloc[0]
    within = cal.loc[cal.calibration_group.notna()]
    assert within.single_strictly_increasing_mapping.all()
    assert within.delta_gini.abs().max() < 1e-12
    decomposition = tables["calibration_gap_decomposition"].set_index("scope").loc["all"]
    assert decomposition.gap_change_due_to_calibration == pytest.approx(
        decomposition.calibration_gini_change_23 - decomposition.calibration_gini_change_22)


def test_arbitrary_raw_score_is_not_treated_as_a_probability():
    frame, cfg = sample_frame(), options()
    for spec in cfg["analysis_levels"]["final_models"].values():
        spec["raw_kind"] = "score"
    tables, _ = run_levels(frame, cfg)
    assert not tables["calibration_metrics"].uncalibrated_probability_available.any()
    assert "brier_uncalibrated_pd" not in tables["calibration_metrics"]


def test_probability_calibration_uses_own_target_and_maturity():
    frame, cfg = sample_frame(), options()
    frame["own_target"] = 1 - frame.y.astype(float)
    frame.loc[frame._window == "recent", "own_target"] = np.nan
    cfg["analysis_levels"]["final_models"]["dub_23"]["pd_target"] = "own_target"
    tables, _ = run_levels(frame, cfg)
    metrics = tables["calibration_metrics"].query("model == 'dub_23'").set_index("scope")
    assert metrics.loc["recent", "n_probability_evaluation"] == 0
    assert metrics.loc["recent", "n"] == 200
    assert metrics.loc["all", "probability_target"] == "own_target"
    assert metrics.loc["all", "probability_target_matches_pd_target"]
    assert metrics.loc["all", "observed_rate"] == pytest.approx(frame.loc[frame.own_target.notna(), "own_target"].mean())


def test_regression_measurement_split_is_independent_of_input_row_order():
    frame, cfg = sample_frame(), options()
    levels = level_config()
    levels["submodels"] = {"early_model": levels["final_models"]}
    levels["regression"] = dict(split_measurement=True, min_defaults=1, min_goods=1)
    original = regression_quality_panel(frame, levels, cfg)
    shuffled = regression_quality_panel(frame.sample(frac=1, random_state=99), levels, cfg)
    pd.testing.assert_frame_equal(original, shuffled)
    assert original.n_outcome.lt(len(frame) / 2).any()


def raw_submodels(frame):
    rng = np.random.default_rng(101)
    versions = {}
    for group in ("early_model", "long_model", "soc_dem"):
        versions[group] = {}
        for name in (("dub_22",) if group == "soc_dem" else ("dub_22", "dub_23")):
            col = group + "_" + name
            frame[col] = rng.normal(size=len(frame))
            versions[group][name] = {"raw_score_col": col, "raw_higher_is_risk": True}
    return versions


def test_uncalibrated_submodels_need_only_scores_and_never_enter_calibration():
    frame, cfg = sample_frame(), options()
    cfg["analysis_levels"]["submodels"] = raw_submodels(frame)
    # Old configuration keys must not require or invent submodel PD columns.
    cfg["analysis_levels"]["submodels"]["early_model"]["dub_22"]["pd_col"] = "missing_obsolete_pd"
    tables, notes = run_levels(frame, cfg)
    sub = tables["levels_metrics"].query("level == 'submodels'")
    assert set(sub.representation) == {"raw"}
    for table in ("calibration_metrics", "calibration_curves", "calibration_gap_decomposition"):
        if "entity" in tables[table]:
            assert set(tables[table].entity) == {"final"}
    quality = tables["submodel_quality"]
    assert set(quality.entity) == {"early_model", "long_model", "soc_dem"}
    soc = quality.query("entity == 'soc_dem' and scope == 'all'").iloc[0]
    assert soc.model == "dub_22"
    assert soc.gini_raw == pytest.approx(gini(frame.y, frame.soc_dem_dub_22))
    assert "gini_raw_ci_low" in quality and "brier_pd" not in quality
    assert not any("early_model/" in note or "soc_dem/" in note for note in notes)


def test_regression_uses_same_raw_submodel_quality_for_both_final_responses():
    frame, cfg = sample_frame(), options()
    levels = cfg["analysis_levels"]
    levels["submodels"] = raw_submodels(frame)
    levels["regression"] = dict(split_measurement=True, min_defaults=1, min_goods=1)
    # Change only final calibration, leaving all raw scores fixed.
    frame.pd_dub_23 = expit(frame.score_dub_23 + np.where(np.arange(len(frame)) % 2, 4, -4))
    panel = regression_quality_panel(frame, levels, cfg)
    assert set(panel.predictor_representation) == {"raw"}
    predictors = [c for c in panel if c.startswith("gini__") or c.startswith("delta_gini__")]
    before = panel.query("representation == 'raw'").set_index("period")
    after = panel.query("representation == 'pd'").set_index("period")
    pd.testing.assert_frame_equal(before[predictors], after[predictors])
    assert not np.allclose(before.response_delta_gini, after.response_delta_gini)


def test_four_way_comparison_uses_identical_complete_cohort():
    frame, cfg = sample_frame(), options()
    frame.loc[0, "score_dub_23"] = np.nan
    tables, _ = run_levels(frame, cfg)
    metrics = tables["levels_metrics"].query("entity == 'final' and scope == 'all'")
    assert metrics.n.tolist() == [len(frame) - 1] * 2
    assert tables["levels_cohort_audit"].n_missing.sum() == 1
    frame.loc[1, "pd_dub_23"] = 1.1
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        run_levels(frame, cfg)


def synthetic_panel(periods=36):
    rng = np.random.default_rng(8)
    x = rng.normal(0, .08, (periods, 3))
    y = .02 + x @ np.array([.7, -.4, .2]) + rng.normal(0, .002, periods)
    one = pd.DataFrame(x, columns=["delta_gini__early_model", "delta_gini__long_model", "gini__soc_dem__dub_22"])
    one["response_delta_gini"] = y
    one["period"] = pd.period_range("2023-01", periods=periods, freq="M").astype(str)
    one["reliable_size"], one["split_measurement"] = True, True
    return pd.concat([one.assign(representation="raw"), one.assign(representation="pd")], ignore_index=True)


def test_regression_reports_hac_intervals_and_real_chronological_holdout():
    coefficients, diagnostics, fitted = fit_quality_regression(synthetic_panel(), {"regression": {}})
    assert diagnostics.status.eq("ok").all()
    assert diagnostics.covariance.eq("HAC").all()
    coefficient = coefficients.query("representation == 'raw'").set_index("term")
    assert coefficient.loc["delta_gini__early_model", "coefficient"] == pytest.approx(.7, abs=.03)
    assert coefficient.loc["delta_gini__long_model", "coefficient"] == pytest.approx(-.4, abs=.03)
    assert coefficient.loc["delta_gini__early_model", "ci_low"] < .7 < coefficient.loc["delta_gini__early_model", "ci_high"]
    assert diagnostics.holdout_rmse.lt(diagnostics.holdout_mean_baseline_rmse).all()
    assert fitted.chronological_holdout_prediction.notna().sum() == 18


def test_regression_refuses_five_months_and_collinear_predictors():
    coef, diagnostics, _ = fit_quality_regression(synthetic_panel(5), {"regression": {}})
    assert coef.empty
    assert diagnostics.status.eq("insufficient_periods").all()
    panel = synthetic_panel()
    panel["delta_gini__early_model"] = panel["delta_gini__long_model"]
    _, diagnostics, _ = fit_quality_regression(panel, {"regression": {}})
    assert diagnostics.status.eq("collinear_predictors").all()


def test_raw_model_verification_allows_a_nonmonotone_calibrated_pd(tmp_path):
    from catboost import CatBoostClassifier
    raw = sample_frame()
    x = raw[["score_dub_22"]].rename(columns={"score_dub_22": "x"})
    model = CatBoostClassifier(iterations=5, depth=2, verbose=False, allow_writing_files=False, thread_count=2)
    model.fit(x, raw.y)
    path = tmp_path / "model.cbm"
    model.save_model(str(path))
    prediction = model.predict(x, prediction_type="RawFormulaVal")
    raw["x"] = x.x
    for name in ("dub_22", "dub_23"):
        raw["score_" + name] = prediction
        raw["pd_" + name] = expit(prediction + np.where(np.arange(len(raw)) % 2, 2, -2))
    cfg = dict(options(), models={m: dict(path=str(path), score_col="pd_" + m) for m in ("dub_22", "dub_23")})
    models, check = load_models_and_scores(raw, cfg, scores_only=False)
    assert len(models) == 2
    assert check.spearman_stored_vs_model.eq(1).all()
    assert set(check.verification_column) == {"score_dub_22", "score_dub_23"}
