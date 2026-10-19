import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostClassifier, Pool

from diagnose import (bootstrap_gap, default_quality, exact_group_gini_shapley,
                      feature_frame, gini, parse_splits, prepare_frame, psi,
                      reranked_defaults, run, top_risk_diagnostics, window_gap_change)


def config():
    return {"id_col": "App_id", "date_col": "issue_dt", "target": "y",
            "max_issue_date": "2026-04-30", "reference_periods": ["2026-02"],
            "recent_periods": ["2026-04"], "models": {
                "dub_22": {"score_col": "p22"}, "dub_23": {"score_col": "p23"}},
            "bootstrap_iterations": 40, "seed": 42}


def raw_frame():
    return pd.DataFrame({"App_id": list("abcdefghij"),
                         "issue_dt": ["2026-02-01"] * 4 + ["2026-04-01"] * 4 + ["2026-05-01"] * 2,
                         "y": [0, 1, 0, 1, 0, 1, 0, 1, np.nan, np.nan],
                         "p22": [0.1, 0.2, 0.3, 0.4] * 2 + [0.1, 0.9],
                         "p23": [0.1, 0.4, 0.3, 0.2] * 2 + [0.2, 0.8]})


def test_default_decomposition_exact_with_ties_and_random_scores():
    rng = np.random.default_rng(7)
    for n in (10, 100, 1000):
        y = np.tile([0, 1], n // 2)
        a, b = rng.integers(0, 8, size=(2, n))
        qa, qb = default_quality(y, a), default_quality(y, b)
        assert 2 * qa.mean() - 1 == pytest.approx(gini(y, a))
        assert 2 * (qb - qa).mean() == pytest.approx(gini(y, b) - gini(y, a))
        # Brute force checks the handling of exact score ties.
        pairs = a[y == 1, None] - a[y == 0]
        assert qa == pytest.approx(((pairs > 0) + 0.5 * (pairs == 0)).mean(axis=1))


def test_maturity_pairing_cutoff_and_duplicate_guard():
    raw = raw_frame()
    raw.loc[0, "p23"] = np.nan
    df, audit = prepare_frame(raw, config())
    assert len(df) == 7
    assert audit.n_paired.sum() == 7
    assert audit.n_eligible_missing_score.sum() == 1
    assert "2026-05" not in set(df._period)
    duplicate = pd.concat([raw, raw.iloc[[0]]])
    with pytest.raises(ValueError, match="уникальным"):
        prepare_frame(duplicate, config())
    raw.loc[1, "y"] = 2
    with pytest.raises(ValueError, match="0/1"):
        prepare_frame(raw, config())


def test_zero_gini_difference_and_bootstrap_for_identical_models():
    raw = raw_frame()
    raw.p23 = raw.p22
    cfg = config()
    frame, _ = prepare_frame(raw, cfg)
    for cluster in (None, "customer"):
        frame["customer"] = ["c1", "c1", "c2", "c3", "c1", "c2", "c4", "c4"]
        cfg["cluster_col"] = cluster
        result = bootstrap_gap(frame, cfg)
        assert result["ci_low"] == result["ci_high"] == 0
        change = window_gap_change(frame, cfg).iloc[0]
        assert change["gap_change_recent_minus_reference"] == 0
        assert change["ci_low"] == change["ci_high"] == 0
    defaults = reranked_defaults(frame, cfg)
    assert (defaults.status == "unchanged").all()


def test_top_quota_net_counts_and_tie_expectation():
    raw = raw_frame()
    raw.p22 = 0.5
    raw.p23 = raw.p22
    cfg = dict(config(), reject_rates=[0.5])
    df, _ = prepare_frame(raw, cfg)
    top, details = top_risk_diagnostics(df, cfg)
    assert (top.net_defaults_caught == 0).all()
    assert (top.defaults_newly_caught == 0).all()
    assert details.empty
    assert top.loc[top.scope == "all", "expected_defaults_random_ties_dub_22"].iloc[0] == 2


def test_group_allocation_sums_to_exact_gini_difference():
    rng = np.random.default_rng(123)
    y = np.tile([0, 1], 40)
    base = rng.normal(size=80)
    changes = {"early": rng.normal(size=80), "fps": rng.normal(size=80), "zero": np.zeros(80)}
    allocation, delta = exact_group_gini_shapley(y, base, changes)
    assert sum(allocation.values()) == pytest.approx(delta)
    assert delta == pytest.approx(gini(y, base + sum(changes.values())) - gini(y, base))
    assert allocation["zero"] == 0


def test_psi_constant_reference_new_values_missingness():
    ref = pd.Series([1.0] * 100)
    assert psi(ref, ref) == 0
    assert psi(ref, pd.Series([2.0] * 100)) > 0
    assert psi(ref, pd.Series([np.nan] * 100)) > 0
    ref = pd.Series(["a", "b"] * 30)
    assert psi(ref, ref) == 0
    assert psi(ref, pd.Series(["unseen"] * 60)) > 0


@pytest.fixture
def fitted(tmp_path):
    rng = np.random.default_rng(42)
    x = pd.DataFrame({"early_model": rng.normal(size=400),
                      "onehot": rng.choice(["A", "B"], 400),
                      "ctr": rng.choice([f"cat{i}" for i in range(15)], 400)})
    y = ((x.early_model > 0).astype(int) + (x.onehot == "A").astype(int) + x.ctr.isin(["cat1", "cat2"]).astype(int) >= 2).astype(int)
    model = CatBoostClassifier(iterations=25, depth=4, one_hot_max_size=2, verbose=False,
                               thread_count=2, allow_writing_files=False, random_seed=2)
    model.fit(x, y, cat_features=["onehot", "ctr"])
    path = tmp_path / "model.json"
    model.save_model(str(path), format="json")
    return model, x, y, json.loads(path.read_text())


def test_split_json_feature_resolution_and_leaf_bit_order(fitted):
    model, x, _, export = fitted
    splits = parse_splits(export, model)
    assert set(splits.split_type) == {"FloatFeature", "OneHotFeature", "OnlineCtr"}
    assert splits.features.str.len().min() > 0
    leaf = model.calc_leaf_indexes(Pool(x, cat_features=["onehot", "ctr"]))
    for row in splits.loc[splits.split_type == "FloatFeature"].itertuples():
        expected = x[row.features].to_numpy(dtype=np.float32) > np.float32(row.border)
        assert np.array_equal((leaf[:, row.tree] >> row.depth) & 1, expected)


def test_feature_mapping_uses_names_and_preserves_order(fitted):
    model, x, _, _ = fitted
    original = x.copy()
    x = x.rename(columns={"early_model": "early_model_dub_22"})
    x = x[list(reversed(x.columns))]
    actual = feature_frame(model, x, {"feature_map": {"early_model": "early_model_dub_22"}})
    pd.testing.assert_frame_equal(original, actual)
    pd.testing.assert_frame_equal(x, x.copy())


def test_end_to_end_score_only_secondary_unripe(tmp_path):
    raw = raw_frame()
    raw["y6"] = raw.y
    raw.loc[raw.issue_dt >= "2026-04-01", "y6"] = np.nan
    source = tmp_path / "data.parquet"
    raw.to_parquet(source)
    cfg = dict(config(), data=str(source), output=str(tmp_path / "report"), secondary_targets=["y6"])
    tables = run(cfg, scores_only=True)
    assert tables["secondary_y6"].set_index("scope").loc["recent", "n"] == 0
    assert (tmp_path / "report" / "summary.md").exists()
    assert (tmp_path / "report" / "overview.png").exists()


def test_full_models_shap_and_tree_reports(fitted, tmp_path):
    model, x, y, _ = fitted
    raw = x.copy()
    raw["App_id"] = [f"application_{i}" for i in range(len(raw))]
    raw["issue_dt"] = ["2026-02-01"] * 200 + ["2026-04-01"] * 200
    raw["y"] = y
    raw["p22"] = raw["p23"] = model.predict_proba(x)[:, 1]
    path, source = tmp_path / "model.cbm", tmp_path / "data.parquet"
    model.save_model(str(path))
    raw.to_parquet(source)
    cfg = dict(config(), data=str(source), output=str(tmp_path / "full_report"),
               bootstrap_iterations=5, diagnostic_sample_per_window=100,
               shap_background_size=8, tree_step=10, max_local_trees=2, thread_count=2)
    for spec in cfg["models"].values():
        spec["path"] = str(path)
    tables = run(cfg)
    assert tables["gini_group_allocation"].gini_delta_allocation.abs().max() < 1e-10
    assert len(tables["tree_branches"]) > 0
    assert set(tables["tree_splits"].split_type) == {"FloatFeature", "OneHotFeature", "OnlineCtr"}
    assert tables["tree_blocks"].groupby("model").tree_end_exclusive.max().eq(model.tree_count_).all()


def test_retraining_refuses_time_leakage(tmp_path):
    from ablation import run_ablation
    raw = raw_frame().iloc[:8].copy()
    source = tmp_path / "same.parquet"
    raw.to_parquet(source)
    cfg = dict(config(), train=str(source), validation=str(source), test=str(source), output=str(tmp_path / "a"))
    with pytest.raises(ValueError, match="временной порядок"):
        run_ablation(cfg)
