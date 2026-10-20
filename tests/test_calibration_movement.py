import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from calibration_movement import plot_dub23_calibration
from diagnose import gini

SEGMENTS = ["sum_request_less_500", "sum_request_500_1000", "sum_request_more_1000k"]


def data_frame():
    rng = np.random.default_rng(23)
    n = 120
    score = rng.normal(size=n)
    segment = np.arange(n) % 3
    return pd.DataFrame({
        "App_id": [f"id_{i}" for i in range(n)],
        "quarter": ["2025Q4"] * 60 + ["2026Q1"] * 60,
        "issue_dt": ["2025-12-01"] * 60 + ["2026-03-01"] * 60,
        "flag_4m_30p_gross": np.tile([0, 0, 1, 0, 1, 0], 20).astype(float),
        "score_dub_23": score,
        "pd_dub_23": expit(score + np.array([-1.5, 0, 1.5])[segment]),
        "segment": np.asarray(SEGMENTS)[segment],
        **{name: (segment == i).astype(int) for i, name in enumerate(SEGMENTS)},
    })


def call(data, **kwargs):
    options = dict(group_col="quarter", reference_value="2025Q4", comparison_value="2026Q1",
                   bootstrap_iterations=5, group_order=["2025Q4", "2026Q1"],
                   min_defaults=2, min_goods=2)
    options.update(kwargs)
    return plot_dub23_calibration(data, **options)


def test_one_call_needs_no_dub22_no_config_and_returns_renderable_figures(tmp_path):
    data = data_frame()
    original = data.copy(deep=True)
    result = call(data, output_dir=tmp_path)
    assert set(result["figures"]) == {"rank_migration", "rank_shift_timeline", "segment_pairs", "raw_pd_maps"}
    assert len(result["saved_files"]) == 10
    for figure in result["figures"].values():
        output = io.BytesIO()
        figure.savefig(output, format="png")
        assert output.tell() > 1000
    assert len(list(tmp_path.glob("*.png"))) == 4
    assert not plt.get_fignums()
    pd.testing.assert_frame_equal(data, original)


def test_ranks_are_within_entire_group_not_within_segments_and_not_sampled():
    data = data_frame()
    full = call(data, scatter_max_goods=3000)
    sampled = call(data, scatter_max_goods=0)
    for name in full["tables"]:
        pd.testing.assert_frame_equal(full["tables"][name], sampled["tables"][name])
    points = full["tables"]["rank_movements"]
    expected = 100 * (data.groupby("quarter").score_dub_23.rank(method="average") - .5) / 60
    assert points._cal_raw_rank.to_numpy() == pytest.approx(expected.to_numpy())
    assert points.groupby("_cal_group")._cal_rank_shift.mean().abs().max() < 1e-10
    for axis, group in zip(sampled["figures"]["rank_migration"].axes, ["2025Q4", "2026Q1"]):
        shown_bads = sum(len(collection.get_offsets()) for collection in axis.collections[1::2])
        assert shown_bads == data.loc[data.quarter == group, "flag_4m_30p_gross"].sum()


def test_pair_matrix_matches_exhaustive_counts_and_global_gini():
    data = data_frame()
    result = call(data)
    pairs = result["tables"]["segment_pairs"]
    for group, part in data.groupby("quarter"):
        current = pairs.loc[pairs.group == group]
        y = part.flag_4m_30p_gross.to_numpy()
        assert current.gini_delta_contribution.sum() == pytest.approx(gini(y, part.pd_dub_23) - gini(y, part.score_dub_23))
        for row in current.itertuples():
            bad = part.loc[(part.flag_4m_30p_gross == 1) & (part.segment == row.bad_segment)]
            good = part.loc[(part.flag_4m_30p_gross == 0) & (part.segment == row.good_segment)]
            before = np.sign(bad.score_dub_23.to_numpy()[:, None] - good.score_dub_23.to_numpy())
            after = np.sign(bad.pd_dub_23.to_numpy()[:, None] - good.pd_dub_23.to_numpy())
            assert row.pairs_bad_good_total == len(bad) * len(good)
            assert row.pairs_improved == int((after > before).sum())
            assert row.pairs_worsened == int((after < before).sum())
        diagonal = current.loc[current.bad_segment == current.good_segment]
        assert diagonal.pair_net_units.eq(0).all()
    within = result["tables"]["gini_by_segment"].query("segment != '__all__'")
    assert within.delta_gini.abs().max() < 1e-12


def test_numeric_non_time_groups_categorical_segments_and_missing_display_group():
    data = data_frame().drop(columns=SEGMENTS)
    data["partition"] = np.where(data.quarter == "2025Q4", 10, 20)
    result = call(data, group_col="partition", reference_value=10, comparison_value=20,
                  group_order=[20, 15, 10], segment_col="segment", segment_order=SEGMENTS)
    assert result["group_order"] == [20, 15, 10]
    assert set(result["tables"]["rank_movements"]._cal_group) == {10, 20}
    assert set(result["tables"]["segment_pairs"].group) == {10, 20}


def test_string_group_values_all_and_reference_are_not_reserved():
    data = data_frame()
    data["partition"] = data.quarter.map({"2025Q4": "all", "2026Q1": "reference"})
    result = call(data, group_col="partition", reference_value="all", comparison_value="reference",
                  group_order=["all", "reference"])
    assert len(result["tables"]["segment_pairs"]) == 18


def test_missing_target_raw_group_and_cutoff_are_audited():
    data = data_frame()
    data.loc[0, "flag_4m_30p_gross"] = np.nan
    data.loc[1, "score_dub_23"] = np.nan
    data.loc[2, "quarter"] = None
    data.loc[3, "issue_dt"] = "2026-04-01"
    result = call(data, date_col="issue_dt", max_issue_date="2026-03-31")
    assert len(result["tables"]["rank_movements"]) == len(data) - 4
    audit = result["tables"]["cohort_audit"]
    assert audit.n_missing_raw_pd.sum() == 1
    assert audit.n_missing_group.sum() == 1
    assert audit.n_used.sum() == len(data) - 4


@pytest.mark.parametrize("change, message", [
    ("flags", "ровно один"), ("flag_na", "0/1"), ("pd", r"\[0, 1\]"),
    ("duplicate", "уникальный"), ("target", "0/1/NA"),
])
def test_invalid_data_fails_clearly(change, message):
    data = data_frame()
    if change == "flags":
        data.loc[0, SEGMENTS] = 1
    elif change == "flag_na":
        data.loc[0, SEGMENTS[0]] = np.nan
    elif change == "pd":
        data.loc[0, "pd_dub_23"] = 1.5
    elif change == "duplicate":
        data.loc[0, "App_id"] = data.loc[1, "App_id"]
    else:
        data.loc[0, "flag_4m_30p_gross"] = 2
    with pytest.raises(ValueError, match=message):
        call(data)


def test_validation_of_group_values_and_order():
    with pytest.raises(ValueError, match="нет группы"):
        call(data_frame(), comparison_value="absent")
    with pytest.raises(ValueError, match="group_order"):
        call(data_frame(), group_order=["2025Q4"])
    with pytest.raises(ValueError, match="два разных"):
        call(data_frame(), comparison_value="2025Q4")
    with pytest.raises(ValueError, match="date_col"):
        call(data_frame(), max_issue_date="2026-03-31")


def test_ties_negative_score_direction_and_ordered_categories():
    data = data_frame()
    data.score_dub_23 = -data.score_dub_23
    data.pd_dub_23 = data.pd_dub_23.round(1)
    data.quarter = pd.Categorical(data.quarter, categories=["2026Q1", "2025Q4"], ordered=True)
    result = call(data, higher_score_higher_risk=False, group_order=None)
    assert result["group_order"] == ["2026Q1", "2025Q4"]
    for _, part in result["tables"]["gini_by_segment"].query("segment == '__all__'").iterrows():
        original = data.loc[data.quarter == part["group"]]
        assert part.gini_raw == pytest.approx(gini(original.flag_4m_30p_gross, -original.score_dub_23))
    assert result["tables"]["segment_pairs"].pairs_correct_to_tie.sum() > 0
