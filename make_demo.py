"""Generate synthetic data and small real CatBoost models for a smoke test.

Every output is synthetic; model deterioration is intentionally simulated.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from catboost import CatBoostClassifier


def make_demo(directory="demo_data", n_per_month=500, iterations=60):
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    frames = []
    for month in pd.period_range("2025-01", "2026-05", freq="M"):
        n = n_per_month
        early, long, soc, common = rng.normal(size=(4, n))
        region = rng.choice(["A", "B", "C", "D", "E", "F"], size=n)
        logit = -2.1 + 1.6 * early + 1.1 * long + 1.0 * soc + 0.5 * common + 0.25 * (region == "A")
        y = rng.binomial(1, 1 / (1 + np.exp(-logit)))
        recent = str(month) >= "2026-03"
        fps = (-1 if recent else 1) * (soc + 0.7 * common) + rng.normal(0, 0.2, n)
        frames.append(pd.DataFrame({
            "App_id": [f"{month}_{i:05d}" for i in range(n)], "issue_dt": month.start_time,
            "flag_4m_30p_gross": y.astype(float),
            "flag_6m_30p_gross": np.maximum(y, rng.binomial(1, 0.04, n)).astype(float),
            "early_model_dub_22": early, "long_model_dub_22": long, "soc_dem_dub_22": soc,
            "early_model_dub_23": early + rng.normal(0, 0.08 if not recent else 0.5, n),
            "long_model_dub_23": long + rng.normal(0, 0.08, n),
            "common_feature_1": common, "region": region, "fps_signal": fps,
            "fps_noise": rng.normal(size=n)}))
    data = pd.concat(frames, ignore_index=True)
    data.loc[data.issue_dt >= "2026-05-01", "flag_4m_30p_gross"] = np.nan
    data.loc[data.issue_dt >= "2026-03-01", "flag_6m_30p_gross"] = np.nan
    train = data.loc[data.issue_dt < "2025-10-01"]
    cfg = json.loads(Path(__file__).with_name("config.example.json").read_text())
    native22 = ["early_model", "long_model", "soc_dem", "common_feature_1", "region"]
    native23 = ["early_model", "long_model", "common_feature_1", "region", "fps_signal", "fps_noise"]
    for name, native in (("dub_22", native22), ("dub_23", native23)):
        sources = [cfg["models"][name]["feature_map"].get(c, c) for c in native]
        x = train[sources].copy()
        x.columns = native
        model = CatBoostClassifier(iterations=iterations, depth=4, learning_rate=0.1,
                                   random_seed=42, thread_count=2, verbose=False,
                                   allow_writing_files=False, one_hot_max_size=2)
        model.fit(x, train.flag_4m_30p_gross, cat_features=["region"])
        path = out / (name + ".cbm")
        model.save_model(str(path))
        cfg["models"][name]["path"] = str(path.resolve())
        all_x = data[sources].copy()
        all_x.columns = native
        data["pd_" + name] = model.predict_proba(all_x)[:, 1]
        data["score_" + name] = model.predict(all_x, prediction_type="RawFormulaVal")
        cfg["analysis_levels"]["final_models"][name]["raw_kind"] = "margin"
        cfg["analysis_levels"]["final_models"][name]["pd_target"] = "flag_4m_30p_gross"
    for component, versions in cfg["analysis_levels"]["submodels"].items():
        for name, spec in versions.items():
            data[spec["pd_col"]] = expit(data[spec["raw_score_col"]] - 1.5)
    path = out / "scoring.parquet"
    data.loc[data.issue_dt >= "2025-12-01"].to_parquet(path, index=False)
    cfg.update(data=str(path.resolve()), output=str((out / "report").resolve()), bootstrap_iterations=100,
               tree_step=10, max_local_trees=8, diagnostic_sample_per_window=1200,
               shap_background_size=16, thread_count=2, segments=["region"])
    (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    for split, mask in (("train", data.issue_dt < "2025-10-01"),
                        ("validation", (data.issue_dt >= "2025-10-01") & (data.issue_dt < "2025-12-01")),
                        ("test", data.issue_dt >= "2025-12-01")):
        data.loc[mask].to_parquet(out / (split + ".parquet"), index=False)
    abl = json.loads(Path(__file__).with_name("ablation.example.json").read_text())
    abl.update({s: str((out / (s + ".parquet")).resolve()) for s in ("train", "validation", "test")})
    abl.update(output=str((out / "ablation_report").resolve()),
               base_features=["early_model_dub_22", "long_model_dub_22", "common_feature_1", "region"],
               categorical_features=["region"], seeds=[42], bootstrap_iterations=30)
    abl["params"].update(iterations=iterations, depth=4, thread_count=2)
    (out / "ablation_config.json").write_text(json.dumps(abl, indent=2), encoding="utf-8")
    print(f"Синтетический пример: {out.resolve()}")
    return cfg


if __name__ == "__main__":
    make_demo()
