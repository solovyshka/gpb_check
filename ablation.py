"""Controlled 2x2 retraining: FPS on/off x soc_dem on/off, fixed common scores.

This identifies conditional predictive effects in an OOT experiment, not a causal
effect on credit default. No test data are used for fitting or early stopping.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from diagnose import bootstrap_gap, gini, prepare_frame, read_table, scopes


def run_ablation(cfg):
    out = Path(cfg["output"])
    out.mkdir(parents=True, exist_ok=True)
    frames, ranges = {}, {}
    for split in ("train", "validation", "test"):
        df = read_table(cfg[split])
        if df[cfg["id_col"]].isna().any() or df[cfg["id_col"]].duplicated().any():
            raise ValueError(f"{split}: дубли/NA App_id")
        dt = pd.to_datetime(df[cfg["date_col"]], errors="raise")
        if dt.isna().any():
            raise ValueError(f"{split}: NA issue_dt")
        if cfg.get("max_issue_date"):
            df = df.loc[dt.dt.normalize() <= pd.Timestamp(cfg["max_issue_date"])].copy()
        df = df.loc[df[cfg["target"]].notna()].copy()
        if set(df[cfg["target"]].unique()) != {0, 1}:
            raise ValueError(f"{split}: нужны оба класса 0/1 и вызревшие таргеты")
        dt = pd.to_datetime(df[cfg["date_col"]])
        ranges[split] = (dt.min(), dt.max())
        frames[split] = df.reset_index(drop=True)
    if not (ranges["train"][1] < ranges["validation"][0]
            and ranges["validation"][1] < ranges["test"][0]):
        raise ValueError("Требуется строгий временной порядок train < validation < OOT")
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if set(frames[a][cfg["id_col"]]) & set(frames[b][cfg["id_col"]]):
            raise ValueError(f"Утечка App_id между {a} и {b}")
    base, soc = cfg["base_features"], cfg["soc_dem_features"]
    fps = cfg.get("fps_features") or [c for c in frames["train"].columns if c.lower().startswith(cfg.get("fps_prefix", "fps").lower())]
    if not base or not soc or not fps:
        raise ValueError("base, soc_dem и fps должны быть непустыми")
    if set(base) & set(soc) or set(base) & set(fps) or set(soc) & set(fps):
        raise ValueError("Группы признаков пересекаются")
    all_features = base + soc + fps
    if len(set(all_features)) != len(all_features):
        raise ValueError("Повтор признаков")
    if set(all_features) & {cfg["id_col"], cfg["date_col"], cfg["target"], "flag_4m_30p_gross", "flag_6m_30p_gross"}:
        raise ValueError("ID, дата и таргеты не должны входить в признаки")
    if not set(cfg.get("categorical_features", [])) <= set(all_features):
        raise ValueError("categorical_features не входят в выбранные признаки")
    for split, df in frames.items():
        missing = set(all_features) - set(df.columns)
        if missing:
            raise ValueError(f"{split}: нет {sorted(missing)}")
        for col in cfg.get("categorical_features", []):
            if df[col].isna().any():
                raise ValueError(f"{split}/{col}: нужен production preprocessing NA")
            df[col] = df[col].astype(str)
    params = dict(cfg["params"])
    if params.get("use_best_model", False) or any(k in params for k in ("early_stopping_rounds", "od_wait", "od_type", "cat_features", "random_seed", "random_state")):
        raise ValueError("Для чистого сравнения фиксируйте iterations; early stopping и random_seed в params запрещены")
    params.update(use_best_model=False, allow_writing_files=False)
    variants = {"base": base, "soc_dem": base + soc, "fps": base + fps, "soc_dem_fps": base + soc + fps}
    metrics, effects, predictions = [], [], []
    for seed in cfg.get("seeds", [42, 1729, 2026]):
        test = frames["test"].copy()
        for variant, features in variants.items():
            model = CatBoostClassifier(**params, random_seed=seed)
            cat = [c for c in cfg.get("categorical_features", []) if c in features]
            model.fit(frames["train"][features], frames["train"][cfg["target"]], cat_features=cat,
                      eval_set=(frames["validation"][features], frames["validation"][cfg["target"]]))
            model.save_model(str(out / f"{variant}_seed{seed}.cbm"))
            for split, df in frames.items():
                pred = model.predict_proba(df[features])[:, 1]
                metrics.append(dict(seed=seed, variant=variant, scope=split, n=len(df), defaults=int(df[cfg["target"]].sum()),
                                    tree_count=model.tree_count_, gini=gini(df[cfg["target"]], pred)))
                if split == "test":
                    test["_pred_" + variant] = pred
        # Reuse pairing/maturity/window checks with base as an anchor; no model fitting here.
        analysis_cfg = dict(cfg, models={"dub_22": {"score_col": "_pred_base"},
                                         "dub_23": {"score_col": "_pred_soc_dem_fps"}}, seed=seed)
        evaluated, _ = prepare_frame(test, analysis_cfg)
        for scope, part in scopes(evaluated):
            y = part[cfg["target"]]
            ginis = {v: gini(y, part["_pred_" + v]) for v in variants}
            for v, value in ginis.items():
                metrics.append(dict(seed=seed, variant=v, scope="oot_" + scope,
                                    n=len(part), defaults=int(y.sum()), tree_count=params.get("iterations", 1000), gini=value))
            comparisons = [("add_fps_without_soc_dem", "base", "fps"),
                           ("add_fps_with_soc_dem", "soc_dem", "soc_dem_fps"),
                           ("add_soc_dem_without_fps", "base", "soc_dem"),
                           ("add_soc_dem_with_fps", "fps", "soc_dem_fps"),
                           ("dub23_structure_minus_dub22_structure", "soc_dem", "fps")]
            for contrast, before, after in comparisons:
                work = part.assign(_risk_dub_22=part["_pred_" + before], _risk_dub_23=part["_pred_" + after])
                ci = bootstrap_gap(work, analysis_cfg)
                effects.append(dict(seed=seed, scope=scope, contrast=contrast, n=len(part),
                                    delta_gini=ginis[after] - ginis[before], **ci))
            effects.append(dict(seed=seed, scope=scope, contrast="fps_soc_dem_interaction", n=len(part),
                                delta_gini=ginis["soc_dem_fps"] - ginis["soc_dem"] - ginis["fps"] + ginis["base"]))
        saved = test[[cfg["id_col"], cfg["date_col"], cfg["target"]] + ["_pred_" + v for v in variants]].copy()
        saved["seed"] = seed
        predictions.append(saved)
    pd.DataFrame(metrics).to_csv(out / "ablation_metrics.csv", index=False)
    pd.DataFrame(effects).to_csv(out / "ablation_effects.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_parquet(out / "ablation_predictions.parquet", index=False)
    resolved = dict(cfg, resolved_fps_features=fps, variants=variants, date_ranges={k: [str(t) for t in v] for k, v in ranges.items()})
    (out / "resolved_config.json").write_text(json.dumps(resolved, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Ablation готов: {out.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_ablation(json.loads(Path(args.config).read_text(encoding="utf-8")))
