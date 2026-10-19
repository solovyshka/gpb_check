"""Paired, unweighted diagnostics of binary CatBoost risk models.

Usage: python diagnose.py --config config.example.json [--scores-only]
All Gini values are on [-1, 1]; multiply by 100 for Gini points.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from ranking import pair_transitions, per_default_pair_statistics, transition_statistics

NAMES = ("dub_22", "dub_23")
GROUPS = ("early_model", "long_model", "soc_dem", "fps", "other")


def read_table(path):
    path = Path(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("Поддерживаются CSV и Parquet")


def gini(y, score, weights=None):
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    good = np.isfinite(score) & pd.notna(y)
    if weights is not None:
        weights = np.asarray(weights)
        good &= weights > 0
    if len(np.unique(y[good])) != 2:
        return np.nan
    return 2 * roc_auc_score(y[good], score[good],
                            sample_weight=None if weights is None else weights[good]) - 1


def default_quality(y, score):
    """For each bad: P(score_bad > score_good) + 0.5 P(tie), exactly.

    Mean over bads is ROC AUC. No sampling and no N_bad x N_good matrix.
    """
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    goods = np.sort(score[y == 0])
    bads = score[y == 1]
    if not len(goods):
        return np.full(len(bads), np.nan)
    left = np.searchsorted(goods, bads, side="left")
    right = np.searchsorted(goods, bads, side="right")
    return (left + right) / (2.0 * len(goods))


def bootstrap_gap(frame, cfg):
    """Paired bootstrap, stratified by y or clustered by customer ID.

    This is a CI conditional on fitted models, not training uncertainty.
    """
    y = frame[cfg["target"]].to_numpy(dtype=int)
    a, b = (frame["_risk_" + m].to_numpy() for m in NAMES)
    if len(np.unique(y)) < 2:
        return {"ci_low": np.nan, "ci_high": np.nan, "bootstrap_valid": 0}
    rng = np.random.default_rng(cfg.get("seed", 42))
    cluster = cfg.get("cluster_col")
    if cluster:
        codes, levels = pd.factorize(frame[cluster], sort=False)
        if (codes < 0).any():
            raise ValueError("cluster_col содержит NA")
    results = []
    for _ in range(cfg.get("bootstrap_iterations", 500)):
        if cluster:
            counts = np.bincount(rng.integers(len(levels), size=len(levels)),
                                 minlength=len(levels))
            w = counts[codes]
        else:
            w = np.zeros(len(y), dtype=int)
            for label in (0, 1):
                ix = np.flatnonzero(y == label)
                w += np.bincount(rng.choice(ix, len(ix), replace=True), minlength=len(y))
        results.append(gini(y, b, w) - gini(y, a, w))
    finite = np.asarray(results)[np.isfinite(results)]
    lo, hi = np.quantile(finite, [0.025, 0.975]) if len(finite) else (np.nan, np.nan)
    return {"ci_low": lo, "ci_high": hi, "bootstrap_valid": len(finite)}


def window_gap_change(df, cfg):
    """Difference in model gaps between recent/reference, with joint cluster bootstrap."""
    part = df.loc[df["_window"].isin(["reference", "recent"])].reset_index(drop=True)
    y = part[cfg["target"]].to_numpy(dtype=int)
    a, b = (part["_risk_" + n].to_numpy() for n in NAMES)
    windows = {w: (part["_window"] == w).to_numpy() for w in ("reference", "recent")}

    def evaluate(weights=None):
        values = {}
        for window, mask in windows.items():
            for name, score in (("22", a), ("23", b)):
                values[window + "_" + name] = gini(y[mask], score[mask], None if weights is None else weights[mask])
        drops = {n: values["recent_" + n] - values["reference_" + n] for n in ("22", "23")}
        return drops["23"] - drops["22"], drops

    delta, drops = evaluate()
    cluster = cfg.get("cluster_col")
    if cluster:
        codes, levels = pd.factorize(part[cluster], sort=False)
        if (codes < 0).any():
            raise ValueError("cluster_col содержит NA")
    rng = np.random.default_rng(cfg.get("seed", 42))
    draws = []
    for _ in range(cfg.get("bootstrap_iterations", 500)):
        if cluster:
            weights = np.bincount(rng.integers(len(levels), size=len(levels)), minlength=len(levels))[codes]
        else:
            weights = np.zeros(len(part), dtype=int)
            for mask in windows.values():
                for label in (0, 1):
                    ix = np.flatnonzero(mask & (y == label))
                    if len(ix):
                        weights += np.bincount(rng.choice(ix, size=len(ix), replace=True), minlength=len(part))
        draws.append(evaluate(weights)[0])
    valid = np.asarray(draws)[np.isfinite(draws)]
    lo, hi = np.quantile(valid, [0.025, 0.975]) if len(valid) else (np.nan, np.nan)
    return pd.DataFrame([dict(gini_change_22=drops["22"], gini_change_23=drops["23"],
                              gap_change_recent_minus_reference=delta,
                              ci_low=lo, ci_high=hi, bootstrap_valid=len(valid))])


def validate_config(cfg):
    for key in ("data", "output", "id_col", "date_col", "target", "models"):
        if key not in cfg:
            raise ValueError(f"В конфиге отсутствует {key}")
    if set(cfg["models"]) != set(NAMES):
        raise ValueError("models должны содержать dub_22 и dub_23")
    if cfg.get("bootstrap_iterations", 500) < 0:
        raise ValueError("bootstrap_iterations должен быть >= 0")
    if cfg.get("tree_step", 25) < 1 or cfg.get("max_local_trees", 30) < 0:
        raise ValueError("Некорректные лимиты анализа деревьев")
    if cfg.get("diagnostic_sample_per_window", 5000) < 2:
        raise ValueError("diagnostic_sample_per_window должен быть >= 2")
    if cfg.get("shap_background_size", 64) < 1:
        raise ValueError("shap_background_size должен быть >= 1")
    if any(not 0 < x < 1 for x in cfg.get("reject_rates", [0.1, 0.2])):
        raise ValueError("reject_rates должны лежать строго между 0 и 1")


def prepare_frame(raw, cfg):
    """Never replaces an unobserved target with 0; paired complete cases only."""
    df = raw.copy()
    required = [cfg["id_col"], cfg["date_col"], cfg["target"]]
    if cfg.get("cluster_col"):
        required.append(cfg["cluster_col"])
    for c in required:
        if c not in df:
            raise ValueError(f"Нет обязательной колонки {c}")
    if df[cfg["id_col"]].isna().any() or df[cfg["id_col"]].duplicated().any():
        raise ValueError("App_id должен быть непустым и уникальным: проверьте join/дубли")
    dt = pd.to_datetime(df[cfg["date_col"]], errors="coerce")
    if dt.isna().any():
        raise ValueError("issue_dt содержит пустые или некорректные даты")
    df["_period"] = dt.dt.to_period("M").astype(str)
    target = pd.to_numeric(df[cfg["target"]], errors="raise")
    if not target.dropna().isin([0, 1]).all():
        raise ValueError("Таргет должен быть 0/1/NA; NA означает неизвестный исход")
    df[cfg["target"]] = target
    cutoff = cfg.get("max_issue_date")
    eligible = np.ones(len(df), dtype=bool)
    if cutoff:
        eligible = dt.dt.normalize() <= pd.Timestamp(cutoff)
    for name in NAMES:
        col = cfg["models"][name].get("score_col")
        if not col or col not in df:
            raise ValueError(f"Нет скора {name}: {col}")
        s = pd.to_numeric(df[col], errors="raise")
        df["_risk_" + name] = s * (1 if cfg["models"][name].get("higher_is_risk", True) else -1)
    finite = np.isfinite(df[["_risk_" + n for n in NAMES]].to_numpy(dtype=float)).all(axis=1)
    labeled = target.notna()
    paired = eligible & labeled & finite
    audit = df[["_period"]].assign(
        n_raw=1, n_within_cutoff=np.asarray(eligible, dtype=int),
        n_target_observed=labeled.astype(int),
        n_target_observed_within_cutoff=(eligible & labeled).astype(int),
        n_eligible_missing_score=(eligible & labeled & ~finite).astype(int),
        n_paired=np.asarray(paired, dtype=int),
    ).groupby("_period", as_index=False).sum()
    result = df.loc[paired].reset_index(drop=True)
    if result.empty:
        raise ValueError("Нет вызревших объектов с двумя скорами")
    periods = sorted(result["_period"].unique())
    recent = cfg.get("recent_periods") or periods[-2:]
    reference = cfg.get("reference_periods") or periods[max(0, len(periods) - 5):-2]
    if set(recent) & set(reference):
        raise ValueError("reference_periods и recent_periods пересекаются")
    if not reference or not recent:
        raise ValueError("Нужны непересекающиеся reference_periods и recent_periods")
    missing = (set(reference) | set(recent)) - set(periods)
    if missing and not cfg.get("allow_missing_window_periods", False):
        raise ValueError(f"Нет парных вызревших наблюдений за периоды {sorted(missing)}")
    result["_window"] = np.select(
        [result["_period"].isin(reference), result["_period"].isin(recent)],
        ["reference", "recent"], default="other")
    cfg["reference_periods"], cfg["recent_periods"] = list(reference), list(recent)
    return result, audit


def scopes(df, include_all=True):
    for period, part in df.groupby("_period", sort=True):
        yield period, part
    for window in ("reference", "recent"):
        yield window, df.loc[df["_window"] == window]
    if include_all:
        yield "all", df


def metric_table(df, cfg, bootstrap=True):
    rows = []
    for scope, part in scopes(df):
        y = part[cfg["target"]].to_numpy(dtype=int)
        a, b = (part["_risk_" + name].to_numpy() for name in NAMES)
        qa, qb = default_quality(y, a), default_quality(y, b)
        diff = qb - qa
        row = dict(scope=scope, n=len(y), defaults=int(y.sum()), goods=int((y == 0).sum()),
                   bad_rate=float(y.mean()) if len(y) else np.nan,
                   gini_dub_22=gini(y, a), gini_dub_23=gini(y, b),
                   defaults_better=int((diff > 1e-12).sum()),
                   defaults_worse=int((diff < -1e-12).sum()),
                   defaults_unchanged=int((np.abs(diff) <= 1e-12).sum()),
                   defaults_not_comparable=int((~np.isfinite(diff)).sum()))
        row["delta_gini_23_minus_22"] = row["gini_dub_23"] - row["gini_dub_22"]
        row.update(transition_statistics(pair_transitions(y, a, b)))
        row["reliable_size"] = row["defaults"] >= cfg.get("min_defaults", 30) and row["goods"] >= cfg.get("min_goods", 30)
        if bootstrap:
            row.update(bootstrap_gap(part, cfg))
        rows.append(row)
    return pd.DataFrame(rows)


def reranked_defaults(df, cfg):
    rows = []
    for period, part in df.groupby("_period", sort=True):
        y = part[cfg["target"]].to_numpy(dtype=int)
        bad = part.loc[y == 1, [cfg["id_col"], "_period", "_window"]].copy()
        for name in NAMES:
            bad["risk_" + name] = part.loc[y == 1, "_risk_" + name].to_numpy()
            bad["quality_" + name] = default_quality(y, part["_risk_" + name])
        bad["quality_delta"] = bad["quality_dub_23"] - bad["quality_dub_22"]
        bad["status"] = np.select([bad["quality_delta"] > 1e-12,
                                   bad["quality_delta"] < -1e-12,
                                   bad["quality_delta"].abs() <= 1e-12],
                                  ["better", "worse", "unchanged"], default="not_comparable")
        # Pair-units are a NET balance, with ties worth 0.5. Not gross pair flips.
        bad["net_pair_units"] = bad["quality_delta"] * (y == 0).sum()
        bad["gini_delta_contribution"] = 2 * bad["quality_delta"] / max(1, y.sum())
        pair_counts = pair_transitions(y, part["_risk_dub_22"], part["_risk_dub_23"])
        for column, values in per_default_pair_statistics(pair_counts).items():
            bad[column] = values
        rows.append(bad)
    return pd.concat(rows, ignore_index=True)


def top_risk_diagnostics(df, cfg):
    summary, migrations = [], []
    for scope, part in scopes(df):
        if not len(part):
            continue
        y = part[cfg["target"]].to_numpy(dtype=int)
        ids = part[cfg["id_col"]].astype(str).to_numpy()
        for rate in cfg.get("reject_rates", [0.1, 0.2]):
            k = min(len(part), math.ceil(rate * len(part)))
            masks, expectations, tie_counts = {}, {}, {}
            for name in NAMES:
                score = part["_risk_" + name].to_numpy()
                # Same deterministic ID tie breaker in both models.
                order = np.lexsort((ids, -score))
                mask = np.zeros(len(part), bool)
                mask[order[:k]] = True
                masks[name] = mask
                threshold = score[order[k - 1]]
                tied = score == threshold
                above = score > threshold
                expectations[name] = float(y[above].sum() + y[tied].sum() * (k - above.sum()) / tied.sum())
                tie_counts[name] = int(tied.sum())
            a, b = masks["dub_22"], masks["dub_23"]
            gained, lost = ~a & b, a & ~b
            row = dict(scope=scope, reject_rate=rate, rejected_n=k,
                       defaults_caught_22=int(y[a].sum()), defaults_caught_23=int(y[b].sum()),
                       defaults_newly_caught=int(y[gained].sum()), defaults_lost=int(y[lost].sum()),
                       goods_newly_rejected=int((1 - y[gained]).sum()),
                       goods_released=int((1 - y[lost]).sum()))
            row["net_defaults_caught"] = row["defaults_newly_caught"] - row["defaults_lost"]
            row["recall_22"] = y[a].sum() / y.sum() if y.sum() else np.nan
            row["recall_23"] = y[b].sum() / y.sum() if y.sum() else np.nan
            for name in NAMES:
                row["expected_defaults_random_ties_" + name] = expectations[name]
                row["boundary_ties_" + name] = tie_counts[name]
            summary.append(row)
            # Detail only within months: no repeated all/reference/recent rows.
            if scope not in {"all", "reference", "recent"}:
                out = part.loc[gained | lost, [cfg["id_col"], "_period", cfg["target"]]].copy()
                out["reject_rate"] = rate
                out["migration"] = np.where(gained[gained | lost], "newly_rejected_23", "released_23")
                migrations.append(out)
    return pd.DataFrame(summary), pd.concat(migrations, ignore_index=True) if migrations else pd.DataFrame()


def feature_group(name):
    name = name.lower()
    for prefix in GROUPS[:-1]:
        if name.startswith(prefix):
            return prefix
    return "other"


def component_metrics(df, cfg):
    rows = []
    for col, spec in cfg.get("components", {}).items():
        if col not in df:
            continue
        for scope, part in scopes(df):
            s = pd.to_numeric(part[col], errors="raise")
            finite = np.isfinite(s)
            y = part.loc[finite, cfg["target"]]
            score = s[finite] * (1 if spec.get("higher_is_risk", True) else -1)
            rows.append(dict(component=col, scope=scope, n=int(finite.sum()),
                             defaults=int(y.sum()), missing_rate=1 - finite.mean(),
                             gini=gini(y, score), reliable_size=int(y.sum()) >= cfg.get("min_defaults", 30)
                             and int((y == 0).sum()) >= cfg.get("min_goods", 30)))
    return pd.DataFrame(rows)


def score_correlations(df, cfg):
    rows = []
    pairs = [("pd_dub_22", "pd_dub_23")]
    for group in ("early_model", "long_model", "soc_dem"):
        pairs.append((group + "_dub_22", group + "_dub_23"))
    for a, b in pairs:
        if a not in df or b not in df:
            continue
        for scope, part in scopes(df):
            aa, bb = (pd.to_numeric(part[c], errors="raise").to_numpy(dtype=float) for c in (a, b))
            finite = np.isfinite(aa) & np.isfinite(bb)
            corr = float(spearmanr(aa[finite], bb[finite]).statistic) if finite.sum() > 1 and np.ptp(aa[finite]) and np.ptp(bb[finite]) else np.nan
            rows.append(dict(scope=scope, column_22=a, column_23=b, n=int(finite.sum()),
                             spearman=corr, equal_values=int(np.isclose(aa[finite], bb[finite], rtol=0, atol=1e-12).sum())))
    return pd.DataFrame(rows)


def psi(ref, recent, bins=10):
    """Reference-only bins, separate NA bucket, smoothed normalized counts."""
    numeric = pd.api.types.is_numeric_dtype(ref)
    if numeric:
        r = pd.to_numeric(ref).replace([np.inf, -np.inf], np.nan)
        t = pd.to_numeric(recent).replace([np.inf, -np.inf], np.nan)
        values = r.dropna().to_numpy(dtype=float)
        if len(np.unique(values)) <= 1:
            # Constant-reference change must not disappear in a single (-inf,inf) bin.
            def encode(s):
                return np.where(s.isna(), "NA", np.where(s == values[0], "same", "changed")) if len(values) else np.where(s.isna(), "NA", "observed")
        else:
            edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)[1:-1]))
            def encode(s):
                return np.where(s.isna(), "NA", np.searchsorted(edges, s.to_numpy(dtype=float), side="right").astype(str))
    else:
        levels = set(ref.dropna().astype(str).value_counts().head(30).index)
        def encode(s):
            text = s.astype(str)
            return np.where(s.isna(), "NA", np.where(text.isin(levels), "V:" + text, "OTHER"))
    a, b = pd.Series(encode(ref)).value_counts(), pd.Series(encode(recent)).value_counts()
    keys = a.index.union(b.index)
    av, bv = a.reindex(keys, fill_value=0).to_numpy() + 0.5, b.reindex(keys, fill_value=0).to_numpy() + 0.5
    av, bv = av / av.sum(), bv / bv.sum()
    return float(np.sum((bv - av) * np.log(bv / av)))


def drift_table(df, columns):
    ref, recent = (df.loc[df["_window"] == s] for s in ("reference", "recent"))
    rows = []
    for col in sorted(set(columns)):
        if col not in df:
            continue
        rows.append(dict(feature=col, group=feature_group(col), psi=psi(ref[col], recent[col]),
                         missing_reference=float(ref[col].isna().mean()),
                         missing_recent=float(recent[col].isna().mean()),
                         unique_reference=int(ref[col].nunique()), unique_recent=int(recent[col].nunique())))
    return pd.DataFrame(rows).sort_values("psi", ascending=False) if rows else pd.DataFrame()


def segment_table(df, cfg):
    rows = []
    for column in cfg.get("segments", []):
        if column not in df:
            raise ValueError(f"Нет сегмента {column}")
        if df[column].nunique() > 100:
            raise ValueError(f"Сегмент {column}: >100 значений; задайте осмысленные интервалы заранее")
        for scope, window in scopes(df, include_all=False):
            for value, part in window.groupby(column, dropna=False, observed=True):
                y = part[cfg["target"]]
                a, b = (gini(y, part["_risk_" + n]) for n in NAMES)
                rows.append(dict(segment=column, value=str(value), scope=scope, n=len(part),
                                 defaults=int(y.sum()), gini_22=a, gini_23=b, delta_gini=b - a,
                                 reliable_size=y.sum() >= cfg.get("min_defaults", 30)
                                 and (y == 0).sum() >= cfg.get("min_goods", 30)))
    return pd.DataFrame(rows)


def feature_frame(model, data, spec):
    """Use model-native names; mapping only selects source data columns."""
    names = list(model.feature_names_)
    if not names or len(set(names)) != len(names):
        raise ValueError("Модель должна содержать уникальные имена признаков")
    if model.get_text_feature_indices() or model.get_embedding_feature_indices():
        raise ValueError("Text/embedding требуют отдельного pipeline; этот шаблон поддерживает numeric/categorical")
    mapping = spec.get("feature_map", {})
    source = [mapping.get(n, n) for n in names]
    missing = set(source) - set(data.columns)
    if missing:
        raise ValueError(f"В данных отсутствуют признаки модели: {sorted(missing)}")
    x = data[source].copy()
    x.columns = names
    for idx in model.get_cat_feature_indices():
        # Do not invent a categorical imputation token different from production.
        col = names[idx]
        if x[col].isna().any():
            raise ValueError(f"Категориальный {col} содержит NA: примените production preprocessing")
        x[col] = x[col].astype(str)
    return x


def load_models_and_scores(raw, cfg, scores_only):
    models, checks = {}, []
    for name, spec in cfg["models"].items():
        path = spec.get("path")
        if path and not scores_only:
            from catboost import CatBoostClassifier
            model = CatBoostClassifier()
            model.load_model(path)
            if list(model.classes_) != [0, 1]:
                raise ValueError(f"{name}: требуется бинарная модель с классами [0, 1]")
            models[name] = model
            # Only score rows used by the primary evaluation; do not require inputs on unripe tails.
            dt = pd.to_datetime(raw[cfg["date_col"]], errors="raise")
            mask = raw[cfg["target"]].notna()
            if cfg.get("max_issue_date"):
                mask &= dt.dt.normalize() <= pd.Timestamp(cfg["max_issue_date"])
            ix = raw.index[mask]
            if not len(ix):
                raise ValueError("Нет вызревших строк для проверки модели")
            model_x = feature_frame(model, raw.loc[ix], spec)
            pred = model.predict_proba(model_x, thread_count=cfg.get("thread_count", 4))[:, 1]
            col = spec.get("score_col")
            if col and col in raw:
                # Calibrated PD may change ranks across segments: verify the RAW output.
                levels = cfg.get("analysis_levels", {})
                raw_spec = levels.get("final_models", {}).get(name, {}) if levels.get("enabled") else {}
                verify_col = spec.get("verification_score_col") or raw_spec.get("raw_score_col") or col
                if verify_col not in raw:
                    raise ValueError(f"{name}: нет колонки сверки {verify_col}")
                stored = pd.to_numeric(raw.loc[ix, verify_col], errors="raise").to_numpy(dtype=float)
                direction = raw_spec.get("raw_higher_is_risk", True) if raw_spec and verify_col == raw_spec.get("raw_score_col") else spec.get("higher_is_risk", True)
                if not direction:
                    stored = -stored
                verify_prediction = model.predict(model_x, prediction_type="RawFormulaVal", thread_count=cfg.get("thread_count", 4))
                valid = np.isfinite(stored)
                # Model artifacts must reproduce the stored score before explaining trees.
                if valid.sum() < 2:
                    raise ValueError(f"{name}: нет скоров для сверки с моделью")
                if np.ptp(stored[valid]) == 0 and np.ptp(verify_prediction[valid]) == 0:
                    rho = 1.0
                else:
                    rho = float(spearmanr(stored[valid], verify_prediction[valid]).statistic)
                checks.append(dict(model=name, verification_column=verify_col, n=int(valid.sum()), spearman_stored_vs_model=rho))
                if not np.isfinite(rho) or rho < spec.get("min_score_spearman", 0.99999):
                    raise ValueError(f"{name}: сохранённые скоры не воспроизводятся моделью (rho={rho:.6g}). Проверьте версию модели, feature_map и preprocessing. Для анализа только скоров: --scores-only")
            else:
                if cfg.get("analysis_levels", {}).get("enabled"):
                    raise ValueError(f"{name}: нужен сохранённый PD {col}; predict_proba не заменяет калиброванный PD")
                col = col or "pd_" + name
                raw[col] = np.nan
                raw.loc[ix, col] = pred
                spec["score_col"], spec["higher_is_risk"] = col, True
        elif not scores_only and path is None:
            # Score-only for an individual model is allowed; paired SHAP requires both.
            continue
    return models, pd.DataFrame(checks)


def diagnostic_sample(df, cfg):
    rng = np.random.default_rng(cfg.get("seed", 42))
    indices = []
    for window in ("reference", "recent"):
        part = df.loc[df["_window"] == window]
        cap = cfg.get("diagnostic_sample_per_window", 5000)
        # Uniform sample independent of target/predictions. Report sample sizes explicitly.
        indices.extend(rng.choice(part.index.to_numpy(), min(len(part), cap), replace=False))
    return df.loc[sorted(indices)].reset_index(drop=True)


def exact_group_gini_shapley(y, baseline, changes):
    """Exact Shapley decomposition of Gini(baseline + sum(changes)) - Gini(baseline).

    Hybrid raw scores explain a ranking difference; they are not retrained models.
    """
    names = list(changes)
    k = len(names)
    if k > 8:
        raise ValueError("Exact group Shapley ограничен 8 группами")
    values = {}
    for mask in range(1 << k):
        score = np.array(baseline, dtype=float, copy=True)
        for j, name in enumerate(names):
            if mask & (1 << j):
                score += changes[name]
        values[mask] = gini(y, score)
    result = {}
    for j, name in enumerate(names):
        result[name] = sum(
            math.factorial(mask.bit_count()) * math.factorial(k - mask.bit_count() - 1) / math.factorial(k)
            * (values[mask | (1 << j)] - values[mask])
            for mask in values if not mask & (1 << j))
    return result, values[(1 << k) - 1] - values[0]


def shap_diagnostics(models, sample, cfg):
    from catboost import Pool
    grouped, raw_scores, group_rows, feature_rows, ablation_rows = {}, {}, [], [], []
    ref = sample.loc[sample["_window"] == "reference"]
    background = ref.sample(min(len(ref), cfg.get("shap_background_size", 64)), random_state=cfg.get("seed", 42))
    for name, model in models.items():
        spec = cfg["models"][name]
        x = feature_frame(model, sample, spec)
        pool = Pool(x, label=sample[cfg["target"]], cat_features=model.get_cat_feature_indices())
        ref_pool = Pool(feature_frame(model, background, spec), label=background[cfg["target"]],
                        cat_features=model.get_cat_feature_indices())
        shap = np.asarray(model.get_feature_importance(
            pool, type="ShapValues", reference_data=ref_pool, thread_count=cfg.get("thread_count", 4)))
        raw = np.asarray(model.predict(pool, prediction_type="RawFormulaVal", thread_count=cfg.get("thread_count", 4)))
        if shap.ndim != 2 or not np.allclose(shap.sum(axis=1), raw, rtol=1e-5, atol=1e-6):
            raise ValueError(f"{name}: SHAP не воспроизводит RawFormulaVal")
        raw_scores[name] = raw
        grouped[name] = {g: np.zeros(len(sample)) for g in GROUPS}
        for j, feature in enumerate(x.columns):
            # Prefer native semantics; source mapping may differ across models.
            group = feature_group(feature)
            grouped[name][group] += shap[:, j]
            for window in ("reference", "recent"):
                mask = (sample["_window"] == window).to_numpy()
                target_values = sample.loc[mask, cfg["target"]]
                gini_full = gini(target_values, raw[mask])
                gini_without = gini(target_values, raw[mask] - shap[mask, j])
                feature_rows.append(dict(model=name, feature=feature,
                                         source_column=spec.get("feature_map", {}).get(feature, feature),
                                         group=group, window=window, n=int(mask.sum()),
                                         mean_abs_shap=float(np.abs(shap[mask, j]).mean()),
                                         mean_shap=float(shap[mask, j].mean()),
                                         gini_full=gini_full,
                                         gini_without_shap_feature=gini_without,
                                         improvement_without_shap_feature=gini_without - gini_full))
        for window in ("reference", "recent"):
            mask = (sample["_window"] == window).to_numpy()
            y = sample.loc[mask, cfg["target"]]
            for group in GROUPS:
                values = grouped[name][group][mask]
                group_rows.append(dict(model=name, group=group, window=window, n=int(mask.sum()),
                                       mean_abs_group_shap=float(np.abs(values).mean()),
                                       mean_group_shap=float(values.mean())))
                full = gini(y, raw[mask])
                without = gini(y, raw[mask] - values)
                ablation_rows.append(dict(model=name, group=group, window=window,
                                          n=int(mask.sum()), gini_full=full,
                                          gini_without_shap_group=without,
                                          improvement_without_group=without - full))
    allocations = []
    details = sample[[cfg["id_col"], "_period", "_window", cfg["target"]]].copy()
    if all(name in models for name in NAMES):
        changes = {g: grouped["dub_23"][g] - grouped["dub_22"][g] for g in GROUPS}
        # Expected SHAP values differ only by a constant, irrelevant to Gini.
        hybrid = raw_scores["dub_22"] + sum(changes.values())
        if not np.allclose(hybrid - raw_scores["dub_23"],
                           (hybrid - raw_scores["dub_23"])[0], atol=1e-5):
            raise ValueError("Группы SHAP не восстанавливают разность raw scores")
        for group, values in changes.items():
            details["delta_raw_shap_" + group] = values
        for window, part in scopes(sample):
            ix = part.index.to_numpy()
            result, gap = exact_group_gini_shapley(part[cfg["target"]], raw_scores["dub_22"][ix],
                                                  {g: v[ix] for g, v in changes.items()})
            if np.isfinite(gap) and not np.isclose(sum(result.values()), gap, atol=1e-9):
                raise AssertionError("Нарушена аддитивность Gini Shapley")
            for group, value in result.items():
                allocations.append(dict(scope=window, group=group, n=len(part),
                                        gini_delta_allocation=value, total_gini_gap=gap))
    return {"shap_features": pd.DataFrame(feature_rows), "shap_groups": pd.DataFrame(group_rows),
            "shap_group_removal": pd.DataFrame(ablation_rows),
            "gini_group_allocation": pd.DataFrame(allocations), "shap_changes_by_application": details}


def parse_splits(export, model):
    """Read symmetric JSON. CTRs resolve by binary split_index, never by threshold alone."""
    if "oblivious_trees" not in export:
        raise ValueError("Локальный разбор сплитов поддерживает только SymmetricTree")
    info, names = export["features_info"], model.feature_names_
    floats = {f["feature_index"]: f for f in info.get("float_features", [])}
    cats = {f["feature_index"]: f for f in info.get("categorical_features", [])}
    binary = {}
    offset = sum(len(f.get("borders", [])) for f in floats.values())
    offset += sum(len(f.get("values", [])) for f in cats.values())
    for ctr in info.get("ctrs", []):
        sources = []
        for el in ctr["elements"]:
            if "cat_feature_index" in el:
                sources.append(names[cats[el["cat_feature_index"]]["flat_feature_index"]])
            elif "float_feature_index" in el:
                sources.append(names[floats[el["float_feature_index"]]["flat_feature_index"]])
        for border in ctr.get("borders", []):
            binary[offset] = (sorted(set(sources)), ctr, border)
            offset += 1
    rows = []
    for tree_idx, tree in enumerate(export["oblivious_trees"]):
        for depth, split in enumerate(tree.get("splits") or []):
            kind = split["split_type"]
            if kind == "FloatFeature":
                features = [names[floats[split["float_feature_index"]]["flat_feature_index"]]]
            elif kind == "OneHotFeature":
                features = [names[cats[split["cat_feature_index"]]["flat_feature_index"]]]
            elif kind == "OnlineCtr":
                if split["split_index"] not in binary:
                    raise ValueError("Не удалось однозначно разрешить CTR split_index")
                features, ctr, border = binary[split["split_index"]]
                if not np.isclose(border, split["border"]):
                    raise ValueError("CTR border не совпал с binary split_index")
            else:
                raise ValueError(f"Неизвестный тип split {kind}")
            rows.append(dict(tree=tree_idx, depth=depth, split_type=kind,
                             features="|".join(features), groups="|".join(sorted({feature_group(f) for f in features})),
                             border=split.get("border", np.nan),
                             split_json=json.dumps(split, ensure_ascii=False)))
    return pd.DataFrame(rows, columns=["tree", "depth", "split_type", "features", "groups", "border", "split_json"])


def tree_diagnostics(model, name, sample, cfg):
    from catboost import Pool
    x = feature_frame(model, sample, cfg["models"][name])
    pool = Pool(x, cat_features=model.get_cat_feature_indices())
    y = sample[cfg["target"]].to_numpy(dtype=int)
    ref, recent = ((sample["_window"] == w).to_numpy() for w in ("reference", "recent"))
    full = np.asarray(model.predict(pool, prediction_type="RawFormulaVal", thread_count=cfg.get("thread_count", 4)))
    step = cfg.get("tree_step", 25)
    stages = model.staged_predict(pool, prediction_type="RawFormulaVal", eval_period=step,
                                  thread_count=cfg.get("thread_count", 4))
    previous = np.zeros(len(sample))
    prev_end, block_rows = 0, []
    for i, raw in enumerate(stages):
        end = min((i + 1) * step, model.tree_count_)
        raw = np.asarray(raw)
        block = raw - previous
        row = dict(model=name, tree_start=prev_end, tree_end_exclusive=end,
                   n_reference=int(ref.sum()), n_recent=int(recent.sum()))
        for window, mask in (("reference", ref), ("recent", recent)):
            value = gini(y[mask], raw[mask])
            row["gini_" + window] = value
            row["gini_increment_" + window] = value - gini(y[mask], previous[mask])
            row["improvement_without_block_" + window] = gini(y[mask], full[mask] - block[mask]) - gini(y[mask], full[mask])
        row["increment_shift_recent_minus_reference"] = row["gini_increment_recent"] - row["gini_increment_reference"]
        block_rows.append(row)
        previous, prev_end = raw, end
    blocks = pd.DataFrame(block_rows)
    if not np.allclose(previous, full, atol=1e-7):
        raise AssertionError("staged_predict не дошёл до полной модели")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            model.save_model(str(path), format="json")
            export = json.loads(path.read_text())
        splits = parse_splits(export, model)
    except ValueError as exc:
        # Preserve general diagnostics for asymmetric/text models; record the limitation.
        return blocks, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), str(exc)
    splits.insert(0, "model", name)
    # Analyze blocks whose removal helps recent ranking most; limit local reports.
    selected = []
    ordered = blocks.sort_values(["improvement_without_block_recent", "increment_shift_recent_minus_reference"], ascending=[False, True])
    for block in ordered.itertuples():
        selected.extend(range(block.tree_start, block.tree_end_exclusive))
    selected = list(dict.fromkeys(selected))[:cfg.get("max_local_trees", 30)]
    branches, leaves = [], []
    # Calculate one tree at a time: avoid materializing N_objects x N_trees.
    for tree_idx in selected:
        leaf = np.asarray(model.calc_leaf_indexes(pool, ntree_start=tree_idx, ntree_end=tree_idx + 1,
                                                  thread_count=cfg.get("thread_count", 4)))[:, 0].astype(int)
        tree = export["oblivious_trees"][tree_idx]
        for leaf_id in np.unique(leaf):
            row = dict(model=name, tree=tree_idx, leaf=int(leaf_id),
                       raw_leaf_value=float(tree["leaf_values"][leaf_id]))
            for window, mask in (("reference", ref), ("recent", recent)):
                at = mask & (leaf == leaf_id)
                row["n_" + window] = int(at.sum())
                row["defaults_" + window] = int(y[at].sum())
                row["bad_rate_" + window] = float(y[at].mean()) if at.any() else np.nan
                row["share_" + window] = float(at.sum() / mask.sum())
            leaves.append(row)
        for split in splits.loc[splits["tree"] == tree_idx].to_dict("records"):
            depth = split["depth"]
            prefix = leaf & ((1 << depth) - 1)
            decision = (leaf >> depth) & 1
            for node in np.unique(prefix):
                for branch in (0, 1):
                    row = dict(split, node_prefix=int(node), branch=branch)
                    for window, mask in (("reference", ref), ("recent", recent)):
                        in_node = mask & (prefix == node)
                        at = in_node & (decision == branch)
                        row["n_" + window] = int(at.sum())
                        row["defaults_" + window] = int(y[at].sum())
                        row["bad_rate_" + window] = float(y[at].mean()) if at.any() else np.nan
                        row["share_within_node_" + window] = float(at.sum() / in_node.sum()) if in_node.any() else np.nan
                        row["share_global_" + window] = float(at.sum() / mask.sum())
                    row["bad_rate_shift"] = row["bad_rate_recent"] - row["bad_rate_reference"]
                    row["branch_share_shift"] = row["share_within_node_recent"] - row["share_within_node_reference"]
                    row["reliable_size"] = all(row["defaults_" + w] >= cfg.get("min_defaults", 30)
                                                and row["n_" + w] - row["defaults_" + w] >= cfg.get("min_goods", 30)
                                                for w in ("reference", "recent"))
                    branches.append(row)
    return blocks, splits, pd.DataFrame(branches), pd.DataFrame(leaves), None


def model_schema(models, cfg):
    rows = []
    for name, model in models.items():
        spec = cfg["models"][name]
        for i, feature in enumerate(model.feature_names_):
            rows.append(dict(model=name, feature=feature,
                             source_column=spec.get("feature_map", {}).get(feature, feature),
                             group=feature_group(feature), categorical=i in model.get_cat_feature_indices()))
    return pd.DataFrame(rows)


def component_swaps(models, sample, cfg):
    """Swap only final-model score inputs; never claims to inspect component trees."""
    if not all(n in models for n in NAMES):
        return pd.DataFrame()
    rows = []
    for name, model in models.items():
        other = next(n for n in NAMES if n != name)
        x = feature_frame(model, sample, cfg["models"][name])
        by_group = {}
        for m in (name, other):
            by_group[m] = {}
            for f in models[m].feature_names_:
                group = feature_group(f)
                if group in {"early_model", "long_model", "soc_dem"}:
                    by_group[m].setdefault(group, []).append(f)
        available = [g for g in by_group[name] if len(by_group[name][g]) == 1 and len(by_group[other].get(g, [])) == 1]
        combinations = [(g,) for g in available]
        if len(available) > 1:
            combinations.append(tuple(available))
        base = model.predict(x, prediction_type="RawFormulaVal", thread_count=cfg.get("thread_count", 4))
        for groups in combinations:
            hybrid = x.copy()
            for group in groups:
                src_native = by_group[other][group][0]
                src_col = cfg["models"][other].get("feature_map", {}).get(src_native, src_native)
                hybrid[by_group[name][group][0]] = sample[src_col].to_numpy()
            prediction = model.predict(hybrid, prediction_type="RawFormulaVal", thread_count=cfg.get("thread_count", 4))
            for scope, part in scopes(sample):
                ix = part.index.to_numpy()
                y = part[cfg["target"]]
                before, after = gini(y, np.asarray(base)[ix]), gini(y, np.asarray(prediction)[ix])
                rows.append(dict(model=name, donor=other, swapped_groups="|".join(groups),
                                 scope=scope, n=len(part), gini_original=before,
                                 gini_with_other_component=after, improvement=after - before))
    return pd.DataFrame(rows)


def write_csv(frame, output, name):
    if len(frame.columns):
        frame.to_csv(output / (name + ".csv"), index=False, encoding="utf-8-sig")


def plot_overview(tables, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    metrics = tables["metrics"]
    monthly = metrics.loc[~metrics["scope"].isin(["reference", "recent", "all"])]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    x = np.arange(len(monthly))
    for name in NAMES:
        axes[0, 0].plot(x, monthly["gini_" + name] * 100, marker="o", label=name)
    axes[0, 0].set_xticks(x, monthly["scope"], rotation=45)
    axes[0, 0].set_title("Monthly Gini: target 30@4")
    axes[0, 0].legend()
    axes[0, 1].bar(x - 0.18, monthly["defaults_better"], width=0.36, label="better")
    axes[0, 1].bar(x + 0.18, monthly["defaults_worse"], width=0.36, label="worse")
    axes[0, 1].set_xticks(x, monthly["scope"], rotation=45)
    axes[0, 1].set_title("Defaults: position relative to non-defaults")
    axes[0, 1].legend()
    allocation = tables.get("gini_group_allocation", pd.DataFrame())
    if len(allocation):
        pivot = allocation.loc[allocation["scope"].isin(["reference", "recent"])].pivot(index="group", columns="scope", values="gini_delta_allocation") * 100
        pivot.plot.bar(ax=axes[1, 0])
        axes[1, 0].set_title("Gini gap allocation (diagnostic sample; descriptive)")
    drift = tables.get("drift", pd.DataFrame())
    if len(drift):
        drift.head(12).sort_values("psi").plot.barh(x="feature", y="psi", ax=axes[1, 1], legend=False)
        axes[1, 1].set_title("Distribution drift: reference -> recent (PSI)")
    fig.savefig(output / "overview.png", dpi=150)
    plt.close(fig)


def build_summary(tables, cfg, warnings):
    metrics = tables["metrics"].set_index("scope")
    recent, ref = metrics.loc["recent"], metrics.loc["reference"]
    change = tables["window_gap_change"].iloc[0]
    lines = ["# Сравнение dub_22 / dub_23", "",
             f"Таргет: `{cfg['target']}`. Выборка: одинаковые заявки, известный таргет, два конечных скора.",
             f"Reference: {', '.join(cfg['reference_periods'])}. Recent: {', '.join(cfg['recent_periods'])}.", "",
             f"Recent: n={int(recent['n'])}, дефолтов={int(recent['defaults'])}.",
             f"Gini dub_22={recent['gini_dub_22']:.4f}; dub_23={recent['gini_dub_23']:.4f}.",
             f"ΔGini (23−22)={recent['delta_gini_23_minus_22']:.4f}; paired bootstrap 95% CI [{recent['ci_low']:.4f}, {recent['ci_high']:.4f}].",
             f"Reference ΔGini={ref['delta_gini_23_minus_22']:.4f}; изменение разрыва={change['gap_change_recent_minus_reference']:.4f}; 95% CI [{change['ci_low']:.4f}, {change['ci_high']:.4f}].", "",
             f"В объединённом recent улучшилось дефолтов: {int(recent['defaults_better'])}; ухудшилось: {int(recent['defaults_worse'])}; без изменения: {int(recent['defaults_unchanged'])}.",
             "Это сравнение относительно недефолтов той же выборки. Помесячные числа в metrics.csv и reranked_defaults.csv используют недефолты своего месяца; их сумма может отличаться от объединённого recent.", "",
             "## Как читать результаты", "",
             "- metrics.csv — Gini, разрыв, доверительный интервал, объём и устойчивость среза.",
             "- top_risk.csv — дополнительно пойманные/потерянные дефолты при одинаковой доле отклонения.",
             "- metrics.csv / reranked_defaults.csv — также точные количества улучшенных/ухудшенных bad–good пар, включая ties; это отдельные числа, а не только чистый баланс.",
             "- component_gini.csv — собственное качество скоров early/long/soc_dem по времени.",
             "- component_swaps.csv — изменение качества при подстановке скоров компонент из другой версии; требует совместимой шкалы.",
             "- gini_group_allocation.csv — описательное разложение разрыва Gini на группы через гибриды raw SHAP. Сумма вкладов равна разрыву на диагностической подвыборке.",
             "- tree_blocks.csv — на каких итерациях меняется Gini и помогает ли удаление блока из готовой модели.",
             "- tree_splits.csv / tree_branches.csv / tree_leaves.csv — условия сплитов, переходы по веткам и листьям, изменение долей и bad rate.",
             "- drift.csv / segments.csv / data_quality.csv — сдвиг распределений, сегменты и фильтрация выборки.", "",
             "## Ограничения интерпретации", "",
             "Негативный вклад группы показывает связь с разницей ранжирования, но не доказывает причину. SHAP и удаление блока не заменяют переобучение.",
             "Сравнение независимых деревьев по одинаковому номеру не имеет смысла. Роутинг читается отдельно для каждой модели.",
             "Добавление FPS, исключение soc_dem, изменения компонентных скоров, обучения и состава выдач могут смешиваться. Для проверки FPS/soc_dem используйте ablation.py на одних train/validation/OOT и одинаковых скоровых входах.",
             "Gini объединённого окна содержит межмесячные пары; основная проверка временного падения — помесячные срезы. Нет данных по отказанным — нет вывода обоём потоке заявок.",
             "Bootstrap оценивает неопределённость на фиксированных моделях. Множественные поисковые проверки сплитов/сегментов не являются подтверждающими тестами."]
    if "levels_metrics" in tables:
        lines += ["", "## Уровни подмоделей, скоров/PD и признаков", "",
                  "- levels_metrics.csv: парные сравнения dub_23−dub_22 отдельно на raw score и на PD для итоговых моделей и подмоделей, CI и точные переходы пар.",
                  "- levels_pair_transitions.csv / levels_defaults.csv: все 9 типов переходов пар и конкретные переранжированные дефолты.",
                  "- calibration_metrics.csv / calibration_curves.csv: изменение Gini при score→PD, монотонность, Brier/LogLoss, reliability и оценка calibration intercept/slope.",
                  "- calibration_gap_decomposition.csv: разрыв Gini на raw, разрыв на PD и точная разница этих разрывов с CI на одной выборке.",
                  "- submodel_regression_panel.csv / submodel_regression_coefficients.csv / submodel_regression_diagnostics.csv: регрессия разрыва Gini через качество подмоделей; HAC CI, проверка коллинеарности и chronological holdout.",
                  "- Уровень признаков: SHAP, группы, сплиты и замены входов описывают raw выход сохранённых CatBoost. Изменение ранжирования внешней калибровкой требует отдельного анализа calibrator.",
                  "Строго возрастающая общая калибровка сохраняет ранжирование; ties и разные калибровки по времени/сегментам могут его изменить. Монотонность калибровки входа подмодели не гарантирует инвариантность уже обученного итогового бустинга при подстановке другой шкалы."]
    if warnings:
        lines += ["", "## Замечания"] + ["- " + w for w in warnings]
    return "\n".join(lines) + "\n"


def run(cfg, scores_only=False):
    validate_config(cfg)
    out = Path(cfg["output"])
    out.mkdir(parents=True, exist_ok=True)
    raw = read_table(cfg["data"])
    if any(str(c).startswith("_risk_") or c in {"_period", "_window"} for c in raw.columns):
        raise ValueError("Входные данные содержат зарезервированные служебные колонки")
    # Fail duplicate joins before expensive model inference.
    if raw[cfg["id_col"]].isna().any() or raw[cfg["id_col"]].duplicated().any():
        raise ValueError("App_id должен быть уникальным и непустым")
    models, checks = load_models_and_scores(raw, cfg, scores_only)
    df, audit = prepare_frame(raw, cfg)
    tables = {"data_quality": audit, "score_reproduction": checks,
              "metrics": metric_table(df, cfg), "reranked_defaults": reranked_defaults(df, cfg),
              "window_gap_change": window_gap_change(df, cfg),
              "component_gini": component_metrics(df, cfg), "segments": segment_table(df, cfg),
              "score_correlations": score_correlations(df, cfg)}
    tables["top_risk"], tables["top_risk_migrations"] = top_risk_diagnostics(df, cfg)
    warnings = []
    from multilevel import plot_levels, run_levels
    level_tables, level_notes = run_levels(df, cfg)
    tables.update(level_tables)
    warnings.extend(level_notes)
    absent = set(cfg.get("components", {})) - set(df.columns)
    if absent:
        warnings.append("Нет компонентных колонок: " + ", ".join(sorted(absent)))
    if not tables["metrics"].set_index("scope").loc["recent", "reliable_size"]:
        warnings.append("В recent мало дефолтов/недефолтов; выводы нестабильны.")
    schema = model_schema(models, cfg)
    tables["model_features"] = schema
    drift_cols = list(cfg.get("components", {})) + ["_risk_" + n for n in NAMES]
    if cfg.get("analysis_levels", {}).get("enabled"):
        levels = cfg["analysis_levels"]
        specifications = list(levels.get("final_models", {}).values())
        specifications += [spec for versions in levels.get("submodels", {}).values() for spec in versions.values()]
        drift_cols += [spec[key] for spec in specifications for key in ("raw_score_col", "pd_col", "uncalibrated_pd_col") if spec.get(key)]
    drift_cols += [c for c in df.columns if c.lower().startswith("fps")]
    if len(schema):
        drift_cols += schema["source_column"].tolist()
    tables["drift"] = drift_table(df, drift_cols)
    for target in cfg.get("secondary_targets", []):
        if target not in raw:
            warnings.append(f"Нет дополнительного таргета {target}")
            continue
        secondary_cfg = dict(cfg, target=target, allow_missing_window_periods=True)
        try:
            secondary, secondary_audit = prepare_frame(raw, secondary_cfg)
        except ValueError as exc:
            # Secondary maturity can lag the primary target; do not invent recent labels.
            if "Нет парных вызревших наблюдений" in str(exc) or "Нет вызревших объектов" in str(exc):
                warnings.append(f"{target}: {exc}. Укажите отдельные окна для этого таргета в отдельном запуске.")
                continue
            raise
        tables["secondary_" + target] = metric_table(secondary, secondary_cfg)
        tables["secondary_quality_" + target] = secondary_audit
        missing_months = (set(cfg["reference_periods"]) | set(cfg["recent_periods"])) - set(secondary["_period"])
        if missing_months:
            warnings.append(f"{target}: нет вызревших парных объектов за {sorted(missing_months)}; соответствующие метрики не оцениваются.")
    if models:
        sample = diagnostic_sample(df, cfg)
        tables["diagnostic_sample_metrics"] = metric_table(sample, dict(cfg, bootstrap_iterations=0), bootstrap=False)
        tables.update(shap_diagnostics(models, sample, cfg))
        for key in ("shap_features", "shap_groups", "shap_group_removal", "gini_group_allocation"):
            tables[key]["explained_output"] = "CatBoost RawFormulaVal"
            tables[key]["external_calibrator_included"] = False
        tables["component_swaps"] = component_swaps(models, sample, cfg)
        block_frames, split_frames, branch_frames, leaf_frames = [], [], [], []
        for name, model in models.items():
            block, split, branch, leaf, warning = tree_diagnostics(model, name, sample, cfg)
            block_frames.append(block)
            split_frames.append(split)
            branch_frames.append(branch)
            leaf_frames.append(leaf)
            if warning:
                warnings.append(f"{name}: {warning}")
        for key, frames in (("tree_blocks", block_frames), ("tree_splits", split_frames),
                            ("tree_branches", branch_frames), ("tree_leaves", leaf_frames)):
            tables[key] = pd.concat(frames, ignore_index=True)
        splits = tables["tree_splits"]
        if len(splits):
            tables["split_profile"] = splits.groupby(["model", "groups", "depth"], as_index=False).agg(
                split_count=("tree", "size"), unique_trees=("tree", "nunique"))
        parameters = {name: model.get_all_params() for name, model in models.items()}
        (out / "model_parameters.json").write_text(json.dumps(parameters, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        warnings.append("Запущен анализ скоров: для SHAP и сплитов нужны файлы моделей.")
    for key, value in tables.items():
        write_csv(value, out, key)
    plot_overview(tables, out)
    if level_tables:
        plot_levels(tables, out)
    (out / "summary.md").write_text(build_summary(tables, cfg, warnings), encoding="utf-8")
    (out / "resolved_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    import catboost
    import sklearn
    manifest = dict(pandas=pd.__version__, numpy=np.__version__, catboost=catboost.__version__,
                    sklearn=sklearn.__version__, input_rows=len(raw), evaluated_rows=len(df),
                    model_sha256={name: hashlib.sha256(Path(cfg["models"][name]["path"]).read_bytes()).hexdigest() for name in models},
                    warnings=warnings)
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Готово: {out.resolve() / 'summary.md'}")
    return tables


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--scores-only", action="store_true")
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text(encoding="utf-8")), args.scores_only)
