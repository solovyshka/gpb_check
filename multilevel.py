"""Submodel, score/PD, and calibration diagnostics on explicitly paired cohorts.

These reports measure associations in fixed predictions. No calibrator is fitted
or applied to production, and no component is replaced by a differently scaled
raw score inside a trained CatBoost model.
"""
from __future__ import annotations

import hashlib
import warnings as pywarnings

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import norm, spearmanr

from ranking import (STATES, WeightedGini, pair_transitions,
                     per_default_pair_statistics, transition_statistics)


def bootstrap_weights(frame, cfg, stratify=True):
    rng = np.random.default_rng(cfg.get("seed", 42))
    y = frame[cfg["target"]].to_numpy(dtype=int)
    cluster = cfg.get("cluster_col")
    if cluster:
        codes, unique = pd.factorize(frame[cluster], sort=False)
        if (codes < 0).any():
            raise ValueError("cluster_col содержит NA")
    for _ in range(cfg.get("bootstrap_iterations", 500)):
        if cluster:
            yield np.bincount(rng.integers(len(unique), size=len(unique)), minlength=len(unique))[codes]
        elif stratify:
            weight = np.zeros(len(frame), dtype=int)
            for label in (0, 1):
                ix = np.flatnonzero(y == label)
                if len(ix):
                    weight += np.bincount(rng.choice(ix, size=len(ix), replace=True), minlength=len(frame))
            yield weight
        else:
            yield np.bincount(rng.integers(len(frame), size=len(frame)), minlength=len(frame))


def metric_ci(frame, cfg, evaluate, stratify=True):
    result = dict(evaluate(None))
    draws = {key: [] for key in result}
    if not len(frame):
        return result
    for weight in bootstrap_weights(frame, cfg, stratify):
        for key, value in evaluate(weight).items():
            if np.isfinite(value):
                draws[key].append(value)
    for key, values in draws.items():
        lo, hi = np.quantile(values, [0.025, 0.975]) if values else (np.nan, np.nan)
        result[key + "_ci_low"], result[key + "_ci_high"] = lo, hi
        result[key + "_bootstrap_valid"] = len(values)
    return result


def all_scopes(frame):
    for month, part in frame.groupby("_period", sort=True):
        yield str(month), part
    for window in ("reference", "recent"):
        yield window, frame.loc[frame["_window"] == window]
    yield "all", frame


def numeric(frame, column):
    return pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)


def raw_risk(frame, spec):
    return numeric(frame, spec["raw_score_col"]) * (1 if spec.get("raw_higher_is_risk", True) else -1)


def paired_cohort(frame, specs, entity, audit):
    columns = []
    for spec in specs:
        columns += [spec["raw_score_col"], spec["pd_col"]]
        if spec.get("uncalibrated_pd_col"):
            columns.append(spec["uncalibrated_pd_col"])
    columns = list(dict.fromkeys(columns))
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{entity}: отсутствуют настроенные колонки {sorted(missing)}")
    valid = np.ones(len(frame), bool)
    for col in columns:
        values = numeric(frame, col)
        valid &= np.isfinite(values)
    for month, ix in frame.groupby("_period").indices.items():
        audit.append(dict(entity=entity, period=month, n_input=len(ix),
                          n_complete=int(valid[ix].sum()), n_missing=int((~valid[ix]).sum())))
    result = frame.loc[valid].copy().reset_index(drop=True)
    for spec in specs:
        probability(result, spec["pd_col"])
        if spec.get("uncalibrated_pd_col"):
            probability(result, spec["uncalibrated_pd_col"])
    return result


def probability(frame, column):
    values = numeric(frame, column)
    if ((values < 0) | (values > 1) | ~np.isfinite(values)).any():
        raise ValueError(f"{column}: PD должен быть конечным числом в [0,1]")
    return values


def before_calibration_probability(frame, spec):
    if spec.get("uncalibrated_pd_col"):
        return probability(frame, spec["uncalibrated_pd_col"])
    kind = spec.get("raw_kind", "score")
    if kind == "probability":
        values = probability(frame, spec["raw_score_col"])
        return values if spec.get("raw_higher_is_risk", True) else 1 - values
    if kind == "margin":
        # RawFormulaVal for binary Logloss models only; do not sigmoid arbitrary scores.
        return expit(raw_risk(frame, spec))
    if kind != "score":
        raise ValueError("raw_kind: ожидается score, probability или margin")
    return None


def compare_rankings(frame, a, b, cfg, metadata, detail=True):
    """Compare two fixed score arrays on one cohort; preserve exact pair counts."""
    from diagnose import default_quality
    y = frame[cfg["target"]].to_numpy(dtype=int)
    qa, qb = default_quality(y, a), default_quality(y, b)
    diff = qb - qa
    matrix = pair_transitions(y, a, b)
    rank_a, rank_b = WeightedGini(y, a), WeightedGini(y, b)

    def evaluate(weight):
        aa, bb = rank_a(weight), rank_b(weight)
        return dict(gini_before=aa, gini_after=bb, delta_gini=bb - aa)

    row = dict(metadata, n=len(frame), defaults=int(y.sum()), goods=int((1 - y).sum()),
               defaults_better=int((diff > 1e-12).sum()), defaults_worse=int((diff < -1e-12).sum()),
               defaults_unchanged=int((np.abs(diff) <= 1e-12).sum()),
               defaults_not_comparable=int((~np.isfinite(diff)).sum()),
               **transition_statistics(matrix), **metric_ci(frame, cfg, evaluate))
    if np.isfinite(row["delta_gini"]) and not np.isclose(row["delta_gini"], row["gini_delta_from_pairs"], atol=1e-10):
        raise AssertionError("Таблица переходов пар не восстанавливает delta Gini")
    row["reliable_size"] = row["defaults"] >= cfg.get("min_defaults", 30) and row["goods"] >= cfg.get("min_goods", 30)
    joint = matrix.sum(axis=0)
    transitions = [dict(metadata, before_state=old, after_state=new, pairs=int(joint[i, j]))
                   for i, old in enumerate(STATES) for j, new in enumerate(STATES)]
    details = pd.DataFrame()
    if detail:
        columns = [cfg["id_col"], "_period", "_window"]
        details = frame.loc[y == 1, columns].copy()
        details["quality_before"], details["quality_after"] = qa, qb
        details["quality_delta"] = diff
        details["status"] = np.select([diff > 1e-12, diff < -1e-12, np.abs(diff) <= 1e-12],
                                       ["better", "worse", "unchanged"], default="not_comparable")
        for key, value in per_default_pair_statistics(matrix).items():
            details[key] = value
        for key, value in metadata.items():
            details[key] = value
    return row, transitions, details


def calibration_mapping(a, p):
    """Test monotonicity in risk direction across all objects, not just bad-good pairs."""
    if not len(a):
        return {}
    order = np.argsort(a, kind="stable")
    aa, pp = a[order], p[order]
    starts = np.flatnonzero(np.r_[True, aa[1:] != aa[:-1]])
    lo, hi = np.minimum.reduceat(pp, starts), np.maximum.reduceat(pp, starts)
    previous_max = np.r_[-np.inf, np.maximum.accumulate(hi)[:-1]]
    within_ties = int((lo != hi).sum())
    violations = int((lo < previous_max).sum())
    nondecreasing = violations == 0 and within_ties == 0
    strict = nondecreasing and bool((lo[1:] > hi[:-1]).all())
    rho = float(spearmanr(a, p).statistic) if len(np.unique(a)) > 1 and len(np.unique(p)) > 1 else np.nan
    return dict(raw_unique=len(starts), pd_unique=len(np.unique(p)),
                raw_equal_groups_with_different_pd=within_ties,
                raw_groups_with_order_violation=violations,
                single_nondecreasing_mapping=nondecreasing,
                single_strictly_increasing_mapping=strict, spearman_raw_pd=rho)


def reliability_rows(y, p, edges, metadata, clustered=False):
    bins = np.minimum(np.searchsorted(edges, p, side="right") - 1, len(edges) - 2)
    rows = []
    z = norm.ppf(0.975)
    for bucket in range(len(edges) - 1):
        mask = bins == bucket
        n = int(mask.sum())
        if not n:
            continue
        rate = float(y[mask].mean())
        center = (rate + z * z / (2 * n)) / (1 + z * z / n)
        half = z * np.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        rows.append(dict(metadata, bin=bucket, pd_left=edges[bucket], pd_right=edges[bucket + 1],
                         n=n, defaults=int(y[mask].sum()), mean_pd=float(p[mask].mean()),
                         observed_rate=rate, rate_ci_low=np.nan if clustered else center - half,
                         rate_ci_high=np.nan if clustered else center + half,
                         rate_ci_method="not_computed_clustered" if clustered else "Wilson_independent_rows"))
    return rows


def calibration_parameters(frame, p, cfg):
    """Diagnostic GLM on evaluation labels, not a new deployed calibration."""
    import statsmodels.api as sm
    from statsmodels.tools.sm_exceptions import PerfectSeparationWarning
    y = frame[cfg["target"]].to_numpy(dtype=int)
    if len(np.unique(y)) < 2 or len(np.unique(p)) < 2:
        return dict(calibration_fit_status="one_class_or_constant_probability")
    x = sm.add_constant(logit(np.clip(p, 1e-12, 1 - 1e-12)))
    try:
        with pywarnings.catch_warnings():
            pywarnings.simplefilter("error", PerfectSeparationWarning)
            model = sm.GLM(y, x, family=sm.families.Binomial())
            if cfg.get("cluster_col"):
                fit = model.fit(cov_type="cluster", cov_kwds={"groups": frame[cfg["cluster_col"]]})
            else:
                fit = model.fit(cov_type="HC0")
        if not fit.converged or not np.isfinite(fit.params).all():
            return dict(calibration_fit_status="not_converged")
        ci = np.asarray(fit.conf_int())
        return dict(calibration_fit_status="ok", calibration_intercept=float(fit.params[0]),
                    calibration_slope=float(fit.params[1]),
                    intercept_ci_low=ci[0, 0], intercept_ci_high=ci[0, 1],
                    slope_ci_low=ci[1, 0], slope_ci_high=ci[1, 1])
    except (np.linalg.LinAlgError, ValueError, PerfectSeparationWarning, ZeroDivisionError) as exc:
        return dict(calibration_fit_status=type(exc).__name__)


def probability_metrics(frame, p, baseline, cfg, metadata, edges):
    y = frame[cfg["target"]].to_numpy(dtype=int)
    probabilities = {"pd": p}
    if baseline is not None:
        probabilities["uncalibrated_pd"] = baseline

    def evaluate(weight):
        result = {}
        if not len(y) or (weight is not None and np.sum(weight) == 0):
            return result
        result["observed_rate"] = np.average(y, weights=weight)
        for name, values in probabilities.items():
            clipped = np.clip(values, 1e-12, 1 - 1e-12)
            result["brier_" + name] = np.average((y - values) ** 2, weights=weight)
            result["logloss_" + name] = np.average(-y * np.log(clipped) - (1 - y) * np.log1p(-clipped), weights=weight)
            result["mean_" + name] = np.average(values, weights=weight)
        if baseline is not None:
            result["brier_change_pd_minus_uncalibrated"] = result["brier_pd"] - result["brier_uncalibrated_pd"]
            result["logloss_change_pd_minus_uncalibrated"] = result["logloss_pd"] - result["logloss_uncalibrated_pd"]
        return result

    row = dict(metadata, **metric_ci(frame, cfg, evaluate, stratify=False))
    curves = []
    for name, values in probabilities.items():
        bins = reliability_rows(y, values, edges, dict(metadata, probability_type=name), clustered=bool(cfg.get("cluster_col")))
        curves.extend(bins)
        row["ece_" + name] = sum(v["n"] * abs(v["mean_pd"] - v["observed_rate"]) for v in bins) / len(y) if len(y) else np.nan
    row["observed_expected_ratio"] = y.mean() / p.mean() if len(y) and p.mean() > 0 else np.nan
    row.update(calibration_parameters(frame, p, cfg))
    return row, curves


def calibration_gap_decomposition(frame, specs, cfg, scope):
    y = frame[cfg["target"]].to_numpy(dtype=int)
    rankers = {}
    for name, spec in specs.items():
        rankers[name + "_raw"] = WeightedGini(y, raw_risk(frame, spec))
        rankers[name + "_pd"] = WeightedGini(y, probability(frame, spec["pd_col"]))

    def evaluate(weight):
        g = {key: value(weight) for key, value in rankers.items()}
        raw = g["dub_23_raw"] - g["dub_22_raw"]
        pd_gap = g["dub_23_pd"] - g["dub_22_pd"]
        c22, c23 = (g[name + "_pd"] - g[name + "_raw"] for name in ("dub_22", "dub_23"))
        return dict(raw_gini_gap=raw, pd_gini_gap=pd_gap, calibration_gini_change_22=c22,
                    calibration_gini_change_23=c23, gap_change_due_to_calibration=pd_gap - raw)

    return dict(scope=scope, n=len(frame), **metric_ci(frame, cfg, evaluate))


def regression_quality_panel(frame, levels, cfg):
    """Measure outcome/submodel quality per non-overlapping period.

    By default predictor and response quality use disjoint, stable customer halves.
    This reduces shared-label measurement noise; it does not imply causality.
    """
    regression = levels.get("regression", {})
    freq = regression.get("frequency", "M")
    # W denotes disjoint calendar weeks; rolling/overlapping windows are not supported.
    if freq not in {"M", "W", "Q"}:
        raise ValueError("regression.frequency: доступны M, W, Q без перекрывающихся окон")
    frame = frame.copy()
    frame["_reg_period"] = pd.to_datetime(frame[cfg["date_col"]]).dt.to_period(freq)
    split_key = cfg.get("cluster_col") or cfg["id_col"]
    if frame[split_key].isna().any():
        raise ValueError("Пустой ID для разделения регрессионных измерений")
    seed = cfg.get("seed", 42)
    # Target-independent assignment; the same customer stays in one half over time.
    frame["_half"] = [hashlib.blake2b((str(seed) + ":" + str(v)).encode(), digest_size=8).digest()[0] % 2 for v in frame[split_key]]
    crossfit = regression.get("split_measurement", True)
    min_bad, min_good = regression.get("min_defaults", 30), regression.get("min_goods", 30)
    all_specs = list(levels["final_models"].values())
    all_specs += [spec for versions in levels.get("submodels", {}).values() for spec in versions.values()]
    columns = {s[c] for s in all_specs for c in ("raw_score_col", "pd_col")}
    missing = columns - set(frame)
    if missing:
        raise ValueError(f"Нет колонок регрессионного panel: {sorted(missing)}")
    finite = np.ones(len(frame), bool)
    for col in columns:
        finite &= np.isfinite(numeric(frame, col))
    frame = frame.loc[finite].copy()
    rows = []
    for period, part in frame.groupby("_reg_period", sort=True):
        predictor = part.loc[part._half == 0] if crossfit else part
        outcome = part.loc[part._half == 1] if crossfit else part
        yp, yo = (d[cfg["target"]].to_numpy(dtype=int) for d in (predictor, outcome))
        size_ok = all((yy == 1).sum() >= min_bad and (yy == 0).sum() >= min_good for yy in (yp, yo))
        for representation in ("raw", "pd"):
            row = dict(period=str(period), representation=representation, n_predictor=len(predictor),
                       n_outcome=len(outcome), defaults_predictor=int(yp.sum()), defaults_outcome=int(yo.sum()),
                       split_measurement=crossfit, reliable_size=size_ok)

            def score(data, spec):
                return raw_risk(data, spec) if representation == "raw" else probability(data, spec["pd_col"])

            final = {m: WeightedGini(yo, score(outcome, spec))() for m, spec in levels["final_models"].items()}
            row["response_delta_gini"] = final["dub_23"] - final["dub_22"]
            for component, versions in levels.get("submodels", {}).items():
                measured = {m: WeightedGini(yp, score(predictor, spec))() for m, spec in versions.items()}
                for m, value in measured.items():
                    row[f"gini__{component}__{m}"] = value
                if set(measured) == {"dub_22", "dub_23"}:
                    row["delta_gini__" + component] = measured["dub_23"] - measured["dub_22"]
            rows.append(row)
    return pd.DataFrame(rows, columns=None if rows else ["period", "representation", "reliable_size", "response_delta_gini"])


def fit_quality_regression(panel, levels):
    """Equal-period OLS with HAC inference; no leakage across chronological holdout."""
    import statsmodels.api as sm
    config = levels.get("regression", {})
    predictors = config.get("predictors", ["delta_gini__early_model", "delta_gini__long_model", "gini__soc_dem__dub_22"])
    if not predictors or len(set(predictors)) != len(predictors):
        raise ValueError("regression.predictors должны быть непустыми и уникальными")
    coefs, diagnostics, fits = [], [], []
    min_periods = max(config.get("min_periods", 12), len(predictors) + 8)
    lags = config.get("hac_lags", 1)
    if lags < 0:
        raise ValueError("hac_lags должен быть >= 0")
    if not 0 < config.get("holdout_fraction", 0.25) < 1:
        raise ValueError("holdout_fraction должен лежать между 0 и 1")
    for representation in ("raw", "pd"):
        data = panel.loc[panel.representation == representation].copy() if len(panel) else pd.DataFrame()
        missing = set(predictors) - set(data)
        base = dict(representation=representation, required_periods=min_periods, predictor_count=len(predictors),
                    covariance="HAC", hac_lags=lags)
        if missing:
            diagnostics.append(dict(base, status="missing_predictors", detail="|".join(sorted(missing)), n_periods=0))
            continue
        data = data.loc[data.reliable_size].dropna(subset=predictors + ["response_delta_gini"]).sort_values("period")
        if len(data) < min_periods:
            diagnostics.append(dict(base, status="insufficient_periods", n_periods=len(data),
                                    detail="Нужны неперекрывающиеся периоды с достаточным числом bad/good в обеих половинах"))
            continue
        periods = pd.PeriodIndex(data.period, freq=config.get("frequency", "M")).asi8
        if not (np.diff(periods) == 1).all():
            diagnostics.append(dict(base, status="non_contiguous_periods", n_periods=len(data),
                                    detail="HAC требует последовательные периоды; проверьте пропуски/размеры срезов"))
            continue
        x = data[predictors].astype(float)
        if (x.std(ddof=0) < 1e-10).any():
            diagnostics.append(dict(base, status="constant_predictor", n_periods=len(data),
                                    detail="|".join(x.columns[x.std(ddof=0) < 1e-10])))
            continue
        # Standardize for numerical conditioning and report both original/standardized coefficients.
        center, scale = x.mean(), x.std(ddof=0)
        design = sm.add_constant((x - center) / scale, has_constant="add")
        condition = float(np.linalg.cond(design))
        if np.linalg.matrix_rank(design) < design.shape[1] or condition > config.get("max_condition_number", 1000):
            diagnostics.append(dict(base, status="collinear_predictors", n_periods=len(data), condition_number=condition))
            continue
        response = data.response_delta_gini
        fit = sm.OLS(response, design).fit(cov_type="HAC", cov_kwds={"maxlags": min(lags, len(data) - 1), "use_correction": True}, use_t=True)
        ci = fit.conf_int()
        for key in predictors:
            coefs.append(dict(representation=representation, term=key,
                              coefficient=float(fit.params[key] / scale[key]),
                              ci_low=float(ci.loc[key, 0] / scale[key]), ci_high=float(ci.loc[key, 1] / scale[key]),
                              coefficient_per_sd=float(fit.params[key]), pvalue=float(fit.pvalues[key]),
                              predictor_mean=float(center[key]), predictor_sd=float(scale[key]), n_periods=len(data)))
        intercept = float(fit.params["const"] - np.sum(fit.params[predictors] * center / scale))
        # Transform the covariance as well for an intercept CI on the original scale.
        transform = np.r_[1.0, -(center / scale).to_numpy()]
        intercept_se = float(np.sqrt(max(0, transform @ np.asarray(fit.cov_params()) @ transform)))
        from scipy.stats import t
        critical = t.ppf(0.975, fit.df_resid)
        coefs.append(dict(representation=representation, term="intercept", coefficient=intercept,
                          ci_low=intercept - critical * intercept_se, ci_high=intercept + critical * intercept_se,
                          n_periods=len(data)))
        diagnostic = dict(base, status="ok", n_periods=len(data), r_squared=float(fit.rsquared),
                          adjusted_r_squared=float(fit.rsquared_adj), condition_number=condition,
                          intercept=intercept, split_measurement=bool(data.split_measurement.all()))
        # Reserve the latest periods for a chronological generalization check when possible.
        holdout = max(3, int(np.ceil(len(data) * config.get("holdout_fraction", 0.25))))
        train_n = len(data) - holdout
        predictions = np.full(len(data), np.nan)
        if train_n >= max(len(predictors) + 5, 8):
            early_x = sm.add_constant(x.iloc[:train_n], has_constant="add")
            if np.linalg.matrix_rank(early_x) == early_x.shape[1]:
                holdout_fit = sm.OLS(response.iloc[:train_n], early_x).fit()
                late_x = sm.add_constant(x.iloc[train_n:], has_constant="add")
                pred = np.asarray(holdout_fit.predict(late_x))
                predictions[train_n:] = pred
                actual = response.iloc[train_n:].to_numpy()
                baseline = np.repeat(response.iloc[:train_n].mean(), holdout)
                diagnostic.update(holdout_periods=holdout, holdout_rmse=float(np.sqrt(np.mean((actual - pred) ** 2))),
                                  holdout_mean_baseline_rmse=float(np.sqrt(np.mean((actual - baseline) ** 2))))
        diagnostics.append(diagnostic)
        for i, (_, row) in enumerate(data.iterrows()):
            fits.append(dict(representation=representation, period=row.period,
                             observed_delta_gini=row.response_delta_gini, fitted_delta_gini=float(fit.fittedvalues.iloc[i]),
                             chronological_holdout_prediction=predictions[i]))
    return (pd.DataFrame(coefs, columns=None if coefs else ["representation", "term", "coefficient", "ci_low", "ci_high"]),
            pd.DataFrame(diagnostics),
            pd.DataFrame(fits, columns=None if fits else ["representation", "period", "observed_delta_gini", "fitted_delta_gini", "chronological_holdout_prediction"]))


def run_levels(frame, cfg):
    """Return independent report tables plus explicit interpretation limitations."""
    levels = cfg.get("analysis_levels", {})
    if not levels.get("enabled", False):
        return {}, []
    if set(levels.get("final_models", {})) != {"dub_22", "dub_23"}:
        raise ValueError("analysis_levels.final_models: нужны dub_22 и dub_23")
    edges = np.asarray(levels.get("pd_bins", [0, .01, .025, .05, .075, .1, .15, .2, .3, .5, 1]), float)
    if edges[0] != 0 or edges[-1] != 1 or not (np.diff(edges) > 0).all():
        raise ValueError("pd_bins должны строго возрастать от 0 до 1")
    audit, metrics, transitions, details, calibration, curves, decomposition, top_tables = [], [], [], [], [], [], [], []
    notes = []
    entities = {"final": levels["final_models"], **levels.get("submodels", {})}
    if len(entities) != len(levels.get("submodels", {})) + 1:
        raise ValueError("Имя подмодели final зарезервировано")
    for entity, versions in entities.items():
        if not versions or set(versions) - {"dub_22", "dub_23"}:
            raise ValueError(f"{entity}: ожидаются dub_22/dub_23")
        common = paired_cohort(frame, list(versions.values()), entity, audit)
        if common.empty:
            notes.append(f"{entity}: нет полной парной выборки score/PD")
            continue
        if entity == "final":
            from diagnose import top_risk_diagnostics
            for representation in ("raw", "pd"):
                work = common.copy()
                for name, spec in versions.items():
                    work["_risk_" + name] = raw_risk(work, spec) if representation == "raw" else probability(work, spec["pd_col"])
                top, _ = top_risk_diagnostics(work, cfg)
                top["representation"] = representation
                top_tables.append(top)
        for scope, part in all_scopes(common):
            if not len(part):
                continue
            # Compare versions at both representations on the same complete rows.
            if set(versions) == {"dub_22", "dub_23"}:
                for representation in ("raw", "pd"):
                    scores = {m: raw_risk(part, s) if representation == "raw" else probability(part, s["pd_col"])
                              for m, s in versions.items()}
                    meta = dict(level="scores" if entity == "final" else "submodels", entity=entity,
                                comparison="dub_23_minus_dub_22", representation=representation, scope=scope)
                    row, pair_rows, default_rows = compare_rankings(part, scores["dub_22"], scores["dub_23"], cfg, meta,
                                                                   detail=scope not in {"all", "recent", "reference"})
                    metrics.append(row)
                    transitions.extend(pair_rows)
                    if len(default_rows):
                        details.append(default_rows)
            if entity == "final":
                decomposition.append(calibration_gap_decomposition(part, versions, cfg, scope))
            for name, spec in versions.items():
                a, p = raw_risk(part, spec), probability(part, spec["pd_col"])
                meta = dict(level="calibration", entity=entity, model=name, scope=scope,
                            comparison="pd_minus_raw", representation="raw_to_pd")
                row, pair_rows, default_rows = compare_rankings(part, a, p, cfg, meta,
                                                               detail=scope not in {"all", "recent", "reference"})
                row.update(calibration_mapping(a, p))
                row["pd_target"] = spec.get("pd_target") or "unknown"
                row["target_matches_pd_target"] = spec.get("pd_target") == cfg["target"]
                row["ranking_target"] = cfg["target"]
                own_target = spec.get("pd_target")
                # Rankability for the downstream early target and probability calibration
                # for a submodel's own target are distinct evaluations.
                probability_target = own_target if own_target in part.columns else cfg["target"]
                prob_frame = part.loc[part[probability_target].notna()]
                if not prob_frame[probability_target].isin([0, 1]).all():
                    raise ValueError(f"{probability_target}: ожидается target 0/1/NA")
                prob_meta = dict(meta, probability_target=probability_target,
                                 probability_target_matches_pd_target=own_target == probability_target)
                prob_cfg = dict(cfg, target=probability_target)
                baseline = before_calibration_probability(prob_frame, spec)
                probs, bins = probability_metrics(prob_frame, probability(prob_frame, spec["pd_col"]), baseline, prob_cfg, prob_meta, edges)
                row.update(probs)
                row["n_probability_evaluation"] = len(prob_frame)
                row["uncalibrated_probability_available"] = baseline is not None
                calibration.append(row)
                transitions.extend(pair_rows)
                curves.extend(bins)
                if len(default_rows):
                    details.append(default_rows)
                # Optional segmented calibrators: compare within those groups as well as globally.
                group_cols = spec.get("calibration_group_cols", [])
                if group_cols:
                    missing = set(group_cols) - set(part)
                    if missing:
                        raise ValueError(f"Нет групп калибровки {sorted(missing)}")
                    for group_key, group_part in part.groupby(group_cols, dropna=False, observed=True):
                        if not isinstance(group_key, tuple):
                            group_key = (group_key,)
                        group_meta = dict(meta, calibration_group="|".join(f"{k}={v}" for k, v in zip(group_cols, group_key)))
                        group_row, _, _ = compare_rankings(group_part, raw_risk(group_part, spec),
                                                          probability(group_part, spec["pd_col"]),
                                                          dict(cfg, bootstrap_iterations=0), group_meta, detail=False)
                        group_row.update(calibration_mapping(raw_risk(group_part, spec), probability(group_part, spec["pd_col"])))
                        calibration.append(group_row)
        for name, spec in versions.items():
            if spec.get("pd_target") not in frame.columns:
                notes.append(f"{entity}/{name}: PD_target={spec.get('pd_target') or 'unknown'} неизвестен/отсутствует; Brier/LogLoss относительно {cfg['target']} не доказывают плохую калибровку для собственного подтаргета.")
    tables = {"levels_metrics": pd.DataFrame(metrics), "levels_pair_transitions": pd.DataFrame(transitions),
              "levels_defaults": pd.concat(details, ignore_index=True) if details else pd.DataFrame(),
              "levels_cohort_audit": pd.DataFrame(audit), "calibration_metrics": pd.DataFrame(calibration),
              "calibration_curves": pd.DataFrame(curves), "calibration_gap_decomposition": pd.DataFrame(decomposition),
              "levels_top_risk": pd.concat(top_tables, ignore_index=True) if top_tables else pd.DataFrame()}
    quality = tables["calibration_metrics"]
    if len(quality):
        if "calibration_group" in quality:
            quality = quality.loc[quality.calibration_group.isna()]
        tables["submodel_quality"] = quality.loc[quality.entity != "final"].copy()
    if levels.get("regression", {}).get("enabled", True):
        panel = regression_quality_panel(frame, levels, cfg)
        coefs, diagnostics, fitted = fit_quality_regression(panel, levels)
        tables.update(submodel_regression_panel=panel, submodel_regression_coefficients=coefs,
                      submodel_regression_diagnostics=diagnostics, submodel_regression_fitted=fitted)
        for row in diagnostics.to_dict("records"):
            if row["status"] != "ok":
                notes.append(f"Регрессия {row['representation']}: {row['status']}; временных точек {row['n_periods']}, требуется минимум {row['required_periods']}.")
        notes.append("Регрессия качества — описательная временная связь, не причинная атрибуция; HAC не устраняет шум/ошибки измерения Gini и смешение состава выдач.")
    return tables, notes


def plot_levels(tables, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    final = tables.get("levels_metrics", pd.DataFrame())
    if final.empty:
        return
    final = final.loc[(final.entity == "final") & ~final.scope.isin(["all", "recent", "reference"])]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    for representation, part in final.groupby("representation"):
        axes[0, 0].plot(part.scope, part.delta_gini * 100,
                        marker="x" if representation == "pd" else "o",
                        linestyle="--" if representation == "pd" else "-",
                        zorder=3 if representation == "pd" else 2, label=representation)
    axes[0, 0].set_title("Final model Gini gap: 23 - 22")
    axes[0, 0].axhline(0, color="gray", linewidth=.8)
    axes[0, 0].legend()
    cal = tables["calibration_gap_decomposition"]
    cal = cal.loc[~cal.scope.isin(["all", "recent", "reference"])]
    axes[0, 1].bar(cal.scope, cal.gap_change_due_to_calibration * 100)
    axes[0, 1].plot(cal.scope, cal.gap_change_due_to_calibration * 100, "o", color="tab:blue")
    axes[0, 1].axhline(0, color="gray", linewidth=.8)
    axes[0, 1].set_title("Gini gap change due to calibration")
    curves = tables["calibration_curves"]
    curves = curves.loc[(curves.entity == "final") & (curves.scope == "recent") & (curves.probability_type == "pd")]
    for model, part in curves.groupby("model"):
        label = model + ": " + str(part.probability_target.iloc[0])
        axes[1, 0].plot(part.mean_pd, part.observed_rate, marker="o", label=label)
    axes[1, 0].plot([0, 1], [0, 1], "--", color="gray")
    axes[1, 0].set(xlabel="Mean PD", ylabel="Observed target rate", title="Reliability: recent")
    axes[1, 0].legend()
    sub = tables["levels_metrics"]
    sub = sub.loc[(sub.entity != "final") & (sub.scope == "recent")]
    if len(sub):
        sub.pivot(index="entity", columns="representation", values="delta_gini").mul(100).plot.bar(ax=axes[1, 1])
        axes[1, 1].set_title("Submodel Gini gaps: recent, 23 - 22")
    for axis in axes[0]:
        axis.tick_params(axis="x", rotation=45)
    fig.savefig(output / "levels_overview.png", dpi=150)
    plt.close(fig)
