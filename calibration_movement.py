"""One-call diagnostics of final dub_23 calibration on arbitrary user-defined groups.

Public entry point: plot_dub23_calibration(data, group_col=..., reference_value=...,
comparison_value=...). No project configuration or dub_22 inputs are required.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from multilevel import calibration_mapping, metric_ci
from ranking import WeightedGini, pair_transitions, transition_statistics

__all__ = ["plot_dub23_calibration"]

_SEGMENTS = ("sum_request_less_500", "sum_request_500_1000", "sum_request_more_1000k")
_ALL = "__all__"
_COLORS = ("#0072B2", "#E69F00", "#009E73")


def _prepare(data, cfg, options):
    columns = [cfg["id_col"], cfg["target"], options["group_col"], options["raw_score_col"], options["pd_col"]]
    if cfg.get("cluster_col"):
        columns.append(cfg["cluster_col"])
    if options.get("date_col"):
        columns.append(options["date_col"])
    columns += [options["segment_col"]] if options["segment_col"] else options["segment_flag_cols"]
    missing = set(columns) - set(data)
    if missing:
        raise ValueError(f"Отсутствуют колонки: {sorted(missing)}")
    df = data[list(dict.fromkeys(columns))].copy()
    if df[cfg["id_col"]].isna().any() or df[cfg["id_col"]].duplicated().any():
        raise ValueError("id_col должен содержать уникальный непустой ID заявки")
    target = pd.to_numeric(df[cfg["target"]], errors="raise")
    if not target.dropna().isin([0, 1]).all():
        raise ValueError("target_col: ожидается 0/1/NA; NA не заменяется нулём")
    raw = pd.to_numeric(df[options["raw_score_col"]], errors="raise").to_numpy(dtype=float)
    pd_score = pd.to_numeric(df[options["pd_col"]], errors="raise").to_numpy(dtype=float)
    eligible = np.ones(len(df), bool)
    if options["max_issue_date"] is not None:
        if not options["date_col"]:
            raise ValueError("Для max_issue_date нужно указать date_col")
        dates = pd.to_datetime(df[options["date_col"]], errors="raise")
        if dates.isna().any():
            raise ValueError("date_col содержит NA")
        eligible = (dates.dt.normalize() <= pd.Timestamp(options["max_issue_date"])).to_numpy()
    observed = target.notna().to_numpy()
    finite = np.isfinite(raw) & np.isfinite(pd_score)
    has_group = df[options["group_col"]].notna().to_numpy()
    keep = eligible & observed & finite & has_group
    audit = pd.DataFrame({
        "group": df[options["group_col"]].to_numpy(), "n_input": 1,
        "n_within_cutoff": eligible.astype(int),
        "n_observed_within_cutoff": (eligible & observed).astype(int),
        "n_missing_raw_pd": (eligible & observed & ~finite).astype(int),
        "n_missing_group": (eligible & observed & finite & ~has_group).astype(int),
        "n_used": keep.astype(int),
    }).groupby("group", dropna=False, observed=True, sort=False, as_index=False).sum()
    df = df.loc[keep].reset_index(drop=True)
    if df.empty:
        raise ValueError("Нет наблюдаемых таргетов с конечными raw/PD и непустой группой")
    if ((pd_score[keep] < 0) | (pd_score[keep] > 1)).any():
        raise ValueError("Калиброванный PD должен находиться в [0, 1]")
    if options["segment_col"]:
        values = df[options["segment_col"]]
        if values.isna().any() or not values.isin(options["segment_order"]).all():
            raise ValueError("В segment_col есть NA или неизвестный сегмент; проверьте segment_order")
        segments = values.astype(str).to_numpy()
    else:
        flags = df[options["segment_flag_cols"]].apply(pd.to_numeric, errors="raise")
        if flags.isna().any().any() or not flags.isin([0, 1]).all().all():
            raise ValueError("Флаги сегментов должны содержать только 0/1 без NA")
        if not flags.sum(axis=1).eq(1).all():
            raise ValueError("У каждой используемой заявки должен быть ровно один активный флаг сегмента")
        segments = np.asarray(options["segment_order"])[np.argmax(flags.to_numpy(), axis=1)]
    df[cfg["target"]] = target.loc[keep].to_numpy(dtype=int)
    df["_cal_group"] = df[options["group_col"]].astype(object)
    df["_cal_segment"] = segments
    df["_cal_raw"] = raw[keep] * (1 if options["higher_score_higher_risk"] else -1)
    df["_cal_pd"] = pd_score[keep]
    grouped = df.groupby("_cal_group", sort=False, observed=True)
    n = grouped[cfg["id_col"]].transform("size")
    for source, output in (("_cal_raw", "_cal_raw_rank"), ("_cal_pd", "_cal_pd_rank")):
        df[output] = 100 * (grouped[source].rank(method="average") - .5) / n
    df["_cal_rank_shift"] = df._cal_pd_rank - df._cal_raw_rank
    groups = pd.unique(df._cal_group).tolist()
    for value in (options["reference_value"], options["comparison_value"]):
        if value not in groups:
            raise ValueError(f"После фильтрации нет группы {value!r}; доступны {groups}")
    if options["group_order"] is None:
        original = data[options["group_col"]]
        if isinstance(original.dtype, pd.CategoricalDtype) and original.dtype.ordered:
            order = [v for v in original.cat.categories if v in groups]
        else:
            order = groups
    else:
        order = list(options["group_order"])
        if len(set(order)) != len(order) or set(groups) - set(order):
            raise ValueError("group_order должен содержать все используемые группы без повторов")
    options["group_order"] = order
    return df, audit


def _summaries(data, cfg, options):
    movement, ginis = [], []
    for group, whole in data.groupby("_cal_group", sort=False, observed=True):
        for segment, part in whole.groupby("_cal_segment", sort=False):
            shift, bad = part._cal_rank_shift, part[cfg["target"]].eq(1)
            movement.append(dict(group=group, segment=segment, n=len(part), defaults=int(bad.sum()),
                                 median_shift=float(shift.median()), mean_shift=float(shift.mean()),
                                 shift_q25=float(shift.quantile(.25)), shift_q75=float(shift.quantile(.75)),
                                 share_moved_up=float((shift > 1e-10).mean()),
                                 share_moved_down=float((shift < -1e-10).mean()),
                                 default_median_shift=float(shift[bad].median()) if bad.any() else np.nan))
        for segment, part in [(_ALL, whole)] + list(whole.groupby("_cal_segment", sort=False)):
            y = part[cfg["target"]].to_numpy(dtype=int)
            rank_raw, rank_pd = WeightedGini(y, part._cal_raw), WeightedGini(y, part._cal_pd)

            def evaluate(weights):
                a, b = rank_raw(weights), rank_pd(weights)
                return dict(gini_raw=a, gini_pd=b, delta_gini=b - a)

            ginis.append(dict(group=group, segment=segment, n=len(part), defaults=int(y.sum()),
                              goods=int((1 - y).sum()),
                              reliable_size=int(y.sum()) >= cfg["min_defaults"] and int((1 - y).sum()) >= cfg["min_goods"],
                              **metric_ci(part, cfg, evaluate),
                              **calibration_mapping(part._cal_raw.to_numpy(), part._cal_pd.to_numpy())))
    return pd.DataFrame(movement), pd.DataFrame(ginis)


def _segment_pairs(data, cfg, options):
    """Exact 3x3 decomposition with one GLOBAL bad*good denominator per group."""
    order, rows = options["segment_order"], []
    for group, part in data.groupby("_cal_group", sort=False, observed=True):
        y = part[cfg["target"]].to_numpy(dtype=int)
        denominator = int(y.sum()) * int((1 - y).sum())
        bad_segments = part.loc[y == 1, "_cal_segment"].to_numpy()
        for good_segment in order:
            good_mask = (y == 0) & part._cal_segment.eq(good_segment).to_numpy()
            selected = part.loc[(y == 1) | good_mask]
            matrix = pair_transitions(selected[cfg["target"]], selected._cal_raw, selected._cal_pd)
            for bad_segment in order:
                stats = transition_statistics(matrix[bad_segments == bad_segment])
                rows.append(dict(group=group, bad_segment=bad_segment, good_segment=good_segment,
                                 defaults_in_bad_segment=int((bad_segments == bad_segment).sum()),
                                 goods_in_good_segment=int(good_mask.sum()), global_pair_denominator=denominator,
                                 gini_delta_contribution=2 * stats["pair_net_units"] / denominator if denominator else np.nan,
                                 **stats))
        if denominator:
            expected = WeightedGini(y, part._cal_pd)() - WeightedGini(y, part._cal_raw)()
            actual = sum(c["gini_delta_contribution"] for c in rows[-9:])
            if not np.isclose(actual, expected, atol=1e-10):
                raise AssertionError("Матрица сегментов не восстанавливает полную разницу Gini")
    return pd.DataFrame(rows)


def _map_bins(data, options):
    rows = []
    for segment in options["segment_order"]:
        reference_raw = data.loc[data._cal_group.eq(options["reference_value"]) & data._cal_segment.eq(segment), "_cal_raw"]
        if reference_raw.empty:
            continue
        cuts = np.unique(np.quantile(reference_raw, np.linspace(0, 1, options["raw_map_bins"] + 1)[1:-1]))
        for group, part in data.loc[data._cal_segment == segment].groupby("_cal_group", sort=False, observed=True):
            indices = np.searchsorted(cuts, part._cal_raw.to_numpy(), side="right")
            for bucket in np.unique(indices):
                selected = part.loc[indices == bucket]
                rows.append(dict(group=group, segment=segment, bin=int(bucket), n=len(selected),
                                 raw_median=float(selected._cal_raw.median()), pd_median=float(selected._cal_pd.median()),
                                 pd_min=float(selected._cal_pd.min()), pd_max=float(selected._cal_pd.max())))
    return pd.DataFrame(rows, columns=None if rows else ["group", "segment", "raw_median", "pd_median"])


def _styles(options):
    return {s: (_COLORS[i], options["segment_labels"].get(s, s)) for i, s in enumerate(options["segment_order"])}


def _draw_scatter(ax, part, cfg, options, x, y):
    good, bad = part.loc[part[cfg["target"]].eq(0)], part.loc[part[cfg["target"]].eq(1)]
    if len(good) > options["scatter_max_goods"]:
        good = good.sample(n=options["scatter_max_goods"], random_state=cfg["seed"])
    for segment, (color, label) in _styles(options).items():
        a, b = good.loc[good._cal_segment == segment], bad.loc[bad._cal_segment == segment]
        ax.scatter(a[x], a[y], s=13, color=color, alpha=.22, linewidths=0, rasterized=True)
        ax.scatter(b[x], b[y], s=55, color=color, marker="X", edgecolors="black", linewidths=.35, zorder=4)
    ax.grid(alpha=.15)


def _legend(fig, options):
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", color=color, linestyle="", label=label) for color, label in _styles(options).values()]
    handles.append(Line2D([], [], marker="X", color="black", linestyle="", label="Дефолт: target=1"))
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9)


def _plot_migration(data, cfg, options):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    for ax, value, title in zip(axes, (options["reference_value"], options["comparison_value"]), ("Референс", "Сравнение")):
        part = data.loc[data._cal_group.eq(value)]
        _draw_scatter(ax, part, cfg, options, "_cal_raw_rank", "_cal_pd_rank")
        ax.plot([0, 100], [0, 100], "--", color="gray", linewidth=1)
        ax.set(xlim=(-2, 102), ylim=(-2, 102), xlabel="Процентиль риска по raw, %",
               title=f"{title}: {value}\nn={len(part):,}, дефолтов={int(part[cfg['target']].sum())}")
        ax.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("Процентиль риска по PD, %")
    fig.suptitle(f"dub_23: перемещение после калибровки · {options['group_col']}", fontsize=14)
    fig.text(.5, .108, "Выше диагонали — место по риску повысилось; ниже — понизилось. 100% = наибольший риск.", ha="center", fontsize=9)
    fig.text(.5, .08, "Ранги — по всем выдачам внутри выбранной группы, сразу по трём сегментам. Прореживаются только недефолты на рисунке.", ha="center", fontsize=8)
    _legend(fig, options)
    fig.tight_layout(rect=(0, .15, 1, .94))
    return fig


def _plot_timeline(summary, gini_summary, options):
    import matplotlib.pyplot as plt
    order = options["group_order"]
    x = np.arange(len(order))
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for segment, (color, label) in _styles(options).items():
        part = summary.loc[summary.segment == segment].set_index("group").reindex(order)
        axes[0].plot(x, part.median_shift, marker="o", color=color, label=label)
        axes[0].fill_between(x, part.shift_q25, part.shift_q75, color=color, alpha=.13)
    axes[0].set(ylabel="Сдвиг ранга, п.п.", title="Какие сегменты смещаются при выбранном разбиении")
    axes[0].legend(fontsize=9)
    part = gini_summary.loc[gini_summary.segment == _ALL].set_index("group").reindex(order)
    axes[1].plot(x, part.delta_gini * 100, color="#333333", marker="o", label="Gini(PD) − Gini(raw)")
    axes[1].fill_between(x, part.delta_gini_ci_low * 100, part.delta_gini_ci_high * 100,
                         color="#0072B2", alpha=.16, label="95% paired bootstrap CI")
    axes[1].set(ylabel="Изменение Gini, п.п.", title="Изменение ранжирования после калибровки")
    axes[1].legend(fontsize=9)
    # Tick labels contain all counts for modest-sized groupings; large plots keep a readable subset.
    ticks = np.unique(np.linspace(0, len(order) - 1, min(len(order), 10), dtype=int))
    labels = []
    for i in ticks:
        row = part.iloc[i]
        labels.append(f"{order[i]}\nn={int(row.n)}, bad={int(row.defaults)}" if pd.notna(row.n) else f"{order[i]}\nнет данных")
    axes[1].set_xticks(ticks, labels, rotation=35, ha="right", fontsize=9)
    axes[1].set_xlabel(options["group_col"])
    for ax in axes:
        ax.axhline(0, color="gray", linewidth=1)
        ax.grid(alpha=.15)
    fig.suptitle("dub_23: сдвиг рейтинга и потеря/прирост Gini", fontsize=14)
    fig.text(.5, .012, "Верхняя полоса: 25–75% сдвигов клиентов (не CI). Нижняя: 95% CI разницы Gini. Группы дискретные, без интерполяции новых данных.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .035, 1, .96))
    return fig


def _plot_maps(data, bins, cfg, options):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharex=True, sharey=True)
    selected_values = [options["reference_value"], options["comparison_value"]]
    raw = data.loc[data._cal_group.isin(selected_values), "_cal_raw"]
    lo, hi = raw.min(), raw.max()
    padding = max((hi - lo) * .03, .01)
    for ax, value in zip(axes, selected_values):
        part = data.loc[data._cal_group.eq(value)]
        _draw_scatter(ax, part, cfg, options, "_cal_raw", "_cal_pd")
        for segment, (color, label) in _styles(options).items():
            selected = bins.loc[bins.group.eq(value) & bins.segment.eq(segment)].sort_values("raw_median")
            ax.plot(selected.raw_median, selected.pd_median, color=color, linewidth=1.8)
        ax.set(xlim=(lo - padding, hi + padding), ylim=(-.02, 1.02),
               xlabel="Raw, ориентированный по риску", title=str(value))
    axes[0].set_ylabel("Калиброванный PD")
    fig.suptitle(f"dub_23: наблюдаемое преобразование raw → PD · {options['group_col']}", fontsize=14)
    fig.text(.5, .09, "Линии — медианы по фиксированным raw-бинам reference; это не восстановленная формула calibrator.", ha="center", fontsize=9)
    _legend(fig, options)
    fig.tight_layout(rect=(0, .14, 1, .94))
    return fig


def _plot_pairs(pairs, options):
    import matplotlib.pyplot as plt
    values = [options["reference_value"], options["comparison_value"]]
    chosen = pairs.loc[pairs.group.isin(values)]
    finite = chosen.gini_delta_contribution.dropna().abs()
    limit = max(float(finite.max()) * 100 if len(finite) else 0, .01)
    order = options["segment_order"]
    labels = options["segment_labels"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    cmap = plt.get_cmap("RdBu").with_extremes(bad="#eeeeee")
    for ax, group in zip(axes, values):
        table = pairs.loc[pairs.group.eq(group)].set_index(["bad_segment", "good_segment"])
        z = np.array([[table.loc[(b, g), "gini_delta_contribution"] * 100 for g in order] for b in order])
        n = np.array([[table.loc[(b, g), "pairs_bad_good_total"] for g in order] for b in order])
        image = ax.imshow(np.ma.masked_where((n == 0) | ~np.isfinite(z), z), cmap=cmap, vmin=-limit, vmax=limit)
        for i, bad_segment in enumerate(order):
            for j, good_segment in enumerate(order):
                row = table.loc[(bad_segment, good_segment)]
                text = (f"{z[i, j]:+.2f} п.п.\n↑{int(row.pairs_improved):,}  ↓{int(row.pairs_worsened):,}"
                        if n[i, j] and np.isfinite(z[i, j]) else "нет пар")
                color = "white" if np.isfinite(z[i, j]) and abs(z[i, j]) > .55 * limit else "black"
                ax.text(j, i, text, ha="center", va="center", fontsize=9, color=color)
        ax.set_xticks(range(3), [labels.get(s, s) for s in order], rotation=20, ha="right", fontsize=8)
        ax.set_yticks(range(3), [labels.get(s, s) for s in order], fontsize=8)
        ax.set(xlabel="Сегмент недефолтного клиента", title=str(group))
    axes[0].set_ylabel("Сегмент дефолтного клиента")
    fig.subplots_adjust(left=.17, right=.88, bottom=.25, top=.88, wspace=.25)
    color_axis = fig.add_axes([.91, .3, .014, .48])
    fig.colorbar(image, cax=color_axis, label="Вклад в ΔGini, п.п.")
    fig.suptitle("dub_23: какие пары изменились при калибровке", fontsize=14)
    fig.text(.5, .025, "Красное — ухудшение, синее — улучшение. ↑/↓ — улучшенные/ухудшенные пары, включая ties. Сумма 9 вкладов = полная ΔGini.", ha="center", fontsize=9)
    return fig


def plot_dub23_calibration(
    data,
    *,
    group_col,
    reference_value,
    comparison_value,
    raw_score_col="score_dub_23",
    pd_col="pd_dub_23",
    target_col="flag_4m_30p_gross",
    id_col="App_id",
    segment_col=None,
    segment_flag_cols=None,
    segment_order=None,
    segment_labels=None,
    group_order=None,
    higher_score_higher_risk=True,
    date_col=None,
    max_issue_date=None,
    cluster_col=None,
    bootstrap_iterations=200,
    min_defaults=30,
    min_goods=30,
    scatter_max_goods=3000,
    raw_map_bins=15,
    random_state=42,
    output_dir=None,
):
    """Return four figures and diagnostic tables using ONLY dub_23 inputs.

    Required
    --------
    data : pandas.DataFrame
        One unique id_col per issuance; raw score, calibrated PD in [0,1],
        target_col=0/1/NA (1=default), group_col, and calibration segment(s).
    group_col : str
        Existing column defining arbitrary cohorts: quarter, month, version,
        campaign, etc. Values are used AS PROVIDED; nothing is converted to months.
    reference_value, comparison_value : scalar
        Two distinct values of group_col, of the same types used in data. Both
        must exist after maturity/completeness filtering.

    Segments
    --------
    Either segment_col contains categorical segment names, or segment_flag_cols
    lists three mutually exclusive 0/1 columns in segment_order order. If neither
    is specified, the three sum_request_* columns are used. No amount thresholds
    are inferred. Set segment_order explicitly for different categorical names.

    Ordering and filtering
    ----------------------
    group_order optionally specifies ALL plotted group values in the desired
    order. Otherwise ordered categoricals retain category order, other values
    retain first appearance. Extra listed groups are displayed as missing data.
    Only known 0/1 targets, finite raw/PD and nonmissing groups are used. Optional
    max_issue_date requires date_col and is inclusive. Ranks are calculated on
    ALL eligible clients within each group, across ALL segments, before sampling.

    Returns
    -------
    dict with 'figures' (rank_migration, rank_shift_timeline, segment_pairs,
    raw_pd_maps), 'tables', 'group_order', 'reference_value', 'comparison_value',
    'messages', 'saved_files'. Figures can be displayed with IPython.display.display.
    No files are written unless output_dir is supplied. Bads are never sampled;
    scatter_max_goods limits only displayed goods, not calculations. IQR bands
    describe clients, while Gini CIs use paired bootstrap (cluster_col if given).
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data должен быть pandas.DataFrame")
    if not pd.api.types.is_scalar(reference_value) or not pd.api.types.is_scalar(comparison_value):
        raise ValueError("reference_value и comparison_value должны быть отдельными значениями")
    if pd.isna(reference_value) or pd.isna(comparison_value) or reference_value == comparison_value:
        raise ValueError("Нужны два разных непустых значения group_col")
    if segment_col and segment_flag_cols is not None:
        raise ValueError("Укажите segment_col ИЛИ segment_flag_cols")
    flags = list(segment_flag_cols if segment_flag_cols is not None else _SEGMENTS)
    order = list(segment_order if segment_order is not None else (_SEGMENTS if segment_col else flags))
    if len(order) != 3 or len(set(order)) != 3 or _ALL in order or not all(isinstance(s, str) for s in order):
        raise ValueError("segment_order: нужны три разных строковых имени; __all__ зарезервирован")
    if not segment_col and (len(flags) != 3 or len(set(flags)) != 3):
        raise ValueError("segment_flag_cols: нужны три разных флага")
    for name, value, minimum in (("bootstrap_iterations", bootstrap_iterations, 0),
                                 ("scatter_max_goods", scatter_max_goods, 0), ("raw_map_bins", raw_map_bins, 2),
                                 ("min_defaults", min_defaults, 1), ("min_goods", min_goods, 1)):
        if not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} должен быть целым числом >= {minimum}")
    cfg = dict(id_col=id_col, target=target_col, cluster_col=cluster_col, seed=random_state,
               bootstrap_iterations=bootstrap_iterations, min_defaults=min_defaults, min_goods=min_goods)
    options = dict(group_col=group_col, reference_value=reference_value, comparison_value=comparison_value,
                   raw_score_col=raw_score_col, pd_col=pd_col, segment_col=segment_col,
                   segment_flag_cols=flags, segment_order=order, segment_labels=segment_labels or {},
                   group_order=None if group_order is None else list(group_order),
                   higher_score_higher_risk=higher_score_higher_risk, date_col=date_col,
                   max_issue_date=max_issue_date, scatter_max_goods=scatter_max_goods, raw_map_bins=raw_map_bins)
    points, audit = _prepare(data, cfg, options)
    summary, ginis = _summaries(points, cfg, options)
    pairs, bins = _segment_pairs(points, cfg, options), _map_bins(points, options)
    tables = dict(rank_movements=points, cohort_audit=audit, rank_summary=summary,
                  gini_by_segment=ginis, segment_pairs=pairs, raw_pd_bins=bins)
    # Close figure managers to avoid duplicate automatic output in Jupyter. The
    # returned Figure objects remain displayable and can still be saved as PNG.
    import matplotlib.pyplot as plt
    figures = {}
    try:
        figures["rank_migration"] = _plot_migration(points, cfg, options)
        figures["rank_shift_timeline"] = _plot_timeline(summary, ginis, options)
        figures["segment_pairs"] = _plot_pairs(pairs, options)
        figures["raw_pd_maps"] = _plot_maps(points, bins, cfg, options)
    finally:
        for fig in figures.values():
            plt.close(fig)
    saved = {}
    if output_dir is not None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        for name, fig in figures.items():
            target = path / (name + ".png")
            fig.savefig(target, dpi=150, bbox_inches="tight")
            saved[name + ".png"] = str(target.resolve())
        for name, table in tables.items():
            target = path / (name + ".csv")
            table.to_csv(target, index=False, encoding="utf-8-sig")
            saved[name + ".csv"] = str(target.resolve())
    messages = []
    if group_order is None:
        messages.append("Порядок групп — порядок ordered categorical либо первого появления. Для временной динамики задайте group_order явно.")
    for value in (reference_value, comparison_value):
        row = ginis.loc[ginis.group.eq(value) & ginis.segment.eq(_ALL)].iloc[0]
        if not row.reliable_size:
            messages.append(f"Группа {value!r}: мало bad/good (bad={int(row.defaults)}, good={int(row.goods)}); Gini и CI могут быть нестабильны.")
    return dict(figures=figures, tables=tables, group_order=options["group_order"],
                reference_value=reference_value, comparison_value=comparison_value,
                messages=messages, saved_files=saved)
