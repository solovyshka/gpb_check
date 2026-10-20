# %% [markdown]
# # Диагностика dub_22 / dub_23 по шагам
#
# Запускайте ячейки сверху вниз через Shift+Enter. Одна ячейка — одна небольшая задача.
# Сначала проверьте весь путь на примере: `USE_DEMO = True`.
# Затем поставьте `USE_DEMO = False` и укажите свой `config.json` в блоке 0.
#
# Подмодели early/long/soc_dem **не калиброваны**: используем только их исходные скоры.
# Калибровку проверяем только у итоговых dub_22/dub_23.
# Все Gini хранятся на шкале [-1, 1]: разница 0.01 — один пункт Gini.
#
# Этот .py содержит те же ячейки, что и walkthrough.ipynb. Маркеры `# %%` понимают
# редакторы с поддержкой Python-ячеек; код можно копировать в свой Jupyter.

# %% [markdown]
# ## 0. Подключаем библиотеки и выбираем режим
#
# Файлы diagnose.py, multilevel.py и ranking.py должны находиться в PROJECT_DIR.
# Если не хватает библиотек, установите requirements-notebook.txt в среду текущего
# ядра. Не выполняйте установку повторно при каждом запуске анализа.

# %%
from pathlib import Path
from copy import deepcopy
import json
import sys

PROJECT_DIR = Path.cwd()  # Если ноутбук в другой папке, задайте путь к проекту.
USE_DEMO = True           # False — использовать свои данные и конфиг.
CONFIG_PATH = PROJECT_DIR / "config.json"
RUN_MODEL_DIAGNOSTICS = True  # False — пропустить .cbm, SHAP и деревья.

assert (PROJECT_DIR / "diagnose.py").exists(), "Укажите PROJECT_DIR — папку с diagnose.py"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# При редактировании функций в .py Jupyter будет подхватывать изменения.
from IPython import get_ipython
from IPython.display import display
shell = get_ipython()
if shell is not None:
    shell.run_line_magic("load_ext", "autoreload")
    shell.run_line_magic("autoreload", "2")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from diagnose import (
    NAMES, read_table, validate_config, prepare_frame, metric_table,
    reranked_defaults, top_risk_diagnostics, window_gap_change, component_metrics,
    score_correlations, drift_table, segment_table, gini, load_models_and_scores,
    model_schema, feature_frame, diagnostic_sample, shap_diagnostics,
    component_swaps, tree_diagnostics, write_csv,
)
from multilevel import (
    all_scopes, paired_cohort, raw_risk, probability, compare_rankings, metric_ci,
    regression_quality_panel, fit_quality_regression, calibration_gap_decomposition,
    calibration_mapping, before_calibration_probability, probability_metrics,
)
from ranking import STATES, WeightedGini, pair_transitions, transition_statistics

pd.set_option("display.max_columns", 30)
print("Python текущего ядра:", sys.executable)

# %% [markdown]
# ### 0.1. Загружаем настройки
#
# В режиме своих данных сначала скопируйте config.example.json в config.json.
# Проверьте пути, периоды, raw_score_col подмоделей и направление скоров.
# Итоговые колонки: score_dub_22, score_dub_23, pd_dub_22, pd_dub_23.
#
# Для быстрой отладки ниже задано 50 bootstrap-повторов. Для финального анализа
# увеличьте число, например, до 500. Перезапустите зависимые ячейки после изменения.

# %%
if USE_DEMO:
    from make_demo import make_demo
    cfg = make_demo(
        directory=str(PROJECT_DIR / "demo_data" / "notebook_demo"),
        n_per_month=300,
        iterations=40,
    )
else:
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

# Относительные пути в конфиге считаем от PROJECT_DIR.
for key in ("data", "output"):
    cfg[key] = str((PROJECT_DIR / cfg[key]).resolve())
for spec in cfg["models"].values():
    if spec.get("path"):
        spec["path"] = str((PROJECT_DIR / spec["path"]).resolve())

cfg["bootstrap_iterations"] = 50
cfg["diagnostic_sample_per_window"] = 500
cfg["shap_background_size"] = 16
cfg["max_local_trees"] = 5
validate_config(cfg)

levels = cfg["analysis_levels"]
tables = {}  # Сюда будем складывать результаты для сохранения в конце.
print("Данные:", cfg["data"])
print("Таргет:", cfg["target"])
print("Последний вызревший месяц:", cfg.get("max_issue_date"))
display(pd.DataFrame(levels["final_models"]).T)

# %% [markdown]
# ## 1. Чтение данных
#
# read_table читает CSV или Parquet. raw — исходная таблица: её не меняем.
# Если данные уже загружены в вашем ноутбуке, замените чтение на `raw = my_df.copy()`.

# %%
raw = read_table(cfg["data"])
print("Размер исходных данных:", raw.shape)
display(raw.head())

# %% [markdown]
# ### 1.1. Проверяем колонки, ID и пропуски
#
# Для начала нужны таргет, ID, дата, четыре итоговых выхода и исходные скоры подмоделей.
# Признаки итоговых .cbm понадобятся только в блоках 8–10.
# NaN таргета означает неизвестный исход, а не отсутствие дефолта.

# %%
required = {cfg["id_col"], cfg["date_col"], cfg["target"]}
for spec in levels["final_models"].values():
    required.update([spec["raw_score_col"], spec["pd_col"]])
for versions in levels.get("submodels", {}).values():
    required.update(spec["raw_score_col"] for spec in versions.values())

missing = sorted(required - set(raw.columns))
assert not missing, f"Нет колонок: {missing}. Исправьте имена в конфиге."
print("Пустых ID:", raw[cfg["id_col"]].isna().sum())
print("Повторяющихся ID:", raw[cfg["id_col"]].duplicated().sum())
display(raw[sorted(required)].isna().sum().rename("missing_count").to_frame())

# %% [markdown]
# ### 1.2. Оставляем вызревшие парные наблюдения
#
# df — заявки с известным таргетом, двумя итоговыми PD и датой до cutoff.
# _period — месяц выдачи; _window — reference/recent/other.
# quality_audit показывает, что исключилось на этом шаге.

# %%
df, quality_audit = prepare_frame(raw, cfg)
tables["data_quality"] = quality_audit
print("После фильтрации:", df.shape)
display(quality_audit)
display(df.groupby("_period")[cfg["target"]].agg(n="size", defaults="sum", bad_rate="mean"))

# %% [markdown]
# ## 2. Анализ Gini итоговых моделей
#
# Чтобы честно сравнить raw и PD, сначала оставляем общие строки с четырьмя выходами.
# final_df — эта общая выборка. raw_frame и pd_frame различаются только тем,
# какие значения используются для ранжирования; набор заявок одинаковый.

# %%
final_audit_rows = []
final_df = paired_cohort(
    df, list(levels["final_models"].values()), "final", final_audit_rows, require_pd=True
)
assert len(final_df), "Нет заявок с четырьмя итоговыми выходами"
display(pd.DataFrame(final_audit_rows))

raw_frame = final_df.copy()
pd_frame = final_df.copy()
for name, spec in levels["final_models"].items():
    raw_frame["_risk_" + name] = raw_risk(final_df, spec)
    pd_frame["_risk_" + name] = probability(final_df, spec["pd_col"])

metrics_raw = metric_table(raw_frame, cfg)
metrics_pd = metric_table(pd_frame, cfg)
tables["metrics_raw"] = metrics_raw
tables["metrics_pd"] = metrics_pd

gini_columns = [
    "scope", "n", "defaults", "gini_dub_22", "gini_dub_23",
    "delta_gini_23_minus_22", "ci_low", "ci_high", "reliable_size",
]
print("Чистые скоры:")
display(metrics_raw[gini_columns])
print("Калиброванные PD:")
display(metrics_pd[gini_columns])

# %% [markdown]
# ### 2.1. Рисуем Gini по месяцам
#
# delta_gini_23_minus_22 < 0 означает, что версия 23 ранжирует хуже версии 22.
# Общий Gini окна включает межмесячные пары: смотрим и месяцы, и recent/reference.

# %%
fig, ax = plt.subplots(figsize=(10, 4))
for representation, table in (("raw", metrics_raw), ("PD", metrics_pd)):
    monthly = table.loc[~table.scope.isin(["all", "reference", "recent"])]
    for name in NAMES:
        ax.plot(monthly.scope, monthly["gini_" + name] * 100,
                marker="o" if representation == "raw" else "x",
                linestyle="-" if representation == "raw" else "--",
                label=name + " / " + representation)
ax.set_ylabel("Gini, пункты")
ax.legend()
ax.tick_params(axis="x", rotation=45)
plt.show()

gap_change = window_gap_change(raw_frame, cfg)
tables["window_gap_change_raw"] = gap_change
display(gap_change)

# %% [markdown]
# ### 2.2. Сколько дефолтов и пар переранжировали
#
# defaults_better/worse — дефолты, у которых чистое положение относительно good улучшилось/ухудшилось.
# pairs_wrong_to_correct / pairs_correct_to_wrong — строгие исправления/ошибки пар.
# pairs_improved/worsened включают также переходы через равные скоры (tie).
# Числа пар точные; CI в предыдущем блоке относятся к Gini.

# %%
count_columns = [
    "scope", "defaults", "defaults_better", "defaults_worse", "defaults_unchanged",
    "pairs_bad_good_total", "pairs_improved", "pairs_worsened",
    "pairs_wrong_to_correct", "pairs_correct_to_wrong", "pair_net_units",
]
display(metrics_raw[count_columns])

# Здесь детализация по заявкам считается внутри каждого месяца.
defaults_raw = reranked_defaults(raw_frame, cfg)
defaults_pd = reranked_defaults(pd_frame, cfg)
tables["defaults_raw"] = defaults_raw
tables["defaults_pd"] = defaults_pd
display(defaults_raw.sort_values("quality_delta").head(20))

# %% [markdown]
# ### 2.3. Разбираем один выбранный срез и все 9 типов переходов пар
#
# Меняйте SCOPE на recent, reference, all или конкретный месяц (например, 2026-04).
# Wrong: у bad риск ниже good. Correct: у bad риск выше good. Tie: скоры равны.
# Это удобно для ручной проверки конкретного окна; его good могут отличаться от good месяца.

# %%
SCOPE = "recent"
REPRESENTATION = "raw"  # "raw" или "pd"
chosen_frame = raw_frame if REPRESENTATION == "raw" else pd_frame
chosen = dict(all_scopes(chosen_frame))[SCOPE].copy()
assert len(chosen), "Выбранный срез пуст"

matrix = pair_transitions(chosen[cfg["target"]], chosen._risk_dub_22, chosen._risk_dub_23)
display(pd.DataFrame(matrix.sum(axis=0), index=STATES, columns=STATES))
display(pd.Series(transition_statistics(matrix), name="value").to_frame())

# %% [markdown]
# ### 2.4. Сколько дефолтов ловим при одинаковой доле отклонения
#
# reject_rates задаются в конфиге. Это offline-проверка на уже наблюдаемых клиентах.

# %%
top_raw, migrations_raw = top_risk_diagnostics(raw_frame, cfg)
top_pd, migrations_pd = top_risk_diagnostics(pd_frame, cfg)
tables.update(top_risk_raw=top_raw, top_risk_pd=top_pd, migrations_raw=migrations_raw)
display(top_raw.loc[top_raw.scope == SCOPE])
display(top_pd.loc[top_pd.scope == SCOPE])

# %% [markdown]
# ## 3. Анализ качества подмоделей и сравнение версий
#
# Только исходные скоры. Для объяснения итогового разрыва все Gini считаем на 30@4,
# хотя сами early/long/soc_dem могли обучаться на разных подтаргетах.
# В каждой подмодели сравниваем версии на одних заявках. У soc_dem есть только версия 22.

# %%
submodel_frames = {}
submodel_audit = []
submodel_quality_rows = []
for component, versions in levels["submodels"].items():
    component_df = paired_cohort(df, list(versions.values()), component,
                                 submodel_audit, require_pd=False)
    submodel_frames[component] = component_df
    for scope, part in all_scopes(component_df):
        for name, spec in versions.items():
            ranker = WeightedGini(part[cfg["target"]], raw_risk(part, spec))
            statistics = metric_ci(part, cfg, lambda weights: {"gini_raw": ranker(weights)})
            submodel_quality_rows.append(dict(
                component=component, model=name, scope=scope, n=len(part),
                defaults=int(part[cfg["target"]].sum()), **statistics,
            ))

submodel_quality = pd.DataFrame(submodel_quality_rows)
tables["submodel_quality"] = submodel_quality
tables["submodel_cohort_audit"] = pd.DataFrame(submodel_audit)
display(submodel_quality.loc[submodel_quality.scope == SCOPE])

# %% [markdown]
# ### 3.1. Выбираем подмодель и сравниваем версии с CI и парами
#
# Для soc_dem сравнения версий нет: её качество доступно в предыдущей таблице.
# Результат sub_comparison можно разбирать как обычный словарь.

# %%
COMPONENT = "early_model"  # Или "long_model".
versions = levels["submodels"][COMPONENT]
assert set(versions) == set(NAMES), "У этой подмодели нет двух версий; смотрите submodel_quality"
sub_part = dict(all_scopes(submodel_frames[COMPONENT]))[SCOPE]

sub_comparison, sub_pairs, sub_defaults = compare_rankings(
    sub_part,
    raw_risk(sub_part, versions["dub_22"]),
    raw_risk(sub_part, versions["dub_23"]),
    cfg,
    metadata=dict(entity=COMPONENT, scope=SCOPE, representation="raw"),
)
display(pd.Series(sub_comparison).to_frame("value"))
display(pd.DataFrame(sub_pairs))
display(sub_defaults.sort_values("quality_delta").head(10))
tables["selected_submodel_comparison"] = pd.DataFrame([sub_comparison])

correlations = score_correlations(df, cfg)
tables["score_correlations"] = correlations
display(correlations.loc[correlations.scope == SCOPE])

# %% [markdown]
# ## 4. Регрессия разрыва Gini через качество подмоделей
#
# Сначала строим panel — одну строку на неперекрывающийся период и тип итогового выхода.
# Предикторы в обеих регрессиях одинаковые: качество исходных скоров подмоделей.
# Response меняется: разрыв итогового Gini на raw либо на PD.
#
# По умолчанию качества подмоделей и итоговый разрыв измеряются на разных половинах
# клиентов одного периода. Это снижает общую ошибку измерения. Проверяйте размеры половин.

# %%
regression_panel = regression_quality_panel(df, levels, cfg)
tables["submodel_regression_panel"] = regression_panel
display(regression_panel)

# %% [markdown]
# ### 4.1. Оцениваем регрессию и смотрим результат
#
# С пяти месяцев коэффициенты не оцениваются: status=insufficient_periods — ожидаемый результат.
# Не уменьшайте порог просто ради получения коэффициентов. Для реального анализа нужна
# более длинная вневыборочная история или обоснованная другая частота с достаточным числом bad/good.
# Коэффициенты описывают связь, а не доказывают причинный вклад подмодели.

# %%
regression_coefficients, regression_diagnostics, regression_fitted = fit_quality_regression(
    regression_panel, levels
)
tables.update(
    submodel_regression_coefficients=regression_coefficients,
    submodel_regression_diagnostics=regression_diagnostics,
    submodel_regression_fitted=regression_fitted,
)
display(regression_diagnostics)
if regression_coefficients.empty:
    print("Коэффициенты не оценены. Причина — в status/detail таблицы выше.")
else:
    display(regression_coefficients)
    display(regression_fitted)

# %% [markdown]
# ## 5. Теряем ли ранжирование из-за калибровки итоговой модели
#
# gap_change_due_to_calibration = разрыв Gini на PD − разрыв Gini на raw.
# Отрицательное значение: после перехода к PD версия 23 относительно 22 выглядит хуже.
# Сравнение выполняется на тех же четырёх выходах и тех же заявках, с paired CI.
# Строго возрастающий общий calibrator сохраняет ранги; ties и разные calibrator по сегментам
# или времени могут менять Gini. По подмоделям этот анализ не делаем.

# %%
calibration_gap_rows = [
    calibration_gap_decomposition(part, levels["final_models"], cfg, scope)
    for scope, part in all_scopes(final_df) if len(part)
]
calibration_gaps = pd.DataFrame(calibration_gap_rows)
tables["calibration_gap_decomposition"] = calibration_gaps
display(calibration_gaps)

# %% [markdown]
# ### 5.1. Проверяем score → PD и качество самих вероятностей
#
# Здесь разбираем только SCOPE, выбранный в блоке 2.3; для другого окна измените SCOPE
# и заново выполните этот блок. pd_target задаёт таргет итогового calibrator.
# Если он неизвестен, Brier/LogLoss на 30@4 — лишь диагностическое сравнение.
# raw_kind=score не позволяет считать Brier исходного балла. Для сравнения вероятностной
# ошибки до/после нужны uncalibrated_pd_col, raw_kind=probability или настоящий logit (margin).

# %%
calibration_part = dict(all_scopes(final_df))[SCOPE]
calibration_rows = []
curve_rows = []
pd_bins = np.asarray(levels["pd_bins"], dtype=float)

for name, spec in levels["final_models"].items():
    score = raw_risk(calibration_part, spec)
    pd_values = probability(calibration_part, spec["pd_col"])
    metadata = dict(entity="final", model=name, scope=SCOPE)
    ranking_stats, pair_rows, default_rows = compare_rankings(
        calibration_part, score, pd_values, cfg, metadata
    )
    ranking_stats.update(calibration_mapping(score, pd_values))

    own_target = spec.get("pd_target")
    probability_target = own_target if own_target in calibration_part.columns else cfg["target"]
    if own_target != probability_target:
        print(name, ": pd_target не задан/отсутствует; вероятность сравнивается с", probability_target)
    probability_part = calibration_part.loc[calibration_part[probability_target].notna()]
    probability_cfg = dict(cfg, target=probability_target)
    before_pd = before_calibration_probability(probability_part, spec)
    probability_stats, bins = probability_metrics(
        probability_part, probability(probability_part, spec["pd_col"]), before_pd,
        probability_cfg, dict(metadata, probability_target=probability_target), pd_bins,
    )
    calibration_rows.append(dict(
        **ranking_stats, **{k: v for k, v in probability_stats.items() if k not in ranking_stats},
        n_probability_evaluation=len(probability_part),
    ))
    curve_rows.extend(bins)

calibration_metrics = pd.DataFrame(calibration_rows)
calibration_curves = pd.DataFrame(curve_rows)
tables.update(calibration_metrics=calibration_metrics, calibration_curves=calibration_curves)
display(calibration_metrics.T)
display(calibration_curves)

# %% [markdown]
# ### 5.2. Рисуем график калибровки
#
# По X — средний PD в бине, по Y — наблюдаемая частота таргета.
# Для осмысленного вывода о калибровке probability_target должен совпадать с таргетом PD.

# %%
fig, ax = plt.subplots(figsize=(6, 5))
if not calibration_curves.empty:
    calibrated = calibration_curves.loc[calibration_curves.probability_type == "pd"]
    for name, part in calibrated.groupby("model"):
        ax.plot(part.mean_pd, part.observed_rate, marker="o", label=name)
ax.plot([0, 1], [0, 1], "--", color="gray")
ax.set(xlabel="Средний PD", ylabel="Частота таргета", title="Калибровка: " + SCOPE)
handles, labels = ax.get_legend_handles_labels()
if handles:
    ax.legend()
plt.show()

# %% [markdown]
# ### 5.3. Одна функция: перемещение dub_23 между сегментами калибровки
#
# Группировка произвольная: существующая колонка квартала, месяца, версии или другого среза.
# Ниже для примера явно создаём quarter; функция сама даты в месяцы НЕ преобразует.
# REFERENCE_VALUE и COMPARISON_VALUE — два значения GROUP_COL, а не даты или PD-пороги.
# В каждой группе ранги считаются по всем трём сегментам вместе. Подмодели не нужны.
#
# Подайте raw (исходные данные), а не final_df: для этого анализа нужны только выходы dub_23,
# и отсутствие dub_22 не должно исключать выдачу. Флаги сегментов: ровно один 1 на заявку.
# Если сегмент хранится одной колонкой, замените segment_flag_cols на segment_col и segment_order.

# %%
from calibration_movement import plot_dub23_calibration

movement_data = raw.copy()
GROUP_COL = "quarter"
movement_data[GROUP_COL] = pd.to_datetime(movement_data[cfg["date_col"]]).dt.to_period("Q").astype(str)
REFERENCE_VALUE = "2025Q4"  # Замените на значение из своих данных.
COMPARISON_VALUE = "2026Q1"
GROUP_ORDER = sorted(movement_data[GROUP_COL].dropna().unique())

movement_result = plot_dub23_calibration(
    movement_data,
    group_col=GROUP_COL,
    reference_value=REFERENCE_VALUE,
    comparison_value=COMPARISON_VALUE,
    raw_score_col="score_dub_23",
    pd_col="pd_dub_23",
    target_col=cfg["target"],
    id_col=cfg["id_col"],
    segment_flag_cols=["sum_request_less_500", "sum_request_500_1000", "sum_request_more_1000k"],
    group_order=GROUP_ORDER,
    higher_score_higher_risk=True,
    date_col=cfg["date_col"],
    max_issue_date=cfg.get("max_issue_date"),
    cluster_col=cfg.get("cluster_col"),
    bootstrap_iterations=cfg["bootstrap_iterations"],
    scatter_max_goods=3000,
    output_dir=PROJECT_DIR / "reports" / "calibration_movement",
)

for name, fig in movement_result["figures"].items():
    print(name)
    display(fig)
for message in movement_result["messages"]:
    print(message)

# %% [markdown]
# ### 5.4. Проверяем расчёты за графиками
#
# rank_movements — все использованные заявки с исходным рангом, новым рангом и сдвигом.
# cohort_audit — фильтрация. gini_by_segment: __all__ означает все три сегмента вместе.
# segment_pairs: строки — сегмент дефолта, столбцы — сегмент недефолта.
# gini_delta_contribution суммируется до общей Gini(PD)−Gini(raw) внутри выбранной группы.
# Верхняя полоса на графике динамики — разброс сдвигов клиентов, нижняя — статистический CI.

# %%
display(movement_result["tables"]["cohort_audit"])
display(movement_result["tables"]["gini_by_segment"])
display(movement_result["tables"]["segment_pairs"].loc[
    lambda table: table["group"].eq(COMPARISON_VALUE)
])
display(movement_result["tables"]["rank_movements"].sort_values("_cal_rank_shift").head(20))

# Если нужна только одна картинка:
# display(movement_result["figures"]["rank_migration"])
# output_dir=None в вызове выше оставляет результаты только в памяти.

# %% [markdown]
# ## 6. Проверяем сдвиги признаков и сегменты
#
# PSI считается между reference и recent. Это индикатор сдвига, не доказательство причины.
# Для сегментов задайте категориальные колонки в cfg['segments']; числовые заранее разбейте на интервалы.

# %%
drift_columns = list(cfg.get("components", {}))
drift_columns += [col for col in df if col.lower().startswith("fps")]
for spec in levels["final_models"].values():
    drift_columns += [spec["raw_score_col"], spec["pd_col"]]

drift = drift_table(df, drift_columns)
segments = segment_table(df, cfg)
tables.update(drift=drift, segments=segments)
display(drift.head(20))
if segments.empty:
    print("Сегменты не заданы: добавьте названия колонок в cfg['segments'].")
else:
    display(segments.loc[segments.scope == SCOPE])

# %% [markdown]
# ## 7. Дополнительный таргет 30@6
#
# Его вызревание может отставать от 30@4. Используем только наблюдаемые 0/1;
# пустой recent не заменяем нулевыми таргетами. Набор клиентов может отличаться от основного.

# %%
SECONDARY_TARGET = "flag_6m_30p_gross"
if SECONDARY_TARGET in raw and raw[SECONDARY_TARGET].notna().any():
    secondary_cfg = dict(deepcopy(cfg), target=SECONDARY_TARGET, allow_missing_window_periods=True)
    secondary_df, secondary_audit = prepare_frame(raw, secondary_cfg)
    secondary_metrics = metric_table(secondary_df, secondary_cfg)
    tables.update(secondary_metrics=secondary_metrics, secondary_audit=secondary_audit)
    display(secondary_metrics[gini_columns])
else:
    print("Дополнительный таргет отсутствует или ещё не наблюдается.")

# %% [markdown]
# ## 8. Загружаем .cbm и проверяем признаки
#
# Для блоков 8–10 нужны обе модели и признаки с тем же preprocessing, что в production.
# Для проверки скоров/регрессии/калибровки они не нужны: поставьте RUN_MODEL_DIAGNOSTICS=False.
# Загруженная модель сверяется с её чистым скором. Подмодели отдельно не переобучаются.

# %%
models = {}
sample = pd.DataFrame()
if RUN_MODEL_DIAGNOSTICS:
    models, score_reproduction = load_models_and_scores(raw.copy(), cfg, scores_only=False)
    assert set(models) == set(NAMES), "Укажите пути к обеим итоговым .cbm в cfg['models']"
    schema = model_schema(models, cfg)
    tables.update(score_reproduction=score_reproduction, model_features=schema)
    display(score_reproduction)
    display(schema)
    sample = diagnostic_sample(final_df, cfg)
    print("Диагностическая подвыборка:", sample.shape)
    display(sample.groupby("_window")[cfg["target"]].agg(n="size", defaults="sum"))
else:
    print("Разбор файлов моделей выключен; блоки 9–10 будут пропущены.")

# %% [markdown]
# ## 9. Признаки и группы: SHAP и изменение ранжирования
#
# Анализируем raw выход итогового CatBoost до внешней калибровки.
# Отрицательный gini_delta_allocation группы связан с ухудшением разрыва на подвыборке.
# improvement_without_shap_feature > 0: вычитание вклада признака улучшило Gini.
# Эти гибриды не равны переобучению без признака и не доказывают причинность.

# %%
if models:
    shap_tables = shap_diagnostics(models, sample, cfg)
    tables.update(shap_tables)
    allocation = shap_tables["gini_group_allocation"]
    display(allocation.loc[allocation.scope == SCOPE])
    feature_sensitivity = shap_tables["shap_features"]
    display(feature_sensitivity.loc[feature_sensitivity.window == "recent"].sort_values(
        "improvement_without_shap_feature", ascending=False
    ).head(20))

# %% [markdown]
# ### 9.1. Меняем скоры подмоделей между версиями
#
# Итоговый бустинг фиксирован: вместо его early_23 пробуем early_22, и аналогично long.
# Полезно только при совместимой шкале компонентных скоров. Не подставляем raw вместо PD.
# Положительный improvement — такая подстановка улучшила ранжирование на срезе.

# %%
if models:
    swaps = component_swaps(models, sample, cfg)
    tables["component_swaps"] = swaps
    display(swaps.loc[swaps.scope == SCOPE].sort_values("improvement", ascending=False))

# %% [markdown]
# ## 10. Где в деревьях ухудшается качество
#
# Для отладки сначала разбираем одну модель. Поменяйте MODEL_TO_INSPECT и выполните блок заново.
# tree_step — сколько деревьев в одном блоке; max_local_trees ограничивает локальные таблицы.
# После просмотра можно поставить tree_step=1 для подробного анализа, но это дольше.
# Номер дерева dub_22 не соответствует тому же номеру dub_23.

# %%
MODEL_TO_INSPECT = "dub_23"
if models:
    tree_blocks, tree_splits, tree_branches, tree_leaves, tree_warning = tree_diagnostics(
        models[MODEL_TO_INSPECT], MODEL_TO_INSPECT, sample, cfg
    )
    tables.update({
        MODEL_TO_INSPECT + "_tree_blocks": tree_blocks,
        MODEL_TO_INSPECT + "_tree_splits": tree_splits,
        MODEL_TO_INSPECT + "_tree_branches": tree_branches,
        MODEL_TO_INSPECT + "_tree_leaves": tree_leaves,
    })
    if tree_warning:
        print("Ограничение локального разбора:", tree_warning)
    display(tree_blocks.sort_values("improvement_without_block_recent", ascending=False).head(10))
    display(tree_splits.head(20))
    if not tree_branches.empty:
        display(tree_branches.sort_values("branch_share_shift", key=abs, ascending=False).head(20))

# %% [markdown]
# ### 10.1. Один конкретный сплит
#
# У дерева и глубины нумерация с нуля. node_prefix различает узлы одного уровня.
# Ветка 1 не обязательно означает больший риск. Смотрим объёмы, bad rate и reliable_size.

# %%
TREE_ID = 0
DEPTH = 0
if models and not tree_splits.empty:
    display(tree_splits.loc[(tree_splits.tree == TREE_ID) & (tree_splits.depth == DEPTH)])
    if not tree_branches.empty:
        branch_slice = tree_branches.loc[
            (tree_branches.tree == TREE_ID) & (tree_branches.depth == DEPTH)
        ]
        if branch_slice.empty:
            print("Это дерево не вошло в локальную выборку. Выберите tree из tree_branches.")
        display(branch_slice)

# %% [markdown]
# ## 11. Сохраняем промежуточные таблицы
#
# Сохраняются только результаты выполненных блоков. Папка reports исключена из Git.
# Перед публикацией своего ноутбука очистите outputs: в них могут остаться ID и данные клиентов.

# %%
NOTEBOOK_OUTPUT = PROJECT_DIR / "reports" / "notebook_debug"
NOTEBOOK_OUTPUT.mkdir(parents=True, exist_ok=True)
for name, table in tables.items():
    write_csv(table, NOTEBOOK_OUTPUT, name)
print("Таблицы сохранены:", NOTEBOOK_OUTPUT)
print("Доступные результаты:", sorted(tables))

# %% [markdown]
# ## 12. Как отлаживать дальше
#
# - Изменили путь, таргет или периоды — повторите ячейки с блока 0.1.
# - Изменили SCOPE — повторите зависящие от него блоки 2.3, 2.4, 3 и 5.
# - Изменили исходники .py — autoreload подхватит изменения, но таблицы нужно пересчитать.
# - После ошибки Jupyter можно выполнить `%debug` в новой ячейке и посмотреть локальные переменные.
# - Чтобы увидеть реализацию функции: `import inspect; print(inspect.getsource(metric_table))`.
# - Для чистого воспроизведения используйте Restart Kernel → Run All.
#
# Ноутбук вызывает функции по шагам. Когда ручной анализ понятен, полный автоматический
# отчёт можно получить отдельно через `diagnose.run(deepcopy(cfg), scores_only=not RUN_MODEL_DIAGNOSTICS)`.
