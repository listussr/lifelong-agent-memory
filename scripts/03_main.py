from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import CONFIG, load_pool, make_rng
from src.utils import make_episode
from src.memory import make_policy
from src.rollout import rollout_loop
from src.readout import (predict, evaluate_accuracy, evaluate_accuracy_no_dup,
                         evaluate_usability, evaluate_precision)

POLICIES = ["nomemory", "recency", "random", "semantic", "kmeans", "oracle"]

READOUT_SEED_OFFSET = 10_000   # чтобы rng голосования не совпадал с rng политики

# Категориальные слоты валидированной палитры: худшая соседняя пара CVD dE 9.1,
# normal-vision 22.9. Пол и потолок — не серии, а опорные линии, поэтому серые.
SERIES_COLORS = {
    "semantic": "#2a78d6",
    "kmeans":   "#eb6834",
    "recency":  "#1baf7a",
    "random":   "#eda100",
}
REF_COLOR = "#898781"
INK       = "#0b0b0b"
INK_2     = "#52514e"
GRID      = "#e1e0d9"
AXIS      = "#c3c2b7"

RU = {
    "nomemory": "без памяти",
    "recency":  "свежесть",
    "random":   "случайные k",
    "semantic": "семантика",
    "kmeans":   "кластеры",
    "oracle":   "oracle",
}


def run_one(episode: List, policy_name: str, emb: np.ndarray, meta: np.ndarray,
            seed: int, budget: int, k: int) -> Dict:
    """
    Один прогон: одна политика на одном эпизоде.
    ---

    Args:
        episode (List): Поток шагов, общий для всех политик этого сида.
        policy_name (str): Имя политики для фабрики.
        emb (np.ndarray): Эмбеддинги пула.
        meta (np.ndarray): Скрытая разметка; уходит только в Oracle.
        seed (int): Сид прогона.
        budget (int): Бюджет памяти.
        k (int): Сколько извлекать.

    Returns:
        Dict: Строка результатов.
    """
    policy = make_policy(
        policy_name, budget, emb, meta, make_rng(seed),
        CONFIG["memory"]["kmeans_clusters"],
        CONFIG["memory"]["kmeans_refit_every"],
        seed,
    )
    roll = rollout_loop(episode, policy, emb, k)
    pred = predict(roll, make_rng(seed + READOUT_SEED_OFFSET))

    accuracy, accuracy_last_third = evaluate_accuracy(roll, pred)
    acc_no_dup, acc_dup_only      = evaluate_accuracy_no_dup(roll, pred)
    hit_at_k, hit_at_k_no_dup     = evaluate_usability(roll)

    return {
        "policy":               policy_name,
        "budget":               budget,
        "k":                    k,
        "accuracy":             accuracy,
        "acc_last_third":       accuracy_last_third,
        "acc_no_dup":           acc_no_dup,
        "acc_dup_only":         acc_dup_only,
        "hit_at_k":             hit_at_k,
        "hit_at_k_no_dup":      hit_at_k_no_dup,
        "precision":            evaluate_precision(roll),
        "exact_dup_rate":       float(roll.exact_dup.mean()),
        "n_retrieved_mean":     float(roll.n_retrieved.mean()),
        "memory_occupancy_max": int(len(policy.items)),
    }


def main() -> None:
    """
    Полная сетка прогонов и запись результатов.
    ---
    """
    texts, meta, emb, index = load_pool(CONFIG["pool"]["dir"])

    alphas  = CONFIG["env"]["alphas"]
    seeds   = CONFIG["seeds"]
    eta     = CONFIG["env"]["eta"]
    n_steps = CONFIG["env"]["n_steps"]
    budget  = CONFIG["memory"]["budget"]
    k       = CONFIG["memory"]["k"]

    print(f"пул: {len(texts)} текстов, эмбеддинги {emb.shape}")
    print(f"сетка: {len(alphas)} alpha x {len(seeds)} сидов x {len(POLICIES)} политик "
          f"= {len(alphas) * len(seeds) * len(POLICIES)} прогонов")
    print(f"бюджет={budget}  k={k}  шагов={n_steps}  eta={eta}\n")

    rows = []
    for alpha in alphas:
        for seed in seeds:
            # 1 эпизод на все политики этого (alpha, seed)
            episode = make_episode(alpha, eta, n_steps, index, make_rng(seed))
            for policy_name in POLICIES:
                row = run_one(episode, policy_name, emb, meta, seed, budget, k)
                row.update(alpha=alpha, seed=seed, eta=eta, n_steps=n_steps)
                rows.append(row)
            done = len(rows)
            total = len(alphas) * len(seeds) * len(POLICIES)
            print(f"  alpha={alpha}  seed={seed}  готово {done}/{total}")

    columns = ["alpha", "seed", "policy", "budget", "k", "eta", "n_steps",
               "accuracy", "acc_last_third", "acc_no_dup", "acc_dup_only",
               "precision", "hit_at_k", "hit_at_k_no_dup",
               "exact_dup_rate", "n_retrieved_mean", "memory_occupancy_max"]
    df = pd.DataFrame(rows)[columns]

    main_csv = Path(CONFIG["output"]["main_csv"])
    main_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(main_csv, index=False)

    summary = (
        df.groupby(["alpha", "policy"], sort=False)
          .agg(accuracy_mean=("accuracy", "mean"),
               accuracy_std=("accuracy", "std"),
               last_third_mean=("acc_last_third", "mean"),
               no_dup_mean=("acc_no_dup", "mean"),
               no_dup_std=("acc_no_dup", "std"),
               dup_only_mean=("acc_dup_only", "mean"),
               precision_mean=("precision", "mean"),
               hit_at_k_mean=("hit_at_k", "mean"),
               exact_dup_mean=("exact_dup_rate", "mean"),
               n_retrieved_mean=("n_retrieved_mean", "mean"))
          .reset_index()
    )
    summary_csv = main_csv.with_name("main_summary.csv")
    summary.to_csv(summary_csv, index=False)

    print(f"\nзаписано: {main_csv}  ({len(df)} строк)")
    print(f"записано: {summary_csv}")

    visualize(df, summary, Path(CONFIG["output"]["figures_dir"]))

    for alpha in alphas:
        print(f"alpha = {alpha}")
        part = summary[summary["alpha"] == alpha]
        for _, r in part.iterrows():
            print(f"  {r['policy']:10} {r['accuracy_mean']:.3f} +- {r['accuracy_std']:.3f}"
                  f"   без дублей={r['no_dup_mean']:.3f}"
                  f"   хвост={r['last_third_mean']:.3f}"
                  f"   точность извлечения={r['precision_mean']:.3f}"
                  f"   дубли={r['exact_dup_mean'] * 100:.1f}%")
        print()

    dup_control(summary)
    check(summary)


def dup_control(summary: pd.DataFrame) -> None:
    """
    Отчёт по контролю на точные дубликаты.
    ---

    Сравнивает эффект alpha, посчитанный на всех шагах и только на шагах без
    дословных совпадений. Если эффект сохраняется — альтернативное объяснение
    «выигрыш это память на дубликаты» закрыто.

    Args:
        summary (pd.DataFrame): Сводная таблица.
    """
    alphas = sorted(summary["alpha"].unique())
    lo, hi = alphas[0], alphas[-1]

    def get(alpha, policy, col):
        row = summary[(summary["alpha"] == alpha) & (summary["policy"] == policy)]
        return float(row[col].iloc[0])

    print("=" * 78)
    print("КОНТРОЛЬ НА ТОЧНЫЕ ДУБЛИКАТЫ")
    print("=" * 78)
    print(f"{'политика':12} {'эффект alpha':>14} {'без дублей':>12} {'сохранилось':>13}"
          f" {'на дублях':>11}")
    for policy in list(SERIES_COLORS) + ["oracle"]:
        full  = get(hi, policy, "accuracy_mean") - get(lo, policy, "accuracy_mean")
        clean = get(hi, policy, "no_dup_mean")   - get(lo, policy, "no_dup_mean")
        share = clean / full * 100 if abs(full) > 1e-9 else float("nan")
        on_dup = get(hi, policy, "dup_only_mean")
        print(f"  {policy:10} {full:+13.3f} {clean:+12.3f} {share:12.0f}% {on_dup:11.3f}")
    print()
    print(f"  «эффект alpha» = точность при alpha={hi} минус при alpha={lo}")
    print(f"  «на дублях»    = точность на шагах с дословным совпадением, alpha={hi}")
    print("=" * 78 + "\n")


def _spread(values: np.ndarray, min_gap: float) -> np.ndarray:
    """
    Раздвигает подписи по вертикали, чтобы они не наезжали друг на друга.
    ---

    Args:
        values (np.ndarray): Исходные позиции.
        min_gap (float): Минимальный зазор.

    Returns:
        np.ndarray: Разведённые позиции.
    """
    out   = np.array(values, dtype=float)
    order = np.argsort(out)
    for i in range(1, len(order)):
        prev, cur = order[i - 1], order[i]
        if out[cur] - out[prev] < min_gap:
            out[cur] = out[prev] + min_gap
    return out


def _style(ax) -> None:
    """
    Общее оформление осей: рецессивная сетка, без верхней и правой рамки.
    ---

    Args:
        ax: Оси matplotlib.
    """
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=9)


def _line_panel(ax, summary: pd.DataFrame, value_col: str, err_col: str,
                title: str, ylabel: str, with_reference: bool) -> None:
    """
    Панель «метрика против alpha»: линии методов плюс опорные пол и потолок.
    ---

    Args:
        ax: Оси matplotlib.
        summary (pd.DataFrame): Сводная таблица.
        value_col (str): Столбец со средним.
        err_col (str): Столбец со стандартным отклонением; None — без усов.
        title (str): Заголовок панели.
        ylabel (str): Подпись оси Y.
        with_reference (bool): Рисовать ли полосу между полом и потолком.
    """
    alphas = sorted(summary["alpha"].unique())

    def series(policy, col):
        part = summary[summary["policy"] == policy].set_index("alpha")
        return np.array([part.loc[a, col] for a in alphas], dtype=float)

    if with_reference:
        floor, ceil = series("nomemory", value_col), series("oracle", value_col)
        ax.fill_between(alphas, floor, ceil, color=REF_COLOR, alpha=0.08, lw=0)
        for values, name in ((ceil, "oracle"), (floor, "nomemory")):
            ax.plot(alphas, values, ls="--", lw=1.6, color=REF_COLOR, zorder=2)
            ax.annotate(RU[name], xy=(alphas[-1], values[-1]),
                        xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8.5, color=INK_2)

    ends, names = [], []
    for policy, color in SERIES_COLORS.items():
        values = series(policy, value_col)
        if err_col is not None:
            ax.errorbar(alphas, values, yerr=series(policy, err_col),
                        color=color, lw=2.0, marker="o", ms=6.5,
                        capsize=3, elinewidth=1.2, zorder=3, label=RU[policy])
        else:
            ax.plot(alphas, values, color=color, lw=2.0, marker="o", ms=6.5,
                    zorder=3, label=RU[policy])
        ends.append(values[-1])
        names.append(policy)

    for y, y_lab, policy in zip(ends, _spread(np.array(ends), 0.028), names):
        ax.annotate(RU[policy], xy=(alphas[-1], y_lab),
                    xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK_2)

    ax.set_xticks(alphas)
    ax.set_xlabel(r"$\alpha$  —  привязка темы к проекту", fontsize=10, color=INK_2)
    ax.set_ylabel(ylabel, fontsize=10, color=INK_2)
    ax.set_title(title, fontsize=12, color=INK, pad=10)
    ax.set_xlim(min(alphas) - 0.05, max(alphas) + 0.32)
    _style(ax)


def visualize(df: pd.DataFrame, summary: pd.DataFrame, figures_dir: Path) -> None:
    """
    Построение и сохранение графиков главного прогона.
    ---

    Сохраняет два файла:
    * main_accuracy.png — точность и точность извлечения против alpha,
    * main_diagnostics.png — связь механизма с результатом и доля точных дубликатов.

    Args:
        df (pd.DataFrame): Полная таблица прогонов.
        summary (pd.DataFrame): Сводная таблица.
        figures_dir (Path): Куда сохранять.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    alphas = sorted(summary["alpha"].unique())

    # ---------- главный рисунок ----------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    _line_panel(ax1, summary, "accuracy_mean", "accuracy_std", "Точность решения", "доля верных действий", with_reference=True)
    ax1.set_ylim(0.25, 1.02)
    _line_panel(ax2, summary, "precision_mean", None, "Точность извлечения (механизм)", "доля извлечённых с верным действием", with_reference=False)
    ax2.set_ylim(0.30, 1.02)
    ax2.axhline(1.0, color=REF_COLOR, ls="--", lw=1.6, zorder=2)
    ax2.annotate(RU["oracle"], xy=(alphas[-1], 1.0), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8.5, color=INK_2)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9.5, labelcolor=INK_2, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(r"Точность против связи семантики с правилом  (5 сидов, усы — ±1$\sigma$)",
                 fontsize=13, color=INK, y=1.0)
    fig.tight_layout()
    fig.savefig(figures_dir / "main_accuracy.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---------- диагностика ----------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    ends = []
    for policy in list(SERIES_COLORS) + ["oracle"]:
        color = SERIES_COLORS.get(policy, REF_COLOR)
        part  = summary[summary["policy"] == policy].set_index("alpha")
        xs = np.array([part.loc[a, "precision_mean"] for a in alphas], float)
        ys = np.array([part.loc[a, "accuracy_mean"] for a in alphas], float)
        ax1.plot(xs, ys, color=color, lw=1.2, alpha=0.7, zorder=2)
        ax1.scatter(xs, ys, color=color, s=55, zorder=3,
                    edgecolor="white", linewidth=1.2)
        ends.append((xs[-1], ys[-1], policy))

    for (x, _, policy), y_lab in zip(ends, _spread(np.array([e[1] for e in ends]), 0.022)):
        ax1.annotate(RU[policy], xy=(x, y_lab), xytext=(8, -3),
                     textcoords="offset points", fontsize=8.5, color=INK_2)
    ax1.set_xlabel("точность извлечения", fontsize=10, color=INK_2)
    ax1.set_ylabel("точность решения", fontsize=10, color=INK_2)
    ax1.set_title("Результат идёт за механизмом", fontsize=12, color=INK, pad=10)
    ax1.set_xlim(0.35, 1.12)
    _style(ax1)
    ax1.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.9)

    ends = []
    for policy in list(SERIES_COLORS) + ["oracle"]:
        color = SERIES_COLORS.get(policy, REF_COLOR)
        part  = summary[summary["policy"] == policy].set_index("alpha")
        ys = np.array([part.loc[a, "exact_dup_mean"] for a in alphas], float) * 100
        ax2.plot(alphas, ys, color=color, lw=2.0, marker="o", ms=6.5, zorder=3)
        ends.append((ys[-1], policy))

    for (_, policy), y_lab in zip(ends, _spread(np.array([e[0] for e in ends]), 1.1)):
        ax2.annotate(RU[policy], xy=(alphas[-1], y_lab), xytext=(6, 0),
                     textcoords="offset points", va="center",
                     fontsize=8.5, color=INK_2)
    ax2.set_xticks(alphas)
    ax2.set_xlabel(r"$\alpha$  —  привязка темы к проекту", fontsize=10, color=INK_2)
    ax2.set_ylabel("доля шагов, %", fontsize=10, color=INK_2)
    ax2.set_title("Точные дубликаты запроса в памяти (конфаунд)",
                  fontsize=12, color=INK, pad=10)
    ax2.set_xlim(min(alphas) - 0.05, max(alphas) + 0.32)
    _style(ax2)

    fig.tight_layout()
    fig.savefig(figures_dir / "main_diagnostics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---------- контроль на дубликаты ----------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    ends = []
    for policy, color in SERIES_COLORS.items():
        part = summary[summary["policy"] == policy].set_index("alpha")
        full  = np.array([part.loc[a, "accuracy_mean"] for a in alphas], float)
        clean = np.array([part.loc[a, "no_dup_mean"] for a in alphas], float)
        ax1.plot(alphas, full, color=color, lw=2.0, marker="o", ms=6.5, zorder=3)
        ax1.plot(alphas, clean, color=color, lw=1.8, ls=":", marker="s", ms=5.5,
                 alpha=0.85, zorder=3)
        ends.append((clean[-1], policy))

    for (_, policy), y_lab in zip(ends, _spread(np.array([e[0] for e in ends]), 0.028)):
        ax1.annotate(RU[policy], xy=(alphas[-1], y_lab), xytext=(6, 0),
                     textcoords="offset points", va="center",
                     fontsize=8.5, color=INK_2)
    ax1.plot([], [], color=INK_2, lw=2.0, marker="o", ms=6.5, label="все шаги")
    ax1.plot([], [], color=INK_2, lw=1.8, ls=":", marker="s", ms=5.5,
             label="без точных дубликатов")
    ax1.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper left")
    ax1.set_xticks(alphas)
    ax1.set_xlabel(r"$\alpha$  —  привязка темы к проекту", fontsize=10, color=INK_2)
    ax1.set_ylabel("доля верных действий", fontsize=10, color=INK_2)
    ax1.set_title(r"Эффект $\alpha$ сохраняется без дословных совпадений",
                  fontsize=12, color=INK, pad=10)
    ax1.set_xlim(min(alphas) - 0.05, max(alphas) + 0.32)
    _style(ax1)

    policies = list(SERIES_COLORS)
    x = np.arange(len(policies))
    part = summary[summary["alpha"] == alphas[-1]].set_index("policy")
    clean_vals = [part.loc[p, "no_dup_mean"] for p in policies]
    dup_vals   = [part.loc[p, "dup_only_mean"] for p in policies]
    ax2.bar(x - 0.19, clean_vals, width=0.36, color=REF_COLOR, alpha=0.55,
            label="шаги без дубликата")
    ax2.bar(x + 0.19, dup_vals, width=0.36,
            color=[SERIES_COLORS[p] for p in policies], label="шаги с дубликатом")
    for xi, (a, b) in enumerate(zip(clean_vals, dup_vals)):
        ax2.text(xi - 0.19, a + 0.015, f"{a:.2f}", ha="center", fontsize=8.5, color=INK_2)
        ax2.text(xi + 0.19, b + 0.015, f"{b:.2f}", ha="center", fontsize=8.5, color=INK_2)
    ax2.set_xticks(x)
    ax2.set_xticklabels([RU[p] for p in policies], fontsize=9.5, color=INK_2)
    ax2.set_ylabel("доля верных действий", fontsize=10, color=INK_2)
    ax2.set_ylim(0, 1.12)
    ax2.set_title(rf"Цена дубликата при $\alpha$ = {alphas[-1]}",
                  fontsize=12, color=INK, pad=10)
    ax2.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper left")
    _style(ax2)

    fig.tight_layout()
    fig.savefig(figures_dir / "main_dup_control.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"записано: {figures_dir / 'main_accuracy.png'}")
    print(f"записано: {figures_dir / 'main_diagnostics.png'}")
    print(f"записано: {figures_dir / 'main_dup_control.png'}\n")


def check(summary: pd.DataFrame) -> None:
    """
    Санити-чеки, без которых результат нельзя интерпретировать.
    ---

    Args:
        summary (pd.DataFrame): Сводная таблица.
    """
    def get(alpha, policy, col="accuracy_mean"):
        row = summary[(summary["alpha"] == alpha) & (summary["policy"] == policy)]
        return float(row[col].iloc[0])

    alphas = sorted(summary["alpha"].unique())
    lo, hi = alphas[0], alphas[-1]
    problems = []

    for alpha in alphas:
        if abs(get(alpha, "nomemory") - 1 / 3) > 0.05:
            problems.append(f"nomemory при alpha={alpha} = {get(alpha, 'nomemory'):.3f}, "
                            f"ожидалось ~0.33 — правило подтекает в текст")
        if get(alpha, "oracle") < 0.85:
            problems.append(f"oracle при alpha={alpha} = {get(alpha, 'oracle'):.3f} — "
                            f"низкий потолок, проверь бюджет и передачу project")

    if abs(get(hi, "oracle") - get(lo, "oracle")) > 0.05:
        problems.append("потолки Oracle при крайних alpha расходятся — "
                        "alpha влияет не только на семантику, интерпретация невозможна")
    if get(hi, "semantic") - get(lo, "semantic") < 0.05:
        problems.append("semantic не проседает при alpha=0 — рубильник не работает")

    print("=" * 70)
    if problems:
        print("ПРОБЛЕМЫ:")
        for p in problems:
            print("  -", p)
    else:
        print("все санити-чеки пройдены")
    print("=" * 70)


if __name__ == "__main__":
    main()
