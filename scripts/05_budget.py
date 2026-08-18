"""
Свип по бюджету памяти: проверка предсказания о вытеснении.
---

Предсказание, записанное ДО прогона: вытеснение помогает при малом бюджете и
почти не помогает при большом. Логика — при большом бюджете вытеснять почти не
приходится, память вмещает всё нужное, и правило выбора жертвы не важно.

Вклад вытеснения измеряется в двух пространствах сразу, чтобы отделить его от
метрики:
    в исходном:         eviction - semantic
    в спроецированном:  full     - metric

Прогон при alpha = 0, где преимущество решенческой памяти максимально.

Запуск из корня проекта:
    python -m scripts.05_budget
"""
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import CONFIG, load_pool, make_rng
from src.utils import make_episode
from src.memory import DecisionEviction, NoMemory, Oracle, Semantic
from src.metric import (build_buckets, collect_episodes, fit_metric,
                        project, projection_matrix, sample_pairs)
from src.rollout import rollout_loop
from src.readout import predict, evaluate_accuracy, evaluate_precision

READOUT_SEED_OFFSET = 10_000

CONFIGS: Dict[str, Tuple[type, str, str]] = {
    "nomemory": (NoMemory,         "orig", "без памяти"),
    "semantic": (Semantic,         "orig", "семантика"),
    "eviction": (DecisionEviction, "orig", "только вытеснение"),
    "metric":   (Semantic,         "proj", "только метрика"),
    "full":     (DecisionEviction, "proj", "полный метод"),
    "oracle":   (Oracle,           "orig", "oracle"),
}

# пары «различаются только вытеснением» — из них считается его вклад
EVICT_PAIRS = [("eviction", "semantic", "в исходном пространстве", "#1baf7a"),
               ("full",     "metric",   "в пространстве метрики",  "#2a78d6")]

SERIES_COLORS = {
    "full":     "#2a78d6",
    "metric":   "#eb6834",
    "eviction": "#1baf7a",
    "semantic": "#eda100",
}
REF_COLOR = "#898781"
INK, INK_2, GRIDC, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"


def build_spaces(emb: np.ndarray, index: Dict, seeds: List[int]) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Исходный и решенческий пул на каждый сид.
    ---

    Повторяет процедуру из 04_ablation, но без контролей: здесь они не нужны.
    Метрика от бюджета не зависит, поэтому обучается один раз до всех прогонов.

    Args:
        emb (np.ndarray): Эмбеддинги пула.
        index (Dict): Индекс пула.
        seeds (List[int]): Оценочные сиды.

    Returns:
        Dict[int, Dict[str, np.ndarray]]: сид -> имя пространства -> пул.
    """
    episodes = collect_episodes(index, CONFIG["env"]["alphas"], 0.0,
                                CONFIG["env"]["n_steps"],
                                CONFIG["metric"]["train_episodes"])
    buckets, groups = build_buckets(episodes)

    spaces = {}
    for seed in seeds:
        rng   = make_rng(seed)
        model = fit_metric(
            emb, lambda b: sample_pairs(buckets, groups, rng, b),
            d_out=CONFIG["metric"]["dim_out"], lr=CONFIG["metric"]["lr"],
            epochs=CONFIG["metric"]["epochs"], batch_size=CONFIG["metric"]["batch_size"],
            margin=CONFIG["metric"]["margin"], neg_weight=CONFIG["metric"]["neg_weight"],
        )
        spaces[seed] = {"orig": emb, "proj": project(emb, projection_matrix(model))}
        print(f"  сид {seed}: метрика обучена")
    return spaces


def run_one(episode: List, name: str, pools: Dict[str, np.ndarray],
            meta: np.ndarray, seed: int, budget: int, k: int) -> Dict:
    """
    Один прогон одной конфигурации при заданном бюджете.
    ---

    Args:
        episode (List): Поток шагов, общий для всех конфигураций этого сида.
        name (str): Ключ из CONFIGS.
        pools (Dict): Пространства извлечения этого сида.
        meta (np.ndarray): Скрытая разметка; уходит только в Oracle.
        seed (int): Сид прогона.
        budget (int): Бюджет памяти.
        k (int): Сколько извлекать.

    Returns:
        Dict: Строка результатов.
    """
    cls, space, _ = CONFIGS[name]
    pool = pools[space]

    policy = Oracle(budget, pool, meta) if cls is Oracle else cls(budget, pool)
    roll   = rollout_loop(episode, policy, pool, k)
    pred   = predict(roll, make_rng(seed + READOUT_SEED_OFFSET))
    accuracy, accuracy_last_third = evaluate_accuracy(roll, pred)

    return {
        "policy":           name,
        "budget":           budget,
        "per_cell":         budget / (CONFIG["env"]["n_projects"] * CONFIG["env"]["n_situations"]),
        "accuracy":         accuracy,
        "acc_last_third":   accuracy_last_third,
        "precision":        evaluate_precision(roll),
        "n_retrieved_mean": float(roll.n_retrieved.mean()),
    }


def main() -> None:
    """
    Свип по бюджету, таблицы, вклад вытеснения и график.
    ---
    """
    texts, meta, emb, index = load_pool(CONFIG["pool"]["dir"])

    alpha   = CONFIG["hypothesis"]["alpha"]        # 0.0 — там преимущество максимально
    seeds   = CONFIG["seeds"]
    budgets = CONFIG["memory"]["budget_sweep"]
    eta     = CONFIG["env"]["eta"]
    n_steps = CONFIG["env"]["n_steps"]
    k       = CONFIG["memory"]["k"]

    print(f"=== ОБУЧЕНИЕ МЕТРИК (alpha прогона = {alpha}) ===")
    spaces = build_spaces(emb, index, seeds)

    print("\n=== ПРОГОН ===")
    total = len(budgets) * len(seeds) * len(CONFIGS)
    rows  = []
    for budget in budgets:
        for seed in seeds:
            episode = make_episode(alpha, eta, n_steps, index, make_rng(seed))
            for name in CONFIGS:
                row = run_one(episode, name, spaces[seed], meta, seed, budget, k)
                row.update(alpha=alpha, seed=seed, eta=eta, k=k)
                rows.append(row)
        print(f"  бюджет={budget:>4}  записей на клетку={budget / 48:.1f}  "
              f"готово {len(rows)}/{total}")

    df = pd.DataFrame(rows)
    out_csv = Path(CONFIG["output"]["main_csv"]).with_name("budget_sweep.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    summary = (
        df.groupby(["budget", "policy"], sort=False)
          .agg(accuracy_mean=("accuracy", "mean"),
               accuracy_std=("accuracy", "std"),
               precision_mean=("precision", "mean"),
               n_retrieved_mean=("n_retrieved_mean", "mean"))
          .reset_index()
    )
    summary.to_csv(out_csv.with_name("budget_sweep_summary.csv"), index=False)
    print(f"\nзаписано: {out_csv}  ({len(df)} строк)")
    print(f"записано: {out_csv.with_name('budget_sweep_summary.csv')}\n")

    for budget in budgets:
        print(f"бюджет = {budget}   ({budget / 48:.1f} записей на клетку "
              f"(проект, ситуация))")
        part = summary[summary["budget"] == budget]
        for _, r in part.iterrows():
            print(f"  {CONFIGS[r['policy']][2]:20} {r['accuracy_mean']:.3f} "
                  f"+- {r['accuracy_std']:.3f}   извлечено={r['n_retrieved_mean']:.1f}")
        print()

    verdict(summary, budgets)
    visualize(summary, budgets, Path(CONFIG["output"]["figures_dir"]))


def verdict(summary: pd.DataFrame, budgets: List[int]) -> None:
    """
    Проверка предсказания: убывает ли вклад вытеснения с ростом бюджета.
    ---

    Args:
        summary (pd.DataFrame): Сводная таблица.
        budgets (List[int]): Значения бюджета по возрастанию.
    """
    def get(budget, policy, col="accuracy_mean"):
        row = summary[(summary["budget"] == budget) & (summary["policy"] == policy)]
        return float(row[col].iloc[0])

    print("=" * 78)
    print("ВКЛАД ВЫТЕСНЕНИЯ ПРОТИВ БЮДЖЕТА")
    print("=" * 78)
    print(f"{'бюджет':>8} {'на клетку':>11} {'в исходном':>13} {'в метрике':>12}")
    contributions = {label: [] for _, _, label, _ in EVICT_PAIRS}
    for budget in budgets:
        line = f"{budget:8} {budget / 48:11.1f}"
        for a, b, label, _ in EVICT_PAIRS:
            delta = get(budget, a) - get(budget, b)
            contributions[label].append(delta)
            line += f" {delta:+13.3f}" if label.startswith("в исходном") else f" {delta:+12.3f}"
        print(line)

    print()
    for label, values in contributions.items():
        trend = values[0] - values[-1]
        print(f"  {label:26} {values[0]:+.3f} -> {values[-1]:+.3f}   "
              f"убывание {trend:+.3f}   "
              f"{'предсказание подтверждается' if trend > 0.01 else 'предсказание НЕ подтверждается'}")
    print("=" * 78 + "\n")


def _style(ax) -> None:
    """Общее оформление осей."""
    ax.grid(axis="y", color=GRIDC, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=9)


def _spread(values: np.ndarray, min_gap: float) -> np.ndarray:
    """Раздвигает подписи по вертикали."""
    out   = np.array(values, dtype=float)
    order = np.argsort(out)
    for i in range(1, len(order)):
        prev, cur = order[i - 1], order[i]
        if out[cur] - out[prev] < min_gap:
            out[cur] = out[prev] + min_gap
    return out


def visualize(summary: pd.DataFrame, budgets: List[int], figures_dir: Path) -> None:
    """
    Две панели: точность против бюджета и вклад вытеснения против бюджета.
    ---

    Args:
        summary (pd.DataFrame): Сводная таблица.
        budgets (List[int]): Значения бюджета.
        figures_dir (Path): Куда сохранять.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)

    def series(policy, col="accuracy_mean"):
        part = summary[summary["policy"] == policy].set_index("budget")
        return np.array([part.loc[b, col] for b in budgets], dtype=float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4))

    floor, ceil = series("nomemory"), series("oracle")
    ax1.fill_between(budgets, floor, ceil, color=REF_COLOR, alpha=0.08, lw=0)
    for values, name in ((ceil, "oracle"), (floor, "nomemory")):
        ax1.plot(budgets, values, ls="--", lw=1.6, color=REF_COLOR, zorder=2)
        ax1.annotate(CONFIGS[name][2], xy=(budgets[-1], values[-1]), xytext=(6, 0),
                     textcoords="offset points", va="center", fontsize=8.5, color=INK_2)

    ends = []
    for policy, color in SERIES_COLORS.items():
        values = series(policy)
        ax1.errorbar(budgets, values, yerr=series(policy, "accuracy_std"),
                     color=color, lw=2.0, marker="o", ms=6.5, capsize=3,
                     elinewidth=1.2, zorder=3, label=CONFIGS[policy][2])
        ends.append((values[-1], policy))

    for (_, policy), y_lab in zip(ends, _spread(np.array([e[0] for e in ends]), 0.03)):
        ax1.annotate(CONFIGS[policy][2], xy=(budgets[-1], y_lab), xytext=(6, 0),
                     textcoords="offset points", va="center", fontsize=8.5, color=INK_2)

    ax1.set_xscale("log", base=2)
    ax1.set_xticks(budgets)
    ax1.set_xticklabels([str(b) for b in budgets])
    ax1.set_xlim(budgets[0] * 0.9, budgets[-1] * 1.9)
    ax1.set_ylim(0.25, 1.02)
    ax1.set_xlabel("бюджет памяти, слотов", fontsize=10, color=INK_2)
    ax1.set_ylabel("доля верных действий", fontsize=10, color=INK_2)
    ax1.set_title(r"Точность против бюджета  ($\alpha$ = 0)", fontsize=12, color=INK, pad=10)
    _style(ax1)

    ax2.axhline(0, color=AXIS, lw=1.2, zorder=1)
    for a, b, label, color in EVICT_PAIRS:
        delta = series(a) - series(b)
        ax2.plot(budgets, delta, color=color, lw=2.0, marker="o", ms=6.5,
                 zorder=3, label=label)
        ax2.annotate(label, xy=(budgets[-1], delta[-1]), xytext=(6, 0),
                     textcoords="offset points", va="center", fontsize=8.5, color=INK_2)

    ax2.set_xscale("log", base=2)
    ax2.set_xticks(budgets)
    ax2.set_xticklabels([f"{b}\n{b / 48:.1f} на клетку" for b in budgets])
    ax2.set_xlim(budgets[0] * 0.9, budgets[-1] * 2.6)
    ax2.set_xlabel("бюджет памяти, слотов", fontsize=10, color=INK_2)
    ax2.set_ylabel("прирост от решенческого вытеснения", fontsize=10, color=INK_2)
    ax2.set_title("Вклад вытеснения: предсказание — убывает с бюджетом",
                  fontsize=12, color=INK, pad=10)
    _style(ax2)

    fig.suptitle(r"Свип по бюджету  (5 сидов, усы — ±1$\sigma$)",
                 fontsize=13, color=INK, y=1.0)
    fig.tight_layout()
    fig.savefig(figures_dir / "budget_sweep.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"записано: {figures_dir / 'budget_sweep.png'}\n")


if __name__ == "__main__":
    main()
