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
    "nomemory":     (NoMemory,         "orig",  "без памяти"),
    "semantic":     (Semantic,         "orig",  "семантика"),
    "metric_clean": (Semantic,         "clean", "метрика, обучена без шума"),
    "metric_noisy": (Semantic,         "noisy", "метрика, обучена при этом eta"),
    "full_noisy":   (DecisionEviction, "noisy", "полный метод"),
    "oracle":       (Oracle,           "orig",  "oracle"),
}

SERIES_COLORS = {
    "metric_clean": "#2a78d6",
    "metric_noisy": "#eb6834",
    "full_noisy":   "#1baf7a",
    "semantic":     "#eda100",
}
REF_COLOR = "#898781"
INK, INK_2, GRIDC, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"


def measure_purity(buckets: Dict, groups: Dict, meta: np.ndarray,
                   rng, n_pairs: int = 4000) -> Tuple[float, float]:
    """
    ДИАГНОСТИКА: чистота обучающего сигнала.
    ---

    Использует скрытое поле project намеренно и только здесь — метрика его не
    видит. Показывает, насколько «совпало действие» остаётся заменителем
    ненаблюдаемого «тот же проект» при данном уровне шума.

    Args:
        buckets (Dict): Из `build_buckets`.
        groups (Dict): Из `build_buckets`.
        meta (np.ndarray): Скрытая разметка пула.
        rng: Numpy генератор случайных чисел.
        n_pairs (int): Сколько пар просмотреть.

    Returns:
        Tuple[float, float]: доля «тот же проект» среди положительных и среди отрицательных.
    """
    i, j, t = sample_pairs(buckets, groups, rng, n_pairs)
    same_project = (meta[i, 0] == meta[j, 0])
    return float(same_project[t > 0].mean()), float(same_project[t < 0].mean())


def build_metric(emb: np.ndarray, index: Dict, eta: float, seed: int) -> np.ndarray:
    """
    Обучение одной метрики при заданном уровне шума.
    ---

    Args:
        emb (np.ndarray): Эмбеддинги пула.
        index (Dict): Индекс пула.
        eta (float): Шум в ОБУЧАЮЩИХ эпизодах.
        seed (int): Сид.

    Returns:
        np.ndarray: Матрица проекции (d_in, d_out).
    """
    episodes = collect_episodes(index, CONFIG["env"]["alphas"], eta,
                                CONFIG["env"]["n_steps"],
                                CONFIG["metric"]["train_episodes"])
    buckets, groups = build_buckets(episodes)
    rng   = make_rng(seed)
    model = fit_metric(
        emb, lambda b: sample_pairs(buckets, groups, rng, b),
        d_out=CONFIG["metric"]["dim_out"], lr=CONFIG["metric"]["lr"],
        epochs=CONFIG["metric"]["epochs"], batch_size=CONFIG["metric"]["batch_size"],
        margin=CONFIG["metric"]["margin"], neg_weight=CONFIG["metric"]["neg_weight"],
    )
    return projection_matrix(model)


def run_one(episode: List, name: str, pools: Dict[str, np.ndarray],
            meta: np.ndarray, seed: int, budget: int, k: int) -> Dict:
    """
    Один прогон одной конфигурации.
    ---

    Args:
        episode (List): Поток шагов, общий для всех конфигураций этого сида.
        name (str): Ключ из CONFIGS.
        pools (Dict): Пространства извлечения.
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
        "policy":         name,
        "accuracy":       accuracy,
        "acc_last_third": accuracy_last_third,
        "precision":      evaluate_precision(roll),
    }


def main() -> None:
    """
    Свип по шуму, диагностика сигнала, проверка гипотезы и графики.
    ---
    """
    texts, meta, emb, index = load_pool(CONFIG["pool"]["dir"])

    alpha   = CONFIG["hypothesis"]["alpha"]
    etas    = CONFIG["hypothesis"]["etas"]
    seeds   = CONFIG["seeds"]
    n_steps = CONFIG["env"]["n_steps"]
    budget  = CONFIG["memory"]["budget"]
    k       = CONFIG["memory"]["k"]
    M       = CONFIG["env"]["n_candidates"]
    P       = CONFIG["env"]["n_projects"]

    print(f"=== ГИПОТЕЗА: свип по eta при alpha = {alpha} ===")
    print(f"    сигнал должен вырождаться при eta = (M-1)/M = {(M - 1) / M:.3f}\n")

    print("=== ЧИСТОТА ОБУЧАЮЩЕГО СИГНАЛА (измерение против теории) ===")
    print(f"{'eta':>6} {'полож. изм.':>13} {'отриц. изм.':>13} "
          f"{'теория P(SP|SA)':>17} {'прирост':>9}")
    purity = {}
    for eta in etas:
        episodes = collect_episodes(index, CONFIG["env"]["alphas"], eta,
                                    n_steps, CONFIG["metric"]["train_episodes"])
        buckets, groups = build_buckets(episodes)
        pos, neg = measure_purity(buckets, groups, meta, make_rng(0))
        q        = (1 - eta) ** 2 + eta ** 2 / (M - 1)
        theory   = q / (q + (P - 1) / M)
        purity[eta] = (pos, neg, theory)
        print(f"{eta:6.2f} {pos:13.3f} {neg:13.3f} {theory:17.3f} "
              f"{pos / (1 / P):8.2f}x")

    print("\n=== ОБУЧЕНИЕ МЕТРИК ===")
    clean = {seed: build_metric(emb, index, 0.0, seed) for seed in seeds}
    print(f"  чистых метрик: {len(clean)}")
    noisy = {(eta, seed): build_metric(emb, index, eta, seed)
             for eta in etas for seed in seeds}
    print(f"  зашумлённых метрик: {len(noisy)}")

    print("\n=== ПРОГОН ===")
    total = len(etas) * len(seeds) * len(CONFIGS)
    rows  = []
    for eta in etas:
        for seed in seeds:
            episode = make_episode(alpha, eta, n_steps, index, make_rng(seed))
            pools = {"orig":  emb,
                     "clean": project(emb, clean[seed]),
                     "noisy": project(emb, noisy[(eta, seed)])}
            for name in CONFIGS:
                row = run_one(episode, name, pools, meta, seed, budget, k)
                row.update(alpha=alpha, eta=eta, seed=seed, budget=budget, k=k,
                           purity_pos=purity[eta][0], purity_neg=purity[eta][1])
                rows.append(row)
        print(f"  eta={eta}  готово {len(rows)}/{total}")

    df = pd.DataFrame(rows)
    out_csv = Path(CONFIG["output"]["hypothesis_csv"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    summary = (
        df.groupby(["eta", "policy"], sort=False)
          .agg(accuracy_mean=("accuracy", "mean"),
               accuracy_std=("accuracy", "std"),
               precision_mean=("precision", "mean"))
          .reset_index()
    )
    summary.to_csv(out_csv.with_name("hypothesis_summary.csv"), index=False)
    print(f"\nзаписано: {out_csv}  ({len(df)} строк)")
    print(f"записано: {out_csv.with_name('hypothesis_summary.csv')}\n")

    for eta in etas:
        print(f"eta = {eta}")
        part = summary[summary["eta"] == eta]
        for _, r in part.iterrows():
            print(f"  {CONFIGS[r['policy']][2]:28} {r['accuracy_mean']:.3f} "
                  f"+- {r['accuracy_std']:.3f}")
        print()

    verdict(summary, etas)
    visualize(summary, purity, etas, Path(CONFIG["output"]["figures_dir"]))


def verdict(summary: pd.DataFrame, etas: List[float]) -> None:
    """
    Проверка гипотезы: убывает ли преимущество с ростом шума.
    ---

    Args:
        summary (pd.DataFrame): Сводная таблица.
        etas (List[float]): Значения eta по возрастанию.
    """
    def get(eta, policy, col="accuracy_mean"):
        row = summary[(summary["eta"] == eta) & (summary["policy"] == policy)]
        return float(row[col].iloc[0])

    print("=" * 82)
    print("ПРОВЕРКА ГИПОТЕЗЫ: преимущество над семантикой против шума")
    print("=" * 82)
    print(f"{'eta':>6} {'семантика':>11} {'метрика чист.':>15} {'метрика шум.':>14} "
          f"{'преим. чист.':>14} {'преим. шум.':>13}")
    adv_clean, adv_noisy = [], []
    for eta in etas:
        base = get(eta, "semantic")
        c    = get(eta, "metric_clean") - base
        n    = get(eta, "metric_noisy") - base
        adv_clean.append(c)
        adv_noisy.append(n)
        print(f"{eta:6.2f} {base:11.3f} {get(eta, 'metric_clean'):15.3f} "
              f"{get(eta, 'metric_noisy'):14.3f} {c:+14.3f} {n:+13.3f}")

    print()
    for label, values in (("обучена без шума", adv_clean), ("обучена при шуме", adv_noisy)):
        drop = values[0] - values[-1]
        print(f"  {label:20} {values[0]:+.3f} -> {values[-1]:+.3f}   "
              f"убывание {drop:+.3f}   "
              f"{'ГИПОТЕЗА ПОДТВЕРЖДАЕТСЯ' if drop > 0.02 else 'не подтверждается'}")

    gap = [n - c for n, c in zip(adv_noisy, adv_clean)]
    print(f"\n  вклад порчи ОБУЧАЮЩЕГО сигнала (шум. минус чист.):")
    print("   ", "  ".join(f"eta={e}: {g:+.3f}" for e, g in zip(etas, gap)))
    print("=" * 82 + "\n")


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


def visualize(summary: pd.DataFrame, purity: Dict, etas: List[float],
              figures_dir: Path) -> None:
    """
    Три панели: точность, преимущество и чистота сигнала.
    ---

    Читаются как цепочка: шум портит сигнал (справа), сигнал определяет
    преимущество (в центре), преимущество определяет точность (слева).

    Args:
        summary (pd.DataFrame): Сводная таблица.
        purity (Dict): eta -> (положительные, отрицательные, теория).
        etas (List[float]): Значения eta.
        figures_dir (Path): Куда сохранять.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)

    def series(policy, col="accuracy_mean"):
        part = summary[summary["policy"] == policy].set_index("eta")
        return np.array([part.loc[e, col] for e in etas], dtype=float)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5.2))

    floor, ceil = series("nomemory"), series("oracle")
    ax1.fill_between(etas, floor, ceil, color=REF_COLOR, alpha=0.08, lw=0)
    for values, name in ((ceil, "oracle"), (floor, "nomemory")):
        ax1.plot(etas, values, ls="--", lw=1.6, color=REF_COLOR, zorder=2)
        ax1.annotate(CONFIGS[name][2], xy=(etas[-1], values[-1]), xytext=(6, 0),
                     textcoords="offset points", va="center", fontsize=8.5, color=INK_2)

    ends = []
    for policy, color in SERIES_COLORS.items():
        values = series(policy)
        ax1.errorbar(etas, values, yerr=series(policy, "accuracy_std"),
                     color=color, lw=2.0, marker="o", ms=6.5, capsize=3,
                     elinewidth=1.2, zorder=3)
        ends.append((values[-1], policy))
    for (_, policy), y_lab in zip(ends, _spread(np.array([e[0] for e in ends]), 0.035)):
        ax1.annotate(CONFIGS[policy][2], xy=(etas[-1], y_lab), xytext=(6, 0),
                     textcoords="offset points", va="center", fontsize=8, color=INK_2)

    ax1.set_xticks(etas)
    ax1.set_xlim(min(etas) - 0.02, max(etas) + 0.24)
    ax1.set_ylim(0.25, 1.02)
    ax1.set_xlabel(r"$\eta$  —  доля лживой обратной связи", fontsize=10, color=INK_2)
    ax1.set_ylabel("доля верных действий", fontsize=10, color=INK_2)
    ax1.set_title("Точность против шума", fontsize=12, color=INK, pad=10)
    _style(ax1)

    ax2.axhline(0, color=AXIS, lw=1.2, zorder=1)
    for policy in ("metric_clean", "metric_noisy"):
        adv = series(policy) - series("semantic")
        ax2.plot(etas, adv, color=SERIES_COLORS[policy], lw=2.0, marker="o",
                 ms=6.5, zorder=3, label=CONFIGS[policy][2])
    ax2.set_xticks(etas)
    ax2.set_xlim(min(etas) - 0.02, max(etas) + 0.02)
    ax2.set_xlabel(r"$\eta$  —  доля лживой обратной связи", fontsize=10, color=INK_2)
    ax2.set_ylabel("преимущество над семантикой", fontsize=10, color=INK_2)
    ax2.set_title("Проверка гипотезы: преимущество убывает",
                  fontsize=12, color=INK, pad=10)
    ax2.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper right")
    _style(ax2)

    pos = [purity[e][0] for e in etas]
    neg = [purity[e][1] for e in etas]
    th  = [purity[e][2] for e in etas]
    ax3.plot(etas, pos, color="#2a78d6", lw=2.0, marker="o", ms=6.5,
             label="положительные пары, измерено")
    ax3.plot(etas, th, color="#2a78d6", lw=1.6, ls=":", marker="s", ms=5,
             alpha=0.8, label="они же, теория")
    ax3.plot(etas, [1 - n for n in neg], color="#eb6834", lw=2.0, marker="o", ms=6.5,
             label="отрицательные пары, чистота")
    ax3.axhline(1 / CONFIG["env"]["n_projects"], color=REF_COLOR, ls="--", lw=1.4)
    ax3.annotate("базовая ставка 1/8", xy=(etas[0], 1 / CONFIG["env"]["n_projects"]),
                 xytext=(0, 6), textcoords="offset points", fontsize=8.5, color=INK_2)
    ax3.set_xticks(etas)
    ax3.set_xlim(min(etas) - 0.02, max(etas) + 0.02)
    ax3.set_ylim(0, 1.08)
    ax3.set_xlabel(r"$\eta$  —  доля лживой обратной связи", fontsize=10, color=INK_2)
    ax3.set_ylabel("доля пар, отражающих истинный проект", fontsize=10, color=INK_2)
    ax3.set_title("Механизм: чистота обучающего сигнала",
                  fontsize=12, color=INK, pad=10)
    ax3.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="center left")
    _style(ax3)

    fig.suptitle(r"Собственная гипотеза: голодание решенческого сигнала при шуме  "
                 r"($\alpha$ = 0, 5 сидов, усы — ±1$\sigma$)",
                 fontsize=13, color=INK, y=1.0)
    fig.tight_layout()
    fig.savefig(figures_dir / "hypothesis.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"записано: {figures_dir / 'hypothesis.png'}\n")


if __name__ == "__main__":
    main()
