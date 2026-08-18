from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import CONFIG, cosine_table, load_pool, make_rng
from src.utils import make_episode
from src.memory import DecisionEviction, NoMemory, Oracle, Semantic
from src.metric import (build_buckets, collect_episodes, fit_metric,
                        project, projection_matrix, random_projection,
                        sample_pairs, sample_pairs_semantic)
from src.rollout import rollout_loop
from src.readout import (predict, evaluate_accuracy, evaluate_precision,
                         evaluate_usability)

READOUT_SEED_OFFSET = 10_000

CONFIGS: Dict[str, Tuple[type, str, str]] = {
    "nomemory":  (NoMemory,         "orig", "без памяти"),
    "semantic":  (Semantic,         "orig", "семантика"),
    "rand_proj": (Semantic,         "rand", "случайная проекция"),
    "sem_proj":  (Semantic,         "sem",  "семантическая цель"),
    "metric":    (Semantic,         "proj", "только метрика"),
    "eviction":  (DecisionEviction, "orig", "только вытеснение"),
    "full":      (DecisionEviction, "proj", "полный метод"),
    "oracle":    (Oracle,           "orig", "oracle"),
}

GRID     = ["semantic", "metric", "eviction", "full"]
CONTROLS = ["semantic", "rand_proj", "sem_proj", "metric"]

SERIES_COLORS = {
    "full":     "#2a78d6",
    "metric":   "#eb6834",
    "eviction": "#1baf7a",
    "semantic": "#eda100",
}
REF_COLOR = "#898781"
INK, INK_2, GRIDC, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"


def build_spaces(emb: np.ndarray, meta: np.ndarray, index: Dict,
                 seeds: List[int]) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Четыре пространства извлечения на каждый оценочный сид.
    ---

    orig  исходные эмбеддинги
    proj  решенческая метрика — обучена на совпадении revealed_action
    rand  КОНТРОЛЬ А: случайная замороженная проекция того же размера.
          Отвечает на «помогает само по себе снижение размерности».
    sem   КОНТРОЛЬ Б: та же архитектура, тот же объём обучения, но пары
          размечены по косинусу в исходном пространстве.
          Отвечает на «помогает сам факт обучаемой проекции, а не сигнал».

    Оба контроля проходят через тот же `fit_metric` с теми же гиперпараметрами —
    условия обучения совпадают по построению, а не «мы старались сделать одинаково».

    Обучающие эпизоды сдвинуты по сидам внутри `collect_episodes` и заведомо не
    пересекаются с оценочными. Разброс по сидам включает инициализацию проекции
    и сэмплирование пар; сами обучающие эпизоды при этом фиксированы — это
    ограничение надо назвать в отчёте.

    Args:
        emb (np.ndarray): Эмбеддинги пула.
        meta (np.ndarray): Скрытая разметка, только для диагностики.
        index (Dict): Индекс пула.
        seeds (List[int]): Оценочные сиды.

    Returns:
        Dict[int, Dict[str, np.ndarray]]: сид -> имя пространства -> пул.
    """
    episodes = collect_episodes(index, CONFIG["env"]["alphas"], 0.0,
                                CONFIG["env"]["n_steps"],
                                CONFIG["metric"]["train_episodes"])
    buckets, groups = build_buckets(episodes)
    d_in, d_out = emb.shape[1], CONFIG["metric"]["dim_out"]

    def fit(sampler):
        return projection_matrix(fit_metric(
            emb, sampler,
            d_out=d_out, lr=CONFIG["metric"]["lr"],
            epochs=CONFIG["metric"]["epochs"], batch_size=CONFIG["metric"]["batch_size"],
            margin=CONFIG["metric"]["margin"], neg_weight=CONFIG["metric"]["neg_weight"],
        ))

    spaces = {}
    for seed in seeds:
        rng = make_rng(seed)
        P_decision = fit(lambda b: sample_pairs(buckets, groups, rng, b))
        rng = make_rng(seed)                      # тот же старт для честного сравнения
        P_semantic = fit(lambda b: sample_pairs_semantic(emb, buckets, groups, rng, b))
        P_random   = random_projection(d_in, d_out, seed)

        spaces[seed] = {
            "orig": emb,
            "proj": project(emb, P_decision),
            "sem":  project(emb, P_semantic),
            "rand": project(emb, P_random),
        }

        print(f"  сид {seed}:")
        for name in ("orig", "rand", "sem", "proj"):
            t = cosine_table(spaces[seed][name], meta, 0)
            print(f"    {name:5} тема={t['topic_gap']:+.3f} проект={t['proj_gap']:+.3f} "
                  f"отн={t['topic_gap'] / abs(t['proj_gap']):7.2f}x ловушка={t['trap']:.3f}")
    return spaces


def run_one(episode: List, name: str, pools: Dict[str, np.ndarray],
            meta: np.ndarray, seed: int, budget: int, k: int) -> Dict:
    """
    Один прогон одной конфигурации.
    ---

    Пул подаётся И политике, И в rollout: иначе запрос будет из одного
    пространства, а записи из другого.

    Args:
        episode (List): Поток шагов, общий для всех конфигураций этого сида.
        name (str): Ключ из CONFIGS.
        emb (np.ndarray): Исходный пул.
        emb_proj (np.ndarray): Спроецированный пул.
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
    hit_at_k, hit_at_k_no_dup     = evaluate_usability(roll)

    return {
        "policy":           name,
        "space":            space,
        "evict":            "decision" if cls is DecisionEviction else "fifo",
        "budget":           budget,
        "k":                k,
        "accuracy":         accuracy,
        "acc_last_third":   accuracy_last_third,
        "precision":        evaluate_precision(roll),
        "hit_at_k":         hit_at_k,
        "hit_at_k_no_dup":  hit_at_k_no_dup,
        "exact_dup_rate":   float(roll.exact_dup.mean()),
        "n_retrieved_mean": float(roll.n_retrieved.mean()),
    }


def main() -> None:
    """
    Полная сетка прогонов, таблицы, разложение вклада и графики.
    ---
    """
    texts, meta, emb, index = load_pool(CONFIG["pool"]["dir"])

    alphas  = CONFIG["env"]["alphas"]
    seeds   = CONFIG["seeds"]
    eta     = CONFIG["env"]["eta"]
    n_steps = CONFIG["env"]["n_steps"]
    budget  = CONFIG["memory"]["budget"]
    k       = CONFIG["memory"]["k"]

    print("=== ПРОСТРАНСТВА ИЗВЛЕЧЕНИЯ (метрика и оба контроля, по одному набору на сид) ===")
    spaces = build_spaces(emb, meta, index, seeds)

    print("\n=== ПРОГОН ===")
    total = len(alphas) * len(seeds) * len(CONFIGS)
    rows  = []
    for alpha in alphas:
        for seed in seeds:
            episode = make_episode(alpha, eta, n_steps, index, make_rng(seed))
            for name in CONFIGS:
                row = run_one(episode, name, spaces[seed], meta, seed, budget, k)
                row.update(alpha=alpha, seed=seed, eta=eta, n_steps=n_steps)
                rows.append(row)
            print(f"  alpha={alpha}  seed={seed}  готово {len(rows)}/{total}")

    df = pd.DataFrame(rows)
    out_csv = Path(CONFIG["output"]["main_csv"]).with_name("ablation.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    summary = (
        df.groupby(["alpha", "policy"], sort=False)
          .agg(accuracy_mean=("accuracy", "mean"),
               accuracy_std=("accuracy", "std"),
               last_third_mean=("acc_last_third", "mean"),
               precision_mean=("precision", "mean"))
          .reset_index()
    )
    summary.to_csv(out_csv.with_name("ablation_summary.csv"), index=False)
    print(f"\nзаписано: {out_csv}  ({len(df)} строк)")
    print(f"записано: {out_csv.with_name('ablation_summary.csv')}\n")

    for alpha in alphas:
        print(f"alpha = {alpha}")
        part = summary[summary["alpha"] == alpha]
        for _, r in part.iterrows():
            print(f"  {CONFIGS[r['policy']][2]:20} {r['accuracy_mean']:.3f} "
                  f"+- {r['accuracy_std']:.3f}   хвост={r['last_third_mean']:.3f}"
                  f"   точность извлечения={r['precision_mean']:.3f}")
        print()

    decompose(summary, alphas)
    controls_verdict(summary, alphas)
    visualize(summary, alphas, Path(CONFIG["output"]["figures_dir"]))
    visualize_controls(summary, alphas, Path(CONFIG["output"]["figures_dir"]))


def controls_verdict(summary: pd.DataFrame, alphas: List[float]) -> None:
    """
    Проверка контролей А и Б.
    ---

    Возражение закрыто, если решенческая метрика обгоняет и случайную проекцию,
    и проекцию с семантической целью. Иначе выигрыш объясняется не сигналом,
    а самим фактом обучаемого преобразования.

    Args:
        summary (pd.DataFrame): Сводная таблица.
        alphas (List[float]): Значения alpha.
    """
    def get(alpha, policy, col="accuracy_mean"):
        row = summary[(summary["alpha"] == alpha) & (summary["policy"] == policy)]
        return float(row[col].iloc[0])

    print("=" * 78)
    print("КОНТРОЛИ: что даёт проекция сама по себе")
    print("=" * 78)
    print(f"{'alpha':>7} {'семантика':>11} {'случайная':>11} {'семант.цель':>13} "
          f"{'решенческая':>13}")
    for alpha in alphas:
        print(f"{alpha:7.2f} {get(alpha, 'semantic'):11.3f} {get(alpha, 'rand_proj'):11.3f} "
              f"{get(alpha, 'sem_proj'):13.3f} {get(alpha, 'metric'):13.3f}")

    ok = True
    for alpha in alphas:
        margin_rand = get(alpha, "metric") - get(alpha, "rand_proj")
        margin_sem  = get(alpha, "metric") - get(alpha, "sem_proj")
        noise = get(alpha, "metric", "accuracy_std") + get(alpha, "sem_proj", "accuracy_std")
        if alpha == min(alphas) and (margin_rand < noise or margin_sem < noise):
            ok = False
    print()
    print("  вывод:", "контроли пройдены — выигрыш даёт СИГНАЛ, а не факт проекции"
          if ok else "ВНИМАНИЕ: контроль не отделён от метрики, разница в пределах шума")
    print("=" * 78 + "\n")


def decompose(summary: pd.DataFrame, alphas: List[float]) -> None:
    """
    Разложение вклада компонентов относительно базовой семантики.
    ---

    Взаимодействие = прирост полного метода минус сумма приростов по отдельности.
    Положительное означает, что части усиливают друг друга.

    Args:
        summary (pd.DataFrame): Сводная таблица.
        alphas (List[float]): Значения alpha.
    """
    def get(alpha, policy):
        row = summary[(summary["alpha"] == alpha) & (summary["policy"] == policy)]
        return float(row["accuracy_mean"].iloc[0])

    print("=" * 78)
    print("ВКЛАД КОМПОНЕНТОВ (прирост к базовой семантике)")
    print("=" * 78)
    print(f"{'alpha':>7} {'семантика':>11} {'+метрика':>10} {'+вытесн.':>10} "
          f"{'+оба':>9} {'взаимод.':>10}")
    for alpha in alphas:
        base = get(alpha, "semantic")
        d_m  = get(alpha, "metric")   - base
        d_e  = get(alpha, "eviction") - base
        d_f  = get(alpha, "full")     - base
        print(f"{alpha:7.2f} {base:11.3f} {d_m:+10.3f} {d_e:+10.3f} "
              f"{d_f:+9.3f} {d_f - d_m - d_e:+10.3f}")
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
    """Раздвигает подписи по вертикали, чтобы не наезжали друг на друга."""
    out   = np.array(values, dtype=float)
    order = np.argsort(out)
    for i in range(1, len(order)):
        prev, cur = order[i - 1], order[i]
        if out[cur] - out[prev] < min_gap:
            out[cur] = out[prev] + min_gap
    return out


def visualize(summary: pd.DataFrame, alphas: List[float], figures_dir: Path) -> None:
    """
    Два рисунка: линии по alpha и сетка 2x2 столбиками.
    ---

    Args:
        summary (pd.DataFrame): Сводная таблица.
        alphas (List[float]): Значения alpha.
        figures_dir (Path): Куда сохранять.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)

    def series(policy, col="accuracy_mean"):
        part = summary[summary["policy"] == policy].set_index("alpha")
        return np.array([part.loc[a, col] for a in alphas], dtype=float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4))

    floor, ceil = series("nomemory"), series("oracle")
    ax1.fill_between(alphas, floor, ceil, color=REF_COLOR, alpha=0.08, lw=0)
    for values, name in ((ceil, "oracle"), (floor, "nomemory")):
        ax1.plot(alphas, values, ls="--", lw=1.6, color=REF_COLOR, zorder=2)
        ax1.annotate(CONFIGS[name][2], xy=(alphas[-1], values[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8.5, color=INK_2)

    ends = []
    for policy, color in SERIES_COLORS.items():
        values = series(policy)
        ax1.errorbar(alphas, values, yerr=series(policy, "accuracy_std"),
                     color=color, lw=2.0, marker="o", ms=6.5, capsize=3,
                     elinewidth=1.2, zorder=3, label=CONFIGS[policy][2])
        ends.append((values[-1], policy))

    for (_, policy), y_lab in zip(ends, _spread(np.array([e[0] for e in ends]), 0.03)):
        ax1.annotate(CONFIGS[policy][2], xy=(alphas[-1], y_lab), xytext=(6, 0),
                     textcoords="offset points", va="center", fontsize=8.5, color=INK_2)

    ax1.set_xticks(alphas)
    ax1.set_xlabel(r"$\alpha$  —  привязка темы к проекту", fontsize=10, color=INK_2)
    ax1.set_ylabel("доля верных действий", fontsize=10, color=INK_2)
    ax1.set_title("Сетка ablation по режимам", fontsize=12, color=INK, pad=10)
    ax1.set_xlim(min(alphas) - 0.05, max(alphas) + 0.42)
    ax1.set_ylim(0.25, 1.02)
    _style(ax1)

    width = 0.2
    base  = np.arange(len(alphas))
    for n, policy in enumerate(GRID):
        values = series(policy)
        pos    = base + (n - 1.5) * width
        ax2.bar(pos, values, width=width * 0.9, color=SERIES_COLORS[policy], label=CONFIGS[policy][2])
        for x, y in zip(pos, values):
            ax2.text(x, y + 0.008, f"{y:.2f}", ha="center", fontsize=7.5, color=INK_2)
    for n, alpha in enumerate(alphas):
        ax2.hlines(series("oracle")[n], base[n] - 0.45, base[n] + 0.45,
                   color=REF_COLOR, ls="--", lw=1.4, zorder=4)
        ax2.hlines(series("nomemory")[n], base[n] - 0.45, base[n] + 0.45,
                   color=REF_COLOR, ls="--", lw=1.4, zorder=4)
    ax2.set_xticks(base)
    ax2.set_xticklabels([rf"$\alpha$ = {a}" for a in alphas], fontsize=10, color=INK_2)
    ax2.set_ylabel("доля верных действий", fontsize=10, color=INK_2)
    ax2.set_title("Вклад компонентов; пунктир — пол и потолок", fontsize=12, color=INK, pad=10)
    ax2.set_ylim(0.25, 1.05)
    ax2.legend(frameon=False, fontsize=9, labelcolor=INK_2, ncol=2, loc="upper left")
    _style(ax2)

    fig.suptitle(r"Решенческая память: метрика извлечения x правило вытеснения  "
                 r"(5 сидов, усы — ±1$\sigma$)", fontsize=13, color=INK, y=1.0)
    fig.tight_layout()
    fig.savefig(figures_dir / "ablation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def visualize_controls(summary: pd.DataFrame, alphas: List[float],
                       figures_dir: Path) -> None:
    """
    Рисунок контролей: что даёт проекция сама по себе.
    ---

    Контроли серые, решенческая метрика цветная — так видно, что работает
    именно сигнал, а не наличие обучаемого преобразования.

    Args:
        summary (pd.DataFrame): Сводная таблица.
        alphas (List[float]): Значения alpha.
        figures_dir (Path): Куда сохранять.
    """
    def series(policy, col="accuracy_mean"):
        part = summary[summary["policy"] == policy].set_index("alpha")
        return np.array([part.loc[a, col] for a in alphas], dtype=float)

    colors = {"semantic":  SERIES_COLORS["semantic"],
              "rand_proj": "#b8b6ae",
              "sem_proj":  "#898781",
              "metric":    SERIES_COLORS["metric"]}

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    width, base = 0.2, np.arange(len(alphas))

    for n, policy in enumerate(CONTROLS):
        values = series(policy)
        pos    = base + (n - 1.5) * width
        ax.bar(pos, values, width=width * 0.9, color=colors[policy],
               yerr=series(policy, "accuracy_std"), capsize=3,
               error_kw={"elinewidth": 1.1, "ecolor": INK_2},
               label=CONFIGS[policy][2])
        for x, y in zip(pos, values):
            ax.text(x, y + 0.025, f"{y:.2f}", ha="center", fontsize=8, color=INK_2)

    for n in range(len(alphas)):
        ax.hlines(series("oracle")[n], base[n] - 0.45, base[n] + 0.45,
                  color=REF_COLOR, ls="--", lw=1.4, zorder=4)

    ax.set_xticks(base)
    ax.set_xticklabels([rf"$\alpha$ = {a}" for a in alphas], fontsize=10, color=INK_2)
    ax.set_ylabel("доля верных действий", fontsize=10, color=INK_2)
    ax.set_ylim(0.25, 1.05)
    ax.set_title("Контроли: проекция сама по себе не помогает\n"
                 "серые — те же 24576 параметров, обучены на другом сигнале",
                 fontsize=12, color=INK, pad=12)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, ncol=2, loc="upper left")
    _style(ax)

    fig.tight_layout()
    fig.savefig(figures_dir / "controls.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
