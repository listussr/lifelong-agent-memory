from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src import CONFIG, load_pool, make_rng
from src.utils import make_episode
from src.memory import NoMemory, Oracle, Semantic, make_policy
from src.metric import (build_buckets, collect_episodes, fit_metric,
                        project, projection_matrix, sample_pairs)
from src.head import (build_dataset, check_empty_memory, collect_head_episodes,
                      fit_head, predict_head)
from src.rollout import rollout_loop
from src.readout import predict, evaluate_accuracy, evaluate_usability

READOUT_SEED_OFFSET = 10_000

CONFIGS: Dict[str, Tuple[str, str, str]] = {
    "nomemory": ("nomemory", "orig", "без памяти"),
    "semantic": ("semantic", "orig", "семантика"),
    "metric":   ("semantic", "proj", "решенческая метрика"),
    "oracle":   ("oracle",   "orig", "oracle"),
}

VOTE_COLOR = "#eda100"
HEAD_COLOR = "#2a78d6"
REF_COLOR  = "#898781"
INK, INK_2, GRIDC, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"


def build_spaces(emb: np.ndarray, index: Dict, seeds: List[int]) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Исходный и решенческий пул на каждый сид.
    ---

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


def train_head(emb: np.ndarray, meta: np.ndarray, index: Dict, device: str):
    """
    Обучение одной головы на смеси политик.
    ---

    Args:
        emb (np.ndarray): Эмбеддинги пула.
        meta (np.ndarray): Скрытая разметка; уходит только в Oracle внутри смеси.
        index (Dict): Индекс пула.
        device (str): cpu или cuda.

    Returns:
        TransformerHead: Обученная голова.
    """
    head_cfg = CONFIG["head"]
    episodes = collect_head_episodes(index, CONFIG["env"]["alphas"], CONFIG["env"]["eta"],
                                     CONFIG["env"]["n_steps"], head_cfg["train_episodes"])
    data = build_dataset(episodes, emb, meta, CONFIG["memory"]["budget"],
                         CONFIG["memory"]["k"], head_cfg["mixture_policies"])
    print(f"  обучающих примеров: {len(data['correct']):,}")

    return fit_head(
        data, emb, make_rng(0),
        d_model=head_cfg["d_model"], n_layers=head_cfg["n_layers"],
        n_heads=head_cfg["n_heads"], ff_mult=head_cfg["ff_mult"],
        dropout=head_cfg["dropout"], norm_first=head_cfg["norm_first"],
        lr=head_cfg["lr"], epochs=head_cfg["epochs"],
        batch_size=head_cfg["batch_size"], device=device,
    )


def run_one(episode: List, name: str, pools: Dict[str, np.ndarray], emb: np.ndarray,
            meta: np.ndarray, head, seed: int, budget: int, k: int, device: str) -> Dict:
    """
    Один прогон: обе версии решающего правила на одном и том же трейсе.
    ---

    Args:
        episode (List): Поток шагов, общий для всех конфигураций этого сида.
        name (str): Ключ из CONFIGS.
        pools (Dict): Пространства извлечения этого сида.
        emb (np.ndarray): Исходные эмбеддинги - вход головы.
        meta (np.ndarray): Скрытая разметка; уходит только в Oracle.
        head: Обученная голова.
        seed (int): Сид прогона.
        budget (int): Бюджет памяти.
        k (int): Сколько извлекать.
        device (str): cpu или cuda.

    Returns:
        Dict: Строка результатов.
    """
    policy_name, space, _ = CONFIGS[name]
    pool = pools[space]

    policy = make_policy(policy_name, budget, pool, meta, make_rng(seed),
                         CONFIG["memory"]["kmeans_clusters"],
                         CONFIG["memory"]["kmeans_refit_every"], seed)
    roll = rollout_loop(episode, policy, pool, k)

    vote = predict(roll, make_rng(seed + READOUT_SEED_OFFSET))
    acc_vote, _ = evaluate_accuracy(roll, vote)

    head_pred = predict_head(head, roll, emb, device)
    acc_head  = float((head_pred == roll.correct).mean())

    hit_at_k, _ = evaluate_usability(roll)

    return {
        "policy":    name,
        "space":     space,
        "acc_vote":  acc_vote,
        "acc_head":  acc_head,
        "hit_at_k":  hit_at_k,
        "gain":      acc_head - acc_vote,
    }


def main() -> None:
    """
    Обучение головы, прогон обеих версий читателя, таблицы и график.
    ---
    """
    texts, meta, emb, index = load_pool(CONFIG["pool"]["dir"])

    alphas  = [min(CONFIG["env"]["alphas"]), max(CONFIG["env"]["alphas"])]
    seeds   = CONFIG["seeds"]
    eta     = CONFIG["env"]["eta"]
    n_steps = CONFIG["env"]["n_steps"]
    budget  = CONFIG["memory"]["budget"]
    k       = CONFIG["memory"]["k"]
    device  = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"=== ПРОСТРАНСТВА ИЗВЛЕЧЕНИЯ ===")
    spaces = build_spaces(emb, index, seeds)

    print(f"\n=== ОБУЧЕНИЕ ГОЛОВЫ (устройство: {device}) ===")
    head = train_head(emb, meta, index, device)
    print(f"  параметров: {sum(p.numel() for p in head.parameters()):,}")

    print("\n=== ПРОГОН ===")
    rows = []
    for alpha in alphas:
        for seed in seeds:
            episode = make_episode(alpha, eta, n_steps, index, make_rng(seed))
            for name in CONFIGS:
                row = run_one(episode, name, spaces[seed], emb, meta, head,
                              seed, budget, k, device)
                row.update(alpha=alpha, seed=seed, eta=eta, budget=budget, k=k)
                rows.append(row)
        print(f"  alpha={alpha}  готово {len(rows)}/{len(alphas) * len(seeds) * len(CONFIGS)}")

    df = pd.DataFrame(rows)
    out_csv = Path(CONFIG["output"]["main_csv"]).with_name("head.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    summary = (
        df.groupby(["alpha", "policy"], sort=False)
          .agg(vote_mean=("acc_vote", "mean"), vote_std=("acc_vote", "std"),
               head_mean=("acc_head", "mean"), head_std=("acc_head", "std"),
               hit_mean=("hit_at_k", "mean"), gain_mean=("gain", "mean"))
          .reset_index()
    )
    summary.to_csv(out_csv.with_name("head_summary.csv"), index=False)
    print(f"\nзаписано: {out_csv}  ({len(df)} строк)")
    print(f"записано: {out_csv.with_name('head_summary.csv')}\n")

    for alpha in alphas:
        print(f"alpha = {alpha}")
        print(f"  {'политика':22} {'голосование':>13} {'голова':>15} "
              f"{'прирост':>9} {'потолок':>9}")
        part = summary[summary["alpha"] == alpha]
        for _, r in part.iterrows():
            print(f"  {CONFIGS[r['policy']][2]:22} {r['vote_mean']:.3f} +-{r['vote_std']:.3f} "
                  f"  {r['head_mean']:.3f} +-{r['head_std']:.3f} "
                  f"{r['gain_mean']:+9.3f} {r['hit_mean']:9.3f}")
        print()

    print("=== ПРОВЕРКА НА УТЕЧКУ (пустая память, усреднение по сидам) ===")
    empties = []
    for seed in seeds:
        episode = make_episode(alphas[0], eta, n_steps, index, make_rng(seed))
        policy  = make_policy("semantic", budget, emb, meta, make_rng(seed), 8, 50, seed)
        roll    = rollout_loop(episode, policy, emb, k)
        empties.append(check_empty_memory(head, emb, roll, device))
    floor = 1.0 / CONFIG["env"]["n_candidates"]
    print(f"  пустая память: {np.mean(empties):.3f} +- {np.std(empties):.3f}   пол: {floor:.3f}")
    print(f"  утечка означала бы значение ЗАМЕТНО ВЫШЕ пола (детерминированный argmax "
          f"даёт разброс в обе стороны)")
    print(f"  вывод: {'утечки нет' if np.mean(empties) < floor + 0.06 else 'ВНИМАНИЕ, УТЕЧКА'}\n")

    verdict(summary, alphas)
    visualize(summary, alphas, Path(CONFIG["output"]["figures_dir"]))


def verdict(summary: pd.DataFrame, alphas: List[float]) -> None:
    """
    Классификация исхода по трём сценариям.
    ---

    Args:
        summary (pd.DataFrame): Сводная таблица.
        alphas (List[float]): Значения alpha.
    """
    def get(alpha, policy, col):
        row = summary[(summary["alpha"] == alpha) & (summary["policy"] == policy)]
        return float(row[col].iloc[0])

    lo = min(alphas)
    gap_vote = get(lo, "metric", "vote_mean") - get(lo, "semantic", "vote_mean")
    gap_head = get(lo, "metric", "head_mean") - get(lo, "semantic", "head_mean")

    print("=" * 78)
    print(f"ГЛАВНЫЙ ВОПРОС: во что превратился разрыв semantic -> metric при alpha = {lo}")
    print("=" * 78)
    print(f"  на голосовании: {gap_vote:+.3f}")
    print(f"  на голове:      {gap_head:+.3f}")
    print(f"  сжатие:         {gap_vote - gap_head:+.3f}"
          f"   ({(1 - gap_head / gap_vote) * 100:.0f}% разрыва закрыто головой)"
          if abs(gap_vote) > 1e-9 else "")
    print()
    if gap_head > 0.15:
        name, text = "А", ("голова не разобрала семантическую выборку; разрыв не "
                           "объясняется примитивностью решающего правила")
    elif gap_head > 0.05:
        name, text = "Б", ("голова частично разобрала выборку; вклад метода "
                           "вычислительный — он переносит различение проектов "
                           "из инференса в память")
    else:
        name, text = "В", ("голова полностью компенсировала; ценность метода "
                           "сводится к экономии на сложности решающего правила")
    print(f"  СЦЕНАРИЙ {name}: {text}")
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


def visualize(summary: pd.DataFrame, alphas: List[float], figures_dir: Path) -> None:
    """
    Столбики: голосование против головы, пунктиром потолок читателя.
    ---

    Args:
        summary (pd.DataFrame): Сводная таблица.
        alphas (List[float]): Значения alpha.
        figures_dir (Path): Куда сохранять.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    names = list(CONFIGS)

    fig, axes = plt.subplots(1, len(alphas), figsize=(6.8 * len(alphas), 5.4), squeeze=False)

    for ax, alpha in zip(axes[0], alphas):
        part  = summary[summary["alpha"] == alpha].set_index("policy")
        base  = np.arange(len(names))
        width = 0.36

        for shift, col, std_col, color, label in (
            (-width / 2, "vote_mean", "vote_std", VOTE_COLOR, "голосование"),
            (+width / 2, "head_mean", "head_std", HEAD_COLOR, "трансформер"),
        ):
            values = np.array([part.loc[n, col] for n in names])
            errors = np.array([part.loc[n, std_col] for n in names])
            ax.bar(base + shift, values, width=width * 0.92, color=color,
                   yerr=errors, capsize=3,
                   error_kw={"elinewidth": 1.1, "ecolor": INK_2}, label=label)
            for x, y in zip(base + shift, values):
                ax.text(x, y + 0.022, f"{y:.2f}", ha="center", fontsize=8, color=INK_2)

        for n, name in enumerate(names):
            ax.hlines(part.loc[name, "hit_mean"], base[n] - 0.45, base[n] + 0.45,
                      color=REF_COLOR, ls="--", lw=1.5, zorder=4)

        ax.hlines([], [], [], color=REF_COLOR, ls="--", lw=1.5,
                  label="потолок читателя (hit@k)")
        ax.set_xticks(base)
        ax.set_xticklabels([CONFIGS[n][2] for n in names], fontsize=9.5, color=INK_2)
        ax.set_ylabel("доля верных действий", fontsize=10, color=INK_2)
        ax.set_ylim(0, 1.12)
        ax.set_title(rf"$\alpha$ = {alpha}", fontsize=12, color=INK, pad=10)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper left")
        _style(ax)

    fig.suptitle(r"Голосование против трансформерной головы  (5 сидов, усы — ±1$\sigma$)",
                 fontsize=13, color=INK, y=1.0)
    fig.tight_layout()
    fig.savefig(figures_dir / "head.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"записано: {figures_dir / 'head.png'}\n")


if __name__ == "__main__":
    main()
