import numpy as np
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Dict, List, Optional, Tuple

from .enviroment import Step
from .utils import NP_RNG, make_episode, make_rng

TRAIN_SEED_OFFSET = 100_000   # обучающие сиды

BucketKey = Tuple[int, int, int]     # (эпизод, ситуация, действие)
GroupKey  = Tuple[int, int]          # (эпизод, ситуация)


class DecisionMetric(nn.Module):
    """
    Линейная проекция эмбеддинга в пространство решений.
    ---
    """

    def __init__(self, d_in: int = 384, d_out: int = 64):
        """
        Args:
            d_in (int): Размерность эмбеддинга пула.
            d_out (int): Размерность пространства решений.
        """
        super().__init__()
        self.proj = nn.Linear(d_in, d_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (B, d_in) эмбеддинги.

        Returns:
            torch.Tensor: (B, d_out) нормированные проекции.
        """
        return F.normalize(self.proj(x), dim=-1)


def collect_episodes(pool_index: Dict, alphas: List[float], eta: float, n_steps: int, n_episodes: int) -> List[List[Step]]:
    """
    Генерация обучающих эпизодов.
    ---

    Args:
        pool_index (Dict): (project, situation, topic) -> список text_id.
        alphas (List[float]): Режимы, по которым чередуются эпизоды.
        eta (float): Шум обратной связи.
        n_steps (int): Длина эпизода.
        n_episodes (int): Сколько эпизодов сгенерировать.

    Returns:
        List[List[Step]]: Список эпизодов.
    """
    episodes = []
    for i in range(n_episodes):
        alpha = alphas[i % len(alphas)]
        rng   = make_rng(TRAIN_SEED_OFFSET + i)
        episodes.append(make_episode(alpha, eta, n_steps, pool_index, rng))
    return episodes


def build_buckets(episodes: List[List[Step]]) -> Tuple[Dict[BucketKey, List[int]], Dict[GroupKey, List[BucketKey]]]:
    """
    Раскладка записей по корзинам для быстрого сэмплирования пар.
    ---

    Args:
        episodes (List[List[Step]]): Обучающие эпизоды.

    Returns:
        Tuple: buckets - (эпизод, ситуация, действие) -> [text_id];
               groups  - (эпизод, ситуация) -> [ключи корзин этой группы].
    """
    buckets: Dict[BucketKey, List[int]] = {}
    groups:  Dict[GroupKey, List[BucketKey]] = {}

    for episode_id, episode in enumerate(episodes):
        for step in episode:
            key = (episode_id, step.situation, step.revealed_action)
            if key not in buckets:
                buckets[key] = []
                groups.setdefault((episode_id, step.situation), []).append(key)
            buckets[key].append(step.text_id)

    return buckets, groups


def sample_pairs(
        buckets: Dict[BucketKey, List[int]],
        groups: Dict[GroupKey, List[BucketKey]],
        rng: NP_RNG, batch_size: int, pos_share: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Разбиение на батчи пар с нужной разметкой.
    ---

    Положительная пара: две записи из одной корзины, то есть один эпизод, одна ситуация, одинаковое сообщённое действие.

    Отрицательная пара: две записи из разных корзин одной группы, то есть один эпизод, одна ситуация, разные действия.

    Args:
        buckets (Dict): Из `build_buckets`.
        groups (Dict): Из `build_buckets`.
        rng (NP_RNG): Numpy генератор случайных чисел.
        batch_size (int): Размер батча.
        pos_share (float): Доля положительных пар.

    Returns:
        Tuple: idx_i, idx_j — индексы в пуле эмбеддингов; target = +1 / -1.
    """
    pos_keys = [k for k, v in buckets.items() if len(v) >= 2]
    neg_keys = [g for g, keys in groups.items() if len(keys) >= 2]
    if not pos_keys or not neg_keys:
        raise ValueError("недостаточно данных для сэмплирования пар")

    n_pos = int(round(batch_size * pos_share))
    idx_i, idx_j, target = [], [], []

    # выбор в одной корзине, т.к. в ней гарантировано одно действие
    for _ in range(n_pos):
        key = pos_keys[rng.integers(len(pos_keys))]
        a, b = rng.choice(buckets[key], size=2, replace=False)
        idx_i.append(int(a))
        idx_j.append(int(b))
        target.append(1.0)

    # сначала выбор из разных корзин, чтобы были разные действия
    for _ in range(batch_size - n_pos):
        group = neg_keys[rng.integers(len(neg_keys))]
        keys  = groups[group]
        ka, kb = rng.choice(len(keys), size=2, replace=False)
        a = buckets[keys[ka]][rng.integers(len(buckets[keys[ka]]))]
        b = buckets[keys[kb]][rng.integers(len(buckets[keys[kb]]))]
        idx_i.append(int(a))
        idx_j.append(int(b))
        target.append(-1.0)

    return (
        np.array(idx_i, np.int64),
        np.array(idx_j, np.int64),
        np.array(target, np.float32)
    )


def sample_pairs_semantic(
        emb: np.ndarray,
        buckets: Dict[BucketKey, List[int]],
        groups: Dict[GroupKey, List[BucketKey]],
        rng: NP_RNG, batch_size: int, oversample: int = 4
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Разбиение на батчи пар с семантической разметкой.
    ---

    Те же архитектура и объём обучения, но пары размечены по косинусу: похожие притягиваются, непохожие отталкиваются.

    Args:
        emb (np.ndarray): Эмбеддинги пула, нормированные.
        buckets (Dict): Из `build_buckets`.
        groups (Dict): Из `build_buckets`.
        rng (NP_RNG): Numpy генератор случайных чисел.
        batch_size (int): Размер батча.
        oversample (int): Во сколько раз больше пар набрать перед отбором краёв.

    Returns:
        Tuple: idx_i, idx_j, target.
    """
    group_keys = [g for g, keys in groups.items() if len(keys) >= 1]
    pool_i, pool_j = [], []

    while len(pool_i) < batch_size * oversample:
        group = group_keys[rng.integers(len(group_keys))]
        ids   = [tid for key in groups[group] for tid in buckets[key]]
        if len(ids) < 2:
            continue
        a, b = rng.choice(ids, size=2, replace=False)
        pool_i.append(int(a)); pool_j.append(int(b))

    pool_i = np.array(pool_i, np.int64)
    pool_j = np.array(pool_j, np.int64)
    sims   = np.sum(emb[pool_i] * emb[pool_j], axis=1)

    order = np.argsort(-sims, kind="stable")
    half  = batch_size // 2
    top, bottom = order[:half], order[-(batch_size - half):]

    idx_i  = np.concatenate([pool_i[top], pool_i[bottom]])
    idx_j  = np.concatenate([pool_j[top], pool_j[bottom]])
    target = np.concatenate([np.ones(len(top), np.float32), -np.ones(len(bottom), np.float32)])
    return idx_i, idx_j, target


def fit_metric(
        emb: np.ndarray,
        sampler: Callable[[int], Tuple[np.ndarray, np.ndarray, np.ndarray]],
        d_out: int = 64, lr: float = 1e-3, epochs: int = 1000,
        batch_size: int = 256, margin: float = 0.2,
        neg_weight: float = 1.0, device: str = "cpu",
        log_every: int = 200,
        on_log: Optional[Callable[[int, float, "DecisionMetric"], None]] = None
    ) -> DecisionMetric:
    """
    Обучение проекции контрастивным лоссом.
    ---

    Args:
        emb (np.ndarray): Эмбеддинги пула (N, d_in), нормированные.
        sampler (Callable): batch_size -> (idx_i, idx_j, target).
        d_out (int): Размерность пространства решений.
        lr (float): Скорость обучения.
        steps (int): Число шагов оптимизации.
        batch_size (int): Размер батча.
        margin (float): Зазор для отрицательных пар.
        neg_weight (float): Вес отрицательных пар. Больше 1 - упор на чистый сигнал.
        device (str): cpu или cuda.
        log_every (int): Через сколько шагов звать on_log.
        on_log (Optional[Callable]): Колбэк для диагностики по ходу обучения.

    Returns:
        DecisionMetric: Обученная модель в режиме eval.
    """
    model   = DecisionMetric(d_in=emb.shape[1], d_out=d_out).to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CosineEmbeddingLoss(margin=margin, reduction="none")
    emb_t   = torch.from_numpy(emb).float().to(device)

    model.train()
    for epoch in range(1, epochs + 1):
        idx_i, idx_j, target = sampler(batch_size)

        z1 = model(emb_t[torch.from_numpy(idx_i).to(device)])
        z2 = model(emb_t[torch.from_numpy(idx_j).to(device)])
        t  = torch.from_numpy(target).to(device)

        per_pair = loss_fn(z1, z2, t)
        weights  = torch.where(t > 0, torch.ones_like(t), torch.full_like(t, neg_weight))
        loss     = (per_pair * weights).sum() / weights.sum()

        opt.zero_grad()
        loss.backward()
        opt.step()

        if on_log is not None and (epoch % log_every == 0 or epoch == 1):
            model.eval()
            on_log(epoch, float(loss.item()), model)
            model.train()

    model.eval()
    return model


def projection_matrix(model: DecisionMetric) -> np.ndarray:
    """
    Матрица проекции в виде, пригодном для `emb @ P`.
    ---

    `nn.Linear` хранит веса в формате (d_out, d_in) и считает `x @ W.T`, поэтому транспонируется.

    Args:
        model (DecisionMetric): Обученная модель.

    Returns:
        np.ndarray: (d_in, d_out).
    """
    return model.proj.weight.detach().cpu().numpy().T.copy()


def random_projection(d_in: int, d_out: int, seed: int = 0) -> np.ndarray:
    """
    Случайная замороженная проекция.
    ---

    Проверяет, не помогает ли само по себе снижение размерности, без всякого обучения.

    Args:
        d_in (int): Исходная размерность.
        d_out (int): Целевая размерность.
        seed (int): Сид.

    Returns:
        np.ndarray: (d_in, d_out).
    """
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(d_in, d_out)) / np.sqrt(d_in)).astype(np.float32)


def project(emb: np.ndarray, P: np.ndarray) -> np.ndarray:
    """
    Проецирование с нормировкой - то же, что делает модель на forward.
    ---

    Args:
        emb (np.ndarray): (..., d_in).
        P (np.ndarray): (d_in, d_out).

    Returns:
        np.ndarray: (..., d_out), строки единичной длины.
    """
    z = emb @ P
    n = np.linalg.norm(z, axis=-1, keepdims=True)
    return z / np.maximum(n, 1e-12)


def check_linear_separability(emb: np.ndarray, meta: np.ndarray, seed: int = 0, test_share: float = 0.3) -> float:
    """
    Диагностика линейной отделимосит маркера проекта.
    ---

    Args:
        emb (np.ndarray): Эмбеддинги пула.
        meta (np.ndarray): (N, 4) - (project, situation, topic, variant).
        seed (int): Сид разбиения.
        test_share (float): Доля тестовой части.

    Returns:
        float: Доля верных предсказаний проекта на отложенной части.
    """
    rng   = np.random.default_rng(seed)
    order = rng.permutation(len(emb))
    cut   = int(len(emb) * (1 - test_share))
    train, test = order[:cut], order[cut:]

    clf = LogisticRegression(max_iter=1000)
    clf.fit(emb[train], meta[train, 0])
    return float(clf.score(emb[test], meta[test, 0]))
