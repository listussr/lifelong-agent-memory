from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .enviroment import Step, actions_len, candidates_len
from .memory import make_policy
from .rollout import Rollout, rollout_loop
from .utils import CONFIG, NP_RNG, make_episode, make_rng

HEAD_SEED_OFFSET = 200_000   # обучающие эпизоды головы: вне оценочных и вне метрики


class TransformerHead(nn.Module):
    """
    Решающее правило: запрос плюс извлечённые записи -> действие.
    ---
    """

    def __init__(self, d_in: int = 384, d_model: int = 128, n_actions: int = actions_len,
                 n_layers: int = 2, n_heads: int = 4, ff_mult: int = 4,
                 dropout: float = 0.1, norm_first: bool = True):
        """
        Args:
            d_in (int): Размерность эмбеддинга пула.
            d_model (int): Внутренняя размерность головы.
            n_actions (int): Размер выходного слоя.
            n_layers (int): Слоёв трансформера.
            n_heads (int): Голов внимания.
            ff_mult (int): Во сколько раз скрытый слой больше d_model.
            dropout (float): Дропаут.
            norm_first (bool): Пренормализация.
        """
        super().__init__()
        self.proj_q   = nn.Linear(d_in, d_model)
        self.proj_m   = nn.Linear(d_in, d_model)
        self.act_emb  = nn.Embedding(n_actions, d_model)
        self.type_emb = nn.Embedding(2, d_model)      # 0 - запрос, 1 - запись

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout, norm_first=norm_first, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out     = nn.Linear(d_model, n_actions)

        block = torch.arange(n_actions) // candidates_len
        self.register_buffer("block", block, persistent=False)

    def forward(self, q_emb: torch.Tensor, m_emb: torch.Tensor,
                m_act: torch.Tensor, m_mask: torch.Tensor,
                situation: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q_emb (torch.Tensor): (B, d_in) эмбеддинг запроса.
            m_emb (torch.Tensor): (B, k, d_in) эмбеддинги записей; пустые слоты обнулены.
            m_act (torch.Tensor): (B, k) действия записей; в пустых слотах -1.
            m_mask (torch.Tensor): (B, k) True = слот ПУСТОЙ.
            situation (torch.Tensor): (B,) тип ситуации запроса.

        Returns:
            torch.Tensor: (B, n_actions) логиты; вне блока ситуации -inf.
        """
        B = q_emb.shape[0]

        query   = self.proj_q(q_emb) + self.type_emb.weight[0]        # (B, d)
        records = (self.proj_m(m_emb)
                   + self.act_emb(m_act.clamp(min=0))
                   + self.type_emb.weight[1])                          # (B, k, d)

        seq = torch.cat([query.unsqueeze(1), records], dim=1)          # (B, k+1, d)

        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=seq.device),
                         m_mask], dim=1)

        hidden = self.encoder(seq, src_key_padding_mask=pad)
        logits = self.out(hidden[:, 0])                                # нулевой элемент по батчу

        allowed = self.block.unsqueeze(0) == situation.unsqueeze(1)
        return logits.masked_fill(~allowed, float("-inf"))


def collect_head_episodes(pool_index: Dict, alphas: List[float], eta: float,
                          n_steps: int, n_episodes: int) -> List[List[Step]]:
    """
    Обучающие эпизоды для головы.
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
        rng   = make_rng(HEAD_SEED_OFFSET + i)
        episodes.append(make_episode(alpha, eta, n_steps, pool_index, rng))
    return episodes


def build_dataset(episodes: List[List[Step]], emb: np.ndarray, meta: np.ndarray,
                  budget: int, k: int, policy_names: List[str]) -> Dict[str, np.ndarray]:
    """
    Датасет из трейса с политиками.
    ---

    Args:
        episodes (List[List[Step]]): Обучающие эпизоды.
        emb (np.ndarray): Эмбеддинги пула; извлечение идёт в ИСХОДНОМ пространстве.
        meta (np.ndarray): Скрытая разметка; уходит только в Oracle.
        budget (int): Бюджет памяти.
        k (int): Сколько извлекать.
        policy_names (List[str]): Смесь политик.

    Returns:
        Dict[str, np.ndarray]: query_ids, ret_ids, ret_actions, ret_mask, situations, correct.
    """
    parts = {key: [] for key in
             ("query_ids", "ret_ids", "ret_actions", "ret_mask", "situations", "correct")}

    for n, episode in enumerate(episodes):
        name   = policy_names[n % len(policy_names)]
        policy = make_policy(name, budget, emb, meta, make_rng(HEAD_SEED_OFFSET + n),
                             CONFIG["memory"]["kmeans_clusters"],
                             CONFIG["memory"]["kmeans_refit_every"], n)
        roll = rollout_loop(episode, policy, emb, k)

        parts["query_ids"].append(roll.query_ids)
        parts["ret_ids"].append(roll.retrieve_ids)
        parts["ret_actions"].append(roll.retrieve_actions)
        parts["ret_mask"].append(roll.retrieve_mask)
        parts["situations"].append(roll.situations)
        parts["correct"].append(roll.correct)

    return {key: np.concatenate(value, axis=0) for key, value in parts.items()}


def _gather(emb_t: torch.Tensor, ret_ids: torch.Tensor, ret_mask: torch.Tensor) -> torch.Tensor:
    """
    Эмбеддинги записей с обнулением пустых слотов.
    ---

    Args:
        emb_t (torch.Tensor): (N, d_in) пул.
        ret_ids (torch.Tensor): (B, k) индексы, -1 в пустых.
        ret_mask (torch.Tensor): (B, k) True = пусто.

    Returns:
        torch.Tensor: (B, k, d_in).
    """
    safe = torch.where(ret_mask, torch.zeros_like(ret_ids), ret_ids)
    out  = emb_t[safe]
    return out.masked_fill(ret_mask.unsqueeze(-1), 0.0)


def fit_head(data: Dict[str, np.ndarray], emb: np.ndarray, rng: NP_RNG,
             d_model: int = 128, n_layers: int = 2, n_heads: int = 4,
             ff_mult: int = 4, dropout: float = 0.1, norm_first: bool = True,
             lr: float = 3e-4, epochs: int = 4, batch_size: int = 256,
             device: str = "cpu", verbose: bool = True) -> TransformerHead:
    """
    Обучение головы обычной классификацией.
    ---

    Args:
        data (Dict[str, np.ndarray]): Из `build_dataset`.
        emb (np.ndarray): Эмбеддинги пула.
        rng (NP_RNG): Генератор для перемешивания.
        device (str): cpu или cuda.
        verbose (bool): Печатать ли прогресс по эпохам.

    Returns:
        TransformerHead: Обученная модель в режиме eval.
    """
    model = TransformerHead(d_in=emb.shape[1], d_model=d_model, n_layers=n_layers,
                            n_heads=n_heads, ff_mult=ff_mult, dropout=dropout,
                            norm_first=norm_first).to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    emb_t   = torch.from_numpy(emb).float().to(device)

    tensors = {
        "query_ids":   torch.from_numpy(data["query_ids"]).long().to(device),
        "ret_ids":     torch.from_numpy(data["ret_ids"]).long().to(device),
        "ret_actions": torch.from_numpy(data["ret_actions"]).long().to(device),
        "ret_mask":    torch.from_numpy(data["ret_mask"]).bool().to(device),
        "situations":  torch.from_numpy(data["situations"]).long().to(device),
        "correct":     torch.from_numpy(data["correct"]).long().to(device),
    }
    n_samples = len(data["correct"])

    model.train()
    for epoch in range(1, epochs + 1):
        order = torch.from_numpy(rng.permutation(n_samples)).to(device)
        total_loss, total_hit = 0.0, 0

        for start in range(0, n_samples, batch_size):
            idx = order[start:start + batch_size]

            q_emb  = emb_t[tensors["query_ids"][idx]]
            m_mask = tensors["ret_mask"][idx]
            m_emb  = _gather(emb_t, tensors["ret_ids"][idx], m_mask)
            target = tensors["correct"][idx]

            logits = model(q_emb, m_emb, tensors["ret_actions"][idx], m_mask, tensors["situations"][idx])
            loss = loss_fn(logits, target)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += float(loss.item()) * len(idx)
            total_hit  += int((logits.argmax(dim=1) == target).sum().item())

        if verbose:
            print(f"    эпоха {epoch}: loss={total_loss / n_samples:.4f}  "
                  f"точность на обучении={total_hit / n_samples:.3f}")

    model.eval()
    return model


@torch.no_grad()
def predict_head(model: TransformerHead, roll: Rollout, emb: np.ndarray,
                 device: str = "cpu") -> np.ndarray:
    """
    Предсказания на весь эпизод ОДНИМ батчем.
    ---

    Args:
        model (TransformerHead): Обученная голова.
        roll (Rollout): Трейс эпизода.
        emb (np.ndarray): Исходные эмбеддинги пула.
        device (str): cpu или cuda.

    Returns:
        np.ndarray: (T,) выбранные действия.
    """
    model.eval()
    emb_t  = torch.from_numpy(emb).float().to(device)
    m_mask = torch.from_numpy(roll.retrieve_mask).bool().to(device)

    logits = model(
        emb_t[torch.from_numpy(roll.query_ids).long().to(device)],
        _gather(emb_t, torch.from_numpy(roll.retrieve_ids).long().to(device), m_mask),
        torch.from_numpy(roll.retrieve_actions).long().to(device),
        m_mask,
        torch.from_numpy(roll.situations).long().to(device),
    )
    return logits.argmax(dim=1).cpu().numpy().astype(np.int16)


@torch.no_grad()
def check_empty_memory(model: TransformerHead, emb: np.ndarray, roll: Rollout,
                       device: str = "cpu") -> float:
    """
    Проверка на пустая память -> минимальная точность.
    ---

    Args:
        model (TransformerHead): Обученная голова.
        emb (np.ndarray): Эмбеддинги пула.
        roll (Rollout): Трасса, из неё берутся запросы и правильные ответы.
        device (str): cpu или cuda.

    Returns:
        float: Доля верных действий при пустой памяти.
    """
    model.eval()
    emb_t = torch.from_numpy(emb).float().to(device)
    T, k  = roll.retrieve_ids.shape

    empty_mask = torch.ones(T, k, dtype=torch.bool, device=device)
    logits = model(
        emb_t[torch.from_numpy(roll.query_ids).long().to(device)],
        torch.zeros(T, k, emb.shape[1], device=device),
        torch.zeros(T, k, dtype=torch.long, device=device),
        empty_mask,
        torch.from_numpy(roll.situations).long().to(device),
    )
    pred = logits.argmax(dim=1).cpu().numpy()
    return float((pred == roll.correct).mean())
