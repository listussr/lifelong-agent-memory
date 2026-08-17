from dataclasses import dataclass
import numpy as np
from typing import List

from .enviroment import Step
from .memory import MemoryPolicy, Item

@dataclass
class Rollout:
 query_ids: np.ndarray        # (T,) индекс запроса
 situations: np.ndarray       # (T,)
 correct: np.ndarray          # (T,) верное действие для метрик
 retrieve_ids: np.ndarray     # (T, k) что извлеклось, -1 в пустых слотах
 retrieve_actions: np.ndarray # (T, k) действия извлечённых
 retrieve_mask: np.ndarray    # (T, k) True = слот пустой
 n_retrieved: np.ndarray      # (T,)
 exact_dup: np.ndarray        # (T,) был ли точный дубликат запроса


def rollout_loop(episode: List[Step], policy: MemoryPolicy,
                 embeddings: np.ndarray, k: int) -> Rollout:
    T = len(episode)
    query_ids   = np.empty(T, np.int32)
    situations  = np.empty(T, np.int16)
    correct     = np.empty(T, np.int16)
    ret_ids     = np.full((T, k), -1, np.int32)
    ret_actions = np.full((T, k), -1, np.int16)
    ret_mask    = np.ones((T, k), bool)          # по умолчанию всё пусто
    n_retrieved = np.zeros(T, np.int16)
    exact_dup   = np.zeros(T, bool)

    policy.reset()
    for t, step in enumerate(episode):
        assert len(policy.items) <= min(t, policy.budget), f"шаг {t}: retrieve вызван после write"

        got = policy.retrieve(embeddings[step.text_id], step.situation, k, project=step.project)

        query_ids[t]   = step.text_id
        situations[t]  = step.situation
        correct[t]     = step.correct_action
        n_retrieved[t] = len(got)
        exact_dup[t]   = any(it.text_id == step.text_id for it in got)

        for slot, it in enumerate(got):
            ret_ids[t, slot]     = it.text_id
            ret_actions[t, slot] = it.revealed_action
            ret_mask[t, slot]    = False

        policy.write(Item(step.text_id, step.situation, step.revealed_action))

    return Rollout(
       query_ids,
       situations,
       correct,
       ret_ids,
       ret_actions,
       ret_mask,
       n_retrieved,
       exact_dup
    )
