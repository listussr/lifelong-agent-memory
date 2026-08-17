import numpy as np
from typing import List, Tuple

from .utils import NP_RNG
from .rollout import Rollout
from .enviroment import candidates_len

def predict_row(actions: np.ndarray, mask: np.ndarray, situation: int, rng: NP_RNG) -> int:
    """
    Выбор наиболее частого кандидата из ситуаций.
    ---

    Выбирается 1 кандидат, который наиболее часто встречался в решениях.

    Если кандидатов несколько или их вообще нет, то выбирается случайное решение.

    Args:
        actions (np.ndarray): Выбранные через `retrieve` действия.
        mask (np.ndarray): Маска для действий.
        situation (int): Индекс ситуации.
        rng (NP_RNG): Numpy генератор случаных чисел.

    Returns:
        int: Индекс выбранного действия.
    """
    if not (~mask).any():
        return int(situation) * candidates_len + int(rng.integers(candidates_len))

    valid_actions  = actions[~mask]
    values, counts = np.unique(valid_actions, return_counts=True)

    best = values[counts == counts.max()]
    return int(best[rng.integers(len(best))]) if len(best) > 1 else int(best[0])

def predict(rollout: Rollout, rng: NP_RNG) -> np.ndarray:
    """
    Предсказания действий для всего трейса.
    ---

    Args:
        rollout (Rollout): Трейс из `rollout.py`.
        rng (NP_RNG): Numpy генератор случаных числе.

    Returns:
        np.ndarray: Массив действий.
    """
    episode_len = len(rollout.query_ids)
    predictions = np.empty((episode_len,), dtype=np.int16)
    for t in range(episode_len):
        predictions[t] = predict_row(
            actions   = rollout.retrieve_actions[t],
            mask      = rollout.retrieve_mask[t],
            situation = rollout.situations[t],
            rng       = rng
        )
    return predictions

def evaluate_accuracy(roll: Rollout, pred: np.ndarray) -> Tuple[float]:
    """
    Подсчёт метрик точности.
    ---

    На выход отдаются точность на всём наборе и точность последней трети.

    
    Полная точность ниже точности последней трети, т.к. в начале трейса идёт заполнение массивов.
    
    Args:
        roll (Rollout): Трейс из `rollout.py`
        pred (np.ndarray): Массив предсказаний.

    Returns:
        Tuple[float]: Точность полная и точность последней трети.
    """
    episode_len = len(pred)
    accuracy = np.mean(roll.correct == pred)
    accuracy_last_third = np.mean(
        roll.correct[-episode_len // 3:] == pred[-episode_len // 3:]
    )
    return accuracy, accuracy_last_third

def evaluate_accuracy_no_dup(roll: Rollout, pred: np.ndarray) -> Tuple[float, float]:
    """
    Контроль на точные дубликаты.
    ---

    На части шагов в памяти лежит текст, дословно совпадающий с запросом. Там косинус
    равен единице, и верный ответ достаётся семантике даром — без всякого обобщения.

    Доля таких шагов растёт вместе с alpha, поэтому без этого контроля часть выигрыша
    при alpha=1 может объясняться памятью на дословные совпадения, а не изучаемым
    механизмом.

    Args:
        roll (Rollout): Трейс из `rollout.py`.
        pred (np.ndarray): Массив предсказаний.

    Returns:
        Tuple[float, float]: Точность на шагах БЕЗ дубликата и точность на шагах С дубликатом.
    """
    clean = ~roll.exact_dup
    accuracy_clean = (float(np.mean(roll.correct[clean] == pred[clean]))
                      if clean.any() else float("nan"))
    accuracy_dup   = (float(np.mean(roll.correct[~clean] == pred[~clean]))
                      if (~clean).any() else float("nan"))
    return accuracy_clean, accuracy_dup


def evaluate_usability(roll: Rollout) -> Tuple[float, float]:
    correct = roll.correct[:, np.newaxis]
    matches = (roll.retrieve_actions == correct) & ~roll.retrieve_mask

    full_metric = np.any(matches, axis=1).mean()

    keep           = ~roll.exact_dup
    correct_no_dup = roll.correct[keep][:, np.newaxis]
    actions_no_dup = roll.retrieve_actions[keep]
    mask_no_dup    = roll.retrieve_mask[keep]

    no_dup_metric = np.any(
        (actions_no_dup == correct_no_dup) & ~mask_no_dup, axis=1
    ).mean() if keep.any() else float("nan")

    return float(full_metric), float(no_dup_metric)


def evaluate_precision(roll: Rollout) -> float:
    correct  = roll.correct[:, np.newaxis]
    matches  = (roll.retrieve_actions == correct) & ~roll.retrieve_mask
    filled   = roll.n_retrieved > 0
    if not filled.any():
        return float("nan")
    per_step = matches[filled].sum(axis=1) / roll.n_retrieved[filled]
    return float(per_step.mean())