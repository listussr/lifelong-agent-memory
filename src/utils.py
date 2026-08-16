import json
import numpy as np
from pathlib import Path
import random
import torch
from typing import Dict, List, Tuple
import yaml

from .enviroment import (
    situation_len,
    projects_len,
    topics_len,
    candidates_len,
    CANDIDATES,
    ACTION_ID,
    Step,
)

NP_RNG = np.random.Generator

def make_rng(seed: int) -> NP_RNG:
    """
    Задание сида по умолчанию
    ---

    Задаёт сид для torch, numpy и random

    Args:
        seed (int): Значение сида.

    Returns:
        NP_RNG: Генератор для numpy.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    return np.random.default_rng(seed)

def sample_topic(project: int, alpha: float, T: int, rng: NP_RNG) -> int:
    """
    Определение контекста для проекта.
    ---

    Args:
        project (int): Номер проекта, для которого выбирается тема.
        alpha (float): Величина случайности выбора контекста. Если 0, то контекст случаен, если 1, то строго закреплён за проектом.
        T (int): Количество тем (должно совпадать с числом проектов).
        rng (NP_RNG): Numpy генератор случайных чисел.

    Returns:
        int: Номер выбранного контекста.
    """
    if rng.random() < alpha:
        return int(project)
    return int(rng.integers(T))

def make_rule_table(rng: NP_RNG) -> np.ndarray:
    """
    Создание таблицы правил для проекта и ситуации.
    ---

    Пересоздаётся на каждый ЭПИЗОД (не на шаг). Внутри эпизода правила
    постоянны, иначе накопленный опыт терял бы смысл.

    Args:
        rng (NP_RNG): Numpy генератор случайных чисел.

    Returns:
        np.ndarray: Таблица переходов.
    """
    return rng.integers(0, candidates_len, size=(projects_len, situation_len))

def make_episode(alpha: float, eta: float, n_steps: int, pool_index: Dict, rng: NP_RNG) -> List[Step]:
    """
    Генерация одного эпизода.
    ---

    Таблица правил своя на каждый эпизод, поэтому её нельзя выучить в весах - можно выучить только то, как доставать правила из памяти.

    Args:
        alpha (float): Сила привязки темы к проекту. 0 - тема случайна, 1 - закреплена.
        eta (float): Вероятность того, что обратная связь соврёт.
        n_steps (int): Длина потока.
        pool_index (Dict): (project, situation, topic) -> список text_id.
        rng (NP_RNG): Numpy генератор случайных чисел.

    Returns:
        List[Step]: Поток шагов эпизода.
    """
    rule = make_rule_table(rng)
    steps = []
    for _ in range(n_steps):
        project   = int(rng.integers(projects_len))
        situation = int(rng.integers(situation_len))
        topic     = sample_topic(project, alpha, topics_len, rng)
        text_id   = int(rng.choice(pool_index[(project, situation, topic)]))
        correct   = ACTION_ID[CANDIDATES[situation][rule[project][situation]]]
        revealed  = correct
        if rng.random() < eta:
            wrong    = [ACTION_ID[n] for n in CANDIDATES[situation]
                        if ACTION_ID[n] != correct]
            revealed = int(rng.choice(wrong))

        steps.append(
            Step(
                text_id=text_id,
                project=project,
                situation=situation,
                topic=topic,
                correct_action=correct,
                revealed_action=revealed
            )
        )
    return steps

def load_pool(out_dir: str) -> Tuple[List[str], np.ndarray, np.ndarray, Dict]:
    """
    Чтение данных из файлов, полученных скриптом.
    ---

    Args:
        out_dir (str): Путь к папке с файлами.

    Returns:
        Tuple[List[str], np.ndarray, np.ndarray, Dict]: Список текстов, список метаданных, эмбеддинги, словарь с метаданными и текстами по ним.
    """
    out   = Path(out_dir)
    texts = json.loads((out / "pool_texts.json").read_text(encoding="utf-8"))
    meta  = np.load(out / "pool_meta.npy")
    emb   = np.load(out / "pool_emb.npy")
    raw   = json.loads((out / "pool_index.json").read_text(encoding="utf-8"))
    index = {tuple(int(x) for x in k.split(",")): v for k, v in raw.items()}
    return texts, meta, emb, index

def cosine_table(emb: np.ndarray, meta: np.ndarray, situation: int) -> Dict:
    """
    Построение таблицы косинусных сходств.
    ---

    Таблица показывает косинусные сходства для 4 случаев:
     1) один проект и контекст в той же теме,
     2) разные проекты и контекст в той же теме,
     3) один проект, но разные контексты (типа ML и тестирование внутри одного проекта),
     4) разные проекты и разные контексты.

     Выходной словарь:
     * `cells`: матрица сходства,
     * `topic_gap`: разница между контекстами,
     * `proj_gap`: разница между проектами,
     * `trap`: случай, когда отличается проект, но описание идентично.

    Args:
        emb (np.ndarray): Матрица эмбеддингов текста.
        meta (np.ndarray): Список метаданных.
        situation (int): Идентификатор ситуации.

    Returns:
        dict: Выходной словарь.
    """
    sel  = np.where(meta[:, 1] == situation)[0]
    E, m = emb[sel], meta[sel]

    C    = E @ E.T # скалярное произведение номированных эмбеддингов = косинусное расстояние
    i, j = np.triu_indices(len(sel), k=1)
    c    = C[i, j]

    same_project = m[i, 0] == m[j, 0]                # одинаковые проекты
    same_topic   = m[i, 2] == m[j, 2]                # одинаковые контексты
    same_both    = same_topic & (m[i, 3] == m[j, 3]) # одинаковые варианты и одинаковые контексты, тема учтена, а проект не нужен
    keep         = ~same_both

    cells = {
        (tl, pl): c[keep & tm & pm].mean()
        for tl, tm in (("та же тема", same_topic), ("другая тема", ~same_topic))
        for pl, pm in (("тот же проект", same_project), ("другой проект", ~same_project))
    }
    return {
        "cells":     cells,
        "topic_gap": c[keep & same_topic].mean() - c[keep & ~same_topic].mean(),
        "proj_gap":  c[keep & same_project].mean() - c[keep & ~same_project].mean(),
        "trap":      c[same_both & ~same_project].mean(),
    }

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
CONFIG = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))