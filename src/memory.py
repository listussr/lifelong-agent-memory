from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from sklearn.cluster import KMeans as SKKMeans

from .utils import NP_RNG
from . import CONFIG


@dataclass(frozen=True)
class Item:
    """Единица опыта. Ни project, ни topic, ни correct_action здесь нет и быть не должно."""
    text_id: int          # индекс в пуле, по нему берётся эмбеддинг
    situation: int        # тип ситуации, читается из текста — доступен честно
    revealed_action: int  # что среда сообщила правильным (возможно, с шумом)


class MemoryPolicy(ABC):
    """
    Базовая политика памяти.
    ---

    Хранение у всех бейзлайнов одинаковое (FIFO), различается только извлечение.
    Так изолируются два фактора: наш метод потом переопределит `_evict`, и получится
    сетка 2x2 «метрика извлечения х правило вытеснения».

    Контракт: `retrieve` вызывается СТРОГО до `write` на том же шаге, иначе текущая
    запись попадёт в собственную выборку вместе с готовым ответом.
    """

    def __init__(self, budget: int, emb: np.ndarray):
        """
        Args:
            budget (int): Максимальное число хранимых записей. Одинаков у всех политик.
            emb (np.ndarray): Матрица эмбеддингов пула (N, dim), нормированная. Только для чтения.
        """
        self.budget            = budget
        self.emb               = emb
        self.items: List[Item] = []

    def reset(self) -> None:
        """
        Очистка памяти между эпизодами.
        ---
        """
        self.items.clear()

    def write(self, item: Item) -> None:
        """
        Запись в память с вытеснением при переполнении.
        ---

        Args:
            item (Item): Новая запись.
        """
        self.items.append(item)
        if len(self.items) > self.budget:
            self._evict()

    def _evict(self) -> None:
        """Вытеснение. По умолчанию FIFO — самая старая запись."""
        self.items.pop(0)

    def _candidates(self, situation: int) -> List[Item]:
        """
        Записи того же типа ситуации.
        ---

        Args:
            situation (int): Тип ситуации текущего запроса.

        Returns:
            List[Item]: Кандидаты в порядке записи.
        """
        return [it for it in self.items if it.situation == situation]

    def _emb_of(self, cand: List[Item]) -> np.ndarray:
        """
        Эмбеддинги кандидатов. Берутся из закешированного пула — энкодер здесь не запускается.

        Args:
            cand (List[Item]): Кандидаты.

        Returns:
            np.ndarray: (len(cand), dim).
        """
        return self.emb[[it.text_id for it in cand]]

    def retrieve(self, query_emb: np.ndarray, situation: int, k: int,
                 project: Optional[int] = None) -> List[Item]:
        """
        Извлечение не более k записей.
        ---

        Args:
            query_emb (np.ndarray): Эмбеддинг текущего запроса (dim,), нормированный.
            situation (int): Тип ситуации текущего запроса.
            k (int): Сколько записей вернуть.
            project (Optional[int]): Скрытое поле. Читает ТОЛЬКО Oracle; остальные обязаны игнорировать.

        Returns:
            List[Item]: Не более k записей той же ситуации.
        """
        return self._select(query_emb, self._candidates(situation), k)

    @abstractmethod
    def _select(self, query_emb: np.ndarray, cand: List[Item], k: int) -> List[Item]:
        """
        Правило отбора k записей из кандидатов. Единственное, что различает бейзлайны.

        Args:
            query_emb (np.ndarray): Эмбеддинг запроса.
            cand (List[Item]): Кандидаты той же ситуации, в порядке записи.
            k (int): Сколько вернуть.

        Returns:
            List[Item]: Отобранные записи.
        """
        raise NotImplementedError


class NoMemory(MemoryPolicy):
    """
    Пол эксперимента: памяти нет. Точность должна лечь на 1/candidates_len.
    """

    def write(self, item: Item) -> None:
        pass

    def _select(self, query_emb: np.ndarray, cand: List[Item], k: int) -> List[Item]:
        return []


class Recency(MemoryPolicy):
    """
    Последние k записей. Простой способ работы с историей из ТЗ.
    """

    def _select(self, query_emb: np.ndarray, cand: List[Item], k: int) -> List[Item]:
        return cand[-k:]


class RandomK(MemoryPolicy):
    """
    Случайные k записей.

    Закрывает возражение «работает сам факт наличия примеров в контексте,
    а не то, какие именно примеры извлечены».
    """

    def __init__(self, budget: int, emb: np.ndarray, rng: NP_RNG):
        """
        Args:
            rng (NP_RNG): Отдельный генератор. Свой, а не генератор эпизода:
                иначе политика сдвигала бы поток и ломала воспроизводимость.
        """
        super().__init__(budget, emb)
        self.rng = rng

    def _select(self, query_emb: np.ndarray, cand: List[Item], k: int) -> List[Item]:
        if len(cand) <= k:
            return cand
        idx = self.rng.choice(len(cand), size=k, replace=False)
        return [cand[i] for i in idx]


class Semantic(MemoryPolicy):
    """
    Top-k по косинусной близости. Главный оппонент нашего метода.

    Эмбеддинги нормированы, поэтому скалярное произведение и есть косинус.
    """

    def _select(self, query_emb: np.ndarray, cand: List[Item], k: int) -> List[Item]:
        if not cand:
            return []
        sims = self._emb_of(cand) @ query_emb
        top  = np.argsort(-sims, kind="stable")[:k]
        return [cand[i] for i in top]


class FeatureKMeans(MemoryPolicy):
    """
    Извлечение через кластеры в пространстве эмбеддингов.

    Кандидат оценивается не сам по себе, а по близости запроса к центроиду его кластера.
    """

    def __init__(self, budget: int, emb: np.ndarray, n_clusters: int,
                 refit_every: int, seed: int = 0):
        """
        Args:
            n_clusters (int): Число кластеров.
            refit_every (int): Через сколько записей пересчитывать центроиды.
            seed (int): Сид для KMeans, ради воспроизводимости.
        """
        super().__init__(budget, emb)
        self.n_clusters  = n_clusters
        self.refit_every = refit_every
        self.seed        = seed
        self.centroids: Optional[np.ndarray] = None
        self._writes     = 0

    def reset(self) -> None:
        super().reset()
        self.centroids = None
        self._writes   = 0

    def write(self, item: Item) -> None:
        super().write(item)
        self._writes += 1
        if self._writes % self.refit_every == 0:
            self._refit()

    def _refit(self) -> None:
        """
        Пересчёт центроидов по всем хранимым записям.
        """
        n = len(self.items)
        if n < self.n_clusters:
            self.centroids = None
            return
        E = self._emb_of(self.items)
        km = SKKMeans(n_clusters=self.n_clusters, n_init=10, random_state=self.seed)
        km.fit(E)
        c = km.cluster_centers_
        self.centroids = c / np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-12)

    def _select(self, query_emb: np.ndarray, cand: List[Item], k: int) -> List[Item]:
        if not cand:
            return []
        E = self._emb_of(cand)
        sims = E @ query_emb
        if self.centroids is None:
            return [cand[i] for i in np.argsort(-sims, kind="stable")[:k]]

        label = np.argmax(E @ self.centroids.T, axis=1)   # ближайший центроид кандидата
        score = (self.centroids @ query_emb)[label]

        order = np.lexsort((-sims, -score))
        return [cand[i] for i in order[:k]]


class Oracle(MemoryPolicy):
    """
    Потолок эксперимента - извлекает записи истинного проекта.
    Единственная политика с доступом к скрытому полю.
    """

    def __init__(self, budget: int, emb: np.ndarray, meta: np.ndarray):
        """
        Args:
            meta (np.ndarray): (N, 4) — project, situation, topic, variant. Скрытая разметка.
        """
        super().__init__(budget, emb)
        self.meta = meta

    def retrieve(self, query_emb: np.ndarray, situation: int, k: int, project: Optional[int] = None) -> List[Item]:
        if project is None:
            raise ValueError("Oracle требует project текущего запроса")
        same = [it for it in self._candidates(situation)
                if int(self.meta[it.text_id, 0]) == project]
        return same[-k:]

    def _select(self, query_emb: np.ndarray, cand: List[Item], k: int) -> List[Item]:
        raise NotImplementedError("Oracle переопределяет retrieve целиком")

class DecisionEviction(Semantic):
    """
    Вытеснение записей на границе забывания.
    ---

    Удаляется запись с наибольшей долей согласных соседей.
    """

    def __init__(self, budget: int, emb: np.ndarray, n_neighbours: Optional[int] = None):
        """
        Args:
            budget (int): Бюджет памяти.
            emb (np.ndarray): Пул эмбеддингов; исходный или спроецированный.
            n_neighbours (Optional[int]): По скольким соседям считать согласие.
        """
        super().__init__(budget, emb)
        self.n_neighbours = (CONFIG["eviction"]["n_neighbors"] if n_neighbours is None else n_neighbours)

    def _evict(self) -> None:
        situation = self.items[-1].situation
        idx = [n for n, it in enumerate(self.items) if it.situation == situation]

        n_nb = min(self.n_neighbours, len(idx) - 1)
        if n_nb < 1:
            self.items.pop(0)               # соседей нет
            return

        E = self.emb[[self.items[n].text_id for n in idx]]
        C = E @ E.T                          # косинусные сходства внутри пула
        np.fill_diagonal(C, -np.inf)         # запись не может быть себе соседом

        neighbours = np.argsort(-C, axis=1)[:, :n_nb]
        actions    = np.array([self.items[n].revealed_action for n in idx])
        agreement  = (actions[neighbours] == actions[:, None]).mean(axis=1)

        # запись с самой большой долей соседей с идентичным решением
        best = np.where(agreement == agreement.max())[0]

        # при равенстве выбор из самой крупной по действию группы
        if len(best) > 1:
            counts = {int(a): int((actions == a).sum()) for a in np.unique(actions[best])}
            top    = max(counts.values())
            best   = np.array([b for b in best if counts[int(actions[b])] == top])

        # при равенстве удаляется самая старая из выбранных
        self.items.pop(idx[int(best.min())])


def make_policy(name: str, budget: int, emb: np.ndarray, meta: np.ndarray, 
                rng: NP_RNG, n_clusters: int = 8, refit_every: int = 50,
                seed: int = 0) -> MemoryPolicy:
    """
    Фабрика политик по имени из конфига.

    Args:
        name (str): nomemory | recency | random | semantic | kmeans | oracle.
        budget (int): Бюджет памяти, одинаковый для всех.
        emb (np.ndarray): Эмбеддинги пула.
        meta (np.ndarray): Скрытая разметка; передаётся только Oracle.
        rng (NP_RNG): Генератор для RandomK.

    Returns:
        MemoryPolicy: Готовая политика.
    """
    name = name.lower()
    if name == "nomemory":
        return NoMemory(budget, emb)
    if name == "recency":
        return Recency(budget, emb)
    if name == "random":
        return RandomK(budget, emb, rng)
    if name == "semantic":
        return Semantic(budget, emb)
    if name == "kmeans":
        return FeatureKMeans(budget, emb, n_clusters, refit_every, seed)
    if name == "oracle":
        return Oracle(budget, emb, meta)
    if name == "eviction":
        return DecisionEviction(budget, emb)
    raise ValueError(f"неизвестная политика: {name}")
