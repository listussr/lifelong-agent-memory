import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Tuple

from src import NP_RNG, make_rng, CONFIG
from src.enviroment import (
    TOPICS,
    TOPIC_NAMES,
    PROJECTS,
    SITUATIONS,
    SITUATION_SLOTS,
    CONTEXT_FRAMES,
    TAIL_FRAMES,
    topics_len,
    situation_len,
)

def pick_slots(situation: int, rng: NP_RNG) -> Dict:
    """
    Сэмлплирование случайного названия технологии для контекста.
    ---

    Args:
        situation (int): Номер ситуации.
        rng (NP_RNG): Numpy генератор случайных чисел.

    Returns:
        Dict: Словарь с ключом для вставки и значением - названием технологии.
    """
    return {
        name: rng.choice(values)
        for name, values in SITUATION_SLOTS[situation].items()
    }

def render_body(situation: int, topic: int, rng: NP_RNG, n_frames: int = 3) -> str:
    """
    Генерация тела текста для проведения исследования.
    ---

    Генерация происходит по шаблону '{контекст}. Сейчас {задача}. {вопрос как сделать}'.

    Пайплайн генерации:
     * выбираются 3 случайных технологии для контекста (контекст уже задан),
     * выбранные технологии вставляются в контекстный шаблон,
     * случайным образом выбирается задача под заданную ситуацию,
     * случайным образом выбирается вопрос о том, как выполнить данную задачу.

    Args:
        situation (int): Id ситуации.
        topic (int): Id темы 
        rng (NP_RNG): Генератор случайных чисел в numpy.
        n_frames (int): Количество рамок для засорения контекстом видимости энкодера. Defaults to 3.

    Returns:
        str: Сгенерированный текст.
    """
    words = TOPICS[TOPIC_NAMES[topic]]
    widx  = rng.choice(len(words), size=3 * n_frames, replace=False)
    fidx  = rng.choice(len(CONTEXT_FRAMES), size=n_frames, replace=False)
    context = " ".join(
        CONTEXT_FRAMES[f].format(w1=words[widx[3*j]],
                                 w2=words[widx[3*j+1]],
                                 w3=words[widx[3*j+2]])
        for j, f in enumerate(fidx)
    )
    task       = rng.choice(SITUATIONS[situation]).format(**pick_slots(situation, rng))
    tail       = rng.choice(TAIL_FRAMES)
    return f"{context} Сейчас {task}. {tail}"

def render(project_name: str, body: str) -> str:
    """
    Генерация полного текста с названием репозитория.
    ---

    Args:
        project_name (str): Название репозитория.
        body (str): Сгенерированное тело текста.

    Returns:
        str: Конечный текст.
    """
    return f"[repo: {project_name}] {body}"

def generate_all(rng: NP_RNG, variants: int) -> Tuple[List[str], np.ndarray, Dict]:
    """
    Генерация текстов всех вариантов.
    ---

    Args:
        rng (NP_RNG): Numpy генератор случайных чисел.
        variants (int): Количество вариантов генерации.

    Returns:
        Tuple[List[str], np.ndarray, Dict]: Список текстов, np массив метаданных генерации, индекс текстов
    """
    texts, meta, index = [], [], {}
    for situation in range(situation_len):
        for topic in range(topics_len):
            for variant in range(variants):
                body = render_body(situation, topic, rng, n_frames = CONFIG['env']['context_frames_per_text'])
                for project, name in PROJECTS.items():
                    index.setdefault((project, situation, topic), []).append(len(texts))
                    texts.append(render(name, body))
                    meta.append((project, situation, topic, variant))
    return texts, np.array(meta, dtype=np.int16), index

def encode_texts(texts: List[str]) -> np.ndarray:
    """
    Перевод текстов в пространство эмбеддингов.
    ---

    Args:
        texts (List[str]): Список текстов.

    Returns:
        np.ndarray: Эмбеддинги всего набора текстов.
    """
    model = SentenceTransformer(CONFIG["encoder"]["name"], device=CONFIG["encoder"]["device"])
    emb = model.encode(
        texts,
        batch_size=CONFIG["encoder"]["batch_size"],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return emb

def save_pool(out_dir: str, texts: List[str], meta: np.ndarray, index: Dict, emb: np.ndarray):
    """
    Сохранение сгенерированных данных.
    ---

    Args:
        out_dir (str): Выходная папка.
        texts (List[str]): Список текстов.
        meta (np.ndarray): Список метаданных.
        index (Dict): Словарь с метаданными: индексами текстов.
        emb (np.ndarray): Список эмбеддингов.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    (out / "pool_texts.json").write_text(
        json.dumps(texts, ensure_ascii=False),
        encoding="utf-8"
    )
    np.save(out / "pool_meta.npy", meta.astype(np.int16))
    np.save(out / "pool_emb.npy",  emb.astype(np.float32))
    (out / "pool_index.json").write_text(
        json.dumps({f"{p},{s},{t}": ids for (p, s, t), ids in index.items()}),
        encoding="utf-8"
    )

if __name__ == "__main__":
    rng = make_rng(CONFIG['seeds'][0])
    texts, meta, index = generate_all(rng, CONFIG['env']['variants_per_body'])
    embeddings = encode_texts(texts)
    save_pool(
        'data',
        texts,
        meta,
        index,
        embeddings,
    )