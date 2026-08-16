import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns
from typing import Dict

from src import load_pool, cosine_table

def visualize(cells: Dict, topic_gap: float, project_gap: float, trap: float) -> None:
    """
    Отрисовка графиков для оценки похожести текстов.
    ---

    Визуализируются:
    * тепловая карта косинусных средних расстояний для таблицы из `src.utils.cosine_table`,
    * столбчатые диаграммы для остальных результатов из `src.utils.cosine_table`.

    Args:
        cells (Dict): см. `src.utils.cosine_table`
        topic_gap (float): см. `src.utils.cosine_table`
        project_gap (float): см. `src.utils.cosine_table`
        trap (float): см. `src.utils.cosine_table`
    """
    fig = plt.figure(figsize=(18, 6))
    
    # ---- 1. Тепловая карта ----
    ax1 = plt.subplot(1, 3, 1)
    rows = ['та же тема', 'другая тема']
    cols = ['тот же проект', 'другой проект']
    matrix = np.array([[cells[(row, col)] for col in cols] for row in rows])
    
    vmin = min(matrix.flatten()) - 0.01
    vmax = max(matrix.flatten()) + 0.01
    
    sns.heatmap(matrix, 
                annot=True,
                fmt='.3f',
                cmap='RdBu_r',
                vmin=vmin,
                vmax=vmax,
                xticklabels=cols,
                yticklabels=rows,
                square=True,
                cbar_kws={'label': 'Косинусное сходство'},
                annot_kws={'size': 11, 'weight': 'bold'},
                ax=ax1)
    ax1.set_title('Сходство по темам и проектам', fontsize=13)
    
    # ---- 2. Gap-метрики ----
    ax2 = plt.subplot(1, 3, 2)
    gap_metrics = {'topic_gap': topic_gap, 'proj_gap': project_gap}
    colors = ['#2E86AB', '#A23B72']
    bars = ax2.bar(gap_metrics.keys(), gap_metrics.values(), 
                   color=colors, edgecolor='black', linewidth=1.5)
    
    for bar, value in zip(bars, gap_metrics.values()):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                f'{value:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.set_ylabel('Разность косинусного сходства', fontsize=11)
    ax2.set_title('Gap-метрики', fontsize=13)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)
    
    # ---- 3. Trap-метрика ----
    ax3 = plt.subplot(1, 3, 3)
    bar = ax3.bar('trap', trap, color='#F18F01', edgecolor='black', linewidth=1.5)
    ax3.text(0, trap + 0.01, f'{trap:.4f}',
            ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax3.set_ylim(0.5, 1.0)
    ax3.set_ylabel('Косинусное сходство', fontsize=11)
    ax3.set_title('Сходство одинаковых текстов\nиз разных проектов', fontsize=13)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    ax3.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig('results/images/combined_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def save_df(cells: Dict, topic_gap: Dict, project_gap: Dict, trap: Dict) -> None:
    """
    Сохранение данных о косинусных сходствах
    ---

    Args:
        cells (Dict): см. `src.utils.cosine_table`
        topic_gap (Dict): см. `src.utils.cosine_table`
        project_gap (Dict): см. `src.utils.cosine_table`
        trap (Dict): см. `src.utils.cosine_table`
    """
    rows = ['та же тема', 'другая тема']
    cols = ['тот же проект', 'другой проект']
    matrix = np.array([[cells[(row, col)] for col in cols] for row in rows])
    df_cells = pd.DataFrame(matrix, index=rows, columns=cols)
    df_cells.to_csv("results/csv/embeddings_similarity.csv")

    df_metrics = pd.DataFrame({
        'topic_gap': [topic_gap],
        'proj_gap': [project_gap],
        'trap': [trap]
    })
    df_metrics.to_csv("results/csv/cosine_metrics.csv", index=False)

if __name__ == '__main__':
    texts, meta, emb, index = load_pool('data')

    result = cosine_table(emb, meta, 0)
    cells = result['cells']

    out = Path('results/')
    out.mkdir(parents=True, exist_ok=True)

    (out / 'csv').mkdir(parents=True, exist_ok=True)
    (out / 'images').mkdir(parents=True, exist_ok=True)

    cells   = result['cells']
    topic   = result['topic_gap']
    project = result['proj_gap']
    trap    = result['trap']

    visualize(cells, topic, project, trap)
    save_df(cells, topic, project, trap)
