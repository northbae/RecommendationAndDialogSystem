import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict

CATEGORY_TREE = {
    'ROOT': ['Политика', 'Экономика', 'Общество', 'Технологии', 'Спорт'],
    'Политика': ['Внутренняя политика', 'Международные отношения', 'Региональная политика'],
    'Внутренняя политика': ['Экономическая', 'Культурная', 'Социальная', 'Экологическая'],
    'Экономика': ['Макроэкономика', 'Бизнес', 'Финансы'],
    'Макроэкономика': ['ВВП', 'Инфляция', 'Курсы валют'],
    'Финансы': ['Банки', 'Фондовый рынок'],
    'Общество': ['Культура', 'Религия', 'Социальные вопросы'],
    'Культура': ['Искусство', 'Музыка', 'Театр', 'Кино'],
    'Технологии': ['Наука', 'IT'],
    'Спорт': ['Индивидуальные виды', 'Командные виды', 'Международные соревнования'],
    'Международные соревнования': ['Олимпиады', 'Чемпионаты мира', 'Кубки']
}


def build_category_graph() -> nx.Graph:
    G = nx.Graph()

    for parent, children in CATEGORY_TREE.items():
        for child in children:
            G.add_edge(parent, child, weight=1)

    return G


def get_leaf_node(category_path: str) -> str:
    return category_path.split('/')[-1]


def compute_tree_distance_matrix(df: pd.DataFrame) -> np.ndarray:
    G = build_category_graph()
    n = len(df)
    tree_dist_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            node1 = get_leaf_node(df.iloc[i]['category'])
            node2 = get_leaf_node(df.iloc[j]['category'])

            try:
                distance = nx.shortest_path_length(G, node1, node2)
                tree_dist_matrix[i, j] = distance
            except nx.NetworkXNoPath:
                tree_dist_matrix[i, j] = 10

    return tree_dist_matrix


def get_category_depth(category_path: str) -> int:
    return len(category_path.split('/'))


def get_common_ancestor(cat1: str, cat2: str) -> str:
    path1 = cat1.split('/')
    path2 = cat2.split('/')

    common = []
    for p1, p2 in zip(path1, path2):
        if p1 == p2:
            common.append(p1)
        else:
            break

    return '/'.join(common) if common else 'ROOT'