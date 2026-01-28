"""
Работа с деревом категорий
"""

from typing import Dict, List, Optional, Tuple
import networkx as nx


class CategoryTree:
    """
    Класс для работы с деревом категорий новостей
    """

    # Определение структуры дерева
    TREE_STRUCTURE = {
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

    def __init__(self):
        self.tree = self.TREE_STRUCTURE
        self.graph = self._build_graph()
        self._all_nodes = None
        self._leaf_nodes = None

    def _build_graph(self) -> nx.Graph:
        """Построить граф из дерева"""
        G = nx.Graph()

        for parent, children in self.tree.items():
            for child in children:
                G.add_edge(parent, child, weight=1)

        return G

    def get_all_nodes(self) -> List[str]:
        """Получить все узлы дерева"""
        if self._all_nodes is None:
            self._all_nodes = list(self.graph.nodes())
        return self._all_nodes

    def get_leaf_nodes(self) -> List[str]:
        """Получить все листовые узлы (конечные категории)"""
        if self._leaf_nodes is None:
            self._leaf_nodes = [
                node for node in self.graph.nodes()
                if self.graph.degree(node) == 1 and node != 'ROOT'
            ]
        return self._leaf_nodes

    def get_children(self, node: str) -> List[str]:
        """Получить дочерние узлы"""
        return self.tree.get(node, [])

    def get_parent(self, node: str) -> Optional[str]:
        """Получить родительский узел"""
        for parent, children in self.tree.items():
            if node in children:
                return parent
        return None

    def get_path_to_root(self, node: str) -> List[str]:
        """Получить путь от узла до корня"""
        path = [node]
        current = node

        while current != 'ROOT':
            parent = self.get_parent(current)
            if parent is None:
                break
            path.append(parent)
            current = parent

        return path[::-1]  # от корня к узлу

    def get_distance(self, node1: str, node2: str) -> int:
        """
        Вычислить расстояние между узлами

        Parameters
        ----------
        node1, node2 : str
            Узлы дерева

        Returns
        -------
        int
            Расстояние (количество рёбер)
        """
        try:
            return nx.shortest_path_length(self.graph, node1, node2)
        except (nx.NodeNotFound, nx.NetworkXNoPath):
            return float('inf')

    def get_common_ancestor(self, node1: str, node2: str) -> str:
        """
        Найти ближайшего общего предка

        Parameters
        ----------
        node1, node2 : str
            Узлы дерева

        Returns
        -------
        str
            Общий предок
        """
        path1 = self.get_path_to_root(node1)
        path2 = self.get_path_to_root(node2)

        # Находим последний общий узел
        common = 'ROOT'
        for p1, p2 in zip(path1, path2):
            if p1 == p2:
                common = p1
            else:
                break

        return common

    def get_level(self, node: str) -> int:
        """
        Получить уровень узла в дереве

        Parameters
        ----------
        node : str
            Узел дерева

        Returns
        -------
        int
            Уровень (0 для ROOT)
        """
        if node == 'ROOT':
            return 0

        path = self.get_path_to_root(node)
        return len(path) - 1

    def parse_category_path(self, path: str) -> Tuple[str, ...]:
        """
        Разобрать путь категории

        Parameters
        ----------
        path : str
            Путь типа 'Политика/Внутренняя политика/Экономическая'

        Returns
        -------
        tuple
            Кортеж узлов
        """
        return tuple(path.split('/'))

    def get_leaf_from_path(self, path: str) -> str:
        """Извлечь листовой узел из пути"""
        return path.split('/')[-1]

    def validate_path(self, path: str) -> bool:
        """
        Проверить, является ли путь корректным

        Parameters
        ----------
        path : str
            Путь категории

        Returns
        -------
        bool
            True если путь корректный
        """
        nodes = self.parse_category_path(path)

        # Проверяем, что все узлы существуют
        for node in nodes:
            if node not in self.get_all_nodes():
                return False

        # Проверяем связность
        for i in range(len(nodes) - 1):
            if nodes[i + 1] not in self.get_children(nodes[i]):
                return False

        return True

    def get_siblings(self, node: str) -> List[str]:
        parent = self.get_parent(node)
        if parent:
            siblings = self.get_children(parent)
            return [s for s in siblings if s != node]
        return []

    def print_tree(self, node: str = 'ROOT', indent: int = 0):
        print("  " * indent + node)

        children = self.get_children(node)
        for child in children:
            self.print_tree(child, indent + 1)

    def to_dict(self) -> Dict:
        return self.tree.copy()

    def get_statistics(self) -> Dict:
        return {
            'total_nodes': len(self.get_all_nodes()),
            'leaf_nodes': len(self.get_leaf_nodes()),
            'max_depth': max(self.get_level(node) for node in self.get_all_nodes()),
            'root_children': len(self.get_children('ROOT'))
        }

    def __repr__(self):
        stats = self.get_statistics()
        return f"CategoryTree(nodes={stats['total_nodes']}, depth={stats['max_depth']}, leaves={stats['leaf_nodes']})"


category_tree = CategoryTree()