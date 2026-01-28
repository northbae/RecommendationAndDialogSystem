"""
Модуль вычисления признаков и мер сходства
"""

from .similarity_metrics import SimilarityCalculator
from .preprocess import FeaturePreprocessor
from .tree_distance import build_category_graph, compute_tree_distance_matrix

__all__ = [
    'SimilarityCalculator',
    'FeaturePreprocessor',
    'build_category_graph',
    'compute_tree_distance_matrix'
]