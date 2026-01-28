"""
Рекомендательная система новостных статей
"""

__version__ = "1.0.0"
__author__ = "Elena"

# Экспортируем основные классы для удобства
from .data.dataset import NewsDataset
from .features.similarity_metrics import SimilarityCalculator
from .recommender.filtering import get_similar_articles
from .recommender.likes_recommender import recommend_from_likes
from .recommender.dislikes_recommender import recommend_with_dislikes

__all__ = [
    'NewsDataset',
    'SimilarityCalculator',
    'get_similar_articles',
    'recommend_from_likes',
    'recommend_with_dislikes'
]