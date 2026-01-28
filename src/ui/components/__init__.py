"""
UI компоненты
"""

from .article_display import ArticleDisplayFormatter, ArticleTable
from .query_history import QueryHistoryManager, HistoryViewer

__all__ = [
    'ArticleDisplayFormatter',
    'ArticleTable',
    'QueryHistoryManager',
    'HistoryViewer'
]