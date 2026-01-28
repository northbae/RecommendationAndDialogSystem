from .linguistic_variable import (
    LengthTerm,
    ImportanceTerm,
    FreshnessTerm,
    TERM_LABELS, # <-- ДОБАВЛЕНО
    get_length_membership,
    get_importance_membership,
    get_freshness_membership
)
from .parser import HybridParser, ParsedQuery
from .search import FuzzySearchEngine, SearchResult
from .dialog_manager import SmartDialogManager, DialogState

__all__ = [
    'LengthTerm', 'ImportanceTerm', 'FreshnessTerm',
    'TERM_LABELS', # <-- ДОБАВЛЕНО
    'get_length_membership', 'get_importance_membership', 'get_freshness_membership',
    'HybridParser', 'ParsedQuery',
    'FuzzySearchEngine', 'SearchResult',
    'SmartDialogManager', 'DialogState'
]