import datetime
from enum import Enum
from typing import Dict, Callable

class LengthTerm(Enum):
    VERY_SHORT = "очень_короткая"
    SHORT = "короткая"
    MEDIUM = "средняя"
    LONG = "длинная"
    VERY_LONG = "очень_длинная"

class ImportanceTerm(Enum):
    LOW = "второстепенная"
    HIGH = "важная"
    CRITICAL = "срочная_главная"

class FreshnessTerm(Enum):
    FRESH = "свежие"
    OLD = "архивные"

TERM_LABELS = {
    LengthTerm.VERY_SHORT: "Очень короткая",
    LengthTerm.SHORT: "Короткая",
    LengthTerm.MEDIUM: "Средняя",
    LengthTerm.LONG: "Длинная",
    LengthTerm.VERY_LONG: "Очень длинная",
    ImportanceTerm.HIGH: "Важная",
}

def _trapmf(x, a, b, c, d):
    return max(min((x - a) / (b - a) if b != a else 1.0, 1, (d - x) / (d - c) if d != c else 1.0), 0)

def _l_function(x, a, b):
    if x <= a: return 1.0
    if x >= b: return 0.0
    return (b - x) / (b - a)

def _r_function(x, a, b):
    if x <= a: return 0.0
    if x >= b: return 1.0
    return (x - a) / (b - a)

def get_length_membership(x: float, term: LengthTerm) -> float:
    if term == LengthTerm.VERY_SHORT: return _l_function(x, 50, 150)
    elif term == LengthTerm.SHORT: return _trapmf(x, 50, 150, 300, 500)
    elif term == LengthTerm.MEDIUM: return _trapmf(x, 300, 500, 1300, 1700)
    elif term == LengthTerm.LONG: return _trapmf(x, 1300, 1700, 3500, 4500)
    elif term == LengthTerm.VERY_LONG: return _r_function(x, 3500, 4500)
    return 0.0

def get_importance_membership(val: float, term: ImportanceTerm) -> float:
    if term == ImportanceTerm.LOW: return _l_function(val, 20, 60)
    elif term == ImportanceTerm.HIGH: return _trapmf(val, 100, 140, 180, 220)
    return 0.0

def get_freshness_membership(date_obj: datetime.datetime, term: FreshnessTerm, now: datetime.datetime) -> float:
    delta = (now - date_obj).total_seconds() / 3600
    if delta < 0: delta = 0
    if term == FreshnessTerm.FRESH: return _l_function(delta, 24, 48)
    elif term == FreshnessTerm.OLD: return _r_function(delta, 72, 168)
    return 0.0