from abc import ABC, abstractmethod
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime


class Query(ABC):

    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.timestamp = datetime.now()
        self.results = None

    @abstractmethod
    def execute(self, df: pd.DataFrame, similarity_df: pd.DataFrame):
        pass

    @abstractmethod
    def describe(self) -> str:
        pass

    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.__class__.__name__,
            'params': self.params,
            'timestamp': self.timestamp.isoformat()
        }


class QueryHistory:

    def __init__(self):
        self.queries: List[Query] = []
        self.current_position = -1

    def add_query(self, query: Query):
        if self.current_position < len(self.queries) - 1:
            self.queries = self.queries[:self.current_position + 1]

        self.queries.append(query)
        self.current_position = len(self.queries) - 1

    def can_undo(self) -> bool:
        return self.current_position > 0

    def can_redo(self) -> bool:
        return self.current_position < len(self.queries) - 1

    def undo(self) -> Query:
        if self.can_undo():
            self.current_position -= 1
            return self.queries[self.current_position]
        return None

    def redo(self) -> Query:
        if self.can_redo():
            self.current_position += 1
            return self.queries[self.current_position]
        return None

    def get_current(self) -> Query:
        if 0 <= self.current_position < len(self.queries):
            return self.queries[self.current_position]
        return None

    def get_last_n(self, n: int = 5) -> List[Query]:
        return self.queries[max(0, len(self.queries) - n):]

    def clear(self):
        self.queries = []
        self.current_position = -1

    def __len__(self):
        return len(self.queries)