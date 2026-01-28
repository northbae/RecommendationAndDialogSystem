from typing import List, Optional
import json
from datetime import datetime
from pathlib import Path

from ...recommender.base import QueryHistory, Query


class QueryHistoryManager:

    def __init__(self, history_file: Optional[str] = None):
        self.history = QueryHistory()
        self.history_file = Path(history_file) if history_file else None

    def add_query(self, query: Query):
        self.history.add_query(query)

        if self.history_file:
            self.save()

    def save(self):
        if not self.history_file:
            return

        history_data = []
        for query in self.history.queries:
            history_data.append(query.to_dict())

        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False)

    def load(self):
        if not self.history_file or not self.history_file.exists():
            return

        with open(self.history_file, 'r', encoding='utf-8') as f:
            history_data = json.load(f)

    def format_history_list(self, n: int = 10) -> str:
        recent = self.history.get_last_n(n)

        if not recent:
            return "История пуста"

        lines = ["История запросов:", "=" * 60]

        for i, query in enumerate(recent, 1):
            marker = "➤" if query == self.history.get_current() else " "
            timestamp = query.timestamp.strftime('%d.%m %H:%M')
            lines.append(f"{marker} {i}. [{timestamp}] {query.describe()}")

        lines.append("=" * 60)
        lines.append(f"Позиция: {self.history.current_position + 1}/{len(self.history)}")
        lines.append(f"Можно откатиться: {'Да' if self.history.can_undo() else 'Нет'}")

        return "\n".join(lines)

    def undo(self) -> Optional[Query]:
        return self.history.undo()

    def redo(self) -> Optional[Query]:
        return self.history.redo()

    def clear(self):
        self.history.clear()

        if self.history_file and self.history_file.exists():
            self.history_file.unlink()

    def get_statistics(self) -> dict:
        if not self.history.queries:
            return {
                'total_queries': 0,
                'unique_types': 0,
                'first_query': None,
                'last_query': None
            }

        query_types = [q.__class__.__name__ for q in self.history.queries]

        return {
            'total_queries': len(self.history.queries),
            'unique_types': len(set(query_types)),
            'first_query': self.history.queries[0].timestamp,
            'last_query': self.history.queries[-1].timestamp,
            'query_types': dict((t, query_types.count(t)) for t in set(query_types))
        }


class HistoryViewer:
    @staticmethod
    def display_query_details(query: Query) -> str:
        lines = [
            "=" * 60,
            f"Тип запроса: {query.__class__.__name__}",
            f"Время: {query.timestamp.strftime('%d.%m.%Y %H:%M:%S')}",
            f"Описание: {query.describe()}",
            "",
            "Параметры:",
        ]

        for key, value in query.params.items():
            lines.append(f"  {key}: {value}")

        lines.append("=" * 60)

        return "\n".join(lines)

    @staticmethod
    def display_timeline(queries: List[Query]) -> str:
        if not queries:
            return "История пуста"

        lines = ["Временная шкала запросов:", ""]

        for i, query in enumerate(queries):
            time_str = query.timestamp.strftime('%H:%M')
            desc = query.describe()[:40]

            lines.append(f"  {time_str}  ●───  {desc}")

            if i < len(queries) - 1:
                lines.append("         │")

        return "\n".join(lines)