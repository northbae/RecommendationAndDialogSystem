import pandas as pd
from .parser import HybridParser
from .search import FuzzySearchEngine
from .linguistic_variable import TERM_LABELS



class DialogState:
    query: str
    intent: str
    filters: dict
    exclusions: dict


class SmartDialogManager:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.parser = HybridParser()
        self.engine = FuzzySearchEngine(df)
        self.liked_articles = set()
        self.disliked_articles = set()

    def process(self, text: str):
        parsed = self.parser.parse(text)

        # Управление состоянием
        if parsed.intent == "reset": return "SIGNAL_RESET"
        if parsed.intent == "undo": return "SIGNAL_UNDO"
        if parsed.intent == "state_change":
            if parsed.sentiment == "like":
                self.liked_articles.add(parsed.target_id)
                return f" Запомнил, что вам понравилась статья #{parsed.target_id}."
            elif parsed.sentiment == "dislike":
                self.disliked_articles.add(parsed.target_id)
                return f" Понял. Не буду рекомендовать похожее на #{parsed.target_id}."

        # Рекомендации
        if parsed.intent == "recommend_personal":
            if self.disliked_articles:
                res = self.engine.find_dissimilar(list(self.disliked_articles))
                return self._format_list(res, "Избегая того, что вам не понравилось:")
            elif self.liked_articles:
                res = self.engine.find_similar(list(self.liked_articles))
                return self._format_list(res, "Вам может понравиться:")
            return "Чтобы я мог что-то посоветовать, оцените статью."

        if parsed.intent == "recommend_similar":
            if parsed.target_id:
                res = self.engine.find_similar([parsed.target_id])
                return self._format_list(res, f" Похожие на #{parsed.target_id}:")
            return "Напишите ID статьи, например: *Похожие на 26*."

        # Поиск
        if parsed.intent == "search":
            res = self.engine.search(parsed.filters, parsed.exclusions)
            return self._format_list(res, " Результаты поиска:")

        # Помощь и болтовня
        if parsed.intent == "help": return " **Справка:**\n- *Найди короткие новости*\n- *Что мне подойдет?*\n- *Похожие на 26*"
        if parsed.intent == "help_domain":
            entity = parsed.domain_entity
            if entity == "категории": return f"**Рубрики:** {', '.join({c.split('/')[0] for c in self.df['category']})}"
            if entity == "авторы": return f"✍**Авторы:** {', '.join(self.df['author'].unique())}"
            return "Я знаю про категории и авторов."

        if parsed.intent == "out_of_domain": return "Только новости."
        if parsed.intent == "offensive": return ":("
        return "Не понял. Попробуйте: *Найди короткие новости*."

    def _format_list(self, results, title: str):
        if not results: return "По вашему запросу ничего не найдено."
        msg = f"**{title}**\n\n"
        for r in results:
            msg += f"- **[#{r.article_id}]** {r.title} (Длина: {r.details['len']})\n"
        return msg