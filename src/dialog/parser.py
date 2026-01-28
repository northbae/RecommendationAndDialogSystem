import re
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import Levenshtein

from .linguistic_variable import LengthTerm, ImportanceTerm


@dataclass
class ParsedQuery:
    intent: str
    filters: Dict[str, Any]
    exclusions: Dict[str, Any]
    original_query: str
    domain_entity: Optional[str] = None
    target_id: Optional[int] = None
    sentiment: Optional[str] = None


class HybridParser:
    """
    Парсер на основе правил и расстояния Левенштейна.
    """

    def __init__(self):
        print("💡 Инициализация парсера (Левенштейн)...")
        self._init_knowledge_base()
        self._init_regex()
        print("✅ Парсер готов!")

    def _init_knowledge_base(self):
        """База знаний ключевых слов и фраз для поиска."""
        self.knowledge_base: List[Tuple[str, Any, List[str]]] = [
            # --- ИНТЕНТЫ ---
            ("intent", "help", ["помощь", "справка", "что умеешь", "команды", "инструкция"]),
            ("intent", "help_domain", ["категории", "рубрики", "авторы", "статистика", "темы", "разделы"]),
            ("intent", "help_examples", ["пример", "образец", "как спросить"]),
            ("intent", "undo", ["назад", "отмени", "верни", "откатить", "предыдущий"]),
            ("intent", "reset", ["сброс", "сначала", "заново", "очисти"]),
            ("intent", "recommend_personal", ["подойдет", "для меня", "на мой вкус", "персональные"]),
            ("intent", "recommend_similar", ["похожие", "подобные", "аналогичные"]),
            ("intent", "search", ["найди", "покажи", "поищи", "выведи", "хочу", "статьи", "новости"]),

            # --- ФИЛЬТРЫ ---
            ("length", LengthTerm.VERY_SHORT, ["очень короткие", "микро", "крошечные"]),
            ("length", LengthTerm.SHORT, ["короткие", "небольшие", "маленькие", "быстрые"]),
            ("length", LengthTerm.MEDIUM, ["средние", "обычные", "нормальные"]),
            ("length", LengthTerm.LONG, ["длинные", "большие", "подробные", "медленные"]),
            ("length", LengthTerm.VERY_LONG, ["очень длинные", "огромные", "лонгриды"]),

            ("importance", ImportanceTerm.HIGH, ["важные", "главные", "топ"]),

            ("date", "DATE_TODAY", ["сегодня", "за день", "свежие"]),
            ("date", "DATE_YESTERDAY", ["вчера"]),
            ("date", "DATE_WEEK", ["за неделю", "недельные"]),
            ("date", "DATE_MONTH", ["за месяц", "месячные"]),

            ("media", "MEDIA_VIDEO", ["с видео", "видеороликом"]),
            ("media", "MEDIA_IMAGE", ["с картинками", "фотографиями", "изображениями"]),

            ("category", "Спорт", ["спорт", "футбол", "хоккей"]),
            ("category", "Экономика", ["экономика", "бизнес", "финансы"]),
            ("category", "Политика", ["политика"]),
            ("category", "Технологии", ["технологии", "it", "ии"]),
            ("category", "Общество", ["общество", "культура", "искусство"]),

            ("author", "Иванов Петр", ["иванова", "иванов"]),
            ("author", "Морозов Андрей", ["морозова", "морозов"]),
        ]

    def _init_regex(self):
        """Regex только для служебных целей"""
        self.like_pattern = re.compile(r'\b(нравится|понравилась|хорошая|супер)\b', re.IGNORECASE)
        self.dislike_pattern = re.compile(r'\b(не\s+нравится|не\s+понравилась|плохая)\b', re.IGNORECASE)
        self.exclusion_pattern = re.compile(r'^\s*(не|кроме|без|исключая)\s*$', re.IGNORECASE)
        self.id_pattern = re.compile(r'(?:стать[юие]|№)?\s*(\d+)', re.IGNORECASE)
        self.domain_question_pattern = re.compile(r'^\s*(какие|кто|что|список|перечисли)\s+', re.IGNORECASE)
        self.unclear_pattern = re.compile(r'^(?:ну\s|эээ|^\s*$)', re.IGNORECASE)
        self.offensive_pattern = re.compile(r'\b(тупой|дурак|идиот)\b', re.IGNORECASE)
        self.similar_pattern = re.compile(r'\b(похожие|подобные|аналогичные)\b', re.IGNORECASE)

    def _find_closest_match(self, token: str) -> Optional[Tuple[str, Any]]:
        """Ищет ближайшее слово/фразу в базе знаний"""
        best_match = None
        max_dist = 1 if len(token) <= 4 else 2
        min_dist = max_dist + 1

        for f_type, f_val, keywords in self.knowledge_base:
            for keyword in keywords:
                dist = Levenshtein.distance(token, keyword)
                if dist < min_dist:
                    min_dist = dist
                    best_match = (f_type, f_val)
        return best_match if best_match and min_dist <= max_dist else None

    def parse(self, query: str) -> ParsedQuery:
        query = query.strip().lower()

        # 1. Быстрые проверки
        if self.unclear_pattern.match(query) or len(query) < 3: return ParsedQuery("unclear", {}, {}, query)
        if self.offensive_pattern.search(query): return ParsedQuery("offensive", {}, {}, query)

        # 2. Извлечение ID и sentiment
        target_id_match = self.id_pattern.search(query)
        target_id = int(target_id_match.group(1)) if target_id_match else None
        sentiment = "dislike" if self.dislike_pattern.search(query) else (
            "like" if self.like_pattern.search(query) else None)

        if sentiment and target_id:
            return ParsedQuery("state_change", {}, {}, query, target_id=target_id, sentiment=sentiment)

        # 3. Приоритетная проверка на вопросы о домене
        if self.domain_question_pattern.match(query):
            # Проверяем, есть ли в этом вопросе фильтр. Если да - это поиск.
            temp_filters, _ = self._extract_filters(query)
            if not temp_filters:  # Если фильтров НЕТ, то это вопрос о домене
                domain_entity = "авторы" if "автор" in query or "пишет" in query else "категории"
                return ParsedQuery("help_domain", {}, {}, query, domain_entity=domain_entity)

        # 4. Извлечение фильтров и интентов
        filters, exclusions, intents = self._extract_all(query)

        # 5. Приоритет поиска похожих
        if self.similar_pattern.search(query) and target_id:
            return ParsedQuery("recommend_similar", {}, {}, query, target_id=target_id)

        # 6. ФИНАЛЬНОЕ РЕШЕНИЕ
        if filters or exclusions:
            return ParsedQuery("search", filters, exclusions, query, 1.0)

        if intents:
            best_intent = max(intents, key=intents.get)
            return ParsedQuery(best_intent, {}, {}, query, 0.9, target_id=target_id)

        return ParsedQuery("unknown", {}, {}, query, 0.0)

    def _extract_all(self, query: str) -> Tuple[Dict, Dict, Dict]:
        """Извлекает все сущности из запроса: фильтры, исключения и интенты"""
        words = query.split()
        tokens = words + [" ".join(words[i:i + 2]) for i in range(len(words) - 1)]

        intents = {}
        filters = {}
        exclusions = {}
        is_exclusion_zone = False

        for token in set(tokens):
            if self.exclusion_pattern.match(token):
                is_exclusion_zone = True
                continue

            match = self._find_closest_match(token)

            if match:
                f_type, f_val = match
                target_dict = exclusions if is_exclusion_zone else filters

                if f_type == "intent":
                    intents[f_val] = intents.get(f_val, 0) + 1
                else:
                    # --- ГЛАВНОЕ ИСПРАВЛЕНИЕ ---
                    if f_type in ["category", "media"]:
                        # Всегда создаем список
                        target_dict.setdefault(f_type, []).append(f_val)
                    else:
                        target_dict[f_type] = f_val

        return filters, exclusions, intents

    def _extract_filters(self, query: str) -> Tuple[Dict, Dict]:
        """Быстро проверяет наличие фильтров в запросе"""
        filters, exclusions, _ = self._extract_all(query)
        return filters, exclusions