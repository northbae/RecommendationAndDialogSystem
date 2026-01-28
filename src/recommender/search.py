import pandas as pd
from typing import Dict, Any, List, Tuple


def search_articles(
        df: pd.DataFrame,
        filters: Dict[str, Any]
) -> pd.DataFrame:
    """
    Точный параметрический поиск.
    """
    mask = pd.Series([True] * len(df))

    # Фильтр по категории (если выбрано)
    if filters.get('categories'):
        mask &= df['category'].isin(filters['categories'])

    # Фильтр по автору (если выбрано)
    if filters.get('authors'):
        mask &= df['author'].isin(filters['authors'])

    # Фильтр по тегам (если есть пересечение хотя бы по одному тегу)
    if filters.get('tags'):
        # Предполагаем, что tags в df это список строк или строка
        # Если это список:
        mask &= df['tags'].apply(lambda x: any(tag in x for tag in filters['tags']))

    # Фильтр по длине контента (диапазон)
    if filters.get('length_range'):
        min_l, max_l = filters['length_range']
        if 'content_length' in df.columns:
            mask &= (df['content_length'] >= min_l) & (df['content_length'] <= max_l)

    return df[mask]


def fuzzy_search_articles(
        df: pd.DataFrame,
        filters: Dict[str, Any],
        n: int = 5
) -> pd.DataFrame:
    #Бонусная задача: поиск "похожего", если точного совпадения нет.
    #Начисляет очки за каждое совпадение критерия.
    scores = pd.Series([0.0] * len(df), index=df.index)

    # Веса критериев
    w_category = 2.0
    w_author = 1.5
    w_tag = 1.0
    w_length = 0.5

    if filters.get('categories'):
        scores += df['category'].isin(filters['categories']).astype(float) * w_category

    if filters.get('authors'):
        scores += df['author'].isin(filters['authors']).astype(float) * w_author

    if filters.get('tags'):
        # Считаем сколько тегов совпало
        def count_tag_matches(article_tags):
            if not isinstance(article_tags, list): return 0
            return sum(1 for t in filters['tags'] if t in article_tags)

        tag_scores = df['tags'].apply(count_tag_matches)
        # Нормируем, чтобы много тегов не перебивали категорию слишком сильно
        if tag_scores.max() > 0:
            tag_scores = tag_scores / tag_scores.max()
        scores += tag_scores * w_tag

    if filters.get('length_range') and 'content_length' in df.columns:
        min_l, max_l = filters['length_range']
        # Если попадает в диапазон - добавляем очки
        len_match = (df['content_length'] >= min_l) & (df['content_length'] <= max_l)
        scores += len_match.astype(float) * w_length

    # Сортируем и берем топ N
    top_indices = scores.sort_values(ascending=False).head(n).index

    # Возвращаем результат, но только если score > 0
    result_df = df.loc[top_indices].copy()
    result_df['similarity_score'] = scores.loc[top_indices]

    return result_df[result_df['similarity_score'] > 0]