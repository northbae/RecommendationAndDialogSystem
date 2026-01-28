import pandas as pd
import numpy as np
from typing import List, Dict, Any
from src.recommender.dislikes_recommender import recommend_with_dislikes


def hybrid_recommendation_search(
        df: pd.DataFrame,
        similarity_df: pd.DataFrame,
        liked_ids: List[int],
        disliked_ids: List[int],
        filters: Dict[str, Any],
        top_n: int = 10
) -> pd.DataFrame:
    result_df = df.copy()

    if liked_ids or disliked_ids:
        try:
            recs = recommend_with_dislikes(
                liked_articles=liked_ids,
                disliked_articles=disliked_ids,
                similarity_df=similarity_df,
                df=df,
                n=len(df),
                exclude_seen=False
            )
            score_map = dict(zip(recs['article_id'], recs['total_score']))

            result_df['rec_score'] = result_df['article_id'].map(score_map).fillna(0)
        except ValueError:
            result_df['rec_score'] = 0.0
    else:
        result_df['rec_score'] = 0.0

    mask = pd.Series([True] * len(result_df))

    if filters.get('categories'):
        mask &= result_df['category'].isin(filters['categories'])

    if filters.get('authors'):
        mask &= result_df['author'].isin(filters['authors'])

    if filters.get('tags'):
        mask &= result_df['tags'].apply(lambda x: any(tag in x for tag in filters['tags']))

    if filters.get('length_range'):
        min_l, max_l = filters['length_range']
        if 'content_length' in result_df.columns:
            mask &= (result_df['content_length'] >= min_l) & (result_df['content_length'] <= max_l)

    filtered_df = result_df[mask].copy()

    if not filtered_df.empty:
        filtered_df = filtered_df.sort_values(by=['rec_score', 'published_at'], ascending=[False, False])
        return filtered_df.head(top_n)

    # Если результат пуст, ищем "похожее", но добавляем rec_score
    else:
        # Начисляем очки за частичные совпадения фильтров
        filter_score = pd.Series([0.0] * len(result_df), index=result_df.index)

        if filters.get('categories'):
            filter_score += result_df['category'].isin(filters['categories']).astype(float) * 2.0

        if filters.get('authors'):
            filter_score += result_df['author'].isin(filters['authors']).astype(float) * 1.5

        # Utility = (Rec_Score * 2) + Filter_Partial_Score
        # Вес рекомендаций берем побольше
        result_df['utility_score'] = (result_df['rec_score'] * 1.0) + filter_score

        # Возвращаем только те, где есть хоть какое-то совпадение по фильтрам или высокий рейтинг
        final_res = result_df[result_df['utility_score'] > 0].sort_values('utility_score', ascending=False)
        return final_res.head(top_n)