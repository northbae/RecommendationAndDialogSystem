import pandas as pd
import numpy as np
from typing import List, Optional


def recommend_with_dislikes(
        liked_articles: List[int],
        disliked_articles: List[int],
        similarity_df: pd.DataFrame,
        df: pd.DataFrame,
        n: int = 10,
        like_weight: float = 1.0,
        dislike_weight: float = 0.5,
        aggregation: str = 'mean',
        exclude_seen: bool = True
) -> pd.DataFrame:

    if not liked_articles and not disliked_articles:
        raise ValueError("нету лайков и дизлайков")

    valid_likes = [aid for aid in liked_articles if aid in similarity_df.index] if liked_articles else []
    valid_dislikes = [aid for aid in disliked_articles if aid in similarity_df.index] if disliked_articles else []

    if liked_articles and not valid_likes:
        raise ValueError("что то не то с id лайков")

    all_articles = similarity_df.index.tolist()

    if exclude_seen:
        seen_articles = set(valid_likes + valid_dislikes)
        candidates = [aid for aid in all_articles if aid not in seen_articles]
    else:
        candidates = all_articles

    if not candidates:
        raise ValueError("Не можем ничего порекомендовать:(")

    scores = {}

    for candidate_id in candidates:
        positive_score = 0.0
        negative_score = 0.0

        if valid_likes:
            like_similarities = [
                similarity_df.loc[candidate_id, liked_id]
                for liked_id in valid_likes
            ]

            if aggregation == 'mean':
                positive_score = np.mean(like_similarities)
            elif aggregation == 'max':
                positive_score = np.max(like_similarities)
            elif aggregation == 'weighted':
                weights = np.array(like_similarities)
                weights = weights / weights.sum() if weights.sum() > 0 else weights
                positive_score = np.sum(np.array(like_similarities) * weights)

        if valid_dislikes:
            dislike_similarities = [
                similarity_df.loc[candidate_id, disliked_id]
                for disliked_id in valid_dislikes
            ]

            if aggregation == 'mean':
                negative_score = np.mean(dislike_similarities)
            elif aggregation == 'max':
                negative_score = np.max(dislike_similarities)
            elif aggregation == 'weighted':
                weights = np.array(dislike_similarities)
                weights = weights / weights.sum() if weights.sum() > 0 else weights
                negative_score = np.sum(np.array(dislike_similarities) * weights)

        final_score = like_weight * positive_score - dislike_weight * negative_score

        scores[candidate_id] = {
            'total_score': final_score,
            'positive_score': positive_score,
            'negative_score': negative_score
        }

    sorted_scores = sorted(
        scores.items(),
        key=lambda x: x[1]['total_score'],
        reverse=True
    )[:n]

    results = []
    for rank, (article_id, score_dict) in enumerate(sorted_scores, 1):
        article = df[df['article_id'] == article_id].iloc[0]

        results.append({
            'rank': rank,
            'article_id': article_id,
            'total_score': score_dict['total_score'],
            'positive_score': score_dict['positive_score'],
            'negative_score': score_dict['negative_score'],
            'category': article['category'],
            'author': article['author'],
            'tags': article['tags']
        })

    return pd.DataFrame(results)