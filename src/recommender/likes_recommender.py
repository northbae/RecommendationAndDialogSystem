import pandas as pd
import numpy as np
from typing import List, Optional


def recommend_from_likes(
        liked_articles: List[int],
        similarity_df: pd.DataFrame,
        df: pd.DataFrame,
        n: int = 10,
        aggregation: str = 'mean',
        exclude_liked: bool = True
) -> pd.DataFrame:
    if not liked_articles:
        raise ValueError("Вы не лайкнули ни одну статью")

    valid_ids = [aid for aid in liked_articles if aid in similarity_df.index]
    if not valid_ids:
        raise ValueError("Ни один ID не найден в матрице сходства")

    all_articles = similarity_df.index.tolist()
    candidates = [aid for aid in all_articles if aid not in valid_ids] if exclude_liked else all_articles

    scores = {}
    for candidate_id in candidates:
        similarities = [similarity_df.loc[candidate_id, liked_id] for liked_id in valid_ids]

        if aggregation == 'mean':
            score = np.mean(similarities)
        elif aggregation == 'max':
            score = np.max(similarities)
        elif aggregation == 'weighted':
            weights = np.array(similarities) / sum(similarities) if sum(similarities) > 0 else np.ones(
                len(similarities))
            score = np.sum(np.array(similarities) * weights)
        else:
            score = np.mean(similarities)

        scores[candidate_id] = {
            'score': score,
            'similarities': {lid: similarity_df.loc[candidate_id, lid] for lid in valid_ids}
        }

    sorted_items = sorted(scores.items(), key=lambda x: x[1]['score'], reverse=True)[:n]

    results = []
    for rank, (article_id, data) in enumerate(sorted_items, 1):
        article = df[df['article_id'] == article_id].iloc[0]
        results.append({
            'rank': rank,
            'article_id': article_id,
            'score': data['score'],
            'category': article['category'],
            'author': article['author'],
            'tags': article['tags'],
            'similarities_to_liked': data['similarities']
        })

    return pd.DataFrame(results)