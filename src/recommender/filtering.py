import pandas as pd
from typing import Union


def get_similar_articles(
        article_id: int,
        similarity_df: pd.DataFrame,
        n: int = 5
) -> pd.Series:

    if article_id not in similarity_df.index:
        raise ValueError(f"Статья {article_id} не найдена в матрице сходства")

    similarities = similarity_df.loc[article_id]

    top_similar = similarities.sort_values(ascending=False).head(n + 1)

    top_similar = top_similar[top_similar.index != article_id].head(n)

    return top_similar