from typing import Optional
import pandas as pd


class ArticleDisplayFormatter:
    @staticmethod
    def format_compact(article: pd.Series) -> str:
        return (
            f"ID: {article['article_id']} | "
            f"{article['category']} | "
            f"{article['author']}"
        )

    @staticmethod
    def format_full(article: pd.Series) -> str:
        media = []
        if article.get('has_video', 0):
            media.append("📹 Видео")
        if article.get('has_image', 0):
            media.append("🖼️ Изображение")

        media_str = ", ".join(media) if media else "Нет медиа"

        return f"""
╔════════════════════════════════════════════════════════════════════════════╗
║ ID: {article['article_id']:4d}                                                                  
║ Категория: {article['category'][:60]}
║ Автор: {article['author']}
║ Теги: {article['tags'][:65]}
║ Дата: {article['published_at']}
║ Длина: {article['content_length']} слов | Комментарии: {article['comment_number']} | Читаемость: {article['readability_index']}
║ География: {article['geographic_scope']} | Медиа: {media_str}
╚════════════════════════════════════════════════════════════════════════════╝
"""

    @staticmethod
    def format_list_item(article: pd.Series, rank: Optional[int] = None,
                        score: Optional[float] = None) -> str:
        prefix = f"{rank}. " if rank else ""
        score_str = f"[{score:.3f}] " if score is not None else ""

        return (
            f"{prefix}{score_str}ID:{article['article_id']:4d} | "
            f"{article['category'][:40]:40s} | {article['author']:15s}"
        )

    @staticmethod
    def format_comparison(article1: pd.Series, article2: pd.Series,
                         similarity: float) -> str:
        return f"""
Сравнение статей:
┌─ Статья #{article1['article_id']}
│  {article1['category']}
│  {article1['tags'][:60]}
│
├─ Сходство: {similarity:.4f}
│
└─ Статья #{article2['article_id']}
   {article2['category']}
   {article2['tags'][:60]}
"""


class ArticleTable:
    @staticmethod
    def create_table(articles_df: pd.DataFrame,
                    columns: Optional[list] = None) -> str:
        if columns is None:
            columns = ['article_id', 'category', 'author', 'comment_number']

        return articles_df[columns].to_string(index=False)

    @staticmethod
    def create_recommendations_table(recommendations_df: pd.DataFrame) -> str:
        header = f"{'Ранг':^6} | {'ID':^6} | {'Рейтинг':^8} | {'Категория':^40} | {'Автор':^15}"
        separator = "-" * len(header)

        rows = [header, separator]

        for _, row in recommendations_df.iterrows():
            line = (
                f"{row['rank']:^6d} | "
                f"{row['article_id']:^6d} | "
                f"{row.get('score', row.get('total_score', 0)):^8.4f} | "
                f"{row['category'][:40]:40s} | "
                f"{row['author'][:15]:15s}"
            )
            rows.append(line)

        return "\n".join(rows)