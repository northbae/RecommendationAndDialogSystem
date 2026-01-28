import pandas as pd
from pathlib import Path
from typing import Optional
from ..utils.config import config


class NewsDataset:
    def __init__(self, csv_path: Optional[str] = None):
        if csv_path is None:
            csv_path = config.get('data.raw_dataset')

        self.csv_path = Path(csv_path)
        self.df = None
        self.df_processed = None

    def load(self) -> pd.DataFrame:
        self.df = pd.read_csv(self.csv_path)
        return self.df

    def preprocess(self) -> pd.DataFrame:
        # тут
        # парсинг дат
        # производные признаки
        # валидация
        if self.df is None:
            self.load()

        df = self.df.copy()

        df['published_at'] = pd.to_datetime(df['published_at'])
        df['published_timestamp'] = df['published_at'].astype('int64') // 10 ** 9

        df['has_many_comments'] = (df['comment_number'] > df['comment_number'].median()).astype(int)
        df['is_long_article'] = (df['content_length'] > df['content_length'].median()).astype(int)
        df['is_easy_read'] = (df['readability_index'] > df['readability_index'].median()).astype(int)

        df['main_category'] = df['category'].apply(lambda x: x.split('/')[0])

        self.df_processed = df
        return df

    def get_by_id(self, article_id: int) -> pd.Series:
        df = self.df_processed if self.df_processed is not None else self.df
        return df[df['article_id'] == article_id].iloc[0]

    def get_by_category(self, category: str) -> pd.DataFrame:
        df = self.df_processed if self.df_processed is not None else self.df
        return df[df['category'].str.startswith(category)]

    def get_by_author(self, author: str) -> pd.DataFrame:
        df = self.df_processed if self.df_processed is not None else self.df
        return df[df['author'] == author]

    @property
    def categories(self):
        df = self.df_processed if self.df_processed is not None else self.df
        return df['category'].unique()

    @property
    def authors(self):
        df = self.df_processed if self.df_processed is not None else self.df
        return df['author'].unique()

    def __len__(self):
        df = self.df_processed if self.df_processed is not None else self.df
        return len(df) if df is not None else 0

    def __repr__(self):
        return f"NewsDataset(n_articles={len(self)}, csv_path='{self.csv_path}')"