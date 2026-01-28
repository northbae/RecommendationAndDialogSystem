import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from typing import Literal, Optional


class FeaturePreprocessor:
    def __init__(self, scaling_method: Literal['minmax', 'standard', 'robust'] = 'minmax'):
        self.scaling_method = scaling_method
        self.scaler = None
        self._init_scaler()

    def _init_scaler(self):
        if self.scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif self.scaling_method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Неизвестный метод: {self.scaling_method}")

    def fit_transform_numeric(self, df: pd.DataFrame,
                              columns: Optional[list] = None) -> np.ndarray:
        if columns is None:
            columns = ['content_length', 'comment_number', 'readability_index']

        data = df[columns].values
        scaled_data = self.scaler.fit_transform(data)

        return scaled_data

    def transform_numeric(self, df: pd.DataFrame,
                          columns: Optional[list] = None) -> np.ndarray:
        if columns is None:
            columns = ['content_length', 'comment_number', 'readability_index']

        data = df[columns].values
        scaled_data = self.scaler.transform(data)

        return scaled_data

    @staticmethod
    def prepare_binary(df: pd.DataFrame,
                       columns: Optional[list] = None) -> np.ndarray:
        if columns is None:
            columns = ['has_video', 'has_image']

        return df[columns].astype(int).values

    @staticmethod
    def prepare_categorical_onehot(df: pd.DataFrame,
                                   column: str) -> np.ndarray:
        from sklearn.preprocessing import OneHotEncoder

        encoder = OneHotEncoder(sparse_output=False)
        encoded = encoder.fit_transform(df[[column]])

        return encoded

    @staticmethod
    def prepare_text_tfidf(df: pd.DataFrame,
                           column: str,
                           max_features: Optional[int] = None) -> np.ndarray:
        from sklearn.feature_extraction.text import TfidfVectorizer

        if column == 'tags':
            vectorizer = TfidfVectorizer(
                tokenizer=lambda x: [tag.strip() for tag in x.split(',')],
                token_pattern=None,
                lowercase=True,
                max_features=max_features
            )
        else:
            vectorizer = TfidfVectorizer(
                lowercase=True,
                max_features=max_features
            )

        tfidf_matrix = vectorizer.fit_transform(df[column])

        return tfidf_matrix.toarray()

    @staticmethod
    def handle_missing_values(df: pd.DataFrame,
                              strategy: Literal['mean', 'median', 'mode', 'drop'] = 'median') -> pd.DataFrame:

        df_clean = df.copy()

        if strategy == 'drop':
            df_clean = df_clean.dropna()
        else:
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns

            for col in numeric_cols:
                if df_clean[col].isnull().any():
                    if strategy == 'mean':
                        df_clean[col].fillna(df_clean[col].mean(), inplace=True)
                    elif strategy == 'median':
                        df_clean[col].fillna(df_clean[col].median(), inplace=True)
                    elif strategy == 'mode':
                        df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)

        return df_clean

    @staticmethod
    def detect_outliers(data: np.ndarray, method: Literal['iqr', 'zscore'] = 'iqr') -> np.ndarray:

        if method == 'iqr':
            Q1 = np.percentile(data, 25, axis=0)
            Q3 = np.percentile(data, 75, axis=0)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = (data < lower_bound) | (data > upper_bound)

        elif method == 'zscore':
            z_scores = np.abs((data - data.mean(axis=0)) / data.std(axis=0))
            outliers = z_scores > 3

        else:
            raise ValueError(f"Неизвестный метод: {method}")

        return outliers.any(axis=1)

    def __repr__(self):
        return f"FeaturePreprocessor(method='{self.scaling_method}')"