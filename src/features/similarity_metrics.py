import numpy as np
import pandas as pd
import scipy.spatial.distance as ssd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Optional, Dict, Literal
from pathlib import Path
import pickle

from ..utils.config import config


class SimilarityCalculator:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.similarity_matrices: Dict[str, pd.DataFrame] = {}
        self._numeric_data_scaled = None
        self._binary_data = None

    def _prepare_numeric_data(self):
        if self._numeric_data_scaled is not None:
            return self._numeric_data_scaled

        numeric_cols = ['content_length', 'comment_number', 'readability_index']

        scaler = MinMaxScaler()
        self._numeric_data_scaled = scaler.fit_transform(self.df[numeric_cols])

        return self._numeric_data_scaled

    def compute_euclidean_similarity(self) -> pd.DataFrame:
        numeric_data = self._prepare_numeric_data()
        n_features = numeric_data.shape[1]

        D = ssd.pdist(numeric_data, metric='euclidean')
        D = ssd.squareform(D)

        S = 1 - D / np.sqrt(n_features)

        sim_df = pd.DataFrame(S, index=self.df['article_id'], columns=self.df['article_id'])
        self.similarity_matrices['euclidean'] = sim_df

        return sim_df

    def compute_manhattan_similarity(self) -> pd.DataFrame:
        numeric_data = self._prepare_numeric_data()
        n_features = numeric_data.shape[1]

        D = ssd.pdist(numeric_data, metric='cityblock')
        D = ssd.squareform(D)

        S = 1 - D / n_features

        sim_df = pd.DataFrame(S, index=self.df['article_id'], columns=self.df['article_id'])
        self.similarity_matrices['manhattan'] = sim_df

        return sim_df

    def compute_cosine_similarity(self) -> pd.DataFrame:
        numeric_data = self._prepare_numeric_data()

        S = cosine_similarity(numeric_data)

        sim_df = pd.DataFrame(S, index=self.df['article_id'], columns=self.df['article_id'])
        self.similarity_matrices['cosine'] = sim_df

        return sim_df

    def compute_chebyshev_similarity(self) -> pd.DataFrame:
        numeric_data = self._prepare_numeric_data()

        D = ssd.pdist(numeric_data, metric='chebyshev')
        D = ssd.squareform(D)

        S = 1 - D / 1

        sim_df = pd.DataFrame(S, index=self.df['article_id'], columns=self.df['article_id'])
        self.similarity_matrices['chebyshev'] = sim_df

        return sim_df

    def compute_minkowski_similarity(self, p: int = 3) -> pd.DataFrame:
        numeric_data = self._prepare_numeric_data()
        n_features = numeric_data.shape[1]

        D = ssd.pdist(numeric_data, metric='minkowski', p=p)
        D = ssd.squareform(D)

        S = 1 - D / (n_features ** (1 / p))

        sim_df = pd.DataFrame(S, index=self.df['article_id'], columns=self.df['article_id'])
        self.similarity_matrices['minkowski'] = sim_df

        return sim_df


    def compute_correlation_similarity(self) -> pd.DataFrame:
        numeric_data = self._prepare_numeric_data()

        D = ssd.pdist(numeric_data, metric='correlation')
        D = ssd.squareform(D)

        S = 1 - D

        sim_df = pd.DataFrame(S, index=self.df['article_id'], columns=self.df['article_id'])
        self.similarity_matrices['correlation'] = sim_df

        return sim_df


    def _prepare_binary_data(self):
        if self._binary_data is not None:
            return self._binary_data

        binary_cols = ['has_video', 'has_image']
        self._binary_data = self.df[binary_cols].astype(int).values

        return self._binary_data

    def compute_binary_jaccard_similarity(self) -> pd.DataFrame:
        binary_data = self._prepare_binary_data()

        D = ssd.pdist(binary_data, metric='jaccard')
        D = ssd.squareform(D)
        S = 1 - D

        sim_df = pd.DataFrame(S, index=self.df['article_id'], columns=self.df['article_id'])
        self.similarity_matrices['binary_jaccard'] = sim_df

        return sim_df

    def compute_binary_dice_similarity(self) -> pd.DataFrame:
        binary_data = self._prepare_binary_data()

        D = ssd.pdist(binary_data, metric='dice')
        D = ssd.squareform(D)
        S = 1 - D

        sim_df = pd.DataFrame(S, index=self.df['article_id'], columns=self.df['article_id'])
        self.similarity_matrices['binary_dice'] = sim_df

        return sim_df

    def compute_binary_hamming_similarity(self) -> pd.DataFrame:
        binary_data = self._prepare_binary_data()

        D = ssd.pdist(binary_data, metric='hamming')
        D = ssd.squareform(D)
        S = 1 - D

        sim_df = pd.DataFrame(S, index=self.df['article_id'], columns=self.df['article_id'])
        self.similarity_matrices['binary_hamming'] = sim_df

        return sim_df

    def compute_binary_smc_similarity(self) -> pd.DataFrame:
        binary_data = self._prepare_binary_data()

        D = ssd.pdist(binary_data, metric='sokalmichener')
        D = ssd.squareform(D)
        S = 1 - D

        sim_df = pd.DataFrame(S, index=self.df['article_id'], columns=self.df['article_id'])
        self.similarity_matrices['binary_smc'] = sim_df

        return sim_df

    def compute_tree_similarity(self) -> pd.DataFrame:
        from .tree_distance import compute_tree_distance_matrix

        tree_dist_matrix = compute_tree_distance_matrix(self.df)

        max_dist = tree_dist_matrix.max()
        S = 1 - tree_dist_matrix / max_dist if max_dist > 0 else tree_dist_matrix

        sim_df = pd.DataFrame(S, index=self.df['article_id'], columns=self.df['article_id'])
        self.similarity_matrices['tree'] = sim_df

        return sim_df

    def compute_tags_similarity(self) -> pd.DataFrame:

        tfidf = TfidfVectorizer(
            tokenizer=lambda x: [tag.strip() for tag in x.split(',')],
            token_pattern=None,
            lowercase=True
        )

        tags_tfidf = tfidf.fit_transform(self.df['tags'])

        S = cosine_similarity(tags_tfidf)

        sim_df = pd.DataFrame(S, index=self.df['article_id'], columns=self.df['article_id'])
        self.similarity_matrices['tags'] = sim_df

        return sim_df


    def compute_geographic_similarity(self) -> pd.DataFrame:
        geo_encoder = OneHotEncoder(sparse_output=False)
        geo_encoded = geo_encoder.fit_transform(self.df[['geographic_scope']])

        D = ssd.pdist(geo_encoded, metric='hamming')
        D = ssd.squareform(D)
        S = 1 - D

        sim_df = pd.DataFrame(S, index=self.df['article_id'], columns=self.df['article_id'])
        self.similarity_matrices['geographic'] = sim_df

        return sim_df

    def compute_author_similarity(self) -> pd.DataFrame:
        author_encoder = OneHotEncoder(sparse_output=False)
        author_encoded = author_encoder.fit_transform(self.df[['author']])

        S = cosine_similarity(author_encoded)

        sim_df = pd.DataFrame(S, index=self.df['article_id'], columns=self.df['article_id'])
        self.similarity_matrices['author'] = sim_df

        return sim_df

    def compute_time_similarity(self, max_hours: int = 168) -> pd.DataFrame:
        if 'published_timestamp' not in self.df.columns:
            self.df['published_timestamp'] = pd.to_datetime(
                self.df['published_at']
            ).astype(np.int64) // 10 ** 9

        timestamps = self.df['published_timestamp'].values.reshape(-1, 1)

        D = ssd.pdist(timestamps, metric='euclidean') / 3600
        D = ssd.squareform(D)

        S = np.maximum(0, 1 - D / max_hours)

        sim_df = pd.DataFrame(S, index=self.df['article_id'], columns=self.df['article_id'])
        self.similarity_matrices['time'] = sim_df

        return sim_df


    def compute_comprehensive_similarity(
            self,
            weights: Optional[Dict[str, float]] = None,
            numeric_metric: str = 'manhattan'
    ) -> pd.DataFrame:

        if weights is None:
            weights = config.similarity_weights

        if 'numeric' not in self.similarity_matrices:
            if numeric_metric == 'euclidean':
                self.compute_euclidean_similarity()
                numeric_key = 'euclidean'
            elif numeric_metric == 'manhattan':
                self.compute_manhattan_similarity()
                numeric_key = 'manhattan'
            elif numeric_metric == 'cosine':
                self.compute_cosine_similarity()
                numeric_key = 'cosine'
            else:
                self.compute_manhattan_similarity()
                numeric_key = 'manhattan'
        else:
            numeric_key = 'numeric'

        if 'binary' not in self.similarity_matrices:
            self.compute_binary_hamming_similarity()
            binary_key = 'binary_hamming'
        else:
            binary_key = 'binary'

        if 'tree' not in self.similarity_matrices:
            self.compute_tree_similarity()

        if 'tags' not in self.similarity_matrices:
            self.compute_tags_similarity()

        if 'geographic' not in self.similarity_matrices:
            self.compute_geographic_similarity()

        if 'author' not in self.similarity_matrices:
            self.compute_author_similarity()

        if 'time' not in self.similarity_matrices:
            self.compute_time_similarity()

        components = {
            'category': self.similarity_matrices['tree'].values,
            'tags': self.similarity_matrices['tags'].values,
            'numeric': self.similarity_matrices.get(numeric_key,
                                                    self.similarity_matrices.get('manhattan')).values,
            'binary': self.similarity_matrices.get(binary_key,
                                                   self.similarity_matrices.get('binary_hamming')).values,
            'geographic': self.similarity_matrices['geographic'].values,
            'author': self.similarity_matrices['author'].values,
            'time': self.similarity_matrices['time'].values
        }

        S_comprehensive = sum(
            weights.get(key, 0) * matrix
            for key, matrix in components.items()
        )

        sim_df = pd.DataFrame(
            S_comprehensive,
            index=self.df['article_id'],
            columns=self.df['article_id']
        )

        self.similarity_matrices['comprehensive'] = sim_df

        #print("вычислили обобщающую")
        return sim_df


    def save_all(self, output_dir: Optional[str] = None):
        if output_dir is None:
            output_dir = config.get('data.similarity_matrices', 'data/processed/similarity_matrices/')

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for name, matrix in self.similarity_matrices.items():
            filepath = output_path / f"{name}_similarity.csv"
            matrix.to_csv(filepath)
            #print(f"Сохранено: {filepath}")

    def load_all(self, input_dir: Optional[str] = None):
        if input_dir is None:
            input_dir = config.get('data.similarity_matrices', 'data/processed/similarity_matrices/')

        input_path = Path(input_dir)

        if not input_path.exists():
            #print(f"Директория {input_path} не найдена")
            return

        for filepath in input_path.glob("*_similarity.csv"):
            name = filepath.stem.replace('_similarity', '')
            matrix = pd.read_csv(filepath, index_col=0)
            self.similarity_matrices[name] = matrix
            #print(f"загружено: {name}")

    def get_similarity_matrix(self, name: str) -> Optional[pd.DataFrame]:
        return self.similarity_matrices.get(name)

    def list_available_matrices(self):
        return list(self.similarity_matrices.keys())

    def __repr__(self):
        return f"SimilarityCalculator(n_articles={len(self.df)}, matrices={len(self.similarity_matrices)})"