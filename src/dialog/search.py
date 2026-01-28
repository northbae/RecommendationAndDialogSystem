import pandas as pd
from typing import List, Dict, Any
from dataclasses import dataclass
from .linguistic_variable import LengthTerm, ImportanceTerm, get_length_membership, get_importance_membership


@dataclass
class SearchResult:
    article_id: int
    title: str
    score: float
    details: Dict[str, Any]


class FuzzySearchEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(self.df['published_at']):
            self.df['published_at'] = pd.to_datetime(self.df['published_at'])
        self.max_date = self.df['published_at'].max().normalize()

    def search(self, filters: Dict, exclusions: Dict, top_n: int = 5) -> List[SearchResult]:
        candidates = self.df.copy()
        candidates['match_score'] = 1.0

        if 'category' in exclusions:
            for ex in exclusions['category']: candidates = candidates[
                ~candidates['category'].str.contains(ex, case=False)]
        if 'media' in exclusions:
            for m in exclusions['media']:
                if m == "MEDIA_VIDEO":
                    candidates = candidates[candidates['has_video'] == 0]
                elif m == "MEDIA_IMAGE":
                    candidates = candidates[candidates['has_image'] == 0]

        if 'category' in filters:
            cond = pd.Series([False] * len(candidates), index=candidates.index)
            for cat in filters['category']: cond |= candidates['category'].str.contains(cat, case=False)
            candidates = candidates[cond]
        if 'author' in filters:
            candidates = candidates[candidates['author'].str.contains(filters['author'], case=False)]
        if 'media' in filters:
            for m in filters['media']:
                if m == "MEDIA_VIDEO":
                    candidates = candidates[candidates['has_video'] == 1]
                elif m == "MEDIA_IMAGE":
                    candidates = candidates[candidates['has_image'] == 1]

        if 'date' in filters:
            d = filters['date']
            if d == "DATE_TODAY":
                candidates = candidates[candidates['published_at'].dt.normalize() == self.max_date]
            elif d == "DATE_YESTERDAY":
                candidates = candidates[
                    candidates['published_at'].dt.normalize() == (self.max_date - pd.Timedelta(days=1))]
            elif d == "DATE_WEEK":
                candidates = candidates[candidates['published_at'] >= (self.max_date - pd.Timedelta(days=7))]
            elif d == "DATE_MONTH":
                candidates = candidates[candidates['published_at'] >= (self.max_date - pd.Timedelta(days=30))]

        if 'length' in filters:
            candidates['match_score'] *= candidates['content_length'].apply(
                lambda x: get_length_membership(x, filters['length']))
        if 'importance' in filters and 'comment_number' in candidates.columns:
            candidates['match_score'] *= candidates['comment_number'].apply(
                lambda x: get_importance_membership(x, filters['importance']))

        candidates = candidates[candidates['match_score'] > 0.01].sort_values(by=['match_score', 'published_at'],
                                                                              ascending=[False, False]).head(top_n)
        return self._to_results(candidates)

    def find_similar(self, liked_ids: List[int], top_n: int = 5) -> List[SearchResult]:
        if not liked_ids: return []
        liked_rows = self.df[self.df['article_id'].isin(liked_ids)]
        if liked_rows.empty: return []

        liked_cats = set(r['category'].split('/')[0] for _, r in liked_rows.iterrows())
        candidates = self.df[~self.df['article_id'].isin(liked_ids)].copy()
        candidates['match_score'] = candidates['category'].apply(
            lambda x: 1.0 if x.split('/')[0] in liked_cats else 0.0)
        return self._to_results(candidates[candidates['match_score'] > 0].head(top_n))

    def find_dissimilar(self, disliked_ids: List[int], top_n: int = 5) -> List[SearchResult]:
        if not disliked_ids: return []
        disliked_rows = self.df[self.df['article_id'].isin(disliked_ids)]
        if disliked_rows.empty: return []
        disliked_cats = set(r['category'].split('/')[0] for _, r in disliked_rows.iterrows())
        candidates = self.df[
            ~self.df['article_id'].isin(disliked_ids) & ~self.df['category'].str.split('/').str[0].isin(
                disliked_cats)].copy()
        candidates['match_score'] = 1.0
        return self._to_results(candidates.head(top_n))

    def _to_results(self, df):
        res = []
        for _, r in df.iterrows():
            media = " 🎥" if r.get('has_video', 0) else ""
            media += " 🖼️" if r.get('has_image', 0) else ""
            res.append(SearchResult(r['article_id'], f"{r['category']} / {r['author']}{media}", r['match_score'],
                                    {"len": r['content_length'], "date": r['published_at'].strftime('%Y-%m-%d')}))
        return res