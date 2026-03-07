"""Hybrid search engine combining BM25 and Dense search using Reciprocal Rank Fusion (RRF)."""

from typing import List, Dict, Any, Optional
import numpy as np


class HybridSearchEngine:
    """Hybrid search engine with Reciprocal Rank Fusion."""

    def __init__(
        self,
        bm25_index,
        dense_index,
        rrf_k: int = 60,
    ):
        """Initialize hybrid search engine.

        Args:
            bm25_index: Instance of BM25Index
            dense_index: Instance of DenseIndex
            rrf_k: Stability parameter for RRF (default 60)
        """
        self.bm25_index = bm25_index
        self.dense_index = dense_index
        self.rrf_k = rrf_k

    def query(
        self,
        text: str,
        top_k: int = 50,
        bm25_top_k: int = 100,
        dense_top_k: int = 100,
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search and fuse results.

        Args:
            text: Query text
            top_k: Final number of results to return
            bm25_top_k: Number of candidates to fetch from BM25
            dense_top_k: Number of candidates to fetch from Dense index

        Returns:
            List of dictionaries: [{'citation': ..., 'score': ...}, ...]
        """
        # 1. Fetch candidates from both indices
        bm25_results = self.bm25_index.search(text, top_k=bm25_top_k)
        dense_results = self.dense_index.search(text, top_k=dense_top_k)

        # 2. Apply Reciprocal Rank Fusion (RRF)
        # We need to map citations to their RRF scores
        rrf_scores = {}

        # Process BM25 rankings
        for rank, res in enumerate(bm25_results):
            citation = res.get("citation")
            if not citation:
                continue

            score = 1.0 / (self.rrf_k + rank + 1)
            rrf_scores[citation] = rrf_scores.get(citation, 0.0) + score

        # Process Dense rankings
        for rank, res in enumerate(dense_results):
            citation = res.get("citation")
            if not citation:
                continue

            score = 1.0 / (self.rrf_k + rank + 1)
            rrf_scores[citation] = rrf_scores.get(citation, 0.0) + score

        # 3. Sort by RRF score
        sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # 4. Limit to top_k
        top_results = sorted_rrf[:top_k]

        if not top_results:
            return []

        # 5. MinMax Normalization of RRF scores to [0, 1] range
        scores = np.array([res[1] for res in top_results])
        if len(scores) > 1:
            min_s, max_s = scores.min(), scores.max()
            if max_s > min_s:
                norm_scores = (scores - min_s) / (max_s - min_s)
            else:
                norm_scores = np.ones_like(scores)
        else:
            norm_scores = np.ones_like(scores)

        # 6. Build final response
        final_results = []
        for (citation, _), norm_score in zip(top_results, norm_scores):
            final_results.append({"citation": citation, "score": float(norm_score)})

        return final_results
