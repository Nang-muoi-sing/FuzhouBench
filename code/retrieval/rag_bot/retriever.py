"""Hybrid retriever: exact-match + BM25 + optional vector search with RRF fusion."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .embeddings import EmbeddingIndex
from .index import (
    SearchIndex,
    normalize_yngping,
    tokenize_chinese,
    tokenize_latin,
    tokenize_yngping,
)
from .loader import Word


_LATIN_NUM_RE = re.compile(r"^[a-zA-Z0-9\s]+$")

# RRF hyperparameter (Cormack et al. 2009). Smaller k gives more weight to top ranks.
RRF_K = 60


@dataclass
class SearchHit:
    word_id: int
    score: float
    match_type: str
    # Debug fields for hybrid retrieval
    bm25_rank: int | None = None
    vector_rank: int | None = None

    def __repr__(self) -> str:
        return f"Hit(wid={self.word_id}, score={self.score:.3f}, via={self.match_type})"


class Retriever:
    def __init__(
        self,
        words: dict[int, Word],
        index: SearchIndex,
        embedding_index: EmbeddingIndex | None = None,
    ):
        self.words = words
        self.index = index
        self.embedding_index = embedding_index
        self._wid_to_pos = {wid: i for i, wid in enumerate(index.word_ids)}

    @property
    def has_vector(self) -> bool:
        return self.embedding_index is not None and self.embedding_index.vectors is not None

    def _is_latin_query(self, q: str) -> bool:
        return bool(_LATIN_NUM_RE.match(q.strip()))

    def search(self, query: str, top_k: int = 10) -> list[SearchHit]:
        """Hybrid search.

        Layer 1 — deterministic lookups:
          - exact text / exact yngping
          - supplement (alternative-form) match
          - prefix / contains
        Layer 2 — fuzzy:
          - BM25 ranking over char n-grams + yngping tokens
          - vector ranking (if available) over semantic docs
          - merged via RRF
        """
        q = query.strip()
        if not q:
            return []

        hits: list[SearchHit] = []
        seen: set[int] = set()

        def add(wid: int, score: float, match_type: str, **meta) -> None:
            if wid in seen or wid not in self.words:
                return
            seen.add(wid)
            hits.append(SearchHit(word_id=wid, score=score, match_type=match_type, **meta))

        if self._is_latin_query(q):
            self._search_yngping_layer(q, add)
        else:
            self._search_chinese_layer(q, add)

        # Fuzzy layer — BM25 + optional vector, RRF-merged
        self._search_fuzzy_layer(q, top_k, add)

        hits.sort(key=lambda h: (-h.score, h.word_id))
        return hits[:top_k]

    # ---- deterministic layers ---------------------------------------------

    def _search_yngping_layer(self, q: str, add) -> None:
        yn = normalize_yngping(q)
        for wid in self.index.yngping_map.get(yn, []):
            add(wid, 1000.0, "exact_yngping")
        for key, wids in self.index.yngping_map.items():
            if key != yn and key.startswith(yn):
                for wid in wids:
                    add(wid, 800.0, "yngping_prefix")
        first = yn.split()[0] if yn else ""
        for wid in self.index.yngping_prefix_map.get(first, []):
            add(wid, 600.0, "yngping_prefix")

    def _search_chinese_layer(self, q: str, add) -> None:
        for wid in self.index.text_map.get(q, []):
            add(wid, 1000.0, "exact_text")
        for wid in self.index.supplement_map.get(q, []):
            add(wid, 900.0, "supplement")
        for key, wids in self.index.text_map.items():
            if key != q and key.startswith(q):
                for wid in wids:
                    add(wid, 750.0, "prefix_text")
        for key, wids in self.index.text_map.items():
            if key != q and q in key and not key.startswith(q):
                for wid in wids:
                    add(wid, 600.0, "contains_text")

    # ---- fuzzy layer (BM25 + vector via RRF) -----------------------------

    def _search_fuzzy_layer(self, q: str, top_k: int, add) -> None:
        # Pull larger candidate pools than top_k so RRF has material to fuse.
        pool_size = max(top_k * 5, 30)

        bm25_ranking = self._bm25_ranked(q, pool_size)
        vector_ranking = self._vector_ranked(q, pool_size) if self.has_vector else []

        if not bm25_ranking and not vector_ranking:
            return

        # Reciprocal Rank Fusion — score-scale-agnostic merge.
        rrf: dict[int, float] = {}
        bm25_ranks: dict[int, int] = {}
        vec_ranks: dict[int, int] = {}

        for rank, wid in enumerate(bm25_ranking):
            rrf[wid] = rrf.get(wid, 0.0) + 1.0 / (RRF_K + rank + 1)
            bm25_ranks[wid] = rank + 1
        for rank, wid in enumerate(vector_ranking):
            rrf[wid] = rrf.get(wid, 0.0) + 1.0 / (RRF_K + rank + 1)
            vec_ranks[wid] = rank + 1

        # Map RRF score into the same ballpark as other layers.
        # Max RRF when a doc is rank 1 in both sources: 2/(K+1) ≈ 0.033.
        # Scale so the best fused hit lands around 500.
        fusion_base = 400.0
        fusion_scale = 15000.0

        if vector_ranking and bm25_ranking:
            match_type = "hybrid"
        elif vector_ranking:
            match_type = "vector"
        else:
            match_type = "bm25"

        # Sort by fused RRF score, add in order.
        fused = sorted(rrf.items(), key=lambda kv: -kv[1])
        for wid, r in fused[: pool_size]:
            score = fusion_base + fusion_scale * r
            add(
                wid,
                score,
                match_type,
                bm25_rank=bm25_ranks.get(wid),
                vector_rank=vec_ranks.get(wid),
            )

    def _bm25_ranked(self, q: str, pool: int) -> list[int]:
        if self.index.bm25 is None:
            return []
        if self._is_latin_query(q):
            tokens = tokenize_yngping(q)
        else:
            tokens = tokenize_chinese(q) + tokenize_latin(q)
        if not tokens:
            return []
        scores = self.index.bm25.get_scores(tokens)
        top_indices = scores.argsort()[::-1][:pool]
        out: list[int] = []
        for pos in top_indices:
            s = float(scores[pos])
            if s <= 0:
                break
            out.append(self.index.word_ids[int(pos)])
        return out

    def _vector_ranked(self, q: str, pool: int) -> list[int]:
        if not self.has_vector:
            return []
        hits = self.embedding_index.search(q, top_k=pool)
        # Cosine sim threshold — very low similarity isn't useful.
        return [wid for wid, sim in hits if sim > 0.3]

    def get(self, wid: int) -> Word | None:
        return self.words.get(wid)
