"""Vector retrieval via sentence embeddings.

Model: BAAI/bge-small-zh-v1.5 (512-dim, Chinese-optimized). Lazy-loaded on first use.
Storage: L2-normalized float32 matrix in .npz + metadata in .meta.pkl.
Search: cosine similarity via a single matrix-vector product (pure numpy, ~10ms for 22k).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .loader import Word

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"

# Model-specific query prefixes for retrieval
BGE_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
GTE_QUERY_PREFIX = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "


def build_semantic_doc(w: Word) -> str:
    """Turn a Word record into a natural-language document for embedding.

    Emphasizes explanations and gloss over raw character; that's where
    the semantic signal lives for a dictionary entry.
    """
    parts: list[str] = []

    head = w.text
    if w.primary_yngping:
        head += f"（{w.primary_yngping}）"
    parts.append(head)

    if w.gloss:
        parts.append(w.gloss)

    # SeeDict explanations
    for e in w.expls[:5]:
        cat = f"[{e.lexical_category}] " if e.lexical_category else ""
        parts.append(f"{cat}{e.content}")
        for s in e.sentences[:2]:
            parts.append(f"例：{s}")

    # Feng definitions — ALWAYS include, not just as fallback.
    # Feng has richer multi-sense definitions and example sentences for 11k+ words.
    for f in w.fengs[:3]:
        for item in (f.content or [])[:4]:
            if isinstance(item, dict) and item.get("expl"):
                parts.append(str(item["expl"]))
                for s in (item.get("sent") or [])[:1]:
                    parts.append(f"例：{s}")

    # Supplement texts — these ARE natural Mandarin synonyms, high semantic value.
    if w.supplements:
        alts = [s.text for s in w.supplements if s.type in ("M", "S")][:8]
        if alts:
            parts.append(f"同义：{'、'.join(alts)}")

    doc = "。".join(p.strip().rstrip("。") for p in parts if p and p.strip())
    return doc[:1200]  # hard cap — bge ctx is 512 tokens, gte-qwen2 supports 32k


class EmbeddingIndex:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model: "SentenceTransformer" | None = None
        self.word_ids: list[int] = []
        self.vectors: np.ndarray | None = None  # shape (N, D), L2-normalized float32
        self.docs: list[str] = []  # stored for debug/introspection

    def _get_model(self) -> "SentenceTransformer":
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            print(f"[embed] loading model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name, trust_remote_code=True)
        return self._model

    def build(self, words: dict[int, Word], batch_size: int = 64) -> None:
        sorted_items = sorted(words.items(), key=lambda kv: kv[0])
        self.word_ids = [wid for wid, _ in sorted_items]
        self.docs = [build_semantic_doc(w) for _, w in sorted_items]

        print(f"[embed] encoding {len(self.docs)} documents (this takes a few minutes on CPU)...")
        model = self._get_model()
        vecs = model.encode(
            self.docs,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        self.vectors = vecs.astype(np.float32)
        print(f"[embed] done. shape={self.vectors.shape}")

    def encode_query(self, query: str) -> np.ndarray:
        model = self._get_model()
        name_lower = self.model_name.lower()
        if "bge" in name_lower:
            q = BGE_QUERY_PREFIX + query
            vec = model.encode([q], normalize_embeddings=True, convert_to_numpy=True)
        elif "gte" in name_lower:
            # GTE-Qwen2 uses prompt_name for query-side instruction
            try:
                vec = model.encode([query], normalize_embeddings=True,
                                   convert_to_numpy=True, prompt_name="query")
            except Exception:
                q = GTE_QUERY_PREFIX + query
                vec = model.encode([q], normalize_embeddings=True, convert_to_numpy=True)
        else:
            vec = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        return vec[0].astype(np.float32)

    def search(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        """Return [(word_id, cosine_similarity)] sorted by score desc."""
        if self.vectors is None or not self.word_ids:
            return []
        q = self.encode_query(query)
        scores = self.vectors @ q  # shape (N,)
        top_idx = np.argsort(-scores)[:top_k]
        return [(self.word_ids[int(i)], float(scores[int(i)])) for i in top_idx]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path.with_suffix(".npz"),
            vectors=self.vectors,
            word_ids=np.array(self.word_ids, dtype=np.int64),
        )
        with open(path.with_suffix(".meta.pkl"), "wb") as f:
            pickle.dump({"model_name": self.model_name, "docs": self.docs}, f)
        print(f"[embed] saved to {path.with_suffix('.npz')}")

    @classmethod
    def load(cls, path: str | Path) -> "EmbeddingIndex":
        path = Path(path)
        npz_path = path.with_suffix(".npz")
        meta_path = path.with_suffix(".meta.pkl")
        if not npz_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"embedding index not found at {path}")
        data = np.load(npz_path)
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        idx = cls(model_name=meta["model_name"])
        idx.vectors = data["vectors"]
        idx.word_ids = data["word_ids"].tolist()
        idx.docs = meta.get("docs", [])
        return idx

    @classmethod
    def try_load(cls, path: str | Path) -> "EmbeddingIndex | None":
        try:
            return cls.load(path)
        except FileNotFoundError:
            return None
