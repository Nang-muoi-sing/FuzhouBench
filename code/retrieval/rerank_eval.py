"""Rerank evaluation: hybrid top-N candidates → bge-reranker-large → top-K.

Compares pre-rerank (hybrid retrieval ranking) vs post-rerank metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_bot.embeddings import EmbeddingIndex, build_semantic_doc
from rag_bot.index import SearchIndex
from rag_bot.loader import Word
from rag_bot.retriever import Retriever

CACHE = Path(__file__).resolve().parent.parent / "data"


def metrics(rrs: List[float], n: int) -> dict:
    """Compute Hit@1/3/5/10 + MRR from per-query reciprocal ranks."""
    h1 = sum(1 for r in rrs if r >= 1.0)
    h3 = sum(1 for r in rrs if r >= 1/3)
    h5 = sum(1 for r in rrs if r >= 1/5)
    h10 = sum(1 for r in rrs if r > 0)  # any positive rank within top-10
    miss = sum(1 for r in rrs if r == 0)
    return {
        "n": n, "miss": miss,
        "hit@1": h1/n, "hit@3": h3/n, "hit@5": h5/n, "hit@10": h10/n,
        "mrr": sum(rrs)/n,
    }


def fmt(m: dict) -> str:
    return (f"n={m['n']:5d}  Hit@1={m['hit@1']:.3f}  Hit@3={m['hit@3']:.3f}  "
            f"Hit@5={m['hit@5']:.3f}  Hit@10={m['hit@10']:.3f}  MRR={m['mrr']:.3f}  Miss={m['miss']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--queries", required=True, help="path to colloquial_queries_*.json")
    p.add_argument("--cache", default=str(CACHE))
    p.add_argument("--reranker", default="BAAI/bge-reranker-large")
    p.add_argument("--candidate-pool", type=int, default=50,
                   help="N candidates from hybrid retrieval to feed reranker")
    p.add_argument("--top-k", type=int, default=10, help="K for Hit@K computation")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    cache = Path(args.cache)

    print("[load] words...")
    with open(cache / "words.pkl", "rb") as f:
        words: Dict[int, Word] = pickle.load(f)
    print("[load] BM25 index...")
    index = SearchIndex.load(cache / "search_index.pkl")
    print("[load] embeddings...")
    emb = EmbeddingIndex.try_load(cache / "embedding_index")
    retriever = Retriever(words, index, embedding_index=emb)

    # Pre-compute semantic docs for all words (same as embedding index docs)
    print("[prep] building doc texts...")
    doc_text: Dict[int, str] = {}
    for wid, w in words.items():
        doc_text[wid] = build_semantic_doc(w)

    # Load queries
    print(f"[load] queries from {args.queries}")
    with open(args.queries, "r", encoding="utf-8") as f:
        raw_queries = json.load(f)
    queries: List[Tuple[str, List[int]]] = []
    for q in raw_queries:
        text = q["query"].strip()
        if len(text) < 2:
            continue
        # Support both formats: {expected_word_ids: [...]} or {word_id: int}
        if "expected_word_ids" in q:
            wids = q["expected_word_ids"]
        elif "word_id" in q:
            wids = [q["word_id"]]
        else:
            continue
        queries.append((text, wids))
    n = len(queries)
    print(f"[load] {n} valid queries (skipped {len(raw_queries) - n})")

    # ----- Stage 1: retrieve top-N candidates per query (hybrid) -----
    print(f"\n[stage1] retrieving hybrid top-{args.candidate_pool} for each query...")
    t0 = time.time()
    candidates: List[List[int]] = []  # candidates[i] = [wid1, wid2, ...] top-N
    pre_rrs: List[float] = []  # baseline (no rerank) reciprocal ranks at top-K
    for q, expected_wids in queries:
        hits = retriever.search(q, top_k=args.candidate_pool)
        wids = [h.word_id for h in hits]
        candidates.append(wids)
        # Compute baseline RR at top-K (just from hybrid)
        expected = set(expected_wids)
        rr = 0.0
        for i, wid in enumerate(wids[:args.top_k]):
            if wid in expected:
                rr = 1.0 / (i + 1)
                break
        pre_rrs.append(rr)
    elapsed = time.time() - t0
    print(f"[stage1] done in {elapsed:.0f}s")

    pre_metrics = metrics(pre_rrs, n)
    print(f"\n  pre-rerank (hybrid top-{args.top_k}):  {fmt(pre_metrics)}")

    # ----- Stage 2: rerank with bge-reranker-large -----
    print(f"\n[stage2] loading reranker: {args.reranker}")
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder(args.reranker, max_length=512)

    print(f"[stage2] reranking {n} queries × up to {args.candidate_pool} candidates...")
    t0 = time.time()
    post_rrs: List[float] = []
    # Process in chunks of queries to keep memory bounded
    chunk_size = 32
    for chunk_start in range(0, n, chunk_size):
        chunk = queries[chunk_start:chunk_start + chunk_size]
        chunk_cands = candidates[chunk_start:chunk_start + chunk_size]

        # Build flat (query, doc) pair list
        pairs: List[List[str]] = []
        pair_owner: List[int] = []  # which query in chunk each pair belongs to
        for i, ((q, _exp), cands) in enumerate(zip(chunk, chunk_cands)):
            for wid in cands:
                pairs.append([q, doc_text.get(wid, "")])
                pair_owner.append(i)

        if pairs:
            scores = reranker.predict(pairs, batch_size=args.batch_size,
                                       show_progress_bar=False)
        else:
            scores = []

        # Distribute scores back to per-query lists
        per_query_scores: List[List[Tuple[int, float]]] = [[] for _ in chunk]
        for idx, score in enumerate(scores):
            i = pair_owner[idx]
            wid = chunk_cands[i][len(per_query_scores[i])]
            per_query_scores[i].append((wid, float(score)))

        # Sort by reranker score, take top-K, compute RR
        for i, (q, expected_wids) in enumerate(chunk):
            scored = sorted(per_query_scores[i], key=lambda kv: -kv[1])
            expected = set(expected_wids)
            rr = 0.0
            for j, (wid, _s) in enumerate(scored[:args.top_k]):
                if wid in expected:
                    rr = 1.0 / (j + 1)
                    break
            post_rrs.append(rr)

        done = min(chunk_start + chunk_size, n)
        elapsed = time.time() - t0
        speed = done / elapsed if elapsed > 0 else 0
        if (done % 320 == 0) or done == n:
            print(f"  [{done}/{n}] {speed:.1f} q/s, ETA {(n-done)/speed:.0f}s")

    elapsed = time.time() - t0
    print(f"[stage2] done in {elapsed:.0f}s ({n/elapsed:.1f} q/s)")

    post_metrics = metrics(post_rrs, n)

    # ----- Report -----
    print(f"\n{'=' * 70}")
    print(f"RESULTS — {args.queries}")
    print(f"{'=' * 70}")
    print(f"  pre-rerank  (hybrid top-{args.top_k}):  {fmt(pre_metrics)}")
    print(f"  post-rerank (bge-reranker top-{args.top_k}): {fmt(post_metrics)}")

    delta = post_metrics["mrr"] - pre_metrics["mrr"]
    pct = 100 * delta / pre_metrics["mrr"] if pre_metrics["mrr"] > 0 else 0
    print(f"\n  Δ MRR: {delta:+.3f} ({pct:+.1f}%)")
    print(f"  Δ Hit@1: {post_metrics['hit@1'] - pre_metrics['hit@1']:+.3f}")
    print(f"  Δ Hit@10: {post_metrics['hit@10'] - pre_metrics['hit@10']:+.3f}")
    print(f"  Δ Miss: {post_metrics['miss'] - pre_metrics['miss']:+d}")

    # Save
    output_path = args.output or (cache / f"rerank_results_{Path(args.queries).stem}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "queries_path": args.queries,
            "reranker": args.reranker,
            "candidate_pool": args.candidate_pool,
            "top_k": args.top_k,
            "pre_rerank": pre_metrics,
            "post_rerank": post_metrics,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n[done] saved to {output_path}")


if __name__ == "__main__":
    main()
