"""Evaluate LLM-generated colloquial queries against BM25/vector/hybrid.

Loads queries from colloquial_queries_llm.json (output of gen_colloquial_queries.py)
and runs the same evaluation as eval_full_dict.py for direct comparison.
"""

from __future__ import annotations

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_bot.embeddings import EmbeddingIndex  # noqa: E402
from rag_bot.index import SearchIndex  # noqa: E402
from rag_bot.loader import Word  # noqa: E402
from rag_bot.retriever import Retriever, SearchHit  # noqa: E402

CACHE = Path(__file__).resolve().parent.parent / "data"


class VectorOnlyRetriever:
    def __init__(self, words, emb):
        self.words = words
        self.embedding_index = emb

    def search(self, query, top_k=10):
        return [SearchHit(word_id=wid, score=sim, match_type="vector")
                for wid, sim in self.embedding_index.search(query, top_k=top_k)]

    def get(self, wid):
        return self.words.get(wid)


def evaluate_batch(retriever, queries: List[Tuple[str, List[int]]], top_k=10):
    hits1 = hits3 = hits5 = hits10 = miss = 0
    total_rr = 0.0
    for query, expected_wids in queries:
        hits = retriever.search(query, top_k=top_k)
        hit_ids = [h.word_id for h in hits]
        expected = set(expected_wids)
        rank = None
        for i, wid in enumerate(hit_ids):
            if wid in expected:
                rank = i + 1
                break
        if rank:
            total_rr += 1.0 / rank
            if rank <= 1: hits1 += 1
            if rank <= 3: hits3 += 1
            if rank <= 5: hits5 += 1
            if rank <= 10: hits10 += 1
        else:
            miss += 1
    n = len(queries)
    return {
        "n": n, "miss": miss,
        "hit@1": hits1/n, "hit@3": hits3/n, "hit@5": hits5/n, "hit@10": hits10/n,
        "mrr": total_rr/n,
    }


def fmt(m):
    return (f"n={m['n']:5d}  Hit@1={m['hit@1']:.3f}  Hit@3={m['hit@3']:.3f}  "
            f"Hit@5={m['hit@5']:.3f}  Hit@10={m['hit@10']:.3f}  MRR={m['mrr']:.3f}  Miss={m['miss']}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--queries", default=str(CACHE / "colloquial_queries_llm.json"))
    p.add_argument("--cache", default=str(CACHE))
    args = p.parse_args()

    cache = Path(args.cache)

    # Load
    print("[load] words...")
    with open(cache / "words.pkl", "rb") as f:
        words = pickle.load(f)
    print("[load] index...")
    index = SearchIndex.load(cache / "search_index.pkl")
    print("[load] embeddings...")
    emb = EmbeddingIndex.try_load(cache / "embedding_index")

    print("[load] LLM queries...")
    with open(args.queries, "r", encoding="utf-8") as f:
        raw_queries = json.load(f)
    print(f"  loaded {len(raw_queries)} queries")

    # Filter out bad queries
    queries: List[Tuple[str, List[int]]] = []
    skipped = 0
    for q in raw_queries:
        query_text = q["query"].strip()
        if len(query_text) < 2:
            skipped += 1
            continue
        queries.append((query_text, [q["word_id"]]))
    print(f"  valid queries: {len(queries)}, skipped: {skipped}")

    # Build retrievers
    r_bm = Retriever(words, index, embedding_index=None)
    r_hy = Retriever(words, index, embedding_index=emb) if emb else None
    r_vec = VectorOnlyRetriever(words, emb) if emb else None

    systems = {"bm25_only": r_bm}
    if r_vec:
        systems["vector_only"] = r_vec
    if r_hy:
        systems["hybrid"] = r_hy

    # Evaluate
    print(f"\n[eval] {len(queries)} queries x {len(systems)} systems")
    results = {}
    for name, ret in systems.items():
        print(f"  evaluating {name}...")
        results[name] = evaluate_batch(ret, queries)

    # Report
    print(f"\n{'='*80}")
    print(f"LLM-GENERATED COLLOQUIAL QUERIES ({len(queries)} queries)")
    print("=" * 80)
    print(f"{'System':20s} {'Hit@1':>8s} {'Hit@3':>8s} {'Hit@5':>8s} {'Hit@10':>8s} {'MRR':>8s} {'Miss':>6s}")
    print("-" * 80)
    for name in systems:
        m = results[name]
        print(f"{name:20s} {m['hit@1']:8.3f} {m['hit@3']:8.3f} {m['hit@5']:8.3f} {m['hit@10']:8.3f} {m['mrr']:8.3f} {m['miss']:5d}")

    # Comparison with previous experiments
    print(f"\n{'='*80}")
    print("COMPARISON WITH PREVIOUS EXPERIMENTS")
    print("=" * 80)
    prev = {
        "Exp5 释义原文 (8880)": {"bm25": 0.799, "vector": 0.591, "hybrid": 0.727},
        "Exp6 规则截短 (8541)": {"bm25": 0.700, "vector": 0.433, "hybrid": 0.586},
        "Exp4 手写200条": {"bm25": 0.389, "vector": 0.639, "hybrid": 0.590},
    }
    print(f"{'Dataset':30s} {'BM25 MRR':>10s} {'Vec MRR':>10s} {'Hyb MRR':>10s} {'Best':>8s}")
    print("-" * 80)
    for label, mrrs in prev.items():
        best = max(mrrs, key=mrrs.get)
        print(f"{label:30s} {mrrs['bm25']:10.3f} {mrrs['vector']:10.3f} {mrrs['hybrid']:10.3f} {best:>8s}")
    cur_mrrs = {
        "bm25": results.get("bm25_only", {}).get("mrr", 0),
        "vector": results.get("vector_only", {}).get("mrr", 0),
        "hybrid": results.get("hybrid", {}).get("mrr", 0),
    }
    best = max(cur_mrrs, key=cur_mrrs.get)
    print(f"{'LLM口语化 (本次)':30s} {cur_mrrs['bm25']:10.3f} {cur_mrrs['vector']:10.3f} {cur_mrrs['hybrid']:10.3f} {best:>8s}")

    # Head-to-head: hybrid vs bm25
    if r_hy and r_bm:
        print(f"\n{'='*80}")
        print("HEAD-TO-HEAD: hybrid vs bm25")
        print("=" * 80)
        hy_wins = bm_wins = ties = 0
        both_miss = only_hy = only_bm = 0
        for q_text, wids in queries:
            expected = set(wids)
            bm_hits = r_bm.search(q_text, top_k=10)
            hy_hits = r_hy.search(q_text, top_k=10)
            def rank_of(hits):
                for i, h in enumerate(hits):
                    if h.word_id in expected:
                        return i + 1
                return None
            br = rank_of(bm_hits)
            hr = rank_of(hy_hits)
            brr = (1.0/br) if br else 0
            hrr = (1.0/hr) if hr else 0
            if hrr > brr: hy_wins += 1
            elif brr > hrr: bm_wins += 1
            else: ties += 1
            if not br and not hr: both_miss += 1
            elif hr and not br: only_hy += 1
            elif br and not hr: only_bm += 1
        print(f"  hybrid赢: {hy_wins}  |  bm25赢: {bm_wins}  |  平: {ties}")
        print(f"  都没找到: {both_miss}  |  只hybrid找到: {only_hy}  |  只bm25找到: {only_bm}")

    # Save metrics
    metrics_path = cache / "eval_llm_queries_results.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[done] metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
