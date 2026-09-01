"""Rerank evaluation with Qwen3-Reranker (yes/no causal LM approach)."""

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
    h1 = sum(1 for r in rrs if r >= 1.0)
    h3 = sum(1 for r in rrs if r >= 1/3)
    h5 = sum(1 for r in rrs if r >= 1/5)
    h10 = sum(1 for r in rrs if r > 0)
    miss = sum(1 for r in rrs if r == 0)
    return {
        "n": n, "miss": miss,
        "hit@1": h1/n, "hit@3": h3/n, "hit@5": h5/n, "hit@10": h10/n,
        "mrr": sum(rrs)/n,
    }


def fmt(m: dict) -> str:
    return (f"n={m['n']:5d}  Hit@1={m['hit@1']:.3f}  Hit@3={m['hit@3']:.3f}  "
            f"Hit@5={m['hit@5']:.3f}  Hit@10={m['hit@10']:.3f}  MRR={m['mrr']:.3f}  Miss={m['miss']}")


# Qwen3-Reranker prompt format from official model card
INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
PREFIX = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def format_input(query: str, doc: str, instruction: str = INSTRUCTION) -> str:
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--queries", required=True)
    p.add_argument("--cache", default=str(CACHE))
    p.add_argument("--reranker", default="Qwen/Qwen3-Reranker-0.6B")
    p.add_argument("--candidate-pool", type=int, default=50)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
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

    print("[prep] building doc texts...")
    doc_text: Dict[int, str] = {wid: build_semantic_doc(w) for wid, w in words.items()}

    print(f"[load] queries from {args.queries}")
    with open(args.queries, "r", encoding="utf-8") as f:
        raw_queries = json.load(f)
    queries: List[Tuple[str, List[int]]] = []
    for q in raw_queries:
        text = q["query"].strip()
        if len(text) < 2:
            continue
        if "expected_word_ids" in q:
            wids = q["expected_word_ids"]
        elif "word_id" in q:
            wids = [q["word_id"]]
        else:
            continue
        queries.append((text, wids))
    n = len(queries)
    print(f"[load] {n} valid queries")

    # Stage 1: hybrid retrieval
    print(f"\n[stage1] retrieving hybrid top-{args.candidate_pool}...")
    t0 = time.time()
    candidates: List[List[int]] = []
    pre_rrs: List[float] = []
    for q, expected_wids in queries:
        hits = retriever.search(q, top_k=args.candidate_pool)
        wids = [h.word_id for h in hits]
        candidates.append(wids)
        expected = set(expected_wids)
        rr = 0.0
        for i, wid in enumerate(wids[:args.top_k]):
            if wid in expected:
                rr = 1.0 / (i + 1); break
        pre_rrs.append(rr)
    print(f"[stage1] done in {time.time()-t0:.0f}s")
    pre_metrics = metrics(pre_rrs, n)
    print(f"  pre-rerank: {fmt(pre_metrics)}")

    # Stage 2: Qwen3-Reranker
    print(f"\n[stage2] loading reranker: {args.reranker}")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.reranker, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(
        args.reranker, torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()

    token_yes = tokenizer.convert_tokens_to_ids("yes")
    token_no = tokenizer.convert_tokens_to_ids("no")
    prefix_ids = tokenizer.encode(PREFIX, add_special_tokens=False)
    suffix_ids = tokenizer.encode(SUFFIX, add_special_tokens=False)
    max_length = 8192

    def score_pairs(pairs: List[Tuple[str, str]]) -> List[float]:
        # pairs = [(query, doc), ...]
        scores = []
        for batch_start in range(0, len(pairs), args.batch_size):
            batch = pairs[batch_start:batch_start + args.batch_size]
            texts = [format_input(q, d) for q, d in batch]
            inputs = tokenizer(texts, return_tensors="pt", padding=True,
                               truncation=True, max_length=max_length - len(prefix_ids) - len(suffix_ids),
                               add_special_tokens=False).to("cuda")
            # Wrap with prefix/suffix
            input_ids_list = []
            attn_list = []
            for i in range(len(batch)):
                ids = prefix_ids + inputs["input_ids"][i].tolist() + suffix_ids
                attn = [1] * len(ids)
                input_ids_list.append(ids)
                attn_list.append(attn)
            # Pad to longest in batch
            max_len = max(len(x) for x in input_ids_list)
            pad_id = tokenizer.pad_token_id or 0
            for i in range(len(input_ids_list)):
                pad_n = max_len - len(input_ids_list[i])
                input_ids_list[i] = [pad_id] * pad_n + input_ids_list[i]
                attn_list[i] = [0] * pad_n + attn_list[i]
            input_ids = torch.tensor(input_ids_list, device="cuda")
            attention_mask = torch.tensor(attn_list, device="cuda")

            with torch.no_grad():
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, -1, :]
                yes_logit = logits[:, token_yes]
                no_logit = logits[:, token_no]
                # Score = P(yes) via softmax over {yes, no}
                stacked = torch.stack([no_logit, yes_logit], dim=1)
                probs = torch.softmax(stacked, dim=1)
                yes_probs = probs[:, 1].cpu().tolist()
                scores.extend(yes_probs)
        return scores

    print(f"[stage2] reranking {n} queries × up to {args.candidate_pool} candidates...")
    t0 = time.time()
    post_rrs: List[float] = []
    chunk_size = 8
    for chunk_start in range(0, n, chunk_size):
        chunk = queries[chunk_start:chunk_start + chunk_size]
        chunk_cands = candidates[chunk_start:chunk_start + chunk_size]

        pairs: List[Tuple[str, str]] = []
        pair_owner: List[int] = []
        for i, ((q, _exp), cands) in enumerate(zip(chunk, chunk_cands)):
            for wid in cands:
                pairs.append((q, doc_text.get(wid, "")))
                pair_owner.append(i)

        if pairs:
            scores = score_pairs(pairs)
        else:
            scores = []

        per_query_scores: List[List[Tuple[int, float]]] = [[] for _ in chunk]
        for idx, score in enumerate(scores):
            i = pair_owner[idx]
            wid = chunk_cands[i][len(per_query_scores[i])]
            per_query_scores[i].append((wid, float(score)))

        for i, (q, expected_wids) in enumerate(chunk):
            scored = sorted(per_query_scores[i], key=lambda kv: -kv[1])
            expected = set(expected_wids)
            rr = 0.0
            for j, (wid, _s) in enumerate(scored[:args.top_k]):
                if wid in expected:
                    rr = 1.0 / (j + 1); break
            post_rrs.append(rr)

        done = min(chunk_start + chunk_size, n)
        elapsed = time.time() - t0
        speed = done / elapsed if elapsed > 0 else 0
        if (done % 80 == 0) or done == n:
            print(f"  [{done}/{n}] {speed:.2f} q/s, ETA {(n-done)/speed:.0f}s")

    elapsed = time.time() - t0
    print(f"[stage2] done in {elapsed:.0f}s ({n/elapsed:.2f} q/s)")

    post_metrics = metrics(post_rrs, n)
    print(f"\n{'=' * 70}")
    print(f"RESULTS — {args.queries}")
    print(f"{'=' * 70}")
    print(f"  pre-rerank  (hybrid top-{args.top_k}):  {fmt(pre_metrics)}")
    print(f"  post-rerank (Qwen3-Reranker top-{args.top_k}): {fmt(post_metrics)}")

    delta = post_metrics["mrr"] - pre_metrics["mrr"]
    pct = 100 * delta / pre_metrics["mrr"] if pre_metrics["mrr"] > 0 else 0
    print(f"\n  Δ MRR: {delta:+.3f} ({pct:+.1f}%)")
    print(f"  Δ Hit@1: {post_metrics['hit@1'] - pre_metrics['hit@1']:+.3f}")
    print(f"  Δ Hit@10: {post_metrics['hit@10'] - pre_metrics['hit@10']:+.3f}")
    print(f"  Δ Miss: {post_metrics['miss'] - pre_metrics['miss']:+d}")

    output_path = args.output or (cache / f"rerank_qwen3_{Path(args.queries).stem}.json")
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
