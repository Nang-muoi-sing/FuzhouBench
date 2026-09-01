"""Reproduce the paper's Table 8 (definition retrieval) on the 200
human-written (native-speaker) Group-B queries, using SiliconFlow-hosted
models for both dense embedding and cross-encoder reranking.

Systems (matching Table 8 rows):
  1. BM25 only                (local rank_bm25, char + 2-gram tokens)
  2. Vector only              (BAAI/bge-large-zh-v1.5 via SiliconFlow)
  3. Hybrid RRF (K=60)        (fuse 1+2 rankings)
  4. Two-stage: RRF top-50 -> BAAI/bge-reranker-v2-m3 (SiliconFlow) -> top-10

Metrics: Hit@1, Hit@5, Hit@10, MRR (reciprocal rank within top-10).

Usage:
  python papers/scripts/table8_human200_siliconflow.py \
      --api-key-file .siliconflow_key \
      --out papers/data/table8_human200_sf.json

Doc embeddings are cached to papers/data/sf_doc_embeddings.npz after the
first run (~23k docs, a few minutes of SiliconFlow embedding calls).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "rag-bot"))

from rag_bot.loader import FoochowDataLoader, Word          # noqa: E402
from rag_bot.index import build_index, tokenize_chinese, tokenize_latin  # noqa: E402
from rag_bot.embeddings import build_semantic_doc            # noqa: E402

SF_BASE = "https://api.siliconflow.cn/v1"
EMB_MODEL = "BAAI/bge-large-zh-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
RRF_K = 60
DOC_CHAR_CAP = 480    # bge ctx is 512 tokens; CJK ~1 token/char


# --- SiliconFlow API helpers ------------------------------------------------

def sf_embed_batch(api_key: str, texts: list[str], retries: int = 4) -> list[list[float]]:
    body = json.dumps({"model": EMB_MODEL, "input": texts}).encode("utf-8")
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                f"{SF_BASE}/embeddings", data=body,
                headers={"Authorization": f"Bearer {api_key}",
                         "Content-Type": "application/json"})
            resp = json.load(urllib.request.urlopen(req, timeout=120))
            data = sorted(resp["data"], key=lambda d: d["index"])
            return [d["embedding"] for d in data]
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


def sf_rerank(api_key: str, query: str, documents: list[str], retries: int = 4) -> list[int]:
    """Return document indices in reranked order (best first)."""
    body = json.dumps({
        "model": RERANK_MODEL, "query": query,
        "documents": documents, "top_n": len(documents),
    }).encode("utf-8")
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                f"{SF_BASE}/rerank", data=body,
                headers={"Authorization": f"Bearer {api_key}",
                         "Content-Type": "application/json"})
            resp = json.load(urllib.request.urlopen(req, timeout=120))
            ranked = sorted(resp["results"], key=lambda r: -r["relevance_score"])
            return [r["index"] for r in ranked]
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


# --- metrics -----------------------------------------------------------------

def rr_at_k(ranking: list[int], expected: set[int], k: int = 10) -> float:
    for i, wid in enumerate(ranking[:k]):
        if wid in expected:
            return 1.0 / (i + 1)
    return 0.0


def metrics(rrs: list[float]) -> dict:
    n = len(rrs)
    return {
        "n": n,
        "hit@1": sum(1 for r in rrs if r >= 1.0) / n,
        "hit@5": sum(1 for r in rrs if r >= 1 / 5) / n,
        "hit@10": sum(1 for r in rrs if r > 0) / n,
        "mrr": sum(rrs) / n,
    }


# --- main ---------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--api-key-file", type=Path, required=True)
    p.add_argument("--queries", type=Path,
                   default=REPO / "rag-bot" / "data" / "eval_mandarin_dataset_v3.json")
    p.add_argument("--resources-dir", type=Path,
                   default=REPO / "foochow-server" / "resources")
    p.add_argument("--emb-cache", type=Path,
                   default=REPO / "papers" / "data" / "sf_doc_embeddings.npz")
    p.add_argument("--candidate-pool", type=int, default=50)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--emb-batch", type=int, default=64)
    p.add_argument("--emb-concurrency", type=int, default=6)
    p.add_argument("--rerank-concurrency", type=int, default=6)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    api_key = args.api_key_file.read_text(encoding="utf-8").strip()

    # ---- load corpus ----
    print("[load] words...", file=sys.stderr)
    loader = FoochowDataLoader(args.resources_dir)
    words: dict[int, Word] = loader.load_all(published_only=True)
    # Keep entries with at least one pronunciation, matching the paper's 22,883 pool.
    words = {wid: w for wid, w in words.items() if w.prons}
    print(f"[load] {len(words)} words with pronunciation", file=sys.stderr)

    print("[bm25] building index...", file=sys.stderr)
    index = build_index(words)
    wid_order: list[int] = index.word_ids

    print("[docs] building semantic docs...", file=sys.stderr)
    docs = {wid: build_semantic_doc(w)[:DOC_CHAR_CAP] for wid, w in words.items()}

    # ---- doc embeddings (cached) ----
    doc_wids = list(words.keys())
    doc_mat = None
    if args.emb_cache.is_file():
        z = np.load(args.emb_cache, allow_pickle=True)
        if "word_ids" in z.files:
            # production embedding cache: word_ids / vectors (bge-large-zh-v1.5,
            # L2-normalized, same build_semantic_doc). Row-align to our corpus.
            cache_wids = z["word_ids"].tolist()
            cache_vecs = z["vectors"]
            model_tag = str(z["embed_model"]) if "embed_model" in z.files else "?"
            pos = {wid: i for i, wid in enumerate(cache_wids)}
            missing = [wid for wid in doc_wids if wid not in pos]
            if missing:
                print(f"[emb] WARN {len(missing)} corpus words absent from cache; "
                      f"they will be unreachable via vector retrieval", file=sys.stderr)
            rows = np.array([pos.get(wid, -1) for wid in doc_wids])
            doc_mat = np.zeros((len(doc_wids), cache_vecs.shape[1]), dtype=np.float32)
            valid = rows >= 0
            doc_mat[valid] = cache_vecs[rows[valid]]
            print(f"[emb] loaded production cache {args.emb_cache} "
                  f"(model={model_tag}, {valid.sum()}/{len(doc_wids)} covered)", file=sys.stderr)
        elif "wids" in z.files and z["wids"].tolist() == doc_wids:
            doc_mat = z["vecs"]
            print(f"[emb] loaded cache {args.emb_cache} {doc_mat.shape}", file=sys.stderr)
        else:
            print("[emb] cache stale (word list changed); re-embedding", file=sys.stderr)

    if doc_mat is None:
        texts = [docs[wid] for wid in doc_wids]
        batches = [texts[i:i + args.emb_batch] for i in range(0, len(texts), args.emb_batch)]
        print(f"[emb] embedding {len(texts)} docs in {len(batches)} batches "
              f"(concurrency={args.emb_concurrency})...", file=sys.stderr)
        results: list = [None] * len(batches)
        t0 = time.time()
        with cf.ThreadPoolExecutor(max_workers=args.emb_concurrency) as pool:
            futs = {pool.submit(sf_embed_batch, api_key, b): i for i, b in enumerate(batches)}
            done = 0
            for fut in cf.as_completed(futs):
                results[futs[fut]] = fut.result()
                done += 1
                if done % 50 == 0:
                    print(f"[emb] {done}/{len(batches)} batches ({time.time()-t0:.0f}s)", file=sys.stderr)
        doc_mat = np.array([v for batch in results for v in batch], dtype=np.float32)
        doc_mat /= np.linalg.norm(doc_mat, axis=1, keepdims=True)
        args.emb_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.emb_cache, wids=np.array(doc_wids), vecs=doc_mat)
        print(f"[emb] cached to {args.emb_cache}  ({time.time()-t0:.0f}s total)", file=sys.stderr)

    wid_to_row = {wid: i for i, wid in enumerate(doc_wids)}

    # ---- load 200 human-written Group-B queries ----
    raw = json.loads(args.queries.read_text(encoding="utf-8"))
    queries = [(q["query"].strip(), set(q["expected_word_ids"]))
               for q in raw if q.get("group") == "B" and len(q["query"].strip()) >= 2]
    print(f"[queries] {len(queries)} human-written Group-B queries", file=sys.stderr)

    # ---- query embeddings ----
    q_texts = [q for q, _ in queries]
    q_vecs = []
    for i in range(0, len(q_texts), args.emb_batch):
        q_vecs.extend(sf_embed_batch(api_key, q_texts[i:i + args.emb_batch]))
    q_mat = np.array(q_vecs, dtype=np.float32)
    q_mat /= np.linalg.norm(q_mat, axis=1, keepdims=True)

    # ---- per-query rankings ----
    pool_n = args.candidate_pool
    k = args.top_k
    bm25_rrs, vec_rrs, rrf_rrs = [], [], []
    rrf_candidates: list[list[int]] = []

    print("[rank] BM25 / vector / RRF...", file=sys.stderr)
    sims_all = q_mat @ doc_mat.T          # (200, N)
    for qi, (q, expected) in enumerate(queries):
        # BM25 ranking
        tokens = tokenize_chinese(q) + tokenize_latin(q)
        scores = index.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:pool_n]
        bm25_ranking = [index.word_ids[int(i)] for i in top_idx if scores[int(i)] > 0]

        # Vector ranking
        sims = sims_all[qi]
        v_idx = np.argsort(sims)[::-1][:pool_n]
        vec_ranking = [doc_wids[int(i)] for i in v_idx]

        # RRF fusion
        rrf: dict[int, float] = {}
        for rank, wid in enumerate(bm25_ranking):
            rrf[wid] = rrf.get(wid, 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, wid in enumerate(vec_ranking):
            rrf[wid] = rrf.get(wid, 0.0) + 1.0 / (RRF_K + rank + 1)
        rrf_ranking = [wid for wid, _ in sorted(rrf.items(), key=lambda kv: -kv[1])][:pool_n]

        bm25_rrs.append(rr_at_k(bm25_ranking, expected, k))
        vec_rrs.append(rr_at_k(vec_ranking, expected, k))
        rrf_rrs.append(rr_at_k(rrf_ranking, expected, k))
        rrf_candidates.append(rrf_ranking)

    # ---- Stage 2: SF rerank ----
    print(f"[rerank] {len(queries)} queries x top-{pool_n} via {RERANK_MODEL}...", file=sys.stderr)

    def rerank_one(qi: int):
        q, expected = queries[qi]
        cands = rrf_candidates[qi]
        cand_docs = [docs[wid] for wid in cands]
        order = sf_rerank(api_key, q, cand_docs)
        reranked = [cands[j] for j in order]
        return qi, rr_at_k(reranked, expected, k)

    rerank_rrs: list = [0.0] * len(queries)
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=args.rerank_concurrency) as pool2:
        futs = [pool2.submit(rerank_one, qi) for qi in range(len(queries))]
        done = 0
        for fut in cf.as_completed(futs):
            qi, rr = fut.result()
            rerank_rrs[qi] = rr
            done += 1
            if done % 50 == 0:
                print(f"[rerank] {done}/{len(queries)} ({time.time()-t0:.0f}s)", file=sys.stderr)

    # ---- report ----
    table = {
        "BM25": metrics(bm25_rrs),
        "Vector (bge-large-zh-v1.5, SF)": metrics(vec_rrs),
        "Hybrid RRF (K=60)": metrics(rrf_rrs),
        f"RRF top-{pool_n} + rerank (bge-reranker-v2-m3, SF)": metrics(rerank_rrs),
    }
    out = {
        "protocol": {
            "queries": "200 human-written Group-B (manual_rewrites.py, eval_mandarin_dataset_v3.json)",
            "corpus": len(words),
            "candidate_pool": pool_n, "top_k": k, "rrf_k": RRF_K,
            "embedding_model": EMB_MODEL, "rerank_model": RERANK_MODEL,
            "backend": "siliconflow",
        },
        "table": table,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.out}", file=sys.stderr)

    print(f"\n{'System':52s}  Hit@1   Hit@5   Hit@10  MRR")
    for name, m in table.items():
        print(f"{name:52s}  {m['hit@1']:.3f}   {m['hit@5']:.3f}   {m['hit@10']:.3f}   {m['mrr']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
