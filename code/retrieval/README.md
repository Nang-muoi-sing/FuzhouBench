# Definition-retrieval pipeline

Code accompanying Section 6.3 (Task 3: Definition Retrieval) of the
manuscript.

## Layout

- `rag_bot/` -- core library (loader, BM25 index, dense embeddings,
  hybrid retriever, doc formatter).
- `build_index.py` -- builds the BM25 search index from the released
  TSVs (`../../data/`).
- `build_embeddings.py` -- builds the dense vector index using
  `BAAI/bge-large-zh-v1.5` (default).
- `eval_full_dict_v3.py` -- evaluates retrieval on the original-source
  query pool (8,880 queries extracted as the first sentence of each
  entry's explanation).
- `eval_llm_queries.py` -- evaluates retrieval on the LLM-generated
  colloquial-Mandarin query pool (11,230 queries; see
  `../../data/retrieval/colloquial_queries_qwen3_30b_a3b.json`).
- `gen_colloquial_queries.py` -- generates the colloquial-Mandarin
  query pool by prompting a local LLM (used here with Qwen3-30B-A3B)
  to rewrite each explanation as a natural Mandarin search phrase.
- `rerank_qwen3.py`, `rerank_eval.py` -- two-stage retrieval: BM25 + dense
  hybrid (RRF, K=60) then cross-encoder rerank with
  `BAAI/bge-reranker-v2-m3` over the top 50 candidates.

## Reproducing the paper's numbers

```bash
python build_index.py                              # BM25 index
python build_embeddings.py --model BAAI/bge-large-zh-v1.5
python eval_full_dict_v3.py                        # original-source pool
python eval_llm_queries.py                          # LLM-colloquial pool
python rerank_eval.py                               # two-stage reranking
```

API keys for any LLM-driven steps are read from `--api-key-file` or
from `ANTHROPIC_API_KEY` env (never hardcoded).
