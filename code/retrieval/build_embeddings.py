"""Build the vector embedding index from TSV data.

First run downloads BAAI/bge-small-zh-v1.5 (~95MB) to HuggingFace cache.
Encoding 22k documents on CPU takes a few minutes.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_bot.embeddings import DEFAULT_MODEL, EmbeddingIndex  # noqa: E402
from rag_bot.loader import FoochowDataLoader  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--resources",
        default=str(Path(__file__).resolve().parent.parent.parent / "foochow-server" / "resources"),
    )
    p.add_argument(
        "--cache",
        default=str(Path(__file__).resolve().parent.parent / "data"),
    )
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    cache = Path(args.cache)
    words_pkl = cache / "words.pkl"

    # Prefer cached Words dict so we don't reparse TSVs
    if words_pkl.exists():
        print(f"[load] using cached words from {words_pkl}")
        with open(words_pkl, "rb") as f:
            words = pickle.load(f)
    else:
        print(f"[load] parsing TSVs from {args.resources}")
        loader = FoochowDataLoader(args.resources)
        words = loader.load_all(published_only=True)
        cache.mkdir(parents=True, exist_ok=True)
        with open(words_pkl, "wb") as f:
            pickle.dump(words, f, protocol=pickle.HIGHEST_PROTOCOL)

    idx = EmbeddingIndex(model_name=args.model)
    idx.build(words, batch_size=args.batch_size)
    idx.save(cache / "embedding_index")
    print("[done]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
