"""One-shot script to build the search index from foochow-server TSV resources."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

# Allow running this script directly without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_bot.index import build_index  # noqa: E402
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
    args = p.parse_args()

    loader = FoochowDataLoader(args.resources)
    words = loader.load_all(published_only=True)
    index = build_index(words)

    cache_dir = Path(args.cache)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_dir / "words.pkl", "wb") as f:
        pickle.dump(words, f, protocol=pickle.HIGHEST_PROTOCOL)
    index.save(cache_dir / "search_index.pkl")
    print(f"[done] cache written to {cache_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
