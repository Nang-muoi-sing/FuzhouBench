"""Build and persist search indexes for Word documents.

Index types:
  - BM25 full-text index over a concatenated searchable string per word
  - Exact-text map: text -> [word_id]
  - Prefix-text map: char -> [word_id] (first-char to word)
  - Yngping map: yngping_normalized -> [word_id]
  - Supplement map: supplement_text -> [word_id]

Tokenization strategy:
  - Chinese chars: each char becomes one token, plus all 2-grams
  - yngping: split by whitespace
  - Latin tokens: lowercase
"""

from __future__ import annotations

import pickle
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from rank_bm25 import BM25Okapi

from .loader import Word


_LATIN_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9]*")
_CJK_RE = re.compile(r"[\u3400-\u9fff\uf900-\ufaff\U00020000-\U0003ffff]")


def tokenize_chinese(text: str) -> list[str]:
    """Char-level tokens + 2-grams for Chinese."""
    if not text:
        return []
    chars = [c for c in text if _CJK_RE.match(c)]
    tokens = list(chars)
    # 2-grams
    for i in range(len(chars) - 1):
        tokens.append(chars[i] + chars[i + 1])
    return tokens


def tokenize_latin(text: str) -> list[str]:
    """Lowercase latin word tokens."""
    if not text:
        return []
    return [m.group(0).lower() for m in _LATIN_TOKEN_RE.finditer(text)]


def tokenize_yngping(text: str) -> list[str]:
    """Whitespace-separated yngping syllables, plus the full string (as phrase token)."""
    if not text:
        return []
    tokens = [t.lower() for t in text.split() if t]
    if len(tokens) > 1:
        tokens.append("".join(tokens))  # joined form for phrase matching
    return tokens


def normalize_yngping(y: str) -> str:
    """Canonical form: lowercase, collapsed whitespace."""
    return " ".join(y.lower().split())


def build_word_tokens(word: Word) -> list[str]:
    """Produce the token stream used for BM25 scoring of a Word."""
    tokens: list[str] = []
    # Headword: add with 3x weight to emphasize exact matches
    tokens.extend(tokenize_chinese(word.text) * 3)
    tokens.extend(tokenize_latin(word.text))

    # Gloss
    tokens.extend(tokenize_chinese(word.gloss))
    tokens.extend(tokenize_latin(word.gloss))

    # All pronunciations
    for p in word.prons:
        tokens.extend(tokenize_yngping(p.yngping))

    # Explanations (content + sentences)
    for e in word.expls:
        tokens.extend(tokenize_chinese(e.content))
        tokens.extend(tokenize_latin(e.content))
        for s in e.sentences:
            tokens.extend(tokenize_chinese(s))

    # Feng entries (content JSON)
    for f in word.fengs:
        tokens.extend(tokenize_yngping(f.literal_pron))
        tokens.extend(tokenize_yngping(f.sandhi_pron))
        for item in f.content:
            if isinstance(item, dict):
                tokens.extend(tokenize_chinese(str(item.get("expl", ""))))
                for s in item.get("sent", []) or []:
                    tokens.extend(tokenize_chinese(str(s)))

    # Supplements (alternative character forms, synonyms-ish)
    for sup in word.supplements:
        tokens.extend(tokenize_chinese(sup.text) * 2)

    # CikLing (historical phonology annotations)
    for ck in word.ciklings:
        tokens.extend(tokenize_chinese(ck.cik_annotation))
        tokens.extend(tokenize_chinese(ck.ling_annotation))

    return tokens


@dataclass
class SearchIndex:
    """Serializable search index over Words."""

    word_ids: list[int] = field(default_factory=list)  # position i -> word_id
    bm25: BM25Okapi | None = None
    text_map: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))
    yngping_map: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))
    yngping_prefix_map: dict[str, list[int]] = field(
        default_factory=lambda: defaultdict(list)
    )  # first syllable -> word_ids
    supplement_map: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "word_ids": self.word_ids,
                    "bm25": self.bm25,
                    "text_map": dict(self.text_map),
                    "yngping_map": dict(self.yngping_map),
                    "yngping_prefix_map": dict(self.yngping_prefix_map),
                    "supplement_map": dict(self.supplement_map),
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load(cls, path: str | Path) -> "SearchIndex":
        with open(path, "rb") as f:
            data = pickle.load(f)
        idx = cls(
            word_ids=data["word_ids"],
            bm25=data["bm25"],
            text_map=defaultdict(list, data["text_map"]),
            yngping_map=defaultdict(list, data["yngping_map"]),
            yngping_prefix_map=defaultdict(list, data["yngping_prefix_map"]),
            supplement_map=defaultdict(list, data["supplement_map"]),
        )
        return idx


def build_index(words: dict[int, Word]) -> SearchIndex:
    """Build a SearchIndex from loaded Words."""
    idx = SearchIndex()
    corpus_tokens: list[list[str]] = []

    sorted_items = sorted(words.items(), key=lambda kv: kv[0])
    for wid, w in sorted_items:
        idx.word_ids.append(wid)

        # Exact/substring-friendly maps
        idx.text_map[w.text].append(wid)

        for p in w.prons:
            norm = normalize_yngping(p.yngping)
            if norm:
                idx.yngping_map[norm].append(wid)
                first_syl = norm.split()[0] if norm else ""
                if first_syl:
                    idx.yngping_prefix_map[first_syl].append(wid)

        for sup in w.supplements:
            if sup.text:
                idx.supplement_map[sup.text].append(wid)

        # BM25 tokens
        tokens = build_word_tokens(w)
        if not tokens:
            tokens = ["_empty_"]  # avoid zero-length docs for BM25 stability
        corpus_tokens.append(tokens)

    print(f"[index] building BM25 over {len(corpus_tokens)} documents...")
    idx.bm25 = BM25Okapi(corpus_tokens)
    print(f"[index] done. text_map={len(idx.text_map)} yngping_map={len(idx.yngping_map)}")
    return idx
