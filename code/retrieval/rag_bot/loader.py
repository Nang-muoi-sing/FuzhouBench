"""Load foochow-server TSV resources into unified Word documents.

Each Word document aggregates:
  - Core fields from WordResource (text, phonology, gloss, tags, is_published)
  - All Pronunciations for this word
  - All Explanations (with example sentences)
  - All FengEntries (Feng Aizhen 1998 dictionary)
  - All CikLingEntries (Qilin Bayin historical phonetic text)
  - All SupplementWords (alternative/related characters)
  - All DFDCharacter entries
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class Pronunciation:
    yngping: str
    is_sandhi: bool
    is_primary: bool
    variant: str = ""
    source_type: str = ""
    note: str = ""


@dataclass
class Explanation:
    order: int
    lexical_category: str
    content: str
    comment: str = ""
    sentences: list[str] = field(default_factory=list)
    source_tags: str = ""


@dataclass
class FengEntry:
    text: str
    literal_pron: str
    sandhi_pron: str
    page_number: int
    page_order: int
    content: list[dict[str, Any]] = field(default_factory=list)  # [{expl, sent}, ...]
    original_comment: str = ""
    comment: str = ""


@dataclass
class CikLingEntry:
    text: str
    tone: str
    cik_initial: str
    cik_final: str
    ling_initial: str
    ling_final: str
    cik_annotation: str = ""
    ling_annotation: str = ""
    li_annotate_cik: str = ""
    li_annotate_ling: str = ""
    comment: str = ""


@dataclass
class SupplementWord:
    text: str
    type: str = ""  # M=main alt, S=secondary, N=non-alt, etc.
    glyph_category: str = ""
    glyph_type: str = ""


@dataclass
class DFDCharacter:
    text: str
    banguace: str
    page_number: int
    column_number: int
    row_number: int
    radical_id: int


@dataclass
class Word:
    id: int
    text: str
    gloss: str = ""
    is_published: bool = False
    phonology_initial: str = ""
    phonology_final: str = ""
    phonology_tone: str = ""
    glyph_comment: str = ""
    pron_comment: str = ""
    expl_comment: str = ""
    note: str = ""
    tags: str = ""

    prons: list[Pronunciation] = field(default_factory=list)
    expls: list[Explanation] = field(default_factory=list)
    fengs: list[FengEntry] = field(default_factory=list)
    ciklings: list[CikLingEntry] = field(default_factory=list)
    supplements: list[SupplementWord] = field(default_factory=list)
    dfd_chars: list[DFDCharacter] = field(default_factory=list)

    @property
    def primary_yngping(self) -> str:
        """Return the primary pronunciation if any, else first sandhi, else empty."""
        for p in self.prons:
            if p.is_primary:
                return p.yngping
        for p in self.prons:
            if p.is_sandhi:
                return p.yngping
        return self.prons[0].yngping if self.prons else ""

    @property
    def all_yngping(self) -> list[str]:
        """All unique yngping strings."""
        seen = []
        for p in self.prons:
            if p.yngping and p.yngping not in seen:
                seen.append(p.yngping)
        return seen

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if pd.isna(v):
        return False
    s = str(v).strip().lower()
    return s in ("1", "true", "t", "yes")


def _parse_int(v: Any, default: int = 0) -> int:
    if pd.isna(v):
        return default
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return default


def _parse_str(v: Any) -> str:
    if pd.isna(v):
        return ""
    return str(v).strip()


def _parse_json(v: Any, default: Any = None) -> Any:
    if pd.isna(v) or not v:
        return default if default is not None else []
    s = str(v).strip()
    if not s or s == "[]":
        return default if default is not None else []
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return default if default is not None else []


class FoochowDataLoader:
    """Load and join all foochow-server TSV resources."""

    def __init__(self, resources_dir: str | Path):
        self.dir = Path(resources_dir)
        if not self.dir.exists():
            raise FileNotFoundError(f"resources dir not found: {self.dir}")

    def _read_tsv(self, name: str) -> pd.DataFrame:
        path = self.dir / name
        return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=[""])

    def load_all(self, published_only: bool = True) -> dict[int, Word]:
        """Return dict of word_id -> Word with all related data joined."""
        print(f"[loader] reading TSVs from {self.dir}")

        words_df = self._read_tsv("WordResource.tsv")
        prons_df = self._read_tsv("PronunciationResource.tsv")
        expls_df = self._read_tsv("ExplanationResource.tsv")
        fengs_df = self._read_tsv("FengResource.tsv")
        cikl_df = self._read_tsv("CikLingResource.tsv")
        supp_df = self._read_tsv("SupplementWordResource.tsv")
        dfd_df = self._read_tsv("DFDCharacterResource.tsv")

        print(
            f"[loader] words={len(words_df)} prons={len(prons_df)} "
            f"expls={len(expls_df)} fengs={len(fengs_df)} "
            f"cikling={len(cikl_df)} supp={len(supp_df)} dfd={len(dfd_df)}"
        )

        # Build core Word records.
        words: dict[int, Word] = {}
        for row in words_df.itertuples(index=False):
            wid = _parse_int(row.id)
            if wid == 0:
                continue
            is_pub = _parse_bool(row.is_published)
            if published_only and not is_pub:
                continue
            words[wid] = Word(
                id=wid,
                text=_parse_str(row.text),
                gloss=_parse_str(row.gloss),
                is_published=is_pub,
                phonology_initial=_parse_str(row.phonology_initial),
                phonology_final=_parse_str(row.phonology_final),
                phonology_tone=_parse_str(row.phonology_tone),
                glyph_comment=_parse_str(row.glyph_comment),
                pron_comment=_parse_str(row.pron_comment),
                expl_comment=_parse_str(row.expl_comment),
                note=_parse_str(row.note),
                tags=_parse_str(row.tags),
            )

        # Attach related records.
        self._attach_prons(words, prons_df)
        self._attach_expls(words, expls_df)
        self._attach_fengs(words, fengs_df)
        self._attach_ciklings(words, cikl_df)
        self._attach_supplements(words, supp_df)
        self._attach_dfd(words, dfd_df)

        # Sort prons by primary first, then sandhi
        for w in words.values():
            w.prons.sort(key=lambda p: (not p.is_primary, not p.is_sandhi))
            w.expls.sort(key=lambda e: e.order)

        print(f"[loader] loaded {len(words)} published words")
        return words

    def _attach_prons(self, words: dict[int, Word], df: pd.DataFrame) -> None:
        for row in df.itertuples(index=False):
            wid = _parse_int(row.word__id)
            w = words.get(wid)
            if not w:
                continue
            w.prons.append(
                Pronunciation(
                    yngping=_parse_str(row.yngping),
                    is_sandhi=_parse_bool(row.is_sandhi),
                    is_primary=_parse_bool(row.is_primary),
                    variant=_parse_str(row.variant),
                    source_type=_parse_str(row.source_type),
                    note=_parse_str(row.note),
                )
            )

    def _attach_expls(self, words: dict[int, Word], df: pd.DataFrame) -> None:
        for row in df.itertuples(index=False):
            wid = _parse_int(row.word__id)
            w = words.get(wid)
            if not w:
                continue
            sentences_raw = _parse_json(row.sentences, default=[])
            if isinstance(sentences_raw, list):
                sentences = [str(s) for s in sentences_raw if s]
            else:
                sentences = []
            w.expls.append(
                Explanation(
                    order=_parse_int(row.order, 1),
                    lexical_category=_parse_str(row.lexical_category),
                    content=_parse_str(row.content),
                    comment=_parse_str(row.comment),
                    sentences=sentences,
                    source_tags=_parse_str(row.source_tags),
                )
            )

    def _attach_fengs(self, words: dict[int, Word], df: pd.DataFrame) -> None:
        for row in df.itertuples(index=False):
            wid = _parse_int(row.word__id)
            w = words.get(wid)
            if not w:
                continue
            content = _parse_json(row.content, default=[])
            if not isinstance(content, list):
                content = []
            w.fengs.append(
                FengEntry(
                    text=_parse_str(row.text),
                    literal_pron=_parse_str(row.literal_pron),
                    sandhi_pron=_parse_str(row.sandhi_pron),
                    page_number=_parse_int(row.page_number),
                    page_order=_parse_int(row.page_order),
                    content=content,
                    original_comment=_parse_str(row.original_comment),
                    comment=_parse_str(row.comment),
                )
            )

    def _attach_ciklings(self, words: dict[int, Word], df: pd.DataFrame) -> None:
        for row in df.itertuples(index=False):
            wid = _parse_int(row.word__id)
            w = words.get(wid)
            if not w:
                continue
            w.ciklings.append(
                CikLingEntry(
                    text=_parse_str(row.text),
                    tone=_parse_str(row.tone),
                    cik_initial=_parse_str(row.cik_initial),
                    cik_final=_parse_str(row.cik_final),
                    ling_initial=_parse_str(row.ling_initial),
                    ling_final=_parse_str(row.ling_final),
                    cik_annotation=_parse_str(row.cik_annotation),
                    ling_annotation=_parse_str(row.ling_annotation),
                    li_annotate_cik=_parse_str(row.li_annotate_cik),
                    li_annotate_ling=_parse_str(row.li_annotate_ling),
                    comment=_parse_str(row.comment),
                )
            )

    def _attach_supplements(self, words: dict[int, Word], df: pd.DataFrame) -> None:
        for row in df.itertuples(index=False):
            wid = _parse_int(row.word__id)
            w = words.get(wid)
            if not w:
                continue
            w.supplements.append(
                SupplementWord(
                    text=_parse_str(row.text),
                    type=_parse_str(row.type),
                    glyph_category=_parse_str(row.glyph_category),
                    glyph_type=_parse_str(row.glyph_type),
                )
            )

    def _attach_dfd(self, words: dict[int, Word], df: pd.DataFrame) -> None:
        for row in df.itertuples(index=False):
            wid = _parse_int(row.word__id)
            w = words.get(wid)
            if not w:
                continue
            w.dfd_chars.append(
                DFDCharacter(
                    text=_parse_str(row.text),
                    banguace=_parse_str(row.banguace),
                    page_number=_parse_int(row.page_number),
                    column_number=_parse_int(row.column_number),
                    row_number=_parse_int(row.row_number),
                    radical_id=_parse_int(row.radical_id),
                )
            )
