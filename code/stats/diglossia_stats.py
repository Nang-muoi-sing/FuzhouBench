"""Compute literary-colloquial diglossia statistics over the FuzhouBench
dictionary, as reported in Section 2 of the manuscript.

Methodology
-----------
Take every Feng entry whose word__text is a multi-character compound
(>= 2 CJK characters, skipping `*` placeholder positions). Align character
positions to the space-separated syllables of `literal_pron` (citation
form, not sandhi). For each unique character, collect the SET of distinct
segmental forms seen across compounds, where the segmental form is the
syllable with tone digits stripped. So `dai53` and `dai55` count as one
segmental form `dai`; `dai53` and `duai55` count as two.

Report
------
- Distribution of distinct segmental forms per unique character.
- Conditional rate restricted to characters seen in >= 10 different
  compounds (frequency control: rare characters have too few observations
  to expose diglossia).
- Top 20 characters by number of distinct forms, for spot-check.

Usage
-----
  python diglossia_stats.py
  python diglossia_stats.py --data-dir ../data

Default --data-dir resolves to a sibling `data/` directory, matching the
supplementary-bundle layout.
"""

from __future__ import annotations

import argparse
import collections
import csv
import re
import sys
from pathlib import Path

TONE_RE = re.compile(r"(\d+)")
PARSER_ARTIFACTS = {"*", "々", "〇"}     # ditto marks etc., not real chars


def compute(word_tsv: Path, feng_tsv: Path):
    word_text = {}
    with word_tsv.open(encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            word_text[r["id"]] = r["text"]

    char_seg: dict[str, set[str]] = collections.defaultdict(set)
    char_occur: collections.Counter = collections.Counter()

    aligned = 0
    mismatch = 0
    with feng_tsv.open(encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            text = r["word__text"]
            lit = r["literal_pron"]
            if not lit:
                continue
            clean_chars = [c for c in text if c != "*" and c.strip()]
            if len(clean_chars) < 2:
                continue
            sylls = lit.split()
            if len(sylls) != len(clean_chars):
                mismatch += 1
                continue
            for ch, syl in zip(clean_chars, sylls):
                char_seg[ch].add(TONE_RE.sub("", syl))
                char_occur[ch] += 1
            aligned += 1

    # Drop parser-artifact characters from the analysis pool.
    for ch in list(char_seg):
        if ch in PARSER_ARTIFACTS:
            del char_seg[ch]
            del char_occur[ch]

    return char_seg, char_occur, aligned, mismatch


def report(char_seg, char_occur, aligned, mismatch, threshold: int = 10) -> None:
    total_unique = len(char_seg)
    ge2 = sum(1 for s in char_seg.values() if len(s) >= 2)
    ge3 = sum(1 for s in char_seg.values() if len(s) >= 3)
    ge4 = sum(1 for s in char_seg.values() if len(s) >= 4)

    chars_freq = {ch for ch, c in char_occur.items() if c >= threshold}
    ge2_freq = sum(1 for ch in chars_freq if len(char_seg[ch]) >= 2)

    pct = lambda n, d: f"{100*n/d:.1f}%" if d else "n/a"

    out = [
        f"Aligned multi-char compounds: {aligned:,}",
        f"Alignment mismatches (skipped): {mismatch:,}",
        f"Unique characters observed:    {total_unique:,}",
        "",
        "Diglossia (distinct segmental forms per unique character):",
        f"  >=2 forms:  {ge2:,} ({pct(ge2, total_unique)})",
        f"  >=3 forms:  {ge3:,} ({pct(ge3, total_unique)})",
        f"  >=4 forms:  {ge4:,} ({pct(ge4, total_unique)})",
        "",
        f"Frequency control: characters appearing in >={threshold} different compounds",
        f"  population:        {len(chars_freq):,}",
        f"  with >=2 forms:    {ge2_freq:,} ({pct(ge2_freq, len(chars_freq))})",
        "",
        "Top 20 characters by number of distinct segmental forms:",
    ]
    print("\n".join(out))

    ranked = sorted(char_seg.items(), key=lambda kv: -len(kv[1]))[:20]
    for ch, forms in ranked:
        forms_preview = sorted(forms)[:6]
        more = " ..." if len(forms) > 6 else ""
        print(f"  {ch} ({len(forms)}, in {char_occur[ch]} compounds): {forms_preview}{more}")


def main() -> int:
    default_data = Path(__file__).resolve().parent.parent / "data"
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data-dir", type=Path, default=default_data,
                   help=f"Directory containing WordResource.tsv and FengResource.tsv "
                        f"(default: {default_data})")
    p.add_argument("--threshold", type=int, default=10,
                   help="Minimum compound count for the frequency-controlled rate (default: 10)")
    args = p.parse_args()

    word_tsv = args.data_dir / "WordResource.tsv"
    feng_tsv = args.data_dir / "FengResource.tsv"
    if not word_tsv.is_file() or not feng_tsv.is_file():
        sys.stderr.write(f"error: expected {word_tsv} and {feng_tsv} to exist\n")
        return 1

    char_seg, char_occur, aligned, mismatch = compute(word_tsv, feng_tsv)
    report(char_seg, char_occur, aligned, mismatch, args.threshold)
    return 0


if __name__ == "__main__":
    sys.exit(main())
