"""Full-sandhi symbolic rule baseline: tone + rime + initial-consonant
assimilation, learned from the train pool.

The paper's context-rule baseline (Table 6, 34.6% word-exact) rewrites
only the tone digits of non-final syllables. Reviewers asked what the
ceiling looks like when the symbolic model also handles the other two
surface alternations that Fuzhou sandhi produces:

  1. Rime alternation on non-final syllables (e.g. kooyng242 -> keoyng53,
     where the vowel changes together with the tone).
  2. Initial-consonant assimilation on non-initial syllables
     (citation s,d,g -> l,l,0 etc., conditioned on how the preceding
     syllable ends).

Components, all learned by majority vote from the same train pool as the
paper's context rule (no hand-written phonology):

  T  (curr_tone, next_tone)          -> sandhi_tone      [paper's rule]
  R  (rime, curr_tone, next_tone)    -> sandhi_rime      backoff: (rime, curr_tone) -> rime; identity
  I  (prev_coda_class, onset)        -> sandhi_onset     backoff: identity
     where prev_coda_class in {vowel, nasal, stop} from the PREDICTED
     surface form of the preceding syllable.

Same test split as the paper: 500 multi-syllable Feng entries, seed 42.

Usage:
  python papers/scripts/sandhi_full_rule.py \
      --resources-dir foochow-server/resources \
      --out papers/data/sandhi_full_rule.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

YNGPING_TOKEN_RE = re.compile(r"([a-z]+)(\d+)")

# Longest-first onset inventory for Yngping.
ONSETS = ["ng", "b", "p", "m", "d", "t", "n", "l", "g", "k", "h", "z", "c", "s", "j", "w", "y"]


def parse_yngping(s: str) -> list[tuple[str, str]]:
    s = s.strip().lower()
    syls = []
    for tok in s.split():
        m = YNGPING_TOKEN_RE.match(tok)
        if m:
            syls.append((m.group(1), m.group(2)))
    return syls


def join_yngping(syls: list[tuple[str, str]]) -> str:
    return " ".join(f"{seg}{tone}" for seg, tone in syls)


def split_onset(seg: str) -> tuple[str, str]:
    """Split segment into (onset, rime). Zero onset -> ('', seg)."""
    for o in ONSETS:
        if seg.startswith(o):
            rest = seg[len(o):]
            # 'ng' can be a whole rime-less syllable? In Yngping 'ng' alone
            # appears as syllabic nasal; treat onset match only if a rime remains.
            if rest:
                return o, rest
    return "", seg


def coda_class(seg: str, tone: str) -> str:
    """Classify how a surface syllable ends: stop (-h/-k), nasal (-ng/-n/-m),
    or open (vowel)."""
    if seg.endswith(("h", "k")):
        return "stop"
    if seg.endswith(("ng", "n", "m")):
        return "nasal"
    return "open"


def score_pair(pred: str, gold: str) -> dict:
    p = parse_yngping(pred)
    g = parse_yngping(gold)
    n = max(len(p), len(g))
    syl_correct = 0
    tone_correct = 0
    for i in range(n):
        ps = p[i] if i < len(p) else ("", "")
        gs = g[i] if i < len(g) else ("", "")
        if ps == gs:
            syl_correct += 1
        if ps[1] == gs[1] and gs[1] != "":
            tone_correct += 1
    return {"n_syl_gold": len(g), "n_syl_correct": syl_correct, "n_tone_correct": tone_correct}


def build_test_set(resources_dir: Path, n: int, seed: int):
    feng = pd.read_csv(
        resources_dir / "FengResource.tsv", sep="\t", dtype=str,
        keep_default_na=False, na_values=[""], encoding="utf-8", engine="python",
    )
    df = feng[feng["literal_pron"].notna() & feng["sandhi_pron"].notna()].copy()
    df["lit_syl"] = df["literal_pron"].apply(lambda s: len(parse_yngping(s)))
    df["san_syl"] = df["sandhi_pron"].apply(lambda s: len(parse_yngping(s)))
    df = df[df["lit_syl"] == df["san_syl"]]
    df = df[df["lit_syl"].between(2, 5)].reset_index(drop=True)

    rng = random.Random(seed)
    idx = list(range(len(df)))
    rng.shuffle(idx)
    test_idx = sorted(idx[:n])
    train_idx = sorted(idx[n:])

    def row_to_dict(i: int) -> dict:
        r = df.iloc[i]
        return {
            "id": int(r["id"]),
            "literal_pron": r["literal_pron"],
            "sandhi_pron": r["sandhi_pron"],
            "syl_count": int(r["lit_syl"]),
        }

    return [row_to_dict(i) for i in test_idx], [row_to_dict(i) for i in train_idx]


# --- learners -----------------------------------------------------------------

def learn_rules(train_pool: list[dict]):
    tone_ctx: dict[tuple[str, str], Counter] = {}
    tone_mode: dict[str, Counter] = {}
    rime_ctx: dict[tuple[str, str, str], Counter] = {}
    rime_mode: dict[tuple[str, str], Counter] = {}
    init_ctx: dict[tuple[str, str], Counter] = {}

    for ex in train_pool:
        lit = parse_yngping(ex["literal_pron"])
        san = parse_yngping(ex["sandhi_pron"])
        if len(lit) != len(san) or len(lit) < 2:
            continue
        for i in range(len(lit)):
            lseg, ltone = lit[i]
            sseg, stone = san[i]
            lon, lri = split_onset(lseg)
            son, sri = split_onset(sseg)

            if i < len(lit) - 1:
                nxt_tone = lit[i + 1][1]
                tone_ctx.setdefault((ltone, nxt_tone), Counter())[stone] += 1
                tone_mode.setdefault(ltone, Counter())[stone] += 1
                rime_ctx.setdefault((lri, ltone, nxt_tone), Counter())[sri] += 1
                rime_mode.setdefault((lri, ltone), Counter())[sri] += 1

            if i > 0:
                # Condition on the SURFACE (sandhi) form of the preceding
                # syllable, which is what phonologically drives assimilation.
                prev_sseg, prev_stone = san[i - 1]
                pclass = coda_class(prev_sseg, prev_stone)
                init_ctx.setdefault((pclass, lon), Counter())[son] += 1

    top = lambda d: {k: c.most_common(1)[0][0] for k, c in d.items()}
    return {
        "tone_ctx": top(tone_ctx), "tone_mode": top(tone_mode),
        "rime_ctx": top(rime_ctx), "rime_mode": top(rime_mode),
        "init_ctx": top(init_ctx),
        "sizes": {k: len(v) for k, v in [
            ("tone_ctx", tone_ctx), ("tone_mode", tone_mode),
            ("rime_ctx", rime_ctx), ("rime_mode", rime_mode),
            ("init_ctx", init_ctx)]},
    }


# --- predictors -----------------------------------------------------------------

def predict_tone_only(literal: str, rules) -> str:
    """Paper's context-rule: tone rewrite only."""
    syls = parse_yngping(literal)
    out = []
    for i, (seg, tone) in enumerate(syls):
        if i < len(syls) - 1:
            nxt = syls[i + 1][1]
            new_tone = rules["tone_ctx"].get((tone, nxt)) or rules["tone_mode"].get(tone, tone)
            out.append((seg, new_tone))
        else:
            out.append((seg, tone))
    return join_yngping(out)


def predict_full(literal: str, rules) -> str:
    """Tone + rime + initial assimilation."""
    syls = parse_yngping(literal)
    out: list[tuple[str, str]] = []
    for i, (seg, tone) in enumerate(syls):
        onset, rime = split_onset(seg)
        new_tone, new_rime, new_onset = tone, rime, onset

        if i < len(syls) - 1:
            nxt = syls[i + 1][1]
            new_tone = rules["tone_ctx"].get((tone, nxt)) or rules["tone_mode"].get(tone, tone)
            new_rime = (rules["rime_ctx"].get((rime, tone, nxt))
                        or rules["rime_mode"].get((rime, tone))
                        or rime)

        if i > 0:
            # Preceding syllable's PREDICTED surface form drives assimilation.
            prev_seg, prev_tone = out[i - 1]
            pclass = coda_class(prev_seg, prev_tone)
            new_onset = rules["init_ctx"].get((pclass, onset), onset)

        out.append((new_onset + new_rime, new_tone))
    return join_yngping(out)


# --- main -----------------------------------------------------------------------

def evaluate(test_set, predict_fn) -> dict:
    agg = {"n_syl_gold": 0, "n_syl_correct": 0, "n_tone_correct": 0, "n_word_exact": 0}
    per_len: dict[int, dict] = {}
    results = []
    for ex in test_set:
        pred = predict_fn(ex["literal_pron"])
        sc = score_pair(pred, ex["sandhi_pron"])
        we = 1 if pred.strip() == ex["sandhi_pron"].strip() else 0
        for k in ("n_syl_gold", "n_syl_correct", "n_tone_correct"):
            agg[k] += sc[k]
        agg["n_word_exact"] += we
        d = per_len.setdefault(ex["syl_count"], {"n": 0, "exact": 0})
        d["n"] += 1
        d["exact"] += we
        results.append({**ex, "pred": pred, "word_exact": we})
    n = len(test_set)
    return {
        "word_exact_accuracy": agg["n_word_exact"] / n,
        "syllable_accuracy": agg["n_syl_correct"] / agg["n_syl_gold"],
        "tone_accuracy": agg["n_tone_correct"] / agg["n_syl_gold"],
        "by_syl_count": {k: {"n": v["n"], "word_exact_acc": v["exact"] / v["n"]}
                         for k, v in sorted(per_len.items())},
        "results": results,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--resources-dir", type=Path, default=Path("foochow-server/resources"))
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    test_set, train_pool = build_test_set(args.resources_dir, args.n, args.seed)
    print(f"[data] test={len(test_set)}  train={len(train_pool)}", file=sys.stderr)

    rules = learn_rules(train_pool)
    print(f"[rules] table sizes: {rules['sizes']}", file=sys.stderr)

    tone_only = evaluate(test_set, lambda lit: predict_tone_only(lit, rules))
    full = evaluate(test_set, lambda lit: predict_full(lit, rules))

    summary = {
        "n_test": len(test_set), "seed": args.seed,
        "rule_table_sizes": rules["sizes"],
        "tone_only_context_rule": {k: v for k, v in tone_only.items() if k != "results"},
        "full_rule_tone_rime_assimilation": {k: v for k, v in full.items() if k != "results"},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"summary": summary,
         "full_rule_results": full["results"]},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.out}", file=sys.stderr)

    print(f"\n{'System':44s}  Word    Syl     Tone")
    for name, m in [("Context-rule (tone only, paper Table 6)", tone_only),
                    ("Full rule (tone + rime + assimilation)", full)]:
        print(f"{name:44s}  {m['word_exact_accuracy']*100:5.2f}   "
              f"{m['syllable_accuracy']*100:5.2f}   {m['tone_accuracy']*100:5.2f}")
    print("\nBy syllable count (full rule):")
    for k, v in full["by_syl_count"].items():
        print(f"  {k}-syl: n={v['n']}, word-exact={v['word_exact_acc']*100:.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
