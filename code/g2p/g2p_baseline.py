"""
G2P (Grapheme-to-Phoneme) baseline evaluation for FuzhouBench.

Task definition:
  Input  : a Chinese word or compound (1 to 4+ characters), e.g. "牛奶"
  Output : its Fuzhou Yngping pronunciation, e.g. "ngu33 nei242"
  Metric : syllable accuracy (exact Yngping syllable match) and
           tone-only accuracy (tone digits match given correct segmentals).

Pipeline:
  1. Build (word, gold_yngping) pairs from PronunciationResource.tsv,
     restricting to is_primary=1 to get one canonical reading per word.
  2. Sample a fixed test set of N words (default 500), stratified by length.
  3. Call each LLM backend with a 5-shot prompt and parse the output.
  4. Score syllable accuracy and tone accuracy.
  5. Write per-model JSON results + a summary table.

Backends supported (selected via --backend):
  - openai      : OpenAI API (gpt-4o, gpt-4o-mini, etc.)
  - anthropic   : Anthropic API (claude-3-5-sonnet, etc.)
  - qwen-remote : Qwen via remote H100 (SSH wrapper, see remote skill)

Run:
  python papers/scripts/g2p_baseline.py --backend openai --model gpt-4o-2024-08-06 --n 500
  python papers/scripts/g2p_baseline.py --backend qwen-remote --model Qwen2.5-72B-Instruct --n 100

The script is intentionally minimal so the design can be reviewed before
filling in API credentials and a full run.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Callable

import pandas as pd


# ---------------------------------------------------------------------------
# Test set construction
# ---------------------------------------------------------------------------

def build_eval_set(
    pron_tsv: Path,
    word_tsv: Path,
    n: int = 500,
    seed: int = 42,
) -> list[dict]:
    """Build a stratified eval set from PronunciationResource + WordResource."""
    pron = pd.read_csv(
        pron_tsv, sep="\t", dtype=str, keep_default_na=False,
        na_values=[""], encoding="utf-8", engine="python",
    )
    word = pd.read_csv(
        word_tsv, sep="\t", dtype=str, keep_default_na=False,
        na_values=[""], encoding="utf-8", engine="python",
    )

    # Keep only primary pronunciations and join to surface text.
    primary = pron[pron["is_primary"] == "1"][["word__id", "yngping"]]
    word_id_to_text = dict(zip(word["id"], word["text"]))
    primary = primary.assign(text=primary["word__id"].map(word_id_to_text))
    primary = primary[primary["text"].notna() & primary["yngping"].notna()]
    primary = primary[primary["text"].str.len().between(1, 8)]

    # Stratify by character length.
    primary["char_len"] = primary["text"].str.len()
    rng = random.Random(seed)
    sampled: list[dict] = []
    target_per_bucket = max(1, n // 4)
    for bucket in (1, 2, 3, 4):
        sub = primary[primary["char_len"] == bucket]
        if len(sub) == 0:
            continue
        take = min(target_per_bucket, len(sub))
        idx = rng.sample(range(len(sub)), take)
        for i in idx:
            row = sub.iloc[i]
            sampled.append({
                "text": row["text"],
                "gold_yngping": row["yngping"],
                "char_len": int(row["char_len"]),
            })

    rng.shuffle(sampled)
    return sampled[:n]


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

FEWSHOT_EXAMPLES = [
    ("牛乳",   "ngu33 neing53"),      # 'cow-milk' = milk; primary reading (sandhi)
    ("講",     "goung33"),             # 'to speak'; single-syl primary
    ("白菜",   "bah21 jai213"),        # 'Chinese cabbage'; primary reading (sandhi, c->j)
    ("做什乇", "zo21 sie53 nooh24"), # 'do-what-thing' = to do what; primary (native speaker)
    ("我",     "nguai33"),             # 'I / me'; single-syl primary
]

PROMPT_TEMPLATE = """You convert Fuzhou Eastern Min words written in Chinese characters
to their Yngping romanization. Each syllable in Yngping is a string of letters
followed by tone digits (one of: 55, 53, 33, 21, 24, 242, 213, 5, 0).
Multi-syllable outputs are space-separated.

Output ONLY the Yngping string, no commentary.

Examples:
{examples}

Word: {word}
Yngping:"""


def format_prompt(word: str) -> str:
    examples_str = "\n".join(f"Word: {w}\nYngping: {y}" for w, y in FEWSHOT_EXAMPLES)
    return PROMPT_TEMPLATE.format(examples=examples_str, word=word)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

_OPENAI_KEY: str | None = None  # set once from --api-key-file at startup


def call_openai(prompt: str, model: str) -> str:
    """Call OpenAI Chat Completions API. Key supplied via --api-key-file.
    For GPT-5 reasoning models we set reasoning_effort='none' so the output
    is not consumed by hidden chain-of-thought tokens."""
    from openai import OpenAI  # type: ignore[import-not-found]
    if _OPENAI_KEY is None:
        raise RuntimeError("OpenAI API key not loaded; pass --api-key-file PATH.")
    client = OpenAI(api_key=_OPENAI_KEY)
    kwargs = dict(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=128,
    )
    if model.startswith("gpt-5") or model.startswith("o"):
        kwargs["reasoning_effort"] = "none"
    resp = client.chat.completions.create(**kwargs)
    return (resp.choices[0].message.content or "").strip()


def call_anthropic(prompt: str, model: str) -> str:
    """Call Anthropic Messages API. Requires ANTHROPIC_API_KEY."""
    import anthropic  # type: ignore[import-not-found]
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=32,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()


def call_qwen_remote(prompt: str, model: str) -> str:
    """Call Qwen on the remote H100 server. Stub; see remote skill for SSH wrapper."""
    raise NotImplementedError(
        "Qwen remote backend not wired yet. See claude/commands/remote.md for the SSH "
        "convention; implement by submitting a one-shot inference job and parsing stdout."
    )


BACKENDS: dict[str, Callable[[str, str], str]] = {
    "openai": call_openai,
    "anthropic": call_anthropic,
    "qwen-remote": call_qwen_remote,
}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

YNGPING_TOKEN_RE = re.compile(r"([a-z]+)(\d+)")


def parse_yngping(s: str) -> list[tuple[str, str]]:
    """Parse a Yngping string into a list of (segment, tone) syllables."""
    s = s.strip().lower()
    syls = []
    for tok in s.split():
        m = YNGPING_TOKEN_RE.match(tok)
        if not m:
            continue
        syls.append((m.group(1), m.group(2)))
    return syls


def score_pair(pred: str, gold: str) -> dict[str, int]:
    """Return per-syllable correctness flags for one (pred, gold) pair."""
    p = parse_yngping(pred)
    g = parse_yngping(gold)
    n = max(len(p), len(g))
    if n == 0:
        return {"n_syl": 0, "n_syl_correct": 0, "n_tone_correct": 0}
    syl_correct = 0
    tone_correct = 0
    for i in range(n):
        ps = p[i] if i < len(p) else ("", "")
        gs = g[i] if i < len(g) else ("", "")
        if ps == gs:
            syl_correct += 1
        if ps[1] == gs[1] and gs[1] != "":
            tone_correct += 1
    return {"n_syl": len(g), "n_syl_correct": syl_correct, "n_tone_correct": tone_correct}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--backend", choices=list(BACKENDS), required=True)
    parser.add_argument("--model", required=True, help="Model identifier passed to the backend.")
    parser.add_argument("--n", type=int, default=500, help="Number of test words.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=None, help="Output JSON path (default: papers/data/g2p_<backend>_<model>.json).")
    parser.add_argument("--api-key-file", type=Path, default=None, help="Path to a file containing the API key (single line, no whitespace).")
    parser.add_argument("--dry-run", action="store_true", help="Build the eval set and print prompts without calling any backend.")
    args = parser.parse_args()

    if args.api_key_file is not None:
        global _OPENAI_KEY
        _OPENAI_KEY = args.api_key_file.read_text(encoding="utf-8").strip()

    project_root: Path = args.project_root
    resources = project_root / "foochow-server" / "resources"
    eval_set = build_eval_set(
        resources / "PronunciationResource.tsv",
        resources / "WordResource.tsv",
        n=args.n,
        seed=args.seed,
    )
    print(f"[g2p] eval set: {len(eval_set)} words", file=sys.stderr)

    if args.dry_run:
        for ex in eval_set[:5]:
            print("---")
            print(format_prompt(ex["text"]))
        print(f"\n[g2p] dry-run: printed 5 of {len(eval_set)} prompts.", file=sys.stderr)
        return 0

    backend_fn = BACKENDS[args.backend]
    results = []
    agg = {"n_syl": 0, "n_syl_correct": 0, "n_tone_correct": 0, "n_words_exact": 0}
    for i, ex in enumerate(eval_set):
        prompt = format_prompt(ex["text"])
        try:
            pred = backend_fn(prompt, args.model)
        except Exception as e:
            print(f"[g2p] backend error on '{ex['text']}': {e}", file=sys.stderr)
            pred = ""
        score = score_pair(pred, ex["gold_yngping"])
        word_exact = 1 if pred.strip() == ex["gold_yngping"].strip() else 0
        results.append({**ex, "pred_yngping": pred, **score, "word_exact": word_exact})
        for k in ("n_syl", "n_syl_correct", "n_tone_correct"):
            agg[k] += score[k]
        agg["n_words_exact"] += word_exact
        if (i + 1) % 25 == 0:
            print(f"[g2p] {i+1}/{len(eval_set)} done", file=sys.stderr)

    summary = {
        "backend": args.backend,
        "model": args.model,
        "n_words": len(results),
        "word_exact_accuracy": agg["n_words_exact"] / max(1, len(results)),
        "syllable_accuracy": agg["n_syl_correct"] / max(1, agg["n_syl"]),
        "tone_accuracy": agg["n_tone_correct"] / max(1, agg["n_syl"]),
    }

    out_path = args.out or (project_root / "papers" / "data" / f"g2p_{args.backend}_{args.model.replace('/', '_')}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"[g2p] wrote {out_path}", file=sys.stderr)

    print("\nSUMMARY")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
