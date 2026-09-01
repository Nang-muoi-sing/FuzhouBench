"""
FuzhouBench v1.0 statistics recount.

Reads:
  foochow-server/resources/*.tsv
  seedict-data/audio/                  (file listing for sanity check)
  seedict-data/audio_eda_summary.json  (canonical audio stats from remote H100 EDA)
  seedict-data/audio_eda.parquet       (optional, for extra audio distributions)

Writes:
  papers/data/v1.0_stats.json   machine-readable, source of truth for paper tables
  papers/data/v1.0_stats.md     human-readable summary

Run:
  python papers/scripts/recount.py
  python papers/scripts/recount.py --project-root /custom/path
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def read_tsv(path: Path) -> pd.DataFrame:
    """Read a TSV with all columns as string to avoid silent dtype coercion."""
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_values=[""],
        quoting=0,  # csv.QUOTE_MINIMAL
        engine="python",
        encoding="utf-8",
    )


def to_int_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)


def value_counts_dict(s: pd.Series, top: int | None = None) -> dict[str, int]:
    vc = s.fillna("").value_counts()
    if top is not None:
        vc = vc.head(top)
    return {str(k): int(v) for k, v in vc.items()}


# ---------------------------------------------------------------------------
# Per-resource stats
# ---------------------------------------------------------------------------

def stats_word(df: pd.DataFrame) -> dict[str, Any]:
    has_initial = df["phonology_initial"].notna().sum()
    has_final = df["phonology_final"].notna().sum()
    has_tone = df["phonology_tone"].notna().sum()
    published = (df["is_published"] == "1").sum()
    return {
        "rows": len(df),
        "published": int(published),
        "with_phonology_initial": int(has_initial),
        "with_phonology_final": int(has_final),
        "with_phonology_tone": int(has_tone),
        "with_gloss": int(df["gloss"].notna().sum()) if "gloss" in df.columns else None,
        "unique_text": int(df["text"].nunique()),
    }


def stats_pronunciation(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": len(df),
        "is_sandhi_true": int((df["is_sandhi"] == "1").sum()),
        "is_primary_true": int((df["is_primary"] == "1").sum()),
        "unique_words_covered": int(df["word__id"].nunique()),
        "source_type_distribution": value_counts_dict(df["source_type"]),
        "variant_top10": value_counts_dict(df["variant"], top=10),
    }


def stats_explanation(df: pd.DataFrame) -> dict[str, Any]:
    nonempty_sentences = df["sentences"].apply(
        lambda x: isinstance(x, str) and len(x) > 2 and x != "[]"
    ).sum()
    return {
        "rows": len(df),
        "unique_words_covered": int(df["word__id"].nunique()),
        "with_example_sentences": int(nonempty_sentences),
        "lexical_category_distribution": value_counts_dict(df["lexical_category"], top=15),
    }


def stats_feng(df: pd.DataFrame) -> dict[str, Any]:
    lit = df["literal_pron"].fillna("")
    san = df["sandhi_pron"].fillna("")
    differs = (lit != san) & (lit != "") & (san != "")
    same = (lit == san) & (lit != "")
    return {
        "rows": len(df),
        "unique_words_covered": int(df["word__id"].nunique()),
        "literal_eq_sandhi": int(same.sum()),
        "literal_differs_from_sandhi": int(differs.sum()),
        "missing_either_side": int(((lit == "") | (san == "")).sum()),
    }


def stats_cikling(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": len(df),
        "unique_words_covered": int(df["word__id"].nunique()),
        "tone_distribution_top10": value_counts_dict(df["tone"], top=10),
        "cik_initial_top10": value_counts_dict(df["cik_initial"], top=10),
        "ling_initial_top10": value_counts_dict(df["ling_initial"], top=10),
    }


def stats_dfd(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": len(df),
        "unique_words_covered": int(df["word__id"].nunique()),
        "with_banguace": int(df["banguace"].notna().sum()),
    }


def stats_supplement(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": len(df),
        "unique_words_covered": int(df["word__id"].nunique()),
        "type_distribution": value_counts_dict(df["type"], top=15),
        "glyph_category_top10": value_counts_dict(df["glyph_category"], top=10),
    }


def stats_recording(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": len(df),
        "speaker_distribution": value_counts_dict(df["speaker"]),
        "unique_md5": int(df["md5"].nunique()),
        "unique_yngping": int(df["yngping"].nunique()),
    }


def stats_user_contrib(df: pd.DataFrame, reviewed: bool) -> dict[str, Any]:
    out = {
        "rows": len(df),
        "status_distribution": value_counts_dict(df["status"], top=10) if "status" in df.columns else {},
        "locale_top10": value_counts_dict(df["locale"], top=10) if "locale" in df.columns else {},
        "reviewed_track": reviewed,
    }
    return out


RESOURCE_HANDLERS = {
    "WordResource.tsv": stats_word,
    "PronunciationResource.tsv": stats_pronunciation,
    "ExplanationResource.tsv": stats_explanation,
    "FengResource.tsv": stats_feng,
    "CikLingResource.tsv": stats_cikling,
    "DFDCharacterResource.tsv": stats_dfd,
    "SupplementWordResource.tsv": stats_supplement,
    "RecordingResource.tsv": stats_recording,
    "UserContribResource.tsv": lambda df: stats_user_contrib(df, reviewed=True),
    "UserSubmitResource.tsv": lambda df: stats_user_contrib(df, reviewed=False),
}


# ---------------------------------------------------------------------------
# Cross-resource coverage (the numbers that go into the paper)
# ---------------------------------------------------------------------------

def coverage_stats(tsv_dfs: dict[str, pd.DataFrame]) -> dict[str, Any]:
    word = tsv_dfs["WordResource.tsv"]
    pron = tsv_dfs["PronunciationResource.tsv"]
    feng = tsv_dfs["FengResource.tsv"]
    rec = tsv_dfs["RecordingResource.tsv"]
    expl = tsv_dfs["ExplanationResource.tsv"]

    word_ids = set(word["id"])
    n_total = len(word_ids)

    words_with_pron = word_ids & set(pron["word__id"])
    words_with_feng = word_ids & set(feng["word__id"])
    words_with_expl = word_ids & set(expl["word__id"])

    # RecordingResource has no word__id but has yngping; pair audio to words via PronunciationResource yngping match
    rec_yngping = set(rec["yngping"].dropna())
    pron_with_audio = pron[pron["yngping"].isin(rec_yngping)]
    words_with_audio = word_ids & set(pron_with_audio["word__id"])

    return {
        "n_word_total": n_total,
        "n_words_with_pronunciation": len(words_with_pron),
        "n_words_with_feng_sandhi": len(words_with_feng),
        "n_words_with_explanation": len(words_with_expl),
        "n_words_with_audio_via_yngping_match": len(words_with_audio),
        "pct_words_with_pronunciation": round(100 * len(words_with_pron) / n_total, 2),
        "pct_words_with_feng_sandhi": round(100 * len(words_with_feng) / n_total, 2),
        "pct_words_with_explanation": round(100 * len(words_with_expl) / n_total, 2),
        "pct_words_with_audio": round(100 * len(words_with_audio) / n_total, 2),
    }


# ---------------------------------------------------------------------------
# Audio (prefer existing remote EDA; cross-check with local file listing)
# ---------------------------------------------------------------------------

def audio_stats(seedict_dir: Path) -> dict[str, Any]:
    summary_json = seedict_dir / "audio_eda_summary.json"
    audio_root = seedict_dir / "audio"

    eda: dict[str, Any] = {}
    if summary_json.exists():
        with summary_json.open(encoding="utf-8") as f:
            eda = json.load(f)

    local_count = None
    per_speaker_local: dict[str, int] = {}
    if audio_root.exists():
        per_speaker_local = {
            sub.name: sum(1 for _ in sub.glob("*.mp3"))
            for sub in audio_root.iterdir()
            if sub.is_dir()
        }
        local_count = sum(per_speaker_local.values())

    return {
        "source_of_truth": "seedict-data/audio_eda_summary.json (remote H100 EDA)",
        "remote_eda": eda,
        "local_file_count": local_count,
        "local_per_speaker": per_speaker_local,
        "local_vs_remote_count_match": (local_count == eda.get("n_files")) if local_count is not None else None,
    }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def fmt_md(stats: dict[str, Any]) -> str:
    """Render stats to a human-readable Markdown block."""
    lines: list[str] = []
    lines.append(f"# FuzhouBench v1.0 — data snapshot")
    lines.append("")
    lines.append(f"Generated: {stats['generated_at']}")
    lines.append(f"Project root: `{stats['project_root']}`")
    lines.append("")

    lines.append("## Text resources (foochow-server/resources/*.tsv)")
    lines.append("")
    lines.append("| Resource | Rows | Notes |")
    lines.append("|---|---|---|")
    for name, body in stats["text"].items():
        if "error" in body and "rows" not in body:
            lines.append(f"| {name} | ERROR | {body['error']} |")
            continue
        rows_val = body.get("rows", "?")
        rows = f"{rows_val:,}" if isinstance(rows_val, int) else str(rows_val)
        notes_bits = []
        for k, v in body.items():
            if k in ("rows",):
                continue
            if isinstance(v, dict):
                top = list(v.items())[:3]
                notes_bits.append(f"{k}: " + ", ".join(f"{a}={b}" for a, b in top))
            else:
                notes_bits.append(f"{k}={v}")
        lines.append(f"| {name} | {rows} | {'; '.join(notes_bits)} |")
    lines.append("")

    lines.append("## Coverage cross-stats")
    lines.append("")
    cov = stats["coverage"]
    if "error" in cov:
        lines.append(f"⚠️ coverage unavailable: {cov['error']}")
    else:
        lines.append(f"- Total unique words: **{cov['n_word_total']:,}**")
        lines.append(f"- With at least one pronunciation: **{cov['n_words_with_pronunciation']:,}** ({cov['pct_words_with_pronunciation']}%)")
        lines.append(f"- With FengResource sandhi annotation: **{cov['n_words_with_feng_sandhi']:,}** ({cov['pct_words_with_feng_sandhi']}%)")
        lines.append(f"- With explanation: **{cov['n_words_with_explanation']:,}** ({cov['pct_words_with_explanation']}%)")
        lines.append(f"- With audio (via yngping match): **{cov['n_words_with_audio_via_yngping_match']:,}** ({cov['pct_words_with_audio']}%)")
    lines.append("")

    lines.append("## Audio")
    lines.append("")
    audio = stats["audio"]
    eda = audio.get("remote_eda", {})
    if eda:
        lines.append(f"- Files (remote EDA): **{eda.get('n_files'):,}**, total **{eda.get('total_hours'):.2f} h**")
        per_h = eda.get("per_speaker_hours", {})
        per_c = eda.get("per_speaker_count", {})
        for spk in sorted(per_h):
            lines.append(f"  - `{spk}`: {per_c.get(spk, '?'):,} files / {per_h[spk]:.2f} h")
        sr = eda.get("samplerate_counts", {})
        if sr:
            lines.append(f"- Sample rates: " + ", ".join(f"{k} Hz × {v:,}" for k, v in sr.items()))
        ch = eda.get("channels_counts", {})
        if ch:
            lines.append(f"- Channels: " + ", ".join(f"{k}ch × {v:,}" for k, v in ch.items()))
        susp = eda.get("suspicious", {})
        if susp:
            lines.append(f"- Quality flags: " + ", ".join(f"{k}={v}" for k, v in susp.items()))
    if audio.get("local_file_count") is not None:
        match = audio.get("local_vs_remote_count_match")
        match_str = "✅" if match else "⚠️ MISMATCH"
        lines.append(f"- Local file count: {audio['local_file_count']:,} ({match_str} vs remote)")
        for spk, n in audio.get("local_per_speaker", {}).items():
            lines.append(f"  - local `{spk}`: {n:,}")
    lines.append("")

    lines.append("## Paper Table 1 candidate (FuzhouBench row)")
    lines.append("")
    words_n = stats["text"].get("WordResource.tsv", {}).get("rows", "?")
    pron_n = stats["text"].get("PronunciationResource.tsv", {}).get("rows", "?")
    feng_n = stats["text"].get("FengResource.tsv", {}).get("rows", "?")
    audio_h = eda.get("total_hours", 0)
    words_s = f"{words_n:,}" if isinstance(words_n, int) else str(words_n)
    pron_s = f"{pron_n:,}" if isinstance(pron_n, int) else str(pron_n)
    feng_s = f"{feng_n:,}" if isinstance(feng_n, int) else str(feng_n)
    lines.append("```")
    lines.append(f"FuzhouBench | Eastern Min | {words_s} words + {pron_s} pronunciations + {audio_h:.2f} h audio | text+audio+IPA | G2P + Sandhi + QA | sandhi-annotated ({feng_s}) | CC-BY-NC-SA 4.0")
    lines.append("```")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Project root (default: two levels up from this script).",
    )
    args = parser.parse_args()

    project_root: Path = args.project_root
    resources_dir = project_root / "foochow-server" / "resources"
    seedict_dir = project_root / "seedict-data"
    out_dir = project_root / "papers" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[recount] project_root = {project_root}", file=sys.stderr)
    print(f"[recount] resources = {resources_dir}", file=sys.stderr)
    print(f"[recount] seedict = {seedict_dir}", file=sys.stderr)

    if not resources_dir.exists():
        print(f"[recount] ERROR: resources dir not found: {resources_dir}", file=sys.stderr)
        return 2

    # Read all TSVs
    tsv_dfs: dict[str, pd.DataFrame] = {}
    text_stats: dict[str, dict[str, Any]] = {}
    for fname, handler in RESOURCE_HANDLERS.items():
        path = resources_dir / fname
        if not path.exists():
            text_stats[fname] = {"error": "file not found"}
            continue
        print(f"[recount] reading {fname} ...", file=sys.stderr)
        try:
            df = read_tsv(path)
            tsv_dfs[fname] = df
        except Exception as e:
            text_stats[fname] = {"error": f"read failed: {type(e).__name__}: {e}"}
            print(f"[recount]   READ FAILED: {e}", file=sys.stderr)
            continue
        try:
            text_stats[fname] = handler(df)
        except Exception as e:
            text_stats[fname] = {"error": f"handler failed: {type(e).__name__}: {e}", "rows": len(df)}
            print(f"[recount]   HANDLER FAILED for {fname}: {e}", file=sys.stderr)

    required = {"WordResource.tsv", "PronunciationResource.tsv", "FengResource.tsv",
                "RecordingResource.tsv", "ExplanationResource.tsv"}
    if required.issubset(tsv_dfs.keys()):
        try:
            coverage = coverage_stats(tsv_dfs)
        except Exception as e:
            print(f"[recount] coverage failed: {e}", file=sys.stderr)
            coverage = {"error": f"{type(e).__name__}: {e}"}
    else:
        missing = required - set(tsv_dfs.keys())
        coverage = {"error": f"missing resources: {sorted(missing)}"}
    audio = audio_stats(seedict_dir)

    stats = {
        "version": "v1.0-draft",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "text": text_stats,
        "coverage": coverage,
        "audio": audio,
    }

    json_path = out_dir / "v1.0_stats.json"
    md_path = out_dir / "v1.0_stats.md"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[recount] wrote {json_path}", file=sys.stderr)
    try:
        with md_path.open("w", encoding="utf-8") as f:
            f.write(fmt_md(stats))
        print(f"[recount] wrote {md_path}", file=sys.stderr)
    except Exception as e:
        print(f"[recount] markdown rendering failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
