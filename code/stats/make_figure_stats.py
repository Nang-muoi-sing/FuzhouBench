"""
Generate Figure 1 of the FuzhouBench paper: a 4-panel dataset composition snapshot.

Panels:
  A. Word entries by character length (1, 2, 3, 4+).
  B. Audio clip duration histogram (with median + p99 markers).
  C. Audio clip count by Fuzhou citation tone.
  D. Cross-resource coverage of the 22,890 word entries.

Inputs:
  papers/data/v1.0_stats.json
  seedict-data/audio_eda_summary.json
  seedict-data/audio_eda.parquet                (optional, for the duration histogram;
                                                 falls back to a quantile-based KDE if
                                                 the parquet is unavailable)
  foochow-server/resources/WordResource.tsv

Output:
  papers/fuzhoubench-draft/figures/fig_stats.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Color palette: muted, BW-printable, color-blind-safe.
PALETTE = {
    "primary":  "#3b6aa0",
    "accent":   "#d97b5b",
    "tone":     "#6a9966",
    "coverage": "#7c5e9a",
    "annot":    "#444444",
}


def panel_word_length(ax, word_tsv: Path) -> None:
    df = pd.read_csv(
        word_tsv, sep="\t", dtype=str, keep_default_na=False, na_values=[""],
        encoding="utf-8", engine="python",
    )
    df = df[df["text"].notna()]
    df = df[df["is_published"] == "1"]
    lens = df["text"].str.len()
    buckets = {"1": (lens == 1).sum(),
               "2": (lens == 2).sum(),
               "3": (lens == 3).sum(),
               "4+": (lens >= 4).sum()}
    xs = list(buckets.keys())
    ys = list(buckets.values())
    bars = ax.bar(xs, ys, color=PALETTE["primary"], edgecolor="white", linewidth=0.6)
    for b, y in zip(bars, ys):
        ax.text(b.get_x() + b.get_width() / 2, y, f"{y:,}",
                ha="center", va="bottom", fontsize=8, color=PALETTE["annot"])
    ax.set_title("(a) Word entries by character length", fontsize=10)
    ax.set_xlabel("character length", fontsize=9)
    ax.set_ylabel("count", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.set_ylim(0, max(ys) * 1.18)


def panel_duration(ax, eda_summary: dict, parquet: Path | None) -> None:
    quant = eda_summary.get("duration_quantiles", {})
    median = float(quant.get("0.5", 1.02))
    p99 = float(quant.get("0.99", 1.42))
    p5 = float(quant.get("0.05", 0.87))

    if parquet is not None and parquet.exists():
        try:
            df = pd.read_parquet(parquet)
            if "duration_sec" in df.columns:
                durs = df["duration_sec"].astype(float).values
            elif "duration" in df.columns:
                durs = df["duration"].astype(float).values
            else:
                durs = None
        except Exception:
            durs = None
    else:
        durs = None

    if durs is not None and len(durs) > 0:
        ax.hist(durs, bins=40, range=(0, 2.0), color=PALETTE["accent"],
                edgecolor="white", linewidth=0.4)
        n = len(durs)
        max_label = f"n = {n:,} clips"
    else:
        # Fallback: bar from quantiles
        xs = ["p5", "p25", "p50", "p75", "p90", "p95", "p99"]
        ys = [float(quant.get(k, 0)) for k in ("0.05", "0.25", "0.5", "0.75", "0.9", "0.95", "0.99")]
        ax.bar(xs, ys, color=PALETTE["accent"], edgecolor="white", linewidth=0.4)
        max_label = "(quantile fallback)"

    ax.set_title("(b) Audio clip duration", fontsize=10)
    ax.set_xlabel("seconds", fontsize=9)
    ax.set_ylabel("count", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.set_xlim(0, 2.0)
    # Draw markers after axis setup so coords are stable.
    ymax = ax.get_ylim()[1]
    ax.axvline(median, color=PALETTE["annot"], linestyle="--", linewidth=0.8)
    ax.text(median + 0.04, ymax * 0.70, f"median {median:.2f}s",
            fontsize=8, color=PALETTE["annot"], va="center")
    ax.text(0.97, 0.40, max_label, transform=ax.transAxes,
            ha="right", va="center", fontsize=8, color=PALETTE["annot"])


def panel_tone(ax, eda_summary: dict) -> None:
    tones = eda_summary.get("tone_top", {})
    # Order tones in the canonical Fuzhou order: open tones then checked + neutral
    canonical = ["55", "53", "33", "21", "24", "242", "213", "5", "0"]
    items = [(t, tones.get(t, 0)) for t in canonical if t in tones]
    xs = [t for t, _ in items]
    ys = [n for _, n in items]
    bars = ax.bar(xs, ys, color=PALETTE["tone"], edgecolor="white", linewidth=0.5)
    for b, y in zip(bars, ys):
        ax.text(b.get_x() + b.get_width() / 2, y, f"{y:,}",
                ha="center", va="bottom", fontsize=7.5, color=PALETTE["annot"])
    ax.set_title("(c) Audio clips by citation tone", fontsize=10)
    ax.set_xlabel("tone", fontsize=9)
    ax.set_ylabel("count", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.set_ylim(0, max(ys) * 1.18)


def panel_coverage(ax, stats: dict) -> None:
    cov = stats.get("coverage", {})
    total = cov.get("n_word_total", 22890)
    rows = [
        ("Pronunciation", cov.get("n_words_with_pronunciation", 0)),
        ("Audio",         cov.get("n_words_with_audio_via_yngping_match", 0)),
        ("Sandhi",        cov.get("n_words_with_feng_sandhi", 0)),
        ("Explanation",   cov.get("n_words_with_explanation", 0)),
    ]
    labels = [r[0] for r in rows]
    counts = [r[1] for r in rows]
    pcts = [100 * c / max(1, total) for c in counts]
    y_pos = np.arange(len(labels))[::-1]
    bars = ax.barh(y_pos, pcts, color=PALETTE["coverage"], edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    for b, p, c in zip(bars, pcts, counts):
        ax.text(p + 1.5, b.get_y() + b.get_height() / 2,
                f"{p:.1f}%  ({c:,})", va="center", fontsize=8, color=PALETTE["annot"])
    ax.set_xlim(0, 115)
    ax.set_xlabel("% of word entries (n = {:,})".format(total), fontsize=9)
    ax.set_title("(d) Cross-resource coverage", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    pr: Path = args.project_root
    stats = json.loads((pr / "papers" / "data" / "v1.0_stats.json").read_text(encoding="utf-8"))
    eda = json.loads((pr / "seedict-data" / "audio_eda_summary.json").read_text(encoding="utf-8"))
    parquet = pr / "seedict-data" / "audio_eda.parquet"
    word_tsv = pr / "foochow-server" / "resources" / "WordResource.tsv"

    plt.rcParams["pdf.fonttype"] = 42         # embed fonts as TrueType
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif", "DejaVu Serif"]

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.3))
    panel_word_length(axes[0, 0], word_tsv)
    panel_duration(axes[0, 1], eda, parquet)
    panel_tone(axes[1, 0], eda)
    panel_coverage(axes[1, 1], stats)
    fig.tight_layout(pad=1.2, h_pad=1.6, w_pad=2.0)

    out = args.out or (pr / "papers" / "fuzhoubench-draft" / "figures" / "fig_stats.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
