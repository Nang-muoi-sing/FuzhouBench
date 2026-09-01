---
license: cc-by-nc-sa-4.0
language:
  - cdo
language_details: cdo (Eastern Min Chinese, Fuzhou variety)
pretty_name: FuzhouBench
tags:
  - fuzhou
  - eastern-min
  - min-dong
  - sinitic
  - low-resource
  - tone-sandhi
  - grapheme-to-phoneme
  - dictionary
  - speech
task_categories:
  - text2text-generation
  - text-retrieval
  - text-to-speech
size_categories:
  - 10K<n<100K
annotations_creators:
  - expert-generated
  - crowdsourced
source_datasets:
  - original
---

# FuzhouBench

**The first public NLP benchmark for Eastern Min Chinese (Fuzhou dialect, ISO 639-3 `cdo`),**
built on [SeeDict](https://www.seedict.com), an open dictionary curated by native-speaker
volunteers of the Nang-muoi-sing (蓝尾星) team.

> Xiaocan Li and Haoyu Lin. 2026. *FuzhouBench: A Benchmark for Eastern Min Chinese with
> Tone Sandhi Annotations.* In Proceedings of EMNLP 2026.

Repository: https://github.com/Nang-muoi-sing/FuzhouBench · Releases: https://github.com/Nang-muoi-sing/FuzhouBench/releases

| Resource | Count |
|---|---|
| Word entries | 22,890 |
| Yngping romanized readings | 51,505 |
| Entries with explicit tone-sandhi annotation (literal + surface form) | 11,276 |
| Explanations | 5,160 |
| Audio clips (two native speakers, dictionary-style) | 10,652 clips, 3.06 h |

## Why Fuzhou is hard

Four properties make Fuzhou structurally distinct from Mandarin and challenging for
current NLP: (1) a tone-sandhi system conditioned on the following syllable's citation
tone; (2) initial-consonant assimilation in non-final positions (citation *s, d, g*
surface as *l, l, ∅*); (3) systematic literary–colloquial diglossia (45% of frequent
characters carry ≥2 distinct segmental readings); (4) substantial lexical divergence
(everyday concepts use entirely different morphemes from Mandarin). FuzhouBench is a
text- and tone-sandhi-centered benchmark supplemented with dictionary-style audio; the
three benchmark tasks are text-only, and the audio serves as paired pronunciation
reference and seed data for future speech work.

## Tasks and headline results

| Task | Input → Output | Best few-shot LLM | Best symbolic / in-domain |
|---|---|---|---|
| **G2P** | characters → Yngping (primary reading) | gpt-5.4-mini 2.40% word-exact | LoRA SFT (Qwen3-1.7B, 340 s) 17.20% |
| **Tone-sandhi prediction** | literal Yngping → surface Yngping | gpt-5.4-mini 27.20% word-exact | full learned rule 44.80% |
| **Definition retrieval** | colloquial Mandarin query → entry | — | BM25 MRR 0.372 → two-stage hybrid + rerank 0.716 |

Retrieval findings replicate on 200 human-written queries (BM25 0.387, two-stage 0.689).

## Download

| What | Where |
|---|---|
| Tables, evaluation splits, results, code, docs | this repository (clone or the `fuzhoubench-vX.Y.Z-data.zip` / `-code.zip` release assets) |
| Audio (10,652 MP3, ~128 MB zipped) | `fuzhoubench-vX.Y.Z-audio.zip` under [Releases](https://github.com/Nang-muoi-sing/FuzhouBench/releases); unzip into `audio/` |

Verify any download against `CHECKSUMS.sha256` (`sha256sum -c CHECKSUMS.sha256`).

## Layout

```
data/          10 TSV tables (UTF-8, tab-separated) — the dictionary snapshot
eval/          official evaluation splits and query pools
results/       every baseline result JSON reported in the paper
audio/         Speaker_1/ (7,405 MP3) and Speaker_2/ (3,247 MP3), ID3 stripped
code/          scripts by task: stats/ g2p/ sandhi/ retrieval/  (MIT)
CHECKSUMS.sha256, LICENSE-DATA, LICENSE-CODE, CITATION.cff
```

### Data tables (star schema on `WordResource`)

| Table | Rows | Content |
|---|---|---|
| `WordResource.tsv` | 22,890 | canonical word entries (headword, tags, phonology, publish flag) |
| `PronunciationResource.tsv` | 51,505 | Yngping readings; `is_primary`, `is_sandhi`, variant, source |
| `FengResource.tsv` | 11,276 | Feng (1998) entries with `literal_pron` and `sandhi_pron` |
| `ExplanationResource.tsv` | 5,160 | Mandarin definitions with example sentences |
| `CikLingResource.tsv` | 11,512 | Qi-Lin Ba-Yin rhyme-book categories |
| `DFDCharacterResource.tsv` | 9,811 | character references (Banguace) |
| `SupplementWordResource.tsv` | 27,871 | alternative forms and Mandarin equivalents |
| `RecordingResource.tsv` | 10,659 | audio index: md5, Yngping, speaker |
| `UserContribResource.tsv` | 1,059 | reviewed community submissions (identity columns removed) |
| `UserSubmitResource.tsv` | 121 | pending community submissions (identity columns removed) |

All tables join on `word__id`; audio joins `PronunciationResource.yngping` to
`RecordingResource.yngping`. **Read every TSV as UTF-8** — many rows contain rare Min
characters that mojibake under GBK or cp1252 defaults.

```python
import pandas as pd
words = pd.read_csv("data/WordResource.tsv", sep="\t", encoding="utf-8", dtype=str)
```

### Evaluation splits (`eval/`)

| File | Use |
|---|---|
| `g2p_test_500.json` | G2P test set: 500 words, 125 per character-length bucket, seed 42 |
| `sandhi_test_500_feng_ids.json` | Feng row ids of the 500 multi-syllable sandhi test items (seed 42); all other Feng rows form the train pool |
| `retrieval_queries_original_8880.json` | original-source query pool (first sentence of each explanation) |
| `retrieval_queries_llm_qwen3_30b_a3b.json` | 11,230 LLM-generated colloquial Mandarin queries |
| `retrieval_queries_human_400.json` | 400 human queries; `group == "B"` is the 200 native-speaker-written colloquial set |
| `blind_naturalness_*.{md,json}` | blind naturalness evaluation sheet (rated) and key |

Scoring for G2P and sandhi: word-exact, per-syllable, and tone-only accuracy over
space-separated Yngping syllables (`code/g2p/*`, `code/sandhi/*`). Retrieval: Hit@k and
MRR within the top 10 (`code/retrieval/*`).

## Reproducing the paper's numbers

`code/stats/recount.py` and `code/stats/make_figure_stats.py` reproduce every count in
the paper's Section 5 and Figure 2 from `results/v1.0_stats.json` and the TSVs. They were
written against the original project layout, so recreate it from this release first:

```bash
mkdir -p repo/foochow-server repo/seedict-data
ln -s "$PWD/data"  repo/foochow-server/resources
ln -s "$PWD/audio" repo/seedict-data/audio
cp results/audio_eda_summary.json repo/seedict-data/
python code/stats/recount.py --project-root repo
```

Baselines: `code/g2p/`, `code/sandhi/`, `code/retrieval/` each carry a usage docstring;
closed-model runs read API keys from `--api-key-file` and never hardcode them.

## Yngping in brief

Yngping (榕拼) writes each syllable as letters followed by a Chao-notation tone value
from {55, 53, 33, 21, 24, 242, 213, 5, 0}; syllables are space-separated
(`huk21 ziu55` = Fuzhou). Multi-syllable words are stored in their **surface (sandhi)**
form as the primary reading; `FengResource` additionally gives the literal (citation)
form for 11,276 entries.

## Intended uses

- Grapheme-to-phoneme, tone-sandhi modeling, and phonological rule induction
- Dictionary / definition retrieval for low-resource Sinitic varieties
- Dialectology and descriptive-phonology research
- Seed data for Fuzhou TTS/ASR research

**Not intended:** training voice-cloning models capable of impersonating the two
speakers; commercial use without permission from the SeeDict community (see license).

## Provenance, consent, and privacy

- The dictionary content was curated by SeeDict volunteers over several years; the two
  audio contributors recorded dictionary-style readings and consented to non-commercial
  research and educational use.
- Contributor-identity columns are removed from the community-submission tables,
  contributor handles are scrubbed from free-text fields, speakers appear as
  `Speaker_1` / `Speaker_2`, and ID3 metadata is stripped from all audio.
- All database identifiers (`id`, `word__id`, and related key columns) are keyed
  pseudonyms, replaced consistently across every table and evaluation file at the
  SeeDict maintainers' request: joins within this release are fully preserved, but
  the identifiers do not correspond to SeeDict production keys.
- The dataset belongs to the SeeDict community, which retains the right to maintain,
  update, and re-license the underlying database. This release is a fixed snapshot
  (v1.0.1) matching the paper; later versions follow semantic versioning.
- Release history: v1.0.1 pseudonymizes all database identifiers (row counts and
  all benchmark content are unchanged from v1.0.0, which was never published as a
  release).

## Known limitations

Two speakers only (one contributes almost exclusively monosyllabic citation forms);
dictionary-style audio (median clip 1.02 s, not connected speech); mostly the urban
Fuzhou variety; LLM-generated queries are measurably less natural than human-written
ones (mean 4.08 vs 4.70 on a 1–5 blind rating); symbolic sandhi coverage collapses
beyond disyllables.

## License

Data: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) (`LICENSE-DATA`).
Code: MIT (`LICENSE-CODE`).

## Citation

```bibtex
@inproceedings{li2026fuzhoubench,
  title     = {FuzhouBench: A Benchmark for Eastern Min Chinese with Tone Sandhi Annotations},
  author    = {Li, Xiaocan and Lin, Haoyu},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year      = {2026},
  publisher = {Association for Computational Linguistics}
}
```

Please also credit the SeeDict community (https://www.seedict.com) when using the data.
