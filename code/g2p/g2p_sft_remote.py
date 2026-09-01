"""LoRA SFT upper bound for Yngping G2P (reviewer-requested).

Fine-tunes Qwen3-1.7B with LoRA on word -> primary-Yngping pairs from the
released dictionary (EXCLUDING the 500 test words), then evaluates on the
same 500-word test set with the same scorer as all other G2P baselines.

Tests the reviewer's "untrained vs unlearnable" distinction: if a small
model fine-tuned on the released data substantially beats few-shot
frontier LLMs, Yngping G2P is learnable and the few-shot failures
reflect missing exposure, not task impossibility.

Run on the remote H100 (from the project root):
  CUDA_VISIBLE_DEVICES=7 venv/bin/python3 papers-scripts/g2p_sft_remote.py \
      --model-path <QWEN3_1.7B_SNAPSHOT> \
      --resources-dir foochow-server/resources \
      --eval-set-file papers-data/batch_eval_set_v2.json \
      --out papers-data/g2p_sft_qwen3_1.7b.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, Trainer,
                          TrainingArguments)

YNGPING_TOKEN_RE = re.compile(r"([a-z]+)(\d+)")

PROMPT_TMPL = "Word: {word}\nYngping:"


def parse_yngping(s: str):
    s = s.strip().lower().split("\n")[0]
    out = []
    for tok in s.split():
        m = YNGPING_TOKEN_RE.match(tok)
        if m:
            out.append((m.group(1), m.group(2)))
    return out


def score_pair(pred: str, gold: str) -> dict:
    p, g = parse_yngping(pred), parse_yngping(gold)
    n = max(len(p), len(g))
    syl = tone = 0
    for i in range(n):
        ps = p[i] if i < len(p) else ("", "")
        gs = g[i] if i < len(g) else ("", "")
        if ps == gs:
            syl += 1
        if ps[1] == gs[1] and gs[1] != "":
            tone += 1
    return {"n_syl_gold": len(g), "n_syl_correct": syl, "n_tone_correct": tone}


def load_pairs(resources_dir: Path, exclude: set[str]) -> list[dict]:
    pron = pd.read_csv(resources_dir / "PronunciationResource.tsv", sep="\t", dtype=str,
                       keep_default_na=False, na_values=[""], encoding="utf-8", engine="python")
    word = pd.read_csv(resources_dir / "WordResource.tsv", sep="\t", dtype=str,
                       keep_default_na=False, na_values=[""], encoding="utf-8", engine="python")
    primary = pron[pron["is_primary"] == "1"][["word__id", "yngping"]]
    id2text = dict(zip(word["id"], word["text"]))
    primary = primary.assign(text=primary["word__id"].map(id2text))
    primary = primary[primary["text"].notna() & primary["yngping"].notna()]
    primary = primary[primary["text"].str.len().between(1, 8)]
    return [{"word": r["text"], "yngping": r["yngping"]}
            for _, r in primary.iterrows() if r["text"] not in exclude]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--resources-dir", type=Path, required=True)
    ap.add_argument("--eval-set-file", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    eval_set = json.loads(args.eval_set_file.read_text(encoding="utf-8"))
    test_words = {ex["text"] for ex in eval_set}
    train_pairs = load_pairs(args.resources_dir, test_words)
    random.shuffle(train_pairs)
    print(f"[sft] train pairs: {len(train_pairs)}  test: {len(eval_set)}", file=sys.stderr)

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def format_example(ex):
        prompt = PROMPT_TMPL.format(word=ex["word"])
        full = prompt + " " + ex["yngping"] + tok.eos_token
        enc = tok(full, truncation=True, max_length=64)
        # Mask the prompt part in labels so loss is on the answer only.
        prompt_len = len(tok(prompt)["input_ids"])
        labels = list(enc["input_ids"])
        labels[:prompt_len] = [-100] * prompt_len
        enc["labels"] = labels
        return enc

    ds = Dataset.from_list(train_pairs).map(format_example,
                                            remove_columns=["word", "yngping"])

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.config.use_cache = False
    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    class Collator(DataCollatorForLanguageModeling):
        def __call__(self, features):
            # pad input_ids/labels to max length in batch
            maxlen = max(len(f["input_ids"]) for f in features)
            batch = {"input_ids": [], "attention_mask": [], "labels": []}
            for f in features:
                pad = maxlen - len(f["input_ids"])
                batch["input_ids"].append(f["input_ids"] + [tok.pad_token_id] * pad)
                batch["attention_mask"].append([1] * len(f["input_ids"]) + [0] * pad)
                batch["labels"].append(f["labels"] + [-100] * pad)
            return {k: torch.tensor(v) for k, v in batch.items()}

    targs = TrainingArguments(
        output_dir="/tmp/g2p_sft_ckpt",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=50,
        save_strategy="no",
        bf16=True,
        report_to=[],
        seed=args.seed,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=Collator(tokenizer=tok, mlm=False))
    t0 = time.time()
    trainer.train()
    train_sec = time.time() - t0
    print(f"[sft] training done in {train_sec:.0f}s", file=sys.stderr)

    # ---- eval ----
    model.eval()
    model.config.use_cache = True
    tok.padding_side = "left"
    results = []
    t0 = time.time()
    for i in range(0, len(eval_set), args.eval_batch_size):
        batch = eval_set[i:i + args.eval_batch_size]
        prompts = [PROMPT_TMPL.format(word=ex["text"]) for ex in batch]
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=40, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        decoded = tok.batch_decode(gen, skip_special_tokens=True)
        for ex, pred in zip(batch, decoded):
            pred = pred.strip().split("\n")[0]
            sc = score_pair(pred, ex["gold_yngping"])
            we = 1 if pred == ex["gold_yngping"].strip() else 0
            results.append({**ex, "pred_yngping": pred, **sc, "word_exact": we})
        print(f"[eval] {len(results)}/{len(eval_set)}", file=sys.stderr)
    eval_sec = time.time() - t0

    sg = sum(r["n_syl_gold"] for r in results)
    by_len: dict[int, dict] = {}
    for r in results:
        d = by_len.setdefault(r["char_len"], {"n": 0, "exact": 0})
        d["n"] += 1
        d["exact"] += r["word_exact"]
    summary = {
        "model_label": "Qwen3-1.7B-LoRA-SFT",
        "model_path": args.model_path,
        "train_pairs": len(train_pairs),
        "epochs": args.epochs, "lr": args.lr, "lora_r": args.lora_r,
        "train_sec": round(train_sec), "eval_sec": round(eval_sec),
        "n_words": len(results),
        "word_exact_accuracy": sum(r["word_exact"] for r in results) / len(results),
        "syllable_accuracy": sum(r["n_syl_correct"] for r in results) / sg,
        "tone_accuracy": sum(r["n_tone_correct"] for r in results) / sg,
        "by_char_len": {k: {"n": v["n"], "word_exact_acc": v["exact"] / v["n"]}
                        for k, v in sorted(by_len.items())},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "results": results},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
