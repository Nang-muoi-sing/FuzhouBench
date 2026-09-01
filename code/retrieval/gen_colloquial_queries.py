"""Generate colloquial Mandarin queries for full dictionary using local LLM.

Uses vLLM offline batch inference with Qwen3-8B to rewrite dictionary
definitions into natural search queries that a real user would type.
"""

from __future__ import annotations

import json
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_bot.loader import Word  # noqa: E402

CACHE = Path(__file__).resolve().parent.parent / "data"

# ---- Simplified Chinese conversion (same as eval_full_dict) ----
_T2S = str.maketrans(
    "說這裏東發動種過對時點長開會買賣飯學書認識話語變還錢歲與讓請該問頭風當後腦臉聲響樣從塊實處質體熱農節氣歷舊習鄉間廟雞鴨魚鳥蟲葉樹華國見親關係傳統寫聯紀類區範圍經濟產廠層選擇滿準備鬧調單歡嘆隻雜藝術醫藥飲機車輪廣場辦運連進遠達歸勞費價條筆紙張號電報陽陰寶貝銀鐵針線繩絲織綿襪褲帶離齡齊嶺額際隊陣險隨雲霧雙壞壓斷獨穩讀鏡鍋鐘鑰匙觀個終衛園圓圖團夠決沒漿潔災為烏無煙燈營獲環畫盤監盡總緒練緊繼續罵舉許詞誰談講議記試貨",
    "说这里东发动种过对时点长开会买卖饭学书认识话语变还钱岁与让请该问头风当后脑脸声响样从块实处质体热农节气历旧习乡间庙鸡鸭鱼鸟虫叶树华国见亲关系传统写联纪类区范围经济产厂层选择满准备闹调单欢叹只杂艺术医药饮机车轮广场办运连进远达归劳费价条笔纸张号电报阳阴宝贝银铁针线绳丝织绵袜裤带离龄齐岭额际队阵险随云雾双坏压断独稳读镜锅钟钥匙观个终卫园圆图团够决没浆洁灾为乌无烟灯营获环画盘监尽总绪练紧继续骂举许词谁谈讲议记试货",
)

_DIALECT_CHARS = set("囝乇厝伊汝卜蜀㑚儥𣍐々～")
_RARE_CJK = re.compile(r"[\u3400-\u4dbf\U00020000-\U0003ffff]")


def simplify(text: str) -> str:
    return text.translate(_T2S)


# ---- Build definitions text for a word ----

def get_definitions(w: Word) -> str:
    """Extract all definition texts for a word, simplified Chinese."""
    parts = []

    for e in w.expls:
        if e.content:
            c = simplify(e.content.replace("～", w.text))
            cat = f"[{simplify(e.lexical_category)}] " if e.lexical_category else ""
            parts.append(f"{cat}{c}")

    for f in w.fengs[:3]:
        for item in (f.content or [])[:3]:
            if isinstance(item, dict) and item.get("expl"):
                parts.append(simplify(str(item["expl"])))

    return "；".join(parts) if parts else ""


def should_skip(w: Word, exclude: Set[str]) -> bool:
    """Check if word should be skipped (no definitions, dialect-specific, etc.)."""
    if not w.expls and not w.fengs:
        return True
    defs = get_definitions(w)
    if not defs or len(defs) < 4:
        return True
    text_s = simplify(w.text)
    if any(c in text_s for c in _DIALECT_CHARS):
        return True
    return False


# ---- Prompt construction ----

SYSTEM_PROMPT = """你是一个普通话母语者，需要在一本福州话词典中查找词条。
根据给出的词条和释义，写出你会在搜索框中输入的自然口语化查询。

要求：
1. 像日常说话一样自然，不要照搬释义原文
2. 简洁（2-8个字最佳，不超过12个字）
3. 用简体中文
4. 如果释义太专业或太生僻，就用最通俗的说法
5. 只输出查询本身，不要加序号、引号或解释"""

# Representative few-shot examples from hand-written rewrites
FEW_SHOT_EXAMPLES = [
    ("做無算", "比喻做事有始无终", "做事有头没尾"),
    ("厝囝厝伲", "形容非常湿", "浑身湿透了"),
    ("僱", "雇工", "花钱请人干活"),
    ("做晬", "孩子满周岁行抓周礼", "小孩周岁抓周"),
    ("喙", "口的通称", "嘴巴"),
    ("菜婆", "指带发修行的尼姑", "带发修行的尼姑"),
    ("箸", "夹菜的竹制餐具；筷子", "夹菜的筷子"),
    ("開面", "出嫁前用线绞去脸上的汗毛", "出嫁前绞脸"),
    ("簫", "管乐器的一种；吹奏乐器", "吹唢呐的乐器"),
    ("歲數", "人的年龄", "多大岁数了"),
    ("蒲扇", "用蒲葵叶制成的扇子", "摇扇子纳凉"),
    ("收拾", "为了管束、惩罚等使吃苦头", "收拾一顿"),
    ("薅草", "拔除田间杂草", "拔田里的草"),
    ("褥", "垫在身体下面的卧具", "垫在身下睡觉的"),
    ("背書", "凭记忆念出读过的书", "背课文"),
    ("舀", "用瓢等取液体", "舀水的东西"),
    ("腿抽筋", "小腿肚痉挛", "腿抽筋了"),
    ("扁擔", "扁平的石头", "扁平的石头"),
    ("蜂箱", "养蜂的木箱", "养蜜蜂的箱子"),
    ("紅麴酒", "用红曲酿造的酒", "红曲酿的酒"),
]


def build_prompt(word_text: str, definition: str) -> str:
    """Build the chat prompt for one word."""
    examples_text = "\n".join(
        f"词条：{w}  释义：{d}  → {q}"
        for w, d, q in FEW_SHOT_EXAMPLES
    )
    return f"""{examples_text}

词条：{simplify(word_text)}  释义：{definition}  →"""


def _clean_query(raw: str, word_text_simplified: str) -> str:
    """Clean LLM output into a usable search query."""
    q = raw
    # Strip Qwen3 thinking tags if present
    q = re.sub(r'<think>.*?</think>', '', q, flags=re.DOTALL).strip()
    q = q.split("\n")[0].strip()
    # Remove markdown, brackets, quotes
    q = re.sub(r'[`\[\]()（）「」『』""''\{\}]', '', q)
    q = q.strip("\"'，。、；：!！?？ ·…—")
    # Remove continuations
    for stop in ["词条", "释义", "→", "怎么说", "怎么用", "是什么意思", "：", ":"]:
        if stop in q:
            idx = q.index(stop)
            if idx > 1:
                q = q[:idx].strip()
            else:
                q = q[idx + len(stop):].strip()
    # Remove pure copies of the word itself
    if q == word_text_simplified:
        q = ""
    # Final cleanup
    q = q.strip("\"'，。、；：!！?？ ·…—")
    if len(q) < 2 or len(q) > 20:
        # Fallback: take first meaningful chunk from raw
        fb = re.sub(r'[`\[\]()（）「」『』""''\{\}]', '', raw)
        fb = fb.split("\n")[0].strip("\"'，。、；：!！?？ ·…—")[:16]
        if len(fb) >= 2:
            q = fb
    return q


# ---- Main ----

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="<HF_CACHE>/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218")
    p.add_argument("--output", default=str(CACHE / "colloquial_queries_llm.json"))
    p.add_argument("--max-words", type=int, default=0, help="Limit for testing (0=all)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-model-len", type=int, default=1536)
    args = p.parse_args()

    # Load data
    print("[load] words...")
    with open(CACHE / "words.pkl", "rb") as f:
        words: Dict[int, Word] = pickle.load(f)

    # Build exclusion set (word texts + supplements — to avoid trivial queries)
    exclude: Set[str] = set()
    for w in words.values():
        exclude.add(w.text)
        exclude.add(simplify(w.text))
        for s in w.supplements:
            exclude.add(s.text)
            exclude.add(simplify(s.text))

    # Prepare entries
    entries: List[Tuple[int, str, str]] = []  # (word_id, word_text, definition)
    for wid, w in sorted(words.items()):
        if should_skip(w, exclude):
            continue
        defs = get_definitions(w)
        entries.append((wid, w.text, defs))

    if args.max_words > 0:
        entries = entries[:args.max_words]

    print(f"[prep] {len(entries)} words to process")

    # Build prompts using transformers tokenizer chat template
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[model] loading: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True,
                                               padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()
    print(f"[model] loaded. device={next(model.parameters()).device}")

    print(f"[gen] generating {len(entries)} queries in batches of {args.batch_size}...")
    t0 = time.time()

    results = []
    for batch_start in range(0, len(entries), args.batch_size):
        batch = entries[batch_start:batch_start + args.batch_size]
        batch_prompts = []

        for wid, text, defs in batch:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(text, defs)},
            ]
            # Qwen3 models support enable_thinking; disable it for direct output
            try:
                prompt_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                prompt_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
            batch_prompts.append(prompt_text)

        # Tokenize batch
        inputs = tokenizer(
            batch_prompts, return_tensors="pt", padding=True, truncation=True,
            max_length=args.max_model_len,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=32,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        for i, (wid, text, defs) in enumerate(batch):
            input_len = inputs["input_ids"][i].shape[0]
            new_tokens = outputs[i][input_len:]
            raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            query = _clean_query(raw, simplify(text))

            results.append({
                "word_id": wid,
                "word_text": text,
                "definition": defs,
                "query": query,
                "raw_output": raw,
            })

        done = min(batch_start + args.batch_size, len(entries))
        elapsed = time.time() - t0
        speed = done / elapsed if elapsed > 0 else 0
        print(f"  [{done}/{len(entries)}] {speed:.1f} q/s  last: {results[-1]['query']}")

    elapsed = time.time() - t0
    print(f"[gen] done in {elapsed:.1f}s ({len(entries)/elapsed:.1f} queries/sec)")

    # Save
    out_path = Path(args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[save] {len(results)} queries saved to {out_path}")

    # Show samples
    print(f"\n--- Samples (first 30) ---")
    for r in results[:30]:
        print(f"  {simplify(r['word_text']):8s}  释义: {r['definition'][:30]:30s}  → {r['query']}")

    # Stats
    lens = [len(r["query"]) for r in results]
    print(f"\n[stats] query lengths: min={min(lens)}, max={max(lens)}, "
          f"avg={sum(lens)/len(lens):.1f}, median={sorted(lens)[len(lens)//2]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
