# -*- coding: utf-8 -*-
"""Full-dictionary B-group evaluation v3: supplement-first colloquial queries.

Strategy:
  Tier 1: For words WITH natural Mandarin supplements →
          use supplement + "怎么说" template (most natural query pattern)
  Tier 2: For words WITHOUT usable supplements →
          use FULL first definition clause (simplified, NOT truncated)

Key change from v2: NO aggressive truncation. Complete meaning > short fragment.
"""

from __future__ import annotations

import io
import json
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_bot.embeddings import EmbeddingIndex  # noqa: E402
from rag_bot.index import SearchIndex  # noqa: E402
from rag_bot.loader import Word  # noqa: E402
from rag_bot.retriever import Retriever, SearchHit  # noqa: E402

CACHE = Path(__file__).resolve().parent.parent / "data"

# ---- Simplified Chinese ----
_T2S = str.maketrans(
    "說這裏東發動種過對時點長開會買賣飯學書認識話語變還錢歲與讓請該問頭風當後腦臉聲響樣從塊實處質體熱農節氣歷舊習鄉間廟雞鴨魚鳥蟲葉樹華國見親關係傳統寫聯紀類區範圍經濟產廠層選擇滿準備鬧調單歡嘆隻雜藝術醫藥飲機車輪廣場辦運連進遠達歸勞費價條筆紙張號電報陽陰寶貝銀鐵針線繩絲織綿襪褲帶離齡齊嶺額際隊陣險隨雲霧雙壞壓斷獨穩讀鏡鍋鐘鑰匙觀個終衛園圓圖團夠決沒漿潔災為烏無煙燈營獲環畫盤監盡總緒練緊繼續罵舉許詞誰談講議記試貨龍歸觸驗騎驚馬願顯飛養餘館饅餅餃飢飽飾裝複規覺覽視記設訪論証詳語誤調謝議護豐貓貧資賺質購趕軍轉輕載適選邊鄰醬釘鋪鋸閃閉閱關際陣隊隱雜離難電題額飄餐館驅驗體鮮麗麵黃龜鑿鏨鬱齒邏輯飼繡損撐擋搶擔擺攤攝攪歡彎歎禮壇壘壺壓嬌嬰嫵嫻嶼鏈鏢鍛錘錶鏽鏡鑼鑰鑲閘閣閱關闊鑄鑽闆闖闡闢閥陰陣陰陸際隊隨隱隸雕雙雜離難霧靈靜韻響頁頓頰頹頻願顛類顯飄飼餃餅餓餐館駕駛騙騷驅驗驚體髮鬆魔鮮鯊鯉鰻鷹鹹鹿麵麻黃點齊齒龍龜",
    "说这里东发动种过对时点长开会买卖饭学书认识话语变还钱岁与让请该问头风当后脑脸声响样从块实处质体热农节气历旧习乡间庙鸡鸭鱼鸟虫叶树华国见亲关系传统写联纪类区范围经济产厂层选择满准备闹调单欢叹只杂艺术医药饮机车轮广场办运连进远达归劳费价条笔纸张号电报阳阴宝贝银铁针线绳丝织绵袜裤带离龄齐岭额际队阵险随云雾双坏压断独稳读镜锅钟钥匙观个终卫园圆图团够决没浆洁灾为乌无烟灯营获环画盘监尽总绪练紧继续骂举许词谁谈讲议记试货龙归触验骑惊马愿显飞养余馆馒饼饺饥饱饰装复规觉览视记设访论证详语误调谢议护丰猫贫资赚质购赶军转轻载适选边邻酱钉铺锯闪闭阅关际阵队隐杂离难电题额飘餐馆驱验体鲜丽面黄龟凿錾郁齿逻辑饲绣损撑挡抢担摆摊摄搅欢弯叹礼坛垒壶压娇婴妩娴屿链镖锻锤表锈镜锣钥镶闸阁阅关阔铸钻板闯阐辟阀阴阵阴陆际队随隐隶雕双杂离难雾灵静韵响页顿颊颓频愿颠类显飘饲饺饼饿餐馆驾驶骗骚驱验惊体发松魔鲜鲨鲤鳗鹰咸鹿面麻黄点齐齿龙龟",
)

def simplify(text: str) -> str:
    return text.translate(_T2S)


# ---- Helpers ----
_RARE_CJK = re.compile(r"[\u3400-\u4dbf\U00020000-\U0003ffff]")
_DIALECT_CHARS = set("囝乇厝伊汝卜蜀㑚儥𣍐々")
_FORMAL_RE = re.compile(
    r"^(〈書〉|〈文〉|〈口〉|〈方〉|〈俗〉|〈敬〉|〈謙〉|"
    r"動賓短語[，,]?\s*|指|即|泛指|特指|又稱|也稱|也叫|俗稱|統稱|"
    r"形容詞[，,]?\s*|副詞[，,]?\s*|量詞[，,]?\s*|名詞[，,]?\s*|動詞[，,]?\s*)"
)


def is_natural_mandarin(text: str) -> bool:
    if not text or len(text) < 2: return False
    if _RARE_CJK.search(text): return False
    if any(c in text for c in _DIALECT_CHARS): return False
    if re.search(r"[a-zA-Z0-9*◇=〖〗]", text): return False
    return True


def is_example(text: str) -> bool:
    return "～" in text or "△" in text


def strip_markers(text: str) -> str:
    text = _FORMAL_RE.sub("", text).strip()
    return re.sub(r"^[，,。；、：\s]+", "", text)


# ---- Tier 1: Supplement-based queries ----

def pick_best_supplement(word: Word, exclude: Set[str]) -> str | None:
    """Pick the most natural Mandarin supplement for this word."""
    candidates = []
    for s in word.supplements:
        if s.type not in ("M", "S"): continue
        t = simplify(s.text)
        if not is_natural_mandarin(t): continue
        if len(t) < 2 or len(t) > 8: continue
        # Score: prefer 2-4 chars, common chars
        score = 10
        if 2 <= len(t) <= 4: score += 5
        elif len(t) <= 6: score += 2
        candidates.append((t, score))
    if not candidates:
        return None
    candidates.sort(key=lambda x: -x[1])
    return candidates[0][0]


def make_supplement_query(supplement: str, exclude: Set[str]) -> str | None:
    """Turn a supplement into a colloquial query using templates."""
    s = supplement
    # Templates in priority order — pick the first that's not in exclude
    templates = [
        f"{s}怎么说",
        f"怎么说{s}",
        f"{s}的说法",
        f"方言{s}",
        f"{s}用方言",
    ]
    for t in templates:
        if t not in exclude and is_natural_mandarin(t) and 4 <= len(t) <= 14:
            return t
    return None


# ---- Tier 2: Definition-based queries ----

def make_definition_query(word: Word, exclude: Set[str]) -> str | None:
    """Extract query from definition. Full clause, NO truncation."""
    # Collect first clause from each definition source
    candidates = []

    for e in word.expls:
        if not e.content or is_example(e.content): continue
        text = strip_markers(e.content)
        # Take first sentence/clause (up to first major punctuation)
        clause = re.split(r"[。；]", text)[0].strip()
        clause = re.split(r"[，、：]", clause)[0].strip() if len(clause) > 15 else clause
        if clause and len(clause) >= 2:
            candidates.append(simplify(clause))

    for f in word.fengs:
        for item in (f.content or []):
            if not isinstance(item, dict): continue
            expl = item.get("expl", "")
            if not expl or is_example(expl): continue
            text = strip_markers(expl)
            clause = re.split(r"[。；]", text)[0].strip()
            clause = re.split(r"[，、：]", clause)[0].strip() if len(clause) > 15 else clause
            if clause and len(clause) >= 2:
                candidates.append(simplify(clause))
            break  # first definition only per feng entry
        if candidates: break

    # Apply pattern transforms
    expanded = []
    for c in candidates:
        expanded.append(c)
        m = re.match(r"形容(.{2,10})", c)
        if m:
            inner = m.group(1).rstrip("的了")
            expanded.append(f"很{inner}")
            expanded.append(f"太{inner}了")
        m = re.match(r"比喻(.{2,12})", c)
        if m: expanded.append(m.group(1))
        m = re.match(r"表示(.{2,10})", c)
        if m: expanded.append(m.group(1))

    # Pick best
    for c in expanded:
        if c not in exclude and is_natural_mandarin(c) and 2 <= len(c) <= 20:
            if word.text not in c and c not in word.text:
                return c
    return None


# ---- Combined query generation ----

def generate_query(word: Word, exclude: Set[str]) -> Tuple[str, str] | None:
    """Generate best colloquial query. Returns (query, tier)."""
    # Tier 1: supplement-based
    supp = pick_best_supplement(word, exclude)
    if supp:
        q = make_supplement_query(supp, exclude)
        if q:
            return q, "tier1_supp"

    # Tier 2: definition-based (full clause)
    q = make_definition_query(word, exclude)
    if q:
        return q, "tier2_def"

    return None


# ---- Evaluation ----
class VectorOnlyRetriever:
    def __init__(self, words, emb):
        self.words = words; self.emb = emb
    def search(self, query, top_k=10):
        return [SearchHit(word_id=wid, score=s, match_type="vector")
                for wid, s in self.emb.search(query, top_k=top_k)]
    def get(self, wid):
        return self.words.get(wid)


def evaluate_batch(retriever, queries, top_k=10):
    hits1 = hits3 = hits5 = hits10 = 0
    total_rr = 0.0; miss = 0
    for q, expected_ids in queries:
        results = retriever.search(q, top_k=top_k)
        expected = set(expected_ids)
        rank = None
        for i, h in enumerate(results):
            if h.word_id in expected:
                rank = i + 1; break
        if rank:
            total_rr += 1.0 / rank
            if rank <= 1: hits1 += 1
            if rank <= 3: hits3 += 1
            if rank <= 5: hits5 += 1
            if rank <= 10: hits10 += 1
        else: miss += 1
    n = len(queries)
    return {"n": n, "miss": miss,
            "hit@1": hits1/n, "hit@3": hits3/n, "hit@5": hits5/n, "hit@10": hits10/n,
            "mrr": total_rr/n}


def main():
    print("[load]...")
    with open(CACHE / "words.pkl", "rb") as f:
        words = pickle.load(f)
    index = SearchIndex.load(CACHE / "search_index.pkl")
    emb = EmbeddingIndex.try_load(CACHE / "embedding_index")

    # Exclusion set
    exclude: Set[str] = set()
    for w in words.values():
        exclude.add(w.text); exclude.add(simplify(w.text))
        for s in w.supplements:
            exclude.add(s.text); exclude.add(simplify(s.text))

    # Generate
    print("[generate]...")
    queries: List[Tuple[str, List[int]]] = []
    tier_cnt = Counter()
    skipped = 0

    for wid, w in sorted(words.items()):
        if not w.expls and not w.fengs:
            continue
        result = generate_query(w, exclude)
        if result is None:
            skipped += 1; continue
        q, tier = result
        queries.append((q, [wid]))
        tier_cnt[tier] += 1

    # Dedup
    qmap: Dict[str, List[int]] = defaultdict(list)
    for q, wids in queries:
        for wid in wids: qmap[q].append(wid)
    queries_dedup = [(q, wids) for q, wids in qmap.items()]

    print(f"  generated: {len(queries)}, deduped: {len(queries_dedup)}, skipped: {skipped}")
    print(f"  tiers: {dict(tier_cnt)}")

    # Save
    ds_path = CACHE / "eval_full_dict_b_v3.json"
    dataset = [{"query": q, "expected_word_ids": wids,
                "expected_texts": [words[wid].text for wid in wids if wid in words]}
               for q, wids in queries_dedup]
    with open(ds_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    # Samples by tier
    tier1_samples = [(q, wids) for (q, wids), (_, t) in zip(queries, [(q, generate_query(words[wids[0]], exclude)) for q, wids in queries[:500]]) if False]
    print(f"\n  --- Tier 1 samples (supplement-based) ---")
    cnt = 0
    for q, wids in queries_dedup:
        if "怎么说" in q and cnt < 15:
            texts = [words[wid].text for wid in wids if wid in words]
            print(f"  {q:24s} → {texts}")
            cnt += 1

    print(f"\n  --- Tier 2 samples (definition-based) ---")
    cnt = 0
    for q, wids in queries_dedup:
        if "怎么说" not in q and cnt < 15:
            texts = [words[wid].text for wid in wids if wid in words]
            print(f"  {q:24s} → {texts}")
            cnt += 1

    # Retrievers
    r_bm = Retriever(words, index, embedding_index=None)
    r_hy = Retriever(words, index, embedding_index=emb) if emb else None
    r_vec = VectorOnlyRetriever(words, emb) if emb else None

    systems = {"bm25_only": r_bm}
    if r_vec: systems["vector_only"] = r_vec
    if r_hy: systems["hybrid"] = r_hy

    # Evaluate
    print(f"\n[eval] {len(queries_dedup)} queries x {len(systems)} systems")
    results = {}
    for name, ret in systems.items():
        print(f"  evaluating {name}...")
        results[name] = evaluate_batch(ret, queries_dedup)

    # Report
    print(f"\n{'='*80}")
    print(f"FULL DICT B-GROUP v3 ({len(queries_dedup)} queries)")
    print("=" * 80)
    print(f"{'系统':20s} {'Hit@1':>8s} {'Hit@3':>8s} {'Hit@5':>8s} {'Hit@10':>8s} {'MRR':>8s} {'Miss':>6s}")
    print("-" * 80)
    for name in systems:
        m = results[name]
        print(f"{name:20s} {m['hit@1']:8.3f} {m['hit@3']:8.3f} {m['hit@5']:8.3f} {m['hit@10']:8.3f} {m['mrr']:8.3f} {m['miss']:5d}")

    # Evaluate ONLY tier1 queries (supplement-based)
    tier1_qs = [(q, wids) for q, wids in queries_dedup if "怎么说" in q]
    tier2_qs = [(q, wids) for q, wids in queries_dedup if "怎么说" not in q]

    for label, qs in [("TIER 1 ONLY (supplement怎么说)", tier1_qs), ("TIER 2 ONLY (definition)", tier2_qs)]:
        if not qs: continue
        print(f"\n{'='*80}")
        print(f"{label} ({len(qs)} queries)")
        print("=" * 80)
        print(f"{'系统':20s} {'Hit@1':>8s} {'Hit@3':>8s} {'Hit@5':>8s} {'Hit@10':>8s} {'MRR':>8s} {'Miss':>6s}")
        print("-" * 80)
        for name, ret in systems.items():
            m = evaluate_batch(ret, qs)
            print(f"{name:20s} {m['hit@1']:8.3f} {m['hit@3']:8.3f} {m['hit@5']:8.3f} {m['hit@10']:8.3f} {m['mrr']:8.3f} {m['miss']:5d}")

    # Head-to-head
    if r_hy and r_bm:
        print(f"\n{'='*80}")
        print("HEAD-TO-HEAD: hybrid vs bm25 (all)")
        print("=" * 80)
        hy_wins = bm_wins = ties = both_miss = only_hy = only_bm = 0
        for q, wids in queries_dedup:
            expected = set(wids)
            bm_hits = r_bm.search(q, top_k=10)
            hy_hits = r_hy.search(q, top_k=10)
            def rank_of(hits):
                for i, h in enumerate(hits):
                    if h.word_id in expected: return i + 1
                return None
            br, hr = rank_of(bm_hits), rank_of(hy_hits)
            brr = (1.0/br) if br else 0
            hrr = (1.0/hr) if hr else 0
            if hrr > brr: hy_wins += 1
            elif brr > hrr: bm_wins += 1
            else: ties += 1
            if not br and not hr: both_miss += 1
            elif hr and not br: only_hy += 1
            elif br and not hr: only_bm += 1
        print(f"  hybrid赢: {hy_wins}  |  bm25赢: {bm_wins}  |  平: {ties}")
        print(f"  都没找到: {both_miss}  |  只hybrid找到: {only_hy}  |  只bm25找到: {only_bm}")

    # Save
    with open(CACHE / "eval_full_dict_v3_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[done]")


if __name__ == "__main__":
    main()
