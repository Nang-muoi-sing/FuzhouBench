"""Format Word records for terminal display and LLM prompts."""

from __future__ import annotations

from .loader import Word


def format_word_terminal(w: Word) -> str:
    """Human-readable terminal format."""
    lines: list[str] = []
    header = f"【{w.text}】"
    py = w.primary_yngping
    if py:
        header += f"  {py}"
    if w.gloss:
        header += f"  — {w.gloss}"
    lines.append(header)

    # All pronunciations
    if len(w.prons) > 1:
        pron_parts = []
        for p in w.prons:
            tag = ""
            if p.is_primary:
                tag = "[主]"
            elif not p.is_sandhi:
                tag = "[本]"
            if p.variant:
                tag += f"[{p.variant}]"
            pron_parts.append(f"{p.yngping}{tag}")
        lines.append("  读音: " + " / ".join(pron_parts))

    # Phonology
    phon = []
    if w.phonology_initial:
        phon.append(f"声母={w.phonology_initial}")
    if w.phonology_final:
        phon.append(f"韵母={w.phonology_final}")
    if w.phonology_tone:
        phon.append(f"调={w.phonology_tone}")
    if phon:
        lines.append("  音韵: " + " ".join(phon))

    # Explanations
    for e in w.expls:
        cat = f"[{e.lexical_category}] " if e.lexical_category else ""
        lines.append(f"  {e.order}. {cat}{e.content}")
        for s in e.sentences:
            lines.append(f"     例: {s}")
        if e.comment:
            lines.append(f"     注: {e.comment}")

    # Feng dictionary entries
    for f in w.fengs:
        if f.content:
            for item in f.content:
                if isinstance(item, dict) and item.get("expl"):
                    lines.append(f"  [冯书] {item['expl']}")
                    for s in item.get("sent", []) or []:
                        lines.append(f"     例: {s}")

    # Supplements (alternative forms)
    if w.supplements:
        alts = [s.text for s in w.supplements if s.type in ("M", "S")]
        if alts:
            lines.append(f"  异体: {' / '.join(alts[:8])}")

    return "\n".join(lines)


def format_word_compact(w: Word) -> str:
    """One-line format for result lists."""
    py = w.primary_yngping
    gloss = w.gloss or (w.expls[0].content[:30] if w.expls else "")
    parts = [f"【{w.text}】"]
    if py:
        parts.append(py)
    if gloss:
        parts.append(f"— {gloss}")
    return " ".join(parts)


def format_word_for_llm(w: Word) -> str:
    """Dense, structured format for LLM context. Keep token count reasonable."""
    lines = [f"词条: {w.text} (id={w.id})"]
    if w.gloss:
        lines.append(f"简释: {w.gloss}")

    if w.prons:
        prons_fmt = []
        for p in w.prons[:6]:
            tags = []
            if p.is_primary:
                tags.append("主要")
            if not p.is_sandhi:
                tags.append("本字音")
            else:
                tags.append("连读音")
            if p.variant:
                tags.append(p.variant)
            prons_fmt.append(f"{p.yngping}({','.join(tags)})")
        lines.append("读音: " + " | ".join(prons_fmt))

    if w.phonology_initial or w.phonology_final or w.phonology_tone:
        lines.append(
            f"音韵: 声母={w.phonology_initial} 韵母={w.phonology_final} 调={w.phonology_tone}"
        )

    if w.expls:
        lines.append("释义:")
        for e in w.expls[:5]:
            cat = f"[{e.lexical_category}]" if e.lexical_category else ""
            lines.append(f"  {e.order}. {cat} {e.content}")
            for s in e.sentences[:3]:
                lines.append(f"     例句: {s}")

    if w.fengs:
        feng_expls = []
        for f in w.fengs[:2]:
            for item in (f.content or [])[:3]:
                if isinstance(item, dict) and item.get("expl"):
                    feng_expls.append(item["expl"])
        if feng_expls:
            lines.append("冯书释义: " + " ; ".join(feng_expls))

    if w.supplements:
        alts = [s.text for s in w.supplements if s.type in ("M", "S")][:6]
        if alts:
            lines.append(f"异体/关联: {' / '.join(alts)}")

    return "\n".join(lines)
