import logging
import os
from typing import List


def expand_with_gemini(seed: str, audience: str = "", language: str = "en", geo: str = "global", max_results: int = 30) -> List[str]:
    try:
        import google.generativeai as genai  # type: ignore
    except Exception:
        return []

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return []

    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        prompt = (
            "Generate a diverse, high-quality list of long-tail SEO keywords and questions as plain text, one per line.\n"
            f"Seed topic: {seed}\nAudience: {audience}\nLanguage: {language}\nGeo: {geo}\n"
            "Rules: lowercase, bigrams/trigrams or longer, focus on search intent, include questions (who/what/why/how/best/vs/for/near me/beginner/advanced/guide/checklist/template)."
            f"Return at most {max_results} items. No numbering, no extra prose."
        )
        resp = model.generate_content(prompt)
        text = getattr(resp, "text", "") or ""
        lines = [ln.strip().lower() for ln in text.splitlines()]
        items = [ln for ln in lines if ln and len(ln.split()) >= 2]
        # Deduplicate while preserving order
        seen = set()
        out = []
        for k in items:
            if k not in seen:
                seen.add(k)
                out.append(k)
        return out[:max_results]
    except Exception as e:
        logging.debug(f"Gemini expansion failed: {e}")
        return []
