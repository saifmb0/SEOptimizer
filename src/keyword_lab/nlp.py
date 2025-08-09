import logging
import re
from collections import Counter
from typing import List, Tuple, Dict, Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords


EN_STOPWORDS = set()
try:
    EN_STOPWORDS = set(stopwords.words("english"))
except Exception:
    # Fallback: minimal set
    EN_STOPWORDS = {"the", "and", "is", "in", "to", "of", "a"}


CLEAN_RE = re.compile(r"[^a-z\s]+")


def clean_text(text: str) -> str:
    text = text.lower()
    text = CLEAN_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    return [t for t in text.split() if t and t not in EN_STOPWORDS]


def ngram_counts(texts: List[str], ngram_range=(2, 3), min_df: int = 2) -> pd.DataFrame:
    if not texts:
        return pd.DataFrame(columns=["ngram", "count"])    
    n_docs = len(texts)
    eff_min_df = min_df if n_docs >= min_df else max(1, n_docs)
    try:
        vectorizer = CountVectorizer(ngram_range=ngram_range, min_df=eff_min_df, stop_words="english")
        X = vectorizer.fit_transform(texts)
    except ValueError:
        # Fallback for very small corpora
        vectorizer = CountVectorizer(ngram_range=ngram_range, min_df=1, stop_words="english")
        X = vectorizer.fit_transform(texts)
    counts = np.asarray(X.sum(axis=0)).ravel()
    ngrams = vectorizer.get_feature_names_out()
    df = pd.DataFrame({"ngram": ngrams, "count": counts})
    df = df.sort_values("count", ascending=False)
    return df


QUESTION_PREFIXES = [
    "how", "what", "best", "vs", "for", "near me", "beginner", "advanced", "guide", "checklist", "template", "why"
]


def generate_questions(phrases: Iterable[str], top_n: int = 50) -> List[str]:
    qs = []
    for p in list(phrases)[:top_n]:
        if len(p.split()) < 2:
            continue
        for pref in QUESTION_PREFIXES:
            q = f"{pref} {p}".strip()
            if len(q.split()) >= 2:
                qs.append(q)
    return qs


def tfidf_top_terms_per_doc(texts: List[str], ngram_range=(2, 3), top_k: int = 10) -> List[str]:
    if not texts:
        return []
    vec = TfidfVectorizer(ngram_range=ngram_range, stop_words="english")
    X = vec.fit_transform(texts)
    terms = vec.get_feature_names_out()
    tops: List[str] = []
    for i in range(X.shape[0]):
        row = X.getrow(i).toarray().ravel()
        idx = np.argsort(-row)[:top_k]
        tops.extend([terms[j] for j in idx if row[j] > 0])
    return list(dict.fromkeys(tops))


def seed_expansions(seed: str, audience: Optional[str] = None) -> List[str]:
    s = clean_text(seed)
    aud = clean_text(audience or "")
    patterns = [
        "how to {s}",
        "what is {s}",
        "best {s}",
        "top {s}",
        "{s} guide",
        "{s} tutorial",
        "{s} checklist",
        "{s} template",
        "{s} vs alternatives",
        "compare {s}",
        "{s} pricing",
        "buy {s}",
        "where to buy {s}",
        "{s} near me",
        "beginner {s} guide",
        "advanced {s}",
    ]
    if aud:
        patterns.extend([
            "best {s} for {a}",
            "{s} for {a}",
            "how to {s} for {a}",
        ])
    cands = []
    for p in patterns:
        q = p.format(s=s, a=aud).strip()
        if len(q.split()) >= 2:
            cands.append(q)
    return list(dict.fromkeys(cands))


def generate_candidates(docs: List[Dict], ngram_min_df: int = 2, top_terms_per_doc: int = 10) -> List[str]:
    cleaned_docs = [clean_text(d.get("text", "")) for d in docs]
    cleaned_docs = [t for t in cleaned_docs if t]
    counts_df = ngram_counts(cleaned_docs, min_df=ngram_min_df)
    ngram_list = counts_df["ngram"].tolist()
    questions = generate_questions(ngram_list, top_n=min(50, len(ngram_list)))
    tfidf_terms = tfidf_top_terms_per_doc(cleaned_docs, top_k=top_terms_per_doc)
    cands = list(dict.fromkeys([*ngram_list, *questions, *tfidf_terms]))
    cands = [c.strip().lower() for c in cands if len(c.split()) >= 2]
    return cands
