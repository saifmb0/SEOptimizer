import json
import logging
import os
from typing import Dict, List, Optional

import numpy as np
from dotenv import load_dotenv
import yaml

from .scrape import acquire_documents, Document
from .nlp import generate_candidates, clean_text, QUESTION_PREFIXES, seed_expansions
from .cluster import cluster_keywords, infer_intent
from .metrics import compute_metrics, opportunity_scores
from .schema import validate_items
from .io import write_json, write_csv
from .llm import expand_with_gemini


def to_funnel_stage(intent: str) -> str:
    if intent == "informational":
        return "TOFU"
    if intent == "transactional":
        return "BOFU"
    return "MOFU"


def run_pipeline(
    seed_topic: str,
    audience: str,
    geo: str = "global",
    language: str = "en",
    competitors: Optional[List[str]] = None,
    business_goals: str = "traffic",
    capabilities: str = "no-paid-apis",
    time_horizon: str = "quarter",
    max_clusters: int = 8,
    max_keywords_per_cluster: int = 12,
    sources: Optional[str] = None,
    query: Optional[str] = None,
    provider: str = "none",
    output: str = "keywords.json",
    save_csv: Optional[str] = None,
    verbose: bool = False,
    config: Optional[Dict] = None,
    dry_run: bool = False,
) -> List[Dict]:
    load_dotenv()

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    nlp_cfg = (config or {}).get("nlp", {})
    scrape_cfg = (config or {}).get("scrape", {})

    # Acquire documents
    docs = acquire_documents(
        sources=sources,
        query=query or seed_topic,
        provider=provider,
        max_serp_results=int(scrape_cfg.get("max_serp_results", 10)),
        timeout=int(scrape_cfg.get("timeout", 10)),
        retries=int(scrape_cfg.get("retries", 2)),
        user_agent=str(scrape_cfg.get("user_agent", os.getenv("USER_AGENT", "keyword-lab/1.0"))),
        dry_run=dry_run,
    )

    # Build pseudo-doc from seed_topic and audience if no content
    if not docs:
        docs = [Document(url="seed", title=seed_topic, text=f"{seed_topic} {audience} {language} {geo}")]

    # Generate keyword candidates from documents
    doc_dicts = [dict(url=d.url, title=d.title, text=d.text) for d in docs]
    candidates = generate_candidates(
        doc_dicts,
        ngram_min_df=int(nlp_cfg.get("ngram_min_df", 2)),
        top_terms_per_doc=int(nlp_cfg.get("top_terms_per_doc", 10)),
    )

    # Seed-based and Gemini expansions
    seed_cands = seed_expansions(seed_topic, audience)
    gemini_cands = expand_with_gemini(seed_topic, audience, language, geo, max_results=50)
    candidates = list(dict.fromkeys([*candidates, *seed_cands, *gemini_cands]))

    # Early exit if no candidates
    if not candidates:
        if output:
            write_json([], output)
        if save_csv:
            write_csv([], save_csv)
        return []

    # Detect question-style keywords for volume boost
    qset = set()
    for kw in candidates:
        for pref in QUESTION_PREFIXES:
            if kw.startswith(pref + " "):
                qset.add(kw)
                break

    # Frequency from docs combined
    from sklearn.feature_extraction.text import CountVectorizer

    cleaned_texts = [clean_text(d["text"]) for d in doc_dicts if d.get("text")]
    vectorizer = CountVectorizer(vocabulary=candidates, ngram_range=(1, 3))
    try:
        X = vectorizer.fit_transform(cleaned_texts)
        sums = np.asarray(X.sum(axis=0)).ravel()
        freq = {kw: int(sums[i]) for i, kw in enumerate(vectorizer.get_feature_names_out())}
    except Exception:
        freq = {kw: 1 for kw in candidates}

    # Cluster
    clusters = cluster_keywords(candidates, max_clusters=max_clusters)

    # Intent mapping
    competitor_list = [c.strip() for c in (competitors or []) if c.strip()]
    intents = {kw: infer_intent(kw, competitor_list) for kw in candidates}

    # Metrics (0–1 scale)
    serp_total = None
    metrics = compute_metrics(candidates, clusters, intents, freq, qset, provider, serp_total)
    opp = opportunity_scores(metrics, intents, business_goals)

    # Assemble per cluster, prioritize opportunity score within clusters
    items: List[Dict] = []
    for cname, kws in clusters.items():
        kws_sorted = sorted(kws, key=lambda k: (opp.get(k, 0), metrics.get(k, {}).get("search_volume", 0)), reverse=True)
        selected = []
        seen = set()
        for kw in kws_sorted:
            if kw in seen:
                continue
            seen.add(kw)
            selected.append(kw)
            if len(selected) >= max_keywords_per_cluster:
                break
        for kw in selected:
            it = {
                "keyword": kw.lower(),
                "cluster": cname.lower(),
                "intent": intents.get(kw, "informational"),
                "funnel_stage": to_funnel_stage(intents.get(kw, "informational")),
                "search_volume": float(metrics.get(kw, {}).get("search_volume", 0.0)),
                "difficulty": float(metrics.get(kw, {}).get("difficulty", 0.0)),
                "estimated": bool(metrics.get(kw, {}).get("estimated", True)),
                "opportunity_score": float(min(1.0, max(0.0, opp.get(kw, 0.0)))),
            }
            items.append(it)

    # Global ranking by opportunity_score (then search_volume desc, then keyword asc)
    items = sorted(items, key=lambda it: (-it["opportunity_score"], -it["search_volume"], it["keyword"]))

    # Validate
    validate_items(items)

    # Persist outputs
    if output:
        write_json(items, output)
    if save_csv:
        write_csv(items, save_csv)

    return items
