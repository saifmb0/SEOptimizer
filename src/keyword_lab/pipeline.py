import json
import logging
import os
from typing import Dict, List, Optional

import numpy as np
from dotenv import load_dotenv

from .scrape import acquire_documents, Document, validate_keywords_with_autocomplete
from .nlp import generate_candidates, clean_text, DEFAULT_QUESTION_PREFIXES, seed_expansions
from .cluster import cluster_keywords, infer_intent
from .metrics import compute_metrics, opportunity_scores
from .schema import validate_items
from .io import write_output
from .llm import expand_with_llm, assign_parent_topics, verify_candidates_with_llm
from .config import load_config, get_intent_rules, get_question_prefixes, config_to_dict


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
    config = None,
    dry_run: bool = False,
    niche: Optional[str] = None,
) -> List[Dict]:
    load_dotenv()

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Convert config to dict for backwards-compatible access
    cfg_dict = config_to_dict(config)
    
    # Get config sections
    nlp_cfg = cfg_dict.get("nlp", {})
    scrape_cfg = cfg_dict.get("scrape", {})
    
    # Get configurable rules (handles both Pydantic and dict)
    intent_rules = get_intent_rules(config)
    question_prefixes = get_question_prefixes(config)

    # Acquire documents (with caching for faster iterations)
    docs = acquire_documents(
        sources=sources,
        query=query or seed_topic,
        provider=provider,
        max_serp_results=int(scrape_cfg.get("max_serp_results", 10)),
        timeout=int(scrape_cfg.get("timeout", 10)),
        retries=int(scrape_cfg.get("retries", 2)),
        user_agent=str(scrape_cfg.get("user_agent", os.getenv("USER_AGENT", "keyword-lab/1.0"))),
        dry_run=dry_run,
        use_cache=bool(scrape_cfg.get("cache_enabled", True)),
        cache_dir=str(scrape_cfg.get("cache_dir", ".keyword_lab_cache")),
    )

    # Build pseudo-doc from seed_topic and audience if no content
    # Use newlines to separate components - this prevents n-gram crossing
    # between unrelated text segments (matching the fix in nlp.py)
    if not docs:
        # Split audience by commas to create separate lines
        audience_parts = [a.strip() for a in audience.split(",") if a.strip()]
        pseudo_text = "\n".join([
            seed_topic,
            *audience_parts,  # Each audience segment on its own line
            f"{language} {geo}",
        ])
        docs = [Document(url="seed", title=seed_topic, text=pseudo_text)]

    # Generate keyword candidates from documents
    doc_dicts = [dict(url=d.url, title=d.title, text=d.text) for d in docs]
    candidates = generate_candidates(
        doc_dicts,
        ngram_min_df=int(nlp_cfg.get("ngram_min_df", 2)),
        top_terms_per_doc=int(nlp_cfg.get("top_terms_per_doc", 10)),
        question_prefixes=question_prefixes,
    )

    # Seed-based and LLM expansions
    seed_cands = seed_expansions(seed_topic, audience)
    llm_cfg = cfg_dict.get("llm", {})
    llm_cands = expand_with_llm(
        seed_topic, audience, language, geo, 
        max_results=int(llm_cfg.get("max_expansion_results", 50)),
        provider=llm_cfg.get("provider", "auto"),
        model=llm_cfg.get("model"),
    )
    candidates = list(dict.fromkeys([*candidates, *seed_cands, *llm_cands]))

    # Apply blacklist and length filtering
    filter_cfg = cfg_dict.get("filtering", {})
    blacklist = [b.lower() for b in filter_cfg.get("blacklist", [])]
    min_words = int(filter_cfg.get("min_words", 2))
    max_words = int(filter_cfg.get("max_words", 6))
    min_chars = int(filter_cfg.get("min_chars", 5))
    
    def passes_filter(kw: str) -> bool:
        kw_lower = kw.lower()
        words = kw.split()
        # Check blacklist
        for bl in blacklist:
            if bl in kw_lower:
                return False
        # Check length constraints
        if len(words) < min_words or len(words) > max_words:
            return False
        if len(kw) < min_chars:
            return False
        return True
    
    candidates_before = len(candidates)
    candidates = [c for c in candidates if passes_filter(c)]
    if candidates_before > len(candidates):
        logging.info(f"Filtered {candidates_before - len(candidates)} keywords (blacklist/length)")

    # Early exit if no candidates
    if not candidates:
        if output:
            write_output([], output, save_csv)
        return []

    # Detect question-style keywords for volume boost
    qset = set()
    for kw in candidates:
        for pref in question_prefixes:
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

    # Cluster with optional silhouette-based K selection
    cluster_cfg = cfg_dict.get("cluster", {})
    clusters = cluster_keywords(
        candidates, 
        max_clusters=max_clusters,
        use_silhouette=bool(cluster_cfg.get("use_silhouette", False)),
        silhouette_k_range=tuple(cluster_cfg.get("silhouette_k_range", [4, 15])),
    )

    # Intent mapping using configurable rules
    competitor_list = [c.strip() for c in (competitors or []) if c.strip()]
    intents = {kw: infer_intent(kw, competitor_list, intent_rules) for kw in candidates}

    # Validate keywords with Google Autocomplete (if enabled)
    validation_cfg = cfg_dict.get("validation", {})
    validated_keywords: Dict[str, bool] = {}
    if validation_cfg.get("autocomplete_enabled", True) and not dry_run:
        # Limit to top candidates by frequency to avoid rate limiting
        top_candidates = sorted(candidates, key=lambda k: freq.get(k, 0), reverse=True)
        max_validate = int(validation_cfg.get("max_autocomplete_checks", 100))
        validated_keywords = validate_keywords_with_autocomplete(
            top_candidates[:max_validate],
            language=language,
            country=geo if len(geo) == 2 else "us",
            delay=float(validation_cfg.get("autocomplete_delay", 0.1)),
        )

    # Metrics (0–1 scale) with autocomplete validation and universal term detection
    # Pass total_docs for detecting terms that appear in every document (likely noise)
    total_docs = len(doc_dicts)
    metrics = compute_metrics(
        candidates, clusters, intents, freq, qset, provider,
        serp_total_results=None,  # TODO: pass per-keyword SERP counts when available
        validated_keywords=validated_keywords,
        total_docs=total_docs,
    )
    
    # Get commercial weight from config if available
    scoring_cfg = cfg_dict.get("scoring", {})
    commercial_weight = float(scoring_cfg.get("commercial_weight", 0.25))
    
    # Calculate opportunity scores with naturalness and universal term penalty
    # This activates the perplexity-based quality filtering from metrics.py
    opp = opportunity_scores(
        metrics, 
        intents, 
        business_goals, 
        niche=niche, 
        commercial_weight=commercial_weight,
        documents=cleaned_texts,         # Pass documents for universal term detection
        use_naturalness=True,            # Activate perplexity-based naturalness scoring
        use_universal_penalty=True,      # Penalize boilerplate/navigation terms
    )

    # Assign parent topics for hub-spoke SEO silo architecture
    parent_topic_cfg = cfg_dict.get("parent_topics", {})
    parent_topics: Dict[str, str] = {}
    if parent_topic_cfg.get("enabled", True) and not dry_run:
        parent_topics = assign_parent_topics(
            candidates,
            provider=llm_cfg.get("provider", "auto"),
            model=llm_cfg.get("model"),
            max_topics=int(parent_topic_cfg.get("max_topics", 10)),
        )

    # LLM Verification Gate: Filter hallucinated/nonsensical keywords
    # This catches "Franken-keywords" that passed earlier heuristic filters
    if llm_cfg.get("verify", False) and not dry_run:
        logging.info("Running LLM hallucination verification...")
        # Get top candidates by opportunity score to limit API costs
        top_candidates = sorted(candidates, key=lambda k: opp.get(k, 0), reverse=True)[:200]
        validity_map = verify_candidates_with_llm(
            top_candidates,
            provider=llm_cfg.get("provider", "auto"),
            model=llm_cfg.get("model"),
        )
        # Filter candidates - keep only those that passed LLM verification
        candidates_before = len(candidates)
        candidates = [c for c in candidates if validity_map.get(c, True)]
        filtered = candidates_before - len(candidates)
        if filtered > 0:
            logging.info(f"LLM verification: Removed {filtered} hallucinated keywords")

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
            m = metrics.get(kw, {})
            # Get parent topic (fallback to first word if not assigned)
            pt = parent_topics.get(kw, parent_topics.get(kw.lower(), kw.split()[0] if kw else "general"))
            
            # Get SERP features if available
            serp_features = m.get("serp_features", [])
            
            it = {
                "keyword": kw.lower(),
                "cluster": cname.lower(),
                "parent_topic": pt.lower() if isinstance(pt, str) else str(pt).lower(),
                "intent": intents.get(kw, "informational"),
                "funnel_stage": to_funnel_stage(intents.get(kw, "informational")),
                "search_volume": float(m.get("search_volume", 0.0)),
                "difficulty": float(m.get("difficulty", 0.0)),
                "ctr_potential": float(m.get("ctr_potential", 1.0)),
                "serp_features": serp_features,
                "estimated": bool(m.get("estimated", True)),
                "validated": bool(m.get("validated", False)),
                "opportunity_score": float(min(1.0, max(0.0, opp.get(kw, 0.0)))),
            }
            items.append(it)

    # Global ranking by opportunity_score (then search_volume desc, then keyword asc)
    items = sorted(items, key=lambda it: (-it["opportunity_score"], -it["search_volume"], it["keyword"]))

    # ORYX Quality Gate: Filter unvalidated low-opportunity keywords
    # Keeps validated keywords OR high-opportunity unvalidated ones
    quality_cfg = cfg_dict.get("quality", {})
    min_opportunity_unvalidated = float(quality_cfg.get("min_opportunity_unvalidated", 0.5))
    filter_unvalidated = bool(quality_cfg.get("filter_unvalidated", True))
    
    if filter_unvalidated:
        items_before = len(items)
        items = [
            it for it in items 
            if it["validated"] or it["opportunity_score"] >= min_opportunity_unvalidated
        ]
        filtered_count = items_before - len(items)
        if filtered_count > 0:
            logging.info(f"Quality gate: Removed {filtered_count} unvalidated low-opportunity keywords (threshold: {min_opportunity_unvalidated})")

    # Validate schema
    validate_items(items)

    # Persist outputs (supports .json, .csv, .xlsx based on extension)
    if output:
        write_output(items, output, save_csv)

    return items
