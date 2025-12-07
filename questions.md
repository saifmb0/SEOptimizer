# Presenter Q&A: Keyword Lab

Use this to answer common questions during your demo.

## What is this tool?
A CLI that expands a seed topic into long‑tail keywords, clusters them by intent, estimates simple SEO metrics (0–1 scale), ranks them, and outputs a compact JSON array.

## What does each number mean?
- search_volume (0–1): Relative interest. 0 is lowest in the set, 1 is highest. It’s a normalized proxy, not exact searches.
- difficulty (0–1): Relative competition. 0 is easiest, 1 is hardest within this run.
- opportunity_score (0–1): Overall attractiveness = search_volume × (1 − difficulty) × business_relevance.
- estimated (true/false): true means metrics are inferred heuristics (no paid SERP data used).

## How did you estimate search_volume?
- We compute a raw popularity proxy from:
  - N‑gram frequency across the input texts
  - TF‑IDF prominence (phrases that stand out in documents)
  - Question boost (how/what/why/how to, etc.)
  - Slight boost for longer long‑tails (≥ bigrams)
- Then we min‑max normalize across all candidates to 0–1 so results are comparable.

## How did you estimate difficulty?
- Heuristic proxy that increases with:
  - Short head terms (≤2 words)
  - Presence of competitive modifiers (best/top/review/compare/pricing)
- Optionally, if a reliable total_results‑style signal were available (not used here), we’d map its log scale and normalize to 0–1.

## What is opportunity_score exactly?
- opportunity_score = search_volume × (1 − difficulty) × business_relevance.
- business_relevance (0.6–1.0) depends on your goals vs. the keyword’s intent (e.g., sales‑leaning goals slightly favor transactional/commercial intents).

## How do you generate long‑tail keywords?
- From your content: bigrams/trigrams via CountVectorizer + TF‑IDF top phrases.
- Question variants: prefixes like how/what/best/vs/for/near me/beginner/advanced/guide/checklist/template.
- Seed‑based expansions: templated patterns derived from the seed + audience.
- Optional LLM expansion (Gemini): generates additional long‑tail ideas/questions (requires `GEMINI_API_KEY`).

## How do you cluster and detect intent?
- Vectorization: SBERT (MiniLM) if available; otherwise TF‑IDF.
- Clustering: KMeans with deterministic random_state; target 6–12 clusters when enough keywords.
- Intent (rule‑based):
  - informational: who/what/why/how/guide/tutorial/tips/checklist/template
  - commercial: best/top/review/compare/vs/alternatives/pricing
  - transactional: buy/discount/coupon/deal/near me
  - navigational: contains competitor/brand tokens (if provided)
- Funnel stage mapping: informational→TOFU, transactional→BOFU, else MOFU.

## Why are keyword and cluster lowercase?
- To keep the output compact/consistent and simplify downstream processing.

## Why is estimated mostly true?
- Because we avoid paid providers and do not use scraped SERP counts. All metrics are inferred from your content and model signals.

## Do you scrape Google directly?
- No. We avoid scraping Google HTML directly and respect robots.txt best‑effort for any URLs you provide.

## How is the list ranked?
- Globally by opportunity_score (desc), then search_volume (desc), then keyword (asc). Within each cluster we also prioritize higher scores.

## What if I get too few keywords?
- Add more source documents (`--sources folder` with .txt/.md),
- Lower `nlp.ngram_min_df` in config,
- Provide `GEMINI_API_KEY` for LLM expansions,
- Include an audience string to enrich seed‑based expansions.

## Can I save CSV?
- Yes, add `--save-csv path.csv`. Same fields as JSON.

## Where do logs and JSON go?
- Logs to stderr. JSON is saved to the file in `--output` and also printed formatted to stdout for convenience.

## Is it deterministic?
- We set `random_state` for clustering to improve reproducibility. Results can still vary if LLM expansions are enabled.

## Can I change the scoring or add features?
- Yes. You can adjust business relevance, add spaCy patterns, switch clustering, or integrate trusted providers. The schema and tests keep outputs consistent.
