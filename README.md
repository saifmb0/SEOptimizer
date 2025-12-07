# Keyword Lab (Presentation Overview)

A simple CLI that turns a seed topic into a ranked list of long‑tail SEO keywords. It reads content, expands ideas, groups similar intents, estimates basic metrics (all on a 0–1 scale), and outputs a compact JSON list you can use immediately.

## What it does (in 1 minute)
- Start with a seed topic (e.g., "best coffee beans").
- Optionally point it at a folder of `.txt/.md` notes or URLs.
- It generates keyword candidates (bigrams/trigrams and questions), expands with a lightweight LLM (Gemini, optional), clusters by similarity/intent, and scores them.
- You get a clean JSON array of ranked keyword ideas you can analyze or feed into dashboards.

## How it works (at a glance)
1) Acquire text
- Local notes or URLs (visible text only, robots.txt respected best‑effort).
- Optional: expand with Gemini (if `GEMINI_API_KEY` is set in `.env`).

2) Generate candidates
- Clean text, extract bigrams/trigrams with TF‑IDF & counts, auto-generate question forms, plus seed‑based expansions.

3) Cluster & tag intent
- Encode keywords and cluster (SBERT if available, else TF‑IDF). Tag intent with simple rules (informational/commercial/transactional/navigational).

4) Score (0–1 scale)
- search_volume: normalized signal from prominence and question boost.
- difficulty: normalized signal from length/head‑term heuristics.
- opportunity_score: search_volume × (1 − difficulty) × business relevance.

5) Output
- A compact JSON array with one object per keyword, ranked by opportunity_score. Also saved to disk; CLI prints the same formatted JSON for convenience.

## What you get
Each item (all lowercase keyword/cluster):

- keyword: string
- cluster: string
- intent: informational | commercial | transactional | navigational
- funnel_stage: TOFU | MOFU | BOFU
- search_volume: 0–1 (estimated)
- difficulty: 0–1 (estimated)
- estimated: boolean
- opportunity_score: 0–1

Example (snippet):
[
  {"keyword": "best coffee beans", "cluster": "cluster-1", "intent": "commercial", "funnel_stage": "MOFU", "search_volume": 0.72, "difficulty": 0.41, "estimated": true, "opportunity_score": 0.42},
  {"keyword": "how to dial in espresso", "cluster": "cluster-0", "intent": "informational", "funnel_stage": "TOFU", "search_volume": 0.66, "difficulty": 0.25, "estimated": true, "opportunity_score": 0.50}
]

## 3‑step demo
1) Create a virtual environment and install:
- python -m venv .venv && source .venv/bin/activate
- pip install -r requirements.txt
- python -m nltk.downloader stopwords

2) (Optional) Enable better expansions:
- cp .env.example .env
- Add your GEMINI_API_KEY to `.env`.

3) Run:
- python -m keyword_lab.cli run --seed-topic "best coffee beans" --audience "home baristas" --output keywords.json
- The CLI also prints the same formatted JSON to stdout for a quick look.

## Talking points (why this is useful)
- Long‑tail focus: Generates bigrams/trigrams and natural questions aligned to searcher intent.
- Clusters by intent: Helps you plan pillar pages vs. supporting content.
- Simple, transparent scoring: All metrics are 0–1 and marked estimated when inferred.
- Clean output: Minimal JSON array, easy to version, diff, and analyze.
- Fast iterations: Good defaults; bring your own notes or URLs; optional LLM expansion.

## Limitations & Ethics
- No paid SERP scraping; respects robots.txt best‑effort; avoids scraping Google HTML directly.
- Metrics are heuristic (estimated=true) unless you integrate a trusted provider.
- Language support is English‑first for stopwords, but it still runs for other languages.

## Need setup details?
See QuickStart with installation, options, config, and troubleshooting:
- QuickStart: ./quickstart.md
