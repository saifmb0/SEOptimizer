# Keyword Lab

A compact Python CLI to discover and cluster long‑tail keywords, estimate basic SEO metrics, and output a compact JSON array. Logs go to stderr, JSON to stdout only when `--output "-"` is used.

## Quick start

- python -m venv .venv && source .venv/bin/activate
- pip install -r requirements.txt
- python -m nltk.downloader stopwords
- cp .env.example .env
- python -m keyword_lab.cli run --seed-topic "best coffee beans" --audience "home baristas" --geo "US" --language "en" --max-clusters 8 --max-keywords-per-cluster 12 --output keywords.json
- cat keywords.json | jq length

## Run tests
- pip install -r requirements.txt
- python -m pytest

## Workflow overview

1) Inputs and configuration
- CLI flags, YAML config (`--config`), and environment variables are merged. Precedence: CLI > config.yaml > defaults. API keys come from `.env` (SERPAPI_KEY, BING_API_KEY).

2) Acquisition (`src/keyword_lab/scrape.py`)
- If `--sources` is a directory: read `.txt/.md` files as documents.
- If `--sources` is a file: treat each line as a URL; fetch their HTML (visible text from h1–h3, p, li). Scripts/styles removed.
- Optional provider results when `--provider` is `serpapi` or `bing` and keys are set: query top results, then fetch each URL.
- Best‑effort robots.txt respected with a custom User‑Agent and retries/timeouts.

3) NLP candidate generation (`src/keyword_lab/nlp.py`)
- Clean text: lowercase, strip punctuation/digits, normalize whitespace.
- Tokenization + English stopword removal (NLTK).
- N‑grams: `CountVectorizer(ngram_range=(2,3), min_df=config.nlp.ngram_min_df)` to get bigram/trigram counts.
- Questions: prefix common intents (how/what/best/vs/for/near me/… ) onto top phrases to produce long‑tails.
- TF‑IDF: per‑document bigram/trigram top terms.
- Merge + de‑dupe to a lowercase candidate keyword list.

4) Clustering and intent (`src/keyword_lab/cluster.py`)
- Vectorize keywords with Sentence‑Transformers (MiniLM) if available, else TF‑IDF.
- KMeans with deterministic `random_state`; k chosen to target 6–12 when volume allows; graceful fallback for few keywords.
- Intent tagging (rule‑based): informational/commercial/transactional/navigational; competitors make navigational intent when matched.

5) Metrics and scoring (`src/keyword_lab/metrics.py`)
- search_volume: heuristic from n‑gram frequency, TF‑IDF prominence, and question boost; scaled ~10–1000. Mark `estimated=true` when provider is `none`.
- difficulty: provider `total_results` if available (log‑mapped), else heuristic based on length/head terms.
- opportunity_score: normalized volume divided by (difficulty+1) times business relevance derived from goals and intent.

6) Selection, validation, and output (`src/keyword_lab/pipeline.py`)
- Rank within each cluster by `opportunity_score` and select up to `--max-keywords-per-cluster`.
- Map intent to funnel stage: informational→TOFU, transactional→BOFU, else MOFU.
- Validate against the compact JSON Schema (`src/keyword_lab/schema.py`).
- Persist the exact JSON array to `--output` (default `keywords.json`). If `--output "-"`, print the JSON array to stdout only.
- Optional CSV mirror with `--save-csv`.

## Disciplined I/O and logging
- All logs go to stderr via Python `logging`.
- Stdout is reserved for the compact JSON array only when `--output "-"` is provided.
- When writing to a file, nothing is printed to stdout.

## Configuration
- Use `config.sample.yaml` as a template. Fields include scraping timeouts/retries, n‑gram `min_df`, top TF‑IDF terms per doc, target cluster counts, etc.
- Environment variables in `.env`:
  - `SERPAPI_KEY`, `BING_API_KEY` (optional)
  - `USER_AGENT` (optional)

## Compact output schema
Each element of the JSON array has exactly these fields (lowercase keyword/cluster):

- keyword: string
- cluster: string
- intent: one of informational|commercial|transactional|navigational
- funnel_stage: TOFU|MOFU|BOFU
- search_volume: integer >= 0
- difficulty: integer 0–100
- estimated: boolean
- opportunity_score: number

See `src/keyword_lab/schema.py` for the JSON Schema used in validation.

## Usage

- Print JSON to stdout only:
  - python -m keyword_lab.cli run --seed-topic "best coffee beans" --audience "home baristas" --output "-"
- Save to file (default `keywords.json`):
  - python -m keyword_lab.cli run --seed-topic "best coffee beans" --audience "home baristas" --output keywords.json
- Also save CSV:
  - python -m keyword_lab.cli run --seed-topic "best coffee beans" --audience "home baristas" --save-csv keywords.csv
- Use sources (file of URLs or directory of .txt/.md files):
  - python -m keyword_lab.cli run --seed-topic "espresso" --sources tests/data

## Providers (optional)
This tool does not scrape Google HTML directly. For SERP data, configure SerpAPI or Bing via environment variables and set `--provider` accordingly.

## Language support
If `--language` != `en`, the pipeline still runs using English stopwords as a first pass.

## Troubleshooting
- Ensure NLTK stopwords downloaded: `python -m nltk.downloader stopwords`.
- If `sentence-transformers` is unavailable, TF‑IDF fallback is automatic.
- Respect robots.txt best‑effort; set a custom `USER_AGENT` in `.env` if needed.
