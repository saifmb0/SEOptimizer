# QuickStart

This guide contains the technical details you need to install, configure, and run Keyword Lab.

## 1) Setup

- Python 3.9+
- Create and activate a virtual environment:
  - python -m venv .venv && source .venv/bin/activate
- Install requirements:
  - pip install -r requirements.txt
- Download NLTK stopwords:
  - python -m nltk.downloader stopwords
- Optional: copy environment file and add keys
  - cp .env.example .env
  - Add GEMINI_API_KEY=<your_key> (optional)

## 2) Run

- Save JSON to file and also print formatted to stdout:
  - python -m keyword_lab.cli run --seed-topic "best coffee beans" --audience "home baristas" --output keywords.json
- Print to stdout only (formatted):
  - python -m keyword_lab.cli run --seed-topic "best coffee beans" --audience "home baristas" --output "-"
- Use local sources (folder of .txt/.md):
  - python -m keyword_lab.cli run --seed-topic "espresso" --audience "home baristas" --sources tests/data --output keywords.json

## 3) Configuration

- CLI flags override config YAML and defaults.
- If present, ./config.yaml is read by default, or pass --config path.
- See config.sample.yaml for all options (timeouts, retries, n-gram min_df, etc.).
- Environment variables in .env:
  - GEMINI_API_KEY (optional) – enables LLM expansions
  - USER_AGENT (optional)

## 4) Output

- JSON array of objects with fields:
  - keyword, cluster, intent, funnel_stage, search_volume (0–1), difficulty (0–1), estimated (bool), opportunity_score (0–1)
- CSV mirror when --save-csv is provided.

## 5) Tests

- Run tests:
  - pytest

## 6) Troubleshooting

- Module not found errors: confirm you’re running from the project root and venv is active.
- NLTK stopwords missing: run "python -m nltk.downloader stopwords".
- Gemini not expanding: ensure GEMINI_API_KEY is set and internet access is available.
- Few/no keywords: add sources, reduce nlp.ngram_min_df in config, or rely on seed/LLM expansions.
