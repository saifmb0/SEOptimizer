# ORYX 🦌 - UAE Keyword Intelligence Engine

> Formerly "Keyword Lab" — Now an enterprise-grade SEO intelligence platform optimized for the Abu Dhabi contracting sector.

A powerful CLI tool that transforms seed topics into validated, ranked SEO keywords optimized for both traditional search and **Generative Engine Optimization (GEO)**. It extracts, validates, clusters, and scores keywords—giving you actionable content strategies.

## ✨ Features

### Core Capabilities
- **Keyword Discovery**: Extract long-tail keywords from documents, URLs, or SERP results
- **Google Autocomplete Validation**: Verify keywords against real search suggestions
- **Semantic Clustering**: Group keywords using Q&A-optimized embeddings
- **Parent Topic Assignment**: Automatic hub-spoke silo architecture via LLM
- **GEO-Centric Intent Classification**: 8 intent categories optimized for AI search

### New in v4.0 - ORYX Enterprise Edition
- 🛡️ **Resilient Scraping** - Tenacity retries with exponential backoff, proxy rotation
- 🧹 **Smart DOM Extraction** - 4-phase noise filtering for legacy UAE sites
- ✅ **Pydantic Validation** - Strict config and schema validation (V2)
- 💾 **Crash Recovery** - SQLite checkpointing for long pipelines
- 🏛️ **Abu Dhabi Entities** - Mussafah, ICAD, KIZAD, Masdar, TAMM, Estidama
- 🔤 **Bilingual Processing** - 80+ Arabic-English construction terms, Gulf dialect
- 🤖 **GEO Module** - Information gain scoring, schema generators, trust signals
- 📊 **Professional Reports** - 8-sheet Excel with charts, conditional formatting
- 📋 **Actionable Recommendations** - Auto-generated content strategy insights

### Previous Releases

<details>
<summary>v3.0 - UAE/Gulf Market Specialization</summary>

- 🇦🇪 **Arabic/English Bilingual Support** - Unicode text preprocessing, Arabic stopwords
- 🌍 **Multilingual Embeddings** - paraphrase-multilingual-MiniLM-L12-v2 for Arabic clustering
- 🔤 **Bilingual Autocomplete** - Fetch suggestions in both English and Arabic
- 🏢 **UAE Entity Extraction** - Emirates, districts, landmarks, free zones
- 💰 **Commercial Intent Scoring** - CPC proxy heuristics for lead-value optimization
- 📉 **SERP Feature CTR Adjustment** - Realistic opportunity scores accounting for featured snippets
- 🎯 **Niche Presets** - Ready-to-use configs for contracting, real estate, legal
- 📋 **GEO Content Briefs** - UAE-specific regulatory requirements and local trust signals
</details>

<details>
<summary>v2.0 - Core Enhancements</summary>

- 🔍 **Google Autocomplete Validation** - Confirm keywords have real search volume
- 📊 **Percentile Ranking** - Fairer scoring that preserves long-tail value
- 🧠 **GEO Intent Categories** - `direct_answer`, `complex_research`, `transactional`, `local`, `comparative`
- 🏗️ **Parent Topic Silos** - Automatic pillar page architecture
- 📝 **Content Brief Generation** - LLM-powered briefs with H1, questions, entities
- ⚠️ **Entity Co-Occurrence** - Detect thin content risk in clusters
- 🧹 **QA Validation** - Automated cleanup (cluster size, word count limits)
- 🚫 **Blacklist Filtering** - Exclude competitor/spam patterns
- 🔍 **PAA Extraction** - People Also Ask questions from SerpAPI
- ⚡ **Weighted HTML Extraction** - Priority extraction of H1-H3, meta, alt text
</details>

## 🏗️ Architecture

```
src/keyword_lab/
├── cli.py           # Typer CLI interface
├── pipeline.py      # Main orchestration
├── config.py        # Pydantic V2 configuration
├── schema.py        # Pydantic V2 data models
├── checkpoint.py    # SQLite crash recovery  [NEW v4.0]
├── scrape.py        # Resilient web scraping  [ENHANCED v4.0]
├── nlp.py           # Text processing & embeddings
├── cluster.py       # K-means clustering
├── llm.py           # LLM integration (Gemini/OpenAI/Anthropic)
├── metrics.py       # Scoring algorithms
├── entities.py      # UAE entity extraction  [EXPANDED v4.0]
├── bilingual.py     # Arabic-English processing  [NEW v4.0]
├── geo.py           # GEO optimization module  [NEW v4.0]
└── io.py            # Professional Excel reports  [ENHANCED v4.0]
```

## 🚀 Quick Start

### Installation
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[ml,excel]"  # Includes ML + Excel exports
python -m nltk.downloader stopwords
```

### Basic Usage
```bash
# Run keyword discovery for UAE contracting
keyword-lab run \
  --seed-topic "villa renovation abu dhabi" \
  --audience "property owners" \
  --geo "ae" \
  --output report.xlsx

# Generate content brief for a cluster
keyword-lab brief keywords.json --cluster "cluster-0"

# Run QA validation
keyword-lab qa keywords.json --min-cluster-size 3 --report
```

### Environment Variables
```bash
# Optional: Enable LLM features
export GEMINI_API_KEY="your-key"      # For Gemini (recommended)
export OPENAI_API_KEY="your-key"      # For OpenAI
export ANTHROPIC_API_KEY="your-key"   # For Anthropic
export SERPAPI_KEY="your-key"         # For PAA extraction
```

## 📋 Output Schema

Each keyword item includes:

| Field | Type | Description |
|-------|------|-------------|
| `keyword` | string | The keyword phrase (lowercase) |
| `cluster` | string | Semantic cluster name |
| `parent_topic` | string | Pillar topic for silo architecture |
| `intent` | enum | Intent category (see below) |
| `funnel_stage` | enum | `TOFU`, `MOFU`, or `BOFU` |
| `search_volume` | 0–1 | Relative search volume (percentile) |
| `difficulty` | 0–1 | Competition difficulty estimate |
| `ctr_potential` | 0–1 | Organic CTR potential (SERP feature adjusted) |
| `serp_features` | array | Detected SERP features affecting CTR |
| `estimated` | boolean | True if metrics are estimated |
| `validated` | boolean | True if confirmed via Autocomplete |
| `opportunity_score` | 0–1 | Combined opportunity metric |
| `geo_suitability` | 0–1 | AI search optimization score |
| `info_gain_score` | 0–1 | Content uniqueness score |
| `uae_entities` | array | Detected UAE entities |
| `emirate` | string | Target emirate |

### Intent Categories

**Traditional SEO:**
- `informational` - Educational queries
- `commercial` - Research/comparison queries
- `transactional` - Purchase intent
- `navigational` - Brand/site navigation

**GEO-Centric (for AI search):**
- `direct_answer` - Simple factual queries (AI snippets)
- `complex_research` - Multi-faceted exploration
- `comparative` - Comparison/recommendation
- `local` - Location-based queries

## 📖 CLI Commands

### `keyword-lab run`
Main pipeline for keyword discovery.

```bash
keyword-lab run \
  --seed-topic "coffee brewing" \
  --audience "home baristas" \
  --geo "us" \
  --language "en" \
  --sources ./docs \
  --config config.yaml \
  --output keywords.json \
  --verbose
```

### `keyword-lab brief`
Generate content briefs from keyword clusters.

```bash
keyword-lab brief keywords.json \
  --cluster "cluster-0" \
  --output brief.md \
  --format markdown
```

### `keyword-lab qa`
Validate and clean pipeline output.

```bash
keyword-lab qa keywords.json \
  --min-cluster-size 3 \
  --max-words 6 \
  --min-opportunity 0.1 \
  --report \
  --dry-run
```

## ⚙️ Configuration

Create a `config.yaml` to customize behavior:

```yaml
# Intent detection rules
intent_rules:
  informational: [how, what, why, guide, tutorial]
  commercial: [best, top, review, compare, vs]
  transactional: [buy, discount, deal, price]

# NLP settings
nlp:
  ngram_min_df: 2
  top_terms_per_doc: 10

# Scraping (v4.0 resilience features)
scrape:
  timeout: 30
  retries: 3
  proxy_enabled: false
  proxy_urls: []
  min_delay_ms: 500
  max_delay_ms: 2000

# Clustering
cluster:
  use_silhouette: true
  max_clusters: 15

# LLM provider
llm:
  provider: auto  # auto, gemini, openai, anthropic, none
  temperature: 0.7

# Geographic targeting (v4.0)
geo:
  region: ae
  locale: en-AE
  primary_emirate: Abu Dhabi
  bilingual: true

# Output
output:
  format: xlsx  # json, csv, xlsx
  include_metadata: true
```

## 🔧 Advanced Features (v4.0)

### Crash Recovery with Checkpoints
Long-running pipelines automatically checkpoint to SQLite:
```python
from keyword_lab.checkpoint import Checkpoint, CheckpointedPipeline

# Automatic checkpointing with context manager
with CheckpointedPipeline("run_20240115_001") as cp:
    if not cp.has_stage("scrape"):
        data = scrape_serps(keywords)
        cp.save_stage("scrape", data, {"urls": len(urls)})
    else:
        data = cp.load_stage("scrape")  # Resume from checkpoint

# List and cleanup old runs
runs = Checkpoint.list_runs()
Checkpoint.cleanup_old_runs(keep_last=10)
```

### Abu Dhabi Entity Extraction
Comprehensive UAE entity recognition for the contracting sector:
```python
from keyword_lab.entities import extract_entities

entities = extract_entities("villa renovation mussafah abu dhabi", geo="ae")
# Returns:
# {
#     "emirate": "Abu Dhabi",
#     "district": "Mussafah",
#     "is_local": True,
#     "is_contracting": True,
#     "location_type": "industrial",
#     "government_entity": None,
#     "certification": None,
# }
```

**Supported Abu Dhabi entities:**
- **Districts**: Mussafah, ICAD 1-3, Khalifa City, MBZ City, Al Raha, Saadiyat
- **Free Zones**: KIZAD, Masdar, ADGM, twofour54, Hub71
- **Government**: TAMM, ADM, ADDED, DMT, Musanada
- **Certifications**: Estidama, Pearl Rating (1-5), PVRS, PCRS
- **Legal Terms**: Musataha, Tawtheeq, NOC

### Bilingual Arabic-English Processing
```python
from keyword_lab.bilingual import (
    expand_bilingual,
    get_arabic_equivalent,
    analyze_bilingual_query,
)

# Get Arabic translation
arabic = get_arabic_equivalent("renovation")  # Returns: "تجديد"

# Expand keyword with bilingual variants
variants = expand_bilingual("villa renovation abu dhabi")
# Returns: ["villa renovation abu dhabi", "فيلا تجديد أبوظبي", ...]

# Analyze query language and content
analysis = analyze_bilingual_query("best contractor abu dhabi")
# Returns intent, language detection, construction terms found
```

### GEO (Generative Engine Optimization)
Optimize content for AI-powered search engines:
```python
from keyword_lab.geo import (
    calculate_geo_suitability,
    calculate_information_gain,
    analyze_trust_signals,
    generate_faq_schema,
    generate_local_business_schema,
)

# Score query suitability for AI search
score = calculate_geo_suitability("how much does villa renovation cost in abu dhabi")
# Returns: 0.75 (high AI answer likelihood)

# Calculate content uniqueness vs competitors
gain = calculate_information_gain(your_content, competitor_contents)
# Returns unique concepts, missing topics, recommendations

# Audit E-E-A-T trust signals
trust = analyze_trust_signals(page_content)
# Returns experience, expertise, authority, trust scores

# Generate structured data
faq_schema = generate_faq_schema([
    ("How much does renovation cost?", "Costs range from 50-200 AED/sqft..."),
    ("Do I need a permit?", "Yes, ADM building permit required..."),
])
business_schema = generate_local_business_schema(
    name="ABC Contracting LLC",
    address={"addressLocality": "Abu Dhabi", "addressRegion": "Abu Dhabi"},
    services=["Villa Renovation", "Fit Out", "MEP"],
)
```

### Professional Excel Reports
Generate stakeholder-ready reports with 8 analysis sheets:
```python
from keyword_lab.io import write_excel

write_excel(
    items=keyword_results,
    xlsx_path="report.xlsx",
    geo="ae",
    report_title="ORYX Q1 Keyword Intelligence Report",
    include_charts=True,
    include_executive_summary=True,
)
```

**Report Sheets:**
1. **Executive Summary** - Key metrics, intent distribution, top clusters
2. **Keywords** - Full data with conditional formatting (color scales)
3. **Cluster Analysis** - Cluster performance with bar charts
4. **Intent Breakdown** - Intent distribution with pie chart
5. **Priority Matrix** - High-opportunity, low-difficulty keywords
6. **GEO Analysis** - AI search optimization scores
7. **Location Analysis** - UAE emirate breakdown
8. **Recommendations** - Auto-generated actionable insights

### Google Autocomplete Validation
Keywords are validated against Google's autocomplete API to confirm real search demand:
```python
from keyword_lab.scrape import validate_keywords_with_autocomplete

validated = validate_keywords_with_autocomplete(
    keywords=["best coffee beans", "how to brew coffee"],
    language="en",
    country="us"
)
# Returns: {"best coffee beans": True, "how to brew coffee": True}
```

### Entity Co-Occurrence Analysis
Detect clusters at risk of thin content:
```python
from keyword_lab.cluster import get_cluster_entity_summary

analysis, healthy, at_risk = get_cluster_entity_summary(clusters)
for cluster in at_risk:
    print(f"⚠️ {cluster}: Thin content risk")
```

### Content Brief Generation
Generate LLM-powered content briefs:
```python
from keyword_lab.cli import generate_brief_with_llm

brief = generate_brief_with_llm(
    cluster_name="coffee-brewing",
    keywords=["how to brew coffee", "best coffee maker"],
    intents={"how to brew coffee": "informational"}
)
```

## 🇦🇪 UAE/Gulf Market Specialization

Keyword Lab v3.0 includes specialized features for the UAE and Gulf markets.

### Using Niche Presets

```bash
# Use the UAE contracting preset
keyword-lab run \
  --seed-topic "villa renovation dubai" \
  --audience "property owners in UAE" \
  --preset presets/contracting_ae.yaml \
  --output keywords.json
```

### GEO-Enhanced Content Briefs

Generate briefs with UAE-specific requirements:

```bash
# Generate a GEO brief for UAE contracting market
keyword-lab geo-brief keywords.json \
  --geo ae \
  --niche contracting \
  --output brief.md
```

The GEO brief includes:
- Geographic entity analysis (Emirates, districts)
- UAE regulatory requirements checklist
- Local trust signals recommendations
- Location variant suggestions

### Entity Extraction

Extract UAE geographic entities from keywords:

```python
from keyword_lab.entities import extract_entities

entities = extract_entities("villa renovation dubai marina", geo="ae")
# Returns: {
#   "emirate": "Dubai",
#   "district": "Dubai Marina",
#   "is_local": True,
#   "location_type": "residential"
# }
```

### Commercial Intent Scoring

Score keywords for lead generation value:

```python
from keyword_lab.metrics import commercial_value

score = commercial_value(
    "villa fit out contractors quote",
    intent="transactional",
    geo="ae",
    niche="contracting"
)
# Returns: 0.85 (high commercial value)
```

### Content Gap Analysis

Identify content opportunities:

```python
from keyword_lab.content_gap import analyze_content_gaps

gaps = analyze_content_gaps(
    target_keywords=["villa renovation dubai", "office fit out abu dhabi"],
    existing_content=[{"url": "...", "keywords": [...]}]
)
```

### UAE Districts & Free Zones

The entity extraction includes:
- **7 Emirates**: Dubai, Abu Dhabi, Sharjah, Ajman, RAK, Fujairah, UAQ
- **25+ Districts**: Dubai Marina, Business Bay, JBR, Al Barsha, etc.
- **Landmarks**: Burj Khalifa, Dubai Mall, Mall of the Emirates
- **Free Zones**: JAFZA, DMCC, DAFZA, TECOM, KIZAD

## 🏗️ Architecture

```
keyword_lab/
├── cli.py          # Typer CLI (run, brief, qa commands)
├── pipeline.py     # Main orchestration
├── scrape.py       # Document acquisition, Autocomplete, PAA
├── nlp.py          # Candidate generation, text processing
├── cluster.py      # Semantic clustering, entity analysis
├── metrics.py      # Percentile ranking, opportunity scoring
├── llm.py          # LLM integration, intent classification
├── qa.py           # Quality assurance validation
├── schema.py       # JSON Schema validation
├── io.py           # Output handling (JSON, CSV, XLSX)
└── config.py       # Configuration management
```

## 📊 Pipeline Flow

```
1. Acquire Documents
   └─> Local files, URLs, SERP results

2. Generate Candidates
   └─> TF-IDF extraction, question generation, LLM expansion

3. Filter & Validate
   └─> Blacklist, length limits, Autocomplete validation

4. Cluster & Classify
   └─> Semantic clustering, intent classification, parent topics

5. Score & Rank
   └─> Percentile ranking, opportunity scoring

6. Output & QA
   └─> JSON/CSV export, QA validation, content briefs
```

## ⚠️ Limitations & Ethics

- **No direct SERP scraping**: Uses APIs and respects robots.txt
- **Estimated metrics**: Marked with `estimated: true` unless from trusted sources
- **Rate limiting**: Built-in delays for Autocomplete validation
- **Language**: English-first for NLP, but runs for other languages

## 📚 Documentation

- [QuickStart Guide](./quickstart.md) - Detailed setup instructions
- [Configuration Reference](./config.sample.yaml) - All configuration options

## 📜 License

MIT License - See LICENSE file for details.

---

## ✨ New in v2.0

### Validation & Quality
- **Google Autocomplete Validation** - Confirm keywords have real search volume by checking against Google's autocomplete suggestions
- **Percentile Ranking** - Fairer scoring using percentile ranking that preserves long-tail keyword value
- **QA Validation Command** - `keyword-lab qa` to clean output (remove small clusters, long keywords)
- **Entity Co-Occurrence Analysis** - Detect thin content risk by analyzing entity density in clusters

### GEO (Generative Engine Optimization)
- **GEO Intent Categories** - 5 AI-search-optimized intents: `direct_answer`, `complex_research`, `transactional`, `local`, `comparative`
- **Parent Topic Silos** - Automatic hub-spoke architecture with LLM-assigned pillar topics
- **Content Brief Generation** - `keyword-lab brief` command for LLM-powered content briefs

### Data Quality
- **Blacklist Filtering** - Exclude competitor/navigational/spam patterns via config
- **PAA Extraction** - Extract "People Also Ask" questions from SerpAPI
- **Weighted HTML Extraction** - Priority extraction of H1-H3, meta descriptions, alt text

### New Output Fields
The output schema now includes:
- `parent_topic` - Pillar topic for silo architecture  
- `validated` - Boolean indicating Autocomplete confirmation

## Additional CLI Commands

### `keyword-lab brief`
Generate content briefs from keyword clusters:
```bash
keyword-lab brief keywords.json --cluster "cluster-0" --output brief.md
```

### `keyword-lab qa`
Validate and clean pipeline output:
```bash
keyword-lab qa keywords.json --min-cluster-size 3 --max-words 6 --report
```
