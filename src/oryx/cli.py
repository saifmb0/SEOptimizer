import json
import logging
import sys
from pathlib import Path
from typing import Optional, List, Dict

import typer
import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.panel import Panel

from .pipeline import run_pipeline
from .config import load_config, ConfigValidationError, config_to_dict, KeywordLabConfig

# Initialize rich console
console = Console()
err_console = Console(stderr=True)

app = typer.Typer(
    add_completion=False, 
    no_args_is_help=False,  # Changed for interactive mode
    help="🦌 ORYX - Discover and cluster SEO keywords"
)


def setup_logging(verbose: bool):
    """Setup logging with rich handler for pretty output."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=err_console, rich_tracebacks=True, show_path=verbose)]
    )


def display_results_table(items: List[dict], max_rows: int = 25):
    """Display results in a pretty ASCII table."""
    if not items:
        console.print("[yellow]No keywords found.[/yellow]")
        return
    
    table = Table(
        title="🔑 Keyword Results",
        show_header=True,
        header_style="bold magenta",
        border_style="dim",
    )
    
    table.add_column("Keyword", style="cyan", no_wrap=False, max_width=40)
    table.add_column("Cluster", style="green")
    table.add_column("Intent", style="yellow")
    table.add_column("Funnel", style="blue")
    table.add_column("Interest", justify="right", style="white")
    table.add_column("Diff", justify="right", style="white")
    table.add_column("Opp", justify="right", style="bold green")
    
    for item in items[:max_rows]:
        table.add_row(
            item["keyword"],
            item["cluster"],
            item["intent"],
            item["funnel_stage"],
            f"{item['relative_interest']:.2f}",
            f"{item['difficulty']:.2f}",
            f"{item['opportunity_score']:.2f}",
        )
    
    if len(items) > max_rows:
        table.add_row(
            f"[dim]... and {len(items) - max_rows} more[/dim]",
            "", "", "", "", "", ""
        )
    
    console.print(table)
    console.print(f"\n[dim]Total: {len(items)} keywords[/dim]")


@app.command()
def run(
    seed_topic: Optional[str] = typer.Option(None, "--seed-topic", help="Seed topic"),
    audience: Optional[str] = typer.Option(None, "--audience", help="Audience"),
    geo: str = typer.Option("global", "--geo"),
    language: str = typer.Option("en", "--language"),
    competitors: str = typer.Option("", "--competitors", help="Comma-separated domains"),
    business_goals: str = typer.Option("traffic, leads", "--business-goals"),
    capabilities: str = typer.Option("no-paid-apis", "--capabilities"),
    time_horizon: str = typer.Option("quarter", "--time-horizon"),
    max_clusters: int = typer.Option(8, "--max-clusters"),
    max_keywords_per_cluster: int = typer.Option(12, "--max-keywords-per-cluster"),
    sources: Optional[str] = typer.Option(None, "--sources"),
    query: Optional[str] = typer.Option(None, "--query"),
    provider: str = typer.Option("none", "--provider", case_sensitive=False),
    output: str = typer.Option("keywords.json", "--output", help="Path to JSON/CSV/XLSX file or '-' for stdout"),
    save_csv: Optional[str] = typer.Option(None, "--save-csv"),
    verbose: bool = typer.Option(False, "--verbose", help="DEBUG logs to stderr"),
    config_path: Optional[str] = typer.Option(None, "--config", help="YAML config (default: ./config.yaml if present)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Log steps without network calls"),
    table_output: bool = typer.Option(False, "--table", help="Display results as a table instead of JSON"),
    preset: Optional[str] = typer.Option(None, "--preset", help="Niche preset file (e.g., presets/contracting_ae.yaml)"),
    niche: Optional[str] = typer.Option(None, "--niche", help="Niche for commercial scoring (contracting, real_estate, legal)"),
):
    """
    Run the keyword discovery pipeline.
    
    If --seed-topic or --audience are not provided, you'll be prompted interactively.
    """
    setup_logging(verbose)

    # Interactive mode: prompt for required fields if not provided
    if seed_topic is None:
        seed_topic = typer.prompt("🌱 Enter seed topic")
    if audience is None:
        audience = typer.prompt("👥 Enter target audience")

    # Load config with defaults and validation
    try:
        cfg = load_config(config_path, preset_path=preset)
    except ConfigValidationError as e:
        err_console.print(f"[bold red]Configuration Error:[/bold red]")
        for error in e.errors:
            err_console.print(f"  [red]•[/red] {error}")
        err_console.print("\n[dim]Check your config.yaml for typos or invalid values.[/dim]")
        sys.exit(2)

    # Convert to dict for easy attribute access (handles both Pydantic and legacy)
    cfg_dict = config_to_dict(cfg)
    
    # Apply preset values ONLY when CLI didn't provide a value
    # CLI arguments take precedence over preset values
    if preset and cfg:
        # Access Pydantic model attributes or dict keys
        # geo: check if it's still the default "global"
        if geo == "global":
            geo = cfg_dict.get("geo", {}).get("region", geo) if isinstance(cfg_dict.get("geo"), dict) else cfg_dict.get("geo", geo)
        # language: check if it's still the default "en"
        if language == "en":
            preset_lang = getattr(cfg, "language", None) or cfg_dict.get("language")
            if preset_lang:
                language = preset_lang
        # audience: was already set via CLI or prompt, only override if user didn't provide
        # Since we prompt for audience if None, we track if it came from CLI
        # by checking if the preset has a different value (keep CLI value)
        # Actually, CLI args always take precedence - don't override audience
        # business_goals: check if it's still the default
        if business_goals == "traffic, leads":
            business_goals = cfg_dict.get("business_goals", business_goals)
        max_clusters = cfg_dict.get("cluster", {}).get("max_clusters", max_clusters) if isinstance(cfg_dict.get("cluster"), dict) else cfg_dict.get("max_clusters", max_clusters)
        max_keywords_per_cluster = cfg_dict.get("cluster", {}).get("max_keywords_per_cluster", max_keywords_per_cluster) if isinstance(cfg_dict.get("cluster"), dict) else cfg_dict.get("max_keywords_per_cluster", max_keywords_per_cluster)
        
    # Get niche from preset or CLI
    effective_niche = niche or cfg_dict.get("niche") if cfg else niche

    comp_list: List[str] = [c.strip() for c in competitors.split(",") if c.strip()]

    # Show a panel with run configuration
    if not verbose:
        niche_display = f" | [bold]Niche:[/bold] {effective_niche}" if effective_niche else ""
        console.print(Panel(
            f"[bold]Seed:[/bold] {seed_topic}\n"
            f"[bold]Audience:[/bold] {audience}\n"
            f"[bold]Geo:[/bold] {geo} | [bold]Language:[/bold] {language}{niche_display}",
            title="🔬 Keyword Lab",
            border_style="blue",
        ))

    items = run_pipeline(
        seed_topic=seed_topic,
        audience=audience,
        geo=geo,
        language=language,
        competitors=comp_list,
        business_goals=business_goals,
        capabilities=capabilities,
        time_horizon=time_horizon,
        max_clusters=max_clusters,
        max_keywords_per_cluster=max_keywords_per_cluster,
        sources=sources,
        query=query or seed_topic,
        provider=provider,
        output=output,
        save_csv=save_csv,
        verbose=verbose,
        config=cfg,
        dry_run=dry_run,
        niche=effective_niche,
    )

    # Output display
    if output == "-":
        # Already printed by io.write_json
        pass
    elif table_output:
        display_results_table(items)
    else:
        # Print formatted JSON to stdout
        console.print_json(json.dumps(items, ensure_ascii=False))
    
    # Summary
    if items:
        console.print(f"\n[green]✓[/green] Generated {len(items)} keywords in {len(set(i['cluster'] for i in items))} clusters")
        if output != "-":
            console.print(f"[dim]Saved to: {output}[/dim]")
    else:
        console.print("[yellow]⚠ No keywords generated. Try adding more sources or adjusting settings.[/yellow]")
        sys.exit(1)


@app.command()
def brief(
    input_file: str = typer.Argument(..., help="Path to keywords JSON file"),
    cluster: Optional[str] = typer.Option(None, "--cluster", "-c", help="Specific cluster to generate brief for"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output markdown file (default: stdout)"),
    provider: str = typer.Option("auto", "--provider", help="LLM provider for brief generation"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """
    Generate a content brief from clustered keywords.
    
    Creates an SEO-optimized content brief with:
    - Suggested H1 title
    - Primary intent analysis
    - Key questions to answer
    - Entity/topic recommendations
    
    Example:
        keyword-lab brief keywords.json --cluster cluster-0 -o brief.md
    """
    from .llm import _detect_provider, HAS_GENAI, HAS_LITELLM
    
    setup_logging(verbose)
    
    # Load keywords file
    try:
        with open(input_file, 'r') as f:
            items = json.load(f)
    except Exception as e:
        err_console.print(f"[red]Error loading {input_file}: {e}[/red]")
        sys.exit(1)
    
    if not items:
        err_console.print("[yellow]No keywords found in file.[/yellow]")
        sys.exit(1)
    
    # Filter by cluster if specified
    if cluster:
        items = [i for i in items if i.get("cluster", "").lower() == cluster.lower()]
        if not items:
            err_console.print(f"[yellow]No keywords found in cluster '{cluster}'[/yellow]")
            sys.exit(1)
    
    # Group by cluster
    clusters_data: Dict[str, List[dict]] = {}
    for item in items:
        c = item.get("cluster", "unknown")
        if c not in clusters_data:
            clusters_data[c] = []
        clusters_data[c].append(item)
    
    # Generate briefs
    briefs = []
    for cluster_name, keywords in clusters_data.items():
        brief_md = _generate_content_brief(cluster_name, keywords, provider)
        briefs.append(brief_md)
    
    full_output = "\n\n---\n\n".join(briefs)
    
    if output:
        Path(output).write_text(full_output)
        console.print(f"[green]✓[/green] Saved brief to {output}")
    else:
        console.print(full_output)


@app.command("geo-brief")
def geo_brief(
    input_file: str = typer.Argument(..., help="Path to keywords JSON file"),
    cluster: Optional[str] = typer.Option(None, "--cluster", "-c", help="Specific cluster to generate brief for"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output markdown file (default: stdout)"),
    geo: str = typer.Option("ae", "--geo", "-g", help="Target geography (ae, sa, qa, etc.)"),
    niche: Optional[str] = typer.Option(None, "--niche", "-n", help="Niche/vertical (contracting, real_estate, etc.)"),
    provider: str = typer.Option("auto", "--provider", help="LLM provider for brief generation"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """
    Generate a GEO-enhanced content brief for UAE/Gulf markets.
    
    Creates a locale-specific SEO content brief with:
    - Geographic entity analysis (Emirates, districts, landmarks)
    - UAE-specific regulatory requirements
    - Local trust signals and schema recommendations
    - Location-based keyword variations
    
    Example:
        keyword-lab geo-brief keywords.json --geo ae --niche contracting -o brief.md
    """
    from .llm import _detect_provider
    
    setup_logging(verbose)
    
    # Load keywords file
    try:
        with open(input_file, 'r') as f:
            items = json.load(f)
    except Exception as e:
        err_console.print(f"[red]Error loading {input_file}: {e}[/red]")
        sys.exit(1)
    
    if not items:
        err_console.print("[yellow]No keywords found in file.[/yellow]")
        sys.exit(1)
    
    # Filter by cluster if specified
    if cluster:
        items = [i for i in items if i.get("cluster", "").lower() == cluster.lower()]
        if not items:
            err_console.print(f"[yellow]No keywords found in cluster '{cluster}'[/yellow]")
            sys.exit(1)
    
    # Group by cluster
    clusters_data: Dict[str, List[dict]] = {}
    for item in items:
        c = item.get("cluster", "unknown")
        if c not in clusters_data:
            clusters_data[c] = []
        clusters_data[c].append(item)
    
    # Show GEO targeting info
    console.print(Panel(
        f"[bold]Geo:[/bold] {geo.upper()}\n"
        f"[bold]Niche:[/bold] {niche or 'general'}\n"
        f"[bold]Clusters:[/bold] {len(clusters_data)}",
        title="🌍 GEO Brief Generator",
        border_style="green",
    ))
    
    # Generate GEO-enhanced briefs
    briefs = []
    for cluster_name, keywords in clusters_data.items():
        brief_md = _generate_geo_brief(cluster_name, keywords, provider, geo, niche)
        briefs.append(brief_md)
    
    full_output = "\n\n---\n\n".join(briefs)
    
    if output:
        Path(output).write_text(full_output)
        console.print(f"[green]✓[/green] Saved GEO brief to {output}")
    else:
        console.print(full_output)


def _generate_content_brief(cluster_name: str, keywords: List[dict], provider: str) -> str:
    """Generate a content brief for a cluster of keywords."""
    from .llm import _detect_provider, HAS_GENAI, HAS_LITELLM
    import os
    
    # Sort by opportunity score
    keywords = sorted(keywords, key=lambda k: k.get("opportunity_score", 0), reverse=True)
    
    # Extract key data
    kw_list = [k["keyword"] for k in keywords]
    intents = set(k.get("intent", "informational") for k in keywords)
    primary_intent = max(intents, key=lambda i: sum(1 for k in keywords if k.get("intent") == i))
    avg_interest = sum(k.get("relative_interest", 0) for k in keywords) / len(keywords)
    avg_difficulty = sum(k.get("difficulty", 0) for k in keywords) / len(keywords)
    questions = [k["keyword"] for k in keywords if any(
        k["keyword"].startswith(q) for q in ["how", "what", "why", "when", "where", "which", "who"]
    )]
    
    # Build brief with or without LLM
    detected_provider = _detect_provider() if provider == "auto" else provider
    
    h1_suggestion = kw_list[0].title() if kw_list else cluster_name.title()
    
    brief = f"""# Content Brief: {cluster_name.replace('-', ' ').title()}

## Overview
- **Primary Intent:** {primary_intent.title()}
- **Keywords:** {len(keywords)}
- **Avg. Search Volume Score:** {avg_volume:.2f}
- **Avg. Difficulty Score:** {avg_difficulty:.2f}

## Suggested H1
> {h1_suggestion}

## Target Keywords
{chr(10).join(f"- {kw}" for kw in kw_list[:10])}
{"" if len(kw_list) <= 10 else f"- ... and {len(kw_list) - 10} more"}

## Questions to Answer
{chr(10).join(f"- {q}" for q in questions[:8]) if questions else "- No question-style keywords detected"}

## Content Recommendations
- **Funnel Stage:** {_get_dominant_funnel(keywords)}
- **Word Count:** {_suggest_word_count(primary_intent)}
- **Content Type:** {_suggest_content_type(primary_intent)}

## SEO Checklist
- [ ] Include primary keyword in H1
- [ ] Address top questions in subheadings
- [ ] Add internal links to related content
- [ ] Include relevant entities/topics
- [ ] Optimize meta description with key terms
"""
    
    # Try to enhance with LLM if available
    if detected_provider and detected_provider != "none":
        try:
            enhanced = _enhance_brief_with_llm(cluster_name, kw_list, detected_provider)
            if enhanced:
                brief += f"\n## AI-Generated Insights\n{enhanced}\n"
        except Exception as e:
            logging.debug(f"LLM brief enhancement failed: {e}")
    
    return brief


def _generate_geo_brief(
    cluster_name: str, 
    keywords: List[dict], 
    provider: str, 
    geo: str = "ae",
    niche: Optional[str] = None,
) -> str:
    """Generate a GEO-enhanced content brief for UAE/Gulf markets."""
    from .llm import _detect_provider, HAS_GENAI, HAS_LITELLM
    from .entities import extract_entities, UAE_EMIRATES
    import os
    
    # Sort by opportunity score
    keywords = sorted(keywords, key=lambda k: k.get("opportunity_score", 0), reverse=True)
    
    # Extract key data
    kw_list = [k["keyword"] for k in keywords]
    intents = set(k.get("intent", "informational") for k in keywords)
    primary_intent = max(intents, key=lambda i: sum(1 for k in keywords if k.get("intent") == i))
    avg_interest = sum(k.get("relative_interest", 0) for k in keywords) / len(keywords)
    avg_difficulty = sum(k.get("difficulty", 0) for k in keywords) / len(keywords)
    avg_ctr = sum(k.get("ctr_potential", 1.0) for k in keywords) / len(keywords)
    
    questions = [k["keyword"] for k in keywords if any(
        k["keyword"].startswith(q) for q in ["how", "what", "why", "when", "where", "which", "who"]
    )]
    
    # Extract geographic entities
    entities = []
    emirates_mentioned = set()
    districts_mentioned = set()
    
    for kw in kw_list:
        ent = extract_entities(kw, geo)
        if ent["is_local"]:
            if ent["emirate"]:
                emirates_mentioned.add(ent["emirate"])
            if ent["district"]:
                districts_mentioned.add(ent["district"])
    
    # Build GEO-enhanced brief
    detected_provider = _detect_provider() if provider == "auto" else provider
    h1_suggestion = kw_list[0].title() if kw_list else cluster_name.title()
    
    # Get locale-specific terms
    locale_terms = []
    if niche == "contracting":
        locale_terms = [
            "Civil Defense approval", "Municipality permit", "NOC",
            "fit-out", "MEP", "HVAC", "turnkey", "BOQ"
        ]
    
    brief = f"""# GEO Content Brief: {cluster_name.replace('-', ' ').title()}

## Market Overview
- **Target Geo:** {UAE_EMIRATES.get(geo, {}).get('name', geo.upper())}
- **Primary Intent:** {primary_intent.title()}
- **Keywords:** {len(keywords)}
- **Avg. Search Volume Score:** {avg_volume:.2f}
- **Avg. Difficulty Score:** {avg_difficulty:.2f}
- **Avg. CTR Potential:** {avg_ctr:.2f}

## Suggested H1
> {h1_suggestion}

## Target Keywords
{chr(10).join(f"- {kw}" for kw in kw_list[:10])}
{"" if len(kw_list) <= 10 else f"- ... and {len(kw_list) - 10} more"}

## Geographic Targeting
- **Emirates Covered:** {', '.join(emirates_mentioned) if emirates_mentioned else 'None specified (consider adding location modifiers)'}
- **Districts/Areas:** {', '.join(list(districts_mentioned)[:5]) if districts_mentioned else 'N/A'}
- **Recommended Location Variants:** {', '.join([f"{kw_list[0]} {e}" for e in list(UAE_EMIRATES.keys())[:3]]) if kw_list else 'N/A'}

## Questions to Answer
{chr(10).join(f"- {q}" for q in questions[:8]) if questions else "- No question-style keywords detected"}

## Content Recommendations
- **Funnel Stage:** {_get_dominant_funnel(keywords)}
- **Word Count:** {_suggest_word_count(primary_intent)}
- **Content Type:** {_suggest_content_type(primary_intent)}

## UAE/Gulf-Specific Requirements
- [ ] Include local regulatory terms ({', '.join(locale_terms[:4]) if locale_terms else 'permits, approvals, licenses'})
- [ ] Reference specific Emirates ({', '.join(list(emirates_mentioned)[:3]) if emirates_mentioned else 'Dubai, Abu Dhabi'})
- [ ] Add local trust signals (DED license, years in UAE, local portfolio)
- [ ] Include UAE-relevant pricing info (AED, VAT considerations)
- [ ] Add location-specific schema markup (LocalBusiness, Service)

## SEO Checklist
- [ ] Include primary keyword in H1
- [ ] Address top questions in H2/H3 subheadings
- [ ] Add internal links to related service pages
- [ ] Include relevant UAE entities/topics
- [ ] Optimize meta description with key terms + location
- [ ] Add FAQ schema for featured snippet potential
- [ ] Include business schema with UAE address
"""
    
    # Try to enhance with LLM if available
    if detected_provider and detected_provider != "none":
        try:
            enhanced = _enhance_geo_brief_with_llm(cluster_name, kw_list, detected_provider, geo, niche)
            if enhanced:
                brief += f"\n## AI-Generated Local Insights\n{enhanced}\n"
        except Exception as e:
            logging.debug(f"LLM GEO brief enhancement failed: {e}")
    
    return brief


def _enhance_geo_brief_with_llm(
    cluster_name: str, 
    keywords: List[str], 
    provider: str,
    geo: str,
    niche: Optional[str],
) -> str:
    """Use LLM to add UAE/Gulf-specific insights to the brief."""
    from .llm import HAS_GENAI, HAS_LITELLM
    import os
    
    niche_context = f" in the {niche} industry" if niche else ""
    
    prompt = f"""Analyze these SEO keywords for UAE{niche_context} and provide:
1. A compelling meta description (150-160 chars) targeting UAE customers
2. 3 UAE-specific entities/topics that MUST be mentioned for local authority
3. 2 local regulations or permits relevant to this topic
4. 1 content angle that competitors in the UAE market likely miss

Keywords: {', '.join(keywords[:15])}
Geo: {geo.upper()}
Niche: {niche or 'general'}

Be concise. Focus on UAE/Gulf market specifics. Format as bullet points."""

    try:
        if provider == "gemini" and HAS_GENAI:
            import google.generativeai as genai
            api_key = os.getenv("GEMINI_API_KEY", "")
            if api_key:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel("gemini-2.0-flash")
                resp = model.generate_content(prompt)
                return getattr(resp, "text", "")
        elif HAS_LITELLM:
            import litellm
            model = "gpt-4o-mini" if os.getenv("OPENAI_API_KEY") else "claude-3-haiku-20240307"
            response = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
            )
            return response.choices[0].message.content or ""
    except Exception as e:
        logging.debug(f"LLM GEO enhancement failed: {e}")
    
    return ""


def _get_dominant_funnel(keywords: List[dict]) -> str:
    """Get the dominant funnel stage from keywords."""
    stages = [k.get("funnel_stage", "MOFU") for k in keywords]
    from collections import Counter
    return Counter(stages).most_common(1)[0][0]


def _suggest_word_count(intent: str) -> str:
    """Suggest word count based on intent."""
    counts = {
        "informational": "1500-2500 words (comprehensive guide)",
        "commercial": "1000-1500 words (comparison/review)",
        "transactional": "500-800 words (product-focused)",
        "navigational": "300-500 words (reference page)",
        "direct_answer": "300-600 words (quick answer + context)",
        "complex_research": "2000-3500 words (in-depth analysis)",
        "comparative": "1200-2000 words (detailed comparison)",
        "local": "600-1000 words (local landing page)",
    }
    return counts.get(intent, "1000-1500 words")


def _suggest_content_type(intent: str) -> str:
    """Suggest content type based on intent."""
    types = {
        "informational": "How-to Guide, Tutorial, or Explainer",
        "commercial": "Comparison Post, Review, or Listicle",
        "transactional": "Product Page, Landing Page, or Pricing Page",
        "navigational": "About Page, Contact, or FAQ",
        "direct_answer": "FAQ Entry, Snippet-optimized Post",
        "complex_research": "Ultimate Guide, Whitepaper, or Long-form Article",
        "comparative": "Versus Post, Comparison Table, or Buyer's Guide",
        "local": "Local Landing Page, Store Locator, or City Guide",
    }
    return types.get(intent, "Blog Post or Article")


def _enhance_brief_with_llm(cluster_name: str, keywords: List[str], provider: str) -> str:
    """Use LLM to add insights to the brief."""
    from .llm import HAS_GENAI, HAS_LITELLM
    import os
    
    prompt = f"""Analyze these SEO keywords for the topic "{cluster_name}" and provide:
1. A compelling meta description (150-160 chars)
2. 3 key entities/topics that MUST be mentioned for topical authority
3. 1 content angle that competitors likely miss

Keywords: {', '.join(keywords[:15])}

Be concise. Format as bullet points."""

    try:
        if provider == "gemini" and HAS_GENAI:
            import google.generativeai as genai
            api_key = os.getenv("GEMINI_API_KEY", "")
            if api_key:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel("gemini-2.0-flash")
                resp = model.generate_content(prompt)
                return getattr(resp, "text", "")
        elif HAS_LITELLM:
            import litellm
            model = "gpt-4o-mini" if os.getenv("OPENAI_API_KEY") else "claude-3-haiku-20240307"
            response = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
            )
            return response.choices[0].message.content or ""
    except Exception as e:
        logging.debug(f"LLM enhancement failed: {e}")
    
    return ""


@app.command()
def qa(
    input_file: str = typer.Argument(
        ...,
        help="Path to keywords JSON file to validate",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output", "-o",
        help="Output path for validated keywords (default: overwrites input)",
    ),
    min_cluster_size: int = typer.Option(
        3,
        "--min-cluster-size",
        help="Minimum keywords per cluster (clusters with fewer are removed)",
    ),
    max_words: int = typer.Option(
        6,
        "--max-words",
        help="Maximum words per keyword (longer keywords are removed)",
    ),
    min_words: int = typer.Option(
        2,
        "--min-words",
        help="Minimum words per keyword (shorter keywords are removed)",
    ),
    min_opportunity: float = typer.Option(
        0.0,
        "--min-opportunity",
        help="Minimum opportunity score (0-1, keywords below are removed)",
    ),
    min_volume: float = typer.Option(
        0.0,
        "--min-volume",
        help="Minimum search volume score (0-1, keywords below are removed)",
    ),
    report: bool = typer.Option(
        False,
        "--report", "-r",
        help="Print detailed QA report",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be removed without making changes",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose output",
    ),
):
    """
    🔍 Run QA validation on keyword output.
    
    Validates and cleans pipeline output by:
    - Removing clusters with fewer than minimum keywords
    - Removing keywords that are too long or too short
    - Removing keywords below score thresholds
    
    Example:
        keyword-lab qa keywords.json --min-cluster-size 3 --max-words 6 --report
    """
    setup_logging(verbose)
    
    # Import QA module
    from .qa import validate_pipeline_output, print_qa_report
    
    # Load input file
    input_path = Path(input_file)
    if not input_path.exists():
        err_console.print(f"[red]Error:[/red] Input file not found: {input_file}")
        raise typer.Exit(1)
    
    try:
        with open(input_path) as f:
            items = json.load(f)
    except json.JSONDecodeError as e:
        err_console.print(f"[red]Error:[/red] Invalid JSON: {e}")
        raise typer.Exit(1)
    
    if not isinstance(items, list):
        err_console.print("[red]Error:[/red] Expected JSON array of keyword items")
        raise typer.Exit(1)
    
    console.print(f"\n[bold]📋 QA Validation[/bold]")
    console.print(f"Input: {input_path}")
    console.print(f"Keywords: {len(items)}")
    console.print("")
    
    # Run validation
    validated_items, qa_report = validate_pipeline_output(
        items,
        min_cluster_size=min_cluster_size,
        max_word_count=max_words,
        min_word_count=min_words,
        min_opportunity_score=min_opportunity,
        min_relative_interest=min_volume,
    )
    
    # Display summary
    removed = qa_report["removed_count"]
    removal_rate = qa_report["removal_rate"]
    
    if removed > 0:
        color = "yellow" if removal_rate < 0.3 else "red"
        console.print(f"[{color}]Removed:[/{color}] {removed} keywords ({removal_rate:.1%})")
    else:
        console.print("[green]No keywords removed - all passed QA[/green]")
    
    console.print(f"[bold green]Final:[/bold green] {len(validated_items)} keywords")
    
    # Print detailed report if requested
    if report:
        console.print("")
        console.print(print_qa_report(qa_report, verbose=verbose))
    
    # Save output
    if not dry_run:
        output_path = Path(output) if output else input_path
        
        with open(output_path, "w") as f:
            json.dump(validated_items, f, indent=2)
        
        console.print(f"\n[dim]Saved to:[/dim] {output_path}")
    else:
        console.print("\n[dim]--dry-run: No changes saved[/dim]")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """
    🔬 Keyword Lab - Discover and cluster SEO keywords from any content.
    
    Run 'keyword-lab run' to start the pipeline.
    """
    if ctx.invoked_subcommand is None:
        # If no command provided, show help
        console.print(ctx.get_help())


if __name__ == "__main__":
    app()
