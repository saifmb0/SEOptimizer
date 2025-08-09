import json
import logging
from pathlib import Path
from typing import Optional, List

import typer
import yaml

from .pipeline import run_pipeline

app = typer.Typer(add_completion=False, no_args_is_help=True, help="keyword-lab")


def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


@app.command()
def run(
    seed_topic: str = typer.Option(..., "--seed-topic", help="Seed topic"),
    audience: str = typer.Option(..., "--audience", help="Audience"),
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
    output: str = typer.Option("keywords.json", "--output", help="Path to JSON file or '-' for stdout"),
    save_csv: Optional[str] = typer.Option(None, "--save-csv"),
    verbose: bool = typer.Option(False, "--verbose", help="DEBUG logs to stderr"),
    config_path: Optional[str] = typer.Option(None, "--config", help="YAML config (default: ./config.yaml if present)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Log steps without network calls"),
):
    setup_logging(verbose)

    cfg = {}
    # Load explicit config or default ./config.yaml if present
    candidate = Path(config_path) if config_path else Path("config.yaml")
    if candidate and candidate.exists():
        try:
            cfg = yaml.safe_load(candidate.read_text()) or {}
            logging.debug(f"Loaded config from {candidate}")
        except Exception as e:
            logging.warning(f"Failed to read config {candidate}: {e}")

    comp_list: List[str] = [c.strip() for c in competitors.split(",") if c.strip()]

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
    )

    # Per user request: print formatted JSON to stdout even when writing to a file
    if output != "-":
        print(json.dumps(items, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
