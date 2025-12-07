import json
import logging
import sys
from pathlib import Path
from typing import Optional, List

import typer
import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.panel import Panel

from .pipeline import run_pipeline
from .config import load_config, ConfigValidationError

# Initialize rich console
console = Console()
err_console = Console(stderr=True)

app = typer.Typer(
    add_completion=False, 
    no_args_is_help=False,  # Changed for interactive mode
    help="🔬 Keyword Lab - Discover and cluster SEO keywords"
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
    table.add_column("Vol", justify="right", style="white")
    table.add_column("Diff", justify="right", style="white")
    table.add_column("Opp", justify="right", style="bold green")
    
    for item in items[:max_rows]:
        table.add_row(
            item["keyword"],
            item["cluster"],
            item["intent"],
            item["funnel_stage"],
            f"{item['search_volume']:.2f}",
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
        cfg = load_config(config_path)
    except ConfigValidationError as e:
        err_console.print(f"[bold red]Configuration Error:[/bold red]")
        for error in e.errors:
            err_console.print(f"  [red]•[/red] {error}")
        err_console.print("\n[dim]Check your config.yaml for typos or invalid values.[/dim]")
        sys.exit(2)

    comp_list: List[str] = [c.strip() for c in competitors.split(",") if c.strip()]

    # Show a panel with run configuration
    if not verbose:
        console.print(Panel(
            f"[bold]Seed:[/bold] {seed_topic}\n"
            f"[bold]Audience:[/bold] {audience}\n"
            f"[bold]Geo:[/bold] {geo} | [bold]Language:[/bold] {language}",
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
