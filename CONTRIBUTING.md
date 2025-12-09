# Contributing to ORYX

Thank you for your interest in contributing to ORYX! This guide will help you get started.

## Development Setup

### 1. Clone the Repository

```bash
git clone https://github.com/saifmb0/oryx.git
cd oryx
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install in Development Mode

Install all development dependencies:

```bash
pip install -e ".[all,dev]"
```

This installs:
- **Core dependencies**: typer, rich, pandas, scikit-learn, etc.
- **Optional features**: sentence-transformers, litellm, openpyxl, markdownify
- **Dev tools**: pytest, pytest-cov, ruff, mypy, pre-commit

### 4. Set Up Pre-commit Hooks

```bash
pre-commit install
```

This ensures code quality checks run automatically before each commit.

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/oryx --cov-report=html

# Run a specific test file
pytest tests/test_schema.py -v
```

## Code Style

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
# Check for issues
ruff check .

# Auto-fix issues
ruff check --fix .

# Format code
ruff format .
```

### Type Checking

```bash
mypy src/oryx
```

## Project Structure

```
oryx/
├── src/oryx/      # Main package source
│   ├── cli.py            # Command-line interface
│   ├── pipeline.py       # Main orchestration
│   ├── nlp.py            # Text preprocessing
│   ├── cluster.py        # Clustering and intent
│   ├── scrape.py         # Web scraping
│   ├── llm.py            # LLM integration
│   ├── io.py             # File I/O
│   ├── schema.py         # Data models
│   ├── metrics.py        # Metrics and scoring
│   └── config.py         # Configuration
├── tests/                # Test suite
├── .github/workflows/    # CI/CD
└── pyproject.toml        # Package config
```

## Making Changes

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes

- Write clear, documented code
- Add type hints where possible
- Update tests for new functionality
- Update documentation if needed

### 3. Test Your Changes

```bash
# Run the test suite
pytest

# Test the CLI manually
oryx --help
oryx run keywords.json -c config.yaml -o output.json
```

### 4. Commit Your Changes

Use [Conventional Commits](https://www.conventionalcommits.org/):

```bash
git commit -m "feat: add new clustering algorithm"
git commit -m "fix: handle empty keyword lists"
git commit -m "docs: update installation instructions"
git commit -m "test: add tests for sitemap parsing"
```

Commit types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Test additions/changes
- `refactor`: Code refactoring
- `chore`: Maintenance tasks

### 5. Push and Create a Pull Request

```bash
git push origin feature/your-feature-name
```

Then open a pull request on GitHub.

## Pull Request Guidelines

- **Keep PRs focused**: One feature or fix per PR
- **Write a clear description**: Explain what and why
- **Add tests**: Cover new functionality
- **Update docs**: If adding user-facing features
- **Pass CI**: All checks must pass

## Adding New Features

### Adding a New Export Format

1. Add the writer function in `src/oryx/io.py`:

```python
def write_newformat(df: pd.DataFrame, path: str) -> None:
    """Write results to NewFormat."""
    # Implementation
```

2. Update `write_output()` to handle the new format
3. Add tests in `tests/test_io.py`

### Adding a New LLM Provider

1. The project uses LiteLLM which supports 100+ providers
2. Users can configure via `llm.model` in their config
3. For direct integration, add to `src/oryx/llm.py`

### Adding New Metrics

1. Add the metric function in `src/oryx/metrics.py`
2. Integrate into `pipeline.py`
3. Update schema if new fields are needed

## Reporting Issues

When reporting bugs, please include:

1. **Python version**: `python --version`
2. **Package version**: `pip show oryx`
3. **OS**: Linux/macOS/Windows
4. **Minimal reproduction**: Config and keywords to reproduce
5. **Full error traceback**

## Questions?

- Open a GitHub issue for bugs or feature requests
- Check existing issues before creating new ones
- Be respectful and constructive

---

Thank you for contributing! 🎉
