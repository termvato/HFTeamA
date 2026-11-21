# HFTeamA

## Requirements

- Python 3.10+

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```bash
hfteama
# or
python -m hfteama
```

## Development

```bash
pytest          # run tests
ruff check .    # lint
ruff format .   # format
```

## Project structure

```
.
├── docs/            # Documentation
├── src/hfteama/     # Package source code
├── tests/           # Test suite
└── pyproject.toml   # Project metadata and tool configuration
```
