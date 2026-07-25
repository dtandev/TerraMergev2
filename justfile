# TerraEye S&I Project Justfile
# Usage: just <command>
# Requires: just (https://github.com/casey/just), uv (https://github.com/astral-sh/uv)

# Default: list available commands
default:
    @just --list

# Install dependencies and set up the project environment
setup:
    @echo "Setting up project environment..."
    uv sync
    uv run pre-commit install
    uv run pre-commit install --hook-type commit-msg
    @echo "Copying .env-example to .env (if not exists)..."
    cp -n .env-example .env || true
    @echo "Setup complete. Edit .env with your values."

# Explain what this project does and how its stages relate
help:
    @echo "TerraMerge: merges Polish cadastral (EGIB) layers, builds H3-hexagon parcel"
    @echo "features, and trains a LightGBM model predicting future parcel/land-use changes."
    @echo ""
    @echo "Pipeline stages (config-gated in conf/, all driven by src/main.py / Hydra):"
    @echo "  prepare   - clean + extract raw .gdb layers to GeoParquet, merge + clean dataset"
    @echo "  features  - attach UZG / transaction-price / MPZP / geometric features to parcels"
    @echo "  hexagons  - aggregate parcels/transactions/MPZP/UZG onto an H3 hex grid"
    @echo "  dataset   - build labels + neighborhood aggregates for modeling"
    @echo "  model     - train the LightGBM model and export predictions"
    @echo ""
    @echo "'just run <base_dir>' chains all five stages end-to-end against one EGIB directory."
    @echo "Run 'just --list' for the exact command reference."

# Run the full pipeline end-to-end (all stages) against a given EGIB base directory
run base_dir *ARGS:
    uv run python -m src.main data.base_dir={{base_dir}} \
        prepare.enabled=true features.enabled=true pipeline.make=true \
        dataset.enabled=true model.enabled=true {{ARGS}}

# Stage 1 only — clean + extract raw .gdb layers, merge layers, clean the dataset
prepare base_dir *ARGS:
    uv run python -m src.main data.base_dir={{base_dir}} \
        prepare.enabled=true features.enabled=false pipeline.make=false \
        dataset.enabled=false model.enabled=false {{ARGS}}

# Stage 2 only — attach UZG / transaction-price / MPZP / geometric features to parcels
features base_dir *ARGS:
    uv run python -m src.main data.base_dir={{base_dir}} \
        prepare.enabled=false features.enabled=true pipeline.make=false \
        dataset.enabled=false model.enabled=false {{ARGS}}

# Stage 3 only — build the H3 hex grid and fill it with parcels/transactions/MPZP/UZG data
hexagons base_dir *ARGS:
    uv run python -m src.main data.base_dir={{base_dir}} \
        prepare.enabled=false features.enabled=false pipeline.make=true \
        dataset.enabled=false model.enabled=false {{ARGS}}

# Stage 4 only — build labels and neighborhood aggregates for the modeling dataset
dataset base_dir *ARGS:
    uv run python -m src.main data.base_dir={{base_dir}} \
        prepare.enabled=false features.enabled=false pipeline.make=false \
        dataset.enabled=true model.enabled=false {{ARGS}}

# Stage 5 only — train the LightGBM model and export predictions
model base_dir *ARGS:
    uv run python -m src.main data.base_dir={{base_dir}} \
        prepare.enabled=false features.enabled=false pipeline.make=false \
        dataset.enabled=false model.enabled=true {{ARGS}}

# Run all tests
test:
    uv run pytest tests/ -v

# Run all quality checks (lint, format, type check)
check:
    uv run ruff check src/ tests/
    uv run ruff format --check src/ tests/
    @echo "All checks passed."

# Auto-fix lint and format issues
fix:
    uv run ruff check --fix src/ tests/
    uv run ruff format src/ tests/

# Run pre-commit on all files
lint:
    uv run pre-commit run --all-files

# Bump version based on conventional commits, update CHANGELOG, tag, and push
release:
    uv run cz bump
    git push && git push --tags
