# TerraMerge

> Geospatial pipeline merging Polish cadastral (EGIB) layers, engineering parcel-level features, and training a LightGBM model to predict future parcel/land-use changes.

---

## Description

TerraMerge ingests yearly EGIB (Polish cadastral) File Geodatabase exports, cleans and merges
their polygon layers into GeoParquet/DuckDB, engineers geometric/UZG/MPZP/transaction-price
features, aggregates them onto an H3 hexagon grid, builds labeled training data, and trains a
LightGBM model to predict future parcel-level land-use changes. See `product.yaml` for the full
input/output contract, and `methodology_audit.md` for a critical look at the modeling approach.

---

## Installation

Requires Python 3.11+, [uv](https://github.com/astral-sh/uv), and [just](https://github.com/casey/just).

```bash
brew install uv just

git clone https://github.com/dtandev/TerraMergev2.git
cd TerraMergev2

just setup      # uv sync + pre-commit hooks + copies .env-example -> .env
nano .env       # fill in your local EGIB dataset paths
```

`just setup` runs `uv sync` (creates `.venv/`, installs all dependencies from `pyproject.toml`),
installs git hooks, and copies `.env-example` to `.env`.

> **GDAL note:** `gdal` is a pip dependency here, but it needs a matching system `libgdal`
> installed (e.g. `brew install gdal` on macOS) — if `uv sync` fails on it, install the system
> library first and re-run.

---

## Configuration

All machine-specific values (the EGIB dataset paths) live in `.env`, not in `conf/*.yaml` — see
`.env-example` for the full list. `conf/config.yaml` and `conf/data/fast.yaml` reference them via
Hydra's `${oc.env:VAR_NAME}` resolver, loaded from `.env` by `python-dotenv` at startup. Everything
else (feature toggles, model hyperparameters, thresholds) lives in `conf/` and can be overridden
on the command line, e.g. `model.n_estimators=800`.

---

## Usage

The pipeline is orchestrated by `src/main.py` (Hydra) and driven through `just`:

```bash
just run <base_dir>        # full pipeline, all 5 stages, end-to-end
just prepare <base_dir>    # stage 1 only — clean/extract/merge raw layers
just features <base_dir>   # stage 2 only — attach UZG/transactions/MPZP/geometric features
just hexagons <base_dir>   # stage 3 only — build the H3 hex grid
just dataset <base_dir>    # stage 4 only — build labels + neighborhood aggregates
just model <base_dir>      # stage 5 only — train the model
```

Every recipe accepts extra Hydra overrides after `<base_dir>`, e.g.:

```bash
just model /data/egib model.n_estimators=800 logging.console_level=DEBUG
```

Equivalent raw invocation (what `just` runs under the hood):

```bash
uv run python -m src.main data.base_dir=/data/egib prepare.enabled=true ...
```

---

## Examples

See [`examples/`](examples/) for sanitized example Hydra configs (`config.example.yaml`,
`data-fast.example.yaml`) — copy the values you need into your own `conf/config.yaml` /
`conf/data/fast.yaml`, or override them on the command line as shown above. (Only these two files
needed path sanitization; the rest of `conf/*.yaml` has no machine-specific values.)

---

## Limitations

TODO: Describe known limitations and edge cases. This section is mandatory — `product.yaml`
"limitations", `audit.md` (code-level findings) and `methodology_audit.md` (modeling-approach
findings) are the starting point.

---

## Input Data Requirements

| Parameter | Requirement |
|---|---|
| Source format | ESRI File Geodatabase (`.gdb`), organized as `rok_YYYY/*.gdb` under `data.base_dir` |
| Target CRS | EPSG:2180 |
| Required layers | TODO: verify this — see `conf/prepare/default.yaml` `layer_name_map` for expected source layer aliases |

---

## Development

```bash
just test     # run tests
just check    # lint + format check
just fix      # auto-fix lint and format issues
just lint     # run all pre-commit hooks on every file
just release  # bump version, update CHANGELOG, tag, and push
```

Commit messages and PR titles follow [Conventional Commits](https://www.conventionalcommits.org/).

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md).
