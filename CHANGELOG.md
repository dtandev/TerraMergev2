# Changelog

All notable changes to this project will be documented here.

Format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), with
version headings in commitizen's `vX.Y.Z` form (matching `tag_format` in pyproject.toml).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## v0.3.0 (2026-09-04)

### Feat

- **mpzp**: enable zoning features via portable symbol-based label mapping (#12)

## v0.2.7 (2026-08-30)

### Fix

- **uzg**: skip stray empty unit dirs when listing obrębs
- **uzg**: treat blank strings as missing when coalescing land-use variants

## v0.2.6 (2026-08-30)

### Fix

- **pipeline**: clean reproducible features for all-years build
- **crs**: normalise every layer to EPSG:2180 at extraction
- **config**: align resolutions, enable core build stages, drop missing merge tables

## v0.2.5 (2026-08-29)

### Refactor

- **config**: replace Hydra with a plain OmegaConf loader

## v0.2.4 (2026-08-29)

### Fix

- **pipeline**: legacy id rename, coalesce dtypes, and dataset schema creation
- **io**: persist full CRS in GeoParquet and read via DuckDB (read_geoparquet)

## v0.2.3 (2026-08-29)

### Refactor

- **modeling**: write outputs via write_geoparquet

## v0.2.2 (2026-08-29)

### Fix

- **duckdb**: avoid egib catalog/schema collision via non-egib db stem

## v0.2.1 (2026-08-29)

### Refactor

- **prepare_data**: clean_dataset writes via write_geoparquet
- **prepare_data**: split polygon extraction into per-format readers

## v0.2.0 (2026-08-29)

### Feat

- **prepare_data**: add readers/gdb_reader for ESRI File Geodatabase
- **common**: add DuckDB-based write_geoparquet helper

### Refactor

- **features**: extract shared MPZP helpers into mpzp_common
- **features**: dedup add_uzg normalisation and cover with tests
- **prepare_data**: make DuckDB extension loading resilient
- **common**: return log_file path from setup_logging

## v0.1.0 - 2026-07-25

### Added
- Baseline S&I scaffolding (product.yaml, justfile, CI workflows, tests/, examples/) generated via `/scaffold`
- TODO: describe what this version of the pipeline actually does end-to-end
