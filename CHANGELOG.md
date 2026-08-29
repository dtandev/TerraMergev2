# Changelog

All notable changes to this project will be documented here.

Format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), with
version headings in commitizen's `vX.Y.Z` form (matching `tag_format` in pyproject.toml).
Versioning follows [Semantic Versioning](https://semver.org/).

---

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
