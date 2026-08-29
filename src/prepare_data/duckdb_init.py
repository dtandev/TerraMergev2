# src/prepare_data/duckdb_init.py
from __future__ import annotations

from pathlib import Path

import duckdb
from loguru import logger
from omegaconf import DictConfig, OmegaConf


def _load_extension(con: duckdb.DuckDBPyConnection, name: str, *, required: bool) -> bool:
    """LOAD a DuckDB extension, INSTALLing it first if it isn't available yet.

    Returns True if the extension is loaded. When ``required`` is False, a failure to
    install (e.g. offline with no cached copy) is logged and swallowed rather than aborting
    init — ``httpfs`` is only needed for remote (http/s3) reads, which local runs never do,
    so a bare init must not hard-fail just because the network is unreachable. The names are
    hardcoded literals, not user input (DuckDB has no parameter binding for INSTALL/LOAD).
    """
    try:
        con.execute(f"LOAD {name};")
        return True
    except (duckdb.CatalogException, duckdb.IOException):
        pass
    try:
        con.execute(f"INSTALL {name};")
        con.execute(f"LOAD {name};")
        return True
    except Exception:
        if required:
            raise
        logger.warning("Nie udało się zainstalować rozszerzenia '{}' — pomijam.", name)
        return False


def run_duckdb_init(cfg: DictConfig) -> None:
    """
    Tworzy/otwiera plik bazy DuckDB i przygotowuje podstawowy schemat.
    Zero tabel, zero ładowania danych – tylko czysta inicjalizacja.
    """
    db_path = Path(
        str(OmegaConf.select(cfg, "data.duckdb_path", default="artifacts/duckdb/egib.duckdb"))
    )
    schema = str(OmegaConf.select(cfg, "duckdb.schema", default="egib"))
    db_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Inicjalizacja DuckDB: {}", db_path)
    con = duckdb.connect(str(db_path))
    try:
        _load_extension(con, "spatial", required=True)
        _load_extension(con, "httpfs", required=False)
        con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema};")

        # --- BEZ 'auto' — ustaw tylko, jeśli podano w configu ---
        threads = OmegaConf.select(cfg, "duckdb.threads", default=None)
        if threads is not None:
            con.execute("SET threads TO ?;", [int(threads)])

        mem_limit = OmegaConf.select(cfg, "duckdb.memory_limit", default=None)  # np. "8GB", "2GB"
        if mem_limit:
            con.execute("PRAGMA memory_limit = ?;", [str(mem_limit)])

        logger.success(f"DuckDB gotowe. Plik utworzony, schema '{schema}' istnieje.")
    finally:
        con.close()
