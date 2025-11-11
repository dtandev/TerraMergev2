# src/prepare_data/duckdb_init.py
from __future__ import annotations

from pathlib import Path
import duckdb
from loguru import logger
from omegaconf import DictConfig, OmegaConf


def run_duckdb_init(cfg: DictConfig) -> None:
    """
    Tworzy/otwiera plik bazy DuckDB i przygotowuje podstawowy schemat.
    Zero tabel, zero ładowania danych – tylko czysta inicjalizacja.
    """
    db_path = Path(str(OmegaConf.select(cfg, "data.duckdb_path", default="artifacts/duckdb/egib.duckdb")))
    schema = str(OmegaConf.select(cfg, "data.duckdb.schema", default="egib"))
    db_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Inicjalizacja DuckDB: {}", db_path)
    con = duckdb.connect(str(db_path))
    try:
        con.execute("INSTALL spatial; LOAD spatial;")
        con.execute("INSTALL httpfs; LOAD httpfs;")
        con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema};")

        # --- BEZ 'auto' — ustaw tylko, jeśli podano w configu ---
        threads = OmegaConf.select(cfg, "data.duckdb.threads", default=None)
        if threads is not None:
            con.execute("SET threads TO ?;", [int(threads)])

        mem_limit = OmegaConf.select(cfg, "data.duckdb.memory_limit", default=None)  # np. "8GB", "2GB"
        if mem_limit:
            con.execute("PRAGMA memory_limit = ?;", [str(mem_limit)])

        logger.success(f"DuckDB gotowe. Plik utworzony, schema '{schema}' istnieje.")
    finally:
        con.close()
