"""Load raw transactions Parquet → DuckDB table **egib."Transakcje"** (no join).

Config keys used
----------------
* `data.duckdb_path`          – target DB file
* `data.transactions_path`    – Parquet file with transactions
* `features.add_transaction_prices`:
    * `enabled`      – gate for this step
    * `id_column`    – parcel identifier (kept as-is)
    * `date_column`  – datetime -> year extraction

Result
------
Creates / replaces table **egib."Transakcje"** with:
* all original columns from Parquet (incl. datetime), plus
* `tx_year` (INT) extracted from `date_column` (for easy joins).
"""

# src/features/load_transactions_to_duckdb.py
from __future__ import annotations

from pathlib import Path

import duckdb
from loguru import logger
from omegaconf import DictConfig

from src.common.config_utils import sel as _sel


def run_load_transactions(cfg: DictConfig) -> None:
    """
    Load transactions from a Parquet file into DuckDB as egib."Transakcje".

    Behavior:
    - Controlled by:
        features.enabled (bool)
        features.add_transaction_prices.enabled (bool)
        features.add_transaction_prices.write_mode ("replace" | "skip")
    - If write_mode == "skip" and the table exists → skip work.
    - Otherwise, (re)create with CREATE OR REPLACE TABLE ... AS SELECT ...

    Notes:
    - tx_year is derived with `EXTRACT(YEAR FROM try_cast(date_col AS DATE))`.
      If date parsing fails, tx_year will be NULL.
    """
    if not _sel(cfg, "features.enabled", default=False) or not _sel(
        cfg, "features.add_transaction_prices.enabled", default=False
    ):
        logger.info("load_transactions disabled – skipping")
        return

    db_path = Path(str(_sel(cfg, "data.duckdb_path", "artifacts/duckdb/egib.duckdb"))).expanduser()
    tx_path = Path(str(_sel(cfg, "data.transactions_path", ""))).expanduser()

    if not db_path.exists():
        logger.error("DuckDB not found: {}", db_path)
        return
    if not tx_path.exists():
        logger.error("Transactions parquet not found: {}", tx_path)
        return

    sec: str = "features.add_transaction_prices"
    date_col: str = str(_sel(cfg, f"{sec}.date_column", "Data"))
    write_mode: str = str(_sel(cfg, f"{sec}.write_mode", "replace")).lower()

    con = duckdb.connect(str(db_path))
    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute("CREATE SCHEMA IF NOT EXISTS egib;")
    logger.info("Connected to {}", db_path.name)

    # Check existence when skipping is requested
    if write_mode == "skip":
        exists = con.execute(
            """
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = 'egib' AND table_name = 'Transakcje'
            """
        ).fetchone()
        if exists:
            logger.info('egib."Transakcje" exists → skipping (write_mode=skip)')
            con.close()
            return

    # Replace (default) or create if missing
    con.execute(
        f"""
        CREATE OR REPLACE TABLE egib."Transakcje" AS
        SELECT
            *,
            EXTRACT(YEAR FROM try_cast({date_col} AS DATE))::INT AS tx_year
        FROM read_parquet('{tx_path.as_posix()}');
        """
    )

    rows = con.execute('SELECT COUNT(*) FROM egib."Transakcje";').fetchone()[0]
    logger.success('egib."Transakcje" loaded – {} rows (write_mode={})', rows, write_mode)
    con.close()
