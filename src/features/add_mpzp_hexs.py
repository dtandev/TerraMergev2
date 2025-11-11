#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
add_mpzp_hexs.py
----------------
1) Wczytuje heksy (cfg.hex.table lub cfg.pipeline.hex.table)
2) Wczytuje MPZP (cfg.add_mpzp_data.table lub cfg.pipeline.add_mpzp_data.table)
3) Liczy udziały klas MPZP per (hex_id, year): mpzp_<KLASA>_share
4) Zapisuje do cfg.add_mpzp_data.out_table (lub fallback) jako GEOMETRY (SRID wg enforce_crs)

Wymagane klucze w configu:
- hex: { table, id_col, geom_col, schema, out_suffix }
- add_mpzp_data: { enabled, table, out_table, klasy_mpzp }
- layer_defaults: { decimals, write_mode, enforce_crs }
- data: { duckdb_path }
"""

from pathlib import Path
from typing import Optional, Dict, List
import time
import duckdb
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely import to_wkb, set_srid  # Shapely >= 2.0
from loguru import logger


# ----------------- Logging setup ----------------- #
def configure_logging(cfg) -> None:
    level = _get_val(cfg, ["logging.level"], default="INFO")
    fmt = _get_val(
        cfg,
        ["logging.format"],
        default="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <7}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                "<level>{message}</level>",
    )
    try:
        logger.remove()
    except Exception:
        pass
    logger.add(lambda m: print(m, end=""), level=level, format=fmt, enqueue=False)


# ----------------- Helpers ----------------- #
def _sel(cfg, path: str, default=None):
    cur = cfg
    for p in path.split("."):
        if cur is None or p not in cur:
            return default
        cur = cur[p]
    return cur

def _get_val(cfg, paths: List[str], default=None):
    """Sprytne pobieranie wartości: przetestuj po kolei ścieżki."""
    for p in paths:
        v = _sel(cfg, p, default=None)
        if v is not None:
            return v
    return default

def connect_duckdb(cfg) -> duckdb.DuckDBPyConnection:
    db_path = Path(_get_val(cfg, ["data.duckdb_path"]))
    t0 = time.perf_counter()
    logger.info("Connecting DuckDB → {}", db_path)
    con = duckdb.connect(str(db_path.expanduser()))
    try:
        con.execute("LOAD spatial;")
        logger.debug("DuckDB spatial extension: loaded.")
    except duckdb.CatalogException:
        logger.warning("DuckDB spatial extension not installed — installing now…")
        con.execute("INSTALL spatial;")
        con.execute("LOAD spatial;")
        logger.debug("DuckDB spatial extension: installed & loaded.")
    logger.success("Connected in {:.3f}s", time.perf_counter() - t0)
    return con

def _detect_srid(con: duckdb.DuckDBPyConnection, table: str, geom_col: str) -> Optional[int]:
    try:
        v = con.execute(
            f"SELECT ST_SRID({geom_col}) FROM {table} WHERE {geom_col} IS NOT NULL LIMIT 1"
        ).fetchone()
        return int(v[0]) if v and v[0] else None
    except Exception:
        return None

def _get_hex_params(cfg):
    """Parametry heksów z cfg.hex.* lub cfg.pipeline.hex.* (fallback)."""
    table    = _get_val(cfg, ["hex.table", "pipeline.hex.table"])
    id_col   = _get_val(cfg, ["hex.id_col", "pipeline.hex.id_col"], default="hex_id")
    geom_col = _get_val(cfg, ["hex.geom_col", "pipeline.hex.geom_col"], default="geometry")
    if not table:
        raise ValueError("Brak nazwy tabeli heksów: oczekiwano 'hex.table' albo 'pipeline.hex.table'.")
    logger.debug("HEX params → table={}, id_col={}, geom_col={}", table, id_col, geom_col)
    return table, id_col, geom_col

def _get_enforce_crs(cfg) -> str:
    crs = _get_val(
        cfg,
        ["layer_defaults.enforce_crs", "pipeline.layer_defaults.enforce_crs",
         "pipeline.geometry.layer_defaults.enforce_crs"],
        default="EPSG:2180",
    )
    logger.debug("enforce_crs = {}", crs)
    return crs


# ----------------- Loaders ----------------- #
def load_hex_gdf(
    con: duckdb.DuckDBPyConnection,
    cfg,
    *,
    limit: Optional[int] = None,
) -> gpd.GeoDataFrame:
    """
    Wczytaj heksy z ID i polem heksa (m2). CRS z ST_SRID (fallback: enforce_crs).
    """
    table, id_col, geom_col = _get_hex_params(cfg)
    enforce = _get_enforce_crs(cfg)

    srid = _detect_srid(con, table, geom_col)
    crs_str = f"EPSG:{srid}" if srid else enforce
    lim = f" LIMIT {int(limit)}" if (isinstance(limit, int) and limit > 0) else ""

    sql = f"""
        SELECT
          "{id_col}" AS hex_id,
          ST_AsWKB({geom_col}) AS _wkb,
          CAST(ST_Area({geom_col}) AS DOUBLE) AS hex_area_m2
        FROM {table}
        {lim}
    """
    logger.info("[HEX] Ładowanie heksów z {} …", table)
    df = con.execute(sql).df()
    logger.debug("[HEX] Otrzymano {} wierszy.", len(df))
    geoms = gpd.GeoSeries.from_wkb(
        df.pop("_wkb").map(lambda b: bytes(b) if b is not None else None),
        crs=crs_str,
    )
    gdf = gpd.GeoDataFrame(df, geometry=geoms, crs=crs_str)[["hex_id", "hex_area_m2", "geometry"]]
    logger.success("[HEX] Gotowe: {} heksów, CRS={}.", len(gdf), gdf.crs)
    return gdf

def load_mpzp_gdf_from_cfg(
    con: duckdb.DuckDBPyConnection,
    cfg,
    *,
    label_col: str = "mpzp_etykieta",
    year_col: str = "year",
    geom_col: str = "geometry",
    limit: Optional[int] = None,
) -> gpd.GeoDataFrame:
    """
    Wczytaj MPZP jako GeoDataFrame (tylko potrzebne kolumny).
    """
    table = _get_val(cfg, ["add_mpzp_data.table", "pipeline.add_mpzp_data.table"], default="egib.MPZP")
    enforce = _get_enforce_crs(cfg)
    srid = _detect_srid(con, table, geom_col)
    crs_str = f"EPSG:{srid}" if srid else enforce

    lim = f" LIMIT {int(limit)}" if (isinstance(limit, int) and limit > 0) else ""
    sql = f"""
        SELECT
            {year_col} AS year,
            {label_col} AS {label_col},
            ST_AsWKB({geom_col}) AS _wkb
        FROM {table}
        {lim}
    """
    logger.info("[MPZP] Ładowanie z {} …", table)
    df = con.execute(sql).df()
    logger.debug("[MPZP] Otrzymano {} wierszy.", len(df))
    geoms = gpd.GeoSeries.from_wkb(
        df.pop("_wkb").map(lambda b: bytes(b) if b is not None else None),
        crs=crs_str,
    )
    gdf = gpd.GeoDataFrame(df, geometry=geoms, crs=crs_str)
    logger.success("[MPZP] Gotowe: {} rekordów, CRS={}.", len(gdf), gdf.crs)
    return gdf


# -------------- Core compute -------------- #
def mpzp_hex_shares(
    gdf_mpzp: gpd.GeoDataFrame,
    gdf_hex: gpd.GeoDataFrame,
    *,
    label_col: str = "mpzp_etykieta",
    year_col: str = "year",
    hex_id_col: str = "hex_id",
    hex_area_col: str = "hex_area_m2",
    classes: Optional[List[str]] = None,
    decimals: int = 3
) -> gpd.GeoDataFrame:
    """
    Udziały klas MPZP per (hex_id, year). Geometria = heks.
    Zwraca: [hex_id, year, mpzp_<KLASA>_share..., geometry]
    """
    logger.info("[MPZP×HEX] Start obliczeń udziałów.")
    if gdf_mpzp.crs != gdf_hex.crs:
        logger.warning("CRS mismatch (MPZP={}, HEX={}) → reprojekcja MPZP do HEX.", gdf_mpzp.crs, gdf_hex.crs)
        gdf_mpzp = gdf_mpzp.to_crs(gdf_hex.crs)

    need = {year_col, label_col, "geometry"}
    missing = need - set(gdf_mpzp.columns)
    if missing:
        msg = f"Brakuje kolumn w MPZP: {missing}"
        logger.error(msg)
        raise ValueError(msg)

    mpzp_use = gdf_mpzp[[year_col, label_col, "geometry"]].copy()
    hex_use  = gdf_hex[[hex_id_col, hex_area_col, "geometry"]].copy()

    logger.info("[MPZP×HEX] Overlay(intersection)…")
    ix = gpd.overlay(mpzp_use, hex_use, how="intersection")
    logger.debug("[MPZP×HEX] Intersections: {}", len(ix))
    if ix.empty:
        logger.warning("[MPZP×HEX] Brak przecięć — zwracam pustą ramkę.")
        out = gpd.GeoDataFrame(columns=[hex_id_col, year_col, "geometry"], geometry=[], crs=gdf_hex.crs)
        if classes:
            for k in classes:
                out[f"mpzp_{k}_share"] = []
        return out

    ix["__ix_area"] = ix.geometry.area.astype(float)

    logger.info("[MPZP×HEX] Agreguję pole per (hex, year, klasa)…")
    by_key = (
        ix.groupby([hex_id_col, year_col, label_col], dropna=False)["__ix_area"]
          .sum()
          .rename("area_in_hex")
          .reset_index()
          .merge(hex_use[[hex_id_col, hex_area_col]], on=hex_id_col, how="left")
    )
    by_key["share"] = (by_key["area_in_hex"] / by_key[hex_area_col]).clip(0.0, 1.0)

    class_values = classes if classes is not None else sorted(by_key[label_col].dropna().unique().tolist())
    logger.debug("[MPZP×HEX] Klasy MPZP: {}", class_values)

    wide = by_key.pivot_table(
        index=[hex_id_col, year_col],
        columns=label_col,
        values="share",
        aggfunc="sum",
        fill_value=0.0,
    )
    for k in class_values:
        if k not in wide.columns:
            wide[k] = 0.0
    wide = wide.reindex(columns=class_values)
    wide.columns = [f"mpzp_{str(c)}_share" for c in wide.columns]
    wide = wide.reset_index()
    logger.debug("[MPZP×HEX] Pivot shape: {}", wide.shape)

    out = wide.merge(hex_use[[hex_id_col, "geometry"]], on=hex_id_col, how="left")
    share_cols = [c for c in out.columns if c.endswith("_share")]
    out[share_cols] = out[share_cols].round(decimals)

    gout = gpd.GeoDataFrame(out, geometry="geometry", crs=gdf_hex.crs)
    logger.success("[MPZP×HEX] Wynik: {} wierszy, {} kolumn.", len(gout), gout.shape[1])
    return gout


# -------------- Save -------------- #
def save_geodf_as_ewkb_geometry(
    db_path: Path,
    gdf: gpd.GeoDataFrame,
    table: str,
    *,
    srid: int = 2180,
    geom_col: str = "geometry",
    write_mode: str = "replace",
    casts: Optional[Dict[str, str]] = None,
) -> int:
    """
    Zapis GeoDataFrame do DuckDB jako GEOMETRY (przez EWKB z SRID).
    """
    logger.info("[SAVE] Zapis do {} (SRID={}, tryb={})…", table, srid, write_mode)
    df = gdf.copy()
    df["__geom_wkb"] = df[geom_col].apply(
        lambda g: to_wkb(set_srid(g, srid), include_srid=True) if g is not None else None
    )
    df = df.drop(columns=[geom_col])

    casts = casts or {}
    cols_sql = ", ".join(
        (f'"{c}"::{casts[c]} AS "{c}"' if c in casts else f'"{c}"')
        for c in df.columns if c != "__geom_wkb"
    )
    geom_sql = 'ST_GeomFromWKB(__geom_wkb) AS "geometry"'
    select_sql = f'SELECT {(cols_sql + ", ") if cols_sql else ""}{geom_sql} FROM __tmp__'

    schema = table.split(".")[0] if "." in table else "main"
    with duckdb.connect(str(db_path)) as con:
        try:
            con.execute("LOAD spatial;")
        except duckdb.CatalogException:
            logger.warning("DuckDB spatial extension not installed — installing now…")
            con.execute("INSTALL spatial;")
            con.execute("LOAD spatial;")

        schema_clean = schema.replace('"', "")
        con.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema_clean}";')

        try:
            con.unregister("__tmp__")
        except Exception:
            pass
        con.register("__tmp__", df)

        if write_mode == "replace":
            con.execute(f"CREATE OR REPLACE TABLE {table} AS {select_sql}")
        elif write_mode in ("create", "append"):
            con.execute(f"CREATE TABLE IF NOT EXISTS {table} AS {select_sql} LIMIT 0")
            con.execute(f"INSERT INTO {table} {select_sql}")
        else:
            raise ValueError("write_mode must be 'replace' | 'append' | 'create'")

        n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        logger.success("[SAVE] Zapisano {:,} wierszy do {}.", n, table)
        return int(n)


# -------------- Orkiestracja -------------- #
def run_add_mpzp_hexs(cfg) -> str:
    """
    Pełny przepływ: load hex + MPZP → shares → save.
    Zwraca nazwę tabeli wynikowej z configu.
    """
    configure_logging(cfg)

    enabled = bool(_get_val(cfg, ["add_mpzp_data.enabled", "pipeline.add_mpzp_data.enabled"], False))
    if not enabled:
        msg = "add_mpzp_data.enabled = false — krok wyłączony w configu."
        logger.error(msg)
        raise RuntimeError(msg)

    con = connect_duckdb(cfg)
    try:
        t0 = time.perf_counter()

        gdf_hex  = load_hex_gdf(con, cfg)
        gdf_mpzp = load_mpzp_gdf_from_cfg(con, cfg)

        classes  = _get_val(cfg, ["add_mpzp_data.klasy_mpzp", "pipeline.add_mpzp_data.klasy_mpzp"], None)
        decimals = int(_get_val(cfg, ["layer_defaults.decimals", "pipeline.layer_defaults.decimals"], 3))
        out_tbl  = _get_val(cfg, ["add_mpzp_data.out_table", "pipeline.add_mpzp_data.out_table"], None)
        if not out_tbl:
            schema = _get_val(cfg, ["hex.schema", "pipeline.hex.schema"], "hex")
            suffix = _get_val(cfg, ["hex.out_suffix", "pipeline.hex.out_suffix"], "rX")
            out_tbl = f"{schema}.mpzp_{suffix}"
        logger.info("[CFG] out_table = {}", out_tbl)

        logger.info("Computing shares for {} (decimals={})…",
                    "preset classes" if classes is not None else "auto-detected classes", decimals)
        res_gdf = mpzp_hex_shares(
            gdf_mpzp, gdf_hex,
            classes=classes,
            label_col="mpzp_etykieta",
            year_col="year",
            hex_id_col="hex_id",
            hex_area_col="hex_area_m2",
            decimals=decimals
        )

        db_path    = Path(_get_val(cfg, ["data.duckdb_path"])).expanduser()
        write_mode = _get_val(cfg, ["layer_defaults.write_mode", "pipeline.layer_defaults.write_mode"], "replace")
        srid_str   = _get_enforce_crs(cfg)  # np. "EPSG:2180"
        srid       = int(str(srid_str).split(":")[-1])

        casts = {}
        if "hex_id" in res_gdf.columns:
            casts["hex_id"] = "VARCHAR"
        if "year" in res_gdf.columns:
            casts["year"] = "INT"

        n_written = save_geodf_as_ewkb_geometry(
            db_path=db_path,
            gdf=res_gdf,
            table=out_tbl,
            srid=srid,
            geom_col="geometry",
            write_mode=write_mode,
            casts=casts,
        )
        logger.success("Pipeline finished: written {:,} rows to {} in {:.3f}s",
                       n_written, out_tbl, time.perf_counter() - t0)
        return out_tbl

    except Exception as e:
        logger.exception("[MPZP] Błąd przetwarzania: {}", e)
        raise
    finally:
        try:
            con.close()
            logger.debug("Połączenie z DuckDB zamknięte.")
        except Exception:
            pass
