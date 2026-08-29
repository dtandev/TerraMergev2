#!/usr/bin/env python3
"""
add_mpzp_hexs.py
----------------
1) Wczytuje heksy (cfg.pipeline.hex.table)
2) Wczytuje MPZP (cfg.pipeline.add_mpzp_data.table)
3) Liczy udziały klas MPZP per (hex_id, year): mpzp_<KLASA>_share
4) Zapisuje do cfg.pipeline.add_mpzp_data.out_table (lub fallback) jako GEOMETRY (SRID wg enforce_crs)

Wymagane klucze w configu:
- pipeline.hex: { table, id_col, geom_col, schema, out_suffix }
- pipeline.add_mpzp_data: { enabled, table, out_table, klasy_mpzp }
- pipeline.layer_defaults: { decimals, write_mode, enforce_crs }
- data: { duckdb_path }
"""

import time
from pathlib import Path

import duckdb
import geopandas as gpd
from loguru import logger

from src.common.config_utils import sel as _sel
from src.common.duckdb_utils import _detect_srid, connect_duckdb, save_geodf_as_ewkb_geometry


# ----------------- Logging setup ----------------- #
def configure_logging(cfg) -> None:
    level = _sel(cfg, "logging.level", "INFO")
    fmt = _sel(
        cfg,
        "logging.format",
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
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
def _get_hex_params(cfg):
    """Parametry heksów z cfg.pipeline.hex.*."""
    table = _sel(cfg, "pipeline.hex.table")
    id_col = _sel(cfg, "pipeline.hex.id_col", "hex_id")
    geom_col = _sel(cfg, "pipeline.hex.geom_col", "geometry")
    if not table:
        raise ValueError("Brak nazwy tabeli heksów: oczekiwano 'pipeline.hex.table'.")
    logger.debug("HEX params → table={}, id_col={}, geom_col={}", table, id_col, geom_col)
    return table, id_col, geom_col


def _get_enforce_crs(cfg) -> str:
    crs = _sel(cfg, "pipeline.layer_defaults.enforce_crs", "EPSG:2180")
    logger.debug("enforce_crs = {}", crs)
    return crs


# ----------------- Loaders ----------------- #
def load_hex_gdf(
    con: duckdb.DuckDBPyConnection,
    cfg,
    *,
    limit: int | None = None,
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
    limit: int | None = None,
) -> gpd.GeoDataFrame:
    """
    Wczytaj MPZP jako GeoDataFrame (tylko potrzebne kolumny).
    """
    table = _sel(cfg, "pipeline.add_mpzp_data.table", "egib.MPZP")
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
    classes: list[str] | None = None,
    decimals: int = 3,
) -> gpd.GeoDataFrame:
    """
    Udziały klas MPZP per (hex_id, year). Geometria = heks.
    Zwraca: [hex_id, year, mpzp_<KLASA>_share..., geometry]
    """
    logger.info("[MPZP×HEX] Start obliczeń udziałów.")
    if gdf_mpzp.crs != gdf_hex.crs:
        logger.warning(
            "CRS mismatch (MPZP={}, HEX={}) → reprojekcja MPZP do HEX.", gdf_mpzp.crs, gdf_hex.crs
        )
        gdf_mpzp = gdf_mpzp.to_crs(gdf_hex.crs)

    need = {year_col, label_col, "geometry"}
    missing = need - set(gdf_mpzp.columns)
    if missing:
        msg = f"Brakuje kolumn w MPZP: {missing}"
        logger.error(msg)
        raise ValueError(msg)

    mpzp_use = gdf_mpzp[[year_col, label_col, "geometry"]].copy()
    hex_use = gdf_hex[[hex_id_col, hex_area_col, "geometry"]].copy()

    logger.info("[MPZP×HEX] Overlay(intersection)…")
    ix = gpd.overlay(mpzp_use, hex_use, how="intersection")
    logger.debug("[MPZP×HEX] Intersections: {}", len(ix))
    if ix.empty:
        logger.warning("[MPZP×HEX] Brak przecięć — zwracam pustą ramkę.")
        out = gpd.GeoDataFrame(
            columns=[hex_id_col, year_col, "geometry"], geometry=[], crs=gdf_hex.crs
        )
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

    class_values = (
        classes if classes is not None else sorted(by_key[label_col].dropna().unique().tolist())
    )
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


# -------------- Orkiestracja -------------- #
def run_add_mpzp_hexs(cfg) -> str:
    """
    Pełny przepływ: load hex + MPZP → shares → save.
    Zwraca nazwę tabeli wynikowej z configu.
    """
    configure_logging(cfg)

    enabled = bool(_sel(cfg, "pipeline.add_mpzp_data.enabled", False))
    if not enabled:
        msg = "pipeline.add_mpzp_data.enabled = false — krok wyłączony w configu."
        logger.error(msg)
        raise RuntimeError(msg)

    db_path = Path(_sel(cfg, "data.duckdb_path")).expanduser()
    con = connect_duckdb(db_path)
    try:
        t0 = time.perf_counter()

        gdf_hex = load_hex_gdf(con, cfg)
        gdf_mpzp = load_mpzp_gdf_from_cfg(con, cfg)

        classes = _sel(cfg, "pipeline.add_mpzp_data.klasy_mpzp", None)
        decimals = int(_sel(cfg, "pipeline.layer_defaults.decimals", 3))
        out_tbl = _sel(cfg, "pipeline.add_mpzp_data.out_table", None)
        if not out_tbl:
            schema = _sel(cfg, "pipeline.hex.schema", "hex")
            suffix = _sel(cfg, "pipeline.hex.out_suffix", "rX")
            out_tbl = f"{schema}.mpzp_{suffix}"
        logger.info("[CFG] out_table = {}", out_tbl)

        logger.info(
            "Computing shares for {} (decimals={})…",
            "preset classes" if classes is not None else "auto-detected classes",
            decimals,
        )
        res_gdf = mpzp_hex_shares(
            gdf_mpzp,
            gdf_hex,
            classes=classes,
            label_col="mpzp_etykieta",
            year_col="year",
            hex_id_col="hex_id",
            hex_area_col="hex_area_m2",
            decimals=decimals,
        )

        write_mode = _sel(cfg, "pipeline.layer_defaults.write_mode", "replace")
        srid_str = _get_enforce_crs(cfg)  # np. "EPSG:2180"
        srid = int(str(srid_str).split(":")[-1])

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
        logger.success(
            "Pipeline finished: written {:,} rows to {} in {:.3f}s",
            n_written,
            out_tbl,
            time.perf_counter() - t0,
        )
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
