#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fill_hexs.py
A/B: wczytaj EGiB (działki)
C:   dołącz GeometricFeatures (wg add_transactions_data)
D:   wczytaj heksy wg hex.res
E:   intersekcja + agregacje (mean ważone polem + dominanta 'jednostka')
     → zapisz do hex.DzialkaEwidencyjna_r{res}
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, List, Dict, Any, Literal

import argparse
import duckdb
import pandas as pd
import geopandas as gpd
import numpy as np
from loguru import logger
from omegaconf import OmegaConf, DictConfig
from shapely import to_wkb, set_srid  # Shapely >= 2.0

# Twój helper do loggera
from src.common.io_utils import setup_logging


# --------------------- Helpers --------------------- #

def _sel(cfg: DictConfig, path: str, default=None):
    cur = cfg
    for part in path.split("."):
        if cur is None or part not in cur:
            return default
        cur = cur[part]
    return cur


# --------------------- DuckDB --------------------- #

def connect_duckdb(cfg: DictConfig) -> duckdb.DuckDBPyConnection:
    db_path = Path(_sel(cfg, "data.duckdb_path")).expanduser()
    logger.info(f"Łączenie z DuckDB: {db_path}")
    con = duckdb.connect(str(db_path))
    try:
        con.execute("LOAD spatial;")
    except duckdb.CatalogException:
        con.execute("INSTALL spatial;")
        con.execute("LOAD spatial;")
    logger.debug("Spatial extension gotowe.")
    return con


def _table_info(con: duckdb.DuckDBPyConnection, table: str) -> pd.DataFrame:
    return con.execute(f"PRAGMA table_info('{table}')").df()


def _read_table_no_geom(
    con: duckdb.DuckDBPyConnection,
    table: str,
    select_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    info = _table_info(con, table)
    names = info["name"].astype(str)
    types = info["type"].astype(str).str.upper()
    non_geom = list(names[~types.str.contains("GEOMETRY")])
    cols = [c for c in (select_cols or non_geom) if c in non_geom]
    sel = ", ".join(f'"{c}"' for c in cols) if cols else "*"
    return con.execute(f"SELECT {sel} FROM {table}").df()


# --------------------- A/B: EGiB --------------------- #

def load_egib_parcels_gdf(
    con: duckdb.DuckDBPyConnection,
    table: str,
    geom_col: str = "geometry",
    limit: Optional[int] = None,
) -> gpd.GeoDataFrame:
    logger.info(f"[A/B] EGiB: {table} (geom={geom_col}, limit={limit})")
    lim = f" LIMIT {int(limit)}" if (isinstance(limit, int) and limit > 0) else ""
    sql = f"""
        SELECT t.*, ST_AsWKB({geom_col}) AS geom_wkb
        FROM {table} AS t
        {lim}
    """
    df = con.execute(sql).df()
    geoms = gpd.GeoSeries.from_wkb(df.pop("geom_wkb").map(lambda b: bytes(b) if b is not None else None),
                                   crs="EPSG:2180")
    gdf = gpd.GeoDataFrame(df, geometry=geoms, crs="EPSG:2180")
    logger.success(f"[A/B] Załadowano {len(gdf)} rekordów EGiB.")
    return gdf


# --------------------- C: JOIN GeometricFeatures --------------------- #

def _dedupe_by_keys(
    df: pd.DataFrame,
    keys: List[str],
    strategy: str = "first",
    reduce_map: Optional[Dict[str, str]] = None,
    order_by: Optional[str] = None,
) -> pd.DataFrame:
    strategy = (strategy or "first").lower()

    if strategy in ("first", "last"):
        if order_by and order_by in df.columns:
            asc = (strategy == "first")
            df_sorted = df.sort_values(by=order_by, ascending=asc, kind="mergesort")
            return df_sorted.drop_duplicates(subset=keys, keep="first")
        keep = "first" if strategy == "first" else "last"
        return df.drop_duplicates(subset=keys, keep=keep)

    # strategia redukcji: użyj reduce_map dla wskazanych kolumn; dla reszty sensowne domyślne
    num = df.select_dtypes("number").columns.difference(keys)
    other = [c for c in df.columns if c not in set(keys) | set(num)]

    agg: Dict[str, Any] = {}
    reduce_map = reduce_map or {}

    # najpierw kolumny numer., które mają własny reducer (np. cena → 'max')
    for c, fn in reduce_map.items():
        if c in df.columns:
            agg[c] = fn

    # pozostałe numeryczne (bez tych już zmapowanych) → domyślnie 'mean'
    for c in num:
        if c not in agg:
            agg[c] = "mean"

    # nienumeryczne → deterministycznie 'first'
    for c in other:
        agg[c] = "first"

    return df.groupby(keys, dropna=False, as_index=False).agg(agg)


def _apply_prefix(df: pd.DataFrame, prefix: str, keys: List[str]) -> pd.DataFrame:
    if not prefix:
        return df
    rename = {c: f"{prefix}{c}" for c in df.columns if c not in set(keys)}
    return df.rename(columns=rename)


def join_parcels_with_gf(
    gdf_parcels: gpd.GeoDataFrame,
    con: duckdb.DuckDBPyConnection,
    cfg: DictConfig,
) -> gpd.GeoDataFrame:
    task = _sel(cfg, "pipeline.add_transactions_data", {}) or {}
    if not bool(task.get("enabled", False)):
        logger.info("[C] JOIN wyłączony – pomijam.")
        return gdf_parcels

    right_table = task.get("join_with")
    if not right_table:
        raise ValueError("[C] Brakuje 'add_transactions_data.join_with' w configu.")

    join_cols   = list(task.get("join_columns") or [])
    join_how    = (task.get("join_how") or "left").lower()
    prefix      = task.get("join_prefix") or ""
    select_cols = task.get("join_select")
    dedupe      = (task.get("join_dedupe", {}) or {}).get("strategy", "first")
    reduce_map  = (task.get("join_dedupe", {}) or {}).get("reduce_map", None)  # NEW

    # NEW: prawe klucze (alternatywne nazwy kolumn po prawej stronie)
    right_on = list(task.get("join_right_on_cols") or task.get("join_right_on") or [])

    if not join_cols:
        join_cols = ["iddzialki", _sel(cfg, "egib.year_col", "year")]

    # jeśli nie podano prawych kluczy, użyj tych samych nazw co po lewej
    if not right_on:
        right_on = join_cols

    logger.info(f"[C] JOIN {right_table} po L:{join_cols} ↔ P:{right_on} (how={join_how})")

    right = _read_table_no_geom(con, right_table, select_cols=select_cols)
    # deduplikujemy po PRAWYCH nazwach (takich, jakie są w right)
    right = _dedupe_by_keys(right, right_on, strategy=dedupe, reduce_map=reduce_map)  # NEW arg
    # prefiksujemy nie-klucze, klucze po PRAWEJ zostają bez prefiksu
    right = _apply_prefix(right, prefix, right_on)

    # MERGE z jawnie wskazanymi lewymi/prawymi kolumnami kluczy
    out = gdf_parcels.merge(right, how=join_how, left_on=join_cols, right_on=right_on)

    logger.success(f"[C] Po JOIN: {out.shape[0]} wierszy, {out.shape[1]} kolumn.")
    return out

# --------------------- D: HEX load --------------------- #

def _detect_srid(con: duckdb.DuckDBPyConnection, table: str, geom_col: str) -> Optional[int]:
    try:
        v = con.execute(f"SELECT ST_SRID({geom_col}) FROM {table} WHERE {geom_col} IS NOT NULL LIMIT 1").fetchone()
        return int(v[0]) if v and v[0] else None
    except Exception:
        return None


def load_hex_gdf(con: duckdb.DuckDBPyConnection, cfg: DictConfig, limit: Optional[int] = None) -> gpd.GeoDataFrame:
    # --- resolve table name ---
    table = _sel(cfg, "pipeline.hex.table", None)
    geom_col = _sel(cfg, "pipeline.hex.geom_col", "geometry")
    id_col   = _sel(cfg, "pipeline.hex.id_col", "hex_id")
    decimals = int(_sel(cfg, "pipeline.geometry.layer_defaults.decimals", 4))
    fallback_crs = _sel(cfg, "pipeline.geometry.layer_defaults.enforce_crs", "EPSG:2180")

    logger.info(f"[D] Wczytuję heksy z tabeli: {table} (id_col='{id_col}')")

    srid = _detect_srid(con, table, geom_col)
    crs_str = f"EPSG:{srid}" if srid else fallback_crs

    lim = f" LIMIT {int(limit)}" if (isinstance(limit, int) and limit > 0) else ""
    sql = f"""
        SELECT
          "{id_col}" AS hex_id,
          ST_AsWKB({geom_col}) AS geom_wkb,
          ROUND(CAST(ST_Area({geom_col}) AS DOUBLE), {decimals}) AS hex_area_m2
        FROM {table}
        {lim}
    """
    df = con.execute(sql).df()
    geoms = gpd.GeoSeries.from_wkb(df.pop("geom_wkb").map(lambda b: bytes(b) if b is not None else None),
                                   crs=crs_str)
    gdf = gpd.GeoDataFrame(df, geometry=geoms, crs=crs_str)
    logger.success(f"[D] Heksy: {len(gdf)} komórek.")
    return gdf


# --------------------- E: Intersect + agregacje --------------------- #

GeometryOut = Literal["hex", "intersection"]

def intersect_and_aggregate_area_weighted(
    gdf_left: gpd.GeoDataFrame,
    gdf_hex: gpd.GeoDataFrame,
    year_col: str = "year",
    hex_id_col: str = "hex_id",
    dominant_col: Optional[str] = "jednostka",
    decimals: int = 3,
    min_cover_fraction: float = 0.0,
    hex_area_col: str = "hex_area_m2",
    geometry_out: GeometryOut = "hex",
    treat_zero_as_na: Optional[List[str]] = None,          # NEW
    extra_nonzero_mean_cols: Optional[List[str]] = None,    # NEW
) -> gpd.GeoDataFrame:
    if gdf_left.crs != gdf_hex.crs:
        gdf_left = gdf_left.to_crs(gdf_hex.crs)

    treat_zero_as_na = set(treat_zero_as_na or [])
    extra_nonzero_mean_cols = set(extra_nonzero_mean_cols or [])

    base_cols = [c for c in [year_col, dominant_col] if c in gdf_left.columns and c]
    num_cols: List[str] = list(
        gdf_left.select_dtypes(include=[np.number]).columns.difference([year_col, dominant_col, hex_id_col])
    )
    left_use = gdf_left[base_cols + num_cols + ["geometry"]].copy()
    hex_use  = gdf_hex[[hex_id_col, "geometry", hex_area_col]].copy()

    logger.info(f"[E] Intersekcja + agregacje (dominanta={dominant_col}, min_cover_fraction={min_cover_fraction})")
    ix = gpd.overlay(left_use, hex_use, how="intersection")
    if ix.empty:
        return gpd.GeoDataFrame(columns=[hex_id_col, year_col] + [f"{c}_mean" for c in num_cols]
                                         + ([dominant_col] if dominant_col else []) + ["geometry"],
                                geometry=[], crs=gdf_hex.crs)

    ix["__ix_area"] = ix.geometry.area.astype(float)
    if min_cover_fraction:
        ix = ix[ix["__ix_area"] / ix[hex_area_col] >= float(min_cover_fraction)]
        if ix.empty:
            return gpd.GeoDataFrame(columns=[hex_id_col, year_col, "geometry"], geometry=[], crs=gdf_hex.crs)

    group_keys = [hex_id_col, year_col]
    g = ix.groupby(group_keys, dropna=False)

    # szkielet
    out = g.size().reset_index()[group_keys]

    # średnie ważone polem
    for col in num_cols:
        vals = ix[col]
        # 0 → NaN dla kolumn wskazanych (nie wliczaj zer do średniej)
        if col in treat_zero_as_na:
            vals = vals.mask(vals == 0, np.nan)

        w   = ix["__ix_area"].where(vals.notna(), 0.0)
        vw  = vals.fillna(0.0) * w
        s_w  = g["__ix_area"].sum()
        s_vw = vw.groupby([ix[hex_id_col], ix[year_col]]).sum()
        mean = (s_vw / s_w.replace(0.0, np.nan)).rename(f"{col}_mean").reset_index()
        out  = out.merge(mean, on=group_keys, how="left")

        # dodatkowo: wariant „bez zer” (mean tylko z vals!=0)
        if col in extra_nonzero_mean_cols:
            mask_nz = vals.notna() & (vals != 0)
            w_nz  = ix["__ix_area"].where(mask_nz, 0.0)
            vw_nz = vals.where(mask_nz, 0.0) * w_nz
            s_w_nz  = w_nz.groupby([ix[hex_id_col], ix[year_col]]).sum()
            s_vw_nz = vw_nz.groupby([ix[hex_id_col], ix[year_col]]).sum()
            mean_nz = (s_vw_nz / s_w_nz.replace(0.0, np.nan)).rename(f"{col}_mean_nz").reset_index()
            out     = out.merge(mean_nz, on=group_keys, how="left")

    # dominanta kategoryczna
    if dominant_col and dominant_col in ix.columns:
        cat = ix.dropna(subset=[dominant_col]).groupby(group_keys + [dominant_col])["__ix_area"].sum().reset_index()
        idx = cat.groupby(group_keys)["__ix_area"].idxmax()
        out = out.merge(cat.loc[idx, group_keys + [dominant_col]], on=group_keys, how="left")

    mean_cols = [c for c in out.columns if c.endswith("_mean")]
    if mean_cols:
        out[mean_cols] = out[mean_cols].round(decimals)
    nz_cols = [c for c in out.columns if c.endswith("_mean_nz")]
    if nz_cols:
        out[nz_cols] = out[nz_cols].round(decimals)

    if geometry_out == "hex":
        geom_df = gdf_hex[[hex_id_col, "geometry"]]
        gout = geom_df.merge(out, on=hex_id_col, how="right")
    else:
        ix_diss = ix.dissolve(by=group_keys, as_index=False, aggfunc="sum")[group_keys + ["geometry"]]
        gout = ix_diss.merge(out, on=group_keys, how="right")

    gout = gpd.GeoDataFrame(gout, geometry="geometry", crs=gdf_hex.crs)
    logger.success(f"[E] Wynik: {len(gout)} wierszy.")
    return gout


# --------------------- Save (EWKB → GEOMETRY) --------------------- #

def save_geodf_as_ewkb_geometry(
    db_path: Path,
    gdf: gpd.GeoDataFrame,
    table: str,
    *,
    srid: int = 2180,
    geom_col: str = "geometry",
    write_mode: str = "replace",        # 'replace' | 'append' | 'create'
    casts: Optional[Dict[str, str]] = None,
) -> int:
    # EWKB z osadzonym SRID → ST_GeomFromWKB w DuckDB
    df = gdf.copy()
    df["geom_wkb"] = df[geom_col].apply(lambda g: to_wkb(set_srid(g, srid), include_srid=True) if g is not None else None)
    df = df.drop(columns=[geom_col])

    casts = casts or {}
    cols_sql = ", ".join(
        (f'"{c}"::{casts[c]} AS "{c}"' if c in casts else f'"{c}"')
        for c in df.columns if c != "geom_wkb"
    )
    geom_sql = 'ST_GeomFromWKB(geom_wkb) AS "geometry"'
    select_sql = f'SELECT {(cols_sql + ", ") if cols_sql else ""}{geom_sql} FROM __tmp__'

    schema = table.split(".")[0] if "." in table else "main"
    with duckdb.connect(str(db_path)) as con:
        con.execute("LOAD spatial;")
        schema = table.split(".")[0] if "." in table else "main"
        schema_clean = schema.replace('"', "")  # lub: schema.strip('"')

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
        logger.success(f"[SAVE] {table}: {n} wierszy (SRID={srid}).")
        return int(n)


# --------------------- Runner --------------------- #

def run_add_transactions_hex(cfg: DictConfig) -> int:
    con = connect_duckdb(cfg)

    gdf_dzialki = load_egib_parcels_gdf(
        con,
        table=_sel(cfg, "egib.table", "egib.DzialkaEwidencyjna"),
        geom_col=_sel(cfg, "egib.geom_col", "geometry"),
    )
    gdf_joined = join_parcels_with_gf(gdf_dzialki, con, cfg)
    gdf_hex    = load_hex_gdf(con, cfg)

    year_col   = _sel(cfg, "egib.year_col", "year")
    hex_id_col = _sel(cfg, "hex.id_col", "hex_id")

    gdf_hex_year = intersect_and_aggregate_area_weighted(
        gdf_left=gdf_joined,
        gdf_hex=gdf_hex,
        year_col=year_col,
        hex_id_col=hex_id_col,
        dominant_col="jednostka",
        decimals=int(_sel(cfg, "layer_defaults.decimals", 3)),
        min_cover_fraction=float(_sel(cfg, "layer_defaults.min_cover_fraction", 0.0)),
        hex_area_col="hex_area_m2",
        geometry_out="hex",
        treat_zero_as_na=_sel(cfg, "pipeline.add_transactions_data.treat_zero_as_na", ["tx_cena"]),        # NEW
        extra_nonzero_mean_cols=_sel(cfg, "pipeline.add_transactions_data.extra_nonzero_mean_cols", ["tx_cena"]),  # NEW
    )
    gdf_hex_year.drop(columns=["tx_year_mean", 'tx_udzial_mean'], inplace=True, errors="ignore")

    out_table = _sel(cfg, "pipeline.add_transactions_data.out_table",
                     f'{_sel(cfg,"pipeline.hex.schema","hex")}.Transakcje{_sel(cfg,"pipeline.hex.out_suffix","r9")}')

    return save_geodf_as_ewkb_geometry(
        db_path=Path(_sel(cfg, "data.duckdb_path")),
        gdf=gdf_hex_year,
        table=out_table,
        srid=2180,
        geom_col="geometry",
        write_mode=_sel(cfg, "layer_defaults.write_mode", "replace"),
        casts={hex_id_col: "VARCHAR", year_col: "INT"},
    )