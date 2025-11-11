# features/add_parcels_data_hexs.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fill_hexs.py
A/B: wczytaj EGiB (działki)
C:   dołącz GeometricFeatures (wg add_parcels_data)
D:   wczytaj heksy wg hex.res
E:   intersekcja + agregacje (mean ważone polem + dominanta 'jednostka')
     → zapisz do hex.DzialkaEwidencyjna_r{res}
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, List, Dict, Any, Literal, Tuple

import argparse
import duckdb
import pandas as pd
import geopandas as gpd
import numpy as np
from loguru import logger
from omegaconf import OmegaConf, DictConfig
from shapely import to_wkb, set_srid, make_valid, set_precision  

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
    logger.info("Spatial extension gotowe.")
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

def _dedupe_by_keys(df: pd.DataFrame, keys: List[str], strategy: str = "first") -> pd.DataFrame:
    if strategy in ("first", "last"):
        return df.drop_duplicates(subset=keys, keep=("first" if strategy == "first" else "last"))
    # prosto: średnia po liczbowych, pierwszy dla pozostałych
    num = df.select_dtypes("number").columns.difference(keys)
    agg = {**{c: "first" for c in df.columns if c not in set(keys) | set(num)},
           **{c: "mean" for c in num}}
    return df.groupby(keys, dropna=False, as_index=False).agg(agg)


def _apply_prefix(df: pd.DataFrame, prefix: str, keys: List[str]) -> pd.DataFrame:
    if not prefix:
        return df
    rename = {c: f"{prefix}{c}" for c in df.columns if c not in set(keys)}
    return df.rename(columns=rename)

def diagnose_hex_coverage_vs_parcels_union(
    gdf_parcels: gpd.GeoDataFrame,
    gdf_hex: gpd.GeoDataFrame,
    *,
    hex_id_col: str = "hex_id",
    hex_area_col: str = "hex_area_m2",
    precision_grid: float = 0.001,
    decimals: int = 3,
) -> pd.DataFrame:
    """
    Compute per-hex coverage of the union of all parcel geometries.

    Returns a DataFrame with:
      - hex_id
      - parcels_union_in_hex_m2
      - cov_union_anyyear  (fraction of hex covered by the parcels union)

    Notes
    -----
    - CRS is aligned to hex CRS.
    - Geometries are made valid and snapped to a precision grid to stabilize area sums.
    """
    if gdf_parcels.crs != gdf_hex.crs:
        gdf_parcels = gdf_parcels.to_crs(gdf_hex.crs)

    # Clean + precision snap
    left = gdf_parcels.copy()
    left.geometry = left.geometry.map(make_valid)
    try:
        left.geometry = left.geometry.map(lambda g: set_precision(g, precision_grid))
    except Exception:
        pass

    hex_use = gdf_hex[[hex_id_col, hex_area_col, "geometry"]].copy()
    hex_use.geometry = hex_use.geometry.map(make_valid)
    try:
        hex_use.geometry = hex_use.geometry.map(lambda g: set_precision(g, precision_grid))
    except Exception:
        pass

    # Union of all parcels (single geometry)
    parcels_union = left.unary_union
    if parcels_union.is_empty:
        logger.warning("[DIAG] Parcels union is empty. Coverage will be 0.")
        out = hex_use[[hex_id_col, hex_area_col]].copy()
        out["parcels_union_in_hex_m2"] = 0.0
        out["cov_union_anyyear"] = 0.0
        return out

    union_gdf = gpd.GeoDataFrame({"geometry": [parcels_union]}, crs=gdf_hex.crs)

    # Intersect per-hex with parcels union
    ix = gpd.overlay(hex_use, union_gdf, how="intersection", keep_geom_type=True)
    if ix.empty:
        out = hex_use[[hex_id_col, hex_area_col]].copy()
        out["parcels_union_in_hex_m2"] = 0.0
        out["cov_union_anyyear"] = 0.0
        return out

    ix["ix_area"] = ix.geometry.area.astype(float)
    agg = (
        ix.groupby(hex_id_col, dropna=False)["ix_area"]
          .sum()
          .rename("parcels_union_in_hex_m2")
          .reset_index()
    )

    out = hex_use[[hex_id_col, hex_area_col]].merge(agg, on=hex_id_col, how="left")
    out["parcels_union_in_hex_m2"] = out["parcels_union_in_hex_m2"].fillna(0.0)
    out["cov_union_anyyear"] = (
        out["parcels_union_in_hex_m2"] / out[hex_area_col].replace(0.0, np.nan)
    ).clip(0.0, 1.0).round(decimals)

    return out
    
def build_parcel_hex_table(
    *,
    gdf_left: gpd.GeoDataFrame,
    gdf_hex: gpd.GeoDataFrame,
    year_col: str = "year",
    hex_id_col: str = "hex_id",
    hex_area_col: str = "hex_area_m2",
    parcel_id_col: str = "iddzialki",
    min_cover_fraction: float = 0.0,
) -> Tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """
    Create a compact per-(hex, year, parcel) table with parcel area assigned to the hex.

    Returns:
      - ix:     intersection GeoDataFrame with "__ix_area" and hex area (diagnostics)
      - phex:   pandas.DataFrame grouped by (hex_id, year, parcel) with:
                * parcel_area_in_hex  – sum of intersection area for a parcel within a hex
                * n_split             – number of pieces per (hex, year) – merged later at group level

    Notes
    -----
    - CRS is aligned to hex CRS.
    - min_cover_fraction filters out tiny intersections by (intersection_area / hex_area).
    """
    # Align CRS
    if gdf_left.crs != gdf_hex.crs:
        gdf_left = gdf_left.to_crs(gdf_hex.crs)

    # Columns we need to carry through overlay
    carry_cols: List[str] = [c for c in [year_col, parcel_id_col] if c in gdf_left.columns]

    left_use = gdf_left[carry_cols + ["geometry"]].copy()
    hex_use  = gdf_hex[[hex_id_col, hex_area_col, "geometry"]].copy()

    # Intersection
    ix = gpd.overlay(left_use, hex_use, how="intersection", keep_geom_type=True)
    if ix.empty:
        # Shape of the empty return to keep code simple downstream
        empty_ix = gpd.GeoDataFrame(columns=[*carry_cols, hex_id_col, hex_area_col, "geometry"], geometry=[], crs=gdf_hex.crs)
        empty_phex = pd.DataFrame(columns=[hex_id_col, year_col, parcel_id_col, "parcel_area_in_hex"])
        return empty_ix, empty_phex

    ix["__ix_area"] = ix.geometry.area.astype(float)

    # Optional coverage filter (area/hex)
    if min_cover_fraction:
        thr = float(min_cover_fraction)
        ix = ix[ix["__ix_area"] / ix[hex_area_col] >= thr]
        if ix.empty:
            empty_ix = gpd.GeoDataFrame(columns=[*carry_cols, hex_id_col, hex_area_col, "geometry"], geometry=[], crs=gdf_hex.crs)
            empty_phex = pd.DataFrame(columns=[hex_id_col, year_col, parcel_id_col, "parcel_area_in_hex"])
            return empty_ix, empty_phex

    # Count geometry splits per (hex, [year]) – diagnostic
    group_keys = [hex_id_col] + ([year_col] if year_col in ix.columns else [])
    n_split = ix.groupby(group_keys, dropna=False).size().rename("n_split").reset_index()

    # Aggregate intersections to parcel level within hex
    need_cols = [c for c in [hex_id_col, year_col, parcel_id_col] if c in ix.columns]
    phex = (
        ix.groupby(need_cols, dropna=False)["__ix_area"]
          .sum()
          .rename("parcel_area_in_hex")
          .reset_index()
    )

    # Attach n_split to phex at (hex, year) level
    phex = phex.merge(n_split, on=group_keys, how="left")

    # Add hex area (will be used for coverage)
    phex = phex.merge(
        ix[[hex_id_col, hex_area_col]].drop_duplicates(),
        on=hex_id_col,
        how="left"
    )

    return ix, phex


def join_parcels_with_gf(
    gdf_parcels: gpd.GeoDataFrame,
    con: duckdb.DuckDBPyConnection,
    cfg: DictConfig,
) -> gpd.GeoDataFrame:
    task = _sel(cfg, "pipeline.add_parcels_data", {}) or {}

    if not bool(task.get("enabled", False)):
        logger.info("[C] JOIN wyłączony – pomijam.")
        return gdf_parcels

    right_table = task.get("join_with")
    if not right_table:
        raise ValueError("[C] Brakuje 'add_parcels_data.join_with' w configu.")

    join_cols   = list(task.get("join_columns") or [])
    join_how    = (task.get("join_how") or "left").lower()
    prefix      = task.get("join_prefix") or ""
    select_cols = task.get("join_select")
    dedupe      = (task.get("join_dedupe", {}) or {}).get("strategy", "first")

    if not join_cols:
        join_cols = ["iddzialki", _sel(cfg, "egib.year_col", "year")]

    logger.info(f"[C] JOIN {right_table} po {join_cols} (how={join_how})")
    right = _read_table_no_geom(con, right_table, select_cols=select_cols)
    right = _dedupe_by_keys(right, join_cols, strategy=dedupe)
    right = _apply_prefix(right, prefix, join_cols)

    out = gdf_parcels.merge(right, how=join_how, on=join_cols)
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
    table = _sel(cfg, "pipeline.hex.table", None)
    geom_col = _sel(cfg, "pipeline.hex.geom_col", "geometry")
    id_col   = _sel(cfg, "pipeline.hex.id_col", "hex_id")
    fallback_crs = _sel(cfg, "pipeline.layer_defaults.enforce_crs", "EPSG:2180")

    logger.info(f"[D] Wczytuję heksy z tabeli: {table} (id_col='{id_col}')")

    srid = _detect_srid(con, table, geom_col)
    crs_str = f"EPSG:{srid}" if srid else fallback_crs

    lim = f" LIMIT {int(limit)}" if (isinstance(limit, int) and limit > 0) else ""
    # ⬇️ bez ST_Area — tylko WKB + id
    sql = f"""
        SELECT
          "{id_col}" AS hex_id,
          ST_AsWKB({geom_col}) AS geom_wkb
        FROM {table}
        {lim}
    """
    df = con.execute(sql).df()
    geoms = gpd.GeoSeries.from_wkb(df.pop("geom_wkb").map(lambda b: bytes(b) if b is not None else None), crs=crs_str)
    gdf = gpd.GeoDataFrame(df, geometry=geoms, crs=crs_str)

    # 🔧 Always re-project geometry to EPSG:2180 so all later steps work in metres
    if gdf.crs != "EPSG:2180":
        logger.warning("[D] Re-projecting hex grid %s ➜ EPSG:2180 (required).", gdf.crs)
        gdf = gdf.to_crs("EPSG:2180")

    # (Re)calculate area **after** the final CRS is set
    gdf["hex_area_m2"] = gdf.geometry.area.astype(float)

    logger.success(f"[D] Hexes: {len(gdf)} cells in EPSG:2180.")
    return gdf

def aggregate_hex_year_stats_from_parcel_hex(
    phex: pd.DataFrame,
    *,
    year_col: str = "year",
    hex_id_col: str = "hex_id",
    hex_area_col: str = "hex_area_m2",
    small_thr_m2: float = 1000.0,   # 0.1 ha
    quantiles: Tuple[float, float] = (0.25, 0.75),
) -> pd.DataFrame:
    """
    From per-(hex, year, parcel) areas compute hex-year stats:
      - n_parcels, sum_area, median_area, q25_area, q75_area
      - p_small_cnt  (share of parcels with area < threshold)
      - p_small_area (share of area held by small parcels)
      - coverage_hex (sum_area / hex_area_m2)
      - n_split      (diagnostic; from phex)
    """
    req = {hex_id_col, year_col, "parcel_area_in_hex", hex_area_col}
    missing = req - set(phex.columns)
    if missing:
        raise KeyError(f"Missing columns in phex: {sorted(missing)}")

    # Base group
    keys = [hex_id_col, year_col]
    grp = phex.groupby(keys, dropna=False)

    # Counts and sums
    n_parcels = grp["parcel_area_in_hex"].count().rename("n_parcels")
    sum_area  = grp["parcel_area_in_hex"].sum().rename("sum_area")

    # Median & quantiles
    med = grp["parcel_area_in_hex"].median().rename("median_area")
    q_low = grp["parcel_area_in_hex"].quantile(quantiles[0]).rename("q25_area")
    q_hi  = grp["parcel_area_in_hex"].quantile(quantiles[1]).rename("q75_area")

    # Small parcel shares
    small_mask = phex["parcel_area_in_hex"] < float(small_thr_m2)
    p_small_cnt = grp.apply(
        lambda g: (g["parcel_area_in_hex"] < float(small_thr_m2)).mean() if len(g) else np.nan,
        include_groups=False,
    ).rename("p_small_cnt")

    small_area_sum = grp.apply(
        lambda g: g.loc[g["parcel_area_in_hex"] < float(small_thr_m2), "parcel_area_in_hex"].sum(),
        include_groups=False,
    ).rename("small_area_sum")

    p_small_area = (small_area_sum / sum_area.replace(0.0, np.nan)).rename("p_small_area")

    # coverage and n_split (diagnostic)
    # hex_area_m2 is constant per hex; join once then compute ratio
    hex_area = phex[[hex_id_col, hex_area_col]].drop_duplicates().set_index(hex_id_col)[hex_area_col]
    out = pd.concat([n_parcels, sum_area, med, q_low, q_hi, p_small_cnt, p_small_area], axis=1).reset_index()

    out = out.merge(
        hex_area.rename_axis(hex_id_col).reset_index(),
        on=hex_id_col, how="left"
    )
    out["coverage_hex"] = (out["sum_area"] / out[hex_area_col].replace(0.0, np.nan)).clip(upper=1.0)

    # n_split at (hex, year) from phex (already merged earlier)
    if "n_split" in phex.columns:
        n_split = grp["n_split"].max().rename("n_split").reset_index()
        out = out.merge(n_split, on=keys, how="left")
    else:
        out["n_split"] = np.nan

    # Minimal contract for weak labels + diagnostics
    cols = [
        hex_id_col, year_col,
        "n_parcels", "sum_area", "median_area",
        "q25_area", "q75_area",
        "p_small_cnt", "p_small_area",
        "coverage_hex", "n_split"
    ]
    return out[cols]

# --------------------- E: Intersect + agregacje --------------------- #

GeometryOut = Literal["hex", "intersection"]
def intersect_and_aggregate_area_weighted(  # noqa: C901
    gdf_left: gpd.GeoDataFrame,
    gdf_hex: gpd.GeoDataFrame,
    *,
    year_col: str = "year",
    hex_id_col: str = "hex_id",
    dominant_col: Optional[str] = "jednostka",
    decimals: int = 3,
    min_cover_fraction: float = 0.0,
    hex_area_col: str = "hex_area_m2",
    geometry_out: GeometryOut = "hex",
    parcel_id_col: str = "iddzialki",
    area_check_tol: float = 1e-6,
    new_cols_prefix: str = "gf_",
) -> gpd.GeoDataFrame:
    """
    Intersect parcels with hexagons and compute area-weighted stats per (hex[, year]).

    Key properties:
    - All *area* computations are performed in EPSG:2180 (meters) to avoid degree^2 vs m^2 issues.
    - Adds dense diagnostics via logger (sizes, CRS, dropped fragments, area conservation).
    - Supports optional dominant category and weighted means by parcel.

    Parameters
    ----------
    gdf_left : GeoDataFrame
        Parcels (may contain `year_col`, `parcel_id_col`, `dominant_col`, numeric features).
    gdf_hex : GeoDataFrame
        Hex polygons with `{hex_id_col}` and `{hex_area_col}` (m²). If `{hex_area_col}` missing, it is computed on the fly.
    year_col : str
        Year column in `gdf_left`. If present and not all-NA, aggregation is per (hex, year).
    hex_id_col : str
        Hex id column name.
    dominant_col : Optional[str]
        Name of categorical column to compute dominant class by area.
    decimals : int
        Rounding for mean and diagnostic columns.
    min_cover_fraction : float
        Drop fragments with (fragment_area / hex_area) < threshold.
    hex_area_col : str
        Hex area column name (m²).
    geometry_out : Literal["hex","intersection"]
        Output geometry: hex polygon or dissolved intersection geometry.
    parcel_id_col : str
        Parcel id column used for parcel-weighted means.
    area_check_tol : float
        Relative tolerance for area conservation check between cut sum and union area.
    new_cols_prefix : str
        Prefix for diagnostic columns.

    Returns
    -------
    GeoDataFrame
        Aggregated table with geometry and diagnostics.
    """
    logger.info("[E] START intersect_and_aggregate_area_weighted")
    logger.info(f"[E] Input left: rows={len(gdf_left)}, CRS={gdf_left.crs}")
    logger.info(f"[E] Input hex : rows={len(gdf_hex)}, CRS={gdf_hex.crs}")

    # --- CRS alignment (keep output in hex CRS) --------------------------- #
    if gdf_left.crs != gdf_hex.crs:
        logger.info("[E] Reproject left → hex CRS (%s)", gdf_hex.crs)
        gdf_left = gdf_left.to_crs(gdf_hex.crs)

    # --- Geometry hygiene ------------------------------------------------ #
    gdf_left = gdf_left.copy()
    gdf_hex  = gdf_hex.copy()

    def _invalid_cnt(gdf: gpd.GeoDataFrame) -> int:
        """
        Count invalid geometries: NA or empty.
        Uses elementwise boolean ops and sums them; avoids casting the whole Series to int.
        """
        # Missing geometry column? Treat as all invalid.
        if "geometry" not in gdf.columns:
            return int(len(gdf))

        s_na = gdf.geometry.isna()
        # .is_empty may raise if geometry contains Nones; guard + fill
        try:
            s_empty = gdf.geometry.is_empty
            # In some GeoPandas versions .is_empty can be object-dtype; coerce to bools
            if hasattr(s_empty, "fillna"):
                s_empty = s_empty.fillna(True)
        except Exception:
            # If we cannot evaluate emptiness, assume not empty where geometry exists
            s_empty = pd.Series(False, index=gdf.index)

        return (s_na | s_empty).astype("int64").sum()

    inv_l_before = _invalid_cnt(gdf_left)
    inv_h_before = _invalid_cnt(gdf_hex)

    gdf_left.geometry = gdf_left.geometry.map(make_valid)
    gdf_hex.geometry  = gdf_hex.geometry.map(make_valid)
    try:
        grid = 0.001  # meters
        gdf_left.geometry = gdf_left.geometry.map(lambda g: set_precision(g, grid))
        gdf_hex.geometry  = gdf_hex.geometry.map(lambda g: set_precision(g, grid))
    except Exception:
        logger.info("[E] set_precision unavailable; skipping.")
    inv_l_after = _invalid_cnt(gdf_left)
    inv_h_after = _invalid_cnt(gdf_hex)
    logger.info("[E] Invalid geometries fixed: left %d→%d, hex %d→%d",
                inv_l_before, inv_l_after, inv_h_before, inv_h_after)


    # --- Group keys ------------------------------------------------------ #
    group_keys: List[str] = [hex_id_col]
    if year_col and (year_col in gdf_left.columns) and (not gdf_left[year_col].isna().all()):
        group_keys.append(year_col)
    logger.info("[E] group_keys=%s", group_keys)

    # --- Column selection ------------------------------------------------ #
    base_candidates = [year_col, dominant_col, parcel_id_col]
    base_cols = [c for c in base_candidates if c and c in gdf_left.columns]
    num_cols: List[str] = list(
        gdf_left.select_dtypes(include=[np.number]).columns.difference(base_candidates)
    )
    left_use = gdf_left[base_cols + ["geometry"] + num_cols].copy()
    hex_cols = [hex_id_col, "geometry"] + ([hex_area_col] if hex_area_col in gdf_hex.columns else [])
    hex_use  = gdf_hex[hex_cols].copy()
    logger.info("[E] left_use rows=%d, hex_use rows=%d", len(left_use), len(hex_use))

    # --- Deduplicate by geometry (post-JOIN duplicates) ------------------ #
    def _dedup_by_geom(df: gpd.GeoDataFrame, keys: List[str]) -> gpd.GeoDataFrame:
        before = len(df)
        tmp = df.copy()
        tmp["_wkb_"] = tmp.geometry.apply(lambda g: g.wkb if g is not None else None)
        tmp = tmp.drop_duplicates(subset=[*(keys or []), "_wkb_"])
        after = len(tmp)
        logger.info("[E] Dedup by geom(keys=%s): %d → %d (removed=%d)", keys, before, after, before - after)
        return tmp.drop(columns=["_wkb_"])

    left_use = _dedup_by_geom(left_use, [c for c in [parcel_id_col, year_col] if c in left_use.columns])
    hex_use  = _dedup_by_geom(hex_use, [hex_id_col])

    # --- Overlay (intersection) ------------------------------------------ #
    logger.info("[E] Overlay(intersection) ...")
    ix = gpd.overlay(left_use, hex_use, how="intersection", keep_geom_type=True)
    logger.info("[E] Overlay done: fragments=%d", len(ix))
    if ix.empty:
        empty_cols = (
            group_keys
            + [f"{c}_mean" for c in num_cols]
            + (["n_split", "n_parcels"] if parcel_id_col in left_use.columns else ["n_split"])
            + ([dominant_col] if dominant_col else [])
            + [f"{new_cols_prefix}cut_area_sum",
               f"{new_cols_prefix}union_area",
               f"{new_cols_prefix}coverage_from_union",
               f"{new_cols_prefix}coverage_from_cut",
               f"{new_cols_prefix}area_conservation_rel_err",
               f"{new_cols_prefix}area_conservation_ok"]
            + ["geometry"]
        )
        logger.warning("[E] Empty overlay result.")
        return gpd.GeoDataFrame(columns=empty_cols, geometry=[], crs=gdf_hex.crs)

    # --- Areas in EPSG:2180 (compute-only reprojection) ------------------ #
    ix_m = ix.to_crs("EPSG:2180")
    ix["__ix_area"] = ix_m.geometry.area.astype(float)

    if hex_area_col not in ix.columns:
        hex_m = hex_use.to_crs("EPSG:2180")
        area_map = dict(zip(hex_m[hex_id_col], hex_m.geometry.area.astype(float)))
        ix[hex_area_col] = ix[hex_id_col].map(area_map).astype(float)

    # Fragment area stats (pre-filter)
    def _q(s: pd.Series, p: float) -> float:
        return float(np.nanquantile(s.values, p)) if len(s) else np.nan

    logger.info("[E] Fragment area stats (m²) pre-filter: "
                 "min=%.2f, q05=%.2f, q25=%.2f, med=%.2f, q75=%.2f, q95=%.2f, max=%.2f, sum=%.2f",
                 float(ix["__ix_area"].min()),
                 _q(ix["__ix_area"], 0.05),
                 _q(ix["__ix_area"], 0.25),
                 float(ix["__ix_area"].median()),
                 _q(ix["__ix_area"], 0.75),
                 _q(ix["__ix_area"], 0.95),
                 float(ix["__ix_area"].max()),
                 float(ix["__ix_area"].sum()))

    # --- Coverage filter -------------------------------------------------- #
    dropped = 0
    if min_cover_fraction:
        thr = float(min_cover_fraction)
        mask = (ix["__ix_area"] / ix[hex_area_col]) >= thr
        dropped = int((~mask).sum())
        kept = int(mask.sum())
        logger.info("[E] min_cover_fraction=%.8f → kept=%d, dropped=%d (%.2f%% dropped)",
                     thr, kept, dropped, 100.0 * dropped / max(1, kept + dropped))
        ix = ix[mask]
        ix_m = ix_m.loc[ix.index]  # keep alignment
        if ix.empty:
            logger.warning("[E] No fragments after coverage filter.")
            return gpd.GeoDataFrame(columns=group_keys + ["geometry"], geometry=[], crs=gdf_hex.crs)

    # --- n_split per group ------------------------------------------------ #
    grp_ix = ix.groupby(group_keys, dropna=False)
    out = grp_ix.size().rename("n_split").reset_index()

    # --- Cut sum & Union area (both in m²) -------------------------------- #
    cut_area_sum = grp_ix["__ix_area"].sum().rename("__cut_area_sum").reset_index()

    ix_diss = ix.dissolve(by=group_keys, as_index=False, aggfunc="sum")[group_keys + ["geometry"]]
    ix_diss_m = ix_diss.to_crs("EPSG:2180")
    ix_diss["__union_area"] = ix_diss_m.geometry.area.astype(float)

    out = out.merge(cut_area_sum, on=group_keys, how="left")
    out = out.merge(ix_diss[group_keys + ["__union_area"]], on=group_keys, how="left")

    # Attach hex areas
    hex_area_df = ix[[hex_id_col, hex_area_col]].drop_duplicates()
    out = out.merge(hex_area_df, on=hex_id_col, how="left")

    # --- Coverage & area conservation ------------------------------------ #
    eps = 1e-12
    out["__coverage_from_union"] = (out["__union_area"] / out[hex_area_col].replace(0.0, np.nan)).clip(upper=1.0)
    out["__coverage_from_cut"]   = (out["__cut_area_sum"] / out[hex_area_col].replace(0.0, np.nan)).clip(upper=1.0)
    out["__area_conservation_rel_err"] = (
        (out["__cut_area_sum"] - out["__union_area"]).abs() / np.maximum(out["__union_area"].abs(), eps)
    )
    out["__area_conservation_ok"] = out["__area_conservation_rel_err"] <= float(area_check_tol)

    bad = out[~out["__area_conservation_ok"]].copy()
    if not bad.empty:
        bad = bad.assign(err=out.loc[bad.index, "__area_conservation_rel_err"]).sort_values("err", ascending=False)
        top = bad.head(5)
        logger.warning("[E] Area conservation failed for %d group(s). Top 5:", len(bad))
        for _, r in top.iterrows():
            _year = r.get(year_col, None)
            logger.warning("    %s | %s | err=%.6f | cut=%.2f | union=%.2f | hexA=%.2f",
                           r[hex_id_col], _year, r["__area_conservation_rel_err"],
                           r["__cut_area_sum"], r["__union_area"], r[hex_area_col])
    else:
        logger.info("[E] Area conservation OK for all groups (tol=%.1e).", area_check_tol)

    # --- Parcel-weighted means ------------------------------------------ #
    if parcel_id_col in ix.columns:
        # per-fragment area (already EPSG:2180)
        ix_m["__frag_area_m2"] = ix_m.geometry.area.astype(float)

        # total parcel area inside hex per (hex[,year], parcel)
        parcel_group_keys: List[str] = group_keys + [parcel_id_col]
        parcel_area_in_hex: pd.Series = (
            ix_m.groupby(parcel_group_keys)["__frag_area_m2"]
                .transform("sum")
                .astype(float)
        )

        # weights = fragment share in parcel area inside the hex
        den = parcel_area_in_hex.replace(0.0, np.nan)
        ix["__w_raw"] = (ix["__ix_area"] / den).clip(lower=0.0)

        w = ix["__w_raw"]
        if w.notna().any():
            logger.info("[E] Weights stats: n=%d, min=%.6f, p05=%.6f, med=%.6f, p95=%.6f, max=%.6f, zero_denoms=%d",
                         w.count(),
                         float(w.min()),
                         float(np.nanquantile(w, 0.05)),
                         float(np.nanmedian(w.values)),
                         float(np.nanquantile(w, 0.95)),
                         float(w.max()),
                         int(den.isna().sum()))
        else:
            logger.info("[E] Weights stats: all-NaN (denominators zero?)")

        parcel_hex = (
            ix.groupby(parcel_group_keys, dropna=False)
              .agg(__w_parcel=("__w_raw", "sum"),
                   **{col: (col, "first") for col in num_cols})
              .reset_index()
        )
        n_parcels = (
            parcel_hex.groupby(group_keys, dropna=False)[parcel_id_col]
            .nunique(dropna=True)
            .rename("n_parcels")
            .reset_index()
        )
        out = out.merge(n_parcels, on=group_keys, how="left")

        for col in num_cols:
            v = parcel_hex[col]
            w_par = parcel_hex["__w_parcel"].where(v.notna(), 0.0)
            by = [parcel_hex[k] for k in group_keys]
            num = (v.fillna(0.0) * w_par).groupby(by).sum()
            den = w_par.groupby(by).sum().replace(0.0, np.nan)
            mean = (num / den).rename(f"{col}_mean").reset_index()
            out = out.merge(mean, on=group_keys, how="left")
    else:
        area_sum = grp_ix["__ix_area"].sum().replace(0.0, np.nan)
        for col in num_cols:
            v = ix[col].astype(float)
            w = ix["__ix_area"].where(v.notna(), 0.0)
            by = [ix[k] for k in group_keys]
            num = (v.fillna(0.0) * w).groupby(by).sum()
            mean = (num / area_sum).rename(f"{col}_mean").reset_index()
            out = out.merge(mean, on=group_keys, how="left")

    # --- Dominant category ------------------------------------------------ #
    if dominant_col and dominant_col in ix.columns:
        cat = (
            ix.dropna(subset=[dominant_col])
              .groupby(group_keys + [dominant_col])["__ix_area"]
              .sum()
              .reset_index()
        )
        idx = cat.groupby(group_keys)["__ix_area"].idxmax()
        dom = cat.loc[idx, group_keys + [dominant_col]]
        out = out.merge(dom, on=group_keys, how="left")

    # --- Rounding & rename diagnostics ----------------------------------- #
    mean_cols = [c for c in out.columns if c.endswith("_mean")]
    if mean_cols:
        out[mean_cols] = out[mean_cols].round(decimals)
    out["__coverage_from_union"] = out["__coverage_from_union"].round(decimals)
    out["__coverage_from_cut"] = out["__coverage_from_cut"].round(decimals)
    out["__area_conservation_rel_err"] = out["__area_conservation_rel_err"].round(decimals)

    rename_map = {
        "__cut_area_sum": f"{new_cols_prefix}cut_area_sum",
        "__union_area": f"{new_cols_prefix}union_area",
        "__coverage_from_union": f"{new_cols_prefix}coverage_from_union",
        "__coverage_from_cut": f"{new_cols_prefix}coverage_from_cut",
        "__area_conservation_rel_err": f"{new_cols_prefix}area_conservation_rel_err",
        "__area_conservation_ok": f"{new_cols_prefix}area_conservation_ok",
    }
    out = out.rename(columns=rename_map)

    # Summary snapshot
    logger.info("[E] Groups=%d | n_split(min/med/max)=(%s,%s,%s) | cov_union(min/med/max)=(%.3f,%.3f,%.3f)",
                 len(out),
                 out["n_split"].min() if len(out) else None,
                 out["n_split"].median() if len(out) else None,
                 out["n_split"].max() if len(out) else None,
                 out[f"{new_cols_prefix}coverage_from_union"].min() if len(out) else float("nan"),
                 out[f"{new_cols_prefix}coverage_from_union"].median() if len(out) else float("nan"),
                 out[f"{new_cols_prefix}coverage_from_union"].max() if len(out) else float("nan"))

    # --- Geometry out ----------------------------------------------------- #
    if geometry_out == "hex":
        out = gdf_hex[[hex_id_col, "geometry"]].merge(out, on=hex_id_col, how="right")
    else:
        out = ix_diss.merge(out, on=group_keys, how="right")

    logger.info("[E] END intersect_and_aggregate_area_weighted → rows=%d, cols=%d", len(out), out.shape[1])
    return gpd.GeoDataFrame(out, geometry="geometry", crs=gdf_hex.crs)


# --------------------- Save (EWKB → GEOMETRY) --------------------- #

def save_hex_stats_table(
    db_path: Path,
    df: pd.DataFrame,
    table: str,
    *,
    hex_id_col: str = "hex_id",
    year_col: str = "year",
    casts: Optional[Dict[str, str]] = None,
    write_mode: Literal["replace", "append", "create"] = "replace",
) -> int:
    """
    Persist a plain (no-geometry) hex-year stats table to DuckDB.
    """
    casts = casts or {}
    with duckdb.connect(str(db_path)) as con:
        schema = table.split(".")[0] if "." in table else "main"
        con.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema.replace(chr(34), "")}";')

        # Prepare a view-like SELECT with explicit casts
        cols_sql = ", ".join(
            f'"{c}"::{casts[c]} AS "{c}"' if c in casts else f'"{c}"'
            for c in df.columns
        )
        try:
            con.unregister("__hexstats__")
        except Exception:
            pass
        con.register("__hexstats__", df)

        if write_mode == "replace":
            con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT {cols_sql} FROM __hexstats__")
        elif write_mode in ("create", "append"):
            con.execute(f"CREATE TABLE IF NOT EXISTS {table} AS SELECT {cols_sql} FROM __hexstats__ LIMIT 0")
            con.execute(f"INSERT INTO {table} SELECT {cols_sql} FROM __hexstats__")
        else:
            raise ValueError("write_mode must be 'replace' | 'append' | 'create'")

        n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        return int(n)


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

def run_add_parcels_data(cfg: DictConfig) -> int:
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

    diag = diagnose_hex_coverage_vs_parcels_union(
        gdf_parcels=gdf_joined,
        gdf_hex=gdf_hex,
        hex_id_col=hex_id_col,
        hex_area_col="hex_area_m2",
        precision_grid=0.001,
        decimals=int(_sel(cfg, "pipeline.layer_defaults.decimals", 3)),
    )

    # Podłącz do gdf_hex (przyda się dalej i do debugowania)
    gdf_hex = gdf_hex.merge(
        diag[[hex_id_col, "parcels_union_in_hex_m2", "cov_union_anyyear"]],
        on=hex_id_col, how="left"
    )

    # Krótki log ostrzegawczy — heksy „pustki”
    low_thr = float(_sel(cfg, "pipeline.layer_defaults.cov_diag_low_thr", 0.1))
    n_low = int((diag["cov_union_anyyear"] < low_thr).sum())
    logger.info(f"[DIAG] Hexes with cov_union_anyyear < {low_thr:.2f}: {n_low} / {len(diag)}")

    gdf_hex_year = intersect_and_aggregate_area_weighted(
        gdf_left=gdf_joined,
        gdf_hex=gdf_hex,
        year_col=year_col,
        hex_id_col=hex_id_col,
        dominant_col="jednostka",
        decimals=int(_sel(cfg, "pipeline.layer_defaults.decimals", 3)),
        min_cover_fraction=float(_sel(cfg, "pipeline.layer_defaults.min_cover_fraction", 0.0)),
        hex_area_col="hex_area_m2",
        geometry_out="hex",
        parcel_id_col=_sel(cfg, "egib.id_col", "iddzialki"),
    )

    ix, phex = build_parcel_hex_table(
        gdf_left=gdf_joined,
        gdf_hex=gdf_hex,
        year_col=year_col,
        hex_id_col=hex_id_col,
        hex_area_col="hex_area_m2",
        parcel_id_col=_sel(cfg, "egib.id_col", "iddzialki"),
        min_cover_fraction=float(_sel(cfg, "pipeline.layer_defaults.min_cover_fraction", 0.0)),
    )

    hex_stats = aggregate_hex_year_stats_from_parcel_hex(
        phex,
        year_col=year_col,
        hex_id_col=hex_id_col,
        hex_area_col="hex_area_m2",
        small_thr_m2=float(_sel(cfg, "features.split.small_thr_m2", 1000.0)),  # 0.1 ha
    )

    out_stats_table = _sel(
        cfg,
        "features.split.hex_stats_out_table",
        f'{_sel(cfg,"pipeline.hex.schema","hex")}."HexStats_{_sel(cfg,"pipeline.hex.out_suffix","r9")}"'
    )

    _ = save_hex_stats_table(
        db_path=Path(_sel(cfg, "data.duckdb_path")),
        df=hex_stats,
        table=out_stats_table,
        casts={hex_id_col: "VARCHAR", year_col: "INT"},
        write_mode=_sel(cfg, "layer_defaults.write_mode", "replace"),
    )

    out_table = _sel(cfg, "add_parcels_data.out_table",
                     f'{_sel(cfg,"pipeline.hex.schema","hex")}.DzialkaEwidencyjna_{_sel(cfg,"pipeline.hex.out_suffix","r9")}')

    return save_geodf_as_ewkb_geometry(
        db_path=Path(_sel(cfg, "data.duckdb_path")),
        gdf=gdf_hex_year,
        table=out_table,
        srid=2180,
        geom_col="geometry",
        write_mode=_sel(cfg, "layer_defaults.write_mode", "replace"),
        casts={hex_id_col: "VARCHAR", year_col: "INT"},
    )