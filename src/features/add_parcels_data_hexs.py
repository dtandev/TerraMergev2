from __future__ import annotations
from pathlib import Path
from typing import Optional, Sequence, Dict, List, Tuple
import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import DictConfig

# =========================
# Small config helper
# =========================

def _sel(cfg: DictConfig, path: str, default=None):
    """
    Safe nested selection from DictConfig using 'dot' path.
    """
    cur = cfg
    for part in path.split("."):
        if cur is None or part not in cur:
            return default
        cur = cur[part]
    return cur


# =========================
# DuckDB I/O
# =========================

def _connect_spatial(db_path: Path) -> duckdb.DuckDBPyConnection:
    """
    Open DuckDB connection with spatial extension.
    """
    con = duckdb.connect(str(db_path))
    con.execute("INSTALL spatial;")
    con.execute("LOAD spatial;")
    return con


def _duckdb_columns(con: duckdb.DuckDBPyConnection, table: str) -> List[str]:
    """
    Return column names for a DuckDB table/view.
    """
    return [r[1] for r in con.execute(f"PRAGMA table_info('{table}')").fetchall()]

def _to_bytes_safe(val: object) -> bytes | None:
    """
    Convert DuckDB BLOB-like values to bytes (handles bytes/bytearray/memoryview/list[int]).
    """
    if val is None:
        return None
    if isinstance(val, bytes):
        return val
    if isinstance(val, (bytearray, memoryview)):
        return bytes(val)
    if isinstance(val, list):
        try:
            return bytes(val)
        except Exception:
            return None
    return None

def load_parcels_joined_2180(
    db_path: Path,
    *,
    parcels_table: str = "egib.DzialkaEwidencyjna",
    gf_table: str = "egib.GeometricFeatures",
    geom_col: str = "geometry",
    join_keys: Tuple[str, str] = ("iddzialki", "year"),
) -> gpd.GeoDataFrame:
    """
    Load parcels LEFT JOIN GeometricFeatures on (iddzialki, year).
    Geometry is fetched as WKB; CRS is set to EPSG:2180 in GeoPandas (no SQL transform).
    """
    id_col, year_col = join_keys
    con = _connect_spatial(db_path)

    # Build a safe SELECT that prefixes GF columns to avoid duplicates.
    gf_cols = [c for c in _duckdb_columns(con, gf_table) if c not in {id_col, year_col}]
    gf_select = ", ".join([f'gf."{c}" AS "gf_{c}"' for c in gf_cols]) if gf_cols else ""

    q = f"""
        SELECT
            de.*,
            ST_AsWKB(de.{geom_col}) AS geom_wkb_2180
            {"," if gf_select else ""}
            {gf_select}
        FROM {parcels_table} AS de
        LEFT JOIN {gf_table} AS gf
          ON de.{id_col} = gf.{id_col} AND de.{year_col} = gf.{year_col}
    """
    df: pd.DataFrame = con.execute(q).fetchdf()
    con.close()

    # Parse WKB → geometry; set CRS to EPSG:2180 (assumption: coords already in 2180)
    wkb = df["geom_wkb_2180"].map(_to_bytes_safe)
    geos = gpd.GeoSeries.from_wkb(wkb, crs="EPSG:2180")

    # Drop the helper WKB and any original raw geometry column if present
    drop_cols = ["geom_wkb_2180"]
    if geom_col in df.columns:
        drop_cols.append(geom_col)

    gdf = gpd.GeoDataFrame(df.drop(columns=drop_cols, errors="ignore"), geometry=geos, crs="EPSG:2180")
    return gdf


def load_parcels_simple_2180(
    db_path: Path,
    *,
    parcels_table: str = "egib.DzialkaEwidencyjna",
    geom_col: str = "geometry",
) -> gpd.GeoDataFrame:
    """
    Load parcels from DuckDB without any join.
    Geometry is fetched as WKB; CRS is set to EPSG:2180 in GeoPandas.
    """
    logger.info("Loading parcels (no join) from table '{}'", parcels_table)
    con = _connect_spatial(db_path)
    q = f"""
        SELECT
            de.*,
            ST_AsWKB(de.{geom_col}) AS geom_wkb_2180
        FROM {parcels_table} AS de
    """
    df: pd.DataFrame = con.execute(q).fetchdf()
    con.close()

    wkb = df["geom_wkb_2180"].map(_to_bytes_safe)
    geos = gpd.GeoSeries.from_wkb(wkb, crs="EPSG:2180")

    drop_cols = ["geom_wkb_2180"]
    if geom_col in df.columns:
        drop_cols.append(geom_col)

    gdf = gpd.GeoDataFrame(df.drop(columns=drop_cols, errors="ignore"), geometry=geos, crs="EPSG:2180")
    logger.debug("Loaded {} parcel rows (no join)", len(gdf))
    return gdf


def load_hex_2180(
    db_path: Path,
    *,
    hex_table: str = "hex.Hexagons_r7",
    geom_col: str = "geometry",
) -> gpd.GeoDataFrame:
    """
    Load H3 hexagons. Geometry is fetched as WKB; CRS is set to EPSG:2180.
    """
    con = _connect_spatial(db_path)
    q = f"""
        SELECT
            hex_id,
            ST_AsWKB({geom_col}) AS geom_wkb_2180
        FROM {hex_table}
    """
    df: pd.DataFrame = con.execute(q).fetchdf()
    con.close()

    wkb = df["geom_wkb_2180"].map(_to_bytes_safe)
    geos = gpd.GeoSeries.from_wkb(wkb, crs="EPSG:2180")
    gdf = gpd.GeoDataFrame(df.drop(columns=["geom_wkb_2180"], errors="ignore"), geometry=geos, crs="EPSG:2180")
    return gdf


# =========================
# Intersection + aggregation
# =========================

def intersect_aggregate_hex_parcels(
    gdf_parcels: gpd.GeoDataFrame,
    gdf_hex: gpd.GeoDataFrame,
    *,
    year_col: str = "year",
    hex_id_col: str = "hex_id",
    include_features: Optional[Sequence[str]] = None,
    categorical_modes: Sequence[str] = ("jednostka",),
    parcel_id_col: str = "iddzialki",
) -> gpd.GeoDataFrame:
    """
    Intersect parcels with hexagons (metric CRS) and compute area-weighted stats.

    For each (hex, year):
      • area-weighted means for numeric features -> <feature>_mean
      • dominant category by area from `categorical_modes` -> 'jednostka'
      • hex_area [m²], coverage_area [m²]
      • n_parcel: number of **unique parcels** intersecting the hex that year
    """
    if gdf_hex.crs is None:
        raise ValueError("gdf_hex CRS is None; expected metric CRS like EPSG:2180.")
    hex_crs = gdf_hex.crs
    if gdf_parcels.crs != hex_crs:
        gdf_parcels = gdf_parcels.to_crs(hex_crs)

    parcels_cols = set(gdf_parcels.columns)
    if parcel_id_col not in parcels_cols:
        raise ValueError(f"Parcel ID column '{parcel_id_col}' not found in gdf_parcels.")

    hex_only = gdf_hex[[hex_id_col, gdf_hex.geometry.name]].copy()
    hex_only["hex_area"] = hex_only.area

    inter = gpd.overlay(gdf_parcels, hex_only, how="intersection", keep_geom_type=True)
    if inter.empty:
        cols = [hex_id_col, year_col, "n_parcel", "jednostka", "hex_area", "coverage_area"]
        return gpd.GeoDataFrame(columns=cols, geometry=gpd.GeoSeries([], crs=hex_crs), crs=hex_crs)

    inter["part_area"] = inter.geometry.area
    if "hex_area" not in inter.columns:
        inter = inter.merge(hex_only[[hex_id_col, "hex_area"]], on=hex_id_col, how="left")
    inter["w_hex"] = inter["part_area"] / inter["hex_area"]

    # === wybór cech numerycznych ===
    if include_features is None:
        include_features = [
            c for c in gdf_parcels.columns
            if c not in {gdf_parcels.geometry.name, year_col, parcel_id_col}
            and np.issubdtype(gdf_parcels[c].dtype, np.number)
        ]
    else:
        include_features = [c for c in include_features if c in parcels_cols]

    # Klucze grupujące (Series, bez DataFrameGroupBy.apply -> brak FutureWarning)
    keys = [inter[hex_id_col], inter[year_col]]

    # === coverage_area (Σ pól części) ===
    coverage_area = inter["part_area"].groupby(keys, observed=True).sum().rename("coverage_area")

    # === średnie ważone (wektorowo) ===
    mean_frames: List[pd.Series] = []
    for col in include_features:
        mask = inter[col].notna()
        sum_wx = (inter.loc[mask, col] * inter.loc[mask, "w_hex"]).groupby(
            [inter.loc[mask, hex_id_col], inter.loc[mask, year_col]], observed=True
        ).sum()
        sum_w = inter.loc[mask, "w_hex"].groupby(
            [inter.loc[mask, hex_id_col], inter.loc[mask, year_col]], observed=True
        ).sum()
        mean_col = (sum_wx / sum_w).rename(f"{col}_mean")
        mean_frames.append(mean_col)

    # === n_parcel (unikalne ID działek w heksie/roku) ===
    n_parcel = inter[parcel_id_col].groupby(keys, observed=True).nunique().rename("n_parcel")

    # === dominanta kategorii (bez apply) ===
    dom_col_name = next((c for c in categorical_modes if c in parcels_cols), None)
    if dom_col_name is not None:
        area_per_cat = inter.groupby(
            [inter[hex_id_col], inter[year_col], inter[dom_col_name]], observed=True
        )["part_area"].sum()
        # idxmax daje MultiIndex (hex, year, cat); wyciągamy poziom kategorii
        idx = area_per_cat.groupby(level=[0, 1]).idxmax()
        dominant = pd.Series(
            data=[t[2] for t in idx.to_list()],
            index=pd.MultiIndex.from_tuples([(t[0], t[1]) for t in idx.to_list()],
                                            names=[hex_id_col, year_col]),
            name="jednostka",
        )
    else:
        dominant = pd.Series(dtype=object, name="jednostka")

    # === złożenie ramki wynikowej ===
    out = pd.concat([coverage_area, *mean_frames, n_parcel, dominant], axis=1).reset_index()

    out = out.merge(
        hex_only[[hex_id_col, "hex_area", hex_only.geometry.name]],
        on=hex_id_col,
        how="left",
    )
    out = out.loc[out["coverage_area"] > 0].copy()

    out_gdf = gpd.GeoDataFrame(
        out.drop(columns=["geometry"], errors="ignore"),
        geometry=out[hex_only.geometry.name],
        crs=hex_crs,
    )

    feature_means = [f"{c}_mean" for c in include_features]
    cols = [hex_id_col, year_col, *feature_means, "n_parcel", "jednostka", "hex_area", "coverage_area", out_gdf.geometry.name]
    cols = [c for c in cols if c in out_gdf.columns]
    return out_gdf[cols]


def _postprocess_coverage_and_rounding(
    gdf_out: gpd.GeoDataFrame,
    *,
    min_cover_fraction: float,
    decimals: int,
    logger=logger,
) -> gpd.GeoDataFrame:
    """
    Add coverage_frac, filter by min_cover_fraction and round numeric *_mean and area columns.
    """
    # Guard: unikamy dzielenia przez zero (teoretycznie hex_area == 0 nie powinno wystąpić, ale lepiej jasno)
    nonzero = gdf_out["hex_area"] > 0
    gdf_out = gdf_out.loc[nonzero].copy()

    gdf_out["coverage_frac"] = gdf_out["coverage_area"] / gdf_out["hex_area"]

    if min_cover_fraction > 0.0:
        before = len(gdf_out)
        gdf_out = gdf_out.loc[gdf_out["coverage_frac"] >= min_cover_fraction].copy()
        logger.info("Coverage filter >= {:.2%}: kept {}/{} rows", min_cover_fraction, len(gdf_out), before)

    # Zaokrąglaj tylko liczby i tylko sensowne kolumny
    numeric_cols = gdf_out.select_dtypes(include=["number"]).columns.tolist()
    to_round = [c for c in numeric_cols if c.endswith("_mean") or c in ("hex_area", "coverage_area", "coverage_frac")]
    if to_round:
        gdf_out[to_round] = gdf_out[to_round].round(decimals)

    return gdf_out


def write_geodf_to_duckdb(
    con: duckdb.DuckDBPyConnection,
    gdf: gpd.GeoDataFrame,
    *,
    table: str,
    geom_col: str = "geometry",
    casts: Optional[Dict[str, str]] = None,
) -> int:
    """
    Write a GeoDataFrame to DuckDB, overwriting the target table.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Open DuckDB connection with spatial extension loaded.
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame with geometry column.
    table : str
        Full table name to create (e.g. 'hex.DzialkaEwidencyjna_r7').
    geom_col : str, default 'geometry'
        Name of geometry column in DuckDB.
    srid : int, default 2180
        SRID assigned to geometry.
    casts : dict[str, str] | None
        Optional SQL type casts, e.g. {'hex_id': 'VARCHAR', 'year': 'INT'}.

    Returns
    -------
    int
        Number of rows written.
    """
    df = gdf.copy()
    df["__wkb__"] = df.geometry.to_wkb()
    non_geo_cols = [c for c in df.columns if c != gdf.geometry.name]
    con.register("df_in", df[non_geo_cols])

    casts = casts or {}
    cols_sql = [
        f'CAST("{c}" AS {casts[c]}) AS "{c}"' if c in casts else f'"{c}"'
        for c in non_geo_cols
        if c != "__wkb__"
    ]
    cols_sql_str = ", ".join(cols_sql)
    sep = ", " if cols_sql_str else ""
    con.execute(f"DROP TABLE IF EXISTS {table}")
    con.execute(f"""
        CREATE TABLE {table} AS
        SELECT
            {cols_sql_str}{sep}
            ST_GeomFromWKB(__wkb__) AS {geom_col}
        FROM df_in
    """)
    return len(df)

# =========================
# Pipeline entry
# =========================

def run_add_parcels_data(cfg: DictConfig) -> None:
    """
    Pipeline step:
      • read EGIB parcels (optionally LEFT JOIN GF),
      • read HEX grid,
      • intersect & aggregate,
      • filter by min_cover_fraction, round decimals,
      • write to DuckDB as hex.DzialkaEwidencyjna_${pipeline.resolution}.
    """
    db_path = Path(_sel(cfg, "data.duckdb_path"))

    if not db_path:
        raise ValueError("Missing cfg.data.duckdb_path")
    
    parcels_table : str = _sel(cfg, "pipeline.egib.table", "")
    geom_col: str = _sel(cfg, "pipeline.egib.geom_col", "geometry")

    join_with: Optional[str] = _sel(cfg, "pipeline.add_parcels_data.join_with", None)
    join_keys: Tuple[str, str] = _sel(cfg, "pipeline.add_parcels_data.join_columns", ("iddzialki", "year"))

    if join_with:
        logger.info(f"Loading parcels joined with {join_with} from DuckDB → {db_path}")
        gdf_parcels = load_parcels_joined_2180(
            db_path,
            parcels_table=parcels_table,              
            gf_table=join_with,
            geom_col=geom_col,
            join_keys=tuple(join_keys)
        )
    else:
        logger.info(f"Loading parcels (no join) from DuckDB → {db_path}")
        gdf_parcels = load_parcels_simple_2180(
            db_path,
            parcels_table=parcels_table,
            geom_col=geom_col
        )

    logger.info(f"Loading hexagons from DuckDB → {db_path}")

    hex_table = _sel(cfg, "pipeline.hex.table", "")
    geom_col: str = _sel(cfg, "pipeline.hex.geom_col", "geometry")

    logger.info(f"Using hex table: {hex_table}")

    gdf_hex = load_hex_2180(
        db_path,
        hex_table=hex_table,
        geom_col=(geom_col)
    )

    logger.info("Intersecting and aggregating parcels into hexagons...")

    year_col: str = _sel(cfg, "pipeline.egib.year_col", "year")
    hex_id_col: str = _sel(cfg, "pipeline.add_parcels_data.hex_id_col", "hex_id")
    parcel_id_col: str = _sel(cfg, "pipeline.egib.parcel_id_col", "iddzialki")

    gdf_out = intersect_aggregate_hex_parcels(
        gdf_parcels=gdf_parcels,
        gdf_hex=gdf_hex,
        year_col=year_col,
        hex_id_col=hex_id_col,
        include_features=None,
        categorical_modes=("jednostka", "obreb"),
        parcel_id_col=parcel_id_col,
    )
    logger.info("Aggregation result: {} rows", len(gdf_out))

    logger.info("Postprocessing: filtering by coverage fraction and rounding...")

    min_cover_fraction: float = _sel(cfg, "pipeline.layer_defaults.min_cover_fraction", 0.1)
    decimals: int = _sel(cfg, "pipeline.layer_defaults.decimals", 2)

    logger.info(f"Using min_cover_fraction = {min_cover_fraction}, decimals = {decimals}")

    gdf_out = _postprocess_coverage_and_rounding(
        gdf_out,
        min_cover_fraction=min_cover_fraction,
        decimals=decimals,
        logger=logger,
    )
    logger.info("Postprocessing result: {} rows", len(gdf_out))

    con = _connect_spatial(db_path)

    out_table = _sel(cfg, "pipeline.add_parcels_data.out_table", "(empty)")   

    logger.info(f"Writing output to DuckDB table {out_table}...")
    rows = write_geodf_to_duckdb(
        con,
        gdf_out,
        table=out_table,
        geom_col=geom_col,
        srid=2180,
        casts={"hex_id": "VARCHAR", "year": "INT"},
    )
    con.close()

    logger.info("Wrote {} rows to {} in DuckDB", rows, out_table)
