# src/features/make_hexagons.py
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import geopandas as gpd
import pandas as pd
import shapely
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from shapely import set_srid, to_wkb
from shapely.geometry import Polygon
from shapely.geometry.multipolygon import MultiPolygon

from src.common.config_utils import sel as _sel

BBox = tuple[float, float, float, float]

# ---------------- Utils ---------------- #


def _resolve_db_path(cfg: DictConfig) -> Path:
    """
    Resolve DuckDB path from config without forcing .resolve().
    """
    p = _sel(cfg, "data.duckdb_path", "egib.duckdb")
    return Path(p).expanduser()


def _configure_logging(cfg: DictConfig) -> None:
    """
    Configure console logging via loguru.
    """
    logger.remove()
    fmt = _sel(
        cfg,
        "logging.format",
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <7}</level> | "
        "<cyan>{name}:{function}:{line}</cyan> | "
        "<level>{message}</level>",
    )
    level_console = _sel(cfg, "logging.console_level", "INFO")
    logger.add(lambda msg: print(msg, end=""), format=fmt, level=level_console)


def _get_hex_node(cfg: DictConfig):
    """
    Read node from either pipeline.make_hexagons or make_hexagons.
    """
    node = OmegaConf.select(cfg, "pipeline.make_hexagons")
    return node if node is not None else OmegaConf.select(cfg, "make_hexagons")


# ---------------- Data loading ---------------- #


def _load_parcels_as_gdf_4326(
    db_path: Path,
    table: str,
    geom_col: str = "geometry",
) -> gpd.GeoDataFrame:
    """
    Read parcels from DuckDB as WKB(2180) and reproject to EPSG:4326 in GeoPandas.

    Parameters
    ----------
    db_path : Path
        Path to DuckDB file.
    table : str
        Fully-qualified table name (e.g., 'egib.kug').
    geom_col : str
        Geometry column name.

    Returns
    -------
    GeoDataFrame
        Geometry in EPSG:4326 (lon, lat).
    """
    con = duckdb.connect(str(db_path))
    con.execute("INSTALL spatial;")
    con.execute("LOAD spatial;")

    q = f"""
        SELECT
            *,
            ST_AsWKB({geom_col}) AS geom_wkb_2180
        FROM {table};
    """
    df: pd.DataFrame = con.execute(q).fetchdf()
    con.close()

    # DuckDB WKB → bytes (handles bytearray/memoryview)
    wkb_bytes = df["geom_wkb_2180"].map(lambda b: bytes(b) if b is not None else None)
    geoms_2180 = gpd.GeoSeries.from_wkb(wkb_bytes, crs="EPSG:2180")

    gdf_2180 = gpd.GeoDataFrame(
        df.drop(columns=["geom_wkb_2180"]),
        geometry=geoms_2180,
        crs="EPSG:2180",
    )
    return gdf_2180.to_crs("EPSG:4326")


# ---------------- H3 helpers ---------------- #


def _import_h3_for_polyfill():
    """
    Import an h3 module exposing `polyfill` and `h3_to_geo_boundary`.

    Returns
    -------
    module
        h3-like module with polyfill(...) and h3_to_geo_boundary(...).
    """
    import importlib

    import h3 as h3_top

    # Some envs (or h3 4.x minimal) might not expose polyfill at top-level:
    if not hasattr(h3_top, "polyfill"):
        h3_top = importlib.import_module("h3.api.basic_str")
    return h3_top


def _build_hexes_for_year_v3(
    gdf_4326: gpd.GeoDataFrame,
    year: int,
    res: int,
) -> gpd.GeoDataFrame:
    """
    Dissolve all parcels for *year* and return an H3 hex grid (EPSG:4326).
    Mirrors the known-working standalone version.

    Parameters
    ----------
    gdf_4326 : GeoDataFrame
        Parcels in EPSG:4326 (lon, lat), must contain a 'year' column.
    year : int
        Target year to select.
    res : int
        H3 resolution (e.g., 8, 9).

    Returns
    -------
    GeoDataFrame
        Columns: ['hex_id', 'res', geometry], CRS='EPSG:4326'.
    """
    if gdf_4326.crs is None or gdf_4326.crs.to_epsg() != 4326:
        raise ValueError("`gdf_4326` must be in EPSG:4326")

    if "year" not in gdf_4326.columns:
        raise KeyError("Missing 'year' column in gdf_4326")

    sub = gdf_4326[gdf_4326["year"] == year]
    if sub.empty:
        raise ValueError(f"No rows with year == {year}")

    # Dissolve robustly (Shapely ≥ 2.0)
    fixed = shapely.make_valid(sub.geometry.values)
    union_geom = shapely.union_all(fixed, grid_size=1e-7)
    if union_geom.is_empty:
        raise ValueError("AOI is empty after dissolve")

    # Normalize to iterable of polygons
    if isinstance(union_geom, MultiPolygon):
        parts = union_geom.geoms
    elif isinstance(union_geom, Polygon):
        parts = (union_geom,)
    else:
        raise TypeError(f"Unsupported geometry type: {union_geom.geom_type}")

    h3_mod = _import_h3_for_polyfill()

    # H3 polyfill for dissolved parts
    cells: set[str] = set()
    for poly in parts:
        gj = json.loads(shapely.to_geojson(poly))  # GeoJSON Polygon
        # enforce GeoJSON axis order (lon, lat)
        cells.update(h3_mod.polyfill(gj, res, geo_json_conformant=True))

    # H3 cells → polygons
    polys: list[Polygon] = []
    ids: list[str] = []
    for cid in cells:
        ring = h3_mod.h3_to_geo_boundary(cid, geo_json=True)  # [[lon, lat], ...]
        poly = shapely.Polygon(ring)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if not poly.is_empty:
            polys.append(poly)
            ids.append(cid)

    return gpd.GeoDataFrame(
        {"hex_id": ids, "res": res},
        geometry=polys,
        crs="EPSG:4326",
    )


# ---------------- Persist to DuckDB ---------------- #


def _save_hexes_to_duckdb_ewkb(
    db_path: Path,
    hex_gdf_2180: gpd.GeoDataFrame,
    table: str,
    srid: int = 2180,
) -> int:
    """
    Persist hexagons into DuckDB using EWKB with embedded SRID.

    Parameters
    ----------
    db_path : Path
        Path to DuckDB file.
    hex_gdf_2180 : GeoDataFrame
        Hex grid in target planar CRS (e.g., EPSG:2180).
    table : str
        Fully-qualified output table name, e.g., 'hex."Hexagons_r8"'.
    srid : int
        SRID to embed in EWKB.

    Returns
    -------
    int
        Number of written rows.
    """
    df = hex_gdf_2180.copy()
    df["geom_wkb"] = df.geometry.apply(lambda g: to_wkb(set_srid(g, srid), include_srid=True))
    df = df[["hex_id", "res", "geom_wkb"]]

    with duckdb.connect(str(db_path)) as con:
        try:
            con.execute("LOAD spatial;")
        except duckdb.CatalogException:
            con.execute("INSTALL spatial;")
            con.execute("LOAD spatial;")

        schema = table.split(".")[0]
        con.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}";')
        con.register("hex_tmp", df)

        con.execute(
            f"""
            CREATE OR REPLACE TABLE {table} AS
            SELECT
                hex_id::VARCHAR          AS hex_id,
                res::INTEGER             AS res,
                ST_GeomFromWKB(geom_wkb) AS geometry  -- SRID embedded in EWKB
            FROM hex_tmp;
            """
        )
        con.unregister("hex_tmp")

        n = con.execute(f"SELECT COUNT(*) FROM {table};").fetchone()[0]
        return int(n)


# ---------------- Core ---------------- #


def run_make_hexagons(cfg: DictConfig) -> None:
    """
    Hydra entrypoint:
    1) Load parcels as WKB(2180) → GeoPandas → to_crs(4326)
    2) Dissolve by `year`
    3) H3 polyfill → hex polygons in EPSG:4326
    4) Save to DuckDB in target SRID (default 2180)
    """
    _configure_logging(cfg)

    hex_cfg = _get_hex_node(cfg)
    if hex_cfg is None or not bool(hex_cfg.get("enabled", True)):
        logger.warning("make_hexagons: disabled or missing config – skipping.")
        return

    schema_in = hex_cfg.get("schema_in", _sel(cfg, "duckdb.schema", "egib"))
    table_in = hex_cfg.get("source_table", "DzialkaEwidencyjna")
    geom_col = hex_cfg.get("geom_column", "geometry")
    year_col = hex_cfg.get("year_column", "year")
    out_schema = hex_cfg.get("out_schema", "hex")
    out_prefix = hex_cfg.get("out_table_prefix", "Hexagons")
    resolutions = list(hex_cfg.get("resolutions", [8, 9]))
    overwrite = bool(hex_cfg.get("overwrite", True))
    target_srid = int(hex_cfg.get("geom_srid", 2180))
    year = int(hex_cfg.get("year"))

    db_path = _resolve_db_path(cfg)
    full_in = f'{schema_in}."{table_in}"'
    logger.info("DuckDB: {}", db_path)
    logger.info("Loading parcels from {} (geometry col: {})", full_in, geom_col)

    gdf_4326 = _load_parcels_as_gdf_4326(db_path, full_in, geom_col=geom_col)

    if year_col not in gdf_4326.columns:
        raise KeyError(f"Missing '{year_col}' column in {full_in}")

    # (opcjonalnie) przemapuj kolumnę roku, jeśli nazwa w tabeli ≠ 'year'
    if year_col != "year":
        gdf_4326 = gdf_4326.rename(columns={year_col: "year"})

    for r in resolutions:
        logger.info("Building H3 polyfill for year={} at R={}…", year, r)
        hex_r_4326 = _build_hexes_for_year_v3(gdf_4326, year=year, res=r)
        logger.info("Cells at R{}: {}", r, len(hex_r_4326))

        hex_r_target = hex_r_4326.to_crs(target_srid)
        out_table = f'{out_schema}."{out_prefix}_r{r}"'

        if overwrite:
            with duckdb.connect(str(db_path)) as con:
                con.execute(f"DROP TABLE IF EXISTS {out_table};")

        n = _save_hexes_to_duckdb_ewkb(db_path, hex_r_target, out_table, srid=target_srid)
        logger.info("Saved {} hexes (GEOMETRY[{}]) to {}.", n, target_srid, out_table)

    logger.success("Finished hex creation: {}", ", ".join(f"r{r}" for r in resolutions))
