from __future__ import annotations

from pathlib import Path

import duckdb
import geopandas as gpd
import pandas as pd
from loguru import logger
from shapely import set_srid, to_wkb


def connect_duckdb(db_path: Path) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection with the spatial extension loaded."""
    logger.info("Łączenie z DuckDB: {}", db_path)
    con = duckdb.connect(str(Path(db_path).expanduser()))
    try:
        con.execute("LOAD spatial;")
        logger.debug("DuckDB spatial extension: loaded.")
    except (duckdb.CatalogException, duckdb.IOException):
        logger.warning("DuckDB spatial extension not installed — installing now…")
        con.execute("INSTALL spatial;")
        con.execute("LOAD spatial;")
        logger.debug("DuckDB spatial extension: installed & loaded.")
    return con


def _detect_srid(con: duckdb.DuckDBPyConnection, table: str, geom_col: str) -> int | None:
    """
    Best-effort SRID detection from an existing table's geometry column.

    Uses ST_CRS(), not ST_SRID() — the DuckDB spatial extension renamed/replaced ST_SRID with
    ST_CRS (returning an "EPSG:<code>" string) at some point; ST_SRID no longer exists in current
    versions. Every prior copy of this helper across the codebase silently returned None here
    (the bare `except` swallowed the resulting CatalogException), always falling back to whatever
    `enforce_crs` default the caller configured instead of the table's actual CRS.

    Known limitation (verified empirically, not fixable from here): in the currently installed
    DuckDB spatial extension build, GEOMETRY('EPSG:...') CRS metadata does not survive a
    connection close/reopen against a file-backed database — only within the same connection that
    wrote it. In practice this means this function reliably returns a real SRID only when called
    on the same connection that just wrote the table; for tables written by a prior run (a new
    connection, as in normal pipeline usage), it will return None and callers fall back to their
    configured `enforce_crs` default. That fallback is therefore not just a safety net here — it
    is, today, the only thing that actually works across separate pipeline runs.
    """
    try:
        v = con.execute(
            f"SELECT ST_CRS({geom_col}) FROM {table} WHERE {geom_col} IS NOT NULL LIMIT 1"
        ).fetchone()
        if not v or not v[0]:
            return None
        crs_str = str(v[0])
        return int(crs_str.split(":")[-1]) if ":" in crs_str else int(crs_str)
    except Exception:
        return None


def save_geodf_as_ewkb_geometry(
    db_path: Path,
    gdf: gpd.GeoDataFrame,
    table: str,
    *,
    srid: int = 2180,
    geom_col: str = "geometry",
    write_mode: str = "replace",
    casts: dict[str, str] | None = None,
) -> int:
    """Write a GeoDataFrame to DuckDB as GEOMETRY, going through EWKB (with SRID)."""
    logger.info("[SAVE] Zapis do {} (SRID={}, tryb={})…", table, srid, write_mode)
    df = gdf.copy()
    df["__geom_wkb"] = df[geom_col].apply(
        lambda g: to_wkb(set_srid(g, srid), include_srid=True) if g is not None else None
    )
    df = df.drop(columns=[geom_col])

    casts = casts or {}
    cols_sql = ", ".join(
        (f'"{c}"::{casts[c]} AS "{c}"' if c in casts else f'"{c}"')
        for c in df.columns
        if c != "__geom_wkb"
    )
    # ST_GeomFromWKB alone does not surface the SRID embedded via to_wkb(include_srid=True) through
    # ST_CRS() in this DuckDB spatial version — wrap with ST_SetCRS so the CRS is actually queryable.
    geom_sql = f"ST_SetCRS(ST_GeomFromWKB(__geom_wkb), 'EPSG:{int(srid)}') AS \"geometry\""
    select_sql = f"SELECT {(cols_sql + ', ') if cols_sql else ''}{geom_sql} FROM __tmp__"

    schema = table.split(".")[0] if "." in table else "main"
    with duckdb.connect(str(db_path)) as con:
        try:
            con.execute("LOAD spatial;")
        except (duckdb.CatalogException, duckdb.IOException):
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
        logger.success("[SAVE] Zapisano {} wierszy do {}.", n, table)
        return int(n)


def write_geoparquet(gdf: gpd.GeoDataFrame, out_path: Path) -> None:
    """
    Write a GeoDataFrame to a GeoParquet file via DuckDB's `COPY ... TO ... (FORMAT PARQUET)`,
    deliberately not through geopandas/pandas' `to_parquet()` (which goes through pyarrow).

    This works around a real, reproducible environment conflict: this project also imports GDAL
    (`osgeo`) to read GDB/GML deliveries, and GDAL's build here bundles its own Arrow/Parquet
    driver. Once `osgeo` has been imported anywhere in the process — regardless of import order —
    pyarrow's own filesystem registry ends up in a broken state, and any later `gdf.to_parquet(...)`
    either raises `pyarrow.lib.ArrowKeyError: Attempted to register factory for scheme 'file' but
    that scheme is already registered` or segfaults outright (verified empirically against real
    data; see audit.md). DuckDB's own parquet writer never touches pyarrow's filesystem registry,
    so it isn't affected — verified to round-trip both data and CRS correctly (EPSG:2178 in, 2178
    out) against a real 6000+ row GeoDataFrame read from a real `.gdb` file.

    Every module in this codebase that reads spatial data via geopandas/GDAL and then calls
    `.to_parquet(...)` is at risk of this in the same process (`src/main.py` imports GDAL-reading
    modules unconditionally at startup) — this helper is the fix; use it instead of
    `gdf.to_parquet(...)` anywhere that combination occurs.
    """
    geom_name = gdf.geometry.name
    crs_epsg = gdf.crs.to_epsg() if gdf.crs is not None else None

    df = pd.DataFrame(gdf)
    # `.apply()` over a zero-row GeoSeries returns an empty Series that KEEPS the geopandas
    # "geometry" extension dtype (nothing to compute, so no coercion to plain object/bytes ever
    # happens) — confirmed against real deliveries where a layer (e.g. restrictions/"RST","OZN")
    # legitimately has 0 features. DuckDB's register() then rejects it: "Data type 'geometry' not
    # recognized". Force object dtype explicitly so empty layers export the same as non-empty ones.
    wkb_col = gdf[geom_name].apply(lambda g: to_wkb(g) if g is not None else None)
    df["__wkb__"] = wkb_col.astype(object)
    df = df.drop(columns=[geom_name])

    con = duckdb.connect(":memory:")
    try:
        try:
            con.execute("LOAD spatial;")
        except (duckdb.CatalogException, duckdb.IOException):
            con.execute("INSTALL spatial;")
            con.execute("LOAD spatial;")

        con.register("__geoparquet_tmp__", df)
        geom_expr = "ST_GeomFromWKB(__wkb__)"
        if crs_epsg:
            geom_expr = f"ST_SetCRS({geom_expr}, 'EPSG:{crs_epsg}')"

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        con.execute(f"""
            COPY (SELECT * EXCLUDE (__wkb__), {geom_expr} AS "{geom_name}" FROM __geoparquet_tmp__)
            TO '{out_path.as_posix()}' (FORMAT PARQUET)
        """)
    finally:
        con.close()
