# === Config ===
from __future__ import annotations

# === Imports ===
import os
from collections.abc import Iterable

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.wkb
from loguru import logger
from omegaconf import DictConfig
from shapely.geometry.base import BaseGeometry

from src.features.features_makeover import FeaturesMakeover

# === Types ===
BBox = tuple[float, float, float, float]

# === Constants ===
PARCEL_GEOM_COL = "geometry"

# === Helpers ===


def _coerce_gdf_geometry(
    df: pd.DataFrame | gpd.GeoDataFrame,
    *,
    geom_col: str = PARCEL_GEOM_COL,
    target_crs: str | None = None,
) -> gpd.GeoDataFrame:
    """
    Ensure geometry column is a proper Shapely-backed GeoSeries.
    Accepts shapely objects or WKB in bytes/bytearray/memoryview/hex string.
    """
    if geom_col not in df.columns:
        raise KeyError(f"Missing geometry column '{geom_col}'")

    s = df[geom_col]

    # Fast path: already shapely
    if isinstance(s.dtype, gpd.array.GeometryDtype) or (
        len(s) > 0 and isinstance(s.iloc[0], BaseGeometry)
    ):
        gser = gpd.GeoSeries(s, crs=None)
    else:

        def _to_bytes(x):
            if x is None or (isinstance(x, float) and pd.isna(x)):
                return None
            if isinstance(x, (bytes, bytearray, memoryview)):
                return bytes(x)
            if isinstance(x, str):
                try:
                    return bytes.fromhex(x)  # hex → bytes
                except Exception:
                    return None
            return None

        wb = s.map(_to_bytes)

        def _loads_wkb(b):
            if b is None:
                return None
            try:
                return shapely.wkb.loads(b)
            except Exception:
                return None

        gser = gpd.GeoSeries(wb.map(_loads_wkb), crs=None)

    out = gpd.GeoDataFrame(df.drop(columns=[geom_col]), geometry=gser)
    # Clean small self-intersections after import/reprojection
    out[geom_col] = out.geometry.buffer(0)

    if target_crs:
        out.set_crs(target_crs, inplace=True, allow_override=True)

    return out


def _sanitize_before_save(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Drop redundant columns
    Safe to call multiple times (errors='ignore').
    """
    to_drop = [
        "shape_length",
        "Shape_Length",
        "shape_area",
        "Shape_Area",
        "obreb",
        "nr_dzialki",
        "jednostka",
    ]
    out = gdf.drop(columns=to_drop, errors="ignore").copy()

    return out


def _gdf_to_wkb_df(gdf: gpd.GeoDataFrame, *, geom_col: str = "geometry") -> pd.DataFrame:
    """
    Return plain pandas.DataFrame with geometry encoded as WKB in '__geom_wkb__'.
    """
    df = pd.DataFrame(gdf.drop(columns=[], errors="ignore"))
    df["__geom_wkb__"] = gdf[geom_col].to_wkb()
    if geom_col in df.columns:
        df = df.drop(columns=[geom_col])
    return df


def _read_wfs_layer(
    base_url: str,
    typename: str,
    version: str,
    srsname: str,
    bbox: BBox | None = None,
) -> gpd.GeoDataFrame:
    """
    Download WFS layer with bbox filter and fallback engines.
    Tries GDAL/pyogrio, Fiona; falls back to HTTP GetFeature.
    """
    from urllib.parse import urlencode

    # 1) GDAL virtual connection
    conn = f"WFS:{base_url}?SERVICE=WFS&VERSION={version}"
    for lyr in (typename, typename.split(":")[-1]):
        for engine in ("pyogrio", "fiona"):
            try:
                return gpd.read_file(conn, layer=lyr, bbox=bbox, engine=engine)
            except Exception:
                pass

    # 2) Raw URL GetFeature
    params = {
        "service": "WFS",
        "request": "GetFeature",
        "version": version,
        ("typeName" if version.startswith("1.") else "typeNames"): typename,
        "srsName": srsname,
        # "outputFormat": "application/json",  # sometimes needed on picky servers
    }
    if bbox is not None:
        # many WFS 1.1.0 servers require CRS at the end of bbox
        params["bbox"] = ",".join(map(str, bbox)) + f",{srsname}"

    url = f"{base_url}?{urlencode(params)}"
    return gpd.read_file(url, engine="pyogrio")


def _require_columns(gdf: gpd.GeoDataFrame, cols: Iterable[str], *, layer_hint: str = "") -> None:
    missing = [c for c in cols if c not in gdf.columns]
    if missing:
        hint = f" in {layer_hint}" if layer_hint else ""
        raise KeyError(f"Missing columns{hint}: {missing}")


# === Core ===
def run_add_mpzp(cfg: DictConfig) -> None:
    """
    Enrich EGIB parcels with MPZP layers (plans and designations) and optionally save to DuckDB.

    Expects in cfg:
      features.add_mpzp.source_layer : str
      features.crs_target  : str  (e.g. 'EPSG:2180')
      features.add_mpzp.add_to_duckdb.enabled : bool
      data.duckdb_path              : str
      wfs.url / wfs.version / wfs.srsname
    """
    # Optional: reduce GDAL warnings for unclosed rings (we still run buffer(0) later)
    os.environ.setdefault("OGR_GEOMETRY_ACCEPT_UNCLOSED_RING", "YES")

    feat = cfg.features.add_mpzp
    wfs = cfg.wfs
    db_path: str = cfg.data.duckdb_path
    src_layer: str = feat.source_layer
    target_crs: str = cfg.features.crs_target

    # --- Load parcels from DuckDB (force spatial ext, then WKB) ---
    logger.info('Loading parcels from DuckDB egib."{}" (as WKB)', src_layer)
    con = duckdb.connect(db_path)
    try:
        # Ensure spatial functions (ST_AsWKB) are available
        con.execute("INSTALL spatial;")
        con.execute("LOAD spatial;")
        try:
            parcels_df = con.execute(
                f'''
                SELECT
                    *,
                    ST_AsWKB({PARCEL_GEOM_COL}) AS __geom_wkb__
                FROM egib."{src_layer}"
                '''
            ).fetch_df()
        except Exception as e:
            logger.warning(
                "ST_AsWKB failed ({}). Falling back to raw geometry column.", type(e).__name__
            )
            parcels_df = con.execute(f'SELECT * FROM egib."{src_layer}"').fetch_df()
            # If no explicit WKB provided, try to reuse existing geometry as WKB-like blob
            if PARCEL_GEOM_COL in parcels_df.columns:
                parcels_df = parcels_df.rename(columns={PARCEL_GEOM_COL: "__geom_wkb__"})
            else:
                raise
    finally:
        con.close()

    if "__geom_wkb__" not in parcels_df.columns:
        raise RuntimeError("DuckDB query did not return __geom_wkb__ nor a usable geometry column")

    raw = parcels_df.drop(columns=[c for c in (PARCEL_GEOM_COL,) if c in parcels_df.columns])
    raw.rename(columns={"__geom_wkb__": PARCEL_GEOM_COL}, inplace=True)

    parcels = _coerce_gdf_geometry(raw, geom_col=PARCEL_GEOM_COL, target_crs=target_crs)

    # Validate geometries
    if parcels.empty:
        logger.warning("Parcels are empty — nothing to do.")
        return
    n_null = int(parcels.geometry.isna().sum())
    if n_null == len(parcels):
        raise ValueError("All parcel geometries are NULL after WKB decoding — cannot compute bbox.")

    # Normalize CRS
    if parcels.crs is None or str(parcels.crs) != str(target_crs):
        parcels = parcels.set_crs(target_crs, allow_override=True)

    bbox_vals = parcels.total_bounds
    if np.any(~np.isfinite(bbox_vals)):
        raise ValueError(f"Invalid bbox from parcels: {bbox_vals}")
    bbox: BBox = tuple(bbox_vals)  # type: ignore[assignment]

    logger.info("Fetching MPZP[plany/przeznaczenia] for bbox={}", bbox)

    # --- WFS layers (→ target CRS) ---
    przezn = _read_wfs_layer(wfs.url, "ms:przeznaczenia", wfs.version, wfs.srsname, bbox)
    plany = _read_wfs_layer(wfs.url, "ms:plany", wfs.version, wfs.srsname, bbox)

    if przezn.crs is None:
        przezn = przezn.set_crs(wfs.srsname, allow_override=True)
    if plany.crs is None:
        plany = plany.set_crs(wfs.srsname, allow_override=True)

    przezn = przezn.to_crs(target_crs)
    plany = plany.to_crs(target_crs)

    # Sanity required cols
    _require_columns(przezn, ["etykieta", PARCEL_GEOM_COL], layer_hint="ms:przeznaczenia")
    _require_columns(
        plany,
        ["data_uchwaly", "oznaczenie", "geotiff", "legenda", PARCEL_GEOM_COL],
        layer_hint="ms:plany",
    )

    # Clean geometries after reprojection
    for gdf in (parcels, przezn, plany):
        gdf[PARCEL_GEOM_COL] = gdf.geometry.buffer(0)

    # --- Spatial joins (simple intersects) ---
    logger.info("Joining przeznaczenia → parcels")

    out = gpd.sjoin(
        parcels,
        przezn[["etykieta", PARCEL_GEOM_COL]].rename(columns={PARCEL_GEOM_COL: "geometry"}),
        how="left",
        predicate="intersects",
    ).drop(columns=["index_right"], errors="ignore")

    # --- Deduplicate/clean columns & derive 'rok_uchwaly' ---
    out = _sanitize_before_save(out)

    logger.info("Joining plany → parcels")
    out = gpd.sjoin(
        out,
        plany[["data_uchwaly", "oznaczenie", "geotiff", "legenda", PARCEL_GEOM_COL]].rename(
            columns={PARCEL_GEOM_COL: "geometry"}
        ),
        how="left",
        predicate="intersects",
    ).drop(columns=["index_right"], errors="ignore")

    # --- Derive MPZP labels ---
    if feat.derive_features:
        logger.info("Deriving MPZP labels")
        out = FeaturesMakeover._sanitize_mpzp_source(
            out,
            src_col="etykieta",
            placeholder="Brak",
        )
        out = FeaturesMakeover._add_mpzp_label_simple(
            out,
            mapping_df=pd.read_csv(cfg.data.mpzp_mapping_csv),
            src_col="etykieta",
            out_col="mpzp_etykieta",
            mapping_orig_col="etykieta_oryginalna",
            mapping_group_col="grupa_glowna",
            placeholder="Brak",
        )
        out = FeaturesMakeover._apply_mpzp_temporal_rule(
            out,
            out_col="mpzp_etykieta",
            plan_date_col="data_uchwaly",
            year_col="year",
            placeholder="Brak",
        )

    # --- Optional save to DuckDB (WKB → GEOMETRY in SQL) ---
    if feat.add_to_duckdb.enabled:
        logger.info("Writing enriched parcels to egib.MPZP (via WKB → GEOMETRY)")
        con = duckdb.connect(db_path)
        try:
            con.execute("INSTALL spatial;")
            con.execute("LOAD spatial;")
            con.execute("CREATE SCHEMA IF NOT EXISTS egib;")

            df_db = _gdf_to_wkb_df(out, geom_col="geometry")
            con.register("tmp_df", df_db)

            cols_no_wkb = ", ".join([f'"{c}"' for c in df_db.columns if c != "__geom_wkb__"])
            con.execute(f"""
                CREATE OR REPLACE TABLE egib.MPZP AS
                SELECT
                    {cols_no_wkb},
                    ST_GeomFromWKB(__geom_wkb__) AS geometry
                FROM tmp_df;
            """)
        finally:
            con.close()

    logger.success("MPZP enrichment complete.")
