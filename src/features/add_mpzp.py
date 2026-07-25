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
    Drop redundant columns and add 'rok_uchwaly' extracted from 'data_uchwaly' (YYYY).
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


def _gdf_to_wkb_df(gdf: gpd.GeoDataFrame, geom_col: str = "geometry") -> pd.DataFrame:
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
    Build MPZP area polygons (not per-parcel):
    - Compute bbox from all parcels.
    - Fetch WFS layers within that bbox.
    - Build AOI mask as unary_union of all parcels.
    - Clip MPZP layers to AOI (no parcel join).
    - Save polygons to DuckDB as egib.MPZP_Przeznaczenia and egib.MPZP_Plany.

    Expected cfg keys:
      features.add_mpzp.source_layer : str
      features.crs_target            : str  (e.g., 'EPSG:2180')
      features.add_mpzp.add_to_duckdb.enabled : bool
      data.duckdb_path               : str
      wfs.url / wfs.version / wfs.srsname
    """
    os.environ.setdefault("OGR_GEOMETRY_ACCEPT_UNCLOSED_RING", "YES")

    feat = cfg.features.add_mpzp
    wfs = cfg.wfs
    db_path: str = cfg.data.duckdb_path
    src_layer: str = feat.source_layer
    target_crs: str = cfg.features.crs_target

    # --- Load parcels (as before, WKB → GeoDataFrame) ---
    logger.info('Loading parcels from DuckDB egib."{}" (as WKB)', src_layer)
    con = duckdb.connect(db_path)
    try:
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

    if parcels.empty:
        logger.warning("Parcels are empty — nothing to do.")
        return

    # Normalize CRS
    parcels = parcels.set_crs(target_crs, allow_override=True)

    # --- BBOX from all parcels (aggregate envelope) ---
    bbox_vals = parcels.total_bounds
    if np.any(~np.isfinite(bbox_vals)):
        raise ValueError(f"Invalid bbox from parcels: {bbox_vals}")
    bbox: BBox = tuple(bbox_vals)  # type: ignore[assignment]
    logger.info("Fetching MPZP[plany/przeznaczenia] for bbox={}", bbox)

    # --- Fetch WFS layers within bbox, then project to target CRS ---
    przezn = _read_wfs_layer(wfs.url, "ms:przeznaczenia", wfs.version, wfs.srsname, bbox)
    plany = _read_wfs_layer(wfs.url, "ms:plany", wfs.version, wfs.srsname, bbox)

    if przezn.crs is None:
        przezn = przezn.set_crs(wfs.srsname, allow_override=True)
    if plany.crs is None:
        plany = plany.set_crs(wfs.srsname, allow_override=True)

    przezn = przezn.to_crs(target_crs)
    plany = plany.to_crs(target_crs)

    _require_columns(przezn, ["etykieta", PARCEL_GEOM_COL], layer_hint="ms:przeznaczenia")
    _require_columns(
        plany,
        ["data_uchwaly", "oznaczenie", "geotiff", "legenda", PARCEL_GEOM_COL],
        layer_hint="ms:plany",
    )

    # Clean possible self-intersections
    for gdf in (parcels, przezn, plany):
        gdf[PARCEL_GEOM_COL] = gdf.geometry.buffer(0)

    # --- AOI mask from aggregated parcels (unary_union) ---
    logger.info("Building AOI mask from all parcels (unary_union)")
    aoi_geom = parcels.unary_union
    if aoi_geom.is_empty:
        logger.warning("AOI union is empty — nothing to clip.")
        return

    # --- Clip MPZP layers to AOI (no parcel-based split) ---
    logger.info("Clipping przeznaczenia to AOI")
    przezn_aoi = gpd.clip(przezn, aoi_geom)
    logger.info("Clipping plany to AOI")
    plany_aoi = gpd.clip(plany, aoi_geom)

    # Optional: tidy columns
    przezn_aoi = _sanitize_before_save(przezn_aoi)
    plany_aoi = _sanitize_before_save(plany_aoi)

    # Derive 'rok_uchwaly' from 'data_uchwaly' if present
    if "data_uchwaly" in plany_aoi.columns:
        # tolerate string or datetime-like
        s = pd.to_datetime(plany_aoi["data_uchwaly"], errors="coerce")
        plany_aoi["rok_uchwaly"] = s.dt.year.astype("Int64")

    # Final clean
    for gdf in (przezn_aoi, plany_aoi):
        gdf[PARCEL_GEOM_COL] = gdf.geometry.buffer(0)

    # --- Write to DuckDB as area polygons (no parcel join) ---
    if feat.add_to_duckdb.enabled:
        logger.info("Writing area polygons to DuckDB: egib.MPZP_Przeznaczenia, egib.MPZP_Plany")
        con = duckdb.connect(db_path)
        try:
            con.execute("INSTALL spatial;")
            con.execute("LOAD spatial;")
            con.execute("CREATE SCHEMA IF NOT EXISTS egib;")

            # Przeznaczenia
            df_pz = _gdf_to_wkb_df(przezn_aoi, geom_col="geometry")
            con.register("tmp_pz", df_pz)
            cols_pz = ", ".join([f'"{c}"' for c in df_pz.columns if c != "__geom_wkb__"])
            con.execute(f"""
                CREATE OR REPLACE TABLE egib.MPZP_Przeznaczenia AS
                SELECT
                    {cols_pz},
                    ST_GeomFromWKB(__geom_wkb__) AS geometry
                FROM tmp_pz;
            """)

            # Plany
            df_pl = _gdf_to_wkb_df(plany_aoi, geom_col="geometry")
            con.register("tmp_pl", df_pl)
            cols_pl = ", ".join([f'"{c}"' for c in df_pl.columns if c != "__geom_wkb__"])
            con.execute(f"""
                CREATE OR REPLACE TABLE egib.MPZP_Plany AS
                SELECT
                    {cols_pl},
                    ST_GeomFromWKB(__geom_wkb__) AS geometry
                FROM tmp_pl;
            """)
        finally:
            con.close()

    logger.success("MPZP area extraction complete (AOI-clipped polygons, no parcel splits).")
