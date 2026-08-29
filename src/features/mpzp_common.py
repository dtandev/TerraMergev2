"""Shared helpers for the MPZP feature builders.

`add_mpzp` (AOI-clipped area polygons) and `add_mpzp_for_parcels` (per-parcel enrichment)
carried byte-for-byte copies of these helpers; they live here now so both import one copy.
"""

from __future__ import annotations

from collections.abc import Iterable
from urllib.parse import urlencode

import geopandas as gpd
import pandas as pd
import shapely.wkb
from shapely.geometry.base import BaseGeometry

BBox = tuple[float, float, float, float]
PARCEL_GEOM_COL = "geometry"


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
    Drop redundant columns. Safe to call multiple times (errors='ignore').
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
    return gdf.drop(columns=to_drop, errors="ignore").copy()


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
