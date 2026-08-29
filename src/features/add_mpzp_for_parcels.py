# === Config ===
from __future__ import annotations

# === Imports ===
import os

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import DictConfig

from src.features.features_makeover import FeaturesMakeover
from src.features.mpzp_common import (
    PARCEL_GEOM_COL,
    BBox,
    _coerce_gdf_geometry,
    _gdf_to_wkb_df,
    _read_wfs_layer,
    _require_columns,
    _sanitize_before_save,
)


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
