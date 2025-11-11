# src/features/add_geom_feat.py
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import time

import duckdb
import geopandas as gpd
import pandas as pd
from loguru import logger
from omegaconf import DictConfig
from pandas.api.types import is_numeric_dtype, is_float_dtype
import hydra

# Jedyny import cech (pakietowy)
import src.features.GeometricFeaturesMaker as GFM  # noqa: F401


# =========================
# Helpers: config & logging
# =========================

def _cfg_get(cfg: DictConfig, path: str, default=None):
    cur: Any = cfg
    for part in path.split("."):
        if cur is None or part not in cur:
            return default
        cur = cur[part]
    return cur

def _to_bytes(x):
    if x is None:
        return None
    if isinstance(x, (bytes, str)):
        return x
    try:
        return x.tobytes()
    except AttributeError:
        return bytes(x)

def _fmt_seconds(sec: float) -> str:
    return f"{sec*1e3:.1f} ms" if sec < 1 else f"{sec:.3f} s"

def _log_mem(prefix: str = "") -> None:
    try:
        import psutil, os
        rss = psutil.Process(os.getpid()).memory_info().rss
        logger.debug("{}RSS ~ {:.1f} MB", f"{prefix} " if prefix else "", rss / (1024**2))
    except Exception:
        pass

def _round_numeric_features(df: pd.DataFrame, *, keys: tuple[str, str], decimals: int) -> pd.DataFrame:
    idc, yc = keys
    cols = [c for c in df.columns if c not in (idc, yc)]
    for c in cols:
        if is_numeric_dtype(df[c]) and is_float_dtype(df[c]):
            df[c] = df[c].round(decimals)
    return df

def _profile_gdf(gdf: gpd.GeoDataFrame, *, crs: str) -> None:
    n = len(gdf)
    geom_types = gdf.geom_type.value_counts(dropna=False).to_dict()
    n_empty = int((gdf.geometry.is_empty | gdf.geometry.isna()).sum())
    try:
        area_desc = gdf.geometry.area.describe(percentiles=[0.5, 0.9]).to_dict()
        logger.info(
            "GDF profile: rows={}, empty/NaN={}, types={}, area[m²]~min={:.2f}, p50={:.2f}, p90={:.2f}, max={:.2f}, CRS={}",
            n, n_empty, geom_types,
            float(area_desc.get("min", float("nan"))),
            float(area_desc.get("50%", float("nan"))),
            float(area_desc.get("90%", float("nan"))),
            float(area_desc.get("max", float("nan"))),
            crs
        )
    except Exception:
        logger.info("GDF profile: rows={}, empty/NaN={}, types={}, CRS={}", n, n_empty, geom_types, crs)

def _profile_features(df: pd.DataFrame, *, keys: Tuple[str, str]) -> None:
    idc, yc = keys
    n = len(df)
    cols = [c for c in df.columns if c not in (idc, yc)]
    nan_ratio = (df[cols].isna().sum() / max(n, 1)).sort_values(ascending=False).head(10)
    dup = df.duplicated(subset=[idc, yc]).sum()
    logger.info("Features matrix: rows={}, features={}, duplicate_keys={} ({}%)",
                n, len(cols), int(dup), f"{(dup/max(n,1))*100:.2f}")
    if len(nan_ratio):
        as_str = ", ".join(f"{k}:{v:.1%}" for k, v in nan_ratio.items())
        logger.debug("Top-10 NaN ratio: {}", as_str)


# ==============
# DuckDB I/O
# ==============

def _connect_duckdb(db_path: Path, *, threads: Optional[int], mem: Optional[str]) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(db_path))
    con.execute("INSTALL spatial;")
    con.execute("LOAD spatial;")
    if threads:
        con.execute(f"PRAGMA threads={int(threads)};")
    if mem:
        con.execute(f"PRAGMA memory_limit='{mem}';")
    logger.info("DuckDB connected: path={}, threads={}, mem={}", db_path, threads, mem)
    return con

def _read_parcels_as_gdf(
    con: duckdb.DuckDBPyConnection, *,
    crs: str, schema: str, src_table: str,
    id_col: str, year_col: str, geom_col: str
) -> gpd.GeoDataFrame:
    t0 = time.perf_counter()
    logger.info('Reading {}."{}" ({} → WKB)…', schema, src_table, geom_col)
    df = con.execute(f"""
        SELECT
            {id_col}::VARCHAR   AS {id_col},
            {year_col}::INTEGER AS {year_col},
            ST_AsWKB({geom_col}) AS {geom_col}
        FROM {schema}."{src_table}";
    """).fetch_df()
    logger.info("Fetched {} rows from DuckDB in {}", len(df), _fmt_seconds(time.perf_counter() - t0))

    if df.empty:
        logger.warning('Source {}."{}" is empty.', schema, src_table)
        return gpd.GeoDataFrame(df, geometry=[], crs=crs)

    t1 = time.perf_counter()
    df[geom_col] = df[geom_col].map(_to_bytes)
    cast_s = _fmt_seconds(time.perf_counter() - t1)

    t2 = time.perf_counter()
    geom = gpd.GeoSeries.from_wkb(df[geom_col], crs=crs)
    gdf = gpd.GeoDataFrame(df.drop(columns=[geom_col]), geometry=geom, crs=crs)
    build_s = _fmt_seconds(time.perf_counter() - t2)
    logger.info("WKB cast={} | GeoDataFrame build={}", cast_s, build_s)

    bad = gdf.geometry.is_empty | gdf.geometry.isna()
    if bad.any():
        logger.warning("Dropping {} rows with empty/NA geometry", int(bad.sum()))
        gdf = gdf.loc[~bad].copy()

    _profile_gdf(gdf, crs=crs)
    _log_mem("after read")
    return gdf


# ======================
# Feature makers control
# ======================

def _available_feature_makers(makers_cfg: Optional[List[str]]) -> List[Any]:
    all_names = [
        "SizeScaleFeatures",
        "ElongationOrientationFeatures",
        "CompactnessCircularityFeatures",
        "EdgeComplexityFeatures",
        "MomentInertiaFeatures",
        "EdgeContextFeatures",
        "ExtremeEnvelopeFeatures",
    ]
    wanted = makers_cfg or all_names
    makers: List[Any] = []
    for name in wanted:
        cls = getattr(GFM, name, None)
        if cls is None:
            logger.warning("Żądany maker {} nie istnieje w GeometricFeaturesMaker — pomijam.", name)
            continue
        makers.append(cls)
    return makers

def _instantiate_maker(maker_cls: Any, mi_samples: int):
    kw: Dict[str, Any] = {"join": False}
    if maker_cls.__name__ == "MomentInertiaFeatures":
        try:
            return maker_cls(sample_points=mi_samples, **kw)
        except TypeError:
            return maker_cls(**kw)
    try:
        return maker_cls(**kw)
    except TypeError:
        return maker_cls()

def _run_maker(inst, gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    name = type(inst).__name__
    t0 = time.perf_counter()
    try:
        out = inst.transform(gdf)
        if not isinstance(out, pd.DataFrame):
            raise TypeError(f"{name}.transform() nie zwrócił DataFrame")
        dt = _fmt_seconds(time.perf_counter() - t0)
        ncols = out.shape[1]
        nan_share = (out.isna().sum().sum() / (out.shape[0] * max(out.shape[1], 1))) if out.size else 0.0
        logger.info("Maker {:>28}: time={}, cols={}, NaN~{:.1%}", name, dt, ncols, nan_share)
        _log_mem(f"after {name}")
        return out
    except Exception as e:
        dt = _fmt_seconds(time.perf_counter() - t0)
        logger.exception("Maker {} FAILED after {} → {}", name, dt, e)
        return pd.DataFrame(index=gdf.index)

def _compute_features(
    gdf: gpd.GeoDataFrame, *,
    id_col: str, year_col: str,
    makers_cfg: Optional[List[str]], mi_samples: int,
    decimals: int
) -> pd.DataFrame:
    makers = _available_feature_makers(makers_cfg)
    if not makers:
        raise RuntimeError("Brak dostępnych klas cech w src.features.GeometricFeaturesMaker")

    logger.info("Computing geometric features using {} maker(s)…", len(makers))
    blocks: List[pd.DataFrame] = []
    for cls in makers:
        inst = _instantiate_maker(cls, mi_samples)
        blocks.append(_run_maker(inst, gdf))

    feat = pd.concat(blocks, axis=1)
    out = pd.concat([gdf[[id_col, year_col]].reset_index(drop=True),
                     feat.reset_index(drop=True)], axis=1)

    key_cols = [id_col, year_col]
    feat_cols = [c for c in out.columns if c not in key_cols]
    out = out[key_cols + sorted(feat_cols)]

    for c in feat_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = _round_numeric_features(out, keys=(id_col, year_col), decimals=decimals)
    _profile_features(out, keys=(id_col, year_col))
    return out


# ==========
# Write back
# ==========

def _write_features_to_duckdb(
    con: duckdb.DuckDBPyConnection, df: pd.DataFrame, *,
    schema: str, dst_table: str, id_col: str, year_col: str
) -> None:
    t0 = time.perf_counter()
    logger.info('Writing to {}."{}" (CREATE OR REPLACE)…', schema, dst_table)
    con.execute(f'CREATE SCHEMA IF NOT EXISTS {schema};')
    con.register("df_features_tmp", df)
    con.execute(f'CREATE OR REPLACE TABLE {schema}."{dst_table}" AS SELECT * FROM df_features_tmp;')
    logger.success('Table {}."{}" ready for joins USING ({}, {}), write_time={}',
                   schema, dst_table, id_col, year_col, _fmt_seconds(time.perf_counter() - t0))


# =============
# Orchestrator
# =============

def run_add_geometric_features(cfg: DictConfig) -> None:
    # Parametry ogólne
    db_path   = Path(str(_cfg_get(cfg, "data.duckdb_path"))).resolve()
    threads   = _cfg_get(cfg, "duckdb.threads", None)
    mem       = _cfg_get(cfg, "duckdb.memory_limit", None)
    crs       = str(_cfg_get(cfg, "_global_.crs_target", _cfg_get(cfg, "features.crs_target", "EPSG:2180")))

    # Parametry kroku
    schema     = _cfg_get(cfg, "features.add_geometric_features.schema", _cfg_get(cfg, "duckdb.schema", "egib"))
    src_table  = _cfg_get(cfg, "features.add_geometric_features.source_table", "DzialkaEwidencyjna")
    dst_table  = _cfg_get(cfg, "features.add_geometric_features.dest_table", "GeometricFeatures")
    id_col     = _cfg_get(cfg, "features.add_geometric_features.id_column", "iddzialki")
    year_col   = _cfg_get(cfg, "features.add_geometric_features.year_column", "year")
    geom_col   = _cfg_get(cfg, "features.add_geometric_features.geom_column", "geometry")
    decimals   = int(_cfg_get(cfg, "features.add_geometric_features.decimals", 3))
    makers_cfg = _cfg_get(cfg, "features.add_geometric_features.makers", None)
    mi_samples = int(_cfg_get(cfg, "features.add_geometric_features.moment_inertia.sample_points", 500))

    logger.info(
        "CFG[geom_feat]: db={}, schema={}, src={}, dst={}, id_col={}, year_col={}, geom_col={}, crs={}, decimals={}, makers={}, mi_samples={}",
        db_path, schema, src_table, dst_table, id_col, year_col, geom_col, crs, decimals, makers_cfg or "ALL", mi_samples
    )
    _log_mem("start")

    con = _connect_duckdb(db_path, threads=threads, mem=mem)
    try:
        gdf = _read_parcels_as_gdf(
            con, crs=crs, schema=schema, src_table=src_table,
            id_col=id_col, year_col=year_col, geom_col=geom_col
        )
        if gdf.empty:
            logger.warning("Brak działek do przetworzenia — kończę krok.")
            return

        t0 = time.perf_counter()
        df_feat = _compute_features(
            gdf, id_col=id_col, year_col=year_col,
            makers_cfg=makers_cfg, mi_samples=mi_samples,
            decimals=decimals
        )
        logger.info("Feature computation total time={}", _fmt_seconds(time.perf_counter() - t0))

        if _cfg_get(cfg, "features.add_geometric_features.add_to_duckdb.enabled", True):
            _write_features_to_duckdb(con, df_feat, schema=schema, dst_table=dst_table,
                                      id_col=id_col, year_col=year_col)
        else:
            logger.info("Konfiguracja: add_to_duckdb.enabled = False → pomijam zapis.")
    finally:
        con.close()
        _log_mem("end")
