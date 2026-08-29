"""Clean GeoParquet files *in‑place* and optionally append the result to DuckDB.

Logging – now verbose again: every sizeable action is logged with `loguru`.
"""

# src/prepare_data/clean_dataset.py
from __future__ import annotations

import re
from pathlib import Path

import duckdb
import geopandas as gpd
import pandas as pd
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.common.duckdb_utils import read_geoparquet, write_geoparquet

# --------------------------------------------------------------------------------------
# UTILS
# --------------------------------------------------------------------------------------


def _std_geom_name(gdf: gpd.GeoDataFrame, target: str = "geometry") -> gpd.GeoDataFrame:
    if not isinstance(gdf, gpd.GeoDataFrame) or gdf.geometry is None:
        raise ValueError("GeoDataFrame bez aktywnej geometrii.")
    cur = gdf.geometry.name
    if cur == target:
        return gdf
    if target in gdf.columns:
        gdf = gdf.drop(columns=[target])
    logger.debug("Geometry column renamed {} ➜ {}", cur, target)
    return gdf.rename(columns={cur: target}).set_geometry(target)


def _lowercase_preserve_geom(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    geom = gdf.geometry.name
    gdf = gdf.rename(columns=str.lower)
    logger.debug("Columns lower‑cased (geometry stays '{}')", geom.lower())
    return gdf.set_geometry(geom.lower())


def _cast_year(series: pd.Series, fallback: int | None) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce").astype("Int64")
    if fallback is not None:
        out = out.fillna(fallback).astype("Int64")
    return out


def _parse_id(df: pd.DataFrame, id_col: str, pat: re.Pattern) -> pd.DataFrame:
    if id_col and id_col in df.columns:
        parsed = df[id_col].astype(str).str.strip().str.extract(pat)
        new_cols = [c for c in parsed.columns if c not in df.columns]
        if new_cols:
            df = pd.concat([df, parsed[new_cols]], axis=1)
            logger.debug("Parsed {} → added cols {}", id_col, new_cols)
    return df


def _deduplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df.columns)
    df = df.loc[:, ~df.columns.duplicated()]
    removed = before - len(df.columns)
    if removed:
        logger.debug("Dropped {} duplicated columns", removed)
    return df


# --------------------------------------------------------------------------------------
# PARQUET SAFETY READER
# --------------------------------------------------------------------------------------


def _read_parquet_safe(path: Path) -> gpd.GeoDataFrame:
    # read_geoparquet reads via DuckDB, not pyarrow. Extraction imports osgeo (GDAL) earlier in
    # the same process, which corrupts pyarrow's filesystem registry so gpd.read_parquet raises
    # ArrowKeyError ('file' scheme already registered) or segfaults. DuckDB sidesteps pyarrow
    # entirely and also decodes a dictionary-encoded 'year' column natively.
    return read_geoparquet(path)


# --------------------------------------------------------------------------------------
# DUCKDB HELPERS
# --------------------------------------------------------------------------------------


def _duck_table_columns(con: duckdb.DuckDBPyConnection, full_tbl: str) -> list[str]:
    cols = con.execute(f"PRAGMA table_info({full_tbl});").fetchall()
    return [c[1] for c in cols]


# ---------------------------------------------------------------------------
#  uniwersalne _append_to_duckdb :
#  • kolumna geometry w GeoDataFrame → WKB (bytes)
#  • w SQL konwertujemy **tylko wtedy**, gdy to naprawdę BLOB
#    (ST_GeomFromWKB  obsługuje zarówno BLOB, jak i już GEOMETRY)
# ---------------------------------------------------------------------------
def _append_to_duckdb(
    con: duckdb.DuckDBPyConnection,
    gdf: gpd.GeoDataFrame,
    layer: str,
    tables_created: set[str],
    schema: str = "egib",
) -> None:
    if gdf.empty:
        logger.debug("DuckDB: skipped empty df for layer '{}'", layer)
        return

    # -- 0) DataFrame → WKB --------------------------------------------------
    df_db = pd.DataFrame(gdf)
    df_db["geometry"] = gdf.geometry.to_wkb()
    df_db = _deduplicate_columns(df_db)

    full_tbl = f'{schema}."{layer}"'
    con.register("_clean_df", df_db)  # tymczasowa tabela w pamięci

    # -- 1) CREATE TABLE (jednorazowo) --------------------------------------
    if layer not in tables_created:
        cols_def = ", ".join(
            "ST_GeomFromWKB(geometry) AS geometry" if c == "geometry" else f'"{c}"'
            for c in df_db.columns
        )
        con.execute(f"""
            CREATE OR REPLACE TABLE {full_tbl} AS
            SELECT {cols_def}
            FROM _clean_df
            WHERE FALSE;                -- pusta tabela z prawidłowym schematem
        """)
        tables_created.add(layer)
        logger.debug("DuckDB: created empty table {}", full_tbl)

    # -- 2) INSERT -----------------------------------------------------------
    tbl_cols = _duck_table_columns(con, full_tbl)
    insert_cols = [c for c in df_db.columns if c in tbl_cols]

    if insert_cols:
        cols_sql = ", ".join(f'"{c}"' for c in insert_cols)
        select_sql = ", ".join(
            "ST_GeomFromWKB(geometry)" if c == "geometry" else f'"{c}"' for c in insert_cols
        )
        con.execute(f"""
            INSERT INTO {full_tbl} ({cols_sql})
            SELECT {select_sql}
            FROM _clean_df;
        """)
        logger.info("DuckDB: +{} rows → {}", len(df_db), full_tbl)

    con.unregister("_clean_df")


# --------------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------------


def run_clean_dataset(cfg: DictConfig) -> None:
    if not bool(OmegaConf.select(cfg, "prepare.clean_dataset.enabled", default=True)):
        logger.info("CLEAN_DATASET disabled in config – skipping.")
        return

    base_dir = Path(OmegaConf.select(cfg, "data.base_dir")).expanduser().resolve()
    root = base_dir / str(OmegaConf.select(cfg, "prepare.output_subdir", default="parquets"))

    layers = list(OmegaConf.select(cfg, "prepare.clean_dataset.layers", default=[]))
    units_filter = set(OmegaConf.select(cfg, "prepare.clean_dataset.units", default=[]))

    id_col = str(
        OmegaConf.select(cfg, "prepare.clean_dataset.id_column", default="iddzialki")
    ).lower()
    regex = str(
        OmegaConf.select(
            cfg,
            "prepare.clean_dataset.id_pattern",
            default=r"^\s*(?P<jednostka>\d{6}_\d)\.(?P<obreb>\d{4})\.(?P<nr_dzialki>\d+(?:/\d+)*)\s*$",
        )
    )

    apply = "prepare.clean_dataset.apply"
    apply_lowercase = bool(OmegaConf.select(cfg, f"{apply}.lowercase", default=False))
    drop_cols = [c.lower() for c in OmegaConf.select(cfg, f"{apply}.drop_columns", default=[])]
    crs_target = OmegaConf.select(cfg, f"{apply}.crs_target", default=None)

    overwrite = bool(OmegaConf.select(cfg, "prepare.clean_dataset.overwrite", default=False))
    write_db = bool(OmegaConf.select(cfg, "prepare.clean_dataset.write_duckdb", default=False))
    db_path = Path(
        str(OmegaConf.select(cfg, "data.duckdb_path", default="artifacts/duckdb/terramerge.duckdb"))
    ).expanduser()

    pat = re.compile(regex)

    if not root.exists():
        logger.error("Katalog danych nie istnieje: {}", root)
        return

    unit_dirs = [
        d for d in root.iterdir() if d.is_dir() and (not units_filter or d.name in units_filter)
    ]
    if not unit_dirs:
        logger.warning("Brak jednostek do przetworzenia w: {}", root)
        return

    con = None
    created: set[str] = set()
    if write_db:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect(str(db_path))
        con.execute("INSTALL spatial; LOAD spatial; CREATE SCHEMA IF NOT EXISTS egib;")
        logger.info("DuckDB output ON → {}", db_path)

    logger.info(
        "CLEAN_DATASET start | layers={} | overwrite={} | write_db={}", layers, overwrite, write_db
    )

    for unit_dir in tqdm(unit_dirs, desc="🏷️  Jednostki"):
        for layer in layers:
            layer_dir = unit_dir / layer
            if not layer_dir.exists():
                logger.debug("Unit '{}' lacks layer '{}'", unit_dir.name, layer)
                continue

            for ydir in [
                p for p in layer_dir.iterdir() if p.is_dir() and p.name.startswith("year=")
            ]:
                try:
                    yr = int(ydir.name.split("=", 1)[-1])
                except ValueError:
                    logger.warning("Unexpected year dir: {}", ydir)
                    continue

                for f in sorted(ydir.glob("*.parquet")):
                    try:
                        gdf = _read_parquet_safe(f)
                        if not isinstance(gdf, gpd.GeoDataFrame) or gdf.geometry is None:
                            logger.error("File without geometry {}", f.name)
                            continue

                        gdf = _std_geom_name(gdf)
                        if apply_lowercase:
                            gdf = _lowercase_preserve_geom(gdf)
                        if drop_cols:
                            before = set(gdf.columns)
                            gdf = gdf.drop(
                                columns=[c for c in drop_cols if c in gdf.columns], errors="ignore"
                            )
                            removed = before - set(gdf.columns)
                            if removed:
                                logger.debug("Dropped cols {} from {}", sorted(removed), f.name)

                        gdf["year"] = _cast_year(gdf.get("year"), fallback=yr)
                        # Extraction already normalises every layer to the target CRS, so this is
                        # normally a no-op. Guard against naive geometry (crs is None): to_crs would
                        # raise "Cannot transform naive geometries" — skip it rather than crash.
                        if (
                            crs_target
                            and gdf.crs is not None
                            and gdf.crs.to_string() != str(crs_target)
                        ):
                            gdf = gdf.to_crs(crs_target, inplace=False)
                        gdf = _parse_id(gdf, id_col=id_col, pat=pat)
                        gdf = _deduplicate_columns(gdf)

                        if overwrite:
                            write_geoparquet(gdf, f)
                            logger.success("Overwritten: {}", f.relative_to(root))
                        else:
                            cleaned = f.with_suffix(".clean.parquet")
                            write_geoparquet(gdf, cleaned)
                            logger.success("Saved clean copy: {}", cleaned.relative_to(root))

                        if write_db and con is not None:
                            _append_to_duckdb(con, gdf, layer, created)

                    except Exception:
                        logger.exception("Cleaning failed for {}", f)

    if con is not None:
        con.close()

    logger.success("CLEAN_DATASET finished – all good ✨")
