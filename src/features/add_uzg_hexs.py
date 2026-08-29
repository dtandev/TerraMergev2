# add_kug_hexs.py  — UZG (użytki) + bonitacja → udziały w heksach, z logowaniem loguru
from pathlib import Path

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
from loguru import logger

from src.common.config_utils import sel as _sel
from src.common.duckdb_utils import _detect_srid, connect_duckdb, save_geodf_as_ewkb_geometry


# ----------------- Logging setup (lekki, opcjonalny) ----------------- #
def configure_logging(cfg) -> None:
    """
    Minimalne ustawienie loggera.
    Jeśli w cfg istnieje logging.level/format — użyje ich; w przeciwnym razie sensowne domyślne.
    """
    level = _sel(cfg, "logging.level", "INFO")
    fmt = _sel(
        cfg,
        "logging.format",
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>",
    )
    try:
        logger.remove()
    except Exception:
        pass
    logger.add(lambda msg: print(msg, end=""), level=level, format=fmt, enqueue=False)


# ----------------- Helpers ----------------- #
def _get_hex_params(cfg):
    """Czyta parametry heksów z cfg.pipeline.hex.*."""
    table = _sel(cfg, "pipeline.hex.table")
    id_col = _sel(cfg, "pipeline.hex.id_col", "hex_id")
    geom_col = _sel(cfg, "pipeline.hex.geom_col", "geometry")
    if not table:
        raise ValueError("Brak nazwy tabeli heksów: oczekiwano 'pipeline.hex.table'.")
    logger.debug(f"HEX params → table={table}, id_col={id_col}, geom_col={geom_col}")
    return table, id_col, geom_col


def _get_enforce_crs(cfg) -> str:
    crs = _sel(cfg, "pipeline.layer_defaults.enforce_crs", "EPSG:2180")
    logger.debug(f"enforce_crs = {crs}")
    return crs


# ----------------- Loaders ----------------- #
def load_hex_gdf(
    con: duckdb.DuckDBPyConnection,
    cfg,
    *,
    limit: int | None = None,
) -> gpd.GeoDataFrame:
    """
    Wczytaj heksy z ID i polem heksa (m2). CRS z ST_SRID (fallback: enforce_crs).
    """
    table, id_col, geom_col = _get_hex_params(cfg)
    enforce = _get_enforce_crs(cfg)

    srid = _detect_srid(con, table, geom_col)
    crs_str = f"EPSG:{srid}" if srid else enforce
    lim = f" LIMIT {int(limit)}" if (isinstance(limit, int) and limit > 0) else ""

    sql = f"""
        SELECT
          "{id_col}" AS hex_id,
          ST_AsWKB({geom_col}) AS _wkb,
          CAST(ST_Area({geom_col}) AS DOUBLE) AS hex_area_m2
        FROM {table}
        {lim}
    """
    logger.info(f"[HEX] Ładowanie heksów z {table} …")
    df = con.execute(sql).df()
    logger.debug(f"[HEX] Otrzymano {len(df)} wierszy.")
    geoms = gpd.GeoSeries.from_wkb(
        df.pop("_wkb").map(lambda b: bytes(b) if b is not None else None),
        crs=crs_str,
    )
    gdf = gpd.GeoDataFrame(df, geometry=geoms, crs=crs_str)
    gdf = gdf[["hex_id", "hex_area_m2", "geometry"]]
    logger.success(f"[HEX] Gotowe: {len(gdf)} heksów, CRS={gdf.crs}.")
    return gdf


def load_kug_gdf_from_cfg(
    con: duckdb.DuckDBPyConnection,
    cfg,
    *,
    label_col: str = "uzg_ozu_simple",
    bon_col: str = "uzg_bon_score",
    year_col: str = "year",
    geom_col: str = "geometry",
    limit: int | None = None,
) -> gpd.GeoDataFrame:
    """
    Wczytaj KUG (cfg.pipeline.add_kug_data.table) jako GeoDataFrame (tylko potrzebne kolumny).
    """
    table = _sel(cfg, "pipeline.add_kug_data.table", "egib.kug")
    enforce = _get_enforce_crs(cfg)
    srid = _detect_srid(con, table, geom_col)
    crs_str = f"EPSG:{srid}" if srid else enforce
    lim = f" LIMIT {int(limit)}" if (isinstance(limit, int) and limit > 0) else ""

    sql = f"""
        SELECT
            {year_col} AS year,
            {label_col} AS {label_col},
            {bon_col}  AS {bon_col},
            ST_AsWKB({geom_col}) AS _wkb
        FROM {table}
        {lim}
    """
    logger.info(f"[KUG] Ładowanie danych z {table} …")
    df = con.execute(sql).df()
    logger.debug(f"[KUG] Otrzymano {len(df)} wierszy.")
    geoms = gpd.GeoSeries.from_wkb(
        df.pop("_wkb").map(lambda b: bytes(b) if b is not None else None),
        crs=crs_str,
    )
    gdf = gpd.GeoDataFrame(df, geometry=geoms, crs=crs_str)
    logger.success(f"[KUG] Gotowe: {len(gdf)} rekordów, CRS={gdf.crs}.")
    return gdf


# -------------- Core compute -------------- #
def kug_hex_shares(
    gdf_kug: gpd.GeoDataFrame,
    gdf_hex: gpd.GeoDataFrame,
    *,
    label_col: str = "uzg_ozu_simple",  # klasa UZG
    bon_col: str = "uzg_bon_score",  # bonitacja (liczbowa)
    year_col: str = "year",
    hex_id_col: str = "hex_id",
    hex_area_col: str = "hex_area_m2",
    classes: list[str] | None = None,  # stały zestaw kolumn
    decimals: int = 4,
    fill_missing_mean_with_zero: bool = False,
) -> gpd.GeoDataFrame:
    """
    Liczy:
      (1) udziały klas UZG per (hex, year),
      (2) średnią ważoną powierzchnią dla bon_col.
    Zwraca: [hex_id, year, uzg_<KLASA>_share..., <bon_col>_mean, geometry]
    """
    logger.info("[KUG×HEX] Start obliczeń udziałów i bonitacji.")
    if gdf_kug.crs != gdf_hex.crs:
        logger.warning(
            f"CRS mismatch (KUG={gdf_kug.crs}, HEX={gdf_hex.crs}) → reprojekcja KUG do HEX."
        )
        gdf_kug = gdf_kug.to_crs(gdf_hex.crs)

    need = {year_col, label_col, "geometry"}
    missing = need - set(gdf_kug.columns)
    if missing:
        msg = f"Brakuje wymaganych kolumn w KUG: {missing}"
        logger.error(msg)
        raise ValueError(msg)
    if bon_col not in gdf_kug.columns:
        logger.debug(f"Kolumna '{bon_col}' nieobecna — policzę tylko udziały, bon_mean będzie NaN.")
        gdf_kug = gdf_kug.assign(**{bon_col: np.nan})

    kug_use = gdf_kug[[year_col, label_col, bon_col, "geometry"]].copy()
    hex_use = gdf_hex[[hex_id_col, hex_area_col, "geometry"]].copy()

    logger.info("[KUG×HEX] Overlay(intersection)…")
    ix = gpd.overlay(kug_use, hex_use, how="intersection")
    logger.debug(f"[KUG×HEX] Intersections: {len(ix)}")
    if ix.empty:
        logger.warning("[KUG×HEX] Brak przecięć — zwracam pustą ramkę.")
        base_cols = [hex_id_col, year_col, f"{bon_col}_mean", "geometry"]
        out = gpd.GeoDataFrame(columns=base_cols, geometry=[], crs=gdf_hex.crs)
        if classes:
            for k in classes:
                out[f"uzg_{k}_share"] = []
        return out

    ix["__ix_area"] = ix.geometry.area.astype(float)

    # Udziały klas
    logger.info("[KUG×HEX] Liczę udziały per (hex, year, klasa)…")
    by_lbl = (
        ix.groupby([hex_id_col, year_col, label_col], dropna=False)["__ix_area"]
        .sum()
        .rename("area_in_hex")
        .reset_index()
        .merge(hex_use[[hex_id_col, hex_area_col]], on=hex_id_col, how="left")
    )
    by_lbl["share"] = (by_lbl["area_in_hex"] / by_lbl[hex_area_col]).clip(0.0, 1.0)

    class_values = (
        classes if classes is not None else (sorted(by_lbl[label_col].dropna().unique().tolist()))
    )
    logger.debug(f"[KUG×HEX] Klasy UZG: {class_values}")

    wide = by_lbl.pivot_table(
        index=[hex_id_col, year_col],
        columns=label_col,
        values="share",
        aggfunc="sum",
        fill_value=0.0,
    )
    for k in class_values:
        if k not in wide.columns:
            wide[k] = 0.0
    wide = wide.reindex(columns=class_values)
    wide.columns = [f"uzg_{str(c)}_share" for c in wide.columns]
    wide = wide.reset_index()
    logger.debug(f"[KUG×HEX] Pivot shape: {wide.shape}")

    # Średnia ważona bonitacji
    logger.info("[KUG×HEX] Liczę średnią ważoną bonitacji…")
    valid = ix[ix[bon_col].notna()].copy()
    if valid.empty:
        logger.warning("[KUG×HEX] Wszystkie wartości bonitacji to NaN — bon_mean będzie puste.")
        bon_mean = pd.DataFrame({hex_id_col: [], year_col: [], f"{bon_col}_mean": []})
    else:
        valid["__w"] = valid["__ix_area"]
        bon_num = (
            (valid[bon_col] * valid["__w"]).groupby([valid[hex_id_col], valid[year_col]]).sum()
        )
        bon_den = valid["__w"].groupby([valid[hex_id_col], valid[year_col]]).sum()
        bon_mean = (bon_num / bon_den.replace(0.0, np.nan)).rename(f"{bon_col}_mean").reset_index()

    out = wide.merge(bon_mean, on=[hex_id_col, year_col], how="left").merge(
        hex_use[[hex_id_col, "geometry"]], on=hex_id_col, how="left"
    )

    share_cols = [c for c in out.columns if c.endswith("_share")]
    out[share_cols] = out[share_cols].round(decimals)
    if f"{bon_col}_mean" in out.columns:
        if fill_missing_mean_with_zero:
            logger.info("[KUG×HEX] Wypełniam NaN w bon_mean zerem (wymuszone flagą).")
            out[f"{bon_col}_mean"] = out[f"{bon_col}_mean"].fillna(0.0)
        out[f"{bon_col}_mean"] = out[f"{bon_col}_mean"].round(decimals)

    gout = gpd.GeoDataFrame(out, geometry="geometry", crs=gdf_hex.crs)
    logger.success(f"[KUG×HEX] Wynik: {len(gout)} wierszy, {gout.shape[1]} kolumn.")
    return gout


# -------------- Orkiestracja -------------- #
def run_add_kug_hexs(cfg) -> str:
    """
    Pełny przepływ UZG: load hex + KUG → udziały + bon_mean → save.
    Zwraca pełną nazwę tabeli wynikowej (cfg.pipeline.add_kug_data.out_table lub fallback).
    """
    configure_logging(cfg)

    enabled = bool(_sel(cfg, "pipeline.add_kug_data.enabled", False))
    if not enabled:
        msg = "pipeline.add_kug_data.enabled = false — krok wyłączony w configu."
        logger.error(msg)
        raise RuntimeError(msg)

    db_path = Path(_sel(cfg, "data.duckdb_path")).expanduser()
    con = connect_duckdb(db_path)
    try:
        gdf_hex = load_hex_gdf(con, cfg)
        gdf_kug = load_kug_gdf_from_cfg(con, cfg)

        classes = _sel(cfg, "pipeline.add_kug_data.klasy_uzg", None)
        decimals = int(_sel(cfg, "pipeline.layer_defaults.decimals", 4))
        out_tbl = _sel(cfg, "pipeline.add_kug_data.out_table", None)
        if not out_tbl:
            schema = _sel(cfg, "pipeline.hex.schema", "hex")
            suffix = _sel(cfg, "pipeline.hex.out_suffix", "rX")
            out_tbl = f"{schema}.kug_{suffix}"
        logger.info(f"[CFG] out_table = {out_tbl}")

        res_gdf = kug_hex_shares(
            gdf_kug,
            gdf_hex,
            label_col="uzg_ozu_simple",
            bon_col="uzg_bon_score",
            year_col="year",
            hex_id_col="hex_id",
            hex_area_col="hex_area_m2",
            classes=classes,
            decimals=decimals,
            fill_missing_mean_with_zero=False,  # zostawiamy NaN jako brak danych
        )

        write_mode = _sel(cfg, "pipeline.layer_defaults.write_mode", "replace")
        srid_str = _get_enforce_crs(cfg)  # np. "EPSG:2180"
        srid = int(str(srid_str).split(":")[-1])

        casts = {}
        if "hex_id" in res_gdf.columns:
            casts["hex_id"] = "VARCHAR"
        if "year" in res_gdf.columns:
            casts["year"] = "INT"

        _ = save_geodf_as_ewkb_geometry(
            db_path=db_path,
            gdf=res_gdf,
            table=out_tbl,
            srid=srid,
            geom_col="geometry",
            write_mode=write_mode,
            casts=casts,
        )
        logger.success(f"[KUG] Wynik zapisany do: {out_tbl}")
        return out_tbl
    except Exception as e:
        logger.exception(f"[KUG] Błąd przetwarzania: {e}")
        raise
    finally:
        try:
            con.close()
            logger.debug("Połączenie z DuckDB zamknięte.")
        except Exception:
            pass
