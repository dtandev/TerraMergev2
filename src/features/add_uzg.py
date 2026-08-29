# src/prepare_data/add_uzg.py
"""
Build a single, cleaned GeoParquet layer that merges
• KKL  – Kontur Klasyfikacyjny
• UZG  – Kontur Użytku Gruntowego

The logic reproduces the “smart” flow, but without liczenia udziałów:
1.  normalise KKL & UZG    (rename, types, year, CRS, make_valid)
2.  append   KKL → UZG     (starsze lata, gdy UZG nie istniał)
3.  fill OZK z KKL         (contains → overlap → backfill po ID → backfill przestrzenny)
4.  zapisz GeoParquet      (…/parquets/<obreb>/kug.parquet)
   + opcjonalnie: CREATE/REPLACE TABLE egib.kug w DuckDB
Everything is driven by Hydra – new node features.add_uzg”.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import duckdb
import geopandas as gpd
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
from loguru import logger
from omegaconf import DictConfig
from pyarrow import compute as pc
from shapely import make_valid

from src.common.config_utils import sel as _sel
from src.features.features_makeover import FeaturesMakeover

# -----------------------------------------------------------------------------#
#  Helpers                                                                     #
# -----------------------------------------------------------------------------#


def _coalesce(df: pd.DataFrame, groups: dict[str, Sequence[str]]) -> pd.DataFrame:
    """Merge many-to-one column variants (first non-NA wins)."""
    for tgt, srcs in groups.items():
        present = [c for c in srcs if c in df.columns]
        if not present:
            continue
        out = df[present[0]].copy()
        for c in present[1:]:
            out = out.fillna(df[c])
        df[tgt] = out
        df.drop(columns=[c for c in present if c != tgt], inplace=True)
    return df


def _load_tree(root: Path, crs_out: str | None) -> gpd.GeoDataFrame:
    """Read all *input_data.parquet under root/year=*/."""
    parts: list[gpd.GeoDataFrame] = []
    for p in sorted(root.rglob("input_data.parquet")):
        try:
            gdf = gpd.read_parquet(p)
        except pa.ArrowTypeError as err:
            # typowy konflikt int64 vs dictionary w kolumnie 'year'
            if "year" in str(err):
                logger.warning("Re-casting <year> in {}", p)
                # > odczytaj fragmenty pojedynczo i rzuć 'year' na int64
                tables = []
                for frag in ds.dataset(p, format="parquet").get_fragments():
                    t = frag.to_table()
                    if pa.types.is_dictionary(t.schema.field("year").type):
                        t = t.set_column(
                            t.schema.get_field_index("year"), "year", pc.cast(t["year"], pa.int64())
                        )
                    tables.append(t)
                table = pa.concat_tables(tables, promote=True)
                gdf = gpd.GeoDataFrame(table.to_pandas(), geometry="geometry")
            else:
                raise

        # year from path
        if "year" not in gdf.columns:
            m = re.search(r"year=(\d{4})", str(p))
            if m:
                gdf["year"] = int(m.group(1))

        # enforce geometry column name
        if "geometry" not in gdf.columns and "GEOMETRY" in gdf.columns:
            gdf = gdf.rename(columns={"GEOMETRY": "geometry"})
        gdf = gdf.set_geometry("geometry")

        parts.append(gdf)

    if not parts:
        raise FileNotFoundError(f"No input_data.parquet under {root}")

    # align columns
    all_cols = sorted(set().union(*(df.columns for df in parts)))
    parts = [df.reindex(columns=all_cols) for df in parts]

    # unify CRS
    g0 = parts[0]
    for i in range(1, len(parts)):
        if parts[i].crs != g0.crs:
            parts[i] = parts[i].to_crs(g0.crs)

    gdf = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry", crs=g0.crs)
    if crs_out and str(gdf.crs) != crs_out:
        gdf = gdf.to_crs(crs_out)
    return gdf


# -----------------------------------------------------------------------------#
#  Normalisers                                                                 #
# -----------------------------------------------------------------------------#

_KKL_RENAME = {
    "G5OZU": "OZU",
    "G5OZK": "OZK",
    "G5IDK": "IDKONTURU",
    "ID": "GML_ID",
    "IDR": "IDENTIFIER",
    "G5PEW": "PRZESTRZENNAZW",
    "G5DTU": "STARTOBIEKT",
    "G5DTW": "STARTWERSJAOBIEKT",
    "GEOMETRY": "geometry",
    "Shape_Area": "SHAPE_AREA",
    "SHAPE_Area": "SHAPE_AREA",
    "Shape_Length": "SHAPE_LENGTH",
    "SHAPE_Length": "SHAPE_LENGTH",
}
_KKL_STRING = {
    "GML_ID",
    "IDENTIFIER",
    "LOKALNYID",
    "PRZESTRZENNAZW",
    "WERSJAID",
    "STARTOBIEKT",
    "STARTWERSJAOBIEKT",
    "IDKONTURU",
    "OZU",
    "OZK",
    "OZNACZENIETYPUGLEBY",
}
_UZG_RENAME = {
    "ID": "GML_ID",
    "IDR": "IDENTIFIER",
    "G5PEW": "PRZESTRZENNAZW",
    "G5DTU": "STARTOBIEKT",
    "G5DTW": "STARTWERSJAOBIEKT",
    "GEOMETRY": "geometry",
    "Shape_Area": "SHAPE_AREA",
    "SHAPE_Area": "SHAPE_AREA",
    "Shape_Length": "SHAPE_LENGTH",
    "SHAPE_Length": "SHAPE_LENGTH",
}
_UZG_COALESCE = {
    "OZU": ["OZU", "G5OZU", "G5OFU", "OFU"],
    "IDUZYTKU": ["IDUZYTKU", "G5IDT"],
}
_UZG_STRING = {
    "GML_ID",
    "IDENTIFIER",
    "LOKALNYID",
    "PRZESTRZENNAZW",
    "WERSJAID",
    "STARTOBIEKT",
    "STARTWERSJAOBIEKT",
    "IDUZYTKU",
    "OZU",
}


def _norm_kkl(df: gpd.GeoDataFrame, crs_out: str) -> gpd.GeoDataFrame:
    if "geometry" not in df.columns and "GEOMETRY" in df.columns:
        df = df.rename(columns={"GEOMETRY": "geometry"})
    df = df.set_geometry("geometry")
    df = df.rename(columns={c: _KKL_RENAME[c] for c in df.columns if c in _KKL_RENAME})
    if "WERSJAID" not in df.columns and "ST_OBJ" in df.columns:
        df["WERSJAID"] = df["ST_OBJ"]
    for c in df.columns.intersection(_KKL_STRING):
        df[c] = df[c].astype("string")
    df["geometry"] = make_valid(df.geometry)
    if crs_out and str(df.crs) != crs_out:
        df = df.to_crs(crs_out)
    return df


def _norm_uzg(df: gpd.GeoDataFrame, crs_out: str) -> gpd.GeoDataFrame:
    if "geometry" not in df.columns and "GEOMETRY" in df.columns:
        df = df.rename(columns={"GEOMETRY": "geometry"})
    df = df.set_geometry("geometry")
    df = _coalesce(df, _UZG_COALESCE)
    df = df.rename(columns={c: _UZG_RENAME[c] for c in df.columns if c in _UZG_RENAME})
    if "WERSJAID" not in df.columns and "ST_OBJ" in df.columns:
        df["WERSJAID"] = df["ST_OBJ"]
    for c in df.columns.intersection(_UZG_STRING):
        df[c] = df[c].astype("string")
    df["geometry"] = make_valid(df.geometry)
    if crs_out and str(df.crs) != crs_out:
        df = df.to_crs(crs_out)
    return df


# -----------------------------------------------------------------------------#
#  Merge logic (strict mode – only “within” + backfill)                         #
# -----------------------------------------------------------------------------#


def _append_early_kkl(kug: gpd.GeoDataFrame, kkl: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    min_year = kug["year"].dropna().astype(int).min()
    early = kkl[kkl["year"] < min_year].copy()
    if early.empty:
        return kug

    early = early.loc[:, ~early.columns.duplicated()]
    early["IDUZYTKU"] = early["IDKONTURU"].astype("string")

    # 👇   upewnij się, że ramka docelowa ma brakujące kolumny (w tym OZK)
    for col in early.columns:
        if col not in kug.columns:
            kug[col] = pd.NA

    # teraz keep-lista będzie zawierać OZK
    keep_cols = [c for c in kug.columns if c in early.columns]
    early = early[keep_cols]

    combined = pd.concat([kug, early], ignore_index=True)
    return gpd.GeoDataFrame(combined, geometry="geometry", crs=kug.crs)


def _fill_ozk_strict(kug: gpd.GeoDataFrame, kkl: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Uzupełnia OZK tylko gdy kontur UZG mieści się w KKL (within), następnie LOCF po ID."""
    if "OZK" not in kug.columns:
        kug = kug.assign(OZK=pd.NA)
    miss = kug["OZK"].isna()
    s = gpd.sjoin(
        kug.loc[miss, ["geometry"]],
        kkl[["OZK", "geometry", "year"]],
        how="left",
        predicate="within",
    )
    s = s.dropna(subset=["OZK"])
    if not s.empty:
        kug.loc[s.index, "OZK"] = s["OZK"].values

    # LOCF po IDUZYTKU (wstecz)
    kug = kug.sort_values(["IDUZYTKU", "year"])
    if "OZK" not in kug.columns:
        kug["OZK"] = pd.NA
    kug["OZK"] = kug["OZK"].astype("string")  # ← rzutowanie od razu
    kug["OZK"] = kug.groupby("IDUZYTKU", dropna=False)["OZK"].ffill()
    kug = kug.sort_index()

    return kug


# -----------------------------------------------------------------------------#
#  Driver                                                                       #
# -----------------------------------------------------------------------------#


def _process_one_obreb(base_dir: Path, obreb: str, target_crs: str, overwrite: bool) -> Path:
    """Return path to output GeoParquet."""
    parq_dir = base_dir / "parquets" / obreb
    kkl_root = parq_dir / "KonturKlasyfikacyjny"
    uzg_root = parq_dir / "KonturUzytkuGruntowego"

    # read & normalise
    gdf_kkl = _norm_kkl(_load_tree(kkl_root, target_crs), target_crs)
    gdf_kug = _norm_uzg(_load_tree(uzg_root, target_crs), target_crs)

    # merge
    gdf_kug = _append_early_kkl(gdf_kug, gdf_kkl)
    gdf_kug = _fill_ozk_strict(gdf_kug, gdf_kkl)

    # --- apply tabular feature makeovers (no geometry ops) ---
    fm = FeaturesMakeover()
    # column names in the normalized frame are uppercase ("OZU", "OZK")
    gdf_kug = fm.add_uzg_ozu_simple(gdf_kug, ozu_col="OZU", out_col="uzg_ozu_simple")
    gdf_kug = fm.add_uzg_bon_score(gdf_kug, ozk_col="OZK", out_col="uzg_bon_score")

    # keep minimal schema + new features
    gdf_kug = gdf_kug[
        ["IDUZYTKU", "OZU", "OZK", "year", "uzg_ozu_simple", "uzg_bon_score", "geometry"]
    ].copy()

    out_file = parq_dir / "kug.parquet"
    if out_file.exists() and not overwrite:
        logger.warning("File exists and overwrite=False: {}", out_file)
    else:
        gdf_kug.to_parquet(out_file, index=False)
        logger.success("Saved {}", out_file)

    return out_file


def run_add_uzg(cfg: DictConfig) -> None:
    """
    • generuje pliki <parquets>/<obreb>/kug.parquet            (jak dotąd)
    • jednym poleceniem ładuje WSZYSTKIE pliki do DuckDB:
        read_parquet([path1, path2, …], union_by_name = true)
    """
    if not _sel(cfg, "features.add_uzg.enabled", False):
        logger.info("STEP[add_uzg] Skipped (disabled)")
        return

    base_dir = Path(_sel(cfg, "data.base_dir")).expanduser().resolve()
    target_crs = str(_sel(cfg, "features.crs_target", "EPSG:2180"))
    overwrite = bool(_sel(cfg, "features.add_uzg.overwrite", True))

    # lista obrębów
    units: list[str] = list(_sel(cfg, "features.add_uzg.units", []))
    parq_root = base_dir / "parquets"
    if not units:
        units = [p.name for p in parq_root.iterdir() if p.is_dir()]

    logger.info("STEP[add_uzg] Start | units={} | CRS={}", units, target_crs)
    outputs: list[Path] = []

    # ---------- 1) generowanie kug.parquet per obręb ----------
    for obreb in units:
        try:
            outputs.append(_process_one_obreb(base_dir, obreb, target_crs, overwrite))
        except Exception:
            logger.exception("Unit FAILED: {}", obreb)
            raise
    logger.success("STEP[add_uzg] Done | {} units", len(outputs))

    # ---------- 2) opcjonalny import DuckDB (jeden strzał) ----------
    if not _sel(cfg, "features.add_uzg.write_duckdb", False):
        return

    db_path = (
        Path(
            _sel(
                cfg,
                "features.add_uzg.duckdb_path",
                _sel(cfg, "data.duckdb_path", base_dir / "egib.duckdb"),
            )
        )
        .expanduser()
        .resolve()
    )

    logger.info("Writing to DuckDB → {}", db_path)
    con = duckdb.connect(db_path)
    con.execute("INSTALL spatial")
    con.execute("LOAD spatial")
    con.execute("CREATE SCHEMA IF NOT EXISTS egib;")

    # lista plików Parquet do wczytania
    paths = ", ".join(f"'{p}'" for p in outputs)

    con.execute(f"""
        CREATE OR REPLACE TABLE egib.kug AS
        SELECT
            IDUZYTKU::VARCHAR AS iduzytku,
            OZU::VARCHAR      AS ozu,
            OZK::VARCHAR      AS ozk,
            year::INTEGER     AS year,
            uzg_ozu_simple::VARCHAR AS uzg_ozu_simple,
            CAST(uzg_bon_score AS DOUBLE) AS uzg_bon_score,
            geometry::GEOMETRY AS geometry      -- ← samo ::GEOMETRY wystarczy
        FROM read_parquet([{paths}], union_by_name = true);
    """)

    total = con.execute("SELECT COUNT(*) FROM egib.kug").fetchone()[0]
    logger.success("DuckDB: tabela egib.kug zawiera {} rekordów", total)
    con.close()
