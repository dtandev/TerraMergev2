# src/modeling/labeling.py

from __future__ import annotations
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import duckdb
from loguru import logger
from omegaconf import DictConfig

__all__ = [
    "run_creating_labels",
]

def _sel(cfg: DictConfig, path: str, default=None):
    """
    Safe nested selection from DictConfig using 'dot' path.
    """
    cur = cfg
    for part in path.split("."):
        if cur is None or part not in cur:
            return default
        cur = cur[part]
    return cur


def _qi(ident: str) -> str:
    """Quote a single SQL identifier for DuckDB."""
    return '"' + ident.replace('"', '""') + '"'

def _split_schema_table(full_name: str) -> Tuple[str, str]:
    """Split 'schema.table' into ('schema','table'); if no dot, assume 'main'."""
    if "." in full_name:
        schema, table = full_name.split(".", 1)
    else:
        schema, table = "main", full_name
    return schema, table

def _save_df_to_duckdb(con: duckdb.DuckDBPyConnection, df: pd.DataFrame, full_table: str) -> None:
    """
    Persist a DataFrame to DuckDB as `schema.table`.
    Creates the schema if missing and overwrites the table.
    """
    schema, table = _split_schema_table(full_table)
    con.execute(f"CREATE SCHEMA IF NOT EXISTS {_qi(schema)};")
    con.register("tmp_df_labels", df)
    con.execute(f"CREATE OR REPLACE TABLE {_qi(schema)}.{_qi(table)} AS SELECT * FROM tmp_df_labels;")
    con.unregister("tmp_df_labels")
    logger.info("Wrote {} rows → {}.{}", len(df), schema, table)


def _q(qualified_name: str) -> str:
    """Poprawne cytowanie schema.table dla DuckDB."""
    if "." not in qualified_name:
        return f'"{qualified_name}"'
    schema, table = qualified_name.split(".", 1)
    return f'"{schema}"."{table}"'


def build_split_labels_full(
    df: pd.DataFrame,
    *,
    hex_col: str = "hex_id",
    year_col: str = "year",
    mean_area_col: str = "shape_area_mean",
    sum_area_col: str = "coverage_area",
    n_parcels_col: Optional[str] = None,
    area_conservation_tol: float = 0.05,
    eps_abs: float = 1000.0,
    extra_horizon: Optional[int] = None,
) -> pd.DataFrame:
    """
    Vectorized split-label builder on a (hex_id, year) panel with no inner functions.

    Logic (computed per hex_id, at year t from transition t-1 -> t):
      delta_mean_area     = mean_area(t) - mean_area(t-1)
      sum_area_rel_change = (sum_area(t) - sum_area(t-1)) / sum_area(t-1), with 0→0 treated as 0
      area_conservation_ok = |sum_area_rel_change| <= area_conservation_tol
      delta_count         = n_parcels(t) - n_parcels(t-1)  (if n_parcels_col provided, else NA)
      split_rule_core     = (delta_mean_area < -eps_abs) & area_conservation_ok & (delta_count>0 if available)
      split_proxy         = split_rule_core
      split_onset         = split_proxy & ~shift(split_proxy, +1)
      y_next              = shift(split_proxy, -1)
      y_next_onset        = y_next & ~shift(y_next, +1)
      y_next_2, y_next_3  = shift(split_proxy, -2), shift(split_proxy, -3)
      y_next_<n>          = shift(split_proxy, -n) if extra_horizon>=4

    Returns
    -------
    DataFrame with:
      [hex_id, year, (n_parcels if present), mean_area_col, sum_area_col,
       delta_count, delta_mean_area, sum_area_rel_change, area_conservation_ok,
       split_rule_core, split_proxy, split_onset, y_next, y_next_onset, y_next_2, y_next_3, (optional y_next_<n>)]
    """
    # --- Select & sort ---
    cols: List[str] = [hex_col, year_col, mean_area_col, sum_area_col]
    has_count = n_parcels_col is not None and n_parcels_col in df.columns
    if has_count:
        cols.append(n_parcels_col)

    data = df[cols].copy()
    data = data.sort_values([hex_col, year_col], kind="mergesort").reset_index(drop=True)

    # --- Group key for vectorized shifts ---
    g = data.groupby(hex_col, sort=False, group_keys=False)

    # --- Deltas ---
    data["delta_mean_area"] = g[mean_area_col].shift(0) - g[mean_area_col].shift(1)

    if has_count:
        # Nullable integer for robustness on missing years
        delta_cnt = g[n_parcels_col].shift(0) - g[n_parcels_col].shift(1)
        data["delta_count"] = delta_cnt.astype("Int64")
    else:
        data["delta_count"] = pd.Series([pd.NA] * len(data), dtype="Int64")

    # --- Relative change of sum_area; treat 0→0 as 0 ---
    prev_sum = g[sum_area_col].shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = (data[sum_area_col].to_numpy() - prev_sum.to_numpy()) / prev_sum.to_numpy()
    rel = np.where((prev_sum.to_numpy() == 0) & (data[sum_area_col].to_numpy() == 0), 0.0, rel)
    data["sum_area_rel_change"] = rel
    data["area_conservation_ok"] = (np.abs(data["sum_area_rel_change"]) <= float(area_conservation_tol))

    # --- Core rule & proxy ---
    cond_drop = data["delta_mean_area"] < -float(eps_abs)
    cond_cons = data["area_conservation_ok"]
    if has_count:
        cond_cnt = data["delta_count"] > 0
        split_core = (cond_drop & cond_cons & cond_cnt)
    else:
        split_core = (cond_drop & cond_cons)

    # Use plain bool to simplify downstream usage
    data["split_rule_core"] = split_core.astype(bool)
    data["split_proxy"] = data["split_rule_core"]

    # --- Onset & forecasts (use shift with fill_value to avoid FutureWarning) ---
    prev_proxy = g["split_proxy"].shift(1, fill_value=False).astype(bool)
    data["split_onset"] = (data["split_proxy"] & (~prev_proxy)).astype(bool)

    data["y_next"]   = g["split_proxy"].shift(-1, fill_value=False).astype(bool)
    data["y_next_2"] = g["split_proxy"].shift(-2, fill_value=False).astype(bool)
    data["y_next_3"] = g["split_proxy"].shift(-3, fill_value=False).astype(bool)

    prev_y = g["y_next"].shift(1, fill_value=False).astype(bool)
    data["y_next_onset"] = (data["y_next"] & (~prev_y)).astype(bool)

    # Optional extra horizon ≥ 4
    extra_col_name: Optional[str] = None
    if extra_horizon is not None and int(extra_horizon) >= 4:
        h = int(extra_horizon)
        extra_col_name = f"y_next_{h}"
        data[extra_col_name] = g["split_proxy"].shift(-h, fill_value=False).astype(bool)

    # --- Assemble output with original column names ---
    out_cols = [
        hex_col, year_col,
        *( [n_parcels_col] if has_count else [] ),
        mean_area_col, sum_area_col,
        "delta_count", "delta_mean_area", "sum_area_rel_change", "area_conservation_ok",
        "split_rule_core", "split_proxy", "split_onset",
        "y_next", "y_next_onset", "y_next_2", "y_next_3",
    ]
    if extra_col_name is not None:
        out_cols.append(extra_col_name)

    return data[out_cols]

def build_uzg_conversion_labels(
    df: pd.DataFrame,
    *,
    hex_col: str = "hex_id",
    year_col: str = "year",
    share_R_col: str = "uzg_R_share",
    share_B_col: str = "uzg_B_share",
    sum_col: str = "sum_uzg",
    area_conservation_tol: float = 0.01,   # 1% dopuszczalna zmiana sumy udziałów
    extra_horizon: Optional[int] = None,
) -> pd.DataFrame:
    """
    Detect and forecast 'odrolnienie' (conversion from R to B) per (hex, year).

    Rules for conversion at transition t-1 → t:
      - ΔR = R(t) - R(t-1) < 0
      - ΔB = B(t) - B(t-1) > 0
      - |Δsum_rel| <= area_conservation_tol

    Returns DataFrame with:
      [hex_id, year,
       delta_R, delta_B, sum_rel_change, area_conservation_ok,
       convert_rule_core, convert_proxy, convert_onset,
       y_next, y_next_onset, y_next_2, y_next_3, (optional y_next_<n>)]
    """
    cols = [hex_col, year_col, share_R_col, share_B_col, sum_col]
    data = df[cols].copy().sort_values([hex_col, year_col]).reset_index(drop=True)
    g = data.groupby(hex_col, sort=False, group_keys=False)

    # --- delty ---
    data["delta_R"] = g[share_R_col].shift(0) - g[share_R_col].shift(1)
    data["delta_B"] = g[share_B_col].shift(0) - g[share_B_col].shift(1)

    prev_sum = g[sum_col].shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = (data[sum_col] - prev_sum) / prev_sum
    rel = np.where((prev_sum == 0) & (data[sum_col] == 0), 0.0, rel)
    data["sum_rel_change"] = rel
    data["area_conservation_ok"] = np.abs(data["sum_rel_change"]) <= area_conservation_tol

    # --- core rule: R↓, B↑, area stable ---
    core = (data["delta_R"] < 0) & (data["delta_B"] > 0) & (data["area_conservation_ok"])
    data["convert_rule_core"] = core.astype(bool)
    data["convert_proxy"] = data["convert_rule_core"]

    # --- onset detection ---
    prev_conv = g["convert_proxy"].shift(1, fill_value=False).astype(bool)
    data["convert_onset"] = (data["convert_proxy"] & (~prev_conv)).astype(bool)

    # --- forecasts ---
    data["y_next"]   = g["convert_proxy"].shift(-1, fill_value=False).astype(bool)
    data["y_next_2"] = g["convert_proxy"].shift(-2, fill_value=False).astype(bool)
    data["y_next_3"] = g["convert_proxy"].shift(-3, fill_value=False).astype(bool)

    prev_y = g["y_next"].shift(1, fill_value=False).astype(bool)
    data["y_next_onset"] = (data["y_next"] & (~prev_y)).astype(bool)

    if extra_horizon is not None and int(extra_horizon) >= 4:
        h = int(extra_horizon)
        coln = f"y_next_{h}"
        data[coln] = g["convert_proxy"].shift(-h, fill_value=False).astype(bool)

    out_cols = [
        hex_col, year_col,
        "delta_R", "delta_B", "sum_rel_change", "area_conservation_ok",
        "convert_rule_core", "convert_proxy", "convert_onset",
        "y_next", "y_next_onset", "y_next_2", "y_next_3",
    ]
    if extra_horizon is not None and int(extra_horizon) >= 4:
        out_cols.append(f"y_next_{int(extra_horizon)}")

    return data[out_cols]


def run_creating_labels(cfg) -> None:
    """
    Główna funkcja tworząca etykiety i zapisująca je do bazy DuckDB.
    """     
    
    # Połączenie z bazą
    db_path = Path(cfg.data.duckdb_path).expanduser()
    con = duckdb.connect(db_path.as_posix())


    if _sel(cfg, "dataset.labels_for_parcels.enabled", False):
        logger.info("Tworzenie etykiet podziału działek...")
        parcels_table = _sel(cfg, "dataset.labels_for_parcels.table", "")
        parcels_df = con.execute(f"SELECT * FROM {_q(parcels_table)};").df()
        logger.info("Wczytano tabelę {} ({} wierszy, {} kolumn)",
                    parcels_table, len(parcels_df), len(parcels_df.columns))

        labels_parcels_df = build_split_labels_full(
            parcels_df,
            hex_col=_sel(cfg, "dataset.labels_for_parcels.hex_col", "hex_id"),
            year_col=_sel(cfg, "dataset.labels_for_parcels.year_col", "year"),
            mean_area_col=_sel(cfg, "dataset.labels_for_parcels.mean_area_col", "shape_area_mean"),
            sum_area_col=_sel(cfg, "dataset.labels_for_parcels.sum_area_col", "coverage_area"),
            n_parcels_col = _sel(cfg, "dataset.labels_for_parcels.parcels_split_col_name", 'n_parcel'),
            area_conservation_tol=_sel(cfg, "dataset.labels_for_parcels.area_conservation_tol", 0.02),
            eps_abs=_sel(cfg, "dataset.labels_for_parcels.eps_abs", 100.0),
            extra_horizon=_sel(cfg, "dataset.labels_for_parcels.extra_horizon", None),
        )

        logger.info("Wygenerowano etykiety podziału działek.")
        logger.info(f'Liczba etykiet: {labels_parcels_df["y_next"].value_counts()}')

        # --- Write to DuckDB ---
        out_table_parcels = _sel(cfg, "dataset.labels_for_parcels.out_table",
                                 f'labels.ParcelLabels_{_sel(cfg, "dataset.resolution", "r8")}')
        if len(labels_parcels_df) == 0:
            logger.warning("Brak wierszy do zapisu (parcels). Zapis pominięty.")
        else:
            if con is not None:
                logger.info(f"Połączono z bazą DuckDB → {db_path}")
                _save_df_to_duckdb(con, labels_parcels_df, out_table_parcels)
                logger.success("Saved labels for parcels.")
                logger.success("STEP[prepare labels] Done")
            else:
                logger.error("Brak połączenia z bazą DuckDB. Zapis pominięty.")


    if _sel(cfg, "dataset.labels_for_uzg.enabled", False):
        logger.info("Tworzenie etykiet konwersji dla uzg...")
        kug_table  = _sel(cfg, "dataset.labels_for_uzg.table", "")
        kug_df = con.execute(f"SELECT * FROM {_q(kug_table)};").df()
        logger.info("Wczytano tabelę {} ({} wierszy, {} kolumn)",
                kug_table, len(kug_df), len(kug_df.columns))
        
        uzg = [
            'uzg_R_share',
            'uzg_Ł_share',
            'uzg_Ps_share',
            'uzg_N_share',
            'uzg_L_share',
            'uzg_dr_share',
            'uzg_B_share',
            'uzg_W_share',
            'uzg_S_share',
            'uzg_T_share',
            'uzg_NB_share',
            'uzg_K_share',
            'uzg_E_share',
            'uzg_O_share',
        ]

        agricultural = [
            'uzg_R_share',
            'uzg_Ł_share',
            'uzg_Ps_share',
            'uzg_S_share',
        ]

        logger.info("Obliczanie sum udziałów dla uzg i gruntów rolnych...")
        kug_df['sum_uzg'] = kug_df[uzg].sum(axis=1)
        kug_df['sum_agri'] = kug_df[agricultural].sum(axis=1)

        agri_classes =_sel(cfg, "dataset.labels_for_uzg.agri_classes", ["uzg_R_share"])
        base_out_table = _sel(cfg, "dataset.labels_for_uzg.out_table",
                              f'labels.kugLabels_{_sel(cfg, "dataset.resolution", "r8")}')


        for cls in agri_classes:
            logger.info(f"Tworzenie etykiet konwersji dla klasy: {cls}")
            labels_uzg_df = build_uzg_conversion_labels(
                kug_df,
                hex_col=_sel(cfg, "dataset.labels_for_uzg.hex_col", "hex_id"),
                year_col=_sel(cfg, "dataset.labels_for_uzg.year_col", "year"),
                share_R_col=cls,
                share_B_col=_sel(cfg, "dataset.labels_for_uzg.share_B_col", "uzg_B_share"),
                sum_col="sum_uzg",
                area_conservation_tol=_sel(cfg, "dataset.labels_for_uzg.area_conservation_tol", 0.01),
                extra_horizon=_sel(cfg, "dataset.labels_for_uzg.extra_horizon", None),
            )
            logger.info("Wygenerowano etykiety podziału uzg.")
            logger.info(f'Liczba etykiet: {labels_uzg_df["y_next"].value_counts()}')

            schema, table = _split_schema_table(base_out_table)
            out_table_with_cls = f"{schema}.{table}_{cls}"

            if len(labels_uzg_df) == 0:
                logger.warning(f"Brak wierszy do zapisu (uzg, {cls}). Zapis pominięty.")
            else:
                if con is not None:
                    logger.info(f"Połączono z bazą DuckDB → {db_path}")
                    _save_df_to_duckdb(con, labels_uzg_df, out_table_with_cls)
                    logger.success("Saved labels for uzg class: {}", cls)
                    logger.success("STEP[prepare labels] Done")
                else:
                    logger.error("Brak połączenia z bazą DuckDB. Zapis pominięty.")

    con.close()
