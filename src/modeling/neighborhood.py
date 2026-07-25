# src/modeling/neighborhood.py
from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import lru_cache
from pathlib import Path
from typing import Literal

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import OmegaConf

from src.common.config_utils import sel as _sel
from src.common.duckdb_utils import connect_duckdb as _connect_spatial

# ──────────────────────────────────────────────────────────────────────────────
# DuckDB I/O helpers
# ──────────────────────────────────────────────────────────────────────────────


def _duckdb_columns(con: duckdb.DuckDBPyConnection, table: str) -> list[str]:
    """
    Return column names for a DuckDB table/view.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Open DuckDB connection.
    table : str
        Fully qualified table name.

    Returns
    -------
    list[str]
        Column name list.
    """
    return [r[1] for r in con.execute(f"PRAGMA table_info('{table}')").fetchall()]


def _to_bytes_safe(val: object) -> bytes | None:
    """
    Convert DuckDB BLOB-like values to bytes (handles bytes/bytearray/memoryview/list[int]).

    Parameters
    ----------
    val : object
        Input value potentially representing binary data.

    Returns
    -------
    bytes | None
        Converted bytes or None if conversion is not possible.
    """
    if val is None:
        return None
    if isinstance(val, bytes):
        return val
    if isinstance(val, (bytearray, memoryview)):
        return bytes(val)
    if isinstance(val, list):
        try:
            return bytes(val)
        except Exception:
            return None
    return None


def write_geodf_to_duckdb(
    con: duckdb.DuckDBPyConnection,
    gdf: gpd.GeoDataFrame,
    *,
    table: str,
    geom_col: str = "geometry",
    srid: int = 2180,
    casts: dict[str, str] | None = None,
) -> int:
    """
    Write a GeoDataFrame to DuckDB, overwriting the target table.

    Takes an already-open `con` (reused across this module's pipeline) rather than a db_path,
    which is why this isn't just a call to src.common.duckdb_utils.save_geodf_as_ewkb_geometry —
    that helper opens its own connection, and DuckDB only allows one read-write connection per
    database file at a time.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Open DuckDB connection with spatial extension loaded.
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame with geometry column.
    table : str
        Full table name to create (e.g. 'hex.DzialkaEwidencyjna_r7').
    geom_col : str, default 'geometry'
        Name of geometry column in DuckDB.
    srid : int, default 2180
        SRID assigned to geometry.
    casts : dict[str, str] | None
        Optional SQL type casts, e.g. {'hex_id': 'VARCHAR', 'year': 'INT'}.

    Returns
    -------
    int
        Number of rows written.
    """
    from shapely import set_srid
    from shapely import to_wkb as _to_wkb

    df = gdf.copy()
    df["__wkb__"] = df.geometry.apply(
        lambda g: _to_wkb(set_srid(g, srid), include_srid=True) if g is not None else None
    )
    non_geo_cols = [c for c in df.columns if c != gdf.geometry.name]
    con.register("df_in", df[non_geo_cols])

    casts = casts or {}
    cols_sql = [
        f'CAST("{c}" AS {casts[c]}) AS "{c}"' if c in casts else f'"{c}"'
        for c in non_geo_cols
        if c != "__wkb__"
    ]
    cols_sql_str = ", ".join(cols_sql)
    sep = ", " if cols_sql_str else ""
    con.execute(f"DROP TABLE IF EXISTS {table}")
    # ST_SetCRS makes the SRID embedded via to_wkb(include_srid=True) actually queryable through
    # ST_CRS() afterwards — ST_GeomFromWKB alone does not surface it in this DuckDB spatial version.
    con.execute(f"""
        CREATE TABLE {table} AS
        SELECT
            {cols_sql_str}{sep}
            ST_SetCRS(ST_GeomFromWKB(__wkb__), 'EPSG:{int(srid)}') AS {geom_col}
        FROM df_in
    """)
    return len(df)


def load_tables_with_labels_2180(
    db_path: Path,
    *,
    tables_to_split: list[str],
    table_with_labels: str,
    label_cols: list[str] = ["split_proxy"],
    geom_col: str = "geometry",
    join_keys: tuple[str, str] = ("hex_id", "year"),
) -> dict[str, gpd.GeoDataFrame]:
    """
    Load multiple DuckDB tables into (Geo)DataFrames and LEFT-join selected label
    columns from `table_with_labels` on `join_keys`. Geometry (if present) is read
    as WKB and assigned EPSG:2180 (assumes coordinates are already in 2180).

    Parameters
    ----------
    db_path : Path
        DuckDB database path.
    tables_to_split : list[str]
        Source tables to load (e.g., 'hex.DzialkaEwidencyjna_r8').
    table_with_labels : str
        Labels table containing at least `join_keys` and requested `label_cols`.
    label_cols : list[str], default ['split_proxy']
        Columns to join from the labels table.
    geom_col : str, default 'geometry'
        Geometry column name to extract as WKB if present.
    join_keys : (str, str), default ('hex_id', 'year')
        Join keys for the merge.

    Returns
    -------
    dict[str, gpd.GeoDataFrame]
        Mapping table_name → GeoDataFrame with merged labels and EPSG:2180 geometry.
    """
    key_a, key_b = join_keys
    con = _connect_spatial(db_path)

    lbl_select_cols = [f'"{key_a}" AS __k1', f'"{key_b}" AS __k2'] + [f'"{c}"' for c in label_cols]
    lbl_q = f"SELECT {', '.join(lbl_select_cols)} FROM {table_with_labels}"
    labels_df: pd.DataFrame = con.execute(lbl_q).fetchdf().drop_duplicates(subset=["__k1", "__k2"])

    out: dict[str, gpd.GeoDataFrame] = {}

    for tbl in tables_to_split:
        cols = set(_duckdb_columns(con, tbl))
        if geom_col in cols:
            q = f'SELECT t.*, ST_AsWKB(t."{geom_col}") AS __wkb_2180 FROM {tbl} AS t'
            df = con.execute(q).fetchdf()
            geos = gpd.GeoSeries.from_wkb(df["__wkb_2180"].map(_to_bytes_safe), crs="EPSG:2180")
            gdf = gpd.GeoDataFrame(
                df.drop(columns=[geom_col, "__wkb_2180"], errors="ignore"),
                geometry=geos,
                crs="EPSG:2180",
            )
        else:
            df = con.execute(f"SELECT * FROM {tbl}").fetchdf()
            gdf = gpd.GeoDataFrame(df)

        if key_a in gdf.columns and key_b in gdf.columns:
            gdf = gdf.merge(
                labels_df,
                how="left",
                left_on=[key_a, key_b],
                right_on=["__k1", "__k2"],
            ).drop(columns=["__k1", "__k2"], errors="ignore")
        else:
            for c in label_cols:
                if c not in gdf.columns:
                    gdf[c] = pd.NA

        out[tbl] = gdf

    con.close()
    return out


def merge_neighborhood_dict_sequential(
    tables: dict[str, gpd.GeoDataFrame],
    *,
    keys: tuple[str, str] = ("hex_id", "year"),
    geom_col: str = "geometry",
    base: str | None = None,
    enforce_one_to_one: bool = True,
    keep_source_prefix: bool = True,
    unify_cols: Iterable[str] = ("split_proxy", "jednostka"),
) -> gpd.GeoDataFrame:
    """
    Sequentially merge a dict of {table_name: GeoDataFrame} into a single GeoDataFrame.

    Rules
    -----
    - Keeps geometry from the base table.
    - Enforces one-to-one on (keys) if requested.
    - For columns in `unify_cols`, avoids prefixing and coalesces duplicates.

    Parameters
    ----------
    tables : dict[str, GeoDataFrame]
        Source tables mapped to GeoDataFrames.
    keys : (str, str), default ('hex_id', 'year')
        Join keys.
    geom_col : str, default 'geometry'
        Geometry column name to preserve from base.
    base : str | None, default None
        Name of base table; if None, the first entry is used.
    enforce_one_to_one : bool, default True
        Enforce uniqueness of keys within each table.
    keep_source_prefix : bool, default True
        Prefix non-key columns with source table name.
    unify_cols : Iterable[str], default ('split_proxy', 'jednostka')
        Columns to coalesce across merges.

    Returns
    -------
    GeoDataFrame
        Merged frame with base CRS.
    """
    key_a, key_b = keys
    items = list(tables.items())
    if not items:
        return gpd.GeoDataFrame(columns=[key_a, key_b, geom_col], geometry=geom_col)

    unify: set[str] = set(unify_cols or ())

    if base is None:
        base_name, merged = items[0][0], items[0][1].copy()
    else:
        base_name = base
        merged = tables[base_name].copy()

    if enforce_one_to_one and merged.duplicated([key_a, key_b]).any():
        raise ValueError(f"Base table '{base_name}' has non-unique keys {keys}.")

    for name, df in items:
        if name == base_name:
            continue

        cols = [c for c in df.columns if c != geom_col]
        for k in keys:
            if k not in cols:
                cols.append(k)
        tmp = df[cols].copy()

        if enforce_one_to_one and tmp.duplicated([key_a, key_b]).any():
            raise ValueError(f"Table '{name}' has non-unique keys {keys}.")

        if keep_source_prefix:
            rename_map = {c: f"{name}__{c}" for c in tmp.columns if c not in (*keys, *unify)}
            tmp = tmp.rename(columns=rename_map)

        merged = merged.merge(
            tmp,
            on=[key_a, key_b],
            how="left",
            validate="one_to_one" if enforce_one_to_one else None,
            suffixes=("", f"__{name}"),
        )

        for col in unify:
            col_new = f"{col}__{name}"
            if col_new in merged.columns:
                if col not in merged.columns:
                    merged[col] = pd.NA
                merged[col] = merged[col].where(merged[col].notna(), merged[col_new])
                merged.drop(columns=[col_new], inplace=True)

    if geom_col not in merged.columns:
        merged[geom_col] = None
    out = gpd.GeoDataFrame(merged, geometry=geom_col, crs=tables[base_name].crs)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# H3 neighbors: exact rings vs rolling disks
# ──────────────────────────────────────────────────────────────────────────────


def _import_h3() -> tuple[object, bool, Callable | None, Callable | None, Callable | None]:
    """
    Import H3 with v4→v3 fallback and return handy callables.

    Returns
    -------
    (h3_module, is_v4, grid_disk_fn, grid_ring_fn, k_ring_fn)
    """
    try:
        from h3.api import basic_str as h3s  # type: ignore

        is_v4 = True
    except Exception:
        import h3 as h3s  # type: ignore

        is_v4 = False

    grid_disk = getattr(h3s, "grid_disk", None) or getattr(h3s, "gridDisk", None)
    grid_ring = getattr(h3s, "grid_ring", None) or getattr(h3s, "gridRing", None)
    k_ring = getattr(h3s, "k_ring", None) or getattr(h3s, "kRing", None)
    return h3s, is_v4, grid_disk, grid_ring, k_ring


@lru_cache(maxsize=100_000)
def _neighbors_exact_ring(cell: str, ring: int) -> list[str]:
    """
    Return neighbors at exact graph distance == ring (not <= ring).

    Parameters
    ----------
    cell : str
        H3 cell id.
    ring : int
        Exact ring distance.

    Returns
    -------
    list[str]
        Neighbor ids at exact ring distance.
    """
    if ring < 1:
        return []
    h3s, is_v4, grid_disk, grid_ring, k_ring = _import_h3()

    if grid_ring is not None:
        try:
            return list(set(grid_ring(cell, ring)))
        except Exception:
            pass

    if grid_disk is not None:
        try:
            kr = set(grid_disk(cell, ring))
            prev = set(grid_disk(cell, ring - 1)) if ring > 1 else {cell}
            return list(kr - prev)
        except Exception:
            pass

    if k_ring is not None:
        if ring == 1:
            return list(set(k_ring(cell, 1)) - {cell})
        kr = set(k_ring(cell, ring))
        prev = set(k_ring(cell, ring - 1))
        return list(kr - prev)

    raise RuntimeError(
        "No suitable H3 neighbor function available (grid_ring/grid_disk/k_ring not found)."
    )


@lru_cache(maxsize=100_000)
def _neighbors_disk(cell: str, ring: int) -> list[str]:
    """
    Return neighbors at graph distance <= ring (disk), excluding `cell`.

    Parameters
    ----------
    cell : str
        H3 cell id.
    ring : int
        Disk radius (<= ring).

    Returns
    -------
    list[str]
        Neighbor ids within disk.
    """
    if ring < 1:
        return []
    h3s, is_v4, grid_disk, grid_ring, k_ring = _import_h3()

    if grid_disk is not None:
        try:
            return list(set(grid_disk(cell, ring)) - {cell})
        except Exception:
            pass

    if grid_ring is not None:
        try:
            acc = set()
            for r in range(1, ring + 1):
                acc |= set(grid_ring(cell, r))
            return list(acc - {cell})
        except Exception:
            pass

    if k_ring is not None:
        return list(set(k_ring(cell, ring)) - {cell})

    raise RuntimeError(
        "No suitable H3 neighbor function available (grid_disk/grid_ring/k_ring not found)."
    )


def build_h3_neighbors_edges(
    cells: Iterable[str],
    R_values: int | Iterable[int],
    *,
    rolling: bool = True,
) -> pd.DataFrame:
    """
    Build H3 neighbor edges limited to the provided set of 'cells'.

    If rolling=True:
        For R=2 include all neighbors within rings {1,2} (disk), tagged as R=2.
    If rolling=False:
        For R=2 include only exact ring==2 neighbors (no ring 1).

    Parameters
    ----------
    cells : Iterable[str]
        H3 cells present in the dataset.
    R_values : int | Iterable[int]
        Radii to compute.
    rolling : bool, default True
        Disk (<=R) if True, exact ring (==R) if False.

    Returns
    -------
    pd.DataFrame
        Columns: ['cell', 'neighbor', 'R'] filtered to provided `cells`.
    """
    cells_list: list[str] = list(dict.fromkeys(cells))
    cells_set = set(cells_list)

    if isinstance(R_values, int):
        rings: tuple[int, ...] = (int(R_values),)
    else:
        rings = tuple(sorted({int(r) for r in R_values if int(r) >= 1}))
    if not rings:
        return pd.DataFrame(columns=["cell", "neighbor", "R"])

    rows: list[tuple[str, str, int]] = []

    if rolling:
        for c in cells_list:
            for R in rings:
                for n in _neighbors_disk(c, R):
                    if n in cells_set:
                        rows.append((c, n, R))
    else:
        for c in cells_list:
            for R in rings:
                for n in _neighbors_exact_ring(c, R):
                    if n in cells_set:
                        rows.append((c, n, R))

    out = pd.DataFrame(rows, columns=["cell", "neighbor", "R"])
    if not out.empty:
        out = out.sort_values(["cell", "R", "neighbor"], kind="mergesort").reset_index(drop=True)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Aggregation over neighbors (same-year t)
# ──────────────────────────────────────────────────────────────────────────────


def _most_frequent_non_null(values: pd.Series) -> str | None:
    """
    Return the most frequent non-null value (mode). If tie, pick the first by count order.

    Parameters
    ----------
    values : pd.Series
        Series of categorical values.

    Returns
    -------
    str | None
        Mode or None if no non-null values.
    """
    s = values.dropna()
    if s.empty:
        return None
    counts = s.value_counts()
    return counts.index[0] if not counts.empty else None


def compute_neighbor_aggregates(
    df: pd.DataFrame,
    *,
    hex_col: str = "hex_id",
    year_col: str = "year",
    categorical_cols: Iterable[str] = ("jednostka",),
    geometry_col: str = "geometry",
    R_values: int | Iterable[int] = (1, 2),
    rolling: bool = True,  # True: disks (<=R); False: exact rings (==R)
    min_neighbors: int = 1,
    prefix: str = "nbr",
) -> pd.DataFrame:
    """
    Compute neighbor aggregates for all numeric columns (same year) and, for each categorical
    col, compute the mode among neighbors. Columns are appended per-R as wide columns:
    {prefix}_r{R}_n, {prefix}_r{R}_{numcol}_mean/median, {prefix}_r{R}_{catcol}_mode.

    Assumptions
    -----------
    - Input df has unique (hex_id, year).
    - geometry is per (hex_id, year) and is passed through unchanged.
    - All numeric columns except keys/geometry/categorical are aggregated with mean/median.
    - Each categorical column is aggregated as mode among neighbors.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with unique (hex_id, year).
    hex_col : str, default 'hex_id'
        Hex id column.
    year_col : str, default 'year'
        Year column.
    categorical_cols : Iterable[str], default ('jednostka',)
        Non-numeric columns for which neighbor mode will be computed.
    geometry_col : str, default 'geometry'
        Geometry column; passed through unchanged.
    R_values : int | Iterable[int], default (1, 2)
        Radii. If rolling=True, R means disk <=R; else exact ring==R.
    rolling : bool, default True
        Disk vs exact ring behavior.
    min_neighbors : int, default 1
        Minimum neighbor count to compute features for a (hex,year,R).
    prefix : str, default 'nbr'
        Prefix for output columns.

    Returns
    -------
    pd.DataFrame
        Original df joined with neighbor-derived features in wide format, one row per (hex_id, year).
    """
    # --- Preconditions ---
    key_counts = df.groupby([hex_col, year_col], dropna=False).size()
    if (key_counts > 1).any():
        raise ValueError("Input (hex_id, year) must be unique. Found duplicates.")

    cat_set = set(categorical_cols or ())
    exclude = {hex_col, year_col, geometry_col} | cat_set
    numeric_cols: list[str] = [
        c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude
    ]

    # Normalize R_values to a sorted unique list of ints
    if isinstance(R_values, (int, np.integer)):
        R_list = [int(R_values)]
    else:
        R_list = sorted({int(r) for r in R_values})

    # Precompute neighbor edges for all R in one call (assumes helper returns columns: ['cell','neighbor','R'])
    edges_all: pd.DataFrame = build_h3_neighbors_edges(
        df[hex_col].unique(), R_values=R_list, rolling=rolling
    )
    result = df.copy()  # accumulate features into original df

    if edges_all.empty:
        # No edges at all -> just add count columns per R as 0
        for R in R_list:
            result[f"{prefix}_r{R}_n"] = 0
        return result

    # For matching neighbors to same-year rows
    df_keys = df[[hex_col, year_col]].rename(columns={hex_col: "__nbr__"})

    # Iterate per-R and append columns in "wide" layout
    for R in R_list:
        edges_R = edges_all[edges_all["R"] == R]
        if edges_R.empty:
            # add count=0 for this R and continue
            result[f"{prefix}_r{R}_n"] = 0
            continue

        # Build (hex,year) x neighbors frame for this R
        baseR = df[[hex_col, year_col]].drop_duplicates()
        frame = baseR.merge(
            edges_R.rename(columns={"cell": hex_col}),
            on=hex_col,
            how="left",
            validate="many_to_many",
        ).rename(columns={"neighbor": "__nbr__"})

        # Count neighbors present in df for the same year
        f_cnt = frame.merge(df_keys, on=["__nbr__", year_col], how="left", suffixes=("", "_hit"))
        cnt_col = f"{prefix}_r{R}_n"
        nbr_n = (
            f_cnt.groupby([hex_col, year_col], dropna=False)["year"]
            .apply(lambda s: s.notna().sum())
            .reset_index(name=cnt_col)
        )

        # Filter keys with enough neighbors
        valid_keys = nbr_n[nbr_n[cnt_col] >= int(min_neighbors)][[hex_col, year_col]]
        # Prepare container for this R's features (always include count)
        r_features = nbr_n.copy()

        # Numeric aggregations (mean/median) among same-year neighbors
        if numeric_cols and not valid_keys.empty:
            nbr_vals = df[[hex_col, year_col] + numeric_cols].rename(columns={hex_col: "__nbr__"})
            f_same = frame.merge(nbr_vals, on=["__nbr__", year_col], how="left")
            g_num = f_same.groupby([hex_col, year_col], dropna=False)[numeric_cols].agg(
                ["mean", "median"]
            )
            # Flatten columns and add r{R} suffix
            g_num.columns = [
                f"{prefix}_r{R}_{c}_{stat}" for c, stat in g_num.columns.to_flat_index()
            ]
            g_num = g_num.reset_index()
            # Keep only valid (hex,year) if min_neighbors applies
            g_num = valid_keys.merge(g_num, on=[hex_col, year_col], how="left")
            r_features = r_features.merge(g_num, on=[hex_col, year_col], how="left")

        # Categorical modes among same-year neighbors
        for cat_col in cat_set:
            if cat_col not in df.columns or valid_keys.empty:
                continue
            nbr_cat = df[[hex_col, year_col, cat_col]].rename(columns={hex_col: "__nbr__"})
            f_cat = frame.merge(nbr_cat, on=["__nbr__", year_col], how="left")
            mode_name = f"{prefix}_r{R}_{cat_col}_mode"
            g_cat = (
                f_cat.groupby([hex_col, year_col], dropna=False)[cat_col]
                .apply(_most_frequent_non_null)
                .reset_index(name=mode_name)
            )
            g_cat = valid_keys.merge(g_cat, on=[hex_col, year_col], how="left")
            r_features = r_features.merge(g_cat, on=[hex_col, year_col], how="left")

        # Merge this R's features into the final result (1:1)
        result = result.merge(r_features, on=[hex_col, year_col], how="left", validate="one_to_one")

    return result


def prune_features(
    df: pd.DataFrame,
    *,
    # what to keep/protect
    protect_cols: Iterable[str] | None = None,
    # cleaning flags
    drop_high_nan: bool = True,
    drop_constant: bool = True,
    drop_quasi_constant: bool = True,
    drop_duplicate_cols: bool = True,
    drop_high_corr: bool = True,
    # parameters
    nan_threshold: float = 0.5,  # drop if share of NaN > threshold
    quasi_constant_unique_ratio: float = 0.01,  # unique_count / nonNaN_count <= thr -> drop
    corr_threshold: float = 0.95,  # |corr| >= thr -> drop one
    corr_method: Literal["pearson", "spearman"] = "pearson",
    # scope
    numeric_only_for_corr: bool = True,
    # report
    return_report: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prune useless or redundant feature columns with configurable steps.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (e.g., merged_with_neighbors).
    protect_cols : Iterable[str] | None
        Columns that must NOT be dropped (e.g., ['hex_id', 'year', 'geometry']).
    drop_high_nan, drop_constant, drop_quasi_constant, drop_duplicate_cols, drop_high_corr : bool
        Enable/disable specific cleaning steps.
    nan_threshold : float
        Drop columns with NaN share > threshold (0..1).
    quasi_constant_unique_ratio : float
        Drop columns where (#unique_nonNaN / #nonNaN) <= threshold.
    corr_threshold : float
        Threshold for absolute correlation to consider columns redundant.
    corr_method : {'pearson','spearman'}
        Correlation method for redundancy pruning.
    numeric_only_for_corr : bool
        If True, compute correlation only among numeric columns (recommended).
    return_report : bool
        If True, return (clean_df, report_df). Report lists dropped columns and reasons.

    Returns
    -------
    (clean_df, report_df)
        clean_df : cleaned dataframe (original order except removed columns),
        report_df: table with columns ['column','reason','detail'].

    Notes
    -----
    • Correlation pruning uses a greedy heuristic:
      - for a highly correlated pair (A,B), drop the one with higher NaN share;
        if tied, drop the one with lower non-NaN count; then longer name; then B.
    • Duplicate detection is exact (after filling NaN with a sentinel), so
      very close-but-not-identical floats nie będą złączone.
    """
    protect: set[str] = set(protect_cols or [])
    report_rows: list[dict[str, str]] = []

    # Work on a copy of columns list to preserve order decisions
    cols = list(df.columns)

    # Step 0: precompute simple stats
    na_share = df.isna().mean()
    non_na_count = df.shape[0] - df.isna().sum()

    # Step 1: high NaN
    to_drop: set[str] = set()
    if drop_high_nan:
        for c in cols:
            if c in protect:
                continue
            if na_share.get(c, 0.0) > nan_threshold:
                to_drop.add(c)
                report_rows.append(
                    {"column": c, "reason": "high_nan", "detail": f"nan_share={na_share[c]:.3f}"}
                )

    # Step 2: constant & quasi-constant (evaluate on non-NaN values)
    if drop_constant or drop_quasi_constant:
        for c in cols:
            if c in protect or c in to_drop:
                continue
            s = df[c].dropna()
            if s.empty:
                # all NaN: if not protected, drop (unless high_nan step already did it)
                if c not in to_drop and drop_high_nan is False:
                    to_drop.add(c)
                    report_rows.append({"column": c, "reason": "all_nan", "detail": ""})
                continue
            nunique = s.nunique(dropna=True)
            if drop_constant and nunique <= 1:
                to_drop.add(c)
                report_rows.append(
                    {"column": c, "reason": "constant", "detail": f"unique={nunique}"}
                )
                continue
            if drop_quasi_constant:
                ratio = nunique / float(len(s))
                if ratio <= quasi_constant_unique_ratio:
                    to_drop.add(c)
                    report_rows.append(
                        {
                            "column": c,
                            "reason": "quasi_constant",
                            "detail": f"unique_ratio={ratio:.5f}",
                        }
                    )

    # Step 3: duplicate columns (exact)
    if drop_duplicate_cols:
        # Fill NaN with a sentinel that won't collide with real values
        sentinel = object()
        # Use tuples of values for hashing (fast but memory-aware)
        seen: dict[tuple, str] = {}
        for c in cols:
            if c in protect or c in to_drop:
                continue
            # Build a tuple that treats NaN as distinct sentinel
            v = tuple(sentinel if pd.isna(x) else x for x in df[c].tolist())
            if v in seen:
                to_drop.add(c)
                report_rows.append(
                    {"column": c, "reason": "duplicate_col", "detail": f"duplicate_of={seen[v]}"}
                )
            else:
                seen[v] = c

    # Step 4: high correlations (numeric scope)
    if drop_high_corr:
        if numeric_only_for_corr:
            num_cols = [
                c
                for c in cols
                if c not in to_drop and c not in protect and pd.api.types.is_numeric_dtype(df[c])
            ]
        else:
            # try coercion: only keep columns convertible to numeric
            num_cols = []
            for c in cols:
                if c in to_drop or c in protect:
                    continue
                if pd.api.types.is_numeric_dtype(df[c]):
                    num_cols.append(c)
                else:
                    # attempt coercion test on a sample
                    try:
                        pd.to_numeric(df[c].dropna().head(50))
                        num_cols.append(c)
                    except Exception:
                        pass

        if len(num_cols) >= 2:
            # Compute correlation with pairwise complete obs
            corr = df[num_cols].corr(method=corr_method, min_periods=1)
            # Greedy selection of columns to drop
            # Evaluate pairs on upper triangle
            tri_pairs: list[tuple[str, str, float]] = []
            for i, a in enumerate(num_cols):
                for b in num_cols[i + 1 :]:
                    val = corr.loc[a, b]
                    if pd.notna(val) and abs(val) >= corr_threshold:
                        tri_pairs.append((a, b, float(val)))

            # Sort by absolute correlation descending to prune the strongest first
            tri_pairs.sort(key=lambda t: abs(t[2]), reverse=True)

            # Helper to decide which to drop
            def _prefer_keep(x: str, y: str) -> str:
                """Return the column to KEEP among (x,y) based on heuristics."""
                # 1) lower NaN share -> keep
                x_na, y_na = na_share.get(x, 0.0), na_share.get(y, 0.0)
                if x_na != y_na:
                    return x if x_na < y_na else y
                # 2) higher non-NaN count -> keep
                x_n, y_n = int(non_na_count.get(x, 0)), int(non_na_count.get(y, 0))
                if x_n != y_n:
                    return x if x_n > y_n else y
                # 3) shorter name -> keep (pure tie-breaker)
                if len(x) != len(y):
                    return x if len(x) < len(y) else y
                # 4) lexicographic
                return x if x < y else y

            dropped_corr: set[str] = set()
            for a, b, v in tri_pairs:
                if a in to_drop or b in to_drop or a in dropped_corr or b in dropped_corr:
                    continue
                if a in protect and b in protect:
                    continue
                # Decide which one to drop (the opposite of "prefer keep")
                keep = _prefer_keep(a, b)
                drop = b if keep == a else a
                if drop in protect:
                    drop = keep  # if chosen drop is protected, drop the other
                    if drop in protect:
                        continue  # both protected -> skip
                to_drop.add(drop)
                dropped_corr.add(drop)
                report_rows.append(
                    {
                        "column": drop,
                        "reason": "high_corr",
                        "detail": f"with={a if drop == b else b}; corr={v:.3f}; keep={keep}",
                    }
                )

    # Build output
    kept_cols = [c for c in cols if c not in to_drop]
    clean = df[kept_cols].copy()

    report_df = pd.DataFrame(report_rows).sort_values(["reason", "column"]).reset_index(drop=True)
    if return_report:
        return clean, report_df
    else:
        # still return a minimal report to comply with type hints
        return clean, report_df


def debug_print_key_duplicates(
    clean_df: pd.DataFrame,
    y_labels: pd.DataFrame,
    *,
    hex_col: str = "hex_id",
    year_col: str = "year",
    sample: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Print and return duplicate-key rows for (year, hex_id) in both left (clean_df) and right (y_labels).

    Parameters
    ----------
    clean_df : pd.DataFrame
        Left dataframe (before merge).
    y_labels : pd.DataFrame
        Right dataframe (source of labels).
    hex_col : str, default 'hex_id'
        Name of the hex id column.
    year_col : str, default 'year'
        Name of the year column.
    sample : int, default 20
        How many sample rows to print from each side.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (left_dups_df, right_dups_df) with only duplicated-key rows (keep=False).
    """
    # Normalize dtypes to avoid "phantom" mismatches
    # (e.g., '2024' as str vs 2024 as int; hex with stray spaces)
    clean_df = clean_df.copy()
    y_labels = y_labels.copy()

    clean_df[year_col] = pd.to_numeric(clean_df[year_col], errors="coerce").astype("Int64")
    y_labels[year_col] = pd.to_numeric(y_labels[year_col], errors="coerce").astype("Int64")

    clean_df[hex_col] = clean_df[hex_col].astype("string").str.strip()
    y_labels[hex_col] = y_labels[hex_col].astype("string").str.strip()

    # Detect duplicates on keys
    left_mask = clean_df.duplicated(subset=[hex_col, year_col], keep=False)
    right_mask = y_labels.duplicated(subset=[hex_col, year_col], keep=False)

    left_dups = clean_df.loc[left_mask].sort_values([year_col, hex_col])
    right_dups = y_labels.loc[right_mask].sort_values([year_col, hex_col])

    # Print quick summary
    logger.info("Left (clean_df) rows: {}", len(clean_df))
    logger.info("Right (y_labels) rows: {}", len(y_labels))

    logger.info(
        "Left duplicate key groups: {}", (left_dups.groupby([hex_col, year_col]).size() > 1).sum()
    )
    logger.info(
        "Right duplicate key groups: {}", (right_dups.groupby([hex_col, year_col]).size() > 1).sum()
    )

    # Show top offenders by group size on each side
    if not left_dups.empty:
        logger.warning("Sample LEFT duplicate groups (size desc):")
        left_grp = (
            left_dups.groupby([hex_col, year_col]).size().sort_values(ascending=False).head(10)
        )
        for (hx, yr), cnt in left_grp.items():
            logger.warning(" LEFT key=(hex_id={}, year={}) -> {} rows", hx, yr, cnt)

        logger.warning(
            "Sample LEFT duplicate rows (first {}):\n{}",
            min(sample, len(left_dups)),
            left_dups.head(sample).to_string(index=False),
        )

    if not right_dups.empty:
        logger.warning("Sample RIGHT duplicate groups (size desc):")
        right_grp = (
            right_dups.groupby([hex_col, year_col]).size().sort_values(ascending=False).head(10)
        )
        for (hx, yr), cnt in right_grp.items():
            logger.warning(" RIGHT key=(hex_id={}, year={}) -> {} rows", hx, yr, cnt)

        logger.warning(
            "Sample RIGHT duplicate rows (first {}):\n{}",
            min(sample, len(right_dups)),
            right_dups.head(sample).to_string(index=False),
        )

    if left_dups.empty and right_dups.empty:
        logger.success("No duplicate (year, hex_id) keys found on either side.")

    return left_dups, right_dups


def run_compute_neighbor_aggregates(cfg) -> None:
    """
    Execute the neighbor-aggregation pipeline driven by the Hydra/OmegaConf config.

    Expected config keys (all under `dataset.calculate_neighborhood` unless stated):
    -------------------------------------------------------------------------------
    data.duckdb_path : str
        Path to DuckDB database.
    tables_to_split : list[str]
        Source tables to load and merge (may include ${...} interpolations).
    table_with_labels : str
        Labels table to LEFT-join into each source table.
    label_cols : list[str]
        Explicit label columns to bring from the labels table.
    keys : list[str]
        Join keys for merges, typically ['hex_id','year'].

    hex_id_col : str, default 'hex_id'
    year_col   : str, default 'year'
    categorical_cols : list[str], default []
        Non-numeric columns for which neighbor mode is computed.
    geom_col   : str, default 'geometry'
    R_values   : list[int] | int, default [1]
    rolling    : bool, default True
    min_neighbors : int, default 1
    prefix     : str, default 'nbr'

    Notes
    -----
    - Interpolations inside `tables_to_split` are resolved via `OmegaConf.to_container(..., resolve=True)`.
    - This function only orchestrates the steps; it does not alter business logic.
    """
    # ─── 0) Feature toggle ──────────────────────────────────────────────────────
    if not _sel(cfg, "dataset.calculate_neighborhood.enabled", False):
        logger.info("Neighbor aggregation step skipped (disabled).")
        return

    logger.info("Starting neighbor aggregation & labels preparation...")

    # ─── 1) Resolve I/O + lists with potential ${...} ──────────────────────────
    db_path = Path(cfg.data.duckdb_path).expanduser()

    # Ensure list interpolation is materialized (e.g., ${dataset.resolution} -> r8)
    tables_to_split = OmegaConf.to_container(
        _sel(cfg, "dataset.calculate_neighborhood.tables_to_split"), resolve=True
    )

    table_with_labels = _sel(cfg, "dataset.calculate_neighborhood.table_with_labels", default="")

    proxy_cols = _sel(cfg, "dataset.calculate_neighborhood.proxy_cols", default=[])

    keys = _sel(cfg, "dataset.calculate_neighborhood.keys", default=[])

    logger.info(
        f"Merging tables ({len(tables_to_split)}): {tables_to_split} | labels: {table_with_labels} | label_cols: {proxy_cols}"
    )

    # ─── 2) Load & merge source tables with labels ─────────────────────────────
    merged_dict = load_tables_with_labels_2180(
        db_path=db_path,
        tables_to_split=tables_to_split,
        table_with_labels=table_with_labels,
        label_cols=proxy_cols,
    )

    logger.debug(f"Loaded {len(merged_dict)} tables into memory.")

    merged_df = merge_neighborhood_dict_sequential(
        merged_dict,
        keys=tuple(keys),
        geom_col="geometry",
        base=None,
        enforce_one_to_one=True,
        keep_source_prefix=False,
    )
    logger.info(f"Merged frame: rows={len(merged_df)}, cols={len(merged_df.columns)}")

    # ─── 3) Read neighbor-aggregation params ───────────────────────────────────
    hex_col = _sel(cfg, "dataset.calculate_neighborhood.hex_id_col", "hex_id")
    year_col = _sel(cfg, "dataset.calculate_neighborhood.year_col", "year")
    categorical_cols = _sel(cfg, "dataset.calculate_neighborhood.categorical_cols", [])
    geom_col = _sel(cfg, "dataset.calculate_neighborhood.geom_col", "geometry")
    R_values = _sel(cfg, "dataset.calculate_neighborhood.R_values", [1])
    rolling = _sel(cfg, "dataset.calculate_neighborhood.rolling", True)
    min_neighbors = _sel(cfg, "dataset.calculate_neighborhood.min_neighbors", 1)
    prefix = _sel(cfg, "dataset.calculate_neighborhood.prefix", "nbr")

    logger.info(
        f"Computing neighbor aggregates | hex_col={hex_col}, year_col={year_col}, geom_col={geom_col}, R={R_values}, "
        f"rolling={rolling}, min_neighbors={min_neighbors}, prefix={prefix}, categorical={categorical_cols}"
    )

    # ─── 4) Compute neighbor aggregates ────────────────────────────────────────
    merged_with_neighbors = compute_neighbor_aggregates(
        merged_df,
        hex_col=hex_col,
        year_col=year_col,
        categorical_cols=tuple(categorical_cols),
        geometry_col=geom_col,
        R_values=R_values,  # any set of rings
        rolling=rolling,  # True: disks (<=R); False: exact rings
        min_neighbors=min_neighbors,
        prefix=prefix,
    )

    # A tiny diagnostic snapshot (avoid dumping full DF to logs)
    sample = merged_with_neighbors.head(3)
    logger.debug(
        f"Sample of merged_with_neighbors (head=3): cols={len(merged_with_neighbors.columns)} | {list(merged_with_neighbors.columns[:10])}"
    )
    logger.info(
        f"Neighbor aggregation finished: rows={len(merged_with_neighbors)}, cols={len(merged_with_neighbors.columns)}",
    )

    # Keep an explicit print for interactive sessions (optional but harmless)
    print(sample)

    # ─── 5) Prune features ─────────────────────────────────────────────────────
    if _sel(cfg, "dataset.prune_features.enabled", False):
        logger.info("Pruning features...")

        protected_cols = OmegaConf.to_container(
            _sel(cfg, "dataset.prune_features.protected_cols", ""), resolve=True
        )

        drop_high_nan = _sel(cfg, "dataset.prune_features.drop_high_nan", True)
        drop_constant = _sel(cfg, "dataset.prune_features.drop_constant", True)
        drop_quasi_constant = _sel(cfg, "dataset.prune_features.drop_quasi_constant", True)
        drop_duplicate_cols = _sel(cfg, "dataset.prune_features.drop_duplicate_cols", True)
        drop_high_corr = _sel(cfg, "dataset.prune_features.drop_high_corr", True)
        nan_threshold = _sel(cfg, "dataset.prune_features.nan_threshold", 0.7)
        quasi_constant_unique_ratio = _sel(
            cfg, "dataset.prune_features.quasi_constant_unique_ratio", 0.01
        )
        corr_threshold = _sel(cfg, "dataset.prune_features.corr_threshold", 0.95)
        corr_method = _sel(cfg, "dataset.prune_features.corr_method", "pearson")
        numeric_only_for_corr = _sel(cfg, "dataset.prune_features.numeric_only_for_corr", True)
        return_report = _sel(cfg, "dataset.prune_features.return_report", True)

        clean_df, report = prune_features(
            merged_with_neighbors,
            protect_cols=protected_cols,
            drop_high_nan=drop_high_nan,
            drop_constant=drop_constant,
            drop_quasi_constant=drop_quasi_constant,
            drop_duplicate_cols=drop_duplicate_cols,
            drop_high_corr=drop_high_corr,
            nan_threshold=nan_threshold,  # np. wytnij kolumny z >50% NaN
            quasi_constant_unique_ratio=quasi_constant_unique_ratio,  # <=1% unikatów wśród nie-NaN
            corr_threshold=corr_threshold,  # usuń |corr| >= 0.95
            corr_method=corr_method,
            numeric_only_for_corr=numeric_only_for_corr,
            return_report=return_report,
        )

        logger.info(
            f"Pruned features: from {len(merged_with_neighbors.columns)} to {len(clean_df.columns)} columns."
        )
        logger.debug(f"Pruning report:\n{report}")
    else:
        logger.info("Feature pruning step skipped (disabled).")
        clean_df = merged_with_neighbors

    con = _connect_spatial(db_path)

    if _sel(cfg, "dataset.join_y_label.enabled", True):
        y_label_col = _sel(cfg, "dataset.join_y_label.y_label_col", "y_next")
        y_labels_df = con.execute(f"SELECT * FROM {table_with_labels}").df()
        logger.info(f"Re-joining y_label column '{y_label_col}' to cleaned dataframe...")

        # left_dups, right_dups = debug_print_key_duplicates(clean_df, y_labels, hex_col="hex_id", year_col="year")
        clean_df_with_y = clean_df.merge(
            y_labels_df[[hex_col, year_col, y_label_col]],
            on=[hex_col, year_col],
            how="left",
            validate="one_to_one",
        )
        logger.info(f"Re-joined y_label column '{y_label_col}' to cleaned dataframe.")
    else:
        logger.info("Re-joining y_label step skipped (disabled).")
        clean_df_with_y = clean_df

    out_table = _sel(cfg, "dataset.calculate_neighborhood.out_table", None)

    clean_df_with_y.to_parquet("cleaned_labels.parquet", index=False)

    if len(clean_df) == 0:
        logger.warning("Brak wierszy do zapisu. Zapis pominięty.")
    else:
        if con is not None:
            logger.info(f"Połączono z bazą DuckDB → {db_path}")
            write_geodf_to_duckdb(con, clean_df_with_y, table=out_table)
            logger.success(f"Saved cleaned labels to {out_table}")
            logger.success("STEP[prepare labels] Done")
        else:
            logger.error("Brak połączenia z bazą DuckDB. Zapis pominięty.")
