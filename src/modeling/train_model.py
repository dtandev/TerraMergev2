from __future__ import annotations

import json
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import geopandas as gpd
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.common.config_utils import sel as _sel
from src.common.duckdb_utils import connect_duckdb as _connect_spatial
from src.common.duckdb_utils import write_geoparquet

# =========================
# Logging & DB
# =========================


def setup_logging(output_dir: Path, name: str = "training") -> Path:
    """
    Initialize loguru sinks: stdout and file in output_dir.
    Returns the log file path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"{name}.log"
    # Remove previous sinks to prevent duplication in notebooks/REPL.
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level="INFO")
    logger.add(
        log_path, level="INFO", encoding="utf-8", enqueue=True, rotation="10 MB", retention=5
    )
    logger.info("Logging to {}", log_path.resolve())
    return log_path


# =========================
# IO helpers
# =========================


def _to_bytes_safe(val: object) -> bytes | None:
    """
    Convert DuckDB BLOB-like values to bytes (bytes/bytearray/memoryview/list[int]).
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


def load_dataset_2180(
    db_path: Path,
    *,
    dataset_table: str,
    geom_col: str = "geometry",
) -> gpd.GeoDataFrame:
    """
    Load dataset from DuckDB as GeoDataFrame in EPSG:2180.
    """
    logger.info("Loading dataset from table '{}'", dataset_table)
    con = _connect_spatial(db_path)
    q = f"""
        SELECT
            de.*,
            ST_AsWKB(de.{geom_col}) AS __geom_wkb__
        FROM {dataset_table} AS de
    """
    df: pd.DataFrame = con.execute(q).fetchdf()
    con.close()

    geos = gpd.GeoSeries.from_wkb(df["__geom_wkb__"].map(_to_bytes_safe), crs="EPSG:2180")
    drop_cols = [c for c in ("__geom_wkb__", geom_col) if c in df.columns]
    gdf = gpd.GeoDataFrame(
        df.drop(columns=drop_cols, errors="ignore"), geometry=geos, crs="EPSG:2180"
    )
    logger.success("Loaded {} rows with geometry (EPSG:2180)", len(gdf))
    return gdf


def save_config_snapshot(
    output_dir: Path, cfg: DictConfig, filename: str = "config_snapshot.yaml"
) -> Path:
    """
    Save a resolved YAML snapshot of the cfg into output_dir/filename.
    Does NOT print the YAML to logs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    yaml_text = OmegaConf.to_yaml(cfg, resolve=True)
    path.write_text(yaml_text, encoding="utf-8")
    logger.success("Saved config snapshot → {}", path.resolve())
    return path


# =========================
# Feature prep (train & inference)
# =========================


def prepare_Xy_for_lgb(
    gdf: pd.DataFrame,
    *,
    label_col: str = "y_next",
    drop_always: tuple = ("hex_id", "geometry", "input_summary"),
    prefer_cats: tuple = ("jednostka_mode", "obreb_mode", "powiat", "dominant_class"),
    cat_cardinality_max: int = 300,
) -> tuple[pd.DataFrame, pd.Series, list[str], dict]:
    """
    Prepare features and label for LightGBM. Returns X, y, categorical_cols, diagnostics.
    """
    df = gdf.copy()
    if label_col not in df.columns:
        raise KeyError(f"Missing label_col '{label_col}' in training data.")

    y = df[label_col].astype(int) if df[label_col].dtype == "boolean" else df[label_col].astype(int)

    # 1) drop hard non-features
    cols_drop = [c for c in drop_always if c in df.columns]
    df = df.drop(columns=cols_drop, errors="ignore")

    # 2) remove label from X
    df = df.drop(columns=[label_col], errors="ignore")

    # 3) text-to-numeric when safe
    obj_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    converted_to_num, still_object = [], []
    for c in obj_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if (s.isna().mean() - df[c].isna().mean()) <= 0.01:
            df[c] = s
            converted_to_num.append(c)
        else:
            still_object.append(c)

    # 4) true categoricals with cardinality control
    cat_cols, dropped_high_card = [], []
    prefer_cats_present = [c for c in prefer_cats if c in still_object]
    for c in still_object:
        nunique = df[c].nunique(dropna=True)
        if (c in prefer_cats_present) or (1 < nunique <= cat_cardinality_max):
            df[c] = df[c].astype("category")
            cat_cols.append(c)
        else:
            df = df.drop(columns=[c])
            dropped_high_card.append(c)

    # 5) normalize pandas nullable ints
    for c in df.columns:
        if pd.api.types.is_integer_dtype(df[c].dtype) and "Int" in str(df[c].dtype):
            df[c] = df[c].astype("float64")

    diagnostics = {
        "dropped_always": cols_drop,
        "converted_to_num": converted_to_num,
        "kept_categoricals": cat_cols,
        "dropped_high_card": dropped_high_card,
        "n_features_final": df.shape[1],
    }
    return df, y, cat_cols, diagnostics


def save_feature_meta(out_dir: Path, feature_cols: list[str], categorical_cols: list[str]) -> Path:
    """
    Save feature layout (columns and categoricals) to JSON for inference parity.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"feature_columns": feature_cols, "categorical_columns": categorical_cols}
    path = out_dir / "feature_meta.json"
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.success("Saved feature meta → {}", path.resolve())
    return path


def load_feature_meta(path: Path) -> dict:
    """
    Load feature meta JSON.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def build_X_like(
    raw_df: pd.DataFrame,
    *,
    label_col: str | None,
    drop_always: Iterable[str],
    reference_columns: Iterable[str],
    categorical_cols: Iterable[str],
) -> pd.DataFrame:
    """
    Transform raw_df into a feature matrix aligned to reference_columns used in training.
    - Drops listed columns.
    - Converts provided categorical_cols to category dtype (if present).
    - Adds any missing reference columns filled with 0.0.
    - Drops any extra columns not in reference_columns.
    """
    df = raw_df.copy()
    # drop non-features + label if present
    cols_to_drop = set(drop_always)
    if label_col and label_col in df.columns:
        cols_to_drop.add(label_col)
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

    # ensure categorical dtype
    for c in categorical_cols:
        if c in df.columns and str(df[c].dtype) != "category":
            df[c] = df[c].astype("category")

    # numeric coercion for obvious numeric strings
    obj_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    for c in obj_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if (s.isna().mean() - df[c].isna().mean()) <= 0.01:
            df[c] = s

    # align to training feature set
    ref = list(reference_columns)
    missing = [c for c in ref if c not in df.columns]
    extra = [c for c in df.columns if c not in ref]

    if missing:
        logger.warning("Adding {} missing feature(s) with 0.0: {}", len(missing), missing[:10])
        for c in missing:
            df[c] = 0.0
    if extra:
        logger.warning("Dropping {} unseen feature(s): {}", len(extra), extra[:10])
        df = df.drop(columns=extra)

    # final column order
    df = df[ref]
    return df


# =========================
# Train / Eval
# =========================


def train_lgbm_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    categorical_cols: Iterable[str] | None,
    model_out_path: Path,
    n_estimators: int = 2000,
    learning_rate: float = 0.02,
    max_depth: int = -1,
    random_state: int = 42,
    early_stopping_rounds: int = 100,
    log_eval_period: int = 50,
) -> tuple[lgb.LGBMClassifier, pd.Series]:
    """
    Train LightGBM binary classifier with early stopping. Save model to model_out_path.
    Returns model and top-30 gain importances.
    """
    model_out_path.parent.mkdir(parents=True, exist_ok=True)

    # scale_pos_weight
    class_counts = y_train.value_counts()
    if (0 not in class_counts) or (1 not in class_counts) or (class_counts[1] == 0):
        raise ValueError("Both classes 0 and 1 must be present in y_train for scale_pos_weight.")
    pos_weight = float(class_counts[0] / class_counts[1])
    logger.info(
        "Class counts → 0: {} | 1: {} | scale_pos_weight={:.4f}",
        int(class_counts[0]),
        int(class_counts[1]),
        pos_weight,
    )

    cat_cols = [c for c in (categorical_cols or []) if c in X_train.columns]
    not_category = [c for c in cat_cols if str(X_train[c].dtype) != "category"]
    if not_category:
        logger.warning(
            "Categoricals not dtype 'category': {}. Consider casting for stability.",
            not_category[:10],
        )

    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        n_jobs=-1,
        scale_pos_weight=pos_weight,
        random_state=random_state,
    )

    callbacks = [
        lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
        lgb.log_evaluation(period=log_eval_period),
    ]
    logger.info(
        "Training LightGBM… n_estimators={}, lr={}, max_depth={}, esr={}",
        n_estimators,
        learning_rate,
        max_depth,
        early_stopping_rounds,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="aucpr",
        categorical_feature=cat_cols,
        callbacks=callbacks,
    )

    best_iter = getattr(model, "best_iteration_", None)
    if best_iter:
        logger.success("Finished. Best iteration: {}", best_iter)
    else:
        logger.success("Finished. No early-stopping best_iteration_ exposed.")

    gain = model.booster_.feature_importance(importance_type="gain")
    imp_gain_top30 = pd.Series(gain, index=X_train.columns).sort_values(ascending=False).head(30)
    logger.info("Top-30 features by GAIN:\n{}", imp_gain_top30.to_string(max_rows=30))

    joblib.dump(model, model_out_path)
    logger.success("Model saved → {}", model_out_path.resolve())

    return model, imp_gain_top30


def fdr_bh(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    Benjamini–Hochberg FDR control. Returns boolean mask for significant hypotheses.
    """
    p = np.asarray(pvals)
    n = p.size
    order = np.argsort(p)
    ranked = np.arange(1, n + 1)
    thresh = alpha * ranked / n
    passed = p[order] <= thresh
    if not np.any(passed):
        return np.zeros_like(p, dtype=bool)
    k_max = int(np.max(np.where(passed)[0]))
    cutoff = p[order][k_max]
    return p <= cutoff


def permutation_significance(
    model: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    scoring: str = "average_precision",
    n_repeats: int = 100,
    random_state: int = 42,
    feature_subset: Iterable[str] | None = None,
    sample_rows: int | None = None,
) -> pd.DataFrame:
    """
    Permutation importance with empirical one-sided p-values and FDR (BH).

    Parameters
    ----------
    feature_subset : iterable[str] | None
        Optional subset of columns to evaluate (speeds up).
    sample_rows : int | None
        Optional number of validation rows to subsample (without replacement).

    Returns
    -------
    DataFrame with columns: feature, importance_mean, importance_std, ci_low, ci_high,
    p_value, significant_FDR_5%
    """
    t0 = time.time()

    X_use = X_val
    y_use = y_val
    if feature_subset is not None:
        cols = [c for c in feature_subset if c in X_val.columns]
        if not cols:
            logger.warning(
                "Feature subset empty after intersection with X_val columns. Falling back to all features."
            )
        else:
            X_use = X_val[cols]
            logger.info("Permutation: using {} feature(s) (subset).", len(cols))
    if sample_rows is not None and sample_rows < len(X_use):
        X_use = X_use.sample(n=sample_rows, random_state=random_state)
        y_use = y_val.loc[X_use.index]
        logger.info("Permutation: subsampled validation to {} rows.", len(X_use))

    pi = permutation_importance(
        model,
        X_use,
        y_use,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    names = np.array(X_use.columns)
    imps_all = pi.importances
    p_emp = (1 + np.sum(imps_all <= 0, axis=1)) / (n_repeats + 1)
    sig_mask = fdr_bh(p_emp, alpha=0.05)
    ci_low = np.percentile(imps_all, 2.5, axis=1)
    ci_high = np.percentile(imps_all, 97.5, axis=1)

    out = (
        pd.DataFrame(
            {
                "feature": names,
                "importance_mean": pi.importances_mean,
                "importance_std": pi.importances_std,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_value": p_emp,
                "significant_FDR_5%": sig_mask,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )

    logger.info(
        "Permutation significance done in {:.2f}s | features={} | repeats={} | rows={}",
        time.time() - t0,
        len(names),
        n_repeats,
        len(X_use),
    )
    return out


def evaluate_lgbm_on_validation(
    model: lgb.LGBMClassifier,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    feature_names: Iterable[str] | None = None,
    topk: Iterable[float] = (0.05, 0.10, 0.20),
    n_perm_repeats: int = 5,
    random_state: int = 42,
    perm_significance: bool = True,
    perm_n_repeats: int = 100,
    perm_compute_brier: bool = True,
    perm_max_features: int | None = 50,  # NEW: limit cech do permutacji (top-N po gain)
    perm_sample_rows: int | None = None,  # NEW: subsample walidacji
) -> dict:
    """
    Validation pack: raw metrics, calibration table, lift@k (raw & isotonic-calibrated),
    permutation importance (AP), optional permutation significance (AP/Brier), and isotonic metrics.
    """
    # RAW probs
    p_raw = model.predict_proba(X_val)[:, 1]

    # Global raw metrics
    ap = float(average_precision_score(y_val, p_raw))
    roc = float(roc_auc_score(y_val, p_raw))
    brier = float(brier_score_loss(y_val, p_raw))
    logger.info("Validation (raw) → AP={:.4f} | ROC-AUC={:.4f} | Brier={:.4f}", ap, roc, brier)

    # Calibration (raw)
    bins = pd.qcut(p_raw, q=10, duplicates="drop")
    calib = (
        pd.DataFrame({"bin": bins, "pred": p_raw, "y": y_val.values})
        .groupby("bin")
        .agg(pred_mean=("pred", "mean"), y_rate=("y", "mean"), n=("y", "size"))
        .reset_index()
    )
    logger.info("Calibration (raw, head):\n{}", calib.head().to_string(index=False))

    # Permutation importance (AP)
    def _ap_scorer(est, X, y):
        p = est.predict_proba(X)[:, 1]
        return average_precision_score(y, p)

    perm = permutation_importance(
        model,
        X_val,
        y_val,
        n_repeats=n_perm_repeats,
        scoring=_ap_scorer,
        random_state=random_state,
    )
    names = list(feature_names) if feature_names is not None else list(X_val.columns)
    perm_imp = pd.Series(perm.importances_mean, index=names).sort_values(ascending=False)
    logger.info("Permutation importance (AP, top-20):\n{}", perm_imp.head(20).to_string())

    # Isotonic calibration (fit on val → optimistic for test evaluation)
    iso = IsotonicRegression(out_of_bounds="clip")
    p_cal = iso.fit_transform(p_raw, y_val)
    ap_raw = float(average_precision_score(y_val, p_raw))
    ap_cal = float(average_precision_score(y_val, p_cal))
    brier_raw = float(brier_score_loss(y_val, p_raw))
    brier_cal = float(brier_score_loss(y_val, p_cal))
    logger.info(
        "Isotonic → AP raw={:.4f} | AP cal={:.4f} | Brier raw={:.4f} | Brier cal={:.4f}",
        ap_raw,
        ap_cal,
        brier_raw,
        brier_cal,
    )

    # Lift@k raw vs cal
    def _lift_at_k(y_true: pd.Series, y_score: np.ndarray, k: float) -> dict:
        n = len(y_true)
        k_n = max(1, int(np.floor(k * n)))
        idx = np.argsort(-y_score)[:k_n]
        capture = int(y_true.iloc[idx].sum())
        total = int(y_true.sum())
        base_rate = (total / n) if n > 0 else 0.0
        capture_rate = (capture / total) if total > 0 else 0.0
        lift = ((capture / k_n) / base_rate) if base_rate > 0 else float("nan")
        return {
            "k": float(k),
            "capture_events": capture,
            "events_total": total,
            "capture_rate": float(capture_rate),
            "lift": float(lift),
        }

    ks = list(topk)
    lift_raw = pd.DataFrame([_lift_at_k(y_val.reset_index(drop=True), p_raw, k) for k in ks])
    lift_cal = pd.DataFrame([_lift_at_k(y_val.reset_index(drop=True), p_cal, k) for k in ks])
    comp = pd.DataFrame(
        {
            "k": ks,
            "capture_rate_raw": lift_raw["capture_rate"].values,
            "capture_rate_cal": lift_cal["capture_rate"].values,
            "lift_raw": lift_raw["lift"].values,
            "lift_cal": lift_cal["lift"].values,
        }
    )
    comp["delta_capture_rate"] = comp["capture_rate_cal"] - comp["capture_rate_raw"]
    comp["delta_lift"] = comp["lift_cal"] - comp["lift_raw"]
    logger.info("Lift comparison:\n{}", comp.to_string(index=False))

    # Optional: permutation significance (AP & optionally Brier)
    signif_block = None
    if perm_significance:
        # wybór top-N po GAIN (szybkie i darmowe)
        gains = pd.Series(
            model.booster_.feature_importance(importance_type="gain"),
            index=list(feature_names) if feature_names is not None else list(X_val.columns),
        ).sort_values(ascending=False)

        feat_subset = None
        if perm_max_features is not None and perm_max_features > 0:
            feat_subset = gains.head(perm_max_features).index.tolist()
            logger.info(
                "Permutation significance: limiting to top-{} features by GAIN.", len(feat_subset)
            )

        logger.info(
            "Running permutation significance (AP) | repeats={} | max_features={} | sample_rows={}",
            perm_n_repeats,
            (len(feat_subset) if feat_subset else "all"),
            (perm_sample_rows or "all"),
        )

        sig_df_ap = permutation_significance(
            model,
            X_val,
            y_val,
            scoring="average_precision",
            n_repeats=perm_n_repeats,
            random_state=random_state,
            feature_subset=feat_subset,
            sample_rows=perm_sample_rows,
        )
        logger.info(
            "Significant @ FDR 5% (AP): {}",
            sig_df_ap.loc[sig_df_ap["significant_FDR_5%"], "feature"].tolist(),
        )

        sig_df_brier = None
        if perm_compute_brier:
            logger.info(
                "Running permutation significance (Brier) | repeats={} | max_features={} | sample_rows={}",
                perm_n_repeats,
                (len(feat_subset) if feat_subset else "all"),
                (perm_sample_rows or "all"),
            )
            sig_df_brier = permutation_significance(
                model,
                X_val,
                y_val,
                scoring="neg_brier_score",
                n_repeats=perm_n_repeats,
                random_state=random_state,
                feature_subset=feat_subset,
                sample_rows=perm_sample_rows,
            )
        signif_block = {"ap": sig_df_ap, "brier": sig_df_brier}

    return {
        "metrics": {"ap": ap, "roc": roc, "brier": brier},
        "calibration": calib,
        "perm_importance": perm_imp,
        "isotonic": {
            "ap_raw": ap_raw,
            "ap_cal": ap_cal,
            "brier_raw": brier_raw,
            "brier_cal": brier_cal,
            "p_raw": p_raw,
            "p_cal": p_cal,
            "model": iso,
        },
        "lift": {"raw": lift_raw, "cal": lift_cal, "comparison": comp},
        "significance": signif_block,
    }


# =========================
# Orchestration (train + inference 2025)
# =========================


def temporal_holdout_split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    year_col: str = "year",
    train_max_year: int = 2022,
    valid_years: tuple[int, ...] = (2023, 2024),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split by explicit years: train on <= train_max_year, validate on valid_years.
    """
    assert year_col in X.columns, f"Missing column '{year_col}' in X."
    train_mask = X[year_col] <= train_max_year
    valid_mask = X[year_col].isin(valid_years)
    assert not (train_mask & valid_mask).any(), "Year overlap between train and valid."

    X_train = X.loc[train_mask].drop(columns=[year_col])
    y_train = y.loc[train_mask]
    X_valid = X.loc[valid_mask].drop(columns=[year_col])
    y_valid = y.loc[valid_mask]
    return X_train, X_valid, y_train, y_valid


def run_training(cfg: DictConfig) -> None:
    """
    Orchestrate: load data, prepare features, train, evaluate, save artifacts,
    run inference on year=2025, and export predictions to GeoParquet.
    """
    db_path = Path(cfg.data.duckdb_path).expanduser()
    dataset_table = _sel(cfg, "model.dataset_table", "dataset.Parcels")
    y_label = _sel(cfg, "model.y_label", "y_next")
    drop_cols = tuple(_sel(cfg, "model.drop_columns", ["hex_id", "geometry"]))
    prefer_cats = tuple(_sel(cfg, "model.prefer_cats", ("jednostka",)))
    cat_card_max = int(_sel(cfg, "model.cat_cardinality_max", 30))
    valid_years = tuple(_sel(cfg, "model.valid_years", [2023, 2024]))
    inference_years = tuple(_sel(cfg, "model.inference_years", [2025]))
    train_max_year = int(_sel(cfg, "model.train_max_year", 2022))

    model_dir = Path(_sel(cfg, "model.model_output_dir", "artifacts/models/0/")).expanduser()
    setup_logging(model_dir, name="training")
    save_config_snapshot(model_dir, cfg)  # zapis do config_snapshot.yaml

    # 1) Load data
    gdf = load_dataset_2180(db_path=db_path, dataset_table=dataset_table, geom_col="geometry")
    if gdf.empty:
        raise ValueError(f"Dataset table '{dataset_table}' is empty.")

    # 2) Split data for training vs. inference
    train_df = gdf[~gdf["year"].isin(inference_years)].copy()
    infer_df = gdf[gdf["year"].isin(inference_years)].copy()
    if infer_df.empty:
        logger.warning("No rows found for inference years {}. Skipping inference.", inference_years)

    # 3) Prepare training matrices
    X_all, y_all, cat_cols, diag = prepare_Xy_for_lgb(
        train_df,
        label_col=y_label,
        drop_always=drop_cols,
        prefer_cats=prefer_cats,
        cat_cardinality_max=cat_card_max,
    )
    logger.info("Prepared training matrix with {} features. Diagnostics: {}", X_all.shape[1], diag)

    # 4) Temporal split
    X_train, X_val, y_train, y_val = temporal_holdout_split(
        X_all, y_all, train_max_year=train_max_year, valid_years=valid_years
    )
    # feature meta includes year dropped from matrices; keep column layout from X_train
    feature_meta_path = save_feature_meta(model_dir, list(X_train.columns), list(cat_cols))

    # 5) Train
    model_path = model_dir / "lgbm_model.joblib"
    model, imp_gain_top30 = train_lgbm_classifier(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        categorical_cols=cat_cols,
        model_out_path=model_path,
        n_estimators=int(_sel(cfg, "model.n_estimators", 500)),
        learning_rate=float(_sel(cfg, "model.learning_rate", 0.05)),
        max_depth=int(_sel(cfg, "model.max_depth", -1)),
        random_state=int(_sel(cfg, "model.random_state", 42)),
        early_stopping_rounds=int(_sel(cfg, "model.early_stopping_rounds", 100)),
        log_eval_period=int(_sel(cfg, "model.log_eval_period", 50)),
    )

    # 6) Evaluate (logs only; artifacts możesz dodać później jeśli chcesz)
    eval_res = evaluate_lgbm_on_validation(
        model,
        X_val,
        y_val,
        feature_names=X_val.columns,
        topk=(0.05, 0.10, 0.20),
        n_perm_repeats=int(_sel(cfg, "model.n_perm_repeats", 5)),
        random_state=int(_sel(cfg, "model.random_state", 42)),
        perm_significance=bool(_sel(cfg, "model.perm_significance", True)),
        perm_n_repeats=int(_sel(cfg, "model.perm_n_repeats", 100)),
        perm_compute_brier=bool(_sel(cfg, "model.perm_compute_brier", True)),
    )
    m = eval_res["metrics"]
    logger.success(
        "Eval summary → AP={:.4f} | ROC-AUC={:.4f} | Brier={:.4f}", m["ap"], m["roc"], m["brier"]
    )

    # 7) Inference for 2025 → GeoParquet
    if not infer_df.empty:
        logger.info("Starting inference for years: {}", inference_years)
        meta = load_feature_meta(feature_meta_path)
        ref_cols = meta["feature_columns"]
        cat_cols_ref = meta["categorical_columns"]

        # build X aligned with training features (year is NOT part of ref_cols)
        X_inf = build_X_like(
            infer_df,
            label_col=y_label if y_label in infer_df.columns else None,
            drop_always=drop_cols,
            reference_columns=ref_cols,
            categorical_cols=cat_cols_ref,
        )
        # predict raw and calibrated (calibrator fitted on validation → optimistic; acceptable here for scoring export)
        p_hat = model.predict_proba(X_inf)[:, 1]
        # optional isotonic calibration using eval_res calibrator trained on val
        iso = eval_res["isotonic"]["model"]
        try:
            p_hat_cal = iso.transform(p_hat)
        except Exception:
            logger.warning("Isotonic transform failed on inference; exporting only raw p_hat.")
            p_hat_cal = None

        # assemble GeoDataFrame for export
        out_cols = ["year"]
        if "hex_id" in infer_df.columns:
            out_cols = ["hex_id"] + out_cols
        export_df = infer_df[out_cols + ["geometry"]].copy()
        export_df["p_hat"] = p_hat
        if p_hat_cal is not None:
            export_df["p_hat_cal"] = p_hat_cal

        gdf_out = gpd.GeoDataFrame(export_df, geometry="geometry", crs="EPSG:2180")

        out_gpq = model_dir / f"predictions_{'_'.join(map(str, inference_years))}.gpq"
        write_geoparquet(gdf_out, out_gpq)
        logger.success("Inference saved → {}", out_gpq.resolve())
    else:
        logger.warning("Inference skipped (no rows for {}).", inference_years)

    logger.success("All done. Artifacts in: {}", model_dir.resolve())
