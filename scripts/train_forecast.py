"""Trening modeli prognozy przekształceń gruntów per horyzont +1/+2/+3.

Dla każdego horyzontu: stroi LightGBM (coordinate descent) na przesuwanym oknie czasowym,
liczy metryki out-of-sample, zapisuje model (joblib), prognozę per hex (CSV) oraz komplet
metryk do JSON. Raport (JSON) powstaje z KODU, nie z modelu językowego — każde wywołanie
jest w pełni udokumentowane i reprodukowalne.

Uruchomienie:
    python scripts/train_forecast.py --config conf/forecast.yaml [--task split|odrol]

Czysty duckdb + lightgbm + sklearn — nie importuje osgeo, więc bez obejścia GDAL/x265.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

warnings.filterwarnings("ignore")

FAMILIES = [
    ("nbr", "nbr (sąsiedztwo)"),
    ("gf_", "gf (geometria)"),
    ("uzg_", "uzg (użytki)"),
    ("mpzp_", "mpzp (plany)"),
    ("tx_", "tx (transakcje)"),
]


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def family_of(col: str) -> str:
    for prefix, label in FAMILIES:
        if col.startswith(prefix):
            return label
    return "stan heksa"


def prepare_frame(con: duckdb.DuckDBPyConnection, cfg: dict, task: str):
    """Ładuje cechy z datasetu i etykiety horyzontowe zadania; zwraca ramkę + listę kategorii."""
    df = con.execute(f"SELECT * EXCLUDE (geometry) FROM {cfg['dataset_table']}").df()
    if "y_next" in df.columns:
        df = df.drop(columns=["y_next"])
    label_tbl = cfg["label_tables"][task]
    lab = con.execute(
        f"SELECT hex_id, year, CAST(y_next AS INT) h1, CAST(y_next_2 AS INT) h2, "
        f"CAST(y_next_3 AS INT) h3 FROM {label_tbl}"
    ).df()
    df = df.merge(lab, on=["hex_id", "year"], how="left")

    cats: list[str] = []
    for col in df.columns:
        if col == "hex_id":
            continue
        if df[col].dtype == bool:
            df[col] = df[col].astype("Int64")
        elif df[col].dtype == object or str(df[col].dtype) == "string":
            num = pd.to_numeric(df[col], errors="coerce")
            if num.notna().mean() >= df[col].notna().mean() - 0.01:
                df[col] = num
            else:
                df[col] = df[col].astype("category")
                cats.append(col)
    return df, cats


def feature_matrix(df: pd.DataFrame, label_col: str):
    """X bez hex_id i bez WSZYSTKICH kolumn etykiet horyzontowych (anty-wyciek); y = label_col."""
    drop = ["hex_id", "h1", "h2", "h3"]
    y = df[label_col].astype(int)
    X = df.drop(columns=[c for c in drop if c in df.columns])
    return X, y


def fit_lgbm(params, Xtr, ytr, Xva, yva, cats, X_predict=None):
    model = lgb.LGBMClassifier(**params)
    model.fit(
        Xtr,
        ytr,
        eval_set=[(Xva, yva)],
        eval_metric="auc",
        categorical_feature=cats,
        callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
    )
    target = Xva if X_predict is None else X_predict
    proba = model.predict_proba(target, num_iteration=model.best_iteration_)[:, 1]
    return model, proba


def tune_lgbm(base_params, grid, Xtr, ytr, Xva, yva, cats):
    best = dict(base_params)
    _, proba = fit_lgbm(best, Xtr, ytr, Xva, yva, cats)
    best_ap = average_precision_score(yva, proba)
    for name, values in grid.items():
        scores = {}
        for value in values:
            trial = dict(best)
            trial[name] = value
            _, pr = fit_lgbm(trial, Xtr, ytr, Xva, yva, cats)
            scores[value] = float(average_precision_score(yva, pr))
        chosen = max(scores, key=scores.get)
        if scores[chosen] >= best_ap:
            best[name] = chosen
            best_ap = scores[chosen]
    return best


def precision_at_k(y, proba, k=0.10):
    n = max(1, int(len(proba) * k))
    idx = np.argsort(proba)[::-1][:n]
    hits = int(np.asarray(y)[idx].sum())
    base = float(np.mean(y))
    return {
        "top_k_n": n,
        "hits": hits,
        "precision": hits / n,
        "base_rate": base,
        "lift": (hits / n / base) if base > 0 else None,
    }


def metric_pack(y, proba):
    return {
        "n": int(len(y)),
        "positives": int(np.sum(y)),
        "AP": float(average_precision_score(y, proba)),
        "ROC_AUC": float(roc_auc_score(y, proba)),
        "Brier": float(brier_score_loss(y, proba)),
        "precision_at_10pct": precision_at_k(y, proba, 0.10),
    }


def run(cfg: dict, task: str) -> dict:
    con = duckdb.connect(
        os.environ.get("TERRAMERGE_DUCKDB_PATH", "artifacts/duckdb/terramerge.duckdb"),
        read_only=True,
    )
    df, cats = prepare_frame(con, cfg, task)
    con.close()

    models_dir = Path(cfg["output"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    base_train_max = int(cfg["eval"]["base_train_max"])
    forecast_year = int(df["year"].max())

    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "task": task,
        "dataset_table": cfg["dataset_table"],
        "label_table": cfg["label_tables"][task],
        "forecast_year": forecast_year,
        "horizons": {},
        "model_files": {},
    }
    forecast_rows = []

    for h in cfg["horizons"]:
        label_col = f"h{h}"
        train_max = base_train_max - (h - 1)
        valid_year, test_year = train_max + 1, train_max + 2

        d = df[df[label_col].notna()].copy()
        X, y = feature_matrix(d, label_col)
        cats_h = [c for c in cats if c in X.columns]
        yr = X["year"]
        drop_year = ["year"]

        m_tr, m_va, m_te = yr <= train_max, yr == valid_year, yr == test_year
        Xtr, ytr = X[m_tr].drop(columns=drop_year), y[m_tr]
        Xva, yva = X[m_va].drop(columns=drop_year), y[m_va]

        base_params = dict(cfg["lgbm"])
        base_params["scale_pos_weight"] = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
        best_params = tune_lgbm(base_params, cfg["tune_grid"], Xtr, ytr, Xva, yva, cats_h)

        eval_model, p_va = fit_lgbm(best_params, Xtr, ytr, Xva, yva, cats_h)
        metrics = {"valid": metric_pack(yva, p_va)}
        if m_te.any() and y[m_te].sum() > 0:
            _, p_te = fit_lgbm(
                best_params, Xtr, ytr, Xva, yva, cats_h, X_predict=X[m_te].drop(columns=drop_year)
            )
            metrics["test"] = metric_pack(y[m_te], p_te)

        perm = permutation_importance(
            eval_model,
            Xva,
            yva,
            scoring="average_precision",
            n_repeats=int(cfg["perm_importance"]["n_repeats"]),
            random_state=42,
            n_jobs=-1,
        )
        imp = pd.Series(perm.importances_mean, index=Xva.columns).sort_values(ascending=False)
        fam_imp = imp.groupby(family_of).sum().sort_values(ascending=False)

        # Model finalny: trening na WSZYSTKICH latach z etykietą (maks. dane), do prognozy.
        Xall, yall = X.drop(columns=drop_year), y
        n_all = len(yall)
        val_frac = max(1, int(n_all * 0.15))
        final_model = lgb.LGBMClassifier(**best_params)
        final_model.fit(
            Xall.iloc[:-val_frac],
            yall.iloc[:-val_frac],
            eval_set=[(Xall.iloc[-val_frac:], yall.iloc[-val_frac:])],
            eval_metric="auc",
            categorical_feature=cats_h,
            callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
        )
        model_path = models_dir / f"{task}_h{h}.joblib"
        joblib.dump(
            {"model": final_model, "features": list(Xall.columns), "categoricals": cats_h},
            model_path,
        )

        # Prognoza dla najnowszego roku cech → mapa ryzyka na +h lat.
        latest = df[df["year"] == forecast_year].copy()
        Xf, _ = feature_matrix(latest, label_col)
        Xf = Xf.drop(columns=drop_year)
        proba = final_model.predict_proba(Xf[list(Xall.columns)])[:, 1]
        for hex_id, pr in zip(latest["hex_id"].to_numpy(), proba):
            forecast_rows.append(
                {
                    "hex_id": hex_id,
                    "task": task,
                    "horizon": h,
                    "base_year": forecast_year,
                    "target_year": forecast_year + h,
                    "probability": round(float(pr), 4),
                }
            )

        report["horizons"][str(h)] = {
            "window": {"train_max": train_max, "valid": valid_year, "test": test_year},
            "n_train": int(len(ytr)),
            "pos_train": int(ytr.sum()),
            "best_params": {k: best_params[k] for k in cfg["tune_grid"]},
            "metrics": metrics,
            "feature_family_importance": {k: round(float(v), 4) for k, v in fam_imp.items()},
            "top_features": {k: round(float(v), 4) for k, v in imp.head(8).items()},
        }
        report["model_files"][str(h)] = str(model_path)
        m = metrics.get("test", metrics["valid"])
        print(
            f"[{task} h{h}] train<= {train_max} test {test_year} | "
            f"AP={m['AP']:.3f} ROC={m['ROC_AUC']:.3f} "
            f"prec@10%={m['precision_at_10pct']['precision']:.1%} "
            f"lift={m['precision_at_10pct']['lift']:.1f} → {model_path.name}"
        )

    metrics_path = Path(cfg["output"]["metrics_json"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    forecast_path = Path(cfg["output"]["forecast_csv"])
    fc = pd.DataFrame(forecast_rows)
    # jeśli plik istnieje dla innego zadania — dołącz, inaczej nadpisz to zadanie
    if forecast_path.exists():
        old = pd.read_csv(forecast_path)
        old = old[old["task"] != task]
        fc = pd.concat([old, fc], ignore_index=True)
    fc.to_csv(forecast_path, index=False)

    print(f"\nmetryki → {metrics_path}\nprognozy → {forecast_path}\nmodele → {models_dir}/")
    return report


def main():
    ap = argparse.ArgumentParser(description="Trening modeli prognozy per horyzont.")
    ap.add_argument("--config", default="conf/forecast.yaml")
    ap.add_argument("--task", default=None, help="split | odrol (nadpisuje config)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    task = args.task or cfg["task"]
    run(cfg, task)


if __name__ == "__main__":
    main()
