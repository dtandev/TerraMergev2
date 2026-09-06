"""Trenuje i stroi dwa modele (LightGBM, LogReg) dla dwóch zadań (podział, odrolnienie)
na wspólnym feature-matrix, z etykietami z dwóch źródeł. Zapisuje wyniki do JSON."""

import json
import os
import warnings
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Pure duckdb + lightgbm + sklearn — nie importuje osgeo, więc NIE wymaga obejścia GDAL/x265.
warnings.filterwarnings("ignore")
RNG = 42
DB = os.environ.get("TERRAMERGE_DUCKDB_PATH", "artifacts/duckdb/terramerge.duckdb")
OUT = os.environ.get("TERRAMERGE_RESULTS", "artifacts/reports/model_results.json")
Path(OUT).parent.mkdir(parents=True, exist_ok=True)

con = duckdb.connect(DB, read_only=True)
df = con.execute('SELECT * EXCLUDE (geometry) FROM dataset."Parcels_neighborhood_r8"').df()
odrol = con.execute(
    "SELECT hex_id, year, CAST(y_next AS INT) AS y_next_odrol, CAST(convert_proxy AS INT) AS convert_proxy "
    'FROM labels."kugLabels_r8_uzg_R_share"'
).df()
df = df.merge(odrol, on=["hex_id", "year"], how="left")

# --- typy: bool->int, string->num-lub-kategoria ---
cat_cols = []
for c in df.columns:
    if c in ("hex_id",):
        continue
    if df[c].dtype == bool:
        df[c] = df[c].astype("Int64")
    elif df[c].dtype == object or str(df[c].dtype) == "string":
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() >= df[c].notna().mean() - 0.01:
            df[c] = s
        else:
            df[c] = df[c].astype("category")
            cat_cols.append(c)

TASKS = {
    "podział (split)": {"label": "y_next", "drop": ["y_next_odrol"]},
    "odrolnienie (R→B)": {"label": "y_next_odrol", "drop": ["y_next"]},
}
DROP_ALWAYS = ["hex_id"]


def prep(task):
    label = task["label"]
    d = df[df[label].notna()].copy()
    y = d[label].astype(int)
    X = d.drop(columns=[c for c in DROP_ALWAYS + [label] + task["drop"] if c in d.columns])
    cats = [c for c in cat_cols if c in X.columns]
    return X, y, cats


def split_years(X, y):
    yr = X["year"]
    m_tr, m_va, m_ho = yr <= 2022, yr == 2023, yr == 2024

    def drop_year(z):
        return z.drop(columns=["year"])

    return (drop_year(X[m_tr]), y[m_tr], drop_year(X[m_va]), y[m_va], drop_year(X[m_ho]), y[m_ho])


def metrics(y, p):
    return {
        "AP": float(average_precision_score(y, p)),
        "ROC_AUC": float(roc_auc_score(y, p)),
        "Brier": float(brier_score_loss(y, p)),
        "lift@10%": lift_at_k(y, p, 0.10),
    }


def lift_at_k(y, p, k):
    n = max(1, int(len(p) * k))
    idx = np.argsort(p)[::-1][:n]
    base = y.mean()
    return float(y.iloc[idx].mean() / base) if base > 0 else float("nan")


# ---------- LightGBM z coordinate-descent ----------
def lgb_fit(params, Xtr, ytr, Xva, yva, cats, Xeval=None):
    m = lgb.LGBMClassifier(**params)
    m.fit(
        Xtr,
        ytr,
        eval_set=[(Xva, yva)],
        eval_metric="auc",
        categorical_feature=cats,
        callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
    )
    Xe = Xva if Xeval is None else Xeval
    p = m.predict_proba(Xe, num_iteration=m.best_iteration_)[:, 1]
    return m, p


def tune_lgb(Xtr, ytr, Xva, yva, cats):
    spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    base = dict(
        objective="binary",
        n_estimators=3000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        scale_pos_weight=spw,
        random_state=RNG,
        n_jobs=-1,
        verbosity=-1,
    )
    best = dict(base)
    _, p = lgb_fit(best, Xtr, ytr, Xva, yva, cats)
    best_ap = average_precision_score(yva, p)
    grid = [
        ("num_leaves", [15, 31, 63, 127]),
        ("max_depth", [-1, 4, 6, 8]),
        ("min_child_samples", [20, 50, 100, 200]),
        ("colsample_bytree", [0.4, 0.6, 0.8, 1.0]),
        ("subsample", [0.6, 0.8, 1.0]),
        ("reg_lambda", [0.0, 1.0, 5.0, 20.0]),
    ]
    trace = []
    for name, vals in grid:
        scores = {}
        for v in vals:
            p_ = dict(best)
            p_[name] = v
            _, pr = lgb_fit(p_, Xtr, ytr, Xva, yva, cats)
            scores[v] = float(average_precision_score(yva, pr))
        bv = max(scores, key=scores.get)
        if scores[bv] >= best_ap:
            best[name] = bv
            best_ap = scores[bv]
        trace.append({"param": name, "scores": scores, "chosen": best[name]})
    return best, best_ap, trace


# ---------- LogReg z przeszukiwaniem C ----------
def tune_logreg(Xtr, ytr, Xva, yva):
    num = Xtr.select_dtypes(include=[np.number]).columns.tolist()
    best_c, best_ap, scores = None, -1, {}
    for C in [0.01, 0.1, 1.0, 10.0]:
        pipe = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=RNG),
        )
        pipe.fit(Xtr[num], ytr)
        p = pipe.predict_proba(Xva[num])[:, 1]
        a = float(average_precision_score(yva, p))
        scores[C] = a
        if a > best_ap:
            best_ap, best_c = a, C
    pipe = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=best_c, class_weight="balanced", max_iter=2000, random_state=RNG),
    )
    pipe.fit(Xtr[num], ytr)
    return pipe, best_c, scores, num


results = {}
for tname, task in TASKS.items():
    X, y, cats = prep(task)
    Xtr, ytr, Xva, yva, Xho, yho = split_years(X, y)
    prevalence = float(yva.mean())
    proxy_col = "split_proxy" if task["label"] == "y_next" else "convert_proxy"
    persist = (
        float(average_precision_score(yva, Xva[proxy_col].fillna(0))) if proxy_col in Xva else None
    )

    # LightGBM
    lgb_params, lgb_ap_va, lgb_trace = tune_lgb(Xtr, ytr, Xva, yva, cats)
    m_lgb, p_va = lgb_fit(lgb_params, Xtr, ytr, Xva, yva, cats)
    _, p_ho = lgb_fit(lgb_params, Xtr, ytr, Xva, yva, cats, Xeval=Xho)
    lgb_m_va, lgb_m_ho = metrics(yva, p_va), metrics(yho, p_ho)

    # permutation importance (LightGBM, na walidacji, scoring=AP)
    perm = permutation_importance(
        m_lgb, Xva, yva, scoring="average_precision", n_repeats=10, random_state=RNG, n_jobs=-1
    )
    imp = pd.Series(perm.importances_mean, index=Xva.columns).sort_values(ascending=False)

    def family(c):
        if c.startswith("nbr"):
            return "nbr"
        for f in ("gf_", "uzg_", "mpzp_", "tx_"):
            if c.startswith(f):
                return f.rstrip("_")
        return "inne"

    fam_imp = imp.groupby(family).sum().sort_values(ascending=False)

    # LogReg
    m_lr, lr_C, lr_scores, num_cols = tune_logreg(Xtr, ytr, Xva, yva)
    p_lr_va = m_lr.predict_proba(Xva[num_cols])[:, 1]
    p_lr_ho = m_lr.predict_proba(Xho[num_cols])[:, 1]
    lr_m_va, lr_m_ho = metrics(yva, p_lr_va), metrics(yho, p_lr_ho)

    results[tname] = {
        "n_train": int(len(ytr)),
        "n_valid": int(len(yva)),
        "n_hold": int(len(yho)),
        "pos_train": int(ytr.sum()),
        "pos_valid": int(yva.sum()),
        "pos_hold": int(yho.sum()),
        "prevalence_valid": prevalence,
        "persistence_AP_valid": persist,
        "lgb_best_params": {
            k: lgb_params[k]
            for k in [
                "num_leaves",
                "max_depth",
                "min_child_samples",
                "colsample_bytree",
                "subsample",
                "reg_lambda",
            ]
        },
        "lgb_valid": lgb_m_va,
        "lgb_hold2024": lgb_m_ho,
        "lgb_trace": lgb_trace,
        "logreg_best_C": lr_C,
        "logreg_C_scores": lr_scores,
        "logreg_valid": lr_m_va,
        "logreg_hold2024": lr_m_ho,
        "perm_top15": {k: float(v) for k, v in imp.head(15).items()},
        "perm_by_family": {k: float(v) for k, v in fam_imp.items()},
        "n_features": int(Xva.shape[1]),
    }
    print(f"\n===== {tname} =====")
    print(
        f"  n_train={len(ytr)} pos={int(ytr.sum())} | valid2023 pos={int(yva.sum())} ({prevalence:.1%})"
    )
    print(f"  BASELINE: losowy(AP)={prevalence:.3f} persistence(AP)={persist:.3f}")
    print(
        f"  LightGBM valid: AP={lgb_m_va['AP']:.3f} ROC={lgb_m_va['ROC_AUC']:.3f} Brier={lgb_m_va['Brier']:.3f} lift@10%={lgb_m_va['lift@10%']:.1f}"
    )
    print(
        f"  LightGBM 2024 : AP={lgb_m_ho['AP']:.3f} ROC={lgb_m_ho['ROC_AUC']:.3f} lift@10%={lgb_m_ho['lift@10%']:.1f}"
    )
    print(f"  LogReg   valid: AP={lr_m_va['AP']:.3f} ROC={lr_m_va['ROC_AUC']:.3f} (C={lr_C})")
    print(f"  najlepsze LGB: {results[tname]['lgb_best_params']}")
    print(f"  ważność wg rodzin cech: {results[tname]['perm_by_family']}")

with open(OUT, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nzapisano {OUT}")
