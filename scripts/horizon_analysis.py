"""Analiza horyzontów wieloletnich (h=1,2,3) dla obu zadań. Przesuwane okno czasowe:
h1: train<=2022, valid 2023, test 2024; h2: -1 rok; h3: -2 lata."""

import json
import os
import warnings
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score

# Pure duckdb + lightgbm + sklearn — nie importuje osgeo, więc NIE wymaga obejścia GDAL/x265.
warnings.filterwarnings("ignore")
RNG = 42
DB = os.environ.get("TERRAMERGE_DUCKDB_PATH", "artifacts/duckdb/terramerge.duckdb")
OUT = os.environ.get("TERRAMERGE_HORIZON_RESULTS", "artifacts/reports/horizon_results.json")
Path(OUT).parent.mkdir(parents=True, exist_ok=True)

con = duckdb.connect(DB, read_only=True)
df = con.execute('SELECT * EXCLUDE (geometry) FROM dataset."Parcels_neighborhood_r8"').df()
# etykiety horyzontowe z obu tabel
sp = con.execute(
    "SELECT hex_id, year, CAST(y_next AS INT) split_h1, CAST(y_next_2 AS INT) split_h2, "
    'CAST(y_next_3 AS INT) split_h3 FROM labels."ParcelLabels_r8"'
).df()
od = con.execute(
    "SELECT hex_id, year, CAST(y_next AS INT) odrol_h1, CAST(y_next_2 AS INT) odrol_h2, "
    "CAST(y_next_3 AS INT) odrol_h3, CAST(convert_proxy AS INT) convert_proxy "
    'FROM labels."kugLabels_r8_uzg_R_share"'
).df()
df = (
    df.drop(columns=["y_next"])
    .merge(sp, on=["hex_id", "year"], how="left")
    .merge(od, on=["hex_id", "year"], how="left")
)
LABELCOLS = ["split_h1", "split_h2", "split_h3", "odrol_h1", "odrol_h2", "odrol_h3"]

cats = []
for c in df.columns:
    if c == "hex_id":
        continue
    if df[c].dtype == bool:
        df[c] = df[c].astype("Int64")
    elif df[c].dtype == object or str(df[c].dtype) == "string":
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() >= df[c].notna().mean() - 0.01:
            df[c] = s
        else:
            df[c] = df[c].astype("category")
            cats.append(c)


def family(c):
    if c.startswith("nbr"):
        return "nbr (sąsiedztwo)"
    for f, lab in [
        ("gf_", "gf (geometria)"),
        ("uzg_", "uzg (użytki)"),
        ("mpzp_", "mpzp (plany)"),
        ("tx_", "tx (transakcje)"),
    ]:
        if c.startswith(f):
            return lab
    return "stan heksa"


def lgb_fit(params, Xtr, ytr, Xva, yva, cc, Xeval=None):
    m = lgb.LGBMClassifier(**params)
    m.fit(
        Xtr,
        ytr,
        eval_set=[(Xva, yva)],
        eval_metric="auc",
        categorical_feature=cc,
        callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
    )
    Xe = Xva if Xeval is None else Xeval
    return m, m.predict_proba(Xe, num_iteration=m.best_iteration_)[:, 1]


def prec_at_k(y, p, k=0.10):
    n = max(1, int(len(p) * k))
    idx = np.argsort(p)[::-1][:n]
    hit = int(np.asarray(y)[idx].sum())
    base = float(np.mean(y))
    return {
        "k_n": n,
        "hits": hit,
        "precision": hit / n,
        "base": base,
        "lift": (hit / n / base) if base > 0 else None,
    }


TASKS = {"split": "split", "odrol": "odrol"}
results = {}
for task in TASKS:
    for h in (1, 2, 3):
        label = f"{task}_h{h}"
        train_max, vy, ty = 2023 - h, 2024 - h, 2025 - h  # przesuwane okno
        d = df[df[label].notna()].copy()
        y = d[label].astype(int)
        X = d.drop(columns=[c for c in ["hex_id"] + LABELCOLS if c in d.columns])
        cc = [c for c in cats if c in X.columns]
        yr = X["year"]
        tr, va, te = yr <= train_max, yr == vy, yr == ty
        Xtr, ytr = X[tr].drop(columns="year"), y[tr]
        Xva, yva = X[va].drop(columns="year"), y[va]
        Xte, yte = X[te].drop(columns="year"), y[te]
        if yva.sum() == 0 or yte.sum() == 0:
            print(f"[skip] {label}: brak pozytywów w valid/test")
            continue
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
            scale_pos_weight=spw,
            random_state=RNG,
            n_jobs=-1,
            verbosity=-1,
        )
        best = dict(base)
        _, p = lgb_fit(best, Xtr, ytr, Xva, yva, cc)
        best_ap = average_precision_score(yva, p)
        for name, vals in [
            ("num_leaves", [15, 31, 63]),
            ("max_depth", [-1, 4, 6]),
            ("min_child_samples", [50, 100, 200]),
        ]:
            sc = {}
            for v in vals:
                pp = dict(best)
                pp[name] = v
                _, pr = lgb_fit(pp, Xtr, ytr, Xva, yva, cc)
                sc[v] = average_precision_score(yva, pr)
            bv = max(sc, key=sc.get)
            if sc[bv] >= best_ap:
                best[name] = bv
                best_ap = sc[bv]
        m, p_va = lgb_fit(best, Xtr, ytr, Xva, yva, cc)
        _, p_te = lgb_fit(best, Xtr, ytr, Xva, yva, cc, Xeval=Xte)
        perm = permutation_importance(
            m, Xva, yva, scoring="average_precision", n_repeats=8, random_state=RNG, n_jobs=-1
        )
        imp = pd.Series(perm.importances_mean, index=Xva.columns).sort_values(ascending=False)
        fam = imp.groupby(family).sum().sort_values(ascending=False)
        results[label] = {
            "task": task,
            "h": h,
            "train_max": train_max,
            "valid_year": vy,
            "test_year": ty,
            "pos_train": int(ytr.sum()),
            "pos_valid": int(yva.sum()),
            "pos_test": int(yte.sum()),
            "AP_valid": float(average_precision_score(yva, p_va)),
            "ROC_valid": float(roc_auc_score(yva, p_va)),
            "AP_test": float(average_precision_score(yte, p_te)),
            "ROC_test": float(roc_auc_score(yte, p_te)),
            "prec10_valid": prec_at_k(yva, p_va),
            "prec10_test": prec_at_k(yte, p_te),
            "fam_imp": {k: float(v) for k, v in fam.items()},
            "top6": {k: float(v) for k, v in imp.head(6).items()},
        }
        r = results[label]
        print(
            f"{label}: train<= {train_max} valid {vy} test {ty} | AP_val={r['AP_valid']:.3f} AP_test={r['AP_test']:.3f} "
            f"ROC_test={r['ROC_test']:.3f} prec@10%_test={r['prec10_test']['precision']:.1%} lift={r['prec10_test']['lift']:.1f}"
        )
        print(
            "    rodziny: "
            + ", ".join(f"{k.split()[0]}={v:.3f}" for k, v in list(r["fam_imp"].items())[:4])
        )

json.dump(results, open(OUT, "w"), indent=2, ensure_ascii=False)
print("zapisano", OUT)
