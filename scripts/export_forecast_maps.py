"""Eksport map predykcji podziału/odrolnienia do pliku .duckdb dla QGIS.

Wczytuje zapisane modele per horyzont (scripts/train_forecast.py) i aplikuje je do cech z
JEDNEGO roku bazowego (ostatni pełny rok obserwacji, conf: forecast_base_year) — dając
JEDNĄ predykcję na hex. Zapisuje jeden plik DuckDB z trzema tabelami przestrzennymi:
forecast_h1 / forecast_h2 / forecast_h3 — każda 1 wiersz na hex, z geometrią (EPSG:2180)
i `probability` = P(procesu) w roku docelowym (rok bazowy + horyzont).
Przykład: baza 2025 → forecast_h1 = P(podziału w 2026), h2 = 2027, h3 = 2028.

W QGIS: dodaj warstwę, stylizuj po `probability`. Jeśli CRS pokaże się jako nieznany —
przypisz EPSG:2180 (DuckDB nie utrwala SRID w kolumnie GEOMETRY).

Uruchomienie:
    python scripts/export_forecast_maps.py --config conf/forecast.yaml [--task split]

Czysty duckdb + lightgbm + joblib — bez osgeo, bez obejścia GDAL.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import duckdb
import joblib
import pandas as pd
import yaml


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def prep_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ten sam preprocessing co przy treningu: bool→int, string→num-lub-kategoria."""
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == bool:
            out[col] = out[col].astype("Int64")
        elif out[col].dtype == object or str(out[col].dtype) == "string":
            num = pd.to_numeric(out[col], errors="coerce")
            if num.notna().mean() >= out[col].notna().mean() - 0.01:
                out[col] = num
            else:
                out[col] = out[col].astype("category")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Eksport map predykcji do .duckdb dla QGIS.")
    ap.add_argument("--config", default="conf/forecast.yaml")
    ap.add_argument("--task", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    task = args.task or cfg["task"]
    crs = cfg["output"]["target_crs"]

    src = duckdb.connect(
        os.environ.get("TERRAMERGE_DUCKDB_PATH", "artifacts/duckdb/terramerge.duckdb"),
        read_only=True,
    )
    src.execute("LOAD spatial;")
    # cechy + hex_id + rok + geometria (WKB) z datasetu
    df = src.execute(f"SELECT *, ST_AsWKB(geometry) AS __wkb FROM {cfg['dataset_table']}").df()
    src.close()
    if "y_next" in df.columns:
        df = df.drop(columns=["y_next"])

    # Rok bazowy prognozy: JEDNA predykcja na hex z ostatniego pełnego roku obserwacji.
    base_year = cfg.get("forecast_base_year") or int(df["year"].max())
    base_year = int(base_year)
    if base_year not in set(df["year"].unique()):
        raise ValueError(
            f"forecast_base_year={base_year} nie istnieje w danych ({sorted(df['year'].unique())})"
        )
    base_mask = df["year"] == base_year

    keys = df.loc[base_mask, ["hex_id", "__wkb"]].copy()
    feats = prep_features(df.drop(columns=["hex_id", "geometry", "__wkb"], errors="ignore"))
    feats_base = feats.loc[base_mask]
    print(f"rok bazowy prognozy: {base_year} | heksów: {len(keys)}")

    out_path = Path(cfg["output"]["maps_duckdb"])
    out_path = out_path.with_stem(f"{out_path.stem}_{task}")  # osobny plik per zadanie
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    out = duckdb.connect(str(out_path))
    out.execute("INSTALL spatial; LOAD spatial;")

    models_dir = Path(cfg["output"]["models_dir"])
    written = []
    for h in cfg["horizons"]:
        bundle = joblib.load(models_dir / f"{task}_h{h}.joblib")
        model, features = bundle["model"], bundle["features"]
        cal_model = bundle.get("calibrated_model")
        missing = [c for c in features if c not in feats_base.columns]
        if missing:
            raise KeyError(f"Brak cech dla modelu h{h}: {missing[:5]}...")
        proba = model.predict_proba(feats_base[features])[:, 1]

        pred = keys.copy()
        pred["base_year"] = base_year
        pred["target_year"] = base_year + h
        pred["horizon"] = h
        pred["task"] = task
        pred["probability"] = proba.round(4)
        # prob_calibrated: realne prawdopodobieństwo z modelu skalibrowanego (CalibratedClassifierCV).
        pred["prob_calibrated"] = (
            cal_model.predict_proba(feats_base[features])[:, 1].round(4)
            if cal_model is not None
            else pred["probability"]
        )
        # risk_percentile: pozycja w rankingu ryzyka 0–100 (do stylizacji/komunikacji).
        pred["risk_percentile"] = (pred["probability"].rank(pct=True) * 100).round(1)

        out.register("pred_df", pred)
        table = f"forecast_h{h}"
        out.execute(f'DROP TABLE IF EXISTS "{table}";')
        out.execute(f"""
            CREATE TABLE "{table}" AS
            SELECT hex_id, base_year, target_year, horizon, task,
                   probability, prob_calibrated, risk_percentile,
                   ST_SetCRS(ST_GeomFromWKB(__wkb), '{crs}') AS geometry
            FROM pred_df;
        """)
        out.unregister("pred_df")
        n = out.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0]
        written.append((table, n))
        print(
            f"  {table}: {n} heksów | prognoza {task} na {base_year + h} (z obserwacji {base_year})"
        )

    out.close()
    print(f"\nzapisano {out_path}")
    print("tabele:", ", ".join(t for t, _ in written))

    # GeoPackage (1 plik, 3 warstwy) — uniwersalny w QGIS, z osadzonym CRS. QGIS czyta .duckdb
    # tylko gdy GDAL ma sterownik DuckDB (rzadki); GPKG działa wszędzie. Best-effort: wymaga GDAL.
    gpkg = _export_gpkg(out_path, cfg["horizons"], crs)
    if gpkg:
        print(f"GeoPackage (zalecany do QGIS): {gpkg} — warstwy forecast_h1/h2/h3, CRS {crs}")
    else:
        print(
            "GeoPackage pominięty (brak GDAL/geopandas). W QGIS użyj .duckdb tylko przy sterowniku DuckDB."
        )
    print(f"QGIS: stylizuj po `probability`, filtruj po `base_year`; jeśli CRS nieznany → {crs}.")


def _export_gpkg(duckdb_path: Path, horizons, crs: str) -> Path | None:
    """Zapisuje tabele forecast_h* z pliku .duckdb do jednego GeoPackage (3 warstwy)."""
    try:
        import geopandas as gpd
        from shapely import from_wkb
    except ImportError:
        return None
    try:
        gpkg = duckdb_path.with_suffix(".gpkg")
        if gpkg.exists():
            gpkg.unlink()
        con = duckdb.connect(str(duckdb_path), read_only=True)
        con.execute("LOAD spatial;")
        for h in horizons:
            d = con.execute(
                f"SELECT hex_id, base_year, target_year, horizon, task, "
                f"probability, prob_calibrated, risk_percentile, "
                f'ST_AsWKB(geometry) AS wkb FROM "forecast_h{h}"'
            ).df()
            geom = from_wkb(d["wkb"].map(lambda b: bytes(b) if b is not None else None))
            gdf = gpd.GeoDataFrame(d.drop(columns=["wkb"]), geometry=geom, crs=crs)
            gdf.to_file(gpkg, layer=f"forecast_h{h}", driver="GPKG")
        con.close()
        return gpkg
    except Exception as exc:  # noqa: BLE001 — best-effort, raportujemy i pomijamy
        print(f"  (GPKG błąd: {exc})")
        return None


if __name__ == "__main__":
    main()
