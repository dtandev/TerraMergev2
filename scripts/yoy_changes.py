"""Zestawienia zmian struktury agrarnej rok-do-roku (r/r) dla powiatu i gmin.

Liczy z bazy DuckDB tabelę per rok (i per gmina) i zapisuje do CSV oraz XLSX. Dane są
źródłem dla interaktywnego dashboardu (artifacts/dashboard/dashboard.html), który czyta
plik CSV. Raport tabelaryczny powstaje z KODU — reprodukowalny, bez udziału modelu językowego.

Uruchomienie:
    python scripts/yoy_changes.py

Czysty duckdb + pandas — bez osgeo, bez obejścia GDAL.
"""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pandas as pd

DB = os.environ.get("TERRAMERGE_DUCKDB_PATH", "artifacts/duckdb/terramerge.duckdb")
OUT_DIR = Path(os.environ.get("TERRAMERGE_REPORTS", "artifacts/reports"))
DATASET = 'dataset."Parcels_neighborhood_r8"'
KUG_LABELS = 'labels."kugLabels_r8_uzg_R_share"'
PARCELS = 'egib."DzialkaEwidencyjna"'


def per_year(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = con.execute(f"""
        SELECT
            d.year AS rok,
            round(avg(d.uzg_R_share), 4)  AS udzial_R,
            round(avg(d.uzg_B_share), 4)  AS udzial_B,
            round(avg(d.uzg_L_share), 4)  AS udzial_L,
            round(avg(d.uzg_Ps_share), 4) AS udzial_Ps,
            round(avg(d.n_parcel), 1)     AS srednia_liczba_dzialek_hex,
            sum(CAST(d.split_proxy AS INT)) AS heksy_z_podzialem,
            sum(CAST(k.convert_proxy AS INT)) AS heksy_z_odrolnieniem
        FROM {DATASET} d
        LEFT JOIN {KUG_LABELS} k USING (hex_id, year)
        GROUP BY d.year ORDER BY d.year
    """).df()

    parcels = con.execute(
        f"SELECT year AS rok, count(*) AS liczba_dzialek FROM {PARCELS} GROUP BY year"
    ).df()
    df = df.merge(parcels, on="rok", how="left")

    # przyrost/ubytek udziałów r/r (punkty procentowe)
    df["delta_B_pp"] = (df["udzial_B"].diff() * 100).round(2)
    df["delta_R_pp"] = (df["udzial_R"].diff() * 100).round(2)
    df["przyrost_dzialek"] = df["liczba_dzialek"].diff()
    cols = [
        "rok",
        "liczba_dzialek",
        "przyrost_dzialek",
        "udzial_R",
        "delta_R_pp",
        "udzial_B",
        "delta_B_pp",
        "udzial_L",
        "udzial_Ps",
        "srednia_liczba_dzialek_hex",
        "heksy_z_podzialem",
        "heksy_z_odrolnieniem",
    ]
    return df[cols]


def per_gmina_year(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute(f"""
        SELECT
            d.jednostka AS gmina,
            d.year AS rok,
            round(avg(d.uzg_R_share), 4) AS udzial_R,
            round(avg(d.uzg_B_share), 4) AS udzial_B,
            sum(CAST(d.split_proxy AS INT)) AS heksy_z_podzialem,
            sum(CAST(k.convert_proxy AS INT)) AS heksy_z_odrolnieniem
        FROM {DATASET} d
        LEFT JOIN {KUG_LABELS} k USING (hex_id, year)
        WHERE d.jednostka IS NOT NULL
        GROUP BY d.jednostka, d.year
        ORDER BY d.jednostka, d.year
    """).df()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(DB, read_only=True)
    con.execute("LOAD spatial;")
    powiat = per_year(con)
    gmina = per_gmina_year(con)
    con.close()

    powiat.to_csv(OUT_DIR / "yoy_powiat.csv", index=False)
    gmina.to_csv(OUT_DIR / "yoy_gmina.csv", index=False)
    try:
        with pd.ExcelWriter(OUT_DIR / "yoy_changes.xlsx", engine="openpyxl") as xl:
            powiat.to_excel(xl, sheet_name="powiat_rok", index=False)
            gmina.to_excel(xl, sheet_name="gmina_rok", index=False)
        xlsx_msg = "yoy_changes.xlsx"
    except ImportError:
        xlsx_msg = "(xlsx pominięty — brak openpyxl; `uv add openpyxl`)"

    print(f"powiat/rok: {len(powiat)} wierszy, gmina/rok: {len(gmina)} wierszy")
    print(f"→ {OUT_DIR}/yoy_powiat.csv, yoy_gmina.csv, {xlsx_msg}")
    print(powiat.to_string(index=False))


if __name__ == "__main__":
    main()
