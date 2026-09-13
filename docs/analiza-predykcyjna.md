# Warstwa predykcyjna — modele, prognozy, mapy, zestawienia r/r

Opis części analitycznej i predykcyjnej (skrypty w `scripts/`, config `conf/forecast.yaml`).
Dwa procesy przekształceń gruntów, na wspólnej macierzy cech heksowych
`dataset.Parcels_neighborhood_r8` (siatka H3 r8, powiat szczycieński, lata 2013–2026):

- **podział** nieruchomości (`task=split`, etykieta z `labels.ParcelLabels_r8`),
- **odrolnienie** R→B (`task=odrol`, etykieta z `labels.kugLabels_r8_uzg_R_share`).

To jeden feature-matrix i dwie etykiety — nie ma osobnego datasetu per zadanie.

## Skrypty

Wszystkie są czyste (`duckdb` + `lightgbm` + `sklearn`), **nie importują `osgeo`**, więc nie
wymagają obejścia GDAL. Czytają bazę z `TERRAMERGE_DUCKDB_PATH`
(domyślnie `artifacts/duckdb/terramerge.duckdb`).

| Skrypt | Co robi | Wynik |
|---|---|---|
| `scripts/train_forecast.py` | trenuje modele per horyzont +1/+2/+3 (LightGBM, strojenie coordinate-descent, przesuwane okno czasowe), kalibruje (`CalibratedClassifierCV`, izotonika) | modele `artifacts/models/forecast/<task>_h{1,2,3}.joblib` + metryki `artifacts/reports/forecast_metrics_<task>.json` |
| `scripts/export_forecast_maps.py` | aplikuje modele do jednego roku bazowego → prognoza per hex, z geometrią | `artifacts/maps/forecast_maps_<task>_<rok>.{duckdb,gpkg}` (3 warstwy: h1/h2/h3) |
| `scripts/yoy_changes.py` | zestawienia zmian rok-do-roku (powiat + gminy) | `artifacts/reports/yoy_powiat.csv`, `yoy_gmina.csv`, `yoy_changes.xlsx` |
| `scripts/model_both.py` | analiza porównawcza obu zadań: LightGBM vs regresja logistyczna vs baseline, permutation importance | `artifacts/reports/model_results.json` (źródło `raport_modele.html`) |
| `scripts/horizon_analysis.py` | jak dokładność i ważność cech zmieniają się z horyzontem 1→3 lata | `artifacts/reports/horizon_results.json` (źródło `raport_horyzonty.html`) |

Uruchomienie (oba zadania, oba etapy):

```bash
for T in split odrol; do
  python scripts/train_forecast.py --config conf/forecast.yaml --task $T
  DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/Cellar/x265/4.2/lib:/opt/homebrew/lib \
  python scripts/export_forecast_maps.py --config conf/forecast.yaml --task $T --base-year 2026
done
python scripts/yoy_changes.py
```

(`DYLD_FALLBACK…` jest potrzebne tylko lokalnie, gdy pakiet `osgeo` w venv jest zepsuty po
`brew upgrade x265` — GDAL nie jest importowany przez te skrypty, ale `geopandas` przy zapisie
GeoPackage tak.)

## Wyniki (test out-of-sample; trening ≤ rok, walidacja +1, test +2)

| proces / horyzont | AP | ROC-AUC | precyzja@10% | lift |
|---|---|---|---|---|
| odrolnienie +1 | 0,37 | 0,91 | 36% | 6,2× |
| odrolnienie +2 | 0,40 | 0,91 | 37% | 6,3× |
| odrolnienie +3 | 0,40 | 0,91 | 38% | 6,4× |
| podział +1 | 0,27 | 0,83 | 32% | 4,0× |
| podział +2 | 0,29 | 0,83 | 32% | 4,1× |
| podział +3 | 0,27 | 0,82 | 32% | 4,0× |

Dokładność prawie nie spada z horyzontem — prognoza 3 lata w przód jest niemal tak dobra jak
roczna, bo o przekształceniu decyduje trwała struktura heksa, nie krótkoterminowy impuls.
Cechy: przy +1 prowadzi autoregresja (stan bieżący), przy +2/+3 przejmują użytki (`uzg_*`) i
sąsiedztwo (`nbr_*`). **MPZP i ceny transakcyjne mają ≈0 wartości predykcyjnej** w tym
powiecie (choć są przydatne w części opisowej).

## Jak czytać mapy prognoz

Każda warstwa (`forecast_h1/h2/h3`) ma jeden wiersz na hex i kolumny:

- `probability` — surowy wynik LightGBM; wąskie pasmo, traktować jako ranking.
- `prob_calibrated` — skalibrowane prawdopodobieństwo (izotonika out-of-fold); interpretowalne
  jako realna szansa. Uwaga: przy bazie 2026 (ostatni, niepełny rok) jest mocno ściśnięte —
  model rzuca z najnowszego roku ostrożniej.
- `risk_percentile` — pozycja w rankingu ryzyka 0–100. **Do stylizacji mapy używać tej kolumny**
  (rozkłada się na pełny zakres niezależnie od roku bazowego).

CRS: EPSG:2180. GeoPackage ma go osadzony; przy odczycie `.duckdb` QGIS może pokazać „nieznany"
— wtedy przypisać ręcznie (DuckDB nie utrwala SRID w kolumnie GEOMETRY).

## Ograniczenia

- Etykiety są proxy na poziomie heksa (agregaty), nie geometrycznym śledzeniem lineage działek.
- `probability` nie jest dosłownym prawdopodobieństwem — do rankingu/mapy, nie do odczytu „X%".
- Wyniki i modele (`artifacts/reports`, `artifacts/maps`, `artifacts/models/forecast`) są
  **gitignorowane** — reprodukowalne przez powyższe skrypty; źródłem prawdy jest kod + config.
