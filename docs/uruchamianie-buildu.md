# Uruchamianie buildu bazy TerraMerge (bez treningu)

Jak samodzielnie zbudować bazę cech dla jednego powiatu — od surowych danych EGiB
(GDB/GML/SHP/SWDE) do tabeli wejściowej modelu `dataset.Parcels_neighborhood_r8`.
Sprawdzone na powiecie szczycieńskim (kod 2817), lata 2013–2024 + 2026, rozdzielczość R=8.

Wynik: `artifacts/duckdb/terramerge.duckdb`, tabela `dataset.Parcels_neighborhood_r8`
(~35 tys. wierszy × ~305 kolumn: cechy geometryczne `gf_*`, udziały użytków `uzg_*`,
agregaty sąsiedzkie `nbr_*`, etykieta `y_next`).

---

## 1. Wymagania jednorazowe

```bash
cd /Users/dariusz.tanajewski/Documents/Code/TerraMergev2
uv sync            # albo: pip install -e . w .venv
```

Wszystkie komendy poniżej wołają `.venv/bin/python` wprost.

## 2. Zawężenie danych do jednego powiatu (APFS clone)

Pipeline iteruje po `base_dir/rok_YYYY/` i przetwarza **każdy** powiat, jaki znajdzie.
Dlatego budujemy osobny `base_dir` zawierający tylko szczycieński. Klonujemy przez APFS
(`cp -Rc`) — natychmiastowe kopiowanie copy-on-write, źródło pozostaje nietknięte.
Symlinki NIE działają (rglob nie wchodzi w dowłania).

Struktura źródła: `…/Urządzeniowo-rolne/extracted/rok_YYYY/2817_szczycienski*`
(nazwy się różnią: goła, `_SWDE`, `_GML`, czasem z „ń"). Katalog docelowy `_run_szczycienski`
już istnieje i zawiera klony wszystkich dostępnych lat + `parquets/` (wynik extract+merge).

Aby zbudować go od zera dla innego powiatu (przykład — kod `NNNN`):

```bash
SRC="/Users/dariusz.tanajewski/Documents/Data/Urządzeniowo-rolne/extracted"
DST="/Users/dariusz.tanajewski/Documents/Data/Urządzeniowo-rolne/_run_NNNN"
mkdir -p "$DST"
for d in "$SRC"/rok_*; do
  y=$(basename "$d")
  match=$(find "$d" -maxdepth 1 -type d -iname 'NNNN_*' | head -1)
  [ -n "$match" ] && mkdir -p "$DST/$y" && cp -Rc "$match" "$DST/$y/"
done
```

Uwaga: dla szczycieńskiego brakuje w źródle roku **2015** (nie ma danych) i **2025**
(rok istnieje, ale bez tego powiatu). To normalne — kod obsługuje luki w latach.

## 3. Zmienne środowiskowe (ścieżki)

Ścieżki podajemy przez env, nie przez CLI (parser dławi się na `ą`/`-` w ścieżce):

```bash
export TERRAMERGE_BASE_DIR="/Users/dariusz.tanajewski/Documents/Data/Urządzeniowo-rolne/_run_szczycienski"
export TERRAMERGE_DUCKDB_PATH="artifacts/duckdb/terramerge.duckdb"
```

`load_dotenv(override=False)` — zmienne z powłoki wygrywają nad `.env`. NIE używaj nazwy
`egib.duckdb` (kolizja katalog↔schema w DuckDB).

## 4. Pełny build (jedna komenda)

Domyślny config włącza już wszystkie potrzebne etapy (rozdzielczości, `make_hexagons.year`,
zapis cech do DuckDB, `tables_to_split` — wszystko poprawne w `conf/`). Wystarczy:

```bash
.venv/bin/python -m src.main \
  prepare.enabled=true prepare.clean.enabled=false \
  features.enabled=true \
  pipeline.make=true \
  dataset.enabled=true \
  model.enabled=false
```

- `prepare.clean.enabled=false` — **zawsze**. Etap `clean` (clean_directories) USUWA
  katalogi swde/gml/shp; na klonie to strata danych klonu.
- `model.enabled=false` — bez treningu.

Czas: rząd kilkunastu minut (najwolniejszy `MomentInertiaFeatures` w geometrii).
Etapy: prepare (extract → merge → clean_dataset) → features (add_uzg, geometric) →
pipeline (make_hexagons → add_parcels_data → add_kug_data) → dataset (labeling →
neighborhood).

## 5. Częściowe re-runy (gdy zmieniłeś tylko fragment)

Etapy da się wyłączać osobno — nie trzeba za każdym razem powtarzać kosztownej ekstrakcji.

| Co zmieniłeś | Co uruchomić (dodatkowe flagi) |
|---|---|
| tylko czyszczenie/atrybuty (clean_dataset) | `prepare.extract.enabled=false prepare.merge.enabled=false` (reszta jak w §4) |
| tylko cechy (add_uzg / geometric) | `prepare.enabled=false` |
| tylko hexagony/dataset (kod pipeline lub labeling) | `prepare.enabled=false features.enabled=false` |
| tylko dataset (labeling + neighborhood) | `prepare.enabled=false features.enabled=false pipeline.make=false` |

Częściowe re-runy czytają z istniejącej bazy DuckDB, więc wymagają wcześniejszego
pełnego przebiegu.

Uruchomienie w tle z logiem:

```bash
nohup .venv/bin/python -m src.main <flagi> > /tmp/build.log 2>&1 &
tail -f /tmp/build.log            # podgląd; Ctrl-C przerywa TYLKO tail, nie build
```

## 6. Podgląd bazy

Gdy baza jest otwarta w QGIS, DuckDB blokuje plik — łącz się **read-only**:

```bash
.venv/bin/python - <<'PY'
import duckdb
con = duckdb.connect('artifacts/duckdb/terramerge.duckdb', read_only=True)
con.execute('LOAD spatial;')
print(con.execute("SELECT table_schema, table_name FROM information_schema.tables ORDER BY 1,2").fetchall())
print(con.execute('SELECT count(*), count(DISTINCT year) FROM dataset."Parcels_neighborhood_r8"').fetchone())
PY
```

Kluczowe tabele: `dataset.Parcels_neighborhood_r8` (wejście modelu), `hex.DzialkaEwidencyjna_r8`,
`hex.kug_r8`, `labels.ParcelLabels_r8`, `labels.kugLabels_r8_*`, `egib.*` (surowe warstwy).

## 7. CRS w QGIS — dlaczego „nieznany" i jak wymusić 2180

Wszystkie dane są przeliczone do **EPSG:2180** przy ekstrakcji (źródłowe GDB są w CS2000/7 ≈
EPSG:2178). Ale DuckDB v1.5.5 **nie zapisuje CRS w kolumnie GEOMETRY** — po zapisie
`ST_CRS(geometry)` zwraca `None`. Nie da się ustawić „domyślnego CRS bazy": geometria na
dysku jest bez układu, choć **współrzędne są w 2180** (dla szczycieńskiego X≈605–669 tys.,
Y≈604–660 tys.; surowe CS2000 miałoby 7-cyfrowy easting).

Trzy sposoby, żeby QGIS pokazał 2180:

1. **Ręcznie w QGIS** — po dodaniu warstwy: Layer Properties → Source → *Assigned CRS* =
   EPSG:2180, potem zapisz projekt. Współrzędne już pasują, ustawiasz tylko etykietę układu.

2. **Eksport do GeoPackage z osadzonym CRS** (najpewniejsze — QGIS otworzy od razu w 2180):

   ```bash
   .venv/bin/python - <<'PY'
   import duckdb
   con = duckdb.connect('artifacts/duckdb/terramerge.duckdb', read_only=True)
   con.execute('LOAD spatial;')
   con.execute("""
     COPY (SELECT * FROM dataset."Parcels_neighborhood_r8")
     TO 'artifacts/parcels_neighborhood_r8.gpkg'
     WITH (FORMAT GDAL, DRIVER 'GPKG', SRS 'EPSG:2180')
   """)
   PY
   ```

   `SRS 'EPSG:2180'` wpisuje układ do pliku GPKG. Zmień nazwę tabeli/pliku wedle potrzeb.

3. **Widok z ST_SetCRS** — `CREATE VIEW v AS SELECT … ST_SetCRS(geometry,'EPSG:2180') AS geometry
   FROM …`. Na poziomie SQL `ST_CRS` zwraca wtedy 2180, ale nie ma pewności, że provider DuckDB
   w QGIS to odczyta — do sprawdzenia u siebie. Jeśli nie zadziała, użyj sposobu 2.

## 8. Znane pułapki

- **KUG a tabela działek.** Dane o użytkach (KUG / `uzg_*`) są w `hex.kug_r8` i
  `labels.kugLabels_r8_*`, a NIE w `dataset.Parcels_neighborhood_r8` (ta jest o podziałach
  działek). Jeśli szukasz KUG w tabeli działek — go tam nie ma z założenia, dla żadnego roku.
  Rok 2017 KUG jest kompletny (2937 hexów, wszystkie udziały wypełnione); realnie brakuje
  roku **2015** (nie ma go w źródle), a 2014 i 2016 mają częściowe pokrycie hexów.
- **Transakcje i MPZP wyłączone** (`add_transaction_prices`, `add_mpzp`) — brak linku do
  transakcji i brak WFS + CSV mapowania. Włącz dopiero, gdy dane będą dostępne, i wtedy dopisz
  ich tabele z powrotem do `dataset.calculate_neighborhood.tables_to_split`.
- **`timeout` nie istnieje na macOS**; do przerwania długiego runu użyj `pkill -f src.main`.
- **Kasowanie klonu** (`rm -rf _run_*`) jest wolne — SWDE to tysiące drobnych plików.
