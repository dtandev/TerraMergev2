# Audyt kodu TerraMergev2 — błędy i zarządzanie konfiguracją

Data audytu: 2026-07-25
Zakres: `rule_them_all.py`, `src/` (25 plików, ~7,4k linii), `conf/` (9 plików YAML), pliki projektowe (`environment.yml`, `pyproject.toml`, `README.md`).

## TL;DR

Kod działa (jest logika, są testy manualne pod danymi), ale **konfiguracja jest systemowo rozjechana z kodem** — to nie jest wrażenie, tylko policzalny fakt: przejrzałem wszystkie ~217 odczytów configu w kodzie i porównałem z faktycznymi kluczami w YAML. Główna przyczyna to niespójne użycie dyrektywy Hydry `# @package _global_`, która przenosi klucze z `conf/features/*.yaml` i `conf/pipeline/*.yaml` do korzenia configu — ale połowa kodu nadal czyta je tak, jakby były zagnieżdżone pod `cfg.features.*` / `cfg.pipeline.*`. Efekt: część kroków **zawsze jest pomijana niezależnie od configu**, część **zawsze się wywala**, a jeden krok modelowania cicho ignoruje realne ustawienia walidacji.

Poniżej: (1) błędy krytyczne (crash / zawsze złe zachowanie), (2) błędy jakości danych/ML, (3) martwy kod i duplikacja, (4) problemy z uruchomieniem projektu, (5) analiza configu + konkretna rekomendacja uproszczenia.

---

## 1. Błędy krytyczne

### 1.1 Systemowy rozjazd configu przez `@package _global_`

`conf/features/default.yaml` i `conf/pipeline/make_hexagons.yaml` zaczynają się od `# @package _global_`. To sprawia, że ich zawartość ląduje w **korzeniu** configu (`cfg.enabled`, `cfg.add_uzg...`, `cfg.hex...`, `cfg.egib...`), a nie pod `cfg.features.*` / `cfg.pipeline.*`. Mimo to większość kodu czyta je z prefiksem `"features."` / `"pipeline."`, który **nie istnieje**. Ponieważ wszędzie używane jest `OmegaConf.select(cfg, "...", default=X)`, brakujący klucz nie rzuca błędu — cicho zwraca `default`. To jest źródło większości poniższych błędów.

Konkretne skutki (potwierdzone przez dwóch agentów przeglądających kod + moją własną weryfikację grepem):

| Plik:linia | Co czyta | Realny klucz | Efekt |
|---|---|---|---|
| `src/features/add_transaction_prices.py:48` | `features.enabled`, `features.add_transaction_prices.enabled` | `enabled` (root) | Krok **zawsze pomijany**, log "disabled – skipping" niezależnie od configu |
| `src/features/add_uzg.py:284` | `features.add_uzg.enabled` | `add_uzg.enabled` (root) | Krok **zawsze pomijany** |
| `src/features/add_mpzp.py:165` / `add_mpzp_for_parcels.py:164` | `cfg.features.add_mpzp` (bezpośredni atrybut, bez fallbacku) | `cfg.add_mpzp` | **Twardy crash** `ConfigAttributeError` — `cfg.features` w ogóle nie istnieje |
| `src/features/add_geometric_features.py:259,262,296` | `features.add_geometric_features.add_to_duckdb.enabled` (default `True`) | `add_geometric_features.add_to_duckdb.enabled: false` | Zamiast szanować `false` z configu, kod **zawsze zapisuje do DuckDB** — odwrotność zamierzonego zachowania |
| `src/features/add_parcels_data_hexs.py:386-456` | `pipeline.parcels_table`, `pipeline.hex_table`, `pipeline.out_table` itd. | `parcels_table=""`, `out_table="(empty)"` | Nawet po naprawieniu importu (patrz 1.2) funkcja wywali się na `SELECT ... FROM  AS de` (pusta nazwa tabeli) |
| `src/features/add_transactions_hex.py:151,203-207,412-413` | `pipeline.hex.table` → `None` | `hex.table` | SQL `FROM None` → crash. Dodatkowo literówka w fallbacku nazwy tabeli: `hex.Transakcjer9` zamiast `hex.Transakcje_r9` |
| `src/modeling/neighborhood.py:1136` | `dataset.join_y_label.enable` (literówka: brakuje "d") | `join_y_label.enabled` | Obecnie niegroźne (default `True` pokrywa się z configiem), ale ustawienie `enabled: false` w YAML **zostanie zignorowane** |

Tylko `add_uzg_hexs.py` i `add_mpzp_hexs.py` mają ręczny podwójny fallback (`_get_val(cfg, ["hex.table", "pipeline.hex.table"])`) — czyli ktoś już się z tym problemem mierzył punktowo, zamiast naprawić go u źródła.

### 1.2 Zły import — martwa gałąź `make_hexagons`

`rule_them_all.py:209`:
```python
from v2.src.features.add_parcels_data_hexs import run_add_parcels_data
```
Pakiet nazywa się `src`, nie `v2.src` — nie ma takiego modułu. Ten import **zawsze rzuci `ModuleNotFoundError`**, jeśli krok `pipeline.add_parcels_data` kiedykolwiek się uruchomi. W praktyce nie widać tego, bo (patrz 1.3) ten krok i tak jest permanentnie wyłączony przez inny bug — dwa niezależne błędy nakładają się na siebie i się nawzajem maskują.

### 1.3 Cztery kroki „make_hexagons" nie da się włączyć przez config

W `rule_them_all.py` linie 191/193 mają ręczny fallback na wypadek `@package _global_` (`_sel(cfg, "pipeline.make", _sel(cfg, "make", False))`), ale linie 206, 224, 240, 253 (`add_parcels_data`, `add_transactions_data`, `add_mpzp_data`, `add_kug_data`) **nie mają** tego fallbacku — czytają wyłącznie `pipeline.xxx.enabled`, który nie istnieje. Efekt: te cztery kroki są **zawsze pomijane**, niezależnie od wartości `enabled` w `conf/pipeline/make_hexagons.yaml`.

### 1.4 `duckdb.threads` / `duckdb.memory_limit` nigdy nie są stosowane

`src/prepare_data/duckdb_init.py:16,27,31` czyta `data.duckdb.schema`, `data.duckdb.threads`, `data.duckdb.memory_limit`. Faktyczne klucze w `conf/config.yaml` to `duckdb.schema`, `duckdb.threads`, `duckdb.memory_limit` (bez `data.` z przodu — sam `rule_them_all.py:50` poprawnie czyta `duckdb.init` bez tego prefiksu, więc niespójność jest nawet w obrębie jednego configu). `schema` przypadkiem działa (default `"egib"` = wartość w YAML), ale **`threads` i `memory_limit` nigdy się nie ustawiają**, nawet jeśli wpiszesz je w configu.

### 1.5 Rozjazd kluczy w `train_model.py`

- `train_model.py:600` czyta `model.test_years` (default `[2023, 2024]`); w `conf/model/default.yaml:6` klucz nazywa się `valid_years: [2024, 2023]`. Kod **zawsze** używa hardcoded defaultu — obecnie akurat ten sam zestaw lat, więc bug jest niewidoczny, ale zmiana `valid_years` w configu nic nie zmieni.
- `train_model.py:653` czyta `model.n_perm_repeats` (nie istnieje w YAML — tam jest `perm_n_repeats`, poprawnie użyty gdzie indziej w tym samym pliku) → szybki permutation-importance zawsze na hardcoded `5`.
- `model.prefer_cats`, `model.cat_cardinality_max` (linie 598-599) nie mają odpowiedników w YAML w ogóle.
- `train_max_year=2022` (linia 630) jest wpisane na sztywno w kodzie, nie ma odpowiednika w configu.

### 1.6 `conf/pipeline/model.yaml` i `eval.yaml` to martwy, zepsuty config

Zweryfikowałem grepem: **żaden plik `.py` w repo nie odwołuje się** do `conf/pipeline/model.yaml` ani `conf/pipeline/eval.yaml` (sekcje `cv`, `stages`, `inputs`, `outputs`, `evaluation`, `alerts`, `reports`). Sekcja `cv: {type: GroupKFold, n_splits: 5, group_col: hex_cell}` sugeruje, że trening miał używać CV z grupowaniem po heksie — ale `train_model.py` robi zwykły split czasowy (`temporal_holdout_split`), bez GroupKFold. Co gorsza, oba pliki interpolują `${contracts.model_dir}`, `${contracts.features_selected_file}` itd., a grupa configu `contracts` **nie istnieje nigdzie** w `conf/` — więc gdyby ktoś je jednak aktywował (`pipeline=model`), Hydra wywali błąd interpolacji przy starcie.

### 1.7 Walidacja modelu jest optymistycznie obciążona (data leakage metodologiczny)

W `train_model.py` ten sam zbiór walidacyjny (`X_val`/`y_val`) służy jednocześnie do **early stopping** LightGBM i do **wszystkich raportowanych metryk** (AP, ROC AUC, Brier) oraz do dopasowania kalibracji izotonicznej, która potem jest używana do skalowania predykcji na danych 2025. To klasyczny "double-dipping" — raportowane metryki będą systematycznie lepsze niż realna skuteczność modelu na nowych danych. W kodzie jest nawet komentarz przyznający, że kalibracja jest "optimistic" — czyli autor był tego świadomy, ale nie ma osobnego zbioru testowego.

---

## 2. Błędy jakości danych / ML

- **`neighborhood.py`, brak try/except przy ładowaniu tabel** (linie ~196-209): `calculate_neighborhood.tables_to_split` w `conf/dataset/default.yaml` wymienia `hex.MPZP_r8` i `hex.Transakcje_r8`, których kroki produkujące (`add_mpzp_data`, `add_transactions_data`) są domyślnie wyłączone. Brak tabeli → `SELECT * FROM {tbl}` rzuca `CatalogException` i **cały krok agregacji sąsiedztwa się wywala**, zamiast pominąć brakującą tabelę.
- **Kolumny boolowskie znikają z agregacji sąsiedztwa** (`neighborhood.py:581`): `df.select_dtypes(include=[np.number])` w pandas **nie obejmuje** `bool` — więc cechy typu `split_proxy` nigdy nie dostają `nbr_r{R}_..._mean`, mimo że docstring twierdzi, że agregowane są "wszystkie kolumny numeryczne". To prawdopodobnie jedna z najbardziej predykcyjnych cech (odsetek sąsiadów, którzy się dzielą) — cicho pomijana.
- **Błąd "luki rocznej" w etykietach** (`labeling.py`, `build_split_labels_full` / `build_uzg_conversion_labels`): `shift(1)`/`shift(-1)` per `hex_id` liczy różnice **po kolejności wierszy**, nie po faktycznym roku. Jeśli dla danego heksu brakuje roku (np. dane z 2022 i 2024, bez 2023), delty (`delta_mean_area`, `delta_R`) liczą się tak, jakby to był jeden rok — zniekształcając `split_proxy`/`convert_proxy`, a przez to wszystkie etykiety `y_next*`.
- **Niedeterministyczna cecha geometryczna** (`GeometricFeaturesMaker.py`, `MomentInertiaFeatures._sample_boundary_points`, ~linia 471): `np.random.uniform` bez seeda → cechy momentu bezwładności **różnią się między uruchomieniami** dla tych samych danych, co utrudnia reprodukcję modelu i debugging.
- **Duplikacja wierszy przy złączeniu przestrzennym MPZP** (`add_mpzp_for_parcels.py:247-265`): join `predicate="intersects"` z warstwami `przezn`/`plany` może namnożyć wiersze, gdy działka nachodzi na >1 poligon planu — bez deduplikacji po joinie liczba wierszy rozjeżdża się z liczbą działek wejściowych, a etykiety dalej licząc na tych rozmnożonych wierszach.
- **Hardcoded output path** (`neighborhood.py:1155`): `clean_df_with_y.to_parquet("cleaned_labels.parquet", index=False)` zapisuje do bieżącego katalogu roboczego, poza jakąkolwiek skonfigurowaną strukturą — kolejne uruchomienia się nadpisują.

---

## 3. Martwy kod i duplikacja

- `src/common/logging_utils.py` i `src/common/validation_utils.py` — **0 bajtów, puste pliki**, nigdzie nieimportowane. Wygląda na to, że miały być miejscem na dokładnie te helpery, które teraz są duplikowane po całym `src/features/` (patrz niżej) — ale nikt ich nie uzupełnił.
- `src/features/add_parcels_data_hexs_bad copy.py` — plik z nazwą zawierającą spację, niemożliwy do zaimportowania jako moduł Pythona; duplikuje ~80% `add_parcels_data_hexs.py` (te same funkcje `_sel`, `connect_duckdb`, `_detect_srid`, `save_geodf_as_ewkb_geometry`) i ma te same błędy configu co oryginał. Nigdzie nieużywany — kandydat do usunięcia albo przeniesienia poza repo produkcyjne.
- Funkcje `_sel`/`_get_val`, `connect_duckdb`, `_detect_srid`, `save_geodf_as_ewkb_geometry` są kopiowane niemal 1:1 w co najmniej 5 plikach (`add_uzg_hexs.py`, `add_mpzp_hexs.py`, `add_transactions_hex.py`, `add_parcels_data_hexs.py` + bad-copy) — naturalny kandydat na wspólny moduł (`src/common/duckdb_utils.py`), zamiast pustych stubów wspomnianych wyżej.
- `neighborhood.py:1160/1166` — sprawdzenie `con is not None` jest martwe (funkcja łącząca zawsze zwraca połączenie albo rzuca wyjątek), więc gałąź `else: logger.error(...)` jest nieosiągalna.
- `train_model.py:130` — `y = df[label_col].astype(int) if df[label_col].dtype == "boolean" else df[label_col].astype(int)` — obie gałęzie warunku robią to samo; jeśli intencją było najpierw obsłużyć `pd.NA` w nullable Boolean, ta logika zaginęła (a `.astype(int)` na `pd.NA` rzuci wyjątek).
- `rule_them_all.py:210-213` — jedyne miejsce w całym pliku, gdzie logger jest wołany ze stylem `%s` zamiast `{}` (loguru obsługuje tylko `{}`) — komunikat wyświetli się z literalnym `%s` zamiast wartości.

---

## 4. Problemy z uruchomieniem / zależnościami projektu

- **`environment.yml:15`** zawiera pakiet `geometrie` — to nie jest realna nazwa pakietu na conda-forge (wygląda na literówkę, prawdopodobnie miało być `shapely`, który i tak przychodzi tranzytywnie z `geopandas`). `conda env create -f environment.yml` **prawdopodobnie się wywali** na tej linii.
- **`pyproject.toml` ma 0 bajtów**, a `README.md` instruuje `pip install -e .` — bez `[build-system]`/`[project]` ta komenda się nie powiedzie. Projekt nie da się zainstalować zgodnie z własną dokumentacją.
- `Makefile` i `LICENSE` — również puste (0 bajtów); nie błąd per se, ale niekompletne rusztowanie projektu.
- **Hardcoded, prywatne ścieżki Windows w wersjonowanym configu** (`conf/config.yaml:13-16`): `D:\EGIB_new\pow_ostrodzki\`, `C:\Users\user\OneDrive\...\TerraMerge\v2\...`. Ten sam `v2` w ścieżce koresponduje z błędnym importem z punktu 1.2 — sugeruje, że repo powstało z przeniesienia/kopii wcześniejszego projektu `TerraMerge/v2`, i część referencji (importy, ścieżki) nie została zaktualizowana po migracji do `TerraMergev2`. To też jest bezpośrednio powiązane z prośbą o uproszczenie configu — patrz sekcja 5.
- `artifacts/models/parcel_r8_full_1y/config_snapshot.yaml` — snapshot uruchomienia zawiera te same prywatne ścieżki, zacommitowany do repo.

---

## 5. Zarządzanie konfiguracją — analiza i rekomendacja

### Dlaczego jest tak, jak jest

To nie jest tylko "dużo plików YAML" — masz cztery nakładające się problemy jednocześnie:

1. **Niespójny `@package _global_`.** Dwie grupy configu (`features`, `pipeline`) są spłaszczane do korzenia, ale kod (pisany prawdopodobnie w różnym czasie / przez różne podejścia) w połowie miejsc zakłada strukturę zagnieżdżoną. To jest **jedna, konkretna, naprawialna przyczyna** większości błędów z sekcji 1.
2. **Brak walidacji configu.** `OmegaConf.select(cfg, "klucz", default=X)` jest używane wszędzie — to sprawia, że literówka w kluczu (`test_years`/`valid_years`, `enable`/`enabled`, `nan_threshold`/`high_nan_threshold`) nigdy nie rzuci błędu, tylko cicho wróci do defaultu wpisanego w kodzie. Innymi słowy: **YAML w tym repo jest w dużej mierze dekoracją** — realne wartości default żyją rozproszone po ~15 plikach `.py`, nie w jednym miejscu.
3. **Osierocone pliki configu** (`conf/pipeline/model.yaml`, `eval.yaml`) referencujące nieistniejącą grupę `contracts` — ktoś zaprojektował strukturę "na przyszłość", ale nigdy jej nie podłączył, i teraz trudno odróżnić "aktywny config" od "szkic".
4. **Dane środowiskowe (ścieżki, sekrety) wmieszane w default config** i zacommitowane — więc każdy, kto klonuje repo, musi ręcznie edytować `conf/config.yaml`, żeby cokolwiek uruchomić, i łatwo przez przypadek zacommitować z powrotem swoją ścieżkę.

### Rekomendacja (od najwyższej dźwigni)

1. **Usuń `@package _global_` z `conf/features/default.yaml` i `conf/pipeline/make_hexagons.yaml`.** Niech żyją pod `cfg.features.*` / `cfg.pipeline.*` — zgodnie z nazwą grupy Hydry, tak jak intuicyjnie czyta je większość kodu. To jeden PR, który naprawia największą liczbę bugów z sekcji 1 na raz. Potem trzeba przejść przez `rule_them_all.py` i pousuwać ręczne dual-fallbacki (`_sel(cfg, "pipeline.make", _sel(cfg, "make", False))`), bo staną się zbędne — a to i tak dobra okazja, żeby jednym grepem (`OmegaConf.select(cfg, "` po całym repo) sprawdzić każdy klucz przeciw realnym plikom YAML, tak jak zrobiłem to w tym audycie.

2. **Wprowadź strukturalny config zamiast gołego `DictConfig`.** Hydra ma natywne wsparcie na to przez `ConfigStore` + `@dataclass` (albo prościej: jeden moduł `src/common/config_schema.py` z `pydantic.BaseModel` per grupa configu — `PrepareConfig`, `FeaturesConfig`, `ModelConfig` itd.), walidowany raz na starcie w `rule_them_all.py`. Zysk: literówka w kluczu albo brakująca sekcja **rzuca błąd przy starcie programu**, a nie cicho wraca do defaultu ukrytego gdzieś w linii 600 `train_model.py`. To jest właściwa odpowiedź na "konfiguracja jest rozproszona" — bo dziś prawda o tym, "jaka wartość faktycznie zadziała", wymaga czytania i YAML, i kodu jednocześnie; po tej zmianie YAML sam w sobie stanie się źródłem prawdy.

3. **Wydziel dane środowiskowe/maszynowe z wersjonowanego configu.** `data.base_dir`, `data.duckdb_path`, `data.transactions_path`, `data.mpzp_mapping_csv` powinny żyć w osobnej grupie Hydry, np. `conf/local/default.yaml`, dodanej do `.gitignore`, z zacommitowanym `conf/local/example.yaml` jako szablonem. To standardowy wzorzec Hydry na "per-maszyna overrides" i rozwiązuje zarówno problem prywatnych ścieżek Windows w repo, jak i to, że każdy nowy współpracownik musi zgadywać, co edytować, żeby cokolwiek uruchomić.

4. **Skonsoliduj duplikowane helpery** (`_sel`, `connect_duckdb`, `_detect_srid`, `save_geodf_as_ewkb_geometry`) do `src/common/` — pliki `logging_utils.py`/`validation_utils.py` już tam czekają puste, dokładnie na to.

5. **Usuń albo podłącz** `conf/pipeline/model.yaml` i `eval.yaml` — dziś to config-widmo: wygląda na aktywny, nic go nie czyta, a przy próbie użycia i tak wybuchnie na `${contracts.*}`.

6. Jeśli chcesz coś zrobić najpierw, zanim usiądziemy do refaktoru configu całościowo: punkt 1 (usunięcie `@package _global_`) to najmniejsza zmiana z największym efektem — naprawia realne, obecnie ciche awarie w produkcji pipeline'u, zanim jeszcze ruszymy architekturę.

---

## Podsumowanie liczbowe

- **7 błędów krytycznych** (crash lub zawsze-złe-zachowanie niezależnie od configu)
- **6 błędów jakości danych/ML** (w tym metodologiczny data leakage w ewaluacji i błąd w konstrukcji etykiet)
- **6 elementów martwego kodu / duplikacji**
- **4 problemy z uruchomieniem projektu** (zależności, puste pliki projektowe, prywatne ścieżki w repo)
- Rdzeń problemu configu: **jedna dyrektywa (`@package _global_`) użyta niespójnie** + brak walidacji schematu configu.
