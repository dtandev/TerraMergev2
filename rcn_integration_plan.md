# Plan integracji RCN (rejestr cen transakcji) z `data.transactions_path`

Data: 2026-07-25
Zakres: `/Users/dariusz.tanajewski/Documents/Data/w-m egib/RCN/` vs. wymagania `src/features/add_transaction_prices.py` + `src/features/add_transactions_hex.py`.

To jest **plan, nie implementacja** — zgodnie z ustaleniem, żadne pliki w `RCN/` ani kod TerraMerge nie zostały zmienione podczas tej analizy.

## Co jest w `RCN/`

105 plików, dwie równoległe dostawy tych samych danych dla 21 powiatów woj. warmińsko-mazurskiego (2801–2819, 2861 M. Elbląg, 2862 M. Olsztyn):

1. **`NNNN_transakcje_ceny.gpkg`** (21 plików) — jeden GeoPackage na powiat, każdy z 3 warstwami:
   `transakcje_dzialki` (transakcje powiązane z działką — **to jest właściwa warstwa** dla TerraMerge), `transakcje_budynki`, `transakcje_lokale`.
2. **`Powiat <NAZWA> NNNN_transakcje_ceny_{lokalowe,niezabudowane,zabudowane_bez_lokali}.gpkg`** + `..._transakcje_grupy.xlsx` (21 × 4 = 84 plików) — ta sama treść rozbita na kategorie, z niewygodnymi nazwami plików (spacje, polskie znaki) do automatyzacji.

**Rekomendacja:** użyj wyłącznie zestawu (1) — jest kompletny, spójny i dużo łatwiejszy do zautomatyzowanego wczytania niż (2).

**Potwierdzone z notatek projektowych** (`terraeye-ops-science-and-innovation/context/other/O011 EGIB/scope/`): zestaw (1) to oryginalny plik dostarczany przez RCN ("tu są wszystkie nieruchomości w jednym: niezabudowane, zabudowane i lokalowe"), a zestaw (2) to ten sam plik ręcznie rozbity na trzy kategorie + arkusz .xlsx do przeglądu — a więc dokładnie pochodna, nie niezależne źródło. Powyższa rekomendacja pokrywa się z tym, jak zespół już rozumie te dane.

## Kontekst zamówienia (dlaczego to ma znaczenie dla modelu)

Z tych samych notatek: to zlecenie to formalne opracowanie **„Analiza zmian struktury agrarnej pod kątem optymalizacji użytkowania gruntów z wykorzystaniem sztucznej inteligencji dla powiatu szczycieńskiego województwa warmińsko-mazurskiego"** — z terminem 5 miesięcy od podpisania umowy. Kluczowe dla architektury modelu:

- **Rozdzielczość hex = 8** jest ustalona kontraktowo (nie tylko domyślną wartością configu `dataset.resolution: r8`) — zgodna z `§1 pkt 3.4` umowy ("przeprowadzenie analiz w układzie siatki heksagonalnej").
- **Zakres geograficzny**: powiat szczycieński (kod `2817`) to właściwy odbiorca analizy ("pow. szczytno do analiz dla odbiorcy"); pozostałe 20 powiatów woj. warmińsko-mazurskiego służą jako szersze dane treningowe/kontekstowe ("woj. w-m do analiz globalnych/na przyszłość — liczymy to, wyciągamy wnioski dla szczytna"). Struktura `rok_YYYY/` obejmująca wszystkie powiaty (już uporządkowana) jest z tym zgodna.
- **Umowa (§2, pkt 3.5–3.6) wprost wymaga DWÓCH ODDZIELNYCH modeli predykcyjnych**: „prawdopodobieństwo odrolnienia" i „prawdopodobieństwo podziału nieruchomości" jako osobne punkty — nie jednego połączonego zdarzenia „podział LUB odrolnienie". To bezpośrednio potwierdza i doprecyzowuje najważniejsze znalezisko z `methodology_audit.md` (sekcja 1): obecny model trenuje się wyłącznie na etykiecie podziału, a etykiety odrolnienia nigdy nie trafiają do datasetu treningowego — to nie tylko niespójność wewnętrzna, to **brakująca, kontraktowo wymagana część dostawy**. Rekomendowane podejście to więc **dwa osobne modele** (nie jeden model na połączone zdarzenie), zgodnie z literą umowy.
- Narracyjny motyw przewodni raportu: lotnisko w Szymanach (leży w powiecie szczycieńskim) — kontekst do interpretacji wyników, nie coś co wpływa na architekturę modelu.

## Zgodność schematu z TerraMerge

Sprawdzone na `2801_transakcje_ceny.gpkg`, warstwa `transakcje_dzialki` (42 971 wierszy w tym jednym powiecie, **1 226 728** wierszy łącznie we wszystkich 21 powiatach):

| Wymaganie TerraMerge | Pole w RCN | Zgodność |
|---|---|---|
| CRS (cały pipeline zakłada EPSG:2180) | `EPSG:2180` (`ETRF2000-PL / CS92`) | ✅ dokładne dopasowanie |
| `iddzialki` (klucz złączenia, format `jednostka_teryt.obreb.nr_dzialki`, np. `280103_2.0066.77` — dokładnie wzorzec z `clean_dataset.py`'s regexu) | `dzi_id_dzialki` = `"280103_2.0066.77"` | ✅ identyczny format, wystarczy zmiana nazwy kolumny |
| `date_column` (domyślnie `"Data"`, parsowane przez `EXTRACT(YEAR FROM try_cast(... AS DATE))`) | `dok_data` (typ DateTime) | ✅ wystarczy ustawić `features.add_transaction_prices.date_column: dok_data` w configu (bez zmiany nazwy) |
| Kolumna ceny — po stronie `add_transactions_hex.py` musi się nazywać **dokładnie `cena`** przed złączeniem (prefiks `tx_` jest doklejany automatycznie przy joinie, więc `cena` → `tx_cena`, zgodnie z `conf/pipeline/make_hexagons.yaml`'s `reduce_map: tx_cena: max` i `treat_zero_as_na: [tx_cena]`) | trzy kandydackie pola: `tran_cena_brutto` (cała transakcja), `dzi_cena_brutto` (cena przypisana do działki — **często `NULL`**, patrz niżej), `nier_cena_brutto` | ⚠️ wymaga decyzji + zmiany nazwy — patrz "Otwarta kwestia" niżej |

## Znaleziony problem jakości danych

`MIN(dok_data)` w powiecie 2801 to `0209-08-17` — ewidentnie błędny rok (prawdopodobnie literówka/błąd wprowadzania danych, "0209" zamiast np. "2009"). `MAX(dok_data)` to `2026-06-12`, co dobrze pokrywa się z zakresem `rok_2013`…`rok_2026`. **Rekomendacja:** przy konwersji odfiltruj rekordy z `dok_data` poza sensownym zakresem (np. `< 1990-01-01` lub `> bieżąca data`), zanim trafią do parqueta — inaczej `tx_year` dla takich wierszy przyjmie absurdalną wartość i wypaczy agregacje rok-po-roku.

## Otwarta kwestia: która kolumna ceny?

Na sprawdzonym przykładzie rekordu (`nier_rodzaj = nieruchomoscLokalowa`, czyli transakcja lokalowa powiązana z działką): `tran_cena_brutto = 160000`, `nier_cena_brutto = 0`, `dzi_cena_brutto = NULL`. To sugeruje, że `dzi_cena_brutto`/`nier_cena_brutto` bywają puste dla transakcji, gdzie działka jest tylko "przy okazji" (sprzedaż lokalu), podczas gdy `tran_cena_brutto` zawsze jest wypełnione (cena całej transakcji). Do potwierdzenia empirycznie (zależy od tego, ile wierszy ma wypełnione `dzi_cena_brutto` w porównaniu z `tran_cena_brutto`), ale rekomendowane podejście:
- Jeśli cel to "cena samej działki" — użyj `dzi_cena_brutto`, z fallbackiem na `tran_cena_brutto` gdy `dzi_cena_brutto IS NULL` i transakcja dotyczy `nier_rodzaj` typu gruntowego (nie lokalowego).
- Jeśli cel to prostszy, zawsze-wypełniony sygnał "ruch cenowy w okolicy działki" — wystarczy `tran_cena_brutto` wprost, bez rozróżniania. Prostsze i mniej podatne na brakujące dane.
- Pole `dzi_sposob_uzyt` (wartości m.in. `gruntyRolne`, `gruntyZabudowaneIZurbanizowane`, `gruntyLesne`, `terenyKomunikacyjne`, `inne`) warto przenieść jako dodatkową cechę — bezpośrednio istotne dla wykrywania "odrolnienia" (sposób użytkowania działki w momencie transakcji).

## Rekomendowany krok konwersji (do zaimplementowania, jeśli zaakceptujesz plan)

Nowy skrypt, np. `src/prepare_data/build_transactions_parquet.py`, wołany raz (lub jako opcjonalny krok w `src/main.py`, analogicznie do istniejących kroków `prepare.*`):

1. Wczytaj warstwę `transakcje_dzialki` z każdego z 21 plików `RCN/NNNN_transakcje_ceny.gpkg` (`geopandas.read_file(path, layer="transakcje_dzialki")`).
2. Odfiltruj rekordy z `dok_data` poza sensownym zakresem dat.
3. Zmień nazwy kolumn: `dzi_id_dzialki` → `iddzialki`; wybierz i ew. przemianuj kolumnę ceny na `cena` zgodnie z decyzją powyżej.
4. Połącz wszystkie 21 powiatów w jeden DataFrame (`pd.concat`).
5. Zapisz jako pojedynczy plik Parquet pod ścieżkę wskazaną w `.env`'s `TERRAMERGE_TRANSACTIONS_PATH` (czyli `data.transactions_path`).
6. Ustaw w configu: `features.add_transaction_prices.date_column: dok_data` (albo przemianuj przy okazji kroku 3 na `Data`, żeby zostać przy domyślnej wartości configu — decyzja stylistyczna, oba działają).

To pasuje do istniejącego wzorca innych kroków `prepare_data` (por. `run_extraction_polygons`) — osobna, jednorazowa funkcja czytająca surowe pliki źródłowe i zapisująca znormalizowany artefakt pośredni, nie zmieniająca niczego w `src/features/add_transaction_prices.py` (ten plik zostaje bez zmian — on i tak oczekuje "gotowego" parquetu).

## Czego NIE robię teraz

Nic z powyższego nie zostało zaimplementowane — to jest wyłącznie plan do Twojej akceptacji. W szczególności nie tknięto: plików w `RCN/`, configu (`conf/*.yaml`), ani `src/features/add_transaction_prices.py` / `add_transactions_hex.py`.
