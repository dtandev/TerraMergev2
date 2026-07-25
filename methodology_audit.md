# Audyt metodologiczny — model predykcji podziału / odrolnienia działki w heksagonie

Data: 2026-07-25
Zakres: `src/modeling/labeling.py`, `src/modeling/neighborhood.py`, `src/modeling/train_model.py`, `conf/dataset/default.yaml`, `conf/model/default.yaml`.

## Cel zdefiniowany przez użytkownika

> Określenie prawdopodobieństwa, że dojdzie do podziału i/lub odrolnienia działki w danym hexagonie.

**Doprecyzowane kontraktowo** (notatki zamówienia O011 EGIB, `terraeye-ops-science-and-innovation/context/other/O011 EGIB/scope/`): umowa wymaga dwóch **oddzielnych** wyników — `P(odrolnienie w hexagonie h)` i `P(podział nieruchomości w hexagonie h)` jako osobne modele/dostawy (§2 pkt 3.5–3.6), nie jednego połączonego `P(split ∨ odrolnienie)`. Poniższy audyt uwzględnia to doprecyzowanie tam, gdzie ma znaczenie (patrz punkt 1).

---

## 1. Krytyczna luka: model uczy się tylko na etykiecie podziału, nie na "podział LUB odrolnienie"

To najważniejsze znalezisko tego audytu i warto je rozstrzygnąć zanim zajmiemy się czymkolwiek innym.

- `labeling.py::build_split_labels_full` buduje etykiety **podziału działki** (`split_proxy`, `y_next`) i zapisuje je do `labels.ParcelLabels_r8`.
- `labeling.py::build_uzg_conversion_labels` buduje **niezależnie** etykiety **odrolnienia** (`convert_proxy`, `y_next`) i zapisuje je do `labels.kugLabels_r8_<klasa>` (osobna tabela na każdą klasę z `agri_classes`).
- `neighborhood.py::run_compute_neighbor_aggregates` dołącza etykietę do finalnego datasetu przez `dataset.calculate_neighborhood.table_with_labels` — a ten klucz w `conf/dataset/default.yaml` wskazuje **wyłącznie** na `labels.ParcelLabels_${dataset.resolution}` (potwierdzone empirycznie: `dataset.calculate_neighborhood.table_with_labels` → `'labels.ParcelLabels_r8'`).
- `train_model.py::run_training` trenuje na `model.dataset_table = "dataset.Parcels_neighborhood_r8"` z `model.y_label = "y_next"` — czyli na etykiecie, która przez cały łańcuch pochodzi wyłącznie z **podziału**.

**Wniosek:** dziś model nie przewiduje "podziału LUB odrolnienia" — przewiduje wyłącznie podział. Etykiety odrolnienia (`kugLabels`) są liczone, ale nigdy nie trafiają do danych treningowych. Jeśli cel biznesowy naprawdę obejmuje oba zjawiska, to jest to brakujący element, nie kwestia doboru algorytmu.

**Potwierdzone kontraktowo, nie tylko domysłem:** notatki projektowe zamówienia O011 EGIB (`terraeye-ops-science-and-innovation/context/other/O011 EGIB/scope/`) pokazują, że umowa (§2, pkt 3.5–3.6) wprost wymaga **dwóch oddzielnych modeli predykcyjnych** — „prawdopodobieństwo odrolnienia" i „prawdopodobieństwo podziału nieruchomości" jako osobne punkty dostawy, nie jednego połączonego zdarzenia. To rozstrzyga poniższą rekomendację: brak modelu odrolnienia to nie kwestia stylu, to brakująca, kontraktowo wymagana część opracowania.

**Rekomendacja:** zbuduj **dwa osobne modele** (nie jeden na połączone zdarzenie `split_proxy ∨ convert_proxy`) — po jednym na każde zjawisko, zgodnie z literą umowy. Model odrolnienia analogicznie do istniejącego modelu podziału: dociągnij `labels.kugLabels_r8_<klasa>` do datasetu treningowego (dziś pomijane przez `dataset.calculate_neighborhood.table_with_labels`, które wskazuje tylko na `ParcelLabels`) i wytrenuj osobny `train_model.py`-owy przebieg z `y_label` wskazującym na etykietę odrolnienia zamiast podziału.

---

## 2. Ramowanie horyzontu czasowego nie odpowiada na zadane pytanie

`y_next`, `y_next_2`, `y_next_3`, `y_next_<extra_horizon>` w obu funkcjach etykietujących liczą **"czy proxy jest aktywne dokładnie w roku t+h"** (przez `shift(-h)`), nie **"czy zdarzenie wystąpiło do roku t+h włącznie"** (skumulowane prawdopodobieństwo/cumulative incidence).

To rozróżnienie ma znaczenie praktyczne: pytanie użytkownika ("jakie jest prawdopodobieństwo, że dojdzie do podziału w danym hexagonie") brzmi jak pytanie o `P(zdarzenie w ciągu najbliższych K lat)`, a obecny model (trenowany na `y_label: "y_next"`, czyli horyzoncie = dokładnie 1 rok) odpowiada tylko na "czy dokładnie za rok".

**Rekomendacja — alternatywne podejście do przetestowania:** dyskretno-czasowy model hazardu (discrete-time survival / hazard model):
- Modeluj `h(t | h, X) = P(zdarzenie w roku t | brak zdarzenia do t-1, cechy X)` jako osobny klasyfikator binarny per rok ekspozycji (technika: "person-period" / "person-year" data expansion — każdy hex-rok bez zdarzenia to jedna obserwacja, z etykietą 1 w roku zdarzenia, 0 wcześniej).
- Skumulowane prawdopodobieństwo do horyzontu K: `P(zdarzenie do t+K) = 1 - ∏_{k=1}^{K}(1 - h(t+k))`.
- Dla dwóch konkurencyjnych zdarzeń (split vs. odrolnienie) — cause-specific hazards (osobny hazard na każdy typ zdarzenia) albo model Fine-Gray dla ryzyk konkurencyjnych (competing risks), jeśli zależy Ci na poprawnym oszacowaniu przy obecności konkurencyjnego zdarzenia "wygaszającego" ryzyko drugiego.
- Zaleta: bezpośrednio odpowiada na pytanie z briefu, naturalnie obsługuje ucinanie (censoring — hexagony, które jeszcze nie "wydarzyły" zdarzenia w dostępnych danych) i pozwala raportować `P(zdarzenie w ciągu 3 lat)` zamiast trzech osobnych, niezależnie wytrenowanych klasyfikatorów `y_next_2`/`y_next_3` (które dziś nawet nie są używane w treningu — liczone w `labeling.py`, ale `train_model.py` czyta tylko `y_label: y_next`).

---

## 3. Jakość etykiet proxy — reguły z ręcznie dobranymi progami

Zarówno `split_proxy`, jak i `convert_proxy` to **heurystyki regułowe** na zagregowanych statystykach hexagonu, nie potwierdzone zdarzenia administracyjne:

- `split_proxy`: spadek średniej powierzchni działki (`delta_mean_area < -eps_abs`) przy zachowanej sumie powierzchni (`area_conservation_ok`) i wzroście liczby działek (`delta_count > 0`).
- `convert_proxy`: spadek udziału klasy rolnej R, wzrost udziału klasy B, przy zachowanej sumie udziałów.

Progi (`eps_abs=100.0`, `area_conservation_tol`) są ustawione ręcznie w configu, bez analizy wrażliwości ani walidacji względem jakiegokolwiek realnego źródła prawdy (np. rzeczywistych decyzji podziałowych/odrolnieniowych, jeśli takie dane są gdziekolwiek dostępne, choćby dla próbki).

**Rekomendacja:** (a) analiza wrażliwości — sprawdź, jak bardzo zmienia się liczba wykrytych zdarzeń przy ±20% zmianie progów; jeśli wynik jest bardzo czuły na próg, to sygnał, że reguła nie jest solidna. (b) jeśli macie dostęp do jakiejkolwiek próbki potwierdzonych podziałów/odrolnień (nawet kilkudziesięciu przypadków z rejestru), zweryfikujcie na niej precyzję/recall reguły proxy, zanim potraktujecie ją jako ground truth do trenowania modelu — dziś model uczy się odgadywać regułę, nie zjawisko.

Dodatkowo: w ramach tego audytu naprawiłem błąd w obu funkcjach etykietujących, w którym `shift()` liczył deltę między **kolejnymi wierszami**, a nie między **kolejnymi latami** — jeśli dla danego hexagonu brakowało roku w danych (np. 2022 i 2024 bez 2023), delta była liczona tak, jakby to była zmiana jednoroczna. Dodałem warunek `year(t) - year(t-1) == 1` gating na `split_rule_core`/`convert_rule_core` (patrz `src/modeling/labeling.py`, `valid_transition`) i test regresyjny w `tests/test_labeling.py`. To nie zmienia charakteru problemu z progami opisanego wyżej, ale usuwa jedno konkretne źródło szumu w etykietach.

---

## 4. Metodologia ewaluacji: brak prawdziwego zbioru testowego (double-dipping)

`train_model.py::run_training` używa **tego samego** zbioru walidacyjnego (`valid_years`) do trzech różnych celów:
1. Early stopping w LightGBM (`eval_set=[(X_val, y_val)]`).
2. Wszystkich raportowanych metryk (AP, ROC-AUC, Brier) w `evaluate_lgbm_on_validation`.
3. Dopasowania kalibracji izotonicznej (`IsotonicRegression.fit_transform(p_raw, y_val)`), która potem skaluje predykcje na dane 2025 (`p_hat_cal`).

Nie ma żadnego trzeciego, w pełni odseparowanego zbioru testowego. Efekt: raportowane AP/ROC/Brier są **optymistycznie obciążone** — model "widział" ten zbiór podczas doboru liczby iteracji (early stopping), więc jego metryki na nim są lepsze niż byłyby na naprawdę nowych danych.

**Rekomendacja:** rolling-origin / walk-forward cross-validation zamiast pojedynczego podziału czasowego — trenuj na latach ≤Y, waliduj (early stopping) na Y+1, testuj (raportowane metryki) na Y+2, przesuwaj okno. To też lepiej odzwierciedla docelowy sposób użycia modelu (coroczny retrening + predykcja na kolejny rok) niż jednorazowy podział 2022/2023-2024.

---

## 5. Generalizacja przestrzenna nigdy nie jest testowana

`prepare_Xy_for_lgb`'s `prefer_cats` domyślnie zawiera `jednostka_mode`, `obreb_mode`, `powiat`, `dominant_class` — czyli identyfikatory jednostek administracyjnych. Są to cechy kategoryczne o niskiej kardynalności, więc łatwo trafiają do modelu (`nunique <= cat_cardinality_max`).

Ryzyko: model może się nauczyć "ten konkretny obręb/gmina ma wysokie ryzyko" zamiast uczyć się przenaszalnych predyktorów geometrycznych/land-use. Ponieważ split czasowy (train ≤2022 / valid 2023-2024) **nie** wyklucza tych samych hexagonów/obrębów z obu zbiorów, model dostaje "darmowy" sygnał lokalizacyjny, który nie musi się przenosić na nowe, niewidziane wcześniej obszary.

**Rekomendacja:** dodaj drugi wariant ewaluacji — spatial holdout (leave-one-region-out: odetnij całe gminy/obręby z treningu, testuj tylko na nich) — obok istniejącego podziału czasowego. Warto też zrobić ablację z/bez `jednostka`/`obreb`/`powiat`, żeby zobaczyć, ile z AP tłumaczy się samą lokalizacją.

**Ten test ma tu konkretne, praktyczne zastosowanie, nie tylko teoretyczne:** z kontekstu zamówienia (O011 EGIB) wynika, że właściwym odbiorcą analizy jest powiat szczycieński (kod `2817`), a pozostałych 20 powiatów woj. warmińsko-mazurskiego dostarczono jako szersze dane treningowe/kontekstowe. To naturalny, bezpośrednio odpowiadający realnemu użyciu spatial holdout: wytrenuj na pozostałych 20 powiatach, testuj wyłącznie na `2817_szczycienski` — to pokaże **dokładnie** to, co ma znaczenie dla odbiorcy: czy model faktycznie generalizuje na powiat, dla którego opracowanie jest tworzone, czy tylko pamięta lokalizacje, które już widział.

---

## 6. Niewykorzystana obsługa niezbalansowania klas

Dziś jedyna obsługa niezbalansowania to `scale_pos_weight` (LightGBM). `imbalanced-learn` jest zadeklarowaną zależnością projektu (`environment.yml`, migrowane teraz do `pyproject.toml`), ale nigdzie w kodzie nieużywane.

**Rekomendacja (tania do przetestowania):** porównaj obecne `scale_pos_weight` z (a) SMOTE/ADASYN na zbiorze treningowym, (b) samym undersamplingiem klasy większościowej — na tym samym podziale walidacyjnym, żeby zobaczyć, czy w ogóle zmienia to AP/kalibrację. Jeśli różnica jest znikoma, to dobra wiadomość — nie trzeba tego utrzymywać jako zależności.

---

## Podsumowanie — priorytetowa lista działań

1. **Zbuduj drugi model — odrolnienie** (punkt 1): kontraktowo wymagane jako osobna dostawa (§2 pkt 3.5 umowy O011 EGIB), dziś w ogóle nieobecne w danych treningowych mimo że etykiety już są liczone.
2. **Przetestuj discrete-time hazard/survival** (punkt 2) jako alternatywę dla obecnego "dokładnie za rok" — to jedyna zmiana, która wprost odpowiada na pytanie z briefu o prawdopodobieństwo w horyzoncie czasowym.
3. **Rolling-origin CV zamiast pojedynczego podziału** (punkt 4) — usuwa optymistyczne obciążenie metryk, tanie do wdrożenia bez zmiany architektury modelu.
4. **Spatial holdout: trenuj na 20 powiatach, testuj na `2817_szczycienski`** (punkt 5) — bezpośrednio sprawdza to, co ma znaczenie dla odbiorcy zamówienia.
5. **Walidacja progów proxy** (punkt 3) — analiza wrażliwości, ewentualnie porównanie z realnymi danymi jeśli dostępne.
6. **Porównanie obsługi niezbalansowania** (punkt 6) — szybki, tani eksperyment poboczny.

Punkty 1-2 mają największy wpływ na to, czy model w ogóle odpowiada na zadane pytanie i spełnia wymogi umowy; punkty 3-6 poprawiają wiarygodność tego, co już jest mierzone.
