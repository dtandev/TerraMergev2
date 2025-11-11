# features/geometric_feat_eng.py

import geopandas as gpd
import pandas as pd
import numpy as np

#######################################################################
# 1. Rozmiar i skala --------------------------------------------------
#######################################################################
class SizeScaleFeatures:
    """
    Klasa oblicza podstawowe cechy związane z **rozmiarem i skalą** działek ewidencyjnych.

    Parametry
    ----------
    geometry_column : str, domyślnie "geometry"
        Nazwa kolumny zawierającej geometrie w GeoDataFrame.
    join : bool, domyślnie True
        Jeśli **True** — wynikowe cechy zostaną dołączone do wejściowego GeoDataFrame.
        Jeśli **False** — zwrócony zostanie osobny `pandas.DataFrame` z nowymi kolumnami.

    Wymagania
    ----------
    • GeoDataFrame musi być w **układzie metrycznym** (np. EPSG:2180).  
      Jeśli CRS jest geograficzny (°), metoda zgłosi wyjątek.

    Cechy generowane przez `transform()`
    ------------------------------------
    area_m2 : float
        Powierzchnia działki w metrach kwadratowych.
        **Za co odpowiada?** Wielkość nominalna parceli (podstawa wszystkich wskaźników intensywności zagospodarowania).
        **Zastosowania w analizie strukturalnej**:
        – Hierarchizacja działek wg wielkości (np. mikro-, małe-, średnie-, makro-parcele).  
        – Normalizacja parametrów zabudowy: FAR (floor-area-ratio), GFA, udział biologicznie czynny.  
        – Filtracja anomalii (działki zerowe lub > 99 percentyla) przed dalszą analizą.

    perimeter_m : float
        Długość obwodu działki w metrach.
        **Za co odpowiada?** „Kontakt” działki ze światem zewnętrznym; proxy kosztu ogrodzenia, uzbrojenia.
        **Zastosowania w analizie strukturalnej**:
        – Szacowanie kosztu infrastruktury liniowej (ogrodzenie, chodnik przyfrontowy).  
        – Razem z area_m2 tworzy wskaźniki kompaktowości (Isoperimetric Quotient, Polsby-Popper).  
        – Wykrywanie działek z bardzo rozczłonkowaną granicą (wysoki P przy niskim A).

    log_area : float
        Logarytm naturalny z powierzchni.
        **Za co odpowiada?** Stabilizuje skośny rozkład dużych/małych działek, zbliżając go do normalnego.
        **Zastosowania w analizie strukturalnej**:
        – Poprawia interpretowalność współczynników regresji liniowych (mniej wrażliwe na outliery).  
        – Przygotowuje dane pod PCA / t-SNE, redukując dominację największych parcel.

    log_perimeter : float
        Logarytm naturalny z obwodu.
        **Za co odpowiada?** Ta sama idea co powyżej, lecz dla krawędzi; wzmacnia zależności addytywne.
        **Zastosowania w analizie strukturalnej**:
        – Modele predykcyjne ceny gruntu, w których stroma relacja (długość ogrodzenia vs. koszt) może być nieliniowa.  
        – Łączenie z log_area do tworzenia cech interakcyjnych (np. log_area / log_perimeter).

    mean_width : float
        Średnia szerokość działki obliczana jako `2 * area / perimeter`.
        **Za co odpowiada?** Przybliża „typową” szerokość prostokąta o tym samym obwodzie i powierzchni.
        **Zastosowania w analizie strukturalnej**:
        – Ocena potencjalnej szerokości frontu zabudowy (istotne w planowaniu ulicznym).  
        – Klasyfikacja morfologii na paskowe/narożne kontra zwarte działki.  
        – Używana jako szybki heurystyczny filtr przy szukaniu parcel o minimalnej szerokości wg MPZP.
        **Niższa wartość** ⇒ paskowa parcelacja; **wyższa** ⇒ front szeroki.
    """

    def __init__(self, geometry_column: str = "geometry", join: bool = True):
        self.geometry_column = geometry_column
        self.join = join

    # -------------------------------------------------------------
    # Metody pomocnicze
    # -------------------------------------------------------------
    def _validate_crs(self, gdf: gpd.GeoDataFrame) -> None:
        """Sprawdza, czy GeoDataFrame ma ustawiony metryczny CRS."""
        if gdf.crs is None:
            raise ValueError("GeoDataFrame nie ma ustawionego CRS – przereprojekuj do układu metrów, np. EPSG:2180.")
        if gdf.crs.is_geographic:
            raise ValueError("CRS wygląda na geograficzny (stopnie). Przekształć najpierw do układu metrycznego.")

    # -------------------------------------------------------------
    # API główne
    # -------------------------------------------------------------
    def transform(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame | pd.DataFrame:
        """Oblicza i zwraca zestaw cech.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            Zbiór działek z kolumną geometry.

        Returns
        -------
        geopandas.GeoDataFrame | pandas.DataFrame
            Rozszerzony GeoDataFrame (lub DataFrame) z pięcioma kolumnami cech.
        """
        self._validate_crs(gdf)

        geom = gdf[self.geometry_column]
        area = geom.area
        perimeter = geom.length

        # Uwaga: perimeter == 0 powinien wystąpić tylko przy niepoprawnych geometriach
        perimeter_safe = perimeter.replace(0, np.nan)

        features = pd.DataFrame({
            "area_ha": (area/10_000).round(4),  # powierzchnia w hektarach
            "perimeter_m": perimeter.round(1),
            "log_area": np.log(area),
            "log_perimeter": np.log(perimeter_safe),
            "mean_width": 2 * area / perimeter_safe
        }, index=gdf.index)

        if self.join:
            # zachowujemy CRS i inne atrybuty GeoDataFrame
            return gdf.join(features)
        return features


import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
import math


#######################################################################
# 2. Wydłużenie i orientacja -----------------------------------------
#######################################################################

class ElongationOrientationFeatures:
    """
    Klasa generuje wskaźniki **wydłużenia i orientacji** działek (grupa 2).

    Parametry
    ----------
    geometry_column : str, domyślnie "geometry"
        Nazwa kolumny z geometrią.
    join : bool, domyślnie True
        Czy dołączyć cechy do wejściowego GeoDataFrame.

    Wymagania
    ----------
    • Dane muszą mieć CRS w metrach.

    Cechy generowane przez `transform()`
    ------------------------------------
    elongation_mrr : float
        Stosunek długości dłuższej krawędzi **minimalnego prostokąta obrotowego** (MRR)
        do krawędzi krótszej (\(> 1\)).  
        **Za co odpowiada?** Kwantyfikuje „smukłość” działki niezależnie od jej rotacji.
        **Zastosowania**: klasyfikacja paskowych parcel, filtr działek o zbyt małej szerokości zabudowy,
        predykcja układu dachu/budynku w modelach ML.
        **Większa wartość** ⇒ działka węższa i dłuższa (bardziej paskowa).  
        **Wartość bliska 1** ⇒ kształt zbliżony do kwadratu / koła.

    orientation_deg : float
        Kąt (0–180°) głównej osi MRR względem osi *wschód* (EPSG), mierzony
        **dodatnio przeciwnie do ruchu wskazówek**.  
        **Za co odpowiada?** Kierunek „frontu” działki, przydatny przy analizie ekspozycji
        (insolacja, widok na ulicę).
        **Zastosowania**: klastrowanie działek pod kątem układu ulic, warunków
        nasłonecznienia, planowania sieci PV.

    aspect_ratio_bbox : float
        Proporcja szerokość / wysokość **nierotowanego** bounding‑boxa (BBOX).  
        **Za co odpowiada?** Tańsza obliczeniowo alternatywa elongation_mrr — wrażliwa na orientację.
        **Zastosowania**: szybkie filtrowanie (np. czy działka jest szeroka na front),
        cecha wejściowa do PCA obok elongation_mrr.
        **Im dalej od 1** tym działka bardziej wydłużona, ale wynik zależy od orientacji w układzie CRS.
    """

    def __init__(self, geometry_column: str = "geometry", join: bool = True):
        self.geometry_column = geometry_column
        self.join = join

    # -------------------------------------------------------------
    def _validate_crs(self, gdf: gpd.GeoDataFrame) -> None:
        if gdf.crs is None or gdf.crs.is_geographic:
            raise ValueError("Geometry must be projected (metry). Użyj np. EPSG:2180.")

    @staticmethod
    def _mrr_properties(poly: Polygon) -> tuple[float, float, float]:
        """Zwraca (elongation, orientation_deg, major_len, minor_len) dla pojedynczego poligonu."""
        mrr = poly.minimum_rotated_rectangle
        coords = list(mrr.exterior.coords)[:4]  # 4 narożniki
        # Dwie kolejne krawędzie
        a = math.dist(coords[0], coords[1])
        b = math.dist(coords[1], coords[2])
        major, minor = (a, b) if a >= b else (b, a)
        # Wektor głównej osi: między środkiem krawędzi dłuższej – uproszczamy: użyjemy v = coords[1]-coords[0] jeśli a>=b else coords[2]-coords[1]
        if a >= b:
            vx, vy = coords[1][0] - coords[0][0], coords[1][1] - coords[0][1]
        else:
            vx, vy = coords[2][0] - coords[1][0], coords[2][1] - coords[1][1]
        angle = math.degrees(math.atan2(vy, vx))
        # normalizuj do 0–180
        if angle < 0:
            angle += 180
        return major / minor if minor else np.nan, angle, major, minor

    # -------------------------------------------------------------
    def transform(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame | pd.DataFrame:
        """Oblicza wskaźniki wydłużenia i orientacji."""
        self._validate_crs(gdf)
        elong, orient = [], []
        bbox_ratio = []
        for geom in gdf[self.geometry_column]:
            if geom.is_empty:
                elong.append(np.nan); orient.append(np.nan); bbox_ratio.append(np.nan)
                continue
            e, ang, *_ = self._mrr_properties(geom)
            elong.append(e)
            orient.append(ang)
            minx, miny, maxx, maxy = geom.bounds
            width = maxx - minx
            height = maxy - miny
            bbox_ratio.append(width / height if height else np.nan)
        features = pd.DataFrame({
            "elongation_mrr": elong,
            "orientation_deg": orient,
            "aspect_ratio_bbox": bbox_ratio
        }, index=gdf.index)
        return gdf.join(features) if self.join else features

#######################################################################
# 3. Kompaktowość i okrągłość ----------------------------------------
#######################################################################

class CompactnessCircularityFeatures:
    """
    **Wskaźniki zwartości (kompaktowości) i „okrągłości” działek** – pozwalają ilościowo
    ocenić, na ile obrys parceli jest regularny, prosty do zagospodarowania, a na ile
    „postrzępiony” lub wklęsły.  
    Wszystkie trzy miary zwracają wartości w **zakresie 0 – 1**.

    • **Wartości bliskie 1** → działka zwarta, zbliżona do figury wypukłej (koło / prostokąt).  
    • **Wartości bliskie 0** → działka nieregularna, z licznymi wcięciami, cyplami, „ogonami”.

    ------------------------------------------------------------------
    Cechy obliczane przez `transform()`
    ------------------------------------------------------------------
    ipq : Isoperimetric Quotient = \(\displaystyle \frac{4\pi A}{P^{2}}\)
        *Porównuje działkę do idealnego koła* – które ma **IPQ = 1.0**.
        • **> 0.70** – kształt bardzo zwarty (koło, kwadrat).  
        • **0.40 – 0.70** – typowe prostokąty, trapezy.  
        • **< 0.40** – obrysy rozciągnięte, „wężowe” lub z zatokami.

        *Zastosowania*: identyfikacja działek trudnych do podziału, ocena ryzyka
        wysokiego kosztu infrastruktury liniowej.

    rectangularity : \(\displaystyle \frac{A}{A_{\text{MRR}}}\)
        Stosunek pola działki do pola **minimalnego prostokąta obrotowego** (MRR).
        • **R ≈ 1.0** – działka prawie dokładnie prostokątna.  
        • **R < 0.6** – znaczna część MRR jest „pusta” – sygnał kształtu litery L, pierścieni itp.

        *Zastosowania*: szybka ocena efektywności zabudowy prostokątnej, filtr pod
        podział wtórny (działki "L"‑kształtne często wymagają scalania/przekształceń).

    solidity : \(\displaystyle \frac{A}{A_{\text{convex hull}}}\)
        Porównuje powierzchnię działki do powierzchni jej **otoczki wypukłej**.
        • **Solidity ≈ 1.0** – obrys prawie wypukły (brak wklęśnięć).  
        • **Solidity < 0.5** – silnie wklęsły kształt (np. litera C, pierścień).

        *Zastosowania*: identyfikacja działek, gdzie obrys zawiera "dziury" lub
        wielokrotne narożniki → potencjalnie większe koszty ogrodzenia / dróg wewn.

    ------------------------------------------------------------------
    Parametry konstruktora
    ------------------------------------------------------------------
    geometry_column : str, domyślnie "geometry"
        Nazwa kolumny z geometrią w GeoDataFrame.
    join : bool, domyślnie True
        • **True** – wynikowe cechy zostaną **dołączone** do wejściowego GeoDataFrame.  
        • **False** – metoda zwróci osobny `pandas.DataFrame`.

    ------------------------------------------------------------------
    Wymagania danych
    ------------------------------------------------------------------
    • GeoDataFrame musi być w **metrycznym CRS** (np. EPSG 2180).  
    • Geometrie powinny być **poprawne** (użyj `buffer(0)` lub `make_valid()` jeśli są błędy).

    ------------------------------------------------------------------
    Interpretacja skrócona (TL;DR)
    ------------------------------------------------------------------
    | Cecha | 0 → … | 1 → … |
    |-------|-------|-------|
    | **ipq** | obr. długi / postrzępiony | koło / kwadrat |
    | **rectangularity** | „L”, pierścień | prostokąt |
    | **solidity** | silnie wklęsły | wypukły |
    """

    def __init__(self, geometry_column: str = "geometry", join: bool = True):
        self.geometry_column = geometry_column
        self.join = join

    # -------------------------------------------------------------
    def _validate_crs(self, gdf):
        if gdf.crs is None or gdf.crs.is_geographic:
            raise ValueError("Wymagany metryczny CRS, np. EPSG:2180")

    # -------------------------------------------------------------
    def transform(self, gdf):
        """Oblicza trzy miary kompaktowości / okrągłości opisane w docstringu."""
        self._validate_crs(gdf)
        geom = gdf[self.geometry_column]
        area = geom.area
        perimeter = geom.length.replace(0, np.nan)
        ipq = 4 * math.pi * area / (perimeter ** 2)
        mrr_area = geom.apply(lambda g: g.minimum_rotated_rectangle.area)
        rectangularity = area / mrr_area.replace(0, np.nan)
        solidity = area / geom.convex_hull.area.replace(0, np.nan)
        feat = pd.DataFrame({
            "ipq": ipq,
            "rectangularity": rectangularity,
            "solidity": solidity
        }, index=gdf.index)
        return gdf.join(feat) if self.join else feat



from shapely.geometry import Polygon, MultiPolygon

#######################################################################
# 4. Złożoność krawędzi ----------------------------------------------
#######################################################################
class EdgeComplexityFeatures:
    """
    **Miary złożoności (chropowatości) obrysu działki**.  
    Pozwalają określić, czy granica parceli jest gładka (prosta) czy bardzo
    „postrzępiona” – co często przekłada się na koszty ogrodzenia, długość linii
    brzegowej, trudności z uzbrojeniem terenu.

    ------------------------------------------------------------------
    Cechy obliczane przez `transform()`
    ------------------------------------------------------------------
    complexity_index : \(\displaystyle \text{CI} = \frac{P}{2\sqrt{\pi A}}\)
        • **CI = 1.0** – idealny okrąg (obrys maksymalnie gładki).  
        • **CI > 1.0** – granica coraz bardziej nieregularna; im wyższa wartość,
          tym większa „chropowatość” i długość ogrodzenia przy tej samej powierzchni.

    vertex_density : \(\displaystyle \frac{n_{\text{wierzchołków}}}{P}\)  
        Liczba wierzchołków wielokąta przypadająca na 1 metr obwodu.
        • **Niska wartość** – granica głównie z długich, prostych odcinków.  
        • **Wysoka wartość** – kontur mocno poszarpany (dużo krótkich segmentów).

        *Zastosowania*: identyfikacja działek nadrzecznych (meandry), z
        fragmentacją brzegów, ocena robocizny przy stawianiu ogrodzenia.

    Interpretacja skrócona
    ----------------------
    | Cecha | Gładki obrys | Postrzępiony obrys |
    |-------|--------------|--------------------|
    | **CI** | 1 | > 1 |
    | **vertex_density** | ≈ 0 | ↑ |
    """

    def __init__(self, geometry_column: str = "geometry", join: bool = True):
        self.geometry_column = geometry_column
        self.join = join

    # -----------------------------------------------------------------
    @staticmethod
    def _count_vertices(geom):
        """Zlicza wierzchołki również dla MultiPolygon."""
        if geom is None or geom.is_empty:
            return np.nan
        if isinstance(geom, Polygon):
            return len(geom.exterior.coords)
        if isinstance(geom, MultiPolygon):
            return sum(len(p.exterior.coords) for p in geom.geoms)
        return np.nan  # inne typy pomijamy

    # -----------------------------------------------------------------
    def transform(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame | pd.DataFrame:
        geom = gdf[self.geometry_column]

        area = geom.area.replace(0, np.nan)
        perimeter = geom.length.replace(0, np.nan)

        complexity_index = perimeter / (2 * np.sqrt(np.pi * area))

        vertex_count = geom.apply(self._count_vertices)
        vertex_density = vertex_count / perimeter

        feat = pd.DataFrame(
            {
                "complexity_index": complexity_index,
                "vertex_density": vertex_density,
            },
            index=gdf.index,
        )

        if self.join:
            return gdf.join(feat)
        return feat


#######################################################################
# 5. Momenty geometryczne --------------------------------------------
#######################################################################
class MomentInertiaFeatures:
    """
    **Momenty geometryczne II rzędu** działki – opisują rozkład „masy” (pola)
    wokół środka ciężkości. Przekłada się to na tzw. *inertia ratio*, niezależne
    od rotacji i skal.

    Uproszczone obliczenie oparte na macierzy kowariancji punktów brzegowych.

    ------------------------------------------------------------------
    Cechy obliczane przez `transform()`
    ------------------------------------------------------------------
    inertia_ratio : \(\lambda_{\text{max}} / \lambda_{\text{min}}\)
        Iloraz największej i najmniejszej wartości własnej macierzy inercji.
        • **≈ 1.0** – masa rozłożona izotropowo (zwarte, okrągłe / kwadratowe kształty).  
        • **≫ 1.0** – masa skupiona wzdłuż jednej osi (wydłużone działki).

    inertia_major : float (m²)
        Największa wartość własna – moment bezwładności względem osi minor.

    inertia_minor : float (m²)
        Najmniejsza wartość własna – moment bezwładności względem osi major.

        *Zastosowania*: feature rotacyjnie niezmienniczy w ML (clustering działek
        podobnych „masowo”), diagnostyka smukłości niezależna od kątowego MRR.

    ------------------------------------------------------------------
    Jak rosną / maleją wartości
    ------------------------------------------------------------------
    • **Wzrost inertia_ratio** → działka staje się bardziej wydłużona / anisotropowa.  
    • **Spadek inertia_ratio → ≈ 1** → działka przechodzi w kształt bardziej zwarty.
    """

    def __init__(self, geometry_column: str = "geometry", join: bool = True, sample_points: int = 500):
        self.geometry_column = geometry_column
        self.join = join
        self.sample_points = sample_points

    # -------------------------------------------------------------
    @staticmethod
    def _boundary_segments(geom):
        """Zwraca listę segmentów [(x1,y1,x2,y2), …] dla Polygon/MultiPolygon."""
        segments = []
        if geom.is_empty:
            return segments
        if isinstance(geom, Polygon):
            coords = list(geom.exterior.coords)
            segments += [(coords[i][0], coords[i][1], coords[i+1][0], coords[i+1][1])
                         for i in range(len(coords)-1)]
        elif isinstance(geom, MultiPolygon):
            for poly in geom.geoms:
                coords = list(poly.exterior.coords)
                segments += [(coords[i][0], coords[i][1], coords[i+1][0], coords[i+1][1])
                             for i in range(len(coords)-1)]
        return segments

    # -------------------------------------------------------------
    def _sample_boundary_points(self, geom):
        segs = self._boundary_segments(geom)
        if not segs:
            return np.empty((0, 2))
        lengths = [math.hypot(x2 - x1, y2 - y1) for x1, y1, x2, y2 in segs]
        cum = np.cumsum([0] + lengths)
        total = cum[-1]
        if total == 0:
            return np.empty((0, 2))
        pts = []
        for _ in range(self.sample_points):
            d = np.random.uniform(0, total)
            idx = np.searchsorted(cum, d, side="right") - 1
            x1, y1, x2, y2 = segs[idx]
            seg_len = lengths[idx]
            t = 0 if seg_len == 0 else (d - cum[idx]) / seg_len
            pts.append((x1 + t * (x2 - x1), y1 + t * (y2 - y1)))
        return np.array(pts)

    # -------------------------------------------------------------
    def transform(self, gdf: gpd.GeoDataFrame):
        inertia_ratio_list, major_list, minor_list = [], [], []
        for geom in gdf[self.geometry_column]:
            pts = self._sample_boundary_points(geom)
            if pts.size == 0:
                inertia_ratio_list.append(np.nan)
                major_list.append(np.nan)
                minor_list.append(np.nan)
                continue
            centroid = pts.mean(axis=0)
            centered = pts - centroid
            cov = np.cov(centered.T)
            eigvals = np.linalg.eigvalsh(cov)
            minor, major = eigvals  # ascending order
            inertia_ratio_list.append(major / minor if minor else np.nan)
            major_list.append(major)
            minor_list.append(minor)
        feat = pd.DataFrame({
            "inertia_ratio": inertia_ratio_list,
            "inertia_major": major_list,
            "inertia_minor": minor_list
        }, index=gdf.index)
        return gdf.join(feat) if self.join else feat


#######################################################################
# 6. Otoczenie brzegu -------------------------------------------------
#######################################################################
class EdgeContextFeatures:
    """
    **Miary „ukrytej” lub wklęsłej części granicy działki** – określają, ile jej
    obwodu i pola „znika” w zagłębieniach względem wypukłej otoczki.

    ------------------------------------------------------------------
    Cechy obliczane przez `transform()`
    ------------------------------------------------------------------
    convexity_deficit_len : float (m)
        Różnica długości: `P_convex_hull − P`.
        • **≈ 0 m** → granica prawie wypukła.  
        • **≫ 0 m** → duża porcja obwodu „ukryta” w zatokach; rosną koszty
          uzbrojenia i długość dróg wewn.

    convexity_deficit_area : float (m²)
        Różnica powierzchni: `A_convex_hull − A`.
        • **≈ 0 m²** → obrys niemal wypukły.  
        • **Wysoka wartość** → duże wcięcia / pierścieniowy kształt.

    convexity_deficit_ratio : float (0–1)
        `convexity_deficit_area / A_convex_hull` – normalizuje miarę do skali 0–1.
        • **0** → brak wklęsłości.  
        • **→ 1** → kształt silnie wklęsły / „pierścień”, znacząca część
          otoczki wypukłej jest pusta.

    ------------------------------------------------------------------
    Interpretacja (TL;DR)
    ------------------------------------------------------------------
    | Cecha | Obrys wypukły | Obrys silnie wklęsły |
    |-------|---------------|----------------------|
    | **convexity_deficit_len** | 0 m | ↑ |
    | **convexity_deficit_area** | 0 m² | ↑ |
    | **convexity_deficit_ratio** | 0 | → 1 |

    Wysokie wartości sugerują, że działka może wymagać **scalania** lub **podziału**
    dla efektywnej zabudowy oraz że koszty linii brzegowych (ogrodzenie, media)
    będą ponadprzeciętne względem powierzchni.
    """

    def __init__(self, geometry_column: str = "geometry", join: bool = True):
        self.geometry_column = geometry_column
        self.join = join

    # -------------------------------------------------------------
    def _validate_crs(self, gdf):
        if gdf.crs is None or gdf.crs.is_geographic:
            raise ValueError("Wymagany metryczny CRS, np. EPSG:2180")

    # -------------------------------------------------------------
    def transform(self, gdf):
        """Oblicza trzy miary ikonweksji dla każdej działki."""
        self._validate_crs(gdf)
        geom = gdf[self.geometry_column]
        perimeter = geom.length
        hull_perim = geom.convex_hull.length
        area = geom.area
        hull_area = geom.convex_hull.area
        deficit_len = hull_perim - perimeter
        deficit_area = hull_area - area
        ratio = deficit_area / hull_area.replace(0, np.nan)
        feat = pd.DataFrame({
            "convexity_deficit_len": deficit_len,
            "convexity_deficit_area": deficit_area,
            "convexity_deficit_ratio": ratio
        }, index=gdf.index)
        return gdf.join(feat) if self.join else feat

    __call__ = transform

#######################################################################
# 7. Ekstremalne minimalne otoczki -----------------------------------
#######################################################################
class ExtremeEnvelopeFeatures:
    """
    **Miary „skrajności” rozciągnięcia działki** – oparte na minimalnym okręgu
    otaczającym i jego relacji do powierzchni parceli.

    ------------------------------------------------------------------
    Cechy obliczane przez `transform()`
    ------------------------------------------------------------------
    min_circle_radius : float (m)
        Promień **minimalnego okręgu otaczającego** (aproksymowany jako maksymalna
        odległość wierzchołka od centroidu).  
        • **Mniejszy promień** → działka kompaktowa.  
        • **Większy promień** → rozciągnięta / długa.

    area_to_circle_ratio : float (0–1)
        `A / (π × r²)` – jak dużą część minimalnego okręgu wypełnia działka.  
        • **≈ 1** → kształt bliski koła / kwadratu (wysoka efektywność pola).  
        • **→ 0** → działka wąska, o małej powierzchni względem zasięgu.

    circle_area_excess : float (m²)
        `(π × r²) − A` – „nadmiar” powierzchni okręgu względem działki.  
        • **Niska** → działka zwarta.  
        • **Wysoka** → rozciągnięta, potencjalnie problematyczna przy zagospodar.

    ------------------------------------------------------------------
    Jak interpretować wartości
    ------------------------------------------------------------------
    | Cecha | Zwarta działka | Długa / nieregularna |
    |-------|---------------|----------------------|
    | **min_circle_radius** | ↓ | ↑ |
    | **area_to_circle_ratio** | → 1 | → 0 |
    | **circle_area_excess** | 0 | ↑ |

    Wysokie `min_circle_radius` i **niskie** `area_to_circle_ratio` sygnalizują
    działki *bardzo* wydłużone lub z dużymi „ogonkami” – co może ograniczyć
    możliwość budowy, wymusić podział lub wpłynąć na wartość rynkową.
    """

    def __init__(self, geometry_column: str = "geometry", join: bool = True):
        self.geometry_column = geometry_column
        self.join = join

    # -------------------------------------------------------------
    def _validate_crs(self, gdf):
        if gdf.crs is None or gdf.crs.is_geographic:
            raise ValueError("Wymagany metryczny CRS, np. EPSG:2180")

    # -------------------------------------------------------------
    @staticmethod
    def _max_radius(geom):
        """Zwraca maksymalną odległość dowolnego wierzchołka od centroidu.
        Obsługuje Polygon i MultiPolygon."""
        if geom.is_empty:
            return np.nan
        centroid = geom.centroid
        cx, cy = centroid.x, centroid.y
        max_r = 0.0
        if isinstance(geom, Polygon):
            rings = [geom.exterior.coords]
        elif isinstance(geom, MultiPolygon):
            rings = [poly.exterior.coords for poly in geom.geoms]
        else:
            return np.nan
        for ring in rings:
            for x, y in ring:
                r = math.hypot(x - cx, y - cy)
                if r > max_r:
                    max_r = r
        return max_r

    # -------------------------------------------------------------
    def transform(self, gdf: gpd.GeoDataFrame):
        """Oblicza promień minimalnego okręgu otaczającego i pochodne miary."""
        self._validate_crs(gdf)
        radii = gdf[self.geometry_column].apply(self._max_radius)
        circle_area = np.pi * radii ** 2
        area = gdf[self.geometry_column].area
        ratio = area / circle_area.replace(0, np.nan)
        excess = circle_area - area
        feat = pd.DataFrame({
            "min_circle_radius": radii,
            "area_to_circle_ratio": ratio,
            "circle_area_excess": excess
        }, index=gdf.index)
        return gdf.join(feat) if self.join else feat
