"""Extracts polygon layers from EGiB deliveries into Parquet, trying multiple source formats.

Per `rok_YYYY` directory, for each administrative unit (jednostka ewidencyjna) found, tries
readers in order GDB -> GML -> SHP -> SWDE, using whichever format is actually present and stopping
at the first that yields usable polygon layers. See `src/prepare_data/readers/` for the
format-specific readers and the plan this was built from for the investigation behind this design.
"""

from __future__ import annotations

import re
from pathlib import Path

import geopandas as gpd
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from osgeo import ogr
from pyproj import CRS
from tqdm import tqdm

from src.common.duckdb_utils import write_geoparquet
from src.prepare_data.readers import gdb_reader, gml_reader, shp_reader, swde_reader

# --- Reguły/stałe domenowe (na później łatwo przenieść do YAML, jeśli zechcesz) ---
YEAR_PATTERN = re.compile(r"^rok_(\d{4})$")
# Ogólny wzorzec jednostki ewidencyjnej TERYT: 6 cyfr + '_' + 1 cyfra (np. 281701_1).
# Był wcześniej zawężony do jednego powiatu (2815, ostródzki) — poprawione, bo łamało to
# wykrywanie jednostek w każdym innym powiecie.
# Otoczone strażnikami (?<!\d)/(?!\d): bez nich dopasowuje też fałszywie do dłuższych ciągów cyfr,
# np. folder "20170303_2802_urzad_marszalkowski" (znacznik daty 2017-03-03 + kod powiatu 2802)
# łapał się jako rzekoma jednostka "170303_2" — potwierdzone na realnych danych (rok_2017,
# powiat 2802 braniewski). Prawdziwe kody jednostek nigdy nie są osadzone w dłuższym ciągu cyfr.
#
# Drugi wariant (bez podkreślnika, np. "2814014.GML") — potwierdzony na realnych danych w co
# najmniej 8 powiatach (2802, 2808, 2810, 2814, 2815, 2818, 2819, 2862): niektóre dostawy zapisują
# ten sam kod TERYT jako 7 cyfr bez separatora zamiast 6+"_"+1. Ostatnia cyfra (rodzaj gminy) w
# systemie TERYT może być tylko 1-5, co ogranicza fałszywe dopasowania do przypadkowych 7-cyfrowych
# ciągów. Bez tego wariantu ekstrakcja milcząco pomijała całe dostawy (np. powiat olsztyński
# 2023-2026), zgłaszając brak danych, mimo że dane istniały — po prostu w tym wariancie nazewnictwa.
#
# Ten wariant jest jednak dużo mniej selektywny niż 6+"_"+1 (brak separatora = tylko struktura cyfr
# odróżnia go od przypadkowej liczby), więc dodatkowo wymagamy prefiksu województwa "28"
# (warmińsko-mazurskie) — potwierdzone, że KAŻDY plik w tym katalogu danych należy do tego
# województwa (nazwa katalogu "w-m egib"), więc to nie jest zawężanie do jednego powiatu (jak
# poprzedni bug), tylko do faktycznego, potwierdzonego zakresu całego zbioru danych. Bez tego
# ograniczenia identyfikatory z zupełnie innych warstw danych (np. BDOT/GESUT — Baza Danych
# Obiektów Topograficznych / sieci uzbrojenia terenu, dołączone do tej samej dostawy GML co EGiB)
# fałszywie dopasowywały się jako rzekome jednostki — potwierdzone na realnych danych (rok_2021,
# powiat 2804 elbląski: pliki "BDOT_4000003.gml", "GESUT_4000004.gml"). Wariant z podkreślnikiem
# zostaje ogólny (bez tego ograniczenia), bo separator już wystarczająco odróżnia go od szumu.
UNIT_PATTERN = re.compile(r"(?<!\d)(?:(\d{6})_(\d)|(28\d{2})(\d{2})([1-5]))(?!\d)")


def _unit_code_from_match(m: re.Match[str]) -> str:
    """Normalize either UNIT_PATTERN branch (with/without '_') to canonical 'NNNNNN_N'."""
    if m.group(1) is not None:
        return f"{m.group(1)}_{m.group(2)}"
    return f"{m.group(3)}{m.group(4)}_{m.group(5)}"


RENAME_2020_TO_2024: dict[str, str] = {
    "G5IDD": "idDzialki",
    "g5idd": "idDzialki",  # <- dodane: legacy bywało też małymi literami
    "SHAPE_Length": "Shape_Length",
    "SHAPE_Leng": "Shape_Length",  # SHP/DBF obcina nazwy pól do 10 znaków
    "SHAPE_Area": "Shape_Area",
    "geometry": "geometry",  # no-op, zostawione dla kompletności
}


# --- tylko poprawiona funkcja ---
def _to_uppercase_columns(
    gdf: gpd.GeoDataFrame,
    *,
    uppercase_geometry: bool = True,
) -> tuple[gpd.GeoDataFrame, dict[str, str]]:
    """UPPERCASE nazw kolumn, zachowując aktywność kolumny geometry. Kolizje → __2, __3, ..."""
    # Kluczowa zmiana: zawsze bierz aktywną nazwę kolumny geometrii; nie sprawdzaj jej obecności w columns,
    # bo po rename() mogłaby "zniknąć" logicznie i stracilibyśmy aktywną geometrię.
    geom_name: str | None = gdf.geometry.name

    seen: set[str] = set()
    rename_map: dict[str, str] = {}

    for col in gdf.columns:
        target = col.upper()
        if not uppercase_geometry and geom_name is not None and col == geom_name:
            target = col  # zachowaj oryginalną nazwę kolumny geometrii

        if target in seen and target != col:
            base = target
            k = 2
            while f"{base}__{k}" in seen:
                k += 1
            target = f"{base}__{k}"

        seen.add(target)
        if target != col:
            rename_map[col] = target

    if rename_map:
        gdf = gdf.rename(columns=rename_map)

    if geom_name and uppercase_geometry:
        new_geom = rename_map.get(geom_name, geom_name)
        if new_geom != geom_name:
            gdf = gdf.set_geometry(new_geom)

    return gdf, rename_map


def _normalize_crs(
    gdf: gpd.GeoDataFrame, *, naive_crs_fallback: str | None, target_crs: str | None
) -> gpd.GeoDataFrame:
    """Bring one layer to a single target CRS (default EPSG:2180 / PUWG 1992).

    The source deliveries are in the CS2000 system (zones EPSG:2176-2179; szczycieński = 2178).
    Formats carry it differently — SWDE has no CRS at all, some GML/SHP deliveries also come in
    "naive" (no CRS tag), and GDB uses an ESRI-flavoured WKT (ESRI:102176) whose to_epsg() is
    None. So: a naive layer is assumed to be in the region's CS2000 zone (naive_crs_fallback,
    from prepare.swde_crs), and then every layer is reprojected to the target. Normalising here,
    once, means every parquet and every downstream DuckDB table shares one CRS — no consumer has
    to reproject, and no later step crashes on a naive geometry.
    """
    if gdf.crs is None and naive_crs_fallback:
        gdf = gdf.set_crs(naive_crs_fallback, allow_override=True)
    if target_crs and gdf.crs is not None:
        target = CRS.from_user_input(target_crs)
        if not gdf.crs.equals(target):
            gdf = gdf.to_crs(target)
    return gdf


def _export_layer_to_parquet(
    gdf: gpd.GeoDataFrame,
    out_path: Path,
    *,
    year: int | None = None,
    uppercase_geometry: bool = True,
    naive_crs_fallback: str | None = None,
    target_crs: str | None = "EPSG:2180",
) -> None:
    """Zapis pojedynczej (już wczytanej) warstwy do Parquet + UPPERCASE kolumn + legacy rename."""
    try:
        gdf = _normalize_crs(gdf, naive_crs_fallback=naive_crs_fallback, target_crs=target_crs)
        # Apply the legacy→modern column rename unconditionally, not just for year<2021: real GDB
        # deliveries keep the legacy names (G5IDD → idDzialki) even in recent years (confirmed on
        # rok_2022 szczycieński), so the year gate left DzialkaEwidencyjna without an `iddzialki`
        # column and every downstream step keyed on it failed. The map only renames columns that
        # exist, so a genuinely modern delivery (already `idDzialki`) is untouched.
        gdf = gdf.rename(columns=RENAME_2020_TO_2024)

        gdf, rename_map = _to_uppercase_columns(gdf, uppercase_geometry=uppercase_geometry)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_geoparquet(gdf, out_path)

        if rename_map:
            logger.info("Zapisano {} ({} zmienionych nazw kolumn)", out_path, len(rename_map))
        else:
            logger.info("Zapisano {}", out_path)
    except Exception:
        logger.exception("Błąd eksportu warstwy → {}", out_path)


# --------------------------------------------------------------------------------------
# Whole-powiat combined deliveries: some real deliveries (confirmed: elbląski, olsztyński,
# nidzicki, działdowski, nowomiejski, piski, gołdapski in the "Urządzeniowo-rolne" dataset,
# rok_2021-2026) put every gmina's rows into ONE .gdb/.gml file instead of one file per gmina --
# the file/containing-folder name carries only the POWIAT code, so UNIT_PATTERN never matches it
# and it was previously invisible to _discover_units entirely.
#
# No physical per-gmina split is needed to make this data usable: every row already carries its
# own gmina code as a prefix on its identifier (confirmed: "iddzialki" like "281401_5.0005.267/1"
# on DzialkaEwidencyjna, "iduzytku" like "281401_5.0005.UG.806" on KonturUzytkuGruntowego) --
# downstream `clean_dataset.py` already re-derives the "jednostka" column from exactly this
# prefix (see its `id_pattern` regex), not from the folder name. So a combined file is extracted
# as-is, as ONE parquet per layer/year like any normal unit, just filed under a pseudo unit code
# ("<powiat>_wspolny") that is never a real TERYT gmina code (real ones always end "_1".."_5",
# this one doesn't) -- everything downstream keeps working unmodified.
_VALID_UNIT_CODE = re.compile(r"^\d{6}_[1-5]$")


def _jew_layer_name(ds: ogr.DataSource) -> str | None:
    """Find a JednostkaEwidencyjna-like layer name, across all 3 known naming conventions."""
    for i in range(ds.GetLayerCount()):
        name = ds.GetLayerByIndex(i).GetName()
        if name == "G5G_JEW" or "jednostkaewidencyjna" in name.lower():
            return name
    return None


def _combined_powiat_codes(path: Path) -> set[str] | None:
    """
    If `path` is a genuine whole-powiat combined delivery (its JednostkaEwidencyjna-like layer
    lists 2+ distinct valid gmina codes, ALL belonging to the SAME powiat), return that set of
    gmina codes. Returns None for anything else:
    - a JEW layer with only 0 or 1 codes is either a normal single-gmina file (already handled
      via UNIT_PATTERN, doesn't need this path) or an unrelated stray/decoy file;
    - a JEW layer whose codes span MULTIPLE powiats is a whole-voivodeship internal
      integration/staging artifact (confirmed real case: files under "INTEGRACJA_BAZ_DANYCH_*"
      folders, e.g. "Warminsko_Mazurskie_na_ATLAS_2021.gdb" with 150 codes across all 21
      powiats) -- not a genuine per-powiat delivery, and processing it would duplicate every
      gmina's data alongside its real delivery. Never guessed as combined, to avoid
      misclassifying noise or double-processing staging files.
    """
    try:
        ds = ogr.Open(str(path), 0)
    except Exception:
        return None
    if ds is None:
        return None

    layer_name = _jew_layer_name(ds)
    if layer_name is None:
        return None

    try:
        gdf = gpd.read_file(str(path), layer=layer_name)
    except Exception:
        logger.debug("Nie udało się odczytać warstwy jednostki z {} (pominięto)", path)
        return None

    id_col = next((c for c in gdf.columns if c.lower() in ("g5idj", "idjednostkiewid")), None)
    if id_col is None:
        return None

    codes = {str(v) for v in gdf[id_col].dropna().unique() if _VALID_UNIT_CODE.match(str(v))}
    if len(codes) < 2:
        return None
    powiat_prefixes = {c[:4] for c in codes}
    if len(powiat_prefixes) != 1:
        logger.debug(
            "Pominięto {} — {} kodów jednostek obejmuje {} różnych powiatów, "
            "to plik integracyjny całego zbioru, nie dostawa jednego powiatu.",
            path.name,
            len(codes),
            len(powiat_prefixes),
        )
        return None
    return codes


def _combined_powiat_prefix(path: Path) -> str | None:
    """4-digit powiat TERYT prefix for a genuine combined delivery, or None -- see
    `_combined_powiat_codes`."""
    codes = _combined_powiat_codes(path)
    return next(iter(codes))[:4] if codes else None


# --------------------------------------------------------------------------------------
# Discovery: for a rok_YYYY dir, find every unit and which formats it's available in.
# --------------------------------------------------------------------------------------


def _discover_units(year_dir: Path) -> dict[str, dict[str, list[Path]]]:
    """
    Returns {unit_code: {"gdb": [path], "gml": [path], "shp": [path, ...], "swde": [path]}}.

    Only formats actually found are present as keys for a given unit. "gdb"/"gml"/"swde" lists
    hold one container path each (per unit, per format found — normally exactly one, but a real
    delivery occasionally has a stray duplicate, see audit.md); "shp" holds every recognized `.shp`
    file for that unit, which may be spread across more than one folder.
    """
    units: dict[str, dict[str, list[Path]]] = {}

    def _unit_of(path: Path) -> str | None:
        m = UNIT_PATTERN.search(path.name) or UNIT_PATTERN.search(str(path))
        return _unit_code_from_match(m) if m else None

    # Pass 1: register every normally-named per-gmina file, across ALL formats (gdb/gml/shp/
    # swde), before looking at any combined-delivery candidate. This must fully complete first --
    # a gmina already covered via, say, a normal .gdb must never be treated as "missing" just
    # because the combined candidate being evaluated happens to be a .gml (confirmed real bug:
    # nowomiejski 2024 already has full per-gmina GDB coverage for all 5 gminy, but a same-year
    # combined *.gml* staging file was still getting registered as a "gap-filler" because the
    # per-format check only looked at gml coverage, which was genuinely absent -- duplicating
    # every one of those 5 gminy's rows once merged). The true test is "does this gmina have ANY
    # working delivery already", not "does it have this exact container format".
    gdb_combined_candidates: list[Path] = []
    for gdb_path in year_dir.rglob("*.gdb"):
        unit = _unit_of(gdb_path)
        if unit:
            units.setdefault(unit, {}).setdefault("gdb", []).append(gdb_path)
        else:
            gdb_combined_candidates.append(gdb_path)

    gml_combined_candidates: list[Path] = []
    for gml_path in year_dir.rglob("*.gml"):
        unit = _unit_of(gml_path)
        if unit:
            units.setdefault(unit, {}).setdefault("gml", []).append(gml_path)
        else:
            gml_combined_candidates.append(gml_path)

    for shp_path in year_dir.rglob("*.shp"):
        unit = _unit_of(shp_path)
        if unit:
            units.setdefault(unit, {}).setdefault("shp", []).append(shp_path)

    # Both 3-letter (.swd) and 4-letter (.swde) extensions are used across real deliveries for
    # the exact same file format — confirmed on real data (260 files across the dataset use
    # ".swde", e.g. rok_2020/2815_ostrodzki/SWDE/opisowa/281501_1.swde). Missing the 4-letter
    # variant silently dropped whole deliveries from discovery.
    seen_swd: set[Path] = set()
    for pattern in ("*.swd", "*.SWD", "*.swde", "*.SWDE"):
        for swd_path in year_dir.rglob(pattern):
            if swd_path in seen_swd:
                continue  # case-insensitive filesystems can match the same file twice
            seen_swd.add(swd_path)
            unit = _unit_of(swd_path)
            if unit:
                units.setdefault(unit, {}).setdefault("swde", []).append(swd_path)

    # Pass 2: now that every normal per-gmina delivery for this year is known, evaluate combined
    # candidates for a genuine gap -- a "_wspolny" pseudo-unit is registered only when at least
    # one of its gminy has NO delivery at all yet (any format).
    def _register_combined_if_gap(path: Path, fmt: str) -> None:
        codes = _combined_powiat_codes(path)
        if not codes:
            return
        missing = {c for c in codes if c not in units}
        if not missing:
            logger.debug(
                "Pominięto {} — wszystkie {} gmin(y) mają już jakąś dostawę w tym roku, "
                "brak luki do wypełnienia.",
                path.name,
                len(codes),
            )
            return
        powiat = next(iter(codes))[:4]
        pseudo_unit = f"{powiat}_wspolny"
        units.setdefault(pseudo_unit, {}).setdefault(fmt, []).append(path)
        logger.info(
            "Wykryto scaloną dostawę całopowiatową: {} → traktowana jako jednostka '{}' "
            "(wypełnia lukę dla {} z {} gmin)",
            path.name,
            pseudo_unit,
            len(missing),
            len(codes),
        )

    for gdb_path in gdb_combined_candidates:
        _register_combined_if_gap(gdb_path, "gdb")
    for gml_path in gml_combined_candidates:
        _register_combined_if_gap(gml_path, "gml")

    return units


def _read_unit_layers(
    candidates: dict[str, list[Path]],
    layer_name_map: dict[str, list[str]],
    swde_crs: str = "EPSG:2178",
) -> tuple[dict[str, gpd.GeoDataFrame], str | None]:
    """
    Try GDB -> GML -> SHP -> SWDE for one unit; return (layers, format_used) for the first
    format that yields anything.

    Within GDB and GML, ALL candidate files for the unit are read and their layers merged
    (union by raw layer name, first candidate wins on a name collision) -- confirmed real
    deliveries ship two complementary .gdb files for the same unit: one carries
    DzialkaEwidencyjna/Budynek/etc. (typically "..._z_GML.gdb"), a separate one carries only
    KonturUzytkuGruntowego/KonturKlasyfikacyjny (named "..._swde.gdb"/"..._kontury_uzytki...gdb"/
    "..._UG_KL...gdb"). Stopping at the first non-empty candidate silently dropped whichever
    layer only the OTHER file carried (real cases recovered by this merge: ostródzki 281509_2
    rok_2023 missing Dzialka; braniewski 280203_4/5 rok_2024, lidzbarski 280901_1/280904_2/
    280905_4 rok_2024, mrągowski 281001_1 rok_2025 all missing UZG).
    """
    if "gdb" in candidates:
        merged: dict[str, gpd.GeoDataFrame] = {}
        for path in candidates["gdb"]:
            for name, gdf in gdb_reader.read_all_layers(path).items():
                merged.setdefault(name, gdf)
        if merged:
            return merged, "gdb"

    if "gml" in candidates:
        merged = {}
        for path in candidates["gml"]:
            for name, gdf in gml_reader.read_all_layers(path).items():
                merged.setdefault(name, gdf)
        if merged:
            return merged, "gml"

    if "shp" in candidates:
        layers = shp_reader.read_all_layers(candidates["shp"], layer_name_map)
        if layers:
            return layers, "shp"

    if "swde" in candidates:
        for path in candidates["swde"]:
            layers = swde_reader.read_all_layers(path, crs=swde_crs)
            if layers:
                return layers, "swde"

    return {}, None


def run_extraction_polygons(cfg: DictConfig) -> None:
    """
    Krok 'run_extraction_polygons':
    - iteruje po katalogach `rok_YYYY` w `cfg.data.base_dir`,
    - dla każdej jednostki ewidencyjnej próbuje kolejno formaty GDB -> GML -> SHP -> SWDE,
    - eksportuje wynik do Parquet w strukturze:
      base_dir / prepare.output_subdir / {unit_code}/{layer}/year=YYYY/{prepare.output_filename}
    """
    base_dir = Path(OmegaConf.select(cfg, "data.base_dir")).expanduser().resolve()
    out_subdir = str(OmegaConf.select(cfg, "prepare.output_subdir", default="parquets"))
    out_dir = base_dir / out_subdir

    uppercase_geometry = bool(OmegaConf.select(cfg, "prepare.uppercase_geometry", default=True))
    rename_legacy = bool(OmegaConf.select(cfg, "prepare.rename_legacy_2020", default=True))
    output_filename = str(
        OmegaConf.select(cfg, "prepare.output_filename", default="input_data.parquet")
    )
    layer_name_map: dict[str, list[str]] = dict(
        OmegaConf.select(cfg, "prepare.layer_name_map", default={})
    )
    # SWDE carries no CRS metadata anywhere in the file (unlike GDB/GML/SHP) — must be configured
    # per delivery region rather than hardcoded. Defaults to EPSG:2178 (CS2000 zone 7), correct for
    # the project's contracted scope (szczycieński / warmińsko-mazurskie).
    swde_crs = str(OmegaConf.select(cfg, "prepare.swde_crs", default="EPSG:2178"))
    # Everything is normalised to this CRS at write time (see _normalize_crs). Default EPSG:2180
    # (PUWG 1992) — the national grid all downstream steps assume.
    target_crs = str(OmegaConf.select(cfg, "prepare.target_crs", default="EPSG:2180"))

    logger.info("START run_extraction_polygons | base_dir={} → out_dir={}", base_dir, out_dir)

    if not base_dir.exists():
        logger.error("Base dir nie istnieje: {}", base_dir)
        return

    year_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and YEAR_PATTERN.match(d.name)])
    if not year_dirs:
        logger.warning("Nie znaleziono katalogów 'rok_YYYY' w: {}", base_dir)

    format_summary: dict[str, int] = {}

    for year_dir in tqdm(year_dirs, desc="📅 Przetwarzanie lat"):
        year = int(YEAR_PATTERN.match(year_dir.name).group(1))  # type: ignore[union-attr]
        units = _discover_units(year_dir)

        if not units:
            logger.warning("Brak rozpoznanych jednostek (GDB/GML/SHP/SWDE) w {}", year_dir)
            continue

        for unit_code in tqdm(sorted(units), desc=f"📁 {year_dir.name}", leave=False):
            layers, format_used = _read_unit_layers(units[unit_code], layer_name_map, swde_crs)

            if not layers:
                available = ", ".join(sorted(units[unit_code]))
                logger.warning(
                    "Jednostka {} w {}: brak użytecznej geometrii (dostępne formaty: {})",
                    unit_code,
                    year_dir.name,
                    available or "brak",
                )
                continue

            logger.info(
                "Jednostka {} w {}: wybrano format '{}' ({} warstw)",
                unit_code,
                year_dir.name,
                format_used,
                len(layers),
            )
            format_summary[format_used] = format_summary.get(format_used, 0) + 1

            for layer_name, gdf in layers.items():
                out_file = out_dir / unit_code / layer_name / f"year={year}" / output_filename
                _export_layer_to_parquet(
                    gdf,
                    out_file,
                    year=year if rename_legacy else None,
                    uppercase_geometry=uppercase_geometry,
                    naive_crs_fallback=swde_crs,
                    target_crs=target_crs,
                )

    logger.success(
        "PREPARE_DATA zakończone. Dane w: {} | jednostek wg formatu: {}",
        out_dir.resolve(),
        format_summary,
    )
