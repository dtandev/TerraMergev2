# src/prepare_data/prepare_data.py
from __future__ import annotations

import re
from pathlib import Path

import geopandas as gpd
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from osgeo import ogr
from tqdm import tqdm

# --- Reguły/stałe domenowe (na później łatwo przenieść do YAML, jeśli zechcesz) ---
YEAR_PATTERN = re.compile(r"^rok_(\d{4})$")
UNIT_PATTERN = re.compile(r"(2815\d{2}_\d)")

RENAME_2020_TO_2024: dict[str, str] = {
    "G5IDD": "idDzialki",
    "g5idd": "idDzialki",  # <- dodane: legacy bywało też małymi literami
    "SHAPE_Length": "Shape_Length",
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


def _extract_polygon_layers(gdb_path: Path) -> list[tuple[str, str]]:
    """Zwraca listę (layer_name, geom_type) dla warstw typu Polygon/MultiPolygon w GDB."""
    ds = ogr.Open(str(gdb_path), 0)
    if ds is None:
        logger.error("Nie można otworzyć GDB: {}", gdb_path)
        return []

    layers: list[tuple[str, str]] = []
    for i in range(ds.GetLayerCount()):
        layer = ds.GetLayerByIndex(i)
        name = layer.GetName()
        geom_type = ogr.GeometryTypeToName(layer.GetGeomType())
        if geom_type in ("Polygon", "Multi Polygon", "MultiPolygon"):
            layers.append((name, geom_type))
        else:
            logger.debug("Pomijam warstwę nie-poligonową: {} ({})", name, geom_type)
    return layers


def _export_to_parquet(
    gdb_path: Path,
    layer_name: str,
    out_path: Path,
    *,
    year: int | None = None,
    uppercase_geometry: bool = True,
) -> None:
    """Eksport pojedynczej warstwy do Parquet + UPPERCASE kolumn + opcjonalne mapowanie legacy."""
    try:
        gdf = gpd.read_file(str(gdb_path), layer=layer_name)

        if year is not None and year < 2021:
            gdf = gdf.rename(columns=RENAME_2020_TO_2024)

        gdf, rename_map = _to_uppercase_columns(gdf, uppercase_geometry=uppercase_geometry)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_parquet(out_path, index=False)

        if rename_map:
            logger.info("Zapisano {} ({} zmienionych nazw kolumn)", out_path, len(rename_map))
        else:
            logger.info("Zapisano {}", out_path)
    except Exception as e:
        logger.exception("Błąd eksportu: {} | layer={} → {}", gdb_path.name, layer_name, e)


def run_extraction_polygons(cfg: DictConfig) -> None:
    """
    Krok 'run_extraction_polygons':
    - iteruje po katalogach `rok_YYYY` w `cfg.data.base_dir`,
    - znajduje pliki .gdb,
    - wykrywa polygonowe warstwy,
    - eksportuje do Parquet w strukturze:
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

    logger.info("START run_extraction_polygons | base_dir={} → out_dir={}", base_dir, out_dir)

    if not base_dir.exists():
        logger.error("Base dir nie istnieje: {}", base_dir)
        return

    year_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and YEAR_PATTERN.match(d.name)])
    if not year_dirs:
        logger.warning("Nie znaleziono katalogów 'rok_YYYY' w: {}", base_dir)

    for year_dir in tqdm(year_dirs, desc="📅 Przetwarzanie lat"):
        year = int(YEAR_PATTERN.match(year_dir.name).group(1))  # type: ignore[union-attr]
        gdb_files = sorted(year_dir.rglob("*.gdb"))

        if not gdb_files:
            logger.warning("Brak plików .gdb w {}", year_dir)
            continue

        for gdb_path in tqdm(gdb_files, desc=f"📁 {year_dir.name}", leave=False):
            unit_match = UNIT_PATTERN.search(gdb_path.name)
            unit_code = unit_match.group(1) if unit_match else "unknown_unit"

            logger.info("Szukam warstw poligonowych: {}", gdb_path)
            layers = _extract_polygon_layers(gdb_path)

            if not layers:
                logger.info("Brak warstw poligonowych w: {}", gdb_path.name)
                continue

            for layer_name, _geom_type in layers:
                out_file = out_dir / unit_code / layer_name / f"year={year}" / output_filename
                logger.info("Eksport: layer={} → {}", layer_name, out_file)
                _export_to_parquet(
                    gdb_path,
                    layer_name,
                    out_file,
                    year=year if rename_legacy else None,
                    uppercase_geometry=uppercase_geometry,
                )

    logger.success("PREPARE_DATA zakończone. Dane w: {}", out_dir.resolve())
